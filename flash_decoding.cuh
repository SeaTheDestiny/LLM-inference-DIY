/**
 * flash_decoding.cuh — Flash Decoding: Two-Stage Split-KV Attention
 * ===================================================================
 * Stage 1: each block handles 1 KV chunk + 1 Q tile
 *   Output: O_partial[num_head][num_splits][Q_seqlen][kHeadDim]
 *           LSE[num_head][num_splits][Q_seqlen]  (log-sum-exp per row)
 *
 * Stage 2: LSE-weighted reduction over KV splits
 *   Input:  O_partial, LSE
 *   Output: O_final[num_head][Q_seqlen][kHeadDim]
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
using namespace nvcuda;

#define WARP_SIZE 32
#define HALF2(val) (reinterpret_cast<half2 *>(&(val))[0])
#define LDST128BITS(val) (reinterpret_cast<float4 *>(&(val))[0])
#define LDST32BITS(val) (reinterpret_cast<half2 *>(&(val))[0])

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                     \
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
               : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr)                                              \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
               : "=r"(R0), "=r"(R1) : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)            \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " \
               "%4, %5}, {%6, %7}, {%8, %9};\n"                                 \
               : "=r"(RD0), "=r"(RD1)                                           \
               : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

// ============================================================
// Stage 1: Per-Chunk Flash Attention (no K-seqlen loop)
// ============================================================
// Each block = 1 warp, handles Q_tile[Br=16,d] × K_chunk[Bc=16,d]
// grid: (num_kv_splits, num_heads * num_q_tiles)
//   blockIdx.x → kv_chunk (0..Nsplits-1)
//   blockIdx.y → head_id * num_q_tiles + q_tile

template <const int kHeadDim, const int kBr = 16, const int kBc = 16>
__global__ void __launch_bounds__(WARP_SIZE)
    flash_decode_stage1_kernel(half *Q, half *K, half *V,
                               half *O_partial, float *LSE,
                               int Q_seqlen, int KV_seqlen,
                               int num_heads, int num_splits_kv) {
    constexpr int kMmaAtomM = 16, kMmaAtomN = 8, kMmaAtomK = 16;

    int kv_chunk   = blockIdx.x;                        // which KV split [0, Nsplits)
    int q_head_idx = blockIdx.y;                        // combined head+q_tile
    int num_q_tiles = (Q_seqlen + kBr - 1) / kBr;
    int head_id   = q_head_idx / num_q_tiles;
    int q_tile    = q_head_idx % num_q_tiles;

    int Q_head_offset = head_id * Q_seqlen * kHeadDim;
    int K_head_offset = head_id * KV_seqlen * kHeadDim;
    int V_head_offset = head_id * KV_seqlen * kHeadDim;

    int Q_tile_start = q_tile * kBr;                    // global Q row start
    int K_chunk_start = kv_chunk * kBc;                 // global K row start

    // O_partial layout: [num_heads][num_splits][Q_seqlen][kHeadDim]
    // LSE layout:       [num_heads][num_splits][Q_seqlen]
    int O_part_base = (head_id * num_splits_kv + kv_chunk) * Q_seqlen * kHeadDim;
    int LSE_base    = (head_id * num_splits_kv + kv_chunk) * Q_seqlen;

    __shared__ half s_q[kBr][kHeadDim], s_k[kBc][kHeadDim], s_v[kBc][kHeadDim];
    int lane_id = threadIdx.x;

    // g(Q) -> s(Q): Br*d halfs, 32 threads, each loads (Br*d/32/8) × float4
    #pragma unroll
    for (int iter = 0; iter < (kHeadDim * kBr) / (WARP_SIZE * 8); iter++) {
        int row = lane_id / 2;
        int col = (lane_id % 2) * 8 + iter * 16;
        int g_row = Q_tile_start + row;
        if (g_row < Q_seqlen)
            LDST128BITS(s_q[row][col]) = LDST128BITS(Q[Q_head_offset + g_row * kHeadDim + col]);
        else {
            float4 z = {0,0,0,0}; LDST128BITS(s_q[row][col]) = z;
        }
    }

    // g(K) -> s(K): Bc*d halfs
    #pragma unroll
    for (int iter = 0; iter < (kHeadDim * kBc) / (WARP_SIZE * 8); iter++) {
        int row = lane_id / 2;
        int col = (lane_id % 2) * 8 + iter * 16;
        int g_row = K_chunk_start + row;
        if (g_row < KV_seqlen)
            LDST128BITS(s_k[row][col]) = LDST128BITS(K[K_head_offset + g_row * kHeadDim + col]);
        else {
            float4 z = {0,0,0,0}; LDST128BITS(s_k[row][col]) = z;
        }
    }

    // g(V) -> s(V): Bc*d halfs
    #pragma unroll
    for (int iter = 0; iter < (kHeadDim * kBc) / (WARP_SIZE * 8); iter++) {
        int row = lane_id / 2;
        int col = (lane_id % 2) * 8 + iter * 16;
        int g_row = K_chunk_start + row;
        if (g_row < KV_seqlen)
            LDST128BITS(s_v[row][col]) = LDST128BITS(V[V_head_offset + g_row * kHeadDim + col]);
        else {
            float4 z = {0,0,0,0}; LDST128BITS(s_v[row][col]) = z;
        }
    }
    __syncthreads();

    // Initialize
    uint32_t R_Q[4], R_K[2], R_O[kHeadDim / kMmaAtomN][2];
    #pragma unroll
    for (int i = 0; i < kHeadDim / kMmaAtomN; i++)
        { R_O[i][0] = 0; R_O[i][1] = 0; }

    float row_max[2] = {-INFINITY, -INFINITY};
    float row_l[2]   = {0.0f, 0.0f};

    // Q@K^T + Softmax + P@V (SINGLE KV chunk, no outer loop!)
    // S = Q[Br,d] @ K^T[d,Bc]
    uint32_t R_S[2][2] = {{0,0},{0,0}};  // Bc/kMmaAtomN=2 for Bc=16

    float scale = 1.0f / sqrtf((float)kHeadDim);

    // Q@K^T MMA over d dimension
    for (int i_dim = 0; i_dim < kHeadDim / kMmaAtomK; i_dim++) {
        int q_row = lane_id % kMmaAtomM;
        int q_col = (lane_id / kMmaAtomM) * 8 + i_dim * kMmaAtomK;
        uint32_t q_addr = __cvta_generic_to_shared(&s_q[q_row][q_col]);
        LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], q_addr);

        #pragma unroll
        for (int kt = 0; kt < 2; kt++) {  // Bc=16, 2 tiles of 8
            int k_row = lane_id % kMmaAtomN + kt * kMmaAtomN;
            int k_col = ((lane_id / kMmaAtomN) % 2) * 8 + i_dim * kMmaAtomK;
            uint32_t k_addr = __cvta_generic_to_shared(&s_k[k_row][k_col]);
            LDMATRIX_X2(R_K[0], R_K[1], k_addr);

            HMMA16816(R_S[kt][0], R_S[kt][1],
                      R_Q[0], R_Q[1], R_Q[2], R_Q[3],
                      R_K[0], R_K[1],
                      R_S[kt][0], R_S[kt][1]);
        }
    }

    // Softmax (local to this chunk, NO rescaling from previous chunks)
    float2 f_res[2][2];
    #pragma unroll
    for (int kt = 0; kt < 2; kt++) {
        f_res[kt][0] = __half22float2(HALF2(R_S[kt][0]));
        f_res[kt][1] = __half22float2(HALF2(R_S[kt][1]));
        f_res[kt][0].x *= scale; f_res[kt][0].y *= scale;
        f_res[kt][1].x *= scale; f_res[kt][1].y *= scale;

        int k_col_x = K_chunk_start + ((lane_id % 4) * 2 + 0) + kt * 8;
        int k_col_y = K_chunk_start + ((lane_id % 4) * 2 + 1) + kt * 8;
        if (k_col_x >= KV_seqlen) { f_res[kt][0].x = -INFINITY; f_res[kt][1].x = -INFINITY; }
        if (k_col_y >= KV_seqlen) { f_res[kt][0].y = -INFINITY; f_res[kt][1].y = -INFINITY; }
    }

    // Find local max
    float m0 = -INFINITY, m1 = -INFINITY;
    #pragma unroll
    for (int kt = 0; kt < 2; kt++) {
        m0 = fmaxf(m0, fmaxf(f_res[kt][0].x, f_res[kt][0].y));
        m1 = fmaxf(m1, fmaxf(f_res[kt][1].x, f_res[kt][1].y));
    }

    // Warp reduce max
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffff, m1, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffff, m1, 2));

    // Exp + sum
    float l0 = 0.0f, l1 = 0.0f;
    #pragma unroll
    for (int kt = 0; kt < 2; kt++) {
        f_res[kt][0].x = expf(f_res[kt][0].x - m0);
        f_res[kt][0].y = expf(f_res[kt][0].y - m0);
        f_res[kt][1].x = expf(f_res[kt][1].x - m1);
        f_res[kt][1].y = expf(f_res[kt][1].y - m1);
        l0 += f_res[kt][0].x + f_res[kt][0].y;
        l1 += f_res[kt][1].x + f_res[kt][1].y;
        HALF2(R_S[kt][0]) = __float22half2_rn(f_res[kt][0]);
        HALF2(R_S[kt][1]) = __float22half2_rn(f_res[kt][1]);
    }

    l0 += __shfl_xor_sync(0xffffffff, l0, 1);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1);
    l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 2);

    // P@V MMA
    #pragma unroll
    for (int i_dim = 0; i_dim < kHeadDim / kMmaAtomN; i_dim++) {
        // P tile: [Br,Bc] = [16,16], kMmaAtomK=16 → 1 p_step
        int p_col = (lane_id / kMmaAtomM) * 8;
        // Use R_S directly (P is 2×16×8 fragments, only 1 p_step since Bc=16)
        int v_row = lane_id % kMmaAtomM;
        int v_col = ((lane_id / kMmaAtomM) % 2) * 8 + i_dim * kMmaAtomN;
        uint32_t v_addr = __cvta_generic_to_shared(&s_v[v_row][v_col]);
        LDMATRIX_X2(R_K[0], R_K[1], v_addr);  // col-major load for V^T

        HMMA16816(R_O[i_dim][0], R_O[i_dim][1],
                  R_S[0][0], R_S[0][1], R_S[1][0], R_S[1][1],
                  R_K[0], R_K[1],
                  R_O[i_dim][0], R_O[i_dim][1]);
    }

    // ---- Write O_partial & LSE ----
    float inv_l0 = __frcp_rn(l0);
    float inv_l1 = __frcp_rn(l1);

    #pragma unroll
    for (int i_dim = 0; i_dim < kHeadDim / kMmaAtomN; i_dim++) {
        float2 f0 = __half22float2(HALF2(R_O[i_dim][0]));
        float2 f1 = __half22float2(HALF2(R_O[i_dim][1]));
        f0.x *= inv_l0; f0.y *= inv_l0;
        f1.x *= inv_l1; f1.y *= inv_l1;

        int o_row = lane_id / 4;
        int o_col = (lane_id % 4) * 2 + i_dim * kMmaAtomN;
        int g_row0 = Q_tile_start + o_row + 0;
        int g_row8 = Q_tile_start + o_row + 8;

        if (g_row0 < Q_seqlen)
            LDST32BITS(O_partial[O_part_base + (o_row+0) * kHeadDim + o_col]) = __float22half2_rn(f0);
        if (g_row8 < Q_seqlen)
            LDST32BITS(O_partial[O_part_base + (o_row+8) * kHeadDim + o_col]) = __float22half2_rn(f1);
    }

    // LSE = log(sum(exp(S - max))) + max = log(l) + m  (for numerical stability in Stage 2)
    // Store log-sum-exp per row: LSE = log(l) + m
    // But actually we store both m and l for Stage 2 reduction:
    // Stage 2 formula: O = Σ(O_i × exp(LSE_i - maxLSE)) / Σ(exp(LSE_i - maxLSE))
    // where LSE_i = log(l_i) + m_i = log(sum(exp(S_i - m_i))) + m_i
    
    // Store: LSE[row] = log(row_l) + row_max  (log-sum-exp)
    {
        int lane_0 = lane_id / 4;        // which row-pair's first row (0..7)
        float lse0 = logf(l0) + m0;
        float lse1 = logf(l1) + m1;

        int g_row0 = Q_tile_start + lane_0 + 0;
        int g_row8 = Q_tile_start + lane_0 + 8;

        if (g_row0 < Q_seqlen && lane_id % 4 == 0)
            LSE[LSE_base + (lane_0+0)] = lse0;
        if (g_row8 < Q_seqlen && lane_id % 4 == 0)
            LSE[LSE_base + (lane_0+8)] = lse1;
    }
}


// ============================================================
// Stage 2: LSE-Weighted Reduction
// ============================================================
// grid: (num_q_tiles * num_heads, 1) — each block = 1 Q tile (16 rows)
// block: 512 threads = 16 warps, each warp handles 1 Q row
// Within each warp: 32 threads cooperate on kHeadDim elements

__global__ void flash_decode_stage2_kernel(
    half *O_partial, float *LSE, half *O_final,
    int Q_seqlen, int num_heads, int num_splits_kv, int kHeadDim) {

    constexpr int kBr = 16;
    int tid = blockIdx.x;
    int num_q_tiles = (Q_seqlen + kBr - 1) / kBr;
    int head_id = tid / num_q_tiles;
    int q_tile = tid % num_q_tiles;
    int q_start = q_tile * kBr;

    int warp_id = threadIdx.x / WARP_SIZE;     // 0..15
    int lane_id = threadIdx.x % WARP_SIZE;     // 0..31
    int q_row = q_start + warp_id;             // each warp = one Q row

    if (q_row >= Q_seqlen) return;

    int d_per_thread = kHeadDim / WARP_SIZE;   // e.g. 128/32=4

    // Step 1: Find max LSE across all KV splits for this row
    float max_lse = -INFINITY;
    for (int s = 0; s < num_splits_kv; s++) {
        int lse_idx = (head_id * num_splits_kv + s) * Q_seqlen + q_row;
        float val = LSE[lse_idx];
        max_lse = fmaxf(max_lse, val);
    }
    // Warp reduce max (all lanes get the same value)
    max_lse = fmaxf(max_lse, __shfl_xor_sync(0xffffffff, max_lse, 1));
    max_lse = fmaxf(max_lse, __shfl_xor_sync(0xffffffff, max_lse, 2));
    max_lse = fmaxf(max_lse, __shfl_xor_sync(0xffffffff, max_lse, 4));
    max_lse = fmaxf(max_lse, __shfl_xor_sync(0xffffffff, max_lse, 8));
    max_lse = fmaxf(max_lse, __shfl_xor_sync(0xffffffff, max_lse, 16));

    // Step 2: Weighted sum over KV splits
    float sum_weights = 0.0f;
    float O_acc[8];  // up to 8 elements per thread (d=256 → 8, d=128 → 4, d=64 → 2)
    #pragma unroll
    for (int i = 0; i < d_per_thread; i++) O_acc[i] = 0.0f;

    for (int s = 0; s < num_splits_kv; s++) {
        int lse_idx = (head_id * num_splits_kv + s) * Q_seqlen + q_row;
        float weight = expf(LSE[lse_idx] - max_lse);
        sum_weights += weight;

        int o_base = (head_id * num_splits_kv + s) * Q_seqlen * kHeadDim + q_row * kHeadDim;
        #pragma unroll
        for (int d_idx = 0; d_idx < d_per_thread; d_idx++) {
            int d_off = lane_id * d_per_thread + d_idx;
            half val = O_partial[o_base + d_off];
            O_acc[d_idx] += __half2float(val) * weight;
        }
    }

    // Step 3: Write final O
    float inv_w = 1.0f / sum_weights;
    int o_final_base = head_id * Q_seqlen * kHeadDim + q_row * kHeadDim;
    #pragma unroll
    for (int d_idx = 0; d_idx < d_per_thread; d_idx++) {
        int d_off = lane_id * d_per_thread + d_idx;
        O_final[o_final_base + d_off] = __float2half(O_acc[d_idx] * inv_w);
    }
}
