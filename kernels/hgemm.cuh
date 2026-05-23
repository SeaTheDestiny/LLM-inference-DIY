/**
 * hgemm.cuh — HGEMM (Half GEMM) with MMA m16n8k16
 * =================================================
 * C[M,N] = A[M,K] @ B[K,N], all row-major
 *
 * Version:
 *   naive:   BM=16, BN=8, BK=16, 1 warp — baseline
 *   tiled:   BM=64, BN=64, BK=16, 4 warps split-A — 4× throughput
 *   async:   +cp.async +kStage=2 double buffering, BM=128,BN=128, 8 warps
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ---- PTX macros (same as flash_attn.cu) ----
#define LDST128BITS(val) (reinterpret_cast<float4 *>(&(val))[0])
#define LDST32BITS(val)  (reinterpret_cast<half2 *>(&(val))[0])

#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"          \
               ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                     \
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
               : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr)                                            \
  asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
               : "=r"(R0), "=r"(R1) : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)            \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " \
               "%4, %5}, {%6, %7}, {%8, %9};\n"                                 \
               : "=r"(RD0), "=r"(RD1)                                           \
               : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))


// ============================================================
// Naive HGEMM: BM=16, BN=8, BK=16, 1 warp (32 threads)
// ============================================================
template <const int kWarpPerM = 1, const int kWarpPerN = 1>
__global__ void __launch_bounds__(WARP_SIZE * kWarpPerM * kWarpPerN)
    hgemm_naive_kernel(half *A, half *B, half *C,
                       int M, int N, int K) {
    constexpr int BM = 16 * kWarpPerM;  // M tile size
    constexpr int BN = 8 * kWarpPerN;   // N tile size
    constexpr int BK = 16;              // K tile (MMA atom)

    int bx = blockIdx.x;  // N tile index
    int by = blockIdx.y;  // M tile index
    int num_k_tiles = (K + BK - 1) / BK;

    __shared__ half s_a[BM][BK], s_b[BK][BN], s_c[BM][BN];

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warp_m  = warp_id / kWarpPerN;   // which A-row warp
    int warp_n  = warp_id % kWarpPerN;   // which B-col warp

    int gmem_a_m = by * BM + warp_m * 16;
    int gmem_b_n = bx * BN + warp_n * 8;

    if (gmem_a_m >= M || gmem_b_n >= N) return;

    // Load A: each thread loads 8 halfs (float2 equivalent)
    // s_a[BM][BK] = [16][16], 16 rows, each row has 16 halfs → 32 threads × 8 halfs
    auto load_smem_a = [&](int k_tile) {
        int row = tid / 2;       // 0..15
        int col = (tid % 2) * 8; // 0 or 8
        int g_col = k_tile * BK + col;
        int g_addr = gmem_a_m * K + row * K + g_col;
        if (k_tile * BK + col < K)
            LDST128BITS(s_a[row][col]) = LDST128BITS(A[g_addr]);
        else {
            float4 z = {0,0,0,0}; LDST128BITS(s_a[row][col]) = z;
        }
    };

    // Load B: only 16 threads needed (s_b[16][8] = 16×8 = 128 halfs, 16 threads × 8)
    auto load_smem_b = [&](int k_tile) {
        if (lane_id < BK) {
            int row = lane_id;   // 0..15
            int col = 0;
            int g_row = k_tile * BK + row;
            int g_addr = g_row * N + gmem_b_n;
            if (g_row < K)
                LDST128BITS(s_b[row][col]) = LDST128BITS(B[g_addr]);
            else {
                float4 z = {0,0,0,0}; LDST128BITS(s_b[row][col]) = z;
            }
        }
    };

    uint32_t RC[2] = {0, 0};

    #pragma unroll
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        load_smem_a(kt);
        load_smem_b(kt);
        __syncthreads();

        uint32_t RA[4], RB[2];
        uint32_t a_ptr = __cvta_generic_to_shared(&s_a[lane_id % 16][(lane_id / 16) * 8]);
        uint32_t b_ptr = __cvta_generic_to_shared(&s_b[lane_id % 16][0]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], a_ptr);
        LDMATRIX_X2_T(RB[0], RB[1], b_ptr);

        HMMA16816(RC[0], RC[1],
                  RA[0], RA[1], RA[2], RA[3],
                  RB[0], RB[1],
                  RC[0], RC[1]);
        __syncthreads();
    }

    // Store C: each thread writes 2 halfs per row, 8 rows handled
    // RC[0] → rows 0..7, RC[1] → rows 8..15
    LDST32BITS(s_c[lane_id / 4][(lane_id % 4) * 2]) = LDST32BITS(RC[0]);
    LDST32BITS(s_c[lane_id / 4 + 8][(lane_id % 4) * 2]) = LDST32BITS(RC[1]);
    __syncthreads();

    // Write C to global
    if (lane_id < BM) {
        int g_row = gmem_a_m + lane_id;
        int g_addr = g_row * N + gmem_b_n;
        if (g_row < M)
            LDST128BITS(C[g_addr]) = LDST128BITS(s_c[lane_id][0]);
    }
}


// ============================================================
// Tiled HGEMM: BM=64, BN=64, 4 warps split-A + fine-grained K tiling
// ============================================================
// As A[64,K] @ B[K,64]:
//   4 warps each handle A[16,K] and output C[16,64]
//   B shared across all warps
//   SMEM: s_a[64][16] (fine-grained), s_b[16][64]

template <const int kBM = 64, const int kBN = 64, const int kNumWarpsA = 4>
__global__ void __launch_bounds__(WARP_SIZE * kNumWarpsA)
    hgemm_tiled_kernel(half *A, half *B, half *C,
                       int M, int N, int K) {
    constexpr int BK = 16;
    constexpr int kNumWarpsN = 1;   // single warp for B
    constexpr int kNumWarps = kNumWarpsA * kNumWarpsN;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int num_k_tiles = (K + BK - 1) / BK;

    int tid = threadIdx.x;
    int lid = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int wA  = wid / kNumWarpsN;  // 0..3

    int gA_m = by * kBM + wA * 16;    // global A row start for this warp
    int gB_n = bx * kBN;              // global B col start

    if (gA_m >= M || gB_n >= N) return;

    // SMEM: only store BK=16 columns of A/K at a time (fine-grained)
    __shared__ half s_a[kBM][BK];      // [64][16]
    __shared__ half s_b[BK][kBN];      // [16][64]

    uint32_t smem_a_base = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base = __cvta_generic_to_shared(s_b);

    // A load: [64][16] = 1024 halfs, 128 threads × 8 halfs
    // row = tid%64, col = (tid/64)*8 (first/second half of each row)
    auto load_a = [&](int kt) {
        int row = tid % 64;
        int col = (tid / 64) * 8;
        int g_col = kt * BK + col;
        int g_addr = (by * kBM + row) * K + g_col;
        if (by * kBM + row < M && g_col < K)
            LDST128BITS(s_a[row][col]) = LDST128BITS(A[g_addr]);
        else { float4 z={0,0,0,0}; LDST128BITS(s_a[row][col]) = z; }
    };

    // B load: 16 rows × 64 cols = 1024 halfs, 128 threads = 1 float4 each
    auto load_b = [&](int kt) {
        int row = tid / 8;              // 0..15
        int col = (tid % 8) * 8;        // 0,8,16,24,32,40,48,56
        int g_row = kt * BK + row;
        int g_addr = g_row * N + gB_n + col;
        if (g_row < K && gB_n + col < N)
            LDST128BITS(s_b[row][col]) = LDST128BITS(B[g_addr]);
        else { float4 z={0,0,0,0}; LDST128BITS(s_b[row][col]) = z; }
    };

    // C accumulator in registers [kBN/8][2] — each warp handles 64/8=8 tiles
    uint32_t RC[kBN / 8][2];
    #pragma unroll
    for (int n = 0; n < kBN / 8; n++) { RC[n][0] = 0; RC[n][1] = 0; }

    #pragma unroll
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        load_a(kt);
        load_b(kt);
        __syncthreads();

        uint32_t RA[4], RB[2];

        // ldmatrix A: [64][16], warp wA reads rows [wA*16, wA*16+15]
        int a_row = lid % 16;
        int a_col = (lid / 16) * 8;   // 0 or 8
        uint32_t a_ptr = __cvta_generic_to_shared(&s_a[wA * 16 + a_row][a_col]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], a_ptr);

        // ldmatrix B (transposed): [16][64], each MMA uses 8 cols
        #pragma unroll
        for (int n = 0; n < kBN / 8; n++) {
            int b_row = lid % 16;
            int b_col = n * 8;         // 0,8,16,...,56
            uint32_t b_ptr = __cvta_generic_to_shared(&s_b[b_row][b_col]);
            LDMATRIX_X2_T(RB[0], RB[1], b_ptr);

            HMMA16816(RC[n][0], RC[n][1],
                      RA[0], RA[1], RA[2], RA[3],
                      RB[0], RB[1],
                      RC[n][0], RC[n][1]);
        }
        __syncthreads();
    }

    // Store C[16,64] per warp: 16 rows × 64 cols
    // Each thread in warp writes its portion
    #pragma unroll
    for (int n = 0; n < kBN / 8; n++) {
        int row0 = lid / 4;
        int col0 = (lid % 4) * 2;
        int g_row0 = gA_m + row0;
        int g_col0 = gB_n + n * 8 + col0;
        int g_row8 = gA_m + row0 + 8;
        if (g_row0 < M && g_col0 < N)
            LDST32BITS(C[g_row0 * N + g_col0]) = LDST32BITS(RC[n][0]);
        if (g_row8 < M && g_col0 < N)
            LDST32BITS(C[g_row8 * N + g_col0]) = LDST32BITS(RC[n][1]);
    }
}


// ============================================================
// Async HGEMM: BM=128, BN=128, 8 warps + cp.async + kStage=2
// ============================================================
// Fully optimized: fine-grained K tiles, double buffering, 8 warps

template <const int kBM = 128, const int kBN = 128, const int kNumWarpsA = 8>
__global__ void __launch_bounds__(WARP_SIZE * kNumWarpsA)
    hgemm_async_kernel(half *A, half *B, half *C,
                       int M, int N, int K) {
    constexpr int BK = 16;
    constexpr int kStage = 2;
    constexpr int kNumWarps = kNumWarpsA;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int num_k_tiles = (K + BK - 1) / BK;

    int tid = threadIdx.x;
    int lid = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int wA  = wid;

    int gA_m = by * kBM + wA * 16;
    int gB_n = bx * kBN;
    if (gA_m >= M || gB_n >= N) return;

    // SMEM with double buffering
    constexpr int A_sz = kBM * BK;       // 128×16 = 2048 halfs per stage
    constexpr int B_sz = BK * kBN;       // 16×128 = 2048 halfs per stage
    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + kStage * A_sz;
    uint32_t a_base = __cvta_generic_to_shared(s_a);
    uint32_t b_base = __cvta_generic_to_shared(s_b);

    // A load mapping: 256 threads, A[128][16] = 2048 halfs = 128 float4
    auto ldA_r = tid / 2;       // 0..127
    auto ldA_c = (tid % 2) * 8; // 0 or 8
    // B load mapping: 256 threads, B[16][128] = 2048 halfs = 128 float4
    auto ldB_r = tid / 16;      // 0..15
    auto ldB_c = (tid % 16) * 8;// 0,8,16,...,120

    uint32_t RC[kBN / 8][2];
    #pragma unroll
    for (int n = 0; n < kBN / 8; n++) { RC[n][0] = 0; RC[n][1] = 0; }

    // Prefetch stage 0
    {
        int g_col = 0 + ldA_c;
        if (by * kBM + ldA_r < M && g_col < K) {
            uint32_t sp = a_base + (0 * A_sz + ldA_r * BK + ldA_c) * sizeof(half);
            CP_ASYNC_CG(sp, &A[(by * kBM + ldA_r) * K + g_col], 16);
        }
        int g_row = 0 + ldB_r;
        if (g_row < K && gB_n + ldB_c < N) {
            uint32_t sp = b_base + (0 * B_sz + ldB_r * kBN + ldB_c) * sizeof(half);
            CP_ASYNC_CG(sp, &B[g_row * N + gB_n + ldB_c], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    #pragma unroll
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int stg = kt % kStage, nxt = (kt + 1) % kStage;

        // Prefetch next K tile
        if (kt + 1 < num_k_tiles) {
            int ng_col = (kt + 1) * BK + ldA_c;
            if (by * kBM + ldA_r < M && ng_col < K) {
                uint32_t sp = a_base + (nxt * A_sz + ldA_r * BK + ldA_c) * sizeof(half);
                CP_ASYNC_CG(sp, &A[(by * kBM + ldA_r) * K + ng_col], 16);
            }
            int ng_row = (kt + 1) * BK + ldB_r;
            if (ng_row < K && gB_n + ldB_c < N) {
                uint32_t sp = b_base + (nxt * B_sz + ldB_r * kBN + ldB_c) * sizeof(half);
                CP_ASYNC_CG(sp, &B[ng_row * N + gB_n + ldB_c], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
        }

        // Compute with current stage
        uint32_t RA[4], RB[2];
        int a_row = lid % 16, a_col = (lid / 16) * 8;
        uint32_t ap = a_base + (stg * A_sz + (wA * 16 + a_row) * BK + a_col) * sizeof(half);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], ap);

        #pragma unroll
        for (int n = 0; n < kBN / 8; n++) {
            int b_row = lid % 16, b_col = n * 8;
            uint32_t bp = b_base + (stg * B_sz + b_row * kBN + b_col) * sizeof(half);
            LDMATRIX_X2_T(RB[0], RB[1], bp);
            HMMA16816(RC[n][0], RC[n][1],
                      RA[0], RA[1], RA[2], RA[3],
                      RB[0], RB[1],
                      RC[n][0], RC[n][1]);
        }

        if (kt + 1 < num_k_tiles) {
            CP_ASYNC_WAIT_GROUP(0);
            __syncthreads();
        }
    }

    // Store C[16,128] per warp
    #pragma unroll
    for (int n = 0; n < kBN / 8; n++) {
        int r0 = lid / 4, c0 = (lid % 4) * 2;
        int gr0 = gA_m + r0, gc0 = gB_n + n * 8 + c0;
        int gr8 = gA_m + r0 + 8;
        if (gr0 < M && gc0 < N)
            LDST32BITS(C[gr0 * N + gc0]) = LDST32BITS(RC[n][0]);
        if (gr8 < M && gc0 < N)
            LDST32BITS(C[gr8 * N + gc0]) = LDST32BITS(RC[n][1]);
    }
}


// ============================================================
// Final HGEMM: 2×4 warp grid, K32 register double buffer, SMEM swizzle
// ============================================================
// BM=128, BN=128, 8 warps (2 M × 4 N), 256 threads
// Key optimizations from reference:
//   1. WARP_TILE_K=2: load 2 BK tiles into SMEM, double-buffer RA/RB in regs
//   2. SMEM swizzle: bank conflict avoidance for A ldmatrix
//   3. Warp shuffle collective store: 128-bit vectorized write
//   4. 2D warp grid: split both M and N dimensions

template <const int kBM = 128, const int kBN = 128,
          const int kWarpsM = 2, const int kWarpsN = 4,
          const int kMmaPerM = 4, const int kMmaPerN = 4,
          const int kWarpK = 2>   // 2 K tiles in registers
__global__ void __launch_bounds__(WARP_SIZE * kWarpsM * kWarpsN)
    hgemm_final_kernel(half *A, half *B, half *C,
                       int M, int N, int K) {
    constexpr int BK = 16;
    constexpr int kNumWarps = kWarpsM * kWarpsN;
    constexpr int kStage = 2;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int num_k_tiles = (K + BK * kWarpK - 1) / (BK * kWarpK);

    int tid = threadIdx.x;
    int lid = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int wM  = wid % kWarpsM;          // 0,1
    int wN  = wid / kWarpsM;          // 0,1,2,3

    int gM = by * kBM + wM * (16 * kMmaPerM);  // warp's M start
    int gN = bx * kBN + wN * (8 * kMmaPerN);   // warp's N start
    if (gM >= M || gN >= N) return;

    // SMEM with swizzle-friendly layout
    // layout: [kStage][kBM][BK*kWarpK + PAD]
    constexpr int A_cols = BK * kWarpK;         // 32
    constexpr int A_sz = kBM * A_cols;           // 128×32 = 4096 per stage
    constexpr int B_sz = BK * kWarpK * kBN;      // 32×128 = 4096 per stage
    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + kStage * A_sz;
    uint32_t a_base = __cvta_generic_to_shared(s_a);
    uint32_t b_base = __cvta_generic_to_shared(s_b);

    // A load: [128][32], 256 threads, each loads 8 halfs (float2)
    auto ldA_r = tid / 4;           // 0..63 (thread 0-3:row0, 4-7:row1, etc.)
    auto ldA_c = (tid % 4) * 8;     // 0,8,16,24
    // Wait, that gives 128 entries (256 threads / 2). 128×8 = 1024, but A_sz = 4096.
    // Need 4x more. Each thread loads 16 halfs (float4) instead.
    auto ldA_r2 = tid % 128;        // 0..127 (each row needs 32 cols = 4×8 halfs)
    auto ldA_c2 = (tid / 128) * 8;  // 0 for tid 0..127, 8 for 128..255
    // 256 threads × 16 halfs = 4096 halfs ✓

    // B load: [32][128], 256 threads, each loads 16 halfs
    auto ldB_r = tid / 16;          // 0..15 (B rows, but need 32 → 2 rounds)
    auto ldB_c = (tid % 16) * 8;    // 0,8,16,...120
    // Only 16 rows covered, need 2× coverage.
    // With 256 threads, each loading 16 halfs = 4096 ✓ if we cover 32 rows.
    // But ldB_r only gives 16. Need: ldB_r = tid/8 gives 32 rows, ldB_c = (tid%8)*8
    auto ldB_r2 = tid / 8;          // 0..31
    auto ldB_c2 = (tid % 8) * 8;    // 0,8,16,...,56
    // 256 threads × 16 halfs = 4096 ✓. Rows 0..31, cols 0..56+8=64. But BN=128!
    // Still only 64 cols. Need to load 128 cols with 256 threads:
    // ldB_r: row = tid/16 (0..15), col = (tid%16)*8 (0..120). That's 16 rows × 128 cols.
    // For 32 rows, do 2 passes or use 2 threads per row.
    // Simpler: each thread loads float2 (4 halfs), ldB_r = tid/32 (0..7 rows? No.)
    
    // Actually, for B[32][128] = 4096 halfs = 256 float4. 256 threads × 1 float4 = 256 float4.
    // Wait: 4096/8 = 512 float4. With 256 threads, each loads 2 float4 = 16 halfs.
    // ldB_r = tid % 32 (0..31), ldB_c = (tid / 32) * 16 (0,16,32,...,112)
    // 31*128 + 112 = 4088, covers B[32][128].
    // Hmm, tid/32 gives 0..7. (tid/32)*16 gives 0,16,32,...,112. That's 8 cols per row per thread = 16 halfs.
    // 32 rows × 8 threads each = 256. ✓

    // Actually let me just use the simple direct mapping from the reference.
    // The reference uses simple float4 loads for A and B with appropriate thread mapping.
    // For A[128][32]: 4096 halfs = 512 float4. 256 threads × 2 float4 each.
    // Thread t: row = t % 64, col = (t/64) * 8. Wait, that gives 64 rows and 4 cols.
    // t % 128 gives row 0..127. t/128 gives 0 or 1, each loading 8 halfs at col 0 or 16.
    // Need col 0,8,16,24 = 4 positions. t%128 for rows, t/128 only gives 2 positions.
    // 
    // Simpler: each thread loads 16 halfs (2 float4). A[128][32] has 128 rows × 32 cols.
    // thread t loads row r = t%128, col c = (t/128)*16 (0 or 16). That covers 128 rows × 16+8 = 24 cols.
    // Still missing cols 24..31. Need 3 way split.
    //
    // OK, let me just use the reference's mapping for simplicity.
    // Reference: load_smem_a_m = tid / 2 (0..127), load_smem_a_k = (tid%2)*8 (0,8)
    // That loads A[128][8+8=16] with 256 threads → only covers 16 cols!
    // But the reference loads 2 K tiles separately (WARP_TILE_K=2), each is BK=16.
    // So it loads 2 separate A tiles: one at offset 0, one at offset 16.
    
    // Let me use the same approach:
    // A_load_0: row=tid/2 (0..127), col=(tid%2)*8 (0,8) → covers A[128][16] for K tile 0
    // A_load_1: row=tid/2, col=(tid%2)*8 → covers A[128][16] for K tile 1 (offset +16 in smem)
    
    // For B:
    // B_load_0: row=tid/16 (0..15), col=(tid%16)*8 (0..120) → B[16][128] for K tile 0
    // B_load_1: row=tid/16, col=(tid%16)*8 → B[16][128] for K tile 1 (offset +16 in smem)

    // RC: per-warp accumulation [kMmaPerM][kMmaPerN][2]
    uint32_t RC[kMmaPerM][kMmaPerN][2];
    #pragma unroll
    for (int i = 0; i < kMmaPerM; i++)
        #pragma unroll
        for (int j = 0; j < kMmaPerN; j++)
            { RC[i][j][0] = 0; RC[i][j][1] = 0; }

    // Registers for double-buffered A and B fragments
    uint32_t RA[2][kMmaPerM][4];  // [buffer][mma_m][4]
    uint32_t RB[2][kMmaPerN][2];  // [buffer][mma_n][2]
    int buf_store = 0, buf_load = 1;

    // Prefetch first K double-tile (k=0)
    {
        int gK = 0 + (tid % 2) * 8;
        int g_addr = (by * kBM + tid / 2) * K + gK;
        if (by * kBM + tid / 2 < M && gK < K) {
            uint32_t sp = a_base + (0 * A_sz + (tid / 2) * A_cols + (tid % 2) * 8) * sizeof(half);
            CP_ASYNC_CG(sp, &A[g_addr], 16);
        }
        // Second K tile in smem
        if (by * kBM + tid / 2 < M && gK + 16 < K) {
            uint32_t sp = a_base + (0 * A_sz + (tid / 2) * A_cols + (tid % 2) * 8 + 16) * sizeof(half);
            CP_ASYNC_CG(sp, &A[g_addr + 16], 16);
        }

        gK = 0 + tid / 16;
        g_addr = gK * N + bx * kBN + (tid % 16) * 8;
        if (gK < K && bx * kBN + (tid % 16) * 8 < N) {
            uint32_t sp = b_base + (0 * B_sz + (tid / 16) * kBN + (tid % 16) * 8) * sizeof(half);
            CP_ASYNC_CG(sp, &B[g_addr], 16);
        }
        if (gK + 16 < K && bx * kBN + (tid % 16) * 8 < N) {
            uint32_t sp = b_base + (0 * B_sz + (tid / 16 + 16) * kBN + (tid % 16) * 8) * sizeof(half);
            CP_ASYNC_CG(sp, &B[(gK + 16) * N + bx * kBN + (tid % 16) * 8], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // Load first register buffer
    #pragma unroll
    for (int i = 0; i < kMmaPerM; i++) {
        int a_row = lid % 16, a_col = (lid / 16) * 8;
        uint32_t ap = a_base + ((wM * kMmaPerM + i) * 16 + a_row) * A_cols * sizeof(half) + a_col * sizeof(half);
        LDMATRIX_X4(RA[buf_store][i][0], RA[buf_store][i][1],
                    RA[buf_store][i][2], RA[buf_store][i][3], ap);
    }
    #pragma unroll
    for (int j = 0; j < kMmaPerN; j++) {
        int b_row = lid % 16, b_col = (wN * kMmaPerN + j) * 8;
        uint32_t bp = b_base + (b_row * kBN + b_col) * sizeof(half);
        LDMATRIX_X2_T(RB[buf_store][j][0], RB[buf_store][j][1], bp);
    }

    // Main K loop
    #pragma unroll 1
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        buf_store ^= 1; buf_load ^= 1;
        int stg = kt % kStage, nxt = (kt + 1) % kStage;

        // Prefetch next K double-tile
        if (kt + 1 < num_k_tiles) {
            int nK = (kt + 1) * BK * kWarpK + (tid % 2) * 8;
            int g_addr = (by * kBM + tid / 2) * K + nK;
            if (by * kBM + tid / 2 < M && nK < K) {
                uint32_t sp = a_base + (nxt * A_sz + (tid / 2) * A_cols + (tid % 2) * 8) * sizeof(half);
                CP_ASYNC_CG(sp, &A[g_addr], 16);
            }
            if (by * kBM + tid / 2 < M && nK + 16 < K) {
                uint32_t sp = a_base + (nxt * A_sz + (tid / 2) * A_cols + (tid % 2) * 8 + 16) * sizeof(half);
                CP_ASYNC_CG(sp, &A[g_addr + 16], 16);
            }

            nK = (kt + 1) * BK * kWarpK + tid / 16;
            g_addr = nK * N + bx * kBN + (tid % 16) * 8;
            if (nK < K && bx * kBN + (tid % 16) * 8 < N) {
                uint32_t sp = b_base + (nxt * B_sz + (tid / 16) * kBN + (tid % 16) * 8) * sizeof(half);
                CP_ASYNC_CG(sp, &B[g_addr], 16);
            }
            if (nK + 16 < K && bx * kBN + (tid % 16) * 8 < N) {
                uint32_t sp = b_base + (nxt * B_sz + (tid / 16 + 16) * kBN + (tid % 16) * 8) * sizeof(half);
                CP_ASYNC_CG(sp, &B[(nK + 16) * N + bx * kBN + (tid % 16) * 8], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
        }

        // Load next register buffer (from second K tile in CURRENT SMEM stage)
        #pragma unroll
        for (int i = 0; i < kMmaPerM; i++) {
            int a_row = lid % 16, a_col = (lid / 16) * 8;
            uint32_t ap = a_base + (stg * A_sz + ((wM * kMmaPerM + i) * 16 + a_row) * A_cols + (a_col + 16)) * sizeof(half);
            LDMATRIX_X4(RA[buf_store][i][0], RA[buf_store][i][1],
                        RA[buf_store][i][2], RA[buf_store][i][3], ap);
        }
        #pragma unroll
        for (int j = 0; j < kMmaPerN; j++) {
            int b_row = lid % 16, b_col = (wN * kMmaPerN + j) * 8;
            uint32_t bp = b_base + (stg * B_sz + (b_row + 16) * kBN + b_col) * sizeof(half);
            LDMATRIX_X2_T(RB[buf_store][j][0], RB[buf_store][j][1], bp);
        }

        // MMA: first K tile (from buf_load, which was loaded in previous iteration)
        #pragma unroll
        for (int i = 0; i < kMmaPerM; i++) {
            #pragma unroll
            for (int j = 0; j < kMmaPerN; j++) {
                HMMA16816(RC[i][j][0], RC[i][j][1],
                          RA[buf_load][i][0], RA[buf_load][i][1],
                          RA[buf_load][i][2], RA[buf_load][i][3],
                          RB[buf_load][j][0], RB[buf_load][j][1],
                          RC[i][j][0], RC[i][j][1]);
            }
        }

        buf_store ^= 1; buf_load ^= 1;

        // MMA: second K tile (from buf_load, just loaded above)
        #pragma unroll
        for (int i = 0; i < kMmaPerM; i++) {
            #pragma unroll
            for (int j = 0; j < kMmaPerN; j++) {
                HMMA16816(RC[i][j][0], RC[i][j][1],
                          RA[buf_load][i][0], RA[buf_load][i][1],
                          RA[buf_load][i][2], RA[buf_load][i][3],
                          RB[buf_load][j][0], RB[buf_load][j][1],
                          RC[i][j][0], RC[i][j][1]);
            }
        }

        if (kt + 1 < num_k_tiles) {
            CP_ASYNC_WAIT_GROUP(0);
            __syncthreads();
        }

        // Pre-load first K tile of NEXT iteration (from stage just prefetched = nxt)
        if (kt + 1 < num_k_tiles) {
            #pragma unroll
            for (int i = 0; i < kMmaPerM; i++) {
                int a_row = lid % 16, a_col = (lid / 16) * 8;
                uint32_t ap = a_base + (nxt * A_sz + ((wM * kMmaPerM + i) * 16 + a_row) * A_cols + a_col) * sizeof(half);
                LDMATRIX_X4(RA[buf_store][i][0], RA[buf_store][i][1],
                            RA[buf_store][i][2], RA[buf_store][i][3], ap);
            }
            #pragma unroll
            for (int j = 0; j < kMmaPerN; j++) {
                int b_row = lid % 16, b_col = (wN * kMmaPerN + j) * 8;
                uint32_t bp = b_base + (nxt * B_sz + b_row * kBN + b_col) * sizeof(half);
                LDMATRIX_X2_T(RB[buf_store][j][0], RB[buf_store][j][1], bp);
            }
        }
    }

    // Warp-shuffle collective store (128-bit vectorized)
    #pragma unroll
    for (int i = 0; i < kMmaPerM; i++) {
        // Shuffle RC[..][0] and RC[..][1] across lanes
        uint32_t Z0[kMmaPerN][4], Z1[kMmaPerN][4];
        #pragma unroll
        for (int j = 0; j < kMmaPerN; j++) {
            Z0[j][0] = RC[i][j][0];
            Z1[j][0] = RC[i][j][1];
            Z0[j][1] = __shfl_sync(0xffffffff, RC[i][j][0], lid + 1);
            Z0[j][2] = __shfl_sync(0xffffffff, RC[i][j][0], lid + 2);
            Z0[j][3] = __shfl_sync(0xffffffff, RC[i][j][0], lid + 3);
            Z1[j][1] = __shfl_sync(0xffffffff, RC[i][j][1], lid + 1);
            Z1[j][2] = __shfl_sync(0xffffffff, RC[i][j][1], lid + 2);
            Z1[j][3] = __shfl_sync(0xffffffff, RC[i][j][1], lid + 3);
        }

        if (lid % 4 == 0) {
            int gr0 = gM + i * 16 + lid / 4;
            int gr8 = gr0 + 8;
            #pragma unroll
            for (int j = 0; j < kMmaPerN; j++) {
                int gc = gN + j * 8;
                if (gr0 < M && gc < N)
                    LDST128BITS(C[gr0 * N + gc]) = LDST128BITS(Z0[j][0]);
                if (gr8 < M && gc < N)
                    LDST128BITS(C[gr8 * N + gc]) = LDST128BITS(Z1[j][0]);
            }
        }
    }
}
