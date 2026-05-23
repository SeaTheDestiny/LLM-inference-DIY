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
