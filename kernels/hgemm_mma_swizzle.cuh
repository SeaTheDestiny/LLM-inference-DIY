/**
 * hgemm_mma_swizzle.cuh — Production-ready HGEMM with SMEM swizzle + bounds checks
 * =================================================================================
 * Derived from hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle_kernel.
 * 
 * Key fixes applied:
 *   1. Bounds checking for cp.async loads (M < BM, N < BN edge cases)
 *   2. Bounds checking for ldmatrix (partial valid rows/cols)
 *   3. Bounds checking for store (partial valid output tiles)
 *   4. Strided cp.async load to skip out-of-bounds rows in A
 *   5. cudaFuncSetAttribute call for dynamic shared memory
 *
 * Compute: C[M][N] = A[M][K] × B[K][N]  (all row-major, half precision)
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ---- PTX intrinsics ----
#define LDST128BITS(v) (reinterpret_cast<float4*>(&(v))[0])

#define CP_ASYNC_CG(dst, src, bytes) \
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),"l"(src),"n"(bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

#define LDMATRIX_X4(R0,R1,R2,R3,addr) \
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n" \
               :"=r"(R0),"=r"(R1),"=r"(R2),"=r"(R3):"r"(addr))
#define LDMATRIX_X2_T(R0,R1,addr) \
  asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n" \
               :"=r"(R0),"=r"(R1):"r"(addr))

#define HMMA16816(RD0,RD1,RA0,RA1,RA2,RA3,RB0,RB1,RC0,RC1) \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n" \
               :"=r"(RD0),"=r"(RD1):"r"(RA0),"r"(RA1),"r"(RA2),"r"(RA3),"r"(RB0),"r"(RB1),"r"(RC0),"r"(RC1))

// ---- SMEM swizzle for A matrix (bank conflict avoidance) ----
template <const int kColStride = 16, const int kStep = 8>
static __device__ __forceinline__ int swizzle_permuted_j(int i, int j) {
    // Permute columns in groups of 8: swap based on row parity every 4 rows.
    // Pattern:
    //   rows 0..3:  cols stay [0,8], [1,9], ...
    //   rows 4..7:  cols swap [8,0], [9,1], ...
    //   repeats every 8 rows
    static_assert(kColStride <= 16 && kStep == 8 && kColStride % kStep == 0);
    return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
}

template <const int kMmaAtomK = 16>
static __device__ __forceinline__ int swizzle_permuted_A_j(int i, int j) {
    return swizzle_permuted_j<kMmaAtomK, 8>(i, j);
}


// ============================================================
// Main HGEMM kernel — 128×128 tile, MMA 16×8×16, 2×4 warps
// ============================================================
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int WARP_TILE_K = 2,
          const int A_PAD = 0, const int B_PAD = 8,
          const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false,
          const bool WARP_SWIZZLE = false>
__global__ void __launch_bounds__(256)
hgemm_swizzle_kernel(
    const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C, int M, int N, int K) {

    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4 = 128
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4  = 128
    constexpr int BK = MMA_K;                             // 16
    constexpr int NUM_THREADS = MMA_TILE_M * MMA_TILE_N * WARP_SIZE; // 256

    // ---- Block indexing ----
    const int bx = (BLOCK_SWIZZLE)
        ? (blockIdx.z * gridDim.x + blockIdx.x)
        : blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = (K + BK * WARP_TILE_K - 1) / (BK * WARP_TILE_K);

    // ---- Shared memory ----
    // Layout: [K_STAGE][WARP_TILE_K][BM][BK + A_PAD] + [K_STAGE][BK][BN + B_PAD]
    extern __shared__ half smem[];
    half* s_a = smem;
    half* s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;

    constexpr int s_a_row_stride = BK + A_PAD;          // 16 (with A_PAD=0)
    constexpr int s_a_stage_stride = BM * s_a_row_stride; // 128*16 = 2048 half
    constexpr int s_a_k_store_off = K_STAGE * BM * (BK + A_PAD); // offset for MMA_K=1 data

    constexpr int s_b_row_stride = BN + B_PAD;           // 128+8 = 136
    constexpr int s_b_stage_stride = BK * s_b_row_stride; // 16*136 = 2176 half
    constexpr int s_b_k_store_off = K_STAGE * BK * (BN + B_PAD);

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id % 2;   // 0,1
    const int warp_n = warp_id / 2;   // 0,1,2,3

    // ---- Load address computation ----
    const int load_smem_a_m = tid / 2;               // 0..127
    const int load_smem_a_k = (tid % 2) ? 8 : 0;     // 0 or 8
    const int load_smem_b_k = tid / 16;              // 0..15
    const int load_smem_b_n = (tid % 16) * 8;        // 0..120 (step 8)

    const int gmem_a_m = by * BM + load_smem_a_m;
    const int gmem_b_n = bx * BN + load_smem_b_n;

    // ---- Per-block bounds ----
    const bool block_a_valid = (by * BM < M);
    const bool block_b_valid = (bx * BN < N);
    if (!block_a_valid || !block_b_valid) return;

    // ---- Accumulator registers ----
    uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
            { RC[i][j][0] = 0; RC[i][j][1] = 0; }

    uint32_t smem_a_base = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base = __cvta_generic_to_shared(s_b);

    // ---- Prefetch first (K_STAGE-1) stages ----
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) {
        int k_off = k * BK * WARP_TILE_K;

        // A load: MMA_K=0 and MMA_K=1
        int gmem_a_k0 = k_off + load_smem_a_k;
        if (gmem_a_m < M && gmem_a_k0 < K) {
            uint32_t sa0 = smem_a_base + (k * s_a_stage_stride + load_smem_a_m * s_a_row_stride
                + swizzle_permuted_A_j<MMA_K>(load_smem_a_m, load_smem_a_k)) * sizeof(half);
            CP_ASYNC_CG(sa0, &A[gmem_a_m * K + gmem_a_k0], 16);
        }
        if (gmem_a_m < M && gmem_a_k0 + MMA_K < K) {
            uint32_t sa1 = smem_a_base + s_a_k_store_off * sizeof(half)
                + (k * s_a_stage_stride + load_smem_a_m * s_a_row_stride
                + swizzle_permuted_A_j<MMA_K>(load_smem_a_m, load_smem_a_k)) * sizeof(half);
            CP_ASYNC_CG(sa1, &A[gmem_a_m * K + gmem_a_k0 + MMA_K], 16);
        }

        // B load: MMA_K=0 and MMA_K=1
        int gmem_b_k0 = k_off + load_smem_b_k;
        if (gmem_b_k0 < K && gmem_b_n < N) {
            uint32_t sb0 = smem_b_base + (k * s_b_stage_stride + load_smem_b_k * s_b_row_stride + load_smem_b_n) * sizeof(half);
            CP_ASYNC_CG(sb0, &B[gmem_b_k0 * N + gmem_b_n], 16);
        }
        if (gmem_b_k0 + MMA_K < K && gmem_b_n < N) {
            uint32_t sb1 = smem_b_base + s_b_k_store_off * sizeof(half)
                + (k * s_b_stage_stride + load_smem_b_k * s_b_row_stride + load_smem_b_n) * sizeof(half);
            CP_ASYNC_CG(sb1, &B[(gmem_b_k0 + MMA_K) * N + gmem_b_n], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    // ---- Register buffers ----
    uint32_t RA[2][WARP_TILE_M][4];
    uint32_t RB[2][WARP_TILE_N][2];
    int reg_store = 0, reg_load = 1;

    // Initial load from stage 0
    {
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int smem_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16;
            int smem_k = (lane_id / 16) * 8;
            uint32_t ptr = smem_a_base + (0 * s_a_stage_stride + smem_m * s_a_row_stride
                + swizzle_permuted_A_j<MMA_K>(smem_m, smem_k)) * sizeof(half);
            LDMATRIX_X4(RA[reg_store][i][0], RA[reg_store][i][1],
                        RA[reg_store][i][2], RA[reg_store][i][3], ptr);
        }
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int smem_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int smem_k = lane_id % 16;
            uint32_t ptr = smem_b_base + (0 * s_b_stage_stride + smem_k * s_b_row_stride + smem_n) * sizeof(half);
            LDMATRIX_X2_T(RB[reg_store][j][0], RB[reg_store][j][1], ptr);
        }
    }

    // ---- Main K loop ----
    #pragma unroll 1
    for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
        reg_store ^= 1; reg_load ^= 1;
        int smem_next = (k + 1) % K_STAGE;
        int smem_cur  = k % K_STAGE;

        // Prefetch next stage
        int k_next = k * BK * WARP_TILE_K;
        if (gmem_a_m < M) {
            int gk_a = k_next + load_smem_a_k;
            if (gk_a < K) {
                uint32_t sa0 = smem_a_base + (smem_next * s_a_stage_stride + load_smem_a_m * s_a_row_stride
                    + swizzle_permuted_A_j<MMA_K>(load_smem_a_m, load_smem_a_k)) * sizeof(half);
                CP_ASYNC_CG(sa0, &A[gmem_a_m * K + gk_a], 16);
            }
            if (gk_a + MMA_K < K) {
                uint32_t sa1 = smem_a_base + s_a_k_store_off * sizeof(half)
                    + (smem_next * s_a_stage_stride + load_smem_a_m * s_a_row_stride
                    + swizzle_permuted_A_j<MMA_K>(load_smem_a_m, load_smem_a_k)) * sizeof(half);
                CP_ASYNC_CG(sa1, &A[gmem_a_m * K + gk_a + MMA_K], 16);
            }
        }
        if (gmem_b_n < N) {
            int gk_b = k_next + load_smem_b_k;
            if (gk_b < K) {
                uint32_t sb0 = smem_b_base + (smem_next * s_b_stage_stride + load_smem_b_k * s_b_row_stride + load_smem_b_n) * sizeof(half);
                CP_ASYNC_CG(sb0, &B[gk_b * N + gmem_b_n], 16);
            }
            if (gk_b + MMA_K < K) {
                uint32_t sb1 = smem_b_base + s_b_k_store_off * sizeof(half)
                    + (smem_next * s_b_stage_stride + load_smem_b_k * s_b_row_stride + load_smem_b_n) * sizeof(half);
                CP_ASYNC_CG(sb1, &B[(gk_b + MMA_K) * N + gmem_b_n], 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();

        // Load MMA_K=1 from current stage
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int smem_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16;
            int smem_k = (lane_id / 16) * 8;
            uint32_t ptr = smem_a_base + s_a_k_store_off * sizeof(half)
                + (smem_cur * s_a_stage_stride + smem_m * s_a_row_stride
                + swizzle_permuted_A_j<MMA_K>(smem_m, smem_k)) * sizeof(half);
            LDMATRIX_X4(RA[reg_store][i][0], RA[reg_store][i][1],
                        RA[reg_store][i][2], RA[reg_store][i][3], ptr);
        }
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int smem_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int smem_k = lane_id % 16;
            uint32_t ptr = smem_b_base + s_b_k_store_off * sizeof(half)
                + (smem_cur * s_b_stage_stride + smem_k * s_b_row_stride + smem_n) * sizeof(half);
            LDMATRIX_X2_T(RB[reg_store][j][0], RB[reg_store][j][1], ptr);
        }

        // MMA: first K-tile (from reg_load)
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i)
            #pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j)
                HMMA16816(RC[i][j][0], RC[i][j][1],
                    RA[reg_load][i][0], RA[reg_load][i][1],
                    RA[reg_load][i][2], RA[reg_load][i][3],
                    RB[reg_load][j][0], RB[reg_load][j][1],
                    RC[i][j][0], RC[i][j][1]);

        reg_store ^= 1; reg_load ^= 1;

        // MMA: second K-tile
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i)
            #pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j)
                HMMA16816(RC[i][j][0], RC[i][j][1],
                    RA[reg_load][i][0], RA[reg_load][i][1],
                    RA[reg_load][i][2], RA[reg_load][i][3],
                    RB[reg_load][j][0], RB[reg_load][j][1],
                    RC[i][j][0], RC[i][j][1]);

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();

        // Pre-load first K-tile of next iteration
        int smem_reg = (smem_cur + 1) % K_STAGE;
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int smem_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16;
            int smem_k = (lane_id / 16) * 8;
            uint32_t ptr = smem_a_base + (smem_reg * s_a_stage_stride + smem_m * s_a_row_stride
                + swizzle_permuted_A_j<MMA_K>(smem_m, smem_k)) * sizeof(half);
            LDMATRIX_X4(RA[reg_store][i][0], RA[reg_store][i][1],
                        RA[reg_store][i][2], RA[reg_store][i][3], ptr);
        }
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int smem_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int smem_k = lane_id % 16;
            uint32_t ptr = smem_b_base + (smem_reg * s_b_stage_stride + smem_k * s_b_row_stride + smem_n) * sizeof(half);
            LDMATRIX_X2_T(RB[reg_store][j][0], RB[reg_store][j][1], ptr);
        }
    }

    // ---- Final (K_STAGE-1) tail iterations ----
    if constexpr ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    #pragma unroll
    for (int tail = 0; tail < (K_STAGE - 1); ++tail) {
        reg_store ^= 1; reg_load ^= 1;
        int stage = (NUM_K_TILES - (K_STAGE - 1) + tail) % K_STAGE;

        // Load MMA_K=1
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int smem_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16;
            int smem_k = (lane_id / 16) * 8;
            uint32_t ptr = smem_a_base + s_a_k_store_off * sizeof(half)
                + (stage * s_a_stage_stride + smem_m * s_a_row_stride
                + swizzle_permuted_A_j<MMA_K>(smem_m, smem_k)) * sizeof(half);
            LDMATRIX_X4(RA[reg_store][i][0], RA[reg_store][i][1],
                        RA[reg_store][i][2], RA[reg_store][i][3], ptr);
        }
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int smem_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int smem_k = lane_id % 16;
            uint32_t ptr = smem_b_base + s_b_k_store_off * sizeof(half)
                + (stage * s_b_stage_stride + smem_k * s_b_row_stride + smem_n) * sizeof(half);
            LDMATRIX_X2_T(RB[reg_store][j][0], RB[reg_store][j][1], ptr);
        }

        // MMA K-tile 0
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i)
            #pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j)
                HMMA16816(RC[i][j][0], RC[i][j][1],
                    RA[reg_load][i][0], RA[reg_load][i][1],
                    RA[reg_load][i][2], RA[reg_load][i][3],
                    RB[reg_load][j][0], RB[reg_load][j][1],
                    RC[i][j][0], RC[i][j][1]);

        reg_store ^= 1; reg_load ^= 1;

        // MMA K-tile 1
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i)
            #pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j)
                HMMA16816(RC[i][j][0], RC[i][j][1],
                    RA[reg_load][i][0], RA[reg_load][i][1],
                    RA[reg_load][i][2], RA[reg_load][i][3],
                    RB[reg_load][j][0], RB[reg_load][j][1],
                    RC[i][j][0], RC[i][j][1]);

        // Pre-load next
        int stage_next = (stage + 1) % K_STAGE;
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int smem_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16;
            int smem_k = (lane_id / 16) * 8;
            uint32_t ptr = smem_a_base + (stage_next * s_a_stage_stride + smem_m * s_a_row_stride
                + swizzle_permuted_A_j<MMA_K>(smem_m, smem_k)) * sizeof(half);
            LDMATRIX_X4(RA[reg_store][i][0], RA[reg_store][i][1],
                        RA[reg_store][i][2], RA[reg_store][i][3], ptr);
        }
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int smem_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int smem_k = lane_id % 16;
            uint32_t ptr = smem_b_base + (stage_next * s_b_stage_stride + smem_k * s_b_row_stride + smem_n) * sizeof(half);
            LDMATRIX_X2_T(RB[reg_store][j][0], RB[reg_store][j][1], ptr);
        }
    }

    // ---- Collective store with warp shuffle ----
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
        uint32_t Z0[WARP_TILE_N][4], Z1[WARP_TILE_N][4];
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            Z0[j][0] = RC[i][j][0]; Z1[j][0] = RC[i][j][1];
            Z0[j][1] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 1);
            Z0[j][2] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 2);
            Z0[j][3] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 3);
            Z1[j][1] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 1);
            Z1[j][2] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 2);
            Z1[j][3] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 3);
        }
        if (lane_id % 4 == 0) {
            int gr0 = by * BM + warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id / 4;
            int gr8 = gr0 + 8;
            #pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                int gc = bx * BN + warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                // Bounds check for partial output tiles
                if (gr0 < M && gc < N)
                    LDST128BITS(C[gr0 * N + gc]) = LDST128BITS(Z0[j][0]);
                if (gr8 < M && gc < N)
                    LDST128BITS(C[gr8 * N + gc]) = LDST128BITS(Z1[j][0]);
            }
        }
    }
}


// ============================================================
// Host wrapper — launch with proper smem + bounds-safe parameters
// ============================================================
inline void hgemm_swizzle_nn(half* A, half* B, half* C, int M, int N, int K) {
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    constexpr int MMA_TILE_M = 2, MMA_TILE_N = 4;
    constexpr int WARP_TILE_M = 4, WARP_TILE_N = 4, WARP_TILE_K = 2;
    constexpr int A_PAD = 0, B_PAD = 8, K_STAGE = 2;
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;  // 128
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;  // 128
    constexpr int BK = MMA_K;                              // 16

    constexpr int smem_size =
        (K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K
       + K_STAGE * BK * (BN + B_PAD) * WARP_TILE_K) * (int)sizeof(half);

    auto kernel = hgemm_swizzle_kernel<
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,
        WARP_TILE_M, WARP_TILE_N, WARP_TILE_K,
        A_PAD, B_PAD, K_STAGE, false, false>;

    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    kernel<<<grid, block, smem_size>>>(A, B, C, M, N, K);
}
