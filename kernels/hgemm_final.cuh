/**
 * hgemm_final.cuh — Optimized HGEMM with SMEM swizzle + multi-stage + block swizzle
 * ================================================================================
 * Based on flash_attn_mma_tiling reference techniques.
 *
 * Features (toggle via template params):
 *   kSwizzleA  — SMEM bank conflict avoidance for A matrix
 *   kStage     — software pipeline depth (2/3/4)
 *   kBlockSwz  — block swizzle stride for L2 cache (0=off)
 *
 * Tile: BM=128, BN=128, 2×4 warp grid, K32 reg double-buffer
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define LDST128BITS(v) (reinterpret_cast<float4*>(&(v))[0])
#define LDST32BITS(v)  (reinterpret_cast<half2*>(&(v))[0])

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


// ============================================================
// Final HGEMM: all optimizations
// ============================================================
template <
    const int kBM = 128, const int kBN = 128,
    const int kWarpsM = 2, const int kWarpsN = 4,
    const int kMmaPerM = 4, const int kMmaPerN = 4,
    const int kWarpK = 2,
    const bool kSwizzleA = false,
    const int kStage = 2,
    const int kBlockSwz = 0>      // 0=off, N=swizzle stride (e.g. 2048)
__global__ void __launch_bounds__(WARP_SIZE * kWarpsM * kWarpsN)
    hgemm_opt_kernel(half *A, half *B, half *C, int M, int N, int K) {

    constexpr int BK = 16;
    constexpr int A_cols = BK * kWarpK;     // 32
    constexpr int A_sz = kBM * A_cols;       // 128×32
    constexpr int B_sz = BK * kWarpK * kBN;  // 32×128

    // Block swizzle: reorder blockIdx.x using z-dimension
    const int bx = (kBlockSwz > 0)
        ? (blockIdx.z * gridDim.x + blockIdx.x)
        : blockIdx.x;
    const int by = blockIdx.y;
    const int num_kt = (K + BK * kWarpK - 1) / (BK * kWarpK);

    int tid = threadIdx.x;
    int lid = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int wM  = wid % kWarpsM;
    int wN  = wid / kWarpsM;
    int gM  = by * kBM + wM * (16 * kMmaPerM);
    int gN  = bx * kBN + wN * (8 * kMmaPerN);
    if (gM >= M || gN >= N) return;

    // ---- SMEM ----
    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + kStage * A_sz;
    uint32_t a_base = __cvta_generic_to_shared(s_a);
    uint32_t b_base = __cvta_generic_to_shared(s_b);

    // ---- SMEM Swizzle Helper ----
    // Permutes A columns within each 16-col K-tile to avoid bank conflicts.
    // Row i, col j → col' = base + swap_8col_groups XOR (row/4)
    auto swz = [](int row, int col) {
        if constexpr (kSwizzleA) {
            int lo = col & 7;                        // low 3 bits preserved
            int grp = (col >> 3) & 1;                 // which 8-col group (0 or 1)
            grp ^= (row >> 2) & 1;                    // XOR with row/4 parity
            return (col & ~15) | (grp << 3) | lo;     // reassemble
        }
        return col;
    };

    // ---- Registers ----
    uint32_t RC[kMmaPerM][kMmaPerN][2];
    #pragma unroll
    for (int i = 0; i < kMmaPerM; i++)
        #pragma unroll
        for (int j = 0; j < kMmaPerN; j++)
            { RC[i][j][0] = 0; RC[i][j][1] = 0; }

    uint32_t RA[2][kMmaPerM][4], RB[2][kMmaPerN][2];
    int bs = 0, bl = 1;  // buf_store, buf_load

    // ---- A/B load helpers ----
    auto load_A = [&](int smem_stage, int k_offset) {
        int row = tid / 2, col = (tid % 2) * 8;
        int gK = k_offset + col;
        uint32_t sp = a_base + (smem_stage * A_sz + row * A_cols + swz(row, col)) * sizeof(half);
        if (by * kBM + row < M && gK < K)
            CP_ASYNC_CG(sp, &A[(by * kBM + row) * K + gK], 16);
        // Second K-tile (col + 16)
        if (gK + 16 < K) {
            uint32_t sp2 = a_base + (smem_stage * A_sz + row * A_cols + swz(row, col + 16)) * sizeof(half);
            CP_ASYNC_CG(sp2, &A[(by * kBM + row) * K + gK + 16], 16);
        }
    };
    auto load_B = [&](int smem_stage, int k_offset) {
        int row0 = tid / 16, col = (tid % 16) * 8;
        int gK0 = k_offset + row0;
        uint32_t sp0 = b_base + (smem_stage * B_sz + row0 * kBN + col) * sizeof(half);
        if (gK0 < K && bx * kBN + col < N)
            CP_ASYNC_CG(sp0, &B[gK0 * N + bx * kBN + col], 16);
        int gK1 = k_offset + 16 + row0;
        if (gK1 < K) {
            uint32_t sp1 = b_base + (smem_stage * B_sz + (row0 + 16) * kBN + col) * sizeof(half);
            CP_ASYNC_CG(sp1, &B[gK1 * N + bx * kBN + col], 16);
        }
    };
    auto ldm_A = [&](int smem_stage, int buf, int mma_i) {
        int row = (wM * kMmaPerM + mma_i) * 16 + lid % 16;
        int col = (lid / 16) * 8;
        uint32_t ap = a_base + (smem_stage * A_sz + row * A_cols + swz(row, col)) * sizeof(half);
        LDMATRIX_X4(RA[buf][mma_i][0], RA[buf][mma_i][1],
                    RA[buf][mma_i][2], RA[buf][mma_i][3], ap);
    };
    auto ldm_A_k1 = [&](int smem_stage, int buf, int mma_i) {
        int row = (wM * kMmaPerM + mma_i) * 16 + lid % 16;
        int col = (lid / 16) * 8 + 16;
        uint32_t ap = a_base + (smem_stage * A_sz + row * A_cols + swz(row, col)) * sizeof(half);
        LDMATRIX_X4(RA[buf][mma_i][0], RA[buf][mma_i][1],
                    RA[buf][mma_i][2], RA[buf][mma_i][3], ap);
    };
    auto ldm_B = [&](int smem_stage, int buf, int mma_j, int ktile_ofs = 0) {
        int row = lid % 16 + ktile_ofs;
        int col = (wN * kMmaPerN + mma_j) * 8;
        uint32_t bp = b_base + (smem_stage * B_sz + row * kBN + col) * sizeof(half);
        LDMATRIX_X2_T(RB[buf][mma_j][0], RB[buf][mma_j][1], bp);
    };
    auto mma = [&](int a_buf, int b_buf) {
        #pragma unroll
        for (int i = 0; i < kMmaPerM; i++)
            #pragma unroll
            for (int j = 0; j < kMmaPerN; j++)
                HMMA16816(RC[i][j][0], RC[i][j][1],
                          RA[a_buf][i][0], RA[a_buf][i][1],
                          RA[a_buf][i][2], RA[a_buf][i][3],
                          RB[b_buf][j][0], RB[b_buf][j][1],
                          RC[i][j][0], RC[i][j][1]);
    };

    // ---- Prefetch first (kStage-1) stages ----
    #pragma unroll
    for (int s = 0; s < kStage - 1; s++) {
        load_A(s, s * BK * kWarpK);
        load_B(s, s * BK * kWarpK);
        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(kStage - 2);
    __syncthreads();

    // ---- Load first register buffer from stage 0 ----
    #pragma unroll
    for (int i = 0; i < kMmaPerM; i++) ldm_A(0, bs, i);
    #pragma unroll
    for (int j = 0; j < kMmaPerN; j++) ldm_B(0, bs, j);

    // ---- Main K loop ----
    #pragma unroll 1
    for (int kt = kStage - 1; kt < num_kt; kt++) {
        bs ^= 1; bl ^= 1;
        int stg = kt % kStage;
        int nxt = (kt + 1) % kStage;

        // Prefetch next stage
        if (kt + 1 < num_kt) {
            load_A(nxt, (kt + 1) * BK * kWarpK);
            load_B(nxt, (kt + 1) * BK * kWarpK);
            CP_ASYNC_COMMIT_GROUP();
        }

        // Load K-tile 1 into buf_store
        #pragma unroll
        for (int i = 0; i < kMmaPerM; i++) ldm_A_k1(stg, bs, i);
        #pragma unroll
        for (int j = 0; j < kMmaPerN; j++) ldm_B(stg, bs, j, 16);

        // MMA: K-tile 0 (from buf_load)
        mma(bl, bl);

        bs ^= 1; bl ^= 1;

        // MMA: K-tile 1 (from buf_load, just loaded)
        mma(bl, bl);

        if (kt + 1 < num_kt) {
            CP_ASYNC_WAIT_GROUP(kStage - 2);
            __syncthreads();
        }

        // Pre-load K-tile 0 of next iteration
        if (kt + 1 < num_kt) {
            #pragma unroll
            for (int i = 0; i < kMmaPerM; i++) ldm_A(nxt, bs, i);
            #pragma unroll
            for (int j = 0; j < kMmaPerN; j++) ldm_B(nxt, bs, j);
        }
    }

    // ---- Process last (kStage-1) K-tiles (if any) ----
    #pragma unroll
    for (int tail = 0; tail < kStage - 1; tail++) {
        int stg = (num_kt - kStage + 1 + tail) % kStage;
        if (num_kt <= kStage - 1 && tail < num_kt) stg = tail;
        if (tail >= num_kt) break;

        bs ^= 1; bl ^= 1;

        #pragma unroll
        for (int i = 0; i < kMmaPerM; i++) ldm_A_k1(stg, bs, i);
        #pragma unroll
        for (int j = 0; j < kMmaPerN; j++) ldm_B(stg, bs, j, 16);

        mma(bl, bl);
        bs ^= 1; bl ^= 1;
        mma(bl, bl);
    }

    // ---- Warp-shuffle collective store ----
    #pragma unroll
    for (int i = 0; i < kMmaPerM; i++) {
        uint32_t Z0[kMmaPerN][4], Z1[kMmaPerN][4];
        #pragma unroll
        for (int j = 0; j < kMmaPerN; j++) {
            Z0[j][0] = RC[i][j][0]; Z1[j][0] = RC[i][j][1];
            Z0[j][1] = __shfl_sync(0xffffffff, RC[i][j][0], lid + 1);
            Z0[j][2] = __shfl_sync(0xffffffff, RC[i][j][0], lid + 2);
            Z0[j][3] = __shfl_sync(0xffffffff, RC[i][j][0], lid + 3);
            Z1[j][1] = __shfl_sync(0xffffffff, RC[i][j][1], lid + 1);
            Z1[j][2] = __shfl_sync(0xffffffff, RC[i][j][1], lid + 2);
            Z1[j][3] = __shfl_sync(0xffffffff, RC[i][j][1], lid + 3);
        }
        if (lid % 4 == 0) {
            int gr0 = gM + i * 16 + lid / 4, gr8 = gr0 + 8;
            #pragma unroll
            for (int j = 0; j < kMmaPerN; j++) {
                int gc = gN + j * 8;
                if (gr0 < M && gc < N) LDST128BITS(C[gr0 * N + gc]) = LDST128BITS(Z0[j][0]);
                if (gr8 < M && gc < N) LDST128BITS(C[gr8 * N + gc]) = LDST128BITS(Z1[j][0]);
            }
        }
    }
}
