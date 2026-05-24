/**
 * flash_decode_q1_opt.cuh — Optimized Flash Decoding for Q=1
 * ===========================================================
 * 4-step optimization (mirrors FA evolution):
 *   v1: Vectorized coalesced loads (fix 4x bandwidth waste)
 *   v2: Large chunk Bc=128 (reduce Stage2 overhead)
 *   v3: Multi-warp (4 warps, 128 threads) + in-block merge
 *   v4: 8 warps + __expf/__fmaf_rn + optimized Stage2
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ============================================================
// Helper: vectorized load/store for half arrays
// ============================================================
#define FD_LDST64(dst, src) \
    *reinterpret_cast<float2*>(&(dst)) = *reinterpret_cast<const float2*>(&(src))
#define FD_ST64(dst, src) \
    *reinterpret_cast<float2*>(&(dst)) = *reinterpret_cast<const float2*>(&(src))

// ============================================================
// v1: Vectorized Coalesced Loads (Bc=16, 1 warp)
// ============================================================
// Fix: original uses scalar loads with stride-4 pattern → 25% BW util
// Now: float2 vectorized load → 100% BW util (4x improvement)

template<int kHeadDim>
__global__ void fd_v1_stage1_kernel(
    const half *Q, const half *K, const half *V,
    half *O_partial, float *LSE,
    int KV_seqlen, int num_chunks) {

    constexpr int kBc = 16;
    constexpr int kDpt = kHeadDim / WARP_SIZE;  // d per thread = 4 for d=128
    int chunk_id = blockIdx.x;
    int kv_start = chunk_id * kBc;
    int lid = threadIdx.x;

    // Vectorized Q load (once per block)
    half R_Q[kDpt];
    FD_LDST64(R_Q[0], Q[lid * kDpt]);

    float scale = 1.0f / sqrtf((float)kHeadDim);
    float row_max = -INFINITY, row_sum = 0.0f;
    float O_acc[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++) O_acc[i] = 0.0f;

    #pragma unroll
    for (int r = 0; r < kBc; r++) {
        int g_row = kv_start + r;
        if (g_row >= KV_seqlen) break;

        // Vectorized K load
        half K_reg[kDpt];
        FD_LDST64(K_reg[0], K[g_row * kHeadDim + lid * kDpt]);

        // Dot product
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            dot += __half2float(R_Q[i]) * __half2float(K_reg[i]);

        // Warp reduce sum (full 32 threads)
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, mask);

        // Online softmax
        float score = dot * scale;
        float m_new = fmaxf(row_max, score);
        float exp_diff = expf(row_max - m_new);
        float p = expf(score - m_new);
        row_sum = row_sum * exp_diff + p;

        // Vectorized V load + accumulate
        half V_reg[kDpt];
        FD_LDST64(V_reg[0], V[g_row * kHeadDim + lid * kDpt]);
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            O_acc[i] = O_acc[i] * exp_diff + p * __half2float(V_reg[i]);

        row_max = m_new;
    }

    // Normalize + vectorized store
    float inv = __frcp_rn(row_sum);
    half O_out[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++)
        O_out[i] = __float2half(O_acc[i] * inv);
    FD_ST64(O_partial[chunk_id * kHeadDim + lid * kDpt], O_out[0]);

    if (lid == 0)
        LSE[chunk_id] = logf(row_sum) + row_max;
}


// ============================================================
// v2: Large Chunk (Bc=128, 1 warp)
// ============================================================
// KV=4096: chunks 256→32, Stage2 merges 8x fewer partials

template<int kHeadDim, int kBc = 128>
__global__ void fd_v2_stage1_kernel(
    const half *Q, const half *K, const half *V,
    half *O_partial, float *LSE,
    int KV_seqlen, int num_chunks) {

    constexpr int kDpt = kHeadDim / WARP_SIZE;
    int chunk_id = blockIdx.x;
    int kv_start = chunk_id * kBc;
    int lid = threadIdx.x;

    half R_Q[kDpt];
    FD_LDST64(R_Q[0], Q[lid * kDpt]);

    float scale = 1.0f / sqrtf((float)kHeadDim);
    float row_max = -INFINITY, row_sum = 0.0f;
    float O_acc[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++) O_acc[i] = 0.0f;

    // Process Bc rows (no unroll — too many iterations)
    for (int r = 0; r < kBc; r++) {
        int g_row = kv_start + r;
        if (g_row >= KV_seqlen) break;

        half K_reg[kDpt];
        FD_LDST64(K_reg[0], K[g_row * kHeadDim + lid * kDpt]);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            dot += __half2float(R_Q[i]) * __half2float(K_reg[i]);

        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, mask);

        float score = dot * scale;
        float m_new = fmaxf(row_max, score);
        float exp_diff = expf(row_max - m_new);
        float p = expf(score - m_new);
        row_sum = row_sum * exp_diff + p;

        half V_reg[kDpt];
        FD_LDST64(V_reg[0], V[g_row * kHeadDim + lid * kDpt]);
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            O_acc[i] = O_acc[i] * exp_diff + p * __half2float(V_reg[i]);

        row_max = m_new;
    }

    float inv = __frcp_rn(row_sum);
    half O_out[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++)
        O_out[i] = __float2half(O_acc[i] * inv);
    FD_ST64(O_partial[chunk_id * kHeadDim + lid * kDpt], O_out[0]);

    if (lid == 0)
        LSE[chunk_id] = logf(row_sum) + row_max;
}


// ============================================================
// v3: Multi-Warp (4 warps, 128 threads, Bc=128)
// ============================================================
// 4 warps each handle 32 KV rows independently.
// In-block merge via SMEM eliminates per-warp global write.
// More warps → more concurrent memory requests → better latency hiding.

template<int kHeadDim, int kBc = 128, int kNumWarps = 4>
__global__ void __launch_bounds__(WARP_SIZE * kNumWarps)
fd_v3_stage1_kernel(
    const half *Q, const half *K, const half *V,
    half *O_partial, float *LSE,
    int KV_seqlen, int num_chunks) {

    constexpr int kDpt = kHeadDim / WARP_SIZE;
    constexpr int kRowsPerWarp = kBc / kNumWarps;  // 32

    int chunk_id = blockIdx.x;
    int kv_start = chunk_id * kBc;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lid = tid % WARP_SIZE;

    // SMEM for cross-warp merge
    __shared__ float smem_lse[kNumWarps];
    __shared__ half  smem_O[kNumWarps * kHeadDim];

    // Vectorized Q load
    half R_Q[kDpt];
    FD_LDST64(R_Q[0], Q[lid * kDpt]);

    float scale = 1.0f / sqrtf((float)kHeadDim);
    float row_max = -INFINITY, row_sum = 0.0f;
    float O_acc[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++) O_acc[i] = 0.0f;

    // Each warp processes its KV rows
    int warp_kv_start = kv_start + warp_id * kRowsPerWarp;
    for (int r = 0; r < kRowsPerWarp; r++) {
        int g_row = warp_kv_start + r;
        if (g_row >= KV_seqlen) break;

        half K_reg[kDpt];
        FD_LDST64(K_reg[0], K[g_row * kHeadDim + lid * kDpt]);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            dot += __half2float(R_Q[i]) * __half2float(K_reg[i]);

        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, mask);

        float score = dot * scale;
        float m_new = fmaxf(row_max, score);
        float exp_diff = expf(row_max - m_new);
        float p = expf(score - m_new);
        row_sum = row_sum * exp_diff + p;

        half V_reg[kDpt];
        FD_LDST64(V_reg[0], V[g_row * kHeadDim + lid * kDpt]);
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            O_acc[i] = O_acc[i] * exp_diff + p * __half2float(V_reg[i]);

        row_max = m_new;
    }

    // Normalize within warp
    float inv = (row_sum > 0.0f) ? __frcp_rn(row_sum) : 0.0f;
    half O_warp[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++)
        O_warp[i] = __float2half(O_acc[i] * inv);

    // Write warp results to SMEM
    FD_ST64(smem_O[warp_id * kHeadDim + lid * kDpt], O_warp[0]);
    if (lid == 0)
        smem_lse[warp_id] = (row_sum > 0.0f) ? (logf(row_sum) + row_max) : -INFINITY;
    __syncthreads();

    // Cross-warp merge (all threads in warp 0 participate)
    if (warp_id == 0) {
        // Find max LSE
        float max_lse = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kNumWarps; w++)
            max_lse = fmaxf(max_lse, smem_lse[w]);

        // Weighted sum
        float sum_w = 0.0f;
        float O_merged[kDpt];
        #pragma unroll
        for (int i = 0; i < kDpt; i++) O_merged[i] = 0.0f;

        #pragma unroll
        for (int w = 0; w < kNumWarps; w++) {
            float weight = expf(smem_lse[w] - max_lse);
            sum_w += weight;
            half O_tmp[kDpt];
            FD_LDST64(O_tmp[0], smem_O[w * kHeadDim + lid * kDpt]);
            #pragma unroll
            for (int i = 0; i < kDpt; i++)
                O_merged[i] += weight * __half2float(O_tmp[i]);
        }

        float inv_w = __frcp_rn(sum_w);
        half O_final[kDpt];
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            O_final[i] = __float2half(O_merged[i] * inv_w);
        FD_ST64(O_partial[chunk_id * kHeadDim + lid * kDpt], O_final[0]);

        if (lid == 0) {
            // Block LSE = log(sum_w) + max_lse
            LSE[chunk_id] = logf(sum_w) + max_lse;
        }
    }
}


// ============================================================
// v4: Final Optimized (8 warps, Bc=256, fast math)
// ============================================================
// - 8 warps (256 threads): max memory parallelism
// - Bc=256: fewer chunks, less Stage2 work
// - __expf / __fmaf_rn: fast intrinsics
// - f32 accumulation throughout (no precision loss)

template<int kHeadDim, int kBc = 256, int kNumWarps = 8>
__global__ void __launch_bounds__(WARP_SIZE * kNumWarps)
fd_v4_stage1_kernel(
    const half *Q, const half *K, const half *V,
    half *O_partial, float *LSE,
    int KV_seqlen, int num_chunks) {

    constexpr int kDpt = kHeadDim / WARP_SIZE;
    constexpr int kRowsPerWarp = kBc / kNumWarps;  // 32

    int chunk_id = blockIdx.x;
    int kv_start = chunk_id * kBc;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lid = tid % WARP_SIZE;

    // SMEM for cross-warp merge
    __shared__ float smem_lse[kNumWarps];
    __shared__ float smem_O[kNumWarps * kHeadDim];  // f32 for precision

    // Vectorized Q load
    half R_Q[kDpt];
    FD_LDST64(R_Q[0], Q[lid * kDpt]);
    // Pre-convert to f32
    float Q_f32[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++)
        Q_f32[i] = __half2float(R_Q[i]);

    float scale = __frcp_rn(sqrtf((float)kHeadDim));
    float row_max = -INFINITY, row_sum = 0.0f;
    float O_acc[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++) O_acc[i] = 0.0f;

    int warp_kv_start = kv_start + warp_id * kRowsPerWarp;
    for (int r = 0; r < kRowsPerWarp; r++) {
        int g_row = warp_kv_start + r;
        if (g_row >= KV_seqlen) break;

        // Vectorized K load
        half K_reg[kDpt];
        FD_LDST64(K_reg[0], K[g_row * kHeadDim + lid * kDpt]);

        // FMA dot product
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            dot = __fmaf_rn(Q_f32[i], __half2float(K_reg[i]), dot);

        // Warp reduce
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, mask);

        // Online softmax with fast math
        float score = dot * scale;
        float m_new = fmaxf(row_max, score);
        float exp_diff = __expf(row_max - m_new);
        float p = __expf(score - m_new);
        row_sum = __fmaf_rn(row_sum, exp_diff, p);

        // Vectorized V load + FMA accumulate
        half V_reg[kDpt];
        FD_LDST64(V_reg[0], V[g_row * kHeadDim + lid * kDpt]);
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            O_acc[i] = __fmaf_rn(O_acc[i], exp_diff, p * __half2float(V_reg[i]));

        row_max = m_new;
    }

    // Write warp results to SMEM (f32 for precision in merge)
    float warp_lse = (row_sum > 0.0f) ? (logf(row_sum) + row_max) : -INFINITY;
    float inv_local = (row_sum > 0.0f) ? __frcp_rn(row_sum) : 0.0f;

    #pragma unroll
    for (int i = 0; i < kDpt; i++)
        smem_O[warp_id * kHeadDim + lid * kDpt + i] = O_acc[i] * inv_local;

    if (lid == 0) smem_lse[warp_id] = warp_lse;
    __syncthreads();

    // Cross-warp merge (all threads participate for bandwidth)
    if (warp_id == 0) {
        float max_lse = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kNumWarps; w++)
            max_lse = fmaxf(max_lse, smem_lse[w]);

        float sum_w = 0.0f;
        float O_merged[kDpt];
        #pragma unroll
        for (int i = 0; i < kDpt; i++) O_merged[i] = 0.0f;

        #pragma unroll
        for (int w = 0; w < kNumWarps; w++) {
            float weight = __expf(smem_lse[w] - max_lse);
            sum_w += weight;
            #pragma unroll
            for (int i = 0; i < kDpt; i++)
                O_merged[i] = __fmaf_rn(weight,
                    smem_O[w * kHeadDim + lid * kDpt + i], O_merged[i]);
        }

        float inv_w = __frcp_rn(sum_w);
        half O_final[kDpt];
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            O_final[i] = __float2half(O_merged[i] * inv_w);
        FD_ST64(O_partial[chunk_id * kHeadDim + lid * kDpt], O_final[0]);

         if (lid == 0)
            LSE[chunk_id] = logf(sum_w) + max_lse;
    }
}


// ============================================================
// v5_fused: Single-Kernel Flash Decoding (NO Stage2!)
// ============================================================
// Eliminates:
//   - Stage2 kernel launch overhead (~5-10 μs)
//   - O_partial global memory write/read
//   - LSE global memory write/read
// All warps in one block, merge via SMEM.
// Uses unnormalized O + (row_max, row_sum) for precision-safe merge.

template<int kHeadDim, int kMaxWarps = 32, int kRowsPerWarp = 32>
__global__ void __launch_bounds__(WARP_SIZE * kMaxWarps)
fd_v5_fused_kernel(
    const half *Q, const half *K, const half *V,
    half *O_final,
    int KV_seqlen,
    int stride = kHeadDim) {

    constexpr int kDpt = kHeadDim / WARP_SIZE;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lid = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    __shared__ float smem_max[kMaxWarps];
    __shared__ float smem_sum[kMaxWarps];
    __shared__ float smem_O[kMaxWarps * kHeadDim];

    // Q in f32 registers
    half R_Q[kDpt];
    FD_LDST64(R_Q[0], Q[lid * kDpt]);
    float Q_f32[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++) Q_f32[i] = __half2float(R_Q[i]);

    float scale = __frcp_rn(sqrtf((float)kHeadDim));
    float row_max = -INFINITY, row_sum = 0.0f;
    float O_acc[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++) O_acc[i] = 0.0f;

    int kv_start = warp_id * kRowsPerWarp;
    if (kv_start < KV_seqlen) {
        for (int r = 0; r < kRowsPerWarp; r++) {
            int g_row = kv_start + r;
            if (g_row >= KV_seqlen) break;

            half K_reg[kDpt];
            FD_LDST64(K_reg[0], K[g_row * stride + lid * kDpt]);

            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < kDpt; i++)
                dot = __fmaf_rn(Q_f32[i], __half2float(K_reg[i]), dot);

            #pragma unroll
            for (int mask = 16; mask >= 1; mask >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, mask);

            float score = dot * scale;
            float m_new = fmaxf(row_max, score);
            float exp_diff = __expf(row_max - m_new);
            float p = __expf(score - m_new);
            row_sum = __fmaf_rn(row_sum, exp_diff, p);

            half V_reg[kDpt];
            FD_LDST64(V_reg[0], V[g_row * stride + lid * kDpt]);
            #pragma unroll
            for (int i = 0; i < kDpt; i++)
                O_acc[i] = __fmaf_rn(O_acc[i], exp_diff, p * __half2float(V_reg[i]));

            row_max = m_new;
        }
    }

    // Write unnormalized results to SMEM
    #pragma unroll
    for (int i = 0; i < kDpt; i++)
        smem_O[warp_id * kHeadDim + lid * kDpt + i] = O_acc[i];
    if (lid == 0) {
        smem_max[warp_id] = row_max;
        smem_sum[warp_id] = row_sum;
    }
    __syncthreads();

    // ALL threads merge (each handles its own kDpt elements)
    float global_max = -INFINITY;
    for (int w = 0; w < num_warps; w++)
        global_max = fmaxf(global_max, smem_max[w]);

    float total_sum = 0.0f;
    float O_merged[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++) O_merged[i] = 0.0f;

    for (int w = 0; w < num_warps; w++) {
        float rescale = __expf(smem_max[w] - global_max);
        total_sum += smem_sum[w] * rescale;
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            O_merged[i] = __fmaf_rn(rescale,
                smem_O[w * kHeadDim + lid * kDpt + i], O_merged[i]);
    }

    // Normalize and write (warp 0)
    if (warp_id == 0) {
        float inv_sum = __frcp_rn(total_sum);
        half O_out[kDpt];
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            O_out[i] = __float2half(O_merged[i] * inv_sum);
        FD_ST64(O_final[lid * kDpt], O_out[0]);
    }
}


// ============================================================
// v5_split: Improved 2-Stage for Large KV
// ============================================================
// Fixes vs v4:
//   1. ALL warps participate in cross-warp merge
//   2. Unnormalized O + (max, sum) — no precision-losing normalize→reweight
//   3. Still multi-block for arbitrarily large KV

template<int kHeadDim, int kBc = 256, int kNumWarps = 8>
__global__ void __launch_bounds__(WARP_SIZE * kNumWarps)
fd_v5_split_stage1_kernel(
    const half *Q, const half *K, const half *V,
    half *O_partial, float *LSE,
    int KV_seqlen, int num_chunks,
    int stride = kHeadDim) {

    constexpr int kDpt = kHeadDim / WARP_SIZE;
    constexpr int kRowsPerWarp = kBc / kNumWarps;

    int chunk_id = blockIdx.x;
    int kv_start = chunk_id * kBc;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lid = tid % WARP_SIZE;

    __shared__ float smem_max[kNumWarps];
    __shared__ float smem_sum[kNumWarps];
    __shared__ float smem_O[kNumWarps * kHeadDim];

    half R_Q[kDpt];
    FD_LDST64(R_Q[0], Q[lid * kDpt]);
    float Q_f32[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++) Q_f32[i] = __half2float(R_Q[i]);

    float scale = __frcp_rn(sqrtf((float)kHeadDim));
    float row_max = -INFINITY, row_sum = 0.0f;
    float O_acc[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++) O_acc[i] = 0.0f;

    int warp_kv_start = kv_start + warp_id * kRowsPerWarp;
    for (int r = 0; r < kRowsPerWarp; r++) {
        int g_row = warp_kv_start + r;
        if (g_row >= KV_seqlen) break;

        half K_reg[kDpt];
        FD_LDST64(K_reg[0], K[g_row * stride + lid * kDpt]);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            dot = __fmaf_rn(Q_f32[i], __half2float(K_reg[i]), dot);

        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, mask);

        float score = dot * scale;
        float m_new = fmaxf(row_max, score);
        float exp_diff = __expf(row_max - m_new);
        float p = __expf(score - m_new);
        row_sum = __fmaf_rn(row_sum, exp_diff, p);

        half V_reg[kDpt];
        FD_LDST64(V_reg[0], V[g_row * stride + lid * kDpt]);
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            O_acc[i] = __fmaf_rn(O_acc[i], exp_diff, p * __half2float(V_reg[i]));

        row_max = m_new;
    }

    // Unnormalized to SMEM
    #pragma unroll
    for (int i = 0; i < kDpt; i++)
        smem_O[warp_id * kHeadDim + lid * kDpt + i] = O_acc[i];
    if (lid == 0) {
        smem_max[warp_id] = row_max;
        smem_sum[warp_id] = row_sum;
    }
    __syncthreads();

    // ALL warps merge
    float global_max = -INFINITY;
    #pragma unroll
    for (int w = 0; w < kNumWarps; w++)
        global_max = fmaxf(global_max, smem_max[w]);

    float total_sum = 0.0f;
    float O_merged[kDpt];
    #pragma unroll
    for (int i = 0; i < kDpt; i++) O_merged[i] = 0.0f;

    #pragma unroll
    for (int w = 0; w < kNumWarps; w++) {
        float rescale = __expf(smem_max[w] - global_max);
        total_sum += smem_sum[w] * rescale;
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            O_merged[i] = __fmaf_rn(rescale,
                smem_O[w * kHeadDim + lid * kDpt + i], O_merged[i]);
    }

    // Normalize and write block result
    if (warp_id == 0) {
        float inv_sum = __frcp_rn(total_sum);
        half O_out[kDpt];
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            O_out[i] = __float2half(O_merged[i] * inv_sum);
        FD_ST64(O_partial[chunk_id * kHeadDim + lid * kDpt], O_out[0]);

        if (lid == 0)
            LSE[chunk_id] = logf(total_sum) + global_max;
    }
}



// ============================================================
// Optimized Stage 2: Multi-warp merge (128 threads)
// ============================================================
// Original: 1 warp (32 threads), scalar loads, non-coalesced
// Optimized: 4 warps (128 threads), vectorized loads, all threads work

template<int kHeadDim>
__global__ void __launch_bounds__(128)
fd_opt_stage2_kernel(
    const half *O_partial, const float *LSE, half *O_final,
    int num_chunks) {

    constexpr int kDpt = kHeadDim / WARP_SIZE;  // elements per thread within a warp
    int tid = threadIdx.x;
    int lid = tid % WARP_SIZE;
    // All 128 threads share the work: each handles kDpt elements of d
    // But we only have d=128 elements total. 128 threads → 1 element each.
    // Use warp 0 for the main computation (same d mapping as Stage1).
    int warp_id = tid / WARP_SIZE;

    // Step 1: Find max LSE (all threads read, warp 0 produces result)
    float max_lse = -INFINITY;
    // Each warp reads a portion of LSE array
    for (int c = warp_id; c < num_chunks; c += 4)
        max_lse = fmaxf(max_lse, LSE[c]);

    // Intra-warp reduce
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1)
        max_lse = fmaxf(max_lse, __shfl_xor_sync(0xffffffff, max_lse, mask));

    // Cross-warp reduce via SMEM
    __shared__ float smem_max[4];
    if (lid == 0) smem_max[warp_id] = max_lse;
    __syncthreads();
    max_lse = smem_max[0];
    for (int w = 1; w < 4; w++)
        max_lse = fmaxf(max_lse, smem_max[w]);

    // Step 2: Weighted sum (warp 0 does the output computation)
    if (warp_id == 0) {
        float sum_w = 0.0f;
        float O_f32[kDpt];
        #pragma unroll
        for (int i = 0; i < kDpt; i++) O_f32[i] = 0.0f;

        for (int c = 0; c < num_chunks; c++) {
            float w = __expf(LSE[c] - max_lse);
            sum_w += w;

            half O_tmp[kDpt];
            FD_LDST64(O_tmp[0], O_partial[c * kHeadDim + lid * kDpt]);
            #pragma unroll
            for (int i = 0; i < kDpt; i++)
                O_f32[i] = __fmaf_rn(w, __half2float(O_tmp[i]), O_f32[i]);
        }

        float inv_w = __frcp_rn(sum_w);
        half O_out[kDpt];
        #pragma unroll
        for (int i = 0; i < kDpt; i++)
            O_out[i] = __float2half(O_f32[i] * inv_w);
        FD_ST64(O_final[lid * kDpt], O_out[0]);
    }
}
