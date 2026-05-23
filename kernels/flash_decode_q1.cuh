/**
 * flash_decode_q1.cuh — Flash Decoding for Q_seqlen=1 (auto-regressive inference)
 * ==================================================================================
 * Stage 1: Pure vectorized dot product, 1 warp per KV chunk (Bc=16 rows)
 *          No MMA needed — Q has only 1 row, MMA m16n8k16 can't be used
 * Stage 2: LSE-weighted merge of partial O from all chunks
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ============================================================
// Stage 1: Per-Chunk Attention for Single Q Token (Q=1, Bc=16)
// ============================================================
// grid: (num_kv_splits, 1)  — one block per KV chunk
// block: 32 threads (1 warp)
// Each chunk: 16 KV rows × d elements
//
// Q is shared across all blocks via global memory (or could be in __constant__)
// Each block: dot(Q[0,:], K[chunk,:]) → 16 scores → softmax → partial O[chunk,:]

__global__ void flash_decode_q1_stage1_kernel(
    half *Q, half *K, half *V,
    half *O_partial, float *LSE,
    int KV_seqlen, int num_chunks, int d) {

    constexpr int kBc = 16;   // 16 KV rows per chunk
    int chunk_id = blockIdx.x;
    int kv_start = chunk_id * kBc;

    int lane_id = threadIdx.x;                    // 0..31
    int d_per_thread = d / WARP_SIZE;             // e.g. 128/32 = 4

    // Load Q[0,:] into registers (shared across all K rows)
    half R_Q[8];  // max d=256 → 8 elements
    #pragma unroll
    for (int i = 0; i < d_per_thread; i++) {
        R_Q[i] = Q[lane_id * d_per_thread + i];
    }

    float scale = 1.0f / sqrtf((float)d);

    // Online softmax per chunk
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Partial O accumulator (f32 for precision)
    float O_acc[8] = {0};  // max 8 elements per thread

    // Process 16 KV rows in this chunk
    #pragma unroll
    for (int kv_row = 0; kv_row < kBc; kv_row++) {
        int g_row = kv_start + kv_row;
        bool valid = (g_row < KV_seqlen);

        // Dot product: Q[0] · K[g_row], vectorized across threads
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < d_per_thread; i++) {
            int d_off = lane_id * d_per_thread + i;
            half k_val = valid ? K[g_row * d + d_off] : __float2half(0.0f);
            dot += __half2float(R_Q[i]) * __half2float(k_val);
        }

        // Warp reduce sum
        dot += __shfl_xor_sync(0xffffffff, dot, 1);
        dot += __shfl_xor_sync(0xffffffff, dot, 2);
        dot += __shfl_xor_sync(0xffffffff, dot, 4);
        dot += __shfl_xor_sync(0xffffffff, dot, 8);
        dot += __shfl_xor_sync(0xffffffff, dot, 16);

        if (!valid) continue;

        float score = dot * scale;
        float m_new = fmaxf(row_max, score);
        float exp_diff = expf(row_max - m_new);
        float p = expf(score - m_new);

        row_sum = row_sum * exp_diff + p;

        // Accumulate O += p * V[g_row]
        #pragma unroll
        for (int i = 0; i < d_per_thread; i++) {
            int d_off = lane_id * d_per_thread + i;
            half v_val = V[g_row * d + d_off];
            O_acc[i] = O_acc[i] * exp_diff + p * __half2float(v_val);
        }

        row_max = m_new;
    }

    // Normalize partial O
    float inv_sum = __frcp_rn(row_sum);
    #pragma unroll
    for (int i = 0; i < d_per_thread; i++) {
        O_acc[i] *= inv_sum;
        int d_off = lane_id * d_per_thread + i;
        O_partial[chunk_id * d + d_off] = __float2half(O_acc[i]);
    }

    // Store LSE = log(row_sum) + row_max
    if (lane_id == 0) {
        LSE[chunk_id] = logf(row_sum) + row_max;
    }
}


// ============================================================
// Stage 2: Merge partial O from all chunks
// ============================================================
// grid: (1, 1)  — single block
// block: 32 threads
// O[row,:] = Σ(O_chunk[row,:] × exp(LSE[chunk] - maxLSE)) / Σexp(...)

__global__ void flash_decode_q1_stage2_kernel(
    half *O_partial, float *LSE, half *O_final,
    int num_chunks, int d) {

    int lane_id = threadIdx.x;
    int d_per_thread = d / WARP_SIZE;

    // Step 1: Find max LSE (warp reduce)
    float max_lse = -INFINITY;
    for (int c = 0; c < num_chunks; c++) {
        max_lse = fmaxf(max_lse, LSE[c]);
    }
    max_lse = fmaxf(max_lse, __shfl_xor_sync(0xffffffff, max_lse, 1));
    max_lse = fmaxf(max_lse, __shfl_xor_sync(0xffffffff, max_lse, 2));
    max_lse = fmaxf(max_lse, __shfl_xor_sync(0xffffffff, max_lse, 4));
    max_lse = fmaxf(max_lse, __shfl_xor_sync(0xffffffff, max_lse, 8));
    max_lse = fmaxf(max_lse, __shfl_xor_sync(0xffffffff, max_lse, 16));

    // Step 2: Weighted sum
    float sum_w = 0.0f;
    float O_f32[8] = {0};  // per-thread accumulators

    for (int c = 0; c < num_chunks; c++) {
        float w = expf(LSE[c] - max_lse);
        sum_w += w;

        #pragma unroll
        for (int i = 0; i < d_per_thread; i++) {
            int d_off = lane_id * d_per_thread + i;
            O_f32[i] += w * __half2float(O_partial[c * d + d_off]);
        }
    }

    // Step 3: Final output
    float inv_w = 1.0f / sum_w;
    #pragma unroll
    for (int i = 0; i < d_per_thread; i++) {
        int d_off = lane_id * d_per_thread + i;
        O_final[d_off] = __float2half(O_f32[i] * inv_w);
    }
}
