#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

// Custom high-performance Softmax + Temperature + Cumulative Sampling Kernel
__global__ void temperature_sampler_kernel(
    int* out_token,             // Output token ID
    const half* logits,         // Raw logits from LM Head [vocab_size]
    float rand_val,             // Random float in [0, 1] for cumulative selection
    float temperature,          // Generation temperature (e.g., 0.7f)
    int vocab_size
) {
    // 512 threads inside a single block is perfect for high-speed block reductions
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int items_per_thread = (vocab_size + num_threads - 1) / num_threads;
    
    __shared__ float s_max_val;
    __shared__ float s_sum_exp;
    
    // 1. Find Max Logit (to avoid exponential overflow)
    float local_max = -INFINITY;
    for (int i = 0; i < items_per_thread; i++) {
        int idx = tid * items_per_thread + i;
        if (idx < vocab_size) {
            float val = __half2float(logits[idx]);
            if (val > local_max) local_max = val;
        }
    }
    
    // Block-level max reduction
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, local_max, mask);
        if (other > local_max) local_max = other;
    }
    
    __shared__ float shared_maxs[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared_maxs[warp_id] = local_max;
    __syncthreads();
    
    if (warp_id == 0) {
        float block_max = (tid < num_threads / 32) ? shared_maxs[tid] : -INFINITY;
        for (int mask = 16; mask >= 1; mask >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, block_max, mask);
            if (other > block_max) block_max = other;
        }
        if (tid == 0) s_max_val = block_max;
    }
    __syncthreads();
    
    // 2. Compute Sum of Exponentials (Partition function)
    float local_sum = 0.0f;
    float temp_inv = 1.0f / (temperature > 1e-5f ? temperature : 1.0f);
    
    for (int i = 0; i < items_per_thread; i++) {
        int idx = tid * items_per_thread + i;
        if (idx < vocab_size) {
            float val = __half2float(logits[idx]);
            local_sum += __expf((val - s_max_val) * temp_inv);
        }
    }
    
    // Block-level sum reduction
    for (int mask = 16; mask >= 1; mask >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, mask);
    }
    
    __shared__ float shared_sums[32];
    if (lane_id == 0) shared_sums[warp_id] = local_sum;
    __syncthreads();
    
    if (warp_id == 0) {
        float block_sum = (tid < num_threads / 32) ? shared_sums[tid] : 0.0f;
        for (int mask = 16; mask >= 1; mask >>= 1) {
            block_sum += __shfl_xor_sync(0xffffffff, block_sum, mask);
        }
        if (tid == 0) s_sum_exp = block_sum;
    }
    __syncthreads();
    
    // 3. Cumulative Probability Selection
    // We want to find the index where cumulative probability exceeds rand_val * s_sum_exp
    __shared__ int s_selected_token;
    if (tid == 0) s_selected_token = vocab_size - 1;
    __syncthreads();
    
    float target_sum = rand_val * s_sum_exp;
    
    // Compute prefix sum per thread and search
    // Since threads process contiguous chunks of logits, we can first compute thread-local sums
    float my_chunk_sum = 0.0f;
    for (int i = 0; i < items_per_thread; i++) {
        int idx = tid * items_per_thread + i;
        if (idx < vocab_size) {
            float val = __half2float(logits[idx]);
            my_chunk_sum += __expf((val - s_max_val) * temp_inv);
        }
    }
    
    // Block-level exclusive scan of my_chunk_sum
    // For simplicity and high speed, let's do a sequential block-wide sweep in shared memory
    __shared__ float s_thread_sums[512];
    s_thread_sums[tid] = my_chunk_sum;
    __syncthreads();
    
    // Thread 0 computes block cumulative offsets
    if (tid == 0) {
        float acc = 0.0f;
        for (int t = 0; t < num_threads; t++) {
            float temp_val = s_thread_sums[t];
            s_thread_sums[t] = acc; // Store start offset for thread t
            acc += temp_val;
        }
    }
    __syncthreads();
    
    // Search inside each thread's chunk
    float start_offset = s_thread_sums[tid];
    float running = start_offset;
    
    for (int i = 0; i < items_per_thread; i++) {
        int idx = tid * items_per_thread + i;
        if (idx < vocab_size) {
            float val = __half2float(logits[idx]);
            float prob = __expf((val - s_max_val) * temp_inv);
            running += prob;
            if (running >= target_sum) {
                atomicMin(&s_selected_token, idx);
            }
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        *out_token = s_selected_token;
    }
}
