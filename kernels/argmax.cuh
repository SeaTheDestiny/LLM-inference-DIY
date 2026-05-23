#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

// ArgMax Kernel (For simple greedy sampling)
__global__ void argmax_kernel(int* out_token, const half* logits, int vocab_size) {
    // Single block, 512 threads is enough for simple argmax
    __shared__ float s_max_val;
    __shared__ int s_max_idx;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int items_per_thread = (vocab_size + num_threads - 1) / num_threads;
    
    float local_max_val = -INFINITY;
    int local_max_idx = -1;
    
    for (int i = 0; i < items_per_thread; i++) {
        int idx = tid * items_per_thread + i;
        if (idx < vocab_size) {
            float val = __half2float(logits[idx]);
            if (val > local_max_val) {
                local_max_val = val;
                local_max_idx = idx;
            }
        }
    }
    
    // Reduce across block
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, local_max_val, mask);
        int other_idx = __shfl_xor_sync(0xffffffff, local_max_idx, mask);
        if (other_val > local_max_val) {
            local_max_val = other_val;
            local_max_idx = other_idx;
        }
    }
    
    __shared__ float shared_vals[32];
    __shared__ int shared_idxs[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
        shared_vals[warp_id] = local_max_val;
        shared_idxs[warp_id] = local_max_idx;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float block_max_val = (tid < num_threads / 32) ? shared_vals[tid] : -INFINITY;
        int block_max_idx = (tid < num_threads / 32) ? shared_idxs[tid] : -1;
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, block_max_val, mask);
            int other_idx = __shfl_xor_sync(0xffffffff, block_max_idx, mask);
            if (other_val > block_max_val) {
                block_max_val = other_val;
                block_max_idx = other_idx;
            }
        }
        if (tid == 0) {
            s_max_val = block_max_val;
            s_max_idx = block_max_idx;
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        *out_token = s_max_idx;
    }
}
