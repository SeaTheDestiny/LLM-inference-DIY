#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// RMSNorm Kernel (Highly optimized, 1 block handles 1 row of hidden_size=2048)
__global__ void rmsnorm_kernel(half* out, const half* in, const half* weight, int d, float epsilon) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int items_per_thread = d / num_threads;
    
    // Allocate shared memory for reduction
    __shared__ float s_variance;
    
    // Thread-local sum of squares
    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < items_per_thread; i++) {
        float val = __half2float(in[tid * items_per_thread + i]);
        local_sum += val * val;
    }
    
    // Block-level reduction (Warp-shuffle & shared memory)
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, mask);
    }
    
    __shared__ float shared_sums[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
        shared_sums[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float block_sum = (tid < num_threads / 32) ? shared_sums[tid] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) {
            block_sum += __shfl_xor_sync(0xffffffff, block_sum, mask);
        }
        if (tid == 0) {
            s_variance = block_sum / d;
        }
    }
    __syncthreads();
    
    // Normalize and scale by weight
    float rsqrt_val = rsqrtf(s_variance + epsilon);
    #pragma unroll
    for (int i = 0; i < items_per_thread; i++) {
        int idx = tid * items_per_thread + i;
        float val = __half2float(in[idx]);
        float w = __half2float(weight[idx]);
        out[idx] = __float2half(val * rsqrt_val * w);
    }
}
