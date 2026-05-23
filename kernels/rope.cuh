#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// RoPE (旋转位置编码) Kernel
__global__ void rope_kernel(half* Q, half* K, int pos, int num_heads, int head_dim) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x; // 0 to 63 (since head_dim = 128, handles pairs)
    
    int idx1 = head_idx * head_dim + 2 * tid;
    int idx2 = idx1 + 1;
    
    float q1 = __half2float(Q[idx1]);
    float q2 = __half2float(Q[idx2]);
    float k1 = __half2float(K[idx1]);
    float k2 = __half2float(K[idx2]);
    
    // Angle calculation matching Qwen base frequency config
    float theta = pos * powf(10000.0f, -2.0f * tid / head_dim);
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    
    Q[idx1] = __float2half(q1 * cos_t - q2 * sin_t);
    Q[idx2] = __float2half(q1 * sin_t + q2 * cos_t);
    K[idx1] = __float2half(k1 * cos_t - k2 * sin_t);
    K[idx2] = __float2half(k1 * sin_t + k2 * cos_t);
}
