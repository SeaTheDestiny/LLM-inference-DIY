#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Bias Addition Kernel (Specifically for QKV projections)
__global__ void add_bias_kernel(half* out, const half* bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __float2half(__half2float(out[idx]) + __half2float(bias[idx]));
    }
}
