#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// SwiGLU FFN Activation Kernel
__global__ void swiglu_kernel(half* out, const half* w1_out, const half* w2_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = __half2float(w1_out[idx]);
        float u = __half2float(w2_out[idx]);
        // SiLU: g * sigmoid(g) = g / (1 + exp(-g))
        float silu = g / (1.0f + __expf(-g));
        out[idx] = __float2half(silu * u);
    }
}
