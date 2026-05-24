#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Mathematically correct SwiGLU FFN Activation Kernel for Qwen: w1 * silu(w2)
__global__ void swiglu_kernel(half* out, const half* w1_out, const half* w2_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float w1_val = __half2float(w1_out[idx]);
        float w2_val = __half2float(w2_out[idx]);
        // SiLU: w2 * sigmoid(w2) = w2 / (1 + exp(-w2))
        float silu_w2 = w2_val / (1.0f + __expf(-w2_val));
        out[idx] = __float2half(w1_val * silu_w2);
    }
}
