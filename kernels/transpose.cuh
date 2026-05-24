#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Custom Optimized 2D Grid Transpose Kernel for FP16
__global__ void transpose_weight_kernel(half* out, const half* in, int rows, int cols) {
    // blockDim: 16x16
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

// Host helper to transpose weight matrix on GPU
inline void transpose_weight_gpu(half*& d_weight, int rows, int cols) {
    half* d_transposed;
    cudaMalloc(&d_transposed, rows * cols * sizeof(half));
    
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    
    transpose_weight_kernel<<<grid, block>>>(d_transposed, d_weight, rows, cols);
    cudaDeviceSynchronize();
    
    // Free old weight and replace pointer
    cudaFree(d_weight);
    d_weight = d_transposed;
}
