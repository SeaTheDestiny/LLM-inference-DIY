#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "hgemm_mma_swizzle.cuh"
int main() {
    half hA=__float2half(2.0f), hB=__float2half(3.0f);
    half *dA,*dB,*dC_sw,*dC_cb;
    cudaMalloc(&dA,sizeof(half)); cudaMemcpy(dA,&hA,sizeof(half),cudaMemcpyHostToDevice);
    cudaMalloc(&dB,sizeof(half)); cudaMemcpy(dB,&hB,sizeof(half),cudaMemcpyHostToDevice);
    cudaMalloc(&dC_sw,sizeof(half)); cudaMalloc(&dC_cb,sizeof(half));
    cublasHandle_t h; cublasCreate(&h);
    half alpha=__float2half(1),beta=__float2half(0);
    // C = A x B: dA=2, dB=3 => 6
    cublasHgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,1,1,1,&alpha,dB,1,dA,1,&beta,dC_cb,1);
    cudaDeviceSynchronize();
    hgemm_swizzle_nn(dA,dB,dC_sw,1,1,1);
    cudaDeviceSynchronize();
    half r_cb,r_sw; cudaMemcpy(&r_cb,dC_cb,sizeof(half),cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_sw,dC_sw,sizeof(half),cudaMemcpyDeviceToHost);
    printf("A=2 B=3 Expected=6\ncuBLAS=%.1f swizzle=%.1f\n",__half2float(r_cb),__half2float(r_sw));
    // Swap A,B in cuBLAS
    cublasHgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,1,1,1,&alpha,dA,1,dB,1,&beta,dC_cb,1);
    cudaDeviceSynchronize(); cudaMemcpy(&r_cb,dC_cb,sizeof(half),cudaMemcpyDeviceToHost);
    printf("cuBLAS(swapped)=%.1f\n",__half2float(r_cb));
    cublasDestroy(h); return 0;
}
