/**
 * tune_hgemm.cu — HGEMM Config Sweep
 * Usage: nvcc -arch=sm_89 -O3 -std=c++17 tune_hgemm.cu -o tune_hgemm -lcublas && ./tune_hgemm
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hgemm.cuh"

double tflops(int M, int N, int K) { return 2.0*M*N*K/1e12; }

template<int BM, int BN, int STAGE, int WK, int WM, int WN>
float run_one(half *dA, half *dB, half *dC, int M, int N, int K, 
              cudaEvent_t start, cudaEvent_t stop, const char* name) {
    constexpr int A_sz = BM * (16 * WK);
    constexpr int B_sz = (16 * WK) * BN;
    constexpr int THR = WARP_SIZE * WM * WN;
    int smem = STAGE * (A_sz + B_sz) * sizeof(half);
    if (smem > 98304) { printf("  %-30s SKIP (smem=%d > 96KB)\n", name, smem); return 0; }
    
    int MT = (M + BM - 1) / BM, NT = (N + BN - 1) / BN;
    dim3 g(NT, MT), b(THR);
    auto fn = hgemm_final_kernel<BM, BN, WM, WN, BM/16/WM, BN/8/WN, WK>;
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    
    fn<<<g, b, smem>>>(dA, dB, dC, M, N, K);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("  %-30s CUDA ERROR: %s\n", name, cudaGetErrorString(err)); return 0; }
    cudaEventRecord(start);
    fn<<<g, b, smem>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("  %-30s CUDA SYNC ERROR: %s\n", name, cudaGetErrorString(err)); return 0; }
    float ms; cudaEventElapsedTime(&ms, start, stop);
    printf("  %-30s %8.3f ms  %6.1f TFLOPS\n", name, ms, tflops(M,N,K)/(ms/1000.0));
    return ms;
}

int main(int argc, char** argv) {
    int M=(argc>1)?atoi(argv[1]):4096, N=(argc>2)?atoi(argv[2]):4096, K=(argc>3)?atoi(argv[3]):4096;
    printf("=== HGEMM Tuning: M=%d N=%d K=%d ===\n\n", M, N, K);
    
    half *dA, *dB, *dC;
    cudaMalloc(&dA, M*K*sizeof(half));
    cudaMalloc(&dB, K*N*sizeof(half));
    cudaMalloc(&dC, M*N*sizeof(half));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("%-30s %10s %10s\n", "Config", "Time(ms)", "TFLOPS");
    printf("--------------------------------------------------------\n");

    // Baseline kernel variants
    {
        // naive (16x8, 1 warp)
        int MT=(M+15)/16, NT=(N+7)/8;
        dim3 g(NT,MT), b(WARP_SIZE);
        hgemm_naive_kernel<1,1><<<g,b>>>(dA,dB,dC,M,N,K);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        hgemm_naive_kernel<1,1><<<g,b>>>(dA,dB,dC,M,N,K);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("  %-30s %8.3f ms  %6.1f TFLOPS\n", "naive(16x8)", ms, tflops(M,N,K)/(ms/1000.0));
    }
    {
        int MT=(M+127)/128, NT=(N+127)/128;
        dim3 g(NT,MT), b(WARP_SIZE*8);
        constexpr int A_sz=128*32, B_sz=32*128; int smem=2*(A_sz+B_sz)*sizeof(half);
        auto fn=hgemm_final_kernel<128,128,2,4,4,4,2>;
        cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
        fn<<<g,b,smem>>>(dA,dB,dC,M,N,K); cudaDeviceSynchronize();
        cudaEventRecord(start);
        fn<<<g,b,smem>>>(dA,dB,dC,M,N,K);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("  %-30s %8.3f ms  %6.1f TFLOPS\n", "final(128x128,S2,K2,W2x4)", ms, tflops(M,N,K)/(ms/1000.0));
    }

    // === Sweep ===
    printf("\n--- Config Sweep ---\n");
    run_one<64,  64,  2, 1, 2, 4>(dA,dB,dC,M,N,K,start,stop,"BM64_BN64_S2_K1_W2x4");
    run_one<64,  64,  2, 2, 2, 4>(dA,dB,dC,M,N,K,start,stop,"BM64_BN64_S2_K2_W2x4");
    run_one<64,  128, 2, 2, 2, 4>(dA,dB,dC,M,N,K,start,stop,"BM64_BN128_S2_K2_W2x4");
    run_one<128, 64,  2, 2, 4, 2>(dA,dB,dC,M,N,K,start,stop,"BM128_BN64_S2_K2_W4x2");
    run_one<128, 128, 2, 2, 4, 2>(dA,dB,dC,M,N,K,start,stop,"BM128_BN128_S2_K2_W4x2");
    run_one<128, 128, 2, 2, 1, 8>(dA,dB,dC,M,N,K,start,stop,"BM128_BN128_S2_K2_W1x8");
    run_one<128, 128, 3, 2, 2, 4>(dA,dB,dC,M,N,K,start,stop,"BM128_BN128_S3_K2_W2x4");
    run_one<128, 128, 4, 2, 2, 4>(dA,dB,dC,M,N,K,start,stop,"BM128_BN128_S4_K2_W2x4");
    // K=1 variants
    run_one<128, 128, 2, 1, 2, 4>(dA,dB,dC,M,N,K,start,stop,"BM128_BN128_S2_K1_W2x4");
    run_one<128, 128, 3, 1, 2, 4>(dA,dB,dC,M,N,K,start,stop,"BM128_BN128_S3_K1_W2x4");

    // cuBLAS
    {
        cublasHandle_t h; cublasCreate(&h);
        half a=__float2half(1), b=__float2half(0);
        cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB,CUDA_R_16F,N,dA,CUDA_R_16F,K,&b,dC,CUDA_R_16F,N,CUBLAS_COMPUTE_16F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB,CUDA_R_16F,N,dA,CUDA_R_16F,K,&b,dC,CUDA_R_16F,N,CUBLAS_COMPUTE_16F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms,start,stop);
        printf("  %-30s %8.3f ms  %6.1f TFLOPS\n", "cuBLAS", ms, tflops(M,N,K)/(ms/1000.0));
        cublasDestroy(h);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
