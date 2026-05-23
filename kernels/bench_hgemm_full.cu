/**
 * bench_hgemm_full.cu — HGEMM Full Tuning: swizzle, K-stage, block swizzle
 * ========================================================================
 * Usage: nvcc -arch=sm_89 -O3 -std=c++17 bench_hgemm_full.cu -o bench_full -lcublas
 *        ./bench_full [M] [N] [K]
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hgemm_final.cuh"

double tflops(int M,int N,int K){ return 2.0*M*N*K/1e12; }

template<bool SWZ, int STG, int BSWZ>
float test_config(half *dA, half *dB, half *dC, int M, int N, int K,
                  cudaEvent_t start, cudaEvent_t stop, const char *name) {
    constexpr int BM=128, BN=128, WM=2, WN=4, WK=2;
    constexpr int A_sz=BM*(16*WK), B_sz=(16*WK)*BN;
    int smem = STG*(A_sz+B_sz)*sizeof(half);
    if (smem > 98304) { printf("  %-35s SKIP (smem=%d>96KB)\n",name,smem); return 0; }
    
    int MT=(M+BM-1)/BM;
    int NT=(N+BN-1)/BN;
    int NZ=(BSWZ>0)?((NT+BSWZ/BN-1)/(BSWZ/BN)):1;
    dim3 g(BSWZ>0?BSWZ/BN:NT, MT, NZ), b(WARP_SIZE*WM*WN);
    
    auto fn = hgemm_opt_kernel<BM,BN,WM,WN,BM/16/WM,BN/8/WN,WK,SWZ,STG,BSWZ>;
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    
    cudaMemset(dC, 0, M*N*sizeof(half));
    fn<<<g,b,smem>>>(dA,dB,dC,M,N,K);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){printf("  %-35s ERROR:%s\n",name,cudaGetErrorString(e));return 0;}
    
    cudaEventRecord(start);
    fn<<<g,b,smem>>>(dA,dB,dC,M,N,K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms,start,stop);
    printf("  %-35s %8.3f ms  %6.1f TFLOPS\n",name,ms,tflops(M,N,K)/(ms/1000.0));
    return ms;
}

int main(int argc,char**argv){
    int M=(argc>1)?atoi(argv[1]):4096;
    int N=(argc>2)?atoi(argv[2]):4096;
    int K=(argc>3)?atoi(argv[3]):4096;
    printf("=== HGEMM Full Tuning: M=%d N=%d K=%d ===\n\n",M,N,K);

    half *dA,*dB,*dC;
    cudaMalloc(&dA,M*K*sizeof(half));
    cudaMalloc(&dB,K*N*sizeof(half));
    cudaMalloc(&dC,M*N*sizeof(half));
    cudaMemset(dA,0x3C,M*K*sizeof(half));
    cudaMemset(dB,0x3C,K*N*sizeof(half));

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("%-35s %10s %10s\n","Config","Time(ms)","TFLOPS");
    printf("-----------------------------------------------------------\n");

    // === Step 1: Swizzle test (kStage=2) ===
    printf("\n-- Step 1: SMEM Swizzle --\n");
    test_config<false,2,0>(dA,dB,dC,M,N,K,start,stop,"S2_no_swizzle");
    test_config<true, 2,0>(dA,dB,dC,M,N,K,start,stop,"S2_swizzle");

    // === Step 2: Multi-stage (kStage=3,4, with swizzle) ===
    printf("\n-- Step 2: K-Stage (with swizzle) --\n");
    test_config<true, 2,0>(dA,dB,dC,M,N,K,start,stop,"S2_swizzle");
    test_config<true, 3,0>(dA,dB,dC,M,N,K,start,stop,"S3_swizzle");
    test_config<true, 4,0>(dA,dB,dC,M,N,K,start,stop,"S4_swizzle");

    // === Step 3: Block Swizzle (with best stage+swizzle) ===
    printf("\n-- Step 3: Block Swizzle (S2+swizzle) --\n");
    test_config<true, 2, 0   >(dA,dB,dC,M,N,K,start,stop,"S2_swz_blk0");
    test_config<true, 2, 1024>(dA,dB,dC,M,N,K,start,stop,"S2_swz_blk1024");
    test_config<true, 2, 2048>(dA,dB,dC,M,N,K,start,stop,"S2_swz_blk2048");

    // === Final best combo ===
    printf("\n-- Best Combos --\n");
    test_config<true, 2, 2048>(dA,dB,dC,M,N,K,start,stop,"S2_swz_blk2048");
    test_config<true, 3, 2048>(dA,dB,dC,M,N,K,start,stop,"S3_swz_blk2048");
    test_config<true, 4, 2048>(dA,dB,dC,M,N,K,start,stop,"S4_swz_blk2048");

    // cuBLAS
    {
        cublasHandle_t h; cublasCreate(&h);
        half a=__float2half(1),b=__float2half(0);
        cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB,CUDA_R_16F,N,dA,CUDA_R_16F,K,&b,dC,CUDA_R_16F,N,CUBLAS_COMPUTE_16F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB,CUDA_R_16F,N,dA,CUDA_R_16F,K,&b,dC,CUDA_R_16F,N,CUBLAS_COMPUTE_16F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms,start,stop);
        printf("\n  %-35s %8.3f ms  %6.1f TFLOPS\n","cuBLAS",ms,tflops(M,N,K)/(ms/1000.0));
        cublasDestroy(h);
    }

    cudaFree(dA);cudaFree(dB);cudaFree(dC);
    return 0;
}
