#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hgemm_mma_swizzle.cuh"

int main() {
    int M=128, N=128, K=16;
    half *hA=(half*)malloc(M*K*sizeof(half)),*hB=(half*)malloc(K*N*sizeof(half));
    half *dA,*dB,*dC_sw,*dC_cb;
    cudaMalloc(&dA,M*K*sizeof(half)); cudaMalloc(&dB,K*N*sizeof(half));
    cudaMalloc(&dC_sw,M*N*sizeof(half)); cudaMalloc(&dC_cb,M*N*sizeof(half));
    srand(42);
    for(int i=0;i<M*K;i++) hA[i]=__float2half(((float)rand()/RAND_MAX-0.5f)*2.0f);
    for(int i=0;i<K*N;i++) hB[i]=__float2half(((float)rand()/RAND_MAX-0.5f)*2.0f);
    cudaMemcpy(dA,hA,M*K*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,K*N*sizeof(half),cudaMemcpyHostToDevice);
    
    cublasHandle_t h; cublasCreate(&h);
    half alpha=__float2half(1),beta=__float2half(0);
    cublasHgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,dB,N,dA,K,&beta,dC_cb,N);
    cudaDeviceSynchronize();
    hgemm_swizzle_nn(dA,dB,dC_sw,M,N,K);
    cudaDeviceSynchronize();
    
    half *r_cb=(half*)malloc(M*N*sizeof(half)),*r_sw=(half*)malloc(M*N*sizeof(half));
    cudaMemcpy(r_cb,dC_cb,M*N*sizeof(half),cudaMemcpyDeviceToHost);
    cudaMemcpy(r_sw,dC_sw,M*N*sizeof(half),cudaMemcpyDeviceToHost);
    
    float max_err=0; int errs=0;
    for(int i=0;i<M*N;i++){
        float d=fabsf(__half2float(r_cb[i])-__half2float(r_sw[i]));
        if(d>max_err)max_err=d;
        if(d>1e-3f)errs++;
    }
    printf("M=%d N=%d K=%d: max_err=%.6f errors=%d/%d\n",M,N,K,max_err,errs,M*N);
    printf("%s\n",errs?"FAIL":"PASS");
    
    // Also test K=32 (two K-tiles)
    K=32; free(hB); hB=(half*)malloc(K*N*sizeof(half)); cudaFree(dB); cudaMalloc(&dB,K*N*sizeof(half));
    for(int i=0;i<K*N;i++) hB[i]=__float2half(((float)rand()/RAND_MAX-0.5f)*2.0f);
    cudaMemcpy(dB,hB,K*N*sizeof(half),cudaMemcpyHostToDevice);
    cublasHgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,dB,N,dA,K,&beta,dC_cb,N);
    cudaDeviceSynchronize();
    hgemm_swizzle_nn(dA,dB,dC_sw,M,N,K);
    cudaDeviceSynchronize();
    cudaMemcpy(r_cb,dC_cb,M*N*sizeof(half),cudaMemcpyDeviceToHost);
    cudaMemcpy(r_sw,dC_sw,M*N*sizeof(half),cudaMemcpyDeviceToHost);
    max_err=0;errs=0;
    for(int i=0;i<M*N;i++){
        float d=fabsf(__half2float(r_cb[i])-__half2float(r_sw[i]));
        if(d>max_err)max_err=d;
        if(d>1e-3f)errs++;
    }
    printf("M=%d N=%d K=%d: max_err=%.6f errors=%d/%d\n",M,N,K,max_err,errs,M*N);
    printf("%s\n",errs?"FAIL":"PASS");
    
    cublasDestroy(h);
    return 0;
}
