/**
 * bench_hgemm.cu — HGEMM Correctness & Performance
 * ================================================
 * Compares: naive | tiled | async vs cuBLAS
 *
 * Usage:
 *   nvcc -arch=sm_89 -O3 bench_hgemm.cu -o bench_hgemm -lcublas
 *   ./bench_hgemm M N K
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hgemm.cuh"

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s:%d\n", cudaGetErrorString(code), file, line);
        exit(1);
    }
}

double tflops_hgemm(int M, int N, int K) {
    return 2.0 * (double)M * (double)N * (double)K / 1e12;
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("=== HGEMM Benchmark: M=%d N=%d K=%d ===\n", M, N, K);
    double total_tflops = tflops_hgemm(M, N, K);

    // Allocate host
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    half *h_C = (half*)malloc(M * N * sizeof(half));

    srand(42);
    for (int i = 0; i < M*K; i++) h_A[i] = __float2half((rand()%1000)/1000.0f - 0.5f);
    for (int i = 0; i < K*N; i++) h_B[i] = __float2half((rand()%1000)/1000.0f - 0.5f);

    // Allocate device
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));

    cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(half), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms;

    // === cuBLAS (reference for correctness + perf) ===
    cublasHandle_t handle;
    cublasCreate(&handle);
    half alpha = __float2half(1.0f), beta = __float2half(0.0f);

    cudaMemset(d_C, 0, M*N*sizeof(half));
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K, &alpha, d_B, CUDA_R_16F, N,
                 d_A, CUDA_R_16F, K, &beta, d_C, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K, &alpha, d_B, CUDA_R_16F, N,
                 d_A, CUDA_R_16F, K, &beta, d_C, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("cuBLAS:        %8.3f ms  %6.1f TFLOPS\n", ms, total_tflops/(ms/1000.0));

    // Save reference
    half *h_C_ref = (half*)malloc(M * N * sizeof(half));
    cudaMemcpy(h_C_ref, d_C, M*N*sizeof(half), cudaMemcpyDeviceToHost);

    auto check_err = [&](half *d_out, const char *name) {
        half *h_out = (half*)malloc(M*N*sizeof(half));
        cudaMemcpy(h_out, d_out, M*N*sizeof(half), cudaMemcpyDeviceToHost);
        float max_err = 0, sum_err = 0;
        for (int i = 0; i < M*N; i++) {
            float e = fabsf(__half2float(h_out[i]) - __half2float(h_C_ref[i]));
            if (e > max_err) max_err = e;
            sum_err += e;
        }
        printf("  %-20s max_err=%.4f mean=%.4f %s\n",
               name, max_err, sum_err/(M*N), (max_err < 2.0f) ? "PASS" : "FAIL");
        free(h_out);
    };

    int num_M_tiles = (M + 15) / 16;
    int num_N_tiles = (N + 7) / 8;
    dim3 grid_naive(num_N_tiles, num_M_tiles);

    // === Naive (BM=16, BN=8) ===
    cudaMemset(d_C, 0, M*N*sizeof(half));
    hgemm_naive_kernel<1,1><<<grid_naive, WARP_SIZE>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    hgemm_naive_kernel<1,1><<<grid_naive, WARP_SIZE>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("naive (16x8):  %8.3f ms  %6.1f TFLOPS\n", ms, total_tflops/(ms/1000.0));
    check_err(d_C, "naive");

    // === Tiled (BM=64, BN=64, 4 warps) ===
    {
        num_M_tiles = (M + 63) / 64;
        num_N_tiles = (N + 63) / 64;
        dim3 g(num_N_tiles, num_M_tiles);
        dim3 b(WARP_SIZE * 4);
        cudaMemset(d_C, 0, M*N*sizeof(half));
        hgemm_tiled_kernel<64,64,4><<<g, b>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        hgemm_tiled_kernel<64,64,4><<<g, b>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf("tiled (64x64): %8.3f ms  %6.1f TFLOPS\n", ms, total_tflops/(ms/1000.0));
        check_err(d_C, "tiled");
    }

    // === Async (BM=128, BN=128, 8 warps, cp.async) ===
    {
        num_M_tiles = (M + 127) / 128;
        num_N_tiles = (N + 127) / 128;
        dim3 g(num_N_tiles, num_M_tiles);
        dim3 b(WARP_SIZE * 8);
        constexpr int A_sz = 128*16, B_sz = 16*128;
        int smem = 2 * (A_sz + B_sz) * sizeof(half);
        auto fn = hgemm_async_kernel<128,128,8>;
        checkCuda(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        cudaMemset(d_C, 0, M*N*sizeof(half));
        fn<<<g, b, smem>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        fn<<<g, b, smem>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf("async(128x128):%8.3f ms  %6.1f TFLOPS\n", ms, total_tflops/(ms/1000.0));
        check_err(d_C, "async");
    }

    // === Final (128x128, 2×4 warp grid, K32 reg buffer, shuffle store) ===
    {
        num_M_tiles = (M + 127) / 128;
        num_N_tiles = (N + 127) / 128;
        dim3 g(num_N_tiles, num_M_tiles);
        dim3 b(WARP_SIZE * 8);
        auto fn = hgemm_final_kernel<128, 128, 2, 4, 4, 4, 2>;
        constexpr int A_sz = 128 * 32, B_sz = 32 * 128;
        int smem = 2 * (A_sz + B_sz) * sizeof(half);
        checkCuda(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        cudaMemset(d_C, 0, M*N*sizeof(half));
        fn<<<g, b, smem>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        fn<<<g, b, smem>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf("final(128x128):%8.3f ms  %6.1f TFLOPS\n", ms, total_tflops/(ms/1000.0));
        check_err(d_C, "final");
    }

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return 0;
}
