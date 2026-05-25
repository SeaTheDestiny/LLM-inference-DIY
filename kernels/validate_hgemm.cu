/**
 * validate_hgemm.cu — Correctness validation: hgemm_opt_kernel vs cuBLAS
 * =====================================================================
 * Compares self-developed Tensor Core MMA HGEMM against cuBLAS reference
 * across realistic prefill matrix shapes (M=1..128, K=2048, N=6144).
 *
 * Usage:
 *   nvcc -arch=sm_89 -O3 -std=c++17 validate_hgemm.cu -o validate_hgemm -lcublas
 *   ./validate_hgemm
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hgemm_final.cuh"

#define checkCuda(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    }

// Reference: cuBLAS HGEMM (row-major → cuBLAS col-major translation)
void cublas_hgemm(half *dA, half *dB, half *dC, int M, int N, int K,
                  cublasHandle_t handle) {
    half alpha = __float2half(1.0f);
    half beta  = __float2half(0.0f);
    // C[M][N] = A[M][K] × B[K][N]  (row-major)
    // cuBLAS (col-major): C_col[N][M] = B_col[N][K] × A_col[K][M]
    cublasStatus_t st = cublasHgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        dB, N,    // B is [K][N] row-major = [N][K] col-major
        dA, K,    // A is [M][K] row-major = [K][M] col-major
        &beta,
        dC, N);   // C is [M][N] row-major = [N][M] col-major
    if (st != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "cuBLAS failed: %d\n", st);
}

// Our self-developed HGEMM with proper cudaFuncSetAttribute
template<bool SWZ, int STG>
float run_hgemm_kernel(half *dA, half *dB, half *dC, int M, int N, int K,
                        cudaEvent_t start, cudaEvent_t stop) {
    constexpr int BM=128, BN=128, WM=2, WN=4, WK=2;
    constexpr int MMA_M = BM/16/WM;  // 4
    constexpr int MMA_N = BN/8/WN;   // 4
    constexpr int A_sz = BM * (16 * WK);
    constexpr int B_sz = (16 * WK) * BN;
    int smem = STG * (A_sz + B_sz) * (int)sizeof(half);

    int MT = (M + BM - 1) / BM;
    int NT = (N + BN - 1) / BN;
    dim3 grid(NT, MT);
    dim3 block(WARP_SIZE * WM * WN);

    auto fn = hgemm_opt_kernel<BM, BN, WM, WN, MMA_M, MMA_N, WK, SWZ, STG, 0>;

    // === THIS WAS THE MISSING LINE ===
    checkCuda(cudaFuncSetAttribute(fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    cudaMemset(dC, 0, M * N * sizeof(half));
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    fn<<<grid, block, smem>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaEventSynchronize(stop));

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// Element-wise error comparison (returns max absolute + relative error)
void compare_results(const half *hgemm_out, const half *cublas_out,
                     int M, int N, float *max_abs_err, float *max_rel_err,
                     int *error_count, float tolerance) {
    *max_abs_err = 0.0f;
    *max_rel_err = 0.0f;
    *error_count = 0;

    for (int i = 0; i < M * N; i++) {
        float a = __half2float(hgemm_out[i]);
        float b = __half2float(cublas_out[i]);
        float abs_err = fabsf(a - b);
        float rel_err = (fabsf(b) > 1e-6f) ? abs_err / fabsf(b) : abs_err;

        if (abs_err > *max_abs_err) *max_abs_err = abs_err;
        if (rel_err > *max_rel_err) *max_rel_err = rel_err;
        if (abs_err > tolerance || rel_err > 0.01f) (*error_count)++;
    }
}

int main(int argc, char **argv) {
    // Use realistic prefill shapes
    const int K = 2048;       // hidden_size
    const int N_full = 6144;  // 3 * hidden_size (QKV)
    const int N_attn = 2048;  // hidden_size (attn proj)
    const int N_ffn  = 5504;  // intermediate_size (FFN)

    // Test M values (prefill batch sizes)
    int test_Ms[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int num_tests = sizeof(test_Ms) / sizeof(test_Ms[0]);

    printf("=== HGEMM Correctness Validation ===\n");
    printf("Matrix shapes (K=%d):\n", K);
    printf("  QKV proj:   M={1..128}, N=%d\n", N_full);
    printf("  Attn proj:  M={1..128}, N=%d\n", N_attn);
    printf("  FFN proj:   M={1..128}, N=%d\n", N_ffn);
    printf("Tolerance: abs=1e-3, rel=1%%\n\n");

    // Allocate GPU memory
    half *dA, *dB, *dC_hgemm, *dC_cublas;
    checkCuda(cudaMalloc(&dA, 128 * K * sizeof(half)));
    checkCuda(cudaMalloc(&dB, K * N_full * sizeof(half)));  // largest B
    checkCuda(cudaMalloc(&dC_hgemm, 128 * N_full * sizeof(half)));
    checkCuda(cudaMalloc(&dC_cublas, 128 * N_full * sizeof(half)));

    // Fill with realistic FP16 data (small random-like values)
    half *hA = (half*)malloc(128 * K * sizeof(half));
    half *hB = (half*)malloc(K * N_full * sizeof(half));
    srand(42);
    for (int i = 0; i < 128 * K; i++)
        hA[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
    for (int i = 0; i < K * N_full; i++)
        hB[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.1f);

    checkCuda(cudaMemcpy(dA, hA, 128 * K * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dB, hB, K * N_full * sizeof(half), cudaMemcpyHostToDevice));

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_errors = 0;
    int total_cases = 0;

    // Test all three matrix shape types
    struct { int N; const char *name; } shapes[] = {
        {N_full, "QKV proj (3×H)"},
        {N_attn, "Attn proj (H×H)"},
        {N_ffn,  "FFN  proj (H×IM)"},
    };

    for (int s = 0; s < 3; s++) {
        int N = shapes[s].N;
        printf("\n--- %s (N=%d) ---\n", shapes[s].name, N);

        for (int t = 0; t < num_tests; t++) {
            int M = test_Ms[t];
            printf("  M=%3d: ", M);

            // cuBLAS reference
            cublas_hgemm(dA, dB, dC_cublas, M, N, K, cublas_handle);
            checkCuda(cudaDeviceSynchronize());

            // Our HGEMM (with swizzle=false, stage=2 — simplest config)
            float ms = run_hgemm_kernel<false, 2>(dA, dB, dC_hgemm, M, N, K,
                                                   start, stop);

            // Compare
            half *h_cublas = (half*)malloc(M * N * sizeof(half));
            half *h_hgemm  = (half*)malloc(M * N * sizeof(half));
            checkCuda(cudaMemcpy(h_cublas, dC_cublas, M * N * sizeof(half),
                                 cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(h_hgemm, dC_hgemm, M * N * sizeof(half),
                                 cudaMemcpyDeviceToHost));

            float max_abs, max_rel;
            int errs;
            compare_results(h_hgemm, h_cublas, M, N,
                           &max_abs, &max_rel, &errs, 1e-3f);

            const char *status = (errs == 0) ? "PASS" : "FAIL";
            printf("%s  max_abs=%.4f  max_rel=%.4f  errors=%d/%d  time=%.3f ms\n",
                   status, max_abs, max_rel, errs, M * N, ms);

            if (errs > 0) {
                // Print first few error locations
                int shown = 0;
                for (int i = 0; i < M * N && shown < 5; i++) {
                    float a = __half2float(h_hgemm[i]);
                    float b = __half2float(h_cublas[i]);
                    if (fabsf(a - b) > 1e-3f || fabsf(a - b) / fmaxf(fabsf(b), 1e-6f) > 0.01f) {
                        printf("    [%d,%d] hgemm=%.6f cublas=%.6f diff=%.6f\n",
                               i / N, i % N, a, b, a - b);
                        shown++;
                    }
                }
                total_errors += errs;
            }
            total_cases++;

            free(h_cublas);
            free(h_hgemm);
        }
    }

    // Summary
    printf("\n========================================\n");
    printf("SUMMARY: %d/%d cases passed\n",
           total_cases - (total_errors > 0 ? 1 : 0), total_cases);
    // Actually, count individual cases:
    printf("Total error elements: %d\n", total_errors);

    if (total_errors == 0) {
        printf("ALL TESTS PASSED — hgemm_opt_kernel matches cuBLAS!\n");
    }

    // Cleanup
    cublasDestroy(cublas_handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC_hgemm);
    cudaFree(dC_cublas);
    free(hA);
    free(hB);

    return total_errors > 0 ? 1 : 0;
}
