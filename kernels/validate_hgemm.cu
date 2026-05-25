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
#include "hgemm_mma_swizzle.cuh"

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

// New swizzle kernel test (with proper SMEM layout + bounds checks)
float run_hgemm_swizzle(half *dA, half *dB, half *dC, int M, int N, int K,
                         cudaEvent_t start, cudaEvent_t stop) {
    cudaMemset(dC, 0, M * N * sizeof(half));
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    hgemm_swizzle_nn(dA, dB, dC, M, N, K);
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
    half *dA, *dB, *dC_hgemm, *dC_swizzle, *dC_cublas;
    checkCuda(cudaMalloc(&dA, 128 * K * sizeof(half)));
    checkCuda(cudaMalloc(&dB, K * N_full * sizeof(half)));  // largest B
    checkCuda(cudaMalloc(&dC_hgemm, 128 * N_full * sizeof(half)));
    checkCuda(cudaMalloc(&dC_swizzle, 128 * N_full * sizeof(half)));
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

    int old_fails = 0, new_fails = 0;

    // Test all three matrix shape types
    struct { int N; const char *name; } shapes[] = {
        {N_full, "QKV proj (3xH)"},
        {N_attn, "Attn proj (HxH)"},
        {N_ffn,  "FFN  proj (HxIM)"},
    };

    for (int s = 0; s < 3; s++) {
        int N = shapes[s].N;
        printf("\n--- %s (N=%d) ---\n", shapes[s].name, N);
        printf("%-9s %5s %8s %8s %8s %7s\n", "Kernel","M","max_abs","max_rel","errors","ms");

        for (int t = 0; t < num_tests; t++) {
            int M = test_Ms[t];
            cublas_hgemm(dA, dB, dC_cublas, M, N, K, cublas_handle);
            checkCuda(cudaDeviceSynchronize());

            // --- OLD kernel ---
            float ms_o = run_hgemm_kernel<false, 2>(dA, dB, dC_hgemm, M, N, K, start, stop);
            half *hc = (half*)malloc(M*N*sizeof(half)), *ho = (half*)malloc(M*N*sizeof(half));
            checkCuda(cudaMemcpy(hc, dC_cublas, M*N*sizeof(half), cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(ho, dC_hgemm,  M*N*sizeof(half), cudaMemcpyDeviceToHost));
            float ao, ro; int eo;
            compare_results(ho, hc, M, N, &ao, &ro, &eo, 1e-3f);
            printf("%-9s %5d %8.4f %8.4f %8d %6.3f\n", eo?"OLD_FAIL":"OLD_PASS", M, ao, ro, eo, ms_o);
            if (eo) old_fails++; free(hc); free(ho);

            // --- NEW swizzle kernel ---
            float ms_n = run_hgemm_swizzle(dA, dB, dC_swizzle, M, N, K, start, stop);
            hc = (half*)malloc(M*N*sizeof(half));
            half *hn = (half*)malloc(M*N*sizeof(half));
            checkCuda(cudaMemcpy(hc, dC_cublas,  M*N*sizeof(half), cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(hn, dC_swizzle, M*N*sizeof(half), cudaMemcpyDeviceToHost));
            float an, rn; int en;
            compare_results(hn, hc, M, N, &an, &rn, &en, 1e-3f);
            printf("%-9s %5d %8.4f %8.4f %8d %6.3f\n", en?"NEW_FAIL":"NEW_PASS", M, an, rn, en, ms_n);
            if (en) new_fails++; free(hc); free(hn);
        }
    }

    printf("\n========================================\n");
    printf("OLD kernel: %d/24 failed\n", old_fails);
    printf("NEW kernel: %d/24 failed\n", new_fails);
    printf(new_fails ? "NEW: still has issues\n" : "NEW: ALL 24 TESTS PASSED!\n");

    // Summary
    printf("\n========================================\n");
    // Cleanup
    cublasDestroy(cublas_handle);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB);
    cudaFree(dC_hgemm); cudaFree(dC_swizzle); cudaFree(dC_cublas);
    free(hA); free(hB);

    return new_fails > 0 ? 1 : 0;
}
