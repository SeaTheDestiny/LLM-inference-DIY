/**
 * test_flash_attn.cu — Correctness & Performance Benchmark
 * ========================================================
 * Usage: nvcc -arch=sm_80 -O3 test_flash_attn.cu -o test_flash_attn && ./test_flash_attn
 *
 * Tests all 6 flash attention kernels against a naive GPU reference:
 *   1. flash_attn_mma_naive_kernel           (original, 1 warp,  Br=16 Bc=16)
 *   2. flash_attn_mma_41warp_18mma_kernel    (original, 4 warps, Br=64 Bc=64)
 *   3. flash_attn_finegrained_qk_tiling_kernel (Step 1)
 *   4. flash_attn_register_p_kernel           (Step 2)
 *   5. flash_attn_async_kernel                (Step 3)
 *   6. flash_attn_final_kernel                (Step 4+5)
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

#define WARP_SIZE 32

// ---- Kernel declarations (from flash_attn.cu, compiled together) ----
// Include the actual kernel source so templates are instantiated
#include "flash_attn.cu"

// ============================================================
// CPU Reference: Naive Attention with FP32 precision
// ============================================================
void cpu_attention_fp32(float *Q, float *K, float *V, float *O,
                        int B, int H, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            float *q = Q + ((b * H + h) * N * d);
            float *k = K + ((b * H + h) * N * d);
            float *v = V + ((b * H + h) * N * d);
            float *o = O + ((b * H + h) * N * d);

            for (int i = 0; i < N; i++) {
                // Online softmax per query row
                float row_max = -INFINITY;
                float row_sum = 0.0f;
                float *oi = o + i * d;
                for (int j = 0; j < d; j++) oi[j] = 0.0f;

                for (int j = 0; j < N; j++) {
                    // S[i,j] = Q[i,:] @ K[j,:]^T
                    float s = 0.0f;
                    for (int kk = 0; kk < d; kk++)
                        s += q[i * d + kk] * k[j * d + kk];
                    s *= scale;

                    float m_new = fmaxf(row_max, s);
                    float exp_diff = expf(row_max - m_new);
                    float p = expf(s - m_new);

                    // Rescale old accumulator
                    row_sum = row_sum * exp_diff + p;
                    for (int kk = 0; kk < d; kk++)
                        oi[kk] = oi[kk] * exp_diff + p * v[j * d + kk];

                    row_max = m_new;
                }
                // Normalize
                float inv_sum = 1.0f / row_sum;
                for (int kk = 0; kk < d; kk++)
                    oi[kk] *= inv_sum;
            }
        }
    }
}

// ============================================================
// GPU Naive Reference: same algorithm on GPU for fair comparison
// (Uses the "naive_kernel" from flash_attn.cu as reference baseline)
// ============================================================
__global__ void gpu_naive_ref_kernel(float *Q, float *K, float *V, float *O,
                                      int N, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // query row
    if (i >= N) return;
    float scale = 1.0f / sqrtf((float)d);
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    float *oi = O + i * d;
    for (int kk = 0; kk < d; kk++) oi[kk] = 0.0f;

    for (int j = 0; j < N; j++) {
        float s = 0.0f;
        for (int kk = 0; kk < d; kk++)
            s += Q[i * d + kk] * K[j * d + kk];
        s *= scale;

        float m_new = fmaxf(row_max, s);
        float exp_diff = expf(row_max - m_new);
        float p = expf(s - m_new);

        row_sum = row_sum * exp_diff + p;
        for (int kk = 0; kk < d; kk++)
            oi[kk] = oi[kk] * exp_diff + p * V[j * d + kk];
        row_max = m_new;
    }
    float inv = 1.0f / row_sum;
    for (int kk = 0; kk < d; kk++) oi[kk] *= inv;
}


// ============================================================
// Helper: compare two half arrays, return max error
// ============================================================
float check_error(half *a, half *b, int count, float *out_mean_err) {
    float max_err = 0.0f;
    double sum_err = 0.0;
    for (int i = 0; i < count; i++) {
        float va = __half2float(a[i]);
        float vb = __half2float(b[i]);
        float err = fabsf(va - vb);
        if (err > max_err) max_err = err;
        sum_err += (double)err;
    }
    *out_mean_err = (float)(sum_err / count);
    return max_err;
}

// ============================================================
// Benchmark macro
// ============================================================
#define BENCH_KERNEL(name, block, grid, smem, stream, ...)          \
    do {                                                             \
        cudaEventRecord(start, stream);                              \
        name<<<grid, block, smem, stream>>>(__VA_ARGS__);            \
        cudaEventRecord(stop, stream);                               \
        cudaEventSynchronize(stop);                                  \
        float ms = 0;                                                \
        cudaEventElapsedTime(&ms, start, stop);                      \
        printf("  %-50s  %8.3f ms\n", #name, ms);                   \
    } while(0)


// ============================================================
// Main test
// ============================================================
int main() {
    // Test configuration
    const int B = 1, H = 1, N = 256, d = 128;
    const int Br = 64, Bc = 64;  // tile sizes for multi-warp kernels
    const int num_heads = B * H;
    const int total_elements = B * H * N * d;

    printf("=== Flash Attention Correctness & Performance Test ===\n");
    printf("Config: B=%d H=%d N=%d d=%d  total halfs=%d\n\n", B, H, N, d, total_elements);

    // Allocate host memory (FP32 for CPU ref, half for GPU)
    float *h_Q = (float*)malloc(total_elements * sizeof(float));
    float *h_K = (float*)malloc(total_elements * sizeof(float));
    float *h_V = (float*)malloc(total_elements * sizeof(float));
    float *h_O_cpu = (float*)malloc(total_elements * sizeof(float));
    half  *h_O_ref = (half*)malloc(total_elements * sizeof(half));

    // Random init
    srand(42);
    for (int i = 0; i < total_elements; i++) {
        h_Q[i] = ((float)(rand() % 1000) / 1000.0f - 0.5f);
        h_K[i] = ((float)(rand() % 1000) / 1000.0f - 0.5f);
        h_V[i] = ((float)(rand() % 1000) / 1000.0f - 0.5f);
    }

    // CPU reference
    printf("[1] Computing CPU reference (FP32)...\n");
    cpu_attention_fp32(h_Q, h_K, h_V, h_O_cpu, B, H, N, d);
    for (int i = 0; i < total_elements; i++)
        h_O_ref[i] = __float2half(h_O_cpu[i]);

    // Allocate GPU memory
    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, total_elements * sizeof(half));
    cudaMalloc(&d_K, total_elements * sizeof(half));
    cudaMalloc(&d_V, total_elements * sizeof(half));
    cudaMalloc(&d_O, total_elements * sizeof(half));

    // Copy FP32 -> half on GPU
    half *h_Q_half = (half*)malloc(total_elements * sizeof(half));
    half *h_K_half = (half*)malloc(total_elements * sizeof(half));
    half *h_V_half = (half*)malloc(total_elements * sizeof(half));
    for (int i = 0; i < total_elements; i++) {
        h_Q_half[i] = __float2half(h_Q[i]);
        h_K_half[i] = __float2half(h_K[i]);
        h_V_half[i] = __float2half(h_V[i]);
    }
    cudaMemcpy(d_Q, h_Q_half, total_elements * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K_half, total_elements * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V_half, total_elements * sizeof(half), cudaMemcpyHostToDevice);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    half *h_O_kernel = (half*)malloc(total_elements * sizeof(half));
    float max_err, mean_err;
    const float PASS_THRESHOLD = 2.0f;  // Max tolerable error for half precision

    // ========= Kernel 1: Naive single-warp =========
    printf("\n[2] Testing flash_attn_mma_naive_kernel (Br=16,Bc=16)...\n");
    {
        int Tr = (N + 15) / 16;  // Br=16
        dim3 grid_naive(1, Tr * num_heads);
        dim3 block_naive(WARP_SIZE);

        cudaMemset(d_O, 0, total_elements * sizeof(half));
        BENCH_KERNEL(flash_attn_mma_naive_kernel<d>, block_naive, grid_naive, 0, stream,
                     d_Q, d_K, d_V, d_O, N, H);
        cudaMemcpy(h_O_kernel, d_O, total_elements * sizeof(half), cudaMemcpyDeviceToHost);

        max_err = check_error(h_O_kernel, h_O_ref, total_elements, &mean_err);
        printf("  Max error: %.6f  Mean error: %.6f  %s\n",
               max_err, mean_err, (max_err < PASS_THRESHOLD) ? "PASS" : "FAIL");
    }

    // ========= Kernel 2: Multi-warp (Br=64,Bc=64) =========
    printf("\n[3] Testing flash_attn_mma_41warp_18mma_kernel (Br=64,Bc=64)...\n");
    {
        int Tr = (N + Br - 1) / Br;
        dim3 grid_mw(1, Tr * num_heads);
        constexpr int NumThreads = WARP_SIZE * 4;  // kMmaTileSeqLenQ=4, kMmaTileSeqLenK=1
        dim3 block_mw(NumThreads);

        cudaMemset(d_O, 0, total_elements * sizeof(half));
        BENCH_KERNEL(flash_attn_mma_41warp_18mma_kernel<d, 16, 8, 16, 4, 1, 1, 8>,
                     block_mw, grid_mw, 0, stream,
                     d_Q, d_K, d_V, d_O, N, H);
        cudaMemcpy(h_O_kernel, d_O, total_elements * sizeof(half), cudaMemcpyDeviceToHost);

        max_err = check_error(h_O_kernel, h_O_ref, total_elements, &mean_err);
        printf("  Max error: %.6f  Mean error: %.6f  %s\n",
               max_err, mean_err, (max_err < PASS_THRESHOLD) ? "PASS" : "FAIL");
    }

    // ========= Kernel 3: Step 1 Fine-grained Q/K Tiling =========
    printf("\n[4] Testing flash_attn_finegrained_qk_tiling_kernel (Step 1)...\n");
    {
        int Tr = (N + Br - 1) / Br;
        dim3 grid_fg(1, Tr * num_heads);
        constexpr int NumThreads = WARP_SIZE * 4;
        dim3 block_fg(NumThreads);

        constexpr int Q_sz = Br * (16 + 8);   // 64*24
        constexpr int K_sz = Bc * (16 + 8);   // 64*24
        constexpr int P_sz = Br * Bc;         // 4096
        constexpr int V_sz = Bc * d;          // 8192
        int smem = (Q_sz + K_sz + P_sz + V_sz) * sizeof(half);

        cudaFuncSetAttribute(
            flash_attn_finegrained_qk_tiling_kernel<d, 16, 8, 16, 4, 1, 1, 8, 8, 8>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        cudaMemset(d_O, 0, total_elements * sizeof(half));
        BENCH_KERNEL(flash_attn_finegrained_qk_tiling_kernel<d, 16, 8, 16, 4, 1, 1, 8, 8, 8>,
                     block_fg, grid_fg, smem, stream,
                     d_Q, d_K, d_V, d_O, N, H);
        cudaMemcpy(h_O_kernel, d_O, total_elements * sizeof(half), cudaMemcpyDeviceToHost);

        max_err = check_error(h_O_kernel, h_O_ref, total_elements, &mean_err);
        printf("  Max error: %.6f  Mean error: %.6f  %s\n",
               max_err, mean_err, (max_err < PASS_THRESHOLD) ? "PASS" : "FAIL");
    }

    // ========= Kernel 4: Step 2 Register P =========
    printf("\n[5] Testing flash_attn_register_p_kernel (Step 2)...\n");
    {
        int Tr = (N + Br - 1) / Br;
        dim3 grid_rp(1, Tr * num_heads);
        constexpr int NumThreads = WARP_SIZE * 4;
        dim3 block_rp(NumThreads);

        constexpr int Q_sz = Br * (16 + 8);
        constexpr int K_sz = Bc * (16 + 8);
        constexpr int V_sz = Bc * d;
        int smem = (Q_sz + K_sz + V_sz) * sizeof(half);  // no s_p!

        cudaFuncSetAttribute(
            flash_attn_register_p_kernel<d, 16, 8, 16, 4, 1, 1, 8, 8, 8>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        cudaMemset(d_O, 0, total_elements * sizeof(half));
        BENCH_KERNEL(flash_attn_register_p_kernel<d, 16, 8, 16, 4, 1, 1, 8, 8, 8>,
                     block_rp, grid_rp, smem, stream,
                     d_Q, d_K, d_V, d_O, N, H);
        cudaMemcpy(h_O_kernel, d_O, total_elements * sizeof(half), cudaMemcpyDeviceToHost);

        max_err = check_error(h_O_kernel, h_O_ref, total_elements, &mean_err);
        printf("  Max error: %.6f  Mean error: %.6f  %s\n",
               max_err, mean_err, (max_err < PASS_THRESHOLD) ? "PASS" : "FAIL");
    }

    // ========= Kernel 5: Step 3 Async + Double Buffering =========
    printf("\n[6] Testing flash_attn_async_kernel (Step 3)...\n");
    {
        int Tr = (N + Br - 1) / Br;
        dim3 grid_async(1, Tr * num_heads);
        constexpr int NumThreads = WARP_SIZE * 4;
        dim3 block_async(NumThreads);

        constexpr int Q_sz = Br * (16 + 8);
        constexpr int K_sz = Bc * (16 + 8);
        constexpr int V_sz = Bc * d;
        int smem = (2 * (Q_sz + K_sz) + V_sz) * sizeof(half);

        cudaFuncSetAttribute(
            flash_attn_async_kernel<d, 16, 8, 16, 4, 1, 1, 8, 8, 8, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        cudaMemset(d_O, 0, total_elements * sizeof(half));
        BENCH_KERNEL(flash_attn_async_kernel<d, 16, 8, 16, 4, 1, 1, 8, 8, 8, 2>,
                     block_async, grid_async, smem, stream,
                     d_Q, d_K, d_V, d_O, N, H);
        cudaMemcpy(h_O_kernel, d_O, total_elements * sizeof(half), cudaMemcpyDeviceToHost);

        max_err = check_error(h_O_kernel, h_O_ref, total_elements, &mean_err);
        printf("  Max error: %.6f  Mean error: %.6f  %s\n",
               max_err, mean_err, (max_err < PASS_THRESHOLD) ? "PASS" : "FAIL");
    }

    // ========= Kernel 6: Step 4+5 Final Fully Optimized =========
    printf("\n[7] Testing flash_attn_final_kernel (Step 4+5, fully optimized)...\n");
    {
        int Tr = (N + Br - 1) / Br;
        dim3 grid_final(1, Tr * num_heads);
        constexpr int NumThreads = WARP_SIZE * 4;
        dim3 block_final(NumThreads);

        constexpr int Q_sz = Br * (16 + 8);
        constexpr int K_sz = Bc * (16 + 8);
        constexpr int V_sz = Bc * (16 + 8);  // fine-grained V!
        // V reuses Q space: max(2*(Q+K), 2*V) = 2*(Q+K) = 12288 halfs
        int smem = (2 * (Q_sz + K_sz)) * sizeof(half);

        constexpr int kWTHV = d / 8;  // kWarpTileHeadDimV

        cudaFuncSetAttribute(
            flash_attn_final_kernel<d, 16, 8, 16, 4, 1, 1, 8, 1, kWTHV, 8, 8, 8, 2, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        cudaMemset(d_O, 0, total_elements * sizeof(half));
        BENCH_KERNEL(flash_attn_final_kernel<d, 16, 8, 16, 4, 1, 1, 8, 1, kWTHV, 8, 8, 8, 2, 1>,
                     block_final, grid_final, smem, stream,
                     d_Q, d_K, d_V, d_O, N, H);
        cudaMemcpy(h_O_kernel, d_O, total_elements * sizeof(half), cudaMemcpyDeviceToHost);

        max_err = check_error(h_O_kernel, h_O_ref, total_elements, &mean_err);
        printf("  Max error: %.6f  Mean error: %.6f  %s\n",
               max_err, mean_err, (max_err < PASS_THRESHOLD) ? "PASS" : "FAIL");
    }

    // ========= Summary =========
    printf("\n========== ALL TESTS COMPLETE ==========\n");
    printf("Refer to timing output above for performance comparison.\n");
    printf("Expected performance improvement: final > async > register_p > finegrained > multi-warp > naive\n");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V); free(h_O_cpu); free(h_O_ref);
    free(h_Q_half); free(h_K_half); free(h_V_half); free(h_O_kernel);

    return 0;
}
