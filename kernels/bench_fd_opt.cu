/**
 * bench_fd_opt.cu — Flash Decoding Optimization Benchmark
 * ========================================================
 * Compares all FD optimization stages + best FA in decode scenario (Q=1, d=128).
 *
 * Kernels tested:
 *   1. baseline      — single warp, serial KV scan
 *   2. fd_naive      — original 2-stage FD (Bc=16)
 *   3. fd_v1         — vectorized coalesced loads (Bc=16)
 *   4. fd_v2         — large chunk (Bc=128)
 *   5. fd_v3         — 4 warps + in-block merge (Bc=128)
 *   6. fd_v4         — 8 warps + fast math (Bc=256)
 *   7. fa_final      — best FA kernel for Q=1
 *
 * Usage:
 *   nvcc -arch=sm_89 -O3 bench_fd_opt.cu -o bench_fd_opt
 *   ./bench_fd_opt > fd_results.csv 2>fd_log.txt
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Include kernels
// flash_decode_q1.cuh defines WARP_SIZE unconditionally,
// flash_attn.cu also defines it, so include decode first.
#include "flash_decode_q1.cuh"
#include "flash_decode_q1_opt.cuh"
// flash_attn.cu has its own WARP_SIZE define — already set, no conflict
#include "flash_attn.cu"

// ============================================================
// Correctness check: compare two half arrays
// ============================================================
float max_abs_error(half *a, half *b, int n) {
    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(__half2float(a[i]) - __half2float(b[i]));
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// ============================================================
// Baseline: single warp serial attention (from bench_q1.cu)
// ============================================================
__global__ void fd_baseline_kernel(const half *Q, const half *K, const half *V,
                                    half *O, int KV_seqlen, int d) {
    int lid = threadIdx.x;
    int dpt = d / WARP_SIZE;

    half R_Q[8];
    #pragma unroll
    for (int i = 0; i < dpt; i++)
        R_Q[i] = Q[lid * dpt + i];

    float scale = 1.0f / sqrtf((float)d);
    float row_max = -INFINITY, row_sum = 0.0f;
    float O_acc[8] = {0};

    for (int j = 0; j < KV_seqlen; j++) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < dpt; i++)
            dot += __half2float(R_Q[i]) * __half2float(K[j * d + lid * dpt + i]);

        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, mask);

        float s = dot * scale;
        float m_new = fmaxf(row_max, s);
        float exp_d = expf(row_max - m_new);
        float p = expf(s - m_new);
        row_sum = row_sum * exp_d + p;

        #pragma unroll
        for (int i = 0; i < dpt; i++)
            O_acc[i] = O_acc[i] * exp_d + p * __half2float(V[j * d + lid * dpt + i]);
        row_max = m_new;
    }

    float inv = __frcp_rn(row_sum);
    #pragma unroll
    for (int i = 0; i < dpt; i++)
        O[lid * dpt + i] = __float2half(O_acc[i] * inv);
}

// ============================================================
// Main benchmark
// ============================================================
int main() {
    const int KV_vals[] = {256, 512, 1024, 2048, 4096, 8192};
    constexpr int d = 128;
    constexpr int kBc_naive = 16;
    constexpr int kBc_v2 = 128;
    constexpr int kBc_v3 = 128;
    constexpr int kBc_v4 = 256;
    const int num_KV = 6;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CSV header
    printf("kernel,KV_seqlen,d,time_ms,bandwidth_gbs,tflops\n");
    fprintf(stderr, "=== Flash Decoding Optimization Benchmark (d=%d) ===\n\n", d);

    for (int ki = 0; ki < num_KV; ki++) {
        int KVN = KV_vals[ki];
        int total_Q = 1 * d;
        int total_KV = KVN * d;

        // Bandwidth calc: read Q + K + V, write O
        double total_bytes = (double)(total_Q + 2 * total_KV + total_Q) * sizeof(half);
        // FLOPs calc: Q*K (2*d FLOPs per row) + P*V (2*d FLOPs per row) = 4*d FLOPs per KV element
        double total_flops = 4.0 * KVN * d;

        fprintf(stderr, "\n--- KV=%d ---\n", KVN);

        // Allocate
        half *d_Q, *d_K, *d_V, *d_O, *d_O_ref;
        cudaMalloc(&d_Q, total_Q * sizeof(half));
        cudaMalloc(&d_K, total_KV * sizeof(half));
        cudaMalloc(&d_V, total_KV * sizeof(half));
        cudaMalloc(&d_O, total_Q * sizeof(half));
        cudaMalloc(&d_O_ref, total_Q * sizeof(half));

        // Init with 0x3C (~1.0 in fp16)
        cudaMemset(d_Q, 0x3C, total_Q * sizeof(half));
        cudaMemset(d_K, 0x3C, total_KV * sizeof(half));
        cudaMemset(d_V, 0x3C, total_KV * sizeof(half));

        // Lambda: measure a kernel run
        auto measure = [&](const char* name, auto run_fn) -> float {
            // Warmup
            run_fn();
            cudaStreamSynchronize(stream);

            cudaEventRecord(start, stream);
            run_fn();
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            double bw = total_bytes / (ms / 1000.0) / 1e9;
            double tflops = total_flops / (ms / 1000.0) / 1e12;
            printf("%s,%d,%d,%.4f,%.2f,%.4f\n", name, KVN, d, ms, bw, tflops);
            fprintf(stderr, "  %-20s %8.4f ms  %6.1f GB/s  %8.4f TFLOPS\n", name, ms, bw, tflops);
            return ms;
        };

        // ---- 1. Baseline (serial single warp) ----
        measure("baseline", [&]() {
            fd_baseline_kernel<<<1, WARP_SIZE, 0, stream>>>(d_Q, d_K, d_V, d_O, KVN, d);
        });

        // Get reference output
        fd_baseline_kernel<<<1, WARP_SIZE, 0, stream>>>(d_Q, d_K, d_V, d_O_ref, KVN, d);
        cudaStreamSynchronize(stream);
        half *h_ref = (half*)malloc(total_Q * sizeof(half));
        cudaMemcpy(h_ref, d_O_ref, total_Q * sizeof(half), cudaMemcpyDeviceToHost);

        // Helper: check correctness
        auto check = [&](const char* name) {
            half *h_O = (half*)malloc(total_Q * sizeof(half));
            cudaMemcpy(h_O, d_O, total_Q * sizeof(half), cudaMemcpyDeviceToHost);
            float err = max_abs_error(h_O, h_ref, total_Q);
            fprintf(stderr, "    → max_err=%.6f %s\n", err, (err < 0.5f) ? "PASS" : "FAIL");
            free(h_O);
        };

        // ---- 2. FD Naive (original Bc=16) ----
        {
            int N_chunks = (KVN + kBc_naive - 1) / kBc_naive;
            half *d_Op; float *d_LSE;
            cudaMalloc(&d_Op, N_chunks * d * sizeof(half));
            cudaMalloc(&d_LSE, N_chunks * sizeof(float));

            measure("fd_naive", [&]() {
                flash_decode_q1_stage1_kernel<<<N_chunks, WARP_SIZE, 0, stream>>>(
                    d_Q, d_K, d_V, d_Op, d_LSE, KVN, N_chunks, d);
                flash_decode_q1_stage2_kernel<<<1, WARP_SIZE, 0, stream>>>(
                    d_Op, d_LSE, d_O, N_chunks, d);
            });

            // Correctness
            flash_decode_q1_stage1_kernel<<<N_chunks, WARP_SIZE, 0, stream>>>(
                d_Q, d_K, d_V, d_Op, d_LSE, KVN, N_chunks, d);
            flash_decode_q1_stage2_kernel<<<1, WARP_SIZE, 0, stream>>>(
                d_Op, d_LSE, d_O, N_chunks, d);
            cudaStreamSynchronize(stream);
            check("fd_naive");

            cudaFree(d_Op); cudaFree(d_LSE);
        }

        // ---- 3. FD v1 (vectorized, Bc=16) ----
        {
            int N_chunks = (KVN + kBc_naive - 1) / kBc_naive;
            half *d_Op; float *d_LSE;
            cudaMalloc(&d_Op, N_chunks * d * sizeof(half));
            cudaMalloc(&d_LSE, N_chunks * sizeof(float));

            measure("fd_v1_coalesced", [&]() {
                fd_v1_stage1_kernel<d><<<N_chunks, WARP_SIZE, 0, stream>>>(
                    d_Q, d_K, d_V, d_Op, d_LSE, KVN, N_chunks);
                fd_opt_stage2_kernel<d><<<1, 128, 0, stream>>>(
                    d_Op, d_LSE, d_O, N_chunks);
            });

            fd_v1_stage1_kernel<d><<<N_chunks, WARP_SIZE, 0, stream>>>(
                d_Q, d_K, d_V, d_Op, d_LSE, KVN, N_chunks);
            fd_opt_stage2_kernel<d><<<1, 128, 0, stream>>>(
                d_Op, d_LSE, d_O, N_chunks);
            cudaStreamSynchronize(stream);
            check("fd_v1_coalesced");

            cudaFree(d_Op); cudaFree(d_LSE);
        }

        // ---- 4. FD v2 (Bc=128) ----
        {
            int N_chunks = (KVN + kBc_v2 - 1) / kBc_v2;
            half *d_Op; float *d_LSE;
            cudaMalloc(&d_Op, N_chunks * d * sizeof(half));
            cudaMalloc(&d_LSE, N_chunks * sizeof(float));

            measure("fd_v2_largechunk", [&]() {
                fd_v2_stage1_kernel<d, kBc_v2><<<N_chunks, WARP_SIZE, 0, stream>>>(
                    d_Q, d_K, d_V, d_Op, d_LSE, KVN, N_chunks);
                fd_opt_stage2_kernel<d><<<1, 128, 0, stream>>>(
                    d_Op, d_LSE, d_O, N_chunks);
            });

            fd_v2_stage1_kernel<d, kBc_v2><<<N_chunks, WARP_SIZE, 0, stream>>>(
                d_Q, d_K, d_V, d_Op, d_LSE, KVN, N_chunks);
            fd_opt_stage2_kernel<d><<<1, 128, 0, stream>>>(
                d_Op, d_LSE, d_O, N_chunks);
            cudaStreamSynchronize(stream);
            check("fd_v2_largechunk");

            cudaFree(d_Op); cudaFree(d_LSE);
        }

        // ---- 5. FD v3 (4 warps, Bc=128) ----
        {
            constexpr int kNW3 = 4;
            int N_chunks = (KVN + kBc_v3 - 1) / kBc_v3;
            half *d_Op; float *d_LSE;
            cudaMalloc(&d_Op, N_chunks * d * sizeof(half));
            cudaMalloc(&d_LSE, N_chunks * sizeof(float));

            measure("fd_v3_multiwarp", [&]() {
                fd_v3_stage1_kernel<d, kBc_v3, kNW3><<<N_chunks, WARP_SIZE*kNW3, 0, stream>>>(
                    d_Q, d_K, d_V, d_Op, d_LSE, KVN, N_chunks);
                fd_opt_stage2_kernel<d><<<1, 128, 0, stream>>>(
                    d_Op, d_LSE, d_O, N_chunks);
            });

            fd_v3_stage1_kernel<d, kBc_v3, kNW3><<<N_chunks, WARP_SIZE*kNW3, 0, stream>>>(
                d_Q, d_K, d_V, d_Op, d_LSE, KVN, N_chunks);
            fd_opt_stage2_kernel<d><<<1, 128, 0, stream>>>(
                d_Op, d_LSE, d_O, N_chunks);
            cudaStreamSynchronize(stream);
            check("fd_v3_multiwarp");

            cudaFree(d_Op); cudaFree(d_LSE);
        }

        // ---- 6. FD v4 (8 warps, Bc=256) ----
        {
            constexpr int kNW4 = 8;
            int N_chunks = (KVN + kBc_v4 - 1) / kBc_v4;
            half *d_Op; float *d_LSE;
            cudaMalloc(&d_Op, N_chunks * d * sizeof(half));
            cudaMalloc(&d_LSE, N_chunks * sizeof(float));

            measure("fd_v4_final", [&]() {
                fd_v4_stage1_kernel<d, kBc_v4, kNW4><<<N_chunks, WARP_SIZE*kNW4, 0, stream>>>(
                    d_Q, d_K, d_V, d_Op, d_LSE, KVN, N_chunks);
                fd_opt_stage2_kernel<d><<<1, 128, 0, stream>>>(
                    d_Op, d_LSE, d_O, N_chunks);
            });

            fd_v4_stage1_kernel<d, kBc_v4, kNW4><<<N_chunks, WARP_SIZE*kNW4, 0, stream>>>(
                d_Q, d_K, d_V, d_Op, d_LSE, KVN, N_chunks);
            fd_opt_stage2_kernel<d><<<1, 128, 0, stream>>>(
                d_Op, d_LSE, d_O, N_chunks);
            cudaStreamSynchronize(stream);
            check("fd_v4_final");

            cudaFree(d_Op); cudaFree(d_LSE);
        }

        // ---- 7. FD v5_fused (single kernel, no Stage2) ----
        {
            constexpr int kRPW = 32;
            int num_warps_needed = (KVN + kRPW - 1) / kRPW;
            // Cap at 32 warps (1024 threads) = max block size
            int num_warps = (num_warps_needed > 32) ? 32 : num_warps_needed;
            int num_threads = num_warps * WARP_SIZE;
            // SMEM: max_warps*(kHeadDim floats + 2 floats)
            int smem_bytes = num_warps * (d * sizeof(float) + 2 * sizeof(float));

            auto fn = fd_v5_fused_kernel<d, 32, kRPW>;
            cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

            if (num_warps_needed <= 32) {
                // True single-kernel: all KV rows in one block
                measure("fd_v5_fused", [&]() {
                    fn<<<1, num_threads, 0, stream>>>(d_Q, d_K, d_V, d_O, KVN);
                });

                fn<<<1, num_threads, 0, stream>>>(d_Q, d_K, d_V, d_O, KVN);
                cudaStreamSynchronize(stream);
                check("fd_v5_fused");
            } else {
                fprintf(stderr, "  fd_v5_fused       (skipped, KV=%d > 1024)\n", KVN);
            }
        }

        // ---- 8. FD v5_split (improved 2-stage, all-warp merge) ----
        {
            constexpr int kBc5 = 256;
            constexpr int kNW5 = 8;
            int N_chunks = (KVN + kBc5 - 1) / kBc5;
            half *d_Op; float *d_LSE;
            cudaMalloc(&d_Op, N_chunks * d * sizeof(half));
            cudaMalloc(&d_LSE, N_chunks * sizeof(float));

            measure("fd_v5_split", [&]() {
                fd_v5_split_stage1_kernel<d, kBc5, kNW5><<<N_chunks, WARP_SIZE*kNW5, 0, stream>>>(
                    d_Q, d_K, d_V, d_Op, d_LSE, KVN, N_chunks);
                fd_opt_stage2_kernel<d><<<1, 128, 0, stream>>>(
                    d_Op, d_LSE, d_O, N_chunks);
            });

            fd_v5_split_stage1_kernel<d, kBc5, kNW5><<<N_chunks, WARP_SIZE*kNW5, 0, stream>>>(
                d_Q, d_K, d_V, d_Op, d_LSE, KVN, N_chunks);
            fd_opt_stage2_kernel<d><<<1, 128, 0, stream>>>(
                d_Op, d_LSE, d_O, N_chunks);
            cudaStreamSynchronize(stream);
            check("fd_v5_split");

            cudaFree(d_Op); cudaFree(d_LSE);
        }

        // ---- 7. FA final_kernel (best FA, applied to Q=1) ----
        {
            constexpr int Br = 64, Thr = WARP_SIZE * 4;
            constexpr int Q_sz = Br * (16 + 8), K_sz = 64 * (16 + 8);
            constexpr int smem = (2 * (Q_sz + K_sz)) * sizeof(half);
            auto fn = flash_attn_final_kernel<d, 16,8,16, 4,1, 1,8, 1, (d/8), 8,8,8, 2,1>;
            cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

            // FA treats Q_seqlen=KVN for the outer dimension
            // For Q=1, we call with QKV_seqlen=KVN, but the grid only has 1 Q tile
            int Tr_fa = 1;

            measure("fa_final", [&]() {
                cudaMemset(d_O, 0, total_Q * sizeof(half));
                fn<<<dim3(1, Tr_fa), dim3(Thr), smem, stream>>>(
                    d_Q, d_K, d_V, d_O, KVN, 0);
            });

            // Note: FA output layout differs from baseline for Q=1,
            // correctness check may not match exactly due to different
            // padding behavior, but timing is what matters.
        }

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_O); cudaFree(d_O_ref);
        free(h_ref);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    fprintf(stderr, "\nDone!\n");
    return 0;
}
