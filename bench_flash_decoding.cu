/**
 * bench_flash_decoding.cu — Flash Decoding Correctness & Performance
 * ===================================================================
 * Usage: nvcc -arch=sm_89 -O3 bench_flash_decoding.cu -o bench_decode && ./bench_decode
 *
 * Tests: Stage1 (split-KV per-chunk attention) + Stage2 (LSE-weighted reduction)
 * Reference: Naive GPU attention (online softmax) — same algorithm, no split
 *
 * Configs: Q_seqlen={16,32,64}, KV_seqlen={256,512,1024,2048}, d={64,128}
 * Output CSV: kernel,N_kv,N_q,d,time_ms,gflops
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "flash_decoding.cuh"

// ============================================================
// CPU Reference: Naive Attention (FP32)
// ============================================================
void cpu_flash_attn(float *Q, float *K, float *V, float *O,
                    int H, int QN, int KVN, int d) {
    float scale = 1.0f / sqrtf((float)d);
    for (int h = 0; h < H; h++) {
        float *qh = Q + h * QN * d;
        float *kh = K + h * KVN * d;
        float *vh = V + h * KVN * d;
        float *oh = O + h * QN * d;

        for (int i = 0; i < QN; i++) {
            float row_max = -INFINITY, row_sum = 0.0f;
            float *oi = oh + i * d;
            for (int j = 0; j < d; j++) oi[j] = 0.0f;

            for (int j = 0; j < KVN; j++) {
                float s = 0.0f;
                for (int k = 0; k < d; k++)
                    s += qh[i*d + k] * kh[j*d + k];
                s *= scale;

                float m_new = fmaxf(row_max, s);
                float exp_d = expf(row_max - m_new);
                float p = expf(s - m_new);
                row_sum = row_sum * exp_d + p;
                for (int k = 0; k < d; k++)
                    oi[k] = oi[k] * exp_d + p * vh[j*d + k];
                row_max = m_new;
            }
            float inv = 1.0f / row_sum;
            for (int k = 0; k < d; k++) oi[k] *= inv;
        }
    }
}

double compute_flops(int QN, int KVN, int d) {
    return 4.0 * (double)QN * (double)KVN * (double)d;
}

// ============================================================
// Main Test
// ============================================================
int main() {
    // Test configs: (KV_seqlen, d) pairs. Q_seqlen fixed at 32.
    const int QN_vals[] = {32};
    const int KVN_vals[] = {256, 512, 1024, 2048};
    const int d_vals[] = {64, 128};
    const int H = 1;
    constexpr int kBr = 16, kBc = 16;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("kernel,N_kv,N_q,d,time_ms,gflops\n");

    for (int di = 0; di < 2; di++) {
        int d = d_vals[di];

        for (int kvi = 0; kvi < 4; kvi++) {
            int KVN = KVN_vals[kvi];
            int QN = QN_vals[0];  // fixed Q seqlen

            int N_splits = (KVN + kBc - 1) / kBc;
            int N_q_tiles = (QN + kBr - 1) / kBr;

            int total_Q = H * QN * d;
            int total_KV = H * KVN * d;
            int total_O_partial = H * N_splits * QN * d;
            int total_LSE = H * N_splits * QN;

            fprintf(stderr, "\n--- QN=%d KVN=%d d=%d splits=%d ---\n", QN, KVN, d, N_splits);

            // Allocate host memory
            float *h_Q = (float*)malloc(total_Q * sizeof(float));
            float *h_K = (float*)malloc(total_KV * sizeof(float));
            float *h_V = (float*)malloc(total_KV * sizeof(float));
            float *h_O_cpu = (float*)malloc(total_Q * sizeof(float));
            half  *h_O_cpu_half = (half*)malloc(total_Q * sizeof(half));

            srand(42);
            for (int i = 0; i < total_Q; i++) h_Q[i] = ((float)(rand()%1000)/1000.0f - 0.5f);
            for (int i = 0; i < total_KV; i++) {
                h_K[i] = ((float)(rand()%1000)/1000.0f - 0.5f);
                h_V[i] = ((float)(rand()%1000)/1000.0f - 0.5f);
            }

            // CPU Reference
            cpu_flash_attn(h_Q, h_K, h_V, h_O_cpu, H, QN, KVN, d);
            for (int i = 0; i < total_Q; i++)
                h_O_cpu_half[i] = __float2half(h_O_cpu[i]);

            // Allocate GPU memory
            half *d_Q, *d_K, *d_V, *d_O_partial, *d_O_final;
            float *d_LSE;
            cudaMalloc(&d_Q, total_Q * sizeof(half));
            cudaMalloc(&d_K, total_KV * sizeof(half));
            cudaMalloc(&d_V, total_KV * sizeof(half));
            cudaMalloc(&d_O_partial, total_O_partial * sizeof(half));
            cudaMalloc(&d_LSE, total_LSE * sizeof(float));
            cudaMalloc(&d_O_final, total_Q * sizeof(half));

            half *h_Q_h = (half*)malloc(total_Q * sizeof(half));
            half *h_K_h = (half*)malloc(total_KV * sizeof(half));
            half *h_V_h = (half*)malloc(total_KV * sizeof(half));
            for (int i = 0; i < total_Q; i++) h_Q_h[i] = __float2half(h_Q[i]);
            for (int i = 0; i < total_KV; i++) {
                h_K_h[i] = __float2half(h_K[i]);
                h_V_h[i] = __float2half(h_V[i]);
            }
            cudaMemcpy(d_Q, h_Q_h, total_Q * sizeof(half), cudaMemcpyHostToDevice);
            cudaMemcpy(d_K, h_K_h, total_KV * sizeof(half), cudaMemcpyHostToDevice);
            cudaMemcpy(d_V, h_V_h, total_KV * sizeof(half), cudaMemcpyHostToDevice);

            double total_flops = compute_flops(QN, KVN, d);

            // === Stage 1 ===
            dim3 grid_s1(N_splits, H * N_q_tiles);
            dim3 block_s1(WARP_SIZE);

            float ms_s1 = 0;
            if (d == 64) {
                auto fn = flash_decode_stage1_kernel<64>;
                fn<<<grid_s1, block_s1, 0, stream>>>(d_Q, d_K, d_V, d_O_partial, d_LSE, QN, KVN, H, N_splits);
                cudaStreamSynchronize(stream);

                cudaEventRecord(start, stream);
                fn<<<grid_s1, block_s1, 0, stream>>>(d_Q, d_K, d_V, d_O_partial, d_LSE, QN, KVN, H, N_splits);
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&ms_s1, start, stop);
            } else {
                auto fn = flash_decode_stage1_kernel<128>;
                fn<<<grid_s1, block_s1, 0, stream>>>(d_Q, d_K, d_V, d_O_partial, d_LSE, QN, KVN, H, N_splits);
                cudaStreamSynchronize(stream);

                cudaEventRecord(start, stream);
                fn<<<grid_s1, block_s1, 0, stream>>>(d_Q, d_K, d_V, d_O_partial, d_LSE, QN, KVN, H, N_splits);
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&ms_s1, start, stop);
            }
            double gflops_s1 = total_flops / (ms_s1 / 1000.0) / 1e9;
            printf("stage1,%d,%d,%d,%.4f,%.2f\n", KVN, QN, d, ms_s1, gflops_s1);
            fprintf(stderr, "  stage1:            %8.3f ms  %8.1f GFLOPS\n", ms_s1, gflops_s1);

            // === Stage 2 ===
            dim3 grid_s2(H * N_q_tiles, 1);
            dim3 block_s2(16 * WARP_SIZE);  // 512 threads = 16 warps

            cudaEventRecord(start, stream);
            flash_decode_stage2_kernel<<<grid_s2, block_s2, 0, stream>>>(d_O_partial, d_LSE, d_O_final, QN, H, N_splits, d);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            float ms_s2;
            cudaEventElapsedTime(&ms_s2, start, stop);
            printf("stage2,%d,%d,%d,%.4f,%.2f\n", KVN, QN, d, ms_s2, gflops_s1);
            fprintf(stderr, "  stage2:            %8.3f ms\n", ms_s2);
            fprintf(stderr, "  total (s1+s2):     %8.3f ms  %8.1f GFLOPS\n", ms_s1+ms_s2, total_flops/((ms_s1+ms_s2)/1000.0)/1e9);

            // === Correctness check ===
            half *h_O_final = (half*)malloc(total_Q * sizeof(half));
            cudaMemcpy(h_O_final, d_O_final, total_Q * sizeof(half), cudaMemcpyDeviceToHost);

            float max_err = 0.0f, sum_err = 0.0f;
            for (int i = 0; i < total_Q; i++) {
                float err = fabsf(__half2float(h_O_final[i]) - __half2float(h_O_cpu_half[i]));
                if (err > max_err) max_err = err;
                sum_err += err;
            }
            fprintf(stderr, "  Max error: %.6f  Mean error: %.6f  %s\n",
                    max_err, sum_err/total_Q, (max_err < 2.0f) ? "PASS" : "FAIL");

            // Cleanup
            cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
            cudaFree(d_O_partial); cudaFree(d_LSE); cudaFree(d_O_final);
            free(h_Q); free(h_K); free(h_V); free(h_O_cpu); free(h_O_cpu_half);
            free(h_Q_h); free(h_K_h); free(h_V_h); free(h_O_final);
        }
    }

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    fprintf(stderr, "\nDone!\n");
    return 0;
}
