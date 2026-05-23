/**
 * bench_q1.cu — Q=1 Flash Decoding vs Baseline (auto-regressive inference)
 * ==========================================================================
 * 场景：batch=1, Q_seqlen=1, KV_seqlen=4096, d=128
 * 对比：
 *   baseline: 1 warp 串行处理全部 4096 KV 行
 *   flash_decode: 256 warps 并行，每个处理 16 行，Stage2 合并
 *
 * Usage: nvcc -arch=sm_89 -O3 bench_q1.cu -o bench_q1 && ./bench_q1
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define WARP_SIZE 32
#include "flash_decode_q1.cuh"
#include "flash_attn.cu"
#include "utils.h"

// ============================================================
// Baseline: Single-warp non-split attention for Q=1
// ============================================================
__global__ void attn_q1_baseline(half *Q, half *K, half *V, half *O,
                                  int KV_seqlen, int d) {
    int lane_id = threadIdx.x;
    int d_per_thread = d / WARP_SIZE;

    half R_Q[8];
    #pragma unroll
    for (int i = 0; i < d_per_thread; i++)
        R_Q[i] = Q[lane_id * d_per_thread + i];

    float scale = 1.0f / sqrtf((float)d);
    float row_max = -INFINITY, row_sum = 0.0f;
    float O_acc[8] = {0};

    for (int j = 0; j < KV_seqlen; j++) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < d_per_thread; i++) {
            int d_off = lane_id * d_per_thread + i;
            dot += __half2float(R_Q[i]) * __half2float(K[j * d + d_off]);
        }
        dot += __shfl_xor_sync(0xffffffff, dot, 1);
        dot += __shfl_xor_sync(0xffffffff, dot, 2);
        dot += __shfl_xor_sync(0xffffffff, dot, 4);
        dot += __shfl_xor_sync(0xffffffff, dot, 8);
        dot += __shfl_xor_sync(0xffffffff, dot, 16);

        float s = dot * scale;
        float m_new = fmaxf(row_max, s);
        float exp_d = expf(row_max - m_new);
        float p = expf(s - m_new);
        row_sum = row_sum * exp_d + p;

        #pragma unroll
        for (int i = 0; i < d_per_thread; i++) {
            int d_off = lane_id * d_per_thread + i;
            O_acc[i] = O_acc[i] * exp_d + p * __half2float(V[j * d + d_off]);
        }
        row_max = m_new;
    }

    float inv = __frcp_rn(row_sum);
    #pragma unroll
    for (int i = 0; i < d_per_thread; i++) {
        O[lane_id * d_per_thread + i] = __float2half(O_acc[i] * inv);
    }
}

// ============================================================
// Main
// ============================================================
int main() {
    const int KV_vals[] = {256, 512, 1024, 2048, 4096, 8192};
    const int d = 128;
    constexpr int kBc = 16;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("kernel,KV_seqlen,d,time_ms,bandwidth_gbs\n");
    fprintf(stderr, "=== Q=1 Flash Decoding vs Baseline (d=128) ===\n\n");

    for (int ki = 0; ki < 6; ki++) {
        int KVN = KV_vals[ki];
        int N_chunks = (KVN + kBc - 1) / kBc;
        int total_Q = 1 * d;
        int total_KV = KVN * d;
        int total_O_partial = N_chunks * d;
        int total_LSE = N_chunks;

        // Memory bandwidth: read Q(d) + K(KVN*d) + V(KVN*d), write O(d) + O_partial + LSE
        double bytes_read = (total_Q + 2 * total_KV) * sizeof(half) + total_LSE * sizeof(float);
        double bytes_written = (total_Q + total_O_partial) * sizeof(half) + total_LSE * sizeof(float);
        double total_bytes = bytes_read + bytes_written;

        fprintf(stderr, "--- KV=%d chunks=%d ---\n", KVN, N_chunks);

        // Alloc
        half *d_Q, *d_K, *d_V, *d_O, *d_O_partial;
        float *d_LSE;
        cudaMalloc(&d_Q, total_Q * sizeof(half));
        cudaMalloc(&d_K, total_KV * sizeof(half));
        cudaMalloc(&d_V, total_KV * sizeof(half));
        cudaMalloc(&d_O, total_Q * sizeof(half));
        cudaMalloc(&d_O_partial, total_O_partial * sizeof(half));
        cudaMalloc(&d_LSE, total_LSE * sizeof(float));
        cudaMemset(d_Q, 0x3C, total_Q * sizeof(half));
        cudaMemset(d_K, 0x3C, total_KV * sizeof(half));
        cudaMemset(d_V, 0x3C, total_KV * sizeof(half));

        // === Baseline ===
        {
            attn_q1_baseline<<<1, WARP_SIZE, 0, stream>>>(d_Q, d_K, d_V, d_O, KVN, d);
            cudaStreamSynchronize(stream);

            cudaEventRecord(start, stream);
            attn_q1_baseline<<<1, WARP_SIZE, 0, stream>>>(d_Q, d_K, d_V, d_O, KVN, d);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            double bw = total_bytes / (ms / 1000.0) / 1e9;
            printf("baseline,%d,%d,%.4f,%.2f\n", KVN, d, ms, bw);
            fprintf(stderr, "  baseline:           %8.3f ms  %6.1f GB/s\n", ms, bw);
        }

        // === Flash Decoding ===
        {
            dim3 grid_s1(N_chunks, 1);
            dim3 block_s1(WARP_SIZE);

            // Stage 1
            flash_decode_q1_stage1_kernel<<<grid_s1, block_s1, 0, stream>>>(d_Q, d_K, d_V, d_O_partial, d_LSE, KVN, N_chunks, d);
            cudaStreamSynchronize(stream);

            cudaEventRecord(start, stream);
            flash_decode_q1_stage1_kernel<<<grid_s1, block_s1, 0, stream>>>(d_Q, d_K, d_V, d_O_partial, d_LSE, KVN, N_chunks, d);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            float ms_s1;
            cudaEventElapsedTime(&ms_s1, start, stop);

            // Stage 2
            cudaEventRecord(start, stream);
            flash_decode_q1_stage2_kernel<<<1, WARP_SIZE, 0, stream>>>(d_O_partial, d_LSE, d_O, N_chunks, d);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            float ms_s2;
            cudaEventElapsedTime(&ms_s2, start, stop);

            float ms_total = ms_s1 + ms_s2;
            double bw = total_bytes / (ms_total / 1000.0) / 1e9;
            printf("decode_s1,%d,%d,%.4f,%.2f\n", KVN, d, ms_s1, bw);
            printf("decode_total,%d,%d,%.4f,%.2f\n", KVN, d, ms_total, bw);
            fprintf(stderr, "  decode: s1=%.3f + s2=%.3f = %.3f ms  %6.1f GB/s\n", ms_s1, ms_s2, ms_total, bw);
        }

        // === FA final_optimized (MMA Br=64, 128 threads) — best single-stage FA for Q=1 ===
        // With Q=1, it processes 1 valid + 63 padded rows. Measures overhead vs decode.
        {
            constexpr int Br = 64, Thr = WARP_SIZE * 4;
            constexpr int Q_sz = Br * (16 + 8), K_sz = 64 * (16 + 8);
            int smem = (2*(Q_sz+K_sz))*sizeof(half);
            auto fn = flash_attn_final_kernel<d, 16,8,16, 4,1, 1,8, 1, (d/8), 8,8,8, 2,1>;
            cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            cudaMemset(d_O, 0, total_Q*sizeof(half));
            fn<<<dim3(1,1), dim3(Thr), smem, stream>>>(d_Q, d_K, d_V, d_O, 1, 1);
            cudaStreamSynchronize(stream);
            cudaEventRecord(start, stream);
            fn<<<dim3(1,1), dim3(Thr), smem, stream>>>(d_Q, d_K, d_V, d_O, 1, 1);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            float ms_fa;
            cudaEventElapsedTime(&ms_fa, start, stop);
            double bw_fa = total_bytes / (ms_fa / 1000.0) / 1e9;
            printf("fa_final,%d,%d,%.4f,%.2f\n", KVN, d, ms_fa, bw_fa);
            fprintf(stderr, "  FA(final,Br=64):    %8.3f ms  %6.1f GB/s\n", ms_fa, bw_fa);
        }

        // Correctness check (baseline = reference)
        half *h_O_base = (half*)malloc(total_Q * sizeof(half));
        half *h_O_decode = (half*)malloc(total_Q * sizeof(half));
        attn_q1_baseline<<<1, WARP_SIZE, 0, stream>>>(d_Q, d_K, d_V, d_O, KVN, d);
        cudaMemcpy(h_O_base, d_O, total_Q * sizeof(half), cudaMemcpyDeviceToHost);

        dim3 grid_s1_c(N_chunks, 1);
        dim3 block_s1_c(WARP_SIZE);
        flash_decode_q1_stage1_kernel<<<grid_s1_c, block_s1_c, 0, stream>>>(d_Q, d_K, d_V, d_O_partial, d_LSE, KVN, N_chunks, d);
        flash_decode_q1_stage2_kernel<<<1, WARP_SIZE, 0, stream>>>(d_O_partial, d_LSE, d_O, N_chunks, d);
        cudaMemcpy(h_O_decode, d_O, total_Q * sizeof(half), cudaMemcpyDeviceToHost);

        float max_err = 0;
        for (int i = 0; i < total_Q; i++) {
            float err = fabsf(__half2float(h_O_decode[i]) - __half2float(h_O_base[i]));
            if (err > max_err) max_err = err;
        }
        fprintf(stderr, "  Max error: %.6f  %s\n", max_err, (max_err < 1.0f) ? "PASS" : "FAIL");

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        cudaFree(d_O_partial); cudaFree(d_LSE);
        free(h_O_base); free(h_O_decode);
    }

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    fprintf(stderr, "\nDone!\n");
    return 0;
}
