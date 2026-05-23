/**
 * bench_decode_vs_fa.cu — Flash Decoding vs Flash Attention (tuned_A)
 * =====================================================================
 * 测试解码场景：小 Q_seqlen ∈ {1,4,16,64,256}, KV_seqlen=4096, d=128
 * 对比 flash decoding (split+reduce) vs 最优 FA kernel (tuned_A)
 * 
 * Usage:
 *   nvcc -arch=sm_89 -O3 -std=c++17 bench_decode_vs_fa.cu -o bench_vs
 *   ./bench_vs > vs_results.csv 2>vs_log.txt
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "flash_decoding.cuh"
#include "flash_attn.cu"

double compute_flops(int QN, int KVN, int d) {
    return 4.0 * (double)QN * (double)KVN * (double)d;
}

int main() {
    const int QN_vals[] = {1, 4, 16, 64, 256};
    const int KVN = 4096;
    const int d = 128;
    const int H = 1;
    constexpr int kBr = 16, kBc = 16;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("kernel,Q_seqlen,KV_seqlen,d,time_ms,gflops\n");
    fprintf(stderr, "=== Flash Decoding vs Flash Attention (KV=4096, d=128) ===\n\n");

    for (int qi = 0; qi < 5; qi++) {
        int QN = QN_vals[qi];
        int N_splits = (KVN + kBc - 1) / kBc;    // 256 splits
        int N_q_tiles = (QN + kBr - 1) / kBr;
        double total_flops = compute_flops(QN, KVN, d);

        int total_Q = H * QN * d;
        int total_KV = H * KVN * d;
        int total_O_partial = H * N_splits * QN * d;
        int total_LSE = H * N_splits * QN;

        fprintf(stderr, "--- Q=%d KV=%d splits=%d q_tiles=%d ---\n", QN, KVN, N_splits, N_q_tiles);

        // Allocate
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

        // === Flash Decoding (Stage1 + Stage2) ===
        {
            dim3 grid_s1(N_splits, H * N_q_tiles);
            dim3 block_s1(WARP_SIZE);
            dim3 grid_s2(H * N_q_tiles, 1);
            dim3 block_s2(16 * WARP_SIZE);

            float ms_total = 0;
            // Stage 1
            {
                auto fn = flash_decode_stage1_kernel<d>;
                fn<<<grid_s1, block_s1, 0, stream>>>(d_Q, d_K, d_V, d_O_partial, d_LSE, QN, KVN, H, N_splits);
                cudaStreamSynchronize(stream);

                cudaEventRecord(start, stream);
                fn<<<grid_s1, block_s1, 0, stream>>>(d_Q, d_K, d_V, d_O_partial, d_LSE, QN, KVN, H, N_splits);
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                float ms;
                cudaEventElapsedTime(&ms, start, stop);
                ms_total += ms;
                printf("decode_s1,%d,%d,%d,%.4f,%.2f\n", QN, KVN, d, ms, total_flops/(ms/1000.0)/1e9);
            }
            // Stage 2
            {
                cudaEventRecord(start, stream);
                flash_decode_stage2_kernel<<<grid_s2, block_s2, 0, stream>>>(d_O_partial, d_LSE, d_O, QN, H, N_splits, d);
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                float ms;
                cudaEventElapsedTime(&ms, start, stop);
                ms_total += ms;
                printf("decode_s2,%d,%d,%d,%.4f,%.2f\n", QN, KVN, d, ms, total_flops/(ms/1000.0)/1e9);
            }
            double gflops = total_flops / (ms_total / 1000.0) / 1e9;
            printf("decode_total,%d,%d,%d,%.4f,%.2f\n", QN, KVN, d, ms_total, gflops);
            fprintf(stderr, "  decode:  s1=%.3f + s2=%.3f = %.3f ms  |  %.1f GFLOPS\n", 
                    ms_total - (ms_total > 0.001f ? 0.0f : 0.0f), 0.0f, ms_total, gflops);
        }

        // === Flash Attention (tuned_A, Br=128,Bc=128, 256 threads) ===
        // tuned_A only works when Q_seqlen >= Br=128 or head_dim matches
        // For QN < 128, need padding or alternative kernel
        // Use final_optimized for QN < 64, tuned_A for QN >= 128
        {
            float ms;
            if (QN >= 128) {
                // tuned_A: Br=128, 256 threads
                constexpr int TBr = 128, TBc = 128;
                constexpr int TThr = WARP_SIZE * 8 * 1;
                constexpr int Q_sz = TBr * (16 + 8);
                constexpr int K_sz = TBc * (16 + 8);
                int smem = (2 * (Q_sz + K_sz)) * sizeof(half);
                int Tr = (QN + TBr - 1) / TBr;
                dim3 grid_fa(1, Tr * H);
                dim3 block_fa(TThr);

                auto fn = flash_attn_tuned_A_kernel<d, 16,8,16, 8,1, 1,16, 1, (d/8), 8,8,8, 2,1>;
                cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

                cudaMemset(d_O, 0, total_Q * sizeof(half));
                fn<<<grid_fa, block_fa, smem, stream>>>(d_Q, d_K, d_V, d_O, QN, H);
                cudaStreamSynchronize(stream);

                cudaEventRecord(start, stream);
                fn<<<grid_fa, block_fa, smem, stream>>>(d_Q, d_K, d_V, d_O, QN, H);
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&ms, start, stop);
            } else {
                // final_optimized: Br=64, 128 threads
                constexpr int Br = 64, Bc = 64;
                constexpr int Thr = WARP_SIZE * 4;
                constexpr int Q_sz = Br * (16 + 8);
                constexpr int K_sz = Bc * (16 + 8);
                int smem = (2 * (Q_sz + K_sz)) * sizeof(half);
                int Tr = (QN + Br - 1) / Br;
                dim3 grid_fa(1, Tr * H);
                dim3 block_fa(Thr);

                auto fn = flash_attn_final_kernel<d, 16,8,16, 4,1, 1,8, 1, (d/8), 8,8,8, 2,1>;
                cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

                cudaMemset(d_O, 0, total_Q * sizeof(half));
                fn<<<grid_fa, block_fa, smem, stream>>>(d_Q, d_K, d_V, d_O, QN, H);
                cudaStreamSynchronize(stream);

                cudaEventRecord(start, stream);
                fn<<<grid_fa, block_fa, smem, stream>>>(d_Q, d_K, d_V, d_O, QN, H);
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&ms, start, stop);
            }
            double gflops = total_flops / (ms / 1000.0) / 1e9;
            printf("flash_attn,%d,%d,%d,%.4f,%.2f\n", QN, KVN, d, ms, gflops);
            fprintf(stderr, "  FA(tuned):         %.3f ms  |  %.1f GFLOPS\n", ms, gflops);
        }

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        cudaFree(d_O_partial); cudaFree(d_LSE);
    }

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    fprintf(stderr, "\nDone!\n");
    return 0;
}
