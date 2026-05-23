/**
 * bench_flash_attn.cu — Multi-scenario Flash Attention Benchmark
 * ================================================================
 * Tests 4 optimized kernels across N={256,512,1024,2048} x d={64,128}
 * Outputs CSV to stdout: kernel,N,d,time_ms,gflops
 * 
 * Usage:
 *   nvcc -arch=sm_89 -O3 bench_flash_attn.cu -o bench_flash_attn
 *   ./bench_flash_attn > results.csv 2>log.txt
 *   python plot_results.py results.csv
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#include "flash_attn.cu"
#include "utils.h"
#include "ref_kernel.cuh"

// ============================================================
// FLOPs: FlashAttention ≈ 4 * N^2 * d
// ============================================================
double compute_flops(int N, int d) {
    return 4.0 * (double)N * (double)N * (double)d;
}

// ============================================================
// Template: run all kernels for a fixed d
// ============================================================
template<int d>
void bench_all_kernels(int N, FILE* csv_out) {
    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int NumThreads = WARP_SIZE * 4;  // 128 threads
    const int H = 1;
    
    int total_elements = H * N * d;
    int Tr = (N + Br - 1) / Br;
    dim3 grid(1, Tr * H);
    dim3 block(NumThreads);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Allocate GPU memory
    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, total_elements * sizeof(half));
    cudaMalloc(&d_K, total_elements * sizeof(half));
    cudaMalloc(&d_V, total_elements * sizeof(half));
    cudaMalloc(&d_O, total_elements * sizeof(half));
    cudaMemset(d_Q, 0x3C, total_elements * sizeof(half));
    cudaMemset(d_K, 0x3C, total_elements * sizeof(half));
    cudaMemset(d_V, 0x3C, total_elements * sizeof(half));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    double total_flops = compute_flops(N, d);
    
    auto measure = [&](auto kernel_fn, const char* name, int smem) -> float {
        // Warmup
        kernel_fn<<<grid, block, smem, stream>>>(d_Q, d_K, d_V, d_O, N, H);
        cudaStreamSynchronize(stream);
        
        // Measure
        cudaEventRecord(start, stream);
        kernel_fn<<<grid, block, smem, stream>>>(d_Q, d_K, d_V, d_O, N, H);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        
        double gflops = total_flops / (ms / 1000.0) / 1e9;
        fprintf(csv_out, "%s,%d,%d,%.4f,%.2f\n", name, N, d, ms, gflops);
        fprintf(stderr, "  %-22s %8.3f ms  %8.1f GFLOPS\n", name, ms, gflops);
        return ms;
    };
    
    // ---- Kernel 1: Fine-grained Q/K Tiling ----
    {
        constexpr int smem = (Br*24 + Bc*24 + Br*Bc + Bc*d) * sizeof(half);
        auto fn = flash_attn_finegrained_qk_tiling_kernel<d, 16,8,16, 4,1, 1,8, 8,8>;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        measure(fn, "finegrained_qk", smem);
    }
    
    // ---- Kernel 2: Register P ----
    {
        constexpr int smem = (Br*24 + Bc*24 + Bc*d) * sizeof(half);
        auto fn = flash_attn_register_p_kernel<d, 16,8,16, 4,1, 1,8, 8,8>;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        measure(fn, "register_p", smem);
    }
    
    // ---- Kernel 3: Async + Double Buffering ----
    {
        constexpr int smem = (2*(Br*24 + Bc*24) + Bc*d) * sizeof(half);
        auto fn = flash_attn_async_kernel<d, 16,8,16, 4,1, 1,8, 8,8, 2>;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        measure(fn, "async_doublebuf", smem);
    }
    
    // ---- Kernel 4: Final Fully Optimized ----
    {
        constexpr int smem = (2*(Br*24 + Bc*24)) * sizeof(half);
        constexpr int kWTHV = d / 8;
        auto fn = flash_attn_final_kernel<d, 16,8,16, 4,1, 1,8, 1, kWTHV, 8,8,8, 2,1>;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        measure(fn, "final_optimized", smem);
    }
    
    // ---- TUNED A: Bigger Tiles (Br=128,Bc=128, 256 threads) ----
    {
        constexpr int TBr = 128, TBc = 128;
        constexpr int TThr = WARP_SIZE * 8 * 1;  // 8 warps
        constexpr int Q_sz = TBr * (16 + 8);     // 128*24=3072
        constexpr int K_sz = TBc * (16 + 8);     // 128*24=3072
        constexpr int smem = (2 * (Q_sz + K_sz)) * sizeof(half);
        constexpr int kWTHV = d / 8;
        const int TTr = (N + TBr - 1) / TBr;
        
        dim3 grid_tuned(TTr, 1);
        dim3 block_tuned(TThr);
        
        auto fn = flash_attn_tuned_A_kernel<d, 16,8,16, 8,1, 1,16, 1, kWTHV, 8,8,8, 2,1>;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        
        cudaMemset(d_O, 0, total_elements * sizeof(half));
        fn<<<grid_tuned, block_tuned, smem, stream>>>(d_Q, d_K, d_V, d_O, N, H);
        cudaStreamSynchronize(stream);
        
        cudaEventRecord(start, stream);
        fn<<<grid_tuned, block_tuned, smem, stream>>>(d_Q, d_K, d_V, d_O, N, H);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        double gflops = total_flops / (ms / 1000.0) / 1e9;
        fprintf(csv_out, "tuned_A,%d,%d,%.4f,%.2f\n", N, d, ms, gflops);
        fprintf(stderr, "  %-22s %8.3f ms  %8.1f GFLOPS\n", "tuned_A", ms, gflops);
    }

    // ---- Kernel Ref: Reference (mma_tiling_qkv, FA2-style) ----
    {
        constexpr int kMmaTileSeqLenQ = (d < 128) ? 4 : 8;
        constexpr int kWarpTileSeqLenK = (d < 128) ? 8 : 16;
        constexpr int kMmaTileSeqLenP = (d < 128) ? 4 : 8;
        constexpr int kWarpTileHeadDimV = d / 8;
        constexpr int kOStorageAccF32 = (d < 256) ? 1 : 0;
        constexpr int kStage = 2;
        constexpr int kPad = 8;
        constexpr int RefThreads = WARP_SIZE * kMmaTileSeqLenQ * 1;
        constexpr int RefBr = 16 * kMmaTileSeqLenQ * 1;   // kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ
        constexpr int RefBc = 8 * 1 * kWarpTileSeqLenK;   // kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK
        const int RefTr = (N + RefBr - 1) / RefBr;
        
        constexpr int QK_smem = kStage * (RefBr*(16+kPad) + RefBc*(16+kPad));
        constexpr int Vr_smem = kStage * (RefBc*(16+kPad));
        constexpr int smem = (QK_smem > Vr_smem ? QK_smem : Vr_smem) * sizeof(half);
        
        dim3 grid_ref(RefTr, 1);  // reference uses blockIdx.x for Q tile
        dim3 block_ref(RefThreads);
        
        auto fn = flash_attn_mma_stages_split_q_tiling_qkv_kernel<
            d, 16,8,16, kMmaTileSeqLenQ,1, kMmaTileSeqLenP,1, 1,kWarpTileSeqLenK, 1,kWarpTileHeadDimV, kOStorageAccF32, kStage, kPad,kPad,kPad>;
        
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        
        cudaMemset(d_O, 0, total_elements * sizeof(half));
        
        // Warmup
        fn<<<grid_ref, block_ref, smem, stream>>>(d_Q, d_K, d_V, d_O, N, H);
        cudaStreamSynchronize(stream);
        
        // Measure
        cudaEventRecord(start, stream);
        fn<<<grid_ref, block_ref, smem, stream>>>(d_Q, d_K, d_V, d_O, N, H);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        double gflops = total_flops / (ms / 1000.0) / 1e9;
        fprintf(csv_out, "reference_mma,%d,%d,%.4f,%.2f\n", N, d, ms, gflops);
        fprintf(stderr, "  %-22s %8.3f ms  %8.1f GFLOPS\n", "reference_mma", ms, gflops);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
}

// ============================================================
// Dispatch based on runtime d
// ============================================================
void bench_d(int d, int N, FILE* csv_out) {
    switch (d) {
        case 64:  bench_all_kernels<64>(N, csv_out);  break;
        case 128: bench_all_kernels<128>(N, csv_out); break;
        default:
            fprintf(stderr, "Error: d=%d not supported\n", d);
    }
}

// ============================================================
// Main
// ============================================================
int main() {
    const int N_vals[] = {256, 512, 1024, 2048};
    const int d_vals[] = {64, 128};
    const int num_N = sizeof(N_vals) / sizeof(N_vals[0]);
    const int num_d = sizeof(d_vals) / sizeof(d_vals[0]);
    
    fprintf(stderr, "=== Flash Attention Multi-Scenario Benchmark ===\n");
    fprintf(stderr, "GPU: RTX 4060 Ti (sm_89)\n");
    fprintf(stderr, "Configs: N in {256,512,1024,2048} x d in {64,128}\n");
    fprintf(stderr, "Kernels: finegrained_qk, register_p, async_doublebuf, final_optimized\n\n");
    
    // CSV header to stdout
    printf("kernel,N,d,time_ms,gflops\n");
    
    for (int di = 0; di < num_d; di++) {
        int d = d_vals[di];
        for (int ni = 0; ni < num_N; ni++) {
            int N = N_vals[ni];
            fprintf(stderr, "\n--- N=%d d=%d ---\n", N, d);
            bench_d(d, N, stdout);
        }
    }
    
    fprintf(stderr, "\nDone! Run: python plot_results.py results.csv\n");
    return 0;
}
