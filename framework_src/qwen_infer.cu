#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Self-developed HGEMM (replaces cuBLAS for prefill batched GEMM)
#include "../kernels/hgemm_mma_swizzle.cuh"

// Global volatile variable to prevent host compiler from optimizing away memory copies
volatile float prevent_opt = 0.0f;

// Include our optimized Flash Decoding kernels from kernels directory
#include "../kernels/flash_decode_q1_opt.cuh"

// Magic number to verify weight compatibility
#define QWDF_MAGIC 0x46445751

// Structure matching python packer header
struct QwenConfig {
    unsigned int magic;
    unsigned int vocab_size;
    unsigned int hidden_size;
    unsigned int num_layers;
    unsigned int num_heads;
    unsigned int intermediate_size;
    unsigned int max_seqlen;
};

// Model Weights Structure allocated in GPU (Device) memory
struct QwenWeights {
    half* wte;            // [vocab_size, hidden_size]
    half* ln_f;           // [hidden_size]
    half* lm_head;        // [vocab_size, hidden_size]
    
    // Per-layer weights
    std::vector<half*> ln_1;          // [hidden_size] x num_layers
    std::vector<half*> qkv_w;         // [3 * hidden_size, hidden_size] x num_layers
    std::vector<half*> qkv_b;         // [3 * hidden_size] x num_layers
    std::vector<half*> attn_proj_w;   // [hidden_size, hidden_size] x num_layers
    std::vector<half*> ln_2;          // [hidden_size] x num_layers
    std::vector<half*> ffn_w1;        // [intermediate_size, hidden_size] x num_layers
    std::vector<half*> ffn_w2;        // [intermediate_size, hidden_size] x num_layers
    std::vector<half*> ffn_proj_w;    // [hidden_size, intermediate_size] x num_layers
};

// ============================================================
// Unified Custom CUDA Operator Library
// ============================================================
#include "../kernels/rmsnorm.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/bias.cuh"
#include "../kernels/argmax.cuh"
#include "../kernels/hgemm_final.cuh"
#include "../kernels/transpose.cuh"
#include "../kernels/fused_qkv_rope_cache.cuh"
#define FLASH_ATTN_CAUSAL  // enable causal mask in FA
#include "../kernels/flash_attn.cu"
#include "../kernels/gather_qkv.cuh"       // interleaved → per-head contiguous

// ============================================================
// Main High-Performance Inference Engine
// ============================================================
class QwenEngine {
private:
    QwenConfig config;
    QwenWeights weights;
    
    // KV Cache in GPU VRAM
    half* d_kv_cache_k;   // [layers, max_seqlen, hidden_size]
    half* d_kv_cache_v;   // [layers, max_seqlen, hidden_size]
    
    // Dynamic Activation Buffers in GPU VRAM
    half* d_x;            // [hidden_size]
    half* d_norm_x;       // [hidden_size]
    half* d_qkv;          // [3 * hidden_size]
    half* d_qkv_out;      // [hidden_size]
    half* d_attn_proj;    // [hidden_size]
    half* d_ffn_w1;       // [intermediate_size]
    half* d_ffn_w2;       // [intermediate_size]
    half* d_ffn_act;      // [intermediate_size]
    half* d_ffn_proj;     // [hidden_size]
    half* d_logits;       // [vocab_size]
    int* d_next_token;    // [1]
    
    // Flash Decoding temporary buffers
    half* d_fd_partial;   // [max_num_chunks, hidden_size]
    float* d_fd_lse;      // [max_num_chunks]

    // Prefill batched buffers (for FA prefill stage)
    half* d_prefill_x;    // [max_seqlen, H]
    half* d_prefill_norm; // [max_seqlen, H]
    half* d_prefill_qkv;  // [max_seqlen, 3H]
    half* d_prefill_out;  // [max_seqlen, H]
    half* d_prefill_q;    // [NH*max_seqlen, headDim] gathered Q for FA
    half* d_prefill_k;    // [NH*max_seqlen, headDim] gathered K
    half* d_prefill_v;    // [NH*max_seqlen, headDim] gathered V
    half* d_prefill_o;    // [NH*max_seqlen, headDim] FA output temp
    half* d_prefill_w1;   // [max_seqlen, IM]
    half* d_prefill_w2;   // [max_seqlen, IM]
    half* d_prefill_act;  // [max_seqlen, IM]
    half* d_prefill_proj; // [max_seqlen, H]

    // Allocate memory helper
    half* alloc_gpu(size_t size) {
        half* ptr;
        cudaMalloc(&ptr, size * sizeof(half));
        return ptr;
    }

public:
    QwenEngine(const std::string& model_path) {
        
        // 1. Read binary weights
        std::ifstream f(model_path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: Cannot open model weights file " << model_path << std::endl;
            exit(1);
        }
        
        f.read(reinterpret_cast<char*>(&config), sizeof(QwenConfig));
        if (config.magic != QWDF_MAGIC) {
            std::cerr << "Error: Magic number mismatch! Expected " << QWDF_MAGIC << ", got " << config.magic << std::endl;
            exit(1);
        }
        
        std::cout << "\n=== Weight Config Loaded Successfully ===" << std::endl;
        std::cout << "  Vocab size:        " << config.vocab_size << std::endl;
        std::cout << "  Hidden size:       " << config.hidden_size << std::endl;
        std::cout << "  Layers count:      " << config.num_layers << std::endl;
        std::cout << "  Heads count:       " << config.num_heads << std::endl;
        std::cout << "  FFN Intermediate:  " << config.intermediate_size << std::endl;
        std::cout << "  Max seqlen:        " << config.max_seqlen << std::endl;
        
        // 2. Allocate & Load weights in GPU VRAM
        std::cout << "\nAllocating VRAM and transferring weights to GPU..." << std::endl;
        
        size_t wte_size = (size_t)config.vocab_size * config.hidden_size;
        weights.wte = alloc_gpu(wte_size);
        std::vector<half> temp_cpu(wte_size);
        f.read(reinterpret_cast<char*>(temp_cpu.data()), wte_size * sizeof(half));
        cudaMemcpy(weights.wte, temp_cpu.data(), wte_size * sizeof(half), cudaMemcpyHostToDevice);
        
        // Per-layer weights
        weights.ln_1.resize(config.num_layers);
        weights.qkv_w.resize(config.num_layers);
        weights.qkv_b.resize(config.num_layers);
        weights.attn_proj_w.resize(config.num_layers);
        weights.ln_2.resize(config.num_layers);
        weights.ffn_w1.resize(config.num_layers);
        weights.ffn_w2.resize(config.num_layers);
        weights.ffn_proj_w.resize(config.num_layers);
        
        size_t layer_ln_size = config.hidden_size;
        size_t layer_qkv_w_size = 3 * (size_t)config.hidden_size * config.hidden_size;
        size_t layer_qkv_b_size = 3 * config.hidden_size;
        size_t layer_proj_w_size = (size_t)config.hidden_size * config.hidden_size;
        size_t layer_ffn_w_size = (size_t)config.intermediate_size * config.hidden_size;
        size_t layer_ffn_proj_w_size = (size_t)config.hidden_size * config.intermediate_size;
        
        for (unsigned int i = 0; i < config.num_layers; i++) {
            // ln_1
            weights.ln_1[i] = alloc_gpu(layer_ln_size);
            temp_cpu.resize(layer_ln_size);
            f.read(reinterpret_cast<char*>(temp_cpu.data()), layer_ln_size * sizeof(half));
            cudaMemcpy(weights.ln_1[i], temp_cpu.data(), layer_ln_size * sizeof(half), cudaMemcpyHostToDevice);
            
            // qkv_w
            weights.qkv_w[i] = alloc_gpu(layer_qkv_w_size);
            temp_cpu.resize(layer_qkv_w_size);
            f.read(reinterpret_cast<char*>(temp_cpu.data()), layer_qkv_w_size * sizeof(half));
            cudaMemcpy(weights.qkv_w[i], temp_cpu.data(), layer_qkv_w_size * sizeof(half), cudaMemcpyHostToDevice);
            
            // qkv_b
            weights.qkv_b[i] = alloc_gpu(layer_qkv_b_size);
            temp_cpu.resize(layer_qkv_b_size);
            f.read(reinterpret_cast<char*>(temp_cpu.data()), layer_qkv_b_size * sizeof(half));
            cudaMemcpy(weights.qkv_b[i], temp_cpu.data(), layer_qkv_b_size * sizeof(half), cudaMemcpyHostToDevice);
            
            // attn_proj_w
            weights.attn_proj_w[i] = alloc_gpu(layer_proj_w_size);
            temp_cpu.resize(layer_proj_w_size);
            f.read(reinterpret_cast<char*>(temp_cpu.data()), layer_proj_w_size * sizeof(half));
            cudaMemcpy(weights.attn_proj_w[i], temp_cpu.data(), layer_proj_w_size * sizeof(half), cudaMemcpyHostToDevice);
            
            // ln_2
            weights.ln_2[i] = alloc_gpu(layer_ln_size);
            temp_cpu.resize(layer_ln_size);
            f.read(reinterpret_cast<char*>(temp_cpu.data()), layer_ln_size * sizeof(half));
            cudaMemcpy(weights.ln_2[i], temp_cpu.data(), layer_ln_size * sizeof(half), cudaMemcpyHostToDevice);
            
            // ffn_w1
            weights.ffn_w1[i] = alloc_gpu(layer_ffn_w_size);
            temp_cpu.resize(layer_ffn_w_size);
            f.read(reinterpret_cast<char*>(temp_cpu.data()), layer_ffn_w_size * sizeof(half));
            cudaMemcpy(weights.ffn_w1[i], temp_cpu.data(), layer_ffn_w_size * sizeof(half), cudaMemcpyHostToDevice);
            
            // ffn_w2
            weights.ffn_w2[i] = alloc_gpu(layer_ffn_w_size);
            temp_cpu.resize(layer_ffn_w_size);
            f.read(reinterpret_cast<char*>(temp_cpu.data()), layer_ffn_w_size * sizeof(half));
            cudaMemcpy(weights.ffn_w2[i], temp_cpu.data(), layer_ffn_w_size * sizeof(half), cudaMemcpyHostToDevice);
            
            // ffn_proj_w
            weights.ffn_proj_w[i] = alloc_gpu(layer_ffn_proj_w_size);
            temp_cpu.resize(layer_ffn_proj_w_size);
            f.read(reinterpret_cast<char*>(temp_cpu.data()), layer_ffn_proj_w_size * sizeof(half));
            cudaMemcpy(weights.ffn_proj_w[i], temp_cpu.data(), layer_ffn_proj_w_size * sizeof(half), cudaMemcpyHostToDevice);
        }
        
        // ln_f
        weights.ln_f = alloc_gpu(layer_ln_size);
        temp_cpu.resize(layer_ln_size);
        f.read(reinterpret_cast<char*>(temp_cpu.data()), layer_ln_size * sizeof(half));
        cudaMemcpy(weights.ln_f, temp_cpu.data(), layer_ln_size * sizeof(half), cudaMemcpyHostToDevice);
        
        // lm_head
        size_t lm_head_size = (size_t)config.vocab_size * config.hidden_size;
        weights.lm_head = alloc_gpu(lm_head_size);
        temp_cpu.resize(lm_head_size);
        f.read(reinterpret_cast<char*>(temp_cpu.data()), lm_head_size * sizeof(half));
        cudaMemcpy(weights.lm_head, temp_cpu.data(), lm_head_size * sizeof(half), cudaMemcpyHostToDevice);
        
        f.close();
        std::cout << "[SUCCESS] Weights fully loaded into VRAM." << std::endl;
        
        // Transpose projection weights in-place for highly optimized HGEMV/HGEMM compatibility
        std::cout << "Transposing projection weights for optimized HGEMV..." << std::endl;
        for (unsigned int i = 0; i < config.num_layers; i++) {
            transpose_weight_gpu(weights.qkv_w[i], 3 * config.hidden_size, config.hidden_size);
            transpose_weight_gpu(weights.attn_proj_w[i], config.hidden_size, config.hidden_size);
            transpose_weight_gpu(weights.ffn_w1[i], config.intermediate_size, config.hidden_size);
            transpose_weight_gpu(weights.ffn_w2[i], config.intermediate_size, config.hidden_size);
            transpose_weight_gpu(weights.ffn_proj_w[i], config.hidden_size, config.intermediate_size);
        }
        transpose_weight_gpu(weights.lm_head, config.vocab_size, config.hidden_size);
        std::cout << "[SUCCESS] Weights successfully transposed." << std::endl;
        
        // 3. Allocate KV Cache & Dynamic Activations
        size_t kv_cache_size = (size_t)config.num_layers * config.max_seqlen * config.hidden_size;
        d_kv_cache_k = alloc_gpu(kv_cache_size);
        d_kv_cache_v = alloc_gpu(kv_cache_size);
        cudaMemset(d_kv_cache_k, 0, kv_cache_size * sizeof(half));
        cudaMemset(d_kv_cache_v, 0, kv_cache_size * sizeof(half));
        
        d_x = alloc_gpu(config.hidden_size);
        d_norm_x = alloc_gpu(config.hidden_size);
        d_qkv = alloc_gpu(3 * config.hidden_size);
        d_qkv_out = alloc_gpu(config.hidden_size);
        d_attn_proj = alloc_gpu(config.hidden_size);
        d_ffn_w1 = alloc_gpu(config.intermediate_size);
        d_ffn_w2 = alloc_gpu(config.intermediate_size);
        d_ffn_act = alloc_gpu(config.intermediate_size);
        d_ffn_proj = alloc_gpu(config.hidden_size);
        d_logits = alloc_gpu(config.vocab_size);
        cudaMalloc(&d_next_token, sizeof(int));
        
        // Flash Decoding temporary buffers (for stage 2 reduction)
        int max_num_chunks = (config.max_seqlen + 255) / 256;
        d_fd_partial = alloc_gpu(max_num_chunks * config.hidden_size);
        cudaMalloc(&d_fd_lse, max_num_chunks * sizeof(float));

        // Prefill batched buffers
        size_t H = config.hidden_size;
        size_t IM = config.intermediate_size;
        size_t NH = config.num_heads;
        d_prefill_x    = alloc_gpu(config.max_seqlen * H);
        d_prefill_norm = alloc_gpu(config.max_seqlen * H);
        d_prefill_qkv  = alloc_gpu(config.max_seqlen * 3 * H);
        d_prefill_out  = alloc_gpu(config.max_seqlen * H);
        d_prefill_q    = alloc_gpu(NH * config.max_seqlen * 128);
        d_prefill_k    = alloc_gpu(NH * config.max_seqlen * 128);
        d_prefill_v    = alloc_gpu(NH * config.max_seqlen * 128);
        d_prefill_o    = alloc_gpu(NH * config.max_seqlen * 128);
        d_prefill_w1   = alloc_gpu(config.max_seqlen * IM);
        d_prefill_w2   = alloc_gpu(config.max_seqlen * IM);
        d_prefill_act  = alloc_gpu(config.max_seqlen * IM);
        d_prefill_proj = alloc_gpu(config.max_seqlen * H);
    }
    
    ~QwenEngine() {
        cudaFree(weights.wte);
        cudaFree(weights.ln_f);
        cudaFree(weights.lm_head);
        
        for (unsigned int i = 0; i < config.num_layers; i++) {
            cudaFree(weights.ln_1[i]);
            cudaFree(weights.qkv_w[i]);
            cudaFree(weights.qkv_b[i]);
            cudaFree(weights.attn_proj_w[i]);
            cudaFree(weights.ln_2[i]);
            cudaFree(weights.ffn_w1[i]);
            cudaFree(weights.ffn_w2[i]);
            cudaFree(weights.ffn_proj_w[i]);
        }
        
        cudaFree(d_kv_cache_k);
        cudaFree(d_kv_cache_v);
        cudaFree(d_x);
        cudaFree(d_norm_x);
        cudaFree(d_qkv);
        cudaFree(d_qkv_out);
        cudaFree(d_attn_proj);
        cudaFree(d_ffn_w1);
        cudaFree(d_ffn_w2);
        cudaFree(d_ffn_act);
        cudaFree(d_ffn_proj);
        cudaFree(d_logits);
        cudaFree(d_next_token);
        cudaFree(d_fd_partial);
        cudaFree(d_fd_lse);
    }
    void reset() {
        size_t H = (size_t)config.hidden_size;
        size_t IM = (size_t)config.intermediate_size;
        size_t NH = (size_t)config.num_heads;
        size_t kv_cache_size = (size_t)config.num_layers * config.max_seqlen * H;
        size_t max_num_chunks = (config.max_seqlen + 255) / 256;

        cudaMemset(d_kv_cache_k, 0, kv_cache_size * sizeof(half));
        cudaMemset(d_kv_cache_v, 0, kv_cache_size * sizeof(half));

        cudaMemset(d_x, 0, H * sizeof(half));
        cudaMemset(d_norm_x, 0, H * sizeof(half));
        cudaMemset(d_qkv, 0, 3 * H * sizeof(half));
        cudaMemset(d_qkv_out, 0, H * sizeof(half));
        cudaMemset(d_attn_proj, 0, H * sizeof(half));
        cudaMemset(d_ffn_w1, 0, IM * sizeof(half));
        cudaMemset(d_ffn_w2, 0, IM * sizeof(half));
        cudaMemset(d_ffn_act, 0, IM * sizeof(half));
        cudaMemset(d_ffn_proj, 0, H * sizeof(half));
        cudaMemset(d_logits, 0, (size_t)config.vocab_size * sizeof(half));
        cudaMemset(d_next_token, 0, sizeof(int));

        cudaMemset(d_fd_partial, 0, max_num_chunks * H * sizeof(half));
        cudaMemset(d_fd_lse, 0, max_num_chunks * sizeof(float));

        cudaMemset(d_prefill_x, 0, (size_t)config.max_seqlen * H * sizeof(half));
        cudaMemset(d_prefill_norm, 0, (size_t)config.max_seqlen * H * sizeof(half));
        cudaMemset(d_prefill_qkv, 0, (size_t)config.max_seqlen * 3 * H * sizeof(half));
        cudaMemset(d_prefill_out, 0, (size_t)config.max_seqlen * H * sizeof(half));
        cudaMemset(d_prefill_q, 0, (size_t)NH * config.max_seqlen * 128 * sizeof(half));
        cudaMemset(d_prefill_k, 0, (size_t)NH * config.max_seqlen * 128 * sizeof(half));
        cudaMemset(d_prefill_v, 0, (size_t)NH * config.max_seqlen * 128 * sizeof(half));
        cudaMemset(d_prefill_o, 0, (size_t)NH * config.max_seqlen * 128 * sizeof(half));
        cudaMemset(d_prefill_w1, 0, (size_t)config.max_seqlen * IM * sizeof(half));
        cudaMemset(d_prefill_w2, 0, (size_t)config.max_seqlen * IM * sizeof(half));
        cudaMemset(d_prefill_act, 0, (size_t)config.max_seqlen * IM * sizeof(half));
        cudaMemset(d_prefill_proj, 0, (size_t)config.max_seqlen * H * sizeof(half));
        cudaDeviceSynchronize();
    }

    // High-performance Matrix Multiplication — 100% self-developed
    void gemm(half* out, const half* in, const half* weight, int m, int n, int k) {
        if (m == 1) {
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            hgemv_kernel<<<blocks, threads>>>(in, weight, out, n, k);
        } else {
            hgemm_swizzle_nn(const_cast<half*>(in), const_cast<half*>(weight), out, m, n, k);
        }
    }

    // ============================================================
    // FA Prefill: batch-process all prompt tokens in one forward pass
    // ============================================================
    void prefill(const std::vector<int>& tokens, int& pos) {
        int N = (int)tokens.size();
        if (N == 0) return;
        int H = (int)config.hidden_size;
        int IM = (int)config.intermediate_size;
        int NH = (int)config.num_heads;
        constexpr int kHD = 128;
        constexpr float eps = 1e-6f;

        // Helper function to print first 8 values to std::cerr
        auto dump_first_8 = [](const std::string& name, half* d_ptr) {
            half tmp[8];
            cudaMemcpy(tmp, d_ptr, 8 * sizeof(half), cudaMemcpyDeviceToHost);
            std::cerr << "[DUMP] " << name << ": ";
            for (int i = 0; i < 8; i++) {
                std::cerr << __half2float(tmp[i]) << " ";
            }
            std::cerr << std::endl;
        };

        const bool debug_finite_checks = (std::getenv("QWEN_DEBUG_FINITE") != nullptr);
        const char* finite_log_path = std::getenv("QWEN_FINITE_LOG");

        auto scan_tensor_finite = [&](const std::string& name, const half* d_ptr, int rows, int cols) {
            if (!debug_finite_checks) return;

            size_t count = (size_t)rows * (size_t)cols;
            std::vector<half> host(count);
            cudaMemcpy(host.data(), d_ptr, count * sizeof(half), cudaMemcpyDeviceToHost);

            std::ofstream log_stream;
            if (finite_log_path != nullptr) {
                log_stream.open(finite_log_path, std::ios::app);
            }

            auto emit = [&](const std::string& line) {
                std::cerr << line << std::endl;
                if (log_stream.is_open()) {
                    log_stream << line << '\n';
                }
            };

            for (size_t idx = 0; idx < count; idx++) {
                float value = __half2float(host[idx]);
                if (!std::isfinite(value)) {
                    int row = (int)(idx / (size_t)cols);
                    int col = (int)(idx % (size_t)cols);
                    emit("[NAN_CHECK] " + name +
                         " first_bad=(row=" + std::to_string(row) +
                         ", col=" + std::to_string(col) +
                         ", idx=" + std::to_string(idx) +
                         ") value=" + std::to_string(value));
                    return;
                }
            }

            emit("[NAN_CHECK] " + name + " all finite (" + std::to_string(count) + " values)");
        };

#define CHECK_CUDA_ERR(msg) \
        { \
            cudaError_t err = cudaGetLastError(); \
            if (err != cudaSuccess) { \
                std::cerr << "[ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl; \
            } \
        }

        // 1. Embed all tokens
        for (int i = 0; i < N; i++) {
            cudaMemcpy(d_prefill_x + (size_t)i * H,
                       weights.wte + (size_t)tokens[i] * H,
                       H * sizeof(half), cudaMemcpyDeviceToDevice);
        }
        cudaDeviceSynchronize();
        CHECK_CUDA_ERR("Embed tokens");

        // 2. Layers
        for (unsigned int l = 0; l < config.num_layers; l++) {
            if (l == 0) {
                dump_first_8("d_prefill_x (input)", d_prefill_x);
            }

            // --- Attention ---
            for (int i = 0; i < N; i++) {
                rmsnorm_kernel<<<1,512>>>(d_prefill_norm + (size_t)i * H,
                    d_prefill_x + (size_t)i * H, weights.ln_1[l], H, eps);
            }
            cudaDeviceSynchronize();
            CHECK_CUDA_ERR("RMSNorm 1");

            gemm(d_prefill_qkv, d_prefill_norm, weights.qkv_w[l], N, 3*H, H);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERR("QKV GEMM");

            if (l == 0) {
                dump_first_8("d_prefill_norm", d_prefill_norm);
                dump_first_8("d_prefill_qkv (before bias)", d_prefill_qkv);
            }

            for (int i = 0; i < N; i++) {
                add_bias_kernel<<<24,256>>>(d_prefill_qkv + (size_t)i * 3 * H, weights.qkv_b[l], 3*H);
                fused_qkv_rope_cache_kernel<<<NH,64>>>(d_prefill_qkv + (size_t)i * 3 * H,
                    d_kv_cache_k + (size_t)l * config.max_seqlen * H,
                    d_kv_cache_v + (size_t)l * config.max_seqlen * H,
                    i, config.max_seqlen, NH, kHD);
            }
            cudaDeviceSynchronize();
            CHECK_CUDA_ERR("QKV Bias + RoPE");

            if (l == 0) {
                dump_first_8("d_prefill_qkv (after rope)", d_prefill_qkv);
            }

            // Gather Q + K + V into per-head contiguous buffers
            gather_q_from_qkv(d_prefill_qkv, d_prefill_q, N, NH, kHD);
            gather_kv_from_cache(d_kv_cache_k + (size_t)l * config.max_seqlen * H,
                                 d_prefill_k, N, NH, kHD);
            gather_kv_from_cache(d_kv_cache_v + (size_t)l * config.max_seqlen * H,
                                 d_prefill_v, N, NH, kHD);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERR("Gather QKV");

            if (l == 0) {
                dump_first_8("d_prefill_q", d_prefill_q);
                dump_first_8("d_prefill_k", d_prefill_k);
                dump_first_8("d_prefill_v", d_prefill_v);
            }

            // Flash Attention per head
            {
                constexpr int Br=128, Bc=128;
                int Tr = (N + Br - 1) / Br;
                dim3 g(1, Tr), b(256);
                constexpr int Q_sz=Br*(16+8), K_sz=Bc*(16+8);
                int smem = 2*(Q_sz+K_sz)*sizeof(half);
                auto fa = flash_attn_tuned_A_kernel<kHD,16,8,16,8,1,1,16,1,0,8,8,8,2,1>;
                cudaFuncSetAttribute(fa, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

                for (int h = 0; h < NH; h++) {
                    half* Qh = d_prefill_q + h * N * kHD;
                    half* Kh = d_prefill_k + h * N * kHD;
                    half* Vh = d_prefill_v + h * N * kHD;
                    half* Oh = d_prefill_o + h * N * kHD;
                    fa<<<g,b,smem>>>(Qh,Kh,Vh,Oh,N,1,kHD,kHD,0,0,0,0);  // all ofs_override=0: per-head pointers
                }
                cudaDeviceSynchronize();
                CHECK_CUDA_ERR("Flash Attention");
            }

            if (l == 0) {
                dump_first_8("d_prefill_o (FA output)", d_prefill_o);
                scan_tensor_finite("d_prefill_o (FA output)", d_prefill_o, N * NH, kHD);
            }

            // Scatter FA output back to interleaved d_prefill_out
            scatter_interleaved_kernel<kHD,256><<<(N*NH*kHD+255)/256, 256>>>(
                d_prefill_o, d_prefill_out, N, NH, H);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERR("Scatter Interleaved");

            if (l == 0) {
                dump_first_8("d_prefill_out", d_prefill_out);
                scan_tensor_finite("d_prefill_out", d_prefill_out, N, H);
            }

            if (l == 0) {
                dump_first_8("weights.attn_proj_w[0]", weights.attn_proj_w[0]);
            }

            // Output proj + residual
            gemm(d_prefill_norm, d_prefill_out, weights.attn_proj_w[l], N, H, H);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERR("Attn Proj GEMM");

            if (l == 0) {
                dump_first_8("d_prefill_norm (after attn_proj gemm)", d_prefill_norm);
                scan_tensor_finite("d_prefill_norm (after attn_proj gemm)", d_prefill_norm, N, H);
            }

            for (int i = 0; i < N; i++) {
                add_bias_kernel<<<8,256>>>(d_prefill_x + (size_t)i * H, d_prefill_norm + (size_t)i * H, H);
            }
            cudaDeviceSynchronize();
            CHECK_CUDA_ERR("Attn Residual Bias");

            if (l == 0) {
                dump_first_8("d_prefill_x (after attn residual)", d_prefill_x);
            }

            // --- FFN ---
            for (int i = 0; i < N; i++) {
                rmsnorm_kernel<<<1,512>>>(d_prefill_norm + (size_t)i * H,
                    d_prefill_x + (size_t)i * H, weights.ln_2[l], H, eps);
            }
            cudaDeviceSynchronize();
            CHECK_CUDA_ERR("RMSNorm 2");

            if (l == 0) {
                dump_first_8("d_prefill_norm (after ln2 rmsnorm)", d_prefill_norm);
            }

            gemm(d_prefill_w1, d_prefill_norm, weights.ffn_w1[l], N, IM, H);
            gemm(d_prefill_w2, d_prefill_norm, weights.ffn_w2[l], N, IM, H);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERR("FFN GEMM 1/2");

            if (l == 0) {
                dump_first_8("d_prefill_w1", d_prefill_w1);
                dump_first_8("d_prefill_w2", d_prefill_w2);
            }

            for (int i = 0; i < N; i++) {
                swiglu_kernel<<<22,256>>>(d_prefill_act + (size_t)i * IM,
                    d_prefill_w1 + (size_t)i * IM, d_prefill_w2 + (size_t)i * IM, IM);
            }
            cudaDeviceSynchronize();

            if (l == 0) {
                dump_first_8("d_prefill_act (swiglu)", d_prefill_act);
            }

            gemm(d_prefill_proj, d_prefill_act, weights.ffn_proj_w[l], N, H, IM);
            cudaDeviceSynchronize();

            if (l == 0) {
                dump_first_8("d_prefill_proj", d_prefill_proj);
            }

            for (int i = 0; i < N; i++) {
                add_bias_kernel<<<8,256>>>(d_prefill_x + (size_t)i * H,
                    d_prefill_proj + (size_t)i * H, H);
            }
            cudaDeviceSynchronize();

            if (l == 0) {
                dump_first_8("d_prefill_x (output of L0)", d_prefill_x);
            }
        }
        pos = N;
    }

    // Execute single-token Decode step (greedy search)
    int step(int token_id, int pos) {
        // 1. Embedding lookup
        // Copys a row of hidden_size elements from wte based on token_id
        cudaMemcpy(d_x, weights.wte + token_id * config.hidden_size, config.hidden_size * sizeof(half), cudaMemcpyDeviceToDevice);
        
        // Constants
        constexpr float epsilon = 1e-6f;
        
        // 2. Transformer layers loop
        for (unsigned int i = 0; i < config.num_layers; i++) {
            // --- A. Self Attention Branch ---
            // RMSNorm
            rmsnorm_kernel<<<1, 512>>>(d_norm_x, d_x, weights.ln_1[i], config.hidden_size, epsilon);
            
            // QKV projection
            gemm(d_qkv, d_norm_x, weights.qkv_w[i], 1, 3 * config.hidden_size, config.hidden_size);
            // Add QKV bias
            add_bias_kernel<<<24, 256>>>(d_qkv, weights.qkv_b[i], 3 * config.hidden_size);
            
            // Fused separation, mathematically correct split-half RoPE rotation, and KV Cache write!
            fused_qkv_rope_cache_kernel<<<config.num_heads, 64>>>(
                d_qkv,
                d_kv_cache_k + (size_t)i * config.max_seqlen * config.hidden_size,
                d_kv_cache_v + (size_t)i * config.max_seqlen * config.hidden_size,
                pos,
                config.max_seqlen,
                config.num_heads,
                128
            );
            
            // Execute attention over historic sequence using our optimized Flash Decoding!
            // Every head is computed in parallel
            int current_seqlen = pos + 1;
            
            // Flash Decoding launch parameters
            constexpr int kHeadDim = 128;
            constexpr int kBc = 256;
            int num_chunks = (current_seqlen + kBc - 1) / kBc;
            
            // Pointers for current layer's KV cache history
            half* cur_layer_cache_k = d_kv_cache_k + (size_t)i * config.max_seqlen * config.hidden_size;
            half* cur_layer_cache_v = d_kv_cache_v + (size_t)i * config.max_seqlen * config.hidden_size;
            
            // Launch parallel Flash Decoding kernels for each head
            for (unsigned int h = 0; h < config.num_heads; h++) {
                half* head_q = d_qkv + h * kHeadDim;
                half* head_k_cache = cur_layer_cache_k + h * kHeadDim;
                half* head_v_cache = cur_layer_cache_v + h * kHeadDim;
                half* head_out = d_qkv_out + h * kHeadDim;
                
                if (num_chunks == 1) {
                    // Use Fused single-block Flash Decoding (sub-microsecond speed!)
                    fd_v5_fused_kernel<kHeadDim, 8, 32><<<1, 256>>>(
                        head_q, head_k_cache, head_v_cache, head_out, current_seqlen, config.hidden_size
                    );
                } else {
                    // Use Split Stage1 + Stage2 Flash Decoding for long context lengths
                    fd_v5_split_stage1_kernel<kHeadDim, kBc, 8><<<num_chunks, 256>>>(
                        head_q, head_k_cache, head_v_cache, d_fd_partial, d_fd_lse, current_seqlen, num_chunks, config.hidden_size
                    );
                    fd_opt_stage2_kernel<kHeadDim><<<1, 128>>>(
                        d_fd_partial, d_fd_lse, head_out, num_chunks
                    );
                }
            }
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "[ERROR] L" << i << " CUDA kernel execution failed: " << cudaGetErrorString(err) << std::endl;
            }
            if (i == 0) {
                half tmp[8];
                cudaMemcpy(tmp, d_qkv_out, 8 * sizeof(half), cudaMemcpyDeviceToHost);
                std::cerr << "[STEP_DUMP] L0 d_qkv_out: ";
                for (int idx = 0; idx < 8; idx++) {
                    std::cerr << __half2float(tmp[idx]) << " ";
                }
                std::cerr << std::endl;
            }
            
            // Attention Output Linear projection (c_proj)
            gemm(d_attn_proj, d_qkv_out, weights.attn_proj_w[i], 1, config.hidden_size, config.hidden_size);
            
            // Add residual connection: x = x + attn_proj(x)
            add_bias_kernel<<<8, 256>>>(d_x, d_attn_proj, config.hidden_size);
            
            // --- B. Feed Forward Network (FFN / SwiGLU) Branch ---
            // RMSNorm
            rmsnorm_kernel<<<1, 512>>>(d_norm_x, d_x, weights.ln_2[i], config.hidden_size, epsilon);
            
            // GEMM for FFN w1 (gate projection) and w2 (up projection)
            gemm(d_ffn_w1, d_norm_x, weights.ffn_w1[i], 1, config.intermediate_size, config.hidden_size);
            gemm(d_ffn_w2, d_norm_x, weights.ffn_w2[i], 1, config.intermediate_size, config.hidden_size);
            
            // Execute SwiGLU Activation in-place
            swiglu_kernel<<<22, 256>>>(d_ffn_act, d_ffn_w1, d_ffn_w2, config.intermediate_size);
            
            // GEMM for FFN output projection (ffn_proj / w3)
            gemm(d_ffn_proj, d_ffn_act, weights.ffn_proj_w[i], 1, config.hidden_size, config.intermediate_size);
            
            // Add residual connection: x = x + ffn_proj(x)
            add_bias_kernel<<<8, 256>>>(d_x, d_ffn_proj, config.hidden_size);
        }
        
        // 3. Final normalization
        rmsnorm_kernel<<<1, 512>>>(d_norm_x, d_x, weights.ln_f, config.hidden_size, epsilon);
        
        // 4. LM Head projection (produces logits)
        gemm(d_logits, d_norm_x, weights.lm_head, 1, config.vocab_size, config.hidden_size);
        
        // 5. Greedy sampling via ArgMax
        argmax_kernel<<<1, 512>>>(d_next_token, d_logits, config.vocab_size);
        
        // Copy next token ID back to CPU
        int next_token_id;
        cudaMemcpy(&next_token_id, d_next_token, sizeof(int), cudaMemcpyDeviceToHost);
        return next_token_id;
    }
};

// ============================================================
// C++ Command Line / Sidecar Endpoint Interface
// ============================================================
int main(int argc, char* argv[]) {
    // Enable raw output streaming with no stdout buffering
    std::cout << std::unitbuf;
    
    std::string model_path = "../model_weights/qwen_1.8b.bin";
    if (argc > 1) {
        model_path = argv[1];
    }
    
    // Initialize Qwen engine
    QwenEngine engine(model_path);
    std::cout << "\n[ENGINE_READY] CUDA Qwen-1.8B Inference Engine is fully ready." << std::endl;
    
    // Simple interactive shell loop for token streaming
    // Format: Web server passes token IDs over stdin in the format: "token_1 token_2 token_3 \n"
    while (true) {
        std::string line;
        if (!std::getline(std::cin, line)) break;
        
        if (line.empty() || line == "exit") break;
        if (line == "reset") {
            engine.reset();
            std::cout << "[RESET_DONE]" << std::endl;
            continue;
        }
        
        // Parse token IDs from standard input
        std::vector<int> prompt_tokens;
        std::string temp = "";
        for (char c : line) {
            if (c == ' ') {
                if (!temp.empty()) {
                    prompt_tokens.push_back(std::stoi(temp));
                    temp = "";
                }
            } else {
                temp += c;
            }
        }
        if (!temp.empty()) {
            prompt_tokens.push_back(std::stoi(temp));
        }
        
        if (prompt_tokens.empty()) continue;
        
        // 1. FA Prefill: batch through causal Flash Attention
        int pos = 0;
        int last_token;
        if (prompt_tokens.size() > 1) {
            std::vector<int> prefill_tokens(prompt_tokens.begin(), prompt_tokens.end() - 1);
            engine.prefill(prefill_tokens, pos);
        }
        last_token = prompt_tokens.back();
        
        // 2. Decode generation stage
        int max_new_tokens = 512;
        // Sync after prefill to ensure all kernels complete cleanly
        cudaDeviceSynchronize();
        std::cout << "[GENERATION_START]" << std::endl;
        
        for (int step_idx = 0; step_idx < max_new_tokens; step_idx++) {
            int next_token = engine.step(last_token, pos);
            pos++;
            
            // Qwen ChatML stop tokens: 151645=<|im_end|>, 151643=<|endoftext|>
            if (next_token == 151645 || next_token == 151643) {
                break;
            }
            
            // Print token and flush immediately for true streaming
            std::cout << next_token << std::endl;
            last_token = next_token;
        }
        // Ensure all GPU kernels finish before next turn (prevents async overlap corruption)
        cudaDeviceSynchronize();
        std::cout << "[GENERATION_END]" << std::endl;
    }
    
    return 0;
}
