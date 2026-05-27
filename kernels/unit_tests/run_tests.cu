#include <cstdint>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

// Helper to compare two half arrays
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

// Include kernels
#include "../rmsnorm.cuh"
#include "../bias.cuh"
#include "../embedding.cuh"
#include "../swiglu.cuh"
#include "../rope.cuh"
#include "../transpose.cuh"
#include "../sampler.cuh"
#include "../argmax.cuh"
#include "../fused_qkv_rope_cache.cuh"
#include "../flash_decode_q1.cuh"
#include "../flash_decode_q1_opt.cuh"

// Include matrix multiplication kernels (using #undef workaround to avoid macro redefinition errors)
#include "../hgemm.cuh"

#undef LDST128BITS
#undef LDST32BITS
#undef CP_ASYNC_CG
#undef CP_ASYNC_COMMIT_GROUP
#undef CP_ASYNC_WAIT_GROUP
#undef LDMATRIX_X4
#undef LDMATRIX_X2_T
#undef HMMA16816

#include "../hgemm_final.cuh"

// ============================================================
// Unit Test Implementations
// ============================================================

bool test_rmsnorm() {
    constexpr int d = 2048;
    constexpr float epsilon = 1e-6f;
    std::vector<half> h_in(d);
    std::vector<half> h_weight(d);
    std::vector<half> h_out_ref(d);
    std::vector<half> h_out_gpu(d);

    // Initialize random input
    float local_sum = 0.0f;
    for (int i = 0; i < d; i++) {
        float val = ((float)(rand() % 1000) / 1000.0f - 0.5f);
        h_in[i] = __float2half(val);
        local_sum += val * val;
        h_weight[i] = __float2half((float)(rand() % 1000) / 1000.0f + 0.5f);
    }
    float variance = local_sum / d;
    float rsqrt_val = 1.0f / sqrtf(variance + epsilon);
    for (int i = 0; i < d; i++) {
        float val = __half2float(h_in[i]);
        float w = __half2float(h_weight[i]);
        h_out_ref[i] = __float2half(val * rsqrt_val * w);
    }

    half *d_in, *d_weight, *d_out;
    cudaMalloc(&d_in, d * sizeof(half));
    cudaMalloc(&d_weight, d * sizeof(half));
    cudaMalloc(&d_out, d * sizeof(half));

    cudaMemcpy(d_in, h_in.data(), d * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), d * sizeof(half), cudaMemcpyHostToDevice);

    // Launch rmsnorm_kernel (blockDim = 512)
    rmsnorm_kernel<<<1, 512>>>(d_out, d_in, d_weight, d, epsilon);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_gpu.data(), d_out, d * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_weight); cudaFree(d_out);

    float max_err, mean_err;
    max_err = check_error(h_out_gpu.data(), h_out_ref.data(), d, &mean_err);
    printf("RMSNorm: Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 1e-2f ? "PASS" : "FAIL");
    return max_err < 1e-2f;
}

bool test_bias() {
    constexpr int size = 6144;
    std::vector<half> h_out(size);
    std::vector<half> h_bias(size);
    std::vector<half> h_ref(size);
    std::vector<half> h_gpu(size);

    for (int i = 0; i < size; i++) {
        float o = ((float)(rand() % 1000) / 1000.0f - 0.5f);
        float b = ((float)(rand() % 1000) / 1000.0f - 0.5f);
        h_out[i] = __float2half(o);
        h_bias[i] = __float2half(b);
        h_ref[i] = __float2half(o + b);
    }

    half *d_out, *d_bias;
    cudaMalloc(&d_out, size * sizeof(half));
    cudaMalloc(&d_bias, size * sizeof(half));

    cudaMemcpy(d_out, h_out.data(), size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), size * sizeof(half), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(d_out, d_bias, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gpu.data(), d_out, size * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_out); cudaFree(d_bias);

    float max_err, mean_err;
    max_err = check_error(h_gpu.data(), h_ref.data(), size, &mean_err);
    printf("Bias (add_bias): Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 1e-3f ? "PASS" : "FAIL");
    return max_err < 1e-3f;
}

bool test_embedding() {
    constexpr int vocab_size = 1000;
    constexpr int hidden_size = 2048;
    constexpr int seq_len = 8;
    
    std::vector<half> h_wte(vocab_size * hidden_size);
    std::vector<int> h_tokens(seq_len);
    std::vector<half> h_ref(seq_len * hidden_size);
    std::vector<half> h_gpu(seq_len * hidden_size);

    for (int i = 0; i < vocab_size * hidden_size; i++) {
        h_wte[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f));
    }
    for (int i = 0; i < seq_len; i++) {
        h_tokens[i] = rand() % vocab_size;
        for (int d = 0; d < hidden_size; d++) {
            h_ref[i * hidden_size + d] = h_wte[h_tokens[i] * hidden_size + d];
        }
    }

    half *d_wte, *d_out;
    int *d_tokens;
    cudaMalloc(&d_wte, vocab_size * hidden_size * sizeof(half));
    cudaMalloc(&d_tokens, seq_len * sizeof(int));
    cudaMalloc(&d_out, seq_len * hidden_size * sizeof(half));

    cudaMemcpy(d_wte, h_wte.data(), vocab_size * hidden_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tokens, h_tokens.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);

    embedding_lookup_kernel<<<seq_len, 512>>>(d_out, d_wte, d_tokens, hidden_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gpu.data(), d_out, seq_len * hidden_size * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_wte); cudaFree(d_tokens); cudaFree(d_out);

    float max_err, mean_err;
    max_err = check_error(h_gpu.data(), h_ref.data(), seq_len * hidden_size, &mean_err);
    printf("Embedding Lookup: Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 1e-5f ? "PASS" : "FAIL");
    return max_err < 1e-5f;
}

bool test_swiglu() {
    constexpr int size = 4096;
    std::vector<half> h_w1(size);
    std::vector<half> h_w2(size);
    std::vector<half> h_ref(size);
    std::vector<half> h_gpu(size);

    for (int i = 0; i < size; i++) {
        float w1 = ((float)(rand() % 1000) / 1000.0f - 0.5f);
        float w2 = ((float)(rand() % 1000) / 1000.0f - 0.5f);
        h_w1[i] = __float2half(w1);
        h_w2[i] = __float2half(w2);
        
        float silu_w2 = w2 / (1.0f + expf(-w2));
        h_ref[i] = __float2half(w1 * silu_w2);
    }

    half *d_w1, *d_w2, *d_out;
    cudaMalloc(&d_w1, size * sizeof(half));
    cudaMalloc(&d_w2, size * sizeof(half));
    cudaMalloc(&d_out, size * sizeof(half));

    cudaMemcpy(d_w1, h_w1.data(), size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2.data(), size * sizeof(half), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads>>>(d_out, d_w1, d_w2, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gpu.data(), d_out, size * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_w1); cudaFree(d_w2); cudaFree(d_out);

    float max_err, mean_err;
    max_err = check_error(h_gpu.data(), h_ref.data(), size, &mean_err);
    printf("SwiGLU: Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 1e-3f ? "PASS" : "FAIL");
    return max_err < 1e-3f;
}

bool test_rope() {
    constexpr int num_heads = 16;
    constexpr int head_dim = 128;
    constexpr int pos = 5;
    constexpr int total = num_heads * head_dim;

    std::vector<half> h_Q(total);
    std::vector<half> h_K(total);
    std::vector<half> h_Q_ref(total);
    std::vector<half> h_K_ref(total);
    std::vector<half> h_Q_gpu(total);
    std::vector<half> h_K_gpu(total);

    for (int i = 0; i < total; i++) {
        h_Q[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f));
        h_K[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f));
    }

    int half_dim = head_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        for (int tid = 0; tid < half_dim; tid++) {
            int idx1 = h * head_dim + tid;
            int idx2 = h * head_dim + half_dim + tid;
            float q1 = __half2float(h_Q[idx1]);
            float q2 = __half2float(h_Q[idx2]);
            float k1 = __half2float(h_K[idx1]);
            float k2 = __half2float(h_K[idx2]);

            float theta = pos * powf(10000.0f, -2.0f * tid / head_dim);
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);

            h_Q_ref[idx1] = __float2half(q1 * cos_t - q2 * sin_t);
            h_Q_ref[idx2] = __float2half(q2 * cos_t + q1 * sin_t);
            h_K_ref[idx1] = __float2half(k1 * cos_t - k2 * sin_t);
            h_K_ref[idx2] = __float2half(k2 * cos_t + k1 * sin_t);
        }
    }

    half *d_Q, *d_K;
    cudaMalloc(&d_Q, total * sizeof(half));
    cudaMalloc(&d_K, total * sizeof(half));

    cudaMemcpy(d_Q, h_Q.data(), total * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), total * sizeof(half), cudaMemcpyHostToDevice);

    rope_kernel<<<num_heads, 64>>>(d_Q, d_K, pos, num_heads, head_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Q_gpu.data(), d_Q, total * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_K_gpu.data(), d_K, total * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_Q); cudaFree(d_K);

    float q_max_err, q_mean_err;
    float k_max_err, k_mean_err;
    q_max_err = check_error(h_Q_gpu.data(), h_Q_ref.data(), total, &q_mean_err);
    k_max_err = check_error(h_K_gpu.data(), h_K_ref.data(), total, &k_mean_err);

    printf("RoPE Q: Max err = %.6f, Mean err = %.6f -> %s\n", q_max_err, q_mean_err, q_max_err < 1e-3f ? "PASS" : "FAIL");
    printf("RoPE K: Max err = %.6f, Mean err = %.6f -> %s\n", k_max_err, k_mean_err, k_max_err < 1e-3f ? "PASS" : "FAIL");
    return (q_max_err < 1e-3f) && (k_max_err < 1e-3f);
}

bool test_transpose() {
    constexpr int rows = 128;
    constexpr int cols = 256;
    constexpr int total = rows * cols;

    std::vector<half> h_in(total);
    std::vector<half> h_ref(total);
    std::vector<half> h_gpu(total);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            h_in[r * cols + c] = __float2half((float)(r * cols + c) / 1000.0f);
            h_ref[c * rows + r] = h_in[r * cols + c];
        }
    }

    half *d_weight;
    cudaMalloc(&d_weight, total * sizeof(half));
    cudaMemcpy(d_weight, h_in.data(), total * sizeof(half), cudaMemcpyHostToDevice);

    transpose_weight_gpu(d_weight, rows, cols);

    cudaMemcpy(h_gpu.data(), d_weight, total * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(d_weight);

    float max_err, mean_err;
    max_err = check_error(h_gpu.data(), h_ref.data(), total, &mean_err);
    printf("Transpose: Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 1e-5f ? "PASS" : "FAIL");
    return max_err < 1e-5f;
}

bool test_sampler() {
    constexpr int vocab_size = 5000;
    constexpr float temperature = 0.7f;
    constexpr float rand_val = 0.35f;

    std::vector<half> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        h_logits[i] = __float2half(((float)(rand() % 1000) / 100.0f - 5.0f));
    }

    // CPU Reference
    float max_logit = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        float val = __half2float(h_logits[i]);
        if (val > max_logit) max_logit = val;
    }
    float sum_exp = 0.0f;
    std::vector<float> probs(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        float val = __half2float(h_logits[i]);
        probs[i] = expf((val - max_logit) / temperature);
        sum_exp += probs[i];
    }

    float target = rand_val * sum_exp;
    float cumulative = 0.0f;
    int ref_token = -1;
    for (int i = 0; i < vocab_size; i++) {
        cumulative += probs[i];
        if (cumulative >= target) {
            ref_token = i;
            break;
        }
    }
    if (ref_token == -1) ref_token = vocab_size - 1;

    half *d_logits;
    int *d_out_token;
    cudaMalloc(&d_logits, vocab_size * sizeof(half));
    cudaMalloc(&d_out_token, sizeof(int));

    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(half), cudaMemcpyHostToDevice);

    temperature_sampler_kernel<<<1, 512>>>(d_out_token, d_logits, rand_val, temperature, vocab_size);
    cudaDeviceSynchronize();

    int gpu_token = -1;
    cudaMemcpy(&gpu_token, d_out_token, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_logits); cudaFree(d_out_token);

    bool pass = (gpu_token == ref_token) || (abs(gpu_token - ref_token) <= 1);
    printf("Sampler: GPU Token = %d, CPU Token = %d -> %s\n", gpu_token, ref_token, pass ? "PASS" : "FAIL");
    return pass;
}

bool test_argmax() {
    constexpr int vocab_size = 10000;
    std::vector<half> h_logits(vocab_size);

    for (int i = 0; i < vocab_size; i++) {
        float val = ((float)(rand() % 10000) / 100.0f - 50.0f);
        h_logits[i] = __float2half(val);
    }

    // Assign a unique dominant maximum value to a random index
    int ref_token = rand() % vocab_size;
    h_logits[ref_token] = __float2half(100.0f);

    half *d_logits;
    int *d_out_token;
    cudaMalloc(&d_logits, vocab_size * sizeof(half));
    cudaMalloc(&d_out_token, sizeof(int));

    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(half), cudaMemcpyHostToDevice);

    argmax_kernel<<<1, 512>>>(d_out_token, d_logits, vocab_size);
    cudaDeviceSynchronize();

    int gpu_token = -1;
    cudaMemcpy(&gpu_token, d_out_token, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_logits); cudaFree(d_out_token);

    bool pass = (gpu_token == ref_token);
    printf("ArgMax: GPU Token = %d, CPU Token = %d -> %s\n", gpu_token, ref_token, pass ? "PASS" : "FAIL");
    return pass;
}

bool test_fused_qkv_rope_cache() {
    constexpr int num_heads = 16;
    constexpr int head_dim = 128;
    constexpr int hidden_size = num_heads * head_dim;
    constexpr int pos = 3;
    constexpr int max_seqlen = 10;

    std::vector<half> h_qkv(3 * hidden_size);
    std::vector<half> h_qkv_ref(3 * hidden_size);
    std::vector<half> h_cache_k_ref(max_seqlen * hidden_size, __float2half(0.0f));
    std::vector<half> h_cache_v_ref(max_seqlen * hidden_size, __float2half(0.0f));
    std::vector<half> h_qkv_gpu(3 * hidden_size);
    std::vector<half> h_cache_k_gpu(max_seqlen * hidden_size);
    std::vector<half> h_cache_v_gpu(max_seqlen * hidden_size);

    for (int i = 0; i < 3 * hidden_size; i++) {
        h_qkv[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f));
        h_qkv_ref[i] = h_qkv[i];
    }

    // CPU Reference
    int half_dim = head_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        for (int tid = 0; tid < half_dim; tid++) {
            int idx1 = h * head_dim + tid;
            int idx2 = h * head_dim + half_dim + tid;
            
            float q1 = __half2float(h_qkv[0 * hidden_size + idx1]);
            float q2 = __half2float(h_qkv[0 * hidden_size + idx2]);
            float k1 = __half2float(h_qkv[1 * hidden_size + idx1]);
            float k2 = __half2float(h_qkv[1 * hidden_size + idx2]);
            float v1 = __half2float(h_qkv[2 * hidden_size + idx1]);
            float v2 = __half2float(h_qkv[2 * hidden_size + idx2]);

            float theta = pos * powf(10000.0f, -2.0f * tid / head_dim);
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);

            h_qkv_ref[0 * hidden_size + idx1] = __float2half(q1 * cos_t - q2 * sin_t);
            h_qkv_ref[0 * hidden_size + idx2] = __float2half(q2 * cos_t + q1 * sin_t);
            
            h_cache_k_ref[pos * hidden_size + idx1] = __float2half(k1 * cos_t - k2 * sin_t);
            h_cache_k_ref[pos * hidden_size + idx2] = __float2half(k2 * cos_t + k1 * sin_t);
            h_cache_v_ref[pos * hidden_size + idx1] = __float2half(v1);
            h_cache_v_ref[pos * hidden_size + idx2] = __float2half(v2);
        }
    }

    half *d_qkv, *d_cache_k, *d_cache_v;
    cudaMalloc(&d_qkv, 3 * hidden_size * sizeof(half));
    cudaMalloc(&d_cache_k, max_seqlen * hidden_size * sizeof(half));
    cudaMalloc(&d_cache_v, max_seqlen * hidden_size * sizeof(half));

    cudaMemcpy(d_qkv, h_qkv.data(), 3 * hidden_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_cache_k, 0, max_seqlen * hidden_size * sizeof(half));
    cudaMemset(d_cache_v, 0, max_seqlen * hidden_size * sizeof(half));

    fused_qkv_rope_cache_kernel<<<num_heads, 64>>>(d_qkv, d_cache_k, d_cache_v, pos, max_seqlen, num_heads, head_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_qkv_gpu.data(), d_qkv, 3 * hidden_size * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cache_k_gpu.data(), d_cache_k, max_seqlen * hidden_size * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cache_v_gpu.data(), d_cache_v, max_seqlen * hidden_size * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_qkv); cudaFree(d_cache_k); cudaFree(d_cache_v);

    float qkv_max_err, qkv_mean_err;
    float k_max_err, k_mean_err;
    float v_max_err, v_mean_err;

    qkv_max_err = check_error(h_qkv_gpu.data(), h_qkv_ref.data(), 3 * hidden_size, &qkv_mean_err);
    k_max_err = check_error(h_cache_k_gpu.data(), h_cache_k_ref.data(), max_seqlen * hidden_size, &k_mean_err);
    v_max_err = check_error(h_cache_v_gpu.data(), h_cache_v_ref.data(), max_seqlen * hidden_size, &v_mean_err);

    printf("Fused QKV RoPE Cache (QKV): Max err = %.6f, Mean err = %.6f -> %s\n", qkv_max_err, qkv_mean_err, qkv_max_err < 1e-3f ? "PASS" : "FAIL");
    printf("Fused QKV RoPE Cache (K Cache): Max err = %.6f, Mean err = %.6f -> %s\n", k_max_err, k_mean_err, k_max_err < 1e-3f ? "PASS" : "FAIL");
    printf("Fused QKV RoPE Cache (V Cache): Max err = %.6f, Mean err = %.6f -> %s\n", v_max_err, v_mean_err, v_max_err < 1e-3f ? "PASS" : "FAIL");

    return (qkv_max_err < 1e-3f) && (k_max_err < 1e-3f) && (v_max_err < 1e-3f);
}

bool test_flash_decode_q1() {
    constexpr int d = 128;
    constexpr int KV_seqlen = 128;
    constexpr int num_chunks = (KV_seqlen + 15) / 16;

    std::vector<half> h_Q(d);
    std::vector<half> h_K(KV_seqlen * d);
    std::vector<half> h_V(KV_seqlen * d);
    std::vector<half> h_ref_O(d);
    std::vector<half> h_gpu_O(d);

    for (int i = 0; i < d; i++) h_Q[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f));
    for (int i = 0; i < KV_seqlen * d; i++) {
        h_K[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f));
        h_V[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f));
    }

    // CPU Reference
    float scale = 1.0f / sqrtf((float)d);
    std::vector<float> scores(KV_seqlen);
    float max_score = -INFINITY;
    for (int i = 0; i < KV_seqlen; i++) {
        float dot = 0.0f;
        for (int j = 0; j < d; j++) {
            dot += __half2float(h_Q[j]) * __half2float(h_K[i * d + j]);
        }
        scores[i] = dot * scale;
        if (scores[i] > max_score) max_score = scores[i];
    }
    float sum_exp = 0.0f;
    std::vector<float> probs(KV_seqlen);
    for (int i = 0; i < KV_seqlen; i++) {
        probs[i] = expf(scores[i] - max_score);
        sum_exp += probs[i];
    }
    for (int i = 0; i < KV_seqlen; i++) probs[i] /= sum_exp;
    std::vector<float> O_f32(d, 0.0f);
    for (int j = 0; j < d; j++) {
        for (int i = 0; i < KV_seqlen; i++) {
            O_f32[j] += probs[i] * __half2float(h_V[i * d + j]);
        }
        h_ref_O[j] = __float2half(O_f32[j]);
    }

    half *d_Q, *d_K, *d_V, *d_O_partial, *d_O_final;
    float *d_LSE;
    cudaMalloc(&d_Q, d * sizeof(half));
    cudaMalloc(&d_K, KV_seqlen * d * sizeof(half));
    cudaMalloc(&d_V, KV_seqlen * d * sizeof(half));
    cudaMalloc(&d_O_partial, num_chunks * d * sizeof(half));
    cudaMalloc(&d_LSE, num_chunks * sizeof(float));
    cudaMalloc(&d_O_final, d * sizeof(half));

    cudaMemcpy(d_Q, h_Q.data(), d * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), KV_seqlen * d * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), KV_seqlen * d * sizeof(half), cudaMemcpyHostToDevice);

    flash_decode_q1_stage1_kernel<<<num_chunks, 32>>>(d_Q, d_K, d_V, d_O_partial, d_LSE, KV_seqlen, num_chunks, d);
    cudaDeviceSynchronize();

    flash_decode_q1_stage2_kernel<<<1, 32>>>(d_O_partial, d_LSE, d_O_final, num_chunks, d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gpu_O.data(), d_O_final, d * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O_partial); cudaFree(d_LSE); cudaFree(d_O_final);

    float max_err, mean_err;
    max_err = check_error(h_gpu_O.data(), h_ref_O.data(), d, &mean_err);
    printf("Flash Decode Q1 (Baseline): Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 5e-2f ? "PASS" : "FAIL");
    return max_err < 5e-2f;
}

bool test_flash_decode_q1_opt() {
    constexpr int d = 128;
    
    // 1. Fused v5
    {
        constexpr int KV_seqlen = 256;
        std::vector<half> h_Q(d);
        std::vector<half> h_K(KV_seqlen * d);
        std::vector<half> h_V(KV_seqlen * d);
        std::vector<half> h_ref_O(d);
        std::vector<half> h_gpu_O(d);

        for (int i = 0; i < d; i++) h_Q[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f));
        for (int i = 0; i < KV_seqlen * d; i++) {
            h_K[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f));
            h_V[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f));
        }

        float scale = 1.0f / sqrtf((float)d);
        std::vector<float> scores(KV_seqlen);
        float max_score = -INFINITY;
        for (int i = 0; i < KV_seqlen; i++) {
            float dot = 0.0f;
            for (int j = 0; j < d; j++) {
                dot += __half2float(h_Q[j]) * __half2float(h_K[i * d + j]);
            }
            scores[i] = dot * scale;
            if (scores[i] > max_score) max_score = scores[i];
        }
        float sum_exp = 0.0f;
        std::vector<float> probs(KV_seqlen);
        for (int i = 0; i < KV_seqlen; i++) {
            probs[i] = expf(scores[i] - max_score);
            sum_exp += probs[i];
        }
        for (int i = 0; i < KV_seqlen; i++) probs[i] /= sum_exp;
        std::vector<float> O_f32(d, 0.0f);
        for (int j = 0; j < d; j++) {
            for (int i = 0; i < KV_seqlen; i++) {
                O_f32[j] += probs[i] * __half2float(h_V[i * d + j]);
            }
            h_ref_O[j] = __float2half(O_f32[j]);
        }

        half *d_Q, *d_K, *d_V, *d_O_final;
        cudaMalloc(&d_Q, d * sizeof(half));
        cudaMalloc(&d_K, KV_seqlen * d * sizeof(half));
        cudaMalloc(&d_V, KV_seqlen * d * sizeof(half));
        cudaMalloc(&d_O_final, d * sizeof(half));

        cudaMemcpy(d_Q, h_Q.data(), d * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K.data(), KV_seqlen * d * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V.data(), KV_seqlen * d * sizeof(half), cudaMemcpyHostToDevice);

        fd_v5_fused_kernel<128, 8, 32><<<1, 256>>>(d_Q, d_K, d_V, d_O_final, KV_seqlen);
        cudaDeviceSynchronize();

        cudaMemcpy(h_gpu_O.data(), d_O_final, d * sizeof(half), cudaMemcpyDeviceToHost);

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O_final);

        float max_err, mean_err;
        max_err = check_error(h_gpu_O.data(), h_ref_O.data(), d, &mean_err);
        printf("Flash Decode Q1 Opt (Fused v5): Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 5e-2f ? "PASS" : "FAIL");
        if (max_err >= 5e-2f) return false;
    }

    // 2. Split v5
    {
        constexpr int KV_seqlen = 1024;
        constexpr int kBc = 256;
        constexpr int num_chunks = (KV_seqlen + kBc - 1) / kBc;
        std::vector<half> h_Q(d);
        std::vector<half> h_K(KV_seqlen * d);
        std::vector<half> h_V(KV_seqlen * d);
        std::vector<half> h_ref_O(d);
        std::vector<half> h_gpu_O(d);

        for (int i = 0; i < d; i++) h_Q[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f));
        for (int i = 0; i < KV_seqlen * d; i++) {
            h_K[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f));
            h_V[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f));
        }

        float scale = 1.0f / sqrtf((float)d);
        std::vector<float> scores(KV_seqlen);
        float max_score = -INFINITY;
        for (int i = 0; i < KV_seqlen; i++) {
            float dot = 0.0f;
            for (int j = 0; j < d; j++) {
                dot += __half2float(h_Q[j]) * __half2float(h_K[i * d + j]);
            }
            scores[i] = dot * scale;
            if (scores[i] > max_score) max_score = scores[i];
        }
        float sum_exp = 0.0f;
        std::vector<float> probs(KV_seqlen);
        for (int i = 0; i < KV_seqlen; i++) {
            probs[i] = expf(scores[i] - max_score);
            sum_exp += probs[i];
        }
        for (int i = 0; i < KV_seqlen; i++) probs[i] /= sum_exp;
        std::vector<float> O_f32(d, 0.0f);
        for (int j = 0; j < d; j++) {
            for (int i = 0; i < KV_seqlen; i++) {
                O_f32[j] += probs[i] * __half2float(h_V[i * d + j]);
            }
            h_ref_O[j] = __float2half(O_f32[j]);
        }

        half *d_Q, *d_K, *d_V, *d_O_partial, *d_O_final;
        float *d_LSE;
        cudaMalloc(&d_Q, d * sizeof(half));
        cudaMalloc(&d_K, KV_seqlen * d * sizeof(half));
        cudaMalloc(&d_V, KV_seqlen * d * sizeof(half));
        cudaMalloc(&d_O_partial, num_chunks * d * sizeof(half));
        cudaMalloc(&d_LSE, num_chunks * sizeof(float));
        cudaMalloc(&d_O_final, d * sizeof(half));

        cudaMemcpy(d_Q, h_Q.data(), d * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K.data(), KV_seqlen * d * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V.data(), KV_seqlen * d * sizeof(half), cudaMemcpyHostToDevice);

        fd_v5_split_stage1_kernel<128, 256, 8><<<num_chunks, 256>>>(d_Q, d_K, d_V, d_O_partial, d_LSE, KV_seqlen, num_chunks);
        cudaDeviceSynchronize();

        fd_opt_stage2_kernel<128><<<1, 128>>>(d_O_partial, d_LSE, d_O_final, num_chunks);
        cudaDeviceSynchronize();

        cudaMemcpy(h_gpu_O.data(), d_O_final, d * sizeof(half), cudaMemcpyDeviceToHost);

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O_partial); cudaFree(d_LSE); cudaFree(d_O_final);

        float max_err, mean_err;
        max_err = check_error(h_gpu_O.data(), h_ref_O.data(), d, &mean_err);
        printf("Flash Decode Q1 Opt (Split v5): Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 5e-2f ? "PASS" : "FAIL");
        if (max_err >= 5e-2f) return false;
    }

    return true;
}

bool test_hgemm() {
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;
    constexpr int total_A = M * K;
    constexpr int total_B = K * N;
    constexpr int total_C = M * N;

    std::vector<half> h_A(total_A);
    std::vector<half> h_B(total_B);
    std::vector<half> h_ref_C(total_C);
    std::vector<half> h_gpu_C(total_C);

    for (int i = 0; i < total_A; i++) h_A[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f) * 0.1f);
    for (int i = 0; i < total_B; i++) h_B[i] = __float2half(((float)(rand() % 1000) / 1000.0f - 0.5f) * 0.1f);

    // CPU Matrix Multiplication
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(h_A[r * K + k]) * __half2float(h_B[k * N + c]);
            }
            h_ref_C[r * N + c] = __float2half(sum);
        }
    }

    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, total_A * sizeof(half));
    cudaMalloc(&d_B, total_B * sizeof(half));
    cudaMalloc(&d_C, total_C * sizeof(half));

    cudaMemcpy(d_A, h_A.data(), total_A * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), total_B * sizeof(half), cudaMemcpyHostToDevice);

    float max_err, mean_err;
    bool all_pass = true;

    // 1. Naive HGEMM
    {
        cudaMemset(d_C, 0, total_C * sizeof(half));
        dim3 grid((N + 7) / 8, (M + 15) / 16);
        dim3 block(WARP_SIZE);
        hgemm_naive_kernel<1, 1><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        cudaMemcpy(h_gpu_C.data(), d_C, total_C * sizeof(half), cudaMemcpyDeviceToHost);
        max_err = check_error(h_gpu_C.data(), h_ref_C.data(), total_C, &mean_err);
        printf("HGEMM Naive: Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 2e-2f ? "PASS" : "FAIL");
        if (max_err >= 2e-2f) all_pass = false;
    }

    // 2. Tiled HGEMM
    {
        cudaMemset(d_C, 0, total_C * sizeof(half));
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        dim3 block(WARP_SIZE * 4);
        hgemm_tiled_kernel<64, 64, 4><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        cudaMemcpy(h_gpu_C.data(), d_C, total_C * sizeof(half), cudaMemcpyDeviceToHost);
        max_err = check_error(h_gpu_C.data(), h_ref_C.data(), total_C, &mean_err);
        printf("HGEMM Tiled: Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 2e-2f ? "PASS" : "FAIL");
        if (max_err >= 2e-2f) all_pass = false;
    }

    // 3. Async HGEMM
    {
        cudaMemset(d_C, 0, total_C * sizeof(half));
        dim3 grid((N + 127) / 128, (M + 127) / 128);
        dim3 block(WARP_SIZE * 8);
        int smem = 2 * (128 * 16 + 16 * 128) * sizeof(half); // 16 KB
        
        auto fn = hgemm_async_kernel<128, 128, 8>;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        fn<<<grid, block, smem>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_gpu_C.data(), d_C, total_C * sizeof(half), cudaMemcpyDeviceToHost);
        max_err = check_error(h_gpu_C.data(), h_ref_C.data(), total_C, &mean_err);
        printf("HGEMM Async: Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 2e-2f ? "PASS" : "FAIL");
        if (max_err >= 2e-2f) all_pass = false;
    }

    // 4. Final HGEMM (hgemm.cuh)
    {
        cudaMemset(d_C, 0, total_C * sizeof(half));
        dim3 grid((N + 127) / 128, (M + 127) / 128);
        dim3 block(WARP_SIZE * 8);
        int smem = 2 * (128 * 32 + 32 * 128) * sizeof(half); // 32 KB
        
        auto fn = hgemm_final_kernel<128, 128, 2, 4, 4, 4, 2, false>;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        fn<<<grid, block, smem>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_gpu_C.data(), d_C, total_C * sizeof(half), cudaMemcpyDeviceToHost);
        max_err = check_error(h_gpu_C.data(), h_ref_C.data(), total_C, &mean_err);
        printf("HGEMM Final (hgemm.cuh): Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 2e-2f ? "PASS" : "FAIL");
        if (max_err >= 2e-2f) all_pass = false;
    }

    // 5. Opt HGEMM (hgemm_final.cuh)
    {
        cudaMemset(d_C, 0, total_C * sizeof(half));
        dim3 grid((N + 127) / 128, (M + 127) / 128);
        dim3 block(WARP_SIZE * 8);
        int smem = 2 * (128 * 32 + 32 * 128) * sizeof(half); // 32 KB
        
        auto fn = hgemm_opt_kernel<128, 128, 2, 4, 4, 4, 2, false, 2, 0>;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        fn<<<grid, block, smem>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_gpu_C.data(), d_C, total_C * sizeof(half), cudaMemcpyDeviceToHost);
        max_err = check_error(h_gpu_C.data(), h_ref_C.data(), total_C, &mean_err);
        printf("HGEMM Opt (hgemm_final.cuh): Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 5e-2f ? "PASS" : "FAIL");
        if (max_err >= 5e-2f) all_pass = false;
    }

    // 6. Opt HGEMV (hgemm_final.cuh)
    {
        std::vector<half> h_A_v(K);
        std::vector<half> h_ref_C_v(N);
        std::vector<half> h_gpu_C_v(N);
        
        for (int i = 0; i < K; i++) h_A_v[i] = h_A[i]; // Use first row of A
        
        // CPU Matrix-Vector Multiplication
        for (int c = 0; c < N; c++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(h_A_v[k]) * __half2float(h_B[k * N + c]);
            }
            h_ref_C_v[c] = __float2half(sum);
        }
        
        half *d_A_v, *d_C_v;
        cudaMalloc(&d_A_v, K * sizeof(half));
        cudaMalloc(&d_C_v, N * sizeof(half));
        cudaMemcpy(d_A_v, h_A_v.data(), K * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemset(d_C_v, 0, N * sizeof(half));
        
        int block_size = 256;
        int grid_size = (N + block_size - 1) / block_size;
        hgemv_kernel<<<grid_size, block_size>>>(d_A_v, d_B, d_C_v, N, K);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_gpu_C_v.data(), d_C_v, N * sizeof(half), cudaMemcpyDeviceToHost);
        max_err = check_error(h_gpu_C_v.data(), h_ref_C_v.data(), N, &mean_err);
        printf("HGEMV Opt (hgemm_final.cuh): Max err = %.6f, Mean err = %.6f -> %s\n", max_err, mean_err, max_err < 1e-3f ? "PASS" : "FAIL");
        if (max_err >= 1e-3f) all_pass = false;
        
        cudaFree(d_A_v); cudaFree(d_C_v);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return all_pass;
}

// ============================================================
// Main Execution Harness
// ============================================================

int main() {
    printf("==================================================\n");
    printf("      Starting Unified Custom CUDA Kernel Unit Tests\n");
    printf("==================================================\n\n");

    srand(42);

    int passed = 0;
    int total = 12;

    if (test_rmsnorm()) passed++;
    printf("--------------------------------------------------\n");
    if (test_bias()) passed++;
    printf("--------------------------------------------------\n");
    if (test_embedding()) passed++;
    printf("--------------------------------------------------\n");
    if (test_swiglu()) passed++;
    printf("--------------------------------------------------\n");
    if (test_rope()) passed++;
    printf("--------------------------------------------------\n");
    if (test_transpose()) passed++;
    printf("--------------------------------------------------\n");
    if (test_sampler()) passed++;
    printf("--------------------------------------------------\n");
    if (test_argmax()) passed++;
    printf("--------------------------------------------------\n");
    if (test_fused_qkv_rope_cache()) passed++;
    printf("--------------------------------------------------\n");
    if (test_flash_decode_q1()) passed++;
    printf("--------------------------------------------------\n");
    if (test_flash_decode_q1_opt()) passed++;
    printf("--------------------------------------------------\n");
    if (test_hgemm()) passed++;
    printf("==================================================\n");

    printf("Unit Tests Completed: %d/%d Passed.\n", passed, total);
    if (passed == total) {
        printf("RESULT: [ALL KERNELS PASSED ACCURACY UNIT TESTS]\n");
    } else {
        printf("RESULT: [SOME KERNELS FAILED ACCURACY UNIT TESTS]\n");
    }
    printf("==================================================\n");

    return (passed == total) ? 0 : 1;
}
