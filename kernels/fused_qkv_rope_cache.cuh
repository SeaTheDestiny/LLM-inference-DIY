#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Fused QKV separation, mathematically correct split-half RoPE rotation, and KV Cache write kernel for Qwen
__global__ void fused_qkv_rope_cache_kernel(
    half* d_qkv,               // QKV output, size = 3 * hidden_size
    half* d_kv_cache_k,        // K cache of current layer, size = max_seqlen * hidden_size
    half* d_kv_cache_v,        // V cache of current layer, size = max_seqlen * hidden_size
    int pos,
    int max_seqlen,
    int num_heads,
    int head_dim
) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x; // 0 to 63
    int hidden_size = num_heads * head_dim;
    int half_dim = head_dim / 2;
    
    int q_offset = 0;
    int k_offset = hidden_size;
    int v_offset = 2 * hidden_size;
    
    int idx1 = head_idx * head_dim + tid;
    int idx2 = head_idx * head_dim + half_dim + tid;
    
    // 1. Read Q, K, V elements
    float q1 = __half2float(d_qkv[q_offset + idx1]);
    float q2 = __half2float(d_qkv[q_offset + idx2]);
    float k1 = __half2float(d_qkv[k_offset + idx1]);
    float k2 = __half2float(d_qkv[k_offset + idx2]);
    float v1 = __half2float(d_qkv[v_offset + idx1]);
    float v2 = __half2float(d_qkv[v_offset + idx2]);
    
    // 2. Perform mathematically correct split-half RoPE rotation on Q and K
    float theta = pos * powf(10000.0f, -2.0f * tid / head_dim);
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    
    half q1_rot = __float2half(q1 * cos_t - q2 * sin_t);
    half q2_rot = __float2half(q2 * cos_t + q1 * sin_t);
    half k1_rot = __float2half(k1 * cos_t - k2 * sin_t);
    half k2_rot = __float2half(k2 * cos_t + k1 * sin_t);
    
    // 3. Write rotated Q back to QKV buffer in-place
    d_qkv[idx1] = q1_rot;
    d_qkv[idx2] = q2_rot;
    
    // 4. Write K and V directly to the KV cache matrix at pos offset
    int cache_offset = pos * hidden_size;
    d_kv_cache_k[cache_offset + idx1] = k1_rot;
    d_kv_cache_k[cache_offset + idx2] = k2_rot;
    d_kv_cache_v[cache_offset + idx1] = __float2half(v1);
    d_kv_cache_v[cache_offset + idx2] = __float2half(v2);
}
