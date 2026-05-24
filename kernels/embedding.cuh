#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Custom Parallel Embedding Lookup Kernel
__global__ void embedding_lookup_kernel(
    half* out,                  // [seq_len, hidden_size]
    const half* wte,            // [vocab_size, hidden_size]
    const int* token_ids,       // [seq_len]
    int hidden_size
) {
    int seq_idx = blockIdx.x;   // Which token in the sequence
    int tid = threadIdx.x;      // Thread inside block
    int num_threads = blockDim.x;
    
    int token_id = token_ids[seq_idx];
    
    // Each thread processes multiple elements if hidden_size > num_threads
    int items_per_thread = (hidden_size + num_threads - 1) / num_threads;
    
    #pragma unroll
    for (int i = 0; i < items_per_thread; i++) {
        int d = tid * items_per_thread + i;
        if (d < hidden_size) {
            out[seq_idx * hidden_size + d] = wte[token_id * hidden_size + d];
        }
    }
}
