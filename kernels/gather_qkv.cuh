/**
 * gather_qkv.cuh — Gather interleaved Q/KV to per-head contiguous layout
 * =========================================================================
 * Input:  QKV [N, 3H]  or  KV-cache [max_seqlen, H]   (heads interleaved)
 * Output: [numHeads * N, headDim]   (each head's tokens contiguous)
 *
 * Used to feed Flash Attention which requires per-head contiguous data.
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <int kHeadDim = 128, int kBlockSize = 256>
__global__ void gather_interleaved_kernel(
    const half* __restrict__ src, half* __restrict__ dst,
    int N, int numHeads, int src_stride_cols)  // src_stride_cols = H or 3H
{
    //           src: [N, totalCols] row-major,  where totalCols = numHeads * kHeadDim (or *3 for QKV)
    // →         dst: [numHeads * N, kHeadDim]  row-major (each head h has N consecutive rows)
    //
    // src[i][h * kHeadDim + j]  →  dst[(h * N + i) * kHeadDim + j]

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = N * numHeads * kHeadDim;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elems; idx += stride) {
        int elem_within_head = idx % kHeadDim;        // j
        int head = (idx / kHeadDim) % numHeads;        // h
        int row  = idx / (kHeadDim * numHeads);        // i (token index within N)

        int src_ofs = row * src_stride_cols + head * kHeadDim + elem_within_head;
        int dst_ofs = (head * N + row) * kHeadDim + elem_within_head;

        dst[dst_ofs] = src[src_ofs];
    }
}

// Scatter FA output back to interleaved layout
template <int kHeadDim = 128, int kBlockSize = 256>
__global__ void scatter_interleaved_kernel(
    const half* __restrict__ src, half* __restrict__ dst,
    int N, int numHeads, int dst_stride_cols)  // dst_stride_cols = H
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = N * numHeads * kHeadDim;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elems; idx += stride) {
        int elem_within_head = idx % kHeadDim;
        int head = (idx / kHeadDim) % numHeads;
        int row  = idx / (kHeadDim * numHeads);

        int src_ofs = (head * N + row) * kHeadDim + elem_within_head;
        int dst_ofs = row * dst_stride_cols + head * kHeadDim + elem_within_head;

        dst[dst_ofs] = src[src_ofs];
    }
}

// Host helper: gather Q from QKV buffer (first H cols of [N, 3H])
inline void gather_q_from_qkv(
    const half* d_qkv, half* d_buf, int N, int numHeads, int kHeadDim,
    cudaStream_t stream = 0)
{
    int total = N * numHeads * kHeadDim;
    int grid = (total + 255) / 256;
    gather_interleaved_kernel<128, 256><<<grid, 256, 0, stream>>>(
        d_qkv, d_buf, N, numHeads, numHeads * kHeadDim * 3);
}

// Host helper: gather K or V from KV cache ([max_seqlen, H])
inline void gather_kv_from_cache(
    const half* d_cache, half* d_buf, int N, int numHeads, int kHeadDim,
    cudaStream_t stream = 0)
{
    int total = N * numHeads * kHeadDim;
    int grid = (total + 255) / 256;
    gather_interleaved_kernel<128, 256><<<grid, 256, 0, stream>>>(
        d_cache, d_buf, N, numHeads, numHeads * kHeadDim);
}
