// utils.h — Minimal stubs needed by flash_attn_mma_tiling_qkv.cu
#pragma once

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

template<typename T, int R, int C>
__device__ inline void fill_2D_regs(T (&arr)[R][C], T val) {
    #pragma unroll
    for (int i = 0; i < R; i++)
        #pragma unroll
        for (int j = 0; j < C; j++)
            arr[i][j] = val;
}

template<typename T, int D1, int D2, int D3>
__device__ inline void fill_3D_regs(T (&arr)[D1][D2][D3], T val) {
    #pragma unroll
    for (int i = 0; i < D1; i++)
        #pragma unroll
        for (int j = 0; j < D2; j++)
            #pragma unroll
            for (int k = 0; k < D3; k++)
                arr[i][j][k] = val;
}

template<typename T, int N>
__device__ inline void fill_1D_regs(T (&arr)[N], T val) {
    #pragma unroll
    for (int i = 0; i < N; i++)
        arr[i] = val;
}

template<typename T, int N>
__device__ inline T warp_reduce_max(T val) {
    static_assert(N == 4, "only N=4 supported");
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    return val;
}

template<typename T, int N>
__device__ inline T warp_reduce_sum(T val) {
    static_assert(N == 4, "only N=4 supported");
    val += __shfl_xor_sync(0xffffffff, val, 1);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    return val;
}

#define CHECK_TORCH_TENSOR_DTYPE(t, dtype) (void)(t); (void)(dtype)

// div_ceil already defined in flash_attn.cu
