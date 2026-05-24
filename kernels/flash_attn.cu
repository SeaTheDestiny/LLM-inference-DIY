#pragma once
#define FLASH_ATTN_CAUSAL
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
using namespace nvcuda;

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#ifndef LDST32BITS
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#endif
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#ifndef LDST128BITS
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#endif
// gmem -> smem
#ifndef CP_ASYNC_COMMIT_GROUP
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#endif
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#ifndef CP_ASYNC_WAIT_GROUP
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#endif
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
#ifndef CP_ASYNC_CG
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
#endif
// smem -> gmem: requires sm_90 or higher.
#define CP_ASYNC_BULK_COMMIT_GROUP()                                           \
  asm volatile("cp.async.bulk.commit_group;\n" ::)
#define CP_ASYNC_BULK_WAIT_ALL() asm volatile("cp.async.bulk.wait_all;\n" ::)
#define CP_ASYNC_BULK_WAIT_GROUP(n)                                            \
  asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_BULK(dst, src, bytes)                                         \
  asm volatile(                                                                \
      "cp.async.bulk.global.shared::cta.bulk_group.L2::128B [%0], [%1], "      \
      "%2;\n" ::"r"(dst),                                                      \
      "l"(src), "n"(bytes))
// ldmatrix
#define LDMATRIX_X1(R, addr)                                                   \
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"        \
               : "=r"(R)                                                       \
               : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr)                                              \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
               : "=r"(R0), "=r"(R1)                                            \
               : "r"(addr))
#ifndef LDMATRIX_X4
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
#endif
#define LDMATRIX_X1_T(R, addr)                                                 \
  asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"  \
               : "=r"(R)                                                       \
               : "r"(addr))
#ifndef LDMATRIX_X2_T
#define LDMATRIX_X2_T(R0, R1, addr)                                            \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"       \
      : "=r"(R0), "=r"(R1)                                                     \
      : "r"(addr))
#endif
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                                    \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, "      \
      "[%4];\n"                                                                \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
// stmatrix: requires sm_90 or higher.
#define STMATRIX_X1(addr, R)                                                   \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(addr),    \
      "r"(R))
#define STMATRIX_X2(addr, R0, R1)                                              \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(      \
          addr),                                                               \
      "r"(R0), "r"(R1))
#define STMATRIX_X4(addr, R0, R1, R2, R3)                                      \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::  \
          "r"(addr),                                                           \
      "r"(R0), "r"(R1), "r"(R2), "r"(R3))
#define STMATRIX_X1_T(addr, R)                                                 \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" ::"r"(    \
          addr),                                                               \
      "r"(R))
#define STMATRIX_X2_T(addr, R0, R1)                                            \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" ::    \
          "r"(addr),                                                           \
      "r"(R0), "r"(R1))
#define STMATRIX_X4_T(addr, R0, R1, R2, R3)                                    \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, "     \
      "%4};\n" ::"r"(addr),                                                    \
      "r"(R0), "r"(R1), "r"(R2), "r"(R3))
// mma m16n8k16
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)            \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "  \
      "%4, %5}, {%6, %7}, {%8, %9};\n"                                         \
      : "=r"(RD0), "=r"(RD1)                                                   \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0),  \
        "r"(RC1))

// warp 内每 4 个线程求最大�?
#define WARP_REDUCE_MAX_4(val)                                                 \
  do {                                                                         \
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));                     \
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));                     \
  } while (0)

// warp 内每 4 个线程求�?
#define WARP_REDUCE_SUM_4(val)                                                 \
  do {                                                                         \
    val += __shfl_xor_sync(0xffffffff, val, 1);                                \
    val += __shfl_xor_sync(0xffffffff, val, 2);                                \
  } while (0)
  
HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }



template <
    const int kHeadDim,          // Headdim, 32,64,128
    const int kMmaAtomM = 16,         // MMA Atom M, 16
    const int kMmaAtomN = 8,         // MMA Atom N, 8
    const int kMmaAtomK = 16,         // MMA Atom K, 16
    const int kWarpTileSeqLenQ = 1,  // 1, more values, M, Br=64*1=64, matmul M
    const int kWarpTileSeqLenK = 2 //
    >
__global__ void __launch_bounds__(WARP_SIZE)
    flash_attn_mma_naive_kernel(half *Q, half *K, half *V,
                                half *O, int QKV_seqlen,
                                int QKV_head) {
    constexpr int Br = kMmaAtomM * kWarpTileSeqLenQ;
    constexpr int Bc = kMmaAtomN * kWarpTileSeqLenK;
    int Tc = div_ceil(QKV_seqlen, Bc);
    int Tr = blockIdx.y;
    //Q.shape = [QKV_head][QKV_seqlen][kHeadDim]
    //K.shape = [QKV_head][QKV_seqlen][kHeadDim]
    //V.shape = [QKV_head][QKV_seqlen][kHeadDim]
    //O.shape = [QKV_head][QKV_seqlen][kHeadDim]
    //Q[QKV_head][Tr * Br][0]
    //K[QKV_head][0][0]
    //V[QKV_head][0][0]
    //O[QKV_head][Tr * Br][0]
    int Q_addr_offset = QKV_head * QKV_seqlen * kHeadDim +
                        Tr * Br * kHeadDim;
    int K_addr_offset = QKV_head * QKV_seqlen * kHeadDim;
    int V_addr_offset = QKV_head * QKV_seqlen * kHeadDim;
    int O_addr_offset = QKV_head * QKV_seqlen * kHeadDim +
                        Tr * Br * kHeadDim;
    __shared__ half s_q[Br][kHeadDim], s_k[Bc][kHeadDim], 
                    s_p[Br][Bc], s_v[Bc][kHeadDim];
    int lane_id = threadIdx.x;
    // br = 16, kHeadDim?
    // g(Q) -> s(Q)
    for(int g_r_dim = 0; g_r_dim < (kHeadDim * Br) / (WARP_SIZE * 8); g_r_dim ++){
      int row = lane_id / 2;
      int col = (lane_id % 2) * 8 + g_r_dim * 16;
      if (Tr * Br + row < QKV_seqlen) {
        LDST128BITS(s_q[row][col]) 
              = LDST128BITS(Q[Q_addr_offset + row * kHeadDim + col]);
      } else {
        float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
        LDST128BITS(s_q[row][col]) = zero;
      }
    }
    uint32_t R_Q[4];
    uint32_t R_K[2];
    
    // 初始化用于累�?P @ V 的结果的 R_O
    uint32_t R_O[kHeadDim / kMmaAtomN][2];
    #pragma unroll
    for (int i = 0; i < kHeadDim / kMmaAtomN; i++) {
        R_O[i][0] = 0;
        R_O[i][1] = 0;
    }

    float row_max[2] = {-INFINITY, -INFINITY};
    float row_l[2] = {0.0f, 0.0f};
    for(int j = 0; j < Tc; j ++){
      uint32_t R_S[kWarpTileSeqLenK][2] = {{0, 0},{0, 0}};
      // g(K) -> s(K)
      for(int g_r_dim = 0; g_r_dim < (kHeadDim * Bc) / (WARP_SIZE * 8); g_r_dim ++){
        int row = lane_id / 2;
        int col = (lane_id % 2) * 8 + g_r_dim * 16;
        if (j * Bc + row < QKV_seqlen) {
          LDST128BITS(s_k[row][col]) 
                = LDST128BITS(K[K_addr_offset + j * Bc * kHeadDim + row * kHeadDim + col]);
        } else {
          float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
          LDST128BITS(s_k[row][col]) = zero;
        }
      }
      //g(V) -> s(V)
      for(int g_r_dim = 0; g_r_dim < (kHeadDim * Bc) / (WARP_SIZE * 8); g_r_dim ++){
        int row = lane_id / 2;
        int col = (lane_id % 2) * 8 + g_r_dim * 16;
        if (j * Bc + row < QKV_seqlen) {
          LDST128BITS(s_v[row][col]) 
                = LDST128BITS(V[V_addr_offset + j * Bc * kHeadDim + row * kHeadDim + col]);
        } else {
          float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
          LDST128BITS(s_v[row][col]) = zero;
        }
      }
      __syncthreads();
      //calculate Q@K^T
      for(int i_dim = 0; i_dim < kHeadDim / kMmaAtomK; i_dim ++){
        //s -> r
        int q_row = lane_id % kMmaAtomM;
        int q_col = (lane_id / kMmaAtomM) * 8 + i_dim * kMmaAtomK;
        uint32_t q_addr = __cvta_generic_to_shared(&s_q[q_row][q_col]);
        LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], q_addr);
        //
        #pragma unroll
        for(int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++){
          int k_row = lane_id % kMmaAtomN + k_tile * kMmaAtomN;
          int k_col = ((lane_id / kMmaAtomN) % 2 ) * 8 + i_dim * kMmaAtomK;
          //15-31 is free!
          uint32_t k_addr = __cvta_generic_to_shared(&s_k[k_row][k_col]);
          LDMATRIX_X2(R_K[0], R_K[1], k_addr);

          HMMA16816(R_S[k_tile][0], R_S[k_tile][1], 
            R_Q[0], R_Q[1], R_Q[2], R_Q[3], 
            R_K[0], R_K[1], 
            R_S[k_tile][0], R_S[k_tile][1]);
        }
      }

      // --- 解析出多�?tile 的数�?(转成 float2) ---
      float2 f_res[kWarpTileSeqLenK][2];
      float scale = 1.0f / sqrtf((float)kHeadDim);
      
      #pragma unroll
      for(int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
        f_res[k_tile][0] = __half22float2(HALF2(R_S[k_tile][0]));
        f_res[k_tile][1] = __half22float2(HALF2(R_S[k_tile][1]));
        
        f_res[k_tile][0].x *= scale; f_res[k_tile][0].y *= scale;
        f_res[k_tile][1].x *= scale; f_res[k_tile][1].y *= scale;

        // 掩码越界部分
        int k_col_x = j * Bc + ((lane_id % 4) * 2 + 0) + k_tile * 8;
        int k_col_y = j * Bc + ((lane_id % 4) * 2 + 1) + k_tile * 8;
        
        if (k_col_x >= QKV_seqlen) { f_res[k_tile][0].x = -INFINITY; f_res[k_tile][1].x = -INFINITY; }
        if (k_col_y >= QKV_seqlen) { f_res[k_tile][0].y = -INFINITY; f_res[k_tile][1].y = -INFINITY; }
      }

      // --- 1. 计算本地线程的最大�?(横跨所有块) ---
      float m_row_0 = -INFINITY;
      float m_row_1 = -INFINITY;
      #pragma unroll
      for(int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
        m_row_0 = fmaxf(m_row_0, fmaxf(f_res[k_tile][0].x, f_res[k_tile][0].y));
        m_row_1 = fmaxf(m_row_1, fmaxf(f_res[k_tile][1].x, f_res[k_tile][1].y));
      }

      WARP_REDUCE_MAX_4(m_row_0);
      WARP_REDUCE_MAX_4(m_row_1);

      float m_new_0 = fmaxf(row_max[0], m_row_0);
      float m_new_1 = fmaxf(row_max[1], m_row_1);
      float exp_m_old_minus_m_new_0 = expf(row_max[0] - m_new_0);
      float exp_m_old_minus_m_new_1 = expf(row_max[1] - m_new_1); 
      
      // --- 2. 减去新最大值并逐元素求指数 ---
      // --- 3. 计算求和用于分母 ---
      float l_row_sum_0 = 0.0f;
      float l_row_sum_1 = 0.0f;
      #pragma unroll
      for(int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
        f_res[k_tile][0].x = expf(f_res[k_tile][0].x - m_new_0);
        f_res[k_tile][0].y = expf(f_res[k_tile][0].y - m_new_0);
        f_res[k_tile][1].x = expf(f_res[k_tile][1].x - m_new_1);
        f_res[k_tile][1].y = expf(f_res[k_tile][1].y - m_new_1);

        l_row_sum_0 += f_res[k_tile][0].x + f_res[k_tile][0].y;
        l_row_sum_1 += f_res[k_tile][1].x + f_res[k_tile][1].y;
      }

      WARP_REDUCE_SUM_4(l_row_sum_0);
      WARP_REDUCE_SUM_4(l_row_sum_1);

      row_l[0] = row_l[0] * exp_m_old_minus_m_new_0 + l_row_sum_0;
      row_l[1] = row_l[1] * exp_m_old_minus_m_new_1 + l_row_sum_1;

      row_max[0] = m_new_0;
      row_max[1] = m_new_1;
       
      // --- 4. 转换�?f16 格式 (half2) 放回 R_S �?s_p �?---
      int p_row = lane_id / 4;
      int p_col = (lane_id % 4) * 2;
      #pragma unroll
      for(int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
        HALF2(R_S[k_tile][0]) = __float22half2_rn(f_res[k_tile][0]);
        HALF2(R_S[k_tile][1]) = __float22half2_rn(f_res[k_tile][1]);
        
        LDST32BITS(s_p[p_row][p_col + k_tile * 8]) = LDST32BITS(R_S[k_tile][0]);
        LDST32BITS(s_p[p_row + 8][p_col + k_tile * 8]) = LDST32BITS(R_S[k_tile][1]);
      }
      __syncthreads();

      // 先把旧的 O 按新�?row_max 衰减，后�?HMMA 直接继续累加
      #pragma unroll
      for (int i = 0; i < kHeadDim / kMmaAtomN; i++) {
        float2 f_o_0 = __half22float2(HALF2(R_O[i][0]));
        float2 f_o_1 = __half22float2(HALF2(R_O[i][1]));
        f_o_0.x *= exp_m_old_minus_m_new_0;
        f_o_0.y *= exp_m_old_minus_m_new_0;
        f_o_1.x *= exp_m_old_minus_m_new_1;
        f_o_1.y *= exp_m_old_minus_m_new_1;
        HALF2(R_O[i][0]) = __float22half2_rn(f_o_0);
        HALF2(R_O[i][1]) = __float22half2_rn(f_o_1);
      }

      for(int i_dim = 0; i_dim < kHeadDim / kMmaAtomN; i_dim ++){
        #pragma unroll
        for (int p_step = 0; p_step < kWarpTileSeqLenK / 2; p_step++) {
          // P 的形状是 16x(kWarpTileSeqLenK*8)，即 16xBc。每次取 16x16
          int p_row = lane_id % kMmaAtomM;
          int p_col = (lane_id / kMmaAtomM) * 8 + p_step * kMmaAtomK; 
          uint32_t p_addr = __cvta_generic_to_shared(&s_p[p_row][p_col]);
          LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], p_addr);

          // �?s_v 中加载对�?16x8 �?B (V 的维度是 Bc x kHeadDim)
          // V 行偏移为 p_step * kMmaAtomK (kMmaAtomK = 16)
          int v_row = lane_id % kMmaAtomM + p_step * kMmaAtomK; 
          int v_col = ((lane_id / kMmaAtomM) % 2 ) * 8 + i_dim * kMmaAtomN;
          uint32_t v_addr = __cvta_generic_to_shared(&s_v[v_row][v_col]);
          LDMATRIX_X2_T(R_K[0], R_K[1], v_addr);

          // 计算 P @ V 
          HMMA16816(R_O[i_dim][0], R_O[i_dim][1], 
                    R_Q[0], R_Q[1], R_Q[2], R_Q[3], 
                    R_K[0], R_K[1], 
                    R_O[i_dim][0], R_O[i_dim][1]);
        }
      }

      __syncthreads();

    }
    // 最后一次归一化后写回 HBM
    float inv_l_0 = __frcp_rn(row_l[0]);
    //快速求倒数近似值，得到 inv_l_0 �?inv_l_1
    float inv_l_1 = __frcp_rn(row_l[1]);

    #pragma unroll
    for (int i_dim = 0; i_dim < kHeadDim / kMmaAtomN; i_dim++) {
      float2 f_o_0 = __half22float2(HALF2(R_O[i_dim][0]));
      float2 f_o_1 = __half22float2(HALF2(R_O[i_dim][1]));
      f_o_0.x *= inv_l_0;
      f_o_0.y *= inv_l_0;
      f_o_1.x *= inv_l_1;
      f_o_1.y *= inv_l_1;

      int o_row = lane_id / 4;
      int o_col = (lane_id % 4) * 2 + i_dim * kMmaAtomN;
      
      int g_row_o_0 = Tr * Br + o_row + 0;
      int g_row_o_8 = Tr * Br + o_row + 8;
      
      if (g_row_o_0 < QKV_seqlen) {
        LDST32BITS(O[O_addr_offset + (o_row + 0) * kHeadDim + o_col]) = __float22half2_rn(f_o_0);
      }
      if (g_row_o_8 < QKV_seqlen) {
        LDST32BITS(O[O_addr_offset + (o_row + 8) * kHeadDim + o_col]) = __float22half2_rn(f_o_1);
      }
    }

  }



template <
    const int kHeadDim,          // Headdim, 32,64,128
    const int kMmaAtomM = 16,         // MMA Atom M, 16
    const int kMmaAtomN = 8,         // MMA Atom N, 8
    const int kMmaAtomK = 16,         // MMA Atom K, 16
    const int kMmaTileSeqLenQ = 4,   // 4, more MMA(warp), M=16*4=64, Q@K^T=[Br(M),
                                 // d(K)]@[d(K),  Bc(N)]
    const int kMmaTileSeqLenK = 1,   // 1, more MMA(warp), N=8*1 =8,  Q@K^T=[Br(M),
                                 // d(K)]@[d(K),  Bc(N)]这里最好是1，不然每行的数据会分布在不同的warp里面
    const int kWarpTileSeqLenQ = 1,  // 1, more values, M, Br=64*1=64, matmul M
    const int kWarpTileSeqLenK = 8 //  8,
    >
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_mma_41warp_18mma_kernel(half *Q, half *K, half *V,
                                half *O, int QKV_seqlen,
                                int QKV_head) {
    constexpr int NumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
    constexpr int Br = kMmaAtomM * kWarpTileSeqLenQ * kMmaTileSeqLenQ;
    constexpr int Bc = kMmaAtomN * kWarpTileSeqLenK * kMmaTileSeqLenK;
    int Tc = div_ceil(QKV_seqlen, Bc);
    int Tr = blockIdx.y;
    //Q.shape = [QKV_head][QKV_seqlen][kHeadDim]
    //K.shape = [QKV_head][QKV_seqlen][kHeadDim]
    //V.shape = [QKV_head][QKV_seqlen][kHeadDim]
    //O.shape = [QKV_head][QKV_seqlen][kHeadDim]
    //Q[QKV_head][Tr * Br][0]
    //K[QKV_head][0][0]
    //V[QKV_head][0][0]
    //O[QKV_head][Tr * Br][0]
    int Q_addr_offset = QKV_head * QKV_seqlen * kHeadDim +
                        Tr * Br * kHeadDim;
    int K_addr_offset = QKV_head * QKV_seqlen * kHeadDim;
    int V_addr_offset = QKV_head * QKV_seqlen * kHeadDim;
    int O_addr_offset = QKV_head * QKV_seqlen * kHeadDim +
                        Tr * Br * kHeadDim;
    __shared__ half s_q[Br][kHeadDim], s_k[Bc][kHeadDim], 
                    s_p[Br][Bc], s_v[Bc][kHeadDim];
    int thread_id = threadIdx.x;
    int lane_id = thread_id % WARP_SIZE;
    
    constexpr int kHalfPer128Bit = 8;
    constexpr int kVecPerRowQ = kHeadDim / kHalfPer128Bit; 
    constexpr int kItersQ = (Br * kHeadDim) / kHalfPer128Bit / NumThreads; 

    // g(Q) -> s(Q)
    #pragma unroll
    for (int iter = 0; iter < kItersQ; ++iter) {
      int vec_idx = iter * NumThreads + thread_id; 
      int row = vec_idx / kVecPerRowQ; 
      int col = (vec_idx % kVecPerRowQ) * kHalfPer128Bit;
      
      if (Tr * Br + row < QKV_seqlen) {
        LDST128BITS(s_q[row][col]) = LDST128BITS(Q[Q_addr_offset + row * kHeadDim + col]);
      } else {
        float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
        LDST128BITS(s_q[row][col]) = zero;
      }
    }
    uint32_t R_Q[4];
    uint32_t R_K[2];
    
    // 初始化用于累�?P @ V 的结果的 R_O
    uint32_t R_O[kHeadDim / kMmaAtomN][2];
    #pragma unroll
    for (int i = 0; i < kHeadDim / kMmaAtomN; i++) {
        R_O[i][0] = 0;
        R_O[i][1] = 0;
    }

    float row_max[2] = {-INFINITY, -INFINITY};
    float row_l[2] = {0.0f, 0.0f};
    
    constexpr int kVecPerRowK = kHeadDim / kHalfPer128Bit; 
    constexpr int kItersK = (Bc * kHeadDim) / kHalfPer128Bit / NumThreads; 
    int warp_id = thread_id / WARP_SIZE;
    int warp_r = warp_id / kMmaTileSeqLenK;
    int warp_c = warp_id % kMmaTileSeqLenK;
    #pragma unroll 1
    for(int j = 0; j < Tc; j ++){
      uint32_t R_S[kWarpTileSeqLenK][2];
      #pragma unroll
      for (int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
        R_S[k_tile][0] = 0;
        R_S[k_tile][1] = 0;
      }
      // g(K) -> s(K)
      #pragma unroll
      for (int iter = 0; iter < kItersK; ++iter) {
        int vec_idx = iter * NumThreads + thread_id; 
        int row = vec_idx / kVecPerRowK; 
        int col = (vec_idx % kVecPerRowK) * kHalfPer128Bit;
        if (j * Bc + row < QKV_seqlen) {
          LDST128BITS(s_k[row][col]) = LDST128BITS(K[K_addr_offset + j * Bc * kHeadDim + row * kHeadDim + col]);
        } else {
          float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
          LDST128BITS(s_k[row][col]) = zero;
        }
      }
      //g(V) -> s(V)
      #pragma unroll
      for (int iter = 0; iter < kItersK; ++iter) {
        int vec_idx = iter * NumThreads + thread_id; 
        int row = vec_idx / kVecPerRowK; 
        int col = (vec_idx % kVecPerRowK) * kHalfPer128Bit;
        if (j * Bc + row < QKV_seqlen) {
          LDST128BITS(s_v[row][col]) = LDST128BITS(V[V_addr_offset + j * Bc * kHeadDim + row * kHeadDim + col]);
        } else {
          float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
          LDST128BITS(s_v[row][col]) = zero;
        }
      }
      __syncthreads();
      //calculate Q@K^T
      #pragma unroll
      for(int i_dim = 0; i_dim < kHeadDim / kMmaAtomK; i_dim ++){
        //s -> r
        int q_row = lane_id % kMmaAtomM + warp_r * kMmaAtomM;
        int q_col = (lane_id / kMmaAtomM) * 8 + i_dim * kMmaAtomK;
        uint32_t q_addr = __cvta_generic_to_shared(&s_q[q_row][q_col]);
        LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], q_addr);
        //
        #pragma unroll
        for(int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++){
          int k_row = lane_id % kMmaAtomN + k_tile * kMmaAtomN + warp_c * kWarpTileSeqLenK * kMmaAtomN;
          int k_col = ((lane_id / kMmaAtomN) % 2 ) * 8 + i_dim * kMmaAtomK;
          //15-31 is free!
          uint32_t k_addr = __cvta_generic_to_shared(&s_k[k_row][k_col]);
          LDMATRIX_X2(R_K[0], R_K[1], k_addr);

          HMMA16816(R_S[k_tile][0], R_S[k_tile][1], 
            R_Q[0], R_Q[1], R_Q[2], R_Q[3], 
            R_K[0], R_K[1], 
            R_S[k_tile][0], R_S[k_tile][1]);
        }
      }

      // --- 解析出多�?tile 的数�?(转成 float2) ---
      float2 f_res[kWarpTileSeqLenK][2];
      float scale = 1.0f / sqrtf((float)kHeadDim);
      
      #pragma unroll
      for(int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
        f_res[k_tile][0] = __half22float2(HALF2(R_S[k_tile][0]));
        f_res[k_tile][1] = __half22float2(HALF2(R_S[k_tile][1]));
        
        f_res[k_tile][0].x *= scale; f_res[k_tile][0].y *= scale;
        f_res[k_tile][1].x *= scale; f_res[k_tile][1].y *= scale;

        // 掩码越界部分
        int k_col_x = j * Bc + ((lane_id % 4) * 2 + 0) + k_tile * 8;
        int k_col_y = j * Bc + ((lane_id % 4) * 2 + 1) + k_tile * 8;
        
        if (k_col_x >= QKV_seqlen) { f_res[k_tile][0].x = -INFINITY; f_res[k_tile][1].x = -INFINITY; }
        if (k_col_y >= QKV_seqlen) { f_res[k_tile][0].y = -INFINITY; f_res[k_tile][1].y = -INFINITY; }
      }

      // --- 1. 计算本地线程的最大�?(横跨所有块) ---
      float m_row_0 = -INFINITY;
      float m_row_1 = -INFINITY;
      #pragma unroll
      for(int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
        m_row_0 = fmaxf(m_row_0, fmaxf(f_res[k_tile][0].x, f_res[k_tile][0].y));
        m_row_1 = fmaxf(m_row_1, fmaxf(f_res[k_tile][1].x, f_res[k_tile][1].y));
      }

      WARP_REDUCE_MAX_4(m_row_0);
      WARP_REDUCE_MAX_4(m_row_1);

      float m_new_0 = fmaxf(row_max[0], m_row_0);
      float m_new_1 = fmaxf(row_max[1], m_row_1);
      float exp_m_old_minus_m_new_0 = expf(row_max[0] - m_new_0);
      float exp_m_old_minus_m_new_1 = expf(row_max[1] - m_new_1); 
      
      // --- 2. 减去新最大值并逐元素求指数 ---
      // --- 3. 计算求和用于分母 ---
      float l_row_sum_0 = 0.0f;
      float l_row_sum_1 = 0.0f;
      #pragma unroll
      for(int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
        f_res[k_tile][0].x = expf(f_res[k_tile][0].x - m_new_0);
        f_res[k_tile][0].y = expf(f_res[k_tile][0].y - m_new_0);
        f_res[k_tile][1].x = expf(f_res[k_tile][1].x - m_new_1);
        f_res[k_tile][1].y = expf(f_res[k_tile][1].y - m_new_1);

        l_row_sum_0 += f_res[k_tile][0].x + f_res[k_tile][0].y;
        l_row_sum_1 += f_res[k_tile][1].x + f_res[k_tile][1].y;
      }

      WARP_REDUCE_SUM_4(l_row_sum_0);
      WARP_REDUCE_SUM_4(l_row_sum_1);

      row_l[0] = row_l[0] * exp_m_old_minus_m_new_0 + l_row_sum_0;
      row_l[1] = row_l[1] * exp_m_old_minus_m_new_1 + l_row_sum_1;

      row_max[0] = m_new_0;
      row_max[1] = m_new_1;
       
      // --- 4. 转换�?f16 格式 (half2) 放回 R_S �?s_p �?---
      int p_row = lane_id / 4 + warp_r * kWarpTileSeqLenQ * kMmaAtomM;
      int p_col = (lane_id % 4) * 2;
      #pragma unroll
      for(int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
        HALF2(R_S[k_tile][0]) = __float22half2_rn(f_res[k_tile][0]);
        HALF2(R_S[k_tile][1]) = __float22half2_rn(f_res[k_tile][1]);
        
        LDST32BITS(s_p[p_row][p_col + k_tile * 8]) = LDST32BITS(R_S[k_tile][0]);
        LDST32BITS(s_p[p_row + 8][p_col + k_tile * 8]) = LDST32BITS(R_S[k_tile][1]);
      }
      __syncthreads();

      // 先把旧的 O 按新�?row_max 衰减，后�?HMMA 直接继续累加
      #pragma unroll
      for (int i = 0; i < kHeadDim / kMmaAtomN; i++) {
        float2 f_o_0 = __half22float2(HALF2(R_O[i][0]));
        float2 f_o_1 = __half22float2(HALF2(R_O[i][1]));
        f_o_0.x *= exp_m_old_minus_m_new_0;
        f_o_0.y *= exp_m_old_minus_m_new_0;
        f_o_1.x *= exp_m_old_minus_m_new_1;
        f_o_1.y *= exp_m_old_minus_m_new_1;
        HALF2(R_O[i][0]) = __float22half2_rn(f_o_0);
        HALF2(R_O[i][1]) = __float22half2_rn(f_o_1);
      }

      #pragma unroll
      for(int i_dim = 0; i_dim < kHeadDim / kMmaAtomN; i_dim ++){
        #pragma unroll
        for (int p_step = 0; p_step < kWarpTileSeqLenK / 2; p_step++) {
          // P 的形状是 16x(kWarpTileSeqLenK*8)，即 16xBc。每次取 16x16
          int p_row = lane_id % kMmaAtomM + warp_r * kWarpTileSeqLenQ * kMmaAtomM;
          int p_col = (lane_id / kMmaAtomM) * 8 + p_step * kMmaAtomK; 
          uint32_t p_addr = __cvta_generic_to_shared(&s_p[p_row][p_col]);
          LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], p_addr);

          // �?s_v 中加载对�?16x8 �?B (V 的维度是 Bc x kHeadDim)
          // V 行偏移为 p_step * kMmaAtomK (kMmaAtomK = 16)
          int v_row = lane_id % kMmaAtomM + p_step * kMmaAtomK; 
          int v_col = ((lane_id / kMmaAtomM) % 2 ) * 8 + i_dim * kMmaAtomN;
          uint32_t v_addr = __cvta_generic_to_shared(&s_v[v_row][v_col]);
          LDMATRIX_X2_T(R_K[0], R_K[1], v_addr);

          // 计算 P @ V 
          HMMA16816(R_O[i_dim][0], R_O[i_dim][1], 
                    R_Q[0], R_Q[1], R_Q[2], R_Q[3], 
                    R_K[0], R_K[1], 
                    R_O[i_dim][0], R_O[i_dim][1]);
        }
      }

      __syncthreads();

    }
    // 最后一次归一化后写回 HBM
    float inv_l_0 = __frcp_rn(row_l[0]);
    //快速求倒数近似值，得到 inv_l_0 �?inv_l_1
    float inv_l_1 = __frcp_rn(row_l[1]);

    #pragma unroll
    for (int i_dim = 0; i_dim < kHeadDim / kMmaAtomN; i_dim++) {
      float2 f_o_0 = __half22float2(HALF2(R_O[i_dim][0]));
      float2 f_o_1 = __half22float2(HALF2(R_O[i_dim][1]));
      f_o_0.x *= inv_l_0;
      f_o_0.y *= inv_l_0;
      f_o_1.x *= inv_l_1;
      f_o_1.y *= inv_l_1;
      
      int o_row = lane_id / 4 + warp_r * kWarpTileSeqLenQ * kMmaAtomM;
      int o_col = (lane_id % 4) * 2 + i_dim * kMmaAtomN;
      
      int g_row_o_0 = Tr * Br + o_row + 0;
      int g_row_o_8 = Tr * Br + o_row + 8;
      
      if (g_row_o_0 < QKV_seqlen) {
        LDST32BITS(O[O_addr_offset + (o_row + 0) * kHeadDim + o_col]) = __float22half2_rn(f_o_0);
      }
      if (g_row_o_8 < QKV_seqlen) {
        LDST32BITS(O[O_addr_offset + (o_row + 8) * kHeadDim + o_col]) = __float22half2_rn(f_o_1);
      }
    }

  }


// ============================================================================
// Step 1: Fine-grained Q/K Tiling
// ============================================================================
// 核心改动 vs 原始 multi-warp kernel:
//   s_q[Br][kHeadDim]  ->  s_q[Br][kMmaAtomK + kPadQ]    (Br*24 代替 Br*128)
//   s_k[Bc][kHeadDim]  ->  s_k[Bc][kMmaAtomK + kPadQ]    (Bc*24 代替 Bc*128)
//   Q@K^T 新增外循环 tile_K_d in [0, kHeadDim/kMmaAtomK), 每次加载16列到SMEM
//   s_p[Br][Bc] / s_v[Bc][kHeadDim] 保持不变 (Step 2/4 再优化)
//   extern __shared__ 动态分配 SMEM + 8元素padding 避免 bank conflict
//   SMEM 用量: (Br*24 + Bc*24 + Br*Bc + Bc*d) * 2 bytes
//   d=128: (1536+1536+4096+8192)*2 = 30720 bytes

template <
    const int kHeadDim,          // Headdim: 32, 64, 128
    const int kMmaAtomM = 16,    // MMA Atom M
    const int kMmaAtomN = 8,     // MMA Atom N
    const int kMmaAtomK = 16,    // MMA Atom K
    const int kMmaTileSeqLenQ = 4,    // 4 warps for Q rows
    const int kMmaTileSeqLenK = 1,    // 1 warp for K cols
    const int kWarpTileSeqLenQ = 1,   // 1, Br = 16*4*1 = 64
    const int kWarpTileSeqLenK = 8,   // 8, Bc = 8*1*8 = 64
    const int kPadQ = 8,              // padding for Q tile
    const int kPadK = 8               // padding for K tile
    >
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_finegrained_qk_tiling_kernel(half *Q, half *K, half *V,
                                            half *O, int QKV_seqlen,
                                            int QKV_head) {
    constexpr int kNumThreads_fg = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;  // 64
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;  // 64
    const int Tc = div_ceil(QKV_seqlen, Bc);
    const int Tr = blockIdx.y;

    int Q_addr_offset = QKV_head * QKV_seqlen * kHeadDim + Tr * Br * kHeadDim;
    int K_addr_offset = QKV_head * QKV_seqlen * kHeadDim;
    int V_addr_offset = QKV_head * QKV_seqlen * kHeadDim;
    int O_addr_offset = QKV_head * QKV_seqlen * kHeadDim + Tr * Br * kHeadDim;

    // --- extern shared memory layout ---
    // [Q_tile] [K_tile] [s_p] [s_v]
    constexpr int Q_tile_size = Br * (kMmaAtomK + kPadQ);   // 64*24 = 1536
    constexpr int K_tile_size = Bc * (kMmaAtomK + kPadK);   // 64*24 = 1536
    extern __shared__ half smem[];
    half *Q_tile_smem = smem;
    half *K_tile_smem = smem + Q_tile_size;
    half *s_p          = smem + Q_tile_size + K_tile_size;
    half *s_v          = smem + Q_tile_size + K_tile_size + Br * Bc;

    uint32_t smem_Q_base = __cvta_generic_to_shared(Q_tile_smem);
    uint32_t smem_K_base = __cvta_generic_to_shared(K_tile_smem);

    int thread_id = threadIdx.x;
    int lane_id   = thread_id % WARP_SIZE;
    int warp_id   = thread_id / WARP_SIZE;
    int warp_r    = warp_id / kMmaTileSeqLenK;  // Q row warps: 0,1,2,3
    int warp_c    = warp_id % kMmaTileSeqLenK;  // K col warp:  0

    // --- Register allocation ---
    uint32_t R_Q[4];
    uint32_t R_K[2];

    uint32_t R_O[kHeadDim / kMmaAtomN][2];
    #pragma unroll
    for (int i = 0; i < kHeadDim / kMmaAtomN; i++) {
        R_O[i][0] = 0;
        R_O[i][1] = 0;
    }

    float row_max[2] = {-INFINITY, -INFINITY};
    float row_l[2]   = {0.0f, 0.0f};

    float scale = 1.0f / sqrtf((float)kHeadDim);
    constexpr int kHalfPer128Bit = 8;

    // --- Outer loop over K seqlen tiles ---
    #pragma unroll 1
    for (int j = 0; j < Tc; j++) {
        uint32_t R_S[kWarpTileSeqLenK][2];
        #pragma unroll
        for (int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
            R_S[k_tile][0] = 0;
            R_S[k_tile][1] = 0;
        }

        // ============ Fine-grained loop over d (kMmaAtomK=16 at a time) ============
        #pragma unroll
        for (int tile_K_d = 0; tile_K_d < kHeadDim / kMmaAtomK; ++tile_K_d) {
            int global_col_offset = tile_K_d * kMmaAtomK;

            // g(Q) -> s(Q): Br*16 halfs, 128 threads -> 1 float4/thread
            {
                int row = thread_id / 2;                       // 0..63
                int col = (thread_id % 2) * kHalfPer128Bit;   // 0 or 8
                if (Tr * Br + row < QKV_seqlen) {
                    LDST128BITS(Q_tile_smem[row * (kMmaAtomK + kPadQ) + col])
                        = LDST128BITS(Q[Q_addr_offset + row * kHeadDim + global_col_offset + col]);
                } else {
                    float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
                    LDST128BITS(Q_tile_smem[row * (kMmaAtomK + kPadQ) + col]) = zero;
                }
            }

            // g(K) -> s(K): Bc*16 halfs, 128 threads -> 1 float4/thread
            {
                int row = thread_id / 2;
                int col = (thread_id % 2) * kHalfPer128Bit;
                if (j * Bc + row < QKV_seqlen) {
                    LDST128BITS(K_tile_smem[row * (kMmaAtomK + kPadK) + col])
                        = LDST128BITS(K[K_addr_offset + j * Bc * kHeadDim + row * kHeadDim + global_col_offset + col]);
                } else {
                    float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
                    LDST128BITS(K_tile_smem[row * (kMmaAtomK + kPadK) + col]) = zero;
                }
            }
            __syncthreads();

            // ---- MMA: Q_tile[Br,16] @ K_tile^T[16,Bc] -> S[Br,Bc] (accumulate) ----
            int q_row = lane_id % kMmaAtomM + warp_r * kMmaAtomM;
            int q_col = (lane_id / kMmaAtomM) * 8;  // 0 or 8
            uint32_t q_addr = smem_Q_base + (q_row * (kMmaAtomK + kPadQ) + q_col) * sizeof(half);
            LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], q_addr);

            #pragma unroll
            for (int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
                int k_row = lane_id % kMmaAtomN + k_tile * kMmaAtomN
                            + warp_c * kWarpTileSeqLenK * kMmaAtomN;
                int k_col = ((lane_id / kMmaAtomN) % 2) * 8;  // 0 or 8
                uint32_t k_addr = smem_K_base + (k_row * (kMmaAtomK + kPadK) + k_col) * sizeof(half);
                LDMATRIX_X2(R_K[0], R_K[1], k_addr);

                HMMA16816(R_S[k_tile][0], R_S[k_tile][1],
                          R_Q[0], R_Q[1], R_Q[2], R_Q[3],
                          R_K[0], R_K[1],
                          R_S[k_tile][0], R_S[k_tile][1]);
            }
            __syncthreads();
        }  // end loop over tile_K_d

        // ---- Online Safe Softmax (same as original multi-warp) ----
        float2 f_res[kWarpTileSeqLenK][2];
        #pragma unroll
        for (int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
            f_res[k_tile][0] = __half22float2(HALF2(R_S[k_tile][0]));
            f_res[k_tile][1] = __half22float2(HALF2(R_S[k_tile][1]));

            f_res[k_tile][0].x *= scale; f_res[k_tile][0].y *= scale;
            f_res[k_tile][1].x *= scale; f_res[k_tile][1].y *= scale;

            int k_col_x = j * Bc + ((lane_id % 4) * 2 + 0) + k_tile * 8;
            int k_col_y = j * Bc + ((lane_id % 4) * 2 + 1) + k_tile * 8;

            if (k_col_x >= QKV_seqlen) { f_res[k_tile][0].x = -INFINITY; f_res[k_tile][1].x = -INFINITY; }
            if (k_col_y >= QKV_seqlen) { f_res[k_tile][0].y = -INFINITY; f_res[k_tile][1].y = -INFINITY; }
        }

        float m_row_0 = -INFINITY;
        float m_row_1 = -INFINITY;
        #pragma unroll
        for (int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
            m_row_0 = fmaxf(m_row_0, fmaxf(f_res[k_tile][0].x, f_res[k_tile][0].y));
            m_row_1 = fmaxf(m_row_1, fmaxf(f_res[k_tile][1].x, f_res[k_tile][1].y));
        }

        WARP_REDUCE_MAX_4(m_row_0);
        WARP_REDUCE_MAX_4(m_row_1);

        float m_new_0 = fmaxf(row_max[0], m_row_0);
        float m_new_1 = fmaxf(row_max[1], m_row_1);
        float exp_m_old_minus_m_new_0 = expf(row_max[0] - m_new_0);
        float exp_m_old_minus_m_new_1 = expf(row_max[1] - m_new_1);

        float l_row_sum_0 = 0.0f;
        float l_row_sum_1 = 0.0f;
        #pragma unroll
        for (int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
            f_res[k_tile][0].x = expf(f_res[k_tile][0].x - m_new_0);
            f_res[k_tile][0].y = expf(f_res[k_tile][0].y - m_new_0);
            f_res[k_tile][1].x = expf(f_res[k_tile][1].x - m_new_1);
            f_res[k_tile][1].y = expf(f_res[k_tile][1].y - m_new_1);

            l_row_sum_0 += f_res[k_tile][0].x + f_res[k_tile][0].y;
            l_row_sum_1 += f_res[k_tile][1].x + f_res[k_tile][1].y;
        }

        WARP_REDUCE_SUM_4(l_row_sum_0);
        WARP_REDUCE_SUM_4(l_row_sum_1);

        row_l[0] = row_l[0] * exp_m_old_minus_m_new_0 + l_row_sum_0;
        row_l[1] = row_l[1] * exp_m_old_minus_m_new_1 + l_row_sum_1;

        row_max[0] = m_new_0;
        row_max[1] = m_new_1;

        // ---- Write P back to s_p[Br][Bc] (same as original) ----
        int p_row = lane_id / 4 + warp_r * kWarpTileSeqLenQ * kMmaAtomM;
        int p_col = (lane_id % 4) * 2;
        #pragma unroll
        for (int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
            HALF2(R_S[k_tile][0]) = __float22half2_rn(f_res[k_tile][0]);
            HALF2(R_S[k_tile][1]) = __float22half2_rn(f_res[k_tile][1]);

            LDST32BITS(s_p[p_row * Bc + p_col + k_tile * 8]) = LDST32BITS(R_S[k_tile][0]);
            LDST32BITS(s_p[(p_row + 8) * Bc + p_col + k_tile * 8]) = LDST32BITS(R_S[k_tile][1]);
        }

        // ---- g(V) -> s(V) [Bc, kHeadDim] (same as original) ----
        constexpr int kVecPerRowV = kHeadDim / kHalfPer128Bit;
        constexpr int kItersV = (Bc * kHeadDim) / kHalfPer128Bit / kNumThreads_fg;
        #pragma unroll
        for (int iter = 0; iter < kItersV; ++iter) {
            int vec_idx = iter * kNumThreads_fg + thread_id;
            int row = vec_idx / kVecPerRowV;
            int col = (vec_idx % kVecPerRowV) * kHalfPer128Bit;
            if (j * Bc + row < QKV_seqlen) {
                LDST128BITS(s_v[row * kHeadDim + col])
                    = LDST128BITS(V[V_addr_offset + j * Bc * kHeadDim + row * kHeadDim + col]);
            } else {
                float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
                LDST128BITS(s_v[row * kHeadDim + col]) = zero;
            }
        }
        __syncthreads();

        // ---- Rescale O (same as original) ----
        #pragma unroll
        for (int i = 0; i < kHeadDim / kMmaAtomN; i++) {
            float2 f_o_0 = __half22float2(HALF2(R_O[i][0]));
            float2 f_o_1 = __half22float2(HALF2(R_O[i][1]));
            f_o_0.x *= exp_m_old_minus_m_new_0;
            f_o_0.y *= exp_m_old_minus_m_new_0;
            f_o_1.x *= exp_m_old_minus_m_new_1;
            f_o_1.y *= exp_m_old_minus_m_new_1;
            HALF2(R_O[i][0]) = __float22half2_rn(f_o_0);
            HALF2(R_O[i][1]) = __float22half2_rn(f_o_1);
        }

        // ---- P@V MMA (same as original) ----
        #pragma unroll
        for (int i_dim = 0; i_dim < kHeadDim / kMmaAtomN; i_dim++) {
            #pragma unroll
            for (int p_step = 0; p_step < kWarpTileSeqLenK / 2; p_step++) {
                int p_row_mma = lane_id % kMmaAtomM + warp_r * kWarpTileSeqLenQ * kMmaAtomM;
                int p_col_mma = (lane_id / kMmaAtomM) * 8 + p_step * kMmaAtomK;
                uint32_t p_addr = __cvta_generic_to_shared(&s_p[p_row_mma * Bc + p_col_mma]);
                LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], p_addr);

                int v_row = lane_id % kMmaAtomM + p_step * kMmaAtomK;
                int v_col = ((lane_id / kMmaAtomM) % 2) * 8 + i_dim * kMmaAtomN;
                uint32_t v_addr = __cvta_generic_to_shared(&s_v[v_row * kHeadDim + v_col]);
                LDMATRIX_X2_T(R_K[0], R_K[1], v_addr);

                HMMA16816(R_O[i_dim][0], R_O[i_dim][1],
                          R_Q[0], R_Q[1], R_Q[2], R_Q[3],
                          R_K[0], R_K[1],
                          R_O[i_dim][0], R_O[i_dim][1]);
            }
        }
        __syncthreads();

    }  // end loop over K seqlen tiles

    // ---- Final rescale & writeback O (same as original) ----
    float inv_l_0 = __frcp_rn(row_l[0]);
    float inv_l_1 = __frcp_rn(row_l[1]);

    #pragma unroll
    for (int i_dim = 0; i_dim < kHeadDim / kMmaAtomN; i_dim++) {
        float2 f_o_0 = __half22float2(HALF2(R_O[i_dim][0]));
        float2 f_o_1 = __half22float2(HALF2(R_O[i_dim][1]));
        f_o_0.x *= inv_l_0;
        f_o_0.y *= inv_l_0;
        f_o_1.x *= inv_l_1;
        f_o_1.y *= inv_l_1;

        int o_row = lane_id / 4 + warp_r * kWarpTileSeqLenQ * kMmaAtomM;
        int o_col = (lane_id % 4) * 2 + i_dim * kMmaAtomN;

        int g_row_o_0 = Tr * Br + o_row + 0;
        int g_row_o_8 = Tr * Br + o_row + 8;

        if (g_row_o_0 < QKV_seqlen) {
            LDST32BITS(O[O_addr_offset + (o_row + 0) * kHeadDim + o_col]) = __float22half2_rn(f_o_0);
        }
        if (g_row_o_8 < QKV_seqlen) {
            LDST32BITS(O[O_addr_offset + (o_row + 8) * kHeadDim + o_col]) = __float22half2_rn(f_o_1);
        }
    }
}


// ============================================================================
// Step 2: P Matrix in Registers (Eliminate s_p[Br][Bc])
// ============================================================================
// 核心改动 vs Step 1:
//   s_p[Br][Bc] 完全消除, P=softmax(Q@K^T) 全程留在寄存器 R_S 中
//   P@V MMA 直接使用 R_S 寄存器作为 A 矩阵, 不再 ldmatrix P from SMEM
//   节省 SMEM: Br*Bc = 4096 half = 8KB, 消除一次 __syncthreads()
//   SMEM 用量: (Br*24 + Bc*24 + Bc*d) * 2 bytes
//   d=128: (1536+1536+8192)*2 = 22528 bytes

template <
    const int kHeadDim,
    const int kMmaAtomM = 16,
    const int kMmaAtomN = 8,
    const int kMmaAtomK = 16,
    const int kMmaTileSeqLenQ = 4,
    const int kMmaTileSeqLenK = 1,
    const int kWarpTileSeqLenQ = 1,
    const int kWarpTileSeqLenK = 8,
    const int kPadQ = 8,
    const int kPadK = 8
    >
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_register_p_kernel(half *Q, half *K, half *V,
                                 half *O, int QKV_seqlen,
                                 int QKV_head) {
    constexpr int kNumThreads_rp = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;  // 64
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;  // 64
    const int Tc = div_ceil(QKV_seqlen, Bc);
    const int Tr = blockIdx.y;

    int Q_addr_offset = QKV_head * QKV_seqlen * kHeadDim + Tr * Br * kHeadDim;
    int K_addr_offset = QKV_head * QKV_seqlen * kHeadDim;
    int V_addr_offset = QKV_head * QKV_seqlen * kHeadDim;
    int O_addr_offset = QKV_head * QKV_seqlen * kHeadDim + Tr * Br * kHeadDim;

    // --- extern SMEM: [Q_tile] [K_tile] [s_v] (NO s_p!) ---
    constexpr int Q_tile_size = Br * (kMmaAtomK + kPadQ);   // 64*24 = 1536
    constexpr int K_tile_size = Bc * (kMmaAtomK + kPadK);   // 64*24 = 1536
    extern __shared__ half smem[];
    half *Q_tile_smem = smem;
    half *K_tile_smem = smem + Q_tile_size;
    half *s_v          = smem + Q_tile_size + K_tile_size;  // V starts after Q,K

    uint32_t smem_Q_base = __cvta_generic_to_shared(Q_tile_smem);
    uint32_t smem_K_base = __cvta_generic_to_shared(K_tile_smem);

    int thread_id = threadIdx.x;
    int lane_id   = thread_id % WARP_SIZE;
    int warp_id   = thread_id / WARP_SIZE;
    int warp_r    = warp_id / kMmaTileSeqLenK;  // Q row warps: 0,1,2,3
    int warp_c    = warp_id % kMmaTileSeqLenK;  // K col warp:  0

    // --- Registers ---
    uint32_t R_Q[4];
    uint32_t R_K[2];

    uint32_t R_O[kHeadDim / kMmaAtomN][2];
    #pragma unroll
    for (int i = 0; i < kHeadDim / kMmaAtomN; i++) {
        R_O[i][0] = 0;
        R_O[i][1] = 0;
    }

    float row_max[2] = {-INFINITY, -INFINITY};
    float row_l[2]   = {0.0f, 0.0f};

    float scale = 1.0f / sqrtf((float)kHeadDim);
    constexpr int kHalfPer128Bit = 8;

    #pragma unroll 1
    for (int j = 0; j < Tc; j++) {
        // R_S holds P = softmax(Q@K^T) in registers throughout the iteration
        uint32_t R_S[kWarpTileSeqLenK][2];
        #pragma unroll
        for (int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
            R_S[k_tile][0] = 0;
            R_S[k_tile][1] = 0;
        }

        // ===== Fine-grained Q@K^T over d =====
        #pragma unroll
        for (int tile_K_d = 0; tile_K_d < kHeadDim / kMmaAtomK; ++tile_K_d) {
            int global_col_offset = tile_K_d * kMmaAtomK;

            // g(Q) -> s(Q): Br*16 halfs, 128 threads -> 1 float4/thread
            {
                int row = thread_id / 2;
                int col = (thread_id % 2) * kHalfPer128Bit;
                if (Tr * Br + row < QKV_seqlen) {
                    LDST128BITS(Q_tile_smem[row * (kMmaAtomK + kPadQ) + col])
                        = LDST128BITS(Q[Q_addr_offset + row * kHeadDim + global_col_offset + col]);
                } else {
                    float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
                    LDST128BITS(Q_tile_smem[row * (kMmaAtomK + kPadQ) + col]) = zero;
                }
            }

            // g(K) -> s(K): Bc*16 halfs, 128 threads -> 1 float4/thread
            {
                int row = thread_id / 2;
                int col = (thread_id % 2) * kHalfPer128Bit;
                if (j * Bc + row < QKV_seqlen) {
                    LDST128BITS(K_tile_smem[row * (kMmaAtomK + kPadK) + col])
                        = LDST128BITS(K[K_addr_offset + j * Bc * kHeadDim + row * kHeadDim + global_col_offset + col]);
                } else {
                    float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
                    LDST128BITS(K_tile_smem[row * (kMmaAtomK + kPadK) + col]) = zero;
                }
            }
            __syncthreads();

            // MMA: Q[Br,16] @ K^T[16,Bc] -> S[Br,Bc]
            int q_row = lane_id % kMmaAtomM + warp_r * kMmaAtomM;
            int q_col = (lane_id / kMmaAtomM) * 8;
            uint32_t q_addr = smem_Q_base + (q_row * (kMmaAtomK + kPadQ) + q_col) * sizeof(half);
            LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], q_addr);

            #pragma unroll
            for (int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
                int k_row = lane_id % kMmaAtomN + k_tile * kMmaAtomN
                            + warp_c * kWarpTileSeqLenK * kMmaAtomN;
                int k_col = ((lane_id / kMmaAtomN) % 2) * 8;
                uint32_t k_addr = smem_K_base + (k_row * (kMmaAtomK + kPadK) + k_col) * sizeof(half);
                LDMATRIX_X2(R_K[0], R_K[1], k_addr);

                HMMA16816(R_S[k_tile][0], R_S[k_tile][1],
                          R_Q[0], R_Q[1], R_Q[2], R_Q[3],
                          R_K[0], R_K[1],
                          R_S[k_tile][0], R_S[k_tile][1]);
            }
            __syncthreads();
        }  // end d loop

        // ===== Online Safe Softmax (P stays in R_S registers!) =====
        float2 f_res[kWarpTileSeqLenK][2];
        #pragma unroll
        for (int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
            f_res[k_tile][0] = __half22float2(HALF2(R_S[k_tile][0]));
            f_res[k_tile][1] = __half22float2(HALF2(R_S[k_tile][1]));

            f_res[k_tile][0].x *= scale; f_res[k_tile][0].y *= scale;
            f_res[k_tile][1].x *= scale; f_res[k_tile][1].y *= scale;

            int k_col_x = j * Bc + ((lane_id % 4) * 2 + 0) + k_tile * 8;
            int k_col_y = j * Bc + ((lane_id % 4) * 2 + 1) + k_tile * 8;

            if (k_col_x >= QKV_seqlen) { f_res[k_tile][0].x = -INFINITY; f_res[k_tile][1].x = -INFINITY; }
            if (k_col_y >= QKV_seqlen) { f_res[k_tile][0].y = -INFINITY; f_res[k_tile][1].y = -INFINITY; }
        }

        float m_row_0 = -INFINITY;
        float m_row_1 = -INFINITY;
        #pragma unroll
        for (int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
            m_row_0 = fmaxf(m_row_0, fmaxf(f_res[k_tile][0].x, f_res[k_tile][0].y));
            m_row_1 = fmaxf(m_row_1, fmaxf(f_res[k_tile][1].x, f_res[k_tile][1].y));
        }

        WARP_REDUCE_MAX_4(m_row_0);
        WARP_REDUCE_MAX_4(m_row_1);

        float m_new_0 = fmaxf(row_max[0], m_row_0);
        float m_new_1 = fmaxf(row_max[1], m_row_1);
        float exp_m_old_minus_m_new_0 = expf(row_max[0] - m_new_0);
        float exp_m_old_minus_m_new_1 = expf(row_max[1] - m_new_1);

        float l_row_sum_0 = 0.0f;
        float l_row_sum_1 = 0.0f;
        #pragma unroll
        for (int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
            f_res[k_tile][0].x = expf(f_res[k_tile][0].x - m_new_0);
            f_res[k_tile][0].y = expf(f_res[k_tile][0].y - m_new_0);
            f_res[k_tile][1].x = expf(f_res[k_tile][1].x - m_new_1);
            f_res[k_tile][1].y = expf(f_res[k_tile][1].y - m_new_1);

            l_row_sum_0 += f_res[k_tile][0].x + f_res[k_tile][0].y;
            l_row_sum_1 += f_res[k_tile][1].x + f_res[k_tile][1].y;
        }

        WARP_REDUCE_SUM_4(l_row_sum_0);
        WARP_REDUCE_SUM_4(l_row_sum_1);

        row_l[0] = row_l[0] * exp_m_old_minus_m_new_0 + l_row_sum_0;
        row_l[1] = row_l[1] * exp_m_old_minus_m_new_1 + l_row_sum_1;

        row_max[0] = m_new_0;
        row_max[1] = m_new_1;

        // === KEY CHANGE: Write P directly to R_S registers, NO s_p SMEM! ===
        #pragma unroll
        for (int k_tile = 0; k_tile < kWarpTileSeqLenK; k_tile++) {
            HALF2(R_S[k_tile][0]) = __float22half2_rn(f_res[k_tile][0]);
            HALF2(R_S[k_tile][1]) = __float22half2_rn(f_res[k_tile][1]);
        }
        // NOTE: no __syncthreads() needed since P never hits SMEM

        // ===== g(V) -> s(V) [Bc, kHeadDim] =====
        constexpr int kVecPerRowV = kHeadDim / kHalfPer128Bit;
        constexpr int kItersV = (Bc * kHeadDim) / kHalfPer128Bit / kNumThreads_rp;
        #pragma unroll
        for (int iter = 0; iter < kItersV; ++iter) {
            int vec_idx = iter * kNumThreads_rp + thread_id;
            int row = vec_idx / kVecPerRowV;
            int col = (vec_idx % kVecPerRowV) * kHalfPer128Bit;
            if (j * Bc + row < QKV_seqlen) {
                LDST128BITS(s_v[row * kHeadDim + col])
                    = LDST128BITS(V[V_addr_offset + j * Bc * kHeadDim + row * kHeadDim + col]);
            } else {
                float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
                LDST128BITS(s_v[row * kHeadDim + col]) = zero;
            }
        }
        __syncthreads();

        // ===== Rescale O =====
        #pragma unroll
        for (int i = 0; i < kHeadDim / kMmaAtomN; i++) {
            float2 f_o_0 = __half22float2(HALF2(R_O[i][0]));
            float2 f_o_1 = __half22float2(HALF2(R_O[i][1]));
            f_o_0.x *= exp_m_old_minus_m_new_0;
            f_o_0.y *= exp_m_old_minus_m_new_0;
            f_o_1.x *= exp_m_old_minus_m_new_1;
            f_o_1.y *= exp_m_old_minus_m_new_1;
            HALF2(R_O[i][0]) = __float22half2_rn(f_o_0);
            HALF2(R_O[i][1]) = __float22half2_rn(f_o_1);
        }

        // ===== P@V MMA: A from R_S registers, B from s_v SMEM =====
        // R_S[k_tile][2] layout: each thread holds fragment of P[16 rows][k_tile*8 cols]
        // For m16n8k16 MMA, A needs 16x16 of P -> two adjacent R_S tiles
        #pragma unroll
        for (int i_dim = 0; i_dim < kHeadDim / kMmaAtomN; i_dim++) {
            #pragma unroll
            for (int p_step = 0; p_step < kWarpTileSeqLenK / 2; p_step++) {
                int w = p_step * 2;  // R_S index pair: [w][w+1] = P[:, p_step*16 : p_step*16+16]

                // Load V[tile, d_tile] from s_v (B matrix, transposed)
                int v_row = lane_id % kMmaAtomM + p_step * kMmaAtomK;
                int v_col = ((lane_id / kMmaAtomM) % 2) * 8 + i_dim * kMmaAtomN;
                uint32_t v_addr = __cvta_generic_to_shared(&s_v[v_row * kHeadDim + v_col]);
                LDMATRIX_X2_T(R_K[0], R_K[1], v_addr);

                // HMMA: A = R_S[w],R_S[w+1] (16x16 P), B = R_K (16x8 V)
                HMMA16816(R_O[i_dim][0], R_O[i_dim][1],
                          R_S[w][0], R_S[w][1],
                          R_S[w+1][0], R_S[w+1][1],
                          R_K[0], R_K[1],
                          R_O[i_dim][0], R_O[i_dim][1]);
            }
        }
        __syncthreads();

    }  // end loop over K seqlen

    // ===== Final rescale & writeback O =====
    float inv_l_0 = __frcp_rn(row_l[0]);
    float inv_l_1 = __frcp_rn(row_l[1]);

    #pragma unroll
    for (int i_dim = 0; i_dim < kHeadDim / kMmaAtomN; i_dim++) {
        float2 f_o_0 = __half22float2(HALF2(R_O[i_dim][0]));
        float2 f_o_1 = __half22float2(HALF2(R_O[i_dim][1]));
        f_o_0.x *= inv_l_0;
        f_o_0.y *= inv_l_0;
        f_o_1.x *= inv_l_1;
        f_o_1.y *= inv_l_1;

        int o_row = lane_id / 4 + warp_r * kWarpTileSeqLenQ * kMmaAtomM;
        int o_col = (lane_id % 4) * 2 + i_dim * kMmaAtomN;

        int g_row_o_0 = Tr * Br + o_row + 0;
        int g_row_o_8 = Tr * Br + o_row + 8;

        if (g_row_o_0 < QKV_seqlen) {
            LDST32BITS(O[O_addr_offset + (o_row + 0) * kHeadDim + o_col]) = __float22half2_rn(f_o_0);
        }
        if (g_row_o_8 < QKV_seqlen) {
            LDST32BITS(O[O_addr_offset + (o_row + 8) * kHeadDim + o_col]) = __float22half2_rn(f_o_1);
        }
    }
}


// ============================================================================
// Step 3: cp.async + Double Buffering (kStage=2)
// ============================================================================
// 核心改动 vs Step 2:
//   同步 LDST128BITS -> 异步 cp.async.cg (bypass L1, cache in L2)
//   SMEM Q/K 双缓冲: stage0 + stage1, 计算与加载重叠
//   预处理 stage0 的 Q,K -> 进入 d-loop: compute(stage0) + prefetch(stage1) 交替
//   SMEM: 2*(Br*24 + Bc*24) + Bc*d = 2*(1536+1536)+8192 = 14336 half = 28672 bytes

template <
    const int kHeadDim,
    const int kMmaAtomM = 16,
    const int kMmaAtomN = 8,
    const int kMmaAtomK = 16,
    const int kMmaTileSeqLenQ = 4,
    const int kMmaTileSeqLenK = 1,
    const int kWarpTileSeqLenQ = 1,
    const int kWarpTileSeqLenK = 8,
    const int kPadQ = 8,
    const int kPadK = 8,
    const int kStage = 2
    >
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_async_kernel(half *Q, half *K, half *V,
                            half *O, int QKV_seqlen,
                            int QKV_head) {
    constexpr int kNumThreads_async = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
    const int Tc = div_ceil(QKV_seqlen, Bc);
    const int Tr = blockIdx.y;

    int Q_gmem_offset = QKV_head * QKV_seqlen * kHeadDim + Tr * Br * kHeadDim;
    int K_gmem_offset = QKV_head * QKV_seqlen * kHeadDim;
    int V_gmem_offset = QKV_head * QKV_seqlen * kHeadDim;
    int O_gmem_offset = QKV_head * QKV_seqlen * kHeadDim + Tr * Br * kHeadDim;

    constexpr int Q_tile_sz = Br * (kMmaAtomK + kPadQ);  // 1536 halfs per stage
    constexpr int K_tile_sz = Bc * (kMmaAtomK + kPadK);  // 1536 halfs per stage
    extern __shared__ half smem[];
    half *Q_smem = smem;                          // kStage * Q_tile_sz
    half *K_smem = smem + kStage * Q_tile_sz;      // kStage * K_tile_sz
    half *s_v    = smem + kStage * (Q_tile_sz + K_tile_sz);

    uint32_t Q_base = __cvta_generic_to_shared(Q_smem);
    uint32_t K_base = __cvta_generic_to_shared(K_smem);

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warp_r  = warp_id / kMmaTileSeqLenK;
    int warp_c  = warp_id % kMmaTileSeqLenK;

    // --- Mapping: tid -> (row, col) within a Q/K tile ---
    int load_row_Q = tid / (kNumThreads_async / Br);           // 0..63
    int load_col_Q = (tid % (kNumThreads_async / Br)) * (kMmaAtomK / (kNumThreads_async / Br)); // 0,8
    int load_row_K = tid / (kNumThreads_async / Bc);
    int load_col_K = (tid % (kNumThreads_async / Bc)) * (kMmaAtomK / (kNumThreads_async / Bc));

    // Registers
    uint32_t R_Q[4], R_K[2];
    uint32_t R_O[kHeadDim / kMmaAtomN][2];
    #pragma unroll
    for (int i = 0; i < kHeadDim / kMmaAtomN; i++) { R_O[i][0] = 0; R_O[i][1] = 0; }

    float row_max[2] = {-INFINITY, -INFINITY};
    float row_l[2]   = {0.0f, 0.0f};
    float scale = 1.0f / sqrtf((float)kHeadDim);
    constexpr int kHalfPer128Bit = 8;

    #pragma unroll 1
    for (int j = 0; j < Tc; j++) {
        uint32_t R_S[kWarpTileSeqLenK][2];
        #pragma unroll
        for (int kt = 0; kt < kWarpTileSeqLenK; kt++) { R_S[kt][0] = 0; R_S[kt][1] = 0; }

        // ---- Prefetch stage 0 for Q and K (d_tile=0) ----
        {
            int g_d = 0 * kMmaAtomK + load_col_Q;
            if (Tr * Br + load_row_Q < QKV_seqlen) {
                uint32_t s_ptr = Q_base + (0 * Q_tile_sz + load_row_Q * (kMmaAtomK + kPadQ) + load_col_Q) * sizeof(half);
                CP_ASYNC_CG(s_ptr, &Q[Q_gmem_offset + load_row_Q * kHeadDim + g_d], 16);
            }
            g_d = 0 * kMmaAtomK + load_col_K;
            if (j * Bc + load_row_K < QKV_seqlen) {
                uint32_t s_ptr = K_base + (0 * K_tile_sz + load_row_K * (kMmaAtomK + kPadK) + load_col_K) * sizeof(half);
                CP_ASYNC_CG(s_ptr, &K[K_gmem_offset + j * Bc * kHeadDim + load_row_K * kHeadDim + g_d], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
            CP_ASYNC_WAIT_GROUP(0);
            __syncthreads();
        }

        // ===== Fine-grained d loop with double buffering =====
        #pragma unroll
        for (int d_tile = 0; d_tile < kHeadDim / kMmaAtomK; ++d_tile) {
            int stage     = d_tile % kStage;       // current compute stage
            int next_stg  = (d_tile + 1) % kStage;  // next prefetch stage
            int g_d       = d_tile * kMmaAtomK;

            // Prefetch next d_tile Q, K into next stage (unless last tile)
            if (d_tile + 1 < kHeadDim / kMmaAtomK) {
                int next_g_d = (d_tile + 1) * kMmaAtomK + load_col_Q;
                if (Tr * Br + load_row_Q < QKV_seqlen) {
                    uint32_t s_ptr = Q_base + (next_stg * Q_tile_sz + load_row_Q * (kMmaAtomK + kPadQ) + load_col_Q) * sizeof(half);
                    CP_ASYNC_CG(s_ptr, &Q[Q_gmem_offset + load_row_Q * kHeadDim + next_g_d], 16);
                }
                next_g_d = (d_tile + 1) * kMmaAtomK + load_col_K;
                if (j * Bc + load_row_K < QKV_seqlen) {
                    uint32_t s_ptr = K_base + (next_stg * K_tile_sz + load_row_K * (kMmaAtomK + kPadK) + load_col_K) * sizeof(half);
                    CP_ASYNC_CG(s_ptr, &K[K_gmem_offset + j * Bc * kHeadDim + load_row_K * kHeadDim + next_g_d], 16);
                }
                CP_ASYNC_COMMIT_GROUP();
            }

            // ---- Compute MMA with current stage ----
            int q_row = lane_id % kMmaAtomM + warp_r * kMmaAtomM;
            int q_col = (lane_id / kMmaAtomM) * 8;
            uint32_t q_addr = Q_base + (stage * Q_tile_sz + q_row * (kMmaAtomK + kPadQ) + q_col) * sizeof(half);
            LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], q_addr);

            #pragma unroll
            for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
                int k_row = lane_id % kMmaAtomN + kt * kMmaAtomN + warp_c * kWarpTileSeqLenK * kMmaAtomN;
                int k_col = ((lane_id / kMmaAtomN) % 2) * 8;
                uint32_t k_addr = K_base + (stage * K_tile_sz + k_row * (kMmaAtomK + kPadK) + k_col) * sizeof(half);
                LDMATRIX_X2(R_K[0], R_K[1], k_addr);

                HMMA16816(R_S[kt][0], R_S[kt][1],
                          R_Q[0], R_Q[1], R_Q[2], R_Q[3],
                          R_K[0], R_K[1],
                          R_S[kt][0], R_S[kt][1]);
            }

            // Wait for next stage to be ready
            if (d_tile + 1 < kHeadDim / kMmaAtomK) {
                CP_ASYNC_WAIT_GROUP(0);
                __syncthreads();
            }
        }  // end d loop

        // ===== Online Safe Softmax (P in R_S registers) =====
        float2 f_res[kWarpTileSeqLenK][2];
        #pragma unroll
        for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
            f_res[kt][0] = __half22float2(HALF2(R_S[kt][0]));
            f_res[kt][1] = __half22float2(HALF2(R_S[kt][1]));
            f_res[kt][0].x *= scale; f_res[kt][0].y *= scale;
            f_res[kt][1].x *= scale; f_res[kt][1].y *= scale;

            int kx = j * Bc + ((lane_id % 4) * 2 + 0) + kt * 8;
            int ky = j * Bc + ((lane_id % 4) * 2 + 1) + kt * 8;
            if (kx >= QKV_seqlen) { f_res[kt][0].x = -INFINITY; f_res[kt][1].x = -INFINITY; }
            if (ky >= QKV_seqlen) { f_res[kt][0].y = -INFINITY; f_res[kt][1].y = -INFINITY; }
        }

        float m0 = -INFINITY, m1 = -INFINITY;
        #pragma unroll
        for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
            m0 = fmaxf(m0, fmaxf(f_res[kt][0].x, f_res[kt][0].y));
            m1 = fmaxf(m1, fmaxf(f_res[kt][1].x, f_res[kt][1].y));
        }
        WARP_REDUCE_MAX_4(m0); WARP_REDUCE_MAX_4(m1);

        float mn0 = fmaxf(row_max[0], m0), mn1 = fmaxf(row_max[1], m1);
        float esc0 = expf(row_max[0] - mn0), esc1 = expf(row_max[1] - mn1);

        float ls0 = 0.0f, ls1 = 0.0f;
        #pragma unroll
        for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
            f_res[kt][0].x = expf(f_res[kt][0].x - mn0);
            f_res[kt][0].y = expf(f_res[kt][0].y - mn0);
            f_res[kt][1].x = expf(f_res[kt][1].x - mn1);
            f_res[kt][1].y = expf(f_res[kt][1].y - mn1);
            ls0 += f_res[kt][0].x + f_res[kt][0].y;
            ls1 += f_res[kt][1].x + f_res[kt][1].y;
        }
        WARP_REDUCE_SUM_4(ls0); WARP_REDUCE_SUM_4(ls1);

        row_l[0] = row_l[0] * esc0 + ls0;
        row_l[1] = row_l[1] * esc1 + ls1;
        row_max[0] = mn0; row_max[1] = mn1;

        // P back to R_S registers
        #pragma unroll
        for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
            HALF2(R_S[kt][0]) = __float22half2_rn(f_res[kt][0]);
            HALF2(R_S[kt][1]) = __float22half2_rn(f_res[kt][1]);
        }

        // ---- g(V) -> s(V) [Bc, kHeadDim] (sync load, same as before) ----
        constexpr int kVecV = kHeadDim / kHalfPer128Bit;
        constexpr int kItV = (Bc * kHeadDim) / kHalfPer128Bit / kNumThreads_async;
        #pragma unroll
        for (int iter = 0; iter < kItV; ++iter) {
            int vi = iter * kNumThreads_async + tid;
            int vr = vi / kVecV, vc = (vi % kVecV) * kHalfPer128Bit;
            if (j * Bc + vr < QKV_seqlen)
                LDST128BITS(s_v[vr * kHeadDim + vc]) = LDST128BITS(V[V_gmem_offset + j * Bc * kHeadDim + vr * kHeadDim + vc]);
            else { float4 z = {0,0,0,0}; LDST128BITS(s_v[vr * kHeadDim + vc]) = z; }
        }
        __syncthreads();

        // ---- Rescale O ----
        #pragma unroll
        for (int i = 0; i < kHeadDim / kMmaAtomN; i++) {
            float2 f0 = __half22float2(HALF2(R_O[i][0])), f1 = __half22float2(HALF2(R_O[i][1]));
            f0.x *= esc0; f0.y *= esc0; f1.x *= esc1; f1.y *= esc1;
            HALF2(R_O[i][0]) = __float22half2_rn(f0); HALF2(R_O[i][1]) = __float22half2_rn(f1);
        }

        // ---- P@V MMA from R_S registers ----
        #pragma unroll
        for (int id = 0; id < kHeadDim / kMmaAtomN; id++) {
            #pragma unroll
            for (int ps = 0; ps < kWarpTileSeqLenK / 2; ps++) {
                int w = ps * 2;
                int vr = lane_id % kMmaAtomM + ps * kMmaAtomK;
                int vc = ((lane_id / kMmaAtomM) % 2) * 8 + id * kMmaAtomN;
                uint32_t va = __cvta_generic_to_shared(&s_v[vr * kHeadDim + vc]);
                LDMATRIX_X2_T(R_K[0], R_K[1], va);

                HMMA16816(R_O[id][0], R_O[id][1],
                          R_S[w][0], R_S[w][1], R_S[w+1][0], R_S[w+1][1],
                          R_K[0], R_K[1],
                          R_O[id][0], R_O[id][1]);
            }
        }
        __syncthreads();
    }

    // ---- Final rescale & writeback ----
    float il0 = __frcp_rn(row_l[0]), il1 = __frcp_rn(row_l[1]);
    #pragma unroll
    for (int id = 0; id < kHeadDim / kMmaAtomN; id++) {
        float2 f0 = __half22float2(HALF2(R_O[id][0])), f1 = __half22float2(HALF2(R_O[id][1]));
        f0.x *= il0; f0.y *= il0; f1.x *= il1; f1.y *= il1;
        int orow = lane_id / 4 + warp_r * kWarpTileSeqLenQ * kMmaAtomM;
        int ocol = (lane_id % 4) * 2 + id * kMmaAtomN;
        if (Tr * Br + orow + 0 < QKV_seqlen)
            LDST32BITS(O[O_gmem_offset + (orow + 0) * kHeadDim + ocol]) = __float22half2_rn(f0);
        if (Tr * Br + orow + 8 < QKV_seqlen)
            LDST32BITS(O[O_gmem_offset + (orow + 8) * kHeadDim + ocol]) = __float22half2_rn(f1);
    }
}


// ============================================================================
// Step 4+5: Fully Optimized Flash Attention Kernel
// ============================================================================
// 累积所有优化:
//   1. 细粒度 Q/K/V tiling (SMEM O(1))
//   2. P 矩阵寄存器化 (R_S)
//   3. cp.async.cg + kStage=2 双缓冲 (Q,K,V)
//   4. V SMEM 复用 Q 空间 (Q@K^T 后 V 覆盖 Q/K)
//   5. __expf / __fmaf_rn / __hmax 高精度内建函数
//   6. O 累积使用 f32 (R_D) 而非 f16
//   7. Warp shuffle collective O store (128-bit 向量化写入)
//   SMEM 峰值: max(2*1536+2*1536, 2*1536) * 2 = 12288 bytes (V复用后)

template <
    const int kHeadDim,
    const int kMmaAtomM = 16,
    const int kMmaAtomN = 8,
    const int kMmaAtomK = 16,
    const int kMmaTileSeqLenQ = 4,
    const int kMmaTileSeqLenK = 1,
    const int kWarpTileSeqLenQ = 1,
    const int kWarpTileSeqLenK = 8,
    const int kWarpTileSeqLenP = 1,
    const int kWarpTileHeadDimV = 0,   // 0 = auto: kHeadDim/kMmaAtomN
    const int kPadQ = 8,
    const int kPadK = 8,
    const int kPadV = 8,
    const int kStage = 2,
    const int kOStorageAccF32 = 1      // 1 = O accumulate in float32
    >
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_final_kernel(half *Q, half *K, half *V,
                            half *O, int QKV_seqlen,
                            int QKV_head) {
    constexpr int kNumThreads_final = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;   // 64
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;   // 64
    constexpr int WarpHeadDimV = (kWarpTileHeadDimV == 0) ? (kHeadDim / kMmaAtomN) : kWarpTileHeadDimV;
    const int Tc = div_ceil(QKV_seqlen, Bc);
    const int Tr = blockIdx.y;

    int Q_gmem_ofs = QKV_head * QKV_seqlen * kHeadDim + Tr * Br * kHeadDim;
    int K_gmem_ofs = QKV_head * QKV_seqlen * kHeadDim;
    int V_gmem_ofs = QKV_head * QKV_seqlen * kHeadDim;
    int O_gmem_ofs = QKV_head * QKV_seqlen * kHeadDim + Tr * Br * kHeadDim;

    // SMEM: [Q_s0][Q_s1][K_s0][K_s1], V reuses Q after Q@K^T
    constexpr int Q_sz = Br * (kMmaAtomK + kPadQ);       // 1536
    constexpr int K_sz = Bc * (kMmaAtomK + kPadK);       // 1536
    constexpr int V_sz = Bc * (kMmaAtomN * 2 + kPadV);   // 1536 (16+8=24 cols)
    extern __shared__ half smem[];
    half *Q_smem = smem;
    half *K_smem = smem + kStage * Q_sz;
    half *V_smem = Q_smem;  // V reuses Q SMEM after Q@K^T

    uint32_t Q_base = __cvta_generic_to_shared(Q_smem);
    uint32_t K_base = __cvta_generic_to_shared(K_smem);
    uint32_t V_base = __cvta_generic_to_shared(V_smem);

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warp_QP = warp_id / kMmaTileSeqLenK;   // 0..3
    int warp_KV = warp_id % kMmaTileSeqLenK;   // 0

    // Mapping tid -> (row, col) for loading Q/K/V tiles
    int ldQ_row = tid / (kNumThreads_final / Br);
    int ldQ_col = (tid % (kNumThreads_final / Br)) * (kMmaAtomK / (kNumThreads_final / Br));
    int ldK_row = tid / (kNumThreads_final / Bc);
    int ldK_col = (tid % (kNumThreads_final / Bc)) * (kMmaAtomK / (kNumThreads_final / Bc));
    int ldV_row = tid / (kNumThreads_final / Bc);
    int ldV_col = (tid % (kNumThreads_final / Bc)) * (kMmaAtomN * 2 / (kNumThreads_final / Bc));

    // Registers
    uint32_t R_Q[4], R_K[2], R_V[2], R_O[2];
    // R_D: final accumulated O in registers [kWarpTileSeqLenP][WarpHeadDimV][2 or 4]
    uint32_t R_D[kWarpTileSeqLenP][WarpHeadDimV][(kOStorageAccF32) ? 4 : 2];
    #pragma unroll
    for (int j = 0; j < WarpHeadDimV; j++) {
        R_D[0][j][0] = 0; R_D[0][j][1] = 0;
        if constexpr (kOStorageAccF32) { R_D[0][j][2] = 0; R_D[0][j][3] = 0; }
    }

    float row_max_old[kWarpTileSeqLenQ][2];
    float row_sum_old[kWarpTileSeqLenQ][2];
    row_max_old[0][0] = -INFINITY; row_max_old[0][1] = -INFINITY;
    row_sum_old[0][0] = 0.0f;      row_sum_old[0][1] = 0.0f;

    float scale = 1.0f / sqrtf((float)kHeadDim);
    constexpr int kHper128 = 8;

    // ===== OUTER LOOP over K seqlen tiles =====
    #pragma unroll 1
    for (int j = 0; j < Tc; j++) {
        uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][2];
        #pragma unroll
        for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
            R_S[0][kt][0] = 0; R_S[0][kt][1] = 0;
        }

        // ---- Prefetch Q,K stage 0 ----
        {
            int gd = 0 + ldQ_col;
            if (Tr * Br + ldQ_row < QKV_seqlen) {
                uint32_t sp = Q_base + (0 * Q_sz + ldQ_row * (kMmaAtomK + kPadQ) + ldQ_col) * sizeof(half);
                CP_ASYNC_CG(sp, &Q[Q_gmem_ofs + ldQ_row * kHeadDim + gd], 16);
            }
            gd = 0 + ldK_col;
            if (j * Bc + ldK_row < QKV_seqlen) {
                uint32_t sp = K_base + (0 * K_sz + ldK_row * (kMmaAtomK + kPadK) + ldK_col) * sizeof(half);
                CP_ASYNC_CG(sp, &K[K_gmem_ofs + j * Bc * kHeadDim + ldK_row * kHeadDim + gd], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
            CP_ASYNC_WAIT_GROUP(0);
            __syncthreads();
        }

        // ===== Q@K^T with double buffering =====
        #pragma unroll
        for (int d_tile = 0; d_tile < kHeadDim / kMmaAtomK; ++d_tile) {
            int stg = d_tile % kStage;
            int nxt = (d_tile + 1) % kStage;

            // Prefetch next Q,K
            if (d_tile + 1 < kHeadDim / kMmaAtomK) {
                int ngd = (d_tile + 1) * kMmaAtomK + ldQ_col;
                if (Tr * Br + ldQ_row < QKV_seqlen) {
                    uint32_t sp = Q_base + (nxt * Q_sz + ldQ_row * (kMmaAtomK + kPadQ) + ldQ_col) * sizeof(half);
                    CP_ASYNC_CG(sp, &Q[Q_gmem_ofs + ldQ_row * kHeadDim + ngd], 16);
                }
                ngd = (d_tile + 1) * kMmaAtomK + ldK_col;
                if (j * Bc + ldK_row < QKV_seqlen) {
                    uint32_t sp = K_base + (nxt * K_sz + ldK_row * (kMmaAtomK + kPadK) + ldK_col) * sizeof(half);
                    CP_ASYNC_CG(sp, &K[K_gmem_ofs + j * Bc * kHeadDim + ldK_row * kHeadDim + ngd], 16);
                }
                CP_ASYNC_COMMIT_GROUP();
            }

            // MMA with current stage
            int q_row = lane_id % kMmaAtomM + warp_QP * (kMmaAtomM * kWarpTileSeqLenQ);
            int q_col = (lane_id / kMmaAtomM) * 8;
            uint32_t qa = Q_base + (stg * Q_sz + q_row * (kMmaAtomK + kPadQ) + q_col) * sizeof(half);
            LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], qa);

            #pragma unroll
            for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
                int k_row = lane_id % kMmaAtomN + kt * kMmaAtomN + warp_KV * kWarpTileSeqLenK * kMmaAtomN;
                int k_col = ((lane_id / kMmaAtomN) % 2) * 8;
                uint32_t ka = K_base + (stg * K_sz + k_row * (kMmaAtomK + kPadK) + k_col) * sizeof(half);
                LDMATRIX_X2(R_K[0], R_K[1], ka);
                HMMA16816(R_S[0][kt][0], R_S[0][kt][1],
                          R_Q[0], R_Q[1], R_Q[2], R_Q[3],
                          R_K[0], R_K[1],
                          R_S[0][kt][0], R_S[0][kt][1]);
            }

            if (d_tile + 1 < kHeadDim / kMmaAtomK) {
                CP_ASYNC_WAIT_GROUP(0);
                __syncthreads();
            }
        }

        // ===== Online Safe Softmax (__expf, __fmaf_rn, __hmax) =====
        float row_max_new[2] = {-INFINITY, -INFINITY};
        float row_sum_new[2] = {0.0f, 0.0f};

        #pragma unroll
        for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
            half *hp = reinterpret_cast<half *>(&R_S[0][kt][0]);
            float t0 = __half2float(__hmax(hp[0], hp[1])) * scale;
            float t1 = __half2float(__hmax(hp[2], hp[3])) * scale;
            row_max_new[0] = max(row_max_new[0], t0);
            row_max_new[1] = max(row_max_new[1], t1);
        }

        row_max_new[0] = fmaxf(row_max_new[0], __shfl_xor_sync(0xffffffff, row_max_new[0], 1));
        row_max_new[1] = fmaxf(row_max_new[1], __shfl_xor_sync(0xffffffff, row_max_new[1], 1));
        row_max_new[0] = fmaxf(row_max_new[0], __shfl_xor_sync(0xffffffff, row_max_new[0], 2));
        row_max_new[1] = fmaxf(row_max_new[1], __shfl_xor_sync(0xffffffff, row_max_new[1], 2));

        float m_new0 = fmaxf(row_max_old[0][0], row_max_new[0]);
        float m_new1 = fmaxf(row_max_old[0][1], row_max_new[1]);

        #pragma unroll
        for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
            half *hp = reinterpret_cast<half *>(&R_S[0][kt][0]);
            float4 pval;
            pval.x = __expf(__fmaf_rn(__half2float(hp[0]), scale, -m_new0));
            pval.y = __expf(__fmaf_rn(__half2float(hp[1]), scale, -m_new0));
            pval.z = __expf(__fmaf_rn(__half2float(hp[2]), scale, -m_new1));
            pval.w = __expf(__fmaf_rn(__half2float(hp[3]), scale, -m_new1));
            row_sum_new[0] += pval.x + pval.y;
            row_sum_new[1] += pval.z + pval.w;
            hp[0] = __float2half_rn(pval.x); hp[1] = __float2half_rn(pval.y);
            hp[2] = __float2half_rn(pval.z); hp[3] = __float2half_rn(pval.w);
        }

        row_sum_new[0] += __shfl_xor_sync(0xffffffff, row_sum_new[0], 1);
        row_sum_new[1] += __shfl_xor_sync(0xffffffff, row_sum_new[1], 1);
        row_sum_new[0] += __shfl_xor_sync(0xffffffff, row_sum_new[0], 2);
        row_sum_new[1] += __shfl_xor_sync(0xffffffff, row_sum_new[1], 2);

        float m_old0 = (j > 0) ? row_max_old[0][0] : m_new0;
        float m_old1 = (j > 0) ? row_max_old[0][1] : m_new1;
        float rescale0 = __expf(m_old0 - m_new0);
        float rescale1 = __expf(m_old1 - m_new1);

        // ===== Prefetch V stage 0, 1 =====
        #pragma unroll
        for (int stg = 0; stg < (kStage - 1); ++stg) {
            int gd = stg * (kMmaAtomN * 2) + ldV_col;
            if (j * Bc + ldV_row < QKV_seqlen) {
                uint32_t sp = V_base + (stg * V_sz + ldV_row * (kMmaAtomN * 2 + kPadV) + ldV_col) * sizeof(half);
                CP_ASYNC_CG(sp, &V[V_gmem_ofs + j * Bc * kHeadDim + ldV_row * kHeadDim + gd], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
        }
        CP_ASYNC_WAIT_GROUP(kStage - 2);
        __syncthreads();

        // ===== P@V with fine-grained V + double buffering =====
        #pragma unroll
        for (int jv = 0; jv < WarpHeadDimV; ++jv) {  // loop over d dimension
            uint32_t R_O_loc[2] = {0, 0};  // clear for each d tile

            int stg_v = (jv / 2) % kStage;
            int nxt_v  = ((jv / 2) + (kStage - 1)) % kStage;

            // Prefetch next V tile
            if (jv % 2 == 0) {
                if ((jv / 2 + 1) < (WarpHeadDimV / 2)) {
                    int ngd = ((jv / 2 + 1) * kMmaAtomN * 2) + ldV_col;
                    if (j * Bc + ldV_row < QKV_seqlen) {
                        uint32_t sp = V_base + (nxt_v * V_sz + ldV_row * (kMmaAtomN * 2 + kPadV) + ldV_col) * sizeof(half);
                        CP_ASYNC_CG(sp, &V[V_gmem_ofs + j * Bc * kHeadDim + ldV_row * kHeadDim + ngd], 16);
                    }
                    CP_ASYNC_COMMIT_GROUP();
                }
            }

            // P@V MMA for each Bc chunk
            #pragma unroll
            for (int tile_V_Bc = 0; tile_V_Bc < Bc / kMmaAtomK; ++tile_V_Bc) {
                int v_d = warp_KV * (kMmaAtomN * WarpHeadDimV) + (jv % 2) * kMmaAtomN;
                int v_row = tile_V_Bc * kMmaAtomK + lane_id % 16;
                uint32_t va = V_base + (stg_v * V_sz + v_row * (kMmaAtomN * 2 + kPadV) + v_d) * sizeof(half);
                LDMATRIX_X2_T(R_V[0], R_V[1], va);

                int w = tile_V_Bc * 2;  // R_S index pair
                HMMA16816(R_O_loc[0], R_O_loc[1],
                          R_S[0][w][0], R_S[0][w][1],
                          R_S[0][w+1][0], R_S[0][w+1][1],
                          R_V[0], R_V[1],
                          R_O_loc[0], R_O_loc[1]);
            }

            // Wait for next V stage
            if (jv % 2 == 1) {
                CP_ASYNC_WAIT_GROUP(kStage - 2);
                __syncthreads();
            }

            // Online rescale: R_D += rescale * R_O_loc
            half *hp_O = reinterpret_cast<half *>(&R_O_loc[0]);
            if constexpr (kOStorageAccF32) {
                float *fp_D = reinterpret_cast<float *>(&R_D[0][jv][0]);
                fp_D[0] = __fmaf_rn(rescale0, fp_D[0], __half2float(hp_O[0]));
                fp_D[1] = __fmaf_rn(rescale0, fp_D[1], __half2float(hp_O[1]));
                fp_D[2] = __fmaf_rn(rescale1, fp_D[2], __half2float(hp_O[2]));
                fp_D[3] = __fmaf_rn(rescale1, fp_D[3], __half2float(hp_O[3]));
            } else {
                half *hp_D = reinterpret_cast<half *>(&R_D[0][jv][0]);
                hp_D[0] = __float2half_rn(__fmaf_rn(rescale0, __half2float(hp_D[0]), __half2float(hp_O[0])));
                hp_D[1] = __float2half_rn(__fmaf_rn(rescale0, __half2float(hp_D[1]), __half2float(hp_O[1])));
                hp_D[2] = __float2half_rn(__fmaf_rn(rescale1, __half2float(hp_D[2]), __half2float(hp_O[2])));
                hp_D[3] = __float2half_rn(__fmaf_rn(rescale1, __half2float(hp_D[3]), __half2float(hp_O[3])));
            }
        }

        // Update running max/sum
        row_sum_old[0][0] = __fmaf_rn(rescale0, row_sum_old[0][0], row_sum_new[0]);
        row_sum_old[0][1] = __fmaf_rn(rescale1, row_sum_old[0][1], row_sum_new[1]);
        row_max_old[0][0] = m_new0;
        row_max_old[0][1] = m_new1;

        __syncthreads();
    }  // end K seqlen loop

    // ===== Final rescale: O = R_D / row_sum =====
    float inv_l0 = __frcp_rn(row_sum_old[0][0]);
    float inv_l1 = __frcp_rn(row_sum_old[0][1]);

    #pragma unroll
    for (int jv = 0; jv < WarpHeadDimV; ++jv) {
        if constexpr (kOStorageAccF32) {
            float *fp = reinterpret_cast<float *>(&R_D[0][jv][0]);
            half  *hp = reinterpret_cast<half *>(&R_D[0][jv][0]);
            hp[0] = __float2half_rn(inv_l0 * fp[0]);
            hp[1] = __float2half_rn(inv_l0 * fp[1]);
            hp[2] = __float2half_rn(inv_l1 * fp[2]);
            hp[3] = __float2half_rn(inv_l1 * fp[3]);
        } else {
            half *hp = reinterpret_cast<half *>(&R_D[0][jv][0]);
            hp[0] = __float2half_rn(inv_l0 * __half2float(hp[0]));
            hp[1] = __float2half_rn(inv_l0 * __half2float(hp[1]));
            hp[2] = __float2half_rn(inv_l1 * __half2float(hp[2]));
            hp[3] = __float2half_rn(inv_l1 * __half2float(hp[3]));
        }
    }

    // ===== Warp Shuffle Collective O Store (128-bit vectorized) =====
    // Reuse R_Q[4] and R_K[2] for collective store buffer
    #pragma unroll
    for (int jv = 0; jv < WarpHeadDimV; ++jv) {
        uint32_t *Z0 = reinterpret_cast<uint32_t *>(&R_Q[0]);
        uint32_t *Z1 = reinterpret_cast<uint32_t *>(&R_K[0]);
        Z0[0] = R_D[0][jv][0];
        Z1[0] = R_D[0][jv][1];
        Z0[1] = __shfl_sync(0xffffffff, R_D[0][jv][0], lane_id + 1, 4);
        Z0[2] = __shfl_sync(0xffffffff, R_D[0][jv][0], lane_id + 2, 4);
        Z0[3] = __shfl_sync(0xffffffff, R_D[0][jv][0], lane_id + 3, 4);
        Z1[1] = __shfl_sync(0xffffffff, R_D[0][jv][1], lane_id + 1, 4);
        Z1[2] = __shfl_sync(0xffffffff, R_D[0][jv][1], lane_id + 2, 4);
        Z1[3] = __shfl_sync(0xffffffff, R_D[0][jv][1], lane_id + 3, 4);

        if (lane_id % 4 == 0) {
            int o_br = warp_QP * (kMmaAtomM * kWarpTileSeqLenP);
            int o_d  = warp_KV * (kMmaAtomN * WarpHeadDimV) + jv * kMmaAtomN;
            int g_row0 = Tr * Br + o_br + lane_id / 4;
            int g_row8 = Tr * Br + o_br + lane_id / 4 + 8;
            if (g_row0 < QKV_seqlen)
                LDST128BITS(O[O_gmem_ofs + (o_br + lane_id / 4) * kHeadDim + o_d]) = LDST128BITS(Z0[0]);
            if (g_row8 < QKV_seqlen)
                LDST128BITS(O[O_gmem_ofs + (o_br + lane_id / 4 + 8) * kHeadDim + o_d]) = LDST128BITS(Z1[0]);
        }
    }
}


// ============================================================================
// TUNING Step A: Bigger Tiles (Br=128, Bc=128) + 8 Warps (256 threads)
// ============================================================================
// 改动 vs final_optimized:
//   kMmaTileSeqLenQ: 4→8  (8 warps, 256 threads, 2× occupancy)
//   kWarpTileSeqLenK: 8→16 (Bc: 64→128, Tc 减半, SMEM 利用率↑)
//   Br: 64→128, Bc: 64→128  (每次 K-seqlen 迭代做 4× 的工作)
//   R_S: [1][8][2]→[1][16][2] (每线程 16 个 K tile)
//   V chunks: 4→8  (Bc/16)
//   SMEM: 2*(128*24+128*24) = 12288 half = 24KB (仍远 < 48KB)

template <
    const int kHeadDim,
    const int kMmaAtomM = 16,
    const int kMmaAtomN = 8,
    const int kMmaAtomK = 16,
    const int kMmaTileSeqLenQ = 8,    // 8 warps for Q rows  ← KEY CHANGE
    const int kMmaTileSeqLenK = 1,
    const int kWarpTileSeqLenQ = 1,
    const int kWarpTileSeqLenK = 16,  // Bc=128              ← KEY CHANGE
    const int kWarpTileSeqLenP = 1,
    const int kWarpTileHeadDimV = 0,   // auto
    const int kPadQ = 8,
    const int kPadK = 8,
    const int kPadV = 8,
    const int kStage = 2,
    const int kOStorageAccF32 = 1
    >
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_tuned_A_kernel(
    // NOTE: when using per-head pointers with interleaved KV-cache,
    //   set stride_KV = hidden_size, K_ofs_override = 0, V_ofs_override = 0.
    //   QKV_head should be 1 (or 0 for the base offset).
    half *Q, half *K, half *V,
    half *O, int QKV_seqlen,
    int QKV_head,
    int stride_Q  = kHeadDim,
    int stride_KV = kHeadDim,
    int K_ofs_override = -1,   // -1 = auto from QKV_head; 0 = per-head pointer
    int V_ofs_override = -1, int Q_ofs_override = -1,
    int O_ofs_override = -1
    ) {
    constexpr int kThrA = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;  // 128
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;  // 128
    constexpr int WHDV = (kWarpTileHeadDimV == 0) ? (kHeadDim / kMmaAtomN) : kWarpTileHeadDimV;
    const int Tc = div_ceil(QKV_seqlen, Bc);
    const int Tr = blockIdx.y;

    // Per-head pointers: QKV_head=1 means Qh/Kh/Vh/Oh point to single head's data.
    // Offset = QKV_head * QKV_seqlen * kHeadDim works as a "skip past this head" offset.
    int Q_ofs = (Q_ofs_override >= 0) ? Q_ofs_override + Tr * Br * stride_Q : QKV_head * QKV_seqlen * stride_Q  + Tr * Br * stride_Q;
    int K_ofs = (K_ofs_override >= 0) ? K_ofs_override : QKV_head * QKV_seqlen * stride_KV;
    int V_ofs = (V_ofs_override >= 0) ? V_ofs_override : QKV_head * QKV_seqlen * stride_KV;
    int O_ofs = (O_ofs_override >= 0) ? O_ofs_override + Tr * Br * stride_Q
              : QKV_head * QKV_seqlen * stride_Q + Tr * Br * stride_Q;

    constexpr int Q_sz = Br * (kMmaAtomK + kPadQ);        // 128*24=3072
    constexpr int K_sz = Bc * (kMmaAtomK + kPadK);        // 128*24=3072
    constexpr int V_sz = Bc * (kMmaAtomN * 2 + kPadV);    // 128*24=3072
    extern __shared__ half smem[];
    half *Q_smem = smem;
    half *K_smem = smem + kStage * Q_sz;
    half *V_smem = Q_smem;

    uint32_t Q_base = __cvta_generic_to_shared(Q_smem);
    uint32_t K_base = __cvta_generic_to_shared(K_smem);
    uint32_t V_base = __cvta_generic_to_shared(V_smem);

    int tid = threadIdx.x;
    int lid = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int wQP = wid / kMmaTileSeqLenK;  // 0..7
    int wKV = wid % kMmaTileSeqLenK;  // 0

    // Load mapping
    int ldQ_r = tid / (kThrA / Br);                    // 0..127
    int ldQ_c = (tid % (kThrA / Br)) * (kMmaAtomK / (kThrA / Br)); // 0,8
    int ldK_r = tid / (kThrA / Bc);
    int ldK_c = (tid % (kThrA / Bc)) * (kMmaAtomK / (kThrA / Bc));
    int ldV_r = tid / (kThrA / Bc);
    int ldV_c = (tid % (kThrA / Bc)) * (kMmaAtomN * 2 / (kThrA / Bc));

    // Registers
    uint32_t R_Q[4], R_K[2], R_V[2], R_O[2];
    uint32_t R_D[kWarpTileSeqLenP][WHDV][(kOStorageAccF32) ? 4 : 2];
    #pragma unroll
    for (int j = 0; j < WHDV; j++) {
        R_D[0][j][0] = 0; R_D[0][j][1] = 0;
        if constexpr (kOStorageAccF32) { R_D[0][j][2] = 0; R_D[0][j][3] = 0; }
    }

    float rm_old[2] = {-INFINITY, -INFINITY};
    float rl_old[2] = {0.0f, 0.0f};
    float scale = 1.0f / sqrtf((float)kHeadDim);

    #pragma unroll 1
    for (int j = 0; j < Tc; j++) {
        uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][2];  // [1][16][2]
        #pragma unroll
        for (int kt = 0; kt < kWarpTileSeqLenK; kt++)
            { R_S[0][kt][0] = 0; R_S[0][kt][1] = 0; }

        // Prefetch Q,K stage 0
        {
            if (Tr * Br + ldQ_r < QKV_seqlen) {
                uint32_t sp = Q_base + (0 * Q_sz + ldQ_r * (kMmaAtomK + kPadQ) + ldQ_c) * sizeof(half);
                CP_ASYNC_CG(sp, &Q[Q_ofs + ldQ_r * stride_Q + ldQ_c], 16);
            }
            if (j * Bc + ldK_r < QKV_seqlen) {
                uint32_t sp = K_base + (0 * K_sz + ldK_r * (kMmaAtomK + kPadK) + ldK_c) * sizeof(half);
                CP_ASYNC_CG(sp, &K[K_ofs + j * Bc * stride_KV + ldK_r * stride_KV + ldK_c], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
            CP_ASYNC_WAIT_GROUP(0);
            __syncthreads();
        }

        // Q@K^T d-loop with double buffering
        #pragma unroll
        for (int dt = 0; dt < kHeadDim / kMmaAtomK; ++dt) {
            int stg = dt % kStage, nxt = (dt + 1) % kStage;

            if (dt + 1 < kHeadDim / kMmaAtomK) {
                int ngd = (dt + 1) * kMmaAtomK + ldQ_c;
                if (Tr * Br + ldQ_r < QKV_seqlen) {
                    uint32_t sp = Q_base + (nxt * Q_sz + ldQ_r * (kMmaAtomK + kPadQ) + ldQ_c) * sizeof(half);
                    CP_ASYNC_CG(sp, &Q[Q_ofs + ldQ_r * stride_Q + ngd], 16);
                }
                ngd = (dt + 1) * kMmaAtomK + ldK_c;
                if (j * Bc + ldK_r < QKV_seqlen) {
                    uint32_t sp = K_base + (nxt * K_sz + ldK_r * (kMmaAtomK + kPadK) + ldK_c) * sizeof(half);
                    CP_ASYNC_CG(sp, &K[K_ofs + j * Bc * stride_KV + ldK_r * stride_KV + ngd], 16);
                }
                CP_ASYNC_COMMIT_GROUP();
            }

            int q_row = lid % kMmaAtomM + wQP * (kMmaAtomM * kWarpTileSeqLenQ);
            int q_col = (lid / kMmaAtomM) * 8;
            uint32_t qa = Q_base + (stg * Q_sz + q_row * (kMmaAtomK + kPadQ) + q_col) * sizeof(half);
            LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], qa);

            #pragma unroll
            for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
                int k_row = lid % kMmaAtomN + kt * kMmaAtomN + wKV * kWarpTileSeqLenK * kMmaAtomN;
                int k_col = ((lid / kMmaAtomN) % 2) * 8;
                uint32_t ka = K_base + (stg * K_sz + k_row * (kMmaAtomK + kPadK) + k_col) * sizeof(half);
                LDMATRIX_X2(R_K[0], R_K[1], ka);
                HMMA16816(R_S[0][kt][0], R_S[0][kt][1],
                          R_Q[0], R_Q[1], R_Q[2], R_Q[3],
                          R_K[0], R_K[1],
                          R_S[0][kt][0], R_S[0][kt][1]);
            }

            if (dt + 1 < kHeadDim / kMmaAtomK) {
                CP_ASYNC_WAIT_GROUP(0);
                __syncthreads();
            }
        }

        // Online Safe Softmax (__hmax, __expf, __fmaf_rn)
        // Causal mask (if enabled): token i only attends to tokens [0..i]
#ifdef FLASH_ATTN_CAUSAL
        #pragma message("FLASH_ATTN_CAUSAL is ACTIVE in tuned_A kernel")
        #pragma unroll
        for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
            half *hp0 = reinterpret_cast<half *>(&R_S[0][kt][0]);
            half *hp1 = reinterpret_cast<half *>(&R_S[0][kt][1]);
            int q0 = Tr * Br + wQP * kMmaAtomM + lid / 4;
            int q1 = q0 + kMmaAtomM / 2;
            int k0 = j * Bc + kt * kMmaAtomN + (lid % 4) * 2;
            int k1 = k0 + 1;
            if (k0 > q0) hp0[0] = __float2half(-65500.0f);
            if (k1 > q0) hp0[1] = __float2half(-65500.0f);
            if (k0 > q1) hp1[0] = __float2half(-65500.0f);
            if (k1 > q1) hp1[1] = __float2half(-65500.0f);
        }
#endif
        float rm_new[2] = {-INFINITY, -INFINITY};
        float rl_new[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
            half *hp = reinterpret_cast<half *>(&R_S[0][kt][0]);
            rm_new[0] = max(rm_new[0], __half2float(__hmax(hp[0], hp[1])) * scale);
            rm_new[1] = max(rm_new[1], __half2float(__hmax(hp[2], hp[3])) * scale);
        }
        rm_new[0] = fmaxf(rm_new[0], __shfl_xor_sync(0xffffffff, rm_new[0], 1));
        rm_new[1] = fmaxf(rm_new[1], __shfl_xor_sync(0xffffffff, rm_new[1], 1));
        rm_new[0] = fmaxf(rm_new[0], __shfl_xor_sync(0xffffffff, rm_new[0], 2));
        rm_new[1] = fmaxf(rm_new[1], __shfl_xor_sync(0xffffffff, rm_new[1], 2));

        float mn0 = fmaxf(rm_old[0], rm_new[0]), mn1 = fmaxf(rm_old[1], rm_new[1]);

        #pragma unroll
        for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
            half *hp = reinterpret_cast<half *>(&R_S[0][kt][0]);
            float4 pv;
            pv.x = __expf(__fmaf_rn(__half2float(hp[0]), scale, -mn0));
            pv.y = __expf(__fmaf_rn(__half2float(hp[1]), scale, -mn0));
            pv.z = __expf(__fmaf_rn(__half2float(hp[2]), scale, -mn1));
            pv.w = __expf(__fmaf_rn(__half2float(hp[3]), scale, -mn1));
            rl_new[0] += pv.x + pv.y;
            rl_new[1] += pv.z + pv.w;
            hp[0] = __float2half_rn(pv.x); hp[1] = __float2half_rn(pv.y);
            hp[2] = __float2half_rn(pv.z); hp[3] = __float2half_rn(pv.w);
        }
        rl_new[0] += __shfl_xor_sync(0xffffffff, rl_new[0], 1);
        rl_new[1] += __shfl_xor_sync(0xffffffff, rl_new[1], 1);
        rl_new[0] += __shfl_xor_sync(0xffffffff, rl_new[0], 2);
        rl_new[1] += __shfl_xor_sync(0xffffffff, rl_new[1], 2);

        float m_old0 = (j > 0) ? rm_old[0] : mn0, m_old1 = (j > 0) ? rm_old[1] : mn1;
        float resc0 = __expf(m_old0 - mn0), resc1 = __expf(m_old1 - mn1);

        // Prefetch V
        #pragma unroll
        for (int stg = 0; stg < (kStage - 1); ++stg) {
            if (j * Bc + ldV_r < QKV_seqlen) {
                uint32_t sp = V_base + (stg * V_sz + ldV_r * (kMmaAtomN * 2 + kPadV) + ldV_c) * sizeof(half);
                CP_ASYNC_CG(sp, &V[V_ofs + j * Bc * stride_KV + ldV_r * stride_KV + stg * (kMmaAtomN * 2) + ldV_c], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
        }
        CP_ASYNC_WAIT_GROUP(kStage - 2);
        __syncthreads();

        // P@V with fine-grained V
        #pragma unroll
        for (int jv = 0; jv < WHDV; ++jv) {
            uint32_t R_O_loc[2] = {0, 0};
            int stg_v = (jv / 2) % kStage, nxt_v = ((jv / 2) + (kStage - 1)) % kStage;

            if (jv % 2 == 0 && (jv / 2 + 1) < (WHDV / 2)) {
                int ngd = ((jv / 2 + 1) * kMmaAtomN * 2) + ldV_c;
                if (j * Bc + ldV_r < QKV_seqlen) {
                    uint32_t sp = V_base + (nxt_v * V_sz + ldV_r * (kMmaAtomN * 2 + kPadV) + ldV_c) * sizeof(half);
                    CP_ASYNC_CG(sp, &V[V_ofs + j * Bc * stride_KV + ldV_r * stride_KV + ngd], 16);
                }
                CP_ASYNC_COMMIT_GROUP();
            }

            #pragma unroll
            for (int tbc = 0; tbc < Bc / kMmaAtomK; ++tbc) {  // 8 chunks ← was 4
                int v_d = wKV * (kMmaAtomN * WHDV) + (jv % 2) * kMmaAtomN;
                int v_r = tbc * kMmaAtomK + lid % 16;
                uint32_t va = V_base + (stg_v * V_sz + v_r * (kMmaAtomN * 2 + kPadV) + v_d) * sizeof(half);
                LDMATRIX_X2_T(R_V[0], R_V[1], va);

                int w = tbc * 2;
                HMMA16816(R_O_loc[0], R_O_loc[1],
                          R_S[0][w][0], R_S[0][w][1],
                          R_S[0][w+1][0], R_S[0][w+1][1],
                          R_V[0], R_V[1],
                          R_O_loc[0], R_O_loc[1]);
            }

            if (jv % 2 == 1) {
                CP_ASYNC_WAIT_GROUP(kStage - 2);
                __syncthreads();
            }

            half *hp_O = reinterpret_cast<half *>(&R_O_loc[0]);
            if constexpr (kOStorageAccF32) {
                float *fp_D = reinterpret_cast<float *>(&R_D[0][jv][0]);
                fp_D[0] = __fmaf_rn(resc0, fp_D[0], __half2float(hp_O[0]));
                fp_D[1] = __fmaf_rn(resc0, fp_D[1], __half2float(hp_O[1]));
                fp_D[2] = __fmaf_rn(resc1, fp_D[2], __half2float(hp_O[2]));
                fp_D[3] = __fmaf_rn(resc1, fp_D[3], __half2float(hp_O[3]));
            } else {
                half *hp_D = reinterpret_cast<half *>(&R_D[0][jv][0]);
                hp_D[0] = __float2half_rn(__fmaf_rn(resc0, __half2float(hp_D[0]), __half2float(hp_O[0])));
                hp_D[1] = __float2half_rn(__fmaf_rn(resc0, __half2float(hp_D[1]), __half2float(hp_O[1])));
                hp_D[2] = __float2half_rn(__fmaf_rn(resc1, __half2float(hp_D[2]), __half2float(hp_O[2])));
                hp_D[3] = __float2half_rn(__fmaf_rn(resc1, __half2float(hp_D[3]), __half2float(hp_O[3])));
            }
        }

        rl_old[0] = __fmaf_rn(resc0, rl_old[0], rl_new[0]);
        rl_old[1] = __fmaf_rn(resc1, rl_old[1], rl_new[1]);
        rm_old[0] = mn0; rm_old[1] = mn1;
        __syncthreads();
    }

    // Final rescale
    float il0 = __frcp_rn(rl_old[0]), il1 = __frcp_rn(rl_old[1]);
    #pragma unroll
    for (int jv = 0; jv < WHDV; ++jv) {
        if constexpr (kOStorageAccF32) {
            float *fp = reinterpret_cast<float *>(&R_D[0][jv][0]);
            half  *hp = reinterpret_cast<half *>(&R_D[0][jv][0]);
            hp[0] = __float2half_rn(il0 * fp[0]); hp[1] = __float2half_rn(il0 * fp[1]);
            hp[2] = __float2half_rn(il1 * fp[2]); hp[3] = __float2half_rn(il1 * fp[3]);
        } else {
            half *hp = reinterpret_cast<half *>(&R_D[0][jv][0]);
            hp[0] = __float2half_rn(il0 * __half2float(hp[0]));
            hp[1] = __float2half_rn(il0 * __half2float(hp[1]));
            hp[2] = __float2half_rn(il1 * __half2float(hp[2]));
            hp[3] = __float2half_rn(il1 * __half2float(hp[3]));
        }
    }

    // Warp Shuffle Collective Store
    #pragma unroll
    for (int jv = 0; jv < WHDV; ++jv) {
        uint32_t *Z0 = reinterpret_cast<uint32_t *>(&R_Q[0]);
        uint32_t *Z1 = reinterpret_cast<uint32_t *>(&R_K[0]);
        Z0[0] = R_D[0][jv][0]; Z1[0] = R_D[0][jv][1];
        Z0[1] = __shfl_sync(0xffffffff, R_D[0][jv][0], lid + 1, 4);
        Z0[2] = __shfl_sync(0xffffffff, R_D[0][jv][0], lid + 2, 4);
        Z0[3] = __shfl_sync(0xffffffff, R_D[0][jv][0], lid + 3, 4);
        Z1[1] = __shfl_sync(0xffffffff, R_D[0][jv][1], lid + 1, 4);
        Z1[2] = __shfl_sync(0xffffffff, R_D[0][jv][1], lid + 2, 4);
        Z1[3] = __shfl_sync(0xffffffff, R_D[0][jv][1], lid + 3, 4);

        if (lid % 4 == 0) {
            int o_br = wQP * (kMmaAtomM * kWarpTileSeqLenP);
            int o_d  = wKV * (kMmaAtomN * WHDV) + jv * kMmaAtomN;
            int gr0 = Tr * Br + o_br + lid / 4;
            int gr8 = Tr * Br + o_br + lid / 4 + 8;
            if (gr0 < QKV_seqlen)
                LDST128BITS(O[O_ofs + (o_br + lid / 4) * stride_Q + o_d]) = LDST128BITS(Z0[0]);
            if (gr8 < QKV_seqlen)
                LDST128BITS(O[O_ofs + (o_br + lid / 4 + 8) * stride_Q + o_d]) = LDST128BITS(Z1[0]);
        }
    }
}
