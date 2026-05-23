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

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
// gmem -> smem
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
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
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
#define LDMATRIX_X1_T(R, addr)                                                 \
  asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"  \
               : "=r"(R)                                                       \
               : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr)                                            \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"       \
      : "=r"(R0), "=r"(R1)                                                     \
      : "r"(addr))
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
    const int kMmaAtomM,         // MMA Atom M, 16
    const int kMmaAtomN,         // MMA Atom N, 8
    const int kMmaAtomK,         // MMA Atom K, 16
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
    constexpr int NumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_mma_41warp_18mma_kernel(half *Q, half *K, half *V,
                                half *O, int QKV_seqlen,
                                int QKV_head) {
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
constexpr int kNumThreads_fg = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
__global__ void __launch_bounds__(kNumThreads_fg)
    flash_attn_finegrained_qk_tiling_kernel(half *Q, half *K, half *V,
                                            half *O, int QKV_seqlen,
                                            int QKV_head) {
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
constexpr int kNumThreads_rp = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
__global__ void __launch_bounds__(kNumThreads_rp)
    flash_attn_register_p_kernel(half *Q, half *K, half *V,
                                 half *O, int QKV_seqlen,
                                 int QKV_head) {
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
