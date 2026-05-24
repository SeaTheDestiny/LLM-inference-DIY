/**
 * flash_attn_causal.cuh — Flash Attention with Causal Mask for Prefill
 * =====================================================================
 * Based on tuned_A kernel from flash_attn.cu.
 * Adds causal masking: token i only attends to tokens 0..i.
 *
 * The ONLY change: insert a mask block between S=Q@K^T and softmax,
 * setting S[i][k] = -inf when global K row > global Q row.
 */

#pragma once
#include "flash_attn.cu"

template <
    const int kHeadDim,
    const int kMmaAtomM = 16,
    const int kMmaAtomN = 8,
    const int kMmaAtomK = 16,
    const int kMmaTileSeqLenQ = 8,
    const int kMmaTileSeqLenK = 1,
    const int kWarpTileSeqLenQ = 1,
    const int kWarpTileSeqLenK = 16,
    const int kWarpTileSeqLenP = 1,
    const int kWarpTileHeadDimV = 0,
    const int kPadQ = 8,
    const int kPadK = 8,
    const int kPadV = 8,
    const int kStage = 2,
    const int kOStorageAccF32 = 1
    >
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    flash_attn_causal_kernel(half *Q, half *K, half *V,
                              half *O, int QKV_seqlen,
                              int QKV_head) {
    constexpr int kThrA = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
    constexpr int WHDV = (kWarpTileHeadDimV == 0) ? (kHeadDim / kMmaAtomN) : kWarpTileHeadDimV;
    const int Tc = div_ceil(QKV_seqlen, Bc);
    const int Tr = blockIdx.y;
    int K_ofs = QKV_head * QKV_seqlen * kHeadDim;
    int O_ofs = QKV_head * QKV_seqlen * kHeadDim + Tr * Br * kHeadDim;

    constexpr int Q_sz = Br * (kMmaAtomK + kPadQ);
    constexpr int K_sz = Bc * (kMmaAtomK + kPadK);
    constexpr int V_sz = Bc * (kMmaAtomN * 2 + kPadV);
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
    int wQP = wid / kMmaTileSeqLenK;
    int wKV = wid % kMmaTileSeqLenK;

    int ldQ_r = tid / (kThrA / Br);
    int ldQ_c = (tid % (kThrA / Br)) * (kMmaAtomK / (kThrA / Br));
    int ldK_r = tid / (kThrA / Bc);
    int ldK_c = (tid % (kThrA / Bc)) * (kMmaAtomK / (kThrA / Bc));
    int ldV_r = tid / (kThrA / Bc);
    int ldV_c = (tid % (kThrA / Bc)) * (kMmaAtomN * 2 / (kThrA / Bc));

    // Registers
    float rm_old[2] = {-INFINITY, -INFINITY};
    float rl_old[2] = {0.0f, 0.0f};
    uint32_t R_S[2][kWarpTileSeqLenK][2];
    uint32_t R_K[2][2];
    uint32_t R_V[4];
    uint32_t R_D[2][WHDV][2];
    #pragma unroll
    for (int jv = 0; jv < WHDV; jv++) { R_D[0][jv][0] = 0; R_D[0][jv][1] = 0; }

    // Q prefetch
    if (Tr * Br + ldQ_r < QKV_seqlen) {
        uint32_t sp = Q_base + (ldQ_r * (kMmaAtomK + kPadQ) + ldQ_c) * sizeof(half);
        CP_ASYNC_CG(sp, &Q[K_ofs + (Tr * Br + ldQ_r) * kHeadDim + ldQ_c], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();

    // Load Q into registers
    uint32_t R_Q[2 * kMmaTileSeqLenQ][4];
    #pragma unroll
    for (int wq = 0; wq < kMmaTileSeqLenQ; wq++) {
        int row = wq * kMmaAtomM + lid % kMmaAtomM;
        int col = (lid / kMmaAtomM) * kMmaAtomK;
        uint32_t qp = Q_base + (row * (kMmaAtomK + kPadQ) + col) * sizeof(half);
        LDMATRIX_X4(R_Q[wq][0], R_Q[wq][1], R_Q[wq][2], R_Q[wq][3], qp);
    }

    float scale = rsqrtf((float)kHeadDim);

    // Main K-tile loop
    #pragma unroll 1
    for (int j = 0; j < Tc; j++) {
        // Prefetch K for next tile
        if (j + 1 < Tc) {
            if (ldK_r < QKV_seqlen) {
                uint32_t sp = K_base + ((j + 1) % kStage * K_sz + ldK_r * (kMmaAtomK + kPadK) + ldK_c) * sizeof(half);
                CP_ASYNC_CG(sp, &K[K_ofs + j * Bc * kHeadDim + ldK_r * kHeadDim + ldK_c], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
        }
        CP_ASYNC_WAIT_GROUP(kStage > 1 ? 1 : 0);
        __syncthreads();

        // Q@K^T via MMA
        #pragma unroll
        for (int dt = 0; dt < kHeadDim / kMmaAtomK; dt++) {
            if (dt > 0) { CP_ASYNC_WAIT_GROUP(0); __syncthreads(); }
            #pragma unroll
            for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
                uint32_t kp = K_base + (j % kStage * K_sz + kt * kMmaAtomN * (kMmaAtomK + kPadK) + (lid % kMmaAtomN) * kMmaAtomK) * sizeof(half);
                LDMATRIX_X2_T(R_K[0], R_K[1], kp);
                HMMA16816(R_S[0][kt][0], R_S[0][kt][1],
                          R_Q[wQP][0], R_Q[wQP][1], R_Q[wQP][2], R_Q[wQP][3],
                          R_K[0], R_K[1],
                          (dt == 0) ? 0 : R_S[0][kt][0], (dt == 0) ? 0 : R_S[0][kt][1]);
            }
            if (dt + 1 < kHeadDim / kMmaAtomK) {
                CP_ASYNC_WAIT_GROUP(0);
                __syncthreads();
            }
        }

        // ============================================================
        // CAUSAL MASK: set S[i][k] = -inf when k > i
        // S[row][col] where row = Q index, col = K row index
        // ============================================================
        #pragma unroll
        for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
            half *hp0 = reinterpret_cast<half *>(&R_S[0][kt][0]);
            half *hp1 = reinterpret_cast<half *>(&R_S[0][kt][1]);
            int q0 = wQP * kMmaAtomM + lid / 4;        // rows 0..7
            int q1 = q0 + kMmaAtomM / 2;                // rows 8..15
            int k_base = j * Bc + kt * kMmaAtomN + (lid % 4) * 2;

            // Group A: rows 0-7
            if (k_base + 0 > q0)      hp0[0] = __float2half(-INFINITY);
            if (k_base + 1 > q0)      hp0[1] = __float2half(-INFINITY);
            // Group B: rows 8-15
            if (k_base + 0 > q1)      hp1[0] = __float2half(-INFINITY);
            if (k_base + 1 > q1)      hp1[1] = __float2half(-INFINITY);
        }

        // Online Safe Softmax (same as tuned_A)
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
                CP_ASYNC_CG(sp, &V[K_ofs + j * Bc * kHeadDim + ldV_r * kHeadDim + stg * (kMmaAtomN * 2) + ldV_c], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
        }
        CP_ASYNC_WAIT_GROUP(kStage - 2);
        __syncthreads();

        // P@V with fine-grained V
        #pragma unroll
        for (int dt = 0; dt < WHDV; dt++) {
            if (dt > 0) { CP_ASYNC_WAIT_GROUP(0); __syncthreads(); }
            #pragma unroll
            for (int kt = 0; kt < kWarpTileSeqLenK; kt++) {
                int v_row = kt * kMmaAtomN + lid % kMmaAtomN;
                int v_col = dt * kMmaAtomN * 2;
                uint32_t vp = V_base + (j % kStage * V_sz + v_row * (kMmaAtomN * 2 + kPadV) + v_col) * sizeof(half);
                LDMATRIX_X2_T(R_V[0], R_V[1], vp);
                HMMA16816(R_D[0][dt][0], R_D[0][dt][1],
                          R_S[0][kt][0], R_S[0][kt][1],
                          R_S[0][kt][0], R_S[0][kt][1],
                          R_V[0], R_V[1],
                          (dt == 0 && kt == 0) ? 0 : R_D[0][dt][0],
                          (dt == 0 && kt == 0) ? 0 : R_D[0][dt][1]);
            }
            if (dt + 1 < WHDV) {
                CP_ASYNC_WAIT_GROUP(0);
                __syncthreads();
            }
        }

        // Rescale old accumulator
        #pragma unroll
        for (int jv = 0; jv < WHDV; ++jv) {
            half *hp = reinterpret_cast<half *>(&R_D[0][jv][0]);
            float il0 = resc0 * rl_old[0], il1 = resc1 * rl_old[1];
            hp[0] = __float2half_rn(il0 * __half2float(hp[0]));
            hp[1] = __float2half_rn(il0 * __half2float(hp[1]));
            hp[2] = __float2half_rn(il1 * __half2float(hp[2]));
            hp[3] = __float2half_rn(il1 * __half2float(hp[3]));
        }
        rm_old[0] = mn0; rm_old[1] = mn1;
        rl_old[0] = rl_new[0] * resc0 + rl_old[0];
        rl_old[1] = rl_new[1] * resc1 + rl_old[1];
    }

    // Final normalization
    #pragma unroll
    for (int jv = 0; jv < WHDV; ++jv) {
        half *hp = reinterpret_cast<half *>(&R_D[0][jv][0]);
        float il0 = __frcp_rn(rl_old[0]), il1 = __frcp_rn(rl_old[1]);
        hp[0] = __float2half_rn(il0 * __half2float(hp[0]));
        hp[1] = __float2half_rn(il0 * __half2float(hp[1]));
        hp[2] = __float2half_rn(il1 * __half2float(hp[2]));
        hp[3] = __float2half_rn(il1 * __half2float(hp[3]));
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
                LDST128BITS(O[O_ofs + (o_br + lid / 4) * kHeadDim + o_d]) = LDST128BITS(Z0[0]);
            if (gr8 < QKV_seqlen)
                LDST128BITS(O[O_ofs + (o_br + lid / 4 + 8) * kHeadDim + o_d]) = LDST128BITS(Z1[0]);
        }
    }
}
