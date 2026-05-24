# Qwen-1.8B CUDA Kernels Unit Tests

This subfolder contains mathematically rigorous correctness and accuracy unit tests for all 13 core CUDA kernels powering the high-performance Qwen-1.8B inference engine.

## Test Coverage & Strategy

The tests compare GPU half-precision execution outputs with high-precision CPU reference calculations to compute both the **Maximum Absolute Error** and **Mean Absolute Error**. The following components are covered:

1. **RMSNorm (`rmsnorm.cuh`)**: Normalizes row vectors via root-mean-square and applies scale weights.
2. **Bias Addition (`bias.cuh`)**: In-place element-wise bias addition.
3. **Embedding Lookup (`embedding.cuh`)**: Token ID vector lookups in half-precision embeddings.
4. **SwiGLU Activation (`swiglu.cuh`)**: High-precision $SiLU(x) \times y$ computation.
5. **RoPE Rotation (`rope.cuh`)**: Rotary Position Embedding rotation on key/query state.
6. **Weight Transpose (`transpose.cuh`)**: Matrix transposition optimized for GPU weights.
7. **Temperature Sampler (`sampler.cuh`)**: Temperature-scaled softmax cumulative probability sweep.
8. **ArgMax reduction (`argmax.cuh`)**: Block-wide warp-shuffled reduction to find the maximum logit index.
9. **Fused QKV/RoPE/Cache (`fused_qkv_rope_cache.cuh`)**: Unified state extraction and caching.
10. **Baseline Flash Decoding (`flash_decode_q1.cuh`)**: Split-key-value baseline attention decoders.
11. **Optimized Flash Decoding (`flash_decode_q1_opt.cuh`)**: Single-block fused & split-stage optimized decoders.
12. **HGEMM Variants (`hgemm.cuh` & `hgemm_final.cuh`)**: Matrix multiplication benchmarks covering Naive, Tiled, Async, Final, and Tensor Core optimized kernels.

---

## How to Compile & Run

### Prerequisites
1. **NVIDIA GPU** supporting CUDA (Compute Capability 8.0 or higher is required for Tensor Core HGEMM; GeForce RTX 4060 Ti SM 8.9 is fully supported).
2. **CUDA Toolkit** (nvcc version 12.0 or higher).
3. **MSVC Host Compiler** (`cl.exe`).

### Compilation Command

If the Visual Studio C++ compiler (`cl.exe`) is not in your system environment path, you can specify it using the `-ccbin` flag in `nvcc`:

```powershell
nvcc -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64" -O3 -arch=sm_89 run_tests.cu -o run_tests.exe
```

### Running the Tests

Simply execute the compiled binary to view the comprehensive test report:

```powershell
.\run_tests.exe
```

All 12 major test suites will run sequentially, outputting their precision differences and validation status.
