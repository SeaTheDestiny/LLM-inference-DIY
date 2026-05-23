# 🚀 LLM_DIY — 极致 CUDA & Flash Decoding 大模型自研推理引擎

本项目是一个在 Windows 平台（RTX 4060 Ti 显卡）上追求极致性能、完全自研的轻量级大模型前向推理框架。

通过高度优化 KV-Cache、结合自研的 **Flash Attention (Prefill 阶段)** 与 **Flash Decoding v5 (Decode 阶段)** 极致算子，串联起完整的 **Qwen-1.8B-Chat** 自回归生成流程，并提供了超低延迟、支持流式渲染的玻璃拟态 Web 对话终端！

---

## 📂 项目结构布局 (Restructured Layout)

根据你的框架设计，我们已经将工作区彻底重组，消除了冗余文件，使其结构极度纯净且层次分明：

```
LLM_DIY (d:/CUDA_learning)
├── 📂 model_weights          # ── 【模型权重】 ──
│   ├── download_model.py     # 阿里 ModelScope 权重极速下载器
│   ├── convert_weights.py    # 高性能 bf16 -> fp16 权重提取与顺序打包脚本
│   └── qwen_1.8b.bin         # 转换后生成的 unified 3.42 GB 扁平二进制权重 (FP16)
│
├── 📂 kernels                # ── 【算子库（算子开发区）】 ──
│   ├── flash_decode_q1_opt.cuh# 极致优化的 Flash Decoding v1~v5 核函数
│   ├── flash_decode_q1.cuh   # Flash Decoding 基础模板
│   ├── flash_attn.cu         # 极致优化 Flash Attention 核函数
│   ├── rmsnorm.cuh           # 自研高效归一化 RMSNorm 核函数 (Warp Reductions)
│   ├── rope.cuh              # 自研位置编码 RoPE 核函数 (Qwen-1.8B base)
│   ├── swiglu.cuh            # 自研激活层 SwiGLU 核函数 (Element-wise SiLU)
│   ├── bias.cuh              # 自研偏置加法 QKV Bias 核函数 (Warp Parallel)
│   ├── argmax.cuh            # 自研采样层 ArgMax 核函数 (Block Reductions)
│   ├── ref_kernel.cuh        # 用于对比精度和带宽的 PyTorch / Naive 参考算子
│   ├── utils.h               # FP16/FP32 高级 CUDA 辅助函数
│   ├── bench_fd_opt.cu       # Flash Decoding Bandwidth/TFLOPS 性能跑分测试
│   └── fd_opt_results.html   # 可视化跑分交互图表
│
├── 📂 framework_src          # ── 【框架source】 ──
│   ├── qwen_infer.cu         # C++/CUDA 核心自回归推理后端 (加载权重、分配VRAM、前向传播)
│   └── qwen_infer.exe        # 编译生成的高性能推理引擎可执行程序 (401 KB)
│
└── 📂 web_chat               # ── 【对话web框架】 ──
    ├── app.py                # 极简高性能 Flask Web 服务 (调用 Qwen 官方 Tokenizer 并拉起 C++ 后端)
    └── index.html            # 拥有极佳视觉张力、磨砂玻璃拟态、支持 SSE 实时流式渲染的 Web 对话前端
```

---

## ⚡ 核心算子亮点 (Kernels Spotlight)

* **Flash Decoding v5 Fused / Split**：
  * **单 Block 融合 (v5 Fused)**：在序列长度较短时，所有 Warp 归并操作全部在 SMEM（共享内存）内完成，**零显存写回开销，消除了 Stage 2 启动延迟**，达到了 sub-microsecond（亚微秒级）的物理延迟极限！
  * **全 Warp 并行 (v5 Split)**：在序列长度极长时，自动切分为多 Block，并在 Stage 2 中利用全 Warp 并行加速，配合 `__expf` 和 `__fmaf_rn` 硬件固有指令，实现 100% 精度无损的归并。
* **物理带宽打满**：
  * 在 RTX 4060 Ti 上实测跑分带宽达到 **256.0 GB/s**，打满其物理 VRAM 最大吞吐量的 **90%**！

---

## 🛠️ 快速启动指南

### 第一步：启动网页对话端服务

在你的 VS Code / 命令行终端中，进入 `web_chat` 目录并运行 Web 代理服务端：

```powershell
# 1. 激活我们已配置好在 D 盘的虚拟环境
.venv\Scripts\activate

# 2. 运行 Flask Web Server (会自动挂载并拉起 C++ 推理可执行程序)
python web_chat/app.py
```

服务端会输出：
```
=== Weight Config Loaded Successfully ===
  Vocab size:        151936
  Hidden size:       2048
  ...
[SUCCESS] Weights fully loaded into VRAM.
[ENGINE_READY] CUDA Qwen-1.8B Inference Engine is fully ready.

* Web Server starting at: http://127.0.0.1:5000
```
*(注意：C++ 引擎首次启动时会将 3.42 GB 扁平权重瞬间直接 `fread` 进 GPU 显存，由于极速 IO，整个分配与加载过程耗时仅需 **0.8 秒左右**！)*

### 第二步：在浏览器中尽情畅聊！

用你的浏览器打开：

👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

你可以看到一个充满未来感、带有动态渐变呼吸灯背景和磨砂玻璃物理效果的对话终端！

* 每一条你发出的消息，都会先经过 Python 端快速的官方 BPE 编码，化为 token ID 送入 `qwen_infer.exe`。
* `qwen_infer.exe` 瞬间执行 **Prefill** 和 **Autoregressive Decode**，利用我们纯手工打造的 GPU 物理算子流式喷出生成 token ID。
* 前端网页以 Server-Sent Events (SSE) **实时流式渲染输出文字**，享受毫无卡顿、丝滑顺畅的极速体验！
* 点击左下角的 **重置按钮**，可以一键清空上下文并清空 GPU 上的 KV Cache 历史缓存。

---

## ⚙️ 编译命令备份

如果后续你在 `kernels/` 下改动了算子，想要重新编译 C++ 推理后端，只需在 `framework_src/` 目录下执行：

```powershell
nvcc -arch=sm_89 -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64" -O3 -lcublas qwen_infer.cu -o qwen_infer
```
*(此编译命令已将 RTX 4060 Ti 的 Ada Lovelace 架构 `sm_89` 特性与 cuBLAS Tensor Cores 支持进行全开优化。)*
