#!/usr/bin/env python3
"""Convert HF Qwen-1.8B-Chat safetensors → custom .bin flat weights for DIY engine."""

import os
import sys
import json
import struct
import argparse
import numpy as np

# Match QwenConfig in framework_src/qwen_infer.cu
QWDF_MAGIC = 0x46445751


def find_safetensors_dir(search_path):
    """Find the directory containing .safetensors files."""
    path = os.path.abspath(search_path)
    for root, dirs, files in os.walk(path):
        safetensors = [f for f in files if f.endswith(".safetensors")]
        if safetensors and "config.json" in files:
            return root
    raise FileNotFoundError(
        f"No safetensors + config.json found under {search_path}. "
        f"Run download_model.py first."
    )


def load_config(safetensors_dir):
    """Load model config.json."""
    config_path = os.path.join(safetensors_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_all_weights(safetensors_dir):
    """Load all safetensors shards into a single numpy fp16 dict.
    Handles both float16 and bfloat16 safetensors natively (no torch needed)."""
    weights = {}
    for fname in sorted(os.listdir(safetensors_dir)):
        if not fname.endswith(".safetensors"):
            continue
        path = os.path.join(safetensors_dir, fname)
        with open(path, "rb") as f:
            data = f.read()
        # Parse header (first 8 bytes = header_len as u64 LE)
        header_len = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8:8 + header_len])
        offset = 8 + header_len
        for key in header:
            if key == "__metadata__":
                continue
            info = header[key]
            dtype = info["dtype"]
            shape = info["shape"]
            start = info["data_offsets"][0]
            end = info["data_offsets"][1]
            raw = data[offset + start:offset + end]
            if dtype == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
            elif dtype == "BF16":
                arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            elif dtype == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
            weights[key] = arr.astype(np.float16)
    return weights


def write_fp16(f, arr):
    """Write numpy array as fp16 bytes."""
    if arr.dtype == np.float16:
        f.write(arr.tobytes())
    elif arr.dtype == np.float32:
        f.write(arr.astype(np.float16).tobytes())
    else:
        f.write(arr.astype(np.float32).astype(np.float16).tobytes())


def convert(safetensors_dir, output_path, max_seqlen=32768):
    """Main conversion logic."""
    sd = find_safetensors_dir(safetensors_dir)
    print(f"[CONVERT] Using safetensors from: {sd}")

    config = load_config(sd)
    weights = load_all_weights(sd)

    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    # Qwen's config.json "intermediate_size" = SwiGLU total (w1+w2 combined).
    # Our engine expects the per-matrix dim (= half of the config value).
    intermediate_size = config.get(
        "intermediate_size", config.get("ffn_hidden_size", hidden_size * 4))
    if intermediate_size % 2 == 0 and intermediate_size > hidden_size * 2:
        intermediate_size = intermediate_size // 2

    print(f"\n  vocab_size={vocab_size}, hidden_size={hidden_size}")
    print(f"  num_layers={num_layers}, num_heads={num_heads}")
    print(f"  intermediate_size={intermediate_size}, max_seqlen={max_seqlen}")

    # Write .bin file
    with open(output_path, "wb") as f:
        # --- Header (28 bytes, little-endian) ---
        f.write(struct.pack("<IIIIIII",
                            QWDF_MAGIC,
                            vocab_size,
                            hidden_size,
                            num_layers,
                            num_heads,
                            intermediate_size,
                            max_seqlen))

        # Resolve HF tensor name → flat array (row-major fp16)
        def get_tensor(name):
            return weights[name].astype(np.float16)

        # --- wte ---
        print("[CONVERT] wte")
        wte = get_tensor("transformer.wte.weight")  # [vocab_size, hidden_size]
        assert wte.shape == (vocab_size, hidden_size), f"wte shape {wte.shape}"
        write_fp16(f, wte)

        # --- Per-layer ---
        for i in range(num_layers):
            prefix = f"transformer.h.{i}"
            print(f"[CONVERT] Layer {i}")

            # ln_1 [hidden_size]
            datas = get_tensor(f"{prefix}.ln_1.weight")
            write_fp16(f, datas)

            # qkv_w: fused c_attn.weight [3*hidden, hidden]
            datas = get_tensor(f"{prefix}.attn.c_attn.weight")
            assert datas.shape == (3 * hidden_size, hidden_size), \
                f"qkv_w shape {datas.shape}"
            write_fp16(f, datas)

            # qkv_b: c_attn.bias [3*hidden]
            datas = get_tensor(f"{prefix}.attn.c_attn.bias")
            assert datas.shape == (3 * hidden_size,), \
                f"qkv_b shape {datas.shape}"
            write_fp16(f, datas)

            # attn_proj_w: c_proj.weight [hidden, hidden]
            datas = get_tensor(f"{prefix}.attn.c_proj.weight")
            assert datas.shape == (hidden_size, hidden_size), \
                f"attn_proj_w shape {datas.shape}"
            write_fp16(f, datas)

            # ln_2 [hidden_size]
            datas = get_tensor(f"{prefix}.ln_2.weight")
            write_fp16(f, datas)

            # ffn_w1: mlp.w1.weight (gate) [intermediate_size, hidden]
            datas = get_tensor(f"{prefix}.mlp.w1.weight")
            assert datas.shape == (intermediate_size, hidden_size), \
                f"ffn_w1 shape {datas.shape}"
            write_fp16(f, datas)

            # ffn_w2: mlp.w2.weight (up) [intermediate_size, hidden]
            datas = get_tensor(f"{prefix}.mlp.w2.weight")
            assert datas.shape == (intermediate_size, hidden_size), \
                f"ffn_w2 shape {datas.shape}"
            write_fp16(f, datas)

            # ffn_proj_w: mlp.c_proj.weight [hidden, intermediate]
            datas = get_tensor(f"{prefix}.mlp.c_proj.weight")
            assert datas.shape == (hidden_size, intermediate_size), \
                f"ffn_proj_w shape {datas.shape}"
            write_fp16(f, datas)

        # --- ln_f ---
        print("[CONVERT] ln_f")
        datas = get_tensor("transformer.ln_f.weight")
        write_fp16(f, datas)

        # --- lm_head ---
        print("[CONVERT] lm_head")
        lm_head = get_tensor("lm_head.weight")  # [vocab_size, hidden_size]
        assert lm_head.shape == (vocab_size, hidden_size), \
            f"lm_head shape {lm_head.shape}"
        write_fp16(f, lm_head)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[OK] Wrote {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen-1.8B-Chat HF → custom .bin for DIY engine")
    parser.add_argument("--model-dir", default=None,
                        help="Directory containing HF model files "
                             "(default: model_weights/qwen_1.8b_chat)")
    parser.add_argument("--output", default=None,
                        help="Output .bin path (default: model_weights/qwen_1.8b.bin)")
    parser.add_argument("--max-seqlen", type=int, default=32768,
                        help="Max sequence length (default: 32768)")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    model_dir = args.model_dir or os.path.join(base, "qwen_1.8b_chat")
    output = args.output or os.path.join(base, "qwen_1.8b.bin")

    convert(model_dir, output, max_seqlen=args.max_seqlen)


if __name__ == "__main__":
    main()
