#!/usr/bin/env python3
"""Download Qwen-1.8B-Chat model from HuggingFace or ModelScope."""

import os
import sys
import argparse

MODEL_ID = "Qwen/Qwen-1_8B-Chat"
MODELSCOPE_ID = "qwen/Qwen-1_8B-Chat"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "qwen_1.8b_chat")


def download_huggingface(output_dir):
    from huggingface_hub import snapshot_download
    print(f"[DOWNLOAD] From HuggingFace: {MODEL_ID}")
    snapshot_download(repo_id=MODEL_ID, local_dir=output_dir,
                      local_dir_use_symlinks=False, resume_download=True,
                      ignore_patterns=["*.msgpack", "*.h5", "*.bin", "*.pt"])
    print(f"[OK] Downloaded to {output_dir}")


def download_modelscope(output_dir):
    from modelscope import snapshot_download
    print(f"[DOWNLOAD] From ModelScope: {MODELSCOPE_ID}")
    snapshot_download(MODELSCOPE_ID, cache_dir=output_dir,
                      local_dir=output_dir)
    print(f"[OK] Downloaded to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download Qwen-1.8B-Chat model")
    parser.add_argument("--source", choices=["hf", "ms", "auto"], default="auto",
                        help="Download source: hf (HuggingFace), ms (ModelScope), auto (try both)")
    parser.add_argument("--output", default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    if os.path.exists(args.output) and os.listdir(args.output):
        print(f"[SKIP] Model already exists at {args.output}")
        return

    os.makedirs(args.output, exist_ok=True)

    if args.source == "auto":
        try:
            download_huggingface(args.output)
            return
        except Exception as e:
            print(f"[WARN] HuggingFace failed: {e}")
            print("[INFO] Trying ModelScope fallback...")
            download_modelscope(args.output)
            return
    elif args.source == "hf":
        download_huggingface(args.output)
    else:
        download_modelscope(args.output)


if __name__ == "__main__":
    main()
