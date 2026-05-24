import os
import subprocess
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_EXE = os.path.abspath(os.path.join(BASE_DIR, "framework_src/qwen_infer.exe"))
MODEL_BIN = os.path.abspath(os.path.join(BASE_DIR, "model_weights/qwen_1.8b.bin"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "model_weights/qwen_1.8b_chat/qwen/Qwen-1_8B-Chat"))

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
    # 1. First Turn Prompt
    prompt1 = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
    tokens1 = tok.encode(prompt1)
    print("Turn 1 Prompt Tokens:", tokens1)

    # 2. Let's see what Qwen actually generates for "你好"!
    # We know from test_engine.py that Qwen generates:
    # 你好 -> [108386, 3837, 104198, 101919, 102661, 99718]
    # which is "你好，我是来自..."
    # But in scratch_multi_turn.py, it got "ãʲôҿ԰"
    # Let's decode "ãʲôҿ԰" back to bytes:
    # "ãʲôҿ԰" looks like a GBK decoding of a UTF-8 string!
    # Let's check!
    text = "你好，我是"
    utf8_bytes = text.encode("utf-8")
    print("UTF-8 bytes:", list(utf8_bytes))
    try:
        gbk_decoded = utf8_bytes.decode("gbk", errors="replace")
        print("GBK Decoded of UTF-8:", repr(gbk_decoded))
    except Exception as e:
        print("Error decoding:", e)

if __name__ == "__main__":
    main()
