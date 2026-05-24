# -*- coding: utf-8 -*-
import os
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "model_weights/qwen_1.8b_chat/qwen/Qwen-1_8B-Chat"))

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    text1 = "你好！有什么我可以帮助你的吗？"
    encoded = tok.encode(text1)
    print("Real Encoded:", encoded)
    decoded = tok.decode(encoded)
    print("Real Decoded back:", repr(decoded))

if __name__ == "__main__":
    main()
