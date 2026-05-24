import os
import sys
import json
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "model_weights/qwen_1.8b_chat/qwen/Qwen-1_8B-Chat"))
ENGINE_EXE = os.path.abspath(os.path.join(BASE_DIR, "framework_src/qwen_infer.exe"))
MODEL_BIN = os.path.abspath(os.path.join(BASE_DIR, "model_weights/qwen_1.8b.bin"))

def main():
    print(f"[TEST] Launching engine: {ENGINE_EXE}...")
    proc = subprocess.Popen(
        [ENGINE_EXE, MODEL_BIN],
        cwd=os.path.dirname(ENGINE_EXE),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Read until [ENGINE_READY]
    while True:
        line = proc.stdout.readline()
        if not line:
            print("[ERROR] Engine exited during startup.")
            sys.exit(1)
        line = line.strip()
        print(f"Engine Log: {line}")
        if "[ENGINE_READY]" in line:
            break

    # Now load tokenizer (lazy import to prevent DLL pollution before subprocess starts)
    print("[TEST] Loading tokenizer...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print("[TEST] Tokenizer loaded.")

    messages = [{"role": "user", "content": "你好，请用中文介绍一下你自己"}]
    formatted_prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted_prompt += "<|im_start|>assistant\n"

    token_ids = tok.encode(formatted_prompt)
    token_str = " ".join(str(tid) for tid in token_ids)
    print(f"[TEST] Encoded prompt tokens: {token_str}")

    # Send reset
    print("[TEST] Sending reset...")
    proc.stdin.write("reset\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        if "[RESET_DONE]" in line:
            break

    # Send prompt
    print("[TEST] Sending prompt...")
    proc.stdin.write(token_str + "\n")
    proc.stdin.flush()

    # Read output
    generated_tokens = []
    started = False
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if "[GENERATION_START]" in line:
            started = True
            line = line.replace("[GENERATION_START]", "").strip()
        if "[GENERATION_END]" in line:
            line = line.replace("[GENERATION_END]", "").strip()
            if line:
                tokens = [int(t) for t in line.split() if t.isdigit()]
                generated_tokens.extend(tokens)
            break

        if started and line:
            tokens = [int(t) for t in line.split() if t.isdigit()]
            generated_tokens.extend(tokens)
            # Print decoded chunk
            for t in tokens:
                print(f"Token ID: {t} -> Decoded: {repr(tok.decode([t]))}")

    print("\n\n[TEST] Generation finished.")
    print(f"Decoded Full Response:\n{tok.decode(generated_tokens)}")
    print(f"Raw Token IDs: {generated_tokens}")

    proc.terminate()

if __name__ == "__main__":
    main()
