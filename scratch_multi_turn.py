import os
import subprocess
import sys
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_EXE = os.path.abspath(os.path.join(BASE_DIR, "framework_src/qwen_infer.exe"))
MODEL_BIN = os.path.abspath(os.path.join(BASE_DIR, "model_weights/qwen_1.8b.bin"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "model_weights/qwen_1.8b_chat/qwen/Qwen-1_8B-Chat"))

def run_prompt(proc, tok, messages):
    formatted_prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted_prompt += "<|im_start|>assistant\n"

    token_ids = tok.encode(formatted_prompt)
    token_str = " ".join(str(tid) for tid in token_ids)
    
    print(f"\n--- Sending Prompt ({len(token_ids)} tokens): {repr(formatted_prompt)} ---")
    
    # Send reset
    proc.stdin.write("reset\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line or "[RESET_DONE]" in line:
            break

    # Send prompt
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
            continue
        if "[GENERATION_END]" in line:
            break
        if started and line:
            t = int(line)
            generated_tokens.append(t)
            # print(f"Token: {t} -> {repr(tok.decode([t]))}")

    full_response = tok.decode(generated_tokens)
    print(f"Generated Response: {repr(full_response)}")
    return full_response

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
    print(f"[TEST] Launching engine...")
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
        if not line or "[ENGINE_READY]" in line:
            break

    # Turn 1
    messages = [{"role": "user", "content": "你好"}]
    resp1 = run_prompt(proc, tok, messages)

    # Turn 2
    messages.append({"role": "assistant", "content": resp1})
    messages.append({"role": "user", "content": "介绍一下你自己"})
    resp2 = run_prompt(proc, tok, messages)

    proc.terminate()

if __name__ == "__main__":
    main()
