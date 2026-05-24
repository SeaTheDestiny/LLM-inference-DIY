import os
import subprocess
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_EXE = os.path.abspath(os.path.join(BASE_DIR, "framework_src/qwen_infer.exe"))
MODEL_BIN = os.path.abspath(os.path.join(BASE_DIR, "model_weights/qwen_1.8b.bin"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "model_weights/qwen_1.8b_chat/qwen/Qwen-1_8B-Chat"))

def run_prompt(proc, tok, token_ids):
    token_str = " ".join(str(tid) for tid in token_ids)
    
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
        if "[DUMP]" in line or "[ERROR]" in line or "[STEP_DUMP]" in line:
            print(line)
            continue
        if "[GENERATION_START]" in line:
            started = True
            continue
        if "[GENERATION_END]" in line:
            break
        if started and line:
            generated_tokens.append(int(line))
            
    return generated_tokens

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
    print("[TEST] Launching engine...")
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

    # Prompt A: 29 tokens (Turn 2 history)
    prompt_A_text = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！有什么我可以帮助你的吗？<|im_end|>\n<|im_start|>user\n介绍一下你自己<|im_end|>\n<|im_start|>assistant\n"
    prompt_A_tokens = tok.encode(prompt_A_text)

    # Scenario 1: Double reset before sending prompt A
    print("\n--- Running Scenario 1: Double Reset ---")
    proc.stdin.write("reset\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line or "[RESET_DONE]" in line:
            break

    # Send Prompt A
    out_tokens_1 = run_prompt(proc, tok, prompt_A_tokens)
    print("Scenario 1 Output tokens length:", len(out_tokens_1))
    print("Scenario 1 Output text:", repr(tok.decode(out_tokens_1)))

    # Scenario 2: Run a small prompt, reset, then run prompt A
    print("\n--- Running Scenario 2: Query -> Reset -> Query ---")
    prompt_small = tok.encode("<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n")
    out_small = run_prompt(proc, tok, prompt_small)
    print("Small Prompt Output text:", repr(tok.decode(out_small)))

    # Now run Prompt A!
    out_tokens_2 = run_prompt(proc, tok, prompt_A_tokens)
    print("Scenario 2 Output tokens length:", len(out_tokens_2))
    print("Scenario 2 Output text:", repr(tok.decode(out_tokens_2)))

    # Write UTF-8 outputs to file
    with open("decoded_utf8.txt", "w", encoding="utf-8") as f_out:
        f_out.write("=== Scenario 1 Output ===\n")
        f_out.write(tok.decode(out_tokens_1) + "\n\n")
        f_out.write("=== Scenario 2 Small Prompt Output ===\n")
        f_out.write(tok.decode(out_small) + "\n\n")
        f_out.write("=== Scenario 2 Prompt A Output ===\n")
        f_out.write(tok.decode(out_tokens_2) + "\n\n")

    proc.terminate()

if __name__ == "__main__":
    main()
