import os
import subprocess
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_EXE = os.path.abspath(os.path.join(BASE_DIR, "framework_src/qwen_infer.exe"))
MODEL_BIN = os.path.abspath(os.path.join(BASE_DIR, "model_weights/qwen_1.8b.bin"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "model_weights/qwen_1.8b_chat/qwen/Qwen-1_8B-Chat"))

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

    # Send reset
    proc.stdin.write("reset\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line or "[RESET_DONE]" in line:
            break

    # Send first prompt
    prompt1_tokens = tok.encode("<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n")
    token_str = " ".join(str(tid) for tid in prompt1_tokens)
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
            generated_tokens.append(int(line))

    proc.terminate()

    print("\nGenerated Token IDs in Turn 1:", generated_tokens)
    resp_text = tok.decode(generated_tokens)
    print("Decoded Turn 1 response text:", repr(resp_text))

    # Let's compare the generated token IDs with what hardcoded "你好！有什么我可以帮助你的吗？" produces!
    hardcoded_text = "你好！有什么我可以帮助你的吗？"
    hardcoded_tokens = tok.encode(hardcoded_text)
    print("\nHardcoded Text Token IDs:", hardcoded_tokens)
    print("Decoded Hardcoded Tokens:", repr(tok.decode(hardcoded_tokens)))

    # Let's check the second turn prompt if we format it using resp_text:
    prompt2_formatted = f"<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n{resp_text}<|im_end|>\n<|im_start|>user\n介绍一下你自己<|im_end|>\n<|im_start|>assistant\n"
    tokens_formatted = tok.encode(prompt2_formatted)
    print("\nTokens produced by formatting Turn 1 response:", tokens_formatted)

    prompt2_hardcoded = f"<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n{hardcoded_text}<|im_end|>\n<|im_start|>user\n介绍一下你自己<|im_end|>\n<|im_start|>assistant\n"
    tokens_hardcoded = tok.encode(prompt2_hardcoded)
    print("Tokens produced by hardcoded text:", tokens_hardcoded)

    print("\nAre they identical?", tokens_formatted == tokens_hardcoded)

if __name__ == "__main__":
    main()
