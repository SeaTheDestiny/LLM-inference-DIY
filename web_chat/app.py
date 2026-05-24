import os
import sys
import time
import json
import subprocess
import threading
from flask import Flask, request, Response, render_template, send_from_directory
from transformers import AutoTokenizer

app = Flask(__name__, static_folder=".", template_folder=".")

# 1. Path Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../model_weights/qwen_1.8b_chat/qwen/Qwen-1_8B-Chat"))
ENGINE_EXE = os.path.abspath(os.path.join(BASE_DIR, "../framework_src/qwen_infer.exe"))
MODEL_BIN = os.path.abspath(os.path.join(BASE_DIR, "../model_weights/qwen_1.8b.bin"))

# Global subprocess handler & lock for thread safety
engine_process = None
engine_lock = threading.Lock()
tokenizer = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        print("[WEB_SERVER] Loading Qwen official tokenizer from local weights directory...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        print("[WEB_SERVER] Tokenizer loaded successfully.")
    return tokenizer

def init_engine():
    global engine_process
    with engine_lock:
        if engine_process is not None:
            return
            
        print(f"[WEB_SERVER] Launching CUDA Inference Engine: {ENGINE_EXE}...")
        if not os.path.exists(ENGINE_EXE):
            print(f"[ERROR] Executable not found at {ENGINE_EXE}. Please compile framework_src first.")
            sys.exit(1)
        if not os.path.exists(MODEL_BIN):
            print(f"[ERROR] Packed model weights not found at {MODEL_BIN}. Please run weight converter first.")
            sys.exit(1)
            
        # Start qwen_infer subprocess in framework_src working dir
        engine_process = subprocess.Popen(
            [os.path.abspath(ENGINE_EXE), os.path.abspath(MODEL_BIN)],
            cwd=os.path.abspath(os.path.join(BASE_DIR, "../framework_src")),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Read stdout until [ENGINE_READY] is printed
        print("[WEB_SERVER] Waiting for CUDA GPU VRAM initialization and weights loading...")
        while True:
            line = engine_process.stdout.readline()
            if not line:
                # Subprocess exited unexpectedly
                stderr_output = engine_process.stderr.read()
                print(f"[ERROR] Engine exited during startup. Stderr:\n{stderr_output}")
                sys.exit(1)
            print(line.strip())
            if "[ENGINE_READY]" in line:
                break
                
        print("[WEB_SERVER] CUDA Inference Engine is connected and ready to stream!")

# Ensure engine is shut down when server exits
import atexit
@atexit.register
def cleanup():
    global engine_process
    if engine_process is not None:
        print("[WEB_SERVER] Shutting down C++/CUDA Inference Engine subprocess...")
        engine_process.terminate()
        engine_process = None

# 2. Main HTML Chat Interface Route
@app.route("/")
def index():
    return render_template("index.html")

# 3. Server Sent Events (SSE) Real-Time Token Generation Route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    
    tok = get_tokenizer()
    
    # 1. Format prompt with official Qwen template structure
    # Standard format: "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    formatted_prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    # End with assistant prompt start
    formatted_prompt += "<|im_start|>assistant\n"
    
    # 2. Encode to Token IDs
    token_ids = tok.encode(formatted_prompt)
    token_str = " ".join(str(tid) for tid in token_ids)
    print(f"[WEB_SERVER] Encoded prompt tokens ({len(token_ids)}): {token_str[:80]}...")
    
    # 3. Stream Generator with timing metrics
    def generate():
        global engine_process
        init_engine()
        prompt_len = len(token_ids)
        ctx_max = 8192

        with engine_lock:
            engine_process.stdin.write("reset\n")
            engine_process.stdin.flush()
            while True:
                line = engine_process.stdout.readline().strip()
                if "[RESET_DONE]" in line:
                    break

            t_start = time.perf_counter()
            t_first = None
            token_count = 0

            engine_process.stdin.write(token_str + "\n")
            engine_process.stdin.flush()

            while True:
                line = engine_process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                if line == "[GENERATION_START]":
                    continue
                if line == "[GENERATION_END]":
                    break

                # Each line is one token ID
                if line.isdigit() or (line.startswith('-') and line[1:].isdigit()):
                    t = int(line)
                    token_count += 1
                    if t_first is None:
                        t_first = time.perf_counter()
                    decoded = tok.decode([t])
                    elapsed = time.perf_counter() - t_start
                    tps = token_count / elapsed if elapsed > 0 else 0
                    ttf_ms = (t_first - t_start) * 1000 if t_first else 0
                    ctx_used = prompt_len + token_count
                    yield f"data: {json.dumps({'text':decoded, 'tokens':token_count, 'tps':round(tps,1), 'ttf_ms':round(ttf_ms,1), 'ctx_used':ctx_used, 'ctx_max':ctx_max})}\n\n"

            yield "data: [DONE]\n\n"
            
    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    # Pre-initialize on startup
    init_engine()
    print("\n* Web Server starting at: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
