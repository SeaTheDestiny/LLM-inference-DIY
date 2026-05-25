import os
import sys
import time
import json
import subprocess
import threading
from flask import Flask, request, Response, render_template

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
        from transformers import AutoTokenizer
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
        # Merge stderr → stdout to avoid pipe buffer deadlock:
        # The C++ engine writes extensive [DUMP]/[STEP_DUMP] debug output to stderr,
        # which would fill the pipe and block the subprocess if not consumed.
        engine_process = subprocess.Popen(
            [os.path.abspath(ENGINE_EXE), os.path.abspath(MODEL_BIN)],
            cwd=os.path.abspath(os.path.join(BASE_DIR, "../framework_src")),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Read stdout (which now includes merged stderr) until [ENGINE_READY] is printed
        print("[WEB_SERVER] Waiting for CUDA GPU VRAM initialization and weights loading...")
        while True:
            line = engine_process.stdout.readline()
            if not line:
                # Subprocess exited unexpectedly (stdout closed)
                print("[ERROR] Engine exited unexpectedly during startup. Check framework_src/stderr.txt for details.")
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
    
    # 1. Build prompt in TEXT-LEVEL ChatML format.
    #    Qwen's tiktoken tokenizer correctly maps <|im_start|> → 151644
    #    and <|im_end|> → 151645 as special tokens when encoding the full text.
    #    Token-level assembly with allowed_special=set() risks BPE mismatches.
    formatted_prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    # End with assistant prompt start (NO <|im_end|> — the model completes it)
    formatted_prompt += "<|im_start|>assistant\n"
    
    # 2. Encode the full prompt text → tokenizer handles special tokens natively
    token_ids = tok.encode(formatted_prompt)
    token_str = " ".join(str(tid) for tid in token_ids)
    print(f"[WEB_SERVER] Encoded prompt tokens ({len(token_ids)}): {token_str[:80]}...")
    
    # 3. Stream Generator with timing metrics
    def generate():
        global engine_process
        # RESTART engine fresh for each request to avoid GPU state corruption
        # that accumulates across multiple resets on the same engine instance.
        with engine_lock:
            if engine_process is not None:
                engine_process.stdin.write("exit\n")
                engine_process.stdin.flush()
                try: engine_process.wait(timeout=3)
                except: engine_process.terminate()
                engine_process = None
        init_engine()
        prompt_len = len(token_ids)
        ctx_max = 8192

        with engine_lock:
            t_start = time.perf_counter()
            t_first = None
            token_count = 0
            # Accumulate all generated tokens so we can decode them TOGETHER.
            # Decoding one token at a time (tok.decode([t])) corrupts multi-byte
            # characters (Chinese, emoji, etc.) that span multiple tokens —
            # each partial token becomes "�", contaminating the next turn's context.
            all_tokens = []

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
                    # Stop at ChatML end tokens per Qwen spec:
                    # 151645 = <|im_end|>, 151643 = <|endoftext|>
                    if t in (151643, 151645):
                        break
                    token_count += 1
                    if t_first is None:
                        t_first = time.perf_counter()
                    all_tokens.append(t)
                    # Decode ALL accumulated tokens together to preserve
                    # multi-byte character integrity across token boundaries.
                    full_text = tok.decode(all_tokens)
                    elapsed = time.perf_counter() - t_start
                    tps = token_count / elapsed if elapsed > 0 else 0
                    ttf_ms = (t_first - t_start) * 1000 if t_first else 0
                    ctx_used = prompt_len + token_count
                    yield f"data: {json.dumps({'text':full_text, 'tokens':token_count, 'tps':round(tps,1), 'ttf_ms':round(ttf_ms,1), 'ctx_used':ctx_used, 'ctx_max':ctx_max})}\n\n"

            yield "data: [DONE]\n\n"
            
    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    # Pre-initialize on startup
    init_engine()
    print("\n* Web Server starting at: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
