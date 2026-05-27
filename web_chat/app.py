import os
import sys
import time
import json
import subprocess
import threading
from flask import Flask, request, Response, render_template, jsonify

app = Flask(__name__, static_folder=".", template_folder=".")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.getenv(
    "QWEN_MODEL_DIR",
    os.path.join(BASE_DIR, "../model_weights/qwen_1.8b_chat")
))
ENGINE_EXE = os.path.abspath(os.getenv(
    "QWEN_ENGINE_EXE",
    os.path.join(BASE_DIR, "../framework_src/qwen_infer")
))
MODEL_BIN = os.path.abspath(os.getenv(
    "QWEN_MODEL_BIN",
    os.path.join(BASE_DIR, "../model_weights/qwen_1.8b.bin")
))

engine_process = None
engine_lock = threading.RLock()
tokenizer = None


def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        print("[WEB_SERVER] Loading Qwen official tokenizer from local weights directory...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        print("[WEB_SERVER] Tokenizer loaded successfully.")
    return tokenizer


def _engine_alive():
    return engine_process is not None and engine_process.poll() is None


def init_engine():
    global engine_process
    with engine_lock:
        if _engine_alive():
            return
        engine_process = None

        print(f"[WEB_SERVER] Launching CUDA Inference Engine: {ENGINE_EXE}...")
        if not os.path.exists(ENGINE_EXE):
            print(f"[ERROR] Executable not found at {ENGINE_EXE}. Compile framework_src first.")
            sys.exit(1)
        if not os.path.exists(MODEL_BIN):
            print(f"[ERROR] Packed model weights not found at {MODEL_BIN}. Run convert_weights.py first.")
            sys.exit(1)

        engine_process = subprocess.Popen(
            [ENGINE_EXE, MODEL_BIN],
            cwd=os.path.join(BASE_DIR, "../framework_src"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        print("[WEB_SERVER] Waiting for CUDA GPU VRAM initialization and weights loading...")
        while True:
            line = engine_process.stdout.readline()
            if not line:
                print("[ERROR] Engine exited unexpectedly during startup.")
                sys.exit(1)
            line = line.strip()
            if line:
                print(line)
            if "[ENGINE_READY]" in line:
                break

        print("[WEB_SERVER] CUDA Inference Engine is connected and ready to stream!")


def _read_until(stdout, marker, label):
    """Read lines from stdout until marker is found.  Returns True on success."""
    while True:
        line = stdout.readline()
        if not line:
            print(f"[ERROR] Engine exited while waiting for {label}")
            return False
        line = line.strip()
        if line and marker in line:
            return True


def _send_command(cmd):
    """Write a command to engine stdin.  Returns True if write succeeded."""
    try:
        engine_process.stdin.write(cmd + "\n")
        engine_process.stdin.flush()
        return True
    except (OSError, ValueError):
        return False


def reset_engine():
    init_engine()
    with engine_lock:
        if not _engine_alive() or engine_process.stdin is None or engine_process.stdout is None:
            raise RuntimeError("CUDA inference engine is not ready for reset")
        if not _send_command("reset"):
            raise RuntimeError("Engine stdin closed")
        if not _read_until(engine_process.stdout, "[RESET_DONE]", "RESET_DONE"):
            raise RuntimeError("CUDA inference engine exited during reset")


import atexit


@atexit.register
def cleanup():
    global engine_process
    if engine_process is not None and engine_process.poll() is None:
        print("[WEB_SERVER] Shutting down C++/CUDA Inference Engine subprocess...")
        engine_process.terminate()
        try:
            engine_process.wait(timeout=5)
        except Exception:
            engine_process.kill()
        engine_process = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    try:
        reset_engine()
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])

    tok = get_tokenizer()
    init_engine()

    formatted_prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted_prompt += "<|im_start|>assistant\n"

    token_ids = tok.encode(formatted_prompt)
    token_str = " ".join(str(tid) for tid in token_ids)
    print(f"[WEB_SERVER] Encoded prompt tokens ({len(token_ids)}): {token_str[:80]}...")

    def generate():
        global engine_process
        prompt_len = len(token_ids)
        ctx_max = 8192

        with engine_lock:
            if not _engine_alive() or engine_process.stdin is None:
                yield "data: {\"error\":\"engine not running\"}\n\n"
                return

            if not _send_command("reset"):
                yield "data: {\"error\":\"engine stdin closed\"}\n\n"
                return
            if not _read_until(engine_process.stdout, "[RESET_DONE]", "RESET_DONE"):
                yield "data: {\"error\":\"engine exited during reset\"}\n\n"
                return

            t_start = time.perf_counter()
            t_first = None
            token_count = 0
            all_tokens = []

            if not _send_command(token_str):
                yield "data: {\"error\":\"engine stdin closed\"}\n\n"
                return

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

                if line.isdigit() or (line.startswith('-') and line[1:].isdigit()):
                    t = int(line)
                    if t in (151643, 151645):
                        break
                    if t < 0 or t >= 151936:
                        continue
                    token_count += 1
                    if t_first is None:
                        t_first = time.perf_counter()
                    all_tokens.append(t)
                    try:
                        full_text = tok.decode(all_tokens)
                    except (OverflowError, ValueError, UnicodeDecodeError):
                        full_text = "[?]"
                    elapsed = time.perf_counter() - t_start
                    tps = token_count / elapsed if elapsed > 0 else 0
                    ttf_ms = (t_first - t_start) * 1000 if t_first else 0
                    ctx_used = prompt_len + token_count
                    yield f"data: {json.dumps({'text': full_text, 'tokens': token_count, 'tps': round(tps, 1), 'ttf_ms': round(ttf_ms, 1), 'ctx_used': ctx_used, 'ctx_max': ctx_max})}\n\n"

            yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    init_engine()
    print("\n* Web Server starting at: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
