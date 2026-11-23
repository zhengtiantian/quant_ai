#!/usr/bin/env python3
"""Colab L4 GPU worker: run llama.cpp with nomic-embed-text-v1.5.Q4_K_M.gguf matching LM Studio."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

CONTENT = Path("/content")
LLAMA_DIR = CONTENT / "llama_bin"
GGUF_PATH = CONTENT / "nomic-embed-text-v1.5.Q4_K_M.gguf"
GGUF_SHA256 = "d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac"

LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b12633/llama-b12633-bin-ubuntu-cuda-12.8-x64.tar.gz"
CUDART_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b12633/cudart-llama-b12633-bin-ubuntu-cuda-12.8-x64.tar.gz"
GGUF_URL = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf"

SERVER_PORT = 8081
SERVER_BASE = f"http://127.0.0.1:{SERVER_PORT}"
BATCH_SIZE = 32
MAX_WORKERS = 8

_THREAD_LOCAL = threading.local()


def get_session() -> requests.Session:
    """Return a per-thread reusable HTTP session."""
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        _THREAD_LOCAL.session = sess
    return sess


def sha256_file(path: Path) -> str:
    """Compute hex SHA256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(4 * 1024 * 1024)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    """Stream-download a URL to dest atomically."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def ensure_llama_and_model() -> tuple[Path, list[Path]]:
    """Ensure llama-server CUDA binaries and exact SHA256-verified Q4_K_M GGUF exist on Colab."""
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    server_bins = list(LLAMA_DIR.rglob("llama-server"))
    if not server_bins:
        for name, url in (("llama.tar.gz", LLAMA_URL), ("cudart.tar.gz", CUDART_URL)):
            arc = CONTENT / name
            print(f"Downloading {name} from {url}...", flush=True)
            download_file(url, arc)
            with tarfile.open(arc, "r:gz") as tf:
                tf.extractall(LLAMA_DIR)
            arc.unlink(missing_ok=True)
        server_bins = list(LLAMA_DIR.rglob("llama-server"))
        if not server_bins:
            raise SystemExit(f"llama-server not found under {LLAMA_DIR}")

    server_bin = sorted(server_bins, key=lambda p: len(p.parts))[0]
    server_bin.chmod(server_bin.stat().st_mode | 0o111)
    so_dirs = sorted({p.parent for p in LLAMA_DIR.rglob("*.so*")})

    if not GGUF_PATH.exists() or sha256_file(GGUF_PATH) != GGUF_SHA256:
        print(f"Downloading GGUF from {GGUF_URL}...", flush=True)
        download_file(GGUF_URL, GGUF_PATH)
        got = sha256_file(GGUF_PATH)
        if got != GGUF_SHA256:
            raise SystemExit(f"GGUF SHA256 mismatch: expected {GGUF_SHA256}, got {got}")
    print(f"Verified GGUF SHA256={GGUF_SHA256} and server={server_bin}", flush=True)
    return server_bin, so_dirs


def l2_norm(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize vectors in float32."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


def start_llama_server(server_bin: Path, so_dirs: list[Path]) -> subprocess.Popen:
    """Launch llama-server in embedding mode on GPU and wait until /health returns 200."""
    env = os.environ.copy()
    lib_paths = [str(d) for d in so_dirs]
    if env.get("LD_LIBRARY_PATH"):
        lib_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths)

    log_path = CONTENT / "llama_server.log"
    log_f = log_path.open("w", encoding="utf-8")
    cmd = [
        str(server_bin),
        "--model", str(GGUF_PATH),
        "--embedding",
        "--pooling", "mean",
        "--host", "127.0.0.1",
        "--port", str(SERVER_PORT),
        "--n-gpu-layers", "99",
        "--ctx-size", "16384",
        "--batch-size", "16384",
        "--ubatch-size", "16384",
        "--parallel", str(MAX_WORKERS),
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)

    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            log_f.flush()
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:])
            raise SystemExit(f"llama-server exited early with rc={rc}:\n{tail}")
        try:
            r = requests.get(f"{SERVER_BASE}/health", timeout=2)
            if r.status_code == 200:
                print(f"llama-server healthy on port {SERVER_PORT}", flush=True)
                return proc
        except requests.RequestException:
            pass
        time.sleep(0.5)

    proc.terminate()
    log_f.flush()
    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:])
    raise SystemExit(f"Timed out waiting for llama-server /health:\n{tail}")


def request_embeddings(texts: list[str], retries: int = 3) -> np.ndarray:
    """Call /v1/embeddings for a batch of texts, bisecting cleanly on HTTP 4xx/5xx."""
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    sess = get_session()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = sess.post(
                f"{SERVER_BASE}/v1/embeddings",
                json={"model": GGUF_PATH.name, "input": texts},
                timeout=120,
            )
            if r.status_code >= 400 and len(texts) > 1:
                mid = len(texts) // 2
                left = request_embeddings(texts[:mid], retries=retries)
                right = request_embeddings(texts[mid:], retries=retries)
                return np.vstack([left, right])
            r.raise_for_status()
            rows = sorted(r.json()["data"], key=lambda d: d["index"])
            return l2_norm(np.asarray([d["embedding"] for d in rows], dtype=np.float32))
        except Exception as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))

    if len(texts) > 1:
        mid = len(texts) // 2
        return np.vstack([
            request_embeddings(texts[:mid], retries=retries),
            request_embeddings(texts[mid:], retries=retries),
        ])
    raise SystemExit(f"Failed embedding text (len={len(texts[0])}): {last_err}")


def main() -> None:
    gpu_info = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if gpu_info.returncode == 0:
        print(f"GPU: {gpu_info.stdout.strip()}", flush=True)

    server_bin, so_dirs = ensure_llama_and_model()
    proc = start_llama_server(server_bin, so_dirs)
    try:
        # 1. Mandatory parity gate vs local LM Studio
        parity_rows = json.loads((CONTENT / "parity_ref.json").read_text(encoding="utf-8"))
        ref_texts = [r["text"] for r in parity_rows]
        ref_vecs = l2_norm(np.asarray([r["embedding"] for r in parity_rows], dtype=np.float32))
        colab_ref = request_embeddings(ref_texts)

        cosines = np.sum(ref_vecs * colab_ref, axis=1)
        min_cos = float(cosines.min())
        mean_cos = float(cosines.mean())
        print(f"Parity cosines: min={min_cos:.6f}, mean={mean_cos:.6f}, vals={[round(float(c), 6) for c in cosines]}", flush=True)
        if min_cos < 0.990 or mean_cos < 0.995:
            raise SystemExit(f"PARITY GATE FAILED: min_cos={min_cos:.6f} (<0.990) or mean_cos={mean_cos:.6f} (<0.995)")

        # 2. Stream exported texts and embed concurrently via ThreadPoolExecutor
        texts: list[str] = []
        with gzip.open(CONTENT / "universe_news_to_embed.jsonl.gz", "rt", encoding="utf-8") as gz:
            for line in gz:
                texts.append(json.loads(line)["text"])
        total = len(texts)
        print(f"Encoding {total:,} texts (batch={BATCH_SIZE}, workers={MAX_WORKERS})...", flush=True)

        out = np.empty((total, colab_ref.shape[1]), dtype=np.float16)
        ranges = [(start, min(start + BATCH_SIZE, total)) for start in range(0, total, BATCH_SIZE)]

        t0 = time.perf_counter()
        done_count = 0
        next_log = 20000

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(request_embeddings, texts[s:e]): (s, e) for s, e in ranges}
            for fut in as_completed(futs):
                s, e = futs[fut]
                vecs = fut.result()
                out[s:e] = vecs.astype(np.float16)
                done_count += (e - s)
                if done_count >= next_log or done_count == total:
                    elapsed = max(time.perf_counter() - t0, 1e-3)
                    rate = done_count / elapsed
                    eta_s = (total - done_count) / rate if rate else 0.0
                    print(f"Encoded {done_count:,}/{total:,} ({rate:,.1f} vec/s, ETA {eta_s:.0f}s)", flush=True)
                    while next_log <= done_count:
                        next_log += 20000

        if not np.isfinite(out).all():
            raise SystemExit("ERROR: Non-finite values found in encoded vector matrix")

        out_path = CONTENT / "universe_vectors_fp16.npy"
        np.save(out_path, out)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"Saved {out_path} shape={out.shape} dtype={out.dtype} ({size_mb:.1f} MB) in {time.perf_counter() - t0:.1f}s", flush=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
