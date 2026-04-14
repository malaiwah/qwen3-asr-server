# Qwen3-ASR server
#
# Base: CUDA 12.8 + Ubuntu 24.04
#   vLLM (qwen-asr[vllm]) installs cu12x torch wheels via pip, so the
#   container CUDA runtime is 12.x regardless of base image version.
#   Using cu128 base gives the widest driver compatibility (≥ 520).
#
#   CUDA 13.x base was tried but requires driver ≥ 570 which excludes
#   vGPU instances and many cloud machines still on driver 550/12.4.
#
# Architecture: server.py proxies all requests to qwen-asr-serve (vLLM)
# on an internal port and strips the "language X\n" prefix from responses.
# Extra flags (--gpu-memory-utilization, --kv-cache-dtype, …) are forwarded
# to vLLM at runtime.

FROM docker.io/nvidia/cuda:12.8.0-devel-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        git curl ffmpeg libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip numpy

# ASR inference stack:
#   qwen-asr[vllm]  → vLLM-backed production server (GPU mode)
#   transformers    → CPU fallback loaded on demand via --cpu
#   httpx           → async proxy client used by server.py
RUN pip install --no-cache-dir "qwen-asr[vllm]" fastapi uvicorn soundfile httpx

# HuggingFace weight cache
ENV HF_HOME=/root/.cache/huggingface
VOLUME ["/root/.cache/huggingface"]

COPY server.py /app/server.py

EXPOSE 8000

# Health check polls the /health endpoint (model_ready field)
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=6 \
    CMD python3 -c "import urllib.request,json,sys; \
d=json.loads(urllib.request.urlopen('http://localhost:8000/health').read()); \
sys.exit(0 if d['model_ready'] else 1)"

# server.py starts our FastAPI wrapper which launches qwen-asr-serve internally.
# All extra flags after -- are forwarded to vLLM.
ENTRYPOINT ["python3", "/app/server.py"]
CMD ["--host", "0.0.0.0", \
     "--port", "8000", \
     "--max-model-len", "4096", \
     "--kv-cache-dtype", "fp8"]
