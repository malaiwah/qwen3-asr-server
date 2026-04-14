FROM docker.io/nvidia/cuda:13.2.0-devel-ubuntu24.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        git curl ffmpeg libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Python venv
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip numpy

# Qwen3-ASR with vLLM backend (production path).
# CPU fallback (transformers-only) is loaded on demand via server.py --cpu.
RUN pip install --no-cache-dir "qwen-asr[vllm]" fastapi uvicorn soundfile

# HuggingFace cache (mount a volume here to persist downloaded weights)
ENV HF_HOME=/root/.cache/huggingface
VOLUME ["/root/.cache/huggingface"]

# Optional: copy server.py for the CPU-fallback wrapper.  In default
# (GPU) mode the entrypoint forwards to qwen-asr-serve directly.
COPY server.py /app/server.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Default: serve with fp8 KV cache, prefix caching, 4096 context.
# gpu-memory-utilization is set at runtime via podman/docker run to allow
# tuning based on what else is on the GPU.
ENTRYPOINT ["qwen-asr-serve", "Qwen/Qwen3-ASR-1.7B"]
CMD ["--host", "0.0.0.0", \
     "--port", "8000", \
     "--max-model-len", "4096", \
     "--kv-cache-dtype", "fp8"]
