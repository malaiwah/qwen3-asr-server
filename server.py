"""Qwen3-ASR server — OpenAI-compatible /v1/audio/transcriptions endpoint.

Architecture
------------
GPU mode (default)
  Launches ``qwen-asr-serve`` (vLLM) on an internal loopback port, waits for
  it to become healthy, then proxies all ``/v1/*`` requests through.

  Post-processing: strips the ``language X\\n`` prefix that Qwen3-ASR prepends
  to every output before returning the response to the caller.

  Why proxy instead of execv?
    ``os.execv`` replaced the process entirely, making response post-processing
    impossible.  The proxy architecture intercepts and cleans every response.

  Extra CLI flags (``--gpu-memory-utilization``, ``--kv-cache-dtype``, …) are
  forwarded verbatim to the vLLM subprocess.

CPU mode (``--cpu``)
  Loads Qwen3-ASR via ``transformers`` directly and serves a minimal subset:
  ``POST /v1/audio/transcriptions``, ``POST /v1/audio/translations``,
  ``GET /health``, ``GET /v1/models``.
  Much slower — for smoke tests / no-GPU environments only.

Performance
-----------
GPU / vLLM: ~250 ms for a 5 s clip, ~600 ms for a 30 s clip.
CPU:        orders of magnitude slower; do not use in production.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse


MODEL_ID = os.getenv("QWEN3_ASR_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")

# Optional bearer-token auth.  Empty/None disables auth entirely.
_API_KEY: str = os.getenv("QWEN_API_KEY", "") or ""

# Paths exempt from auth (standard ops endpoints + OpenAPI docs).
_AUTH_EXEMPT_PATHS = frozenset({
    "/", "/health", "/metrics",
    "/docs", "/redoc", "/openapi.json",
})

# Internal port for the vLLM subprocess — never exposed to callers.
_VLLM_INTERNAL_PORT = 18000
_VLLM_BASE_URL = f"http://127.0.0.1:{_VLLM_INTERNAL_PORT}"

# Strips language prefix from ASR output.  Qwen3-ASR prepends one of:
#   "language English\n"        (older vLLM / qwen-asr versions)
#   "language English<asr_text>" (newer versions use XML-style tag separator)
# Both forms are matched and removed.
_LANG_PREFIX_RE = re.compile(
    r"^language\s+\S+[,\s]*(?:\n|<asr_text>)", re.IGNORECASE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Shared state
# -----------------------------------------------------------------------

_model_ready: bool = False
_DEVICE: str = "cpu"

# GPU path
_vllm_process: Optional[subprocess.Popen] = None
_vllm_extra_args: list[str] = []

# CPU path
_cpu_processor = None
_cpu_model = None


# -----------------------------------------------------------------------
# Language prefix stripping
# -----------------------------------------------------------------------

def _strip_language_prefix(text: str) -> str:
    """Remove the 'language English\\n' prefix Qwen3-ASR prepends to output.

    Examples:
        "language English\\nHello world"  → "Hello world"
        "language French,\\nBonjour"       → "Bonjour"
        "Hello world"                      → "Hello world"  (no-op)
    """
    return _LANG_PREFIX_RE.sub("", text, count=1).strip()


def _clean_transcription(data: dict) -> dict:
    """Strip language prefix from all text fields in a transcription payload."""
    if isinstance(data.get("text"), str):
        data["text"] = _strip_language_prefix(data["text"])
    for seg in data.get("segments", []):
        if isinstance(seg.get("text"), str):
            seg["text"] = _strip_language_prefix(seg["text"])
    return data


def _clean_chat_completion(data: dict) -> dict:
    """Strip language prefix from assistant message content in chat response."""
    for choice in data.get("choices", []):
        content = choice.get("message", {}).get("content")
        if isinstance(content, str):
            choice["message"]["content"] = _strip_language_prefix(content)
    return data


# -----------------------------------------------------------------------
# GPU path: vLLM subprocess management
# -----------------------------------------------------------------------

async def _wait_for_vllm(timeout: int = 240) -> bool:
    """Poll vLLM /health until ready or timeout."""
    import httpx
    async with httpx.AsyncClient() as client:
        for i in range(timeout):
            try:
                r = await client.get(f"{_VLLM_BASE_URL}/health", timeout=3.0)
                if r.status_code == 200:
                    return True
            except Exception:  # noqa: BLE001
                pass
            if _vllm_process is not None and _vllm_process.poll() is not None:
                logger.error("vLLM subprocess exited early (rc=%d)", _vllm_process.returncode)
                return False
            if i > 0 and i % 15 == 0:
                logger.info("Waiting for vLLM to load model … (%ds)", i)
            await asyncio.sleep(1)
    return False


async def _start_vllm() -> None:
    global _vllm_process, _model_ready

    qwen_bin = shutil.which("qwen-asr-serve")
    if not qwen_bin:
        raise RuntimeError(
            "qwen-asr-serve not found in PATH. "
            "Install with: pip install 'qwen-asr[vllm]'"
        )

    cmd = [
        qwen_bin, MODEL_ID,
        "--host", "127.0.0.1",
        "--port", str(_VLLM_INTERNAL_PORT),
        *_vllm_extra_args,
    ]
    logger.info("Starting vLLM backend: %s", " ".join(cmd))
    _vllm_process = subprocess.Popen(
        cmd, stdout=sys.stdout, stderr=sys.stderr, start_new_session=False
    )

    logger.info("Waiting for vLLM (PID %d) to load model …", _vllm_process.pid)
    ready = await _wait_for_vllm()
    if not ready:
        _vllm_process.kill()
        raise RuntimeError("vLLM backend failed to start within timeout.")

    _model_ready = True
    logger.info("✓ vLLM backend ready — serving %s", MODEL_ID)


def _stop_vllm() -> None:
    global _vllm_process
    if _vllm_process is None:
        return
    pid = _vllm_process.pid
    logger.info("Stopping vLLM (PID %d) …", pid)
    try:
        _vllm_process.send_signal(signal.SIGTERM)
        _vllm_process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logger.warning("vLLM SIGTERM timed out — sending SIGKILL")
        _vllm_process.kill()
        _vllm_process.wait()
    _vllm_process = None
    logger.info("vLLM stopped.")


# -----------------------------------------------------------------------
# CPU path: transformers model management
# -----------------------------------------------------------------------

def _load_model_cpu() -> None:
    global _cpu_processor, _cpu_model, _model_ready
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    logger.warning(
        "Loading %s on CPU — very slow (~0.01× real-time). "
        "For production latency start without --cpu.",
        MODEL_ID,
    )
    t0 = time.time()
    _cpu_processor = AutoProcessor.from_pretrained(MODEL_ID)
    _cpu_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).to("cpu")
    logger.info("CPU model loaded in %.1fs", time.time() - t0)
    _model_ready = True


def _read_audio(upload) -> tuple[np.ndarray, int]:
    import soundfile as sf
    raw = upload.file.read()
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def _transcribe_cpu_sync(audio: np.ndarray, sr: int, language: Optional[str]) -> str:
    import torch
    inputs = _cpu_processor(audio, sampling_rate=sr, return_tensors="pt")
    forced = None
    if language:
        try:
            forced = _cpu_processor.get_decoder_prompt_ids(language=language, task="transcribe")
        except Exception:  # noqa: BLE001
            pass
    with torch.inference_mode():
        ids = _cpu_model.generate(**inputs, forced_decoder_ids=forced, max_new_tokens=448)
    raw = _cpu_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return _strip_language_prefix(raw)


# -----------------------------------------------------------------------
# HTTP proxy helper
# -----------------------------------------------------------------------

async def _proxy(
    request: Request,
    target_path: str,
    *,
    post_process_json=None,
) -> Response:
    """Forward request to internal vLLM, optionally transform JSON response.

    Args:
        request:          Original FastAPI request.
        target_path:      Path on the internal vLLM server.
        post_process_json: Callable(dict) → dict applied to successful JSON responses.
    """
    import httpx

    body = await request.body()
    skip = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}

    try:
        async with httpx.AsyncClient() as client:
            r = await client.request(
                method=request.method,
                url=f"{_VLLM_BASE_URL}{target_path}",
                content=body,
                headers=headers,
                params=dict(request.query_params),
                timeout=300.0,
            )
    except httpx.RequestError as exc:
        raise HTTPException(503, f"ASR backend unavailable: {exc}") from exc

    # Optionally post-process JSON
    if (
        post_process_json is not None
        and r.status_code == 200
        and "application/json" in r.headers.get("content-type", "")
    ):
        try:
            data = post_process_json(r.json())
            return JSONResponse(data, status_code=200)
        except Exception:  # noqa: BLE001
            pass  # fall through to raw response on parse errors

    # Pass through as-is (errors, non-JSON, etc.)
    return Response(
        content=r.content,
        status_code=r.status_code,
        headers=dict(r.headers),
        media_type=r.headers.get("content-type"),
    )


# -----------------------------------------------------------------------
# Prometheus metrics (in-process, no external deps beyond prometheus_client)
# -----------------------------------------------------------------------

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    _METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover — prometheus_client is a hard dep in prod
    _METRICS_AVAILABLE = False

if _METRICS_AVAILABLE:
    _M_REQUESTS = Counter(
        "qwen3_asr_requests_total",
        "Total HTTP requests handled by qwen3-asr-server.",
        ["method", "path", "status"],
    )
    _M_LATENCY = Histogram(
        "qwen3_asr_request_duration_seconds",
        "HTTP request latency (wall clock, seconds).",
        ["method", "path"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    _M_INFLIGHT = Gauge(
        "qwen3_asr_requests_in_flight",
        "HTTP requests currently being processed.",
    )
    _M_MODEL_READY = Gauge(
        "qwen3_asr_model_ready",
        "1 if the backend is fully initialised and ready to serve, else 0.",
    )
    _M_BACKEND_INFO = Gauge(
        "qwen3_asr_backend_info",
        "Backend descriptor (value is always 1; dimensions in labels).",
        ["device", "model_id"],
    )


def _route_template(request: Request) -> str:
    """Return the registered route template for a request (low-cardinality label)."""
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return route.path
    return request.url.path


# -----------------------------------------------------------------------
# FastAPI application
# -----------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if _METRICS_AVAILABLE:
        _M_MODEL_READY.set(0)
        _M_BACKEND_INFO.labels(device=_DEVICE, model_id=MODEL_ID).set(1)
    if _DEVICE == "cpu":
        await asyncio.to_thread(_load_model_cpu)
    else:
        await _start_vllm()
    if _METRICS_AVAILABLE:
        _M_MODEL_READY.set(1 if _model_ready else 0)
    yield
    if _METRICS_AVAILABLE:
        _M_MODEL_READY.set(0)
    if _DEVICE != "cpu":
        _stop_vllm()


app = FastAPI(title="Qwen3-ASR", lifespan=lifespan)


# -----------------------------------------------------------------------
# Auth + metrics middleware
# -----------------------------------------------------------------------

@app.middleware("http")
async def _auth_and_metrics(request: Request, call_next):
    path = request.url.path

    # Bearer-token auth (enabled when QWEN_API_KEY / --api-key is set).
    if _API_KEY and path not in _AUTH_EXEMPT_PATHS:
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {_API_KEY}"
        if auth != expected:
            return JSONResponse(
                {"error": {"message": "Invalid or missing API key.", "type": "invalid_request_error", "code": "invalid_api_key"}},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

    if not _METRICS_AVAILABLE:
        return await call_next(request)

    t0 = time.perf_counter()
    _M_INFLIGHT.inc()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        _M_INFLIGHT.dec()
        template = _route_template(request)
        elapsed = time.perf_counter() - t0
        _M_LATENCY.labels(method=request.method, path=template).observe(elapsed)
        _M_REQUESTS.labels(method=request.method, path=template, status=str(status_code)).inc()


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus exposition endpoint.  Not authenticated (standard ops practice)."""
    if not _METRICS_AVAILABLE:
        return PlainTextResponse(
            "# prometheus_client not installed\n",
            status_code=501,
        )
    # Refresh model-readiness gauge at scrape time (cheap).
    _M_MODEL_READY.set(1 if _model_ready else 0)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -----------------------------------------------------------------------
# Transcription endpoints
# -----------------------------------------------------------------------

@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request):
    """Whisper-compatible transcription.

    Accepts multipart/form-data with fields:
      - ``file``            (required) — audio file (WAV, MP3, OGG, FLAC, …)
      - ``model``           (optional) — model ID, default Qwen/Qwen3-ASR-1.7B
      - ``language``        (optional) — ISO-639-1 hint (e.g. "en", "fr")
      - ``response_format`` (optional) — json | text | verbose_json | srt | vtt
      - ``prompt``          (optional) — vocabulary hint / context string

    The 'language X\\n' prefix that Qwen3-ASR prepends internally is stripped
    automatically before the response is sent.
    """
    if not _model_ready:
        raise HTTPException(503, "Model is still loading. Check /health and retry.")

    if _DEVICE != "cpu":
        # Prime the body cache FIRST.  request.form() reads via stream() which
        # marks the stream consumed; a subsequent request.body() in _proxy would
        # then raise RuntimeError("Stream consumed").  Calling request.body()
        # first caches the raw bytes in request._body; stream() thereafter
        # yields from that cache, so both form() and _proxy work correctly.
        await request.body()
        form = await request.form()
        response_format = (form.get("response_format") or "json").lower()

        proxy_result = await _proxy(request, "/v1/audio/transcriptions",
                                    post_process_json=_clean_transcription)

        # vLLM always returns JSON; reformat to plain text when requested.
        if response_format == "text" and proxy_result.status_code == 200:
            try:
                data = json.loads(proxy_result.body)
                return Response(content=data.get("text", ""), media_type="text/plain")
            except Exception:
                pass
        return proxy_result

    # CPU path
    form = await request.form()
    upload = form.get("file")
    if upload is None:
        raise HTTPException(400, "Missing 'file' field in multipart form data")
    language = form.get("language")
    response_format = form.get("response_format", "json")

    audio, sr = _read_audio(upload)
    text = await asyncio.to_thread(_transcribe_cpu_sync, audio, sr, language)

    if response_format == "text":
        return Response(content=text, media_type="text/plain")
    return JSONResponse({"text": text})


@app.post("/v1/audio/translations")
async def translate(request: Request):
    """Whisper-compatible translations endpoint (API compat shim).

    Qwen3-ASR transcribes in the detected language; it does **not** translate
    to English.  This endpoint proxies to /v1/audio/transcriptions and adds
    a ``_note`` field explaining the limitation.

    For actual translation, pass the transcript text to a downstream LLM.
    """
    if not _model_ready:
        raise HTTPException(503, "Model is still loading. Check /health and retry.")

    def _clean_and_note(data: dict) -> dict:
        data = _clean_transcription(data)
        data["_note"] = (
            "Qwen3-ASR does not translate to English; this is a transcription "
            "in the detected language. Use a downstream LLM for translation."
        )
        return data

    if _DEVICE != "cpu":
        return await _proxy(request, "/v1/audio/transcriptions",
                            post_process_json=_clean_and_note)

    # CPU: reuse transcription logic
    return await transcribe(request)


# -----------------------------------------------------------------------
# Context-primed transcription via Chat API
# -----------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Context-primed transcription via the chat completions API.

    Send audio as an ``input_audio`` content part in the user message.
    A system prompt can bias vocabulary, language, or named entities.

    Example::

        {
          "model": "Qwen/Qwen3-ASR-1.7B",
          "messages": [
            {"role": "system", "content": "Speaker uses Python, Kubernetes, AWS."},
            {"role": "user", "content": [
              {"type": "input_audio",
               "input_audio": {"data": "<base64>", "format": "wav"}}
            ]}
          ],
          "temperature": 0.0
        }

    The language prefix is stripped from the assistant message content.
    Requires the vLLM backend (GPU mode).
    """
    if not _model_ready:
        raise HTTPException(503, "Model is still loading. Check /health and retry.")
    if _DEVICE == "cpu":
        raise HTTPException(
            501,
            "Context-primed transcription (/v1/chat/completions) requires the vLLM "
            "backend. Start without --cpu to enable it.",
        )
    return await _proxy(request, "/v1/chat/completions",
                        post_process_json=_clean_chat_completion)


# -----------------------------------------------------------------------
# Info / introspection endpoints
# -----------------------------------------------------------------------

@app.get("/health")
async def health():
    vllm_alive = False
    if _DEVICE != "cpu" and _vllm_process is not None:
        if _vllm_process.poll() is None:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    r = await client.get(f"{_VLLM_BASE_URL}/health", timeout=2.0)
                    vllm_alive = r.status_code == 200
            except Exception:  # noqa: BLE001
                pass

    ready = _model_ready and (vllm_alive if _DEVICE != "cpu" else True)
    status = "healthy" if ready else ("loading" if _model_ready else "starting")
    return {
        "status": status,
        "model_ready": ready,
        "model_id": MODEL_ID,
        "device": _DEVICE,
        "backend": "vllm" if _DEVICE != "cpu" else "transformers (CPU — slow)",
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen_asr",
            }
        ],
    }


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR server (GPU: vLLM proxy | CPU: transformers fallback)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Unknown flags are forwarded to qwen-asr-serve (vLLM) in GPU mode.

Useful vLLM flags:
  --gpu-memory-utilization 0.55   lower when sharing GPU with TTS (default: 0.9)
  --max-model-len 4096
  --kv-cache-dtype fp8            halves KV-cache VRAM
  --max-num-seqs 4

GPU sharing with qwen3-tts-server (16 GB card):
  Start TTS first (fixed ~4.4 GB footprint), then ASR with:
    --gpu-memory-utilization 0.55
  vLLM sizes its KV cache to whatever VRAM is left after TTS.
""",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (transformers). Very slow — smoke tests only.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Require 'Authorization: Bearer <key>' on /v1/* endpoints. "
             "Overrides the QWEN_API_KEY env var when provided. "
             "/health and /metrics remain unauthenticated.",
    )
    parser.add_argument("--log-level", default="info")
    args, extra = parser.parse_known_args()

    global _DEVICE, _vllm_extra_args, _API_KEY
    if args.api_key is not None:
        _API_KEY = args.api_key
    if _API_KEY:
        logger.info("API-key auth enabled — /v1/* endpoints require 'Authorization: Bearer <key>'.")

    gpu_available = False
    if not args.cpu:
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            pass

    if args.cpu or not gpu_available:
        if not args.cpu:
            logger.warning(
                "No CUDA device detected — falling back to CPU (transformers). "
                "Very slow. Install 'qwen-asr[vllm]' and ensure CUDA is available "
                "for production use."
            )
        _DEVICE = "cpu"
    else:
        _DEVICE = "cuda"
        _vllm_extra_args = extra

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
