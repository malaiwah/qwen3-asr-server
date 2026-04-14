"""Thin wrapper around qwen-asr-serve (vLLM backend) with optional CPU fallback.

The default deployment runs the upstream ``qwen-asr-serve`` CLI directly
(see Containerfile).  This module exists so users without a GPU can still
exercise the API surface:

    python server.py --cpu

CPU mode loads the Qwen3-ASR HuggingFace checkpoint via ``transformers``
and serves a minimal subset of the OpenAI-compatible audio endpoints
(``/v1/audio/transcriptions`` and ``/v1/audio/translations``).  It is
**slow** (multi-second per short clip) and intended for smoke tests only.

For production use, prefer the vLLM-backed entrypoint baked into the
container.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse


MODEL_ID = os.getenv("QWEN3_ASR_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


app = FastAPI(title="Qwen3-ASR (CPU fallback)")
processor = None
model = None
model_ready = False
_DEVICE = "cpu"


def _load_model_cpu() -> None:
    """Load the model on CPU using transformers."""
    global processor, model, model_ready
    import torch  # local import — keeps cold start fast for --help
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    logger.warning(
        "Loading %s on CPU — this is SLOW. For production, use the vLLM "
        "entrypoint (qwen-asr-serve) instead.",
        MODEL_ID,
    )
    start = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).to("cpu")
    logger.info("CPU model loaded in %.1fs", time.time() - start)
    model_ready = True


def _read_audio(upload: UploadFile) -> tuple[np.ndarray, int]:
    """Decode an uploaded audio file into mono float32 + sample rate."""
    import soundfile as sf
    raw = upload.file.read()
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def _transcribe_cpu(audio: np.ndarray, sr: int, language: str | None = None) -> str:
    import torch
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    forced = None
    if language:
        try:
            forced = processor.get_decoder_prompt_ids(language=language, task="transcribe")
        except Exception:
            forced = None
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            forced_decoder_ids=forced,
            max_new_tokens=448,
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# -----------------------------------------------------------------------
# Endpoints (CPU-only fallback subset)
# -----------------------------------------------------------------------

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL_ID),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
):
    if not model_ready:
        raise HTTPException(503, "Model loading")
    audio, sr = _read_audio(file)
    text = _transcribe_cpu(audio, sr, language=language)
    if response_format == "text":
        return text
    return JSONResponse({"text": text})


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model_ready else "loading",
        "model_ready": model_ready,
        "model_id": MODEL_ID,
        "device": _DEVICE,
        "backend": "transformers (CPU fallback)",
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID, "object": "model", "created": int(time.time()),
            "owned_by": "qwen_asr",
        }],
    }


@app.on_event("startup")
async def _startup():
    if _DEVICE == "cpu":
        _load_model_cpu()


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-ASR server (CPU fallback wrapper)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode. Otherwise this script execs the production "
             "vLLM entrypoint (qwen-asr-serve).",
    )
    parser.add_argument("--log-level", default="info")
    args, extra = parser.parse_known_args()

    if not args.cpu:
        # GPU path: hand off to the upstream vLLM CLI so users get the
        # full /v1/* surface (chat/completions for context-primed ASR,
        # streaming, fp8 KV cache, etc.).
        try:
            import shutil
            qwen_bin = shutil.which("qwen-asr-serve")
            if not qwen_bin:
                raise FileNotFoundError("qwen-asr-serve not found in PATH")
        except Exception as exc:
            logger.error(
                "GPU mode requires the qwen-asr[vllm] package: %s. "
                "Use --cpu for the slow fallback.", exc,
            )
            return 2
        argv = [qwen_bin, MODEL_ID, "--host", args.host, "--port", str(args.port), *extra]
        logger.info("Execing: %s", " ".join(argv))
        os.execv(qwen_bin, argv)
        return 0  # unreachable

    # CPU fallback
    global _DEVICE
    _DEVICE = "cpu"
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
