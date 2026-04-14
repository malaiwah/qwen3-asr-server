#!/usr/bin/env python3
"""Benchmark a running qwen3-asr-server.

Measures steady-state wall-clock latency and real-time factor (RTF) for each
provided audio file.  First run of each file is treated as warmup and excluded.

Usage:
    ./benchmark.py audio.wav
    ./benchmark.py short.wav long.wav --runs 5
    ./benchmark.py audio.wav --url http://my-gpu-host:8002

Audio duration is read from the file headers (via soundfile or ffprobe) and
RTF is computed as audio_seconds / wall_seconds.
"""
from __future__ import annotations

import argparse
import io
import mimetypes
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

import urllib.error
import urllib.request


def _audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf
        with sf.SoundFile(str(path)) as f:
            return len(f) / f.samplerate
    except Exception:
        pass
    import subprocess
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], text=True)
    return float(out.strip())


def _multipart_post(url: str, file_path: Path, fields: dict[str, str], api_key: str | None) -> bytes:
    """Send a multipart/form-data POST with no external deps."""
    boundary = f"----qwen3asrbench{uuid.uuid4().hex}"
    body = bytearray()
    for key, val in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{val}\r\n".encode()
    mime, _ = mimetypes.guess_type(str(file_path))
    mime = mime or "application/octet-stream"
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_path.name}\"\r\nContent-Type: {mime}\r\n\r\n".encode()
    body += file_path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()

    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=bytes(body), method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def _format_avg(label: str, walls: list[float], audio_s: float) -> str:
    avg = statistics.mean(walls)
    rtf = audio_s / avg if avg > 0 else 0.0
    stdev = statistics.pstdev(walls) if len(walls) > 1 else 0.0
    return (
        f"  → avg over {len(walls)} runs: wall={avg*1000:7.1f}ms "
        f"(±{stdev*1000:4.1f}ms)  audio={audio_s:5.2f}s  RTF={rtf:5.2f}x"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="Audio files to benchmark (WAV/MP3/OGG/FLAC).")
    parser.add_argument("--url", default="http://localhost:8002", help="Base URL of the ASR server.")
    parser.add_argument("--runs", type=int, default=5, help="Runs per file (first is warmup, excluded).")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--language", default=None, help='Hint, e.g. "English", "French".')
    parser.add_argument("--api-key", default=os.getenv("QWEN_API_KEY") or None,
                        help="Bearer token (reads QWEN_API_KEY env if unset).")
    args = parser.parse_args()

    url = args.url.rstrip("/") + "/v1/audio/transcriptions"
    print(f"qwen3-asr-server benchmark — {args.url}  runs={args.runs}  model={args.model}\n")

    files = [Path(f) for f in args.files]
    for f in files:
        if not f.is_file():
            print(f"No such file: {f}", file=sys.stderr)
            return 1

    any_failed = False
    for f in files:
        try:
            audio_s = _audio_duration_seconds(f)
        except Exception as exc:
            print(f"[{f.name}]  cannot read duration: {exc}  — skipping", file=sys.stderr)
            any_failed = True
            continue
        print(f"[{f.name}]  audio={audio_s:.2f}s  size={f.stat().st_size} bytes")

        fields = {"model": args.model, "response_format": "json"}
        if args.language:
            fields["language"] = args.language

        kept: list[float] = []
        for i in range(args.runs):
            t0 = time.perf_counter()
            try:
                _ = _multipart_post(url, f, fields, args.api_key)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:300]
                print(f"  run {i+1}: HTTP {exc.code} — {body}", file=sys.stderr)
                any_failed = True
                break
            except urllib.error.URLError as exc:
                print(f"  run {i+1}: network error: {exc.reason}", file=sys.stderr)
                return 2
            wall = time.perf_counter() - t0
            rtf = audio_s / wall if wall > 0 else 0.0
            tag = " (warmup)" if i == 0 else ""
            print(f"  run {i+1}{tag}: wall={wall*1000:7.1f}ms  RTF={rtf:5.2f}x")
            if i > 0:
                kept.append(wall)
        if kept:
            print(_format_avg("avg", kept, audio_s))
        print()

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
