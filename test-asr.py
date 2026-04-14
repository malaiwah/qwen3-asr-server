#!/usr/bin/env python3
"""Quick CLI tester for a running qwen3-asr-server.

Usage:
    ./test-asr.py audio.wav
    ./test-asr.py audio.wav --url http://my-gpu-host:8002
    ./test-asr.py audio.wav --language English --context "Hermes Agent, Honcho, oikos"
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
import uuid
from pathlib import Path

import urllib.error
import urllib.request


def _multipart_post(url: str, file_path: Path, fields: dict[str, str]) -> bytes:
    """Send a multipart/form-data POST without external dependencies."""
    boundary = f"----qwen3asrtest{uuid.uuid4().hex}"
    body = bytearray()
    for key, val in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{val}\r\n".encode()
    mime, _ = mimetypes.guess_type(str(file_path))
    mime = mime or "application/octet-stream"
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_path.name}\"\r\nContent-Type: {mime}\r\n\r\n".encode()
    body += file_path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url, data=bytes(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="Path to a WAV/MP3/OGG/FLAC file (mono or stereo).")
    parser.add_argument("--url", default="http://localhost:8002", help="Base URL of the ASR server.")
    parser.add_argument("--language", default=None, help='Hint, e.g. "English", "French". Optional.')
    parser.add_argument("--context", default=None, help="Context-priming text for /v1/chat/completions path (vocabulary biasing).")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    args = parser.parse_args()

    audio = Path(args.audio)
    if not audio.is_file():
        print(f"No such file: {audio}", file=sys.stderr)
        return 1

    base = args.url.rstrip("/")
    start = time.time()

    try:
        if args.context:
            # Context-primed path via chat/completions (vocabulary biasing)
            mime, _ = mimetypes.guess_type(str(audio))
            fmt = (audio.suffix.lstrip(".") or "wav").lower()
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": args.context},
                    {"role": "user", "content": [
                        {"type": "input_audio", "input_audio": {
                            "data": base64.b64encode(audio.read_bytes()).decode(),
                            "format": fmt,
                        }},
                    ]},
                ],
                "temperature": 0.0,
            }
            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
        else:
            # Whisper-compatible path via multipart upload
            url = f"{base}/v1/audio/transcriptions"
            fields = {"model": args.model}
            if args.language:
                fields["language"] = args.language
            raw = _multipart_post(url, audio, fields)
            data = json.loads(raw)
            text = data.get("text", data)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Could not reach server at {args.url}: {exc.reason}", file=sys.stderr)
        return 2

    wall = (time.time() - start) * 1000
    print(text)
    print(f"\n[transcribed in {wall:.0f} ms]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
