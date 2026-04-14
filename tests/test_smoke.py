"""Lightweight smoke tests — no GPU, no model load required.

Validates module structure, language prefix stripping, API routing, and
proxy helpers so CI can catch regressions without CUDA hardware.
"""
from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest  # noqa: F401 — kept for pytest.raises

# Ensure the repo root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_server():
    return importlib.import_module("server")


# -----------------------------------------------------------------------
# Module-level smoke
# -----------------------------------------------------------------------

def test_module_imports():
    server = _import_server()
    assert hasattr(server, "app")
    assert hasattr(server, "MODEL_ID")
    assert server.MODEL_ID.startswith("Qwen/Qwen3-ASR")


# -----------------------------------------------------------------------
# Language prefix stripping (critical correctness check)
# -----------------------------------------------------------------------

def test_strip_language_prefix_english():
    server = _import_server()
    assert server._strip_language_prefix("language English\nHello world") == "Hello world"


def test_strip_language_prefix_french():
    server = _import_server()
    assert server._strip_language_prefix("language French\nBonjour le monde") == "Bonjour le monde"


def test_strip_language_prefix_with_comma():
    server = _import_server()
    # Variants the model sometimes emits
    assert server._strip_language_prefix("language English,\nHello") == "Hello"


def test_strip_language_prefix_case_insensitive():
    server = _import_server()
    assert server._strip_language_prefix("Language English\nHello") == "Hello"


def test_strip_language_prefix_noop():
    server = _import_server()
    # Text without prefix should pass through unchanged
    assert server._strip_language_prefix("Hello world") == "Hello world"
    assert server._strip_language_prefix("") == ""


def test_clean_transcription_strips_text_field():
    server = _import_server()
    data = {"text": "language English\nHello world"}
    result = server._clean_transcription(data)
    assert result["text"] == "Hello world"


def test_clean_transcription_strips_segment_text():
    server = _import_server()
    data = {
        "text": "language English\nHello",
        "segments": [
            {"text": "language English\nHello", "start": 0.0, "end": 1.0},
        ],
    }
    result = server._clean_transcription(data)
    assert result["segments"][0]["text"] == "Hello"


def test_clean_chat_completion():
    server = _import_server()
    data = {
        "choices": [
            {"message": {"role": "assistant", "content": "language English\nHello from chat"}}
        ]
    }
    result = server._clean_chat_completion(data)
    assert result["choices"][0]["message"]["content"] == "Hello from chat"


# -----------------------------------------------------------------------
# Route registration
# -----------------------------------------------------------------------

def test_routes_registered():
    server = _import_server()
    paths = {route.path for route in server.app.routes}
    expected = {
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/chat/completions",
        "/v1/models",
        "/health",
    }
    missing = expected - paths
    assert not missing, f"Missing routes: {missing}"


# -----------------------------------------------------------------------
# Health endpoint (no model loaded)
# -----------------------------------------------------------------------

def test_health_payload_when_starting():
    server = _import_server()
    payload = asyncio.run(server.health())
    assert "model_ready" in payload
    assert "model_id" in payload
    assert "device" in payload
    assert "backend" in payload
    assert "status" in payload
    assert payload["model_ready"] is False
