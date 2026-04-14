"""Lightweight smoke tests — no model load, no GPU required."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure the repo root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_server():
    return importlib.import_module("server")


def test_module_imports():
    server = _import_server()
    assert hasattr(server, "app")
    assert hasattr(server, "MODEL_ID")
    assert server.MODEL_ID.startswith("Qwen/Qwen3-ASR")


def test_routes_registered():
    server = _import_server()
    paths = {route.path for route in server.app.routes}
    expected = {"/v1/audio/transcriptions", "/v1/models", "/health"}
    missing = expected - paths
    assert not missing, f"Missing routes: {missing}"


def test_health_payload_shape():
    """The /health endpoint should respond even before the model is loaded."""
    import asyncio
    server = _import_server()
    payload = asyncio.run(server.health())
    assert "model_ready" in payload
    assert "model_id" in payload
    assert "device" in payload
