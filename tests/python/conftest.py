"""Shared test fixtures for Python tests."""

import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add 'web/' to sys.path so modules can do 'from config import ...' etc.
# Append (not insert) to avoid shadowing stdlib modules.
_web_dir = str(Path(__file__).resolve().parent.parent.parent / "web")
if _web_dir not in sys.path:
    sys.path.append(_web_dir)

from config import CONFIG, PRESETS, validate, apply_preset
from code_explorer import get_file_tree, read_file_content, build_project_context


@pytest.fixture
def client():
    """FastAPI TestClient for server endpoints. Mocks TrainingSession."""
    import server as server_mod
    # Replace imported TrainingSession with a mock
    server_mod.session = MagicMock()
    server_mod.session.state = "idle"
    server_mod.session.current_step = 0
    server_mod.session.current_loss = 0.0
    server_mod.session.current_time = 0.0
    server_mod.session.total_steps = 100
    server_mod.session.weight_stats = {}
    server_mod.session.start = AsyncMock(return_value={"status": "started"})
    server_mod.session.stop = AsyncMock(return_value={"status": "stopped", "step": 50})

    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        yield c


@pytest.fixture
def sample_config():
    """Valid training config for tests."""
    return {
        "data_path": "data/alice.txt",
        "steps": 100,
        "batch": 4,
        "seq": 64,
        "dmodel": 64,
        "layers": 1,
        "lr": 0.0003,
        "seed": 1,
        "norm_type": "layernorm",
        "mlp_type": "gelu",
        "tokenizer": "byte",
    }
