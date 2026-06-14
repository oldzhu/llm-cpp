"""Tests for scripts/import_safetensors.py — safetensors roundtrip."""

import json
import os
import struct
import sys
from pathlib import Path

import pytest

# Add scripts/ to path
_scripts = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

# Try importing the converter
try:
    import import_safetensors as importer
    HAVE_IMPORTER = True
except ImportError as e:
    HAVE_IMPORTER = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not HAVE_IMPORTER, reason=f"Import script deps missing: {IMPORT_ERROR if not HAVE_IMPORTER else ''}")
class TestBinaryFormat:
    def test_write_binary_format(self, tmp_path):
        """Verify binary checkpoint format is written correctly."""
        import numpy as np
        output = str(tmp_path / "test_ckpt")
        cfg = {"vocab_size": 256, "seq_len": 64, "d_model": 64, "n_layers": 1,
               "norm_type": 0, "mlp_type": 0, "swiglu_interm": 0, "n_experts": 4, "top_k": 2,
               "lr": 0.0003, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01,
               "imported_from": "test"}

        # Create dummy tensors
        tensors = []
        tensors.append(("wte", np.random.randn(256, 64).astype(np.float32)))
        tensors.append(("wpe", np.random.randn(64, 64).astype(np.float32)))

        importer.write_checkpoint(output, cfg, tensors)

        # Verify JSON
        with open(output + ".json") as f:
            j = json.load(f)
        assert j["vocab_size"] == 256
        assert j["d_model"] == 64
        assert j["step"] == 0
        assert j["has_optim_state"] == False

        # Verify binary
        with open(output + ".bin", "rb") as f:
            magic = f.read(8)
            assert magic == b"BGPTCKPT"
            version = struct.unpack("<I", f.read(4))[0]
            assert version == 4
            has_opt = struct.unpack("<I", f.read(4))[0]
            assert has_opt == 0
            step = struct.unpack("<Q", f.read(8))[0]
            assert step == 0
            nparams = struct.unpack("<I", f.read(4))[0]
            assert nparams == 2


@pytest.mark.skipif(not HAVE_IMPORTER, reason="Import script deps missing")
class TestSafetensorsRead:
    def test_can_parse_config(self, tmp_path):
        """Verify config.json parsing from HF model dir."""
        import numpy as np
        # Create a mock HF model directory
        cfg = {"vocab_size": 50257, "n_ctx": 1024, "n_embd": 768, "n_layer": 6}
        cfg_path = tmp_path / "config.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        # Create tiny safetensors
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            pytest.skip("safetensors/torch not installed")

        tensors = {"wte.weight": torch.randn(50257, 768)}
        sf_path = tmp_path / "model.safetensors"
        save_file(tensors, str(sf_path))

        # Load
        loaded = importer.load_safetensors(str(tmp_path))
        assert "wte.weight" in loaded
        assert loaded["wte.weight"].shape == (50257, 768)
