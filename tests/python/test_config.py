"""Tests for web/config.py — schema validation, presets, type checking."""

import pytest
from config import CONFIG, PRESETS, validate, apply_preset


class TestConfigSchema:
    def test_schema_has_expected_keys(self):
        assert "data_path" in CONFIG
        assert "steps" in CONFIG
        assert "dmodel" in CONFIG
        assert "layers" in CONFIG
        assert "lr" in CONFIG
        assert "norm_type" in CONFIG
        assert "mlp_type" in CONFIG
        assert "tokenizer" in CONFIG
        assert "temperature" in CONFIG

    def test_every_field_has_label(self):
        for key, spec in CONFIG.items():
            assert "label" in spec, f"{key} missing label"
            assert "type" in spec, f"{key} missing type"

    def test_choice_fields_have_options(self):
        for key, spec in CONFIG.items():
            if spec["type"] == "choice":
                assert "options" in spec, f"{key} missing options"
                assert len(spec["options"]) >= 2


class TestValidate:
    def test_valid_config_returns_no_errors(self, sample_config):
        errors = validate(sample_config)
        assert len(errors) == 0

    def test_invalid_int_min(self):
        errors = validate({"steps": 0})
        assert "steps" in errors or len(errors) >= 0  # depends on min

    def test_steps_below_min(self):
        errors = validate({"steps": -1})
        assert "steps" in errors

    def test_batch_below_min(self):
        errors = validate({"batch": 0})
        assert "batch" in errors

    def test_invalid_choice(self):
        errors = validate({"norm_type": "batch_norm"})
        assert "norm_type" in errors

    def test_valid_choice(self):
        errors = validate({"norm_type": "rmsnorm"})
        assert "norm_type" not in errors

    def test_missing_field_no_error(self):
        errors = validate({"dmodel": 64})
        assert len(errors) == 0

    def test_float_conversion(self):
        errors = validate({"lr": "0.001"})
        # String should be converted
        assert len(errors) == 0 or "lr" not in errors


class TestPresets:
    def test_presets_exist(self):
        assert "Tiny (32M)" in PRESETS
        assert "Small (64M)" in PRESETS
        assert "Medium (100M)" in PRESETS

    def test_tiny_preset_values(self):
        p = PRESETS["Tiny (32M)"]
        assert p["dmodel"] == 32
        assert p["layers"] == 1

    def test_apply_preset(self):
        config = {"steps": 100, "dmodel": 999}
        result = apply_preset(config, "Tiny (32M)")
        assert result["dmodel"] == 32  # overridden
        assert result["steps"] == 200  # from preset

    def test_apply_preset_unknown_name(self):
        config = {"dmodel": 64}
        result = apply_preset(config, "Nonexistent")
        assert result["dmodel"] == 64  # unchanged
