"""Tests for web server — content types, error handling, edge cases."""

import pytest


class TestServerContent:
    def test_home_html_content_type(self, client):
        resp = client.get("/")
        assert "text/html" in resp.headers.get("content-type", "")

    def test_static_css_loads(self, client):
        resp = client.get("/static/style.css")
        assert resp.status_code == 200

    def test_static_js_loads(self, client):
        resp = client.get("/static/app.js")
        assert resp.status_code == 200

    def test_static_i18n_loads(self, client):
        resp = client.get("/static/i18n.js")
        assert resp.status_code == 200

    def test_404_not_found(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_api_train_status_no_training(self, client):
        resp = client.get("/api/train/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "idle"


class TestConfigAPI:
    def test_config_schema_has_new_fields(self, client):
        resp = client.get("/api/config/schema")
        data = resp.json()
        # Verify new attention/position fields exist
        assert "attn_type" in data
        assert "n_heads" in data
        assert "n_kv" in data
        assert "pos_type" in data
        assert "qk_norm" in data
        assert "swin_win" in data
        assert "n_mtp" in data
        assert "n_shared" in data

    def test_config_schema_mla_option(self, client):
        resp = client.get("/api/config/schema")
        data = resp.json()
        assert "mla" in data["attn_type"]["options"]

    def test_config_schema_pos_options(self, client):
        resp = client.get("/api/config/schema")
        data = resp.json()
        opts = data["pos_type"]["options"]
        for o in ["wpe","rope","alibi","nope"]:
            assert o in opts


class TestWebSocketEndpoint:
    def test_ws_endpoint_accepts(self, client):
        # Verify the WebSocket route exists (FastAPI returns 426 without upgrade)
        with client.websocket_connect("/ws/train") as ws:
            # Just connect and disconnect — proves endpoint works
            pass
