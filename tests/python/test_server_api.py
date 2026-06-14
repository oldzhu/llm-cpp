"""Tests for web/server.py API routes."""

import pytest


class TestServerRoutes:
    def test_home_page_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "build-llm-using-cpp" in resp.text

    def test_code_page_returns_200(self, client):
        resp = client.get("/code")
        assert resp.status_code == 200

    def test_chat_page_returns_200(self, client):
        resp = client.get("/chat")
        assert resp.status_code == 200

    def test_config_schema_returns_200(self, client):
        resp = client.get("/api/config/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "data_path" in data
        assert isinstance(data["data_path"], dict)

    def test_config_presets_returns_200(self, client):
        resp = client.get("/api/config/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert "Tiny (32M)" in data

    def test_train_start_returns_status(self, client, sample_config):
        resp = client.post("/api/train/start", json={"config": sample_config})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"

    def test_train_stop_returns_status(self, client):
        resp = client.post("/api/train/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"

    def test_train_status_returns_structure(self, client):
        resp = client.get("/api/train/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "state" in data
        assert "step" in data
        assert "loss" in data
        assert "time" in data

    def test_code_tree_returns_200(self, client):
        resp = client.get("/api/code/tree")
        assert resp.status_code == 200

    def test_code_file_returns_content(self, client):
        resp = client.get("/api/code/file?path=src/model.h")
        assert resp.status_code == 200
        data = resp.json()
        assert "lines" in data
        assert "filename" in data

    def test_code_file_invalid_path(self, client):
        resp = client.get("/api/code/file?path=../../../etc/passwd")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_chat_send_no_key(self, client):
        resp = client.post("/api/chat/send", json={"messages": [{"role":"user","content":"Hi"}]})
        assert resp.status_code == 200
        data = resp.json()
        # Should return error about missing API key
        assert "error" in data

    def test_chat_explain_no_key(self, client):
        resp = client.post("/api/chat/explain", json={"file":"src/ops.cpp","start_line":1,"end_line":5})
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_page_html_contains_tabs(self, client):
        resp = client.get("/")
        assert "config" in resp.text.lower()
        assert "monitor" in resp.text.lower()
