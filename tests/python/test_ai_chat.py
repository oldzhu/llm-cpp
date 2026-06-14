"""Tests for web/ai_chat.py — chat function, settings passthrough, error handling."""

import asyncio
import pytest

from ai_chat import chat, explain_code


class TestAIChatNoKey:
    def test_no_api_key_returns_error(self):
        result = asyncio.run(chat([{"role": "user", "content": "Hi"}]))
        assert "error" in result
        assert "API key" in result["error"]

    def test_custom_settings_no_key(self):
        result = asyncio.run(chat([{"role": "user", "content": "Hi"}],
                                  settings={"api_url": "http://localhost:9999"}))
        assert "error" in result

    def test_url_with_fake_key_gives_api_error(self):
        settings = {"api_url": "https://api.example.com", "api_key": "sk-fake-key"}
        result = asyncio.run(chat([{"role": "user", "content": "Hi"}], settings=settings))
        assert "error" in result  # API call will fail

    def test_explain_no_key(self):
        result = asyncio.run(explain_code("src/ops.cpp", 1, 5, settings={}))
        assert "error" in result

    def test_explain_nonexistent_file(self):
        result = asyncio.run(explain_code("nonexistent.cpp", 1, 5, settings={"api_key":"sk-fake"}))
        assert "error" in result
