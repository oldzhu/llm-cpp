"""E2E test fixtures — starts web server, provides playwright browser."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def server():
    """Start web server for E2E tests."""
    server_script = PROJECT_ROOT / "web" / "server.py"
    python = sys.executable
    proc = subprocess.Popen([python, str(server_script)], cwd=str(PROJECT_ROOT),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Wait for server to start
    base = "http://127.0.0.1:8080"
    for _ in range(30):
        try:
            import urllib.request
            urllib.request.urlopen(f"{base}/api/train/status", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def browser():
    """Playwright browser instance."""
    # Use pytest-playwright if available, otherwise manual
    try:
        from pytest_playwright import configure
    except ImportError:
        pass
    with __import__("playwright.sync_api", fromlist=["sync_playwright"]).sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


@pytest.fixture
def page(browser, server):
    """Fresh page for each test."""
    ctx = browser.new_context()
    pg = ctx.new_page()
    pg.set_default_timeout(10000)
    pg.goto(server, wait_until="domcontentloaded")
    yield pg
    ctx.close()
