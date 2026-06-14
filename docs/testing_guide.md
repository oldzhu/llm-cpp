> [简体中文](README.zh-CN.md)

# Test Infrastructure Guide

This project has a multi-layer test suite. Run all tests with a single command.

## Quick Start

```powershell
# All layers (one command)
.\test_all.ps1
```

## Layer 1: C++ Tests (ctest)

Always available. No extra dependencies.

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release
```

Currently: **20 tests**

---

## Layer 2: Python Tests (pytest)

Requires: `pip install pytest pytest-asyncio httpx`

```powershell
pytest tests/python/ -v
```

Currently: **44 tests** (config, server API, code explorer, AI chat, import script)

---

## Layer 3: JavaScript Tests (vitest)

Requires: [Node.js](https://nodejs.org/) + `npm install`

```powershell
npm install
npm test
```

Test files in `tests/js/`:
- `i18n.test.js` — translation lookup, interpolation, fallback
- `settings.test.js` — settings load/save, modal generation
- `architecture.test.js` — component data completeness
- `code.test.js` — selection logic, markdown rendering
- `charts.test.js` — data push, shift limit, reset

---

## Layer 4: E2E Tests (Playwright)

Requires: `pip install pytest-playwright` + `python -m playwright install chromium`

```powershell
# Start server first (separate terminal)
python web/server.py

# Run E2E tests
pytest tests/e2e/ -v
```

Test files in `tests/e2e/`:
- `web_ui.spec.js` — page loads, tabs switch, config form
- `training.spec.js` — start/stop training, status changes
- `code_explorer.spec.js` — file tree, viewer, line selection
- `architecture.spec.js` — diagram render, hover tooltip, click navigate
- `i18n.spec.js` — language toggle, all labels change
- `settings.spec.js` — gear icon, modal, save persistence
- `chat.spec.js` — send message, response display
- `monitor.spec.js` — chart canvas, weight table

---

## Test-First Workflow

For every new feature:

1. **Write the test FIRST** (red — test fails)
2. **Implement the feature** (green — test passes)
3. **Run ALL existing tests** — nothing broken
4. **Commit only when all green**

This is documented in `docs/project_lifecycle_guidelines.md`.
