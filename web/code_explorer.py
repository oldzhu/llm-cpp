"""Code explorer API — browse and read project source files."""

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCAN_DIRS = {
    "src": str(PROJECT_ROOT / "src"),
    "docs": str(PROJECT_ROOT / "docs"),
    "tests": str(PROJECT_ROOT / "tests"),
    "root": str(PROJECT_ROOT),
}

ROOT_FILES = ["CMakeLists.txt", "README.md", "README.zh-CN.md"]


def get_file_tree() -> dict[str, Any]:
    """Returns directory tree for all scanned dirs."""
    tree: dict[str, Any] = {}
    for name, path in SCAN_DIRS.items():
        tree[name] = _scan_dir(Path(path), Path(path))
    # Add root-level files
    root_entries = []
    for f in ROOT_FILES:
        p = PROJECT_ROOT / f
        if p.exists():
            root_entries.append({"name": f, "type": "file", "path": f})
    tree["root files"] = root_entries
    return tree


def _scan_dir(root: Path, rel: Path) -> list[dict]:
    entries: list[dict] = []
    try:
        for child in sorted(rel.iterdir()):
            if child.name.startswith("."):
                continue
            if child.name in ("build", "__pycache__", "data", "shaders", "web", "chat"):
                continue
            entry: dict = {
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
                "path": str(child.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
            if child.is_dir():
                entry["children"] = _scan_dir(root, child)
            entries.append(entry)
    except PermissionError:
        pass
    return entries


def read_file_content(rel_path: str) -> dict:
    """Read a file's content with line numbers."""
    full = PROJECT_ROOT / rel_path
    if not full.exists() or not full.is_file():
        return {"error": "File not found"}
    if not str(full.resolve()).startswith(str(PROJECT_ROOT.resolve())):
        return {"error": "Access denied"}
    try:
        text = full.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {"error": "Cannot read file"}
    return {
        "path": rel_path,
        "filename": full.name,
        "lines": text.split("\n"),
        "total_lines": len(text.split("\n")),
        "size": full.stat().st_size,
    }


def build_project_context() -> str:
    """Build a condensed project context for AI chat."""
    parts = []
    parts.append("Project: build-llm-using-cpp — C++ GPT-style Transformer from scratch")
    parts.append("Key source files and their roles:")
    for name, path in SCAN_DIRS.items():
        if Path(path).exists():
            parts.append(f"\n{name}/:")
            for f in sorted(Path(path).glob("**/*.cpp")) + sorted(Path(path).glob("**/*.h")):
                rel = str(f.relative_to(PROJECT_ROOT)).replace("\\", "/")
                parts.append(f"  {rel}")
    return "\n".join(parts)
