"""Tests for web/code_explorer.py — file tree, file content, path security."""

import os
import pytest
from code_explorer import get_file_tree, read_file_content, build_project_context


class TestFileTree:
    def test_tree_returns_dict(self):
        tree = get_file_tree()
        assert isinstance(tree, dict)
        assert "src" in tree

    def test_src_contains_model_files(self):
        tree = get_file_tree()
        src_files = [entry["name"] for entry in tree.get("src", []) if isinstance(entry, dict)]
        # model.h should exist
        names = []
        for entry in tree.get("src", []):
            if isinstance(entry, dict):
                names.append(entry.get("name", ""))
                if entry.get("type") == "dir" and entry.get("children"):
                    for child in entry["children"]:
                        names.append(child.get("name", ""))
        assert "model.h" in names or any("model.h" in n for n in names)

    def test_tree_entries_have_required_keys(self):
        tree = get_file_tree()
        for entry in tree.get("src", []):
            if isinstance(entry, dict):
                assert "name" in entry
                assert "type" in entry
                assert "path" in entry


class TestFileContent:
    def test_read_model_header(self):
        data = read_file_content("src/model.h")
        assert "error" not in data
        assert data["filename"] == "model.h"

    def test_read_returns_lines(self):
        data = read_file_content("src/model.h")
        assert isinstance(data["lines"], list)
        assert len(data["lines"]) > 0
        assert data["total_lines"] > 0
        assert data["size"] > 0

    def test_read_nonexistent_file(self):
        data = read_file_content("src/nonexistent.cpp")
        assert "error" in data

    def test_path_traversal_blocked(self):
        data = read_file_content("../CMakeLists.txt")
        assert "error" in data

    def test_absolute_path_blocked(self):
        data = read_file_content("C:/Windows/System32/drivers/etc/hosts")
        assert "error" in data


class TestProjectContext:
    def test_build_context_returns_string(self):
        ctx = build_project_context()
        assert isinstance(ctx, str)
        assert len(ctx) > 100
        assert "build-llm-using-cpp" in ctx

    def test_context_lists_source_files(self):
        ctx = build_project_context()
        assert "src/" in ctx
