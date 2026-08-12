"""Failed ingestions must not permanently consume maker memory slots."""

from __future__ import annotations

import ast
from pathlib import Path


def test_upload_limits_use_active_count_not_all_rows():
    """Routes must ignore status=error when enforcing the 3-memory cap."""
    root = Path(__file__).resolve().parents[1]
    memories_src = (root / "app/routes/memories.py").read_text(encoding="utf-8")
    queries_src = (root / "app/db/queries.py").read_text(encoding="utf-8")

    assert "count_active_by_kind" in memories_src
    assert "count_by_kind(conn" not in memories_src.replace("count_active_by_kind", "")

    tree = ast.parse(queries_src)
    fn_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }
    assert "count_active_by_kind" in fn_names

    # Active count SQL must exclude error rows
    assert "status <> 'error'" in queries_src or "status != 'error'" in queries_src
