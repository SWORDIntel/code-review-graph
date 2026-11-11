"""Compatibility entry point for the native QIHSE CodeGraph exporter.

The CLI and direct callers share one versioned freshness and XXH3 identity
contract. Keeping a separate serializer here would publish graphs that the
native importer cannot accept.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .exports import _qihse_identity, export_qihse


def _worktree_digest(root: Path) -> str:
    """Return the legacy HEAD diff/status component of the current identity."""
    return _qihse_identity(Path(root).resolve())[1]


def summarize_qihse_snapshot(output: Path) -> dict:
    """Read the small header/trailer to preserve the CLI's summary output."""
    with output.open("rb") as source:
        header = json.loads(source.readline())
        source.seek(max(0, output.stat().st_size - 4096))
        trailer = json.loads(source.readlines()[-1])
    return {
        "format": header["format"], "file": output.name,
        "nodes": trailer["nodes"], "edges": trailer["edges"],
        "unresolved_edges": trailer["unresolved_edges"],
        "head": header["head"], "worktree": header["worktree"],
        "freshness": header["freshness"], "worktree_v2": header["worktree_v2"],
        "sha256": trailer["sha256"],
    }


def export_qihse_snapshot(store: Any, output_path: Path, repo_root: Path) -> dict:
    """Export through the native producer and retain the old summary result."""
    output = export_qihse(store, Path(output_path), Path(repo_root))
    return summarize_qihse_snapshot(output)
