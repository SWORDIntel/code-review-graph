"""Canonical, evidence-preserving CRG -> QIHSE CodeGraph v1 snapshot.

Do not use visualization.export_graph_data here: its name resolution heuristic
may choose the first candidate for an ambiguous call. This transport preserves
CRG's raw source/target identities and explicitly marks unresolved relations.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


FORMAT = "qihse-codegraph-v1"
MAX_RECORD_BYTES = 256 * 1024


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30,
    )
    return result.stdout


def _relative(root: Path, value: str) -> str:
    """Do not publish local usernames, full paths or paths outside the repo."""
    if not value:
        return ""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve().relative_to(root).as_posix()
    except ValueError:
        return ""


def _stable_ref(root: Path, qualified_name: str) -> str:
    """Stable across filesystems for CRG's repo-relative qualified identifiers."""
    ref = qualified_name
    prefix = str(root)
    if ref.startswith(prefix + os.sep):
        ref = ref[len(prefix) + 1:]
    elif ref == prefix:
        ref = "."
    return hashlib.sha256(ref.encode("utf-8")).hexdigest()


def _node_record(root: Path, node: Any) -> dict:
    file_path = _relative(root, node.file_path)
    return {
        "t": "n",
        "id": _stable_ref(root, node.qualified_name),
        "file": file_path,
        "kind": str(node.kind)[:64],
        "name": str(node.name)[:512],
        "line_start": int(node.line_start or 0),
        "line_end": int(node.line_end or 0),
        "language": str(node.language or "")[:48],
        "is_test": bool(node.is_test),
        "file_hash": str(node.file_hash or "")[:64],
    }


def _edge_record(root: Path, edge: Any, mapped: dict[str, str]) -> dict:
    source = mapped.get(edge.source_qualified)
    target = mapped.get(edge.target_qualified)
    tier = str(edge.confidence_tier or "UNKNOWN")[:32]
    resolution = str(edge.target_resolution or "direct").lower()[:24]
    confidence = max(0.0, min(float(edge.confidence), 1.0))
    # A missing/ambiguous endpoint must never silently become an authoritative
    # call. CRG sometimes has a target spelling that names no indexed node.
    trusted = bool(
        source and target and resolution == "direct"
        and tier.upper() not in {"UNKNOWN", "INFERRED", "HEURISTIC", "AMBIGUOUS", "UNRESOLVED"}
        and confidence >= 0.8
    )
    relation = str(edge.kind)[:48]
    location = _relative(root, edge.file_path)
    line = int(edge.line or 0)
    edge_id = hashlib.sha256(
        json.dumps([source or "", target or "", relation, location, line],
                   separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    return {
        "t": "e", "id": edge_id, "source": source or "",
        "target": target or "", "relation": relation,
        "file": location, "line": line, "confidence": confidence,
        "tier": tier, "resolution": resolution,
        "traversable": trusted,
    }


def export_qihse_snapshot(store: Any, output_path: Path, repo_root: Path) -> dict:
    """Write a validated, atomic, deterministic JSONL snapshot.

    Export requires current CRG file hashes to agree with disk. A concurrent
    edit or differing git state aborts instead of publishing a stale graph.
    Unresolved relationships remain visible as non-traversable evidence.
    """
    root = repo_root.resolve(strict=True)
    if not (root / ".git").exists():
        raise ValueError("QIHSE graph export requires a Git working tree")

    previous_head = _git(root, "rev-parse", "--verify", "HEAD").strip().decode("ascii")
    previous_status = hashlib.sha256(_git(root, "status", "--porcelain=v1",
                                          "-z", "--untracked-files=normal")).hexdigest()

    # Index freshness cannot be inferred from CRG timestamps; validate bytes.
    for original, expected in sorted(store.get_file_hashes().items()):
        rel = _relative(root, original)
        if not rel:
            raise ValueError("CRG index contains a path outside the repository")
        source = root / rel
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"CRG graph stale (missing/nonregular source): {rel}")
        if expected:
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"CRG graph stale (source changed): {rel}; run update")

    nodes = store.get_all_nodes(exclude_files=False)
    mapped: dict[str, str] = {}
    ids: set[str] = set()
    for node in nodes:
        key = _stable_ref(root, node.qualified_name)
        if key in ids:
            raise ValueError("duplicate canonical graph node identity")
        ids.add(key)
        mapped[node.qualified_name] = key

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    counts = {"nodes": 0, "edges": 0, "unresolved_edges": 0}
    digest = hashlib.sha256()

    try:
        fd, tmp = tempfile.mkstemp(prefix=".codegraph.", suffix=".tmp",
                                   dir=output_path.parent)
        temp_path = Path(tmp)
        with os.fdopen(fd, "wb") as out:
            def write(record: dict, *, hashed: bool = True) -> None:
                encoded = (
                    json.dumps(record, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True) + "\n"
                ).encode("ascii")
                if len(encoded) > MAX_RECORD_BYTES:
                    raise ValueError("codegraph record exceeds 256 KiB")
                out.write(encoded)
                if hashed:
                    digest.update(encoded)

            write({
                "t": "h", "format": FORMAT, "version": 1,
                "head": previous_head, "worktree": previous_status,
            })
            for node in sorted(nodes, key=lambda item: mapped[item.qualified_name]):
                write(_node_record(root, node))
                counts["nodes"] += 1
            for edge in sorted(store.get_all_edges(), key=lambda e:
                               (e.source_qualified, e.target_qualified, e.kind, e.line)):
                row = _edge_record(root, edge, mapped)
                write(row)
                counts["edges"] += 1
                if not row["traversable"]:
                    counts["unresolved_edges"] += 1

            current_head = _git(root, "rev-parse", "--verify", "HEAD").strip().decode("ascii")
            current_status = hashlib.sha256(_git(root, "status", "--porcelain=v1",
                                                 "-z", "--untracked-files=normal")).hexdigest()
            if (current_head, current_status) != (previous_head, previous_status):
                raise ValueError("Git worktree changed during graph export; retry")

            write({"t": "end", **counts, "sha256": digest.hexdigest()}, hashed=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp_path, output_path)
        temp_path = None
        directory = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return {
        "format": FORMAT, "path": str(output_path), **counts,
        "head": previous_head, "worktree": previous_status,
        "sha256": digest.hexdigest(),
    }
