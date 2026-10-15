"""CodeGraph export tests: precise provenance, privacy, atomic stale rejection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from code_review_graph.qihse_export import export_qihse_snapshot


class FakeGraph:
    def __init__(self, root: Path, *, unresolved=False):
        path = root / "src/a.py"
        self.expected = hashlib.sha256(path.read_bytes()).hexdigest()
        self.nodes = [
            SimpleNamespace(
                qualified_name=f"{path}::target", kind="Function", name="target",
                file_path=str(path), line_start=1, line_end=3, language="python",
                is_test=False, file_hash=self.expected,
            ),
            SimpleNamespace(
                qualified_name=f"{path}::caller", kind="Function", name="caller",
                file_path=str(path), line_start=5, line_end=7, language="python",
                is_test=False, file_hash=self.expected,
            ),
        ]
        self.edges = [
            SimpleNamespace(
                source_qualified=f"{path}::caller",
                target_qualified=f"{path}::external" if unresolved else f"{path}::target",
                kind="CALLS", file_path=str(path), line=6,
                confidence=0.95, confidence_tier="EXTRACTED",
                target_resolution="unresolved" if unresolved else "direct",
            )
        ]

    def get_file_hashes(self):
        return {self.nodes[0].file_path: self.expected}

    def get_all_nodes(self, exclude_files=True):
        assert exclude_files is False
        return self.nodes

    def get_all_edges(self):
        return self.edges


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src/a.py").write_text(
        "def target():\n    pass\n\ndef caller():\n    return target()\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=test@example.invalid",
                    "-c", "user.name=Test", "commit", "-qm", "baseline"],
                   cwd=root, check=True)
    return root


def test_canonical_export(tmp_path):
    root = _repo(tmp_path)
    output = root / ".code-review-graph" / "graph.qihse.jsonl"
    result = export_qihse_snapshot(FakeGraph(root), output, root)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["t"] for r in records] == ["h", "n", "n", "e", "end"]
    assert result["nodes"] == 2 and result["edges"] == 1
    assert records[3]["traversable"] is True
    assert records[3]["resolution"] == "direct"
    assert records[-1]["sha256"] == hashlib.sha256(
        b"".join((json.dumps(row, sort_keys=True, ensure_ascii=True,
                             separators=(",", ":")) + "\n").encode("ascii")
                 for row in records[:-1])
    ).hexdigest()
    # Native persistence receives only repo-relative paths.
    assert str(root) not in output.read_text()
    assert records[1]["file"] == "src/a.py"


def test_unresolved_is_never_guessed(tmp_path):
    root = _repo(tmp_path)
    output = root / "graph.jsonl"
    result = export_qihse_snapshot(FakeGraph(root, unresolved=True), output, root)
    edge = [json.loads(line) for line in output.read_text().splitlines()
            if json.loads(line)["t"] == "e"][0]
    assert result["unresolved_edges"] == 1
    assert edge["target"] == "" and not edge["traversable"]
    assert edge["source"]


def test_stale_index_never_overwrites_snapshot(tmp_path):
    root = _repo(tmp_path)
    output = root / "graph.jsonl"
    store = FakeGraph(root)
    export_qihse_snapshot(store, output, root)
    original = output.read_bytes()
    (root / "src/a.py").write_text("updated()\n")
    with pytest.raises(ValueError, match="stale"):
        export_qihse_snapshot(store, output, root)
    assert output.read_bytes() == original


def test_ambiguous_edge_not_promoted(tmp_path):
    root = _repo(tmp_path)
    output = root / "graph.jsonl"
    store = FakeGraph(root)
    store.edges[0].confidence_tier = "HEURISTIC"
    export_qihse_snapshot(store, output, root)
    edge = [json.loads(line) for line in output.read_text().splitlines()
            if json.loads(line)["t"] == "e"][0]
    assert edge["traversable"] is False
