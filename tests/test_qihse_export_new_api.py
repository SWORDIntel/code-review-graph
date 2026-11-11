"""QIHSE CodeGraph export preserves graph evidence and worktree identity."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from code_review_graph.exports import export_qihse
from code_review_graph.graph import GraphEdge, GraphNode


class _Store:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def get_all_nodes(self, exclude_files=True):
        assert exclude_files is False
        return self.nodes

    def get_all_edges(self):
        return self.edges


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True,
    ).stdout


def test_nested_untracked_git_identity_tracks_its_contents(repository):
    nested = repository / "nested"
    nested.mkdir()
    _git(nested, "init", "-q")
    source = nested / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    _git(nested, "add", "module.py")
    _git(nested, "-c", "user.email=test@example.invalid", "-c", "user.name=test",
         "commit", "-qm", "initial")
    listing = _git(repository, "ls-files", "--others", "--exclude-standard", "-z",
                   "--", ".", ":(exclude).code-review-graph")
    assert listing == b"nested/\0"

    first = _header(repository)
    source.write_text("value = 2\n", encoding="utf-8")
    second = _header(repository)
    assert first["worktree"] == second["worktree"]
    assert first["worktree_v2"] != second["worktree_v2"]

    nested_status = _git(nested, "status", "--porcelain=v1", "-z",
                         "--untracked-files=normal", "--", ".",
                         ":(exclude).code-review-graph")
    nested_diff = _git(nested, "diff", "--no-ext-diff", "--no-textconv",
                       "--binary", "HEAD", "--", ".", ":(exclude).code-review-graph")
    nested_index = _git(nested, "ls-files", "--stage", "-z", "--", ".",
                        ":(exclude).code-review-graph")
    nested_v2 = hashlib.sha256(nested_status + b"\0" + nested_diff + b"\0" +
                               nested_index + b"\0").hexdigest().encode("ascii")
    nested_head = _git(nested, "rev-parse", "HEAD").strip()
    nested_content = hashlib.sha256(b"qihse-nested-git-v1\0" + nested_head + b"\0" +
                                    nested_v2).hexdigest().encode("ascii")
    outer_status = _git(repository, "status", "--porcelain=v1", "-z",
                        "--untracked-files=normal", "--", ".",
                        ":(exclude).code-review-graph")
    outer_diff = _git(repository, "diff", "--no-ext-diff", "--no-textconv",
                      "--binary", "HEAD", "--", ".", ":(exclude).code-review-graph")
    outer_index = _git(repository, "ls-files", "--stage", "-z", "--", ".",
                       ":(exclude).code-review-graph")
    assert second["worktree_v2"] == hashlib.sha256(
        outer_status + b"\0" + outer_diff + b"\0" + outer_index + b"\0" +
        b"nested/\0r\0" + nested_content + b"\0"
    ).hexdigest()


def test_qihse_export_records_integrity_and_traversal(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    source = repo / "a.py"
    source.write_text("def first(): pass\n", encoding="utf-8")
    _git(repo, "add", "a.py")
    _git(repo, "-c", "user.email=test@example.invalid", "-c", "user.name=test",
         "commit", "-qm", "initial")
    source.write_text("def first(): return second()\n", encoding="utf-8")

    def node(ident, name, file_path):
        return GraphNode(ident, "Function", name, f"a.py::{name}", file_path, 1, 1,
                         "python", None, None, None, False,
                         hashlib.sha256(source.read_bytes()).hexdigest(), {})

    direct = GraphEdge(1, "CALLS", "a.py::first", "a.py::second", str(source), 1,
                       {}, 0.95, "EXTRACTED", "direct")
    uncertain = GraphEdge(2, "CALLS", "a.py::first", "external", "a.py", 1,
                          {}, 0.4, "INFERRED", "unresolved")
    store = _Store([node(1, "first", str(source)), node(2, "second", "a.py")],
                   [direct, uncertain])
    output = repo / ".code-review-graph" / "graph.qihse.jsonl"
    export_qihse(store, output, repo)
    raw_lines = output.read_bytes().splitlines(keepends=True)
    records = [json.loads(line) for line in raw_lines]

    assert [record["t"] for record in records] == ["h", "n", "n", "e", "e", "end"]
    assert records[0]["format"] == "qihse-codegraph-v1"
    assert records[0]["head"] == _git(repo, "rev-parse", "HEAD").decode().strip()
    status = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=normal",
                  "--", ".", ":(exclude).code-review-graph")
    diff = _git(repo, "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD",
                "--", ".", ":(exclude).code-review-graph")
    index = _git(repo, "ls-files", "--stage", "-z", "--", ".",
                 ":(exclude).code-review-graph")
    assert records[0]["freshness"] == "git-status-diff-index-untracked-v2"
    assert records[0]["worktree"] == hashlib.sha256(status + b"\0" + diff).hexdigest()
    assert records[0]["worktree_v2"] == hashlib.sha256(
        status + b"\0" + diff + b"\0" + index + b"\0"
    ).hexdigest()
    assert records[1]["file"] == records[2]["file"] == records[3]["file"] == "a.py"
    edges = [row for row in records if row["t"] == "e"]
    assert sum(row["traversable"] for row in edges) == 1
    assert next(row for row in edges if not row["traversable"])["target"] == ""
    assert records[-1] == {
        "t": "end", "nodes": 2, "edges": 2, "unresolved_edges": 1,
        "sha256": hashlib.sha256(b"".join(raw_lines[:-1])).hexdigest(),
    }


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_bytes(b"def original(): pass\n")
    (repo / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.email=test@example.invalid", "-c", "user.name=test",
         "commit", "-qm", "initial")
    return repo


def _header(repo, store=None):
    output = repo / ".code-review-graph" / "graph.qihse.jsonl"
    export_qihse(store if store is not None else _Store([], []), output, repo)
    return json.loads(output.read_bytes().splitlines()[0])


@pytest.mark.parametrize("name", [
    "new.py", "nested/a b\nfile.py", "binary.dat", os.fsdecode(b"nonutf8-\xff.py"),
])
def test_qihse_untracked_content_changes_identity(repository, name):
    source = repository / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"first\0contents")
    before_status = _git(repository, "status", "--porcelain=v1", "-z")
    before = _header(repository)
    pathspec = ("--", ".", ":(exclude).code-review-graph")
    expected = hashlib.sha256(
        _git(repository, "status", "--porcelain=v1", "-z", "--untracked-files=normal",
             *pathspec) + b"\0"
        + _git(repository, "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD",
               *pathspec) + b"\0"
        + _git(repository, "ls-files", "--stage", "-z", *pathspec) + b"\0"
        + os.fsencode(name) + b"\0f\0"
        + hashlib.sha256(b"first\0contents").hexdigest().encode("ascii") + b"\0"
    ).hexdigest()
    assert before["worktree_v2"] == expected
    source.write_bytes(b"other\0contents")
    assert _git(repository, "status", "--porcelain=v1", "-z") == before_status
    assert _header(repository)["worktree_v2"] != before["worktree_v2"]


def test_qihse_untracked_symlink_hashes_target_not_destination(repository):
    target = repository.parent / "outside.py"
    target.write_bytes(b"outside contents")
    link = repository / "new.py"
    link.symlink_to(target)
    before = _header(repository)
    target.write_bytes(b"changed outside contents")
    assert _header(repository)["worktree_v2"] == before["worktree_v2"]
    link.unlink()
    link.symlink_to("missing-target")
    assert _header(repository)["worktree_v2"] != before["worktree_v2"]


def test_qihse_untracked_kind_changes_identity(repository):
    source = repository / "new.py"
    source.write_bytes(b"missing-target")
    before = _header(repository)
    source.unlink()
    source.symlink_to("missing-target")
    assert _header(repository)["worktree_v2"] != before["worktree_v2"]


def test_qihse_index_changes_identity_with_same_worktree_and_status(repository):
    source = repository / "a.py"
    source.write_bytes(b"def staged_one(): pass\n")
    _git(repository, "add", "a.py")
    source.write_bytes(b"def working(): pass\n")
    before = _header(repository)
    before_status = _git(repository, "status", "--porcelain=v1", "-z")
    before_diff = _git(repository, "diff", "--binary", "HEAD")
    source.write_bytes(b"def staged_two(): pass\n")
    _git(repository, "add", "a.py")
    source.write_bytes(b"def working(): pass\n")
    assert _git(repository, "status", "--porcelain=v1", "-z") == before_status
    assert _git(repository, "diff", "--binary", "HEAD") == before_diff
    assert _header(repository)["worktree_v2"] != before["worktree_v2"]


def test_qihse_ignored_and_export_files_do_not_change_identity(repository):
    before = _header(repository)
    ignored = repository / "ignored"
    ignored.mkdir()
    (ignored / "a.py").write_bytes(b"ignored bytes")
    (repository / ".code-review-graph" / "scratch.py").write_bytes(b"export bytes")
    assert _header(repository) == before


@pytest.mark.parametrize("change", ["tracked", "untracked", "index", "head"])
def test_qihse_repository_changed_during_export_preserves_destination(repository, change):
    output = repository / ".code-review-graph" / "graph.qihse.jsonl"
    if change == "untracked":
        (repository / "new.py").write_bytes(b"def original(): pass\n")
    elif change == "index":
        (repository / "a.py").write_bytes(b"def staged(): pass\n")
        _git(repository, "add", "a.py")
        (repository / "a.py").write_bytes(b"def working(): pass\n")
    _header(repository)
    original = output.read_bytes()

    class ChangingStore(_Store):
        def get_all_nodes(self, exclude_files=True):
            if change == "tracked":
                (repository / "a.py").write_bytes(b"def changed(): pass\n")
            elif change == "untracked":
                (repository / "new.py").write_bytes(b"def changed(): pass\n")
            elif change == "index":
                (repository / "a.py").write_bytes(b"def staged_changed(): pass\n")
                _git(repository, "add", "a.py")
                (repository / "a.py").write_bytes(b"def working(): pass\n")
            else:
                _git(repository, "-c", "user.email=test@example.invalid", "-c", "user.name=test",
                     "commit", "--allow-empty", "-qm", "new head")
            return super().get_all_nodes(exclude_files)

    with pytest.raises(ValueError, match="repository changed during QIHSE export"):
        export_qihse(ChangingStore([], []), output, repository)
    assert output.read_bytes() == original
    assert list(output.parent.glob(".graph.qihse.jsonl.*.tmp")) == []


@pytest.mark.parametrize("change", ["tracked", "untracked", "missing", "assume-unchanged"])
def test_qihse_stale_graph_cannot_acquire_current_identity(repository, change):
    source = repository / ("new.py" if change == "untracked" else "a.py")
    if not source.exists():
        source.write_bytes(b"def original(): pass\n")
    indexed_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    node = GraphNode(1, "File", source.name, source.name, str(source), 1, 1,
                     "python", None, None, None, False, indexed_hash, {})
    store = _Store([node], [])
    output = repository / ".code-review-graph" / "graph.qihse.jsonl"
    export_qihse(store, output, repository)
    retained = output.read_bytes()
    if change == "missing":
        source.unlink()
    else:
        if change == "assume-unchanged":
            _git(repository, "update-index", "--assume-unchanged", "a.py")
        source.write_bytes(b"def changed(): pass\n")
    with pytest.raises(ValueError, match="indexed source"):
        export_qihse(store, output, repository)
    assert output.read_bytes() == retained
    assert not list(output.parent.glob(".graph.qihse.jsonl.*.tmp"))


def test_qihse_source_hash_changed_during_export_preserves_destination(repository, monkeypatch):
    source = repository / "a.py"
    node = GraphNode(1, "File", source.name, source.name, str(source), 1, 1,
                     "python", None, None, None, False,
                     hashlib.sha256(source.read_bytes()).hexdigest(), {})
    output = repository / ".code-review-graph" / "graph.qihse.jsonl"
    export_qihse(_Store([node], []), output, repository)
    retained = output.read_bytes()

    real_fsync = os.fsync

    def edit_after_snapshot_write(fd):
        real_fsync(fd)
        source.write_bytes(b"def changed(): pass\n")

    monkeypatch.setattr(os, "fsync", edit_after_snapshot_write)
    with pytest.raises(ValueError, match="indexed source changed"):
        export_qihse(_Store([node], []), output, repository)
    assert output.read_bytes() == retained
    assert not list(output.parent.glob(".graph.qihse.jsonl.*.tmp"))


def _stable_store(repo):
    source = repo / "a.py"
    nodes = [GraphNode(i, "Function", name, f"a.py::{name}", str(source), 1, 1,
                       "python", None, None, None, False,
                       hashlib.sha256(source.read_bytes()).hexdigest(), {})
             for i, name in enumerate(("first", "second"), 1)]
    edges = [GraphEdge(i, "CALLS", "a.py::first", target, str(source), line,
                       {}, confidence, tier, resolution)
             for i, (target, line, confidence, tier, resolution) in enumerate([
                 ("a.py::second", 1, 0.95, "EXTRACTED", "direct"),
                 ("a.py::second", 1, 0.95, "EXTRACTED", "direct"),
                 ("a.py::second", 2, 0.95, "EXTRACTED", "direct"),
                 ("external-one", 1, 0.4, "INFERRED", "unresolved"),
                 ("external-two", 1, 0.4, "INFERRED", "unresolved"),
             ], 1)]
    return _Store(nodes, edges)


def test_qihse_rebuild_row_ids_and_input_order_do_not_change_export(repository):
    from dataclasses import replace
    store = _stable_store(repository)
    output = repository / ".code-review-graph" / "graph.qihse.jsonl"
    export_qihse(store, output, repository)
    before = output.read_bytes()
    store.nodes = [replace(n, id=1000-i) for i, n in enumerate(reversed(store.nodes))]
    store.edges = [replace(e, id=5000-i) for i, e in enumerate(reversed(store.edges))]
    export_qihse(store, output, repository)
    assert output.read_bytes() == before
    rows = [json.loads(line) for line in before.splitlines()]
    edges = [row for row in rows if row["t"] == "e"]
    assert len(edges) == len({row["id"] for row in edges}) == 5
    assert sum(not row["traversable"] for row in edges) == 2
    assert rows[0]["identity"] == "qihse-semantic-xxh3-128-v1"
    assert rows[0]["id_algorithm"] == "xxh3-128"
    assert rows[0]["id_bits"] == 128
    assert rows[0]["id_seed"] == 0
    assert rows[0]["id_encoding"] == "xxh128-canonical"
    assert rows[0]["id_collision"] == "counter-xxh3-128-v1"
    assert rows[0]["id_collision_attempts"] == 256
    for row in rows:
        if row["t"] in ("n", "e"):
            assert len(row["id"]) == 32
            assert all(c in "0123456789abcdef" for c in row["id"])
        if row["t"] == "e":
            assert not row["source"] or len(row["source"]) == 32
            assert not row["target"] or len(row["target"]) == 32


@pytest.mark.parametrize("collide", ["node", "edge"])
def test_qihse_digest_collision_preserves_previous_export(repository, monkeypatch, collide):
    from code_review_graph import exports
    store = _stable_store(repository)
    output = repository / ".code-review-graph" / "graph.qihse.jsonl"
    export_qihse(store, output, repository)
    previous = output.read_bytes()
    real_digest = exports._qihse_id_digest
    prefix = b"qihse-node-128-v1\0" if collide == "node" else b"qihse-edge-site-v2\0"

    def collide_selected_kind(raw):
        identity = raw
        if raw.startswith(exports._QIHSE_ID_COLLISION_DOMAIN):
            offset = len(exports._QIHSE_ID_COLLISION_DOMAIN)
            length = int.from_bytes(raw[offset:offset + 8], "big")
            identity = raw[offset + 8:offset + 8 + length]
        return "0" * 32 if identity.startswith(prefix) else real_digest(raw)

    monkeypatch.setattr(exports, "_qihse_id_digest", collide_selected_kind)
    monkeypatch.setattr(exports, "_qihse_id_digests",
                        lambda rows: [exports._qihse_id_digest(row) for row in rows])
    with pytest.raises(ValueError, match="identity collision"):
        export_qihse(store, output, repository)
    assert output.read_bytes() == previous
    assert not list(output.parent.glob(".graph.qihse.jsonl.*.tmp"))


def test_qihse_identity_framing_distinguishes_embedded_separators():
    from code_review_graph.exports import _qihse_edge_identity
    assert _qihse_edge_identity(("a\0b", "c"), 0) != _qihse_edge_identity(("a", "b\0c"), 0)
    assert _qihse_edge_identity(("a",), 0) != _qihse_edge_identity(("a",), 1)


def test_qihse_128_initial_collisions_resolve_stably(repository, monkeypatch):
    from dataclasses import replace

    from code_review_graph import exports
    real_digest = exports._qihse_id_digest

    def collide_initial(raw):
        if not raw.startswith(exports._QIHSE_ID_COLLISION_DOMAIN):
            return "ab" * 16
        return real_digest(raw)

    monkeypatch.setattr(exports, "_qihse_id_digest", collide_initial)
    monkeypatch.setattr(exports, "_qihse_id_digests",
                        lambda rows: [exports._qihse_id_digest(row) for row in rows])
    store = _stable_store(repository)
    output = repository / ".code-review-graph" / "graph.qihse.jsonl"
    export_qihse(store, output, repository)
    first = output.read_bytes()
    records = [json.loads(line) for line in first.splitlines()]
    public = [row["id"] for row in records if row["t"] in ("n", "e")]
    assert len(public) == len(set(public)) == 7
    assert "ab" * 16 in public
    nodes = {row["id"] for row in records if row["t"] == "n"}
    for edge in (row for row in records if row["t"] == "e"):
        assert not edge["source"] or edge["source"] in nodes
        assert not edge["target"] or edge["target"] in nodes
    store.nodes = [replace(row, id=2000-i) for i, row in enumerate(reversed(store.nodes))]
    store.edges = [replace(row, id=9000-i) for i, row in enumerate(reversed(store.edges))]
    export_qihse(store, output, repository)
    assert output.read_bytes() == first


def test_qihse_128_digest_matches_fixed_domain_and_width():
    from code_review_graph import exports
    identity = exports._qihse_node_identity("a.py::function")
    assert identity == b"qihse-node-128-v1\0" + (14).to_bytes(8, "big") + b"a.py::function"
    assert exports._qihse_id_digest(identity) == "667c4a9389b288d9195ffe608145c761"
    assert exports._qihse_id_digests([identity])[0] == exports._qihse_id_digest(identity)
    allocated = {}
    assert exports._qihse_checked_id(identity, allocated) == exports._qihse_checked_id(
        identity, allocated,
    )
    assert len(allocated) == 1


def test_qihse_128_collision_resolver_compares_full_preimages(monkeypatch):
    from code_review_graph import exports
    real_digest = exports._qihse_id_digest
    monkeypatch.setattr(exports, "_qihse_id_digest", lambda raw:
                        real_digest(raw) if raw.startswith(exports._QIHSE_ID_COLLISION_DOMAIN)
                        else "42" * 16)
    a, b = exports._qihse_node_identity("first"), exports._qihse_node_identity("second")
    registry = {}
    first = exports._qihse_checked_id(a, registry)
    second = exports._qihse_checked_id(b, registry)
    assert first == "42" * 16
    assert second != first and len(second) == 32
    assert registry[first] == a and registry[second] == b


def test_qihse_xxh3_vectors_and_batch_boundaries():
    from code_review_graph.qihse_xxh3 import hash_many, hash_one
    assert hash_one(b"") == "99aa06d3014798d86001c324468d497f"
    assert hash_one(b"abc") == "06b05ab6733a618578af5f94892f3950"
    rows = [b"", b"abc", bytes(range(256))] * 1500
    assert hash_many(rows) == [hash_one(row) for row in rows]
    assert hash_many([]) == []


def test_qihse_xxh3_missing_library_is_explicit(monkeypatch, tmp_path):
    from code_review_graph import qihse_xxh3
    qihse_xxh3._binding.cache_clear()
    try:
        with monkeypatch.context() as scoped:
            scoped.setenv("QIHSE_CODEGRAPH_XXH3_LIBRARY", str(tmp_path / "absent.so"))
            with pytest.raises(RuntimeError, match="requires the tuned XXH3 native library"):
                qihse_xxh3.hash_one(b"abc")
    finally:
        qihse_xxh3._binding.cache_clear()


def test_qihse_xxh3_scalar_auto_equivalence(monkeypatch):
    from code_review_graph import qihse_xxh3
    qihse_xxh3._binding.cache_clear()
    try:
        expected = qihse_xxh3.hash_many([b"", b"abc", bytes(range(256)) * 32])
        with monkeypatch.context() as scoped:
            scoped.setenv("QIHSE_CODEGRAPH_XXH3_BACKEND", "scalar")
            qihse_xxh3._binding.cache_clear()
            assert qihse_xxh3.hash_many([b"", b"abc", bytes(range(256)) * 32]) == expected
    finally:
        qihse_xxh3._binding.cache_clear()
