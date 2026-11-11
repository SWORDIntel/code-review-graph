"""QIHSE incremental export parity, explicit source providers and progress."""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from code_review_graph.exports import export_qihse
from code_review_graph.graph import GraphStore
from code_review_graph.incremental import get_db_path
from code_review_graph.tools.build import build_or_update_graph


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True)


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "callee.py").write_text("def target():\n    return 1\n")
    (repo / "caller.py").write_text(
        "from callee import target\n\ndef caller():\n    return target()\n"
    )
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "baseline",
    )
    return repo


def exported(repo, path, full):
    result = build_or_update_graph(repo_root=str(repo), full_rebuild=full)
    assert result["status"] == "ok", result
    store = GraphStore(get_db_path(repo))
    try:
        export_qihse(store, path, repo)
    finally:
        store.close()
    return path.read_bytes()


def test_renamed_imported_file_reparses_unchanged_consumer_before_stale_purge(repository, tmp_path):
    repo = repository
    exported(repo, tmp_path / "before", True)
    (repo / "callee.py").rename(repo / "renamed.py")
    incremental = exported(repo, tmp_path / "incremental", False)
    full = exported(repo, tmp_path / "full", True)
    assert incremental == full
    assert b"callee.py::target" not in incremental


def test_deleted_imported_file_matches_full_rebuild(repository, tmp_path):
    repo = repository
    exported(repo, tmp_path / "before", True)
    (repo / "callee.py").unlink()
    assert exported(repo, tmp_path / "incremental", False) == exported(
        repo, tmp_path / "full", True
    )


def test_reverted_uncommitted_source_hash_is_reconciled(repository, tmp_path):
    repo = repository
    exported(repo, tmp_path / "before", True)
    original = (repo / "callee.py").read_bytes()
    (repo / "callee.py").write_text("def renamed():\n    return 2\n")
    exported(repo, tmp_path / "edited", False)
    (repo / "callee.py").write_bytes(original)
    assert exported(repo, tmp_path / "incremental", False) == exported(
        repo, tmp_path / "full", True
    )


def test_explicit_identity_provider_supports_verified_non_git_workspace(tmp_path):
    repo = tmp_path / "workspace"
    repo.mkdir()
    (repo / ".code-review-graph").mkdir()
    (repo / "source.py").write_text("def target():\n    return 1\n")
    result = build_or_update_graph(repo_root=str(repo), full_rebuild=True)
    assert result["status"] == "ok", result
    calls = []

    def identity(root):
        calls.append(root)
        return ("0" * 64, "a" * 64, "b" * 64)

    store = GraphStore(get_db_path(repo))
    try:
        output = tmp_path / "export"
        export_qihse(store, output, repo, source_identity_provider=identity)
        header = json.loads(output.read_bytes().splitlines()[0])
        assert header["head"] == "0" * 64 and header["worktree"] == "a" * 64
        assert calls == [repo, repo]
        assert not (repo / ".git").exists()
    finally:
        store.close()


def test_provider_change_preserves_prior_export(repository, tmp_path):
    repo = repository
    output = tmp_path / "export"
    output.write_bytes(b"previous")
    build_or_update_graph(repo_root=str(repo), full_rebuild=True)
    identities = iter([("0" * 64, "a" * 64, "b" * 64), ("0" * 64, "c" * 64, "d" * 64)])
    store = GraphStore(get_db_path(repo))
    try:
        with pytest.raises(ValueError, match="changed during QIHSE export"):
            export_qihse(
                store, output, repo, source_identity_provider=lambda repo: next(identities)
            )
        assert output.read_bytes() == b"previous"
    finally:
        store.close()


def test_actual_parse_and_export_progress_events(repository, tmp_path, monkeypatch):
    # Read the pipe after tiny fixture work, safely below its kernel capacity.
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK)
    monkeypatch.setenv("QIHSE_CODEGRAPH_PROGRESS_FD", str(write_fd))
    try:
        output = tmp_path / "export"
        exported(repository, output, True)
        events = [json.loads(raw) for raw in os.read(read_fd, 65536).splitlines()]
        parsed = [e for e in events if e["phase"] == "parse"]
        records = [e for e in events if e["phase"] == "export" and e["unit"] == "records"]
        byte_events = [e for e in events if e["phase"] == "export" and e["unit"] == "bytes"]
        assert parsed[0]["completed"] == 0 and parsed[-1]["completed"] == parsed[-1]["total"] == 2
        assert (
            records[-1]["completed"]
            == records[-1]["total"]
            == len(output.read_bytes().splitlines())
        )
        assert byte_events[-1]["completed"] == output.stat().st_size
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_exact_invalidation_limit_requires_full_rebuild_before_mutation(
    repository,
    tmp_path,
    monkeypatch,
):
    from code_review_graph import incremental

    repo = repository
    exported(repo, tmp_path / "before", True)
    store = GraphStore(get_db_path(repo))
    try:
        previous = [node.qualified_name for node in store.get_all_nodes(exclude_files=False)]
    finally:
        store.close()
    (repo / "callee.py").write_text("def changed_name():\n    return 2\n")
    monkeypatch.setattr(
        incremental,
        "find_dependents",
        lambda *args, **kwargs: incremental.DependentList([], truncated=True),
    )
    result = build_or_update_graph(repo_root=str(repo), exact_dependencies=True)
    assert result["status"] == "error" and result["full_rebuild_required"] is True
    store = GraphStore(get_db_path(repo))
    try:
        assert [
            node.qualified_name for node in store.get_all_nodes(exclude_files=False)
        ] == previous
    finally:
        store.close()


def test_export_rows_hold_one_wal_snapshot_across_a_concurrent_writer(repository, tmp_path):
    repo = repository
    previous = exported(repo, tmp_path / "before", True)
    store = GraphStore(get_db_path(repo))
    writer = GraphStore(get_db_path(repo))
    original_nodes = store.get_all_nodes

    def after_nodes(*args, **kwargs):
        rows = original_nodes(*args, **kwargs)
        writer._conn.execute("UPDATE edges SET confidence=0.125")
        return rows

    store.get_all_nodes = after_nodes
    try:
        output = tmp_path / "concurrent"
        export_qihse(store, output, repo)
        assert output.read_bytes() == previous
        assert {edge.confidence for edge in writer.get_all_edges()} == {0.125}
    finally:
        store.close()
        writer.close()


def test_same_connection_writer_takeover_denies_export_and_preserves_writer(repository, tmp_path):
    repo = repository
    exported(repo, tmp_path / "before", True)
    output = tmp_path / "output"
    output.write_bytes(b"previous")
    store = GraphStore(get_db_path(repo))
    original_nodes = store.get_all_nodes

    def steal_snapshot(*args, **kwargs):
        rows = original_nodes(*args, **kwargs)
        store._begin_immediate()
        store._conn.execute("UPDATE edges SET confidence=0.25")
        store._conn.commit()
        return rows

    store.get_all_nodes = steal_snapshot
    try:
        with pytest.raises(ValueError, match="snapshot changed"):
            export_qihse(store, output, repo)
        assert output.read_bytes() == b"previous"
        assert {edge.confidence for edge in store.get_all_edges()} == {0.25}
    finally:
        store.close()


def test_degraded_or_incomplete_snapshot_is_not_exported(repository, tmp_path, monkeypatch):
    from contextlib import nullcontext

    from code_review_graph.build_state import BUILD_IN_PROGRESS, BUILD_STATE_KEY

    repo = repository
    exported(repo, tmp_path / "before", True)
    output = tmp_path / "output"
    output.write_bytes(b"previous")
    store = GraphStore(get_db_path(repo))
    try:
        store.set_metadata(BUILD_STATE_KEY, BUILD_IN_PROGRESS)
        with pytest.raises(ValueError, match="incomplete source graph"):
            export_qihse(store, output, repo)
        monkeypatch.setattr(store, "_read_transaction", nullcontext)
        with pytest.raises(ValueError, match="could not pin"):
            export_qihse(store, output, repo)
        assert output.read_bytes() == b"previous"
    finally:
        store.close()


def test_identical_materialized_roots_have_identical_node_edge_ids_and_bytes(tmp_path):
    def identity(root):
        return ("0" * 64, "a" * 64, "b" * 64)

    exports = []
    for name in ("view-one", "another-temporary-view"):
        repo = tmp_path / name
        repo.mkdir()
        (repo / ".code-review-graph").mkdir()
        (repo / "callee.py").write_text("def target():\n    return 1\n")
        (repo / "caller.py").write_text(
            "from callee import target\n\ndef caller():\n"
            "    return target() + missing_external()\n",
        )
        result = build_or_update_graph(repo_root=str(repo), full_rebuild=True)
        assert result["status"] == "ok", result
        output = tmp_path / (name + ".jsonl")
        store = GraphStore(get_db_path(repo))
        try:
            export_qihse(store, output, repo, source_identity_provider=identity)
        finally:
            store.close()
        exports.append(output.read_bytes())
    assert exports[0] == exports[1]
    records = [json.loads(line) for line in exports[0].splitlines()]
    assert any(row["t"] == "e" and row["traversable"] for row in records)
    assert any(row["t"] == "e" and not row["traversable"] for row in records)


def test_exact_dependency_mode_includes_references_and_directory_scoped_imports(tmp_path):
    from types import SimpleNamespace

    from code_review_graph.constants import IMPORT_SCOPE_KEY, IMPORT_SCOPE_TREE
    from code_review_graph.incremental import _single_hop_dependents

    source = str(tmp_path / "pkg" / "source.py")
    caller = str(tmp_path / "caller.py")
    package_caller = str(tmp_path / "tree.py")
    reverse_caller = str(tmp_path / "external_test.py")
    node = SimpleNamespace(qualified_name=source + "::symbol")
    reference = SimpleNamespace(kind="REFERENCES", file_path=caller)
    tree = SimpleNamespace(
        kind="IMPORTS_FROM", file_path=package_caller, extra={IMPORT_SCOPE_KEY: IMPORT_SCOPE_TREE}
    )

    class Store:
        def get_nodes_by_file(self, file_path):
            return [node]

        def get_edges_by_target(self, target):
            return (
                [reference]
                if target == node.qualified_name
                else ([tree] if target == str(tmp_path) else [])
            )

        def get_edges_by_source(self, endpoint):
            return (
                [SimpleNamespace(kind="TESTED_BY", file_path=reverse_caller)]
                if endpoint == node.qualified_name
                else []
            )

        def tree_scope_bridges(self, files):
            return {str(tmp_path)}

    assert _single_hop_dependents(Store(), source) == set()
    assert _single_hop_dependents(Store(), source, all_relationships=True) == {
        caller,
        package_caller,
        reverse_caller,
    }
