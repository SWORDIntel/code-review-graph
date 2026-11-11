"""Additional export formats: JSON, GraphML, Neo4j Cypher, Obsidian, SVG."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from .build_state import BUILD_STATE_KEY, INCOMPLETE_BUILD_STATES
from .graph import GraphStore, _sanitize_name
from .qihse_progress import report as _report_progress
from .qihse_xxh3 import hash_many as _xxh3_many
from .qihse_xxh3 import hash_one as _xxh3_one
from .visualization import export_graph_data

logger = logging.getLogger(__name__)


def _qihse_repo_path(value: str, repo_root: Path) -> str:
    """Require a graph path that stays inside the repository."""
    if Path(value).is_absolute():
        try:
            value = Path(value).relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"graph path is outside repository: {value!r}") from exc
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or any(
        part in ("", ".", "..") for part in value.split("/")
    ):
        raise ValueError(f"unsafe graph path: {value!r}")
    return value


def _qihse_git(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, check=True,
    )
    return result.stdout


_QIHSE_FRESHNESS = "git-status-diff-index-untracked-v2"


def _qihse_identity(repo_root: Path) -> tuple[str, str, str]:
    """Bind tracked changes, index entries and nonignored untracked contents."""
    head = _qihse_git(repo_root, "rev-parse", "HEAD").decode("ascii").strip()
    pathspec = ("--", ".", ":(exclude).code-review-graph")
    status = _qihse_git(
        repo_root, "status", "--porcelain=v1", "-z", "--untracked-files=normal", *pathspec,
    )
    diff = _qihse_git(
        repo_root, "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", *pathspec,
    )
    index = _qihse_git(repo_root, "ls-files", "--stage", "-z", *pathspec)
    untracked = _qihse_git(repo_root, "ls-files", "--others", "--exclude-standard", "-z",
                           *pathspec)
    digest = hashlib.sha256(status + b"\0" + diff + b"\0" + index + b"\0")
    for name in untracked.split(b"\0"):
        if not name:
            continue
        path = os.path.join(os.fsencode(repo_root), name)
        mode = os.lstat(path).st_mode
        content = hashlib.sha256()
        if stat.S_ISLNK(mode):
            kind = b"l"
            content.update(os.readlink(path))
        elif stat.S_ISREG(mode):
            kind = b"f"
            with open(path, "rb") as source:
                for chunk in iter(lambda: source.read(131072), b""):
                    content.update(chunk)
        else:
            raise ValueError(f"unsupported untracked file type: {os.fsdecode(name)!r}")
        digest.update(name + b"\0" + kind + b"\0" + content.hexdigest().encode("ascii") + b"\0")
    return head, hashlib.sha256(status + b"\0" + diff).hexdigest(), digest.hexdigest()


_QIHSE_ID_COLLISION_ATTEMPTS = 256
_QIHSE_ID_COLLISION_DOMAIN = b"qihse-id-collision-xxh3-v1\0"
_QIHSE_PUBLIC_ID = re.compile(r"[0-9a-f]{32}")


def _qihse_id_digest(preimage: bytes) -> str:
    """Standard unseeded XXH3-128 in canonical byte order."""
    return _xxh3_one(preimage)


def _qihse_id_digests(preimages: list[bytes]) -> list[str]:
    return _xxh3_many(preimages)


def _qihse_node_identity(qualified_name: str) -> bytes:
    name = qualified_name.encode("utf-8")
    return b"qihse-node-128-v1\0" + len(name).to_bytes(8, "big") + name


def _qihse_checked_id(
    preimage: bytes, identities: dict[str, bytes], initial: str | None = None,
) -> str:
    """Resolve collisions by deterministic rehashing; never merge distinct facts.

    Attempt zero hashes the original framed identity. Retries hash the explicit
    collision domain, original byte length and bytes, and an unsigned 64-bit
    big-endian counter. Canonical export order makes allocation reproducible.
    The registry always compares the complete original identity, not a digest.
    """
    for attempt in range(_QIHSE_ID_COLLISION_ATTEMPTS):
        framed = preimage if attempt == 0 else (
            _QIHSE_ID_COLLISION_DOMAIN + len(preimage).to_bytes(8, "big") + preimage
            + attempt.to_bytes(8, "big")
        )
        digest = initial if attempt == 0 and initial is not None else _qihse_id_digest(framed)
        if not isinstance(digest, str) or not _QIHSE_PUBLIC_ID.fullmatch(digest):
            raise ValueError("invalid QIHSE public identity digest")
        previous = identities.get(digest)
        if previous is None or previous == preimage:
            identities[digest] = preimage
            return digest
    raise ValueError("QIHSE graph identity collision resolution exhausted")


def _qihse_edge_identity(fields: tuple[str, ...], occurrence: int) -> bytes:
    """Versioned, length-framed semantic site, independent of SQLite row IDs."""
    framed = bytearray(b"qihse-edge-site-v2\0")
    for field in fields:
        encoded = field.encode("utf-8")
        framed.extend(len(encoded).to_bytes(8, "big"))
        framed.extend(encoded)
    framed.extend(occurrence.to_bytes(8, "big"))
    return bytes(framed)




def _qihse_qualified_identity(value: str, repo_root: Path) -> str:
    """Remove only this workspace's absolute prefix from qualified identities.

    A materialized editor view has another temporary directory on every run.
    Persisting that prefix would rename every public ID without a source change.
    External unresolved identities remain opaque, not basename-collapsed.
    """
    file_part, separator, symbol = value.partition('::')
    path = Path(file_part)
    if path.is_absolute():
        try:
            file_part = path.relative_to(repo_root).as_posix()
        except ValueError:
            return value
    return file_part + separator + symbol


def _qihse_graph_records(store: GraphStore):
    """Capture complete source rows from one non-degrading SQLite snapshot."""
    if not isinstance(store, GraphStore):
        # Lightweight immutable source providers used by callers/tests do not
        # have a SQLite connection or a build-state journal.
        return store.get_all_nodes(exclude_files=False), store.get_all_edges()
    with store._read_transaction():
        token = store._active_read_txn
        changes = store._conn.total_changes
        if token is None or not store._conn.in_transaction:
            raise ValueError('QIHSE export could not pin a source-index read snapshot')
        if store.get_metadata(BUILD_STATE_KEY) in INCOMPLETE_BUILD_STATES:
            raise ValueError('QIHSE export refuses an incomplete source graph')
        nodes = store.get_all_nodes(exclude_files=False)
        edges = store.get_all_edges()
        if (store._active_read_txn is not token or not store._conn.in_transaction
                or store._conn.total_changes != changes):
            raise ValueError('Source-index snapshot changed during QIHSE row capture')
    return nodes, edges


def export_qihse(
    store: GraphStore, output_path: Path, repo_root: Path, *,
    source_identity_provider: Callable[[Path], tuple[str, str, str]] | None = None,
) -> Path:
    """Export a repository-bound, integrity-checked qihse-codegraph-v1 stream."""
    repo_root = repo_root.resolve()
    identity_provider = source_identity_provider or _qihse_identity
    head, worktree, worktree_v2 = identity_provider(repo_root)
    nodes, edges = _qihse_graph_records(store)
    nodes.sort(key=lambda node: _qihse_qualified_identity(node.qualified_name, repo_root))
    source_hashes: dict[str, str] = {}
    for node in nodes:
        if not node.file_hash:
            if node.kind == "File" and not node.extra.get("virtual"):
                raise ValueError(f"indexed source has no content hash: {node.file_path!r}")
            continue
        if len(node.file_hash) != 64 or any(c not in "0123456789abcdef" for c in node.file_hash):
            raise ValueError(f"invalid indexed source hash: {node.file_path!r}")
        path = _qihse_repo_path(node.file_path, repo_root)
        if path in source_hashes and source_hashes[path] != node.file_hash:
            raise ValueError(f"inconsistent indexed source hashes: {path!r}")
        source_hashes[path] = node.file_hash

    def verify_sources() -> None:
        for path, expected in source_hashes.items():
            source = repo_root / path
            if not source.resolve().is_relative_to(repo_root.resolve()) or not source.is_file():
                raise ValueError(f"indexed source is unavailable or outside repository: {path!r}")
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(131072), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(
                    f"indexed source changed; refresh the graph before export: {path!r}",
                )

    verify_sources()
    identities: dict[str, bytes] = {}
    node_ids: dict[str, str] = {}
    qualified_names = [_qihse_qualified_identity(node.qualified_name, repo_root) for node in nodes]
    node_frames = [_qihse_node_identity(name) for name in qualified_names]
    node_candidates = _qihse_id_digests(node_frames)
    for name, frame, candidate in zip(qualified_names, node_frames, node_candidates, strict=True):
        if name in node_ids:
            raise ValueError("duplicate QIHSE qualified node identity")
        node_ids[name] = _qihse_checked_id(
            frame, identities, candidate,
        )

    # Sort equal sites by their exported evidence, then number occurrences. Identical
    # parallel records remain distinct without introducing database allocation order.
    edge_rows = []
    for edge in edges:
        path = _qihse_repo_path(edge.file_path, repo_root)
        source_name = _qihse_qualified_identity(edge.source_qualified, repo_root)
        target_name = _qihse_qualified_identity(edge.target_qualified, repo_root)
        site = (edge.kind, source_name, target_name, path,
                str(max(0, edge.line)))
        source, target = node_ids.get(source_name, ""), node_ids.get(
            target_name, "",
        )
        tier = edge.confidence_tier or ""
        resolution = edge.target_resolution or (
            "direct" if source and target else "unresolved"
        )
        confidence = max(0.0, min(1.0, edge.confidence))
        evidence = json.dumps([tier, resolution, confidence], separators=(",", ":"))
        edge_rows.append((site, evidence, edge))
    edge_rows.sort(key=lambda row: (row[0], row[1]))
    edge_frames = []
    previous_site = None
    occurrence = 0
    for site, _evidence, _edge in edge_rows:
        occurrence = occurrence + 1 if site == previous_site else 0
        previous_site = site
        edge_frames.append(_qihse_edge_identity(site, occurrence))
    edge_candidates = _qihse_id_digests(edge_frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    digest = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)

            records_written = 0
            bytes_written = 0
            total_records = len(nodes) + len(edges) + 2
            _report_progress("export", "records", 0, total_records)

            def write_record(record: dict, *, final: bool = False) -> None:
                nonlocal records_written, bytes_written
                raw = (
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                if len(raw) > 262144:
                    raise ValueError("QIHSE record exceeds 262144 bytes")
                handle.write(raw)
                records_written += 1
                bytes_written += len(raw)
                if final or records_written % 256 == 0:
                    _report_progress("export", "records", records_written, total_records)
                    _report_progress("export", "bytes", bytes_written)
                if not final:
                    digest.update(raw)

            write_record({
                "t": "h", "format": "qihse-codegraph-v1", "version": 1,
                "identity": "qihse-semantic-xxh3-128-v1",
                "id_algorithm": "xxh3-128", "id_bits": 128,
                "id_seed": 0, "id_encoding": "xxh128-canonical",
                "id_collision": "counter-xxh3-128-v1",
                "id_collision_attempts": _QIHSE_ID_COLLISION_ATTEMPTS,
                "head": head, "worktree": worktree,
                "freshness": _QIHSE_FRESHNESS, "worktree_v2": worktree_v2,
            })
            for node in nodes:
                file_path = _qihse_repo_path(node.file_path, repo_root)
                node_name = (_qihse_qualified_identity(node.name, repo_root)
                             if node.kind == "File" else node.name)
                if len(node.kind) > 64 or len(node_name) > 512:
                    raise ValueError("graph node exceeds QIHSE field limit")
                node_id = node_ids[_qihse_qualified_identity(node.qualified_name, repo_root)]
                write_record({
                    "t": "n", "id": node_id,
                    "file": file_path, "kind": node.kind, "name": node_name,
                    "line_start": max(0, node.line_start),
                    "line_end": max(0, node.line_end),
                    "language": node.language, "is_test": node.is_test,
                    "file_hash": node.file_hash or "",
                })
            unresolved_edges = 0
            for (site, _evidence, edge), frame, candidate in zip(
                edge_rows, edge_frames, edge_candidates, strict=True,
            ):
                file_path = _qihse_repo_path(edge.file_path, repo_root)
                if len(edge.kind) >= 32:
                    raise ValueError("graph edge relation exceeds QIHSE field limit")
                source = node_ids.get(site[1], "")
                target = node_ids.get(site[2], "")
                tier = edge.confidence_tier or ""
                resolution = edge.target_resolution or (
                    "direct" if source and target else "unresolved"
                )
                confidence = max(0.0, min(1.0, edge.confidence))
                traversable = bool(
                    source and target and tier == "EXTRACTED"
                    and resolution == "direct" and confidence >= 0.8
                )
                unresolved_edges += not traversable
                edge_id = _qihse_checked_id(
                    frame, identities, candidate,
                )
                write_record({
                    "t": "e", "id": edge_id, "source": source,
                    "target": target, "relation": edge.kind, "file": file_path,
                    "line": max(0, edge.line), "confidence": confidence,
                    "tier": tier, "resolution": resolution,
                    "traversable": traversable,
                })
            write_record({
                "t": "end", "nodes": len(nodes), "edges": len(edges),
                "unresolved_edges": unresolved_edges, "sha256": digest.hexdigest(),
            }, final=True)
            handle.flush()
            os.fsync(handle.fileno())
        verify_sources()
        if identity_provider(repo_root) != (head, worktree, worktree_v2):
            raise ValueError("repository changed during QIHSE export")
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output_path


class MissingOptionalDependencyError(ImportError):
    """An export needs a package that a default install does not ship.

    Subclasses ``ImportError`` so existing callers keep working, but is
    specific enough that a CLI can catch it and print the one-line install
    hint instead of a traceback — without swallowing an ImportError raised
    by a genuinely broken install of a required package.
    """


# -------------------------------------------------------------------
# JSON export
# -------------------------------------------------------------------

def export_json(store: GraphStore, output_path: Path) -> Path:
    """Export the complete local graph payload as UTF-8 JSON atomically.

    The payload can contain absolute local paths and code-structure metadata.
    Callers are responsible for deciding whether it is safe to publish.
    """
    data = export_graph_data(store)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    logger.info("JSON exported to %s", output_path)
    return output_path


# -------------------------------------------------------------------
# GraphML export (for Gephi, yEd, Cytoscape)
# -------------------------------------------------------------------

#: The GraphML 1.0 namespace. Every GraphML reader keys off this exact URI.
GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
#: Canonical location of the schema, for the ``xsi:schemaLocation`` pair.
GRAPHML_SCHEMA = "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"

#: Characters XML 1.0 forbids outright — not even as a character reference.
#:
#: This is exactly the complement of the XML 1.0 ``Char`` production below
#: U+0020: everything from U+0000 to U+001F except tab, newline and carriage
#: return. U+007F (DEL) is deliberately *not* here. ``Char`` admits the whole
#: of ``[#x20-#xD7FF]``, so DEL is a legal XML 1.0 character, and
#: ``_sanitize_name`` keeps it in a node name. Dropping it would rewrite the
#: identity this export carries: two names differing only by a DEL would come
#: back from the file as one, which is the newline-collision bug this export
#: already avoids for ``node/@id``, in a smaller form.
_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _xml_text(value: object) -> str:
    """Escape a value for XML element content.

    Drops the control characters XML 1.0 cannot represent at all, then
    escapes the markup characters. Element content (unlike an attribute
    value) is not whitespace-normalised, so a newline survives verbatim.
    """
    return html.escape(_XML_ILLEGAL.sub("", str(value)), quote=False)


def export_graphml(store: GraphStore, output_path: Path) -> Path:
    """Export the graph as GraphML XML for Gephi/yEd/Cytoscape.

    The document declares the official GraphML namespace and validates
    against the GraphML 1.0 schema. That schema types ``node/@id`` as an
    ``NMTOKEN``, which cannot hold the ``/`` in a qualified name, and XML
    attribute-value normalisation would fold a newline inside a name onto a
    space and silently merge two distinct nodes. So the identity travels in
    a ``<data key="qualified_name">`` element and the ids are synthetic.

    Returns the path to the written file.
    """
    data = export_graph_data(store)
    nodes = data["nodes"]
    edges = data["edges"]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<graphml xmlns="{GRAPHML_NS}"',
        '  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        f'  xsi:schemaLocation="{GRAPHML_NS} {GRAPHML_SCHEMA}">',
        '  <key id="qualified_name" for="node" '
        'attr.name="qualified_name" attr.type="string"/>',
        '  <key id="name" for="node" attr.name="name" '
        'attr.type="string"/>',
        '  <key id="kind" for="node" attr.name="kind" '
        'attr.type="string"/>',
        '  <key id="file" for="node" attr.name="file" '
        'attr.type="string"/>',
        '  <key id="language" for="node" attr.name="language" '
        'attr.type="string"/>',
        '  <key id="community" for="node" attr.name="community" '
        'attr.type="int"/>',
        '  <key id="edge_kind" for="edge" attr.name="kind" '
        'attr.type="string"/>',
        '  <graph id="code-review-graph" edgedefault="directed">',
    ]

    # Qualified name -> NMTOKEN id, so edges can reference declared nodes.
    node_ids: dict[str, str] = {}
    for index, n in enumerate(nodes):
        node_ids.setdefault(n["qualified_name"], f"n{index}")

    for index, n in enumerate(nodes):
        lines.append(f'    <node id="n{index}">')
        lines.append('      <data key="qualified_name">'
                     f'{_xml_text(n["qualified_name"])}</data>')
        lines.append('      <data key="name">'
                     f'{_xml_text(n.get("name", ""))}</data>')
        lines.append('      <data key="kind">'
                     f'{_xml_text(n.get("kind", ""))}</data>')
        lines.append('      <data key="file">'
                     f'{_xml_text(n.get("file_path", ""))}</data>')
        lines.append('      <data key="language">'
                     f'{_xml_text(n.get("language", "") or "")}</data>')
        cid = n.get("community_id")
        if cid is not None:
            lines.append(f'      <data key="community">{int(cid)}</data>')
        lines.append('    </node>')

    written_edges = 0
    for e in edges:
        src = node_ids.get(e["source"])
        tgt = node_ids.get(e["target"])
        if src is None or tgt is None:
            # A dangling endpoint would break the schema's keyref on
            # edge/@source and edge/@target, so drop the edge instead.
            logger.debug(
                "GraphML: dropping edge with an unknown endpoint (%r -> %r)",
                e["source"], e["target"],
            )
            continue
        lines.append(
            f'    <edge id="e{written_edges}" source="{src}" '
            f'target="{tgt}">'
        )
        lines.append('      <data key="edge_kind">'
                     f'{_xml_text(e.get("kind", ""))}</data>')
        lines.append('    </edge>')
        written_edges += 1

    lines.append('  </graph>')
    lines.append('</graphml>')

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("GraphML exported to %s (%d nodes, %d edges)",
                output_path, len(nodes), written_edges)
    return output_path


# -------------------------------------------------------------------
# Neo4j Cypher export
# -------------------------------------------------------------------

def export_neo4j_cypher(store: GraphStore, output_path: Path) -> Path:
    """Export the graph as Neo4j Cypher CREATE statements.

    Returns the path to the written file.
    """
    data = export_graph_data(store)
    nodes = data["nodes"]
    edges = data["edges"]

    lines = [
        "// Generated by code-review-graph",
        "// Import: paste into Neo4j Browser or run via cypher-shell",
        "",
    ]

    # Create nodes
    for n in nodes:
        kind = n.get("kind", "Node")
        props = {
            "qualified_name": n["qualified_name"],
            "name": n.get("name", ""),
            "file_path": n.get("file_path", ""),
            "language": n.get("language", "") or "",
        }
        cid = n.get("community_id")
        if cid is not None:
            props["community_id"] = cid
        props_str = _cypher_props(props)
        lines.append(f"CREATE (:{kind} {props_str});")

    lines.append("")

    # Create edges via MATCH
    for e in edges:
        kind = e.get("kind", "RELATES_TO")
        src_qn = _cypher_escape(e["source"])
        tgt_qn = _cypher_escape(e["target"])
        lines.append(
            f"MATCH (a {{qualified_name: '{src_qn}'}}), "
            f"(b {{qualified_name: '{tgt_qn}'}}) "
            f"CREATE (a)-[:{kind}]->(b);"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Neo4j Cypher exported to %s (%d nodes, %d edges)",
                output_path, len(nodes), len(edges))
    return output_path


def _cypher_escape(s: str) -> str:
    """Escape a string for Cypher single-quoted literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _cypher_props(d: dict) -> str:
    """Format a dict as Cypher property map.

    ``bool`` is tested before ``int`` because it is a subclass of ``int``:
    the other order makes the boolean branch unreachable and emits Python's
    ``True``/``False`` instead of Cypher's ``true``/``false``.
    """
    parts = []
    for k, v in d.items():
        if isinstance(v, str):
            parts.append(f"{k}: '{_cypher_escape(v)}'")
        elif isinstance(v, bool):
            parts.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}: {v}")
    return "{" + ", ".join(parts) + "}"


# -------------------------------------------------------------------
# Obsidian vault export
# -------------------------------------------------------------------

def export_obsidian_vault(
    store: GraphStore, output_dir: Path
) -> Path:
    """Export the graph as an Obsidian vault with wikilinks.

    Creates:
    - One .md per node with YAML frontmatter and [[wikilinks]]
    - _COMMUNITY_*.md overview notes per community
    - _INDEX.md with links to all nodes

    Returns the output directory path.
    """
    data = export_graph_data(store)
    nodes = data["nodes"]
    edges = data["edges"]
    communities = data.get("communities", [])

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build adjacency for wikilinks
    neighbors: dict[str, list[dict]] = {}
    for e in edges:
        src = e["source"]
        tgt = e["target"]
        kind = e.get("kind", "RELATES_TO")
        neighbors.setdefault(src, []).append(
            {"target": tgt, "kind": kind}
        )
        neighbors.setdefault(tgt, []).append(
            {"target": src, "kind": kind}
        )

    # Node name -> slug mapping
    slugs: dict[str, str] = {}
    for n in nodes:
        slug = _obsidian_slug(n.get("name", n["qualified_name"]))
        # Handle collisions
        base_slug = slug
        counter = 1
        while slug in slugs.values():
            slug = f"{base_slug}-{counter}"
            counter += 1
        slugs[n["qualified_name"]] = slug

    # Write node pages
    for n in nodes:
        qn = n["qualified_name"]
        slug = slugs[qn]
        name = n.get("name", qn)

        frontmatter = {
            "kind": n.get("kind", ""),
            "file": n.get("file_path", ""),
            "language": n.get("language", "") or "",
            "community": n.get("community_id"),
            "tags": [n.get("kind", "").lower()],
        }

        lines = ["---"]
        for k, v in frontmatter.items():
            if isinstance(v, list):
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - {_yaml_scalar(item)}")
            elif v is not None:
                lines.append(f"{k}: {_yaml_scalar(v)}")
        lines.append("---")
        lines.append(f"# {_sanitize_name(name)}")
        lines.append("")
        lines.append(f"**Kind:** {n.get('kind', '')}")
        lines.append(f"**File:** `{n.get('file_path', '')}`")
        lines.append("")

        # Wikilinks to neighbors
        nbrs = neighbors.get(qn, [])
        if nbrs:
            lines.append("## Connections")
            lines.append("")
            seen = set()
            for nb in nbrs:
                tgt_slug = slugs.get(nb["target"])
                if tgt_slug and tgt_slug not in seen:
                    seen.add(tgt_slug)
                    tgt_name = tgt_slug.replace("-", " ").title()
                    lines.append(
                        f"- {nb['kind']}: "
                        f"[[{tgt_slug}|{tgt_name}]]"
                    )

        page_path = output_dir / f"{slug}.md"
        page_path.write_text("\n".join(lines), encoding="utf-8")

    # Write community overview pages
    community_map: dict[int, list[str]] = {}
    for n in nodes:
        cid = n.get("community_id")
        if cid is not None:
            community_map.setdefault(cid, []).append(
                n["qualified_name"]
            )

    for c in communities:
        cid = c.get("id")
        cname = c.get("name", f"community-{cid}")
        members = community_map.get(cid, [])

        lines = [f"# Community: {_sanitize_name(cname)}", ""]
        lines.append(f"**Size:** {c.get('size', len(members))}")
        lines.append(f"**Cohesion:** {c.get('cohesion', 0):.2f}")
        lang = c.get("dominant_language", "")
        if lang:
            lines.append(f"**Language:** {lang}")
        lines.append("")
        lines.append("## Members")
        lines.append("")
        for qn in members[:50]:
            slug = slugs.get(qn)
            if slug:
                lines.append(f"- [[{slug}]]")

        page_path = output_dir / f"_COMMUNITY_{cid}.md"
        page_path.write_text("\n".join(lines), encoding="utf-8")

    # Write index
    index_lines = ["# Code Graph Index", ""]
    index_lines.append(f"**Nodes:** {len(nodes)}")
    index_lines.append(f"**Edges:** {len(edges)}")
    index_lines.append(
        f"**Communities:** {len(communities)}"
    )
    index_lines.append("")
    index_lines.append("## All Nodes")
    index_lines.append("")
    for n in sorted(nodes, key=lambda x: x.get("name", "")):
        slug = slugs.get(n["qualified_name"])
        if slug:
            index_lines.append(
                f"- [[{slug}]] ({n.get('kind', '')})"
            )

    (output_dir / "_INDEX.md").write_text(
        "\n".join(index_lines), encoding="utf-8"
    )

    logger.info(
        "Obsidian vault exported to %s (%d pages)",
        output_dir, len(nodes)
    )
    return output_dir


def _yaml_scalar(value: object) -> str:
    """Render a frontmatter value as a scalar any YAML parser will accept.

    A bare scalar cannot carry ``": "``, a leading ``#``, a quote or a
    newline, and a legal POSIX file path can contain all of them, so
    anything that is not a number is emitted double-quoted. JSON string
    syntax is a subset of YAML's double-quoted style, so ``json.dumps``
    produces exactly the escaping YAML expects. ``bool`` is tested before
    ``int`` because it is a subclass of ``int``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=False)


def _obsidian_slug(name: str) -> str:
    """Convert a name to an Obsidian-friendly filename slug."""
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:100] or "unnamed"


# -------------------------------------------------------------------
# SVG export (matplotlib-based)
# -------------------------------------------------------------------

def export_svg(store: GraphStore, output_path: Path) -> Path:
    """Export a static SVG graph visualization.

    Requires matplotlib (optional dependency).
    Returns the path to the written file.

    Raises:
        MissingOptionalDependencyError: If matplotlib is not installed.
            Callers on a user-facing surface should print the message and
            exit non-zero rather than let it become a traceback.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "SVG export requires matplotlib. "
            'Run: pip install "code-review-graph[eval]"'
        ) from exc

    import networkx as nx

    data = export_graph_data(store)
    nodes_data = data["nodes"]
    edges_data = data["edges"]

    nxg: nx.DiGraph = nx.DiGraph()  # type: ignore[type-arg]
    for n in nodes_data:
        nxg.add_node(
            n["qualified_name"],
            label=n.get("name", ""),
            kind=n.get("kind", ""),
        )
    for e in edges_data:
        if e["source"] in nxg and e["target"] in nxg:
            nxg.add_edge(e["source"], e["target"])

    if nxg.number_of_nodes() == 0:
        raise ValueError("Graph is empty, nothing to export")

    # Color by kind
    kind_colors = {
        "File": "#6c757d",
        "Class": "#0d6efd",
        "Function": "#198754",
        "Type": "#ffc107",
        "Test": "#dc3545",
    }
    colors = [
        kind_colors.get(
            nxg.nodes[n].get("kind", ""), "#adb5bd"
        )
        for n in nxg.nodes()
    ]

    # Node names are code identifiers, never TeX. With matplotlib's default
    # ``text.parse_math`` a name carrying two '$' (legal in JavaScript, PHP,
    # Perl, shell and in file paths) is parsed as mathtext and an unknown
    # symbol aborts the whole export, so every label is drawn literally.
    # ``parse_math`` is read when a Text artist is built *and* when it is
    # rendered, so the context has to cover savefig too.
    with matplotlib.rc_context({"text.parse_math": False}):
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        pos = nx.spring_layout(
            nxg, k=2 / (nxg.number_of_nodes() ** 0.5),
            iterations=50, seed=42
        )

        # Limit labels to avoid clutter
        labels = {}
        if nxg.number_of_nodes() <= 100:
            labels = {
                n: nxg.nodes[n].get("label", n.split("::")[-1])
                for n in nxg.nodes()
            }

        nx.draw_networkx_nodes(
            nxg, pos, ax=ax, node_color=colors,
            node_size=30, alpha=0.8
        )
        nx.draw_networkx_edges(
            nxg, pos, ax=ax, alpha=0.2,
            arrows=True, arrowsize=5
        )
        if labels:
            nx.draw_networkx_labels(
                nxg, pos, labels=labels, ax=ax,
                font_size=6
            )

        ax.set_title("Code Review Graph", fontsize=14)
        ax.axis("off")

        fig.savefig(
            str(output_path), format="svg",
            bbox_inches="tight", dpi=150
        )
        plt.close(fig)

    logger.info("SVG exported to %s (%d nodes)",
                output_path, nxg.number_of_nodes())
    return output_path
