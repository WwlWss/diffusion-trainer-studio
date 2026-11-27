"""Verify current superproject and gitlinks, not historical fixed SHAs.

Checkout consistency only; not attestation or a clean-worktree policy.
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import subprocess

REQUIRED_GITLINKS = ("frontend", "sd-scripts", "mikazuki/dataset-tag-editor")
CHECKOUT_MARKER = "DTS_E2_CHECKOUT_JSON "
_SHA = re.compile(r"[0-9a-f]{40}\Z")


class CheckoutError(RuntimeError):
    """A missing, malformed or mismatched checkout cannot be qualified."""


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "--literal-pathspecs", "-C", str(root), *args],
            capture_output=True, timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise CheckoutError(f"Git checkout command failed: {root}: {args!r}") from exc


def _sha(value: str) -> str:
    if type(value) is not str or not _SHA.fullmatch(value):
        raise CheckoutError(f"Invalid full Git object identity: {value!r}")
    return value


def _git_sha(root: Path, spec: str) -> str:
    try:
        return _sha(_git(root, "rev-parse", "--verify", spec).decode("ascii").strip())
    except UnicodeError as exc:
        raise CheckoutError(f"Invalid Git identity encoding: {root}: {spec}") from exc


def _require_root(root: Path) -> Path:
    try:
        root = root.resolve(strict=True)
        top = Path(os.fsdecode(_git(root, "rev-parse", "--show-toplevel").rstrip(b"\r\n")))
        if not root.is_dir() or not top.is_absolute() or not root.samefile(top):
            raise CheckoutError(f"Not the actual Git worktree root: {root} (found {top})")
    except OSError as exc:
        raise CheckoutError(f"Missing or invalid Git worktree: {root}") from exc
    return root


def _paths(required_paths):
    paths = tuple(required_paths)
    if not paths or any(
        type(rel) is not str or not rel or not re.fullmatch(r"[A-Za-z0-9_./-]+", rel)
        or PurePosixPath(rel).is_absolute()
        or any(part in ("", ".", "..") for part in rel.split("/"))
        for rel in paths
    ) or len(set(paths)) != len(paths):
        raise CheckoutError(f"Invalid required gitlink paths: {paths!r}")
    return paths


def read_gitlinks(repo_root: Path, commit: str, required_paths=REQUIRED_GITLINKS) -> dict[str, str]:
    """Read exact required paths from the committed tree, not the index."""
    repo_root = _require_root(repo_root)
    commit = _sha(commit)
    paths = _paths(required_paths)
    raw = _git(repo_root, "ls-tree", "-z", "--full-tree", commit, "--", *paths)
    if not raw.endswith(b"\0"):
        raise CheckoutError("Missing or malformed NUL-delimited gitlink records")
    expected = {}
    for entry in raw[:-1].split(b"\0"):
        try:
            metadata, name = entry.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split(" ")
            rel = name.decode("utf-8")
        except (ValueError, UnicodeError) as exc:
            raise CheckoutError("Malformed Git tree record") from exc
        if rel not in paths or rel in expected or mode != "160000" or kind != "commit":
            raise CheckoutError(f"Unexpected or non-gitlink Git tree entry: {rel}: {mode} {kind}")
        expected[rel] = _sha(oid)
    if set(expected) != set(paths):
        raise CheckoutError(f"Required gitlinks missing: {sorted(set(paths) - set(expected))}")
    return {rel: expected[rel] for rel in paths}


def verify_submodule_checkouts(repo_root: Path, expected_gitlinks: dict[str, str]) -> dict[str, str]:
    repo_root = _require_root(repo_root)
    actual = {}
    for rel in _paths(expected_gitlinks):
        expected = _sha(expected_gitlinks[rel])
        child = repo_root / rel
        try:
            if child.is_symlink() or child.resolve(strict=True) != child:
                raise CheckoutError(f"Unexpected submodule path resolution: {rel}")
        except OSError as exc:
            raise CheckoutError(f"Missing submodule checkout: {rel}") from exc
        # An absent .git marker can cause git to discover the parent root.
        child = _require_root(child)
        observed = _git_sha(child, "HEAD^{commit}")
        if observed != expected:
            raise CheckoutError(f"Current submodule mismatch: {rel}: expected {expected}, got {observed}")
        actual[rel] = observed
    return actual


def verify_current_checkout(repo_root: Path) -> dict:
    repo_root = _require_root(repo_root)
    head = _git_sha(repo_root, "HEAD^{commit}")
    tree = _git_sha(repo_root, "HEAD^{tree}")
    event_sha = os.environ.get("GITHUB_SHA")
    if event_sha is None and os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        raise CheckoutError("Missing GitHub event commit in Actions")
    if event_sha is not None and _sha(event_sha) != head:
        raise CheckoutError(f"Event checkout identity mismatch: expected {event_sha}, got {head}")
    expected = read_gitlinks(repo_root, head)
    actual = verify_submodule_checkouts(repo_root, expected)
    if _git_sha(repo_root, "HEAD^{commit}") != head:
        raise CheckoutError("Superproject HEAD changed during checkout verification")
    return {"head": head, "tree": tree, "event_sha": event_sha,
            "expected_gitlinks": expected, "actual_gitlinks": actual}
