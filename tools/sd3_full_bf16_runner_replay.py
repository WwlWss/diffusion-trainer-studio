"""C4-C2-P1-B durable one-use nonce MECHANISM, never Runner authority.

SQLite unique constraints and BEGIN IMMEDIATE serialize claims across OS
processes. This class accepts an injected TEST path: it is NOT a production
protected trust root or permission to call original Accelerator.load_state.
No process that can edit the DB may treat this ledger as an authority.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Any


_AUDIENCE = "dts.sd3c2b.c4c2.resume-load"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_SCHEMA = "dts.sd3c2b.c4c2p1.nonce-ledger-v1"


def _reject(reason: str) -> None:
    raise ValueError("C4-C2-P1-B replay mechanism refused: " + reason)


def _safe_path(raw: Path | str) -> Path:
    if not isinstance(raw, (Path, str)):
        _reject("ledger path must be a Path or string")
    p = Path(raw)
    if not p.is_absolute() or any(part in (".", "..") for part in p.parts):
        _reject("ledger path not absolute/canonical")
    if p.resolve(strict=False) != p:
        _reject("ledger path has symlink/reparse/alias components")
    if not p.parent.is_dir():
        _reject("ledger parent missing")
    if p.exists():
        info = p.lstat()
        if not stat.S_ISREG(info.st_mode):
            _reject("ledger is not a regular file")
    return p


def _hex64(value: Any, field: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        _reject("invalid " + field)
    return value


def _identity(value: Any, field: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        _reject("invalid " + field)
    return value


class RunnerNonceLedger:
    """Context-managed atomic durable replay primitive, scoped Host ONLY."""

    def __init__(self, db_path: Path | str):
        self._path = _safe_path(db_path)
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> "RunnerNonceLedger":
        if self._connection is not None:
            _reject("ledger already entered")
        _safe_path(self._path)
        try:
            connection = sqlite3.connect(
                str(self._path), timeout=10.0, isolation_level=None
            )
            self._connection = connection
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA synchronous=FULL")
            # Initialization and all nonce claims share one SQLite writer lock.
            connection.execute("BEGIN IMMEDIATE")
            check = connection.execute("PRAGMA integrity_check").fetchone()
            if check != ("ok",):
                _reject("ledger integrity validation failed")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS nonce_ledger_meta ("
                "schema_id TEXT NOT NULL PRIMARY KEY)"
            )
            rows = connection.execute(
                "SELECT schema_id FROM nonce_ledger_meta"
            ).fetchall()
            if rows:
                if rows != [(_SCHEMA,)]:
                    _reject("ledger schema is untrusted")
            else:
                connection.execute(
                    "INSERT INTO nonce_ledger_meta (schema_id) VALUES (?)",
                    (_SCHEMA,),
                )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS consumed_nonces ("
                "one_use_nonce TEXT PRIMARY KEY NOT NULL,"
                "token_id TEXT NOT NULL UNIQUE,"
                "issuer_id TEXT NOT NULL,"
                "audience TEXT NOT NULL,"
                "ticket_sha256 TEXT NOT NULL,"
                "issued_at_unix INTEGER NOT NULL,"
                "expires_at_unix INTEGER NOT NULL)"
            )
            # Ensure existing databases do not replace the unique constraints.
            cols = {
                row[1] for row in connection.execute(
                    "PRAGMA table_info(consumed_nonces)"
                ).fetchall()
            }
            expected_cols = {
                "one_use_nonce", "token_id", "issuer_id", "audience",
                "ticket_sha256", "issued_at_unix", "expires_at_unix",
            }
            if cols != expected_cols:
                _reject("ledger existing table does not match schema")
            unique_columns = {
                tuple(row[2] for row in connection.execute(
                    f"PRAGMA index_info({row[1]})"
                ).fetchall())
                for row in connection.execute(
                    "PRAGMA index_list(consumed_nonces)"
                ).fetchall()
                if row[2]
            }
            if ("one_use_nonce",) not in unique_columns or (
                "token_id",) not in unique_columns:
                _reject("ledger uniqueness constraints missing")
            connection.execute("COMMIT")
            return self
        except (sqlite3.Error, ValueError, OSError) as exc:
            self.close()
            if isinstance(exc, ValueError):
                raise
            _reject("ledger could not be opened safely: " + type(exc).__name__)

    def consume_once(
        self, *,
        issuer_id: str,
        audience: str,
        ticket_sha256: str,
        one_use_nonce: str,
        token_id: str,
        issued_at_unix: int,
        expires_at_unix: int,
        now_unix: int,
    ) -> dict[str, Any]:
        connection = self._connection
        if connection is None:
            _reject("ledger is not open")
        _identity(issuer_id, "issuer_id")
        if type(audience) is not str or audience != _AUDIENCE:
            _reject("wrong audience")
        _hex64(ticket_sha256, "ticket_sha256")
        _hex64(one_use_nonce, "one_use_nonce")
        _hex64(token_id, "token_id")
        if (type(issued_at_unix) is not int
                or type(expires_at_unix) is not int
                or type(now_unix) is not int
                or issued_at_unix < 0
                or not issued_at_unix <= now_unix < expires_at_unix
                or expires_at_unix - issued_at_unix > 60):
            _reject("invalid or expired one-use time window")
        _safe_path(self._path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO consumed_nonces "
                "(one_use_nonce, token_id, issuer_id, audience, "
                "ticket_sha256, issued_at_unix, expires_at_unix) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (one_use_nonce, token_id, issuer_id, audience,
                 ticket_sha256, issued_at_unix, expires_at_unix),
            )
            connection.execute("COMMIT")
        except sqlite3.Error as exc:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            _reject("nonce replay or ledger transaction failure: "
                    + type(exc).__name__)
        return {
            "status": "nonce_consumed_unqualified",
            "physical_cuda_qualified": False,
        }

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
            finally:
                connection.close()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = ["RunnerNonceLedger"]
