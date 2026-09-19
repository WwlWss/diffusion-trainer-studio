"""Exposure-time Multi-Caption resolver shared by stable/dev trainers.

The resolver compiles source captions into a temporary SQLite index once during
Dataset construction.  DataLoader workers therefore pickle only a small
resolver object instead of a potentially multi-gigabyte JSON dictionary, and
Separate Files / Multi-Line modes do not reopen caption files on every exposure.
"""

from __future__ import annotations

import atexit
from collections import OrderedDict
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import tempfile
from typing import Any, Iterable
import uuid


@dataclass
class CaptionProcessingOptions:
    caption_separator: str = ","
    secondary_separator: str | None = None
    enable_wildcard: bool = False
    caption_prefix: str | None = None
    caption_suffix: str | None = None
    shuffle_caption: bool = False
    keep_tokens: int = 0
    keep_tokens_separator: str = ""
    token_warmup_min: int = 1
    token_warmup_step: float = 0
    caption_dropout_rate: float = 0
    caption_dropout_every_n_epochs: int = 0
    caption_tag_dropout_rate: float = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "CaptionProcessingOptions":
        raw = dict(raw or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError("Multi-Caption: unsupported processing fields: " + ", ".join(unknown))
        return cls(**raw)


@dataclass(frozen=True)
class CaptionGroup:
    name: str
    enabled: bool
    weight: float
    source: dict[str, Any]
    processing: CaptionProcessingOptions


@dataclass(frozen=True)
class ResolvedCaption:
    group_name: str
    caption: str
    processing: CaptionProcessingOptions


def _cleanup_index(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


class MultiCaptionResolver:
    CACHE_LIMIT = 2048

    def __init__(
        self,
        config_path: str,
        policy: dict[str, Any],
        image_infos: Iterable[Any],
        *,
        deterministic: bool = False,
    ):
        self.config_path = str(config_path)
        self.policy = policy
        self.deterministic = bool(deterministic)
        if policy.get("version") != 1:
            raise ValueError(f"Multi-Caption: unsupported policy version {policy.get('version')!r}")
        if (policy.get("selection") or {}).get("mode") != "weighted_one":
            raise ValueError("Multi-Caption: only weighted_one selection is supported")

        self.storage = dict(policy.get("storage") or {})
        self.storage_mode = str(self.storage.get("mode") or "")
        if self.storage_mode not in {"files", "multiline", "json", "jsonl"}:
            raise ValueError(f"Multi-Caption: unsupported storage mode {self.storage_mode!r}")

        raw_groups = policy.get("groups")
        if not isinstance(raw_groups, dict) or not raw_groups:
            raise ValueError("Multi-Caption: at least one caption group is required")
        self.groups: list[CaptionGroup] = []
        self._groups_by_name: dict[str, CaptionGroup] = {}
        for name, raw in raw_groups.items():
            raw = dict(raw or {})
            group = CaptionGroup(
                name=str(name),
                enabled=bool(raw.get("enabled", True)),
                weight=float(raw.get("weight", 1.0)),
                source=dict(raw.get("source") or {}),
                processing=CaptionProcessingOptions.from_dict(raw.get("processing")),
            )
            if group.weight < 0:
                raise ValueError(f"Multi-Caption: group {group.name!r} has negative weight")
            self.groups.append(group)
            self._groups_by_name[group.name] = group

        self._image_key_mode = str(self.storage.get("image_key_mode") or "relative_path")
        self._json_root: Path | None = None
        self._conn: sqlite3.Connection | None = None
        self._candidate_cache: OrderedDict[str, dict[str, str]] = OrderedDict()

        infos = [info for info in image_infos if not bool(getattr(info, "is_reg", False))]
        self._db_path = self._build_index(infos)

    @classmethod
    def from_file(
        cls,
        config_path: str,
        image_infos: Iterable[Any],
        *,
        deterministic: bool = False,
    ) -> "MultiCaptionResolver":
        path = Path(config_path)
        if not path.is_file():
            raise ValueError(f"Multi-Caption config does not exist: {config_path}")
        try:
            policy = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Multi-Caption config could not be read: {config_path}: {exc}") from exc
        if not isinstance(policy, dict):
            raise ValueError("Multi-Caption config must contain a JSON object")
        return cls(config_path, policy, image_infos, deterministic=deterministic)

    def clone(self, *, deterministic: bool) -> "MultiCaptionResolver":
        cloned = copy.deepcopy(self)
        cloned.deterministic = bool(deterministic)
        cloned._conn = None
        cloned._candidate_cache = OrderedDict()
        return cloned

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_conn"] = None
        state["_candidate_cache"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._conn = None
        self._candidate_cache = OrderedDict()

    @staticmethod
    def _identity(image_path: str) -> str:
        return os.path.normcase(os.path.abspath(image_path))

    def _candidate_paths(self, image_path: str, extension: str) -> list[str]:
        base_name = os.path.splitext(image_path)[0]
        base_name_face_det = base_name
        tokens = base_name.split("_")
        if len(tokens) >= 5:
            base_name_face_det = "_".join(tokens[:-4])
        paths = [base_name + extension]
        fallback = base_name_face_det + extension
        if fallback != paths[0]:
            paths.append(fallback)
        return paths

    @staticmethod
    def _read_text(path: str) -> str | None:
        if not os.path.isfile(path):
            return None
        try:
            return Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Multi-Caption caption file is not UTF-8: {path}") from exc

    @staticmethod
    def _json_pointer(record: Any, key: str) -> Any:
        # A leading slash opts into RFC 6901-style traversal. Plain keys retain
        # backward-compatible top-level lookup, including keys containing dots.
        if not key.startswith("/"):
            return record.get(key) if isinstance(record, dict) else None
        current = record
        for token in key.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict):
                current = current.get(token)
            elif isinstance(current, list):
                try:
                    current = current[int(token)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
            if current is None:
                return None
        return current

    def _make_json_lookup_key(self, image_path: str) -> str:
        path = Path(image_path)
        if self._image_key_mode == "filename":
            return path.name
        if self._image_key_mode == "stem":
            return path.stem
        root = self._json_root
        if root is None:
            raise ValueError("Multi-Caption JSON root was not initialized")
        try:
            resolved_path = path.resolve()
            resolved_root = root.resolve()
            key = os.path.relpath(str(resolved_path), str(resolved_root))
        except OSError:
            key = os.path.relpath(str(path), str(root))
        return key.replace("\\", "/")

    def _new_index_path(self) -> Path:
        root = Path(tempfile.gettempdir()) / "dts-mc"
        root.mkdir(parents=True, exist_ok=True)
        # Keep Windows paths short. The file is run-local and removed at normal
        # interpreter exit; stale crash leftovers are harmless and overwritten
        # by neither another job nor another process.
        return root / f"{uuid.uuid4().hex[:16]}.sqlite3"

    def _insert(self, conn: sqlite3.Connection, image_key: str, group_name: str, caption: Any) -> None:
        if caption is None:
            return
        if isinstance(caption, list):
            caption = self._groups_by_name[group_name].processing.caption_separator.join(
                str(item) for item in caption
            )
        value = str(caption).strip()
        if not value:
            return
        conn.execute(
            "INSERT OR REPLACE INTO captions(image_key, group_name, caption) VALUES (?, ?, ?)",
            (image_key, group_name, value),
        )

    def _build_index(self, image_infos: list[Any]) -> str:
        target = self._new_index_path()
        conn = sqlite3.connect(str(target))
        try:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute(
                "CREATE TABLE captions ("
                "image_key TEXT NOT NULL, "
                "group_name TEXT NOT NULL, "
                "caption TEXT NOT NULL, "
                "PRIMARY KEY(image_key, group_name))"
            )

            if self.storage_mode in {"files", "multiline"}:
                self._index_files(conn, image_infos)
            else:
                self._index_json(conn, image_infos)

            conn.execute("CREATE INDEX captions_image_key ON captions(image_key)")
            conn.commit()
        except Exception:
            conn.close()
            target.unlink(missing_ok=True)
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass
        atexit.register(_cleanup_index, str(target))
        return str(target)

    def _index_files(self, conn: sqlite3.Connection, image_infos: list[Any]) -> None:
        seen: set[str] = set()
        for info in image_infos:
            image_path = str(getattr(info, "absolute_path", "") or "")
            image_key = self._identity(image_path)
            if image_key in seen:
                continue
            seen.add(image_key)

            if self.storage_mode == "files":
                for group in self.groups:
                    extension = str(group.source.get("extension") or "")
                    content = None
                    for path in self._candidate_paths(image_path, extension):
                        content = self._read_text(path)
                        if content is not None:
                            break
                    if content is None:
                        continue
                    lines = content.splitlines()
                    if group.processing.enable_wildcard:
                        value = "\n".join(line.strip() for line in lines if line.strip())
                    else:
                        value = lines[0].strip() if lines else ""
                    self._insert(conn, image_key, group.name, value)
            else:
                extension = str(self.storage.get("extension") or ".txt")
                content = None
                for path in self._candidate_paths(image_path, extension):
                    content = self._read_text(path)
                    if content is not None:
                        break
                physical_lines = [] if content is None else content.splitlines()
                for group in self.groups:
                    line = int(group.source.get("line") or 0)
                    value = physical_lines[line - 1].strip() if 1 <= line <= len(physical_lines) else None
                    self._insert(conn, image_key, group.name, value)

    def _index_json(self, conn: sqlite3.Connection, image_infos: list[Any]) -> None:
        source = Path(str(self.storage.get("path") or ""))
        if not source.is_file():
            raise ValueError(f"Multi-Caption dedicated JSON/JSONL does not exist: {source}")
        configured_root = self.storage.get("root")
        self._json_root = Path(str(configured_root)).resolve() if configured_root else source.parent.resolve()

        wanted: dict[str, str] = {}
        for info in image_infos:
            image_path = str(getattr(info, "absolute_path", "") or "")
            lookup = self._make_json_lookup_key(image_path)
            identity = self._identity(image_path)
            previous = wanted.get(lookup)
            if previous is not None and previous != identity:
                raise ValueError(
                    f"Multi-Caption image lookup key collision {lookup!r}. "
                    "Use relative_path mode or a less ambiguous key."
                )
            wanted[lookup] = identity

        def insert_record(lookup: str, record: dict[str, Any]) -> None:
            identity = wanted.get(lookup)
            if identity is None:
                return
            for group in self.groups:
                key = str(group.source.get("key") or "")
                self._insert(conn, identity, group.name, self._json_pointer(record, key))

        if self.storage_mode == "json":
            try:
                raw = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Multi-Caption JSON could not be read: {source}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError("Multi-Caption JSON must map image keys to objects")
            for lookup in wanted:
                record = raw.get(lookup)
                if record is not None:
                    if not isinstance(record, dict):
                        raise ValueError(f"Multi-Caption JSON record {lookup!r} must be an object")
                    insert_record(lookup, record)
            del raw
        else:
            image_field = str(self.storage.get("image_key_field") or "image")
            seen: set[str] = set()
            try:
                with source.open("r", encoding="utf-8") as fp:
                    for lineno, line in enumerate(fp, 1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Multi-Caption JSONL invalid line {lineno}: {exc}") from exc
                        if not isinstance(record, dict):
                            raise ValueError(f"Multi-Caption JSONL line {lineno} must be an object")
                        lookup = str(record.get(image_field) or "")
                        if not lookup:
                            raise ValueError(f"Multi-Caption JSONL line {lineno} is missing {image_field!r}")
                        if lookup in seen:
                            raise ValueError(f"Multi-Caption JSONL duplicate image key: {lookup!r}")
                        seen.add(lookup)
                        insert_record(lookup, record)
            except OSError as exc:
                raise ValueError(f"Multi-Caption JSONL could not be read: {source}: {exc}") from exc

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            if not os.path.isfile(self._db_path):
                raise ValueError("Multi-Caption caption index disappeared before training completed")
            self._conn = sqlite3.connect(self._db_path)
        return self._conn

    def _candidates(self, image_path: str) -> dict[str, str]:
        image_key = self._identity(image_path)
        cached = self._candidate_cache.get(image_key)
        if cached is not None:
            self._candidate_cache.move_to_end(image_key)
            return cached

        rows = self._connection().execute(
            "SELECT group_name, caption FROM captions WHERE image_key = ?",
            (image_key,),
        ).fetchall()
        result = {str(name): str(caption) for name, caption in rows}
        self._candidate_cache[image_key] = result
        self._candidate_cache.move_to_end(image_key)
        while len(self._candidate_cache) > self.CACHE_LIMIT:
            self._candidate_cache.popitem(last=False)
        return result

    def choose(self, *, image_path: str, image_key: str | None = None) -> ResolvedCaption:
        del image_key
        candidates = self._candidates(image_path)
        valid: list[tuple[CaptionGroup, str]] = []
        for group in self.groups:
            if not group.enabled or group.weight <= 0:
                continue
            caption = candidates.get(group.name)
            if caption is None or not caption.strip():
                continue
            valid.append((group, caption.strip()))

        if not valid:
            raise ValueError(
                f"Multi-Caption: no enabled non-empty caption group is available for image {image_path!r}"
            )

        weights = [item[0].weight for item in valid]
        if self.deterministic:
            material = (image_path + "\0" + "\0".join(
                f"{item[0].name}:{item[0].weight:g}" for item in valid
            )).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            group, caption = random.Random(seed).choices(valid, weights=weights, k=1)[0]
        else:
            group, caption = random.choices(valid, weights=weights, k=1)[0]
        return ResolvedCaption(group.name, caption, group.processing)


def configure_multi_caption_dataset_groups(
    config_path: str,
    train_dataset_group: Any,
    val_dataset_group: Any | None = None,
) -> MultiCaptionResolver:
    groups = [group for group in (train_dataset_group, val_dataset_group) if group is not None]
    image_infos = []
    for group in groups:
        if not hasattr(group, "datasets"):
            raise ValueError("Multi-Caption v1 does not support custom dataset_class / MinimalDataset")
        for dataset in group.datasets:
            image_infos.extend(
                info for info in dataset.image_data.values()
                if not bool(getattr(info, "is_reg", False))
            )

    train_resolver = MultiCaptionResolver.from_file(config_path, image_infos, deterministic=False)
    for dataset in train_dataset_group.datasets:
        dataset.set_multi_caption_resolver(train_resolver)

    if val_dataset_group is not None:
        val_resolver = train_resolver.clone(deterministic=True)
        for dataset in val_dataset_group.datasets:
            dataset.set_multi_caption_resolver(val_resolver)
    return train_resolver
