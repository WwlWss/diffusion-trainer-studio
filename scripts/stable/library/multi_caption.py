"""Optional exposure-time Multi-Caption resolver for the stable dataset tree.

This module is deliberately independent from DTS host modules so stable trainer
scripts remain directly runnable. Standard training never imports or constructs
this resolver unless --multi_caption_config is present.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable


@dataclass(frozen=True)
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


class MultiCaptionResolver:
    CACHE_LIMIT = 8192

    def __init__(self, config_path: str, policy: dict[str, Any], image_infos: Iterable[Any]):
        self.config_path = str(config_path)
        self.policy = policy
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
        self.groups = []
        for name, raw in raw_groups.items():
            raw = dict(raw or {})
            source = dict(raw.get("source") or {})
            group = CaptionGroup(
                name=str(name),
                enabled=bool(raw.get("enabled", True)),
                weight=float(raw.get("weight", 1.0)),
                source=source,
                processing=CaptionProcessingOptions.from_dict(raw.get("processing")),
            )
            if group.weight < 0:
                raise ValueError(f"Multi-Caption: group {group.name!r} has negative weight")
            self.groups.append(group)

        self._candidate_cache: OrderedDict[str, dict[str, str | None]] = OrderedDict()
        self._records: dict[str, dict[str, Any]] | None = None
        self._json_root: Path | None = None
        self._image_key_mode = str(self.storage.get("image_key_mode") or "relative_path")
        self._dataset_lookup_keys: dict[str, str] = {}

        infos = [info for info in image_infos if not bool(getattr(info, "is_reg", False))]
        if self.storage_mode in {"json", "jsonl"}:
            self._prepare_json_records(infos)

    @classmethod
    def from_file(cls, config_path: str, image_infos: Iterable[Any]) -> "MultiCaptionResolver":
        path = Path(config_path)
        if not path.is_file():
            raise ValueError(f"Multi-Caption config does not exist: {config_path}")
        try:
            policy = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Multi-Caption config could not be read: {config_path}: {exc}") from exc
        if not isinstance(policy, dict):
            raise ValueError("Multi-Caption config must contain a JSON object")
        return cls(config_path, policy, image_infos)

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

    def _cache_get(self, image_path: str) -> dict[str, str | None] | None:
        value = self._candidate_cache.get(image_path)
        if value is not None:
            self._candidate_cache.move_to_end(image_path)
        return value

    def _cache_put(self, image_path: str, value: dict[str, str | None]) -> None:
        self._candidate_cache[image_path] = value
        self._candidate_cache.move_to_end(image_path)
        while len(self._candidate_cache) > self.CACHE_LIMIT:
            self._candidate_cache.popitem(last=False)

    def _load_file_candidates(self, image_path: str) -> dict[str, str | None]:
        cached = self._cache_get(image_path)
        if cached is not None:
            return cached

        result: dict[str, str | None] = {}
        if self.storage_mode == "files":
            for group in self.groups:
                extension = str(group.source.get("extension") or "")
                value = None
                for path in self._candidate_paths(image_path, extension):
                    content = self._read_text(path)
                    if content is not None:
                        value = content.splitlines()[0].strip() if content.splitlines() else ""
                        break
                result[group.name] = value
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
                result[group.name] = value

        self._cache_put(image_path, result)
        return result

    def _make_lookup_key(self, image_path: str) -> str:
        path = Path(image_path)
        if self._image_key_mode == "filename":
            return path.name
        if self._image_key_mode == "stem":
            return path.stem
        root = self._json_root
        if root is None:
            raise ValueError("Multi-Caption JSON root was not initialized")
        try:
            key = os.path.relpath(str(path.resolve()), str(root.resolve()))
        except OSError:
            key = os.path.relpath(str(path), str(root))
        return key.replace("\\", "/")

    def _prepare_json_records(self, image_infos: list[Any]) -> None:
        source = Path(str(self.storage.get("path") or ""))
        if not source.is_file():
            raise ValueError(f"Multi-Caption dedicated JSON/JSONL does not exist: {source}")
        configured_root = self.storage.get("root")
        self._json_root = Path(str(configured_root)).resolve() if configured_root else source.parent.resolve()

        lookup_owner: dict[str, str] = {}
        for info in image_infos:
            image_path = str(getattr(info, "absolute_path", "") or "")
            key = self._make_lookup_key(image_path)
            previous = lookup_owner.get(key)
            if previous is not None and previous != image_path:
                raise ValueError(
                    f"Multi-Caption image lookup key collision {key!r}: {previous!r} and {image_path!r}. "
                    "Use relative_path mode or a less ambiguous key."
                )
            lookup_owner[key] = image_path
            self._dataset_lookup_keys[image_path] = key

        wanted = set(lookup_owner)
        records: dict[str, dict[str, Any]] = {}
        if self.storage_mode == "json":
            try:
                raw = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Multi-Caption JSON could not be read: {source}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError("Multi-Caption JSON must map image keys to objects")
            for key in wanted:
                value = raw.get(key)
                if value is not None:
                    if not isinstance(value, dict):
                        raise ValueError(f"Multi-Caption JSON record {key!r} must be an object")
                    records[key] = value
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
                        key = str(record.get(image_field) or "")
                        if not key:
                            raise ValueError(f"Multi-Caption JSONL line {lineno} is missing {image_field!r}")
                        if key in seen:
                            raise ValueError(f"Multi-Caption JSONL duplicate image key: {key!r}")
                        seen.add(key)
                        if key in wanted:
                            records[key] = record
            except OSError as exc:
                raise ValueError(f"Multi-Caption JSONL could not be read: {source}: {exc}") from exc
        self._records = records

    def _load_json_candidates(self, image_path: str) -> dict[str, str | None]:
        if self._records is None:
            raise ValueError("Multi-Caption JSON records were not initialized")
        key = self._dataset_lookup_keys.get(image_path)
        if key is None:
            key = self._make_lookup_key(image_path)
        record = self._records.get(key) or {}
        result: dict[str, str | None] = {}
        for group in self.groups:
            value = record.get(str(group.source.get("key") or ""))
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            result[group.name] = None if value is None else str(value).strip()
        return result

    def choose(self, *, image_path: str, image_key: str | None = None) -> ResolvedCaption | None:
        del image_key
        candidates = (
            self._load_json_candidates(image_path)
            if self.storage_mode in {"json", "jsonl"}
            else self._load_file_candidates(image_path)
        )
        valid: list[tuple[CaptionGroup, str]] = []
        for group in self.groups:
            if not group.enabled or group.weight <= 0:
                continue
            caption = candidates.get(group.name)
            if caption is None or not caption.strip():
                continue
            valid.append((group, caption.strip()))
        if not valid:
            return None

        group, caption = random.choices(
            valid,
            weights=[item[0].weight for item in valid],
            k=1,
        )[0]
        return ResolvedCaption(group.name, caption, group.processing)


def configure_multi_caption_dataset_groups(config_path: str, *dataset_groups: Any) -> MultiCaptionResolver:
    groups = [group for group in dataset_groups if group is not None]
    datasets = []
    image_infos = []
    for group in groups:
        if not hasattr(group, "datasets"):
            raise ValueError("Multi-Caption v1 does not support custom dataset_class / MinimalDataset")
        for dataset in group.datasets:
            datasets.append(dataset)
            image_infos.extend(
                info for info in dataset.image_data.values()
                if not bool(getattr(info, "is_reg", False))
            )

    resolver = MultiCaptionResolver.from_file(config_path, image_infos)
    for dataset in datasets:
        dataset.set_multi_caption_resolver(resolver)
    return resolver
