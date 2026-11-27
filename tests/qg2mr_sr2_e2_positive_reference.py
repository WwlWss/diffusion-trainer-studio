"""Independent positive Host reference via public unittest module loading.

Select Host files before import, not by the historical Torch import errors.
Full unpartitioned discovery remains in pinned CPU CI.
"""
from __future__ import annotations

import fnmatch
import importlib
import importlib.util
from pathlib import Path
import re
import sys
import unittest

PATTERN = "test_parameter_policy*.py"


class ReferenceError(RuntimeError):
    pass


class _ReferenceLoader(unittest.TestLoader):
    def discover(self, start_dir, pattern="test*.py", top_level_dir=None):
        raise ReferenceError("Unreviewed recursive reference discovery")


def discover_reference(test_dir: Path, cpu_files):
    if importlib.util.find_spec("torch") is not None:
        raise ReferenceError("Positive Host reference requires no Torch")
    test_dir = test_dir.resolve(strict=True)
    excluded = frozenset(cpu_files)
    entries = sorted(test_dir.iterdir(), key=lambda p: p.name)
    if any(p.is_dir() and (p / "__init__.py").is_file() for p in entries):
        raise ReferenceError("Unreviewed nested test package in Host reference")
    files = {p.name for p in entries if p.is_file() and fnmatch.fnmatch(p.name, PATTERN)}
    if len(excluded) != 2 or not excluded <= files:
        raise ReferenceError(f"Missing exact CPU ownership files: {sorted(excluded - files)}")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", name) for name in files):
        raise ReferenceError("Invalid reference module filename")
    selected = sorted(files - excluded)
    if not selected:
        raise ReferenceError("Empty reference Host selection")
    stems = {Path(name).stem for name in excluded}
    aliases = stems | {"tests." + stem for stem in stems}

    def require_no_cpu():
        if aliases & sys.modules.keys():
            raise ReferenceError(f"CPU-only modules loaded in reference: {sorted(aliases & sys.modules.keys())}")

    require_no_cpu()
    # Fresh process, same top-level identities as unittest discover -s tests.
    if str(test_dir.parent) not in sys.path:
        sys.path.insert(0, str(test_dir.parent))
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))
    loader = _ReferenceLoader()
    suite = unittest.TestSuite()
    for filename in selected:
        module = importlib.import_module(Path(filename).stem)
        origin = getattr(module, "__file__", None)
        if origin is None or Path(origin).resolve() != (test_dir / filename).resolve():
            raise ReferenceError(f"Unexpected reference module source: {filename}: {origin}")
        suite.addTests(loader.loadTestsFromModule(module, pattern=PATTERN))
        require_no_cpu()
    if loader.errors:
        raise ReferenceError("Reference discovery errors:\n" + "\n".join(loader.errors))

    def walk(item):
        if isinstance(item, unittest.TestCase):
            yield item.id()
        elif isinstance(item, unittest.BaseTestSuite):
            for child in item:
                yield from walk(child)
        else:
            raise ReferenceError(f"Unsupported reference discovery object: {type(item)!r}")

    ids = list(walk(suite))
    if (not ids or any(type(i) is not str or not i
                       or i.startswith("unittest.loader._FailedTest.")
                       or i.split(".")[0] in stems
                       or any(i.startswith("tests." + stem + ".") for stem in stems)
                       for i in ids)
            or len(ids) != len(set(ids))):
        raise ReferenceError("Invalid, duplicate, failed or CPU-owned Host reference identity")
    return suite, ids
