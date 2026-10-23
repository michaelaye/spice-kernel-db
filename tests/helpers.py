"""Helpers shared by more than one test module.

Plain classes and constants, not fixtures, so they must be imported
explicitly -- conftest.py only injects fixtures. Extracted when the
6400-line tests/test_kernel_db.py was split by subsystem.
"""

from __future__ import annotations
import hashlib
import logging
import shutil
import textwrap
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from spice_kernel_db import KernelDB, parse_metakernel
from spice_kernel_db.config import Config, load_config, save_config, show_config
from spice_kernel_db.hashing import (
    canonicalize_mission,
    classify_kernel,
    guess_mission,
    sha256_file,
)
from spice_kernel_db.parser import parse_metakernel_text, write_metakernel
from spice_kernel_db.remote import (
    Probe,
    ProbeResult,
    RemoteMetakernel,
    check_mk_availability,
    list_remote_metakernels,
    list_remote_missions,
    resolve_kernel_urls,
    SPICE_SERVERS,
    _DIR_ENTRY_RE,
    _VERSION_TAG_RE,
)


class _CursorTrap:
    """Wraps a duckdb cursor; raises on a sentinel SQL substring match."""

    def __init__(self, cur, trap: str, raise_after: int = 1):
        self._c = cur
        self._trap = trap.upper()
        self._raise_after = raise_after
        self._hits = 0

    def execute(self, sql, *args, **kw):
        if self._trap in sql.upper():
            self._hits += 1
            if self._hits >= self._raise_after:
                raise RuntimeError("simulated crash")
        return self._c.execute(sql, *args, **kw)

    def __enter__(self):
        self._c.__enter__()
        return self

    def __exit__(self, *a):
        return self._c.__exit__(*a)


class _ConWrapper:
    """Delegates to a real duckdb connection; cursor() returns a trap."""

    def __init__(self, real_con, trap: str, raise_after: int = 1):
        self._real = real_con
        self._trap = trap
        self._raise_after = raise_after

    def __getattr__(self, name):
        return getattr(self._real, name)

    def cursor(self):
        return _CursorTrap(self._real.cursor(), self._trap, self._raise_after)


FIXTURE_REGISTRY = Path(__file__).parent / "fixtures" / "mission_registry_min.toml"


class _FakeResp:
    """Minimal urlopen() stand-in: yields `data` once, then EOF."""

    def __init__(self, data: bytes):
        self._data = data
        self.headers = {"Content-Length": str(len(data))}
        self._sent = False

    def read(self, n: int = -1) -> bytes:
        if self._sent:
            return b""
        self._sent = True
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
