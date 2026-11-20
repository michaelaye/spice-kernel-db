"""File hashing and kernel classification utilities."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# Extension -> SPICE kernel type
KERNEL_TYPES: dict[str, str] = {
    ".bc": "ck",
    ".bsp": "spk",
    ".bpc": "pck",
    ".tpc": "pck",
    ".tf": "fk",
    ".ti": "ik",
    ".tls": "lsk",
    ".tsc": "sclk",
    ".bds": "dsk",
    ".tm": "mk",
}

# All extensions we consider kernel files
KERNEL_EXTENSIONS: set[str] = set(KERNEL_TYPES.keys())


def classify_kernel(filename: str) -> str:
    """Return the SPICE kernel type from the file extension."""
    _, ext = os.path.splitext(filename.lower())
    return KERNEL_TYPES.get(ext, "unknown")


# Known mission identifiers for path-based guessing
_KNOWN_MISSIONS = (
    "juice", "mro", "mex", "vex", "rosetta", "cassini", "juno",
    "messenger", "dawn", "maven", "insight", "odyssey", "lro",
    "clipper", "bepi", "exomars", "europa", "envision", "hera",
    "lucy", "psyche", "osiris-rex", "new-horizons", "ladee",
)

_GENERIC_PREFIXES = (
    "de4", "de3", "jup", "sat", "mar", "ura", "nep", "plu",
    "naif", "pck", "earth", "gm_",
)


def canonicalize_mission(name: str | None) -> str:
    """Normalise a mission name to its canonical form (H4).

    All mission storage and comparison goes through this — without it,
    ``JUICE``/``juice``/``Juice`` ended up as separate ``missions`` rows
    on case-sensitive filesystems and as colliding directories on
    case-insensitive ones. We pick uppercase as canonical with two
    exceptions: ``generic`` and ``unknown`` stay lowercase because the
    rest of the code (notably ``guess_mission``) treats them as
    sentinels for "no specific mission."
    """
    if not name:
        return "unknown"
    s = name.strip()
    if s.lower() in ("generic", "unknown"):
        return s.lower()
    return s.upper()


_KERNEL_TYPE_DIRS = frozenset({"ck", "spk", "pck", "fk", "ik", "lsk", "sclk", "dsk", "mk"})


def guess_mission(filepath: str, roots: list[Path] | None = None) -> str:
    """Best-effort mission name from path components.

    Under one of *roots* (the configured kernel roots, #48) the mission is
    the first component after the root: skd's own store is laid out as
    ``<root>/<MISSION>/<type>/<file>``, and reading it any other way gave
    the store directory as the mission (#43). Elsewhere: a directory before
    ``kernels/`` that is not a generic infrastructure name, then a
    kernel-type directory's parent, then filename patterns. Result is run
    through :func:`canonicalize_mission` for consistent casing.
    """
    path = Path(filepath)
    if roots:
        resolved = path.expanduser().resolve()
        for r in roots:
            try:
                rel = resolved.relative_to(r)
            except ValueError:
                continue
            if len(rel.parts) > 1 and rel.parts[0].lower() not in _KERNEL_TYPE_DIRS:
                return canonicalize_mission(rel.parts[0])
            # Under a root but in no mission directory: only the filename can say.
            return _guess_from_filename(path)
    parts = path.parts
    # Pattern: .../MISSION/kernels/...
    for i, part in enumerate(parts):
        if part.lower() == "kernels" and i > 0:
            candidate = parts[i - 1]
            if candidate.lower() not in ("naif", "pub", "data", "spice"):
                return canonicalize_mission(candidate)

    # Pattern: .../MISSION/<kernel_type>/file (no intermediate kernels/ dir)
    for i, part in enumerate(parts):
        if part.lower() in _KERNEL_TYPE_DIRS and i > 0:
            candidate = parts[i - 1]
            if candidate.lower() not in (
                "naif", "pub", "data", "spice", "kernels",
            ) and "generic" not in candidate.lower():
                return canonicalize_mission(candidate)

    return _guess_from_filename(path)


def _guess_from_filename(p: str | Path) -> str:
    """Known mission names, then generic prefixes, in the filename alone."""
    name = Path(p).stem.lower()
    for mission in _KNOWN_MISSIONS:
        if mission in name:
            return canonicalize_mission(mission)
    if name.startswith(_GENERIC_PREFIXES):
        return "generic"
    return "unknown"


# NAIF versioned metakernel snapshots: _v461_20251127_001.tm or _V324_20260206_002.TM
VERSION_TAG_RE = re.compile(r"(_[vV]\d+_\d{8}_\d{3})\.[tT][mM]$")


def strip_version_tag(filename: str) -> str:
    """``juice_crema_5_2_v473_20260819_001.tm`` -> ``juice_crema_5_2.tm``.

    A filename without a version tag is returned unchanged. The extension
    keeps its original case.
    """
    match = VERSION_TAG_RE.search(filename)
    if not match:
        return filename
    return filename[: match.start()] + filename[filename.rfind("."):]
