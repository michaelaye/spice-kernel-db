"""File hashing and kernel classification utilities."""

from __future__ import annotations

import hashlib
import os
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


def guess_mission(filepath: str) -> str:
    """Best-effort mission name from path components.

    Heuristic: look for a directory named before 'kernels/' that isn't
    a generic infrastructure name, then fall back to filename patterns.
    """
    parts = Path(filepath).parts
    # Pattern: .../MISSION/kernels/...
    for i, p in enumerate(parts):
        if p.lower() == "kernels" and i > 0:
            candidate = parts[i - 1]
            if candidate.lower() not in ("naif", "pub", "data", "spice"):
                return candidate

    # Fallback: known mission names in filename
    name = Path(filepath).stem.lower()
    for mission in _KNOWN_MISSIONS:
        if mission in name:
            return mission.upper()

    if name.startswith(_GENERIC_PREFIXES):
        return "generic"

    return "unknown"
