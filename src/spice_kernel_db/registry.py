"""Curated registry of alternate metakernel directory locations per mission.

Loaded from ``mission_registry.toml`` shipped inside the package. Used by the
mission-add flow to discover metakernel directories for missions whose layout
deviates from the default ``{server_url}{MISSION}/kernels/mk/`` path.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

_REGISTRY_RESOURCE = "mission_registry.toml"


@dataclass(frozen=True)
class MissionEntry:
    candidates: tuple[str, ...] = ()
    planetarypy: bool = False


def _parse_registry(path: Path) -> Mapping[str, MissionEntry]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    parsed = {
        name: MissionEntry(
            candidates=tuple(entry.get("candidates", [])),
            planetarypy=bool(entry.get("planetarypy", False)),
        )
        for name, entry in data.items()
        if isinstance(entry, dict)
    }
    return MappingProxyType(parsed)


@cache
def load_registry() -> Mapping[str, MissionEntry]:
    """Load the bundled mission registry as a read-only mapping."""
    return _parse_registry(Path(str(files("spice_kernel_db").joinpath(_REGISTRY_RESOURCE))))


def load_registry_from(path: Path) -> Mapping[str, MissionEntry]:
    """Load a registry from an arbitrary path (for tests / overrides)."""
    return _parse_registry(path)


def registry_candidates(mission: str, server_url: str) -> list[str]:
    """Return registry-defined candidate metakernel URLs for *mission*.

    Placeholders ``{server}`` and ``{m}`` are expanded. Order is preserved
    and duplicates are removed. Returns an empty list if the mission is
    not in the registry.
    """
    if not server_url.endswith("/"):
        server_url += "/"
    entry = load_registry().get(mission)
    if entry is None:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for tmpl in entry.candidates:
        url = tmpl.format(server=server_url, m=mission)
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def is_planetarypy_managed(mission: str) -> bool:
    """Return True if the registry marks *mission* as planetarypy-managed."""
    entry = load_registry().get(mission)
    return bool(entry and entry.planetarypy)
