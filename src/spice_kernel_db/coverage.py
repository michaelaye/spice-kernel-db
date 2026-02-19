"""SPK body coverage analysis using SpiceyPy.

Pure functions that query SPICE SPK files for body ephemeris coverage.
SpiceyPy is imported lazily inside each function so the rest of the
package works without it installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Mapping of common body names (lowercase) to list of (NAIF ID, description).
# Multiple entries per name handle ambiguity (e.g. "earth" → body center vs barycenter).
NAIF_BODIES: dict[str, list[tuple[int, str]]] = {
    "sun": [(10, "Sun")],
    "mercury": [(199, "Mercury"), (1, "Mercury Barycenter")],
    "venus": [(299, "Venus"), (2, "Venus Barycenter")],
    "earth": [(399, "Earth"), (3, "Earth-Moon Barycenter")],
    "moon": [(301, "Moon")],
    "luna": [(301, "Moon")],
    "mars": [(499, "Mars"), (4, "Mars Barycenter")],
    "phobos": [(401, "Phobos")],
    "deimos": [(402, "Deimos")],
    "jupiter": [(599, "Jupiter"), (5, "Jupiter Barycenter")],
    "io": [(501, "Io")],
    "europa": [(502, "Europa")],
    "ganymede": [(503, "Ganymede")],
    "callisto": [(504, "Callisto")],
    "saturn": [(699, "Saturn"), (6, "Saturn Barycenter")],
    "titan": [(606, "Titan")],
    "enceladus": [(602, "Enceladus")],
    "mimas": [(601, "Mimas")],
    "uranus": [(799, "Uranus"), (7, "Uranus Barycenter")],
    "neptune": [(899, "Neptune"), (8, "Neptune Barycenter")],
    "triton": [(801, "Triton")],
    "pluto": [(999, "Pluto"), (9, "Pluto Barycenter")],
    "charon": [(901, "Charon")],
    "ceres": [(2000001, "Ceres")],
    "vesta": [(2000004, "Vesta")],
    "bennu": [(2101955, "Bennu")],
    "ryugu": [(2162173, "Ryugu")],
    "67p": [(1000012, "67P/Churyumov-Gerasimenko")],
    "churyumov-gerasimenko": [(1000012, "67P/Churyumov-Gerasimenko")],
    "3i/atlas": [(1004083, "3I/ATLAS (Horizons SPK)"), (90004923, "3I/ATLAS (Horizons record)")],
    "atlas": [(615, "Atlas (Saturn XV)"), (1004083, "3I/ATLAS (Horizons SPK)"), (90004923, "3I/ATLAS (Horizons record)")],
    "ssb": [(0, "Solar System Barycenter")],
}


def resolve_body_id(name_or_id: str) -> list[tuple[int, str]]:
    """Resolve a body name or numeric ID to a list of (NAIF ID, description) pairs.

    If *name_or_id* is a valid integer string, returns a single-element list
    with that ID. Otherwise looks up the name (case-insensitive) in ``NAIF_BODIES``.
    Returns an empty list if no match is found.
    """
    try:
        body_id = int(name_or_id)
        return [(body_id, f"NAIF ID {body_id}")]
    except ValueError:
        pass
    return NAIF_BODIES.get(name_or_id.strip().lower(), [])


@dataclass
class CoverageInterval:
    """A single time interval of ephemeris coverage."""

    et_start: float
    et_end: float
    utc_start: str | None = None
    utc_end: str | None = None


@dataclass
class KernelCoverageResult:
    """Coverage result for one kernel file."""

    filename: str
    kernel_type: str
    body_found: bool = False
    intervals: list[CoverageInterval] = field(default_factory=list)
    error: str | None = None


def spk_bodies(spk_path: str | Path) -> list[int]:
    """Return sorted list of all body IDs in an SPK file."""
    import spiceypy as spice

    ids = spice.spkobj(str(spk_path))
    return sorted(int(i) for i in ids)


def spk_coverage(
    spk_path: str | Path,
    body_id: int,
    lsk_path: str | Path | None = None,
) -> list[CoverageInterval]:
    """Return coverage intervals for a body in an SPK file.

    If *lsk_path* is provided, it is temporarily furnished so that
    ``spice.timout`` can convert ET to UTC strings. It is unloaded
    immediately after use.

    ``spkcov`` reads directly from the file — no kernel pool load needed.
    """
    import spiceypy as spice

    cover = spice.stypes.SPICEDOUBLE_CELL(2000)
    spice.spkcov(str(spk_path), body_id, cover)

    n = spice.wncard(cover)
    if n == 0:
        return []

    # Optionally furnish LSK for UTC conversion
    furnished_lsk = False
    if lsk_path is not None:
        try:
            spice.furnsh(str(lsk_path))
            furnished_lsk = True
        except Exception:
            pass

    intervals = []
    for i in range(n):
        et_start, et_end = spice.wnfetd(cover, i)
        utc_start = None
        utc_end = None
        if furnished_lsk:
            try:
                utc_start = spice.timout(
                    et_start, "YYYY-MON-DD HR:MN", 20,
                )
                utc_end = spice.timout(
                    et_end, "YYYY-MON-DD HR:MN", 20,
                )
            except Exception:
                pass
        intervals.append(CoverageInterval(
            et_start=et_start, et_end=et_end,
            utc_start=utc_start, utc_end=utc_end,
        ))

    if furnished_lsk:
        try:
            spice.unload(str(lsk_path))
        except Exception:
            pass

    return intervals


def check_coverage(
    filenames: list[str],
    resolved_paths: list[str | None],
    kernel_types: list[str],
    body_id: int,
    lsk_path: str | Path | None = None,
) -> list[KernelCoverageResult]:
    """Check body coverage across a list of resolved kernel files.

    Non-SPK kernels are skipped (included in results with body_found=False).
    Missing files produce an error entry. SPK files are queried via
    ``spk_coverage``.
    """
    results = []
    for fname, path, ktype in zip(filenames, resolved_paths, kernel_types):
        if ktype != "spk":
            results.append(KernelCoverageResult(
                filename=fname, kernel_type=ktype,
            ))
            continue

        if path is None or not Path(path).is_file():
            results.append(KernelCoverageResult(
                filename=fname, kernel_type=ktype,
                error=f"File not found: {fname}",
            ))
            continue

        try:
            intervals = spk_coverage(path, body_id, lsk_path=lsk_path)
            results.append(KernelCoverageResult(
                filename=fname,
                kernel_type=ktype,
                body_found=len(intervals) > 0,
                intervals=intervals,
            ))
        except Exception as e:
            results.append(KernelCoverageResult(
                filename=fname, kernel_type=ktype,
                error=str(e),
            ))

    return results
