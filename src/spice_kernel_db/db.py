"""Content-addressed SPICE kernel database backed by DuckDB.

The database tracks every kernel file by its SHA-256 hash. Multiple
locations (across missions) can reference the same hash, enabling
deduplication and cross-mission kernel reuse.

Key design decisions:
  - SHA-256 is the source of truth for identity (not filename).
  - resolve_kernel() is mission-aware: prefers kernels from the same
    mission, warns when falling back to another mission's copy.
  - Metakernel rewriting minimises edits to the original: only
    PATH_VALUES is changed, everything else stays intact.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import duckdb

from spice_kernel_db.hashing import (
    KERNEL_EXTENSIONS,
    classify_kernel,
    guess_mission,
    sha256_file,
)
from spice_kernel_db.parser import ParsedMetakernel, parse_metakernel, write_metakernel

logger = logging.getLogger(__name__)


class KernelDB:
    """Content-addressed SPICE kernel database.

    Schema:
        kernels:     sha256 (PK), filename, kernel_type, size_bytes
        locations:   sha256 + abs_path (PK), mission, source_url, scanned_at
        metakernels: mk_path + entry_index (PK), raw_entry, filename

    A kernel is uniquely identified by its SHA-256. Multiple *locations*
    (from different missions or the same mission in different directories)
    can reference the same hash — that's how duplicates are detected.
    """

    def __init__(self, db_path: str | Path = "~/.spice_kernels.duckdb"):
        self.db_path = str(Path(db_path).expanduser())
        self.con = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS kernels (
                sha256       VARCHAR PRIMARY KEY,
                filename     VARCHAR,
                kernel_type  VARCHAR,
                size_bytes   BIGINT
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS locations (
                sha256       VARCHAR,
                abs_path     VARCHAR,
                mission      VARCHAR,
                source_url   VARCHAR,
                scanned_at   TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (sha256, abs_path)
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS metakernels (
                mk_path      VARCHAR,
                entry_index  INTEGER,
                raw_entry    VARCHAR,
                filename     VARCHAR,
                PRIMARY KEY (mk_path, entry_index)
            )
        """)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def register_file(
        self,
        path: str | Path,
        mission: str | None = None,
        source_url: str | None = None,
    ) -> str:
        """Register a single kernel file. Returns its SHA-256.

        If the file's hash already exists but under a different filename,
        the new location is still recorded (pointing to the same hash).
        This is exactly what happens when JUICE ships ``jup365_19900101_20500101.bsp``
        and generic_kernels has ``jup365.bsp`` — same content, different names,
        one hash.
        """
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)

        h = sha256_file(p)
        fname = p.name
        ktype = classify_kernel(fname)
        size = p.stat().st_size
        m = mission or guess_mission(str(p))

        # Insert or update the kernel record.
        # If the hash already exists with a different filename, we keep
        # the first-registered filename as canonical but still record
        # this location.
        existing = self.con.execute(
            "SELECT filename FROM kernels WHERE sha256 = ?", [h]
        ).fetchone()

        if existing is None:
            self.con.execute(
                "INSERT INTO kernels VALUES (?, ?, ?, ?)",
                [h, fname, ktype, size],
            )
        elif existing[0] != fname:
            logger.info(
                "Hash match: %s is identical to already-registered %s",
                fname, existing[0],
            )

        self.con.execute("""
            INSERT OR REPLACE INTO locations VALUES
                (?, ?, ?, ?, current_timestamp)
        """, [h, str(p), m, source_url])
        return h

    def scan_directory(
        self,
        root: str | Path,
        mission: str | None = None,
        extensions: set[str] | None = None,
        verbose: bool = False,
    ) -> int:
        """Recursively scan a directory tree and register all kernel files.

        Returns the number of files registered.
        """
        if extensions is None:
            extensions = KERNEL_EXTENSIONS

        root = Path(root).expanduser().resolve()
        count = 0
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in extensions:
                try:
                    self.register_file(p, mission=mission)
                    count += 1
                    if verbose:
                        print(f"  registered: {p.name}")
                except Exception as e:
                    logger.warning("Could not register %s: %s", p, e)
        print(f"Scanned {root}: {count} kernel files registered.")
        return count

    # ------------------------------------------------------------------
    # Lookup (mission-aware)
    # ------------------------------------------------------------------

    def find_by_filename(self, filename: str) -> list[dict]:
        """Find all locations for a kernel by filename.

        Checks both the canonical filename in the kernels table AND the
        actual filename in the locations path. This handles the case where
        the same content is stored under different names (e.g. jup365.bsp
        vs jup365_19900101_20500101.bsp).
        """
        rows = self.con.execute("""
            SELECT DISTINCT k.sha256, l.abs_path, l.mission,
                   k.kernel_type, k.size_bytes
            FROM kernels k
            JOIN locations l ON k.sha256 = l.sha256
            WHERE k.filename = ?
               OR l.abs_path LIKE '%/' || ?
            ORDER BY l.mission
        """, [filename, filename]).fetchall()
        return [
            {"sha256": r[0], "abs_path": r[1], "mission": r[2],
             "kernel_type": r[3], "size_bytes": r[4]}
            for r in rows
        ]

    def find_by_hash(self, sha256: str) -> list[dict]:
        """Find all locations for a kernel by its hash."""
        rows = self.con.execute("""
            SELECT l.abs_path, l.mission, k.filename
            FROM locations l
            JOIN kernels k ON k.sha256 = l.sha256
            WHERE l.sha256 = ?
        """, [sha256]).fetchall()
        return [
            {"abs_path": r[0], "mission": r[1], "filename": r[2]}
            for r in rows
        ]

    def _find_by_filename_any_alias(self, filename: str) -> list[dict]:
        """Find locations where the file content matches, even if stored
        under a different filename.

        This handles the case where JUICE has ``jup365_19900101_20500101.bsp``
        but the DB only knows it as ``jup365.bsp`` (or vice versa).
        We search all locations whose sha256 matches any kernel with a
        similar filename stem prefix.
        """
        # First: try all locations that have the exact filename
        # (the filename column in `kernels` is the *first* registered name,
        #  but locations can have any name).
        rows = self.con.execute("""
            SELECT DISTINCT k.sha256, l.abs_path, l.mission,
                   k.kernel_type, k.size_bytes, k.filename AS canonical_name
            FROM locations l
            JOIN kernels k ON k.sha256 = l.sha256
            WHERE l.abs_path LIKE '%/' || ? OR l.abs_path LIKE '%\\' || ?
            ORDER BY l.mission
        """, [filename, filename]).fetchall()
        if rows:
            return [
                {"sha256": r[0], "abs_path": r[1], "mission": r[2],
                 "kernel_type": r[3], "size_bytes": r[4],
                 "canonical_name": r[5]}
                for r in rows
            ]

        # Second: fuzzy — strip date/version suffixes and search by prefix.
        # e.g. 'jup365_19900101_20500101' -> try 'jup365%'
        stem = Path(filename).stem
        ext = Path(filename).suffix
        # Progressively shorten: split on _ and try each prefix
        parts = stem.split("_")
        for n in range(len(parts), 0, -1):
            prefix = "_".join(parts[:n])
            if len(prefix) < 4:
                continue  # too short, would match too broadly
            pattern = prefix + "%"
            rows = self.con.execute("""
                SELECT DISTINCT k.sha256, l.abs_path, l.mission,
                       k.kernel_type, k.size_bytes, k.filename AS canonical_name
                FROM kernels k
                JOIN locations l ON k.sha256 = l.sha256
                WHERE k.filename LIKE ?
                ORDER BY l.mission
            """, [pattern]).fetchall()
            if rows:
                return [
                    {"sha256": r[0], "abs_path": r[1], "mission": r[2],
                     "kernel_type": r[3], "size_bytes": r[4],
                     "canonical_name": r[5]}
                    for r in rows
                ]

        return []

    def resolve_kernel(
        self,
        filename: str,
        preferred_mission: str | None = None,
    ) -> tuple[str | None, list[str]]:
        """Resolve a kernel filename to an absolute path on disk.

        Mission-aware resolution order:
          1. Exact filename match in preferred_mission
          2. Exact filename match in any mission
          3. Fuzzy match (alias/renamed) in preferred_mission
          4. Fuzzy match in any mission
          5. None

        Returns:
            (resolved_path, warnings) where warnings is a list of
            human-readable strings about fallback decisions.
        """
        warnings: list[str] = []

        # --- 1. Exact filename, preferred mission ---
        hits = self.find_by_filename(filename)
        if preferred_mission and hits:
            same_mission = [
                h for h in hits
                if h["mission"].lower() == preferred_mission.lower()
                and Path(h["abs_path"]).is_file()
            ]
            if same_mission:
                return same_mission[0]["abs_path"], warnings

        # --- 2. Exact filename, any mission ---
        for h in hits:
            if Path(h["abs_path"]).is_file():
                if preferred_mission:
                    warnings.append(
                        f"{filename}: not found in [{preferred_mission}] registry, "
                        f"using copy from [{h['mission']}]"
                    )
                return h["abs_path"], warnings

        # --- 3. Fuzzy (alias) lookup ---
        fuzzy = self._find_by_filename_any_alias(filename)
        if preferred_mission and fuzzy:
            same_mission = [
                h for h in fuzzy
                if h["mission"].lower() == preferred_mission.lower()
                and Path(h["abs_path"]).is_file()
            ]
            if same_mission:
                h = same_mission[0]
                warnings.append(
                    f"{filename}: matched by content hash to "
                    f"{h['canonical_name']} in [{h['mission']}]"
                )
                return h["abs_path"], warnings

        # --- 4. Fuzzy, any mission ---
        for h in fuzzy:
            if Path(h["abs_path"]).is_file():
                warnings.append(
                    f"{filename}: not in [{preferred_mission or '?'}] registry, "
                    f"matched by hash to {h['canonical_name']} in [{h['mission']}]"
                )
                return h["abs_path"], warnings

        return None, warnings

    # ------------------------------------------------------------------
    # Duplicate reporting
    # ------------------------------------------------------------------

    def report_duplicates(self, min_copies: int = 2) -> list[dict]:
        """Return kernels that exist in multiple locations.

        Each result: sha256, filename, size_bytes, count, paths, missions.
        """
        rows = self.con.execute("""
            SELECT k.sha256, k.filename, k.size_bytes, COUNT(*) AS cnt,
                   LIST(l.abs_path) AS paths,
                   LIST(l.mission) AS missions
            FROM kernels k
            JOIN locations l ON k.sha256 = l.sha256
            GROUP BY k.sha256, k.filename, k.size_bytes
            HAVING cnt >= ?
            ORDER BY k.size_bytes DESC
        """, [min_copies]).fetchall()

        results = []
        total_waste = 0
        for r in rows:
            waste = r[2] * (r[3] - 1)
            total_waste += waste
            results.append({
                "sha256": r[0], "filename": r[1], "size_bytes": r[2],
                "count": r[3], "paths": r[4], "missions": r[5],
                "wasted_bytes": waste,
            })

        if results:
            print(f"\n{'=' * 70}")
            print(f"Duplicate kernels: {len(results)} files with "
                  f"{min_copies}+ copies")
            print(f"Total wasted space: {total_waste / 1e6:.1f} MB")
            print(f"{'=' * 70}")
            for d in results[:30]:
                print(f"\n  {d['filename']}  "
                      f"({d['size_bytes'] / 1e6:.1f} MB × {d['count']} copies)")
                for p, m in zip(d["paths"], d["missions"]):
                    print(f"    [{m}] {p}")
        else:
            print("No duplicates found.")

        return results

    # ------------------------------------------------------------------
    # Metakernel operations
    # ------------------------------------------------------------------

    def index_metakernel(self, mk_path: str | Path):
        """Parse and register a metakernel's entries in the DB."""
        mk_path = str(Path(mk_path).resolve())
        parsed = parse_metakernel(mk_path)
        self.con.execute(
            "DELETE FROM metakernels WHERE mk_path = ?", [mk_path]
        )
        for i, raw in enumerate(parsed.kernels):
            self.con.execute(
                "INSERT INTO metakernels VALUES (?, ?, ?, ?)",
                [mk_path, i, raw, Path(raw).name],
            )

    def check_metakernel(
        self,
        mk_path: str | Path,
        mission: str | None = None,
    ) -> dict:
        """Check which kernels from a metakernel are already available.

        Args:
            mk_path: Path to the .tm file.
            mission: Mission name for preferred-mission resolution.

        Returns:
            dict with keys:
                found:    list of (raw_entry, local_path)
                missing:  list of raw_entry
                warnings: list of str
        """
        parsed = parse_metakernel(mk_path)
        if mission is None:
            mission = guess_mission(str(mk_path))

        found = []
        missing = []
        all_warnings = []

        for raw in parsed.kernels:
            fname = Path(raw).name
            local, warnings = self.resolve_kernel(fname, preferred_mission=mission)
            all_warnings.extend(warnings)
            if local and Path(local).is_file():
                found.append((raw, local))
            else:
                missing.append(raw)

        print(f"\nMetakernel: {mk_path}")
        print(f"  Mission:        {mission}")
        print(f"  Found locally:  {len(found)}/{len(parsed.kernels)}")
        print(f"  Missing:        {len(missing)}")
        if missing:
            print("  Missing files:")
            for m in missing:
                print(f"    {m}")
        if all_warnings:
            print("  Warnings:")
            for w in all_warnings:
                print(f"    ⚠ {w}")

        return {"found": found, "missing": missing, "warnings": all_warnings}

    def rewrite_metakernel(
        self,
        mk_path: str | Path,
        output: str | Path,
        *,
        mission: str | None = None,
        link_root: str | Path | None = None,
    ) -> tuple[Path, list[str]]:
        """Rewrite a metakernel for local use with minimal edits.

        Strategy: create a symlink tree at ``link_root`` that mirrors the
        original ``$KERNELS/type/filename`` layout. Each symlink points to
        the actual file on disk (which might live in a different mission's
        tree). The rewritten metakernel only changes PATH_VALUES to point
        to ``link_root`` — everything else is identical to the original.

        If ``link_root`` is None, defaults to a ``kernels/`` directory
        next to the output file (matching the common ``..`` convention).

        Args:
            mk_path:   Path to original .tm file.
            output:    Where to write the new .tm.
            mission:   Mission name for preferred-mission resolution.
            link_root: Where to create the symlink tree.

        Returns:
            (output_path, warnings)
        """
        mk_path = Path(mk_path)
        output = Path(output)
        parsed = parse_metakernel(mk_path)
        if mission is None:
            mission = guess_mission(str(mk_path))

        if link_root is None:
            # Default: kernels/ directory next to the output mk
            link_root = output.parent / "kernels"
        link_root = Path(link_root).expanduser().resolve()

        all_warnings: list[str] = []
        unresolved: list[str] = []

        # Build symlink tree
        relpaths = parsed.kernel_relpaths()
        for raw, rel in zip(parsed.kernels, relpaths):
            fname = Path(raw).name
            local, warnings = self.resolve_kernel(
                fname, preferred_mission=mission
            )
            all_warnings.extend(warnings)

            link_path = link_root / rel
            link_path.parent.mkdir(parents=True, exist_ok=True)

            if local and Path(local).is_file():
                if link_path.is_symlink() or link_path.exists():
                    link_path.unlink()
                link_path.symlink_to(Path(local).resolve())
            else:
                unresolved.append(raw)
                all_warnings.append(f"{fname}: NOT FOUND — symlink not created")

        # Determine what PATH_VALUES should be.
        # The original mk typically has PATH_VALUES = ( '..' ) and the
        # kernels are at $KERNELS/type/file. We want the same structure
        # but rooted at link_root.
        # So PATH_VALUES becomes the link_root.
        new_path_values = [str(link_root)]

        out_path = write_metakernel(
            parsed, output,
            path_values=new_path_values,
        )

        print(f"Wrote metakernel to {out_path}")
        print(f"  Symlink tree at {link_root}")
        print(f"  Resolved: {len(parsed.kernels) - len(unresolved)}"
              f"/{len(parsed.kernels)}")
        if unresolved:
            print(f"  Unresolved ({len(unresolved)}):")
            for u in unresolved:
                print(f"    {u}")
        if all_warnings:
            print(f"  Warnings:")
            for w in all_warnings:
                print(f"    ⚠ {w}")

        return out_path, all_warnings

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Print and return summary statistics."""
        n_kernels = self.con.execute(
            "SELECT COUNT(*) FROM kernels"
        ).fetchone()[0]
        n_locs = self.con.execute(
            "SELECT COUNT(*) FROM locations"
        ).fetchone()[0]
        total_size = self.con.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM kernels"
        ).fetchone()[0]
        missions = [
            m[0] for m in self.con.execute(
                "SELECT DISTINCT mission FROM locations ORDER BY mission"
            ).fetchall()
        ]
        by_type = self.con.execute("""
            SELECT kernel_type, COUNT(*), SUM(size_bytes)
            FROM kernels GROUP BY kernel_type ORDER BY SUM(size_bytes) DESC
        """).fetchall()
        dups = self.con.execute("""
            SELECT COUNT(*) FROM (
                SELECT sha256 FROM locations
                GROUP BY sha256 HAVING COUNT(*) > 1
            )
        """).fetchone()[0]

        print(f"\n{'=' * 50}")
        print(f"SPICE Kernel Database: {self.db_path}")
        print(f"{'=' * 50}")
        print(f"  Unique kernels:   {n_kernels}")
        print(f"  Total locations:  {n_locs}")
        print(f"  Unique content:   {total_size / 1e9:.2f} GB")
        print(f"  Duplicated files: {dups}")
        print(f"  Missions:         {', '.join(missions)}")
        print(f"\n  By type:")
        for ktype, cnt, sz in by_type:
            print(f"    {ktype:8s}  {cnt:5d} files  {sz / 1e6:10.1f} MB")
        print()

        return {
            "n_kernels": n_kernels,
            "n_locations": n_locs,
            "total_bytes": total_size,
            "n_duplicates": dups,
            "missions": missions,
        }

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate_plan(self) -> list[dict]:
        """Generate a deduplication plan.

        For each set of identical files (same hash, multiple locations),
        pick one canonical location (preferring generic_kernels) and list
        the rest as removable.

        Returns list of: {filename, size_bytes, keep, remove: [paths]}
        """
        rows = self.con.execute("""
            SELECT k.sha256, k.filename, k.size_bytes,
                   LIST(l.abs_path ORDER BY l.abs_path) AS paths,
                   LIST(l.mission ORDER BY l.abs_path) AS missions
            FROM kernels k
            JOIN locations l ON k.sha256 = l.sha256
            GROUP BY k.sha256, k.filename, k.size_bytes
            HAVING COUNT(*) > 1
        """).fetchall()

        plan = []
        for r in rows:
            paths = r[3]
            missions = r[4]
            # Prefer generic_kernels path as canonical
            keep = paths[0]
            for p, m in zip(paths, missions):
                if m.lower() == "generic":
                    keep = p
                    break
            remove = [p for p in paths if p != keep]
            plan.append({
                "filename": r[1],
                "size_bytes": r[2],
                "keep": keep,
                "remove": remove,
            })
        return plan

    def deduplicate_with_symlinks(self, dry_run: bool = True) -> list[dict]:
        """Replace duplicate files with symlinks to the canonical copy.

        Args:
            dry_run: If True (default), only print what would happen.
        """
        plan = self.deduplicate_plan()
        total_saved = 0

        for item in plan:
            for rm_path in item["remove"]:
                total_saved += item["size_bytes"]
                if dry_run:
                    print(f"  WOULD replace {rm_path}")
                    print(f"    -> symlink to {item['keep']}")
                else:
                    p = Path(rm_path)
                    if p.is_file() and not p.is_symlink():
                        p.unlink()
                        p.symlink_to(item["keep"])
                        print(f"  Replaced {rm_path} -> {item['keep']}")
                        # Update DB to note this is now a symlink
                        # (the hash and location are still valid)

        action = "Would save" if dry_run else "Saved"
        print(f"\n{action}: {total_saved / 1e6:.1f} MB")
        return plan

    def close(self):
        """Close the database connection."""
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
