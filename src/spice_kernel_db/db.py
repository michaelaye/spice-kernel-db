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
import shutil
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
from spice_kernel_db.parser import (
    ParsedMetakernel,
    parse_metakernel,
    parse_metakernel_text,
    write_metakernel,
)
from spice_kernel_db.remote import (
    download_kernel,
    download_kernels_parallel,
    fetch_metakernel,
    query_remote_sizes,
    resolve_kernel_urls,
)

logger = logging.getLogger(__name__)


def _format_size(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"


class KernelDB:
    """Content-addressed SPICE kernel database.

    Schema:
        kernels:     sha256 (PK), filename, kernel_type, size_bytes
        locations:   sha256 + abs_path (PK), mission, source_url, scanned_at
        metakernel_entries:  mk_path + entry_index (PK), raw_entry, filename
        metakernel_registry: mk_path (PK), mission, source_url, filename, acquired_at

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
            CREATE TABLE IF NOT EXISTS metakernel_entries (
                mk_path      VARCHAR,
                entry_index  INTEGER,
                raw_entry    VARCHAR,
                filename     VARCHAR,
                PRIMARY KEY (mk_path, entry_index)
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS metakernel_registry (
                mk_path      VARCHAR PRIMARY KEY,
                mission      VARCHAR,
                source_url   VARCHAR,
                filename     VARCHAR,
                acquired_at  TIMESTAMP DEFAULT current_timestamp
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
        archive_dir: str | Path | None = None,
    ) -> str:
        """Register a single kernel file. Returns its SHA-256.

        If the file's hash already exists but under a different filename,
        the new location is still recorded (pointing to the same hash).
        This is exactly what happens when JUICE ships ``jup365_19900101_20500101.bsp``
        and generic_kernels has ``jup365.bsp`` — same content, different names,
        one hash.

        If ``archive_dir`` is set, the file is moved to
        ``archive_dir/mission/kernel_type/filename`` and a symlink is left
        at the original location.
        """
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)

        h = sha256_file(p)
        fname = p.name
        ktype = classify_kernel(fname)
        size = p.stat().st_size
        m = mission or guess_mission(str(p))

        # Archive: move to central location, leave symlink
        if archive_dir is not None:
            dest = Path(archive_dir).expanduser().resolve() / m / ktype / fname
            if p != dest:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest))
                try:
                    p.symlink_to(dest)
                except OSError:
                    logger.warning(
                        "Could not create symlink at %s (unsupported OS?)", p,
                    )
                p = dest

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
        archive_dir: str | Path | None = None,
    ) -> int:
        """Recursively scan a directory tree and register all kernel files.

        If ``archive_dir`` is set, files are moved into the archive and
        symlinks are left at the original locations.

        Returns the number of files registered.
        """
        if extensions is None:
            extensions = KERNEL_EXTENSIONS

        root = Path(root).expanduser().resolve()
        count = 0
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in extensions:
                try:
                    self.register_file(
                        p, mission=mission, archive_dir=archive_dir,
                    )
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
            "DELETE FROM metakernel_entries WHERE mk_path = ?", [mk_path]
        )
        for i, raw in enumerate(parsed.kernels):
            self.con.execute(
                "INSERT INTO metakernel_entries VALUES (?, ?, ?, ?)",
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
    # Acquire (remote metakernel)
    # ------------------------------------------------------------------

    def acquire_metakernel(
        self,
        url: str,
        download_dir: str | Path | None = None,
        mission: str | None = None,
        yes: bool = False,
    ) -> dict:
        """Fetch a remote metakernel, show status, and download missing kernels.

        Args:
            url: URL to a remote .tm metakernel file.
            download_dir: Where to store downloaded kernels. Preserves the
                remote subdirectory structure (lsk/, spk/, etc.).
            mission: Override auto-detected mission name.
            yes: If True, skip the confirmation prompt.

        Returns:
            dict with keys: found, missing, downloaded, warnings
        """
        # 0. Resolve download_dir early (needed for saving .tm)
        if download_dir is None:
            download_dir = Path("~/.local/share/spice-kernel-db/kernels").expanduser()
        download_dir = Path(download_dir).expanduser().resolve()

        # 1. Fetch and parse
        print(f"Fetching {url} ...")
        text = fetch_metakernel(url)
        parsed = parse_metakernel_text(text, url)
        if mission is None:
            mission = guess_mission(url)

        # 1b. Save .tm file to disk and register
        mk_filename = url.rsplit("/", 1)[-1]
        mk_dest = download_dir / mission / "mk" / mk_filename
        mk_dest.parent.mkdir(parents=True, exist_ok=True)
        mk_dest.write_text(text)
        self.index_metakernel(mk_dest)
        self.con.execute("""
            INSERT OR REPLACE INTO metakernel_registry
            VALUES (?, ?, ?, ?, current_timestamp)
        """, [str(mk_dest), mission, url, mk_filename])

        # 2. Resolve kernel URLs
        kernel_urls = resolve_kernel_urls(url, parsed)
        relpaths = parsed.kernel_relpaths()
        filenames = parsed.kernel_filenames()

        # 3. Check local DB for each kernel
        found_indices: list[int] = []
        missing_indices: list[int] = []
        for i, fname in enumerate(filenames):
            local, _ = self.resolve_kernel(fname, preferred_mission=mission)
            if local and Path(local).is_file():
                found_indices.append(i)
            else:
                missing_indices.append(i)

        # 4. Query remote sizes for ALL kernels (parallel HEAD)
        sizes = query_remote_sizes(kernel_urls)

        # 5. Display table
        mk_name = url.rsplit("/", 1)[-1]
        total = len(filenames)
        n_found = len(found_indices)
        n_missing = len(missing_indices)
        download_bytes = sum(
            sizes.get(kernel_urls[i]) or 0 for i in missing_indices
        )

        print(f"\nMetakernel: {mk_name} ({total} kernels)\n")
        name_width = max((len(filenames[i]) for i in range(total)), default=20)
        name_width = max(name_width, 20)
        print(f"  {'Kernel':<{name_width}}  {'Size':>10}  Status")
        print(f"  {'─' * name_width}  {'─' * 10}  {'─' * 8}")
        for i in range(total):
            fname = filenames[i]
            sz = sizes.get(kernel_urls[i])
            sz_str = _format_size(sz) if sz is not None else "unknown"
            status = "in db" if i in found_indices else "missing"
            print(f"  {fname:<{name_width}}  {sz_str:>10}  {status}")

        print(
            f"\n  Total: {total} | In DB: {n_found} | "
            f"Missing: {n_missing} | Download: {_format_size(download_bytes)}"
        )

        if n_missing == 0:
            print("\n  All kernels already in database.")
            return {
                "found": found_indices,
                "missing": [],
                "downloaded": [],
                "warnings": [],
            }

        # 6. Prompt
        if not yes:
            answer = input(
                f"\nDownload {n_missing} missing kernels "
                f"({_format_size(download_bytes)})? [y/N]: "
            ).strip().lower()
            if answer not in ("y", "yes"):
                print("Aborted.")
                return {
                    "found": found_indices,
                    "missing": missing_indices,
                    "downloaded": [],
                    "warnings": [],
                }

        # 7. Download in parallel and register
        tasks = []
        task_info: dict[str, str] = {}  # filename -> source_url
        for i in missing_indices:
            kurl = kernel_urls[i]
            relpath = relpaths[i]
            dest = download_dir / mission / relpath
            fname = filenames[i]
            tasks.append((kurl, dest, fname))
            task_info[fname] = kurl

        dl_paths, warnings = download_kernels_parallel(tasks)

        # Register downloaded files
        downloaded: list[str] = []
        for dest in dl_paths:
            kurl = task_info[dest.name]
            self.register_file(dest, mission=mission, source_url=kurl)
            downloaded.append(str(dest))

        print(f"\n  Downloaded: {len(downloaded)}/{n_missing}")
        if warnings:
            for w in warnings:
                print(f"  ⚠ {w}")

        return {
            "found": found_indices,
            "missing": missing_indices,
            "downloaded": downloaded,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Metakernel listing / info
    # ------------------------------------------------------------------

    def list_metakernels(self, mission: str | None = None) -> list[dict]:
        """List all tracked metakernels, optionally filtered by mission."""
        if mission:
            rows = self.con.execute("""
                SELECT r.filename, r.mission, r.source_url, r.acquired_at,
                       r.mk_path, COUNT(e.entry_index) AS n_kernels
                FROM metakernel_registry r
                LEFT JOIN metakernel_entries e ON e.mk_path = r.mk_path
                WHERE r.mission = ?
                GROUP BY r.mk_path, r.filename, r.mission, r.source_url, r.acquired_at
                ORDER BY r.mission, r.filename
            """, [mission]).fetchall()
        else:
            rows = self.con.execute("""
                SELECT r.filename, r.mission, r.source_url, r.acquired_at,
                       r.mk_path, COUNT(e.entry_index) AS n_kernels
                FROM metakernel_registry r
                LEFT JOIN metakernel_entries e ON e.mk_path = r.mk_path
                GROUP BY r.mk_path, r.filename, r.mission, r.source_url, r.acquired_at
                ORDER BY r.mission, r.filename
            """).fetchall()

        results = []
        for r in rows:
            filename, mis, source_url, acquired_at, mk_path, n_kernels = r
            # Count how many kernels are available in the DB
            available = self.con.execute("""
                SELECT COUNT(*) FROM metakernel_entries e
                WHERE e.mk_path = ?
                  AND EXISTS (
                    SELECT 1 FROM kernels k
                    WHERE k.filename = e.filename
                  )
            """, [mk_path]).fetchone()[0]
            results.append({
                "filename": filename,
                "mission": mis,
                "source_url": source_url,
                "acquired_at": acquired_at,
                "mk_path": mk_path,
                "n_kernels": n_kernels,
                "n_available": available,
            })

        if not results:
            print("No tracked metakernels.")
            return results

        # Print summary table
        name_w = max(len(r["filename"]) for r in results)
        name_w = max(name_w, 12)
        mis_w = max(len(r["mission"]) for r in results)
        mis_w = max(mis_w, 7)

        print("\nTracked metakernels:\n")
        print(f"  {'Mission':<{mis_w}}  {'Metakernel':<{name_w}}  {'Kernels':>7}  {'Available':>9}  Source")
        print(f"  {'─' * mis_w}  {'─' * name_w}  {'─' * 7}  {'─' * 9}  {'─' * 20}")
        for r in results:
            avail_str = f"{r['n_available']}/{r['n_kernels']}"
            source = r["source_url"] or ""
            # Shorten source URL to host
            if "://" in source:
                source_short = source.split("://", 1)[1].split("/", 1)[0]
            else:
                source_short = source
            print(
                f"  {r['mission']:<{mis_w}}  {r['filename']:<{name_w}}"
                f"  {r['n_kernels']:>7}  {avail_str:>9}  {source_short}"
            )
        print()
        return results

    def info_metakernel(self, name: str) -> dict | None:
        """Show detailed info about a tracked metakernel.

        Looks up by filename (or path) in metakernel_registry.
        """
        row = self.con.execute("""
            SELECT mk_path, mission, source_url, filename, acquired_at
            FROM metakernel_registry
            WHERE filename = ? OR mk_path = ?
        """, [name, name]).fetchone()

        if row is None:
            print(f"Metakernel not found: {name}")
            return None

        mk_path, mission, source_url, filename, acquired_at = row

        # Get kernel entries
        entries = self.con.execute("""
            SELECT entry_index, raw_entry, filename
            FROM metakernel_entries
            WHERE mk_path = ?
            ORDER BY entry_index
        """, [mk_path]).fetchall()

        # For each entry, check if in kernels table
        kernel_info = []
        n_in_db = 0
        for _, raw_entry, entry_fname in entries:
            k_row = self.con.execute("""
                SELECT kernel_type, size_bytes FROM kernels
                WHERE filename = ?
            """, [entry_fname]).fetchone()
            if k_row:
                ktype, size_bytes = k_row
                status = "in db"
                n_in_db += 1
            else:
                ktype = classify_kernel(entry_fname)
                size_bytes = None
                status = "missing"
            kernel_info.append({
                "filename": entry_fname,
                "kernel_type": ktype,
                "size_bytes": size_bytes,
                "status": status,
            })

        n_missing = len(entries) - n_in_db

        # Print detailed info
        print(f"\nMetakernel: {filename}")
        print(f"  Mission:      {mission}")
        if source_url:
            print(f"  Source:       {source_url}")
        print(f"  Acquired:     {acquired_at}")
        print(f"  Local path:   {mk_path}")

        if kernel_info:
            name_w = max(len(k["filename"]) for k in kernel_info)
            name_w = max(name_w, 10)
            print(f"\n  {'Kernel':<{name_w}}  {'Type':<9}  {'Size':>10}  Status")
            print(f"  {'─' * name_w}  {'─' * 9}  {'─' * 10}  {'─' * 7}")
            for k in kernel_info:
                sz_str = _format_size(k["size_bytes"]) if k["size_bytes"] is not None else "—"
                print(
                    f"  {k['filename']:<{name_w}}  {k['kernel_type']:<9}"
                    f"  {sz_str:>10}  {k['status']}"
                )

        print(f"\n  Total: {len(entries)} | In DB: {n_in_db} | Missing: {n_missing}")
        print()

        return {
            "filename": filename,
            "mission": mission,
            "source_url": source_url,
            "acquired_at": acquired_at,
            "mk_path": mk_path,
            "kernels": kernel_info,
            "n_kernels": len(entries),
            "n_in_db": n_in_db,
            "n_missing": n_missing,
        }

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
