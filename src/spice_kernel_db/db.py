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
from pathlib import Path
from typing import Optional

import duckdb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

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
    SPICE_SERVERS,
    download_kernel,
    download_kernels_parallel,
    fetch_metakernel,
    list_remote_metakernels,
    list_remote_missions,
    query_remote_sizes,
    resolve_kernel_urls,
)

logger = logging.getLogger(__name__)


def _validate_path_values(
    path_values: list[str],
    mk_dir: Path,
    download_dir: Path,
) -> None:
    """Validate that resolved PATH_VALUES stay within download_dir.

    Raises ValueError if any resolved path escapes the download directory.
    This prevents path traversal attacks via malicious metakernels.
    """
    for v in path_values:
        resolved = (mk_dir / v).resolve()
        try:
            resolved.relative_to(download_dir)
        except ValueError:
            raise ValueError(
                f"PATH_VALUE '{v}' resolves to {resolved} which is "
                f"outside download directory {download_dir}"
            )


def _should_skip_download(
    dest: Path,
    remote_size: int | None,
    db_hash: str | None,
    force: bool,
) -> bool:
    """Check whether a file on disk can be skipped (already correct).

    Returns True only if the file exists, size matches remote, AND
    the SHA-256 hash matches the database record.
    """
    if force:
        return False
    if not dest.is_file():
        return False
    if not remote_size or dest.stat().st_size != remote_size:
        return False
    if not db_hash:
        return False
    actual_hash = sha256_file(dest)
    return actual_hash == db_hash


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

    def __init__(self, db_path: str | Path | None = None, read_only: bool = False):
        if db_path is None:
            from spice_kernel_db.config import load_config
            config = load_config()
            db_path = config.db_path if config else "~/.spice_kernels.duckdb"
        self.db_path = str(Path(db_path).expanduser())
        self.read_only = read_only
        self.con = duckdb.connect(self.db_path, read_only=read_only)
        if not read_only:
            self._init_schema()

    def release(self):
        """Close the DB connection to release the lock.

        Use before long-running operations (downloads) so other
        processes can read the DB. Call :meth:`reacquire` to reopen.
        """
        self.con.close()
        self.con = None

    def reacquire(self):
        """Reopen the DB connection after :meth:`release`."""
        self.con = duckdb.connect(self.db_path, read_only=self.read_only)

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
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS missions (
                name         VARCHAR PRIMARY KEY,
                server_url   VARCHAR,
                mk_dir_url   VARCHAR,
                dedup        BOOLEAN DEFAULT TRUE,
                added_at     TIMESTAMP DEFAULT current_timestamp
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS kernel_coverage (
                sha256          VARCHAR NOT NULL,
                body_id         INTEGER NOT NULL,
                interval_index  INTEGER NOT NULL,
                et_start        DOUBLE NOT NULL,
                et_end          DOUBLE NOT NULL,
                PRIMARY KEY (sha256, body_id, interval_index)
            )
        """)

    # ------------------------------------------------------------------
    # Mission management
    # ------------------------------------------------------------------

    def add_mission(
        self,
        name: str,
        server_url: str,
        mk_dir_url: str,
        dedup: bool = True,
    ) -> None:
        """Register a mission in the database.

        Args:
            name: Mission name (e.g. "JUICE").
            server_url: Root server URL (e.g. NASA or ESA).
            mk_dir_url: Full URL to the mission's mk/ directory.
            dedup: Whether deduplication is enabled for this mission.
        """
        self.con.execute("""
            INSERT OR REPLACE INTO missions (name, server_url, mk_dir_url, dedup)
            VALUES (?, ?, ?, ?)
        """, [name, server_url, mk_dir_url, dedup])

    def list_missions(self) -> list[dict]:
        """List all configured missions."""
        rows = self.con.execute("""
            SELECT name, server_url, mk_dir_url, dedup, added_at
            FROM missions ORDER BY name
        """).fetchall()
        results = []
        for name, server_url, mk_dir_url, dedup, added_at in rows:
            # Determine server label
            server_label = "custom"
            for label, url in SPICE_SERVERS.items():
                if server_url == url:
                    server_label = label
                    break
            results.append({
                "name": name,
                "server_url": server_url,
                "server_label": server_label,
                "mk_dir_url": mk_dir_url,
                "dedup": dedup,
                "added_at": added_at,
            })
        return results

    def remove_mission(self, name: str) -> bool:
        """Remove a mission from the database (case-insensitive, prefix-matchable).

        Uses the same fuzzy lookup as ``get_mission``. Returns True if found.
        """
        mission = self.get_mission(name)
        if mission is None:
            return False
        self.con.execute("DELETE FROM missions WHERE name = ?", [mission["name"]])
        return True

    def get_mission(self, name: str) -> dict | None:
        """Look up a mission by name (case-insensitive, prefix-matchable).

        Tries exact match first, then case-insensitive prefix match.
        Returns None if no match or if the prefix is ambiguous.
        """
        row = self.con.execute("""
            SELECT name, server_url, mk_dir_url, dedup, added_at
            FROM missions WHERE LOWER(name) = LOWER(?)
        """, [name]).fetchone()
        if not row:
            # Try prefix match
            rows = self.con.execute("""
                SELECT name, server_url, mk_dir_url, dedup, added_at
                FROM missions WHERE LOWER(name) LIKE LOWER(?) || '%'
            """, [name]).fetchall()
            if len(rows) == 1:
                row = rows[0]
        if not row:
            return None
        name, server_url, mk_dir_url, dedup, added_at = row
        server_label = "custom"
        for label, url in SPICE_SERVERS.items():
            if server_url == url:
                server_label = label
                break
        return {
            "name": name,
            "server_url": server_url,
            "server_label": server_label,
            "mk_dir_url": mk_dir_url,
            "dedup": dedup,
            "added_at": added_at,
        }

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def register_file(
        self,
        path: str | Path,
        mission: str | None = None,
        source_url: str | None = None,
        archive_dir: str | Path | None = None,
        expected_hash: str | None = None,
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

        If ``expected_hash`` is provided, the computed hash is verified
        against it before storing. Raises ValueError on mismatch (Issue 7).
        """
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)

        h = sha256_file(p)

        # Issue 7: Verify expected hash if provided
        if expected_hash is not None and h != expected_hash:
            raise ValueError(
                f"Hash mismatch for {p.name}: computed {h[:16]}..., "
                f"expected {expected_hash[:16]}..."
            )
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
            # Issue 11: Check if same filename exists with different hash
            existing_by_name = self.con.execute(
                "SELECT sha256 FROM kernels WHERE filename = ?", [fname]
            ).fetchone()
            if existing_by_name and existing_by_name[0] != h:
                old_hash = existing_by_name[0]
                logger.warning(
                    "Kernel %s hash changed: %s -> %s. "
                    "Updating to new version (old record preserved).",
                    fname, old_hash[:16], h[:16],
                )
                # Insert the new hash record (old one remains for history)
                self.con.execute(
                    "INSERT INTO kernels VALUES (?, ?, ?, ?)",
                    [h, fname, ktype, size],
                )
            else:
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
    ) -> tuple[int, set[str]]:
        """Recursively scan a directory tree and register all kernel files.

        If ``archive_dir`` is set, files are moved into the archive and
        symlinks are left at the original locations.

        Returns ``(count, missions_found)`` — the number of files registered
        and the set of mission names detected during the scan.
        """
        if extensions is None:
            extensions = KERNEL_EXTENSIONS

        root = Path(root).expanduser().resolve()
        count = 0
        error_count = 0
        missions_found: set[str] = set()
        mk_files: list[tuple[Path, str]] = []  # (path, mission)
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in extensions:
                try:
                    m = mission or guess_mission(str(p))
                    self.register_file(
                        p, mission=m, archive_dir=archive_dir,
                    )
                    count += 1
                    missions_found.add(m)
                    if verbose:
                        print(f"  registered: {p.name}")
                    if p.suffix.lower() == ".tm":
                        mk_files.append((p, m))
                except (PermissionError, MemoryError, KeyboardInterrupt):
                    # Issue 8: Re-raise fatal errors
                    raise
                except OSError as e:
                    # Issue 8: Catch non-fatal OS errors (file not found, etc.)
                    error_count += 1
                    logger.warning("Could not register %s: %s", p, e)
                except Exception:
                    # Issue 8: Re-raise unexpected exceptions
                    raise

        # Index metakernels into the registry
        for mk_path, m in mk_files:
            try:
                self.index_metakernel(mk_path)
                self.con.execute("""
                    INSERT OR REPLACE INTO metakernel_registry
                    VALUES (?, ?, NULL, ?, current_timestamp)
                """, [str(mk_path), m, mk_path.name])
            except (PermissionError, MemoryError, KeyboardInterrupt):
                raise
            except OSError as e:
                error_count += 1
                logger.warning("Could not index metakernel %s: %s", mk_path, e)
            except Exception:
                raise

        print(f"Scanned {root}: {count} kernel files registered.")
        if mk_files:
            print(f"  Indexed {len(mk_files)} metakernels.")
        if error_count:
            print(f"  Errors: {error_count} files could not be registered.")
        return count, missions_found

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

    def _find_by_path_suffix(self, filename: str) -> list[dict]:
        """Find locations where the abs_path ends with the given filename.

        This catches kernels that are registered in the DB under a different
        canonical name but exist on disk at a path ending with *filename*.
        """
        rows = self.con.execute("""
            SELECT DISTINCT k.sha256, l.abs_path, l.mission,
                   k.kernel_type, k.size_bytes, k.filename AS canonical_name
            FROM locations l
            JOIN kernels k ON k.sha256 = l.sha256
            WHERE l.abs_path LIKE '%/' || ? OR l.abs_path LIKE '%\\' || ?
            ORDER BY l.mission
        """, [filename, filename]).fetchall()
        return [
            {"sha256": r[0], "abs_path": r[1], "mission": r[2],
             "kernel_type": r[3], "size_bytes": r[4],
             "canonical_name": r[5]}
            for r in rows
        ]

    def resolve_kernel(
        self,
        filename: str,
        preferred_mission: str | None = None,
    ) -> tuple[str | None, list[str]]:
        """Resolve a kernel filename to an absolute path on disk.

        Mission-aware resolution order:
          1. Exact filename match in preferred_mission
          2. Exact filename match in any mission
          3. Path-suffix match (file on disk registered under different name)
          4. None — suggest ``spice-kernel-db scan`` to re-index

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

        # --- 3. Path-suffix match ---
        path_hits = self._find_by_path_suffix(filename)
        if preferred_mission and path_hits:
            same_mission = [
                h for h in path_hits
                if h["mission"].lower() == preferred_mission.lower()
                and Path(h["abs_path"]).is_file()
            ]
            if same_mission:
                h = same_mission[0]
                if h["canonical_name"] != filename:
                    warnings.append(
                        f"{filename}: found on disk, registered as "
                        f"{h['canonical_name']} in [{h['mission']}]"
                    )
                return h["abs_path"], warnings

        for h in path_hits:
            if Path(h["abs_path"]).is_file():
                if h["canonical_name"] != filename:
                    warnings.append(
                        f"{filename}: found on disk, registered as "
                        f"{h['canonical_name']} in [{h['mission']}]"
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
            console.print(Panel(
                f"[bold]{len(results)}[/bold] files with {min_copies}+ copies\n"
                f"Total wasted space: [bold]{total_waste / 1e6:.1f} MB[/bold]",
                title="Duplicate kernels",
            ))
            for d in results[:30]:
                console.print(
                    f"\n  [bold]{d['filename']}[/bold]  "
                    f"({d['size_bytes'] / 1e6:.1f} MB × {d['count']} copies)"
                )
                for p, m in zip(d["paths"], d["missions"]):
                    console.print(f"    [{m}] {p}")
        else:
            console.print("No duplicates found.")

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
        verbose: bool = False,
    ) -> dict:
        """Check which kernels from a metakernel are already available.

        Args:
            mk_path: Path to the .tm file.
            mission: Mission name for preferred-mission resolution.
            verbose: If True, print full per-file warnings instead of summary.

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
            if verbose:
                print("  Warnings:")
                for w in all_warnings:
                    print(f"    ⚠ {w}")
            else:
                # Categorise and summarise
                n_hash = sum(1 for w in all_warnings if "matched by content hash" in w or "matched by hash" in w)
                n_cross = sum(1 for w in all_warnings if "using copy from" in w)
                n_other = len(all_warnings) - n_hash - n_cross
                parts = []
                if n_hash:
                    parts.append(f"    Resolved via dedup:     {n_hash} kernels (same content, different filename)")
                if n_cross:
                    parts.append(f"    Cross-mission:          {n_cross} kernels")
                if n_other:
                    parts.append(f"    Other warnings:         {n_other}")
                if parts:
                    print("  Warnings:")
                    for p in parts:
                        print(p)
                    print("    (use -v for full details)")

        # Check for remote updates
        self._check_remote_staleness(mk_path, mission)

        return {"found": found, "missing": missing, "warnings": all_warnings}

    def _check_remote_staleness(
        self,
        mk_path: str | Path,
        mission: str | None,
    ) -> None:
        """Print a notice if the remote metakernel is newer than the local copy.

        This is a read-only check — it never downloads. Network errors are
        silently swallowed so ``check`` keeps working offline.
        """
        mk_path_str = str(mk_path)
        # Look up source_url and acquired_at from registry
        row = self.con.execute("""
            SELECT source_url, acquired_at, filename
            FROM metakernel_registry WHERE mk_path = ?
        """, [mk_path_str]).fetchone()
        if not row:
            return
        source_url, acquired_at, mk_filename = row
        if not acquired_at:
            return

        # Determine mk_dir_url
        mk_dir_url = None
        if source_url:
            mk_dir_url = source_url.rsplit("/", 1)[0] + "/"
        elif mission:
            m = self.get_mission(mission)
            if m and m.get("mk_dir_url"):
                mk_dir_url = m["mk_dir_url"]
        if not mk_dir_url:
            return

        try:
            entries = list_remote_metakernels(mk_dir_url)
        except Exception:
            return

        # Find matching remote entry
        acq_str = str(acquired_at)[:16]  # "YYYY-MM-DD HH:MM"
        for entry in entries:
            if entry.filename == mk_filename:
                if entry.date > acq_str:
                    print(
                        f"\n  Remote update available: server modified {entry.date}"
                        f" (acquired {acq_str})"
                    )
                    print(
                        f"  Run 'spice-kernel-db update {mk_filename}' to fetch the latest version."
                    )
                break

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

            link_path = (link_root / rel).resolve()

            # Issue 9: Validate that link_path stays within link_root
            try:
                link_path.relative_to(link_root)
            except ValueError:
                logger.warning(
                    "Path traversal in kernel relpath '%s': resolved to %s "
                    "which is outside link_root %s. Skipping.",
                    rel, link_path, link_root,
                )
                all_warnings.append(
                    f"{fname}: path traversal — resolved outside link_root, skipped"
                )
                continue

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

    def _snapshot_kernel_hashes(
        self, filenames: list[str], mission: str,
    ) -> dict[str, set[str]]:
        """Take a snapshot of kernel hashes for given filenames.

        Returns a mapping of filename -> set of sha256 hashes.
        Used for race condition detection (Issue 5).
        """
        snapshot: dict[str, set[str]] = {}
        for fname in filenames:
            rows = self.con.execute(
                "SELECT sha256 FROM kernels WHERE filename = ?",
                [fname],
            ).fetchall()
            snapshot[fname] = {r[0] for r in rows}
        return snapshot

    def _check_state_changed(
        self,
        pre_snapshot: dict[str, set[str]],
        filenames: list[str],
        mission: str,
    ) -> bool:
        """Check if kernel records changed since the pre-snapshot.

        Returns True if any change detected, and logs a warning.
        """
        current = self._snapshot_kernel_hashes(filenames, mission)
        changed_files = [
            f for f in filenames
            if pre_snapshot.get(f, set()) != current.get(f, set())
        ]
        if changed_files:
            logger.warning(
                "DB state changed during download for %d kernel(s): %s. "
                "Another process may have modified the database.",
                len(changed_files),
                ", ".join(changed_files[:5]),
            )
            return True
        return False

    def _link_existing_kernels(
        self,
        indices: list[int],
        filenames: list[str],
        relpaths: list[str],
        download_dir: Path,
        mission: str,
    ) -> int:
        """Create symlinks for kernels already in the DB.

        For each kernel index in *indices*, resolve its local path and
        create a symlink at ``download_dir / mission / relpath`` so the
        metakernel's relative paths work without rewriting.

        Returns the number of symlinks created.
        """
        n_linked = 0
        for i in indices:
            expected = download_dir / mission / relpaths[i]
            if expected.exists() or expected.is_symlink():
                continue
            local, _ = self.resolve_kernel(
                filenames[i], preferred_mission=mission,
            )
            if local and Path(local).is_file():
                # Issue 3: Verify hash before creating symlink
                db_row = self.con.execute(
                    "SELECT sha256 FROM kernels WHERE filename = ?",
                    [filenames[i]],
                ).fetchone()
                if db_row:
                    actual_hash = sha256_file(local)
                    if actual_hash != db_row[0]:
                        logger.warning(
                            "Hash mismatch for %s: expected %s, got %s. "
                            "Skipping symlink.",
                            filenames[i], db_row[0][:16], actual_hash[:16],
                        )
                        continue
                # Issue 6: Wrap symlink creation in try/except
                try:
                    expected.parent.mkdir(parents=True, exist_ok=True)
                    expected.symlink_to(Path(local).resolve())
                    n_linked += 1
                except OSError as e:
                    logger.warning(
                        "Could not create symlink for %s at %s: %s",
                        filenames[i], expected, e,
                    )
        return n_linked

    def get_metakernel(
        self,
        url: str,
        download_dir: str | Path | None = None,
        mission: str | None = None,
        yes: bool = False,
        force: bool = False,
    ) -> dict:
        """Fetch a remote metakernel, show status, and download missing kernels.

        Args:
            url: URL to a remote .tm metakernel file.
            download_dir: Where to store downloaded kernels. Preserves the
                remote subdirectory structure (lsk/, spk/, etc.).
            mission: Override auto-detected mission name.
            yes: If True, skip the confirmation prompt.
            force: If True, treat all kernels as missing and re-download.

        Returns:
            dict with keys: found, missing, downloaded, warnings
        """
        # 0. Resolve download_dir early (needed for saving .tm)
        if download_dir is None:
            download_dir = Path("~/.local/share/spice-kernel-db/kernels").expanduser()
        download_dir = Path(download_dir).expanduser().resolve()

        # 1. Fetch and parse
        console.print(f"Fetching [bold]{url}[/bold] ...")
        text = fetch_metakernel(url)
        parsed = parse_metakernel_text(text, url)
        if mission is None:
            mission = guess_mission(url)
        # Resolve to canonical mission name from DB (e.g. "juice" → "JUICE")
        m = self.get_mission(mission)
        if m:
            mission = m["name"]

        # 1b. Save .tm file to disk with absolute PATH_VALUES so it works
        # from any working directory (SPICE resolves paths relative to CWD,
        # not relative to the .tm file).
        mk_filename = url.rsplit("/", 1)[-1]
        mk_dest = download_dir / mission / "mk" / mk_filename
        mk_dest.parent.mkdir(parents=True, exist_ok=True)

        # Resolve each PATH_VALUE relative to the mk/ directory and make absolute
        # Validate that resolved paths stay within download_dir (Issue 1)
        mk_dir = mk_dest.parent
        _validate_path_values(parsed.path_values, mk_dir, download_dir)
        abs_path_values = [
            str((mk_dir / v).resolve()) for v in parsed.path_values
        ]
        write_metakernel(parsed, mk_dest, path_values=abs_path_values)
        self.index_metakernel(mk_dest)
        self.con.execute("""
            INSERT OR REPLACE INTO metakernel_registry
            VALUES (?, ?, ?, ?, current_timestamp)
        """, [str(mk_dest), mission, url, mk_filename])

        # 1c. Write .spice-server marker for scan auto-detection
        mk_dir_url = url.rsplit("/", 1)[0] + "/"
        server_url = ""
        if m:
            server_url = m["server_url"]
        else:
            for label, surl in SPICE_SERVERS.items():
                if url.startswith(surl):
                    server_url = surl
                    break
        marker = download_dir / mission / ".spice-server"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            f"server_url={server_url}\nmk_dir_url={mk_dir_url}\n"
        )

        # 2. Resolve kernel URLs
        kernel_urls = resolve_kernel_urls(url, parsed)
        relpaths = parsed.kernel_relpaths()
        filenames = parsed.kernel_filenames()

        # 3. Check local DB for each kernel
        # When dedup is disabled for this mission, skip DB lookups and
        # download every kernel fresh — no symlinks to existing copies.
        mission_dedup = True
        if m:
            mission_dedup = m.get("dedup", True)

        found_indices: list[int] = []
        missing_indices: list[int] = []
        if force or not mission_dedup:
            # Force / no-dedup: treat everything as missing, download fresh
            missing_indices = list(range(len(filenames)))
        else:
            for i, fname in enumerate(filenames):
                local, _ = self.resolve_kernel(fname, preferred_mission=mission)
                if local and Path(local).is_file():
                    found_indices.append(i)
                else:
                    missing_indices.append(i)

        # 4. Query remote sizes (parallel HEAD) — skip for --force
        if force:
            sizes: dict = {}
        else:
            # Issue 5: Snapshot kernel state before releasing lock
            pre_snapshot = self._snapshot_kernel_hashes(filenames, mission)
            # Release lock during network I/O so other processes aren't blocked.
            self.release()
            try:
                sizes = query_remote_sizes(kernel_urls)
            finally:
                self.reacquire()
            # Issue 5: Check if state changed during lock release
            self._check_state_changed(pre_snapshot, filenames, mission)

        # 5. Display table
        mk_name = url.rsplit("/", 1)[-1]
        total = len(filenames)
        n_found = len(found_indices)
        n_missing = len(missing_indices)
        download_bytes = sum(
            sizes.get(kernel_urls[i]) or 0 for i in missing_indices
        )

        table = Table(title=f"Metakernel: {mk_name} ({total} kernels)")
        table.add_column("Kernel")
        table.add_column("Size", justify="right")
        table.add_column("Status")
        for i in range(total):
            fname = filenames[i]
            sz = sizes.get(kernel_urls[i])
            sz_str = _format_size(sz) if sz is not None else "unknown"
            status = "in db" if i in found_indices else "missing"
            style = "green" if status == "in db" else "red"
            table.add_row(fname, sz_str, f"[{style}]{status}[/{style}]")
        console.print(table)

        console.print(Panel(
            f"Total: [bold]{total}[/bold] | "
            f"In DB: [green]{n_found}[/green] | "
            f"Missing: [red]{n_missing}[/red] | "
            f"Download: [bold]{_format_size(download_bytes)}[/bold]",
            title=mk_name,
        ))

        if n_missing == 0:
            n_linked = 0
            if mission_dedup:
                n_linked = self._link_existing_kernels(
                    found_indices, filenames, relpaths, download_dir, mission,
                )
            lines = ["[green]All kernels already in database.[/green]"]
            if n_linked:
                lines.append(f"Linked {n_linked} existing kernels into download tree.")
            lines.append(f"\nMetakernel ready: [bold]{mk_dest}[/bold]")
            console.print(Panel("\n".join(lines), title="Result"))
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
                console.print("[dim]Aborted.[/dim]")
                n_linked = 0
                if mission_dedup and found_indices:
                    n_linked = self._link_existing_kernels(
                        found_indices, filenames, relpaths, download_dir, mission,
                    )
                lines = []
                if n_linked:
                    lines.append(f"Linked {n_linked} existing kernels into download tree.")
                lines.append(f"Metakernel ready: [bold]{mk_dest}[/bold]")
                console.print(Panel("\n".join(lines), title="Result"))
                return {
                    "found": found_indices,
                    "missing": missing_indices,
                    "downloaded": [],
                    "warnings": [],
                }

        # 7. Build download list, skipping files with correct size AND hash (Issue 2)
        tasks = []
        task_info: dict[str, str] = {}  # filename -> source_url
        already_on_disk: list[Path] = []
        for i in missing_indices:
            kurl = kernel_urls[i]
            relpath = relpaths[i]
            dest = download_dir / mission / relpath
            fname = filenames[i]
            remote_size = sizes.get(kurl)
            # Look up expected hash from DB
            db_row = self.con.execute(
                "SELECT sha256 FROM kernels WHERE filename = ?", [fname]
            ).fetchone()
            db_hash = db_row[0] if db_row else None
            if _should_skip_download(dest, remote_size, db_hash, force):
                already_on_disk.append(dest)
                task_info[fname] = kurl
                continue
            tasks.append((kurl, dest, fname))
            task_info[fname] = kurl

        if already_on_disk:
            console.print(f"  [dim]Skipped: {len(already_on_disk)} already downloaded[/dim]")
            download_bytes -= sum(f.stat().st_size for f in already_on_disk)

        # Issue 5: Snapshot before releasing for download
        pre_dl_snapshot = self._snapshot_kernel_hashes(filenames, mission)
        # Release the DB lock during download so other processes can
        # run read-only queries (resolve, list, check, etc.).
        self.release()
        try:
            dl_paths, warnings = download_kernels_parallel(
                tasks, total_bytes=download_bytes,
            )
        finally:
            self.reacquire()
        # Issue 5: Check state after download reacquire
        self._check_state_changed(pre_dl_snapshot, filenames, mission)

        # Register all files (downloaded + already on disk)
        downloaded: list[str] = []
        for dest in [*already_on_disk, *dl_paths]:
            kurl = task_info[dest.name]
            self.register_file(dest, mission=mission, source_url=kurl)
            downloaded.append(str(dest))

        # 8. Create symlinks for "in db" kernels so the metakernel works locally
        # Only when dedup is enabled — otherwise all kernels were downloaded fresh.
        n_linked = 0
        if mission_dedup and found_indices:
            n_linked = self._link_existing_kernels(
                found_indices, filenames, relpaths, download_dir, mission,
            )

        lines = [f"Downloaded: [bold]{len(dl_paths)}/{n_missing}[/bold]"]
        if already_on_disk:
            lines.append(f"Skipped:    {len(already_on_disk)} (already on disk)")
        if n_linked:
            lines.append(f"Linked:     {n_linked} existing kernels")
        if warnings:
            for w in warnings:
                lines.append(f"[yellow]⚠ {w}[/yellow]")
        lines.append(f"\nMetakernel ready: [bold]{mk_dest}[/bold]")
        console.print(Panel("\n".join(lines), title="Result"))

        return {
            "found": found_indices,
            "missing": missing_indices,
            "downloaded": downloaded,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Update (re-fetch remote metakernel)
    # ------------------------------------------------------------------

    def update_metakernel(
        self,
        mk_path_or_name: str | Path,
        mission: str | None = None,
        download_dir: str | Path | None = None,
        yes: bool = False,
        force: bool = False,
    ) -> dict:
        """Re-fetch a metakernel from its source URL and download new kernels.

        Looks up the source URL from ``metakernel_registry``, falling back to
        the mission's ``mk_dir_url`` + filename if no source URL is stored.

        Args:
            mk_path_or_name: mk_path or filename of a registered metakernel.
            mission: Override mission name.
            download_dir: Override kernel download directory.
            yes: Skip confirmation prompt.

        Returns:
            dict from ``get_metakernel``.
        """
        # Look up by mk_path first, then by filename
        row = self.con.execute("""
            SELECT mk_path, source_url, filename, mission
            FROM metakernel_registry
            WHERE mk_path = ? OR filename = ?
        """, [str(mk_path_or_name), str(mk_path_or_name)]).fetchone()

        if not row:
            raise LookupError(
                f"Metakernel not found in registry: {mk_path_or_name}"
            )

        mk_path, source_url, mk_filename, reg_mission = row
        mission = mission or reg_mission

        if not source_url:
            # Try to derive from mission's mk_dir_url
            m = self.get_mission(mission) if mission else None
            if m and m.get("mk_dir_url"):
                source_url = m["mk_dir_url"] + mk_filename
            else:
                raise LookupError(
                    "This metakernel was added via scan, not downloaded. "
                    "Use 'get' with a URL."
                )

        result = self.get_metakernel(
            source_url,
            download_dir=download_dir,
            mission=mission,
            yes=yes,
            force=force,
        )

        # Re-scan the kernel directories referenced by the metakernel
        # so that new/renamed files on disk get indexed.  Skip if --force
        # was used, since get_metakernel already re-downloaded and
        # registered everything.
        if force:
            return result

        from spice_kernel_db import parse_metakernel as _parse_mk

        # Find the local .tm — either the original mk_path or the one
        # get_metakernel just wrote.
        local_mk = Path(mk_path) if Path(mk_path).is_file() else None
        if local_mk is None:
            # get_metakernel writes to download_dir/mission/mk/filename
            if download_dir is None:
                download_dir = Path(
                    "~/.local/share/spice-kernel-db/kernels"
                ).expanduser()
            candidate = (
                Path(download_dir).expanduser().resolve()
                / mission / "mk" / mk_filename
            )
            if candidate.is_file():
                local_mk = candidate

        if local_mk is not None:
            try:
                parsed_local = _parse_mk(str(local_mk))
                # Resolve PATH_VALUES relative to the mk directory
                mk_dir = local_mk.resolve().parent
                scan_dirs = set()
                for pv in parsed_local.path_values:
                    resolved = (mk_dir / pv).resolve()
                    if resolved.is_dir():
                        scan_dirs.add(resolved)
                for sd in sorted(scan_dirs):
                    console.print(
                        f"[dim]Re-scanning {sd} for updated kernels...[/dim]"
                    )
                    self.scan_directory(str(sd), mission=mission)
            except Exception as e:
                logger.warning("Post-update rescan failed: %s", e)

        return result

    # ------------------------------------------------------------------
    # Metakernel listing / info
    # ------------------------------------------------------------------

    def list_metakernels(self, mission: str | None = None) -> list[dict]:
        """List all tracked metakernels, optionally filtered by mission.

        The *mission* filter is case-insensitive with prefix matching.
        """
        if mission:
            rows = self.con.execute("""
                SELECT r.filename, r.mission, r.source_url, r.acquired_at,
                       r.mk_path, COUNT(e.entry_index) AS n_kernels
                FROM metakernel_registry r
                LEFT JOIN metakernel_entries e ON e.mk_path = r.mk_path
                WHERE LOWER(r.mission) LIKE LOWER(?) || '%'
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
            results.append({
                "filename": filename,
                "mission": mis,
                "source_url": source_url,
                "acquired_at": acquired_at,
                "mk_path": mk_path,
                "n_kernels": n_kernels,
            })

        if not results:
            print("No tracked metakernels.")
            return results

        # Compute content fingerprints to detect identical metakernels
        import hashlib
        fingerprints: dict[str, str] = {}  # mk_path -> fingerprint
        for r in results:
            entry_rows = self.con.execute("""
                SELECT filename FROM metakernel_entries
                WHERE mk_path = ? ORDER BY filename
            """, [r["mk_path"]]).fetchall()
            key = "\n".join(row[0] for row in entry_rows)
            fingerprints[r["mk_path"]] = hashlib.md5(key.encode()).hexdigest()

        # Group by fingerprint: map fingerprint -> list of filenames
        from collections import defaultdict
        fp_groups: dict[str, list[str]] = defaultdict(list)
        for r in results:
            fp_groups[fingerprints[r["mk_path"]]].append(r["filename"])

        # For each result, find its duplicate reference (first in group)
        for r in results:
            fp = fingerprints[r["mk_path"]]
            group = fp_groups[fp]
            if len(group) > 1 and r["filename"] != group[0]:
                r["identical_to"] = group[0]
            else:
                r["identical_to"] = None

        # Print summary table
        table = Table(title="Tracked metakernels")
        table.add_column("Mission")
        table.add_column("Metakernel")
        table.add_column("Kernels", justify="right")
        table.add_column("Source")
        for r in results:
            source = r["source_url"] or ""
            if "://" in source:
                source_short = source.split("://", 1)[1].split("/", 1)[0]
            else:
                source_short = source
            name_cell = r["filename"]
            if r["identical_to"]:
                name_cell += f"\n[dim]↳ identical to {r['identical_to']}[/dim]"
            table.add_row(
                r["mission"], name_cell,
                str(r["n_kernels"]), source_short,
            )
        console.print(table)
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

        # For each entry, check availability using resolve_kernel
        kernel_info = []
        n_in_db = 0
        for _, raw_entry, entry_fname in entries:
            local, _ = self.resolve_kernel(
                entry_fname, preferred_mission=mission,
            )
            if local and Path(local).is_file():
                # Look up size from the resolved location
                k_row = self.con.execute("""
                    SELECT k.kernel_type, k.size_bytes FROM kernels k
                    JOIN locations l ON k.sha256 = l.sha256
                    WHERE l.abs_path = ?
                """, [local]).fetchone()
                if k_row:
                    ktype, size_bytes = k_row
                else:
                    ktype = classify_kernel(entry_fname)
                    size_bytes = None
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
            table = Table()
            table.add_column("Kernel")
            table.add_column("Type")
            table.add_column("Size", justify="right")
            table.add_column("Status")
            for k in kernel_info:
                sz_str = _format_size(k["size_bytes"]) if k["size_bytes"] is not None else "—"
                table.add_row(
                    k["filename"], k["kernel_type"], sz_str, k["status"],
                )
            console.print(table)

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

    def remove_metakernel(self, name: str) -> bool:
        """Remove a metakernel from the registry and its entries.

        Looks up by filename or mk_path (case-insensitive prefix match on
        filename). Does NOT delete the .tm file from disk.

        Returns True if found and removed, False if not found.
        """
        row = self.con.execute("""
            SELECT mk_path, filename FROM metakernel_registry
            WHERE filename = ? OR mk_path = ?
               OR LOWER(filename) LIKE LOWER(?) || '%'
        """, [name, name, name]).fetchone()

        if not row:
            return False

        mk_path, filename = row
        self.con.execute(
            "DELETE FROM metakernel_entries WHERE mk_path = ?", [mk_path],
        )
        self.con.execute(
            "DELETE FROM metakernel_registry WHERE mk_path = ?", [mk_path],
        )
        return True

    # ------------------------------------------------------------------
    # Browse remote metakernels
    # ------------------------------------------------------------------

    def browse_remote_metakernels(
        self,
        mk_dir_url: str,
        mission: str | None = None,
        show_versioned: bool = False,
    ) -> list[dict]:
        """Scan a remote NAIF mk/ directory and show available metakernels.

        Groups metakernels by base name (stripping version tags like
        ``_v461_20251127_001``), counts versioned snapshots, and checks
        which base metakernels have been locally acquired.

        Args:
            mk_dir_url: URL to a mission's ``mk/`` directory.
            mission: Override auto-detected mission name.
            show_versioned: If True, list versioned snapshots under each
                base metakernel.

        Returns:
            List of dicts with keys: ``base_name``, ``n_versions``,
            ``latest_date``, ``is_local``, ``filenames``.
        """
        entries = list_remote_metakernels(mk_dir_url)
        if mission is None:
            mission = guess_mission(mk_dir_url)

        # Group by base_name
        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)
        for entry in entries:
            groups[entry.base_name].append(entry)

        # Get locally acquired metakernels with acquired_at for this mission
        if mission:
            local_rows = self.con.execute(
                "SELECT filename, source_url, acquired_at FROM metakernel_registry WHERE LOWER(mission) LIKE LOWER(?) || '%'",
                [mission],
            ).fetchall()
        else:
            local_rows = []
        local_info: dict[str, tuple[str | None, str | None]] = {}  # filename -> (source_url, acquired_at_str)
        for fname, source_url, acquired_at in local_rows:
            acq_str = str(acquired_at)[:16] if acquired_at else None  # "YYYY-MM-DD HH:MM"
            local_info[fname] = (source_url, acq_str)

        results: list[dict] = []
        has_outdated = False
        for base_name in sorted(groups):
            group = groups[base_name]
            latest_date = max(e.date for e in group)
            # Determine local status: "yes", "outdated", or "no"
            local_status = "no"
            for e in group:
                if e.filename in local_info:
                    source_url, acq_str = local_info[e.filename]
                    if source_url is None:
                        # Scan-acquired — can't determine staleness
                        local_status = "yes"
                    elif acq_str is None or e.date > acq_str:
                        local_status = "outdated"
                        has_outdated = True
                    else:
                        local_status = "yes"
                    break
            # Separate current (no version tag) from versioned snapshots
            current = [e for e in group if e.version_tag is None]
            versioned = sorted(
                [e for e in group if e.version_tag is not None],
                key=lambda e: e.filename,
            )
            results.append({
                "base_name": base_name,
                "n_versions": len(group),
                "latest_date": latest_date,
                "is_local": local_status != "no",
                "local_status": local_status,
                "filenames": [e.filename for e in group],
                "current": current,
                "versioned": versioned,
            })

        # Print table
        n_unique = len(results)
        n_files = len(entries)
        n_local = sum(1 for r in results if r["is_local"])

        # Shorten URL for display
        display_url = mk_dir_url
        if len(display_url) > 60:
            display_url = display_url[:57] + "..."

        print(f"\nRemote metakernels: {mission} ({display_url})\n")

        if not results:
            print("  No .tm files found.")
            return results

        def _local_str(status: str) -> str:
            if status == "outdated":
                return "[yellow]outdated[/yellow]"
            return status

        if show_versioned:
            # Expanded view: show each file, versioned snapshots indented
            table = Table()
            table.add_column("Metakernel")
            table.add_column("Date")
            table.add_column("Size", justify="right")
            table.add_column("Local")
            for r in results:
                ls = _local_str(r["local_status"])
                if r["current"]:
                    e = r["current"][0]
                    table.add_row(e.filename, e.date, e.size, ls)
                else:
                    table.add_row(r["base_name"], r["latest_date"], "", ls)
                for e in r["versioned"]:
                    table.add_row(
                        f"  [dim]snapshot: {e.filename}[/dim]",
                        f"[dim]{e.date}[/dim]",
                        f"[dim]{e.size}[/dim]",
                    )
            console.print(table)
        else:
            # Compact view: one line per unique metakernel
            table = Table()
            table.add_column("Metakernel")
            table.add_column("Versions", justify="right")
            table.add_column("Latest")
            table.add_column("Local")
            for r in results:
                ls = _local_str(r["local_status"])
                table.add_row(
                    r["base_name"], str(r["n_versions"]),
                    r["latest_date"], ls,
                )
            console.print(table)

        print(
            f"\n  Total: {n_unique} unique | {n_files} files"
            f" | {n_local} locally acquired"
        )
        if has_outdated:
            print("  Run 'spice-kernel-db get <name>' to update outdated metakernels.")
        if not show_versioned and n_files > n_unique:
            print("  Use --show-versioned to also show the identical but versioned snapshot metakernels.")
        print()

        return results

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

        console.print(Panel(
            f"Unique kernels:   [bold]{n_kernels}[/bold]\n"
            f"Total locations:  {n_locs}\n"
            f"Unique content:   {total_size / 1e9:.2f} GB\n"
            f"Duplicated files: {dups}\n"
            f"Missions:         {', '.join(missions)}",
            title=f"SPICE Kernel Database: {self.db_path}",
        ))
        type_table = Table(title="By type")
        type_table.add_column("Type")
        type_table.add_column("Files", justify="right")
        type_table.add_column("Size", justify="right")
        for ktype, cnt, sz in by_type:
            type_table.add_row(ktype, str(cnt), f"{sz / 1e6:.1f} MB")
        console.print(type_table)

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
        """Generate a deduplication plan respecting per-mission settings.

        For each set of identical files (same hash, multiple locations),
        pick one canonical location (preferring generic_kernels) and list
        the rest as removable — but only for missions with dedup enabled.

        Missions not in the ``missions`` table default to dedup=True.

        Returns list of: {filename, size_bytes, keep, remove: [paths]}
        """
        # Get missions with dedup explicitly disabled
        no_dedup = {
            r[0].lower()
            for r in self.con.execute(
                "SELECT name FROM missions WHERE dedup = FALSE"
            ).fetchall()
        }

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
            # Only remove files from dedup-enabled missions
            remove = [
                p for p, m in zip(paths, missions)
                if p != keep and m.lower() not in no_dedup
            ]
            if remove:
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

    def prune(self, dry_run: bool = True) -> list[str]:
        """Remove stale locations where the file no longer exists on disk.

        Also removes kernels that have zero remaining locations after pruning.

        Args:
            dry_run: If True (default), only report what would be removed.

        Returns:
            List of pruned abs_path strings.
        """
        rows = self.con.execute(
            "SELECT sha256, abs_path FROM locations"
        ).fetchall()

        stale: list[tuple[str, str]] = []
        for sha256, abs_path in rows:
            if not Path(abs_path).exists():
                stale.append((sha256, abs_path))

        if not stale:
            print("No stale locations found.")
            return []

        action = "Would remove" if dry_run else "Removed"
        pruned_paths = []
        for sha256, abs_path in stale:
            print(f"  {action}: {abs_path}")
            pruned_paths.append(abs_path)
            if not dry_run:
                self.con.execute(
                    "DELETE FROM locations WHERE sha256 = ? AND abs_path = ?",
                    [sha256, abs_path],
                )

        # Remove orphaned kernels (no remaining locations)
        if not dry_run:
            orphaned = self.con.execute("""
                SELECT k.sha256, k.filename FROM kernels k
                LEFT JOIN locations l ON k.sha256 = l.sha256
                WHERE l.sha256 IS NULL
            """).fetchall()
            for sha256, filename in orphaned:
                self.con.execute(
                    "DELETE FROM kernels WHERE sha256 = ?", [sha256],
                )
                print(f"  Removed orphaned kernel record: {filename}")

        print(f"\n{action}: {len(stale)} stale location(s)")
        return pruned_paths

    # ------------------------------------------------------------------
    # Coverage analysis
    # ------------------------------------------------------------------

    def coverage_metakernel(
        self,
        mk_path: str | Path,
        body_id: int,
        mission: str | None = None,
    ) -> list:
        """Check body coverage for all SPK kernels in a metakernel.

        Follows the ``check_metakernel`` pattern: parse → resolve → classify
        → delegate to ``coverage.check_coverage()``. Results are stored in
        the ``kernel_coverage`` table.

        If *mk_path* does not exist on disk, falls back to looking up the
        filename in metakernel_registry.

        Returns:
            list of KernelCoverageResult
        """
        from spice_kernel_db.coverage import check_coverage

        mk = Path(mk_path)
        if not mk.is_file():
            # Try registry lookup by filename
            row = self.con.execute(
                "SELECT mk_path FROM metakernel_registry WHERE filename = ?",
                [str(mk_path)],
            ).fetchone()
            if row:
                mk = Path(row[0])
            else:
                raise FileNotFoundError(
                    f"Metakernel not found: {mk_path}"
                )

        parsed = parse_metakernel(mk)
        if mission is None:
            mission = guess_mission(str(mk))

        filenames = parsed.kernel_filenames()
        resolved_paths: list[str | None] = []
        kernel_types: list[str] = []

        # Auto-discover LSK for UTC display
        lsk_path: str | None = None

        for raw in parsed.kernels:
            fname = Path(raw).name
            ktype = classify_kernel(fname)
            kernel_types.append(ktype)
            local, _ = self.resolve_kernel(fname, preferred_mission=mission)
            resolved_paths.append(local)
            if ktype == "lsk" and local and Path(local).is_file():
                lsk_path = local

        results = check_coverage(
            filenames, resolved_paths, kernel_types, body_id,
            lsk_path=lsk_path,
        )

        # Store coverage in DB
        for res in results:
            if res.intervals and res.kernel_type == "spk":
                # Look up sha256 for the resolved path
                idx = filenames.index(res.filename)
                rpath = resolved_paths[idx]
                if rpath:
                    sha_row = self.con.execute(
                        "SELECT sha256 FROM locations WHERE abs_path = ?",
                        [rpath],
                    ).fetchone()
                    if sha_row:
                        self.store_coverage(
                            sha_row[0], body_id, res.intervals,
                        )

        return results

    def store_coverage(
        self,
        sha256: str,
        body_id: int,
        intervals: list,
    ) -> None:
        """Upsert coverage intervals into kernel_coverage table."""
        # Delete existing intervals for this kernel+body
        self.con.execute(
            "DELETE FROM kernel_coverage WHERE sha256 = ? AND body_id = ?",
            [sha256, body_id],
        )
        for i, iv in enumerate(intervals):
            self.con.execute(
                "INSERT INTO kernel_coverage VALUES (?, ?, ?, ?, ?)",
                [sha256, body_id, i, iv.et_start, iv.et_end],
            )

    def query_coverage(
        self,
        body_id: int,
        et_start: float | None = None,
        et_end: float | None = None,
    ) -> list[dict]:
        """Query kernels that cover a body, optionally within a time range.

        Uses interval overlap: et_start < row.et_end AND et_end > row.et_start

        Returns:
            list of dicts with sha256, filename, et_start, et_end
        """
        if et_start is not None and et_end is not None:
            rows = self.con.execute("""
                SELECT DISTINCT kc.sha256, k.filename,
                       kc.et_start, kc.et_end
                FROM kernel_coverage kc
                JOIN kernels k ON k.sha256 = kc.sha256
                WHERE kc.body_id = ?
                  AND kc.et_start < ?
                  AND kc.et_end > ?
                ORDER BY kc.et_start
            """, [body_id, et_end, et_start]).fetchall()
        else:
            rows = self.con.execute("""
                SELECT DISTINCT kc.sha256, k.filename,
                       kc.et_start, kc.et_end
                FROM kernel_coverage kc
                JOIN kernels k ON k.sha256 = kc.sha256
                WHERE kc.body_id = ?
                ORDER BY kc.et_start
            """, [body_id]).fetchall()

        return [
            {
                "sha256": r[0], "filename": r[1],
                "et_start": r[2], "et_end": r[3],
            }
            for r in rows
        ]

    def close(self):
        """Close the database connection."""
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
