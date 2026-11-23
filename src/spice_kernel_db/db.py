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

from typing import NamedTuple

import http.client
import logging
import os
import shutil
import urllib.error
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from spice_kernel_db.config import kernel_roots
from spice_kernel_db.hashing import (
    KERNEL_EXTENSIONS,
    canonicalize_mission,
    classify_kernel,
    guess_mission,
    root_containing,
    sha256_file,
    strip_version_tag,
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
    _fetch_metakernel,
    list_remote_metakernels,
    list_remote_missions,
    query_remote_sizes,
    server_label_for,
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


def _safe_join(root: Path, relpath: str) -> Path | None:
    """Join *relpath* onto *root* and confirm the result stays inside root.

    Returns the resolved absolute path on success, or None if *relpath*
    is absolute, contains traversal that escapes *root*, or otherwise
    resolves outside *root*. Uses lexical normalisation rather than
    `Path.resolve()` so existing symlinks cannot mask traversal intent.
    """
    import os.path
    if not relpath:
        return None
    if os.path.isabs(relpath):
        return None
    root_abs = Path(os.path.abspath(root))
    joined = os.path.normpath(os.path.join(str(root_abs), relpath))
    try:
        Path(joined).relative_to(root_abs)
    except ValueError:
        return None
    return Path(joined)


def _atomic_symlink(target: str | Path, link_path: Path) -> None:
    """Create or replace a symlink at *link_path* atomically (C7).

    Writes a temp symlink in the same directory then ``os.replace``
    onto *link_path* — this is atomic on POSIX even when *link_path*
    already exists. Concurrent readers never see a missing link.
    """
    import os
    import uuid
    tmp = link_path.with_name(link_path.name + f".tmp.{uuid.uuid4().hex[:8]}")
    try:
        os.symlink(os.fspath(target), tmp)
        os.replace(tmp, link_path)
    except Exception:  # broad on purpose: cleanup only, re-raised below
        if tmp.is_symlink() or tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _validate_relpaths(
    relpaths: list[str], root: Path,
) -> tuple[list[Path | None], list[str]]:
    """Validate every entry in *relpaths* against *root*.

    Returns (safe_paths, errors) — safe_paths[i] is the validated absolute
    Path or None if entry i was unsafe. errors is a list of human-readable
    descriptions for every unsafe entry. The caller decides whether to
    abort or skip individual entries based on context.
    """
    safe: list[Path | None] = []
    errors: list[str] = []
    for rel in relpaths:
        p = _safe_join(root, rel)
        if p is None:
            errors.append(f"unsafe relpath {rel!r}: escapes {root}")
        safe.append(p)
    return safe, errors


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


class ConcurrentModificationError(RuntimeError):
    """Another process modified the kernel records during a download.

    ``KernelDB.get_metakernel`` (and so ``update_metakernel`` and the module-level ``get``) releases
    the database lock while it talks to the network, so that other processes can read in the
    meantime. Afterwards it checks that the kernel records it looked at are unchanged. If another
    process changed them, the operation aborts with this error instead of silently overwriting that
    process's work (C5).

    Retry after the other writer has finished. If the error persists, another ``get`` or ``scan`` is
    still running. Subclass of ``RuntimeError``.
    """


class MetakernelNotTracked(LookupError):
    """A metakernel reference matched nothing that skd tracks.

    Raised when a reference is neither a file on disk nor a match in the metakernel registry, or
    when the matching registry row points at a file that is gone. Every method that resolves a
    metakernel reference can raise it, among them ``KernelDB.resolve_metakernel``,
    ``update_metakernel`` and ``info_metakernel``. Subclass of ``LookupError``.

    The default message carries the hint ``spice-kernel-db get <name>``. When the row's file is gone
    it names ``clean --registry`` and ``get`` instead. The CLI prints the message and exits with
    code 1.

    Issue #11: ``skd check <name>.tm`` used to hand the bare name straight to ``parse_metakernel``,
    which raised ``FileNotFoundError`` from inside the parser and printed a traceback. The name had
    not been downloaded yet.

    Attributes
    ----------
    name : str
        The reference as it was given.
    """

    def __init__(self, name: str, detail: str | None = None):
        self.name = name
        super().__init__(
            detail
            or f"'{name}' is not tracked locally; fetch it with "
               f"`spice-kernel-db get {name}`"
        )


class AmbiguousMetakernel(LookupError):
    """A metakernel reference matched several registry rows.

    Raised by ``KernelDB.resolve_metakernel`` (and so by every method that takes a metakernel
    reference) when the reference matches more than one row, for example a base name while two
    versioned snapshots are tracked. The resolver never picks the first match (#45). Subclass of
    ``LookupError``.

    The caller lists ``candidates`` and asks for a more specific name. The CLI does that and exits
    with code 2.

    Attributes
    ----------
    name : str
        The reference as it was given.
    candidates : list[str]
        The registry filenames that matched.
    """

    def __init__(self, name: str, candidates: list[str]):
        self.name = name
        self.candidates = candidates
        listed = ", ".join(candidates)
        super().__init__(
            f"'{name}' matches {len(candidates)} tracked metakernels: {listed}. "
            f"Pass one of them."
        )


class AmbiguousKernelRoot(LookupError):
    """Nothing says which configured kernel root a download belongs in.

    Raised by ``KernelDB.root_for`` when several kernel roots are configured and the available facts
    do not single one out. ``root_for`` derives the root from the metakernel's own path, then from
    the one configured root, then from where the mission's kernels already are. With several roots
    and none of those facts it never picks the first (decided 2026-11-21). ``get_metakernel``,
    ``update_metakernel`` and ``scan_directory(archive=True)`` call ``root_for``, so they raise it
    too. Subclass of ``LookupError``.

    At a terminal the CLI lists ``roots`` and asks. In a script it exits with code 2, and
    ``--download-dir`` is the explicit answer.

    Attributes
    ----------
    mission : str | None
        The mission the download is for, or ``None`` when no mission is involved.
    roots : list[Path]
        The candidates. When more than one root already holds kernels of the mission these are those
        roots, otherwise every configured root.
    """

    def __init__(self, mission: str | None, roots: list[Path], stored: bool = False):
        self.mission = mission
        self.roots = roots
        self.what = f"mission {mission}" if mission else "this download"
        listed = "\n  ".join(str(r) for r in roots)
        lead = (
            f"{self.what[0].upper()}{self.what[1:]} is stored in {len(roots)} kernel roots:"
            if stored else
            f"No configured kernel root holds {self.what} yet, and {len(roots)} roots are configured:"
        )
        super().__init__(f"{lead}\n  {listed}\nName the root.")


class RegistryRow(NamedTuple):
    """One tracked metakernel, as a row of the metakernel registry.

    Returned by ``KernelDB.registry_rows`` and ``KernelDB.registry_row``. The class is part of the
    public API because the module-level functions and planetarypy read it. A ``NamedTuple``, so a
    row unpacks in field order.

    Attributes
    ----------
    mk_path : str
        The metakernel file on disk.
    source_url : str | None
        Where ``get`` fetched it from. ``None`` for a file that was scanned or imported.
    filename : str
        The file's name.
    mission : str
        The mission the row belongs to.
    acquired_at : object
        The acquisition time stored with the row (a DuckDB timestamp).
    official : bool
        ``False`` for a metakernel assembled locally (``skd import --unofficial``), ``True``
        otherwise. Defaults to ``True``.
    """

    mk_path: str
    source_url: str | None
    filename: str
    mission: str
    acquired_at: object
    official: bool = True  # False for a metakernel assembled locally (#48)


def _check_verdicts(present: dict, resolves: dict, server: dict | None) -> list[dict]:
    """The three verdicts of ``check`` as ``{bit, ok, line}`` rows (#46).

    ``ok`` is True, False or None (not checked). A kernel missing from disk
    is verdict 1. Verdict 2 covers what the PATH_VALUES walk finds beyond
    that: traversal, size or hash against the database, an ambiguous name,
    and entries pointing at paths that do not exist when every kernel is
    present somewhere else, which is the case ``rewrite`` fixes.
    """
    n_missing = len(present["missing"])
    n_total = len(present["found"]) + n_missing
    line = f"present         {n_total - n_missing}/{n_total} kernels on disk"
    if n_missing:
        line += f", {n_missing} missing (spice-kernel-db get)"
    rows = [{"bit": 1, "ok": n_missing == 0, "line": line}]

    absent = {"NOT_FOUND", "DANGLING"}
    entries = resolves["entries"]
    hard = [e for e in entries if e["status"] not in {"OK", "UNREGISTERED"} | absent]
    misplaced = [e for e in entries if e["status"] in absent]
    unregistered = sum(1 for e in entries if e["status"] == "UNREGISTERED")
    if hard:
        ok, line = False, f"resolves        {len(hard)} entries do not resolve soundly (see table)"
    elif misplaced and not n_missing:
        ok, line = False, (f"resolves        {len(misplaced)} entries point at paths that do "
                           f"not exist (spice-kernel-db rewrite)")
    elif unregistered:
        ok, line = True, (f"resolves        every entry resolves; {unregistered} not in the "
                          f"database (spice-kernel-db scan)")
    else:
        ok, line = True, "resolves        every entry resolves and matches the database"
    rows.append({"bit": 2, "ok": ok, "line": line})

    if server is None:
        rows.append({"bit": 4, "ok": None, "line": "matches remote  not checked (pass --remote)"})
    elif server["bad"]:
        rows.append({"bit": 4, "ok": False,
                     "line": f"matches remote  {server['bad']} differ from the server"})
    elif not server["ok"]:
        detail = server["entries"][0]["detail"] if server["entries"] else "nothing to compare"
        rows.append({"bit": 4, "ok": None, "line": f"matches remote  could not compare: {detail}"})
    else:
        line = f"matches remote  {server['ok']} match the server"
        if server["unchecked"]:
            line += f", {server['unchecked']} unchecked"
        rows.append({"bit": 4, "ok": True, "line": line})
    return rows


class MetakernelUnreachableError(LookupError):
    """The remote copy of a tracked metakernel is gone.

    Raised by ``KernelDB.update_metakernel`` when the remote server answers with a permanent HTTP
    error (403, 404 or 410). The usual cause is that NAIF rotated the file into
    ``former_versions/``. The local registry row is then an obsolete pointer. Subclass of
    ``LookupError``.

    To clean it up, call ``db.prune_metakernels(dry_run=False)`` or run
    ``spice-kernel-db clean --registry``. ``update_all`` does not raise this error. It reports the
    row in its result.

    Attributes
    ----------
    url : str
        The URL that was requested.
    status : int
        The HTTP status code.
    filename : str
        The metakernel's filename.
    """

    def __init__(self, url: str, status: int, filename: str):
        self.url = url
        self.status = status
        self.filename = filename
        super().__init__(
            f"Metakernel {filename!r} is unreachable: HTTP {status} for {url}"
        )


def _prefer_exact_name(hits: list[dict], filename: str) -> list[dict]:
    """Order *hits* so locations actually named *filename* come first.

    `find_by_filename` joins on sha256, so two byte-identical files under
    different names share one `kernels` row and either name matches both
    `locations` rows. Without this the winner was whichever path the scanner
    happened to walk last, and resolving `juice_crema_5_2.tm` could hand back
    `juice_crema_5_2_v473_20260819_001.tm` (issue #9).

    Content-only matches are kept, just after the exact ones, because they are
    the point of the sha256 join: `jup365.bsp` resolving to a location named
    `jup365_19900101_20500101.bsp` is a legitimate hit when no file of the
    requested name is registered. Each group keeps `find_by_filename`'s
    deterministic order. C8: the comparison is case-insensitive.
    """
    wanted = filename.lower()
    # sorted() is stable, so each group keeps find_by_filename's ordering.
    return sorted(hits, key=lambda h: Path(h["abs_path"]).name.lower() != wanted)


def _warn_if_renamed(warnings: list[str], filename: str, hit: dict) -> None:
    """Record that *hit* is a content match under a different name."""
    actual = Path(hit["abs_path"]).name
    if actual.lower() != filename.lower():
        warnings.append(
            f"{filename}: no file of that name registered, using "
            f"byte-identical {actual} in [{hit['mission']}]"
        )


def _format_size(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"


class KernelDB:
    """Content-addressed SPICE kernel database.

    Wraps a DuckDB database and provides scanning, lookup, metakernel tracking, downloading,
    rewriting and deduplication. Every kernel file is identified by its SHA-256 hash, not by its
    filename. Several locations (from different missions, or the same mission in different
    directories) can reference the same hash, which is how duplicates are detected.

    The library is quiet. No method prints unless the instance was given a ``rich.console.Console``
    (``KernelDB(console=...)``), and warnings go through the ``spice_kernel_db`` logger, so a script
    sees them only if it configures logging. The CLI is this API plus a console and one log handler.

    For one operation against the configured database, the module-level functions ``check``,
    ``get``, ``scan`` and the others open the database, do one thing and close it again. Keep a
    ``KernelDB`` for several operations in a row. An instance is a context manager that closes the
    connection on exit.

    Parameters
    ----------
    db_path : str | Path | None, default None
        The DuckDB file to open, created if it does not exist. ``None`` reads the path from
        ``~/.config/spice-kernel-db/config.toml``. Without a config file it is
        ``~/.local/share/spice-kernel-db/kernels.duckdb``.
    read_only : bool, default False
        Open the database for queries only, and skip the schema initialisation. Several read-only
        connections can be open at once. DuckDB's file lock means none can open while another
        process has the database open for writing, for example during ``get``.
    kernel_roots : list[str | Path] | None, default None
        The kernel roots used to guess a mission from a path. ``None`` reads them from the
        configuration the first time they are needed. The resolved list is available as the
        ``kernel_roots`` property. Pass it to keep a script independent of the user's configuration.
    console : rich.console.Console | None, default None
        The one place the instance writes to: status tables, notes, progress bars and the download
        question of ``get_metakernel``. Without one nothing is printed, progress bars are off and
        every question is answered no, so a script passes ``yes=True`` where it wants a download.
        The CLI passes its own console.

    Attributes
    ----------
    db_path : str
        The expanded path of the open database file.
    read_only : bool
        Whether the database was opened read-only.
    console : rich.console.Console | None
        The console given at construction.
    kernel_roots : list[Path]
        The kernel roots, as expanded, resolved paths. Read from the configuration on first use
        unless they were passed to the constructor.

    Raises
    ------
    LegacyDatabaseFound
        ``db_path`` is ``None``, there is no config file, and a database sits at the pre-0.19
        default ``~/.spice_kernels.duckdb`` while the new default does not exist. The message names
        the move command. Before 0.21.0 that database was opened with a ``FutureWarning``. Since
        0.21.0 it is not, so that an empty database is not started quietly beside it.
    ConfigError
        ``db_path`` is ``None`` and the config file does not parse.
    duckdb.IOException
        ``read_only`` is ``True`` and another process holds the database open for writing.

    See Also
    --------
    check, get, scan, resolve_metakernel, metakernels_covering, register_metakernel :
        The module-level functions that wrap one call each.

    Notes
    -----
    The schema is created when the database is opened for writing. It has these tables:

    ``kernels``
        One row per unique file content: ``sha256`` (primary key), ``filename`` (the first name
        seen), ``kernel_type``, ``size_bytes`` and ``superseded_by``.
    ``locations``
        Where files live on disk: ``sha256`` and ``abs_path`` (primary key), ``mission``,
        ``source_url`` and ``scanned_at``. References ``kernels``.
    ``kernel_aliases``
        A view of ``locations``: every distinct on-disk basename a content hash appears under.
    ``metakernel_entries``
        The kernel entries inside each ``.tm`` file: ``mk_path`` and ``entry_index`` (primary key),
        ``raw_entry`` and ``filename``.
    ``metakernel_registry``
        The tracked metakernel files: ``mk_path`` (primary key), ``mission``, ``source_url``,
        ``filename``, ``acquired_at`` and ``official``.
    ``missions``
        The configured missions: ``name`` (primary key), ``server_url``, ``mk_dir_url``, ``dedup``
        and ``added_at``.
    ``kernel_coverage``
        Cached SPK coverage intervals per body: ``sha256``, ``body_id`` and ``interval_index``
        (primary key), ``et_start`` and ``et_end``.

    Examples
    --------
    >>> from spice_kernel_db import KernelDB
    >>> with KernelDB(read_only=True) as db:
    ...     path = db.resolve_metakernel("juice_crema_5_2.tm")
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        read_only: bool = False,
        kernel_roots: list[str | Path] | None = None,
        console: Console | None = None,
    ):
        """Open (or create) the database.

        When ``db_path`` is ``None`` the path is read from
        ``~/.config/spice-kernel-db/config.toml``, and without a config file it is
        ``~/.local/share/spice-kernel-db/kernels.duckdb``. The schema is initialised automatically
        unless ``read_only`` is set. The class docstring describes the tables.

        Parameters
        ----------
        db_path : str | Path | None, default None
            The DuckDB file to open, created if it does not exist. ``None`` takes the path from the
            config file, or the default above.
        read_only : bool, default False
            Open the database for queries only, and skip the schema initialisation. Several
            read-only connections can be open at once. DuckDB's file lock means none can open while
            another process has the database open for writing, for example during ``get``.
        kernel_roots : list[str | Path] | None, default None
            The kernel roots used to guess a mission from a path. ``None`` reads them from the
            configuration the first time they are needed. The resolved list is available as the
            ``db.kernel_roots`` property. Pass it to keep a script independent of the user's
            configuration.
        console : rich.console.Console | None, default None
            The one place this class writes to (#49, #6): tables, notes, progress bars and the
            download question all go to it. Without one (the default, for library use) nothing is
            printed, progress bars are off and every question is answered no, so a script passes
            ``yes=True`` where it wants a download. Warnings go through the ``spice_kernel_db``
            logger either way.

        Raises
        ------
        LegacyDatabaseFound
            ``db_path`` is ``None``, there is no config file, and a database sits at the pre-0.19
            default ``~/.spice_kernels.duckdb`` while the new default does not exist. Before 0.21.0
            that database was opened with a ``FutureWarning``. Since 0.21.0 it is not, and the
            message names the move command, rather than quietly starting an empty database beside
            it.
        ConfigError
            ``db_path`` is ``None`` and the config file does not parse.
        duckdb.IOException
            The database is locked by another process, or the path cannot be opened.
        """
        # Imported here, not at module level: ``duckdb`` costs ~30 ms to
        # import and ``skd --version`` never opens a database (#26).
        import duckdb

        self.console = console

        if db_path is None:
            from spice_kernel_db.config import default_db_path, load_config
            config = load_config()
            db_path = config.db_path if config else default_db_path()
        self.db_path = str(Path(db_path).expanduser())
        self.read_only = read_only
        # The kernel roots (#48): given, or read from the config on first use.
        self._kernel_roots: list[Path] | None = (
            [Path(r).expanduser().resolve() for r in kernel_roots]
            if kernel_roots is not None else None
        )
        if not read_only:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.db_path, read_only=read_only)
        if not read_only:
            self._init_schema()
        # A database from before #48 opened read-only cannot be migrated;
        # its registry rows read as official until the first write opens it.
        self._registry_has_official = (
            not read_only or "official" in self._columns("metakernel_registry")
        )

    def release(self):
        """Close the DB connection to release the lock.

        Use before long-running operations (downloads) so other
        processes can read the DB. Call :meth:`reacquire` to reopen.
        """
        self.con.close()
        self.con = None

    def reacquire(
        self, *, max_retries: int = 6, initial_delay: float = 0.1,
    ):
        """Reopen the DB connection after :meth:`release`.

        DuckDB enforces single-writer access via a file lock. If another
        process took the lock during our release window, retry with
        exponential backoff (~6.4s max with defaults) before giving up
        — short bursts of contention are common when two CLI processes
        overlap their network phases (C5).
        """
        import duckdb
        import time as _time
        delay = initial_delay
        last_exc: Exception | None = None
        for _ in range(max_retries):
            try:
                self.con = duckdb.connect(self.db_path, read_only=self.read_only)
                return
            except duckdb.IOException as e:  # the lock, or an unopenable path
                last_exc = e
                _time.sleep(delay)
                delay *= 2
        # Final attempt: let the original exception propagate
        self.con = duckdb.connect(self.db_path, read_only=self.read_only)
        # Unreachable on success; keep last_exc for static analyzers
        _ = last_exc

    def _columns(self, table: str) -> set[str]:
        return {r[1] for r in self.con.execute(f"PRAGMA table_info({table})").fetchall()}

    def _show(self, *renderables) -> None:
        """Print rich renderables or markup to the caller's console, if any."""
        if self.console is not None:
            self.console.print(*renderables)

    def _say(self, text: str = "") -> None:
        """Print plain text to the caller's console, if any: no markup, no wrapping."""
        if self.console is not None:
            self.console.print(text, markup=False, highlight=False, emoji=False, soft_wrap=True)

    def _confirm(self, question: str) -> bool:
        """Ask a yes/no question on the caller's console; no console, or a closed stdin, means no."""
        if self.console is None:
            return False
        try:
            answer = self.console.input(question, markup=False, emoji=False)
        except EOFError:
            return False
        return answer.strip().lower() in ("y", "yes")

    def _official(self, prefix: str = "") -> str:
        """The ``official`` column to select, or TRUE on a read-only pre-#48 database."""
        return f"{prefix}official" if self._registry_has_official else "TRUE AS official"

    @property
    def kernel_roots(self) -> list[Path]:
        """The configured kernel roots, as expanded, resolved paths.

        Read from the configuration the first time they are needed and kept for the life of the
        instance (#48). A list passed as ``kernel_roots`` to the constructor takes the place of the
        configuration, which keeps a script independent of the user's setup.

        Returns
        -------
        list[Path]
            The kernel roots, in configured order.
        """
        if self._kernel_roots is None:
            self._kernel_roots = kernel_roots()
        return self._kernel_roots

    def _guess_mission(self, path: str | Path) -> str:
        """``guess_mission`` with this database's roots, for every file path."""
        return guess_mission(str(path), self.kernel_roots)

    def root_for(self, mission: str | None, mk_path: str | Path | None = None) -> Path:
        """The kernel root a download belongs in (decided 2026-11-21).

        This is the one answer to "which kernel root does this download belong in". In order:

        1. the root that contains *mk_path*, since a tracked metakernel is re-fetched into its own
           store;
        2. the only configured root;
        3. the one root that already holds kernels of *mission*.

        When several roots remain it raises ``AmbiguousKernelRoot`` instead of picking the first.
        The CLI turns that into a picker at a terminal and exit code 2 in a script. There is no
        configured download target.

        ``get_metakernel`` and ``update_metakernel`` call it when ``download_dir`` is not given (a
        URL that is already tracked uses that row's root), and ``scan_directory(archive=True)``
        calls it for every file outside the roots.

        Parameters
        ----------
        mission : str | None
            The mission the download is for. Matched case-insensitively against the missions of the
            registered locations.
        mk_path : str | Path | None, default None
            A tracked metakernel's path. The root containing it wins.

        Returns
        -------
        Path
            The kernel root.

        Raises
        ------
        AmbiguousKernelRoot
            Several roots are configured and neither *mk_path* nor the mission's existing kernels
            single one out.
        """
        roots = self.kernel_roots
        if mk_path is not None and (root := root_containing(mk_path, roots)) is not None:
            return root
        if len(roots) == 1:
            return roots[0]
        holding = [
            r for r in roots if mission and self.con.execute(
                "SELECT 1 FROM locations WHERE LOWER(mission) = LOWER(?) "
                "AND starts_with(abs_path, ?) LIMIT 1", [mission, f"{r}/"],
            ).fetchone()
        ]
        if len(holding) == 1:
            return holding[0]
        raise AmbiguousKernelRoot(mission, holding or roots, stored=bool(holding))

    def _init_schema(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS kernels (
                sha256        VARCHAR PRIMARY KEY,
                filename      VARCHAR,
                kernel_type   VARCHAR,
                size_bytes    BIGINT,
                superseded_by VARCHAR
            )
        """)
        # Migration for DBs created before superseded_by existed
        cols = {r[1] for r in self.con.execute("PRAGMA table_info(kernels)").fetchall()}
        if "superseded_by" not in cols:
            self.con.execute("ALTER TABLE kernels ADD COLUMN superseded_by VARCHAR")
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
        # Dedup alias trail: `kernel_aliases` is a VIEW derived from
        # `locations`, not a stored table — every distinct on-disk basename a
        # content hash appears under is an alias for it. Deriving it keeps it
        # always-correct with zero sync; the trade-off (accepted) is that a
        # name is forgotten once its file is pruned. Drop any real table of
        # that name left by an earlier version before defining the view.
        if self.con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'kernel_aliases' AND table_type = 'BASE TABLE'"
        ).fetchone()[0] > 0:
            self.con.execute("DROP TABLE kernel_aliases")
        self.con.execute("""
            CREATE OR REPLACE VIEW kernel_aliases AS
            SELECT DISTINCT sha256, regexp_extract(abs_path, '[^/]+$') AS filename
            FROM locations
            WHERE regexp_extract(abs_path, '[^/]+$') <> ''
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
                acquired_at  TIMESTAMP DEFAULT current_timestamp,
                official     BOOLEAN DEFAULT TRUE
            )
        """)
        # Migration for DBs created before official existed (#48)
        if "official" not in self._columns("metakernel_registry"):
            self.con.execute(
                "ALTER TABLE metakernel_registry ADD COLUMN official BOOLEAN DEFAULT TRUE")
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
        # H4: warn at startup if the DB already contains case-duplicate
        # mission rows (e.g. legacy state from before canonicalisation).
        self._warn_about_mission_case_duplicates()

    def _warn_about_mission_case_duplicates(self) -> None:
        """Surface a one-time startup warning if `missions` and/or
        `locations` contain rows whose mission names differ only by
        case. The user can then choose to merge them explicitly — we
        deliberately don't auto-merge because data ops should be
        intentional (H4)."""
        import duckdb

        try:
            dup_groups = self.con.execute("""
                SELECT LOWER(name) AS k, LIST(name) AS variants
                FROM missions
                GROUP BY LOWER(name)
                HAVING COUNT(*) > 1
            """).fetchall()
        except duckdb.CatalogException:  # no missions table: nothing to check
            return
        if dup_groups:
            for _, variants in dup_groups:
                logger.warning(
                    "Mission case duplicates detected in missions table: %s. "
                    "These should be merged manually (e.g. remove one and "
                    "re-add the canonical form) — automatic resolution "
                    "would silently move data between rows.",
                    variants,
                )

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

        If the mission already exists, the existing entry is replaced.

        Parameters
        ----------
        name : str
            Mission name, for example ``"JUICE"``. It is canonicalised (H4), so ``juice`` and
            ``Juice`` collapse to the same row. The names ``generic`` and ``unknown`` stay
            lowercase, every other name becomes uppercase.
        server_url : str
            Root server URL, for example the NASA or ESA SPICE archive.
        mk_dir_url : str
            Full URL of the mission's ``mk/`` directory.
        dedup : bool, default True
            Whether deduplication is enabled for this mission. ``deduplicate_plan`` respects it.
        """
        name = canonicalize_mission(name)
        self.con.execute("""
            INSERT OR REPLACE INTO missions (name, server_url, mk_dir_url, dedup)
            VALUES (?, ?, ?, ?)
        """, [name, server_url, mk_dir_url, dedup])

    def list_missions(self) -> list[dict]:
        """List all configured missions.

        Returns
        -------
        list[dict]
            One dict per mission, ordered by name, with the keys ``name``, ``server_url``,
            ``server_label`` (``"NASA"``, ``"ESA"`` or ``"custom"``), ``mk_dir_url``, ``dedup`` and
            ``added_at``.
        """
        rows = self.con.execute("""
            SELECT name, server_url, mk_dir_url, dedup, added_at
            FROM missions ORDER BY name
        """).fetchall()
        results = []
        for name, server_url, mk_dir_url, dedup, added_at in rows:
            results.append({
                "name": name,
                "server_url": server_url,
                "server_label": server_label_for(server_url),
                "mk_dir_url": mk_dir_url,
                "dedup": dedup,
                "added_at": added_at,
            })
        return results

    def remove_mission(self, name: str) -> bool:
        """Remove a mission from the database.

        The lookup is the one ``get_mission`` uses: case-insensitive, and a unique prefix is enough.

        Parameters
        ----------
        name : str
            Mission name or unique prefix of one.

        Returns
        -------
        bool
            ``True`` if the mission was found and removed, ``False`` if not.
        """
        mission = self.get_mission(name)
        if mission is None:
            return False
        self.con.execute("DELETE FROM missions WHERE name = ?", [mission["name"]])
        return True

    def get_mission(self, name: str) -> dict | None:
        """Look up a mission by name.

        Tries an exact case-insensitive match first, then a case-insensitive prefix match.

        Parameters
        ----------
        name : str
            Mission name or prefix of one.

        Returns
        -------
        dict | None
            The mission as a dict with the same keys as ``list_missions`` returns. ``None`` if
            nothing matches or if the prefix matches several missions.
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
        return {
            "name": name,
            "server_url": server_url,
            "server_label": server_label_for(server_url),
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
        """Register a single kernel file and return its SHA-256.

        If the file's content is already in the database under a different filename, the new
        location is still recorded and points to the same hash. That is what happens when JUICE
        ships ``jup365_19900101_20500101.bsp`` and generic_kernels has ``jup365.bsp``: same content,
        different names, one hash. The match is logged at the ``INFO`` level.

        If the path was registered before under another hash, that location is stale. It is deleted,
        and the old kernel row is marked superseded by the new hash. This applies to every stale row
        at the path, not only one.

        Parameters
        ----------
        path : str | Path
            The kernel file.
        mission : str | None, default None
            The mission to file it under, canonicalised. ``None`` guesses it with ``guess_mission``
            and the database's kernel roots.
        source_url : str | None, default None
            Where the file was downloaded from, stored with the location.
        archive_dir : str | Path | None, default None
            If given, the file is moved to ``<archive_dir>/<mission>/<kernel_type>/<filename>`` and
            a symlink is left at the original location.
        expected_hash : str | None, default None
            If given, the computed SHA-256 is compared with it before anything is stored (Issue 7).

        Returns
        -------
        str
            The file's SHA-256 hex digest.

        Raises
        ------
        FileNotFoundError
            *path* is not a file.
        ValueError
            The computed hash differs from *expected_hash*.
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
        # H4: canonicalise mission name at the storage boundary
        m = canonicalize_mission(mission) if mission else self._guess_mission(p)

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

        # Detect "upstream content update at the same path": this same
        # abs_path was previously registered with a different sha256.
        # That's the genuine supersession case (C3 fix). Two different
        # paths sharing a filename is NOT supersession — both are real
        # kernels.
        # Every row at this path with another hash is stale: a path holds one
        # file. More than one such row has existed (#48: a 0-byte registration
        # from an interrupted download beside the real one), and taking only
        # the first left the other in place through every later scan.
        stale_hashes = [
            s for (s,) in self.con.execute(
                "SELECT sha256 FROM locations WHERE abs_path = ?", [str(p)]
            ).fetchall() if s != h
        ]
        superseded_old_hash: str | None = stale_hashes[0] if stale_hashes else None

        # Insert or update the kernel record.
        # If the hash already exists with a different filename, we keep
        # the first-registered filename as canonical but still record
        # this location.
        existing = self.con.execute(
            "SELECT filename FROM kernels WHERE sha256 = ?", [h]
        ).fetchone()

        with self.con.cursor() as cur:
            cur.execute("BEGIN")
            try:
                if existing is None:
                    # Issue 11: Check if same filename exists with different hash
                    # C8: case-insensitive comparison so APFS/HFS+ collisions
                    # are detected the same way Linux ext4 would see them.
                    existing_by_name = cur.execute(
                        "SELECT sha256 FROM kernels WHERE "
                        "LOWER(filename) = LOWER(?) "
                        "AND superseded_by IS NULL",
                        [fname],
                    ).fetchone()
                    if existing_by_name and existing_by_name[0] != h:
                        old_hash = existing_by_name[0]
                        if superseded_old_hash == old_hash:
                            logger.warning(
                                "Kernel %s content changed at %s: %s -> %s. "
                                "Marking old version superseded.",
                                fname, p, old_hash[:16], h[:16],
                            )
                        else:
                            logger.warning(
                                "Kernel %s exists with a different hash (%s) "
                                "from a different path; registering new hash %s "
                                "without superseding (both kept active).",
                                fname, old_hash[:16], h[:16],
                            )
                    cur.execute(
                        "INSERT INTO kernels (sha256, filename, kernel_type, "
                        "size_bytes, superseded_by) VALUES (?, ?, ?, ?, NULL)",
                        [h, fname, ktype, size],
                    )
                elif existing[0] != fname:
                    logger.info(
                        "Hash match: %s is identical to already-registered %s",
                        fname, existing[0],
                    )

                # If this same path previously held other hashes, those
                # locations rows are stale (the file on disk has one hash).
                # Delete them, and mark the old kernels rows superseded.
                for old in stale_hashes:
                    cur.execute(
                        "DELETE FROM locations WHERE sha256 = ? AND abs_path = ?",
                        [old, str(p)],
                    )
                    cur.execute(
                        "UPDATE kernels SET superseded_by = ? "
                        "WHERE sha256 = ? AND superseded_by IS NULL",
                        [h, old],
                    )

                # Issue 29: un-supersede content that is demonstrably present.
                #
                # `superseded_by` had exactly one writer -- the block above --
                # and it only ever SET the column, so the state was
                # unrecoverable. Whenever anything wrote non-kernel content to
                # a tracked path (a pre-0.17.0 interrupted download, a Dropbox
                # conflict copy, an editor), the real kernel's hash was marked
                # superseded by that junk. A later correct download restored
                # the file and re-inserted this location, but the flag stayed,
                # and find_by_filename() excludes superseded rows. So resolve()
                # reported "not found" for a byte-perfect file on disk,
                # _should_skip_download() got db_hash=None, and every `update`
                # re-downloaded it. Forever, with no shipped way to repair it.
                #
                # The condition is deliberately narrow: clear the flag only
                # when the superseding content exists in NO live location. A
                # genuine upstream update whose newer file is on disk keeps its
                # supersession, otherwise resolve() could start handing back a
                # real older version. Content that supersedes nothing findable
                # was never a version, only damage.
                #
                # This also makes `scan` the repair path for existing damage:
                # re-registering a file heals its row.
                superseded_now = cur.execute(
                    "SELECT superseded_by FROM kernels WHERE sha256 = ?", [h]
                ).fetchone()
                if superseded_now and superseded_now[0]:
                    newer_hash = superseded_now[0]
                    newer_paths = cur.execute(
                        "SELECT abs_path FROM locations WHERE sha256 = ?",
                        [newer_hash],
                    ).fetchall()
                    if not any(Path(r[0]).is_file() for r in newer_paths):
                        cur.execute(
                            "UPDATE kernels SET superseded_by = NULL "
                            "WHERE sha256 = ?", [h],
                        )
                        logger.warning(
                            "Kernel %s (%s) was marked superseded by %s, which "
                            "exists in no live location. Clearing the flag: "
                            "the file is present at %s.",
                            fname, h[:16], newer_hash[:16], p,
                        )

                cur.execute("""
                    INSERT OR REPLACE INTO locations VALUES
                        (?, ?, ?, ?, current_timestamp)
                """, [h, str(p), m, source_url])
                cur.execute("COMMIT")
            except Exception:  # broad on purpose: roll back, then re-raise
                cur.execute("ROLLBACK")
                raise
        return h

    def aliases(self, name_or_hash: str) -> dict | None:
        """Follow the deduplication trail for a kernel.

        Content-addressed deduplication stores one physical file per hash under a single canonical
        name. This method surfaces every other name that content is known by, so the deduplication
        is followable.

        Aliases are derived from ``locations``, not stored separately. The read-only view
        ``kernel_aliases (sha256, filename)`` exposes every distinct on-disk basename each content
        hash appears under, so a deduplicated kernel (identical bytes under several names) is
        followable with no extra bookkeeping. The trade-off is that a name is forgotten once its
        file is pruned.

        The companion ``alias_counts(shas)`` maps each hash to its number of known names.
        ``alias_counts_by_name(filenames)`` maps each filename to how many other names its content
        is known under. It is used to render the ``(+N aliases)`` annotation in kernel listings.

        Parameters
        ----------
        name_or_hash : str
            A full or partial SHA-256 (at least 6 hex characters), the canonical filename, or any
            alias filename the content has been registered under. Filenames match
            case-insensitively.

        Returns
        -------
        dict | None
            ``None`` if nothing matches. Otherwise a dict with the keys:

            - ``sha256`` (str)
            - ``canonical`` (str): the first-seen filename, ``kernels.filename``
            - ``kernel_type`` (str)
            - ``size_bytes`` (int)
            - ``superseded`` (bool)
            - ``aliases`` (list[str]): every filename this content is known under
            - ``locations`` (list[dict]): each with ``abs_path``, ``mission`` and ``source_url``
        """
        key = name_or_hash.strip()
        sha: str | None = None
        # The alias table may be absent on a database that hasn't been
        # migrated yet (read-only opens skip _init_schema). Degrade to a
        # canonical-name-only view rather than erroring.
        has_aliases = self.con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'kernel_aliases'"
        ).fetchone()[0] > 0

        # Hash (full or >=6-char prefix)?
        if len(key) >= 6 and all(c in "0123456789abcdef" for c in key.lower()):
            row = self.con.execute(
                "SELECT sha256 FROM kernels "
                "WHERE sha256 = ? OR sha256 LIKE ? || '%'",
                [key.lower(), key.lower()],
            ).fetchone()
            if row:
                sha = row[0]

        # Filename (alias first, then canonical), case-insensitive.
        if sha is None:
            row = None
            if has_aliases:
                row = self.con.execute(
                    "SELECT sha256 FROM kernel_aliases "
                    "WHERE LOWER(filename) = LOWER(?) LIMIT 1",
                    [key],
                ).fetchone()
            if row is None:
                row = self.con.execute(
                    "SELECT sha256 FROM kernels "
                    "WHERE LOWER(filename) = LOWER(?) LIMIT 1",
                    [key],
                ).fetchone()
            if row:
                sha = row[0]

        if sha is None:
            return None

        krow = self.con.execute(
            "SELECT filename, kernel_type, size_bytes, superseded_by "
            "FROM kernels WHERE sha256 = ?",
            [sha],
        ).fetchone()
        if has_aliases:
            names = [
                r[0] for r in self.con.execute(
                    "SELECT filename FROM kernel_aliases "
                    "WHERE sha256 = ? ORDER BY LOWER(filename)",
                    [sha],
                ).fetchall()
            ]
        else:
            names = [krow[0]] if krow and krow[0] else []
        locations = [
            {"abs_path": r[0], "mission": r[1], "source_url": r[2]}
            for r in self.con.execute(
                "SELECT abs_path, mission, source_url FROM locations "
                "WHERE sha256 = ? ORDER BY abs_path",
                [sha],
            ).fetchall()
        ]
        return {
            "sha256": sha,
            "canonical": krow[0] if krow else None,
            "kernel_type": krow[1] if krow else None,
            "size_bytes": krow[2] if krow else None,
            "superseded": bool(krow[3]) if krow and krow[3] else False,
            "aliases": names,
            "locations": locations,
        }

    def alias_counts(self, shas: list[str]) -> dict[str, int]:
        """Map each SHA-256 to how many filenames it is known under.

        Used to annotate listings with ``(+N aliases)`` without a query per row. A count above 1
        means the kernel has been deduplicated.

        Parameters
        ----------
        shas : list[str]
            Full SHA-256 hex digests.

        Returns
        -------
        dict[str, int]
            Hash to number of known filenames. A hash with no known name is omitted.
        """
        if not shas:
            return {}
        if self.con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'kernel_aliases'"
        ).fetchone()[0] == 0:
            return {}
        placeholders = ", ".join("?" for _ in shas)
        rows = self.con.execute(
            f"SELECT sha256, count(*) FROM kernel_aliases "
            f"WHERE sha256 IN ({placeholders}) GROUP BY sha256",
            shas,
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def alias_counts_by_name(self, filenames: list[str]) -> dict[str, int]:
        """For each filename, count how many other names its content is known under.

        Used to annotate kernel listings with ``(+N aliases)``.

        Parameters
        ----------
        filenames : list[str]
            Kernel basenames. Matching is case-insensitive.

        Returns
        -------
        dict[str, int]
            ``{filename: extra_count}``, with the keys exactly as passed in. Only names whose
            content has at least one alias (``extra_count >= 1``) are included. Names with no
            aliases, or that are not in the database, are omitted.
        """
        if not filenames:
            return {}
        if self.con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'kernel_aliases'"
        ).fetchone()[0] == 0:
            return {}
        lowered = {n.lower(): n for n in filenames}
        placeholders = ", ".join("?" for _ in lowered)
        name_to_sha = {
            r[0]: r[1] for r in self.con.execute(
                f"SELECT LOWER(filename), sha256 FROM kernel_aliases "
                f"WHERE LOWER(filename) IN ({placeholders})",
                list(lowered.keys()),
            ).fetchall()
        }
        counts = self.alias_counts(list(set(name_to_sha.values())))
        out: dict[str, int] = {}
        for low, original in lowered.items():
            sha = name_to_sha.get(low)
            if sha and counts.get(sha, 0) > 1:
                out[original] = counts[sha] - 1
        return out

    def scan_directory(
        self,
        root: str | Path,
        mission: str | None = None,
        extensions: set[str] | None = None,
        verbose: bool = False,
        archive: bool = False,
        archive_roots: dict[str, Path] | None = None,
    ) -> tuple[int, set[str]]:
        """Recursively scan a directory tree and register all kernel files.

        Files are registered with ``register_file``. Every ``.tm`` file found is also indexed and
        added to the metakernel registry.

        If *mission* is not given, it is detected for each file from its path with ``guess_mission``
        and the database's kernel roots: under a root, the first directory below the root;
        elsewhere, the path structure (``.../<MISSION>/kernels/...``). ``register_file`` guesses the
        same way when it is given no mission.

        With *archive*, each file is moved into a kernel root, laid out as
        ``<root>/<mission>/<type>/<filename>``, and a symlink is left at the original location. The
        root is the file's own when it already sits in a configured root, else ``root_for`` of its
        mission (2026-11-21). Every file's root is settled before anything moves, so an
        ``AmbiguousKernelRoot`` leaves the tree untouched. *archive_roots* (``{mission: root}``) is
        how the CLI answers it and calls again. ``scan_roots`` takes the same two arguments.

        Parameters
        ----------
        root : str | Path
            The directory to scan.
        mission : str | None, default None
            File everything under this mission instead of detecting it.
        extensions : set[str] | None, default None
            The file extensions to register, lowercase and with the dot. ``None`` means every known
            SPICE kernel extension: ``.bc``, ``.bsp``, ``.bpc``, ``.tpc``, ``.tf``, ``.ti``,
            ``.tls``, ``.tsc``, ``.bds`` and ``.tm``.
        verbose : bool, default False
            Write each registered file name on the instance's console, if it has one.
        archive : bool, default False
            Move the files into a kernel root, as described above.
        archive_roots : dict[str, Path] | None, default None
            The root to use for each mission when ``archive`` is set and ``root_for`` cannot decide.

        Returns
        -------
        tuple[int, set[str]]
            ``(count, missions_found)``: the number of files registered and the set of mission names
            detected during the scan.

        Raises
        ------
        AmbiguousKernelRoot
            With *archive*, a file outside the roots belongs to a mission that ``root_for`` cannot
            place and *archive_roots* does not answer.
        """
        if extensions is None:
            extensions = KERNEL_EXTENSIONS

        root = Path(root).expanduser().resolve()
        count = 0
        error_count = 0
        missions_found: set[str] = set()
        mk_files: list[tuple[Path, str]] = []  # (path, mission)
        # H4: canonicalise user-supplied mission once up-front
        if mission is not None:
            mission = canonicalize_mission(mission)
        files = [p for p in sorted(root.rglob("*"))
                 if p.is_file() and p.suffix.lower() in extensions]
        missions = {p: mission or self._guess_mission(p) for p in files}
        archive_for: dict[Path, Path] = {}
        if archive:
            chosen = dict(archive_roots or {})
            for p in files:
                own = root_containing(p, self.kernel_roots)
                if own is None:
                    m = missions[p]
                    own = chosen.get(m) or chosen.setdefault(m, self.root_for(m))
                archive_for[p] = own
        for p in files:
                try:
                    m = missions[p]
                    self.register_file(
                        p, mission=m, archive_dir=archive_for.get(p),
                    )
                    count += 1
                    missions_found.add(m)
                    if verbose:
                        self._say(f"  registered: {p.name}")
                    if p.suffix.lower() == ".tm":
                        mk_files.append((p, m))
                except (PermissionError, MemoryError, KeyboardInterrupt):
                    # Issue 8: Re-raise fatal errors
                    raise
                except OSError as e:
                    # Issue 8: Catch non-fatal OS errors (file not found, etc.)
                    error_count += 1
                    logger.warning("Could not register %s: %s", p, e)

        # Index metakernels into the registry
        for mk_path, m in mk_files:
            try:
                self._track_metakernel(mk_path, m)
            except (PermissionError, MemoryError, KeyboardInterrupt):
                raise
            except OSError as e:
                error_count += 1
                logger.warning("Could not index metakernel %s: %s", mk_path, e)

        self._say(f"Scanned {root}: {count} kernel files registered.")
        if mk_files:
            self._say(f"  Indexed {len(mk_files)} metakernels.")
        if error_count:
            self._say(f"  Errors: {error_count} files could not be registered.")
        return count, missions_found

    def scan_roots(
        self, verbose: bool = False, archive: bool = False,
        archive_roots: dict[str, Path] | None = None,
    ) -> tuple[int, set[str]]:
        """Run ``scan_directory`` over every configured kernel root.

        The roots in ``db.kernel_roots`` are scanned in order, as ``skd scan`` does with no
        directory (#48). A root that is not an existing directory is skipped with a warning to the
        ``spice_kernel_db`` logger: ``Kernel root <path> does not exist; skipped.``

        Parameters
        ----------
        verbose : bool, default False
            As for ``scan_directory``.
        archive : bool, default False
            As for ``scan_directory``.
        archive_roots : dict[str, Path] | None, default None
            As for ``scan_directory``.

        Returns
        -------
        tuple[int, set[str]]
            The total count of registered files and the union of the missions found.
        """
        count, missions = 0, set()
        for root in self.kernel_roots:
            if not root.is_dir():
                logger.warning("Kernel root %s does not exist; skipped.", root)
                continue
            n, found = self.scan_directory(
                root, verbose=verbose, archive=archive, archive_roots=archive_roots)
            count += n
            missions |= found
        return count, missions

    # ------------------------------------------------------------------
    # Lookup (mission-aware)
    # ------------------------------------------------------------------

    def _kernels_at(self, paths: list[Path]) -> dict[str, tuple[str, int]]:
        """``{abs_path: (sha256, size_bytes)}`` for every registered location
        among *paths*, each also looked up by its realpath (#48).

        A registered path pins the file, whatever other missions or roots
        call a file by the same name; the filename is only a fallback for
        an unregistered path.
        """
        keys: list[str] = []
        for p in paths:
            keys.append(str(p))
            if p.is_symlink():
                keys.append(os.path.realpath(p))
        if not keys:
            return {}
        rows = self.con.execute(
            "SELECT l.abs_path, k.sha256, k.size_bytes FROM locations l "
            "JOIN kernels k ON k.sha256 = l.sha256 "
            f"WHERE l.abs_path IN ({', '.join('?' * len(keys))})",
            keys,
        ).fetchall()
        found: dict[str, tuple[str, int]] = {}
        for abs_path, sha, size in rows:
            # Where a path still carries several rows (see register_file), the
            # row whose size is the file's is the one for the file that is there.
            if abs_path in found:
                try:
                    on_disk = os.stat(abs_path).st_size
                except OSError:
                    continue
                if found[abs_path][1] == on_disk or size != on_disk:
                    continue
            found[abs_path] = (sha, size)
        return found

    def _kernel_at(self, path: Path) -> tuple[str, int] | None:
        found = self._kernels_at([path])
        return found.get(str(path)) or (
            found.get(os.path.realpath(path)) if path.is_symlink() else None)

    def find_by_filename(self, filename: str) -> list[dict]:
        """Find all locations for a kernel by filename.

        Checks both the canonical filename in the ``kernels`` table and the actual filename in the
        location path. This handles the same content stored under different names (for example
        ``jup365.bsp`` and ``jup365_19900101_20500101.bsp``).

        Superseded kernels (``kernels.superseded_by IS NOT NULL``) are excluded. The comparison is
        case-insensitive (C8): kernel names are treated identically on case-insensitive (HFS+, APFS)
        and case-sensitive (ext4, NTFS) filesystems. Identity is the SHA-256 anyway, and name casing
        is metadata.

        The order is deterministic: by mission, then most recently scanned location, then path.
        Callers see stable results even when several non-superseded kernels share a filename.

        Parameters
        ----------
        filename : str
            The kernel's basename.

        Returns
        -------
        list[dict]
            One dict per location with the keys ``sha256``, ``abs_path``, ``mission``,
            ``kernel_type`` and ``size_bytes``.
        """
        rows = self.con.execute("""
            SELECT DISTINCT k.sha256, l.abs_path, l.mission,
                   k.kernel_type, k.size_bytes, l.scanned_at
            FROM kernels k
            JOIN locations l ON k.sha256 = l.sha256
            WHERE (LOWER(k.filename) = LOWER(?)
                   OR LOWER(l.abs_path) LIKE '%/' || LOWER(?))
              AND k.superseded_by IS NULL
            ORDER BY l.mission, l.scanned_at DESC, l.abs_path
        """, [filename, filename]).fetchall()
        return [
            {"sha256": r[0], "abs_path": r[1], "mission": r[2],
             "kernel_type": r[3], "size_bytes": r[4]}
            for r in rows
        ]

    def find_by_hash(self, sha256: str) -> list[dict]:
        """Find all locations for a kernel by its SHA-256 hash.

        Parameters
        ----------
        sha256 : str
            The full hex digest.

        Returns
        -------
        list[dict]
            One dict per location with the keys ``abs_path``, ``mission`` and ``filename``.
        """
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
        Superseded kernels are excluded; ordering is deterministic.
        C8: comparison is case-insensitive.
        """
        rows = self.con.execute("""
            SELECT DISTINCT k.sha256, l.abs_path, l.mission,
                   k.kernel_type, k.size_bytes, k.filename AS canonical_name,
                   l.scanned_at
            FROM locations l
            JOIN kernels k ON k.sha256 = l.sha256
            WHERE (LOWER(l.abs_path) LIKE '%/' || LOWER(?)
                   OR LOWER(l.abs_path) LIKE '%\\' || LOWER(?))
              AND k.superseded_by IS NULL
            ORDER BY l.mission, l.scanned_at DESC, l.abs_path
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

        The resolution is mission-aware (see the design notes on mission-aware resolution). In
        priority order:

        1. Exact filename match in *preferred_mission*. No warning.
        2. Exact filename match in any mission. Warns when a *preferred_mission* was given.
        3. Path-suffix match: a file on disk that is registered under a different name. Warns when
           the registered name differs from *filename*.
        4. Metakernel registry match, for ``.tm`` files acquired with ``get``. Warns only on a
           cross-mission hit.
        5. Not found: ``(None, [])``. No hint is produced, so the caller decides whether to suggest
           ``scan``.

        Steps 1 and 2 match on content, so a name can have several hits when two byte-identical
        files are registered under different names. A location actually named *filename* always wins
        over a byte-identical file registered under another name (issue #9). A differently-named
        copy is used only when no file of the requested name is registered, and says so in a
        warning.

        Parameters
        ----------
        filename : str
            The kernel's basename.
        preferred_mission : str | None, default None
            The mission to prefer in steps 1, 3 and 4.

        Returns
        -------
        tuple[str | None, list[str]]
            ``(path, warnings)``. *path* is the resolved absolute path, or ``None`` if nothing was
            found. *warnings* is a list of human-readable strings about the fallback decisions that
            were taken.
        """
        warnings: list[str] = []

        # --- 1. Exact filename, preferred mission ---
        hits = _prefer_exact_name(self.find_by_filename(filename), filename)
        if preferred_mission and hits:
            same_mission = [
                h for h in hits
                if h["mission"].lower() == preferred_mission.lower()
                and Path(h["abs_path"]).is_file()
            ]
            if same_mission:
                h = same_mission[0]
                _warn_if_renamed(warnings, filename, h)
                return h["abs_path"], warnings

        # --- 2. Exact filename, any mission ---
        for h in hits:
            if Path(h["abs_path"]).is_file():
                if preferred_mission:
                    warnings.append(
                        f"{filename}: not found in [{preferred_mission}] registry, "
                        f"using copy from [{h['mission']}]"
                    )
                _warn_if_renamed(warnings, filename, h)
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

        # --- 4. Metakernel registry (for .tm files acquired via 'get') ---
        if filename.lower().endswith(".tm"):
            mk_rows = self.con.execute(
                "SELECT mk_path, mission FROM metakernel_registry "
                "WHERE filename = ?",
                [filename],
            ).fetchall()
            if preferred_mission:
                for mk_path, mis in mk_rows:
                    if (mis.lower() == preferred_mission.lower()
                            and Path(mk_path).is_file()):
                        return mk_path, warnings
            for mk_path, mis in mk_rows:
                if Path(mk_path).is_file():
                    if preferred_mission:
                        warnings.append(
                            f"{filename}: not found in [{preferred_mission}] "
                            f"registry, using copy from [{mis}]"
                        )
                    return mk_path, warnings

        return None, warnings

    # ------------------------------------------------------------------
    # Duplicate reporting
    # ------------------------------------------------------------------

    def report_duplicates(self, min_copies: int = 2) -> list[dict]:
        """Find and report kernels that exist in multiple locations.

        With a console, a summary panel and the first 30 duplicated kernels are printed.

        Parameters
        ----------
        min_copies : int, default 2
            The minimum number of locations for a kernel to be reported.

        Returns
        -------
        list[dict]
            One dict per kernel, largest first, with the keys ``sha256``, ``filename``,
            ``size_bytes``, ``count``, ``paths``, ``missions`` and ``wasted_bytes``.
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
            self._show(Panel(
                f"[bold]{len(results)}[/bold] files with {min_copies}+ copies\n"
                f"Total wasted space: [bold]{total_waste / 1e6:.1f} MB[/bold]",
                title="Duplicate kernels",
            ))
            for d in results[:30]:
                self._show(
                    f"\n  [bold]{d['filename']}[/bold]  "
                    f"({d['size_bytes'] / 1e6:.1f} MB × {d['count']} copies)"
                )
                for p, m in zip(d["paths"], d["missions"]):
                    self._say(f"    [{m}] {p}")
        else:
            self._show("No duplicates found.")

        return results

    # ------------------------------------------------------------------
    # Metakernel operations
    # ------------------------------------------------------------------

    def _track_metakernel(
        self, mk_path: Path, mission: str, source_url: str | None = None,
        official: bool | None = None,
    ) -> None:
        """Index *mk_path*'s entries and write (or replace) its registry row.

        The one registration sequence behind ``scan``, ``get`` and
        ``import``. The registry row is the authority for everything it
        holds (#43, #48): ``get`` and ``import`` state mission and
        ``official`` and replace the row; a scan (``official=None``) only
        guessed, so it writes a row for a file seen for the first time and
        otherwise leaves the row alone. The re-scan that follows every
        ``get`` used to rewrite it, wiping the source URL and the
        acquisition time (2026-11-21).
        """
        self.index_metakernel(mk_path)
        if official is None and self.con.execute(
            "SELECT 1 FROM metakernel_registry WHERE mk_path = ?", [str(mk_path)],
        ).fetchone():
            return
        self.con.execute("""
            INSERT OR REPLACE INTO metakernel_registry
            (mk_path, mission, source_url, filename, acquired_at, official)
            VALUES (?, ?, ?, ?, current_timestamp, ?)
        """, [str(mk_path), mission, source_url, mk_path.name,
              True if official is None else official])

    def index_metakernel(self, mk_path: str | Path):
        """Parse a metakernel and store its entries in the ``metakernel_entries`` table.

        The existing entries of the metakernel are replaced. C6: the ``DELETE`` and the ``INSERT``
        statements run in one transaction. A crash between them used to leave an empty
        ``metakernel_entries`` set for this ``mk_path``, which made the kernel list silently
        incomplete.

        Parameters
        ----------
        mk_path : str | Path
            The ``.tm`` file.
        """
        mk_path = str(Path(mk_path).resolve())
        parsed = parse_metakernel(mk_path)
        with self.con.cursor() as cur:
            cur.execute("BEGIN")
            try:
                cur.execute(
                    "DELETE FROM metakernel_entries WHERE mk_path = ?",
                    [mk_path],
                )
                for i, raw in enumerate(parsed.kernels):
                    cur.execute(
                        "INSERT INTO metakernel_entries VALUES (?, ?, ?, ?)",
                        [mk_path, i, raw, Path(raw).name],
                    )
                cur.execute("COMMIT")
            except Exception:  # broad on purpose: roll back, then re-raise
                cur.execute("ROLLBACK")
                raise

    def registry_rows(self, mission: str | None = None) -> list[RegistryRow]:
        """Every tracked metakernel, ordered by mission and filename.

        This is the one registry query behind ``resolve_metakernel`` and the CLI pickers (#45).

        The ``official`` column was added in a later release. It is created when a database from
        before that release is first opened for writing. Opened read-only (``ls``, ``show``,
        ``doctor``, ``check``) before that, every row reads as official.

        Parameters
        ----------
        mission : str | None, default None
            Case-insensitive prefix filter on the mission.

        Returns
        -------
        list[RegistryRow]
            The rows. ``official`` is ``False`` for a metakernel imported with ``official=False``
            and ``True`` otherwise.
        """
        where, params = "", []
        if mission:
            where = "WHERE LOWER(mission) LIKE LOWER(?) || '%'"
            params = [mission]
        rows = self.con.execute(
            f"SELECT mk_path, source_url, filename, mission, acquired_at, {self._official()} "
            f"FROM metakernel_registry {where} ORDER BY mission, filename",
            params,
        ).fetchall()
        return [RegistryRow(*r) for r in rows]

    def _mission_of(self, mk_path: Path, mission: str | None) -> str:
        """*mission* as given, else the registry's mission for *mk_path*, else
        a guess from the path with the configured roots (#43, #48)."""
        if mission is not None:
            return mission
        row = self.con.execute(
            "SELECT mission FROM metakernel_registry WHERE mk_path = ?", [str(mk_path)],
        ).fetchone()
        return row[0] if row and row[0] else self._guess_mission(mk_path)

    def registry_row(
        self, ref: str | Path, mission: str | None = None,
    ) -> RegistryRow | None:
        """The registry row *ref* names, or None when no row matches.

        *ref* may be a registry filename, an ``mk_path``, a ``source_url``
        or a version-stripped base name (``juice_crema_5_2.tm`` for
        ``juice_crema_5_2_v473_20260819_001.tm``), case-insensitive, with
        or without ``.tm``. The file need not exist, so a stale row can
        still be shown, updated or removed. Raises ``AmbiguousMetakernel``
        when several rows match; never picks the first.
        """
        low = str(ref).lower()
        wanted = {low, low + ".tm"}
        rows = self.registry_rows(mission)
        matches = [
            r for r in rows
            if r.filename.lower() in wanted
            or r.mk_path.lower() == low
            or (r.source_url or "").lower() == low
        ]
        # A base name matches every tracked snapshot of it. The tag is
        # stripped from the registry side only, so asking for a specific
        # version that is not tracked never lands on another version.
        matches = matches or [
            r for r in rows if strip_version_tag(r.filename).lower() in wanted
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise AmbiguousMetakernel(str(ref), [r.filename for r in matches])
        return matches[0]

    def resolve_metakernel(
        self, ref: str | Path, mission: str | None = None,
    ) -> Path:
        """The on-disk path of the metakernel *ref* names (#45).

        *ref* can be a path that exists, a registry filename, a registry ``mk_path``, the
        ``source_url`` the metakernel was fetched from, or a version-stripped base name
        (``juice_crema_5_2.tm`` for ``juice_crema_5_2_v473_20260819_001.tm``). Names are
        case-insensitive, with or without ``.tm``. A path that exists is returned as it is. Anything
        else goes through ``registry_row``.

        Every command that reads a metakernel resolves through this method, so a name that works for
        ``check`` works for ``show`` and ``rewrite`` too (#41).

        Parameters
        ----------
        ref : str | Path
            The metakernel reference.
        mission : str | None, default None
            Narrow the registry to one mission (case-insensitive prefix match).

        Returns
        -------
        Path
            The metakernel file on disk.

        Raises
        ------
        MetakernelNotTracked
            Nothing matches, or the matched file is gone.
        AmbiguousMetakernel
            Several registry rows match. The method never picks the first of several.

        Examples
        --------
        >>> db.resolve_metakernel("juice_crema_5_2.tm")
        PosixPath('.../JUICE/mk/juice_crema_5_2_v473_20260819_001.tm')
        """
        candidate = Path(ref)
        if candidate.is_file():
            return candidate
        row = self.registry_row(ref, mission=mission)
        if row is None:
            raise MetakernelNotTracked(str(ref))
        path = Path(row.mk_path)
        if not path.is_file():
            raise MetakernelNotTracked(
                str(ref),
                detail=f"'{ref}' is registered at {path} but the file is gone; "
                       f"run `spice-kernel-db clean --registry` or "
                       f"`spice-kernel-db get {ref}`",
            )
        return path

    def verify_against_server(
        self,
        mk_path: str | Path,
        mission: str | None = None,
        *,
        check_hash: bool = False,
        check_spice: bool = False,
    ) -> dict:
        """Compare each local kernel of a metakernel against the server.

        `check` only asked whether a file of each name existed, so a local file
        holding the wrong content counted as present. On
        juice_crema_5_2_v473_20260819_001.tm that reported "Found locally:
        102/102" while 10 files differed from the server, one of them a
        245 MB DSK present locally as 11 MB (#15).

        The checks, cheapest first:

        * size -- one HEAD per kernel. A size mismatch already PROVES the files
          differ, so this alone catches every case in the issue.
        * hash -- only with *check_hash*, and only for kernels whose size
          matched, since a mismatch is already decided. Downloads each kernel,
          so it is opt-in.
        * spice -- with *check_spice*, furnsh each local kernel in an isolated
          kernel pool and report what SPICE refuses to read. Needs spiceypy.

        Returns ``{'entries': [...], 'ok': int, 'bad': int, 'unchecked': int}``.
        Status is one of ok, SIZE_MISMATCH, HASH_MISMATCH, UNREADABLE,
        NO_REMOTE or MISSING_LOCAL.
        """
        from spice_kernel_db.hashing import sha256_file
        from spice_kernel_db.remote import query_remote_sizes, resolve_kernel_urls

        resolved_mk = self.resolve_metakernel(mk_path, mission=mission)
        parsed = parse_metakernel(resolved_mk)
        mission = self._mission_of(resolved_mk, mission)

        row = self.con.execute(
            "SELECT source_url FROM metakernel_registry WHERE mk_path = ?",
            [str(resolved_mk)],
        ).fetchone()
        mk_url = row[0] if row else None
        if not mk_url:
            return {
                "entries": [{
                    "filename": Path(resolved_mk).name,
                    "status": "NO_REMOTE",
                    "detail": "no source_url recorded; cannot compare to a server",
                }],
                "ok": 0, "bad": 0, "unchecked": len(parsed.kernels),
            }

        # Resolve the remote URLs from the REMOTE metakernel, not the local
        # one. rewrite_metakernel() replaces PATH_VALUES with absolute local
        # paths, and resolve_kernel_urls() interprets PATH_VALUES relative to
        # the metakernel URL -- so a rewritten .tm yields URLs like
        # https://server/tmp/kernels/lsk/naif0012.tls. The remote copy is the
        # authority for where the files live on the server anyway.
        from spice_kernel_db.remote import fetch_metakernel
        from spice_kernel_db.parser import parse_metakernel_text

        try:
            remote_text = fetch_metakernel(mk_url)
        except OSError as exc:
            # Only network/IO failures are "could not fetch". Anything else is
            # a bug here and must not be disguised as an unreachable server.
            return {
                "entries": [{
                    "filename": Path(resolved_mk).name,
                    "status": "NO_REMOTE",
                    "detail": f"could not fetch {mk_url}: {exc}",
                }],
                "ok": 0, "bad": 0, "unchecked": len(parsed.kernels),
            }
        remote_parsed = parse_metakernel_text(remote_text, source=mk_url)

        urls_by_name = {
            Path(raw).name: url
            for raw, url in zip(
                remote_parsed.kernels,
                resolve_kernel_urls(mk_url, remote_parsed),
            )
        }
        locals_: dict[str, str | None] = {}
        for raw in parsed.kernels:
            fname = Path(raw).name
            local, _ = self.resolve_kernel(fname, preferred_mission=mission)
            locals_[fname] = local

        sizes = query_remote_sizes(sorted(urls_by_name.values()), console=self.console)
        entries: list[dict] = []
        ok = bad = unchecked = 0
        size_matched: list[tuple[str, str, str]] = []

        for raw in parsed.kernels:
            fname = Path(raw).name
            url = urls_by_name.get(fname)
            if url is None:
                entries.append({"filename": fname, "status": "NO_REMOTE",
                                "detail": "not listed in the remote metakernel"})
                unchecked += 1
                continue
            local = locals_.get(fname)
            if not local or not Path(local).is_file():
                entries.append({"filename": fname, "status": "MISSING_LOCAL",
                                "detail": "not present locally"})
                bad += 1
                continue
            remote_size = sizes.get(url)
            if remote_size is None:
                entries.append({"filename": fname, "status": "NO_REMOTE",
                                "detail": f"server did not report a size for {url}"})
                unchecked += 1
                continue
            local_size = Path(local).stat().st_size
            if local_size != remote_size:
                entries.append({
                    "filename": fname, "status": "SIZE_MISMATCH",
                    "detail": f"local {local_size:,} bytes, server "
                              f"{remote_size:,} bytes",
                })
                bad += 1
                continue
            size_matched.append((fname, local, url))

        if check_hash:
            import tempfile
            import urllib.request

            from spice_kernel_db.remote import CONNECT_TIMEOUT

            for fname, local, url in size_matched:
                try:
                    with tempfile.NamedTemporaryFile() as tmp:
                        with urllib.request.urlopen(
                            url, timeout=CONNECT_TIMEOUT
                        ) as resp:
                            while chunk := resp.read(1 << 20):
                                tmp.write(chunk)
                        tmp.flush()
                        remote_hash = sha256_file(tmp.name)
                except (OSError, http.client.HTTPException) as exc:
                    entries.append({"filename": fname, "status": "NO_REMOTE",
                                    "detail": f"could not fetch for hashing: {exc}"})
                    unchecked += 1
                    continue
                if sha256_file(local) != remote_hash:
                    entries.append({
                        "filename": fname, "status": "HASH_MISMATCH",
                        "detail": "same size as the server but different content",
                    })
                    bad += 1
                else:
                    entries.append({"filename": fname, "status": "ok",
                                    "detail": "size and hash match"})
                    ok += 1
        else:
            for fname, _local, _url in size_matched:
                entries.append({"filename": fname, "status": "ok",
                                "detail": "size matches"})
                ok += 1

        if check_spice:
            spice_entries = self._spice_readability(locals_)
            entries.extend(spice_entries)
            bad += sum(1 for e in spice_entries if e["status"] == "UNREADABLE")

        return {"entries": entries, "ok": ok, "bad": bad, "unchecked": unchecked}

    @staticmethod
    def _spice_readability(locals_: dict[str, str | None]) -> list[dict]:
        """furnsh each local kernel in isolation; report what SPICE refuses."""
        try:
            import spiceypy
        except ImportError:
            return [{
                "filename": "-", "status": "NO_REMOTE",
                "detail": "spiceypy not installed; skipped readability check",
            }]
        from spiceypy.utils.exceptions import SpiceyError

        out = []
        for fname, local in locals_.items():
            if not local or not Path(local).is_file():
                continue
            try:
                spiceypy.kclear()
                spiceypy.furnsh(str(local))
            except SpiceyError as exc:
                out.append({"filename": fname, "status": "UNREADABLE",
                            "detail": str(exc).splitlines()[0][:120]})
            finally:
                spiceypy.kclear()
        return out

    def check_metakernel(
        self,
        mk_path: str | Path,
        mission: str | None = None,
    ) -> dict:
        """Check which kernels from a metakernel are already available locally.

        Each ``KERNELS_TO_LOAD`` entry is looked up with ``resolve_kernel``, preferring the
        metakernel's mission. The method prints nothing and touches no network. The CLI renders its
        result, and ``check_report`` adds the other verdicts (#46).

        Parameters
        ----------
        mk_path : str | Path
            Any reference ``resolve_metakernel`` accepts, so a path to the ``.tm`` file or a tracked
            name.
        mission : str | None, default None
            Mission name for preferred-mission resolution. By default it is read from the registry
            row of a tracked metakernel, and only an untracked file is guessed from its path.

        Returns
        -------
        dict
            With the keys:

            - ``mk_path``: the resolved path, as a string
            - ``mission``: the mission used for resolution
            - ``found``: list of ``(raw_entry, local_path)`` tuples
            - ``missing``: list of ``raw_entry`` strings
            - ``warnings``: list of warning strings

        Raises
        ------
        MetakernelNotTracked
            *mk_path* matches nothing.
        AmbiguousMetakernel
            *mk_path* matches several registry rows.
        """
        mk_path = self.resolve_metakernel(mk_path, mission=mission)
        parsed = parse_metakernel(mk_path)
        mission = self._mission_of(mk_path, mission)

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

        return {"mk_path": str(mk_path), "mission": mission, "found": found,
                "missing": missing, "warnings": all_warnings}

    def check_report(
        self,
        mk_path: str | Path,
        *,
        mission: str | None = None,
        hash_: bool = False,
        remote: bool = False,
        spice: bool = False,
    ) -> dict:
        """Everything the ``check`` command reports, as data (#46).

        Runs ``check_metakernel`` (present), ``verify_metakernel`` (resolves, with SHA-256 checks
        when *hash_* is set) and, only when *remote* is set, ``verify_against_server`` and
        ``newer_remote_version``. Those two are the only network calls. ``check_metakernel`` itself
        prints nothing and touches no network.

        The three verdicts, each a bit of the exit code, are:

        - 1, ``present``: every kernel of the metakernel is on disk;
        - 2, ``resolves``: every entry resolves soundly through ``PATH_VALUES`` and matches the
          database;
        - 4, ``matches remote``: every local kernel matches the server (only with *remote*).

        Parameters
        ----------
        mk_path : str | Path
            Any reference ``resolve_metakernel`` accepts.
        mission : str | None, default None
            Mission name for preferred-mission resolution.
        hash_ : bool, default False
            Verify SHA-256 hashes: recompute them for the ``resolves`` verdict and, with *remote*,
            compare them against the server.
        remote : bool, default False
            Compare the local kernels with the server. This is the one network path.
        spice : bool, default False
            With *remote*, also furnsh each local kernel in an isolated kernel pool and report what
            SPICE refuses to read. Needs SpiceyPy.

        Returns
        -------
        dict
            With the keys:

            - ``mk_path``: the resolved path, as a string
            - ``present``: the raw result of ``check_metakernel``
            - ``resolves``: the raw result of ``verify_metakernel``
            - ``remote``: the raw result of ``verify_against_server``, or ``None`` without *remote*
            - ``newer``: the newer-version notice, or ``None``
            - ``verdicts``: the three verdicts as ``{bit, ok, line}`` rows. ``ok`` is ``True``,
              ``False``, or ``None`` when the verdict was not checked
            - ``exit_code``: the sum of the failed verdicts' bits, 0 to 7

        Raises
        ------
        MetakernelNotTracked
            *mk_path* matches nothing.
        AmbiguousMetakernel
            *mk_path* matches several registry rows.
        """
        present = self.check_metakernel(mk_path, mission=mission)
        resolves = self.verify_metakernel(mk_path, deep=hash_)
        server = newer = None
        if remote:
            server = self.verify_against_server(
                mk_path, mission=mission, check_hash=hash_, check_spice=spice,
            )
            newer = self.newer_remote_version(present["mk_path"], present["mission"])
        verdicts = _check_verdicts(present, resolves, server)
        return {
            "mk_path": present["mk_path"],
            "present": present,
            "resolves": resolves,
            "remote": server,
            "newer": newer,
            "verdicts": verdicts,
            "exit_code": sum(v["bit"] for v in verdicts if v["ok"] is False),
        }

    def newer_remote_version(
        self,
        mk_path: str | Path,
        mission: str | None,
    ) -> str | None:
        """A notice when the server lists a newer copy of the metakernel, else None.

        One directory listing, nothing downloaded. Network errors are
        swallowed, so the caller gets None offline. Only ``check --remote``
        asks (#46).
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
        except (OSError, http.client.HTTPException) as exc:  # best-effort hint
            logger.debug("Could not list %s for newer versions: %s", mk_dir_url, exc)
            return

        # H6: parse to datetime and compare numerically — the old
        # lexicographic string comparison on "YYYY-MM-DD HH:MM"
        # silently dropped seconds and broke at the minute boundary.
        from datetime import datetime as _dt
        acq_dt: _dt | None = None
        if isinstance(acquired_at, _dt):
            acq_dt = acquired_at
        else:
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    acq_dt = _dt.strptime(str(acquired_at)[:len(fmt) + 6], fmt)
                    break
                except ValueError:
                    continue
        if acq_dt is None:
            return

        for entry in entries:
            if entry.filename == mk_filename:
                try:
                    remote_dt = _dt.strptime(entry.date, "%Y-%m-%d %H:%M")
                except ValueError:
                    break
                if remote_dt > acq_dt:
                    return (
                        f"Remote update available: server modified {entry.date}"
                        f" (acquired {acq_dt:%Y-%m-%d %H:%M}). "
                        f"Run 'spice-kernel-db get {mk_filename}' to fetch it."
                    )
                break
        return None

    def rewrite_metakernel(
        self,
        mk_path: str | Path,
        output: str | Path,
        *,
        mission: str | None = None,
        link_root: str | Path | None = None,
    ) -> tuple[Path, list[str]]:
        """Rewrite a metakernel for local use with minimal edits.

        The method creates a symlink tree at *link_root* that mirrors the original
        ``$KERNELS/<type>/<filename>`` layout. Each symlink points to the actual file on disk, which
        may live in a different mission's tree. Only ``PATH_VALUES`` in the output ``.tm`` file is
        changed, to point to *link_root*. ``KERNELS_TO_LOAD``, ``PATH_SYMBOLS`` and all comments are
        preserved. See the design notes on minimal metakernel edits for the rationale.

        A kernel that cannot be found gets no symlink and a warning. A kernel path that would escape
        *link_root* is skipped with a warning.

        Parameters
        ----------
        mk_path : str | Path
            The original metakernel: a path, or any reference ``resolve_metakernel`` accepts.
        output : str | Path
            Where to write the new ``.tm`` file.
        mission : str | None, default None
            Mission name for preferred-mission resolution.
        link_root : str | Path | None, default None
            Where to create the symlink tree. ``None`` means a ``kernels/`` directory next to
            *output*, matching the common ``..`` convention.

        Returns
        -------
        tuple[Path, list[str]]
            ``(output_path, warnings)``.

        Raises
        ------
        MetakernelNotTracked
            *mk_path* matches nothing.
        AmbiguousMetakernel
            *mk_path* matches several registry rows.
        """
        mk_path = self.resolve_metakernel(mk_path, mission=mission)
        output = Path(output)
        parsed = parse_metakernel(mk_path)
        mission = self._mission_of(mk_path, mission)

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

            # Issue 9 / C1: validate relpath lexically against link_root
            link_path = _safe_join(link_root, rel)
            if link_path is None:
                logger.warning(
                    "Path traversal in kernel relpath '%s': escapes "
                    "link_root %s. Skipping.", rel, link_root,
                )
                all_warnings.append(
                    f"{fname}: path traversal — relpath escapes link_root, skipped"
                )
                continue

            link_path.parent.mkdir(parents=True, exist_ok=True)

            if local and Path(local).is_file():
                # C7: atomic symlink swap so concurrent readers never
                # observe a missing link.
                _atomic_symlink(Path(local).resolve(), link_path)
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

        self._say(f"Wrote metakernel to {out_path}")
        self._say(f"  Symlink tree at {link_root}")
        self._say(f"  Resolved: {len(parsed.kernels) - len(unresolved)}"
              f"/{len(parsed.kernels)}")
        if unresolved:
            self._say(f"  Unresolved ({len(unresolved)}):")
            for u in unresolved:
                self._say(f"    {u}")
        if all_warnings:
            self._say(f"  Warnings:")
            for w in all_warnings:
                self._say(f"    ⚠ {w}")

        return out_path, all_warnings

    # ------------------------------------------------------------------
    # Verify (cross-check a metakernel against the DB)
    # ------------------------------------------------------------------

    def verify_metakernel(
        self,
        mk_path: str | Path,
        *,
        deep: bool = False,
    ) -> dict:
        """Deeply cross-check a metakernel against the database.

        Each entry in ``KERNELS_TO_LOAD`` is substituted through ``PATH_VALUES`` and
        ``PATH_SYMBOLS`` to the absolute path SPICE will load. That path is validated against the
        ``kernels`` row of the registered location at that path, or of the file a symlink there
        leads to. Only a path the database has not registered falls back to the ``kernels`` rows
        whose filename matches (case-insensitive).

        Checks, with the entry ``status`` each failure gets:

        - ``PATH_VALUES`` entries that are not absolute, or not existing directories
          (``BAD_PATH_VALUE``);
        - traversal escape from the ``PATH_VALUES`` root (``TRAVERSAL``, C1);
        - dangling symlinks (``DANGLING``), missing files (``NOT_FOUND``) and paths that are not
          files (``NOT_FILE``);
        - size mismatch against ``kernels.size_bytes`` (``SIZE_MISMATCH``);
        - SHA-256 mismatch against ``kernels.sha256``, only with ``deep=True`` (``HASH_MISMATCH``);
        - ambiguous resolution (``AMBIGUOUS``, C3): the path is not registered and several
          non-superseded ``kernels`` rows share the filename. A registered path is never ambiguous,
          so a same-named kernel with other content in another mission or kernel root is not flagged
          (see the design notes on ``superseded_by`` semantics);
        - files the database does not know (``UNREGISTERED``).

        An entry that passes has status ``OK``. This is the recommended cross-check after every
        ``get``, ``update`` and ``rewrite``. Run it with ``deep=True`` periodically to catch content
        corruption that preserves the size.

        Parameters
        ----------
        mk_path : str | Path
            The metakernel: a path, or any reference ``resolve_metakernel`` accepts. Typically a
            rewritten one with absolute ``PATH_VALUES``.
        deep : bool, default False
            Recompute the SHA-256 of every entry. Without it only stat-based checks run, which is
            fast.

        Returns
        -------
        dict
            With the keys ``entries``, ``ok`` (the number of entries that passed), ``fail`` (the
            number that did not), ``fatal``, ``mk_path`` and ``deep``. Each entry is a dict with
            ``raw``, ``resolved``, ``status`` and ``detail`` (and ``sha256`` for a verified one in
            deep mode). ``fatal`` is ``True`` if any entry is in a P0 state: ``BAD_PATH_VALUE``,
            ``TRAVERSAL``, ``DANGLING``, ``NOT_FOUND``, ``NOT_FILE``, ``AMBIGUOUS``,
            ``SIZE_MISMATCH`` or ``HASH_MISMATCH``. ``UNREGISTERED`` counts as a failure but is not
            fatal.

        Raises
        ------
        MetakernelNotTracked
            *mk_path* matches nothing.
        AmbiguousMetakernel
            *mk_path* matches several registry rows.
        """
        from os.path import abspath, normpath
        mk_path = self.resolve_metakernel(mk_path)
        parsed = parse_metakernel(mk_path)

        entries: list[dict] = []
        ok = 0
        fail = 0
        fatal = False

        # Validate PATH_VALUES first — they must be absolute existing dirs
        path_value_roots: list[Path] = []
        for v in parsed.path_values:
            if not v.startswith("/"):
                entries.append({
                    "raw": f"PATH_VALUE={v!r}",
                    "resolved": v,
                    "status": "BAD_PATH_VALUE",
                    "detail": "PATH_VALUE is not absolute; SPICE resolves "
                              "against CWD which makes the .tm non-portable",
                })
                fail += 1
                fatal = True
                continue
            root = Path(v)
            if not root.is_dir():
                entries.append({
                    "raw": f"PATH_VALUE={v!r}",
                    "resolved": str(root),
                    "status": "BAD_PATH_VALUE",
                    "detail": f"PATH_VALUE directory does not exist: {root}",
                })
                fail += 1
                fatal = True
                continue
            path_value_roots.append(Path(abspath(root)))

        for raw in parsed.kernels:
            resolved_str = parsed.resolve(raw)
            resolved = Path(abspath(normpath(resolved_str)))
            entry: dict = {"raw": raw, "resolved": str(resolved)}

            # Traversal check: resolved must be inside at least one PATH_VALUE
            if path_value_roots and not any(
                str(resolved).startswith(str(r) + "/") or resolved == r
                for r in path_value_roots
            ):
                entry["status"] = "TRAVERSAL"
                entry["detail"] = (
                    f"resolves to {resolved} which is outside every "
                    f"PATH_VALUE root"
                )
                entries.append(entry)
                fail += 1
                fatal = True
                continue

            expected_name = resolved.name

            # Existence and dangling check
            if resolved.is_symlink() and not resolved.exists():
                entry["status"] = "DANGLING"
                entry["detail"] = (
                    f"symlink target missing: "
                    f"{Path(resolved).resolve(strict=False)}"
                )
                entries.append(entry)
                fail += 1
                fatal = True
                continue
            if not resolved.exists():
                entry["status"] = "NOT_FOUND"
                entry["detail"] = "no file at this path"
                entries.append(entry)
                fail += 1
                fatal = True
                continue
            if not resolved.is_file():
                entry["status"] = "NOT_FILE"
                entry["detail"] = "path exists but is not a regular file"
                entries.append(entry)
                fail += 1
                fatal = True
                continue

            # A registered path pins the file (#48: two stores, three
            # different naif0012.tls); only an unregistered path falls back
            # to the filename, where several active rows are an ambiguity.
            located = self._kernel_at(resolved)
            if located is not None:
                kernels_rows = [located]
            else:
                # (C8: case-insensitive for cross-FS portability)
                kernels_rows = self.con.execute(
                    "SELECT sha256, size_bytes FROM kernels "
                    "WHERE LOWER(filename) = LOWER(?) "
                    "AND superseded_by IS NULL",
                    [expected_name],
                ).fetchall()

            if len(kernels_rows) == 0:
                entry["status"] = "UNREGISTERED"
                entry["detail"] = (
                    f"no kernels row for filename {expected_name!r}; "
                    f"run scan to register"
                )
                entries.append(entry)
                fail += 1
                continue

            if len(kernels_rows) > 1:
                # C3: multiple active candidates → ambiguous resolution
                entry["status"] = "AMBIGUOUS"
                hashes = ", ".join(r[0][:8] for r in kernels_rows)
                entry["detail"] = (
                    f"{len(kernels_rows)} active kernels rows for "
                    f"{expected_name!r} (sha256s: {hashes}); resolution "
                    f"is not uniquely determined"
                )
                entries.append(entry)
                fail += 1
                fatal = True
                continue

            expected_sha, expected_size = kernels_rows[0]

            # Quick size check (cheap)
            actual_size = resolved.stat().st_size
            if actual_size != expected_size:
                entry["status"] = "SIZE_MISMATCH"
                entry["detail"] = (
                    f"size {actual_size} != expected {expected_size}"
                )
                entries.append(entry)
                fail += 1
                fatal = True
                continue

            if deep:
                actual_sha = sha256_file(resolved)
                if actual_sha != expected_sha:
                    entry["status"] = "HASH_MISMATCH"
                    entry["detail"] = (
                        f"sha256 {actual_sha[:16]} != expected "
                        f"{expected_sha[:16]}"
                    )
                    entries.append(entry)
                    fail += 1
                    fatal = True
                    continue
                entry["sha256"] = actual_sha
                entry["status"] = "OK"
                entry["detail"] = "hash verified"
            else:
                entry["status"] = "OK"
                entry["detail"] = "size matches (quick check)"

            entries.append(entry)
            ok += 1

        return {
            "entries": entries,
            "ok": ok,
            "fail": fail,
            "fatal": fatal,
            "mk_path": str(mk_path),
            "deep": deep,
        }

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

        C5: previously this only logged a warning and the caller
        ignored the return value, so detected races silently corrupted
        the DB. It now raises :class:`ConcurrentModificationError` so
        the operation aborts cleanly and the user can retry.

        Returns False if no change detected; never returns True.
        """
        current = self._snapshot_kernel_hashes(filenames, mission)
        changed_files = [
            f for f in filenames
            if pre_snapshot.get(f, set()) != current.get(f, set())
        ]
        if changed_files:
            sample = ", ".join(changed_files[:5])
            logger.warning(
                "DB state changed during download for %d kernel(s): %s. "
                "Another process may have modified the database.",
                len(changed_files), sample,
            )
            raise ConcurrentModificationError(
                f"DB modified concurrently during operation "
                f"({len(changed_files)} kernel(s) changed: {sample}). "
                f"Retry after the other operation completes."
            )
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
        mission_root = download_dir / mission
        for i in indices:
            # C1: validate relpath against download_dir/mission lexically
            expected = _safe_join(mission_root, relpaths[i])
            if expected is None:
                logger.warning(
                    "Refusing unsafe relpath %r for %s: escapes %s",
                    relpaths[i], filenames[i], mission_root,
                )
                continue
            # H1: a dangling symlink (target moved/deleted) was previously
            # skipped, leaving SPICE pointed at nothing. Unlink so the
            # block below can recreate it; healthy links still no-op out.
            if expected.is_symlink() and not expected.exists():
                try:
                    expected.unlink()
                except OSError as e:
                    logger.warning(
                        "Could not remove dangling symlink %s: %s",
                        expected, e,
                    )
                    continue
            elif expected.exists() or expected.is_symlink():
                continue
            local, _ = self.resolve_kernel(
                filenames[i], preferred_mission=mission,
            )
            if local and Path(local).is_file():
                # C2: verify hash by the RESOLVED LOCAL PATH, not by the
                # requested filename. The resolved path was returned from
                # `locations`, which stores sha256 as the join key — so a
                # direct lookup tells us exactly what hash should be there.
                # If this lookup misses, the resolver returned a path that
                # isn't tracked in locations — refuse rather than silently
                # symlinking.
                resolved_abs = str(Path(local).resolve())
                loc_row = self.con.execute(
                    "SELECT sha256 FROM locations WHERE abs_path = ?",
                    [resolved_abs],
                ).fetchone()
                if loc_row is None:
                    # Fallback: try the raw local path too (resolve may have
                    # followed a symlink the DB doesn't know about)
                    loc_row = self.con.execute(
                        "SELECT sha256 FROM locations WHERE abs_path = ?",
                        [str(Path(local))],
                    ).fetchone()
                if loc_row is None:
                    logger.warning(
                        "Cannot verify hash for %s: resolved path %s has no "
                        "locations row. Skipping symlink.",
                        filenames[i], resolved_abs,
                    )
                    continue
                actual_hash = sha256_file(local)
                if actual_hash != loc_row[0]:
                    logger.warning(
                        "Hash mismatch for %s at %s: expected %s, got %s. "
                        "Skipping symlink.",
                        filenames[i], resolved_abs,
                        loc_row[0][:16], actual_hash[:16],
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

    def _create_metakernel_alias(
        self,
        alias_path: Path,
        target: Path,
        mission: str,
        source_url: str,
        alias_filename: str,
    ) -> None:
        """Create a symlink ``alias_path → target.name`` and register it.

        If ``alias_path`` already exists as a regular file (not a symlink),
        leave it alone and emit a warning — refuse to clobber user data.
        If it exists as a symlink, replace it to point at the new target.
        """
        if alias_path.exists() and not alias_path.is_symlink():
            self._show(
                f"[yellow]Alias {alias_filename} already exists as a regular "
                f"file in {alias_path.parent} — not overwriting.[/yellow]"
            )
            return
        try:
            # C7: atomic symlink replace
            _atomic_symlink(target.name, alias_path)
        except OSError as e:
            self._show(
                f"[yellow]Could not create alias symlink "
                f"{alias_path}: {e}[/yellow]"
            )
            return
        self.con.execute("""
            INSERT OR REPLACE INTO metakernel_registry
            (mk_path, mission, source_url, filename, acquired_at)
            VALUES (?, ?, ?, ?, current_timestamp)
        """, [str(alias_path), mission, source_url, alias_filename])
        self._show(
            f"  Created alias [bold]{alias_filename}[/bold] → {target.name}"
        )

    def get_metakernel(
        self,
        url: str,
        download_dir: str | Path | None = None,
        mission: str | None = None,
        yes: bool = False,
        force: bool = False,
        alias_filename: str | None = None,
        verbose: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """Fetch a remote metakernel, show its status, and download the missing kernels.

        The steps are:

        1. Fetch and parse the remote ``.tm`` file.
        2. Resolve each kernel entry to a full URL, using ``PATH_VALUES`` relative to the metakernel
           URL.
        3. Check each kernel against the local database.
        4. Query the remote file sizes with parallel HTTP HEAD requests.
        5. Show a summary table with kernel names, sizes and status (``in db`` or ``missing``) on
           the instance's console.
        6. Ask for confirmation on that console, unless ``yes=True``. With no console there is
           nobody to ask. The call then stops after the status step like ``dry_run``, writes
           nothing, and the result carries ``aborted: True``.
        7. Download the missing kernels, preserving the subdirectory structure, and register them.
        8. Create symlinks for the kernels already in the database, so the metakernel works
           immediately. This applies when deduplication is enabled for the mission. Without it every
           kernel is downloaded fresh.

        The metakernel itself is saved as ``<download_dir>/<mission>/mk/<filename>`` with absolute
        ``PATH_VALUES``, so it works from any working directory. Downloaded files are placed under
        ``<download_dir>/<mission>/<relpath>``.

        The table is shown whenever there is a console. It precedes the download question, so it is
        what you decide on.

        Parameters
        ----------
        url : str
            URL of a remote ``.tm`` metakernel file.
        download_dir : str | Path | None, default None
            The kernel root to download into. ``None`` means ``root_for(mission)``: the configured
            root that already holds kernels of the mission, or the only configured root. A URL that
            is already tracked uses that row's root.
        mission : str | None, default None
            Override the mission name, which is otherwise guessed from the URL.
        yes : bool, default False
            Skip the confirmation prompt.
        force : bool, default False
            Treat all kernels as missing and download them again.
        alias_filename : str | None, default None
            A second name to expose this metakernel under, for example the version-stripped
            ``juice_crema_5_2.tm`` when the actual file is ``juice_crema_5_2_v470_20260415_001.tm``.
            A symlink is created at ``mk/<alias_filename>`` pointing to the downloaded file, and a
            second ``metakernel_registry`` row is added so ``which <alias_filename>`` works. When
            the server redirects to a differently named file, the requested name is used as the
            alias automatically.
        verbose : bool, default False
            List each kernel's resolved URL beneath the status table.
        dry_run : bool, default False
            Stop after the status table. Nothing is written to disk or to the database and no kernel
            is downloaded. This is what ``skd update`` runs without ``--execute`` (#47).

        Returns
        -------
        dict
            With the keys ``found`` and ``missing`` (the indices of the metakernel's kernel entries
            that are in the database and that are not), ``downloaded`` (the paths of the registered
            files) and ``warnings``. After a dry run the dict also has ``dry_run: True`` and
            ``downloaded`` is empty. It has ``aborted: True`` when nothing was downloaded because
            there was no console to ask on, or because the question was answered no. In the second
            case the registry is restored to its state before the call.

        Raises
        ------
        AmbiguousKernelRoot
            ``download_dir`` is not given and ``root_for`` cannot choose among several roots. Pass
            ``download_dir`` then.
        ConcurrentModificationError
            Another process modified the kernel records while the lock was released for the network
            I/O.
        ValueError
            A ``PATH_VALUES`` entry of the remote metakernel resolves outside the download
            directory.
        """
        if mission is None:
            mission = guess_mission(url)
        # H4: canonicalise before any filesystem path or DB row uses it
        mission = canonicalize_mission(mission)
        # Resolve to canonical mission name from DB (e.g. "juice" → "JUICE")
        m = self.get_mission(mission)
        if m:
            mission = m["name"]

        # With no console there is nobody to ask, so without ``yes`` the answer
        # is no before anything is written: the dry-run path, reported as aborted.
        unasked = not yes and not dry_run and self.console is None
        if unasked:
            dry_run = True

        # Where it lands, decided before any network I/O: the root holding
        # the file this URL fetched before, else the root that holds the
        # mission, else the only root. Several roots and no such fact raise
        # AmbiguousKernelRoot; there is no configured download target
        # (decided 2026-11-21).
        if download_dir is None:
            try:
                tracked = self.registry_row(url)
            except LookupError:  # the URL names several rows
                tracked = None
            download_dir = self.root_for(mission, tracked.mk_path if tracked else None)
        download_dir = Path(download_dir).expanduser().resolve()

        # 1. Fetch and parse
        self._show(f"Fetching [bold]{url}[/bold] ...")
        text, final_url = _fetch_metakernel(url)
        parsed = parse_metakernel_text(text, final_url)

        # 1b. Save .tm file to disk with absolute PATH_VALUES so it works
        # from any working directory (SPICE resolves paths relative to CWD,
        # not relative to the .tm file).
        # Use final_url so HTTP redirects (e.g. alias → versioned snapshot)
        # land under the canonical versioned filename on disk.
        mk_filename = final_url.rsplit("/", 1)[-1]
        requested_filename = url.rsplit("/", 1)[-1]
        # Auto-detect an HTTP-redirect alias the caller did not pass explicitly
        if alias_filename is None and requested_filename != mk_filename:
            alias_filename = requested_filename
        mk_dest = download_dir / mission / "mk" / mk_filename

        # Resolve each PATH_VALUE relative to the mk/ directory and make absolute
        # Validate that resolved paths stay within download_dir (Issue 1)
        mk_dir = mk_dest.parent
        _validate_path_values(parsed.path_values, mk_dir, download_dir)
        abs_path_values = [
            str((mk_dir / v).resolve()) for v in parsed.path_values
        ]
        # Snapshot prior state so we can roll back cleanly if the user
        # declines the kernel download. Without this, declining left a
        # registry row + indexed entries + on-disk .tm but no kernels,
        # which `ls` then listed as if the metakernel were ready.
        mk_existed_before = mk_dest.exists()
        prior_mk_content = mk_dest.read_bytes() if mk_existed_before else None
        if not dry_run:
            mk_dest.parent.mkdir(parents=True, exist_ok=True)
            write_metakernel(parsed, mk_dest, path_values=abs_path_values)
            self._track_metakernel(mk_dest, mission, final_url, official=True)

            # 1b-alias. Expose the metakernel under a second name via symlink.
            if alias_filename and alias_filename != mk_filename:
                alias_path = mk_dir / alias_filename
                self._create_metakernel_alias(
                    alias_path, mk_dest, mission, url, alias_filename,
                )

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

        # C1: validate every kernel relpath against download_dir/mission
        # BEFORE any download or symlink work. A hostile metakernel using
        # `$KERNELS/../../foo` would otherwise let us write arbitrary
        # paths on the user's filesystem. Abort the whole operation if
        # any one entry is unsafe — partial downloads on a hostile .tm
        # would still expose the user.
        mission_root = (download_dir / mission).resolve()
        _, relpath_errors = _validate_relpaths(relpaths, mission_root)
        if relpath_errors:
            raise ValueError(
                "Refusing to process metakernel with path-traversing "
                "KERNELS_TO_LOAD entries:\n  " + "\n  ".join(relpath_errors)
            )

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
                sizes = query_remote_sizes(kernel_urls, console=self.console)
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

        # (+N aliases): flag in-db kernels whose content is shared under
        # other filenames, pointing the user at `aliases <name>`.
        alias_extra = self.alias_counts_by_name(
            [filenames[i] for i in found_indices]
        ) if self.console is not None else {}
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
            label = fname
            n_alias = alias_extra.get(fname)
            if n_alias:
                label = (
                    f"{fname} [cyan](+{n_alias} "
                    f"alias{'es' if n_alias > 1 else ''})[/cyan]"
                )
            table.add_row(label, sz_str, f"[{style}]{status}[/{style}]")
        self._show(table)

        if verbose:
            self._show("[dim]Resolved URLs:[/dim]")
            for relpath, kurl in zip(relpaths, kernel_urls):
                self._show(f"  [dim]{relpath} -> {kurl}[/dim]")

        self._show(Panel(
            f"Total: [bold]{total}[/bold] | "
            f"In DB: [green]{n_found}[/green] | "
            f"Missing: [red]{n_missing}[/red] | "
            f"Download: [bold]{_format_size(download_bytes)}[/bold]",
            title=mk_name,
        ))

        if dry_run:
            return {
                "found": found_indices,
                "missing": missing_indices,
                "downloaded": [],
                "warnings": [],
                "dry_run": True,
                **({"aborted": True} if unasked else {}),
            }

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
            self._show(Panel("\n".join(lines), title="Result"))
            return {
                "found": found_indices,
                "missing": [],
                "downloaded": [],
                "warnings": [],
            }

        # 6. Prompt
        if not yes and not self._confirm(
            f"\nDownload {n_missing} missing kernels "
            f"({_format_size(download_bytes)})? [y/N]: "
        ):
            # Roll back the speculative registration. Without this,
            # the metakernel would appear in `ls` as if it were
            # ready, even though its required kernels were never
            # downloaded.
            if mk_existed_before and prior_mk_content is not None:
                # Update flow: restore the previous .tm content and
                # re-index its entries. The registry row was
                # REPLACEd in place; values are essentially the
                # same modulo the acquired_at timestamp, which we
                # accept as harmless drift.
                mk_dest.write_bytes(prior_mk_content)
                self.index_metakernel(mk_dest)
            else:
                # First-time get: full rollback.
                self.con.execute(
                    "DELETE FROM metakernel_entries WHERE mk_path = ?",
                    [str(mk_dest)],
                )
                self.con.execute(
                    "DELETE FROM metakernel_registry WHERE mk_path = ?",
                    [str(mk_dest)],
                )
                if alias_filename and alias_filename != mk_filename:
                    alias_path = mk_dir / alias_filename
                    if alias_path.is_symlink():
                        try:
                            alias_path.unlink()
                        except OSError:
                            pass
                    self.con.execute(
                        "DELETE FROM metakernel_registry "
                        "WHERE mk_path = ?",
                        [str(alias_path)],
                    )
                try:
                    mk_dest.unlink()
                except OSError:
                    pass
            self._show(Panel(
                f"Acquisition aborted — [bold]{n_missing}[/bold] "
                f"kernel(s) would have been downloaded.\n"
                f"Nothing was added to the database.\n\n"
                f"Re-run [bold]spice-kernel-db get[/bold] when "
                f"you're ready to download.",
                title="Aborted",
                border_style="yellow",
            ))
            return {
                "found": found_indices,
                "missing": missing_indices,
                "downloaded": [],
                "warnings": [],
                "aborted": True,
            }

        # 7. Build download list, skipping files with correct size AND hash (Issue 2)
        tasks = []
        # H5: key by the full destination path, not basename. Two
        # KERNELS_TO_LOAD entries with the same basename under different
        # relpaths previously collided here, attributing the wrong
        # source_url to whichever registered second.
        task_info: dict[Path, str] = {}
        already_on_disk: list[Path] = []
        for i in missing_indices:
            kurl = kernel_urls[i]
            relpath = relpaths[i]
            # C1: this was already validated above in `_validate_relpaths`,
            # but defense-in-depth — refuse to write outside mission_root.
            dest = _safe_join(mission_root, relpath)
            if dest is None:
                raise ValueError(
                    f"Refusing unsafe download dest for {filenames[i]!r}"
                )
            fname = filenames[i]
            remote_size = sizes.get(kurl)
            # Expected hash: the registered row at this path, else the
            # filename's active row (case-insensitive, C8)
            at_dest = self._kernel_at(dest) if dest.exists() else None
            if at_dest is not None:
                db_hash = at_dest[0]
            else:
                db_row = self.con.execute(
                    "SELECT sha256 FROM kernels "
                    "WHERE LOWER(filename) = LOWER(?) "
                    "AND superseded_by IS NULL",
                    [fname],
                ).fetchone()
                db_hash = db_row[0] if db_row else None
            if _should_skip_download(dest, remote_size, db_hash, force):
                already_on_disk.append(dest)
                task_info[dest] = kurl
                continue
            tasks.append((kurl, dest, fname))
            task_info[dest] = kurl

        if already_on_disk:
            self._show(f"  [dim]Skipped: {len(already_on_disk)} already downloaded[/dim]")
            download_bytes -= sum(f.stat().st_size for f in already_on_disk)

        # Issue 5: Snapshot before releasing for download
        pre_dl_snapshot = self._snapshot_kernel_hashes(filenames, mission)
        # Release the DB lock during download so other processes can
        # run read-only queries (resolve, list, check, etc.).
        self.release()
        try:
            dl_results, warnings = download_kernels_parallel(
                tasks, total_bytes=download_bytes, console=self.console,
            )
        finally:
            self.reacquire()
        # Issue 5: Check state after download reacquire
        self._check_state_changed(pre_dl_snapshot, filenames, mission)

        # C4: streamed hashes from the downloader. Pass to register_file
        # as expected_hash so a TOCTOU between "download finished" and
        # "register_file re-hashes from disk" cannot let tampered bytes
        # become canonical.
        download_hashes: dict[Path, str] = {p: s for p, s in dl_results}
        dl_paths: list[Path] = [p for p, _ in dl_results]

        # Register all files (downloaded + already on disk)
        # H5: look up source_url by full dest path, not basename
        downloaded: list[str] = []
        for dest in [*already_on_disk, *dl_paths]:
            kurl = task_info[dest]
            self.register_file(
                dest, mission=mission, source_url=kurl,
                expected_hash=download_hashes.get(dest),
            )
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
        self._show(Panel("\n".join(lines), title="Result"))

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
        verbose: bool = False,
        dry_run: bool = False,
        rescan: bool = True,
    ) -> dict:
        """Re-fetch a metakernel from its source URL and download new or missing kernels.

        The source URL comes from the ``metakernel_registry``. If no source URL is stored, the
        mission's ``mk_dir_url`` plus the filename is used. After downloading, the kernel
        directories referenced by the metakernel are rescanned, so newly downloaded files are
        indexed at once. The rescan does not happen after a dry run or with ``force``, since
        ``get_metakernel`` has already registered everything then.

        Parameters
        ----------
        mk_path_or_name : str | Path
            The ``mk_path`` or filename of a registered metakernel, or any other form
            ``resolve_metakernel`` takes.
        mission : str | None, default None
            Override the mission name, which is otherwise read from the registry row.
        download_dir : str | Path | None, default None
            Override the kernel root. By default it is the one that holds the metakernel
            (``root_for``).
        yes : bool, default False
            Skip the confirmation prompt.
        force : bool, default False
            Download everything again.
        verbose : bool, default False
            Name each directory as it is rescanned. Otherwise the count is reported in one line
            (#4).
        dry_run : bool, default False
            Report what would change and write nothing (#47). No rescan happens.
        rescan : bool, default True
            Re-index the directories the metakernel points at after the download. ``update_all``
            passes ``False`` and scans each distinct directory once at the end.

        Returns
        -------
        dict
            The dict ``get_metakernel`` returns. After a real run (neither ``dry_run`` nor
            ``force``) it also has ``scan_dirs``, the set of directories the metakernel points at.

        Raises
        ------
        MetakernelNotTracked
            The metakernel is not in the registry.
        AmbiguousMetakernel
            The reference matches several registry rows.
        LookupError
            The metakernel was added by ``scan`` or ``import`` without a source URL, and its mission
            has no ``mk/`` URL.
        MetakernelUnreachableError
            The remote returns HTTP 403, 404 or 410. This is typically NAIF rotating the file into
            ``former_versions/``. Use ``prune_metakernels`` to clean up the stale registry row.
        AmbiguousKernelRoot
            ``download_dir`` is not given and ``root_for`` cannot choose.
        """
        row = self.registry_row(mk_path_or_name, mission=mission)
        if row is None:
            raise MetakernelNotTracked(str(mk_path_or_name))

        mk_path, source_url, mk_filename, reg_mission = (
            row.mk_path, row.source_url, row.filename, row.mission,
        )
        mission = mission or reg_mission

        if not source_url:
            # Try to derive from mission's mk_dir_url
            m = self.get_mission(mission) if mission else None
            if m and m.get("mk_dir_url"):
                source_url = m["mk_dir_url"] + mk_filename
            else:
                raise LookupError(
                    f"{mk_filename} was added by scan or import and has no source "
                    f"URL, and its mission has no mk/ URL. Configure the mission "
                    f"(spice-kernel-db mission add) or pass the URL to get."
                )

        download_dir = download_dir or self.root_for(mission, mk_path)
        try:
            result = self.get_metakernel(
                source_url,
                download_dir=download_dir,
                mission=mission,
                yes=yes,
                force=force,
                dry_run=dry_run,
            )
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                # NAIF (and ESA) routinely rotate old versioned
                # metakernel snapshots into former_versions/, so the
                # registry's source_url goes permanently 404. Raise a
                # structured error the CLI can pretty-print.
                raise MetakernelUnreachableError(
                    source_url, e.code, mk_filename,
                ) from e
            raise

        # Re-scan the kernel directories referenced by the metakernel
        # so that new/renamed files on disk get indexed.  Skip if --force
        # was used, since get_metakernel already re-downloaded and
        # registered everything; a dry run changed nothing to re-scan.
        if force or dry_run:
            return result
        result["scan_dirs"] = self._metakernel_dirs(mk_path, mission, mk_filename, download_dir)
        if rescan:
            self._rescan(result["scan_dirs"], mission, verbose=verbose)
        return result

    def _metakernel_dirs(
        self, mk_path: str, mission: str, mk_filename: str,
        download_dir: str | Path,
    ) -> set[Path]:
        """The directories a tracked metakernel's ``PATH_VALUES`` point at.

        The .tm is either *mk_path* or the file ``get_metakernel`` just wrote
        under ``download_dir/mission/mk/``.
        """
        local_mk = Path(mk_path) if Path(mk_path).is_file() else None
        if local_mk is None:
            candidate = (
                Path(download_dir).expanduser().resolve() / mission / "mk" / mk_filename
            )
            if candidate.is_file():
                local_mk = candidate
        if local_mk is None:
            return set()
        try:
            parsed_local = parse_metakernel(str(local_mk))
        except OSError as e:
            logger.warning("Could not read %s for the rescan: %s", local_mk, e)
            return set()
        mk_dir = local_mk.resolve().parent
        return {
            (mk_dir / pv).resolve() for pv in parsed_local.path_values
            if (mk_dir / pv).resolve().is_dir()
        }

    def _rescan(self, scan_dirs: set[Path], mission: str | None, *, verbose: bool) -> None:
        """``scan_directory`` over *scan_dirs*, each once (#47: ``update``
        collects the directories of every tracked metakernel first)."""
        import duckdb

        for sd in sorted(scan_dirs):
            if verbose:
                self._show(f"[dim]Re-scanning {sd} for updated kernels...[/dim]")
            try:
                self.scan_directory(str(sd), mission=mission)
            except (OSError, duckdb.Error) as e:
                logger.warning("Post-update rescan of %s failed: %s", sd, e)
        if scan_dirs and not verbose:
            self._show(
                f"[dim]Re-scanned {len(scan_dirs)} "
                f"director{'y' if len(scan_dirs) == 1 else 'ies'} "
                f"for updated kernels.[/dim]"
            )

    def update_all(self, dry_run: bool = False, verbose: bool = False) -> list[dict]:
        """Re-fetch every tracked metakernel from its source, as ``skd update`` does (#47).

        Each metakernel lands in the root that holds it (``root_for``). There is no directory
        argument, since one directory for metakernels from several roots would undo that. With
        ``dry_run`` each metakernel's status table is shown and nothing is written. Otherwise the
        missing kernels are downloaded without a per-metakernel prompt (the CLI's ``--execute`` is
        the confirmation), and the directories the metakernels point at are re-indexed once at the
        end. The ``skd update`` command passes ``dry_run=True`` unless ``--execute`` is given.

        Alias rows, where a symlink stands next to its target, are skipped. A source that has gone
        from the server, a source that cannot be reached, and a row with no source URL are reported
        in the summary table and never raise.

        Parameters
        ----------
        dry_run : bool, default False
            Show what would be downloaded and write nothing.
        verbose : bool, default False
            Name each directory as it is rescanned.

        Returns
        -------
        list[dict]
            One dict per registry row walked, with the keys ``filename``, ``mission``, ``count`` and
            ``error``. ``count`` is the number of missing kernels in a dry run and the number of
            downloaded kernels otherwise. ``error`` is ``None`` or a one-line reason.
        """
        report: list[dict] = []
        scan_dirs: dict[Path, str] = {}
        for row in self.registry_rows():
            if Path(row.mk_path).is_symlink():
                continue  # the alias of a row walked under its own name
            entry = {"filename": row.filename, "mission": row.mission,
                     "count": 0, "error": None}
            self._show(f"\n[bold]{row.filename}[/bold] ({row.mission})")
            try:
                result = self.update_metakernel(
                    row.mk_path, yes=True, verbose=verbose, dry_run=dry_run, rescan=False,
                )
                entry["count"] = len(result["missing" if dry_run else "downloaded"])
                for sd in result.get("scan_dirs", ()):
                    scan_dirs.setdefault(sd, row.mission)
            except MetakernelUnreachableError as e:
                entry["error"] = f"source gone (HTTP {e.status}); spice-kernel-db clean --registry"
            except LookupError as e:
                entry["error"] = str(e)
            except (urllib.error.URLError, OSError) as e:
                entry["error"] = f"could not reach the source: {e}"
            report.append(entry)

        for sd, mission in sorted(scan_dirs.items()):
            self._rescan({sd}, mission, verbose=verbose)

        table = Table(title="update: all tracked metakernels" + (" (dry run)" if dry_run else ""))
        table.add_column("Mission")
        table.add_column("Metakernel")
        table.add_column("Missing" if dry_run else "Downloaded", justify="right")
        table.add_column("Note")
        for e in report:
            table.add_row(e["mission"] or "", e["filename"],
                          "" if e["error"] else str(e["count"]), e["error"] or "")
        self._show(table)
        return report

    def import_metakernel(
        self, mk_path: str | Path, mission: str | None = None, official: bool = True,
    ) -> dict:
        """Track a metakernel that is already on disk, without a download (#47).

        This is what ``skd import`` does. The method indexes the metakernel's entries, registers
        every kernel it resolves to that exists on disk, and adds the registry row with no source
        URL (as ``scan`` does for the ``.tm`` files it meets).

        ``official=False`` marks a metakernel assembled locally (#48), such as a subsetter's output
        or the one planetarypy builds in ``archived_kernels.py``, as ``skd import --unofficial``
        does. The flag is stored in the registry, ``ls`` appends ``unofficial`` to the name, and
        ``show`` prints ``Official: no``. Importing a path again replaces its row, so the flag
        follows the latest call.

        Parameters
        ----------
        mk_path : str | Path
            The ``.tm`` file.
        mission : str | None, default None
            The mission. Defaults to the one guessed from the path with ``guess_mission``, taking
            the configured kernel roots into account.
        official : bool, default True
            ``False`` marks a metakernel assembled locally rather than published by an archive.

        Returns
        -------
        dict
            With the keys ``mk_path``, ``filename``, ``mission``, ``n_entries``, ``n_registered``
            and ``missing``, the entries whose file is not on disk.

        Raises
        ------
        FileNotFoundError
            *mk_path* is not a file.
        """
        p = Path(mk_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"No such metakernel: {mk_path}")
        mission = canonicalize_mission(mission or self._guess_mission(p))
        m = self.get_mission(mission)
        if m:
            mission = m["name"]
        parsed = parse_metakernel(p)
        missing: list[str] = []
        n_registered = 0
        for raw in parsed.kernels:
            kernel = Path(parsed.resolve(raw))
            if kernel.is_file():
                self.register_file(kernel, mission=mission)
                n_registered += 1
            else:
                missing.append(raw)
        self._track_metakernel(p, mission, official=official)
        return {
            "mk_path": str(p),
            "filename": p.name,
            "mission": mission,
            "n_entries": len(parsed.kernels),
            "n_registered": n_registered,
            "missing": missing,
        }

    # ------------------------------------------------------------------
    # Metakernel listing / info
    # ------------------------------------------------------------------

    def list_metakernels(
        self, mission: str | None = None, show: bool = True,
    ) -> list[dict]:
        """List all tracked metakernels, optionally filtered by mission.

        Each entry includes the number of kernel entries. Alias rows (symlink metakernels created by
        ``get``) inherit their target's entry count and content fingerprint, so the listing reads
        consistently. Both rows show the same kernel count, and one of them is annotated
        ``↳ identical to <target>``.

        With a console, a summary table is shown. ``show=False`` suppresses it even with a console.
        That argument is legacy since #49, because the library is quiet without a console, and is
        kept because planetarypy's released code passes it.

        Parameters
        ----------
        mission : str | None, default None
            Case-insensitive prefix filter on the mission.
        show : bool, default True
            Show the summary table on the instance's console, if it has one.

        Returns
        -------
        list[dict]
            One dict per metakernel with the keys ``filename``, ``mission``, ``source_url``,
            ``acquired_at``, ``mk_path``, ``n_kernels``, ``official`` and ``identical_to``.
            ``identical_to`` is the name of another row with the same content fingerprint, else
            ``None``.
        """
        where, params = "", []
        if mission:
            where = "WHERE LOWER(r.mission) LIKE LOWER(?) || '%'"
            params = [mission]
        rows = self.con.execute(f"""
            SELECT r.filename, r.mission, r.source_url, r.acquired_at,
                   r.mk_path, COUNT(e.entry_index) AS n_kernels, {self._official("r.")}
            FROM metakernel_registry r
            LEFT JOIN metakernel_entries e ON e.mk_path = r.mk_path
            {where}
            GROUP BY r.mk_path, r.filename, r.mission, r.source_url, r.acquired_at{", r.official" if self._registry_has_official else ""}
            ORDER BY r.mission, r.filename
        """, params).fetchall()

        results = []
        for r in rows:
            filename, mis, source_url, acquired_at, mk_path, n_kernels, official = r
            results.append({
                "filename": filename,
                "mission": mis,
                "source_url": source_url,
                "acquired_at": acquired_at,
                "mk_path": mk_path,
                "n_kernels": n_kernels,
                "official": bool(official),
            })

        if not results:
            if show:
                self._say("No tracked metakernels.")
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

        # Alias rows (created by `_create_metakernel_alias` since v0.12.0)
        # live in `metakernel_registry` at the SYMLINK path but their
        # `metakernel_entries` are stored under the resolved target path.
        # That makes the COUNT(*) join above return 0 for aliases. Detect
        # them here and inherit the target's count + fingerprint so the
        # listing reads consistently — "↳ identical to <target>".
        known_paths = {r["mk_path"]: r for r in results}
        for r in results:
            p = Path(r["mk_path"])
            if not p.is_symlink():
                continue
            try:
                target = str(p.resolve())
            except OSError:
                continue
            tgt_r = known_paths.get(target)
            if tgt_r is None or target == r["mk_path"]:
                continue
            if r["n_kernels"] == 0:
                r["n_kernels"] = tgt_r["n_kernels"]
            fingerprints[r["mk_path"]] = fingerprints[target]

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

        if not show:
            return results

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
            if not r["official"]:
                name_cell += "  [yellow]unofficial[/yellow]"
            if r["identical_to"]:
                name_cell += f"\n[dim]↳ identical to {r['identical_to']}[/dim]"
            table.add_row(
                r["mission"], name_cell,
                str(r["n_kernels"]), source_short,
            )
        self._show(table)
        return results

    def metakernels_covering(
        self,
        body_id: int,
        et: float | None = None,
        mission: str | None = None,
    ) -> list[dict]:
        """Tracked metakernels whose SPK kernels cover a body.

        Meant for code that needs to pick a metakernel rather than show one. The method prints
        nothing and writes nothing, so it works on a ``read_only=True`` database. Coverage is read
        from the SPK files each metakernel loads that exist on disk. Identical aliases are skipped.
        With *et* only the metakernels covering that ephemeris time are returned. Without it, any
        coverage counts.

        Parameters
        ----------
        body_id : int
            NAIF ID of the body, for example ``-121`` for BepiColombo MPO.
        et : float | None, default None
            Ephemeris time that must be covered. ``None`` accepts any coverage at all.
        mission : str | None, default None
            Case-insensitive mission prefix filter.

        Returns
        -------
        list[dict]
            The matching ``list_metakernels`` rows, each with an added ``intervals`` key: the
            ``(et_start, et_end)`` windows for the body.

        Raises
        ------
        ImportError
            SpiceyPy is not installed.

        Examples
        --------
        >>> import spiceypy
        >>> db = KernelDB(read_only=True)
        >>> spiceypy.furnsh("naif0012.tls")
        >>> rows = db.metakernels_covering(-121, spiceypy.utc2et("2027-06-01"))  # BepiColombo MPO
        """
        from spice_kernel_db.coverage import spk_coverage

        matches = []
        for row in self.list_metakernels(mission=mission, show=False):
            mk_path = Path(row["mk_path"])
            if row.get("identical_to") or not mk_path.is_file():
                continue
            parsed = parse_metakernel(mk_path)
            intervals = []
            for raw in parsed.kernels:
                path = Path(parsed.resolve(raw))
                if path.suffix.lower() == ".bsp" and path.is_file():
                    intervals += [
                        (iv.et_start, iv.et_end) for iv in spk_coverage(path, body_id)
                    ]
            if not intervals:
                continue
            if et is not None and not any(start <= et <= end for start, end in intervals):
                continue
            matches.append({**row, "intervals": intervals})
        return matches

    def info_metakernel(
        self, name: str, kernel_type: str | None = None, mission: str | None = None,
    ) -> dict:
        """Show detailed information about a tracked metakernel, as ``skd show`` does (#47).

        Displays the registry row and, for each kernel the metakernel lists, its status (``in db``
        or ``missing``) with kernel type and size.

        Parameters
        ----------
        name : str
            Any form ``resolve_metakernel`` accepts, or the path of a ``.tm`` file that is not
            tracked. For an untracked file the row fields in the result are ``None`` and the file is
            read directly.
        kernel_type : str | None, default None
            Limit the printed kernel table to one type. The counts stay whole.
        mission : str | None, default None
            Narrow the registry lookup to one mission.

        Returns
        -------
        dict
            With the keys ``filename``, ``mission``, ``source_url``, ``acquired_at``, ``mk_path``,
            ``official``, ``kernels``, ``n_kernels``, ``n_in_db`` and ``n_missing``. ``kernels`` is
            a list of dicts with ``filename``, ``kernel_type``, ``size_bytes`` and ``status``.

        Raises
        ------
        MetakernelNotTracked
            *name* matches no registry row and is not a file on disk.
        AmbiguousMetakernel
            *name* matches several registry rows.
        """
        row = self.registry_row(name, mission=mission)
        if row is None:
            if not Path(name).is_file():
                raise MetakernelNotTracked(str(name))
            mk_path, source_url, filename, acquired_at = str(name), None, Path(name).name, None
            official = True
        else:
            mk_path, source_url, filename, mission, acquired_at, official = row

        entries = self.con.execute("""
            SELECT entry_index, raw_entry, filename
            FROM metakernel_entries
            WHERE mk_path = ?
            ORDER BY entry_index
        """, [mk_path]).fetchall()
        if not entries and Path(mk_path).is_file():
            # Untracked, or a row whose entries were never indexed: read the file.
            parsed = parse_metakernel(mk_path)
            entries = list(zip(range(len(parsed.kernels)), parsed.kernels,
                               parsed.kernel_filenames()))

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
        self._say(f"\nMetakernel: {filename}")
        if row is None:
            self._say(f"  Not tracked; spice-kernel-db import {mk_path} tracks it.")
        else:
            self._say(f"  Mission:      {mission}")
            if source_url:
                self._say(f"  Source:       {source_url}")
            self._say(f"  Acquired:     {acquired_at}")
            if not official:
                self._say("  Official:     no (assembled locally, not an archive metakernel)")
        self._say(f"  Local path:   {mk_path}")

        shown = kernel_info
        if kernel_type:
            shown = [k for k in kernel_info if k["kernel_type"] == kernel_type.lower()]
        if shown:
            table = Table()
            table.add_column("Kernel")
            table.add_column("Type")
            table.add_column("Size", justify="right")
            table.add_column("Status")
            for k in shown:
                sz_str = _format_size(k["size_bytes"]) if k["size_bytes"] is not None else "—"
                table.add_row(
                    k["filename"], k["kernel_type"], sz_str, k["status"],
                )
            self._show(table)

        self._say(f"\n  Total: {len(entries)} | In DB: {n_in_db} | Missing: {n_missing}")
        self._say()

        return {
            "filename": filename,
            "mission": mission,
            "source_url": source_url,
            "acquired_at": acquired_at,
            "official": official,
            "mk_path": mk_path,
            "kernels": kernel_info,
            "n_kernels": len(entries),
            "n_in_db": n_in_db,
            "n_missing": n_missing,
        }

    def remove_metakernel(self, name: str) -> bool:
        """Remove a metakernel from the registry and its entries.

        *name* is any form ``resolve_metakernel`` accepts. Does NOT delete
        the .tm file from disk.

        Returns True if found and removed, False if not found.
        """
        row = self.registry_row(name)
        if row is None:
            return False

        mk_path = row.mk_path
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
        sort_by: str = "name",
        filter: str | None = None,
        archived: bool = False,
    ) -> list[dict]:
        """Scan a remote NAIF ``mk/`` directory and show the available metakernels.

        Groups the entries by base name (stripping version tags like ``_v461_20251127_001``), counts
        the versioned snapshots, and checks which base metakernels have been acquired locally. A
        summary table is printed on the instance's console, if it has one.

        Parameters
        ----------
        mk_dir_url : str
            URL of a mission's ``mk/`` directory.
        mission : str | None, default None
            Override the mission name, which is otherwise detected.
        show_versioned : bool, default False
            List the versioned snapshots under each base metakernel.
        sort_by : str, default "name"
            Row ordering. ``"name"`` sorts alphabetically by base name. ``"date"`` sorts by latest
            remote modification date, ascending, so the most recently updated metakernels appear at
            the bottom.
        filter : str | None, default None
            Case-insensitive substring applied to entry filenames before grouping. Useful for
            narrowing large listings such as the ~1,400-entry JUICE ``former_versions/`` archive.
        archived : bool, default False
            Treat each entry as a distinct historical metakernel, with no base-name aggregation. Use
            it for browsing ``former_versions/`` directories, where every file is an independent
            frozen-in-time metakernel rather than a snapshot of a current file.

        Returns
        -------
        list[dict]
            One dict per row with the keys ``base_name``, ``n_versions``, ``latest_date``,
            ``is_local`` and ``filenames``.
        """
        entries = list_remote_metakernels(mk_dir_url)
        if filter:
            needle = filter.lower()
            entries = [e for e in entries if needle in e.filename.lower()]
        if mission is None:
            mission = guess_mission(mk_dir_url)

        # Group by base_name. In archive mode each entry is its own group
        # — archived snapshots are not byte-identical named copies of a
        # current file (the original premise behind grouping), so the
        # base-name aggregation does not apply.
        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)
        if archived:
            for entry in entries:
                groups[entry.filename].append(entry)
        else:
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
            # Separate current (no version tag) from versioned snapshots.
            # In archive mode, each entry is its own standalone metakernel
            # (not a snapshot of a current file), so treat the lone group
            # member as "current" regardless of its version tag.
            if archived:
                current = list(group)
                versioned: list = []
            else:
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

        if sort_by == "date":
            results.sort(key=lambda r: (r["latest_date"], r["base_name"]))
        elif sort_by != "name":
            raise ValueError(
                f"sort_by must be 'name' or 'date', got {sort_by!r}"
            )

        # Print table
        n_unique = len(results)
        n_files = len(entries)
        n_local = sum(1 for r in results if r["is_local"])

        self._say(f"\nRemote metakernels: {mission}")
        self._say(f"  {mk_dir_url}\n")

        if not results:
            self._say("  No .tm files found.")
            return results

        def _local_str(status: str) -> str:
            if status == "outdated":
                return "[yellow]outdated[/yellow]"
            return status

        if archived:
            # Archive view: one row per file, no grouping. Each archived
            # metakernel is a distinct historical artifact, not a copy of
            # something else, so the "Versions" column doesn't apply.
            table = Table()
            table.add_column("Metakernel")
            table.add_column("Date")
            table.add_column("Size", justify="right")
            table.add_column("Local")
            for r in results:
                ls = _local_str(r["local_status"])
                e = r["current"][0]
                table.add_row(e.filename, e.date, e.size, ls)
            self._show(table)
        elif show_versioned:
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
            self._show(table)
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
            self._show(table)

        if archived:
            self._say(
                f"\n  Total: {n_files} archived files"
                f" | {n_local} locally acquired"
            )
        else:
            self._say(
                f"\n  Total: {n_unique} unique | {n_files} files"
                f" | {n_local} locally acquired"
            )
        if has_outdated:
            self._say("  Run 'spice-kernel-db get <name>' to update outdated metakernels.")
        if not archived and not show_versioned and n_files > n_unique:
            self._say("  Use --show-versioned to also show the identical but versioned snapshot metakernels.")
        self._say()

        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def poisoned_superseded(self) -> list[dict]:
        """Find kernels flagged as superseded by content that exists in no live location.

        These kernels' own file is present on disk, so ``which`` should find it, but
        ``superseded_by`` points at a hash with no file anywhere. ``find_by_filename`` therefore
        skips the row, so the kernel resolves as "not found" and every ``update`` downloads it
        again. #29 made the state repairable and #31 makes it visible. ``scan`` over the kernel
        directory repairs every row listed here. A genuine upstream update whose newer file is on
        disk is not listed.

        Returns
        -------
        list[dict]
            One dict per kernel with the keys ``sha256``, ``filename``, ``superseded_by`` and
            ``abs_path`` (a present location).
        """
        rows = self.con.execute("""
            SELECT k.sha256, k.filename, k.superseded_by, l.abs_path
            FROM kernels k JOIN locations l USING (sha256)
            WHERE k.superseded_by IS NOT NULL
            ORDER BY k.filename, l.abs_path
        """).fetchall()
        out: list[dict] = []
        seen: set[str] = set()
        for sha, fname, newer, abs_path in rows:
            if sha in seen or not Path(abs_path).is_file():
                continue
            newer_paths = self.con.execute(
                "SELECT abs_path FROM locations WHERE sha256 = ?", [newer]
            ).fetchall()
            if any(Path(r[0]).is_file() for r in newer_paths):
                continue  # a real newer version: supersession is correct
            seen.add(sha)
            out.append({"sha256": sha, "filename": fname,
                        "superseded_by": newer, "abs_path": abs_path})
        return out

    def stale_locations(self) -> list[tuple[str, str]]:
        """List the registered locations whose file is gone from disk.

        This is the one walk behind ``clean`` and ``doctor``.

        Returns
        -------
        list[tuple[str, str]]
            ``(sha256, abs_path)`` for every such location.
        """
        rows = self.con.execute(
            "SELECT sha256, abs_path FROM locations ORDER BY abs_path"
        ).fetchall()
        return [(sha, path) for sha, path in rows if not Path(path).exists()]

    def doctor(self) -> dict:
        """Report the health of the database, with no metakernel argument (#46).

        The CLI's ``doctor`` prints the result and exits 1 only when ``poisoned`` is not empty. That
        is the one state that is corruption, since every ``update`` re-downloads those kernels. A
        ``scan_directory`` of the directory holding a conflicting path keeps the row that matches
        the file on disk and supersedes the others. ``register_file`` supersedes every stale row at
        the path, not only one.

        Returns
        -------
        dict
            With the keys:

            - ``n_kernels``, ``n_locations`` and ``n_metakernels``: the row counts;
            - ``poisoned``: the ``poisoned_superseded()`` rows;
            - ``missing_locations``: location paths whose file is gone, which ``clean`` removes;
            - ``conflicting_locations``: paths with more than one ``locations`` row, that is, one
              file registered under more than one hash (#48);
            - ``missing_metakernels``: registry filenames whose file is gone, which ``rm`` removes.
        """
        registry = self.registry_rows()
        return {
            "n_kernels": self.con.execute("SELECT COUNT(*) FROM kernels").fetchone()[0],
            "n_locations": self.con.execute("SELECT COUNT(*) FROM locations").fetchone()[0],
            "n_metakernels": len(registry),
            "poisoned": self.poisoned_superseded(),
            "missing_locations": [path for _sha, path in self.stale_locations()],
            "conflicting_locations": [
                r[0] for r in self.con.execute(
                    "SELECT abs_path FROM locations GROUP BY abs_path "
                    "HAVING COUNT(*) > 1 ORDER BY abs_path").fetchall()
            ],
            "missing_metakernels": [
                r.filename for r in registry if not Path(r.mk_path).is_file()
            ],
        }

    def stats(self) -> dict:
        """Print and return summary statistics.

        With a console, a summary panel and a table by kernel type are printed.

        Returns
        -------
        dict
            With the keys ``n_kernels``, ``n_locations``, ``total_bytes`` (the size of the unique
            content), ``n_duplicates`` (the number of hashes with more than one location),
            ``n_poisoned`` and ``missions``.
        """
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
        n_poisoned = len(self.poisoned_superseded())
        poisoned_line = (
            f"[yellow]Poisoned rows:    {n_poisoned}[/yellow]  "
            f"(repair: scan <directory>)"
            if n_poisoned else f"Poisoned rows:    {n_poisoned}"
        )

        self._show(Panel(
            f"Unique kernels:   [bold]{n_kernels}[/bold]\n"
            f"Total locations:  {n_locs}\n"
            f"Unique content:   {total_size / 1e9:.2f} GB\n"
            f"Duplicated files: {dups}\n"
            f"{poisoned_line}\n"
            f"Missions:         {', '.join(missions)}",
            title=f"SPICE Kernel Database: {self.db_path}",
        ))
        type_table = Table(title="By type")
        type_table.add_column("Type")
        type_table.add_column("Files", justify="right")
        type_table.add_column("Size", justify="right")
        for ktype, cnt, sz in by_type:
            type_table.add_row(ktype, str(cnt), f"{sz / 1e6:.1f} MB")
        self._show(type_table)

        return {
            "n_kernels": n_kernels,
            "n_locations": n_locs,
            "total_bytes": total_size,
            "n_duplicates": dups,
            "n_poisoned": n_poisoned,
            "missions": missions,
        }

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate_plan(self) -> list[dict]:
        """Generate a deduplication plan without executing it.

        For each set of identical files (same hash, several locations), one canonical copy is
        chosen, preferring the ``generic`` mission, and the rest are listed as removable. The plan
        respects the per-mission ``dedup`` setting of the ``missions`` table: files of a mission
        with deduplication disabled are never listed as removable. A mission that is not in the
        table defaults to ``dedup=True``.

        Returns
        -------
        list[dict]
            One dict per set of identical files that has something to remove, with the keys
            ``filename``, ``size_bytes``, ``keep`` (the canonical path) and ``remove`` (the list of
            removable paths).
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

        The plan is the one ``deduplicate_plan`` returns. With a console, each replacement and the
        space saved are printed.

        Parameters
        ----------
        dry_run : bool, default True
            With ``True`` (the default) only print what would happen. Regular files are replaced
            only with ``dry_run=False``.

        Returns
        -------
        list[dict]
            The plan, as ``deduplicate_plan`` returns it.
        """
        plan = self.deduplicate_plan()
        total_saved = 0

        for item in plan:
            for rm_path in item["remove"]:
                total_saved += item["size_bytes"]
                if dry_run:
                    self._say(f"  WOULD replace {rm_path}")
                    self._say(f"    -> symlink to {item['keep']}")
                else:
                    p = Path(rm_path)
                    if p.is_file() and not p.is_symlink():
                        p.unlink()
                        p.symlink_to(item["keep"])
                        self._say(f"  Replaced {rm_path} -> {item['keep']}")
                        # Update DB to note this is now a symlink
                        # (the hash and location are still valid)

        action = "Would save" if dry_run else "Saved"
        self._say(f"\n{action}: {total_saved / 1e6:.1f} MB")
        return plan

    def prune(self, dry_run: bool = True) -> list[str]:
        """Remove stale locations where the file no longer exists on disk.

        Also removes kernels that have zero remaining locations after pruning.

        Args:
            dry_run: If True (default), only report what would be removed.

        Returns:
            List of pruned abs_path strings.
        """
        stale = self.stale_locations()
        if not stale:
            self._say("No stale locations found.")
            return []

        action = "Would remove" if dry_run else "Removed"
        pruned_paths = []
        for sha256, abs_path in stale:
            self._say(f"  {action}: {abs_path}")
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
                self._say(f"  Removed orphaned kernel record: {filename}")

        self._say(f"\n{action}: {len(stale)} stale location(s)")
        return pruned_paths

    def prune_orphan_symlinks(self, dry_run: bool = True) -> list[str]:
        """Find symlinks under the download trees whose target no longer exists.

        The method walks every mission's download tree recursively, derived from
        ``metakernel_registry.mk_path`` (the parent of the ``mk/`` directory), and finds dangling
        symlinks. These accumulate when an underlying kernel store moves, or after ``prune`` or
        ``clean --locations`` removes a ``locations`` row. The symlinks pointing at that file are
        not themselves tracked in ``locations``, so they survive as junk.

        Healthy symlinks and regular files are left alone.

        Parameters
        ----------
        dry_run : bool, default True
            With ``True`` (the default) only report. With ``False``, unlink every dangling symlink.

        Returns
        -------
        list[str]
            The absolute paths that were, or with ``dry_run`` would be, removed.
        """
        # Derive download tree roots from the registry. mk_path is
        # typically ``<download_dir>/<mission>/mk/<filename>.tm``, so
        # the mission's tree is two levels up. Skip any mk_path that
        # doesn't fit the convention.
        roots: set[Path] = set()
        for (mk_path,) in self.con.execute(
            "SELECT mk_path FROM metakernel_registry"
        ).fetchall():
            p = Path(mk_path)
            if p.parent.name == "mk":
                roots.add(p.parent.parent)

        if not roots:
            self._say("No download trees to walk (no registered metakernels).")
            return []

        dangling: list[Path] = []
        for root in roots:
            if not root.is_dir():
                continue
            for sub in root.rglob("*"):
                if sub.is_symlink() and not sub.exists():
                    dangling.append(sub)

        if not dangling:
            self._say("No orphan symlinks found.")
            return []

        action = "Would remove" if dry_run else "Removed"
        for d in dangling:
            try:
                target = os.readlink(d)
            except OSError:
                target = "?"
            self._say(f"  {action}: {d}  →  {target}")
            if not dry_run:
                try:
                    d.unlink()
                except OSError as e:
                    logger.warning("Could not unlink %s: %s", d, e)

        self._say(f"\n{action}: {len(dangling)} orphan symlink(s)")
        return [str(p) for p in dangling]

    def prune_metakernels(
        self,
        dry_run: bool = True,
        delete_files: bool = False,
        timeout: float = 10.0,
    ) -> list[dict]:
        """Find registry rows whose remote URL is permanently gone and optionally remove them.

        A HEAD request is sent to each row's URL, and a permanent HTTP error (403, 404 or 410) marks
        the row as dead. This typically happens when NAIF rotates old versioned snapshots into
        ``former_versions/``.

        The probe URL is derived per row: the explicit ``source_url`` if set, otherwise
        ``mission.mk_dir_url + filename`` (the same fallback as ``update_metakernel``). Rows with
        neither, typically locally created metakernels with no upstream, are noted in the output and
        skipped. Use ``skd rm <name>`` to drop them manually.

        Transient failures (timeouts, DNS errors, 5xx) are deliberately never classified as dead.
        Leaving a registry row stale is always safer than deleting it on a network blip.

        Parameters
        ----------
        dry_run : bool, default True
            With ``True`` (the default) only report what would be removed. With ``False`` the
            matching rows are removed from ``metakernel_registry`` and ``metakernel_entries``.
        delete_files : bool, default False
            With ``dry_run=False``, also unlink the on-disk ``.tm`` file. Symlink trees under the
            mission's download directory are shared across metakernels and are never cleaned
            automatically.
        timeout : float, default 10.0
            Timeout of each HEAD request, in seconds.

        Returns
        -------
        list[dict]
            One dict per dead metakernel with the keys ``mk_path``, ``filename``, ``mission``,
            ``source_url`` and ``status_code``.
        """
        import urllib.request

        # Walk every registry row. We deliberately do NOT filter on
        # source_url here: rows added via `scan` have NULL source_url
        # but can still be probed by deriving a URL from the mission's
        # mk_dir_url + filename, mirroring `update_metakernel`'s
        # fallback. Without this, the most common case (locally-scanned
        # metakernels) is silently skipped.
        rows = self.con.execute("""
            SELECT mk_path, filename, mission, source_url
            FROM metakernel_registry
            ORDER BY mission, filename
        """).fetchall()

        if not rows:
            self._say("No metakernels in the registry.")
            return []

        # Cache mk_dir_url per mission to avoid N×get_mission queries.
        mk_dir_url_cache: dict[str, str | None] = {}
        def _mk_dir_url(mission_name: str | None) -> str | None:
            if not mission_name:
                return None
            if mission_name not in mk_dir_url_cache:
                m = self.get_mission(mission_name)
                mk_dir_url_cache[mission_name] = (
                    m.get("mk_dir_url") if m else None
                )
            return mk_dir_url_cache[mission_name]

        dead: list[dict] = []
        no_url: list[dict] = []
        with Progress(
            transient=True,
        ) as progress:
            tid = progress.add_task(
                "Checking remote metakernels", total=len(rows),
            )
            for mk_path, filename, mission, source_url in rows:
                progress.advance(tid)

                # Derive an effective URL: explicit source_url first,
                # then mission.mk_dir_url + filename (same precedence
                # as update_metakernel).
                effective_url = source_url
                if not effective_url:
                    base = _mk_dir_url(mission)
                    if base and filename:
                        effective_url = (
                            base if base.endswith("/") else base + "/"
                        ) + filename
                if not effective_url:
                    no_url.append({
                        "mk_path": mk_path,
                        "filename": filename,
                        "mission": mission,
                    })
                    continue

                try:
                    req = urllib.request.Request(effective_url, method="HEAD")
                    with urllib.request.urlopen(req, timeout=timeout):
                        continue  # alive
                except urllib.error.HTTPError as e:
                    if e.code in (403, 404, 410):
                        dead.append({
                            "mk_path": mk_path,
                            "filename": filename,
                            "mission": mission,
                            "source_url": effective_url,
                            "status_code": e.code,
                        })
                    else:
                        logger.warning(
                            "HEAD %s returned HTTP %d — not pruning",
                            effective_url, e.code,
                        )
                except (urllib.error.URLError, OSError) as e:
                    logger.warning(
                        "HEAD %s failed (%s) — not pruning, may be transient",
                        effective_url, e,
                    )

        if no_url:
            self._show(
                f"[dim]{len(no_url)} row(s) had no probeable URL "
                f"(no source_url and no mission mk_dir_url). "
                f"Use `spice-kernel-db rm <name>` to drop them manually.[/dim]"
            )

        if not dead:
            self._say("No dead metakernels found.")
            return []

        # Render results
        table = Table(title=f"Unreachable metakernels ({len(dead)})")
        table.add_column("Mission")
        table.add_column("Filename")
        table.add_column("HTTP")
        table.add_column("URL", overflow="fold")
        for d in dead:
            table.add_row(
                d["mission"] or "",
                d["filename"],
                str(d["status_code"]),
                d["source_url"],
            )
        self._show(table)

        if dry_run:
            self._show(
                f"\n[dim]Dry run — pass --execute to remove the "
                f"{len(dead)} registry row(s) above.[/dim]"
            )
            if not delete_files:
                self._show(
                    "[dim]Add --delete-files to also unlink the .tm files "
                    "on disk.[/dim]"
                )
            return dead

        # Actually remove
        for d in dead:
            self.con.execute(
                "DELETE FROM metakernel_entries WHERE mk_path = ?",
                [d["mk_path"]],
            )
            self.con.execute(
                "DELETE FROM metakernel_registry WHERE mk_path = ?",
                [d["mk_path"]],
            )
            if delete_files:
                p = Path(d["mk_path"])
                if p.is_symlink() or p.is_file():
                    try:
                        p.unlink()
                        self._say(f"  Deleted file: {p}")
                    except OSError as e:
                        self._show(
                            f"[yellow]Could not delete {p}: {e}[/yellow]"
                        )

        self._show(
            f"\n[bold]Removed {len(dead)} registry row(s)"
            f"{' + files' if delete_files else ''}.[/bold]"
        )
        return dead

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

        mk = self.resolve_metakernel(mk_path, mission=mission)
        parsed = parse_metakernel(mk)
        mission = self._mission_of(mk, mission)

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
