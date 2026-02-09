"""Remote SPICE kernel operations — fetch, resolve, and download."""

from __future__ import annotations

import re
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from spice_kernel_db.parser import ParsedMetakernel

# Known SPICE archive servers
SPICE_SERVERS: dict[str, str] = {
    "NASA": "https://naif.jpl.nasa.gov/pub/naif/",
    "ESA": "https://spiftp.esac.esa.int/data/SPICE/",
}

# Regex for Apache mod_autoindex directory entries (trailing /)
# Handles both plain-text (NASA) and table-based (ESA) formats.
_DIR_ENTRY_RE = re.compile(
    r'<a href="([^"]+/)">[^<]+</a>'
    r".+?"
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})"
)

# Regex for NAIF versioned metakernel snapshots: _v461_20251127_001.tm or _V324_20260206_002.TM
_VERSION_TAG_RE = re.compile(r"(_[vV]\d+_\d{8}_\d{3})\.[tT][mM]$")

# Regex for Apache mod_autoindex .tm file entries.
# Handles both plain-text (NASA) and table-based (ESA) formats.
# Case-insensitive on the .tm extension (ESA uses .TM for some missions).
_DIR_LISTING_RE = re.compile(
    r'<a href="([^"]+\.[tT][mM])">[^<]+</a>'
    r".+?"
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})"
    r".+?"
    r"(\d[\d.]*[KMGT]?)"
)


@dataclass
class RemoteMetakernel:
    """A metakernel entry parsed from a remote Apache directory listing."""

    filename: str
    url: str
    date: str  # "2025-11-27 09:30"
    size: str  # "12K"
    base_name: str  # filename with version tag stripped
    version_tag: str | None  # "v461_20251127_001" or None


def list_remote_metakernels(mk_dir_url: str) -> list[RemoteMetakernel]:
    """Parse an Apache directory listing to extract .tm metakernels.

    Fetches the HTML from *mk_dir_url*, extracts ``.tm`` file entries,
    strips version tags to compute ``base_name``, and returns the list
    sorted by (base_name, filename).

    Entries under ``former_versions/`` are excluded.
    """
    if not mk_dir_url.endswith("/"):
        mk_dir_url += "/"

    with urllib.request.urlopen(mk_dir_url) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    results: list[RemoteMetakernel] = []
    for m in _DIR_LISTING_RE.finditer(html):
        filename, date, size = m.group(1), m.group(2), m.group(3)

        # Skip entries from subdirectories (e.g. former_versions/)
        if "/" in filename:
            continue

        # Extract version tag if present
        tag_match = _VERSION_TAG_RE.search(filename)
        if tag_match:
            version_tag = tag_match.group(1).lstrip("_")  # "v461_20251127_001"
            ext = filename[filename.rfind("."):]  # preserve original case (.tm or .TM)
            base_name = filename[: tag_match.start()] + ext
        else:
            version_tag = None
            base_name = filename

        results.append(RemoteMetakernel(
            filename=filename,
            url=urljoin(mk_dir_url, filename),
            date=date,
            size=size,
            base_name=base_name,
            version_tag=version_tag,
        ))

    results.sort(key=lambda r: (r.base_name, r.filename))
    return results


def list_remote_missions(server_url: str) -> list[str]:
    """List available mission directories from a SPICE archive server.

    Parses the Apache directory listing at *server_url* and returns
    mission directory names (e.g. ``['CASSINI', 'JUICE', 'MRO']``).

    Directories that don't look like missions (e.g. ``toolkit/``,
    ``cosmographia/``, ``misc/``) are included — the caller can
    filter if needed.
    """
    if not server_url.endswith("/"):
        server_url += "/"

    with urllib.request.urlopen(server_url) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    missions: list[str] = []
    for m in _DIR_ENTRY_RE.finditer(html):
        dirname = m.group(1).rstrip("/")
        # Skip parent directory link and hidden dirs
        if dirname in (".", "..") or dirname.startswith("."):
            continue
        missions.append(dirname)

    missions.sort()
    return missions


def check_mk_availability(
    server_url: str, missions: list[str], *, max_workers: int = 16
) -> dict[str, bool]:
    """Check which missions have a ``kernels/mk/`` directory.

    Sends parallel HEAD requests to ``{server_url}{mission}/kernels/mk/``
    and returns a mapping of mission name → availability.
    """
    if not server_url.endswith("/"):
        server_url += "/"

    def _check(name: str) -> tuple[str, bool]:
        url = f"{server_url}{name}/kernels/mk/"
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=5):
                return name, True
        except Exception:
            return name, False

    result: dict[str, bool] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        transient=True,
    ) as progress:
        task_id = progress.add_task("Checking metakernel directories", total=len(missions))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_check, m): m for m in missions}
            for future in as_completed(futures):
                name, available = future.result()
                result[name] = available
                progress.advance(task_id)
    return result


def fetch_metakernel(url: str) -> str:
    """Download metakernel text content from a URL."""
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8", errors="replace")


def resolve_kernel_urls(
    mk_url: str, parsed: ParsedMetakernel
) -> list[str]:
    """Resolve each KERNELS_TO_LOAD entry to a full URL.

    The metakernel's PATH_VALUES are interpreted relative to the
    metakernel's own directory URL, then substituted into each
    kernel entry.

    Example:
        mk_url  = "https://naif.jpl.nasa.gov/.../mk/juice.tm"
        PATH_VALUES  = ('..')
        PATH_SYMBOLS = ('KERNELS')
        entry = '$KERNELS/lsk/naif0012.tls'
        → "https://naif.jpl.nasa.gov/.../kernels/lsk/naif0012.tls"
    """
    # Base directory of the metakernel URL
    if not mk_url.endswith("/"):
        mk_dir = mk_url.rsplit("/", 1)[0] + "/"
    else:
        mk_dir = mk_url

    # Build symbol → resolved URL base mapping
    symbol_urls: dict[str, str] = {}
    for sym, val in zip(parsed.path_symbols, parsed.path_values):
        # Resolve PATH_VALUE (e.g. '..') relative to mk directory
        resolved = urljoin(mk_dir, val)
        if not resolved.endswith("/"):
            resolved += "/"
        symbol_urls[sym] = resolved

    urls: list[str] = []
    for raw in parsed.kernels:
        url = raw
        for sym, base_url in symbol_urls.items():
            url = url.replace(f"${sym}/", base_url).replace(f"${sym}", base_url)
        urls.append(url)

    return urls


def _head_size(url: str) -> tuple[str, int | None]:
    """Return (url, content_length_or_None) via HTTP HEAD."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as resp:
            cl = resp.headers.get("Content-Length")
            return url, int(cl) if cl else None
    except Exception:
        return url, None


def query_remote_sizes(
    urls: list[str], *, max_workers: int = 8
) -> dict[str, int | None]:
    """Query Content-Length for multiple URLs in parallel."""
    out: dict[str, int | None] = {}
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total} files"),
        transient=True,
    ) as progress:
        task_id = progress.add_task("Querying sizes", total=len(urls))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_head_size, u): u for u in urls}
            for future in as_completed(futures):
                url, size = future.result()
                out[url] = size
                progress.advance(task_id)
    return out


def download_kernel(
    url: str, dest: Path, *, progress: Progress | None = None,
    task_id: int | None = None,
) -> Path:
    """Download a single kernel file to *dest*.

    Creates parent directories as needed. If *progress* and *task_id*
    are given, updates the progress bar with bytes written. Returns *dest*.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                if progress is not None and task_id is not None:
                    progress.advance(task_id, len(chunk))
    return dest


def download_kernels_parallel(
    tasks: list[tuple[str, Path, str]],
    *,
    max_workers: int = 8,
    total_bytes: int | None = None,
) -> tuple[list[Path], list[str]]:
    """Download multiple kernels in parallel with a byte-level progress bar.

    Args:
        tasks: List of (url, dest_path, filename) tuples.
        max_workers: Maximum concurrent downloads.
        total_bytes: Total expected bytes (for progress bar). If None,
            the bar counts files instead of bytes.

    Returns:
        (downloaded_paths, warnings)
    """
    if total_bytes and total_bytes > 0:
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ]
    else:
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} files"),
        ]

    downloaded: list[Path] = []
    warnings: list[str] = []
    with Progress(*columns) as progress:
        if total_bytes and total_bytes > 0:
            pid = progress.add_task("Downloading", total=total_bytes)
        else:
            pid = progress.add_task("Downloading", total=len(tasks))

        def _do(task: tuple[str, Path, str]) -> tuple[str, Path | None, str | None]:
            url, dest, filename = task
            try:
                download_kernel(
                    url, dest,
                    progress=progress if total_bytes else None,
                    task_id=pid if total_bytes else None,
                )
                if not total_bytes:
                    progress.advance(pid)
                return filename, dest, None
            except Exception as e:
                return filename, None, str(e)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_do, t): t for t in tasks}
            for future in as_completed(futures):
                fname, dest, error = future.result()
                if error is None:
                    downloaded.append(dest)
                else:
                    warnings.append(f"{fname}: {error}")

    return downloaded, warnings
