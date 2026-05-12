"""Remote SPICE kernel operations — fetch, resolve, and download."""

from __future__ import annotations

import re
import shutil
import urllib.error
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


# Fallback candidate template probed when no registry candidate matches.
# Placeholders: {server}, {m}.
#
# An empirical survey of NAIF (https://naif.jpl.nasa.gov/pub/naif/) found that
# every mission either has the standard ``kernels/mk/`` directory or no
# curated metakernel folder at all — none of the obvious alternatives
# (``spice_kernels/mk/``, ``data/spice/mk/``, ``data/mk/``,
# ``kernels/mk/former_versions/``) is in use. So the default list contains
# only the standard path. Real alternate locations (typically PDS bundles)
# should be added explicitly via the curated registry or ``--mk-dir-url``.
DEFAULT_ALT_MK_PATHS: tuple[str, ...] = (
    "{server}{m}/kernels/mk/",
)


def _head_ok(url: str, timeout: float = 5.0) -> bool:
    """Return True if a HEAD request to *url* responds successfully."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


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
        return name, _head_ok(url)

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


def probe_mk_candidates(
    urls: list[str], *, max_workers: int = 8, timeout: float = 5.0
) -> list[str]:
    """Probe candidate metakernel directory URLs in parallel.

    Sends a HEAD request to each URL and returns those that respond
    successfully, **preserving the input order** (which encodes priority).
    Duplicates in the input are collapsed.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)

    if not ordered:
        return []
    if len(ordered) == 1:
        return ordered if _head_ok(ordered[0], timeout) else []

    status: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_head_ok, u, timeout): u for u in ordered}
        for future in as_completed(futures):
            status[futures[future]] = future.result()
    return [u for u in ordered if status.get(u)]


def discover_mk_url(
    server_url: str,
    mission: str,
    *,
    registry_candidates: list[str] | None = None,
    include_default: bool = True,
    timeout: float = 5.0,
) -> list[str]:
    """Discover live metakernel directory URLs for *mission*.

    Composes registry-supplied candidates (highest priority) with the
    default ``kernels/mk/`` probe (unless ``include_default=False``),
    expands placeholders, dedups, and returns those that respond 200.
    Order is preserved (priority).
    """
    if not server_url.endswith("/"):
        server_url += "/"

    candidates: list[str] = list(registry_candidates or [])
    if include_default:
        for tmpl in DEFAULT_ALT_MK_PATHS:
            candidates.append(tmpl.format(server=server_url, m=mission))

    return probe_mk_candidates(candidates, timeout=timeout)


def server_label_for(server_url: str) -> str:
    """Reverse-lookup the canonical label for a server URL ('NASA', 'ESA', or 'custom')."""
    for label, url in SPICE_SERVERS.items():
        if url == server_url:
            return label
    return "custom"


def fetch_metakernel(url: str) -> str:
    """Download metakernel text content from a URL."""
    return _fetch_metakernel(url)[0]


def _fetch_metakernel(url: str) -> tuple[str, str]:
    """Download metakernel; return (text, final_url_after_redirects)."""
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8", errors="replace")
        return text, resp.geturl()


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

    # H3: replace longest symbols first so e.g. $KERNELS_DATA doesn't
    # get clobbered by an earlier $KERNELS replacement.
    ordered = sorted(symbol_urls.items(), key=lambda kv: -len(kv[0]))

    urls: list[str] = []
    for raw in parsed.kernels:
        url = raw
        for sym, base_url in ordered:
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
    task_id: int | None = None, expected_hash: str | None = None,
) -> tuple[Path, str]:
    """Download a single kernel file to *dest*, hashing as we go.

    Creates parent directories as needed. If *progress* and *task_id*
    are given, updates the progress bar with bytes written. Returns
    ``(dest, sha256_hex)`` — the SHA-256 is computed in the same
    streaming pass as the file write, so the caller can trust it
    matches the bytes that landed on disk without a second read pass.

    If *expected_hash* is provided, the streamed hash is compared
    against it and an IOError is raised on mismatch (C4: detects
    in-transit corruption or proxy/CDN mangling when an external
    checksum is known a priori). The partial file is removed.

    Raises IOError if the download is incomplete (bytes written does not
    match Content-Length), if zero bytes are received, or if
    *expected_hash* was given and does not match.
    """
    import hashlib
    dest.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    hasher = hashlib.sha256()
    try:
        with urllib.request.urlopen(url) as resp:
            content_length = resp.headers.get("Content-Length")
            expected_size = int(content_length) if content_length else None
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    hasher.update(chunk)
                    bytes_written += len(chunk)
                    if progress is not None and task_id is not None:
                        progress.advance(task_id, len(chunk))
        # Verify download completeness (Issue 4)
        if bytes_written == 0:
            raise IOError(
                f"Zero-byte download from {url}"
            )
        if expected_size is not None and bytes_written != expected_size:
            raise IOError(
                f"Incomplete download from {url}: "
                f"got {bytes_written} bytes, expected {expected_size}"
            )
        actual_hash = hasher.hexdigest()
        # C4: when caller passed an authoritative checksum, refuse to
        # accept a download whose streamed hash disagrees.
        if expected_hash is not None and actual_hash != expected_hash:
            raise IOError(
                f"Hash mismatch on download from {url}: got "
                f"{actual_hash[:16]}..., expected {expected_hash[:16]}..."
            )
    except Exception:
        # Clean up partial file on any error
        if dest.exists():
            dest.unlink()
        raise
    return dest, actual_hash


def download_kernels_parallel(
    tasks: list[tuple[str, Path, str]],
    *,
    max_workers: int = 8,
    total_bytes: int | None = None,
    expected_hashes: dict[str, str] | None = None,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """Download multiple kernels in parallel with a byte-level progress bar.

    Args:
        tasks: List of (url, dest_path, filename) tuples.
        max_workers: Maximum concurrent downloads.
        total_bytes: Total expected bytes (for progress bar). If None,
            the bar counts files instead of bytes.
        expected_hashes: Optional mapping ``url -> sha256_hex``. When set
            and a URL is present, the streamed hash is checked against
            this value during download; mismatches abort that task.
            Use this to enforce manifest/published checksums (C4).

    Returns:
        ``(downloaded, warnings)`` — ``downloaded`` is a list of
        ``(dest_path, sha256_hex)`` tuples for every successful download.
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

    downloaded: list[tuple[Path, str]] = []
    warnings: list[str] = []
    with Progress(*columns) as progress:
        if total_bytes and total_bytes > 0:
            pid = progress.add_task("Downloading", total=total_bytes)
        else:
            pid = progress.add_task("Downloading", total=len(tasks))

        def _do(
            task: tuple[str, Path, str],
        ) -> tuple[str, Path | None, str | None, str | None]:
            url, dest, filename = task
            try:
                exp = expected_hashes.get(url) if expected_hashes else None
                _, sha = download_kernel(
                    url, dest,
                    progress=progress if total_bytes else None,
                    task_id=pid if total_bytes else None,
                    expected_hash=exp,
                )
                if not total_bytes:
                    progress.advance(pid)
                return filename, dest, sha, None
            except urllib.error.HTTPError as e:
                # Issue 10: Classify HTTP errors by code (must be before OSError)
                if e.code in (403, 404, 410):
                    return filename, None, None, f"[FATAL] HTTP {e.code}: {e.reason}"
                elif e.code >= 500:
                    return filename, None, None, f"[RETRIABLE] HTTP {e.code}: {e.reason}"
                else:
                    return filename, None, None, f"[ERROR] HTTP {e.code}: {e.reason}"
            except urllib.error.URLError as e:
                # Network errors — potentially retriable
                return filename, None, None, f"[RETRIABLE] {e.reason}"
            except (IOError, OSError) as e:
                # Issue 10: Disk/IO problems — ERROR severity
                return filename, None, None, f"[ERROR] {e}"
            except Exception as e:
                return filename, None, None, f"[WARNING] {e}"

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_do, t): t for t in tasks}
            for future in as_completed(futures):
                fname, dest, sha, error = future.result()
                if error is None:
                    downloaded.append((dest, sha))
                else:
                    warnings.append(f"{fname}: {error}")

    return downloaded, warnings
