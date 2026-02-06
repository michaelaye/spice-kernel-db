"""Remote SPICE kernel operations — fetch, resolve, and download."""

from __future__ import annotations

import re
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

from tqdm.contrib.concurrent import thread_map

from spice_kernel_db.parser import ParsedMetakernel

# Regex for NAIF versioned metakernel snapshots: _v461_20251127_001.tm
_VERSION_TAG_RE = re.compile(r"(_v\d+_\d{8}_\d{3})\.tm$")

# Regex for Apache mod_autoindex directory listing entries
_DIR_LISTING_RE = re.compile(
    r'<a href="([^"]+\.tm)">[^<]+</a>\s+'
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s+"
    r"(\S+)"
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
            base_name = filename[: tag_match.start()] + ".tm"
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
    results = thread_map(
        _head_size, urls, max_workers=max_workers,
        desc="Querying sizes", unit="file",
    )
    return {url: size for url, size in results}


def download_kernel(url: str, dest: Path) -> Path:
    """Download a single kernel file to *dest*.

    Creates parent directories as needed. Returns *dest*.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    return dest


def _download_task(
    task: tuple[str, Path, str],
) -> tuple[str, Path | None, str | None]:
    """Download a single kernel, returning (filename, dest_or_None, error_or_None)."""
    url, dest, filename = task
    try:
        download_kernel(url, dest)
        return filename, dest, None
    except Exception as e:
        return filename, None, str(e)


def download_kernels_parallel(
    tasks: list[tuple[str, Path, str]],
    *,
    max_workers: int = 8,
) -> tuple[list[Path], list[str]]:
    """Download multiple kernels in parallel with a progress bar.

    Args:
        tasks: List of (url, dest_path, filename) tuples.
        max_workers: Maximum concurrent downloads.

    Returns:
        (downloaded_paths, warnings)
    """
    results = thread_map(
        _download_task, tasks, max_workers=max_workers,
        desc="Downloading", unit="file",
    )

    downloaded: list[Path] = []
    warnings: list[str] = []
    for fname, dest, error in results:
        if error is None:
            downloaded.append(dest)
        else:
            warnings.append(f"{fname}: {error}")

    return downloaded, warnings
