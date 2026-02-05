"""Remote SPICE kernel operations — fetch, resolve, and download.

All networking uses stdlib urllib so no extra dependencies are needed.
"""

from __future__ import annotations

import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

from spice_kernel_db.parser import ParsedMetakernel


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
    """Query Content-Length for multiple URLs in parallel.

    Returns {url: size_bytes} where size_bytes is None if unavailable.
    """
    results: dict[str, int | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_head_size, u): u for u in urls}
        for future in as_completed(futures):
            url, size = future.result()
            results[url] = size
    return results


def download_kernel(url: str, dest: Path) -> Path:
    """Download a single kernel file to *dest*.

    Creates parent directories as needed. Returns *dest*.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    return dest
