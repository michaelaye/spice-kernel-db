"""Content-addressed SPICE kernel database for deduplication and metakernel rewriting."""

from spice_kernel_db.config import Config, ensure_config
from spice_kernel_db.db import (
    ConcurrentModificationError,
    KernelDB,
    MetakernelNotTracked,
    MetakernelUnreachableError,
)
from spice_kernel_db.parser import parse_metakernel, parse_metakernel_text
from spice_kernel_db.remote import (
    fetch_metakernel,
    list_remote_metakernels,
    resolve_kernel_urls,
)

__all__ = [
    "ConcurrentModificationError",
    "Config",
    "KernelDB",
    "MetakernelNotTracked",
    "MetakernelUnreachableError",
    "ensure_config",
    "fetch_metakernel",
    "list_remote_metakernels",
    "parse_metakernel",
    "parse_metakernel_text",
    "resolve_kernel_urls",
]


def __getattr__(name: str):
    # ``__version__`` is looked up on first access: importlib.metadata costs
    # ~20 ms to import and most imports of the package never read it (#26).
    if name == "__version__":
        from importlib.metadata import version

        return version("spice-kernel-db")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
