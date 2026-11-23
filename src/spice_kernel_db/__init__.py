"""Content-addressed SPICE kernel database for deduplication and metakernel rewriting.

The names below are the public API, listed in ``docs/index.qmd`` § Public API,
documented in the generated reference, and held stable within a major version
(#49). Everything else is internal.
"""

from spice_kernel_db.api import (
    check,
    get,
    metakernels_covering,
    register_metakernel,
    resolve_metakernel,
    scan,
)
from spice_kernel_db.config import (
    Config,
    ConfigError,
    LegacyDatabaseFound,
    add_root,
    ensure_config,
    kernel_roots,
    remove_root,
)
from spice_kernel_db.db import (
    AmbiguousKernelRoot,
    AmbiguousMetakernel,
    ConcurrentModificationError,
    KernelDB,
    MetakernelNotTracked,
    MetakernelUnreachableError,
    RegistryRow,
)
from spice_kernel_db.parser import parse_metakernel, parse_metakernel_text

__all__ = [
    "AmbiguousKernelRoot",
    "AmbiguousMetakernel",
    "ConcurrentModificationError",
    "Config",
    "ConfigError",
    "KernelDB",
    "LegacyDatabaseFound",
    "MetakernelNotTracked",
    "MetakernelUnreachableError",
    "RegistryRow",
    "add_root",
    "check",
    "ensure_config",
    "get",
    "kernel_roots",
    "metakernels_covering",
    "parse_metakernel",
    "parse_metakernel_text",
    "register_metakernel",
    "remove_root",
    "resolve_metakernel",
    "scan",
]


def __getattr__(name: str):
    # ``__version__`` is looked up on first access: importlib.metadata costs
    # ~20 ms to import and most imports of the package never read it (#26).
    if name == "__version__":
        from importlib.metadata import version

        return version("spice-kernel-db")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
