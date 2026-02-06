"""Content-addressed SPICE kernel database for deduplication and metakernel rewriting."""

from spice_kernel_db.config import Config, ensure_config
from spice_kernel_db.db import KernelDB
from spice_kernel_db.parser import parse_metakernel, parse_metakernel_text
from spice_kernel_db.remote import (
    fetch_metakernel,
    list_remote_metakernels,
    resolve_kernel_urls,
)

__all__ = [
    "Config",
    "KernelDB",
    "ensure_config",
    "fetch_metakernel",
    "list_remote_metakernels",
    "parse_metakernel",
    "parse_metakernel_text",
    "resolve_kernel_urls",
]
__version__ = "0.3.0"
