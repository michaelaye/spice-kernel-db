"""Content-addressed SPICE kernel database for deduplication and metakernel rewriting."""

from spice_kernel_db.config import Config, ensure_config
from spice_kernel_db.db import KernelDB
from spice_kernel_db.parser import parse_metakernel, parse_metakernel_text
from spice_kernel_db.remote import fetch_metakernel, resolve_kernel_urls

__all__ = [
    "Config",
    "KernelDB",
    "ensure_config",
    "fetch_metakernel",
    "parse_metakernel",
    "parse_metakernel_text",
    "resolve_kernel_urls",
]
__version__ = "0.1.0"
