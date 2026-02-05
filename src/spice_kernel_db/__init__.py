"""Content-addressed SPICE kernel database for deduplication and metakernel rewriting."""

from spice_kernel_db.db import KernelDB
from spice_kernel_db.parser import parse_metakernel

__all__ = ["KernelDB", "parse_metakernel"]
__version__ = "0.1.0"
