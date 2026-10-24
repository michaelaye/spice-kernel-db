"""The Typer application object and the per-invocation state it carries.

Command modules import ``app`` from here and register themselves with
``@app.command(...)``; ``cli/__init__.py`` imports those modules so the
registration happens, then defines the root callback and ``main()``.
Keeping ``app`` in its own module avoids a circular import.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import typer

from spice_kernel_db.config import Config
from spice_kernel_db.db import KernelDB

# Help panels, in the order they appear in ``--help``.
BROWSE = "browse & acquire"
INSPECT = "inspect"
TRANSFORM = "transform"
CONFIGURE = "configure"

app = typer.Typer(
    name="spice-kernel-db",
    help="Browse, get, and manage SPICE kernels and metakernels",
    epilog="Quick start: mission add → browse MISSION → get → check",
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@dataclass
class State:
    """Root options and the loaded config, shared by every command."""

    db: str
    verbose: bool
    config: Config


def state(ctx: typer.Context) -> State:
    return ctx.obj


@contextmanager
def open_db(ctx: typer.Context, *, read_only: bool = False) -> Iterator[KernelDB]:
    """Open the configured database for the duration of one command.

    Read-only commands can share the DB with a running writer, so each
    command says which kind of connection it needs.
    """
    db = KernelDB(state(ctx).db, read_only=read_only)
    try:
        yield db
    finally:
        db.close()
