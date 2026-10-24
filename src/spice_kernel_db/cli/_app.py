"""The Typer application object and the per-invocation state it carries.

Command modules import ``app`` from here and register themselves with
``@app.command(...)``; ``cli/__init__.py`` imports those modules so the
registration happens, then defines the root callback and ``main()``.
Keeping ``app`` in its own module avoids a circular import.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

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
    """Root options plus the config, shared by every command.

    The config is loaded on first use, not when the root callback runs:
    Click invokes the root callback before a subcommand parses ``--help``,
    so eager loading made ``skd get --help`` start first-time setup on a
    machine without a config file.
    """

    db_override: str | None
    verbose: bool
    load_config: Callable[[], Config]
    _config: Config | None = field(default=None, init=False, repr=False)

    @property
    def config(self) -> Config:
        if self._config is None:
            self._config = self.load_config()
        return self._config

    @property
    def db(self) -> str:
        return self.db_override or self.config.db_path


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
