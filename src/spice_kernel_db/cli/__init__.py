"""Command-line interface for spice-kernel-db.

The commands live in one module per help panel; importing them here
registers them on ``app``. ``main()`` is the console-script entry point.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Optional

import typer

from spice_kernel_db.config import ConfigError
from spice_kernel_db.db import (
    AmbiguousKernelRoot,
    AmbiguousMetakernel,
    KernelDB,
    MetakernelNotTracked,
)

from ._app import State, app
from .prompts import _metakernel_table, console

# Register the commands, in the order the help panels should appear.
from . import browse, inspect_cmds, transform, configure  # noqa: E402, F401, I001
from .configure import ensure_config_or_setup


# Exit code for a config file that does not parse: outside 0-7, which `check`
# uses as verdict bits, and the sysexits.h value for EX_CONFIG.
EXIT_CONFIG = 78

_LOG = logging.getLogger("spice_kernel_db")


def exit_code(exc: BaseException) -> int | None:
    """The exit code for one of this package's exceptions, or None for any other.

    ``main()`` uses it, and so does a program that mounts
    ``spice_kernel_db.cli.app`` under its own command (planetarypy's
    ``plp spice``): it catches what escapes its Typer app and asks here, so
    both front doors exit the same way (#45, #49; planetarypy #51).
    """
    if isinstance(exc, (AmbiguousMetakernel, AmbiguousKernelRoot)):
        return 2  # the candidates are listed; the caller names one
    if isinstance(exc, MetakernelNotTracked):
        return 1
    if isinstance(exc, ConfigError):
        return EXIT_CONFIG  # never one of check's verdict bits
    return None


@contextmanager
def logging_to_stderr() -> Iterator[None]:
    """For one CLI run, the library's warnings (INFO under ``-v``) as bare lines
    on stderr, what Python's last-resort handler printed before #49, so a long
    path is neither wrapped nor coloured. Removed on exit, so an embedding
    process keeps its own logging. ``main()`` runs inside it, and so does a
    host CLI that mounts ``app`` (planetarypy's ``plp``): part of the host
    contract with ``app`` and ``exit_code``.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(handler)
    try:
        yield
    finally:
        _LOG.removeHandler(handler)


def _pkg_version() -> str:
    # importlib.metadata is imported on demand: it costs ~20 ms and only
    # ``--version`` and the no-args summary need it (#26).
    from importlib.metadata import version

    return version("spice-kernel-db")


def _version(flag: bool) -> None:
    if flag:
        print(f"spice-kernel-db {_pkg_version()}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    db: Annotated[Optional[str], typer.Option(
        "--db", help="Path to DuckDB database file (default: from config)")] = None,
    verbose: Annotated[bool, typer.Option(
        "-v", "--verbose", help="Show detailed output (per-file warnings, etc.)")] = False,
    version: Annotated[bool, typer.Option(
        "--version", help="Show program's version number and exit",
        callback=_version, is_eager=True)] = False,
) -> None:
    if ctx.invoked_subcommand is None:
        # No subcommand — show a useful summary if possible, else help.
        _show_default_summary()
        return
    _LOG.setLevel(logging.INFO if verbose else logging.WARNING)
    # The loader is handed over, not called: a command loads the config the
    # first time it needs it, so `<command> --help` never prompts.
    ctx.obj = State(db_override=db, verbose=verbose, load_config=ensure_config_or_setup)


def _show_default_summary() -> None:
    """Show a useful summary when invoked with no subcommand.

    Lists locally acquired metakernels if any exist, otherwise prints
    a quick-start guide.
    """
    from spice_kernel_db.config import load_config

    console.print(
        f"[bold]spice-kernel-db[/bold] {_pkg_version()}\n"
    )

    config = load_config()
    if not config:
        console.print(
            "No configuration found. Run [bold]spice-kernel-db config --setup[/bold] to get started.\n"
        )
        return

    db_path = Path(config.db_path).expanduser()
    if not db_path.is_file():
        console.print(
            "No database yet. Quick start:\n"
            "  [bold]spice-kernel-db mission add[/bold]    # configure a mission\n"
            "  [bold]spice-kernel-db browse[/bold] MISSION  # see available metakernels\n"
            "  [bold]spice-kernel-db get[/bold]             # download one interactively\n"
        )
        return

    with KernelDB(db_path, read_only=True) as db:
        rows = db.registry_rows()

    if rows:
        console.print(_metakernel_table(rows, index_style="dim"))
        console.print(
            "\n[dim]Use 'spice-kernel-db check <filename>' to verify kernel availability.\n"
            "Use 'spice-kernel-db which <kernel>' to get a kernel's local path.\n"
            "Use 'spice-kernel-db get <filename>' to fetch its latest version.\n"
            "Run 'spice-kernel-db --help' for all commands.[/dim]\n"
        )
    else:
        console.print(
            "Database exists but no metakernels acquired yet.\n\n"
            "  [bold]spice-kernel-db browse[/bold] MISSION  # see available metakernels\n"
            "  [bold]spice-kernel-db get[/bold]             # download one interactively\n"
        )


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point.

    Runs the app without Click's standalone mode so that a successful command
    returns normally (the tests call ``main([...])`` directly), then exits
    the way the argparse CLI did: 0 for ``--help`` and ``--version``, 2 for a
    usage error, 130 for Ctrl-C (Typer turns it into ``Exit(130)``).
    """
    try:
        with logging_to_stderr():
            rc = app(args=argv, prog_name="spice-kernel-db", standalone_mode=False)
    except (AmbiguousMetakernel, AmbiguousKernelRoot, MetakernelNotTracked, ConfigError) as exc:
        # The one mapping for every command (#45), shared with host CLIs.
        print(str(exc), file=sys.stderr)
        sys.exit(exit_code(exc))
    except typer.Abort:
        print("Aborted.", file=sys.stderr)
        sys.exit(1)
    except typer.TyperException as exc:
        from typer import rich_utils  # pulls in rich.markdown; error path only

        rich_utils.rich_format_error(exc)
        sys.exit(exc.exit_code)
    if isinstance(rc, int):  # Exit codes only; a command's return value is not one
        sys.exit(rc)
