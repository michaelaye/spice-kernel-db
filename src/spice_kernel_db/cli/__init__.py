"""Command-line interface for spice-kernel-db.

The commands live in one module per help panel; importing them here
registers them on ``app``. ``main()`` is the console-script entry point.
"""

from __future__ import annotations

import sys
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Annotated, Optional

import typer
from typer import rich_utils

from spice_kernel_db.config import ensure_config
from spice_kernel_db.db import KernelDB

from ._app import State, app
from .prompts import _metakernel_table, _registry_rows, console

# Register the commands, in the order the help panels should appear.
from . import browse, inspect_cmds, transform, configure  # noqa: E402, F401, I001


def _version(flag: bool) -> None:
    if flag:
        print(f"spice-kernel-db {pkg_version('spice-kernel-db')}")
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
    # ensure_config is handed over, not called: a command loads the config
    # the first time it needs it, so `<command> --help` never prompts.
    ctx.obj = State(db_override=db, verbose=verbose, load_config=ensure_config)


def _show_default_summary() -> None:
    """Show a useful summary when invoked with no subcommand.

    Lists locally acquired metakernels if any exist, otherwise prints
    a quick-start guide.
    """
    from spice_kernel_db.config import load_config

    console.print(
        f"[bold]spice-kernel-db[/bold] {pkg_version('spice-kernel-db')}\n"
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
        rows = _registry_rows(db)

    if rows:
        console.print(_metakernel_table(rows, index_style="dim"))
        console.print(
            "\n[dim]Use 'spice-kernel-db check <filename>' to verify kernel availability.\n"
            "Use 'spice-kernel-db resolve <kernel>' to get a kernel's local path.\n"
            "Use 'spice-kernel-db update <filename>' to fetch the latest version.\n"
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
        rc = app(args=argv, prog_name="spice-kernel-db", standalone_mode=False)
    except typer.Abort:
        print("Aborted.", file=sys.stderr)
        sys.exit(1)
    except typer.TyperException as exc:
        rich_utils.rich_format_error(exc)
        sys.exit(exc.exit_code)
    if isinstance(rc, int):  # Exit codes only; a command's return value is not one
        sys.exit(rc)
