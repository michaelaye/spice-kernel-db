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

from spice_kernel_db._version import __version__
from spice_kernel_db.config import ConfigError, LegacyDatabaseFound
from spice_kernel_db.db import (
    AmbiguousKernelRoot,
    AmbiguousMetakernel,
    ConcurrentModificationError,
    KernelDB,
    MetakernelNotTracked,
    MetakernelUnreachableError,
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
    """The exit code for one of this package's exceptions, or ``None`` for any other.

    This is part of the host contract with ``spice_kernel_db.cli.app`` and ``logging_to_stderr``. A
    program with its own Typer CLI mounts skd's command tree as one of its groups with
    ``app.add_typer(spice_kernel_db.cli.app, name=...)``. This is what planetarypy does as
    ``plp spice``. The host catches what escapes its Typer app and asks here, so both front doors
    exit the same way. ``main()`` uses it too (#45, #49; planetarypy #51). These three names change
    only with a major version. The rest of ``spice_kernel_db.cli`` is internal.

    The mapping:

    - 1: ``MetakernelNotTracked``, and ``ConcurrentModificationError`` (another process changed the
      kernel records during a download; run the command again).
    - 2: ``AmbiguousMetakernel`` and ``AmbiguousKernelRoot``, where the candidates are listed and
      the caller names one, and ``MetakernelUnreachableError``, where the source is gone (HTTP 403,
      404 or 410) and the caller decides what to drop.
    - 78: ``ConfigError`` and ``LegacyDatabaseFound``, the conventional code for a configuration
      error. It is never one of ``check``'s verdict bits, which use 0 to 7.
    - ``None``: any exception that is not skd's. A host re-raises it, so a bug keeps its traceback.

    Parameters
    ----------
    exc : BaseException
        The exception that escaped the app.

    Returns
    -------
    int | None
        The exit code, or ``None`` when the exception has no mapping.

    Examples
    --------
    >>> import sys
    >>> from spice_kernel_db.cli import exit_code, logging_to_stderr
    >>> try:
    ...     with logging_to_stderr():
    ...         app(standalone_mode=False)  # the host's Typer app
    ... except Exception as exc:
    ...     code = exit_code(exc)
    ...     if code is None:
    ...         raise
    ...     print(exc, file=sys.stderr)
    ...     sys.exit(code)
    """
    if isinstance(exc, (AmbiguousMetakernel, AmbiguousKernelRoot, MetakernelUnreachableError)):
        return 2  # the candidates are listed, or the source is gone; the caller decides
    if isinstance(exc, (MetakernelNotTracked, ConcurrentModificationError)):
        return 1
    if isinstance(exc, (ConfigError, LegacyDatabaseFound)):
        return EXIT_CONFIG  # a configuration matter, never one of check's verdict bits
    return None


@contextmanager
def logging_to_stderr() -> Iterator[None]:
    """For one CLI run, print the library's warnings as bare lines on stderr.

    INFO messages are included under ``-v``. This is what Python's last-resort handler printed
    before #49, so a long path is neither wrapped nor coloured. The handler is removed on exit, so
    an embedding process keeps its own logging.

    The library is quiet: its warnings go through the ``spice_kernel_db`` logger, and a script sees
    them only if it configures logging. ``main()`` runs inside this context manager, and so does a
    host CLI that mounts ``spice_kernel_db.cli.app`` (planetarypy's ``plp``). It is part of the host
    contract with ``app`` and ``exit_code``, and changes only with a major version.

    Yields
    ------
    None
        A context manager. Run the app inside the ``with`` block.

    Examples
    --------
    >>> from spice_kernel_db.cli import logging_to_stderr
    >>> with logging_to_stderr():
    ...     app()  # the host's Typer app
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(handler)
    try:
        yield
    finally:
        _LOG.removeHandler(handler)


def _version(flag: bool) -> None:
    if flag:
        print(f"spice-kernel-db {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    db: Annotated[Optional[str], typer.Option(
        "--db", help="Path to DuckDB database file (default: from config)")] = None,
    verbose: Annotated[bool, typer.Option(
        "-v", "--verbose", help="Show detailed output (per-file warnings, update's per-metakernel tables)")] = False,
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
        f"[bold]spice-kernel-db[/bold] {__version__}\n"
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
            "  [bold]spice-kernel-db browse[/bold] MISSION  # see available metakernels (adds the mission)\n"
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
    except typer.Abort:
        print("Aborted.", file=sys.stderr)
        sys.exit(1)
    except typer.TyperException as exc:
        from typer import rich_utils  # pulls in rich.markdown; error path only

        rich_utils.rich_format_error(exc)
        sys.exit(exc.exit_code)
    except Exception as exc:
        # The one mapping for every skd exception (#45), shared with host CLIs;
        # anything else is a bug and keeps its traceback.
        code = exit_code(exc)
        if code is None:
            raise
        print(str(exc), file=sys.stderr)
        sys.exit(code)
    if isinstance(rc, int):  # Exit codes only; a command's return value is not one
        sys.exit(rc)
