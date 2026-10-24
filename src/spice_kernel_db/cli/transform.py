"""Commands that change the database or the files on disk: scan, rewrite, dedup, prune."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from spice_kernel_db.db import KernelDB
from spice_kernel_db.remote import SPICE_SERVERS, server_label_for

from ._app import TRANSFORM, app, open_db, state
from .prompts import _ask, _confirm

@app.command(rich_help_panel=TRANSFORM, help="Scan a directory for kernel files")
def scan(
    ctx: typer.Context,
    directory: Annotated[str, typer.Argument(help="Root directory to scan recursively")],
    mission: Annotated[Optional[str], typer.Option(
        "--mission", help="Override auto-detected mission name")] = None,
    archive: Annotated[bool, typer.Option(
        "--archive",
        help="Move files to the configured kernel directory and leave symlinks")] = False,
) -> None:
    st = state(ctx)
    with open_db(ctx) as db:
        count, missions_found = db.scan_directory(
            directory,
            mission=mission,
            verbose=st.verbose,
            archive_dir=st.config.kernel_dir if archive else None,
        )
        _configure_new_missions(
            db, missions_found, Path(directory).expanduser().resolve(),
        )
        db.stats()


@app.command(rich_help_panel=TRANSFORM, help="Rewrite a metakernel with a local symlink tree")
def rewrite(
    ctx: typer.Context,
    metakernel: Annotated[str, typer.Argument(help="Path to original .tm file")],
    output: Annotated[Optional[str], typer.Option(
        "-o", "--output", help="Output .tm path (default: <stem>_local.tm)")] = None,
    link_root: Annotated[Optional[str], typer.Option(
        "--link-root", help="Root for symlink tree (default: kernels/ next to output)")] = None,
    mission: Annotated[Optional[str], typer.Option(
        "--mission", help="Preferred mission for kernel resolution")] = None,
) -> None:
    if output is None:
        mk = Path(metakernel)
        output = str(mk.with_stem(mk.stem + "_local"))
    with open_db(ctx) as db:
        db.rewrite_metakernel(metakernel, output, mission=mission, link_root=link_root)


@app.command(rich_help_panel=TRANSFORM, help="Deduplicate kernel files using symlinks")
def dedup(
    ctx: typer.Context,
    execute: Annotated[bool, typer.Option(
        "--execute", help="Actually replace files (default: dry run)")] = False,
    yes: Annotated[bool, typer.Option(
        "-y", "--yes", help="Skip confirmation prompt when --execute is used")] = False,
) -> None:
    with open_db(ctx) as db:
        if execute and not yes:
            # Show dry run first, then ask
            db.deduplicate_with_symlinks(dry_run=True)
            if not _confirm("\nProceed with deduplication? [y/N]: "):
                print("Aborted.")
                return
        db.deduplicate_with_symlinks(dry_run=not execute)


@app.command(rich_help_panel=TRANSFORM,
             help="Remove stale DB entries for files no longer on disk "
                  "(or, with --metakernels, registry rows whose source URL is dead)")
def prune(
    ctx: typer.Context,
    execute: Annotated[bool, typer.Option(
        "--execute", help="Actually remove entries (default: dry run)")] = False,
    prune_metakernels: Annotated[bool, typer.Option(
        "--metakernels", "--mk",
        help="Prune metakernel_registry rows whose remote URL "
             "returns 403/404/410 (NAIF often rotates old snapshots into "
             "former_versions/)")] = False,
    prune_orphan_symlinks: Annotated[bool, typer.Option(
        "--orphan-symlinks",
        help="Find and remove dangling symlinks under each mission's "
             "download tree (left behind when a kernel store moves or "
             "after default prune removes the location row)")] = False,
    delete_files: Annotated[bool, typer.Option(
        "--delete-files",
        help="With --metakernels --execute: also unlink the on-disk .tm files")] = False,
) -> None:
    if prune_metakernels and prune_orphan_symlinks:
        print("--metakernels and --orphan-symlinks are mutually exclusive.", file=sys.stderr)
        sys.exit(1)
    with open_db(ctx) as db:
        if prune_metakernels:
            db.prune_metakernels(dry_run=not execute, delete_files=delete_files)
        elif prune_orphan_symlinks:
            db.prune_orphan_symlinks(dry_run=not execute)
        else:
            db.prune(dry_run=not execute)


def _read_spice_server_marker(directory: Path) -> dict[str, str] | None:
    """Read a .spice-server marker file from a directory.

    Returns a dict with 'server_url' and 'mk_dir_url' keys, or None.
    """
    marker = directory / ".spice-server"
    if not marker.is_file():
        return None
    data: dict[str, str] = {}
    for line in marker.read_text().splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            data[key.strip()] = val.strip()
    if "server_url" in data and "mk_dir_url" in data:
        return data
    return None


def _configure_new_missions(
    db: KernelDB, missions_found: set[str], scan_root: Path,
) -> None:
    """Configure missions found during scan that aren't in the DB yet.

    Checks for .spice-server marker files first, prompts user as fallback.
    """
    skip_names = {"generic", "unknown"}
    server_names = list(SPICE_SERVERS.keys())

    for name in sorted(missions_found):
        if name.lower() in skip_names:
            continue
        if db.get_mission(name):
            continue

        # Look for .spice-server marker in <root>/<mission>/ and <root>/<mission>/kernels/
        marker = None
        for candidate in [scan_root / name, scan_root / name / "kernels"]:
            marker = _read_spice_server_marker(candidate)
            if marker:
                break

        if marker:
            mk_dir_url = marker["mk_dir_url"]
            server_url = marker["server_url"]
            db.add_mission(name, server_url, mk_dir_url)
            print(f"  Configured {name} from .spice-server ({server_label_for(server_url)}).")
            continue

        # No marker — prompt user
        print(f"\n  {name} is not yet configured for remote access.")
        for i, sname in enumerate(server_names, 1):
            print(f"    [{i}] {sname}  ({SPICE_SERVERS[sname]})")
        print(f"    [{len(server_names) + 1}] Skip")
        while True:
            choice = _ask(f"  Select [1-{len(server_names) + 1}]: ").strip()
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(server_names):
                    server_label = server_names[idx - 1]
                    server_url = SPICE_SERVERS[server_label]
                    mk_dir_url = f"{server_url}{name}/kernels/mk/"
                    db.add_mission(name, server_url, mk_dir_url)
                    print(f"  Configured {name} ({server_label}).")
                    break
                elif idx == len(server_names) + 1:
                    break
            print("  Invalid choice.")
