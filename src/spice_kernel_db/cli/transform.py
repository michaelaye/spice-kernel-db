"""Maintain panel: scan, import, rewrite, rm, clean, dedup (reset lives in configure.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from spice_kernel_db.db import AmbiguousKernelRoot, KernelDB, MetakernelNotTracked
from spice_kernel_db.remote import SPICE_SERVERS, server_label_for

from ._app import MAINTAIN, app, open_db, rename_note, state
from .prompts import _confirm, _pick_root, _require_metakernel


@app.command(rich_help_panel=MAINTAIN, epilog="See also: import, config add-root, mission add",
             help="Index every kernel file under a directory "
                  "(no directory: every configured kernel root)")
def scan(
    ctx: typer.Context,
    directory: Annotated[Optional[str], typer.Argument(
        help="Directory to scan recursively (default: each kernel root in config)")] = None,
    mission: Annotated[Optional[str], typer.Option(
        "--mission", help="Override auto-detected mission name")] = None,
    archive: Annotated[bool, typer.Option(
        "--archive",
        help="Move files into the kernel root that holds their mission "
             "(or the only root) and leave symlinks")] = False,
) -> None:
    st = state(ctx)
    with open_db(ctx) as db:
        chosen: dict[str, Path] = {}
        while True:
            try:
                if directory:
                    _, missions_found = db.scan_directory(
                        directory, mission=mission, verbose=st.verbose,
                        archive=archive, archive_roots=chosen)
                    scanned = [Path(directory).expanduser().resolve()]
                else:
                    _, missions_found = db.scan_roots(
                        verbose=st.verbose, archive=archive, archive_roots=chosen)
                    scanned = db.kernel_roots
                break
            except AmbiguousKernelRoot as e:
                # --archive, several roots, a mission none of them holds: ask
                # once per such mission (exit 2 in a script). Nothing has
                # moved yet; the scan starts over with the answer.
                chosen[e.mission] = _pick_root(
                    e, hint="--mission NAME makes every file that mission.")
        _note_new_missions(db, missions_found, scanned)
        db.stats()


@app.command("import", rich_help_panel=MAINTAIN, epilog="See also: scan, get, check",
             help="Track a metakernel that is already on disk, without downloading")
def import_(
    ctx: typer.Context,
    metakernel: Annotated[str, typer.Argument(help="Path to the .tm file")],
    mission: Annotated[Optional[str], typer.Option(
        "--mission", help="Override auto-detected mission name")] = None,
    unofficial: Annotated[bool, typer.Option(
        "--unofficial",
        help="Mark it as assembled locally (a subsetter's output), not an "
             "archive metakernel; ls and show say so")] = False,
) -> None:
    with open_db(ctx) as db:
        info = db.import_metakernel(metakernel, mission=mission, official=not unofficial)
    print(f"Tracked {info['filename']} ({info['mission']}): "
          f"{info['n_registered']}/{info['n_entries']} kernels on disk and registered.")
    if info["missing"]:
        print(f"  Not on disk ({len(info['missing'])}):")
        for raw in info["missing"]:
            print(f"    {raw}")
        print(f"  spice-kernel-db get {info['filename']} downloads them once the "
              f"mission's server is configured.")


@app.command(rich_help_panel=MAINTAIN, epilog="See also: check, which",
             help="Rewrite a metakernel with a local symlink tree")
def rewrite(
    ctx: typer.Context,
    metakernel: Annotated[Optional[str], typer.Argument(
        help="Original metakernel: path, registry filename or base name "
             "(omit to pick from the tracked metakernels at a terminal)")] = None,
    output: Annotated[Optional[str], typer.Option(
        "-o", "--output", help="Output .tm path (default: <stem>_local.tm)")] = None,
    link_root: Annotated[Optional[str], typer.Option(
        "--link-root", help="Root for symlink tree (default: kernels/ next to output)")] = None,
    mission: Annotated[Optional[str], typer.Option(
        "--mission", help="Preferred mission for kernel resolution")] = None,
) -> None:
    with open_db(ctx) as db:
        mk = _require_metakernel(metakernel, db, mission)
        if mk is None:
            return
        mk_path = db.resolve_metakernel(mk, mission=mission)
        if output is None:
            output = str(mk_path.with_stem(mk_path.stem + "_local"))
        db.rewrite_metakernel(mk_path, output, mission=mission, link_root=link_root)


@app.command(rich_help_panel=MAINTAIN, epilog="See also: ls, clean",
             help="Stop tracking a metakernel (the file stays on disk)")
def rm(
    ctx: typer.Context,
    metakernel: Annotated[Optional[str], typer.Argument(
        help="Registry filename, base name or path "
             "(omit to pick from the tracked metakernels at a terminal)")] = None,
    mission: Annotated[Optional[str], typer.Option(
        "--mission", help="Mission the name belongs to")] = None,
) -> None:
    with open_db(ctx) as db:
        mk = _require_metakernel(metakernel, db, mission)
        if mk is None:
            return
        # Matched against the registry, not through the file resolver:
        # removing a row whose file is gone is the point.
        if not db.remove_metakernel(mk):
            raise MetakernelNotTracked(mk)
    print(f"Metakernel '{mk}' removed from registry.")


@app.command(rich_help_panel=MAINTAIN, epilog="See also: doctor, rm",
             help="Drop database rows and symlinks that point at nothing "
                  "(dry run unless --execute)")
def clean(
    ctx: typer.Context,
    execute: Annotated[bool, typer.Option(
        "--execute", help="Actually remove (default: report only)")] = False,
    locations: Annotated[bool, typer.Option(
        "--locations",
        help="Kernel locations whose file is gone, and kernel rows left with "
             "no location")] = False,
    symlinks: Annotated[bool, typer.Option(
        "--symlinks",
        help="Dangling symlinks under each mission's download tree")] = False,
    registry: Annotated[bool, typer.Option(
        "--registry",
        help="Tracked metakernels whose source URL answers 403/404/410 and that "
             "have no copy in former_versions/ (the only class that contacts the server)")] = False,
    delete_files: Annotated[bool, typer.Option(
        "--delete-files",
        help="With --registry --execute: also unlink the on-disk .tm files")] = False,
) -> None:
    if not (locations or symlinks or registry):
        locations = symlinks = True  # the two local classes
    with open_db(ctx) as db:
        if locations:
            db.prune(dry_run=not execute)
        if symlinks:
            db.prune_orphan_symlinks(dry_run=not execute)
        if registry:
            db.prune_metakernels(dry_run=not execute, delete_files=delete_files)


@app.command(hidden=True, help="Renamed: use clean")
def prune(
    ctx: typer.Context,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    prune_metakernels: Annotated[bool, typer.Option("--metakernels", "--mk")] = False,
    prune_orphan_symlinks: Annotated[bool, typer.Option("--orphan-symlinks")] = False,
    delete_files: Annotated[bool, typer.Option("--delete-files")] = False,
) -> None:
    rename_note("prune", "clean")
    print("  --metakernels is --registry, --orphan-symlinks is --symlinks", file=sys.stderr)
    clean(ctx, execute=execute,
          locations=not (prune_metakernels or prune_orphan_symlinks),
          symlinks=prune_orphan_symlinks, registry=prune_metakernels,
          delete_files=delete_files)


@app.command(rich_help_panel=MAINTAIN, epilog="See also: stats, clean",
             help="Deduplicate kernel files using symlinks (dry run unless --execute)")
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


def _note_new_missions(
    db: KernelDB, missions_found: set[str], scan_roots: list[Path],
) -> None:
    """Configure missions found during scan that are not in the DB yet.

    A ``.spice-server`` marker (written by ``get``) configures the mission
    on the spot. Without one the command to run is printed; ``scan`` never
    prompts, so it is safe in a script (#47).
    """
    skip_names = {"generic", "unknown"}
    servers = "|".join(SPICE_SERVERS.values())

    for name in sorted(missions_found):
        if name.lower() in skip_names or db.get_mission(name):
            continue

        marker = None
        for scan_root in scan_roots:
            for candidate in [scan_root / name, scan_root / name / "kernels"]:
                marker = marker or _read_spice_server_marker(candidate)

        if marker:
            db.add_mission(name, marker["server_url"], marker["mk_dir_url"])
            print(f"  Configured {name} from .spice-server "
                  f"({server_label_for(marker['server_url'])}).")
            continue

        print(f"  {name} is not configured for remote access. To browse and get "
              f"its metakernels:\n"
              f"    spice-kernel-db mission add {name} --server-url <{servers}>")
