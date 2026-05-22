"""Command-line interface for spice-kernel-db."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from importlib.metadata import version as pkg_version

from rich.console import Console
from rich.table import Table

from spice_kernel_db.config import (
    ensure_config,
    setup_interactive,
    show_config,
)
from spice_kernel_db.db import KernelDB, MetakernelUnreachableError
from spice_kernel_db import planetarypy_bridge, registry
from spice_kernel_db.remote import (
    SPICE_SERVERS,
    _head_ok,
    check_mk_availability,
    discover_mk_url,
    list_remote_metakernels,
    list_remote_missions,
    probe_mk_candidates,
    server_label_for,
)

console = Console()


def main(argv: list[str] | None = None):
    # ANSI green for command names when outputting to a terminal
    if sys.stdout.isatty():
        _g, _r = "\033[1;32m", "\033[0m"
    else:
        _g, _r = "", ""

    _EPILOG = f"""\
commands:
  browse & acquire:
    {_g}browse{_r}          Browse remote metakernels in a NAIF mk/ directory
    {_g}get{_r}             Download missing kernels for a remote metakernel
    {_g}update{_r}          Re-fetch a metakernel and download new kernels

  inspect:
    {_g}list{_r}            List kernels contained in a metakernel
    {_g}check{_r}           Check which kernels in a metakernel are available locally
    {_g}coverage{_r}        Check SPK body coverage in a metakernel
    {_g}metakernels{_r}     List tracked metakernels or show details  (alias: {_g}mk{_r})
    {_g}resolve{_r}         Find local path for a kernel filename
    {_g}stats{_r}           Show database statistics
    {_g}duplicates{_r}      Report duplicate kernels

  transform:
    {_g}scan{_r}            Scan a directory for kernel files
    {_g}rewrite{_r}         Rewrite a metakernel with a local symlink tree
    {_g}dedup{_r}           Deduplicate kernel files using symlinks
    {_g}prune{_r}           Remove stale DB entries for deleted files

  configure:
    {_g}mission{_r}         Manage configured missions
    {_g}config{_r}          Show or update configuration
    {_g}reset{_r}           Delete the database and start fresh

quick start:
    mission add → browse MISSION → get → check
"""

    parser = argparse.ArgumentParser(
        prog="spice-kernel-db",
        usage="%(prog)s [-h] [--db DB] {command} ...",
        description="Browse, get, and manage SPICE kernels and metakernels",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {pkg_version('spice-kernel-db')}",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to DuckDB database file (default: from config)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Show detailed output (per-file warnings, etc.)",
    )
    sub = parser.add_subparsers(dest="command", metavar="{command}", help=argparse.SUPPRESS)

    # --- scan ---
    p_scan = sub.add_parser("scan", help="Scan a directory for kernel files")
    p_scan.add_argument("directory", help="Root directory to scan recursively")
    p_scan.add_argument(
        "--mission", help="Override auto-detected mission name"
    )
    p_scan.add_argument(
        "--archive", action="store_true",
        help="Move files to the configured kernel directory and leave symlinks",
    )

    # --- stats ---
    sub.add_parser("stats", help="Show database statistics")

    # --- duplicates ---
    sub.add_parser("duplicates", help="Report duplicate kernels")

    # --- check ---
    p_check = sub.add_parser(
        "check",
        help="Check which kernels in a metakernel are available locally",
        epilog="See also: list, mk, resolve",
    )
    p_check.add_argument(
        "metakernel", nargs="?", default=None,
        help="Path to .tm file (omit to select from local registry)",
    )
    p_check.add_argument("--mission", help="Filter by mission name")

    # --- verify ---
    p_verify = sub.add_parser(
        "verify",
        help="Deep cross-check a metakernel against the database",
        epilog="See also: check",
    )
    p_verify.add_argument(
        "metakernel", nargs="?", default=None,
        help="Path to .tm file (omit to select from local registry)",
    )
    p_verify.add_argument(
        "--deep", action="store_true",
        help="Recompute sha256 of every kernel (slow; default: size-only)",
    )
    p_verify.add_argument(
        "--strict", action="store_true",
        help="Non-zero exit on any non-OK finding (default: only on FATAL)",
    )
    p_verify.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of a table",
    )
    p_verify.add_argument("--mission", help="Filter by mission (for registry lookup)")

    # --- rewrite ---
    p_rewrite = sub.add_parser(
        "rewrite",
        help="Rewrite a metakernel with a local symlink tree",
    )
    p_rewrite.add_argument("metakernel", help="Path to original .tm file")
    p_rewrite.add_argument(
        "-o", "--output", default=None,
        help="Output .tm path (default: <stem>_local.tm)",
    )
    p_rewrite.add_argument(
        "--link-root",
        help="Root for symlink tree (default: kernels/ next to output)",
    )
    p_rewrite.add_argument("--mission", help="Preferred mission for kernel resolution")

    # --- dedup ---
    p_dedup = sub.add_parser(
        "dedup", help="Deduplicate kernel files using symlinks"
    )
    p_dedup.add_argument(
        "--execute", action="store_true",
        help="Actually replace files (default: dry run)",
    )
    p_dedup.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt when --execute is used",
    )

    # --- prune ---
    p_prune = sub.add_parser(
        "prune",
        help="Remove stale DB entries for files no longer on disk "
             "(or, with --metakernels, registry rows whose source URL is dead)",
    )
    p_prune.add_argument(
        "--execute", action="store_true",
        help="Actually remove entries (default: dry run)",
    )
    p_prune.add_argument(
        "--metakernels", "--mk", action="store_true",
        dest="prune_metakernels",
        help="Prune metakernel_registry rows whose remote URL "
             "returns 403/404/410 (NAIF often rotates old snapshots into "
             "former_versions/)",
    )
    p_prune.add_argument(
        "--orphan-symlinks", action="store_true",
        dest="prune_orphan_symlinks",
        help="Find and remove dangling symlinks under each mission's "
             "download tree (left behind when a kernel store moves or "
             "after default prune removes the location row)",
    )
    p_prune.add_argument(
        "--delete-files", action="store_true",
        help="With --metakernels --execute: also unlink the on-disk .tm files",
    )

    # --- resolve ---
    p_resolve = sub.add_parser(
        "resolve",
        help="Find local path for a kernel filename",
        epilog="See also: check, list",
    )
    p_resolve.add_argument(
        "filename", nargs="?", default=None,
        help="Kernel filename to resolve "
             "(omit to pick a tracked metakernel interactively)",
    )
    p_resolve.add_argument("--mission", help="Preferred mission for resolution")
    p_resolve.add_argument(
        "--metakernel", dest="resolve_mk", default=None,
        help="Resolve all kernels in a .tm metakernel",
    )

    # --- metakernels ---
    p_mk = sub.add_parser(
        "metakernels", aliases=["mk"],
        help="List tracked metakernels or show details",
        epilog="See also: check, list",
    )
    p_mk.add_argument(
        "name", nargs="?",
        help="Show info for a specific metakernel (by filename)",
    )
    p_mk.add_argument("--mission", help="Filter by mission name (prefix match)")
    p_mk.add_argument(
        "--remove", action="store_true",
        help="Remove the named metakernel from the registry (does not delete files)",
    )

    # --- get ---
    p_get = sub.add_parser(
        "get",
        help="Download missing kernels for a remote metakernel",
        epilog="See also: browse, update",
    )
    p_get.add_argument(
        "url", nargs="?", default=None,
        help="Full URL, or .tm filename (requires a configured mission). Omit to select interactively.",
    )
    p_get.add_argument(
        "--download-dir", default=None,
        help="Directory for downloaded kernels (default: from config)",
    )
    p_get.add_argument("--mission", help="Override auto-detected mission name")
    p_get.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt",
    )
    p_get.add_argument(
        "--force", action="store_true",
        help="Re-download all kernels even if already on disk",
    )

    # --- update ---
    p_update = sub.add_parser(
        "update",
        help="Re-fetch a metakernel from its source and download new kernels",
        epilog="See also: get, browse",
    )
    p_update.add_argument(
        "metakernel", nargs="?", default=None,
        help="Metakernel filename or path (omit to select interactively)",
    )
    p_update.add_argument("--mission", help="Override mission name")
    p_update.add_argument(
        "--download-dir", default=None,
        help="Directory for downloaded kernels (default: from config)",
    )
    p_update.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt",
    )
    p_update.add_argument(
        "--force", action="store_true",
        help="Re-download all kernels even if already on disk",
    )

    # --- browse ---
    p_browse = sub.add_parser(
        "browse",
        help="Browse remote metakernels in a NAIF mk/ directory",
        epilog="See also: get, mission add",
    )
    p_browse.add_argument(
        "url", nargs="?", default=None,
        help="URL to a mission's mk/ directory, or a mission name (omit to list known directories)",
    )
    p_browse.add_argument("--mission", help="Override auto-detected mission name")
    p_browse.add_argument(
        "--show-versioned", action="store_true",
        help="Show versioned snapshots under each metakernel",
    )
    p_browse.add_argument(
        "--sort", dest="sort_by", choices=["name", "date"], default="name",
        help="Row ordering: 'name' (default, alphabetical) or 'date' "
             "(by latest remote date ascending — newest at the bottom)",
    )
    p_browse.add_argument(
        "--archived", action="store_true",
        help="Browse the mission's former_versions/ subdirectory instead "
             "of the live mk/ listing — for older versioned metakernels "
             "that have been superseded",
    )
    p_browse.add_argument(
        "--filter", dest="filter", metavar="SUBSTRING", default=None,
        help="Case-insensitive substring filter on metakernel filenames "
             "(useful for narrowing large archive listings, e.g. "
             "--filter crema_5_1)",
    )

    # --- mission ---
    p_mission = sub.add_parser(
        "mission",
        help="Manage configured missions",
    )
    mission_sub = p_mission.add_subparsers(dest="mission_command")
    p_mission_add = mission_sub.add_parser(
        "add",
        help="Add a new mission (interactive when no arguments)",
    )
    p_mission_add.add_argument(
        "name", nargs="?",
        help="Mission name. When given, the prompts are skipped.",
    )
    p_mission_add.add_argument(
        "--server-url",
        help="Archive server base URL. Required with positional name unless "
             "the name is unambiguous on a known server.",
    )
    p_mission_add.add_argument(
        "--mk-dir-url",
        help="Override the metakernel directory URL (bypasses auto-discovery).",
    )
    p_mission_add.add_argument(
        "--no-dedup", action="store_true",
        help="Disable deduplication for this mission.",
    )
    p_mission_add.add_argument(
        "--use-planetarypy", action="store_true",
        help="Delegate kernel management to planetarypy if the [planetarypy] "
             "extra is installed and the mission is registry-flagged.",
    )
    mission_sub.add_parser("list", help="List configured missions")
    p_mission_rm = mission_sub.add_parser("remove", help="Remove a mission")
    p_mission_rm.add_argument("name", help="Mission name to remove")

    # --- list ---
    p_list = sub.add_parser(
        "list",
        help="List kernels contained in a metakernel",
        epilog="See also: check, mk",
    )
    p_list.add_argument(
        "metakernel", nargs="?", default=None,
        help="Path to .tm file (omit to select from local registry)",
    )
    p_list.add_argument("--mission", help="Filter by mission name")
    p_list.add_argument(
        "--type", dest="kernel_type", default=None,
        help="Filter by kernel type (ck, spk, fk, ik, pck, lsk, sclk, dsk)",
    )

    # --- coverage ---
    p_cov = sub.add_parser(
        "coverage",
        help="Check SPK body coverage in a metakernel",
    )
    p_cov.add_argument(
        "body",
        help="NAIF body ID (e.g. 399) or name (e.g. Earth, Mars)",
    )
    p_cov.add_argument(
        "metakernel", nargs="?", default=None,
        help="Path to .tm file or registry filename (omit to select interactively)",
    )
    p_cov.add_argument("--mission", help="Preferred mission for kernel resolution")

    # --- config ---
    p_config = sub.add_parser("config", help="Show or update configuration")
    p_config.add_argument(
        "--setup", action="store_true",
        help="Re-run interactive setup",
    )
    p_config.add_argument(
        "config_args", nargs="*", default=[],
        help="'set <key> <value>' or 'get <key>' (keys: db_path, kernel_dir)",
    )

    # --- reset ---
    p_reset = sub.add_parser("reset", help="Delete the database and start fresh")
    p_reset.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args(argv)

    if not args.command:
        # No subcommand — show a useful summary if possible, else help.
        _show_default_summary()
        return

    # --- Load config (deferred past --help / --version / no-args) ---
    config = ensure_config()
    if args.db is None:
        args.db = config.db_path
    if hasattr(args, "download_dir") and args.download_dir is None:
        args.download_dir = config.kernel_dir

    # --- Commands that don't need a DB connection ---
    if args.command == "config":
        if args.setup:
            setup_interactive()
        elif args.config_args:
            from spice_kernel_db.config import save_config
            ca = args.config_args
            if ca[0] == "get" and len(ca) == 2:
                key = ca[1]
                if hasattr(config, key):
                    print(getattr(config, key))
                else:
                    print(f"Unknown key: {key}. Valid keys: db_path, kernel_dir", file=sys.stderr)
                    sys.exit(1)
            elif ca[0] == "set" and len(ca) == 3:
                key, value = ca[1], ca[2]
                if hasattr(config, key):
                    setattr(config, key, value)
                    save_config(config)
                    print(f"{key} = {value}")
                else:
                    print(f"Unknown key: {key}. Valid keys: db_path, kernel_dir", file=sys.stderr)
                    sys.exit(1)
            else:
                print("Usage: spice-kernel-db config [set <key> <value> | get <key>]", file=sys.stderr)
                sys.exit(1)
        else:
            show_config(config)
        return

    if args.command == "reset":
        db_path = Path(args.db).expanduser()
        if not db_path.is_file():
            print(f"No database found at {db_path}")
            return
        if not args.yes:
            answer = input(
                f"Delete database at {db_path}? [y/N]: "
            ).strip().lower()
            if answer not in ("y", "yes"):
                print("Aborted.")
                return
        db_path.unlink()
        kernel_dir = Path(config.kernel_dir).expanduser()
        print(f"Database deleted: {db_path}")
        print(f"Kernel files in {kernel_dir} are untouched.\n")
        print("To re-index existing kernels:")
        print(f"  spice-kernel-db scan {kernel_dir}")
        return

    # --- All other commands need a DB connection ---
    # Read-only commands can share the DB with a running writer.
    read_only_commands = {"stats", "duplicates", "check", "list", "resolve",
                          "metakernels", "mk", "coverage"}
    read_only = args.command in read_only_commands
    # `mk --remove` is a destructive variant of an otherwise read-only
    # command; it needs a writable connection.
    if (
        args.command in ("mk", "metakernels")
        and getattr(args, "remove", False)
    ):
        read_only = False
    db = KernelDB(args.db, read_only=read_only)
    try:
        if args.command == "scan":
            archive_dir = config.kernel_dir if args.archive else None
            count, missions_found = db.scan_directory(
                args.directory,
                mission=args.mission,
                verbose=args.verbose,
                archive_dir=archive_dir,
            )
            _configure_new_missions(
                db, missions_found, Path(args.directory).expanduser().resolve(),
            )
            db.stats()

        elif args.command == "stats":
            db.stats()

        elif args.command == "duplicates":
            db.report_duplicates()

        elif args.command == "list":
            metakernel = _require_metakernel(args.metakernel, db, args.mission)
            if metakernel is None:
                return
            _list_kernels(metakernel, kernel_type=args.kernel_type)

        elif args.command == "check":
            metakernel = _require_metakernel(args.metakernel, db, args.mission)
            if metakernel is None:
                return
            db.check_metakernel(
                metakernel, mission=args.mission, verbose=args.verbose,
            )

        elif args.command == "verify":
            metakernel = _require_metakernel(args.metakernel, db, args.mission)
            if metakernel is None:
                return
            result = db.verify_metakernel(metakernel, deep=args.deep)
            if args.json:
                import json as _json
                print(_json.dumps(result, indent=2, default=str))
            else:
                _render_verify_table(result)
            # Exit code: non-zero on fatal (P0) issues by default; on any
            # non-OK finding with --strict.
            bad = result["fatal"] or (args.strict and result["fail"] > 0)
            if bad:
                import sys as _sys
                _sys.exit(1)

        elif args.command == "rewrite":
            output = args.output
            if output is None:
                mk = Path(args.metakernel)
                output = str(mk.with_stem(mk.stem + "_local"))
            db.rewrite_metakernel(
                args.metakernel,
                output,
                mission=args.mission,
                link_root=args.link_root,
            )

        elif args.command == "dedup":
            if args.execute and not args.yes:
                # Show dry run first, then ask
                db.deduplicate_with_symlinks(dry_run=True)
                answer = input(
                    "\nProceed with deduplication? [y/N]: "
                ).strip().lower()
                if answer not in ("y", "yes"):
                    print("Aborted.")
                    return
            db.deduplicate_with_symlinks(dry_run=not args.execute)

        elif args.command == "prune":
            if args.prune_metakernels and args.prune_orphan_symlinks:
                print(
                    "--metakernels and --orphan-symlinks are mutually exclusive.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if args.prune_metakernels:
                db.prune_metakernels(
                    dry_run=not args.execute,
                    delete_files=args.delete_files,
                )
            elif args.prune_orphan_symlinks:
                db.prune_orphan_symlinks(dry_run=not args.execute)
            else:
                db.prune(dry_run=not args.execute)

        elif args.command == "resolve":
            if args.resolve_mk:
                # Batch mode — resolve all kernels in a metakernel
                from spice_kernel_db import parse_metakernel as _parse_mk
                parsed = _parse_mk(args.resolve_mk)
                all_ok = True
                for kernel in parsed.kernel_filenames():
                    path, warnings = db.resolve_kernel(
                        kernel, preferred_mission=args.mission,
                    )
                    if path:
                        print(f"{kernel}\t{path}")
                    else:
                        print(f"{kernel}\tNOT FOUND", file=sys.stderr)
                        all_ok = False
                    for w in warnings:
                        print(f"  ⚠ {w}", file=sys.stderr)
                if not all_ok:
                    sys.exit(1)
            elif args.filename:
                path, warnings = db.resolve_kernel(
                    args.filename, preferred_mission=args.mission,
                )
                if path:
                    print(path)
                else:
                    print(f"Not found: {args.filename}", file=sys.stderr)
                    print(
                        f"  Hint: the kernel may exist on disk but not in the database.\n"
                        f"  Run 'spice-kernel-db scan <directory>' to re-index.",
                        file=sys.stderr,
                    )
                for w in warnings:
                    print(f"  ⚠ {w}", file=sys.stderr)
            else:
                # No filename — offer interactive picker over local metakernels
                picked = _interactive_pick_local_metakernel(db, args.mission)
                if picked is None:
                    sys.exit(1)
                print(picked)

        elif args.command in ("metakernels", "mk"):
            if args.remove:
                if not args.name:
                    print("Usage: spice-kernel-db mk --remove <name>", file=sys.stderr)
                    sys.exit(1)
                if db.remove_metakernel(args.name):
                    print(f"Metakernel '{args.name}' removed from registry.")
                else:
                    print(f"Metakernel '{args.name}' not found.", file=sys.stderr)
            elif args.name:
                db.info_metakernel(args.name)
            else:
                db.list_metakernels(mission=args.mission)

        elif args.command == "coverage":
            try:
                body_id = _resolve_body_interactive(args.body)
            except ImportError:
                print(
                    "SpiceyPy is required for coverage analysis.\n"
                    "Install it with: pip install spice-kernel-db[spice]",
                    file=sys.stderr,
                )
                sys.exit(1)
            if body_id is None:
                return

            metakernel = _require_metakernel(args.metakernel, db, args.mission)
            if metakernel is None:
                return

            try:
                results = db.coverage_metakernel(
                    metakernel, body_id, mission=args.mission,
                )
            except ImportError:
                print(
                    "SpiceyPy is required for coverage analysis.\n"
                    "Install it with: pip install spice-kernel-db[spice]",
                    file=sys.stderr,
                )
                sys.exit(1)
            _print_coverage_table(results, body_id, metakernel)

        elif args.command == "get":
            url = args.url
            alias_filename: str | None = None
            if url is None:
                # Interactive selection — fetch remote listing
                selection = _interactive_select_metakernel(db, args.mission)
                if not selection:
                    return
                url, alias_filename = selection
                args.mission = args.mission or _guess_mission_from_url(url)
            elif not url.startswith("http"):
                # Treat as a filename — resolve mission mk_dir_url
                mk_dir_url, mission_name = _resolve_mission_mk_dir(
                    db, args.mission,
                )
                if not mk_dir_url:
                    return
                filename = url
                live_url = mk_dir_url + filename
                archive_url = mk_dir_url + "former_versions/" + filename
                if _head_ok(live_url):
                    url = live_url
                elif _head_ok(archive_url):
                    console.print(
                        f"[dim]Not in mk/, using archived copy: "
                        f"former_versions/{filename}[/dim]"
                    )
                    url = archive_url
                else:
                    print(
                        f"Metakernel '{filename}' not found.\n"
                        f"  Tried:\n"
                        f"    {live_url}\n"
                        f"    {archive_url}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                args.mission = args.mission or mission_name
            db.get_metakernel(
                url,
                download_dir=args.download_dir,
                mission=args.mission,
                yes=args.yes,
                force=args.force,
                alias_filename=alias_filename,
            )

        elif args.command == "update":
            metakernel = _require_metakernel(args.metakernel, db, args.mission)
            if metakernel is None:
                return
            try:
                db.update_metakernel(
                    metakernel,
                    mission=args.mission,
                    download_dir=args.download_dir,
                    yes=args.yes,
                    force=args.force,
                )
            except MetakernelUnreachableError as e:
                from rich.panel import Panel as _Panel
                console.print(_Panel(
                    f"[red]The remote metakernel is no longer available.[/red]\n\n"
                    f"  URL:      {e.url}\n"
                    f"  Status:   HTTP {e.status}\n"
                    f"  Filename: {e.filename}\n\n"
                    f"NAIF (and ESA) routinely rotate old versioned snapshots "
                    f"into a [italic]former_versions/[/italic] subdirectory, "
                    f"which makes the original URL 404.\n\n"
                    f"To clean up the stale registry row, run:\n"
                    f"  [bold]spice-kernel-db prune --metakernels[/bold]\n"
                    f"or remove just this one:\n"
                    f"  [bold]spice-kernel-db mk --remove {e.filename}[/bold]",
                    title="Metakernel unreachable",
                    border_style="red",
                ))
                sys.exit(2)
            except LookupError as e:
                print(str(e), file=sys.stderr)
                sys.exit(1)

        elif args.command == "browse":
            def _maybe_archive(u: str) -> str:
                return u.rstrip("/") + "/former_versions/" if args.archived else u

            if args.url:
                url = args.url
                # If not a URL, treat as a mission name
                if not url.startswith("http"):
                    m = db.get_mission(url)
                    if not m:
                        print(
                            f"No configured mission '{url}'.\n"
                            f"Use 'spice-kernel-db mission add' to configure it.",
                            file=sys.stderr,
                        )
                        return
                    url = m["mk_dir_url"]
                    args.mission = args.mission or m["name"]
                url = _maybe_archive(url)
                try:
                    db.browse_remote_metakernels(
                        url,
                        mission=args.mission,
                        show_versioned=args.show_versioned,
                        sort_by=args.sort_by,
                        filter=args.filter,
                        archived=args.archived,
                    )
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        print(
                            f"No metakernel directory found at:\n  {url}\n\n"
                            f"This mission may not publish metakernels at this location.\n"
                            f"Use 'spice-kernel-db mission remove' and re-add with the correct URL.",
                            file=sys.stderr,
                        )
                    else:
                        raise
                except urllib.error.URLError as e:
                    print(
                        f"Could not connect to server:\n  {url}\n  {e.reason}",
                        file=sys.stderr,
                    )
            else:
                # No args — pick a mission to browse interactively
                missions = db.list_missions()
                if not missions:
                    print("No configured missions.")
                    print("Use 'spice-kernel-db mission add' to set one up.")
                    return
                if len(missions) == 1:
                    m = missions[0]
                    console.print(
                        f"Browsing [bold]{m['name']}[/bold]...\n"
                    )
                    db.browse_remote_metakernels(
                        _maybe_archive(m["mk_dir_url"]),
                        mission=m["name"],
                        show_versioned=args.show_versioned,
                        sort_by=args.sort_by,
                        filter=args.filter,
                        archived=args.archived,
                    )
                else:
                    table = Table(title="Select a mission to browse")
                    table.add_column("#", justify="right", style="bold")
                    table.add_column("Mission")
                    table.add_column("mk/ URL")
                    for i, m in enumerate(missions, 1):
                        table.add_row(str(i), m["name"], m["mk_dir_url"])
                    console.print(table)
                    try:
                        raw = input(
                            f"\nSelect mission [1-{len(missions)}]: "
                        ).strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        return
                    try:
                        idx = int(raw)
                        if 1 <= idx <= len(missions):
                            m = missions[idx - 1]
                            console.print(
                                f"\nBrowsing [bold]{m['name']}[/bold]...\n"
                            )
                            db.browse_remote_metakernels(
                                _maybe_archive(m["mk_dir_url"]),
                                mission=m["name"],
                                show_versioned=args.show_versioned,
                                sort_by=args.sort_by,
                                filter=args.filter,
                                archived=args.archived,
                            )
                            return
                    except ValueError:
                        pass
                    console.print(
                        "[red]Invalid selection.[/red]", stderr=True,
                    )

        elif args.command == "mission":
            _handle_mission(db, args)

    finally:
        db.close()


def _render_verify_table(result: dict) -> None:
    """Render a verify_metakernel result as a rich table + summary panel."""
    from rich.panel import Panel
    from rich.table import Table

    mk = result["mk_path"]
    deep = result["deep"]
    total = result["ok"] + result["fail"]

    style_for = {
        "OK":              "green",
        "DANGLING":        "red",
        "NOT_FOUND":       "red",
        "NOT_FILE":        "red",
        "HASH_MISMATCH":   "red",
        "TRAVERSAL":       "red",
        "AMBIGUOUS":       "red",
        "BAD_PATH_VALUE":  "red",
        "SIZE_MISMATCH":   "red",
        "UNREGISTERED":    "yellow",
    }

    table = Table(title=f"verify {mk} ({'deep' if deep else 'quick'})")
    table.add_column("Entry", overflow="fold")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    for e in result["entries"]:
        status = e["status"]
        color = style_for.get(status, "white")
        table.add_row(
            e["raw"],
            f"[{color}]{status}[/{color}]",
            e["detail"],
        )
    console.print(table)

    if result["fatal"]:
        title_style = "red"
        verdict = "FAIL (fatal issues)"
    elif result["fail"]:
        title_style = "yellow"
        verdict = "WARN (non-fatal issues)"
    else:
        title_style = "green"
        verdict = "OK"
    console.print(Panel(
        f"OK: [green]{result['ok']}[/green] / "
        f"Issues: [red]{result['fail']}[/red] / "
        f"Total: [bold]{total}[/bold]",
        title=f"[{title_style}]{verdict}[/{title_style}]",
    ))


def _show_default_summary():
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

    db = KernelDB(db_path, read_only=True)
    try:
        rows = db.con.execute("""
            SELECT r.mk_path, r.filename, r.mission, r.acquired_at
            FROM metakernel_registry r
            ORDER BY r.mission, r.filename
        """).fetchall()
    finally:
        db.close()

    if rows:
        table = Table(title="Local metakernels")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Filename")
        table.add_column("Mission")
        table.add_column("Acquired")
        for i, (mk_path, filename, mission, acquired) in enumerate(rows, 1):
            acq_str = str(acquired)[:16] if acquired else ""
            table.add_row(str(i), filename, mission, acq_str)
        console.print(table)
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


def _list_kernels(metakernel: str, kernel_type: str | None = None):
    """Parse a metakernel and list its kernels in a rich table."""
    from spice_kernel_db import parse_metakernel

    parsed = parse_metakernel(metakernel)

    # Classify kernels by directory (ck/, spk/, fk/, etc.)
    kernels = parsed.kernels
    if kernel_type:
        kt = kernel_type.lower()
        kernels = [k for k in kernels if f"/{kt}/" in k.lower()]

    # Count by type
    type_counts: dict[str, int] = {}
    for k in parsed.kernels:
        parts = k.replace("\\", "/").split("/")
        for part in parts:
            if part in ("ck", "spk", "fk", "ik", "pck", "lsk", "sclk", "dsk"):
                type_counts[part] = type_counts.get(part, 0) + 1
                break

    console.print(f"\n[bold]Metakernel:[/bold] {metakernel}")
    console.print(f"[bold]Kernels:[/bold] {len(parsed.kernels)} total")
    if type_counts:
        summary = ", ".join(f"{v} {k}" for k, v in sorted(type_counts.items()))
        console.print(f"[dim]({summary})[/dim]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Type", style="bold")
    table.add_column("Kernel")

    for i, k in enumerate(kernels, 1):
        # Extract type from path
        parts = k.replace("\\", "/").split("/")
        ktype = ""
        for part in parts:
            if part in ("ck", "spk", "fk", "ik", "pck", "lsk", "sclk", "dsk"):
                ktype = part
                break
        # Show just the filename
        filename = parts[-1] if parts else k
        table.add_row(str(i), ktype, filename)

    console.print(table)


def _print_coverage_table(results, body_id: int, metakernel: str):
    """Print a rich table summarising body coverage results."""
    from rich.panel import Panel

    spk_count = sum(1 for r in results if r.kernel_type == "spk")
    cov_count = sum(1 for r in results if r.body_found)

    console.print(Panel(
        f"Body ID:      [bold]{body_id}[/bold]\n"
        f"Metakernel:   {metakernel}\n"
        f"SPK files:    {spk_count}\n"
        f"With coverage: {cov_count}",
        title="Coverage summary",
    ))

    table = Table(title="Kernel coverage")
    table.add_column("Kernel", no_wrap=True)
    table.add_column("Type")
    table.add_column("Coverage")
    table.add_column("Start (UTC)")
    table.add_column("End (UTC)")

    for r in results:
        if r.kernel_type != "spk":
            table.add_row(r.filename, r.kernel_type, "N/A", "", "")
            continue
        if r.error:
            # Truncate long SPICE tracebacks to the key message
            err_msg = r.error
            if len(err_msg) > 60:
                # Extract the short SPICE error if present
                for marker in ("SPICE(", "Input file"):
                    idx = err_msg.find(marker)
                    if idx >= 0:
                        err_msg = err_msg[idx:idx + 80].split("\n")[0]
                        break
                else:
                    err_msg = err_msg[:60] + "…"
            table.add_row(
                r.filename, r.kernel_type,
                f"[red]{err_msg}[/red]", "", "",
            )
            continue
        if not r.body_found:
            table.add_row(
                r.filename, r.kernel_type,
                "[dim]not found[/dim]", "", "",
            )
            continue
        for i, iv in enumerate(r.intervals):
            label = r.filename if i == 0 else ""
            type_label = r.kernel_type if i == 0 else ""
            cov_str = "[green]yes[/green]" if i == 0 else "[green]gap[/green]"
            table.add_row(
                label, type_label, cov_str,
                iv.utc_start or f"{iv.et_start:.3f} ET",
                iv.utc_end or f"{iv.et_end:.3f} ET",
            )

    console.print(table)


def _require_metakernel(
    metakernel: str | None, db: KernelDB, mission: str | None,
) -> str | None:
    """Return *metakernel* if given, otherwise prompt the user to pick one."""
    if metakernel is not None:
        return metakernel
    return _select_local_metakernel(db, mission)


def _resolve_body_interactive(name_or_id: str) -> int | None:
    """Resolve a body name/ID string to a single NAIF ID, prompting if ambiguous.

    Returns the integer body ID, or None on failure / user cancellation.
    """
    from spice_kernel_db.coverage import resolve_body_id

    candidates = resolve_body_id(name_or_id)
    if not candidates:
        print(
            f"Unknown body: '{name_or_id}'. "
            f"Use a NAIF ID (e.g. 399) or a known name (e.g. Earth).",
            file=sys.stderr,
        )
        return None

    if len(candidates) == 1:
        return candidates[0][0]

    # Disambiguation prompt
    console.print(f"\n'{name_or_id}' matches multiple NAIF bodies:\n")
    for i, (bid, desc) in enumerate(candidates, 1):
        console.print(f"  [{i}] {desc}  (NAIF ID {bid})")
    console.print()
    try:
        raw = input(f"Select [1-{len(candidates)}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    try:
        idx = int(raw)
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1][0]
    except ValueError:
        pass
    print("Invalid selection.", file=sys.stderr)
    return None


def _select_local_metakernel(
    db: KernelDB, mission: str | None,
) -> str | None:
    """List locally registered metakernels and let the user pick one.

    Returns the mk_path of the selected metakernel, or None on failure.
    Prints a hint about ``browse`` and ``get`` when no metakernels are found
    or when the user wants to fetch more.
    """
    rows = db.con.execute("""
        SELECT r.mk_path, r.filename, r.mission, r.acquired_at
        FROM metakernel_registry r
        {}
        ORDER BY r.mission, r.filename
    """.format(
        "WHERE LOWER(r.mission) LIKE LOWER(?) || '%'" if mission else "",
    ), [mission] if mission else []).fetchall()

    if not rows:
        mission_hint = f" for mission '{mission}'" if mission else ""
        console.print(
            f"[yellow]No locally acquired metakernels{mission_hint}.[/yellow]\n\n"
            f"[dim]To browse and download metakernels:\n"
            f"  spice-kernel-db browse [MISSION]\n"
            f"  spice-kernel-db get [METAKERNEL][/dim]\n",
            stderr=True,
        )
        return None

    table = Table(title="Local metakernels")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Filename")
    table.add_column("Mission")
    table.add_column("Acquired")
    for i, (mk_path, filename, mis, acquired) in enumerate(rows, 1):
        acq_str = str(acquired)[:16] if acquired else ""
        table.add_row(str(i), filename, mis, acq_str)
    console.print(table)

    console.print(
        "\n[dim]To fetch more metakernels: "
        "spice-kernel-db browse [MISSION][/dim]\n"
    )

    try:
        raw = input(f"Select metakernel [1-{len(rows)}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    try:
        idx = int(raw)
        if 1 <= idx <= len(rows):
            return rows[idx - 1][0]  # mk_path
    except ValueError:
        pass
    console.print("[red]Invalid selection.[/red]", stderr=True)
    return None


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

        # Look for .spice-server marker in the mission directory
        # Check both <root>/<mission>/ and <root>/<mission>/kernels/
        marker = None
        for candidate in [scan_root / name, scan_root / name / "kernels"]:
            marker = _read_spice_server_marker(candidate)
            if marker:
                break

        if marker:
            mk_dir_url = marker["mk_dir_url"]
            server_url = marker["server_url"]
            server_label = "custom"
            for label, url in SPICE_SERVERS.items():
                if server_url == url:
                    server_label = label
                    break
            db.add_mission(name, server_url, mk_dir_url)
            print(f"  Configured {name} from .spice-server ({server_label}).")
            continue

        # No marker — prompt user
        print(f"\n  {name} is not yet configured for remote access.")
        for i, sname in enumerate(server_names, 1):
            print(f"    [{i}] {sname}  ({SPICE_SERVERS[sname]})")
        print(f"    [{len(server_names) + 1}] Skip")
        while True:
            choice = input(f"  Select [1-{len(server_names) + 1}]: ").strip()
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


def _resolve_mission_mk_dir(
    db: KernelDB, mission_name: str | None,
) -> tuple[str | None, str | None]:
    """Resolve a mission name to an mk/ directory URL.

    Looks up the missions table. Returns (mk_dir_url, mission_name)
    or (None, None) on failure.
    """
    if mission_name:
        m = db.get_mission(mission_name)
        if m:
            return m["mk_dir_url"], m["name"]

    # No mission specified — check if exactly one is configured
    missions = db.list_missions()
    if len(missions) == 1:
        m = missions[0]
        return m["mk_dir_url"], m["name"]

    if not missions:
        console.print(
            "[yellow]No missions configured.[/yellow]\n"
            "[dim]Use 'spice-kernel-db mission add' first.[/dim]",
            stderr=True,
        )
        return None, None

    # Multiple missions — let user pick interactively
    table = Table(title="Select a mission")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Mission")
    table.add_column("mk/ URL")
    for i, m in enumerate(missions, 1):
        table.add_row(str(i), m["name"], m["mk_dir_url"])
    console.print(table)

    try:
        raw = input(f"\nSelect mission [1-{len(missions)}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None, None
    try:
        idx = int(raw)
        if 1 <= idx <= len(missions):
            m = missions[idx - 1]
            return m["mk_dir_url"], m["name"]
    except ValueError:
        pass
    console.print("[red]Invalid selection.[/red]", stderr=True)
    return None, None


def _guess_mission_from_url(url: str) -> str | None:
    """Extract mission name from a metakernel URL."""
    from spice_kernel_db.hashing import guess_mission
    return guess_mission(url)


def _interactive_pick_local_metakernel(
    db: KernelDB, mission_filter: str | None,
) -> str | None:
    """Show a picker over locally tracked metakernels and return the chosen path.

    Reads ``metakernel_registry`` (optionally filtered by mission with
    case-insensitive prefix matching) and prompts the user to pick one.
    Returns the ``mk_path`` of the selection, or ``None`` if there are
    no entries, the file no longer exists on disk, or the user cancels.
    """
    if mission_filter:
        rows = db.con.execute(
            "SELECT mk_path, filename, mission FROM metakernel_registry "
            "WHERE LOWER(mission) LIKE LOWER(?) || '%' "
            "ORDER BY mission, filename",
            [mission_filter],
        ).fetchall()
    else:
        rows = db.con.execute(
            "SELECT mk_path, filename, mission FROM metakernel_registry "
            "ORDER BY mission, filename"
        ).fetchall()

    if not rows:
        msg = "No tracked metakernels"
        if mission_filter:
            msg += f" for mission matching '{mission_filter}'"
        msg += ". Use 'spice-kernel-db get' to acquire one."
        console.print(f"[yellow]{msg}[/yellow]", stderr=True)
        return None

    table = Table(title="Tracked metakernels")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Mission")
    table.add_column("Metakernel")
    for i, (_, filename, mis) in enumerate(rows, 1):
        table.add_row(str(i), mis, filename)
    console.print(table)

    try:
        raw = input(f"\nSelect metakernel [1-{len(rows)}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    try:
        idx = int(raw)
        if 1 <= idx <= len(rows):
            mk_path = rows[idx - 1][0]
            if not Path(mk_path).is_file():
                console.print(
                    f"[red]File missing on disk: {mk_path}[/red]",
                    stderr=True,
                )
                return None
            return mk_path
    except ValueError:
        pass
    console.print("[red]Invalid selection.[/red]", stderr=True)
    return None


def _interactive_select_metakernel(
    db: KernelDB, mission_name: str | None,
) -> tuple[str, str | None] | None:
    """Fetch remote metakernels and let the user pick one interactively.

    Returns ``(url, alias_filename)`` for the selected metakernel, or
    ``None`` on failure. ``alias_filename`` is the version-stripped base
    name (e.g. ``juice_crema_5_2.tm``) when the picked row groups one or
    more versioned snapshots, otherwise ``None``.
    """
    mk_dir_url, resolved_name = _resolve_mission_mk_dir(db, mission_name)
    if not mk_dir_url:
        return None

    console.print(f"Fetching metakernels from [bold]{mk_dir_url}[/bold] ...")
    try:
        entries = list_remote_metakernels(mk_dir_url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            console.print(
                f"[red]No metakernel directory found at:[/red]\n  {mk_dir_url}",
                stderr=True,
            )
        else:
            raise
        return None

    if not entries:
        console.print("[red]No metakernels found.[/red]", stderr=True)
        return None

    # Group by base_name, show latest version per group
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for entry in entries:
        groups[entry.base_name].append(entry)

    # Check which are locally acquired
    local_rows = db.con.execute(
        "SELECT filename FROM metakernel_registry WHERE LOWER(mission) LIKE LOWER(?) || '%'",
        [resolved_name or ""],
    ).fetchall()
    local_filenames = {r[0] for r in local_rows}

    # Build selection list — one row per base_name, link to latest version
    # (url, base_name, latest_filename, is_local)
    choices: list[tuple[str, str, str, bool]] = []
    for base_name in sorted(groups):
        group = groups[base_name]
        latest = max(group, key=lambda e: e.filename)
        is_local = any(e.filename in local_filenames for e in group)
        choices.append((latest.url, base_name, latest.filename, is_local))

    table = Table(title=f"Available metakernels — {resolved_name or 'remote'}")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Metakernel")
    table.add_column("Versions", justify="right")
    table.add_column("Latest")
    table.add_column("Local", justify="center")
    for i, (base_name, group) in enumerate(sorted(groups.items()), 1):
        latest = max(group, key=lambda e: e.filename)
        is_local = any(e.filename in local_filenames for e in group)
        n_ver = str(len(group)) if len(group) > 1 else ""
        local_col = "[green]\u2713[/green]" if is_local else ""
        table.add_row(str(i), base_name, n_ver, latest.date, local_col)
    console.print(table)

    try:
        raw = input(f"\nSelect metakernel [1-{len(choices)}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    try:
        idx = int(raw)
        if 1 <= idx <= len(choices):
            url, base_name, latest_filename, _ = choices[idx - 1]
            alias = base_name if base_name != latest_filename else None
            return url, alias
    except ValueError:
        pass
    console.print("[red]Invalid selection.[/red]", stderr=True)
    return None


def _handle_mission(db: KernelDB, args):
    """Handle the 'mission' subcommand group."""
    if args.mission_command == "list":
        missions = db.list_missions()
        if not missions:
            print("No configured missions.")
            print("Use 'spice-kernel-db mission add' to set one up.")
            return
        table = Table(title="Configured missions")
        table.add_column("Mission")
        table.add_column("Server")
        table.add_column("Dedup")
        table.add_column("mk/ URL")
        for m in missions:
            dedup_str = "yes" if m["dedup"] else "no"
            table.add_row(
                m["name"], m["server_label"], dedup_str, m["mk_dir_url"],
            )
        console.print(table)

    elif args.mission_command == "remove":
        if db.remove_mission(args.name):
            print(f"Mission '{args.name}' removed.")
        else:
            print(f"Mission '{args.name}' not found.", file=sys.stderr)

    elif args.mission_command == "add":
        if args.mk_dir_url or args.name:
            _mission_add_noninteractive(db, args)
        else:
            _mission_add_interactive(db, args)

    else:
        print("Usage: spice-kernel-db mission {add,list,remove}")


def _choose_server() -> tuple[str, str]:
    """Prompt for a SPICE archive server. Returns ``(label, server_url)``."""
    server_names = list(SPICE_SERVERS.keys())
    print("\nAvailable SPICE archive servers:\n")
    for i, name in enumerate(server_names, 1):
        print(f"  [{i}] {name}  ({SPICE_SERVERS[name]})")
    print()
    while True:
        choice = input(f"Select server [1-{len(server_names)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(server_names):
            server_label = server_names[int(choice) - 1]
            return server_label, SPICE_SERVERS[server_label]
        print("Invalid choice.")


def _match_ci(choice: str, names: list[str]) -> str | None:
    """Return the first case-insensitive match for *choice* in *names*, or None."""
    return next((n for n in names if n.lower() == choice.lower()), None)


def _select_mk_url_for_unsupported(server_url: str, mission_name: str) -> str | None:
    """Consult the curated registry for a mission lacking the default mk/.

    ``check_mk_availability`` already proved the default ``kernels/mk/`` path
    fails for this mission, so we probe only registry-supplied candidates.
    Prompts the user to pick when multiple hits are found.
    """
    reg_candidates = registry.registry_candidates(mission_name, server_url)
    if not reg_candidates:
        return None
    print(f"\nProbing registered alternate metakernel locations for {mission_name}...")
    hits = discover_mk_url(
        server_url, mission_name,
        registry_candidates=reg_candidates, include_default=False,
    )
    if not hits:
        return None
    if len(hits) == 1:
        print(f"Found: {hits[0]}")
        return hits[0]
    print(f"\nFound {len(hits)} alternate metakernel locations:\n")
    for i, url in enumerate(hits, 1):
        print(f"  [{i}] {url}")
    while True:
        choice = input(f"Select [1-{len(hits)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(hits):
            return hits[int(choice) - 1]
        print("Invalid choice.")


def _choose_mission(server_url: str, server_label: str) -> tuple[str, str] | None:
    """Prompt for a mission. Returns ``(mission_name, mk_dir_url)`` or None on abort."""
    print(f"\nFetching missions from {server_label}...")
    all_missions = list_remote_missions(server_url)
    if not all_missions:
        print("No missions found.", file=sys.stderr)
        return None

    mk_status = check_mk_availability(server_url, all_missions)
    supported = [m for m in all_missions if mk_status.get(m)]
    unsupported = [m for m in all_missions if not mk_status.get(m)]

    if not supported and not unsupported:
        print("No missions with metakernel directories found.", file=sys.stderr)
        return None

    if supported:
        print(f"\nAvailable missions ({len(supported)} with metakernels):\n")
        for i, name in enumerate(supported, 1):
            print(f"  [{i:>3}] {name}")

    if unsupported:
        print(
            f"\nNo default mk/ directory ({len(unsupported)} missions — type "
            f"the name to consult the curated registry):"
        )
        print(f"  {', '.join(unsupported)}")

    print()
    while True:
        choice = input(
            f"Select mission [1-{len(supported)}] or type name: "
        ).strip()
        if choice.isdigit() and 1 <= int(choice) <= len(supported):
            name = supported[int(choice) - 1]
            return name, f"{server_url}{name}/kernels/mk/"
        if (name := _match_ci(choice, supported)):
            return name, f"{server_url}{name}/kernels/mk/"
        if (name := _match_ci(choice, unsupported)):
            url = _select_mk_url_for_unsupported(server_url, name)
            if url is None:
                print(
                    f"\nNo metakernel directory found for {name}. "
                    f"See docs/troubleshooting.qmd for how to contribute "
                    f"alternate locations to mission_registry.toml.",
                    file=sys.stderr,
                )
                return None
            return name, url
        print("Invalid choice.")


def _choose_dedup(mission_name: str) -> bool:
    """Prompt for deduplication preference. Returns True when enabled."""
    answer = input(
        f"\nEnable deduplication for {mission_name}? [Y/n]: "
    ).strip().lower()
    return answer not in ("n", "no")


def _maybe_offer_planetarypy(mission_name: str, *, force: bool = False) -> bool:
    """Offer planetarypy delegation when applicable. Returns True if user opted in."""
    if not registry.is_planetarypy_managed(mission_name):
        return False
    if not planetarypy_bridge.is_available():
        if force:
            print(
                f"--use-planetarypy requested but planetarypy is not installed. "
                f"Install with: pip install 'spice-kernel-db[planetarypy]'",
                file=sys.stderr,
            )
        return False
    if force:
        opted_in = True
    else:
        answer = input(
            f"\n{mission_name} is registered as planetarypy-managed. "
            f"Delegate to planetarypy? [y/N]: "
        ).strip().lower()
        opted_in = answer in ("y", "yes")
    if not opted_in:
        return False
    print(
        f"planetarypy delegation requested but full integration is not yet "
        f"implemented (tracked at {planetarypy_bridge.tracking_issue()}). "
        f"Falling back to normal discovery.",
        file=sys.stderr,
    )
    return False


def _save_mission(
    db: KernelDB, mission_name: str, server_label: str, server_url: str,
    mk_dir_url: str, dedup: bool,
) -> None:
    """Persist the mission row and print a confirmation block."""
    db.add_mission(mission_name, server_url, mk_dir_url, dedup)
    dedup_str = "enabled" if dedup else "disabled"
    print(f"\nMission '{mission_name}' ({server_label}) configured.")
    print(f"  Deduplication: {dedup_str}")
    print(f"  mk/ directory: {mk_dir_url}")
    print(f"\nNext steps:")
    print(f"  spice-kernel-db browse {mission_name}")
    print(f"  spice-kernel-db get <metakernel>.tm --mission {mission_name}")


def _mission_add_interactive(db: KernelDB, args) -> None:
    """Interactive mission setup: choose server → pick mission → configure."""
    server_label, server_url = _choose_server()
    chosen = _choose_mission(server_url, server_label)
    if chosen is None:
        return
    mission_name, mk_dir_url = chosen
    _maybe_offer_planetarypy(mission_name, force=bool(args.use_planetarypy))
    dedup = False if args.no_dedup else _choose_dedup(mission_name)
    _save_mission(db, mission_name, server_label, server_url, mk_dir_url, dedup)


def _infer_server_url(name: str, server_url_arg: str | None) -> str | None:
    """Infer the server URL from --server-url, or a unique match in SPICE_SERVERS.

    Returns None if nothing can be inferred (no --server-url and remote calls fail).
    """
    if server_url_arg:
        return server_url_arg if server_url_arg.endswith("/") else server_url_arg + "/"

    from concurrent.futures import ThreadPoolExecutor

    urls = list(SPICE_SERVERS.values())
    def _has_mission(url: str) -> bool:
        try:
            return any(m.lower() == name.lower() for m in list_remote_missions(url))
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        found = list(pool.map(_has_mission, urls))
    for url, ok in zip(urls, found):
        if ok:
            return url
    return None


def _mission_add_noninteractive(db: KernelDB, args) -> None:
    """Non-interactive mission add. Either --mk-dir-url, or discover from --server-url."""
    if not args.name:
        print(
            "mission add: --mk-dir-url requires a positional mission name.",
            file=sys.stderr,
        )
        sys.exit(2)

    mission_name = args.name
    server_url = _infer_server_url(mission_name, args.server_url)
    if server_url is None:
        print(
            f"mission add: cannot determine server URL for '{mission_name}'. "
            f"Pass --server-url explicitly.",
            file=sys.stderr,
        )
        sys.exit(2)
    server_label = server_label_for(server_url)

    _maybe_offer_planetarypy(mission_name, force=bool(args.use_planetarypy))

    if args.mk_dir_url:
        mk_dir_url = args.mk_dir_url
        if not mk_dir_url.endswith("/"):
            mk_dir_url += "/"
        if not probe_mk_candidates([mk_dir_url]):
            print(
                f"mission add: --mk-dir-url did not respond to HEAD: {mk_dir_url}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        reg_candidates = registry.registry_candidates(mission_name, server_url)
        hits = discover_mk_url(
            server_url, mission_name, registry_candidates=reg_candidates,
        )
        if not hits:
            print(
                f"mission add: no metakernel directory found for "
                f"'{mission_name}' on {server_url}. Re-run with --mk-dir-url, "
                f"or see docs/troubleshooting.qmd.",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(hits) > 1:
            print(
                f"mission add: multiple metakernel directories responded for "
                f"'{mission_name}'. Re-run with --mk-dir-url <one of>:",
                file=sys.stderr,
            )
            for url in hits:
                print(f"  {url}", file=sys.stderr)
            sys.exit(1)
        mk_dir_url = hits[0]

    dedup = not args.no_dedup
    _save_mission(db, mission_name, server_label, server_url, mk_dir_url, dedup)
