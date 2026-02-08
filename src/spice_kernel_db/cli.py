"""Command-line interface for spice-kernel-db."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.table import Table

from spice_kernel_db.config import (
    ensure_config,
    setup_interactive,
    show_config,
)
from spice_kernel_db.db import KernelDB
from spice_kernel_db.remote import SPICE_SERVERS, list_remote_missions

console = Console()


def main(argv: list[str] | None = None):
    config = ensure_config()

    parser = argparse.ArgumentParser(
        prog="spice-kernel-db",
        description="Browse, get, and manage SPICE kernels and metakernels",
    )
    parser.add_argument(
        "--db", default=config.db_path,
        help=f"Path to DuckDB database file (default: {config.db_path})",
    )
    sub = parser.add_subparsers(dest="command")

    # --- scan ---
    p_scan = sub.add_parser("scan", help="Scan a directory for kernel files")
    p_scan.add_argument("directory", help="Root directory to scan recursively")
    p_scan.add_argument(
        "--mission", help="Override auto-detected mission name"
    )
    p_scan.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print each registered file",
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
        "check", help="Check which kernels in a metakernel are available locally"
    )
    p_check.add_argument("metakernel", help="Path to .tm file")
    p_check.add_argument("--mission", help="Override mission name")

    # --- rewrite ---
    p_rewrite = sub.add_parser(
        "rewrite",
        help="Rewrite a metakernel with a local symlink tree",
    )
    p_rewrite.add_argument("metakernel", help="Path to original .tm file")
    p_rewrite.add_argument(
        "-o", "--output", required=True,
        help="Output .tm path",
    )
    p_rewrite.add_argument(
        "--link-root",
        help="Root for symlink tree (default: kernels/ next to output)",
    )
    p_rewrite.add_argument("--mission", help="Override mission name")

    # --- dedup ---
    p_dedup = sub.add_parser(
        "dedup", help="Deduplicate kernel files using symlinks"
    )
    p_dedup.add_argument(
        "--execute", action="store_true",
        help="Actually replace files (default: dry run)",
    )

    # --- resolve ---
    p_resolve = sub.add_parser(
        "resolve", help="Find local path for a kernel filename"
    )
    p_resolve.add_argument("filename", help="Kernel filename to resolve")
    p_resolve.add_argument("--mission", help="Preferred mission")

    # --- metakernels ---
    p_mk = sub.add_parser(
        "metakernels", help="List tracked metakernels or show details",
    )
    p_mk.add_argument(
        "name", nargs="?",
        help="Show info for a specific metakernel (by filename)",
    )
    p_mk.add_argument("--mission", help="Filter by mission name")

    # --- get ---
    p_get = sub.add_parser(
        "get",
        help="Download missing kernels for a remote metakernel",
    )
    p_get.add_argument(
        "url",
        help="URL to a remote .tm metakernel, or just a filename (e.g. juice_ops.tm)",
    )
    p_get.add_argument(
        "--download-dir", default=config.kernel_dir,
        help=f"Directory for downloaded kernels (default: {config.kernel_dir})",
    )
    p_get.add_argument("--mission", help="Override auto-detected mission name")
    p_get.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt",
    )

    # --- browse ---
    p_browse = sub.add_parser(
        "browse",
        help="Browse remote metakernels in a NAIF mk/ directory",
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

    # --- mission ---
    p_mission = sub.add_parser(
        "mission",
        help="Manage configured missions",
    )
    mission_sub = p_mission.add_subparsers(dest="mission_command")
    mission_sub.add_parser("add", help="Add a new mission (interactive)")
    mission_sub.add_parser("list", help="List configured missions")
    p_mission_rm = mission_sub.add_parser("remove", help="Remove a mission")
    p_mission_rm.add_argument("name", help="Mission name to remove")

    # --- config ---
    p_config = sub.add_parser("config", help="Show or update configuration")
    p_config.add_argument(
        "--setup", action="store_true",
        help="Re-run interactive setup",
    )

    # --- reset ---
    p_reset = sub.add_parser("reset", help="Delete the database and start fresh")
    p_reset.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    # --- Commands that don't need a DB connection ---
    if args.command == "config":
        if args.setup:
            setup_interactive()
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
    db = KernelDB(args.db)
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

        elif args.command == "check":
            db.check_metakernel(args.metakernel, mission=args.mission)

        elif args.command == "rewrite":
            db.rewrite_metakernel(
                args.metakernel,
                args.output,
                mission=args.mission,
                link_root=args.link_root,
            )

        elif args.command == "dedup":
            db.deduplicate_with_symlinks(dry_run=not args.execute)

        elif args.command == "resolve":
            path, warnings = db.resolve_kernel(
                args.filename, preferred_mission=args.mission,
            )
            if path:
                print(path)
            else:
                print(f"Not found: {args.filename}", file=sys.stderr)
            for w in warnings:
                print(f"  ⚠ {w}", file=sys.stderr)

        elif args.command == "metakernels":
            if args.name:
                db.info_metakernel(args.name)
            else:
                db.list_metakernels(mission=args.mission)

        elif args.command == "get":
            url = args.url
            if not url.startswith("http"):
                # Treat as a filename — resolve mission mk_dir_url
                mk_dir_url, mission_name = _resolve_mission_mk_dir(
                    db, args.mission,
                )
                if not mk_dir_url:
                    return
                url = mk_dir_url + url
                args.mission = args.mission or mission_name
            db.get_metakernel(
                url,
                download_dir=args.download_dir,
                mission=args.mission,
                yes=args.yes,
            )

        elif args.command == "browse":
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
                try:
                    db.browse_remote_metakernels(
                        url,
                        mission=args.mission,
                        show_versioned=args.show_versioned,
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
            else:
                # No args — list configured missions
                missions = db.list_missions()
                if not missions:
                    print("No configured missions.")
                    print("Use 'spice-kernel-db mission add' to set one up.")
                    return
                print("\nConfigured missions:\n")
                for m in missions:
                    print(f"  {m['name']}  {m['mk_dir_url']}")

        elif args.command == "mission":
            _handle_mission(db, args)

    finally:
        db.close()


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
        print(
            "No missions configured. "
            "Use 'spice-kernel-db mission add' first.",
            file=sys.stderr,
        )
    else:
        names = ", ".join(m["name"] for m in missions)
        print(
            f"Multiple missions configured: {names}. "
            f"Use --mission to specify which one.",
            file=sys.stderr,
        )
    return None, None


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
        _mission_add_interactive(db)

    else:
        print("Usage: spice-kernel-db mission {add,list,remove}")


def _mission_add_interactive(db: KernelDB):
    """Interactive mission setup: choose server → pick mission → configure."""
    # 1. Choose server
    server_names = list(SPICE_SERVERS.keys())
    print("\nAvailable SPICE archive servers:\n")
    for i, name in enumerate(server_names, 1):
        print(f"  [{i}] {name}  ({SPICE_SERVERS[name]})")
    print()
    while True:
        choice = input(f"Select server [1-{len(server_names)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(server_names):
            server_label = server_names[int(choice) - 1]
            break
        print("Invalid choice.")
    server_url = SPICE_SERVERS[server_label]

    # 2. List missions
    print(f"\nFetching missions from {server_label}...")
    missions = list_remote_missions(server_url)
    if not missions:
        print("No missions found.", file=sys.stderr)
        return

    print(f"\nAvailable missions ({len(missions)}):\n")
    for i, name in enumerate(missions, 1):
        print(f"  [{i:>3}] {name}")
    print()
    while True:
        choice = input(f"Select mission [1-{len(missions)}] or type name: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(missions):
            mission_name = missions[int(choice) - 1]
            break
        elif choice in missions:
            mission_name = choice
            break
        # Case-insensitive match
        matches = [m for m in missions if m.lower() == choice.lower()]
        if matches:
            mission_name = matches[0]
            break
        print("Invalid choice.")

    # 3. Compute mk/ directory URL and validate it exists
    mk_dir_url = f"{server_url}{mission_name}/kernels/mk/"
    try:
        req = urllib.request.Request(mk_dir_url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10):
            pass
    except (urllib.error.HTTPError, urllib.error.URLError):
        print(f"\n  Warning: {mk_dir_url} not found (404).")
        print(f"  This mission may not have metakernels on this server.")
        custom = input(
            "  Enter correct mk/ URL (or press Enter to save anyway): "
        ).strip()
        if custom:
            mk_dir_url = custom if custom.endswith("/") else custom + "/"

    # 4. Dedup preference
    dedup_answer = input(
        f"\nEnable deduplication for {mission_name}? [Y/n]: "
    ).strip().lower()
    dedup = dedup_answer not in ("n", "no")

    # 5. Store
    db.add_mission(mission_name, server_url, mk_dir_url, dedup)
    dedup_str = "enabled" if dedup else "disabled"
    print(f"\nMission '{mission_name}' ({server_label}) configured.")
    print(f"  Deduplication: {dedup_str}")
    print(f"  mk/ directory: {mk_dir_url}")
    print(f"\nNext steps:")
    print(f"  spice-kernel-db browse {mission_name}")
    print(f"  spice-kernel-db get <metakernel>.tm --mission {mission_name}")
