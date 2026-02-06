"""Command-line interface for spice-kernel-db."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spice_kernel_db.config import (
    ensure_config,
    setup_interactive,
    show_config,
)
from spice_kernel_db.db import KernelDB


def main(argv: list[str] | None = None):
    config = ensure_config()

    parser = argparse.ArgumentParser(
        prog="spice-kernel-db",
        description="SPICE kernel deduplication database and metakernel rewriter",
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

    # --- acquire ---
    p_acquire = sub.add_parser(
        "acquire",
        help="Download missing kernels for a remote metakernel",
    )
    p_acquire.add_argument(
        "url",
        help="URL to a remote .tm metakernel, or just a filename (e.g. juice_ops.tm)",
    )
    p_acquire.add_argument(
        "--download-dir", default=config.kernel_dir,
        help=f"Directory for downloaded kernels (default: {config.kernel_dir})",
    )
    p_acquire.add_argument("--mission", help="Override auto-detected mission name")
    p_acquire.add_argument(
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
            db.scan_directory(
                args.directory,
                mission=args.mission,
                verbose=args.verbose,
                archive_dir=archive_dir,
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

        elif args.command == "acquire":
            url = args.url
            if not url.startswith("http"):
                # Treat as a filename — resolve to full URL via known dirs
                known = db.list_known_mk_dirs(quiet=True)
                if args.mission:
                    known = [
                        d for d in known
                        if d["mission"].lower() == args.mission.lower()
                    ]
                if not known:
                    mission_hint = f" for mission '{args.mission}'" if args.mission else ""
                    print(
                        f"No known mk/ directory{mission_hint}. "
                        f"Use a full URL or browse a mission first.",
                        file=sys.stderr,
                    )
                    return
                if len(known) > 1:
                    missions = ", ".join(d["mission"] for d in known)
                    print(
                        f"Multiple missions known: {missions}. "
                        f"Use --mission to specify which one.",
                        file=sys.stderr,
                    )
                    return
                url = known[0]["mk_dir_url"] + url
                args.mission = args.mission or known[0]["mission"]
            db.acquire_metakernel(
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
                    known = db.list_known_mk_dirs(quiet=True)
                    match = [
                        d for d in known
                        if d["mission"].lower() == url.lower()
                    ]
                    if not match:
                        print(
                            f"No known mk/ directory for mission '{url}'.",
                            file=sys.stderr,
                        )
                        return
                    url = match[0]["mk_dir_url"]
                    args.mission = args.mission or match[0]["mission"]
                db.browse_remote_metakernels(
                    url,
                    mission=args.mission,
                    show_versioned=args.show_versioned,
                )
            else:
                db.list_known_mk_dirs()

    finally:
        db.close()
