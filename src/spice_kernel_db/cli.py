"""Command-line interface for spice-kernel-db."""

from __future__ import annotations

import argparse
import sys

from spice_kernel_db.config import ensure_config
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

    # --- acquire ---
    p_acquire = sub.add_parser(
        "acquire",
        help="Download missing kernels for a remote metakernel",
    )
    p_acquire.add_argument("url", help="URL to a remote .tm metakernel")
    p_acquire.add_argument(
        "--download-dir", default=config.kernel_dir,
        help=f"Directory for downloaded kernels (default: {config.kernel_dir})",
    )
    p_acquire.add_argument("--mission", help="Override auto-detected mission name")
    p_acquire.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    db = KernelDB(args.db)
    try:
        if args.command == "scan":
            db.scan_directory(
                args.directory, mission=args.mission, verbose=args.verbose,
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

        elif args.command == "acquire":
            db.acquire_metakernel(
                args.url,
                download_dir=args.download_dir,
                mission=args.mission,
                yes=args.yes,
            )

    finally:
        db.close()
