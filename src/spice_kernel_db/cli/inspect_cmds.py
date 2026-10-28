"""Read-only commands: list, check, verify, coverage, metakernels, resolve, aliases, stats, duplicates."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, NoReturn, Optional

import typer
from rich.panel import Panel
from rich.table import Table

from spice_kernel_db.db import KernelDB, MetakernelNotTracked, _format_size

from ._app import INSPECT, app, open_db, state
from .prompts import (
    _interactive_pick_deduplicated_kernel,
    _interactive_pick_local_metakernel,
    _require_metakernel,
    _resolve_body_interactive,
    console,
)

MetakernelArg = Annotated[
    Optional[str],
    typer.Argument(help="Path to .tm file (omit to select from local registry)"),
]


@app.command("list", rich_help_panel=INSPECT, epilog="See also: check, mk",
             help="List kernels contained in a metakernel")
def list_(
    ctx: typer.Context,
    metakernel: MetakernelArg = None,
    mission: Annotated[Optional[str], typer.Option(
        "--mission", help="Filter by mission name")] = None,
    kernel_type: Annotated[Optional[str], typer.Option(
        "--type", help="Filter by kernel type (ck, spk, fk, ik, pck, lsk, sclk, dsk)")] = None,
) -> None:
    with open_db(ctx, read_only=True) as db:
        mk = _require_metakernel(metakernel, db, mission)
        if mk is None:
            return
        _list_kernels(mk, db=db, kernel_type=kernel_type)


@app.command(rich_help_panel=INSPECT, epilog="See also: list, mk, resolve",
             help="Check which kernels in a metakernel are available locally")
def check(
    ctx: typer.Context,
    metakernel: MetakernelArg = None,
    full: Annotated[bool, typer.Option(
        "--full",
        help="Also compare each local kernel against the server (size). "
             "Without this, check only asks whether a file of each name exists")] = False,
    hash_: Annotated[bool, typer.Option(
        "--hash",
        help="With --full, also compare sha256 for kernels whose size matched. "
             "Downloads each one, so it is slow")] = False,
    spice: Annotated[bool, typer.Option(
        "--spice",
        help="With --full, furnsh each local kernel and report what SPICE "
             "cannot read (requires spiceypy)")] = False,
    mission: Annotated[Optional[str], typer.Option(
        "--mission", help="Filter by mission name")] = None,
) -> None:
    verbose = state(ctx).verbose
    with open_db(ctx, read_only=True) as db:
        mk = _require_metakernel(metakernel, db, mission)
        if mk is None:
            return
        try:
            result = db.check_metakernel(mk, mission=mission, verbose=verbose)
            if full:
                report = db.verify_against_server(
                    mk, mission=mission, check_hash=hash_, check_spice=spice,
                )
                _print_server_check(report, verbose=verbose)
                if report["bad"]:
                    sys.exit(1)
            elif hash_ or spice:
                print("check: --hash and --spice require --full", file=sys.stderr)
                sys.exit(2)
            if result["poisoned"]:
                sys.exit(1)
        except MetakernelNotTracked as exc:
            print(f"check: {exc}", file=sys.stderr)
            sys.exit(1)


@app.command(rich_help_panel=INSPECT, epilog="See also: check",
             help="Deep cross-check a metakernel against the database")
def verify(
    ctx: typer.Context,
    metakernel: MetakernelArg = None,
    deep: Annotated[bool, typer.Option(
        "--deep", help="Recompute sha256 of every kernel (slow; default: size-only)")] = False,
    strict: Annotated[bool, typer.Option(
        "--strict", help="Non-zero exit on any non-OK finding (default: only on FATAL)")] = False,
    json_: Annotated[bool, typer.Option(
        "--json", help="Emit machine-readable JSON instead of a table")] = False,
    mission: Annotated[Optional[str], typer.Option(
        "--mission", help="Filter by mission (for registry lookup)")] = None,
) -> None:
    with open_db(ctx) as db:
        mk = _require_metakernel(metakernel, db, mission)
        if mk is None:
            return
        result = db.verify_metakernel(mk, deep=deep)
        if json_:
            import json
            print(json.dumps(result, indent=2, default=str))
        else:
            _render_verify_table(result)
        # Non-zero on fatal (P0) issues by default; on any finding with --strict.
        if result["fatal"] or (strict and result["fail"] > 0):
            sys.exit(1)


@app.command(rich_help_panel=INSPECT, help="Check SPK body coverage in a metakernel")
def coverage(
    ctx: typer.Context,
    body: Annotated[str, typer.Argument(
        help="NAIF body ID (e.g. 399) or name (e.g. Earth, Mars)")],
    metakernel: Annotated[Optional[str], typer.Argument(
        help="Path to .tm file or registry filename (omit to select interactively)")] = None,
    mission: Annotated[Optional[str], typer.Option(
        "--mission", help="Preferred mission for kernel resolution")] = None,
) -> None:
    with open_db(ctx, read_only=True) as db:
        try:
            body_id = _resolve_body_interactive(body)
        except ImportError:
            _spiceypy_required()
        if body_id is None:
            return
        mk = _require_metakernel(metakernel, db, mission)
        if mk is None:
            return
        try:
            results = db.coverage_metakernel(mk, body_id, mission=mission)
        except ImportError:
            _spiceypy_required()
        _print_coverage_table(results, body_id, mk)


def _spiceypy_required() -> NoReturn:
    print(
        "SpiceyPy is required for coverage analysis.\n"
        "Install it with: pip install spice-kernel-db[spice]",
        file=sys.stderr,
    )
    sys.exit(1)


_MK_HELP = "List tracked metakernels or show details (alias: mk)"
_MK_EPILOG = "See also: check, list"


@app.command("metakernels", rich_help_panel=INSPECT, help=_MK_HELP, epilog=_MK_EPILOG)
@app.command("mk", hidden=True, help=_MK_HELP, epilog=_MK_EPILOG)
def metakernels(
    ctx: typer.Context,
    name: Annotated[Optional[str], typer.Argument(
        help="Show info for a specific metakernel (by filename)")] = None,
    mission: Annotated[Optional[str], typer.Option(
        "--mission", help="Filter by mission name (prefix match)")] = None,
    remove: Annotated[bool, typer.Option(
        "--remove",
        help="Remove the named metakernel from the registry (does not delete files)")] = False,
) -> None:
    # --remove is the one destructive variant of an otherwise read-only command.
    with open_db(ctx, read_only=not remove) as db:
        if remove:
            if not name:
                print("Usage: spice-kernel-db mk --remove <name>", file=sys.stderr)
                sys.exit(1)
            if db.remove_metakernel(name):
                print(f"Metakernel '{name}' removed from registry.")
            else:
                print(f"Metakernel '{name}' not found.", file=sys.stderr)
        elif name:
            db.info_metakernel(name)
        else:
            db.list_metakernels(mission=mission)


@app.command(rich_help_panel=INSPECT, epilog="See also: check, list",
             help="Find local path for a kernel filename")
def resolve(
    ctx: typer.Context,
    filename: Annotated[Optional[str], typer.Argument(
        help="Kernel filename to resolve "
             "(omit to pick a tracked metakernel interactively)")] = None,
    mission: Annotated[Optional[str], typer.Option(
        "--mission", help="Preferred mission for resolution")] = None,
    resolve_mk: Annotated[Optional[str], typer.Option(
        "--metakernel", help="Resolve all kernels in a .tm metakernel")] = None,
) -> None:
    with open_db(ctx, read_only=True) as db:
        if resolve_mk:
            # Batch mode — resolve all kernels in a metakernel
            from spice_kernel_db import parse_metakernel
            all_ok = True
            for kernel in parse_metakernel(resolve_mk).kernel_filenames():
                path, warnings = db.resolve_kernel(kernel, preferred_mission=mission)
                if path:
                    print(f"{kernel}\t{path}")
                else:
                    print(f"{kernel}\tNOT FOUND", file=sys.stderr)
                    all_ok = False
                for w in warnings:
                    print(f"  ⚠ {w}", file=sys.stderr)
            if not all_ok:
                sys.exit(1)
        elif filename:
            path, warnings = db.resolve_kernel(filename, preferred_mission=mission)
            if path:
                print(path)
            else:
                print(f"Not found: {filename}", file=sys.stderr)
                print(
                    "  Hint: the kernel may exist on disk but not in the database.\n"
                    "  Run 'spice-kernel-db scan <directory>' to re-index.",
                    file=sys.stderr,
                )
            for w in warnings:
                print(f"  ⚠ {w}", file=sys.stderr)
        else:
            picked = _interactive_pick_local_metakernel(db, mission)
            if picked is None:
                sys.exit(1)
            print(picked)


@app.command(rich_help_panel=INSPECT, epilog="See also: resolve, duplicates",
             help="Show every filename a deduplicated kernel is known under")
def aliases(
    ctx: typer.Context,
    name_or_hash: Annotated[Optional[str], typer.Argument(
        help="A kernel filename (canonical or alias) or a SHA-256 "
             "(full or >=6-char prefix). Omit to pick from deduplicated "
             "kernels interactively.")] = None,
) -> None:
    with open_db(ctx, read_only=True) as db:
        target = name_or_hash
        if target is None:
            target = _interactive_pick_deduplicated_kernel(db)
            if target is None:
                return
        info = db.aliases(target)
        if info is None:
            print(f"Not found: {target}", file=sys.stderr)
            print(
                "  Hint: give a known kernel filename (canonical or alias) "
                "or a SHA-256 prefix.",
                file=sys.stderr,
            )
            sys.exit(1)
        _print_aliases(info)


@app.command(rich_help_panel=INSPECT, help="Show database statistics")
def stats(ctx: typer.Context) -> None:
    with open_db(ctx, read_only=True) as db:
        db.stats()


@app.command(rich_help_panel=INSPECT, help="Report duplicate kernels")
def duplicates(ctx: typer.Context) -> None:
    with open_db(ctx, read_only=True) as db:
        db.report_duplicates()


# --- rendering -----------------------------------------------------------

def _print_aliases(info: dict) -> None:
    sz = _format_size(info["size_bytes"]) if info["size_bytes"] else "?"
    header = (
        f"[bold]{info['sha256'][:16]}…[/bold]  "
        f"({info['kernel_type'] or '?'}, {sz})"
    )
    if info["superseded"]:
        header += "  [yellow](superseded)[/yellow]"
    console.print(Panel(header, title=info["canonical"] or "kernel"))

    tbl = Table(title=f"Known under {len(info['aliases'])} name(s)")
    tbl.add_column("Filename")
    tbl.add_column("")
    for name in info["aliases"]:
        marker = "canonical" if name == info["canonical"] else "alias"
        style = "green" if marker == "canonical" else "dim"
        tbl.add_row(name, f"[{style}]{marker}[/{style}]")
    console.print(tbl)

    loc_tbl = Table(title=f"On disk ({len(info['locations'])} location(s))")
    # overflow="fold" wraps long paths instead of ellipsizing them, so
    # the distinguishing tail stays visible even in a narrow terminal.
    loc_tbl.add_column("Path", overflow="fold")
    loc_tbl.add_column("Kind")
    loc_tbl.add_column("Mission")
    for loc in info["locations"]:
        kind = _location_kind(loc["abs_path"])
        kind_disp = "[red]missing[/red]" if kind == "missing" else kind
        loc_tbl.add_row(loc["abs_path"], kind_disp, loc["mission"] or "")
    console.print(loc_tbl)


def _location_kind(path: str) -> str:
    """Describe an on-disk location: a real file, a symlink (→ target name),
    or missing. Used to annotate the `aliases` on-disk listing."""
    p = Path(path)
    if p.is_symlink():
        try:
            return f"→ {p.readlink().name}"
        except OSError:
            return "→ ?"
    return "real file" if p.exists() else "missing"


def _render_verify_table(result: dict) -> None:
    """Render a verify_metakernel result as a rich table + summary panel."""
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
        table.add_row(e["raw"], f"[{color}]{status}[/{color}]", e["detail"])
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


_KERNEL_TYPES = ("ck", "spk", "fk", "ik", "pck", "lsk", "sclk", "dsk")


def _kernel_type_of(path: str) -> str:
    """The kernel type named by a directory component of *path*, or ''."""
    for part in path.replace("\\", "/").split("/"):
        if part in _KERNEL_TYPES:
            return part
    return ""


def _list_kernels(metakernel: str, db: KernelDB | None = None,
                  kernel_type: str | None = None) -> None:
    """Parse a metakernel and list its kernels in a rich table.

    When ``db`` is given, kernels whose content is deduplicated are
    annotated with ``(+N aliases)``.
    """
    from spice_kernel_db import parse_metakernel

    parsed = parse_metakernel(metakernel)

    kernels = parsed.kernels
    if kernel_type:
        kt = kernel_type.lower()
        kernels = [k for k in kernels if f"/{kt}/" in k.lower()]

    type_counts: dict[str, int] = {}
    for k in parsed.kernels:
        ktype = _kernel_type_of(k)
        if ktype:
            type_counts[ktype] = type_counts.get(ktype, 0) + 1

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

    display_names = [k.replace("\\", "/").rsplit("/", 1)[-1] for k in kernels]
    alias_extra = db.alias_counts_by_name(display_names) if db is not None else {}

    for i, (k, filename) in enumerate(zip(kernels, display_names), 1):
        n_alias = alias_extra.get(filename)
        label = filename
        if n_alias:
            label = (
                f"{filename} [cyan](+{n_alias} "
                f"alias{'es' if n_alias > 1 else ''})[/cyan]"
            )
        table.add_row(str(i), _kernel_type_of(k), label)

    console.print(table)


def _print_coverage_table(results, body_id: int, metakernel: str) -> None:
    """Print a rich table summarising body coverage results."""
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
                for marker in ("SPICE(", "Input file"):
                    idx = err_msg.find(marker)
                    if idx >= 0:
                        err_msg = err_msg[idx:idx + 80].split("\n")[0]
                        break
                else:
                    err_msg = err_msg[:60] + "…"
            table.add_row(r.filename, r.kernel_type, f"[red]{err_msg}[/red]", "", "")
            continue
        if not r.body_found:
            table.add_row(r.filename, r.kernel_type, "[dim]not found[/dim]", "", "")
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


def _print_server_check(report: dict, *, verbose: bool = False) -> None:
    """Render verify_against_server()'s result.

    Problems are always listed: the point of --full is that a silent
    "Found locally: 102/102" hid ten wrong files (#15).
    """
    bad = [e for e in report["entries"] if e["status"] not in ("ok", "NO_REMOTE")]
    unchecked = [e for e in report["entries"] if e["status"] == "NO_REMOTE"]
    print("\n  Against the server:")
    print(f"    Verified:     {report['ok']}")
    print(f"    Problems:     {report['bad']}")
    print(f"    Unchecked:    {report['unchecked']}")
    for e in bad:
        print(f"      {e['status']:<14} {e['filename']}  ({e['detail']})")
    if unchecked and verbose:
        for e in unchecked:
            print(f"      {e['status']:<14} {e['filename']}  ({e['detail']})")
    elif unchecked:
        print(f"      (use -v to list the {len(unchecked)} unchecked)")
