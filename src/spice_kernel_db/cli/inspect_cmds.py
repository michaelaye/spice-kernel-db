"""Read-only commands: list, check (verify is its hidden alias), doctor, coverage, metakernels, resolve, aliases, stats, duplicates."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, NoReturn, Optional

import typer
from rich.panel import Panel
from rich.table import Table

from spice_kernel_db.db import KernelDB, _format_size

from ._app import INSPECT, app, open_db, state
from .prompts import (
    _interactive_pick_deduplicated_kernel,
    _require_metakernel,
    _resolve_body_interactive,
    _select_local_metakernel,
    console,
)

MetakernelArg = Annotated[
    Optional[str],
    typer.Argument(
        help="Path to a .tm file, registry filename or versionless base name "
             "(omit to pick from the tracked metakernels at a terminal)"),
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
        _list_kernels(str(db.resolve_metakernel(mk, mission=mission)), db=db,
                      kernel_type=kernel_type)


def _print_present(present: dict, *, verbose: bool) -> None:
    """Render ``check_metakernel``'s result: counts, missing names, warnings."""
    found, missing, warnings = present["found"], present["missing"], present["warnings"]
    print(f"\nMetakernel: {present['mk_path']}")
    print(f"  Mission:        {present['mission']}")
    print(f"  Found locally:  {len(found)}/{len(found) + len(missing)}")
    print(f"  Missing:        {len(missing)}")
    if missing:
        print("  Missing files:")
        for m in missing:
            print(f"    {m}")
    if not warnings:
        return
    if verbose:
        print("  Warnings:")
        for w in warnings:
            print(f"    ⚠ {w}")
        return
    n_hash = sum(1 for w in warnings if "matched by content hash" in w or "matched by hash" in w)
    n_cross = sum(1 for w in warnings if "using copy from" in w)
    n_other = len(warnings) - n_hash - n_cross
    parts = []
    if n_hash:
        parts.append(f"    Resolved via dedup:     {n_hash} kernels (same content, different filename)")
    if n_cross:
        parts.append(f"    Cross-mission:          {n_cross} kernels")
    if n_other:
        parts.append(f"    Other warnings:         {n_other}")
    if parts:
        print("  Warnings:")
        for p in parts:
            print(p)
        print("    (use -v for full details)")


def _run_check(ctx: typer.Context, metakernel: str | None, *, remote: bool,
               hash_: bool, spice: bool, json_: bool, mission: str | None) -> None:
    """The one implementation behind ``check`` and its ``verify`` alias (#46)."""
    verbose = state(ctx).verbose
    if spice and not remote:
        print("check: --spice needs --remote", file=sys.stderr)
        sys.exit(2)
    with open_db(ctx, read_only=True) as db:
        mk = _require_metakernel(metakernel, db, mission)
        if mk is None:
            return
        report = db.check_report(mk, mission=mission, hash_=hash_, remote=remote, spice=spice)
    if json_:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_present(report["present"], verbose=verbose)
        if report["resolves"]["fail"]:
            _render_verify_table(report["resolves"])
        if report["remote"] is not None:
            _print_server_check(report["remote"], verbose=verbose)
        if report["newer"]:
            print(f"\n  {report['newer']}")
        print()
        for v in report["verdicts"]:
            mark = "✓" if v["ok"] else ("✗" if v["ok"] is False else "–")
            print(f"  {mark} {v['line']}")
    if report["exit_code"]:
        sys.exit(report["exit_code"])


@app.command(rich_help_panel=INSPECT,
             help="Check a metakernel: kernels present, entries resolve, "
                  "and with --remote, matches the server",
             epilog="Exit code: 1 a kernel is missing, 2 the metakernel does not resolve, "
                    "4 differs from the server (--remote); summed, so 5 is missing plus "
                    "differs. See also: doctor, list, rewrite")
def check(
    ctx: typer.Context,
    metakernel: MetakernelArg = None,
    remote: Annotated[bool, typer.Option(
        "--remote",
        help="Also compare each kernel's size with the server (one HEAD per kernel). "
             "The only network check; exit bit 4")] = False,
    hash_: Annotated[bool, typer.Option(
        "--hash",
        help="Compare sha256 against the database, and with --remote against the "
             "server (downloads each kernel, so slow)")] = False,
    spice: Annotated[bool, typer.Option(
        "--spice",
        help="With --remote, furnsh each local kernel and report what SPICE "
             "cannot read (requires spiceypy)")] = False,
    json_: Annotated[bool, typer.Option(
        "--json", help="Emit the three verdicts and their evidence as JSON")] = False,
    mission: Annotated[Optional[str], typer.Option(
        "--mission", help="Preferred mission for kernel resolution")] = None,
) -> None:
    _run_check(ctx, metakernel, remote=remote, hash_=hash_, spice=spice,
               json_=json_, mission=mission)


@app.command(hidden=True, help="Renamed: use check (--deep is --hash)")
def verify(
    ctx: typer.Context,
    metakernel: MetakernelArg = None,
    deep: Annotated[bool, typer.Option("--deep")] = False,
    strict: Annotated[bool, typer.Option("--strict")] = False,
    json_: Annotated[bool, typer.Option("--json")] = False,
    mission: Annotated[Optional[str], typer.Option("--mission")] = None,
) -> None:
    print("note: 'verify' is now 'check' (--deep is --hash); exit 2 means the "
          "metakernel does not resolve", file=sys.stderr)
    if strict:
        print("note: --strict has no equivalent and is ignored", file=sys.stderr)
    _run_check(ctx, metakernel, remote=False, hash_=deep, spice=False,
               json_=json_, mission=mission)


@app.command(rich_help_panel=INSPECT,
             help="Database health: poisoned rows, files gone from disk",
             epilog="Exits 1 only on poisoned supersession rows; a file gone from disk is "
                    "listed with the command that cleans it up. See also: check, prune, stats")
def doctor(
    ctx: typer.Context,
    json_: Annotated[bool, typer.Option("--json", help="Emit the report as JSON")] = False,
) -> None:
    with open_db(ctx, read_only=True) as db:
        report = db.doctor()
    if json_:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"  Kernels: {report['n_kernels']}   Locations: {report['n_locations']}   "
              f"Tracked metakernels: {report['n_metakernels']}")
        poisoned = report["poisoned"]
        print(f"  Poisoned supersession rows: {len(poisoned)}")
        for row in poisoned:
            print(f"    {row['filename']}  ({row['abs_path']})")
        if poisoned:
            print("    Present on disk, yet flagged as superseded by content that\n"
                  "    exists nowhere: every `update` re-downloads them.\n"
                  "    Repair: spice-kernel-db scan <kernel_dir>")
        gone = report["missing_locations"]
        print(f"  Kernel locations whose file is gone: {len(gone)}"
              + ("  (spice-kernel-db prune --execute)" if gone else ""))
        for path in gone:
            print(f"    {path}")
        gone_mk = report["missing_metakernels"]
        print(f"  Tracked metakernels whose file is gone: {len(gone_mk)}"
              + ("  (spice-kernel-db mk --remove <name>)" if gone_mk else ""))
        for name in gone_mk:
            print(f"    {name}")
    if report["poisoned"]:
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
        help="Show info for one metakernel (path, registry filename or base name)")] = None,
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
            # Matched against the registry directly, not through the file
            # resolver: removing a row whose file is gone is the point.
            if db.remove_metakernel(name):
                print(f"Metakernel '{name}' removed from registry.")
            else:
                print(f"Metakernel '{name}' not found.", file=sys.stderr)
                sys.exit(1)
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
        "--metakernel",
        help="Resolve all kernels in a metakernel (path, registry filename "
             "or base name)")] = None,
) -> None:
    with open_db(ctx, read_only=True) as db:
        if resolve_mk:
            # Batch mode — resolve all kernels in a metakernel
            from spice_kernel_db import parse_metakernel
            mk = db.resolve_metakernel(resolve_mk, mission=mission)
            all_ok = True
            for kernel in parse_metakernel(mk).kernel_filenames():
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
            picked = _select_local_metakernel(db, mission)
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
    """Render a verify_metakernel result as a rich table; the verdict line
    that follows carries the summary (#46)."""
    mk = result["mk_path"]
    deep = result["deep"]

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

    table = Table(title=f"check {mk}: entries against the database ({'hash' if deep else 'size'})")
    table.add_column("Entry", overflow="fold")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    for e in result["entries"]:
        status = e["status"]
        color = style_for.get(status, "white")
        table.add_row(e["raw"], f"[{color}]{status}[/{color}]", e["detail"])
    console.print(table)



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

    Problems are always listed: the point of --remote is that a silent
    "Found locally: 102/102" hid ten wrong files (#15). The counts are on
    the verdict line (#46).
    """
    bad = [e for e in report["entries"] if e["status"] not in ("ok", "NO_REMOTE")]
    unchecked = [e for e in report["entries"] if e["status"] == "NO_REMOTE"]
    if not bad and not unchecked:
        return
    print("\n  Against the server:")
    for e in bad:
        print(f"      {e['status']:<14} {e['filename']}  ({e['detail']})")
    if unchecked and verbose:
        for e in unchecked:
            print(f"      {e['status']:<14} {e['filename']}  ({e['detail']})")
    elif unchecked:
        print(f"      (use -v to list the {len(unchecked)} unchecked)")
