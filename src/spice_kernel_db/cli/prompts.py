"""Terminal prompts and interactive pickers shared by several commands."""

from __future__ import annotations

import sys
import urllib.error

from rich.console import Console
from rich.table import Table

from spice_kernel_db.db import KernelDB, RegistryRow, _format_size
from spice_kernel_db.remote import list_remote_metakernels

console = Console()
err_console = Console(stderr=True)


def _interactive() -> bool:
    """True when stdin is a terminal, so a menu can be shown (#14, #45)."""
    return sys.stdin.isatty()


def _menu() -> Console:
    """Where a menu's table goes: stdout at a terminal, stderr otherwise, so
    a script that gets exit 2 keeps a clean stdout (#45)."""
    return console if _interactive() else err_console


def _require_interactive(what: str, hint: str | None = None) -> None:
    """Exit 2 when *what* needs a terminal and stdin is not one.

    Issue #14: the interactive menus called input() unconditionally, so
    `ssh host skd mission add`, a CI step or any redirected stdin produced a
    raw EOFError traceback instead of an explanation. *hint* says what to
    pass instead; the default is the mission-add form.
    """
    if not _interactive():
        hint = hint or ("Pass the values instead, e.g. "
                        "spice-kernel-db mission add <NAME> --server-url <URL>")
        print(f"{what} needs a terminal for its menu.\n  {hint}", file=sys.stderr)
        sys.exit(2)


def _ask(prompt: str) -> str:
    """input() that reports EOF instead of raising it (#14)."""
    try:
        return input(prompt)
    except EOFError:
        print(
            "\nstdin closed while waiting for input; aborting.",
            file=sys.stderr,
        )
        sys.exit(2)


def _confirm(prompt: str) -> bool:
    """A [y/N] prompt. EOF counts as "no", which is the safe default (#14)."""
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        print("\nstdin closed; assuming no.", file=sys.stderr)
        return False
    return answer in ("y", "yes")


INVALID = "[red]Invalid selection.[/red]"


def _pick_index(
    prompt: str, n: int, *, invalid: str | None = None, quiet_eof: bool = False,
) -> int | None:
    """Ask for a 1-based index into *n* rows and return it 0-based.

    Returns None when the user cancels (Ctrl-C, or EOF when *quiet_eof*;
    otherwise ``_ask`` exits 2 on EOF) and on a bad entry, printing
    *invalid* to stderr in that case when one is given.

    Every picker in the CLI comes through here, so this is where a script
    or notebook (stdin not a terminal) gets its answer: the candidates were
    just listed, and the command exits 2 instead of waiting for input (#45).
    """
    _require_interactive(
        f"Picking one of the {n} candidates listed above",
        hint="Pass it as the argument instead.",
    )
    try:
        raw = (input(prompt) if quiet_eof else _ask(prompt)).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    try:
        idx = int(raw)
    except ValueError:
        idx = 0
    if 1 <= idx <= n:
        return idx - 1
    if invalid:
        err_console.print(invalid)
    return None


def _metakernel_table(rows: list[RegistryRow], *, index_style: str = "bold") -> Table:
    """The numbered "Local metakernels" table over ``KernelDB.registry_rows``."""
    table = Table(title="Local metakernels")
    table.add_column("#", justify="right", style=index_style)
    table.add_column("Filename")
    table.add_column("Mission")
    table.add_column("Acquired")
    for i, row in enumerate(rows, 1):
        acq_str = str(row.acquired_at)[:16] if row.acquired_at else ""
        table.add_row(str(i), row.filename, row.mission, acq_str)
    return table


def _mission_table(missions: list[dict], title: str) -> Table:
    """The numbered mission picker table (``#``, Mission, mk/ URL)."""
    table = Table(title=title)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Mission")
    table.add_column("mk/ URL")
    for i, m in enumerate(missions, 1):
        table.add_row(str(i), m["name"], m["mk_dir_url"])
    return table


def _require_metakernel(
    metakernel: str | None, db: KernelDB, mission: str | None,
) -> str | None:
    """*metakernel* as given, or the user's pick when it was omitted.

    Resolution happens once, in the ``KernelDB`` method that reads the
    file; ``main()`` turns its two errors into exit codes (#45). Returns
    None when the user cancelled the picker.
    """
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

    _menu().print(f"\n'{name_or_id}' matches multiple NAIF bodies:\n")
    for i, (bid, desc) in enumerate(candidates, 1):
        _menu().print(f"  [{i}] {desc}  (NAIF ID {bid})")
    _menu().print()
    idx = _pick_index(f"Select [1-{len(candidates)}]: ", len(candidates),
                      invalid="Invalid selection.")
    if idx is None:
        return None
    return candidates[idx][0]


def _select_local_metakernel(
    db: KernelDB, mission: str | None,
) -> str | None:
    """List locally registered metakernels and let the user pick one.

    Returns the mk_path of the selected metakernel, or None on failure.
    Prints a hint about ``browse`` and ``get`` when no metakernels are found
    or when the user wants to fetch more. Without a terminal the table goes
    to stderr and ``_pick_index`` exits 2, so a script's stdout stays clean.
    """
    rows = db.registry_rows(mission)

    if not rows:
        mission_hint = f" for mission '{mission}'" if mission else ""
        err_console.print(
            f"[yellow]No locally acquired metakernels{mission_hint}.[/yellow]\n\n"
            f"[dim]To browse and download metakernels:\n"
            f"  spice-kernel-db browse [MISSION]\n"
            f"  spice-kernel-db get [METAKERNEL][/dim]\n",
        )
        return None

    _menu().print(_metakernel_table(rows))
    _menu().print(
        "\n[dim]To fetch more metakernels: "
        "spice-kernel-db browse [MISSION][/dim]\n"
    )

    idx = _pick_index(f"Select metakernel [1-{len(rows)}]: ", len(rows), invalid=INVALID)
    if idx is None:
        return None
    return rows[idx].mk_path


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
        err_console.print(
            "[yellow]No missions configured.[/yellow]\n"
            "[dim]Use 'spice-kernel-db mission add' first.[/dim]",
        )
        return None, None

    # Multiple missions — let user pick interactively
    _menu().print(_mission_table(missions, "Select a mission"))

    idx = _pick_index(f"\nSelect mission [1-{len(missions)}]: ", len(missions),
                      invalid=INVALID)
    if idx is None:
        return None, None
    m = missions[idx]
    return m["mk_dir_url"], m["name"]


def _interactive_pick_deduplicated_kernel(db: KernelDB) -> str | None:
    """Show a picker over deduplicated kernels and return the chosen name.

    Lists kernels whose content is shared under more than one filename
    (the only ones for which `which` shows a non-trivial trail), with an
    alias count, sorted most-aliased first. Returns the canonical filename
    of the selection, or ``None`` if there are none or the user cancels.
    """
    has_aliases = db.con.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_name = 'kernel_aliases'"
    ).fetchone()[0] > 0
    rows = []
    if has_aliases:
        rows = db.con.execute("""
            SELECT k.filename, k.kernel_type, k.size_bytes,
                   count(a.filename) AS n_names
            FROM kernels k
            JOIN kernel_aliases a ON a.sha256 = k.sha256
            GROUP BY k.sha256, k.filename, k.kernel_type, k.size_bytes
            HAVING count(a.filename) > 1
            ORDER BY n_names DESC, LOWER(k.filename)
        """).fetchall()

    if not rows:
        err_console.print(
            "[yellow]No deduplicated kernels in the database — every kernel "
            "has a unique name.\nPass a name or hash explicitly: "
            "spice-kernel-db which <name|hash>[/yellow]"
        )
        return None

    table = Table(title="Deduplicated kernels")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Kernel")
    table.add_column("Aliases", justify="right")
    table.add_column("Type")
    table.add_column("Size", justify="right")
    for i, (fn, kt, sz, n) in enumerate(rows, 1):
        size_str = _format_size(sz) if sz else ""
        table.add_row(str(i), fn, f"+{n - 1}", kt or "", size_str)
    _menu().print(table)

    idx = _pick_index(f"\nSelect kernel [1-{len(rows)}]: ", len(rows))
    if idx is None:
        return None
    return rows[idx][0]


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

    _menu().print(f"Fetching metakernels from [bold]{mk_dir_url}[/bold] ...")
    try:
        entries = list_remote_metakernels(mk_dir_url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            err_console.print(
                f"[red]No metakernel directory found at:[/red]\n  {mk_dir_url}",
            )
        else:
            raise
        return None

    if not entries:
        err_console.print("[red]No metakernels found.[/red]")
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

    # One row per base_name, linking to the latest version.
    table = Table(title=f"Available metakernels — {resolved_name or 'remote'}")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Metakernel")
    table.add_column("Versions", justify="right")
    table.add_column("Latest")
    table.add_column("Local", justify="center")
    choices: list[tuple[str, str | None]] = []  # (url, alias_filename)
    for i, (base_name, group) in enumerate(sorted(groups.items()), 1):
        latest = max(group, key=lambda e: e.filename)
        is_local = any(e.filename in local_filenames for e in group)
        n_ver = str(len(group)) if len(group) > 1 else ""
        local_col = "[green]✓[/green]" if is_local else ""
        table.add_row(str(i), base_name, n_ver, latest.date, local_col)
        alias = base_name if base_name != latest.filename else None
        choices.append((latest.url, alias))
    _menu().print(table)

    idx = _pick_index(f"\nSelect metakernel [1-{len(choices)}]: ", len(choices),
                      invalid=INVALID)
    if idx is None:
        return None
    return choices[idx]
