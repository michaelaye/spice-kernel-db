"""Configuration commands: mission add/list/remove, config, reset."""

from __future__ import annotations

import http.client
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.table import Table

from spice_kernel_db import registry
from spice_kernel_db.config import setup_interactive, show_config
from spice_kernel_db.db import KernelDB
from spice_kernel_db.remote import (
    SPICE_SERVERS,
    check_mk_availability,
    discover_mk_url,
    list_remote_missions,
    probe_mk_candidates,
    server_label_for,
)

from ._app import CONFIGURE, app, open_db, state
from .prompts import _ask, _confirm, _require_interactive, console

mission_app = typer.Typer()
app.add_typer(mission_app, name="mission", rich_help_panel=CONFIGURE,
              help="Manage configured missions")


@mission_app.callback(invoke_without_command=True)
def _mission(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        names = ",".join(ctx.command.commands)
        print(f"Usage: spice-kernel-db mission {{{names}}}")


@mission_app.command("add", help="Add a new mission (interactive when no arguments)")
def mission_add(
    ctx: typer.Context,
    name: Annotated[Optional[str], typer.Argument(
        help="Mission name. When given, the prompts are skipped.")] = None,
    server_url: Annotated[Optional[str], typer.Option(
        "--server-url",
        help="Archive server base URL. Required with positional name unless "
             "the name is unambiguous on a known server.")] = None,
    mk_dir_url: Annotated[Optional[str], typer.Option(
        "--mk-dir-url",
        help="Override the metakernel directory URL (bypasses auto-discovery).")] = None,
    no_dedup: Annotated[bool, typer.Option(
        "--no-dedup", help="Disable deduplication for this mission.")] = False,
) -> None:
    with open_db(ctx) as db:
        if mk_dir_url or name:
            _mission_add_noninteractive(db, name, server_url, mk_dir_url, no_dedup)
        else:
            _mission_add_interactive(db, no_dedup)


@mission_app.command("list", help="List configured missions")
def mission_list(ctx: typer.Context) -> None:
    with open_db(ctx) as db:
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
        table.add_row(m["name"], m["server_label"], dedup_str, m["mk_dir_url"])
    console.print(table)


@mission_app.command("remove", help="Remove a mission")
def mission_remove(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Mission name to remove")],
) -> None:
    with open_db(ctx) as db:
        if db.remove_mission(name):
            print(f"Mission '{name}' removed.")
        else:
            print(f"Mission '{name}' not found.", file=sys.stderr)


@app.command(rich_help_panel=CONFIGURE, help="Show or update configuration")
def config(
    ctx: typer.Context,
    setup: Annotated[bool, typer.Option(
        "--setup", help="Re-run interactive setup")] = False,
    config_args: Annotated[Optional[list[str]], typer.Argument(
        help="'set <key> <value>' or 'get <key>' (keys: db_path, kernel_dir)")] = None,
) -> None:
    cfg = state(ctx).config
    if setup:
        setup_interactive()
        return
    if not config_args:
        show_config(cfg)
        return
    verb, *rest = config_args
    if verb == "get" and len(rest) == 1:
        key = rest[0]
        _require_config_key(cfg, key)
        print(getattr(cfg, key))
    elif verb == "set" and len(rest) == 2:
        from spice_kernel_db.config import save_config
        key, value = rest
        _require_config_key(cfg, key)
        setattr(cfg, key, value)
        save_config(cfg)
        print(f"{key} = {value}")
    else:
        print("Usage: spice-kernel-db config [set <key> <value> | get <key>]", file=sys.stderr)
        sys.exit(1)


def _require_config_key(cfg, key: str) -> None:
    if not hasattr(cfg, key):
        print(f"Unknown key: {key}. Valid keys: db_path, kernel_dir", file=sys.stderr)
        sys.exit(1)


@app.command(rich_help_panel=CONFIGURE, help="Delete the database and start fresh")
def reset(
    ctx: typer.Context,
    yes: Annotated[bool, typer.Option(
        "-y", "--yes", help="Skip confirmation prompt")] = False,
) -> None:
    st = state(ctx)
    db_path = Path(st.db).expanduser()
    if not db_path.is_file():
        print(f"No database found at {db_path}")
        return
    if not yes and not _confirm(f"Delete database at {db_path}? [y/N]: "):
        print("Aborted.")
        return
    db_path.unlink()
    kernel_dir = Path(st.config.kernel_dir).expanduser()
    print(f"Database deleted: {db_path}")
    print(f"Kernel files in {kernel_dir} are untouched.\n")
    print("To re-index existing kernels:")
    print(f"  spice-kernel-db scan {kernel_dir}")


# --- mission add: the wizard and its non-interactive twin -----------------

class AmbiguousServer(Exception):
    """More than one known SPICE server carries the requested mission."""

    def __init__(self, mission: str, urls: list[str]):
        self.mission = mission
        self.urls = urls
        super().__init__(f"{mission} is on {len(urls)} servers")


def _choose_server() -> tuple[str, str]:
    """Prompt for a SPICE archive server. Returns ``(label, server_url)``."""
    server_names = list(SPICE_SERVERS.keys())
    print("\nAvailable SPICE archive servers:\n")
    for i, name in enumerate(server_names, 1):
        print(f"  [{i}] {name}  ({SPICE_SERVERS[name]})")
    print()
    while True:
        choice = _ask(f"Select server [1-{len(server_names)}]: ").strip()
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
        choice = _ask(f"Select [1-{len(hits)}]: ").strip()
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
        choice = _ask(
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
    # Default-yes prompt, so EOF must mean yes here, not the _confirm default.
    try:
        answer = input(
            f"\nEnable deduplication for {mission_name}? [Y/n]: "
        ).strip().lower()
    except EOFError:
        print("\nstdin closed; keeping the default (enabled).", file=sys.stderr)
        return True
    return answer not in ("n", "no")


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
    print("\nNext steps:")
    print(f"  spice-kernel-db browse {mission_name}")
    print(f"  spice-kernel-db get <metakernel>.tm --mission {mission_name}")


def _mission_add_interactive(db: KernelDB, no_dedup: bool) -> None:
    """Interactive mission setup: choose server → pick mission → configure."""
    _require_interactive("mission add")
    server_label, server_url = _choose_server()
    chosen = _choose_mission(server_url, server_label)
    if chosen is None:
        return
    mission_name, mk_dir_url = chosen
    dedup = False if no_dedup else _choose_dedup(mission_name)
    _save_mission(db, mission_name, server_label, server_url, mk_dir_url, dedup)


def _servers_hosting(name: str) -> list[str]:
    """Every known server that lists *name*, in SPICE_SERVERS order."""
    from concurrent.futures import ThreadPoolExecutor

    urls = list(SPICE_SERVERS.values())

    def _has_mission(url: str) -> bool:
        try:
            return _match_ci(name, list_remote_missions(url)) is not None
        except (OSError, http.client.HTTPException):  # server down, or not a listing
            return False

    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        found = list(pool.map(_has_mission, urls))
    return [url for url, ok in zip(urls, found) if ok]


def _infer_server_url(name: str, server_url_arg: str | None) -> str | None:
    """Infer the server URL from --server-url, or a UNIQUE match in SPICE_SERVERS.

    Returns None when nothing can be inferred. Raises AmbiguousServer when more
    than one known server carries the mission.

    Issue #13: this used to return the first server that had the mission, so
    `skd mission add JUICE` silently configured NASA's NAIF mirror even though
    JUICE is published on ESA first and the mirror can lag. The help text had
    always promised that a positional name needs --server-url "unless the name
    is unambiguous on a known server"; now the code enforces that.
    """
    if server_url_arg:
        return server_url_arg if server_url_arg.endswith("/") else server_url_arg + "/"

    hits = _servers_hosting(name)
    if len(hits) > 1:
        raise AmbiguousServer(name, hits)
    return hits[0] if hits else None


def _mission_add_noninteractive(
    db: KernelDB, name: str | None, server_url_arg: str | None,
    mk_dir_url: str | None, no_dedup: bool,
) -> None:
    """Non-interactive mission add. Either --mk-dir-url, or discover from --server-url."""
    if not name:
        print(
            "mission add: --mk-dir-url requires a positional mission name.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        server_url = _infer_server_url(name, server_url_arg)
    except AmbiguousServer as exc:
        listed = "\n".join(
            f"    --server-url {u}   ({server_label_for(u)})" for u in exc.urls
        )
        print(
            f"mission add: '{name}' is published on "
            f"{len(exc.urls)} known servers:\n{listed}\n"
            f"  Re-run with one of them. ESA missions are often published on "
            f"ESA first, so the NAIF mirror can lag.",
            file=sys.stderr,
        )
        sys.exit(2)
    if server_url is None:
        print(
            f"mission add: cannot determine server URL for '{name}'. "
            f"Pass --server-url explicitly.",
            file=sys.stderr,
        )
        sys.exit(2)
    server_label = server_label_for(server_url)

    if mk_dir_url:
        if not mk_dir_url.endswith("/"):
            mk_dir_url += "/"
        if not probe_mk_candidates([mk_dir_url]):
            print(
                f"mission add: --mk-dir-url did not respond to HEAD: {mk_dir_url}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        reg_candidates = registry.registry_candidates(name, server_url)
        hits = discover_mk_url(server_url, name, registry_candidates=reg_candidates)
        if not hits:
            print(
                f"mission add: no metakernel directory found for "
                f"'{name}' on {server_url}. Re-run with --mk-dir-url, "
                f"or see docs/troubleshooting.qmd.",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(hits) > 1:
            print(
                f"mission add: multiple metakernel directories responded for "
                f"'{name}'. Re-run with --mk-dir-url <one of>:",
                file=sys.stderr,
            )
            for url in hits:
                print(f"  {url}", file=sys.stderr)
            sys.exit(1)
        mk_dir_url = hits[0]

    _save_mission(db, name, server_label, server_url, mk_dir_url, not no_dedup)
