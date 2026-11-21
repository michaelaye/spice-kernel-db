"""Acquire panel: browse, get, update (all tracked metakernels)."""

from __future__ import annotations

import sys
import urllib.error
from enum import Enum
from typing import Annotated, Optional

import typer
from rich.panel import Panel

from spice_kernel_db.db import AmbiguousKernelRoot, KernelDB, MetakernelUnreachableError
from spice_kernel_db.hashing import guess_mission
from spice_kernel_db.remote import Probe, ProbeResult, probe_url

from ._app import ACQUIRE, app, open_db, rename_note, state
from .prompts import (
    INVALID,
    _interactive_select_metakernel,
    _mission_table,
    _pick_index,
    _pick_root,
    _resolve_mission_mk_dir,
    _menu,
    console,
)

MissionOverride = Annotated[Optional[str], typer.Option(
    "--mission", help="Override auto-detected mission name")]
DownloadDir = Annotated[Optional[str], typer.Option(
    "--download-dir", help="Kernel root to download into (default: the root that "
                           "already holds the metakernel or its mission)")]
Yes = Annotated[bool, typer.Option("-y", "--yes", help="Skip confirmation prompt")]
Force = Annotated[bool, typer.Option(
    "--force", help="Re-download all kernels even if already on disk")]


class SortBy(str, Enum):
    name = "name"
    date = "date"


@app.command(rich_help_panel=ACQUIRE, epilog="See also: get, mission add",
             help="Browse remote metakernels in a NAIF mk/ directory")
def browse(
    ctx: typer.Context,
    url: Annotated[Optional[str], typer.Argument(
        help="URL to a mission's mk/ directory, or a mission name "
             "(omit to list known directories)")] = None,
    mission: MissionOverride = None,
    show_versioned: Annotated[bool, typer.Option(
        "--show-versioned", help="Show versioned snapshots under each metakernel")] = False,
    sort_by: Annotated[SortBy, typer.Option(
        "--sort",
        help="Row ordering: 'name' (default, alphabetical) or 'date' "
             "(by latest remote date ascending — newest at the bottom)")] = SortBy.name,
    archived: Annotated[bool, typer.Option(
        "--archived",
        help="Browse the mission's former_versions/ subdirectory instead "
             "of the live mk/ listing — for older versioned metakernels "
             "that have been superseded")] = False,
    filter_: Annotated[Optional[str], typer.Option(
        "--filter", metavar="SUBSTRING",
        help="Case-insensitive substring filter on metakernel filenames "
             "(useful for narrowing large archive listings, e.g. "
             "--filter crema_5_1)")] = None,
) -> None:
    def _maybe_archive(u: str) -> str:
        return u.rstrip("/") + "/former_versions/" if archived else u

    def _show(db: KernelDB, mk_dir_url: str, mission_name: str | None) -> None:
        db.browse_remote_metakernels(
            mk_dir_url,
            mission=mission_name,
            show_versioned=show_versioned,
            sort_by=sort_by.value,
            filter=filter_,
            archived=archived,
        )

    with open_db(ctx) as db:
        if url:
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
                mission = mission or m["name"]
            url = _maybe_archive(url)
            try:
                _show(db, url, mission)
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                print(
                    f"No metakernel directory found at:\n  {url}\n\n"
                    f"This mission may not publish metakernels at this location.\n"
                    f"Use 'spice-kernel-db mission remove' and re-add with the correct URL.",
                    file=sys.stderr,
                )
            except urllib.error.URLError as e:
                print(f"Could not connect to server:\n  {url}\n  {e.reason}", file=sys.stderr)
            return

        # No args — pick a mission to browse interactively
        missions = db.list_missions()
        if not missions:
            print("No configured missions.")
            print("Use 'spice-kernel-db mission add' to set one up.")
            return
        if len(missions) == 1:
            m = missions[0]
            console.print(f"Browsing [bold]{m['name']}[/bold]...\n")
            _show(db, _maybe_archive(m["mk_dir_url"]), m["name"])
            return

        _menu().print(_mission_table(missions, "Select a mission to browse"))
        # quiet_eof: this menu has always returned silently on EOF.
        idx = _pick_index(f"\nSelect mission [1-{len(missions)}]: ", len(missions),
                          invalid=INVALID, quiet_eof=True)
        if idx is None:
            return
        m = missions[idx]
        console.print(f"\nBrowsing [bold]{m['name']}[/bold]...\n")
        _show(db, _maybe_archive(m["mk_dir_url"]), m["name"])


def _locate_remote_metakernel(mk_dir_url: str, filename: str) -> str:
    """URL of *filename* under mk/ or former_versions/; exits when neither answers."""
    live_url = mk_dir_url + filename
    archive_url = mk_dir_url + "former_versions/" + filename
    live = probe_url(live_url)
    archive = Probe(ProbeResult.ABSENT) if live.ok else probe_url(archive_url)
    if live.ok:
        return live_url
    if archive.ok:
        console.print(
            f"[dim]Not in mk/, using archived copy: "
            f"former_versions/{filename}[/dim]"
        )
        return archive_url
    if live.unreachable or archive.unreachable:
        # The server never gave us a status, so we do not know whether the
        # file is there. Saying "not found" here was issue #10: a slow ESA
        # handshake read as a missing file.
        unreachable = live if live.unreachable else archive
        print(
            f"Could not reach the server for '{filename}' "
            f"({unreachable.detail}).\n"
            f"  Tried:\n"
            f"    {live_url}  [{live.detail or 'ok'}]\n"
            f"    {archive_url}  [{archive.detail or 'ok'}]\n"
            f"  The metakernel may still exist. Retry, or raise the "
            f"timeout if this server is habitually slow.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"Metakernel '{filename}' not found.\n"
        f"  Tried:\n"
        f"    {live_url}  [{live.detail}]\n"
        f"    {archive_url}  [{archive.detail}]",
        file=sys.stderr,
    )
    sys.exit(1)


@app.command(rich_help_panel=ACQUIRE, epilog="See also: browse, update, import",
             help="Download a metakernel and the kernels it needs that are missing")
def get(
    ctx: typer.Context,
    url: Annotated[Optional[str], typer.Argument(
        help="Full URL, a tracked metakernel (re-fetched from its source), or a "
             ".tm filename on the configured mission's server. "
             "Omit to pick from the server listing at a terminal.")] = None,
    download_dir: DownloadDir = None,
    mission: MissionOverride = None,
    yes: Yes = False,
    force: Force = False,
) -> None:
    st = state(ctx)
    with open_db(ctx) as db:
        alias_filename: str | None = None
        if url is None:
            # Interactive selection — fetch remote listing
            selection = _interactive_select_metakernel(db, mission)
            if not selection:
                return
            url, alias_filename = selection
            mission = mission or guess_mission(url)
        elif not url.startswith("http"):
            if db.registry_row(url, mission=mission) is not None:
                # Tracked already: re-fetch from the stored source, download
                # what is missing, re-index (what `update <name>` did, #47).
                _refetch(db, url, mission=mission, download_dir=download_dir,
                         yes=yes, force=force, verbose=st.verbose)
                return
            # A filename on the mission's server
            mk_dir_url, mission_name = _resolve_mission_mk_dir(db, mission)
            if not mk_dir_url:
                return
            url = _locate_remote_metakernel(mk_dir_url, url)
            mission = mission or mission_name
        kwargs = {"mission": mission, "yes": yes, "force": force,
                  "alias_filename": alias_filename, "verbose": st.verbose}
        try:
            db.get_metakernel(url, download_dir=download_dir, **kwargs)
        except AmbiguousKernelRoot as e:
            # Several roots and none holds this mission yet: ask (exit 2 in
            # a script), then fetch again into the chosen root.
            chosen = _pick_root(e, hint=f"spice-kernel-db get {url} --download-dir <root>")
            db.get_metakernel(url, download_dir=chosen, **kwargs)


def _refetch(db: KernelDB, name: str, **kwargs) -> None:
    """``update_metakernel`` with the no-source and unreachable-source messages."""
    try:
        db.update_metakernel(name, **kwargs)
    except AmbiguousKernelRoot:
        raise  # main() maps it to exit 2; it is a LookupError too
    except LookupError as e:
        # Tracked by scan or import: no source URL and no mission mk/ URL.
        print(f"{e}", file=sys.stderr)
        sys.exit(1)
    except MetakernelUnreachableError as e:
        console.print(Panel(
            f"[red]The remote metakernel is no longer available.[/red]\n\n"
            f"  URL:      {e.url}\n"
            f"  Status:   HTTP {e.status}\n"
            f"  Filename: {e.filename}\n\n"
            f"NAIF (and ESA) routinely rotate old versioned snapshots "
            f"into a [italic]former_versions/[/italic] subdirectory, "
            f"which makes the original URL 404.\n\n"
            f"To drop every registry row whose source is gone, run:\n"
            f"  [bold]spice-kernel-db clean --registry[/bold]\n"
            f"or remove just this one:\n"
            f"  [bold]spice-kernel-db rm {e.filename}[/bold]",
            title="Metakernel unreachable",
            border_style="red",
        ))
        sys.exit(2)


@app.command(rich_help_panel=ACQUIRE, epilog="See also: get, check",
             help="Re-fetch every tracked metakernel from its source and report "
                  "what would change; downloads only with --execute")
def update(
    ctx: typer.Context,
    metakernel: Annotated[Optional[str], typer.Argument(hidden=True)] = None,
    execute: Annotated[bool, typer.Option(
        "--execute", help="Download the missing kernels (default: report only)")] = False,
    download_dir: DownloadDir = None,
    # The three below serve only the pre-#47 form `update <name>`.
    mission: Annotated[Optional[str], typer.Option("--mission", hidden=True)] = None,
    yes: Annotated[bool, typer.Option("-y", "--yes", hidden=True)] = False,
    force: Annotated[bool, typer.Option("--force", hidden=True)] = False,
) -> None:
    if metakernel is not None:
        rename_note(f"update {metakernel}", f"get {metakernel}")
        get(ctx, url=metakernel, download_dir=download_dir, mission=mission,
            yes=yes, force=force)
        return
    with open_db(ctx, read_only=not execute) as db:
        db.update_all(download_dir=download_dir, dry_run=not execute, verbose=state(ctx).verbose)
    if not execute:
        console.print("[dim]Dry run: pass --execute to download.[/dim]")
