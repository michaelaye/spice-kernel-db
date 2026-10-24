"""Remote commands: browse, get, update."""

from __future__ import annotations

import sys
import urllib.error
from enum import Enum
from typing import Annotated, Optional

import typer
from rich.panel import Panel

from spice_kernel_db.db import KernelDB, MetakernelUnreachableError
from spice_kernel_db.hashing import guess_mission
from spice_kernel_db.remote import Probe, ProbeResult, probe_url

from ._app import BROWSE, app, open_db, state
from .prompts import (
    INVALID,
    _interactive_select_metakernel,
    _mission_table,
    _pick_index,
    _require_metakernel,
    _resolve_mission_mk_dir,
    console,
)

MissionOverride = Annotated[Optional[str], typer.Option(
    "--mission", help="Override auto-detected mission name")]
DownloadDir = Annotated[Optional[str], typer.Option(
    "--download-dir", help="Directory for downloaded kernels (default: from config)")]
Yes = Annotated[bool, typer.Option("-y", "--yes", help="Skip confirmation prompt")]
Force = Annotated[bool, typer.Option(
    "--force", help="Re-download all kernels even if already on disk")]


class SortBy(str, Enum):
    name = "name"
    date = "date"


@app.command(rich_help_panel=BROWSE, epilog="See also: get, mission add",
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

        console.print(_mission_table(missions, "Select a mission to browse"))
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


@app.command(rich_help_panel=BROWSE, epilog="See also: browse, update",
             help="Download missing kernels for a remote metakernel")
def get(
    ctx: typer.Context,
    url: Annotated[Optional[str], typer.Argument(
        help="Full URL, or .tm filename (requires a configured mission). "
             "Omit to select interactively.")] = None,
    download_dir: DownloadDir = None,
    mission: MissionOverride = None,
    yes: Yes = False,
    force: Force = False,
) -> None:
    st = state(ctx)
    if download_dir is None:
        download_dir = st.config.kernel_dir
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
            # Treat as a filename — resolve mission mk_dir_url
            mk_dir_url, mission_name = _resolve_mission_mk_dir(db, mission)
            if not mk_dir_url:
                return
            url = _locate_remote_metakernel(mk_dir_url, url)
            mission = mission or mission_name
        db.get_metakernel(
            url,
            download_dir=download_dir,
            mission=mission,
            yes=yes,
            force=force,
            alias_filename=alias_filename,
            verbose=st.verbose,
        )


@app.command(rich_help_panel=BROWSE, epilog="See also: get, browse",
             help="Re-fetch a metakernel from its source and download new kernels")
def update(
    ctx: typer.Context,
    metakernel: Annotated[Optional[str], typer.Argument(
        help="Metakernel filename or path (omit to select interactively)")] = None,
    mission: MissionOverride = None,
    download_dir: DownloadDir = None,
    yes: Yes = False,
    force: Force = False,
) -> None:
    st = state(ctx)
    if download_dir is None:
        download_dir = st.config.kernel_dir
    with open_db(ctx) as db:
        mk = _require_metakernel(metakernel, db, mission)
        if mk is None:
            return
        try:
            db.update_metakernel(
                mk,
                mission=mission,
                download_dir=download_dir,
                yes=yes,
                force=force,
                verbose=st.verbose,
            )
        except MetakernelUnreachableError as e:
            console.print(Panel(
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
