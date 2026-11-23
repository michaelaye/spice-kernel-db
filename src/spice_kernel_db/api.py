"""The module-level API: one call, against the configured database (#49).

Each function opens the database named in the config, does one thing and
closes it. Scripts that do several things in a row keep a ``KernelDB``
instead. Nothing here prints; pass ``console=`` to ``get`` for the
status table and the download question.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from spice_kernel_db.db import KernelDB


def resolve_metakernel(ref: str | Path, mission: str | None = None) -> Path:
    """The on-disk path of a tracked metakernel, from any form of reference.

    Opens the database named in the config read-only, does one thing and closes it. For several
    operations in a row keep a ``KernelDB``. This function calls ``KernelDB.resolve_metakernel``.

    Parameters
    ----------
    ref : str | Path
        A path that exists, a registry filename, a registry ``mk_path``, the ``source_url`` the
        metakernel was fetched from, or a version-stripped base name (``juice_crema_5_2.tm`` for
        ``juice_crema_5_2_v473_20260819_001.tm``). Names are case-insensitive, with or without
        ``.tm``.
    mission : str | None, default None
        Narrow the registry to one mission (prefix match).

    Returns
    -------
    Path
        The metakernel file on disk.

    Raises
    ------
    MetakernelNotTracked
        Nothing matches, or the matched file is gone.
    AmbiguousMetakernel
        Several registry rows match. The function never picks the first of several.
    """
    with KernelDB(read_only=True) as db:
        return db.resolve_metakernel(ref, mission=mission)


def metakernels_covering(
    body_id: int, et: float | None = None, mission: str | None = None,
) -> list[dict]:
    """Tracked metakernels whose SPK kernels cover a body (at *et*, if given).

    Opens the database named in the config read-only, does one thing and closes it. This function
    calls ``KernelDB.metakernels_covering``. Coverage is read from the SPK files each metakernel
    loads that exist on disk. Identical aliases are skipped.

    Parameters
    ----------
    body_id : int
        NAIF ID of the body, for example ``-121`` for BepiColombo MPO.
    et : float | None, default None
        Ephemeris time that must be covered. ``None`` accepts any coverage at all.
    mission : str | None, default None
        Case-insensitive mission prefix filter.

    Returns
    -------
    list[dict]
        The tracked metakernels as ``KernelDB.list_metakernels`` returns them, each with an added
        ``intervals`` key: the ``(et_start, et_end)`` windows for the body.

    Raises
    ------
    ImportError
        SpiceyPy is not installed.

    Examples
    --------
    >>> import spiceypy
    >>> from spice_kernel_db import metakernels_covering
    >>> spiceypy.furnsh("naif0012.tls")
    >>> rows = metakernels_covering(-121, spiceypy.utc2et("2027-06-01"))  # BepiColombo MPO
    """
    with KernelDB(read_only=True) as db:
        return db.metakernels_covering(body_id, et=et, mission=mission)


def check(
    ref: str | Path, *, mission: str | None = None,
    hash_: bool = False, remote: bool = False, spice: bool = False,
) -> dict:
    """``skd check`` as data: the verdicts and their evidence.

    Opens the database named in the config read-only, does one thing and closes it. This function
    calls ``KernelDB.check_report``, whose documentation describes the verdicts and the result in
    full.

    Parameters
    ----------
    ref : str | Path
        The metakernel: a path, or any reference ``resolve_metakernel`` accepts.
    mission : str | None, default None
        Mission name for preferred-mission resolution.
    hash_ : bool, default False
        Verify SHA-256 hashes: recompute them for the ``resolves`` verdict and, with *remote*,
        compare them against the server.
    remote : bool, default False
        Compare the local kernels with the server. This is the one network path.
    spice : bool, default False
        With *remote*, also furnsh each local kernel in an isolated kernel pool and report what
        SPICE refuses to read. Needs SpiceyPy.

    Returns
    -------
    dict
        With the keys ``mk_path``, ``present``, ``resolves``, ``remote``, ``newer``, ``verdicts``
        (three ``{bit, ok, line}`` rows) and ``exit_code``, the sum of the failed verdicts' bits.

    Raises
    ------
    MetakernelNotTracked
        *ref* matches nothing.
    AmbiguousMetakernel
        *ref* matches several registry rows.
    """
    with KernelDB(read_only=True) as db:
        return db.check_report(ref, mission=mission, hash_=hash_, remote=remote, spice=spice)


def get(
    ref: str, *, mission: str | None = None, yes: bool = False,
    download_dir: str | Path | None = None, force: bool = False,
    console: Console | None = None,
) -> dict:
    """``skd get``: fetch a metakernel by URL, or re-fetch a tracked one by name.

    Opens the database named in the config, does one thing and closes it. For several operations in
    a row keep a ``KernelDB``. A *ref* that starts with ``http://`` or ``https://`` is fetched with
    ``KernelDB.get_metakernel``. Any other *ref* names a tracked metakernel and is re-fetched from
    its stored source with ``KernelDB.update_metakernel``.

    Without ``yes=True`` and without a *console* to ask on, nothing is downloaded and the result
    carries ``aborted: True``. A name that is not tracked raises ``MetakernelNotTracked``. Looking
    it up on the mission's server is the CLI's job (``skd get <name>`` does that, with a picker).

    Parameters
    ----------
    ref : str
        A URL, or any reference to a tracked metakernel.
    mission : str | None, default None
        Override the mission name.
    yes : bool, default False
        Download without asking.
    download_dir : str | Path | None, default None
        The kernel root to download into. By default ``KernelDB.root_for`` decides.
    force : bool, default False
        Download everything again.
    console : rich.console.Console | None, default None
        Where the status table and the download question go. Without one nothing is printed and the
        question is answered no.

    Returns
    -------
    dict
        The dict ``KernelDB.get_metakernel`` returns, with the keys ``found``, ``missing``,
        ``downloaded`` and ``warnings``, and ``aborted: True`` when nothing was downloaded because
        nobody was asked or the answer was no. A re-fetch by name also has ``scan_dirs``.

    Raises
    ------
    MetakernelNotTracked
        *ref* is not a URL and matches nothing tracked.
    AmbiguousMetakernel
        *ref* matches several registry rows.
    AmbiguousKernelRoot
        Several roots are configured, ``download_dir`` is not given and ``root_for`` cannot choose.
    MetakernelUnreachableError
        A re-fetch by name finds the source gone from the server (HTTP 403, 404 or 410).
    """
    with KernelDB(console=console) as db:
        if ref.startswith(("http://", "https://")):  # the CLI's test, cli/browse.py
            return db.get_metakernel(
                ref, download_dir=download_dir, mission=mission, yes=yes, force=force,
            )
        return db.update_metakernel(
            ref, mission=mission, download_dir=download_dir, yes=yes, force=force,
        )


def scan(
    directory: str | Path | None = None, *, mission: str | None = None,
) -> int:
    """Index the kernel files under *directory*, or under every configured root.

    Opens the database named in the config, does one thing and closes it. This function calls
    ``KernelDB.scan_directory``, or ``KernelDB.scan_roots`` when *directory* is ``None``.

    Parameters
    ----------
    directory : str | Path | None, default None
        The directory to index. ``None`` indexes every configured kernel root.
    mission : str | None, default None
        File everything under this mission instead of detecting it from the path. It applies only
        together with *directory*. Scanning the configured roots always detects the mission.

    Returns
    -------
    int
        The number of files registered.
    """
    with KernelDB() as db:
        if directory is None:
            count, _ = db.scan_roots()
        else:
            count, _ = db.scan_directory(directory, mission=mission)
        return count


def register_metakernel(
    path: str | Path, *, mission: str | None = None, official: bool = True,
) -> dict:
    """Track a metakernel already on disk, as ``skd import`` does.

    Opens the database named in the config, does one thing and closes it. This function calls
    ``KernelDB.import_metakernel``. It indexes the metakernel's entries, registers every kernel it
    resolves to that exists on disk, and adds the registry row with no source URL.

    Parameters
    ----------
    path : str | Path
        The ``.tm`` file.
    mission : str | None, default None
        The mission. Defaults to the one guessed from the path, taking the configured kernel roots
        into account.
    official : bool, default True
        ``False`` marks a metakernel assembled locally, such as the output of NAIF's subsetter, so
        that ``ls`` and ``show`` say so.

    Returns
    -------
    dict
        With the keys ``mk_path``, ``filename``, ``mission``, ``n_entries``, ``n_registered`` and
        ``missing``, the entries whose file is not on disk.

    Raises
    ------
    FileNotFoundError
        *path* is not a file.
    """
    with KernelDB() as db:
        return db.import_metakernel(path, mission=mission, official=official)
