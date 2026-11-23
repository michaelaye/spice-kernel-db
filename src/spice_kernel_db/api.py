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

    *ref* is a path, a registry filename, a version-stripped base name or a
    source URL (``KernelDB.resolve_metakernel``). Raises
    ``MetakernelNotTracked`` or ``AmbiguousMetakernel``.
    """
    with KernelDB(read_only=True) as db:
        return db.resolve_metakernel(ref, mission=mission)


def metakernels_covering(
    body_id: int, et: float | None = None, mission: str | None = None,
) -> list[dict]:
    """Tracked metakernels whose SPK kernels cover *body_id* (at *et*, if given)."""
    with KernelDB(read_only=True) as db:
        return db.metakernels_covering(body_id, et=et, mission=mission)


def check(
    ref: str | Path, *, mission: str | None = None,
    hash_: bool = False, remote: bool = False, spice: bool = False,
) -> dict:
    """``skd check`` as data: the verdicts and their evidence (``KernelDB.check_report``)."""
    with KernelDB(read_only=True) as db:
        return db.check_report(ref, mission=mission, hash_=hash_, remote=remote, spice=spice)


def get(
    ref: str, *, mission: str | None = None, yes: bool = False,
    download_dir: str | Path | None = None, force: bool = False,
    console: Console | None = None,
) -> dict:
    """``skd get``: fetch a metakernel by URL, or re-fetch a tracked one by name.

    Without ``yes=True`` and without a *console* to ask on, nothing is
    downloaded and the result carries ``aborted: True``. A name that is not
    tracked raises ``MetakernelNotTracked``; looking it up on the mission's
    server is the CLI's job (``skd get <name>`` does that, with a picker).
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

    Returns the number of files registered.
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
    """Track a metakernel already on disk (``skd import``).

    ``official=False`` marks one assembled locally, such as the output of
    NAIF's subsetter, so ``ls`` and ``show`` say so.
    """
    with KernelDB() as db:
        return db.import_metakernel(path, mission=mission, official=official)
