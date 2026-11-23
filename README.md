# spice-kernel-db

[![CI](https://github.com/michaelaye/spice-kernel-db/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelaye/spice-kernel-db/actions/workflows/ci.yml)

Browse, get, and manage SPICE kernels and metakernels across NASA and ESA mission archives.

> **⚠️ 0.30 cleaned up the command line.** If you used spice-kernel-db before
> 0.30, these are the changes you will meet (the version jumped from 0.22 to
> mark them):
>
> - **Verbs were renamed:** `list` → `show`, `metakernels`/`mk` → `ls`,
>   `resolve` and `aliases` → `which`, `duplicates` → `stats`, `verify` → `check`,
>   `prune` → `clean`. The old names still run, print a one-line note, and go
>   away in 1.0.
> - **`update` walks every tracked metakernel** and reports; it downloads only
>   with `--execute`. `get <name>` re-fetches one.
> - **Several kernel roots:** `config.toml` holds `kernel_dirs = [...]` (a file
>   with the old `kernel_dir` still loads). Where a download lands is derived
>   from what is already on disk; there is no download target to configure.
> - **Scripts can read the exit code:** `check` returns a sum of bits (1 a
>   kernel missing, 2 an entry does not resolve, 4 differs from the server),
>   an untracked name is 1, an ambiguous name or root is 2, a broken config
>   file is 78; a missing argument shows a picker at a terminal and exits 2
>   without one.
> - **The library is quiet** and `spice_kernel_db.__all__` is its public API:
>   `KernelDB(console=None)` prints nothing and downloads only with `yes=True`.
>
> The [CHANGELOG](https://github.com/michaelaye/spice-kernel-db/blob/main/CHANGELOG.md#0300---2026-11-23)
> has the full list; the [docs](https://michaelaye.github.io/spice-kernel-db/)
> are now generated from the code.

> **⚠️ CRITICAL: Update to v0.10.0 or later.** Versions before 0.10.0 contained a
> fuzzy filename matching bug that could silently create symlinks between
> *completely different* SPICE kernel files, causing **silent scientific data
> corruption** — SPICE loads valid but *wrong* data with no errors. **All users
> who used `spice-kernel-db get` or `spice-kernel-db update` are affected**,
> regardless of the dedup setting — the symlink creation code did not honor the
> per-mission dedup flag (also fixed in 0.10.0). Users who only used `scan` and
> single-kernel lookup (`which` today) are not affected. After updating,
> remove all symlinks in your kernel directories and re-run
> `spice-kernel-db update --execute`. See [CHANGELOG.md](https://github.com/michaelaye/spice-kernel-db/blob/main/CHANGELOG.md)
> for recovery steps.

## What this tool does

1. **Mission setup**: Configure missions from NASA NAIF or ESA SPICE servers with an interactive dialog.

2. **Browse & get**: Browse available metakernels for a mission, then get one — the tool downloads all missing kernels automatically and makes the metakernel ready to use locally.

3. **Metakernel rewriting**: Rewrites `.tm` files for local use with **minimal edits** — only `PATH_VALUES` is changed, everything else stays identical to the original. A symlink tree bridges the gap between where the metakernel expects files and where they actually live on disk.

4. **Deduplication** (optional): Identifies identical kernel files across missions using SHA-256 hashing and replaces duplicates with symlinks. Per-mission opt-in — you can deduplicate some missions while keeping others untouched.

## Documentation

Full documentation is at [michaelaye.github.io/spice-kernel-db](https://michaelaye.github.io/spice-kernel-db/) and built with [Quarto](https://quarto.org/).

## Installation

```bash
pip install spice-kernel-db
```

Or with conda:

```bash
conda install -c michaelaye spice-kernel-db
```

Or from source:

```bash
git clone https://github.com/michaelaye/spice-kernel-db
cd spice-kernel-db
pip install -e ".[dev]"
```

## Quick start

### Set up a mission

```bash
spice-kernel-db mission add
```

Interactive dialog: choose a server (NASA NAIF / ESA SPICE) → pick a mission from the list → configure deduplication preference.

### Browse available metakernels

```bash
spice-kernel-db browse JUICE
```

Shows all `.tm` files in the mission's remote `mk/` directory, grouped by base name with version counts.

### Get a metakernel

```bash
spice-kernel-db get juice_ops.tm
```

Downloads the metakernel, checks which kernels you already have, downloads the missing ones in parallel, and creates symlinks so the `.tm` file works immediately.

### Use with spiceypy

```python
import spiceypy as spice
from spice_kernel_db import KernelDB

db = KernelDB()
mks = db.list_metakernels(mission="JUICE")
spice.furnsh(mks[0]["mk_path"])
```

### Python API

```python
from spice_kernel_db import KernelDB

db = KernelDB()

# Browse remote metakernels
db.browse_remote_metakernels(
    "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/",
    mission="JUICE",
)

# Get a metakernel (downloads missing kernels automatically)
db.get_metakernel(
    "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/juice_ops.tm",
    mission="JUICE",
)

# Optionally deduplicate across missions
db.deduplicate_with_symlinks(dry_run=True)   # preview
db.deduplicate_with_symlinks(dry_run=False)  # execute
```

## Supported servers

| Server | URL |
|--------|-----|
| NASA NAIF | `https://naif.jpl.nasa.gov/pub/naif/` |
| ESA SPICE | `https://spiftp.esac.esa.int/data/SPICE/` |

Both use the same `<server>/<MISSION>/kernels/mk/` directory structure.

## Dependencies

- Python >= 3.11
- [DuckDB](https://duckdb.org/) >= 1.0
- [rich](https://rich.readthedocs.io/) >= 13.0

## License

MIT
