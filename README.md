# spice-kernel-db

[![CI](https://github.com/michaelaye/spice-kernel-db/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelaye/spice-kernel-db/actions/workflows/ci.yml)

Browse, get, and manage SPICE kernels and metakernels across NASA and ESA mission archives.

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

- Python >= 3.10
- [DuckDB](https://duckdb.org/) >= 1.0
- [rich](https://rich.readthedocs.io/) >= 13.0

## License

MIT
