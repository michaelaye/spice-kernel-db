# spice-kernel-db

Content-addressed SPICE kernel database for deduplication and metakernel rewriting across missions.

## The problem

NAIF SPICE kernel archives for different missions often ship identical copies of generic kernels (`naif0012.tls`, `pck00011.tpc`, `de432s.bsp`, etc.). If you work with multiple missions, you end up storing the same files many times. And when you want to use a metakernel (`.tm`) from one mission, it expects all kernels to live under a specific directory tree — even if you already have the files downloaded for another mission.

## What this tool does

1. **Deduplication**: Identifies identical kernel files across missions using SHA-256 hashing. Same content = same hash, regardless of filename (handles cases like `jup365.bsp` vs `jup365_19900101_20500101.bsp`).

2. **Metakernel rewriting**: Rewrites `.tm` files for local use with **minimal edits** — only `PATH_VALUES` is changed, everything else (header comments, `KERNELS_TO_LOAD` entries) stays identical to the original. A symlink tree bridges the gap between where the metakernel expects files and where they actually live on disk.

3. **Mission-aware resolution**: When resolving kernel paths, prefers copies from the same mission. Falls back to other missions with a clear warning.

## Installation

```bash
pip install spice-kernel-db
```

Or from source:

```bash
git clone https://github.com/michaelaye/spice-kernel-db
cd spice-kernel-db
pip install -e ".[dev]"
```

## Quick start

### Python API

```python
from spice_kernel_db import KernelDB

db = KernelDB("~/.spice_kernels.duckdb")

# Register your local kernel trees
db.scan_directory("/data/spice/generic_kernels", mission="generic")
db.scan_directory("/data/spice/JUICE/kernels")
db.scan_directory("/data/spice/MRO/kernels")

# See what's duplicated
db.report_duplicates()

# Check a metakernel: what do you already have, what's missing?
result = db.check_metakernel(
    "juice_crema_5_1_150lb_23_1_a3.tm",
    mission="JUICE",
)
# result["found"], result["missing"], result["warnings"]

# Rewrite metakernel for local use (creates symlink tree,
# only changes PATH_VALUES in the .tm file)
db.rewrite_metakernel(
    "juice_crema_5_1_150lb_23_1_a3.tm",
    output="juice_local.tm",
    mission="JUICE",
    link_root="/data/spice/unified_kernels",
)

# Replace duplicate files with symlinks to save disk space
db.deduplicate_with_symlinks(dry_run=True)   # preview first
db.deduplicate_with_symlinks(dry_run=False)   # do it
```

### CLI

```bash
# Scan directories
spice-kernel-db scan /data/spice/generic_kernels --mission generic
spice-kernel-db scan /data/spice/JUICE/kernels
spice-kernel-db scan /data/spice/MRO/kernels

# View stats and duplicates
spice-kernel-db stats
spice-kernel-db duplicates

# Check metakernel availability
spice-kernel-db check juice_crema_5_1.tm --mission JUICE

# Rewrite metakernel (creates symlink tree + rewritten .tm)
spice-kernel-db rewrite juice_crema_5_1.tm -o juice_local.tm

# Resolve a single kernel filename
spice-kernel-db resolve naif0012.tls --mission JUICE

# Deduplicate (dry run by default)
spice-kernel-db dedup
spice-kernel-db dedup --execute
```

## How it works

### Content-addressed storage

Every kernel file is identified by its SHA-256 hash. The DuckDB database has two core tables:

- **`kernels`**: `(sha256, filename, kernel_type, size_bytes)` — one row per unique file content
- **`locations`**: `(sha256, abs_path, mission, source_url)` — one row per place a file exists on disk

When `jup365.bsp` (from `generic_kernels`) and `jup365_19900101_20500101.bsp` (from JUICE) have identical content, they share the same `sha256` in `kernels` but have separate entries in `locations`.

### Mission-aware resolution

`resolve_kernel("naif0012.tls", preferred_mission="JUICE")` follows this priority:

1. Exact filename match in JUICE locations → ✅ no warning
2. Exact filename match in any other mission → ⚠️ warning
3. Fuzzy match (same hash, different name) in JUICE → ⚠️ warning
4. Fuzzy match in any other mission → ⚠️ warning
5. Not found → returns `None`

### Minimal metakernel edits

The `rewrite_metakernel()` command:

1. Parses the original `.tm` file
2. For each `$KERNELS/type/filename` entry, resolves the filename to a local path
3. Creates a symlink at `link_root/type/filename` → actual file location
4. Writes a new `.tm` where **only `PATH_VALUES` is changed** to point to `link_root`

The `KERNELS_TO_LOAD` list, `PATH_SYMBOLS`, and all comments/header text remain identical to the original.

## Database location

By default the database lives at `~/.spice_kernels.duckdb`. Override with `--db` on the CLI or the `db_path` constructor argument.

## Dependencies

- Python ≥ 3.10
- [DuckDB](https://duckdb.org/) ≥ 1.0

## License

MIT
