# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0] - 2026-03-25

### Added

- **`--version` flag** — `spice-kernel-db --version` now prints the installed version.
- **`skd` short alias** — `skd` is now a second entry point for `spice-kernel-db`, for quicker invocation.
- **No-args summary** — running `spice-kernel-db` (or `skd`) with no subcommand now shows locally acquired metakernels and a quick-start guide instead of just the help text.
- **`mk --remove` flag** — `spice-kernel-db mk --remove <name>` removes a metakernel from the registry and its entries. Does not delete files from disk.
- **`prune` command** — `spice-kernel-db prune` finds and removes stale DB entries for files that no longer exist on disk. Also cleans up orphaned kernel records with no remaining locations. Dry-run by default (`--execute` to apply).
- **Pre-release review document** — `docs/review.qmd` captures the full CLI/API/docs audit.

### Changed

- **Deferred config loading** — `ensure_config()` is no longer called before argument parsing. `--help`, `--version`, and no-args invocation now work on a fresh install without triggering the interactive setup wizard.
- **`update_metakernel()` raises `LookupError`** instead of calling `sys.exit(1)`. The library layer (`db.py`) no longer calls `sys.exit()` — all exit decisions are made in the CLI layer (`cli.py`). This makes `KernelDB` safe to use as a Python API.
- **`_resolve_body_interactive()` returns `None`** on unknown bodies and invalid selections instead of calling `sys.exit(1)`.
- **`browse` (no args)** now picks a mission interactively and browses its remote metakernels, instead of duplicating the `mission list` table.
- Development Status classifier upgraded from Alpha to Beta.
- Test suite expanded from 136 to 150 tests.

### Fixed

- **Documentation overhaul** — removed all references to removed fuzzy filename matching from design.qmd, api.qmd, and CLAUDE.md. Updated API signatures for `scan_directory`, `register_file`, `get_metakernel`, `check_metakernel`, `browse_remote_metakernels`, and `KernelDB` constructor. Added missing `update_metakernel` API docs. Fixed Python version requirement in README (3.11, not 3.10). Updated Quarto site description. Added `kernel_coverage` table to schema docs.

## [0.10.1] - 2026-03-20

### Fixed

- **Dedup flag now enforced** — `_link_existing_kernels()` in `get_metakernel()` now checks the per-mission `dedup` setting before creating symlinks. Previously the dedup flag was ignored, and symlinks were created unconditionally for all missions.
- **Corrected warning text** — warnings in `get_metakernel()` and `update_metakernel()` now accurately describe the operation being performed.

## [0.10.0] - 2026-03-20

### ⚠️ CRITICAL UPDATE — Data Corruption Bug

**All users should update immediately.** Versions prior to 0.10.0 contained a
fuzzy filename matching algorithm in `resolve_kernel()` that could silently
create symlinks between completely different SPICE kernel files. This caused
**silent scientific data corruption** — SPICE loaded valid but *wrong* kernel
data (e.g., a different mission trajectory), producing plausible but incorrect
results with no error messages.

**Who is affected:** All users who used `get` or `update` are affected,
regardless of the per-mission dedup setting — the symlink creation code
(`_link_existing_kernels`) did **not** honor the `dedup` flag and ran
unconditionally. The fuzzy match could link to the wrong file when kernel
filenames shared a common prefix (e.g., all JUICE kernels start with `juice_`).
Users who only used `scan` and `resolve` are **not affected**. v0.10.0 also
fixes the dedup flag being ignored, so `dedup=False` missions now download
all kernels fresh without symlinks.

**Impact:** Any kernel directory managed by `spice-kernel-db` prior to 0.10.0
may contain corrupted symlinks. After updating, run:

```bash
# 1. Remove all symlinks (they may point to wrong kernels)
find /path/to/your/kernels -type l -delete
# 2. Re-download missing kernels
spice-kernel-db update
# 3. Verify all kernels load correctly
spice-kernel-db check <your-metakernel.tm>
```

### Security

- **CRITICAL: Path traversal via metakernel PATH_VALUES** — `get_metakernel()` and `update_metakernel()` now validate that resolved PATH_VALUES stay within the download directory. Malicious metakernels with paths like `/../../../tmp/evil` are rejected with a `ValueError`. New `_validate_path_values()` helper.
- **CRITICAL: Size-only skip check for existing files** — Step 7 of `get_metakernel()` now verifies SHA-256 hash in addition to file size before skipping a download. A file with correct size but wrong content is no longer silently accepted. New `_should_skip_download()` helper.
- **CRITICAL: Symlink creation without hash validation** — `_link_existing_kernels()` now computes the SHA-256 of the resolved file and verifies it matches the database record before creating a symlink. Mismatched files are skipped with a warning.
- **CRITICAL: No partial download detection** — `download_kernel()` now tracks bytes written vs the `Content-Length` header. Incomplete downloads raise `IOError` and the partial file is deleted. Zero-byte downloads are also rejected.
- **HIGH: Race condition — DB state change during download** — After each `release()`/`reacquire()` cycle in `get_metakernel()`, kernel records are compared to a pre-release snapshot. A warning is logged if another process modified the database during the lock release.
- **HIGH: Symlink creation errors silently ignored** — `_link_existing_kernels()` now wraps `symlink_to()` in `try/except OSError`. Failed symlinks log a warning and do not increment the link counter.
- **HIGH: No pre-registration hash verification** — `register_file()` accepts an optional `expected_hash` parameter. When provided, the computed hash is verified before storing. Raises `ValueError` on mismatch.
- **MEDIUM: Broad except Exception masks real errors** — `scan_directory()` now catches `OSError` specifically (excluding `PermissionError`) and re-raises unexpected exceptions. Error count is reported at the end of the scan.
- **MEDIUM: Path traversal in rewrite_metakernel** — `rewrite_metakernel()` now validates that each resolved link path stays within `link_root`. Paths that escape are skipped with a warning.
- **MEDIUM: Thread pool download failures — no severity distinction** — `download_kernels_parallel()` now classifies exceptions by severity: `[FATAL]` for HTTP 403/404, `[RETRIABLE]` for HTTP 5xx and network errors, `[ERROR]` for disk/IO problems.
- **MEDIUM: Same filename, different hash silently ignored** — `register_file()` now detects when a filename already exists with a different hash and inserts a new record for the updated content. The old hash record is preserved for history. A warning is logged about the version change.

### Added

- **`list` command** — `spice-kernel-db list [metakernel]` lists all kernels in a metakernel, grouped by type. Supports `--type` filter. Without arguments, shows a picker from registered metakernels.
- **`--force` flag** for `get` and `update` — re-downloads all kernels regardless of what's on disk, skipping DB lookups and remote size queries for faster startup.
- **DB lock release during downloads** — the DuckDB connection is released during network I/O (size queries and file downloads), allowing other processes to run read-only commands (`resolve`, `list`, `check`, `stats`) while a download is in progress.
- **Read-only DB mode** — read-only commands now open the database in read-only mode for better concurrent access.
- **Post-update rescan** — `update` now automatically rescans the kernel directories referenced by the metakernel's PATH_VALUES, so newly downloaded kernels are immediately indexed without manual `scan`.
- **Resolve failure hint** — `resolve` now suggests running `scan` when a kernel is not found.

### Removed

- **Fuzzy filename matching** — the `_find_by_filename_any_alias()` method has been removed entirely. `resolve_kernel()` now uses exact filename match and path-suffix match only. This eliminates the class of bugs where unrelated kernels with similar name prefixes were incorrectly linked.

### Changed

- Test suite expanded from 114 to 136 tests.
- Warning messages in `resolve_kernel()` now correctly say "matched by filename prefix" instead of the misleading "matched by hash".

## [0.9.1] - 2026-02-20

### Added

- Conda package on anaconda.org/michaelaye — `conda install -c michaelaye spice-kernel-db`.
- `conda/meta.yaml` recipe committed to repo for reproducible builds.
- Conda install instructions in README and docs.
- Full release pipeline documented in CLAUDE.md (PyPI + GitHub release + conda + anaconda upload).

## [0.9.0] - 2026-02-20

### Added

- `update` command — re-fetch a metakernel from its source URL and download new/missing kernels. Resolves the URL from the metakernel registry so you don't have to remember it.
- `mk` alias for `metakernels` command.
- `browse` 3-state local indicator: `yes` (up to date), `outdated` (remote newer than acquired), `no` (not acquired). Prints a hint when outdated metakernels are found.
- `check` remote staleness notice — queries the remote server and prints a notice when a newer version is available (read-only, no automatic download).
- `check -v` / `--verbose` flag for full per-file warnings. Default output now shows a compact summary (dedup count, cross-mission count) instead of flooding with per-file lines.
- Categorized `--help` output — subcommands grouped into browse & acquire, inspect, transform, and configure sections with color-coded command names.
- Interactive mission picker — `get` and `browse` without `--mission` now show a numbered mission table instead of an error when multiple missions are configured.

### Changed

- `get` output uses `rich.Panel` for summary and result sections instead of scattered `print` calls.
- `browse` without arguments uses `rich.Table` for the mission listing.
- Interactive selectors (`get`, `check`, `update`, `coverage`) use styled `console.print` throughout for consistent formatting.
- CLI docs (`docs/cli.qmd`) updated for all new features.

## [0.8.1] - 2026-02-19

### Fixed

- Fuzzy kernel resolution no longer matches across file types. Previously, searching for a `.bsp` file could match a `.tm` metakernel with the same name prefix (e.g. `juice_crema_5_1_150lb_23_1_a3_2_v01.bsp` resolved to `juice_crema_5_1_150lb_23_1_a3_2.tm`). This caused `coverage` to feed a text metakernel to SpiceyPy's `spkcov` as if it were a binary SPK, triggering `SPICE(INVALIDARCHITECTURE)`. The fuzzy query now filters by file extension (case-insensitive), so `.bsp` queries only match `.bsp` files.

## [0.8.0] - 2026-02-19

### Added

- `coverage` CLI command and `coverage_metakernel()` API for checking SPK body coverage in metakernels. Uses SpiceyPy's `spkcov` to report which SPK files contain data for a given NAIF body, including gap detection.
- `kernel_coverage` database table caching body coverage intervals per kernel (content-addressed by SHA-256, reusable across sessions).
- Body name resolution — `coverage` accepts ASCII names like `Earth` or `3i/atlas` in addition to numeric NAIF IDs. Ambiguous names (e.g. "earth" → body center vs barycenter) prompt for disambiguation.
- `NAIF_BODIES` lookup table with 30+ common solar system bodies.
- `store_coverage()` and `query_coverage()` API methods for persisting and querying coverage data.
- SpiceyPy as optional dependency: `pip install spice-kernel-db[spice]`
- Coverage documentation page (`docs/coverage.qmd`)
- Interactive metakernel selection for `check` and `coverage` commands — omit the metakernel argument to pick from locally acquired metakernels, with hints to `browse`/`get` when none are available.

### Changed

- Quarto documentation theme: sandstone (light) + superhero (dark) with dark-mode table styling.
- `check` metakernel argument is now optional (interactive selection when omitted).
- CLI helpers refactored: `_require_metakernel()` and `_resolve_body_interactive()` extracted to avoid code duplication.
- Coverage time display uses compact `YYYY-MON-DD HR:MN` format instead of fractional seconds.
- Coverage table Kernel column uses `no_wrap` for full filename visibility.
- Coverage table error messages truncated to key SPICE error instead of full traceback.
- Test suite expanded from 95 to 114 tests.

## [0.7.1] - 2026-02-09

### Fixed

- `KernelDB()` now reads the configured database path from `config.toml` instead of defaulting to a hardcoded `~/.spice_kernels.duckdb`. Previously the Python API connected to a different database than the CLI, causing `list_metakernels()` to return empty results.
- `get_metakernel` now writes absolute `PATH_VALUES` in saved `.tm` files. SPICE resolves paths relative to the current working directory, not the metakernel file location, so the original relative `'..'` only worked if you happened to `cd` into the `mk/` directory. Metakernels now work with `spice.furnsh()` from any directory.

## [0.7.0] - 2026-02-09

### Changed

- `mission add` now probes each mission for a `kernels/mk/` directory before listing, using parallel HEAD requests. Missions without metakernels are shown separately as "not yet supported" with a reference to [#2](https://github.com/michaelaye/spice-kernel-db/issues/2). This avoids configuring missions that cannot actually be browsed or downloaded.
- Metakernel file matching is now case-insensitive on the `.tm` extension — ESA missions like MEX that use uppercase `.TM` filenames are now correctly parsed, including version tag extraction (e.g. `_V324_20260206_002.TM`).
- `base_name` for versioned metakernels now preserves the original extension case instead of forcing lowercase `.tm`.
- Test suite expanded from 92 to 95 tests.

## [0.6.0] - 2026-02-08

### Added

- Interactive metakernel selection for `get` — run without arguments to pick from a numbered menu
- `rich` library for all output formatting (tables, panels, progress bars)
- Dependency roadmap section in design docs

### Changed

- Replaced `tqdm` with `rich.progress` for download and query progress bars
- Converted all 8 manual f-string tables to `rich.Table` / `rich.Panel`
- `mission remove` now uses case-insensitive prefix matching (consistent with other commands)
- `metakernels --mission` filter now uses case-insensitive prefix matching
- `browse` local status query now matches mission names case-insensitively
- Test suite expanded from 89 to 92 tests

### Fixed

- `mission remove mro` no longer fails when the stored name is `MRO`
- `metakernels --mission bepi` no longer returns empty when stored as `BEPICOLOMBO`

## [0.5.0] - 2026-02-06

### Added

- Multi-server support: `mission add` interactive setup for NASA NAIF and ESA SPICE servers
- `mission list` and `mission remove` commands for managing configured missions
- Per-mission deduplication control via `missions` table `dedup` column
- Case-insensitive prefix matching for mission names (e.g. `bepi` matches `BEPICOLOMBO`)
- `.spice-server` marker files written during `get` for automatic mission detection on scan
- `scan` now auto-detects unconfigured missions and prompts for server selection
- `scan` indexes `.tm` files into `metakernel_registry` automatically
- Symlinks for "in db" kernels after `get` so metakernels work immediately
- Skip already-downloaded files during `get` (size-based comparison)
- Byte-level download progress bar with `tqdm` for detailed feedback
- ESA SPICE server HTML parsing (table-based Apache directory listings)
- Mission auto-detection from `MISSION/<kernel_type>/` directory layout (no `kernels/` required)
- Hero SVG redesigned as three-panel Browse → Get → Use workflow

### Changed

- Renamed `acquire` CLI command to `get` (shorter, less ambiguous)
- Renamed `acquire_metakernel()` API to `get_metakernel()`
- `get_metakernel()` resolves abbreviated mission names to canonical form before storing
- `scan_directory()` returns `(count, missions_found)` tuple
- Removed backward-compatibility code (`list_known_mk_dirs`, fallback lookups)
- `--show-versioned` hint updated to explain versioned snapshot metakernels
- Metakernel auto-adaptation is now the default (not optional rewrite step)
- Test suite expanded from 62 to 89 tests

### Fixed

- ESA server directory listings now parsed correctly (both NASA and ESA HTML formats)
- ESA `.tm` file listings parsed correctly (table-based format)
- Mission detection for directories without `kernels/` subdirectory (e.g. `BEPICOLOMBO/ck/`)

## [0.4.0] - 2026-02-06

### Added

- `browse` CLI command to scan remote NAIF `mk/` directories, grouping
  metakernels by base name and showing version counts and local status
- `browse` without arguments lists known `mk/` directory URLs from prior acquires
- `browse` accepts mission names (e.g. `browse JUICE`) in addition to full URLs
- `--show-versioned` flag on `browse` to display individual versioned snapshots
- `acquire` now accepts filenames (e.g. `acquire juice_ops.tm`) resolving via
  known `mk/` directories, with `--mission` for disambiguation
- `list_known_mk_dirs()` API for querying previously seen remote directories
- `browse_remote_metakernels()` API with compact and expanded display modes
- `list_remote_metakernels()` and `RemoteMetakernel` dataclass for parsing
  Apache directory listings
- `tqdm` dependency for progress bars during parallel downloads and size queries
- `download_kernels_parallel()` for concurrent kernel downloads with progress

### Changed

- Parallel downloads use `tqdm.contrib.concurrent.thread_map` instead of
  manual `ThreadPoolExecutor` loops
- Removed redundant "Available" column from `metakernels` listing (acquired
  metakernels always have all kernels downloaded)
- `info_metakernel()` uses `resolve_kernel()` for consistent availability checks
- Test suite expanded from 49 to 62 tests

## [0.3.0] - 2026-02-06

### Added

- `metakernels` CLI command and `list_metakernels()` / `info_metakernel()` API
  for listing tracked metakernels with kernel counts and per-kernel status
- `metakernel_registry` DB table tracking acquired metakernel files (mission,
  source URL, local path, acquisition timestamp)
- `acquire` now saves the `.tm` file to disk and registers it in the database
- `config` CLI command to show current settings or re-run interactive setup
- `reset` CLI command to delete the database (kernel files are preserved)
- `--archive` flag on `scan` to move kernels into the configured archive
  directory and leave symlinks at original locations
- `show_config()` helper in config module

### Changed

- Renamed `metakernels` DB table to `metakernel_entries` for clarity (the table
  stores individual kernel entries within a metakernel, not metakernels themselves)
- `reset` and `config` commands run without opening a DB connection
- Test suite expanded from 36 to 49 tests

## [0.2.0] - 2026-02-06

### Added

- `acquire` CLI command and `acquire_metakernel()` API for fetching remote
  metakernels, displaying a status table, and downloading missing kernels
- First-run interactive configuration saved to
  `~/.config/spice-kernel-db/config.toml`
- `parse_metakernel_text()` for parsing metakernels from string content
  (e.g. fetched from URLs)
- `remote.py` module with URL resolution, parallel HTTP HEAD size queries,
  and kernel download (stdlib only, no new dependencies)
- GitHub Actions workflow for deploying Quarto documentation to GitHub Pages

### Changed

- CLI now uses configured defaults from `config.toml` instead of hardcoded
  `~/.spice_kernels.duckdb`

## [0.1.0] - 2026-02-05

### Added

- Content-addressed SPICE kernel database backed by DuckDB
- SHA-256 hashing for deduplication across missions
- Mission-aware kernel resolution with fuzzy filename matching
- Metakernel parser with `PATH_VALUES`, `PATH_SYMBOLS`, `KERNELS_TO_LOAD`
  extraction
- Metakernel rewriting with symlink tree strategy (minimal edits, only
  `PATH_VALUES` changed)
- Deduplication with symlinks (dry-run by default)
- CLI commands: `scan`, `stats`, `duplicates`, `check`, `rewrite`, `dedup`,
  `resolve`
- Quarto documentation site (motivation, design, usage guide, CLI and API
  reference)
- Comprehensive test suite (30 tests)

[0.11.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.10.1...v0.11.0
[0.10.1]: https://github.com/michaelaye/spice-kernel-db/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.9.1...v0.10.0
[0.9.1]: https://github.com/michaelaye/spice-kernel-db/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/michaelaye/spice-kernel-db/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/michaelaye/spice-kernel-db/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/michaelaye/spice-kernel-db/releases/tag/v0.1.0
