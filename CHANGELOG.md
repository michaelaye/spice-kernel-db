# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.3.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/michaelaye/spice-kernel-db/releases/tag/v0.1.0
