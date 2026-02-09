# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run full test suite
pytest tests/

# Run a single test class or test
pytest tests/test_kernel_db.py::TestResolveKernel
pytest tests/test_kernel_db.py::TestResolveKernel::test_exact_match_preferred_mission

# Preview documentation locally
cd docs && quarto preview
```

There is no linter or formatter configured. The CI workflow only builds and deploys Quarto docs to GitHub Pages.

## Architecture

Content-addressed SPICE kernel database. Every kernel file is identified by its SHA-256 hash, not its filename. Multiple filesystem locations (across missions) can reference the same hash, enabling deduplication and cross-mission reuse.

### DuckDB Schema (4 tables)

- **`kernels`** — unique file content: `(sha256 PK, filename, kernel_type, size_bytes)`
- **`locations`** — where files live on disk: `(sha256 + abs_path PK, mission, source_url, scanned_at)` → references `kernels`
- **`metakernel_entries`** — kernel entries within a `.tm` file: `(mk_path + entry_index PK, raw_entry, filename)`
- **`metakernel_registry`** — tracked metakernel files: `(mk_path PK, mission, source_url, filename, acquired_at)`

### Module Responsibilities

| Module | Role |
|--------|------|
| `db.py` | `KernelDB` class — all DB operations, scanning, resolution, metakernel rewriting, get, browse |
| `parser.py` | `ParsedMetakernel` dataclass; parse/write `.tm` files; symbol resolution |
| `hashing.py` | `sha256_file()`, `classify_kernel()` (extension→type), `guess_mission()` (path heuristic) |
| `remote.py` | Network operations: fetch metakernels, resolve URLs, parallel HEAD/download with rich.progress |
| `config.py` | `Config` dataclass, TOML persistence at `~/.config/spice-kernel-db/config.toml` |
| `cli.py` | argparse CLI dispatching to `KernelDB` methods |

### Key Design Decisions

**Mission-aware resolution** (`resolve_kernel`) uses a 4-step fallback:
1. Exact filename in preferred mission
2. Exact filename in any mission
3. Fuzzy prefix match in preferred mission
4. Fuzzy prefix match in any mission

Each fallback step emits a warning. Fuzzy matching handles NAIF naming variations like `jup365.bsp` vs `jup365_19900101_20500101.bsp`.

**Metakernel rewriting** only modifies `PATH_VALUES` — the header, `PATH_SYMBOLS`, `KERNELS_TO_LOAD`, and all comments are preserved verbatim. A symlink tree bridges between where the metakernel expects files and where they actually live.

**Remote browsing** parses Apache `mod_autoindex` HTML directory listings. Version tags like `_v461_20251127_001` are stripped to group versioned snapshots under their base metakernel name.

## Testing Patterns

Tests use `pytest-tmp-files` for temporary directory fixtures. The main fixture `tmp_spice_tree` creates a realistic multi-mission SPICE directory tree with known duplicates. Network operations are mocked with `unittest.mock.patch`. Metakernel content is written as multi-line strings in tests.

## Release Process

**IMPORTANT: Always update the changelog when bumping the version. Never commit a version bump without a corresponding changelog entry.**

1. Update version in `pyproject.toml`
2. Add entry to `CHANGELOG.md` (Keep a Changelog format)
3. Add comparison link at bottom of changelog
4. Commit as "Bump version to X.Y.Z and add changelog"
5. Build and publish: `rm -rf dist/ && python -m build && python -m twine upload dist/*`
