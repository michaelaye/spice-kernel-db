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

There is no linter or formatter configured. CI runs pytest across Python 3.11–3.14; the publish workflow deploys Quarto docs to GitHub Pages (chained after CI passes).

## Architecture

Content-addressed SPICE kernel database. Every kernel file is identified by its SHA-256 hash, not its filename. Multiple filesystem locations (across missions) can reference the same hash, enabling deduplication and cross-mission reuse.

### DuckDB Schema (5 tables)

- **`kernels`** — unique file content: `(sha256 PK, filename, kernel_type, size_bytes)`
- **`locations`** — where files live on disk: `(sha256 + abs_path PK, mission, source_url, scanned_at)` → references `kernels`
- **`metakernel_entries`** — kernel entries within a `.tm` file: `(mk_path + entry_index PK, raw_entry, filename)`
- **`metakernel_registry`** — tracked metakernel files: `(mk_path PK, mission, source_url, filename, acquired_at)`
- **`missions`** — configured missions: `(name PK, server_url, mk_dir_url, dedup, added_at)`

### Module Responsibilities

| Module | Role |
|--------|------|
| `db.py` | `KernelDB` class — all DB operations, scanning, resolution, metakernel rewriting, get, browse |
| `parser.py` | `ParsedMetakernel` dataclass; parse/write `.tm` files; symbol resolution |
| `hashing.py` | `sha256_file()`, `classify_kernel()` (extension→type), `guess_mission()` (path heuristic) |
| `remote.py` | Network operations: fetch metakernels, resolve URLs, parallel HEAD/download with rich.progress |
| `config.py` | `Config` dataclass, TOML persistence at `~/.config/spice-kernel-db/config.toml` |
| `cli.py` | argparse CLI with interactive mission setup, metakernel selection, and dispatching to `KernelDB` |

### Key Design Decisions

**Mission-aware resolution** (`resolve_kernel`) uses a 4-step fallback:
1. Exact filename in preferred mission
2. Exact filename in any mission
3. Fuzzy prefix match in preferred mission
4. Fuzzy prefix match in any mission

Each fallback step emits a warning. Fuzzy matching handles NAIF naming variations like `jup365.bsp` vs `jup365_19900101_20500101.bsp`.

**Metakernel rewriting** only modifies `PATH_VALUES` — the header, `PATH_SYMBOLS`, `KERNELS_TO_LOAD`, and all comments are preserved verbatim. A symlink tree bridges between where the metakernel expects files and where they actually live.

**Remote browsing** parses Apache `mod_autoindex` HTML directory listings. Version tags like `_v461_20251127_001` are stripped to group versioned snapshots under their base metakernel name.

**Case-insensitive fuzzy matching** — all mission name lookups (get, browse, metakernels, mission remove) use case-insensitive prefix matching. `get_mission()` resolves via the `missions` table; SQL queries on `metakernel_registry` use `LOWER(mission) LIKE LOWER(?) || '%'` directly.

**Output formatting** uses `rich` throughout — `rich.Table` and `rich.Panel` for all tabular output, `rich.progress.Progress` for download and query progress bars (thread-safe with `ThreadPoolExecutor`).

## Testing Patterns

Tests use `pytest-tmp-files` for temporary directory fixtures. The main fixture `tmp_spice_tree` creates a realistic multi-mission SPICE directory tree with known duplicates. Network operations are mocked with `unittest.mock.patch`. Metakernel content is written as multi-line strings in tests.

## Release Process

**IMPORTANT: Always update the changelog when bumping the version. Never commit a version bump without a corresponding changelog entry.**

1. Commit all changes with descriptive messages
2. Update version in `pyproject.toml` (minor for features, patch for fixes)
3. Add entry to `CHANGELOG.md` (Keep a Changelog format) with comparison link at bottom
4. Commit as "Bump version to X.Y.Z: brief description"
5. Push to GitHub: `git push`
6. Publish to PyPI: `rm -rf dist/ && python -m build && python -m twine upload dist/*`
7. Create GitHub release: `gh release create vX.Y.Z --title "vX.Y.Z" --notes "..."`
8. Build conda package: `grayskull pypi spice-kernel-db` (in /tmp), then `conda-build /tmp/spice-kernel-db --output-folder /tmp/conda-output --no-anaconda-upload`
9. Upload to anaconda.org: `anaconda --site anaconda upload /tmp/conda-output/noarch/spice-kernel-db-X.Y.Z-py_0.conda`

**Notes:**
- The `--site anaconda` flag is required for `anaconda` commands (bypasses an interactive prompt that fails in non-TTY environments)
- `grayskull`, `conda-build`, and `anaconda-client` are installed in the `py314` conda env
- The conda recipe maintainer in `meta.yaml` should be `michaelaye` (grayskull defaults to a placeholder)
