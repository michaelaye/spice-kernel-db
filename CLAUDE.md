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

# Build / preview the documentation (great-docs: generated API reference + docs/ guide)
make docs
make serve
```

There is no linter or formatter configured. CI runs pytest across Python 3.11–3.14; the publish workflow builds the site with great-docs and deploys it to GitHub Pages (chained after CI passes). The API reference is generated from docstrings (numpy style) for the names in `great-docs.yml` `reference:`, which `tests/test_api.py` keeps equal to `spice_kernel_db.__all__`; the guide pages are the numbered `docs/*.qmd` files, the tutorial notebook in `notebooks/` is converted by `make notebooks`.

## Architecture

Content-addressed SPICE kernel database. Every kernel file is identified by its SHA-256 hash, not its filename. Multiple filesystem locations (across missions) can reference the same hash, enabling deduplication and cross-mission reuse.

### DuckDB Schema (6 tables)

- **`kernels`** — unique file content: `(sha256 PK, filename, kernel_type, size_bytes)`
- **`locations`** — where files live on disk: `(sha256 + abs_path PK, mission, source_url, scanned_at)` → references `kernels`
- **`metakernel_entries`** — kernel entries within a `.tm` file: `(mk_path + entry_index PK, raw_entry, filename)`
- **`metakernel_registry`** — tracked metakernel files: `(mk_path PK, mission, source_url, filename, acquired_at, official)`; `official` is FALSE for a metakernel assembled locally (`import --unofficial`), and a tracked metakernel's mission is read from this row, never guessed from its path
- **`missions`** — configured missions: `(name PK, server_url, mk_dir_url, dedup, added_at)`
- **`kernel_coverage`** — cached SPK body coverage intervals: `(sha256 + body_id PK, et_start, et_end, source)`

### Module Responsibilities

| Module | Role |
|--------|------|
| `db.py` | `KernelDB` class — all DB operations, scanning, resolution, metakernel rewriting, get, browse. Writes only to the `console` it was given (none by default: library use is silent, #49); warnings go to the `spice_kernel_db` logger |
| `parser.py` | `ParsedMetakernel` dataclass; parse/write `.tm` files; symbol resolution |
| `hashing.py` | `sha256_file()`, `classify_kernel()` (extension→type), `guess_mission(filepath, roots=None)` (path heuristic; the first directory below a configured kernel root wins) |
| `remote.py` | Network operations: fetch metakernels, resolve URLs, parallel HEAD/download with rich.progress |
| `config.py` | `Config` (`kernel_dirs`, the list of kernel roots; no root is special, `KernelDB.root_for` derives where a download lands), TOML persistence at `~/.config/spice-kernel-db/config.toml`, `kernel_roots()` / `add_root()` / `remove_root()`; raises `ConfigError`, never prints or prompts |
| `api.py` | Module-level `get`, `check`, `scan`, `register_metakernel`, `resolve_metakernel`, `metakernels_covering`: one call against the configured database; `__init__.py`'s `__all__` is the public API (#49), mirrored in `docs/index.qmd` § Public API and in `great-docs.yml` `reference:`, both checked by `tests/test_api.py` |
| `cli/` | Typer CLI, one module per help panel: `browse.py` (browse, get, update), `inspect_cmds.py` (ls, show, which, coverage, stats, check, doctor, plus the hidden aliases list, metakernels/mk, resolve, aliases, duplicates, verify), `transform.py` (scan, import, rewrite, rm, clean, dedup, and the hidden prune), `configure.py` (mission, config, reset); `prompts.py` holds the shared interactive pickers, `_app.py` the `app` object and per-invocation state, `__init__.py` the root options and `main()` |

### Key Design Decisions

**Mission-aware resolution** (`resolve_kernel`) uses a 3-step fallback:
1. Exact filename in preferred mission
2. Exact filename in any mission
3. Path-suffix match (file exists on disk under a different canonical name)

Each fallback step emits a warning. Fuzzy prefix matching was removed in v0.10.0 due to silent data corruption risk.

Steps 1 and 2 go through `find_by_filename`, which joins on `sha256`, so identical content under two names produces one `kernels` row and a `locations` row per name — and either name matches both. `_prefer_exact_name` puts the location actually named `filename` first, so a name always resolves to its own file; the differently-named copy is used only when nothing of that name is registered, with a warning (issue #9).

**Metakernel rewriting** only modifies `PATH_VALUES` — the header, `PATH_SYMBOLS`, `KERNELS_TO_LOAD`, and all comments are preserved verbatim. A symlink tree bridges between where the metakernel expects files and where they actually live.

**Remote browsing** parses Apache `mod_autoindex` HTML directory listings. Version tags like `_v461_20251127_001` are stripped to group versioned snapshots under their base metakernel name.

**Case-insensitive fuzzy matching** — all mission name lookups (get, browse, ls, mission remove) use case-insensitive prefix matching. `get_mission()` resolves via the `missions` table; SQL queries on `metakernel_registry` use `LOWER(mission) LIKE LOWER(?) || '%'` directly.

**Output formatting** uses `rich` throughout — `rich.Table` and `rich.Panel` for all tabular output, `rich.progress.Progress` for download and query progress bars (thread-safe with `ThreadPoolExecutor`).

## Development rules

**Docs ship with the feature, not after.** Every new CLI subcommand, every new flag on an existing subcommand, every new public method on `KernelDB`, and every new behavior visible to the user (new error class, new exit code, new output format) must land in the same change as the corresponding documentation update. Concretely:

- New or changed CLI subcommand → entry in `docs/05-cli.qmd` (table of options + at least one example).
- New CLI flag → row added to the relevant subcommand's options table in `docs/05-cli.qmd`.
- New public name (`__init__.py` export) → a numpy docstring that says everything a user needs, and an entry under `reference:` in `great-docs.yml` (the test fails otherwise). A new `KernelDB` method needs only the docstring.
- New user-visible error or recovery path → entry in `docs/08-troubleshooting.qmd`.
- New public function, or a change to how a script uses the library (quiet library, `yes=`, `console=`, exit codes) → the worked examples in `docs/04-python.qmd` and the Python snippets in `docs/03-usage.qmd` follow in the same change.
- New feature → `CHANGELOG.md` entry under the unreleased section (or current dev version).

A pull request that adds features without updating docs is incomplete. If you spot an existing gap while doing other work, fix it in the same change rather than filing a TODO.

**Issue fixes go straight to `main`.** No feature branch or pull request is needed for work on a tracked issue; commit on `main` with `Closes #N` and push (Michael, 2026-10-28).

## Testing Patterns

Tests use `pytest-tmp-files` for temporary directory fixtures. The main fixture `tmp_spice_tree` creates a realistic multi-mission SPICE directory tree with known duplicates. Network operations are mocked with `unittest.mock.patch`. Metakernel content is written as multi-line strings in tests.

## Release Process

**IMPORTANT: Always update the changelog when bumping the version. Never commit a version bump without a corresponding changelog entry.**

The version string lives in **three** places and all three move together:
`src/spice_kernel_db/_version.py`, `conda/recipe.yaml` (`context.version`) and `CITATION.cff`
(`tests/test_cli.py::test_version_is_the_same_in_all_three_places` fails otherwise).
`pyproject.toml` has no version of its own: hatchling reads `_version.py` at build time, and
the package imports it, so an editable install reports the current version without a reinstall.

0. **Audit the README and the guide pages against the release before tagging.** The README is the PyPI page and the docs home; read it top to bottom and fix every statement the release makes stale: how the docs are built, the verbs and config keys the examples use, the Python examples (quiet library: `yes=True`, `console=`), the dependency list, the version note. (Michael, 2026-11-23: 0.30.0 shipped a README that still said the docs were built with Quarto and Python examples that no longer downloaded anything.) Then walk the release's CHANGELOG section entry by entry: each change must show in the guide pages (`docs/*.qmd`), the README, and the LLM support files. Those are `llms.txt`, `llms-full.txt` and `skill.md`, generated by `make docs` from the docstrings and the hand-written `skill:` keys in `great-docs.yml` (`decision_table`, `gotchas`, `best_practices`). Edit the keys when a change alters what an agent should call or watch out for, run `make docs`, and read the generated `great-docs/_site/skill.md` and `llms.txt` before tagging (Michael, 2026-11-27).
1. Commit all changes with descriptive messages
2. Update the version in all three files above (minor for features, patch for fixes)
3. Add entry to `CHANGELOG.md` (Keep a Changelog format) with comparison link at bottom
4. Commit as "Bump version to X.Y.Z: brief description"
5. Push to GitHub: `git push`
6. Publish to PyPI: `rm -rf dist/ && python -m build && python -m twine upload dist/*`
7. Create GitHub release: `gh release create vX.Y.Z --title "vX.Y.Z" --notes-file <file> --verify-tag`
8. Put the **published** sdist's sha256 in `conda/recipe.yaml`, then build:
   ```bash
   curl -s https://pypi.org/pypi/spice-kernel-db/X.Y.Z/json \
     | jq -r '.urls[] | select(.packagetype=="sdist") | .digests.sha256'
   rattler-build build --recipe conda/recipe.yaml -c conda-forge --output-dir /tmp/skd-conda-out
   ```
9. Upload to anaconda.org:
   ```bash
   export ANACONDA_API_KEY=<token>
   rattler-build upload anaconda -o michaelaye /tmp/skd-conda-out/noarch/spice-kernel-db-X.Y.Z-*.conda
   ```

**Notes:**
- **Tag before building any artefact, and build from the tag.** A 0.22.x lesson:
  the Typer port sat on `main` under the already-released 0.22.0 version string,
  so a build from `main` would have published something the tag and the Zenodo
  record did not describe. 0.22.1 then shipped a tag containing a stray build
  directory that `git add -A` swept in. Check `git status` and the sdist's
  contents before uploading, not after.
- **`rattler-build` replaced `conda-build` here** (2026-10-26). It is a
  standalone Rust binary with no Python or conda-build dependency, so nothing
  has to live in `base` or in a work env. Installed via
  `conda install -n base -c conda-forge rattler-build` (the earlier pixi install was gone by 2026-11-23), so the binary is `~/miniforge3/bin/rattler-build`. The recipe is
  `conda/recipe.yaml` in the v1 format (CEP 13/14); the old
  `conda/meta.yaml` and the `grayskull`/`anaconda-client` steps are gone.
- The v1 format is also conda-forge's direction, so the recipe should mostly
  carry over if this ever moves to conda-forge. As of early 2026 conda-forge
  still accepts both formats.
- conda-forge calls the DuckDB bindings **`python-duckdb`**, while
  `pyproject.toml` asks for `duckdb`. The recipe must use the conda name.
- The recipe's tests exercise **both console scripts plus `--version`**. 0.22.0
  moved the CLI to Typer, and an import-only test would not notice an entry
  point breaking.
