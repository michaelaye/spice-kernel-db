# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.0] - 2026-05-20

### Added

- **`browse --sort {name,date}`.** New flag on `spice-kernel-db browse`
  to order rows by the **Latest** remote modification date instead of
  alphabetically. `--sort date` sorts ascending, so the most recently
  updated metakernels appear at the bottom of the table — convenient
  for spotting fresh updates without scrolling. Mirrored on
  `KernelDB.browse_remote_metakernels(..., sort_by=...)`.

## [0.14.0] - 2026-05-13

### Added

- **Non-interactive `mission add`.** Positional mission name plus new flags
  let you add a mission without prompts:

  ```bash
  spice-kernel-db mission add MRO \
    --server-url https://naif.jpl.nasa.gov/pub/naif/ \
    --mk-dir-url https://naif.jpl.nasa.gov/pub/naif/MRO/kernels/mk/
  ```

  New options on `mission add`:
  - `--server-url <URL>` — archive server base URL (inferred from known
    servers when the mission is unambiguous).
  - `--mk-dir-url <URL>` — explicit metakernel directory URL. HEAD-checked,
    then stored as-is. Useful when you've located a metakernel directory
    on a PDS node or mission website that isn't on NAIF.
  - `--no-dedup` — disable deduplication for this mission.
  - `--use-planetarypy` — opt-in delegation flag (see below).

- **Curated mission registry (`mission_registry.toml`)** — extension point
  for verified alternate metakernel directory locations. Mission names map
  to candidate URL templates (`{server}` / `{m}` placeholders supported;
  absolute URLs accepted too) plus an optional `planetarypy = true` flag.
  The bundled file ships empty by design — an empirical NAIF survey found
  that the speculative patterns (`spice_kernels/mk/`, `data/spice/mk/`,
  etc.) you might guess from naming convention are not actually in use
  anywhere on NAIF. The registry grows only through verified
  contributions. The `mission add` flow consults it before falling back to
  the standard `kernels/mk/` probe.

- **Optional `[planetarypy]` extra** and `planetarypy_bridge` module.
  Most active NASA missions (MRO, MAVEN, JUNO, LRO, MARS2020, MSL, ORX,
  …) don't publish a mission-wide metakernel on NAIF — their kernels are
  served by PDS nodes, which the
  [planetarypy](https://github.com/michaelaye/planetarypy) library
  already manages. The bridge is the architectural hook for delegating to
  planetarypy when both (a) the registry marks a mission as managed by
  planetarypy and (b) the user passes `--use-planetarypy`. The
  current implementation is a stub that prints a notice and falls back
  to normal discovery; full delegation is tracked as a follow-up.

- **Public API additions:** `remote.discover_mk_url`,
  `remote.probe_mk_candidates`, `remote.DEFAULT_ALT_MK_PATHS`, the
  `registry` module (`MissionEntry`, `load_registry`,
  `registry_candidates`, `is_planetarypy_managed`), and the
  `planetarypy_bridge` module.

### Changed

- The interactive `mission add` flow now splits the mission list into
  "with default `kernels/mk/`" and "no default `mk/`". Typing the name of
  a "no default `mk/`" mission consults the curated registry and offers
  any hits as a numbered selection; the previous "not yet supported"
  message is gone.

### Notes for users

The empirical NAIF survey behind this release found only 8 of 76 real
missions ship a metakernel directory at the conventional path (and 7 of
those 8 are ESA-mirror entries). The remaining 68 NASA missions publish
individual per-type kernels but no curated `.tm` file. For those, your
real options are (a) point `--mk-dir-url` at a metakernel you've found
elsewhere (e.g., a PDS bundle), or (b) wait for the planetarypy bridge to
materialise. Synthesising a metakernel from per-type kernel listings is
sketched in `Plans/did-we-have-a-abstract-kite.md` (Tier 2) and is a
separate follow-up.

### Tests

- 232 → 254: `TestRegistry`, `TestProbeMkCandidates`, `TestDiscoverMkUrl`,
  `TestPlanetarypyBridge`, `TestMissionAddNoninteractive`. New fixture
  `tests/fixtures/mission_registry_min.toml`.

## [0.13.4] - 2026-05-12

### Fixed

- **`mk` / `list_metakernels` shows correct kernel count for alias rows.**
  When `get` creates an alias symlink (e.g. `juice_crema_5_2.tm` →
  `juice_crema_5_2_v470_20260415_001.tm`), the alias's `metakernel_registry`
  row used the symlink path, but `metakernel_entries` were stored under
  the resolved target path. The JOIN in `list_metakernels` returned
  `n_kernels=0` for the alias, and the content-fingerprint identity
  detection missed the relationship.

  `list_metakernels` now detects alias rows (mk_path is a symlink to
  another known mk_path) and inherits the target's entry count and
  fingerprint, so both rows display the same kernel count and one is
  annotated `↳ identical to <target>` — consistent with how real-file
  duplicates already render.

### Tests

- 230 → 232: `TestListMetakernelsAliasAware` (covers the symlink-alias
  inheritance and confirms non-alias symlinks aren't affected).

## [0.13.3] - 2026-05-12

### Added

- **`prune --orphan-symlinks`** (CLI) and `KernelDB.prune_orphan_symlinks()`
  (API) — walks each mission's download tree (derived from
  `metakernel_registry.mk_path`) and finds dangling symlinks whose
  target has disappeared. These accumulate over time as the upstream
  kernel store shifts or after default `prune` removes a location row
  (the symlinks pointing at that file aren't themselves in `locations`
  and so survive as junk). `--orphan-symlinks` is mutually exclusive
  with `--metakernels`; run them sequentially if you need both.

### Tests

- 226 → 230: `TestPruneOrphanSymlinks` (4 tests covering dry-run vs
  execute, healthy-symlink preservation, empty-registry early-out, and
  CLI mutual-exclusion).

## [0.13.2] - 2026-05-12

### Fixed

- **`prune --metakernels` now finds rows added via `scan`.** The 0.13.1
  implementation filtered the registry with `WHERE source_url IS NOT NULL`,
  which silently skipped every metakernel that had been scanned in from
  a local tree (the most common case — `scan_directory` writes
  `source_url=NULL`). It now derives a probe URL per row using the same
  fallback as `update_metakernel`: explicit `source_url` first, then
  `mission.mk_dir_url + filename`. Rows with neither — typically
  hand-rolled metakernels — are surfaced in the output as
  "no probeable URL" rather than silently ignored.

  Discovered in the wild: `update juice_crema_5_1_150lb_23_1_a3_2_v462_20260223_001.tm`
  correctly raised `MetakernelUnreachableError` (and pointed the user at
  `prune --metakernels`), but the subsequent prune run failed to list
  that very metakernel because it had been originally scanned in rather
  than `get`-acquired, so its `source_url` was NULL.

### Tests

- 224 → 226: `TestPruneMetakernelsNullSourceUrl` (2 tests covering the
  mk_dir_url-fallback path and the "no probeable URL" skip path).

## [0.13.1] - 2026-05-12

### Added

- **`MetakernelUnreachableError`** (new exception, exported from the
  top-level package) — raised by `KernelDB.update_metakernel()` when
  the remote returns HTTP 403/404/410. Attributes: `url`, `status`,
  `filename`. The CLI catches it and prints a rich red panel with
  recovery instructions instead of a raw stack trace; exit code is
  `2` (distinct from `1` for the generic `LookupError`) so scripts
  can detect the case.
- **`prune --metakernels` (CLI) and `KernelDB.prune_metakernels()`** —
  HEAD-probes every `source_url` in `metakernel_registry` and lists
  rows that return 403/404/410 (NAIF rotates old versioned snapshots
  into `former_versions/`, making the original URL permanently
  unreachable). `--execute` removes the rows; `--delete-files` also
  unlinks the on-disk `.tm` files. Transient errors (timeouts,
  DNS failures, 5xx) are never treated as dead — leaving a row in
  place is always safer than deleting on a network blip.

### Documentation

- New CLI subcommand entries for `verify` (shipped in 0.13.0 without
  docs — gap fixed) and `prune` (shipped in 0.11.0 without docs —
  gap fixed), plus the new `prune --metakernels` mode.
- New troubleshooting entries for "Metakernel unreachable / HTTP 404
  on update" and "`verify` reports unexpected statuses".
- New `docs/api.qmd` entries for `verify_metakernel`,
  `prune_metakernels`, and an Exceptions section covering both
  `ConcurrentModificationError` and `MetakernelUnreachableError`.
- New rule in `CLAUDE.md`: every user-visible change (new CLI
  command/flag, new public `KernelDB` method, new error class, new
  exit code) must land documentation in the same commit. Codifies
  why the 0.13.0 verify/prune doc gaps existed and won't again.

### Tests

- 224 tests (+8 over 0.13.0): `TestUpdateUnreachableMetakernel`
  (2), `TestPruneMetakernels` (5), `TestPruneMetakernelsCLI` (1).

## [0.13.0] - 2026-05-12

A wide-ranging audit-and-fix release driven by a parallel-agent
adversarial review of the architecture. Eight P0 findings (including a
remotely-triggered path-traversal write-anywhere primitive) and six P1
findings are closed. The new `verify` command lets you cross-check any
metakernel against the database. Test suite grows from 161 to 216.

See `Plans/2026-05-12-redteam-findings.md` for the full audit, data-
migration notes, and design questions for the (still-deferred)
per-metakernel dedup opt-out.

### Added

- **`verify` command** — `spice-kernel-db verify [<mk>]` deeply
  cross-checks a metakernel against the DB: traversal, dangling
  symlinks, size, sha256 (with `--deep`), ambiguous resolution, plus
  PATH_VALUES validity. `--strict` exits non-zero on any non-OK
  finding, `--json` emits machine-readable output, otherwise a rich
  table.
- **Streaming sha256 during download** — `download_kernel` returns
  the hash it computed while writing the bytes. `download_kernels_parallel`
  returns `(path, sha256)` tuples; both also accept an optional
  `expected_hash`/`expected_hashes` parameter that aborts the download
  on mismatch (foundation for future manifest-based integrity).
- **`canonicalize_mission()`** in `hashing` — single source of truth
  for mission-name casing. Real missions canonicalised uppercase;
  `generic` / `unknown` sentinels stay lowercase.
- **`ConcurrentModificationError`** exported from the package — raised
  when a `get` detects another writer mutated the DB during its
  network phase.
- **`kernels.superseded_by` column** — auto-migrated on open. Marks
  the old `kernels` row when the same path is re-registered with new
  content, so resolution skips superseded versions deterministically.

### Fixed

#### Security (P0)

- **Path-traversal write-anywhere via hostile metakernel (C1)** —
  `_link_existing_kernels` and the download-dest builder used
  `download_dir / mission / relpath` with no traversal check. A
  metakernel containing `$KERNELS/../../foo` could plant
  symlinks or HTTP-body bytes at arbitrary user-writable paths
  (`~/.ssh/authorized_keys` worst case). New `_safe_join` /
  `_validate_relpaths` helpers refuse the entire `get` on any
  escaping entry; `rewrite_metakernel`'s existing check is refactored
  onto the same helper.

#### Silent wrong-kernel-into-SPICE (P0)

- **Hash verification gate bypass (C2)** — `_link_existing_kernels`
  looked up the expected sha256 by the *requested* filename, which
  returns NULL whenever `resolve_kernel` falls back via path-suffix
  match (the documented "jup365.bsp ↔ jup365_19900101_20500101.bsp"
  case). Hash check is now keyed by the *resolved local path* (joined
  through `locations.abs_path`); a missing row is a hard skip with
  warning, not a silent pass.
- **Nondeterministic resolution on filename collisions (C3)** —
  multiple active `kernels` rows for the same filename are now
  deterministic (`ORDER BY mission, scanned_at DESC, abs_path`) and
  `find_by_filename` / `_find_by_path_suffix` filter out superseded
  rows. `verify` reports `AMBIGUOUS` when more than one active row
  remains.
- **Post-download hash trust (C4)** — `register_file` previously
  re-hashed the file from disk to determine its canonical sha256.
  Hashes are now computed during the download itself and plumbed
  through as `expected_hash`, closing the TOCTOU window and making
  manifest-based verification trivial to add.

#### DB integrity (P0)

- **Race-detection acted upon (C5)** — `_check_state_changed` raised
  the new `ConcurrentModificationError` (was previously discarded).
  `reacquire()` retries DuckDB lock contention with exponential
  backoff (~6.4 s max).
- **Transactional multi-statement ops (C6)** — `BEGIN/COMMIT/ROLLBACK`
  around `register_file` (multi-INSERT, possibly two `kernels` rows)
  and `index_metakernel` (DELETE + N INSERTs). A crash mid-op no
  longer leaves orphans.
- **Atomic file mutations (C7)** — new `_atomic_write_text` (in
  `parser.py`) and `_atomic_symlink` (in `db.py`) use the
  tmp+`os.replace` pattern. Applied to `write_metakernel`,
  `rewrite_metakernel`, and `_create_metakernel_alias`. A SPICE
  `furnsh` running concurrently with a rewrite no longer sees a
  zero-byte or torn `.tm` file.

#### Cross-platform (P0)

- **Case-insensitive filename queries (C8)** — `find_by_filename`,
  `_find_by_path_suffix`, `register_file`'s Issue-11 detection, and
  `verify_metakernel`'s kernel lookup all use `LOWER(filename) =
  LOWER(?)`. APFS/HFS+/NTFS and ext4 now behave the same — kernel
  identity is the sha256 anyway; casing is metadata.

#### Recoverable bugs (P1)

- **Dangling-symlink repair (H1)** — `_link_existing_kernels` no
  longer skips when `is_symlink()` is true but the target is gone.
  Dangling links are unlinked and recreated; `get` reports honest
  success.
- **Robust metakernel parser (H2)** — replaced the regex-based parser
  with a line-aware lexer that honors SPICE's `''` quote escape,
  balanced parens (including `)` inside strings), and accepts
  `\begindata`/`\begintext` markers only at line starts. Previously,
  a `\begindata` example *inside a comment block*, or any `)` in a
  kernel path, silently truncated `KERNELS_TO_LOAD` — leading SPICE
  to load fewer kernels than declared, with no error.
- **PATH_SYMBOL prefix collision (H3)** — substitutions now sort by
  descending symbol length, so `$KERNELS_DATA` is no longer clobbered
  by an earlier `$KERNELS` replacement. Applied in
  `ParsedMetakernel.resolve`, `kernel_relpaths`, and
  `resolve_kernel_urls`.
- **Mission case canonicalisation (H4)** — every storage boundary
  (`register_file`, `scan_directory`, `add_mission`, `get_metakernel`)
  funnels through `canonicalize_mission()`. Opening a legacy DB with
  e.g. both `JUICE` and `juice` rows emits a startup warning so the
  user knows to merge — we deliberately don't auto-merge because data
  ops should be explicit.
- **`task_info` keyed by full dest path (H5)** — a `.tm` with the same
  basename under two different relpaths no longer scrambles
  `source_url` attribution between the two `locations` rows.
- **Datetime-based staleness comparison (H6)** — the staleness check
  parses both timestamps to `datetime` instead of comparing 16-char
  string prefixes; sub-minute precision restored, and the previous
  off-by-one-minute spurious warnings are gone.

### Changed

- **`guess_mission` results are canonicalised** — path-derived
  candidates are run through `canonicalize_mission()` before return.
  E.g. `/data/generic_kernels/kernels/...` now yields
  `GENERIC_KERNELS` instead of the verbatim `generic_kernels`.
- **`download_kernel` return type** — now `(Path, sha256_hex)` instead
  of just `Path`. Internal callers updated; external code that
  depended on the path-only return needs to unpack.
- **`download_kernels_parallel` return type** — now
  `(list[(Path, sha256)], warnings)` instead of
  `(list[Path], warnings)`.
- **Test suite expanded from 161 to 216 tests.**

## [0.12.0] - 2026-05-11

### Added

- **Alias symlinks on `get`** — when the interactive picker groups versioned snapshots under a base name (e.g. `juice_crema_5_2.tm` covers `juice_crema_5_2_v470_20260415_001.tm`), `get` now creates a relative symlink at the alias name and registers a second `metakernel_registry` row so both names resolve. Also catches HTTP redirects via `resp.geturl()`. Refuses to clobber regular files at the alias path; replaces existing symlinks.
- **Interactive `resolve` picker** — running `spice-kernel-db resolve` with no arguments now opens a numbered table of tracked metakernels (filtered by `--mission` if given) and prints the selected `mk_path`, instead of exiting with a usage error.

## [0.11.1] - 2026-05-11

### Fixed

- **`resolve` finds tracked metakernels** — `spice-kernel-db resolve foo.tm` now returns the path of a `.tm` file acquired via `get` (previously returned "Not found" because `resolve_kernel` only consulted the `kernels`/`locations` tables and ignored `metakernel_registry`). Mission preference and cross-mission fallback warnings work the same as for regular kernels.

## [0.11.0] - 2026-03-26

### Added

- **`--version` flag** — `spice-kernel-db --version` now prints the installed version.
- **`skd` short alias** — `skd` is now a second entry point for `spice-kernel-db`, for quicker invocation.
- **No-args summary** — running `spice-kernel-db` (or `skd`) with no subcommand now shows locally acquired metakernels and a quick-start guide instead of just the help text.
- **`mk --remove` flag** — `spice-kernel-db mk --remove <name>` removes a metakernel from the registry and its entries. Does not delete files from disk.
- **`prune` command** — `spice-kernel-db prune` finds and removes stale DB entries for files that no longer exist on disk. Also cleans up orphaned kernel records with no remaining locations. Dry-run by default (`--execute` to apply).
- **`resolve --metakernel`** — batch-resolve all kernels in a `.tm` file, printing `filename\tpath` for each.
- **`config set/get`** — `config set db_path /new/path` and `config get db_path` for scripted, non-interactive configuration.
- **Global `-v/--verbose`** flag on the root parser, replacing per-command `-v` on `scan` and `check`.
- **Troubleshooting page** — `docs/troubleshooting.qmd` with 9 common issues and solutions.
- **Quick-start workflow** in the `--help` epilog: `mission add → browse → get → check`.
- **"See also" cross-references** in subcommand help text (check, list, resolve, mk, get, update, browse).
- **Pre-release review document** — `docs/review.qmd` captures the full CLI/API/docs audit.

### Changed

- **Deferred config loading** — `ensure_config()` is no longer called before argument parsing. `--help`, `--version`, and no-args invocation now work on a fresh install without triggering the interactive setup wizard.
- **`update_metakernel()` raises `LookupError`** instead of calling `sys.exit(1)`. The library layer (`db.py`) no longer calls `sys.exit()` — all exit decisions are made in the CLI layer (`cli.py`). This makes `KernelDB` safe to use as a Python API.
- **`_resolve_body_interactive()` returns `None`** on unknown bodies and invalid selections instead of calling `sys.exit(1)`.
- **`browse` (no args)** now picks a mission interactively and browses its remote metakernels, instead of duplicating the `mission list` table.
- **`rewrite -o` is now optional** — defaults to `<stem>_local.tm` when omitted.
- **`coverage` positional renamed** from `body_id` to `body` with clearer help text.
- **`dedup --execute` now prompts** for confirmation (shows dry-run preview first). Use `-y` to skip.
- **`--mission` help text standardized** — "Override" for commands that set the mission, "Filter" for listing commands, "Preferred mission" for resolution commands.
- **`get` help text clarified** — explains that filenames require a configured mission.
- Development Status classifier upgraded from Alpha to Beta.
- Test suite expanded from 136 to 157 tests.

### Fixed

- **Corrupt config handling** — `load_config()` now catches `TOMLDecodeError` with a friendly error message instead of a raw traceback.
- **`browse` network error** — catches `URLError` (network unreachable) with "Could not connect to server" message.
- **`coverage` ImportError** — now caught around body resolution too, not just the coverage analysis call.
- **Documentation overhaul** — removed all references to removed fuzzy filename matching from design.qmd, api.qmd, and CLAUDE.md. Updated API signatures for `scan_directory`, `register_file`, `get_metakernel`, `check_metakernel`, `browse_remote_metakernels`, and `KernelDB` constructor. Added missing `update_metakernel` API docs. Fixed Python version requirement in README (3.11, not 3.10). Updated Quarto site description. Added `kernel_coverage` table to schema docs. Updated usage.qmd browse output format. Added `list` command to usage guide.

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

[0.15.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.13.4...v0.14.0
[0.13.4]: https://github.com/michaelaye/spice-kernel-db/compare/v0.13.3...v0.13.4
[0.13.3]: https://github.com/michaelaye/spice-kernel-db/compare/v0.13.2...v0.13.3
[0.13.2]: https://github.com/michaelaye/spice-kernel-db/compare/v0.13.1...v0.13.2
[0.13.1]: https://github.com/michaelaye/spice-kernel-db/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/michaelaye/spice-kernel-db/compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com/michaelaye/spice-kernel-db/compare/v0.11.0...v0.11.1
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
