"""The database: scanning, resolution, dedup, hashing, coverage, config.

Split out of the former tests/test_kernel_db.py.
"""

from __future__ import annotations
import hashlib
import logging
import shutil
import textwrap
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from spice_kernel_db import KernelDB, parse_metakernel
from spice_kernel_db.config import Config, load_config, save_config, show_config
from spice_kernel_db.hashing import (
    canonicalize_mission,
    classify_kernel,
    guess_mission,
    sha256_file,
)
from spice_kernel_db.parser import parse_metakernel_text, write_metakernel
from spice_kernel_db.remote import (
    Probe,
    ProbeResult,
    RemoteMetakernel,
    check_mk_availability,
    list_remote_metakernels,
    list_remote_missions,
    resolve_kernel_urls,
    SPICE_SERVERS,
    _DIR_ENTRY_RE,
    _VERSION_TAG_RE,
)


class TestHashing:
    def test_sha256_file(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        h = sha256_file(f)
        assert len(h) == 64
        assert h == sha256_file(f)  # deterministic

    def test_sha256_different_content(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"hello")
        f2.write_bytes(b"world")
        assert sha256_file(f1) != sha256_file(f2)

    def test_sha256_same_content_different_name(self, tmp_path):
        f1 = tmp_path / "jup365.bsp"
        f2 = tmp_path / "jup365_19900101_20500101.bsp"
        content = b"same bytes"
        f1.write_bytes(content)
        f2.write_bytes(content)
        assert sha256_file(f1) == sha256_file(f2)

    def test_classify_kernel(self):
        assert classify_kernel("naif0012.tls") == "lsk"
        assert classify_kernel("de432s.bsp") == "spk"
        assert classify_kernel("juice_v44.tf") == "fk"
        assert classify_kernel("juice_sc.bc") == "ck"
        assert classify_kernel("pck00011.tpc") == "pck"
        assert classify_kernel("juice.ti") == "ik"
        assert classify_kernel("juice.tsc") == "sclk"
        assert classify_kernel("juice.bds") == "dsk"
        assert classify_kernel("random.xyz") == "unknown"

    def test_guess_mission(self):
        # H4: results are canonicalised via canonicalize_mission().
        # Real mission names → uppercase; "generic"/"unknown" sentinels
        # stay lowercase.
        assert guess_mission("/data/JUICE/kernels/lsk/naif.tls") == "JUICE"
        assert guess_mission("/data/MRO/kernels/ck/mro.bc") == "MRO"
        assert guess_mission("/data/juice/kernels/lsk/naif.tls") == "JUICE"
        # Path-based: directory before 'kernels/' is canonicalised
        assert guess_mission("/data/generic_kernels/kernels/lsk/naif.tls") == "GENERIC_KERNELS"
        # Fallback to filename-based when no 'kernels/' in path
        assert guess_mission("/naif/generic_kernels/lsk/naif0012.tls") == "generic"


class TestDBScan:
    def test_scan_counts(self, populated_db, tmp_spice_tree):
        stats = populated_db.stats()
        assert stats["n_locations"] > stats["n_kernels"]  # duplicates exist
        assert "JUICE" in stats["missions"]
        assert "MRO" in stats["missions"]
        assert "generic" in stats["missions"]

    def test_register_same_content_different_name(
        self, populated_db, tmp_spice_tree
    ):
        """jup365.bsp and jup365_19900101_20500101.bsp have same hash."""
        hits_generic = populated_db.find_by_filename("jup365.bsp")
        hits_juice = populated_db.find_by_filename(
            "jup365_19900101_20500101.bsp"
        )

        # Both should exist in the DB
        assert len(hits_generic) >= 1
        assert len(hits_juice) >= 1

        # But they share the same hash
        # (the first registered filename is canonical, but both locations
        #  are recorded)
        generic_hashes = {h["sha256"] for h in hits_generic}
        juice_paths = [
            str(tmp_spice_tree / "JUICE" / "kernels" / "spk"
                / "jup365_19900101_20500101.bsp")
        ]
        # The juice file's location should point to the same hash
        juice_loc_hashes = set()
        for jp in juice_paths:
            rows = populated_db.con.execute(
                "SELECT sha256 FROM locations WHERE abs_path = ?", [jp]
            ).fetchall()
            juice_loc_hashes.update(r[0] for r in rows)

        assert generic_hashes & juice_loc_hashes  # intersection non-empty

    def test_scan_returns_missions_found(self, db, tmp_path):
        """scan_directory returns (count, missions_found) tuple."""
        # Create a small directory with one kernel per mission
        (tmp_path / "TESTMISSION" / "kernels" / "lsk").mkdir(parents=True)
        (tmp_path / "TESTMISSION" / "kernels" / "lsk" / "test.tls").write_text("data")
        count, missions = db.scan_directory(tmp_path)
        assert count == 1
        assert "TESTMISSION" in missions


class TestResolveKernel:
    def test_resolve_exact_same_mission(self, populated_db, tmp_spice_tree):
        """Kernel in JUICE tree resolves from JUICE mission without warning."""
        path, warnings = populated_db.resolve_kernel(
            "juice_v44.tf", preferred_mission="JUICE"
        )
        assert path is not None
        assert "juice_v44.tf" in path
        assert len(warnings) == 0

    def test_resolve_generic_from_juice_mission(
        self, populated_db, tmp_spice_tree
    ):
        """naif0012.tls exists in both JUICE and generic; JUICE copy preferred."""
        path, warnings = populated_db.resolve_kernel(
            "naif0012.tls", preferred_mission="JUICE"
        )
        assert path is not None
        assert Path(path).is_file()
        # Should prefer JUICE copy → no warning
        assert "JUICE" in path
        assert len(warnings) == 0

    def test_resolve_missing_in_mission_falls_back(
        self, populated_db, tmp_spice_tree
    ):
        """MRO-specific kernel not found in JUICE → falls back with warning."""
        path, warnings = populated_db.resolve_kernel(
            "mro_sc_2024.bc", preferred_mission="JUICE"
        )
        assert path is not None
        assert "MRO" in path
        assert len(warnings) == 1
        assert "not found in [JUICE]" in warnings[0]

    def test_resolve_nonexistent(self, populated_db):
        """Totally unknown kernel returns None."""
        path, warnings = populated_db.resolve_kernel("does_not_exist.bsp")
        assert path is None

    def test_resolve_without_preferred_mission(self, populated_db):
        """Without preferred mission, any copy is fine."""
        path, warnings = populated_db.resolve_kernel("naif0012.tls")
        assert path is not None
        assert Path(path).is_file()

    def test_resolve_metakernel_from_registry(
        self, populated_db, tmp_spice_tree
    ):
        """A .tm file tracked only in metakernel_registry is resolvable."""
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        populated_db.con.execute(
            "INSERT OR REPLACE INTO metakernel_registry "
            "VALUES (?, ?, ?, ?, current_timestamp)",
            [str(mk), "JUICE", "https://example.com/juice_test.tm",
             "juice_test.tm"],
        )
        path, warnings = populated_db.resolve_kernel(
            "juice_test.tm", preferred_mission="JUICE"
        )
        assert path == str(mk)
        assert warnings == []

    def test_resolve_metakernel_cross_mission_warns(
        self, populated_db, tmp_spice_tree
    ):
        """Resolving a .tm registered in another mission emits a warning."""
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        populated_db.con.execute(
            "INSERT OR REPLACE INTO metakernel_registry "
            "VALUES (?, ?, ?, ?, current_timestamp)",
            [str(mk), "JUICE", "https://example.com/juice_test.tm",
             "juice_test.tm"],
        )
        path, warnings = populated_db.resolve_kernel(
            "juice_test.tm", preferred_mission="MRO"
        )
        assert path == str(mk)
        assert len(warnings) == 1
        assert "not found in [MRO]" in warnings[0]


    def test_resolve_prefers_the_name_asked_for(self, db, tmp_spice_tree):
        """Two byte-identical files under different names each resolve to
        themselves, not to whichever the scanner walked last (issue #9)."""
        mk_dir = tmp_spice_tree / "JUICE" / "kernels" / "mk"
        pointer = mk_dir / "juice_crema_5_2.tm"
        snapshot = mk_dir / "juice_crema_5_2_v473_20260819_001.tm"
        content = "KPL/MK\n\nFAKE shared metakernel content\n"
        pointer.write_text(content)
        snapshot.write_text(content)
        db.scan_directory(tmp_spice_tree / "JUICE" / "kernels")

        for wanted in (pointer, snapshot):
            path, warnings = db.resolve_kernel(
                wanted.name, preferred_mission="JUICE"
            )
            assert path == str(wanted)
            assert warnings == []

    def test_resolve_prefers_exact_name_without_preferred_mission(
        self, db, tmp_spice_tree
    ):
        """The same holds when no mission is given, where resolution falls
        through to the 'any mission' step."""
        mk_dir = tmp_spice_tree / "JUICE" / "kernels" / "mk"
        pointer = mk_dir / "juice_crema_5_2.tm"
        snapshot = mk_dir / "juice_crema_5_2_v473_20260819_001.tm"
        content = "KPL/MK\n\nFAKE shared metakernel content\n"
        pointer.write_text(content)
        snapshot.write_text(content)
        db.scan_directory(tmp_spice_tree / "JUICE" / "kernels")

        for wanted in (pointer, snapshot):
            path, warnings = db.resolve_kernel(wanted.name)
            assert path == str(wanted)
            assert warnings == []

    def test_resolve_still_uses_identical_content_under_another_name(
        self, populated_db, tmp_spice_tree
    ):
        """With no file of the requested name registered, the sha256 join is
        still the point: jup365.bsp resolves to the time-ranged copy, and says
        so in a warning."""
        generic = (tmp_spice_tree / "generic_kernels" / "spk" / "satellites"
                   / "jup365.bsp")
        ranged = (tmp_spice_tree / "JUICE" / "kernels" / "spk"
                  / "jup365_19900101_20500101.bsp")
        assert sha256_file(generic) == sha256_file(ranged)
        generic.unlink()

        path, warnings = populated_db.resolve_kernel("jup365.bsp")
        assert path == str(ranged)
        assert len(warnings) == 1
        assert "jup365_19900101_20500101.bsp" in warnings[0]


class TestDuplicates:
    def test_report_duplicates(self, populated_db):
        dups = populated_db.report_duplicates()
        assert len(dups) > 0
        filenames = {d["filename"] for d in dups}
        assert "naif0012.tls" in filenames
        assert "pck00011.tpc" in filenames
        assert "de432s.bsp" in filenames

    def test_no_false_duplicates(self, populated_db):
        """Mission-specific kernels should NOT appear as duplicates."""
        dups = populated_db.report_duplicates()
        filenames = {d["filename"] for d in dups}
        assert "juice_v44.tf" not in filenames
        assert "mro_sc_2024.bc" not in filenames

    def test_dedup_plan_prefers_generic(self, populated_db):
        plan = populated_db.deduplicate_plan()
        for item in plan:
            # If generic is among the copies, it should be the kept one
            is_in_generic = any("generic" in p for p in [item["keep"]])
            has_generic_copy = any(
                "generic" in p
                for p in item["remove"] + [item["keep"]]
            )
            if has_generic_copy:
                assert is_in_generic, (
                    f"Generic copy should be preferred for {item['filename']}"
                )


class TestDedupWithSymlinks:
    def test_dry_run_does_not_modify(
        self, populated_db, tmp_spice_tree
    ):
        # Count files before
        all_files = list(tmp_spice_tree.rglob("*"))
        symlinks_before = sum(1 for f in all_files if f.is_symlink())

        populated_db.deduplicate_with_symlinks(dry_run=True)

        all_files_after = list(tmp_spice_tree.rglob("*"))
        symlinks_after = sum(1 for f in all_files_after if f.is_symlink())
        assert symlinks_before == symlinks_after

    def test_execute_creates_symlinks(
        self, populated_db, tmp_spice_tree
    ):
        populated_db.deduplicate_with_symlinks(dry_run=False)

        # naif0012.tls in JUICE should now be a symlink
        juice_lsk = (
            tmp_spice_tree / "JUICE" / "kernels" / "lsk" / "naif0012.tls"
        )
        assert juice_lsk.is_symlink()
        # But it should still be readable
        assert juice_lsk.read_text() == "FAKE LSK naif0012"


class TestConfig:
    def test_save_and_load(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_file = config_dir / "config.toml"
        monkeypatch.setattr("spice_kernel_db.config.CONFIG_DIR", config_dir)
        monkeypatch.setattr("spice_kernel_db.config.CONFIG_FILE", config_file)

        cfg = Config(db_path="/tmp/test.duckdb", kernel_dir="/tmp/kernels")
        save_config(cfg)
        assert config_file.is_file()

        loaded = load_config()
        assert loaded is not None
        assert loaded.db_path == "/tmp/test.duckdb"
        assert loaded.kernel_dir == "/tmp/kernels"

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        config_file = tmp_path / "nonexistent" / "config.toml"
        monkeypatch.setattr("spice_kernel_db.config.CONFIG_FILE", config_file)
        assert load_config() is None


class TestArchiveOnScan:
    def test_archive_moves_file_and_creates_symlink(self, tmp_path):
        """scan with archive_dir moves files and leaves symlinks."""
        # Setup: a random directory with a kernel
        src_dir = tmp_path / "random_project"
        src_dir.mkdir()
        kernel_file = src_dir / "naif0012.tls"
        kernel_file.write_text("FAKE LSK naif0012")

        archive_dir = tmp_path / "archive"

        db = KernelDB(tmp_path / "test.duckdb")
        db.scan_directory(
            src_dir, mission="JUICE", archive_dir=archive_dir,
        )

        # File should now be in the archive
        archived = archive_dir / "JUICE" / "lsk" / "naif0012.tls"
        assert archived.is_file()
        assert archived.read_text() == "FAKE LSK naif0012"

        # Original location should be a symlink
        assert kernel_file.is_symlink()
        assert kernel_file.resolve() == archived.resolve()

        # DB should point to the archive path
        hits = db.find_by_filename("naif0012.tls")
        assert len(hits) == 1
        assert str(archived) in hits[0]["abs_path"]

        db.close()

    def test_archive_skips_already_archived(self, tmp_path):
        """If file is already at the archive destination, no move needed."""
        archive_dir = tmp_path / "archive"
        dest = archive_dir / "JUICE" / "lsk" / "naif0012.tls"
        dest.parent.mkdir(parents=True)
        dest.write_text("FAKE LSK naif0012")

        db = KernelDB(tmp_path / "test.duckdb")
        db.scan_directory(
            archive_dir, mission="JUICE", archive_dir=archive_dir,
        )

        # File should still be a regular file, not a symlink
        assert dest.is_file()
        assert not dest.is_symlink()

        hits = db.find_by_filename("naif0012.tls")
        assert len(hits) == 1
        db.close()

    def test_archive_multiple_types(self, tmp_path):
        """Archive correctly sorts different kernel types into subdirs."""
        src_dir = tmp_path / "mixed"
        src_dir.mkdir()
        (src_dir / "naif0012.tls").write_text("FAKE LSK")
        (src_dir / "de432s.bsp").write_bytes(b"FAKE SPK" + b"\x00" * 100)
        (src_dir / "juice_v44.tf").write_text("FAKE FK")

        archive_dir = tmp_path / "archive"

        db = KernelDB(tmp_path / "test.duckdb")
        db.scan_directory(src_dir, mission="TEST", archive_dir=archive_dir)

        assert (archive_dir / "TEST" / "lsk" / "naif0012.tls").is_file()
        assert (archive_dir / "TEST" / "spk" / "de432s.bsp").is_file()
        assert (archive_dir / "TEST" / "fk" / "juice_v44.tf").is_file()

        # All originals should be symlinks
        assert (src_dir / "naif0012.tls").is_symlink()
        assert (src_dir / "de432s.bsp").is_symlink()
        assert (src_dir / "juice_v44.tf").is_symlink()

        db.close()


class TestPerMissionDedup:
    def test_dedup_respects_disabled_mission(self, populated_db, tmp_spice_tree):
        """Files in a no-dedup mission should NOT be removed."""
        # Mark JUICE as dedup=False
        populated_db.add_mission(
            "JUICE", SPICE_SERVERS["ESA"],
            "https://example.com/JUICE/mk/", dedup=False,
        )

        plan = populated_db.deduplicate_plan()

        # No plan entry should remove a JUICE path
        for item in plan:
            for rm_path in item["remove"]:
                assert "JUICE" not in rm_path, (
                    f"JUICE file should not be removed when dedup=False: {rm_path}"
                )

    def test_dedup_includes_enabled_mission(self, populated_db, tmp_spice_tree):
        """Files in a dedup-enabled mission CAN be removed."""
        # Mark MRO as dedup=True (default), JUICE as dedup=True
        populated_db.add_mission(
            "JUICE", SPICE_SERVERS["ESA"],
            "https://example.com/JUICE/mk/", dedup=True,
        )
        populated_db.add_mission(
            "MRO", SPICE_SERVERS["NASA"],
            "https://example.com/MRO/mk/", dedup=True,
        )

        plan = populated_db.deduplicate_plan()
        # With all missions dedup-enabled, plan should have entries
        assert len(plan) > 0

    def test_dedup_default_is_enabled(self, populated_db, tmp_spice_tree):
        """Missions not in the missions table default to dedup=True."""
        # No missions configured at all
        plan_default = populated_db.deduplicate_plan()

        # Should be the same as having all dedup enabled
        assert len(plan_default) > 0
        # naif0012.tls is duplicated across generic, JUICE, MRO
        filenames = {item["filename"] for item in plan_default}
        assert "naif0012.tls" in filenames

    def test_dedup_mixed_settings(self, populated_db, tmp_spice_tree):
        """With mixed settings, only dedup-enabled files are removed."""
        # JUICE dedup off, MRO not configured (defaults to on)
        populated_db.add_mission(
            "JUICE", SPICE_SERVERS["ESA"],
            "https://example.com/JUICE/mk/", dedup=False,
        )

        plan = populated_db.deduplicate_plan()

        # Collect all removed paths
        removed_paths = []
        for item in plan:
            removed_paths.extend(item["remove"])

        # JUICE paths should never appear in removals
        juice_removed = [p for p in removed_paths if "JUICE" in p]
        assert len(juice_removed) == 0

        # MRO paths CAN appear in removals (dedup defaults to True)
        # naif0012.tls exists in generic, JUICE, and MRO
        # With JUICE protected, the plan should still remove MRO copies
        mro_removed = [p for p in removed_paths if "MRO" in p]
        assert len(mro_removed) > 0


class TestCoverage:
    """Tests for the coverage analysis feature.

    All tests mock spiceypy to avoid needing real binary SPK files.
    """

    def test_non_spk_kernels_skipped(self):
        """check_coverage skips non-SPK kernels without calling spk_coverage."""
        from spice_kernel_db.coverage import check_coverage

        with patch("spice_kernel_db.coverage.spk_coverage") as mock_cov:
            results = check_coverage(
                filenames=["naif0012.tls", "juice_v44.tf"],
                resolved_paths=["/fake/naif0012.tls", "/fake/juice_v44.tf"],
                kernel_types=["lsk", "fk"],
                body_id=399,
            )

        mock_cov.assert_not_called()
        assert len(results) == 2
        assert all(not r.body_found for r in results)
        assert all(r.error is None for r in results)

    def test_missing_spk_sets_error(self):
        """Missing SPK file produces an error entry."""
        from spice_kernel_db.coverage import check_coverage

        results = check_coverage(
            filenames=["missing.bsp"],
            resolved_paths=[None],
            kernel_types=["spk"],
            body_id=399,
        )
        assert len(results) == 1
        assert results[0].error is not None
        assert "not found" in results[0].error.lower()
        assert not results[0].body_found

    def test_body_found_single_interval(self, tmp_path):
        """Body found with a single coverage interval."""
        from spice_kernel_db.coverage import (
            CoverageInterval,
            check_coverage,
        )

        fake_spk = tmp_path / "test.bsp"
        fake_spk.write_bytes(b"FAKE SPK")

        mock_interval = CoverageInterval(
            et_start=0.0, et_end=1e8,
            utc_start="2000-JAN-01 12:00",
            utc_end="2003-FEB-27 08:47",
        )

        with patch(
            "spice_kernel_db.coverage.spk_coverage",
            return_value=[mock_interval],
        ):
            results = check_coverage(
                filenames=["test.bsp"],
                resolved_paths=[str(fake_spk)],
                kernel_types=["spk"],
                body_id=399,
            )

        assert len(results) == 1
        assert results[0].body_found is True
        assert len(results[0].intervals) == 1
        assert results[0].intervals[0].et_start == 0.0

    def test_body_found_multiple_intervals_gap(self, tmp_path):
        """Body found with multiple intervals (gap in coverage)."""
        from spice_kernel_db.coverage import (
            CoverageInterval,
            check_coverage,
        )

        fake_spk = tmp_path / "gapped.bsp"
        fake_spk.write_bytes(b"FAKE SPK")

        intervals = [
            CoverageInterval(et_start=0.0, et_end=1e8),
            CoverageInterval(et_start=2e8, et_end=3e8),
        ]

        with patch(
            "spice_kernel_db.coverage.spk_coverage",
            return_value=intervals,
        ):
            results = check_coverage(
                filenames=["gapped.bsp"],
                resolved_paths=[str(fake_spk)],
                kernel_types=["spk"],
                body_id=399,
            )

        assert results[0].body_found is True
        assert len(results[0].intervals) == 2

    def test_body_not_found_empty_intervals(self, tmp_path):
        """Body not in SPK file → empty intervals, body_found=False."""
        from spice_kernel_db.coverage import check_coverage

        fake_spk = tmp_path / "nobody.bsp"
        fake_spk.write_bytes(b"FAKE SPK")

        with patch(
            "spice_kernel_db.coverage.spk_coverage",
            return_value=[],
        ):
            results = check_coverage(
                filenames=["nobody.bsp"],
                resolved_paths=[str(fake_spk)],
                kernel_types=["spk"],
                body_id=90004923,
            )

        assert results[0].body_found is False
        assert len(results[0].intervals) == 0

    def test_spiceypy_import_error(self, tmp_path):
        """ImportError from spiceypy gives a clear message."""
        from spice_kernel_db.coverage import check_coverage

        fake_spk = tmp_path / "test.bsp"
        fake_spk.write_bytes(b"FAKE SPK")

        with patch(
            "spice_kernel_db.coverage.spk_coverage",
            side_effect=ImportError("No module named 'spiceypy'"),
        ):
            results = check_coverage(
                filenames=["test.bsp"],
                resolved_paths=[str(fake_spk)],
                kernel_types=["spk"],
                body_id=399,
            )

        assert results[0].error is not None
        assert "spiceypy" in results[0].error.lower()

    def test_lsk_autodiscovery_passes_lsk_path(self, tmp_path):
        """check_coverage forwards lsk_path when provided."""
        from spice_kernel_db.coverage import check_coverage

        fake_spk = tmp_path / "test.bsp"
        fake_spk.write_bytes(b"FAKE SPK")

        with patch(
            "spice_kernel_db.coverage.spk_coverage",
            return_value=[],
        ) as mock_cov:
            check_coverage(
                filenames=["test.bsp"],
                resolved_paths=[str(fake_spk)],
                kernel_types=["spk"],
                body_id=399,
                lsk_path="/fake/naif0012.tls",
            )

        mock_cov.assert_called_once_with(
            str(fake_spk), 399, lsk_path="/fake/naif0012.tls",
        )

    def test_coverage_metakernel_integration(
        self, populated_db, tmp_spice_tree,
    ):
        """coverage_metakernel resolves, classifies, and delegates correctly."""
        from spice_kernel_db.coverage import CoverageInterval

        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"

        mock_interval = CoverageInterval(et_start=0.0, et_end=1e8)

        with patch(
            "spice_kernel_db.coverage.spk_coverage",
            return_value=[mock_interval],
        ):
            results = populated_db.coverage_metakernel(
                mk, body_id=399, mission="JUICE",
            )

        # 11 kernels in juice_test.tm
        assert len(results) == 11
        spk_results = [r for r in results if r.kernel_type == "spk"]
        # de432s.bsp, jup365_*.bsp, juice_crema_*.bsp = 3 SPK files
        assert len(spk_results) == 3
        assert all(r.body_found for r in spk_results)

    def test_store_and_query_coverage(self, populated_db):
        """store_coverage persists, query_coverage retrieves."""
        from spice_kernel_db.coverage import CoverageInterval

        intervals = [
            CoverageInterval(et_start=0.0, et_end=1e8),
            CoverageInterval(et_start=2e8, et_end=3e8),
        ]

        # Use a known sha256 from the DB
        row = populated_db.con.execute(
            "SELECT sha256 FROM kernels LIMIT 1"
        ).fetchone()
        sha = row[0]

        populated_db.store_coverage(sha, 399, intervals)

        # Query without time range
        hits = populated_db.query_coverage(399)
        assert len(hits) == 2
        assert hits[0]["sha256"] == sha

        # Query with overlapping time range
        hits = populated_db.query_coverage(399, et_start=0.5e8, et_end=0.9e8)
        assert len(hits) == 1  # only first interval overlaps

        # Query with non-overlapping time range
        hits = populated_db.query_coverage(399, et_start=1e8, et_end=2e8)
        assert len(hits) == 0

    def test_query_coverage_wrong_body(self, populated_db):
        """query_coverage returns empty for unknown body."""
        hits = populated_db.query_coverage(99999)
        assert hits == []

    def test_store_coverage_upserts(self, populated_db):
        """Calling store_coverage twice replaces previous intervals."""
        from spice_kernel_db.coverage import CoverageInterval

        row = populated_db.con.execute(
            "SELECT sha256 FROM kernels LIMIT 1"
        ).fetchone()
        sha = row[0]

        populated_db.store_coverage(
            sha, 399, [CoverageInterval(et_start=0.0, et_end=1e8)],
        )
        assert len(populated_db.query_coverage(399)) == 1

        # Upsert with different intervals
        populated_db.store_coverage(
            sha, 399, [
                CoverageInterval(et_start=5e8, et_end=6e8),
                CoverageInterval(et_start=7e8, et_end=8e8),
            ],
        )
        hits = populated_db.query_coverage(399)
        assert len(hits) == 2
        assert hits[0]["et_start"] == 5e8


class TestBodyNameResolution:
    """Tests for resolve_body_id and NAIF_BODIES lookup."""

    def test_numeric_id_passthrough(self):
        """Numeric string returns single-element list with that ID."""
        from spice_kernel_db.coverage import resolve_body_id
        result = resolve_body_id("399")
        assert len(result) == 1
        assert result[0][0] == 399

    def test_negative_numeric_id(self):
        """Negative NAIF IDs (spacecraft) work."""
        from spice_kernel_db.coverage import resolve_body_id
        result = resolve_body_id("-61")
        assert len(result) == 1
        assert result[0][0] == -61

    def test_name_case_insensitive(self):
        """Body names are case-insensitive."""
        from spice_kernel_db.coverage import resolve_body_id
        for name in ("Earth", "earth", "EARTH", "eArTh"):
            result = resolve_body_id(name)
            assert len(result) >= 1
            ids = [r[0] for r in result]
            assert 399 in ids

    def test_ambiguous_name_returns_multiple(self):
        """Ambiguous names like 'earth' return body center + barycenter."""
        from spice_kernel_db.coverage import resolve_body_id
        result = resolve_body_id("earth")
        assert len(result) == 2
        ids = {r[0] for r in result}
        assert 399 in ids  # Earth body center
        assert 3 in ids    # Earth-Moon barycenter

    def test_unambiguous_name_returns_one(self):
        """Unambiguous names return a single match."""
        from spice_kernel_db.coverage import resolve_body_id
        result = resolve_body_id("moon")
        assert len(result) == 1
        assert result[0][0] == 301

    def test_unknown_name_returns_empty(self):
        """Unknown names return an empty list."""
        from spice_kernel_db.coverage import resolve_body_id
        result = resolve_body_id("krypton")
        assert result == []

    def test_common_bodies_present(self):
        """Key solar system bodies are in the lookup table."""
        from spice_kernel_db.coverage import NAIF_BODIES
        expected = [
            "sun", "mercury", "venus", "earth", "mars",
            "jupiter", "saturn", "uranus", "neptune",
            "moon", "europa", "ganymede", "titan",
        ]
        for name in expected:
            assert name in NAIF_BODIES, f"{name} missing from NAIF_BODIES"

    def test_3i_atlas_in_table(self):
        """3I/ATLAS is in the lookup table with both Horizons IDs."""
        from spice_kernel_db.coverage import resolve_body_id
        result = resolve_body_id("3i/atlas")
        assert len(result) == 2
        ids = {r[0] for r in result}
        assert 1004083 in ids   # Horizons SPK ID
        assert 90004923 in ids  # Horizons record ID


class TestPrune:
    """Tests for prune command."""

    def test_prune_removes_stale_locations(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")

        # Create a real file and register it
        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("FAKE LSK")
        h = db.register_file(kernel, mission="TEST")

        # Verify it's registered
        locs = db.find_by_hash(h)
        assert len(locs) == 1

        # Delete the file from disk
        kernel.unlink()

        # Dry run — should report but not delete
        pruned = db.prune(dry_run=True)
        assert len(pruned) == 1
        assert db.con.execute("SELECT COUNT(*) FROM locations").fetchone()[0] == 1

        # Execute — should actually delete
        pruned = db.prune(dry_run=False)
        assert len(pruned) == 1
        assert db.con.execute("SELECT COUNT(*) FROM locations").fetchone()[0] == 0
        # Orphaned kernel should also be removed
        assert db.con.execute("SELECT COUNT(*) FROM kernels").fetchone()[0] == 0
        db.close()

    def test_prune_nothing_to_do(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")

        # Create a real file and register it
        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("FAKE LSK")
        db.register_file(kernel, mission="TEST")

        # File still exists — nothing to prune
        pruned = db.prune(dry_run=False)
        assert len(pruned) == 0
        db.close()


class TestPruneOrphanSymlinks:
    """`prune --orphan-symlinks` walks each mission's download tree and
    finds symlinks whose target has disappeared. These accumulate when
    the upstream kernel store moves or after default `prune` deletes
    location rows for missing files."""

    def _seed_tree(self, tmp_path: Path) -> tuple[KernelDB, Path, Path]:
        """Build a download-tree shape that mimics what `get` produces:
        download_dir/<mission>/mk/<file.tm> + download_dir/<mission>/lsk/...
        Returns (db, dangling_link, healthy_link)."""
        db = KernelDB(tmp_path / "test.duckdb")
        dl = tmp_path / "downloads"
        mission_root = dl / "JUICE"
        (mission_root / "mk").mkdir(parents=True)
        (mission_root / "lsk").mkdir()

        # The mk file (referenced via metakernel_registry so we can
        # derive the mission root)
        mk = mission_root / "mk" / "juice.tm"
        mk.write_text("\\begintext\n")
        db.con.execute(
            "INSERT INTO metakernel_registry (mk_path, mission, source_url, "
            "filename, acquired_at) VALUES (?, 'JUICE', NULL, ?, "
            "current_timestamp)",
            [str(mk), "juice.tm"],
        )

        # A healthy symlink pointing at an existing kernel
        kernel = tmp_path / "store" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("LSK")
        healthy = mission_root / "lsk" / "naif0012.tls"
        healthy.symlink_to(kernel)

        # A dangling symlink (target never existed)
        dangling = mission_root / "lsk" / "ghost.tls"
        dangling.symlink_to(tmp_path / "nowhere" / "ghost.tls")

        return db, dangling, healthy

    def test_dry_run_reports_dangling_only(self, tmp_path, capsys):
        db, dangling, healthy = self._seed_tree(tmp_path)
        result = db.prune_orphan_symlinks(dry_run=True)
        assert len(result) == 1
        assert str(dangling) in result
        # Healthy link untouched in dry run AND in result
        assert healthy.is_symlink() and healthy.exists()
        assert dangling.is_symlink()  # not yet removed
        captured = capsys.readouterr().out
        assert "Would remove" in captured
        db.close()

    def test_execute_unlinks_dangling_keeps_healthy(self, tmp_path):
        db, dangling, healthy = self._seed_tree(tmp_path)
        result = db.prune_orphan_symlinks(dry_run=False)
        assert len(result) == 1
        assert not dangling.exists() and not dangling.is_symlink()
        # Healthy symlink survives
        assert healthy.is_symlink() and healthy.exists()
        db.close()

    def test_no_registered_metakernels_skips_walk(self, tmp_path, capsys):
        """With nothing in metakernel_registry there's no tree to walk."""
        db = KernelDB(tmp_path / "test.duckdb")
        result = db.prune_orphan_symlinks(dry_run=True)
        assert result == []
        captured = capsys.readouterr().out
        assert "No download trees" in captured
        db.close()

    def test_mutually_exclusive_with_metakernels_in_cli(self, tmp_path):
        from spice_kernel_db.cli import main
        db = KernelDB(tmp_path / "test.duckdb")
        db.close()
        with patch("spice_kernel_db.cli.ensure_config", return_value=Config()):
            with pytest.raises(SystemExit) as exc:
                main([
                    "--db", str(tmp_path / "test.duckdb"),
                    "prune", "--metakernels", "--orphan-symlinks",
                ])
        assert exc.value.code == 1


class TestRegistry:
    def test_load_registry_returns_entries(self, fixture_registry):
        entries = fixture_registry.load_registry()
        assert "LRO" in entries
        assert "MSL" in entries
        assert len(entries["LRO"].candidates) == 2

    def test_registry_candidates_expands_placeholders(self, fixture_registry):
        urls = fixture_registry.registry_candidates(
            "LRO", "https://naif.jpl.nasa.gov/pub/naif/"
        )
        assert urls == [
            "https://naif.jpl.nasa.gov/pub/naif/LRO/data/spice/mk/",
            "https://naif.jpl.nasa.gov/pub/naif/LRO/kernels/mk/",
        ]

    def test_registry_candidates_adds_trailing_slash(self, fixture_registry):
        urls = fixture_registry.registry_candidates(
            "LRO", "https://naif.jpl.nasa.gov/pub/naif"  # no trailing /
        )
        assert urls[0].startswith("https://naif.jpl.nasa.gov/pub/naif/LRO/")

    def test_registry_candidates_unknown_mission(self, fixture_registry):
        assert fixture_registry.registry_candidates("APOLLO", "https://x/") == []

    def test_bundled_registry_loads(self):
        """The TOML shipped inside the package must at least parse."""
        from spice_kernel_db import registry as reg
        reg.load_registry.cache_clear()
        assert reg.load_registry() is not None

    def test_registry_is_read_only(self, fixture_registry):
        """The cached registry mapping must not be mutable — prevents one
        caller from poisoning the cache for everyone else."""
        entries = fixture_registry.load_registry()
        with pytest.raises(TypeError):
            entries["NEW_MISSION"] = fixture_registry.MissionEntry()


class TestLocationKind:
    """`aliases` on-disk listing distinguishes real files from symlinks."""

    def test_real_symlink_and_missing(self, tmp_path):
        from spice_kernel_db.cli.inspect_cmds import _location_kind
        real = tmp_path / "r.bc"
        real.write_bytes(b"x")
        link = tmp_path / "l.bc"
        link.symlink_to(real)
        assert _location_kind(str(real)) == "real file"
        assert _location_kind(str(link)) == "→ r.bc"
        assert _location_kind(str(tmp_path / "gone.bc")) == "missing"


class TestDefaultDbPath:
    """Without a config file: new default, with the old path found until 0.21.0."""

    @pytest.fixture(autouse=True)
    def _home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    def test_new_default_when_nothing_exists(self):
        import warnings

        from spice_kernel_db.config import DEFAULT_DB_PATH, default_db_path

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert default_db_path() == DEFAULT_DB_PATH

    def test_legacy_database_now_raises_instead_of_being_opened(self, tmp_path):
        """0.21.0 stopped opening it, as the FutureWarning promised.

        It raises rather than quietly creating an empty database next to a
        populated legacy one, which would look like data loss.
        """
        from spice_kernel_db.config import LegacyDatabaseFound, default_db_path

        (tmp_path / ".spice_kernels.duckdb").write_bytes(b"")
        with pytest.raises(LegacyDatabaseFound) as exc:
            default_db_path()
        msg = str(exc.value)
        assert "~/.spice_kernels.duckdb" in msg
        assert "config --setup" in msg and "mv " in msg

    def test_legacy_file_is_ignored_once_the_new_one_exists(self, tmp_path):
        """No error when the user has already migrated."""
        from spice_kernel_db.config import DEFAULT_DB_PATH, default_db_path

        (tmp_path / ".spice_kernels.duckdb").write_bytes(b"")
        new = tmp_path / ".local/share/spice-kernel-db/kernels.duckdb"
        new.parent.mkdir(parents=True)
        new.write_bytes(b"")
        assert default_db_path() == DEFAULT_DB_PATH

    def test_new_database_wins_over_legacy(self, tmp_path):
        import warnings

        from spice_kernel_db.config import DEFAULT_DB_PATH, default_db_path

        (tmp_path / ".spice_kernels.duckdb").write_bytes(b"")
        new = tmp_path / ".local/share/spice-kernel-db/kernels.duckdb"
        new.parent.mkdir(parents=True)
        new.write_bytes(b"")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert default_db_path() == DEFAULT_DB_PATH

    def test_kerneldb_without_config_creates_the_new_default(self, tmp_path):
        with patch("spice_kernel_db.config.load_config", return_value=None):
            db = KernelDB()
        try:
            assert db.db_path == str(tmp_path / ".local/share/spice-kernel-db/kernels.duckdb")
            assert Path(db.db_path).is_file()
        finally:
            db.close()


class TestRemoteSizesAreActuallyChecked:
    """The size column was never asserted on.

    Before the offline guard these tests opened real sockets to NAIF, and
    whatever Content-Length came back went unverified; after it, the lookup
    fails and returns nothing, equally unverified. Either way, no coverage.
    """

    def test_query_remote_sizes_is_patchable_and_returns_what_it_is_given(
        self, remote_sizes
    ):
        """Pin the seam the fixture relies on, so a rename is caught here."""
        from spice_kernel_db import remote

        url = "https://example.com/x.bsp"
        with remote_sizes({url: 4096}):
            assert remote.query_remote_sizes([url]) == {url: 4096}

    def test_offline_guard_blocks_a_real_connection(self):
        """The guard itself must work, or everything above is theatre."""
        import socket

        from tests._netguard import NetworkUseInOfflineTest

        s = socket.socket()
        try:
            with pytest.raises(NetworkUseInOfflineTest):
                s.connect(("naif.jpl.nasa.gov", 443))
        finally:
            s.close()

    def test_localhost_is_still_allowed(self):
        """Blocking loopback would break unrelated libraries."""
        import socket

        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        cli = socket.socket()
        try:
            cli.connect(("127.0.0.1", port))  # must not raise
        finally:
            cli.close()
            srv.close()
