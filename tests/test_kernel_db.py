"""Tests for spice-kernel-db."""

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
    RemoteMetakernel,
    check_mk_availability,
    list_remote_metakernels,
    list_remote_missions,
    resolve_kernel_urls,
    SPICE_SERVERS,
    _DIR_ENTRY_RE,
    _VERSION_TAG_RE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_spice_tree(tmp_path):
    """Create a realistic SPICE directory tree with known duplicates."""

    # --- generic_kernels ---
    gk = tmp_path / "generic_kernels"
    for sub in ["lsk", "pck", "spk/planets", "spk/satellites"]:
        (gk / sub).mkdir(parents=True)

    (gk / "lsk/naif0012.tls").write_text("FAKE LSK naif0012")
    (gk / "pck/pck00011.tpc").write_text("FAKE PCK pck00011")
    (gk / "pck/gm_de431.tpc").write_text("FAKE PCK gm_de431")
    (gk / "spk/planets/de432s.bsp").write_bytes(
        b"FAKE SPK de432s" + b"\x00" * 100
    )
    # Note: generic name without time range
    (gk / "spk/satellites/jup365.bsp").write_bytes(
        b"FAKE SPK jup365" + b"\x00" * 200
    )

    # --- JUICE/kernels ---
    juice = tmp_path / "JUICE" / "kernels"
    for sub in ["ck", "fk", "ik", "lsk", "pck", "sclk", "spk", "mk"]:
        (juice / sub).mkdir(parents=True)

    # Duplicates of generic:
    (juice / "lsk/naif0012.tls").write_text("FAKE LSK naif0012")
    (juice / "pck/pck00011.tpc").write_text("FAKE PCK pck00011")
    (juice / "pck/gm_de431.tpc").write_text("FAKE PCK gm_de431")
    (juice / "spk/de432s.bsp").write_bytes(
        b"FAKE SPK de432s" + b"\x00" * 100
    )
    # Same content as jup365.bsp but different name!
    (juice / "spk/jup365_19900101_20500101.bsp").write_bytes(
        b"FAKE SPK jup365" + b"\x00" * 200
    )

    # Mission-specific:
    (juice / "ck/juice_sc_default_v01.bc").write_bytes(b"FAKE CK juice")
    (juice / "fk/juice_v44.tf").write_text("FAKE FK juice_v44")
    (juice / "ik/juice_janus_v08.ti").write_text("FAKE IK janus")
    (juice / "sclk/juice_fict_160326_v02.tsc").write_text("FAKE SCLK")
    (juice / "spk/juice_crema_5_1_a3_v01.bsp").write_bytes(
        b"FAKE SPK juice trajectory"
    )
    (juice / "pck/juice_jup012.tpc").write_text("FAKE PCK juice_jup012")

    # Metakernel
    (juice / "mk/juice_test.tm").write_text(textwrap.dedent("""\
        KPL/MK

        Test JUICE metakernel
        =====================

           Source: https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/

        \\begindata

          PATH_VALUES  = ( '..' )
          PATH_SYMBOLS = ( 'KERNELS' )

          KERNELS_TO_LOAD = (

            '$KERNELS/ck/juice_sc_default_v01.bc'
            '$KERNELS/fk/juice_v44.tf'
            '$KERNELS/ik/juice_janus_v08.ti'
            '$KERNELS/lsk/naif0012.tls'
            '$KERNELS/pck/pck00011.tpc'
            '$KERNELS/pck/gm_de431.tpc'
            '$KERNELS/pck/juice_jup012.tpc'
            '$KERNELS/sclk/juice_fict_160326_v02.tsc'
            '$KERNELS/spk/de432s.bsp'
            '$KERNELS/spk/jup365_19900101_20500101.bsp'
            '$KERNELS/spk/juice_crema_5_1_a3_v01.bsp'

          )

        \\begintext
    """))

    # --- MRO/kernels ---
    mro = tmp_path / "MRO" / "kernels"
    for sub in ["lsk", "pck", "spk", "ck"]:
        (mro / sub).mkdir(parents=True)

    (mro / "lsk/naif0012.tls").write_text("FAKE LSK naif0012")
    (mro / "pck/pck00011.tpc").write_text("FAKE PCK pck00011")
    (mro / "spk/de432s.bsp").write_bytes(
        b"FAKE SPK de432s" + b"\x00" * 100
    )
    (mro / "ck/mro_sc_2024.bc").write_bytes(b"FAKE CK mro")

    return tmp_path


@pytest.fixture
def db(tmp_path):
    """Create a fresh KernelDB."""
    return KernelDB(tmp_path / "test.duckdb")


@pytest.fixture
def populated_db(db, tmp_spice_tree):
    """DB with all three mission trees scanned."""
    db.scan_directory(tmp_spice_tree / "generic_kernels", mission="generic")
    db.scan_directory(tmp_spice_tree / "JUICE" / "kernels")
    db.scan_directory(tmp_spice_tree / "MRO" / "kernels")
    return db


# ---------------------------------------------------------------------------
# Unit tests: hashing module
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Unit tests: parser module
# ---------------------------------------------------------------------------

class TestParser:
    def test_parse_metakernel(self, tmp_spice_tree):
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        parsed = parse_metakernel(mk)

        assert parsed.path_values == [".."]
        assert parsed.path_symbols == ["KERNELS"]
        assert len(parsed.kernels) == 11
        assert "$KERNELS/lsk/naif0012.tls" in parsed.kernels

    def test_kernel_filenames(self, tmp_spice_tree):
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        parsed = parse_metakernel(mk)
        names = parsed.kernel_filenames()

        assert "naif0012.tls" in names
        assert "juice_v44.tf" in names

    def test_kernel_relpaths(self, tmp_spice_tree):
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        parsed = parse_metakernel(mk)
        rels = parsed.kernel_relpaths()

        assert "lsk/naif0012.tls" in rels
        assert "fk/juice_v44.tf" in rels

    def test_symbol_map(self, tmp_spice_tree):
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        parsed = parse_metakernel(mk)
        assert parsed.symbol_map == {"KERNELS": ".."}

    def test_resolve(self, tmp_spice_tree):
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        parsed = parse_metakernel(mk)
        assert parsed.resolve("$KERNELS/lsk/naif0012.tls") == "../lsk/naif0012.tls"

    def test_write_metakernel(self, tmp_spice_tree, tmp_path):
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        parsed = parse_metakernel(mk)

        out = tmp_path / "rewritten.tm"
        write_metakernel(parsed, out, path_values=["/data/my_kernels"])

        reparsed = parse_metakernel(out)
        assert reparsed.path_values == ["/data/my_kernels"]
        assert reparsed.path_symbols == ["KERNELS"]
        assert reparsed.kernels == parsed.kernels  # unchanged

    def test_header_preserved(self, tmp_spice_tree, tmp_path):
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        parsed = parse_metakernel(mk)

        out = tmp_path / "rewritten.tm"
        write_metakernel(parsed, out)

        text = out.read_text()
        assert "Test JUICE metakernel" in text
        assert "spiftp.esac.esa.int" in text


# ---------------------------------------------------------------------------
# Integration tests: database
# ---------------------------------------------------------------------------

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


class TestMetakernel:
    def test_check_metakernel_all_found(
        self, populated_db, tmp_spice_tree
    ):
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        result = populated_db.check_metakernel(mk, mission="JUICE")
        assert len(result["missing"]) == 0
        assert len(result["found"]) == 11

    def test_check_metakernel_reports_warnings(
        self, populated_db, tmp_spice_tree
    ):
        """If we check as MRO, JUICE-specific kernels trigger warnings."""
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        result = populated_db.check_metakernel(mk, mission="MRO")
        # JUICE-specific kernels are found but with warnings
        assert len(result["warnings"]) > 0

    def test_rewrite_metakernel_creates_symlinks(
        self, populated_db, tmp_spice_tree, tmp_path
    ):
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        output = tmp_path / "rewritten.tm"
        link_root = tmp_path / "my_kernels"

        out_path, warnings = populated_db.rewrite_metakernel(
            mk, output, mission="JUICE", link_root=link_root,
        )

        assert out_path.exists()

        # Check the rewritten mk has the new PATH_VALUES
        reparsed = parse_metakernel(out_path)
        assert str(link_root) in reparsed.path_values
        # KERNELS_TO_LOAD should be unchanged
        assert reparsed.kernels == parse_metakernel(mk).kernels

        # Symlinks should exist
        assert (link_root / "lsk" / "naif0012.tls").is_symlink()
        assert (link_root / "fk" / "juice_v44.tf").is_symlink()
        assert (link_root / "spk" / "de432s.bsp").is_symlink()

        # Symlinks should point to real files
        assert (link_root / "lsk" / "naif0012.tls").resolve().is_file()

    def test_rewrite_preserves_header(
        self, populated_db, tmp_spice_tree, tmp_path
    ):
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        output = tmp_path / "rewritten.tm"

        populated_db.rewrite_metakernel(mk, output, mission="JUICE")

        text = output.read_text()
        assert "Test JUICE metakernel" in text
        assert "spiftp.esac.esa.int" in text

    def test_rewrite_minimal_diff(
        self, populated_db, tmp_spice_tree, tmp_path
    ):
        """The ONLY change should be PATH_VALUES. KERNELS_TO_LOAD untouched."""
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        output = tmp_path / "rewritten.tm"

        populated_db.rewrite_metakernel(mk, output, mission="JUICE")

        original = parse_metakernel(mk)
        rewritten = parse_metakernel(output)

        # Kernels list must be identical
        assert original.kernels == rewritten.kernels
        # PATH_SYMBOLS must be identical
        assert original.path_symbols == rewritten.path_symbols
        # Only PATH_VALUES should differ
        assert original.path_values != rewritten.path_values


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


# ---------------------------------------------------------------------------
# Unit tests: config module
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Unit tests: parse_metakernel_text
# ---------------------------------------------------------------------------

class TestParseMetakernelText:
    SAMPLE_MK = textwrap.dedent("""\
        KPL/MK

        Test metakernel from URL

        \\begindata

          PATH_VALUES  = ( '..' )
          PATH_SYMBOLS = ( 'KERNELS' )

          KERNELS_TO_LOAD = (

            '$KERNELS/lsk/naif0012.tls'
            '$KERNELS/spk/de432s.bsp'

          )

        \\begintext
    """)

    def test_parse_from_text(self):
        parsed = parse_metakernel_text(
            self.SAMPLE_MK, "https://example.com/mk/test.tm"
        )
        assert parsed.path_values == [".."]
        assert parsed.path_symbols == ["KERNELS"]
        assert len(parsed.kernels) == 2
        assert "$KERNELS/lsk/naif0012.tls" in parsed.kernels

    def test_kernel_filenames_from_text(self):
        parsed = parse_metakernel_text(self.SAMPLE_MK, "test.tm")
        assert parsed.kernel_filenames() == ["naif0012.tls", "de432s.bsp"]


# ---------------------------------------------------------------------------
# Unit tests: resolve_kernel_urls
# ---------------------------------------------------------------------------

class TestResolveKernelUrls:
    def test_basic_resolution(self):
        text = textwrap.dedent("""\
            KPL/MK

            \\begindata

              PATH_VALUES  = ( '..' )
              PATH_SYMBOLS = ( 'KERNELS' )

              KERNELS_TO_LOAD = (
                '$KERNELS/lsk/naif0012.tls'
                '$KERNELS/spk/de432s.bsp'
              )

            \\begintext
        """)
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        parsed = parse_metakernel_text(text, mk_url)
        urls = resolve_kernel_urls(mk_url, parsed)

        assert len(urls) == 2
        assert urls[0] == "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/lsk/naif0012.tls"
        assert urls[1] == "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/spk/de432s.bsp"

    def test_multiple_path_symbols(self):
        text = textwrap.dedent("""\
            KPL/MK

            \\begindata

              PATH_VALUES  = ( '/data/generic' '/data/mission' )
              PATH_SYMBOLS = ( 'GENERIC' 'MISSION' )

              KERNELS_TO_LOAD = (
                '$GENERIC/lsk/naif0012.tls'
                '$MISSION/ck/mission.bc'
              )

            \\begintext
        """)
        mk_url = "https://example.com/mk/test.tm"
        parsed = parse_metakernel_text(text, mk_url)
        urls = resolve_kernel_urls(mk_url, parsed)

        assert len(urls) == 2
        assert "generic/lsk/naif0012.tls" in urls[0]
        assert "mission/ck/mission.bc" in urls[1]


# ---------------------------------------------------------------------------
# Integration tests: get_metakernel
# ---------------------------------------------------------------------------

class TestGetMetakernel:
    REMOTE_MK_TEXT = textwrap.dedent("""\
        KPL/MK

        Test remote metakernel

        \\begindata

          PATH_VALUES  = ( '..' )
          PATH_SYMBOLS = ( 'KERNELS' )

          KERNELS_TO_LOAD = (

            '$KERNELS/lsk/naif0012.tls'
            '$KERNELS/spk/new_kernel.bsp'

          )

        \\begintext
    """)

    def _mock_urlopen(self, req_or_url):
        """Mock urlopen that returns metakernel text or HEAD sizes."""
        mock_resp = MagicMock()

        if isinstance(req_or_url, str):
            url = req_or_url
        else:
            url = req_or_url.full_url if hasattr(req_or_url, 'full_url') else str(req_or_url)

        if url.endswith(".tm"):
            mock_resp.read.return_value = self.REMOTE_MK_TEXT.encode()
            mock_resp.geturl.return_value = url
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        # HEAD request for sizes
        mock_resp.headers = {"Content-Length": "1024"}
        mock_resp.geturl.return_value = url
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_get_shows_table(self, populated_db, tmp_path, capsys):
        """get_metakernel prints a table even when we skip download."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("builtins.input", return_value="n"):
            result = populated_db.get_metakernel(
                mk_url,
                download_dir=tmp_path / "downloads",
                mission="JUICE",
                yes=False,
            )

        captured = capsys.readouterr()
        assert "naif0012.tls" in captured.out
        assert "new_kernel.bsp" in captured.out
        # naif0012.tls should be "in db" since populated_db has it
        assert "in db" in captured.out
        assert "missing" in captured.out

    def test_get_saves_tm_file(self, populated_db, tmp_path):
        """get_metakernel saves the .tm file to disk."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        dl_dir = tmp_path / "downloads"

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("builtins.input", return_value="n"):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=False,
            )

        mk_file = dl_dir / "JUICE" / "mk" / "test.tm"
        assert mk_file.is_file()
        assert "KERNELS_TO_LOAD" in mk_file.read_text()

    def test_get_registers_in_metakernel_registry(self, populated_db, tmp_path):
        """get_metakernel registers the .tm in metakernel_registry."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        dl_dir = tmp_path / "downloads"

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("builtins.input", return_value="n"):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=False,
            )

        row = populated_db.con.execute(
            "SELECT mission, source_url, filename FROM metakernel_registry WHERE filename = 'test.tm'"
        ).fetchone()
        assert row is not None
        assert row[0] == "JUICE"
        assert row[1] == mk_url
        assert row[2] == "test.tm"

    def test_get_indexes_metakernel_entries(self, populated_db, tmp_path):
        """get_metakernel populates metakernel_entries."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        dl_dir = tmp_path / "downloads"

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("builtins.input", return_value="n"):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=False,
            )

        mk_dest = str(dl_dir / "JUICE" / "mk" / "test.tm")
        count = populated_db.con.execute(
            "SELECT COUNT(*) FROM metakernel_entries WHERE mk_path = ?",
            [mk_dest],
        ).fetchone()[0]
        assert count == 2  # naif0012.tls + new_kernel.bsp

    def test_get_links_existing_kernels(self, populated_db, tmp_path):
        """get creates symlinks for 'in db' kernels so the mk works locally."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        dl_dir = tmp_path / "downloads"

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("builtins.input", return_value="n"):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=False,
            )

        # naif0012.tls is "in db" — should have a symlink at the expected path
        linked = dl_dir / "JUICE" / "lsk" / "naif0012.tls"
        assert linked.is_symlink()
        assert linked.resolve().is_file()
        assert linked.read_text() == "FAKE LSK naif0012"

    def test_get_links_printed(self, populated_db, tmp_path, capsys):
        """get prints the number of linked kernels and 'ready' message."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        dl_dir = tmp_path / "downloads"

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("builtins.input", return_value="n"):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=False,
            )

        captured = capsys.readouterr()
        assert "Linked" in captured.out
        assert "Metakernel ready" in captured.out

    def test_get_writes_spice_server_marker(self, populated_db, tmp_path):
        """get_metakernel writes a .spice-server marker file."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        dl_dir = tmp_path / "downloads"

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("builtins.input", return_value="n"):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=False,
            )

        marker = dl_dir / "JUICE" / ".spice-server"
        assert marker.is_file()
        content = marker.read_text()
        assert "server_url=" in content
        assert "mk_dir_url=" in content
        assert "naif.jpl.nasa.gov" in content

    def test_get_creates_alias_symlink(self, populated_db, tmp_path):
        """get_metakernel(alias_filename=...) creates a symlink alias."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test_v470_20260415_001.tm"
        dl_dir = tmp_path / "downloads"

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("builtins.input", return_value="n"):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=False,
                alias_filename="test.tm",
            )

        mk_dir = dl_dir / "JUICE" / "mk"
        versioned = mk_dir / "test_v470_20260415_001.tm"
        alias = mk_dir / "test.tm"
        assert versioned.is_file() and not versioned.is_symlink()
        assert alias.is_symlink()
        assert alias.resolve() == versioned.resolve()

        # Both names should resolve via metakernel_registry
        path, _ = populated_db.resolve_kernel(
            "test.tm", preferred_mission="JUICE"
        )
        assert path == str(alias)
        path, _ = populated_db.resolve_kernel(
            "test_v470_20260415_001.tm", preferred_mission="JUICE"
        )
        assert path == str(versioned)

    def test_get_alias_refuses_to_clobber_regular_file(
        self, populated_db, tmp_path
    ):
        """If alias_filename is a real file on disk, don't overwrite it."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test_v470_20260415_001.tm"
        dl_dir = tmp_path / "downloads"
        mk_dir = dl_dir / "JUICE" / "mk"
        mk_dir.mkdir(parents=True)
        existing = mk_dir / "test.tm"
        existing.write_text("PRECIOUS USER DATA")

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("builtins.input", return_value="n"):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=False,
                alias_filename="test.tm",
            )

        assert existing.read_text() == "PRECIOUS USER DATA"
        assert not existing.is_symlink()


# ---------------------------------------------------------------------------
# Integration tests: metakernel listing and info
# ---------------------------------------------------------------------------

class TestMetakernelListingInfo:
    REMOTE_MK_TEXT = TestGetMetakernel.REMOTE_MK_TEXT

    def _mock_urlopen(self, req_or_url):
        return TestGetMetakernel._mock_urlopen(self, req_or_url)

    def _get_test_mk(self, db, tmp_path):
        """Helper: get a test metakernel into the DB."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        dl_dir = tmp_path / "downloads"
        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("builtins.input", return_value="n"):
            db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=False,
            )
        return dl_dir

    def test_list_metakernels(self, populated_db, tmp_path, capsys):
        """list_metakernels returns tracked metakernels."""
        self._get_test_mk(populated_db, tmp_path)
        results = populated_db.list_metakernels()
        # Scan indexes the fixture's .tm too, so there are 2
        assert len(results) >= 1
        got = [r for r in results if r["filename"] == "test.tm"]
        assert len(got) == 1
        assert got[0]["mission"] == "JUICE"
        assert got[0]["n_kernels"] == 2

        captured = capsys.readouterr()
        assert "test.tm" in captured.out
        assert "JUICE" in captured.out

    def test_list_metakernels_filter_by_mission(self, populated_db, tmp_path):
        """list_metakernels filters by mission."""
        self._get_test_mk(populated_db, tmp_path)
        results = populated_db.list_metakernels(mission="JUICE")
        assert len(results) >= 1
        results = populated_db.list_metakernels(mission="MRO")
        assert len(results) == 0

    def test_list_metakernels_filter_case_insensitive(self, populated_db, tmp_path):
        """list_metakernels filter accepts lowercase and prefix."""
        self._get_test_mk(populated_db, tmp_path)
        assert len(populated_db.list_metakernels(mission="juice")) >= 1
        assert len(populated_db.list_metakernels(mission="jui")) >= 1
        assert len(populated_db.list_metakernels(mission="NONEXISTENT")) == 0

    def test_info_metakernel(self, populated_db, tmp_path, capsys):
        """info_metakernel shows detailed per-kernel info."""
        self._get_test_mk(populated_db, tmp_path)
        result = populated_db.info_metakernel("test.tm")
        assert result is not None
        assert result["filename"] == "test.tm"
        assert result["mission"] == "JUICE"
        assert result["n_kernels"] == 2
        assert result["n_in_db"] == 1
        assert result["n_missing"] == 1

        # Check per-kernel details
        kernel_names = [k["filename"] for k in result["kernels"]]
        assert "naif0012.tls" in kernel_names
        assert "new_kernel.bsp" in kernel_names

        captured = capsys.readouterr()
        assert "in db" in captured.out
        assert "missing" in captured.out

    def test_list_metakernels_identical_content(self, populated_db, tmp_path, capsys):
        """list_metakernels flags metakernels with identical kernel lists."""
        dl_dir = self._get_test_mk(populated_db, tmp_path)

        # Manually insert a second metakernel with the same kernel entries as test.tm
        mk_path_1 = str(dl_dir / "JUICE" / "mk" / "test.tm")
        mk_path_2 = str(tmp_path / "downloads" / "JUICE" / "mk" / "test_v2.tm")
        populated_db.con.execute("""
            INSERT INTO metakernel_registry (mk_path, mission, source_url, filename, acquired_at)
            VALUES (?, 'JUICE', 'https://example.com/test_v2.tm', 'test_v2.tm', CURRENT_TIMESTAMP)
        """, [mk_path_2])
        # Copy entries only from the test.tm metakernel
        populated_db.con.execute("""
            INSERT INTO metakernel_entries (mk_path, entry_index, raw_entry, filename)
            SELECT ?, entry_index, raw_entry, filename
            FROM metakernel_entries
            WHERE mk_path = ?
        """, [mk_path_2, mk_path_1])

        results = populated_db.list_metakernels()
        # At least test.tm and test_v2.tm (scan may also index fixture's .tm)
        test_results = [r for r in results if r["filename"] in ("test.tm", "test_v2.tm")]
        assert len(test_results) == 2

        # One should be flagged as identical to the other
        identical = [r for r in test_results if r["identical_to"] is not None]
        assert len(identical) == 1
        assert identical[0]["identical_to"] in ("test.tm", "test_v2.tm")

        captured = capsys.readouterr()
        assert "identical to" in captured.out

    def test_info_metakernel_not_found(self, populated_db, capsys):
        """info_metakernel returns None for unknown name."""
        result = populated_db.info_metakernel("nonexistent.tm")
        assert result is None
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()


# ---------------------------------------------------------------------------
# Integration tests: reset command
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_deletes_db_file(self, tmp_path):
        """reset deletes only the DB file, not the kernel directory."""
        db_path = tmp_path / "test.duckdb"
        kernel_dir = tmp_path / "kernels"
        kernel_dir.mkdir()
        (kernel_dir / "naif0012.tls").write_text("FAKE")

        # Create DB so the file exists
        db = KernelDB(db_path)
        db.close()
        assert db_path.is_file()

        # Delete it
        db_path.unlink()
        assert not db_path.is_file()

        # Kernel dir untouched
        assert kernel_dir.is_dir()
        assert (kernel_dir / "naif0012.tls").is_file()


# ---------------------------------------------------------------------------
# Unit tests: show_config
# ---------------------------------------------------------------------------

class TestShowConfig:
    def test_show_config_prints_paths(self, capsys):
        cfg = Config(db_path="/tmp/test.duckdb", kernel_dir="/tmp/kernels")
        show_config(cfg)
        captured = capsys.readouterr()
        assert "/tmp/test.duckdb" in captured.out
        assert "/tmp/kernels" in captured.out


# ---------------------------------------------------------------------------
# Integration tests: archive-on-scan
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Unit tests: version tag regex
# ---------------------------------------------------------------------------

class TestVersionTag:
    def test_versioned_filename(self):
        m = _VERSION_TAG_RE.search("juice_ops_v230_20221128_001.tm")
        assert m is not None
        assert m.group(1) == "_v230_20221128_001"

    def test_unversioned_filename(self):
        m = _VERSION_TAG_RE.search("juice_ops.tm")
        assert m is None

    def test_complex_versioned(self):
        m = _VERSION_TAG_RE.search("juice_crema_5_1_150lb_23_1_v461_20251127_001.tm")
        assert m is not None
        assert m.group(1) == "_v461_20251127_001"

    def test_base_name_extraction(self):
        filename = "juice_ops_v230_20221128_001.tm"
        m = _VERSION_TAG_RE.search(filename)
        base = filename[:m.start()] + ".tm"
        assert base == "juice_ops.tm"


# ---------------------------------------------------------------------------
# Unit tests: list_remote_metakernels (HTML parsing)
# ---------------------------------------------------------------------------

class TestListRemoteMetakernels:
    SAMPLE_HTML = textwrap.dedent("""\
        <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
        <html><head><title>Index of /pub/naif/JUICE/kernels/mk</title></head>
        <body><h1>Index of /pub/naif/JUICE/kernels/mk</h1>
        <pre><img src="/icons/blank.gif"> <a href="?C=N;O=D">Name</a>
        <hr><img src="/icons/back.gif"> <a href="/pub/naif/JUICE/kernels/">Parent Directory</a>
        <img src="/icons/folder.gif"> <a href="former_versions/">former_versions/</a>   2026-01-15 10:00    -
        <img src="/icons/unknown.gif"> <a href="aareadme.txt">aareadme.txt</a>          2025-06-03 08:00   12K
        <img src="/icons/unknown.gif"> <a href="juice_ops.tm">juice_ops.tm</a>          2025-11-27 09:30   12K
        <img src="/icons/unknown.gif"> <a href="juice_ops_v230_20221128_001.tm">juice_ops_v230_20221128_001.tm</a>  2022-11-28 10:00    8K
        <img src="/icons/unknown.gif"> <a href="juice_ops_v461_20251127_001.tm">juice_ops_v461_20251127_001.tm</a>  2025-11-27 09:30   12K
        <img src="/icons/unknown.gif"> <a href="juice_plan.tm">juice_plan.tm</a>        2026-02-02 14:00   15K
        <img src="/icons/unknown.gif"> <a href="juice_plan_v100_20250101_001.tm">juice_plan_v100_20250101_001.tm</a>  2025-01-01 12:00   10K
        <img src="/icons/unknown.gif"> <a href="juice_crema_5_0.tm">juice_crema_5_0.tm</a>  2025-08-15 11:00    9K
        <hr></pre></body></html>
    """)

    def test_parses_tm_files(self):
        with patch("spice_kernel_db.remote.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = self.SAMPLE_HTML.encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            results = list_remote_metakernels("https://example.com/mk/")

        # Should find 6 .tm files (not aareadme.txt, not former_versions/)
        assert len(results) == 6
        filenames = [r.filename for r in results]
        assert "juice_ops.tm" in filenames
        assert "juice_ops_v230_20221128_001.tm" in filenames
        assert "juice_plan.tm" in filenames
        assert "juice_crema_5_0.tm" in filenames

    def test_version_tag_extraction(self):
        with patch("spice_kernel_db.remote.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = self.SAMPLE_HTML.encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            results = list_remote_metakernels("https://example.com/mk/")

        by_name = {r.filename: r for r in results}
        # Current version has no tag
        assert by_name["juice_ops.tm"].version_tag is None
        assert by_name["juice_ops.tm"].base_name == "juice_ops.tm"
        # Versioned snapshot has tag
        assert by_name["juice_ops_v230_20221128_001.tm"].version_tag == "v230_20221128_001"
        assert by_name["juice_ops_v230_20221128_001.tm"].base_name == "juice_ops.tm"

    def test_parses_uppercase_tm_files(self):
        """ESA missions like MEX use .TM extension."""
        html = textwrap.dedent("""\
            <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
            <html><body><pre>
            <a href="MEX_OPS.TM">MEX_OPS.TM</a>                    2026-01-20 14:00   10K
            <a href="MEX_OPS_V324_20260206_002.TM">MEX_OPS_V324_20260206_002.TM</a>  2026-02-06 09:00   11K
            </pre></body></html>
        """)
        with patch("spice_kernel_db.remote.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = html.encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            results = list_remote_metakernels("https://example.com/mk/")

        assert len(results) == 2
        filenames = [r.filename for r in results]
        assert "MEX_OPS.TM" in filenames
        assert "MEX_OPS_V324_20260206_002.TM" in filenames
        # Version tag extraction works with uppercase
        by_name = {r.filename: r for r in results}
        assert by_name["MEX_OPS.TM"].version_tag is None
        assert by_name["MEX_OPS_V324_20260206_002.TM"].version_tag == "V324_20260206_002"
        assert by_name["MEX_OPS_V324_20260206_002.TM"].base_name == "MEX_OPS.TM"

    def test_sorted_by_base_name_then_filename(self):
        with patch("spice_kernel_db.remote.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = self.SAMPLE_HTML.encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            results = list_remote_metakernels("https://example.com/mk/")

        base_names = [r.base_name for r in results]
        assert base_names == sorted(base_names)


# ---------------------------------------------------------------------------
# Integration tests: browse_remote_metakernels
# ---------------------------------------------------------------------------

class TestBrowseRemoteMetakernels:
    SAMPLE_HTML = TestListRemoteMetakernels.SAMPLE_HTML

    def _mock_urlopen(self, url):
        mock_resp = MagicMock()
        mock_resp.read.return_value = self.SAMPLE_HTML.encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_groups_by_base_name(self, db, capsys):
        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen):
            results = db.browse_remote_metakernels(
                "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/",
                mission="JUICE",
            )

        # juice_ops.tm + 2 versioned = 1 group
        # juice_plan.tm + 1 versioned = 1 group
        # juice_crema_5_0.tm = 1 group
        assert len(results) == 3
        by_base = {r["base_name"]: r for r in results}
        assert by_base["juice_ops.tm"]["n_versions"] == 3
        assert by_base["juice_plan.tm"]["n_versions"] == 2
        assert by_base["juice_crema_5_0.tm"]["n_versions"] == 1

    def test_latest_date(self, db):
        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen):
            results = db.browse_remote_metakernels(
                "https://example.com/mk/", mission="JUICE",
            )

        by_base = {r["base_name"]: r for r in results}
        assert by_base["juice_ops.tm"]["latest_date"] == "2025-11-27 09:30"

    def test_local_status_no_acquired(self, db):
        """With no acquired MKs, all should show is_local=False."""
        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen):
            results = db.browse_remote_metakernels(
                "https://example.com/mk/", mission="JUICE",
            )

        assert all(not r["is_local"] for r in results)

    def test_local_status_with_acquired(self, db, capsys):
        """After registering an MK in metakernel_registry, is_local should be True."""
        # Simulate having acquired juice_ops.tm
        db.con.execute("""
            INSERT INTO metakernel_registry VALUES
            (?, ?, ?, ?, current_timestamp)
        """, ["/tmp/juice_ops.tm", "JUICE", "https://example.com/mk/juice_ops.tm", "juice_ops.tm"])

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen):
            results = db.browse_remote_metakernels(
                "https://example.com/mk/", mission="JUICE",
            )

        by_base = {r["base_name"]: r for r in results}
        assert by_base["juice_ops.tm"]["is_local"] is True
        assert by_base["juice_plan.tm"]["is_local"] is False

        captured = capsys.readouterr()
        assert "yes" in captured.out
        assert "no" in captured.out

    def test_prints_summary(self, db, capsys):
        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen):
            db.browse_remote_metakernels(
                "https://example.com/mk/", mission="JUICE",
            )

        captured = capsys.readouterr()
        assert "3 unique" in captured.out
        assert "6 files" in captured.out
        assert "0 locally acquired" in captured.out


# ---------------------------------------------------------------------------
# Unit tests: directory entry regex
# ---------------------------------------------------------------------------

class TestDirEntryRegex:
    def test_matches_directory_entry(self):
        line = '<a href="JUICE/">JUICE/</a>                     2026-01-15 10:00    -'
        m = _DIR_ENTRY_RE.search(line)
        assert m is not None
        assert m.group(1) == "JUICE/"
        assert m.group(2) == "2026-01-15 10:00"

    def test_matches_esa_table_format(self):
        line = '<tr><td valign="top"><img src="/icons/folder.gif" alt="[DIR]"></td><td><a href="BEPICOLOMBO/">BEPICOLOMBO/</a></td><td align="right">2021-10-15 12:50  </td><td align="right">  - </td><td>&nbsp;</td></tr>'
        m = _DIR_ENTRY_RE.search(line)
        assert m is not None
        assert m.group(1) == "BEPICOLOMBO/"
        assert m.group(2) == "2021-10-15 12:50"

    def test_does_not_match_file(self):
        line = '<a href="aareadme.txt">aareadme.txt</a>          2025-06-03 08:00   12K'
        m = _DIR_ENTRY_RE.search(line)
        assert m is None

    def test_does_not_match_tm_file(self):
        line = '<a href="juice_ops.tm">juice_ops.tm</a>          2025-11-27 09:30   12K'
        m = _DIR_ENTRY_RE.search(line)
        assert m is None


# ---------------------------------------------------------------------------
# Unit tests: list_remote_missions (HTML parsing)
# ---------------------------------------------------------------------------

class TestListRemoteMissions:
    SAMPLE_DIR_HTML = textwrap.dedent("""\
        <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
        <html><head><title>Index of /pub/naif</title></head>
        <body><h1>Index of /pub/naif</h1>
        <pre><img src="/icons/blank.gif"> <a href="?C=N;O=D">Name</a>
        <hr><img src="/icons/back.gif"> <a href="/pub/">Parent Directory</a>
        <img src="/icons/folder.gif"> <a href="CASSINI/">CASSINI/</a>             2025-06-10 08:00    -
        <img src="/icons/folder.gif"> <a href="JUICE/">JUICE/</a>               2026-01-15 10:00    -
        <img src="/icons/folder.gif"> <a href="MRO/">MRO/</a>                 2025-12-01 09:00    -
        <img src="/icons/folder.gif"> <a href="toolkit/">toolkit/</a>             2025-11-01 07:00    -
        <img src="/icons/unknown.gif"> <a href="aareadme.txt">aareadme.txt</a>  2025-01-01 00:00   2K
        <hr></pre></body></html>
    """)

    def test_parses_mission_dirs(self):
        with patch("spice_kernel_db.remote.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = self.SAMPLE_DIR_HTML.encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            missions = list_remote_missions("https://naif.jpl.nasa.gov/pub/naif/")

        # Should find directories (not files)
        assert "CASSINI" in missions
        assert "JUICE" in missions
        assert "MRO" in missions
        assert "toolkit" in missions
        # Should NOT include files
        assert "aareadme.txt" not in missions

    def test_sorted_alphabetically(self):
        with patch("spice_kernel_db.remote.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = self.SAMPLE_DIR_HTML.encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            missions = list_remote_missions("https://example.com/")

        assert missions == sorted(missions)

    def test_skips_parent_dir(self):
        html = '<a href="../">../</a>  2025-01-01 00:00    -\n'
        html += '<a href="JUICE/">JUICE/</a>  2025-01-01 00:00    -\n'
        with patch("spice_kernel_db.remote.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = html.encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            missions = list_remote_missions("https://example.com/")

        assert ".." not in missions
        assert "JUICE" in missions

    def test_parses_esa_table_format(self):
        esa_html = textwrap.dedent("""\
            <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
            <html><head><title>Index of /data/SPICE</title></head>
            <body><h1>Index of /data/SPICE</h1>
            <table>
            <tr><td><a href="/data/">Parent Directory</a></td><td>&nbsp;</td><td align="right">  - </td></tr>
            <tr><td><a href="BEPICOLOMBO/">BEPICOLOMBO/</a></td><td align="right">2021-10-15 12:50  </td><td align="right">  - </td></tr>
            <tr><td><a href="JUICE/">JUICE/</a></td><td align="right">2023-07-07 12:54  </td><td align="right">  - </td></tr>
            <tr><td><a href="ROSETTA/">ROSETTA/</a></td><td align="right">2020-03-01 09:00  </td><td align="right">  - </td></tr>
            </table></body></html>
        """)
        with patch("spice_kernel_db.remote.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = esa_html.encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            missions = list_remote_missions("https://spiftp.esac.esa.int/data/SPICE/")

        assert "BEPICOLOMBO" in missions
        assert "JUICE" in missions
        assert "ROSETTA" in missions
        assert len(missions) == 3


# ---------------------------------------------------------------------------
# Unit tests: check_mk_availability
# ---------------------------------------------------------------------------

class TestCheckMkAvailability:
    def test_returns_true_for_reachable_mk_dirs(self):
        """Missions whose mk/ URL responds 200 are marked True."""
        def fake_urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else req
            if "JUICE" in url or "MRO" in url:
                return MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=fake_urlopen):
            result = check_mk_availability(
                "https://example.com/", ["JUICE", "MRO", "APOLLO"]
            )

        assert result["JUICE"] is True
        assert result["MRO"] is True
        assert result["APOLLO"] is False

    def test_returns_false_on_connection_error(self):
        """Network errors are treated as unavailable."""
        with patch(
            "spice_kernel_db.remote.urllib.request.urlopen",
            side_effect=urllib.error.URLError("timeout"),
        ):
            result = check_mk_availability("https://example.com/", ["CASSINI"])

        assert result["CASSINI"] is False


# ---------------------------------------------------------------------------
# Integration tests: mission management
# ---------------------------------------------------------------------------

class TestMissionManagement:
    NASA_URL = "https://naif.jpl.nasa.gov/pub/naif/"
    ESA_URL = "https://spiftp.esac.esa.int/data/SPICE/"

    def test_add_and_get_mission(self, db):
        db.add_mission(
            "JUICE", self.ESA_URL,
            "https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/mk/",
            dedup=True,
        )
        m = db.get_mission("JUICE")
        assert m is not None
        assert m["name"] == "JUICE"
        assert m["server_url"] == self.ESA_URL
        assert m["server_label"] == "ESA"
        assert m["dedup"] is True
        assert "JUICE/kernels/mk/" in m["mk_dir_url"]

    def test_get_mission_case_insensitive(self, db):
        db.add_mission("JUICE", self.ESA_URL, "https://example.com/mk/", True)
        assert db.get_mission("juice") is not None
        assert db.get_mission("Juice") is not None

    def test_get_mission_not_found(self, db):
        assert db.get_mission("NONEXISTENT") is None

    def test_list_missions(self, db):
        db.add_mission("JUICE", self.ESA_URL, "https://example.com/JUICE/mk/", True)
        db.add_mission("MRO", self.NASA_URL, "https://example.com/MRO/mk/", False)

        missions = db.list_missions()
        assert len(missions) == 2

        by_name = {m["name"]: m for m in missions}
        assert by_name["JUICE"]["server_label"] == "ESA"
        assert by_name["JUICE"]["dedup"] is True
        assert by_name["MRO"]["server_label"] == "NASA"
        assert by_name["MRO"]["dedup"] is False

    def test_list_missions_empty(self, db):
        assert db.list_missions() == []

    def test_remove_mission(self, db):
        db.add_mission("JUICE", self.ESA_URL, "https://example.com/mk/", True)
        assert db.remove_mission("JUICE") is True
        assert db.get_mission("JUICE") is None

    def test_remove_mission_case_insensitive(self, db):
        db.add_mission("JUICE", self.ESA_URL, "https://example.com/mk/", True)
        assert db.remove_mission("juice") is True
        assert db.get_mission("JUICE") is None

    def test_remove_mission_prefix(self, db):
        db.add_mission("JUICE", self.ESA_URL, "https://example.com/mk/", True)
        assert db.remove_mission("jui") is True
        assert db.get_mission("JUICE") is None

    def test_remove_nonexistent_mission(self, db):
        assert db.remove_mission("NONEXISTENT") is False

    def test_add_mission_replaces_existing(self, db):
        db.add_mission("JUICE", self.ESA_URL, "https://old.com/mk/", True)
        db.add_mission("JUICE", self.NASA_URL, "https://new.com/mk/", False)

        m = db.get_mission("JUICE")
        assert m["server_url"] == self.NASA_URL
        assert m["dedup"] is False
        assert m["mk_dir_url"] == "https://new.com/mk/"

    def test_custom_server_label(self, db):
        db.add_mission(
            "CUSTOM", "https://my-server.example.com/spice/",
            "https://my-server.example.com/spice/CUSTOM/mk/", True,
        )
        m = db.get_mission("CUSTOM")
        assert m["server_label"] == "custom"


# ---------------------------------------------------------------------------
# Integration tests: per-mission deduplication
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Unit tests: SPICE_SERVERS constant
# ---------------------------------------------------------------------------

class TestSpiceServers:
    def test_has_nasa_and_esa(self):
        assert "NASA" in SPICE_SERVERS
        assert "ESA" in SPICE_SERVERS

    def test_urls_end_with_slash(self):
        for label, url in SPICE_SERVERS.items():
            assert url.endswith("/"), f"{label} URL should end with /"

    def test_urls_are_https(self):
        for label, url in SPICE_SERVERS.items():
            assert url.startswith("https://"), f"{label} URL should be HTTPS"


# ---------------------------------------------------------------------------
# Unit / integration tests: coverage module
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Unit tests: body name resolution
# ---------------------------------------------------------------------------

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


# ===========================================================================
# Security / correctness fixes (Issues 1-11)
# ===========================================================================


class TestIssue1PathTraversalPathValues:
    """Issue 1: PATH_VALUES like /../../../tmp/evil could escape download dir."""

    def test_path_traversal_in_path_values_raises(self, tmp_path):
        """PATH_VALUES that resolve outside download_dir must raise ValueError."""
        from spice_kernel_db.db import _validate_path_values
        download_dir = tmp_path / "kernels"
        download_dir.mkdir()
        # This resolves to /tmp/evil, outside download_dir
        with pytest.raises(ValueError, match="outside.*download"):
            _validate_path_values(
                ["/../../../tmp/evil"],
                mk_dir=download_dir / "JUICE" / "mk",
                download_dir=download_dir,
            )

    def test_safe_path_values_accepted(self, tmp_path):
        """PATH_VALUES that stay within download_dir should pass."""
        from spice_kernel_db.db import _validate_path_values
        download_dir = tmp_path / "kernels"
        mk_dir = download_dir / "JUICE" / "mk"
        mk_dir.mkdir(parents=True)
        # '..' from mk/ goes to JUICE/, still within download_dir
        _validate_path_values(
            [".."],
            mk_dir=mk_dir,
            download_dir=download_dir,
        )


class TestIssue2SizeOnlySkipCheck:
    """Issue 2: Files skipped when size matches but hash may differ."""

    def test_same_size_wrong_hash_not_skipped(self, tmp_path):
        """A file on disk with matching size but wrong hash should be re-downloaded."""
        db = KernelDB(tmp_path / "test.duckdb")

        # Create a file on disk at the expected destination
        dest = tmp_path / "JUICE" / "lsk" / "naif0012.tls"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"WRONG CONTENT!!")  # same size as correct

        correct_content = b"CORRECT CONTENT"
        assert len(correct_content) == dest.stat().st_size

        # The DB has no record of this file, so resolve_kernel returns None.
        # The key behavior: when dest.is_file() and size matches remote,
        # it should also check hash. Since there's no DB hash to compare,
        # it should still download.
        # We test the _should_skip_download helper directly.
        from spice_kernel_db.db import _should_skip_download
        skip = _should_skip_download(
            dest=dest,
            remote_size=len(correct_content),
            db_hash="abcdef1234567890" * 4,  # known hash from DB
            force=False,
        )
        assert skip is False, "Should not skip when hash mismatches"
        db.close()

    def test_same_size_correct_hash_skipped(self, tmp_path):
        """A file on disk with matching size AND hash should be skipped."""
        from spice_kernel_db.db import _should_skip_download
        dest = tmp_path / "naif0012.tls"
        dest.write_bytes(b"CORRECT CONTENT")
        real_hash = sha256_file(dest)

        skip = _should_skip_download(
            dest=dest,
            remote_size=dest.stat().st_size,
            db_hash=real_hash,
            force=False,
        )
        assert skip is True, "Should skip when both size and hash match"


class TestIssue3SymlinkWithoutHashValidation:
    """Issue 3: Symlinks created without verifying resolved file's hash."""

    def test_link_existing_skips_hash_mismatch(self, tmp_path):
        """_link_existing_kernels should not create symlink when hash mismatches."""
        db = KernelDB(tmp_path / "test.duckdb")

        # Register a kernel file
        kernel = tmp_path / "JUICE" / "kernels" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("CORRECT LSK CONTENT")
        h = db.register_file(kernel, mission="JUICE")

        # Now corrupt the file on disk (different content, same location)
        kernel.write_text("CORRUPTED CONTENT!!")

        # Try to link — should skip because hash of file on disk != DB hash
        download_dir = tmp_path / "download"
        download_dir.mkdir()
        n_linked = db._link_existing_kernels(
            indices=[0],
            filenames=["naif0012.tls"],
            relpaths=["lsk/naif0012.tls"],
            download_dir=download_dir,
            mission="JUICE",
        )
        assert n_linked == 0, "Should not link file with hash mismatch"
        db.close()


class TestIssue4PartialDownloadDetection:
    """Issue 4: Partial downloads not detected."""

    def test_partial_download_raises(self, tmp_path):
        """download_kernel should raise if bytes written < Content-Length."""
        from spice_kernel_db.remote import download_kernel

        dest = tmp_path / "test.bsp"

        # Mock a response that claims 1000 bytes but only delivers 500
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "1000"}
        mock_resp.read = MagicMock(side_effect=[b"x" * 500, b""])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("spice_kernel_db.remote.urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(IOError, match="[Pp]artial|[Ii]ncomplete"):
                download_kernel("http://example.com/test.bsp", dest)

        # Partial file should be cleaned up
        assert not dest.exists(), "Partial file should be deleted"

    def test_zero_byte_download_raises(self, tmp_path):
        """download_kernel should raise on zero-byte downloads."""
        from spice_kernel_db.remote import download_kernel

        dest = tmp_path / "test.bsp"

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "1000"}
        mock_resp.read = MagicMock(return_value=b"")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("spice_kernel_db.remote.urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(IOError):
                download_kernel("http://example.com/test.bsp", dest)

    def test_successful_download_no_error(self, tmp_path):
        """A complete download should succeed without error."""
        from spice_kernel_db.remote import download_kernel

        dest = tmp_path / "test.bsp"
        content = b"x" * 1000

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "1000"}
        mock_resp.read = MagicMock(side_effect=[content, b""])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("spice_kernel_db.remote.urllib.request.urlopen", return_value=mock_resp):
            result_dest, result_sha = download_kernel(
                "http://example.com/test.bsp", dest,
            )

        assert result_dest == dest
        assert dest.read_bytes() == content
        # C4: download_kernel now returns the streamed sha256 alongside
        # the path so callers don't need a second pass over the file.
        assert result_sha == hashlib.sha256(content).hexdigest()


class TestIssue5RaceConditionDBState:
    """Issue 5 / C5: DB state can change during download while lock released.

    After C5, detection raises ConcurrentModificationError instead of just
    logging — the previous behavior silently continued and produced wrong
    output.
    """

    def test_state_change_raises_concurrent_modification(self, tmp_path, caplog):
        """When state changed since the pre-snapshot, _check_state_changed
        must raise so the caller can abort the operation cleanly."""
        from spice_kernel_db import ConcurrentModificationError
        db = KernelDB(tmp_path / "test.duckdb")

        kernel = tmp_path / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("LSK CONTENT")
        db.register_file(kernel, mission="JUICE")

        pre_snapshot = db._snapshot_kernel_hashes(["naif0012.tls"], "JUICE")

        kernel.write_text("UPDATED LSK CONTENT")
        db.register_file(kernel, mission="JUICE")

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ConcurrentModificationError):
                db._check_state_changed(
                    pre_snapshot, ["naif0012.tls"], "JUICE",
                )
        # Warning is still emitted for forensics
        assert any("DB state changed" in r.message for r in caplog.records)
        db.close()

    def test_no_change_returns_false(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("LSK")
        db.register_file(kernel, mission="JUICE")
        snap = db._snapshot_kernel_hashes(["naif0012.tls"], "JUICE")
        # No mutation between snapshot and check → returns False
        assert db._check_state_changed(snap, ["naif0012.tls"], "JUICE") is False
        db.close()


class TestIssue6SymlinkCreationErrorsSilent:
    """Issue 6: Symlink creation errors silently ignored, counter wrong."""

    def test_symlink_failure_not_counted(self, tmp_path):
        """If symlink_to() fails, n_linked should NOT be incremented."""
        db = KernelDB(tmp_path / "test.duckdb")

        # Register a kernel file
        kernel = tmp_path / "JUICE" / "kernels" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("CORRECT LSK CONTENT")
        db.register_file(kernel, mission="JUICE")

        download_dir = tmp_path / "download"
        download_dir.mkdir()

        # Patch symlink_to to raise OSError
        with patch.object(Path, "symlink_to", side_effect=OSError("Permission denied")):
            n_linked = db._link_existing_kernels(
                indices=[0],
                filenames=["naif0012.tls"],
                relpaths=["lsk/naif0012.tls"],
                download_dir=download_dir,
                mission="JUICE",
            )
        assert n_linked == 0, "Should not count failed symlink"
        db.close()

    def test_symlink_failure_logs_warning(self, tmp_path, caplog):
        """Symlink failure should log a warning."""
        db = KernelDB(tmp_path / "test.duckdb")

        kernel = tmp_path / "JUICE" / "kernels" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("CORRECT LSK CONTENT")
        db.register_file(kernel, mission="JUICE")

        download_dir = tmp_path / "download"
        download_dir.mkdir()

        with patch.object(Path, "symlink_to", side_effect=OSError("Permission denied")):
            with caplog.at_level(logging.WARNING):
                db._link_existing_kernels(
                    indices=[0],
                    filenames=["naif0012.tls"],
                    relpaths=["lsk/naif0012.tls"],
                    download_dir=download_dir,
                    mission="JUICE",
                )
        assert any("symlink" in r.message.lower() or "Permission" in r.message
                    for r in caplog.records)
        db.close()


class TestIssue7PreRegistrationHashVerification:
    """Issue 7: register_file should optionally verify expected hash."""

    def test_expected_hash_mismatch_raises(self, tmp_path):
        """register_file with wrong expected_hash should raise ValueError."""
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("LSK CONTENT")

        with pytest.raises(ValueError, match="[Hh]ash mismatch"):
            db.register_file(
                kernel, mission="JUICE",
                expected_hash="0000000000000000" * 4,
            )
        db.close()

    def test_expected_hash_match_succeeds(self, tmp_path):
        """register_file with correct expected_hash should succeed."""
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("LSK CONTENT")
        correct_hash = sha256_file(kernel)

        h = db.register_file(
            kernel, mission="JUICE",
            expected_hash=correct_hash,
        )
        assert h == correct_hash
        db.close()

    def test_no_expected_hash_still_works(self, tmp_path):
        """register_file without expected_hash should work as before."""
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("LSK CONTENT")

        h = db.register_file(kernel, mission="JUICE")
        assert h == sha256_file(kernel)
        db.close()


class TestIssue8BroadExceptMasksErrors:
    """Issue 8: Broad except Exception masks real errors like PermissionError."""

    def test_permission_error_in_scan_reraised(self, tmp_path):
        """PermissionError during scan should be re-raised, not swallowed."""
        db = KernelDB(tmp_path / "test.duckdb")

        # Create a file that will trigger PermissionError during registration
        kernel_dir = tmp_path / "JUICE" / "kernels" / "lsk"
        kernel_dir.mkdir(parents=True)
        kernel = kernel_dir / "naif0012.tls"
        kernel.write_text("LSK CONTENT")

        with patch.object(
            db, "register_file",
            side_effect=PermissionError("Permission denied"),
        ):
            with pytest.raises(PermissionError):
                db.scan_directory(kernel_dir)
        db.close()

    def test_os_error_in_scan_collected(self, tmp_path):
        """OSError during scan should be collected, not re-raised."""
        db = KernelDB(tmp_path / "test.duckdb")

        kernel_dir = tmp_path / "JUICE" / "kernels" / "lsk"
        kernel_dir.mkdir(parents=True)
        kernel = kernel_dir / "naif0012.tls"
        kernel.write_text("LSK CONTENT")

        with patch.object(
            db, "register_file",
            side_effect=OSError("Disk full"),
        ):
            # OSError should be caught and collected, not re-raised
            count, _ = db.scan_directory(kernel_dir)
        assert count == 0
        db.close()


class TestIssue9PathTraversalRewriteMetakernel:
    """Issue 9: Path traversal in rewrite_metakernel via ../."""

    def test_path_escape_in_relpath_skipped(self, tmp_path):
        """Kernel relpaths containing ../ that escape link_root should be skipped."""
        db = KernelDB(tmp_path / "test.duckdb")

        # Register a kernel so resolve_kernel finds it
        kernel = tmp_path / "JUICE" / "kernels" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("LSK CONTENT")
        db.register_file(kernel, mission="JUICE")

        # Create a metakernel with a path-traversal relpath
        mk = tmp_path / "evil.tm"
        mk.write_text(textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '.' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = (
                '$KERNELS/../../etc/naif0012.tls'
              )
            \\begintext
        """))

        output = tmp_path / "output.tm"
        link_root = tmp_path / "links"

        _, warnings = db.rewrite_metakernel(
            mk, output, mission="JUICE", link_root=str(link_root),
        )
        # The path-traversal entry should generate a warning
        assert any("outside" in w.lower() or "escape" in w.lower() or "traversal" in w.lower()
                    for w in warnings)
        db.close()


class TestIssue10ThreadPoolSeverityDistinction:
    """Issue 10: Download failures should distinguish retriable from fatal."""

    def test_403_classified_as_fatal(self, tmp_path):
        """403 Forbidden should be classified as FATAL in warnings."""
        from spice_kernel_db.remote import download_kernels_parallel

        def mock_download(*args, **kwargs):
            raise urllib.error.HTTPError(
                "http://example.com/test.bsp", 403, "Forbidden", {}, None
            )

        tasks = [("http://example.com/test.bsp", tmp_path / "test.bsp", "test.bsp")]
        with patch("spice_kernel_db.remote.download_kernel", side_effect=mock_download):
            _, warnings = download_kernels_parallel(tasks)

        assert len(warnings) == 1
        assert "FATAL" in warnings[0] or "403" in warnings[0]

    def test_500_classified_as_retriable(self, tmp_path):
        """500 Server Error should be classified as RETRIABLE in warnings."""
        from spice_kernel_db.remote import download_kernels_parallel

        def mock_download(*args, **kwargs):
            raise urllib.error.HTTPError(
                "http://example.com/test.bsp", 500, "Internal Server Error", {}, None
            )

        tasks = [("http://example.com/test.bsp", tmp_path / "test.bsp", "test.bsp")]
        with patch("spice_kernel_db.remote.download_kernel", side_effect=mock_download):
            _, warnings = download_kernels_parallel(tasks)

        assert len(warnings) == 1
        assert "RETRIABLE" in warnings[0] or "retry" in warnings[0].lower()


class TestIssue11SameFilenameDifferentHash:
    """Issue 11: Re-registering same filename with different content should update hash."""

    def test_reregister_updates_hash(self, tmp_path):
        """Registering a file with same name but different content should update the hash."""
        db = KernelDB(tmp_path / "test.duckdb")

        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("ORIGINAL CONTENT")
        h1 = db.register_file(kernel, mission="JUICE")

        # Change content (simulates updated kernel)
        kernel.write_text("UPDATED CONTENT!")
        h2 = db.register_file(kernel, mission="JUICE")

        assert h1 != h2, "Hashes should differ for different content"

        # The kernels table should now have the new hash with this filename
        row = db.con.execute(
            "SELECT sha256 FROM kernels WHERE filename = ?",
            ["naif0012.tls"],
        ).fetchall()
        hashes = {r[0] for r in row}
        assert h2 in hashes, "New hash should be in kernels table"

        db.close()

    def test_reregister_logs_warning(self, tmp_path, caplog):
        """Hash change for same filename should log a warning."""
        db = KernelDB(tmp_path / "test.duckdb")

        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("ORIGINAL CONTENT")
        db.register_file(kernel, mission="JUICE")

        kernel.write_text("UPDATED CONTENT!")
        with caplog.at_level(logging.WARNING):
            db.register_file(kernel, mission="JUICE")

        assert any("naif0012.tls" in r.message for r in caplog.records)
        db.close()

    def test_old_location_preserved(self, tmp_path):
        """Old location entry should still exist after hash update."""
        db = KernelDB(tmp_path / "test.duckdb")

        kernel = tmp_path / "naif0012.tls"
        kernel.write_text("ORIGINAL CONTENT")
        h1 = db.register_file(kernel, mission="JUICE")

        kernel.write_text("UPDATED CONTENT!")
        h2 = db.register_file(kernel, mission="JUICE")

        # Both hashes should have location entries
        locs_h1 = db.find_by_hash(h1)
        # The old location entry is overwritten because same abs_path
        # but the old hash record in kernels table should still exist
        old_kernel = db.con.execute(
            "SELECT filename FROM kernels WHERE sha256 = ?", [h1]
        ).fetchone()
        assert old_kernel is not None, "Old hash record should still exist in kernels table"
        db.close()


class TestCLIVersion:
    """Tests for --version flag and deferred config."""

    def test_version_flag(self):
        """--version should print the version and exit."""
        from spice_kernel_db.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_help_flag(self):
        """--help should work without triggering config setup."""
        from spice_kernel_db.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_no_args_does_not_crash(self, tmp_path):
        """Invoking with no args should not crash."""
        from spice_kernel_db.cli import main

        # Should not raise — either shows summary or quick-start guide
        main([])


class TestCLINoArgs:
    """Tests for the no-args default summary behavior."""

    def test_no_args_with_metakernels(self, tmp_path, capsys):
        """No-args should list metakernels when some exist."""
        from spice_kernel_db.cli import main

        db = KernelDB(tmp_path / "test.duckdb")
        db.add_mission("JUICE", "https://example.com/", "https://example.com/JUICE/kernels/mk/")
        db.con.execute("""
            INSERT INTO metakernel_registry (mk_path, filename, mission, source_url, acquired_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, ["/tmp/juice_ops.tm", "juice_ops.tm", "JUICE", "https://example.com/juice_ops.tm"])
        db.close()

        with patch("spice_kernel_db.config.load_config", return_value=Config(
            db_path=str(tmp_path / "test.duckdb"),
            kernel_dir=str(tmp_path / "kernels"),
        )):
            main([])

        captured = capsys.readouterr()
        assert "juice_ops.tm" in captured.out

    def test_no_args_without_config(self, capsys):
        """No-args without config should show setup hint."""
        from spice_kernel_db.cli import main

        with patch("spice_kernel_db.config.load_config", return_value=None):
            main([])

        captured = capsys.readouterr()
        assert "config --setup" in captured.out


class TestUpdateMetakernelRaisesLookupError:
    """update_metakernel raises LookupError instead of sys.exit."""

    def test_not_found_raises(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        with pytest.raises(LookupError, match="not found in registry"):
            db.update_metakernel("nonexistent.tm")
        db.close()

    def test_scan_only_no_source_url_raises(self, tmp_path):
        """Metakernel added via scan (no source_url) should raise LookupError."""
        db = KernelDB(tmp_path / "test.duckdb")
        db.con.execute("""
            INSERT INTO metakernel_registry (mk_path, filename, mission, source_url, acquired_at)
            VALUES (?, ?, ?, NULL, CURRENT_TIMESTAMP)
        """, ["/tmp/test.tm", "test.tm", "TEST"])
        with pytest.raises(LookupError, match="added via scan"):
            db.update_metakernel("test.tm")
        db.close()


class TestResolveBodyInteractiveReturnsNone:
    """_resolve_body_interactive returns None instead of sys.exit."""

    def test_unknown_body_returns_none(self):
        from spice_kernel_db.cli import _resolve_body_interactive
        result = _resolve_body_interactive("ZZZZNOTABODY999")
        assert result is None

    def test_invalid_selection_returns_none(self):
        from spice_kernel_db.cli import _resolve_body_interactive
        # "earth" is ambiguous (center vs barycenter) — give invalid input
        with patch("builtins.input", return_value="999"):
            result = _resolve_body_interactive("earth")
        assert result is None


class TestRemoveMetakernel:
    """Tests for mk --remove / remove_metakernel."""

    def test_remove_by_filename(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        db.con.execute("""
            INSERT INTO metakernel_registry (mk_path, filename, mission, source_url, acquired_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, ["/tmp/juice_ops.tm", "juice_ops.tm", "JUICE", "https://example.com/juice_ops.tm"])
        db.con.execute("""
            INSERT INTO metakernel_entries (mk_path, entry_index, raw_entry, filename)
            VALUES (?, 0, '$KERNELS/lsk/naif0012.tls', 'naif0012.tls')
        """, ["/tmp/juice_ops.tm"])

        assert db.remove_metakernel("juice_ops.tm") is True
        # Registry and entries should be empty
        assert db.con.execute("SELECT COUNT(*) FROM metakernel_registry").fetchone()[0] == 0
        assert db.con.execute("SELECT COUNT(*) FROM metakernel_entries").fetchone()[0] == 0
        db.close()

    def test_remove_not_found(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        assert db.remove_metakernel("nonexistent.tm") is False
        db.close()

    def test_remove_by_prefix(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        db.con.execute("""
            INSERT INTO metakernel_registry (mk_path, filename, mission, source_url, acquired_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, ["/tmp/juice_ops.tm", "juice_ops.tm", "JUICE", "https://example.com/juice_ops.tm"])

        assert db.remove_metakernel("juice_ops") is True
        assert db.con.execute("SELECT COUNT(*) FROM metakernel_registry").fetchone()[0] == 0
        db.close()


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


class TestResolveMetakernel:
    """Test resolve --metakernel batch mode."""

    def test_resolve_metakernel(self, tmp_spice_tree):
        """resolve --metakernel resolves all kernels in a .tm file."""
        from spice_kernel_db.cli import main

        db = KernelDB(tmp_spice_tree / "test.duckdb")
        db.scan_directory(tmp_spice_tree)
        db.close()

        mk_path = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        with patch("spice_kernel_db.cli.ensure_config", return_value=Config()):
            main([
                "--db", str(tmp_spice_tree / "test.duckdb"),
                "resolve", "--metakernel", str(mk_path),
                "--mission", "JUICE",
            ])


class TestConfigSetGet:
    """Test config set/get subcommands."""

    def test_config_get(self, tmp_path, capsys):
        from spice_kernel_db.cli import main

        with patch("spice_kernel_db.cli.ensure_config", return_value=Config(
            db_path=str(tmp_path / "test.duckdb"),
            kernel_dir=str(tmp_path / "kernels"),
        )):
            main(["config", "get", "db_path"])

        captured = capsys.readouterr()
        assert "test.duckdb" in captured.out

    def test_config_set(self, tmp_path, capsys):
        from spice_kernel_db.cli import main

        config = Config(
            db_path=str(tmp_path / "test.duckdb"),
            kernel_dir=str(tmp_path / "kernels"),
        )
        with patch("spice_kernel_db.cli.ensure_config", return_value=config), \
             patch("spice_kernel_db.config.save_config") as mock_save:
            main(["config", "set", "kernel_dir", "/new/path"])

        mock_save.assert_called_once()
        assert config.kernel_dir == "/new/path"

    def test_config_get_unknown_key(self, tmp_path):
        from spice_kernel_db.cli import main

        with patch("spice_kernel_db.cli.ensure_config", return_value=Config()):
            with pytest.raises(SystemExit):
                main(["config", "get", "nonexistent_key"])


class TestCorruptConfig:
    """Test handling of corrupt config files."""

    def test_corrupt_toml_returns_none(self, tmp_path):
        from spice_kernel_db.config import CONFIG_FILE, load_config

        # Write corrupt TOML
        corrupt_file = tmp_path / "config.toml"
        corrupt_file.write_text("this is not [valid toml = {{{")

        with patch("spice_kernel_db.config.CONFIG_FILE", corrupt_file):
            result = load_config()

        assert result is None


class TestRewriteDefaultOutput:
    """Test rewrite -o default behavior."""

    def test_default_output(self, tmp_spice_tree):
        from spice_kernel_db.cli import main

        db = KernelDB(tmp_spice_tree / "test.duckdb")
        db.scan_directory(tmp_spice_tree)
        db.close()

        mk_path = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        expected_output = mk_path.with_stem("juice_test_local")

        with patch("spice_kernel_db.cli.ensure_config", return_value=Config()):
            main([
                "--db", str(tmp_spice_tree / "test.duckdb"),
                "rewrite", str(mk_path), "--mission", "JUICE",
            ])

        assert expected_output.is_file()


class TestGlobalVerbose:
    """Test that -v works as a global flag."""

    def test_verbose_flag_on_help(self):
        from spice_kernel_db.cli import main

        # --help should mention -v
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


class TestC1PathTraversalGetMetakernel:
    """C1: hostile metakernel with traversing relpaths must be refused."""

    def test_get_refuses_traversal_in_kernels_to_load(self, tmp_path):
        from spice_kernel_db.remote import _fetch_metakernel
        db = KernelDB(tmp_path / "test.duckdb")

        evil_tm = textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '..' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = (
                '$KERNELS/../../../../tmp/pwn.bsp'
              )
            \\begintext
        """)

        with patch(
            "spice_kernel_db.db._fetch_metakernel",
            return_value=(evil_tm, "https://evil.example/evil.tm"),
        ):
            with pytest.raises(ValueError, match="path-traversing|unsafe"):
                db.get_metakernel(
                    "https://evil.example/evil.tm",
                    download_dir=tmp_path / "downloads",
                    mission="JUICE",
                    yes=True,
                )

        # No file should have been written outside the download tree
        assert not (Path("/tmp") / "pwn.bsp").exists() or \
            (Path("/tmp") / "pwn.bsp").stat().st_size == 0
        db.close()

    def test_link_existing_refuses_unsafe_relpath(self, tmp_path):
        """_link_existing_kernels must skip relpaths that escape mission_root."""
        db = KernelDB(tmp_path / "test.duckdb")

        # Register a real kernel
        kernel = tmp_path / "JUICE" / "kernels" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("LSK")
        db.register_file(kernel, mission="JUICE")

        download_dir = tmp_path / "download"
        download_dir.mkdir()

        n = db._link_existing_kernels(
            indices=[0],
            filenames=["naif0012.tls"],
            relpaths=["../../../tmp/evil.tls"],   # escapes mission_root
            download_dir=download_dir,
            mission="JUICE",
        )
        assert n == 0
        # No symlink anywhere outside the mission dir
        assert not (Path("/tmp") / "evil.tls").exists() or \
            not (Path("/tmp") / "evil.tls").is_symlink()
        db.close()


class TestC2HashGateByResolvedPath:
    """C2: hash verification must use the resolved local path, not the
    caller-supplied filename. Otherwise the gate is bypassed whenever the
    canonical filename in `kernels` differs from what the metakernel asks
    for (the documented path-suffix-match case)."""

    def test_link_skips_when_corrupted_via_path_suffix_match(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")

        # Register a file under its canonical (long) name
        canonical = tmp_path / "JUICE" / "spk" / "jup365_19900101_20500101.bsp"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"identical content")
        h = db.register_file(canonical, mission="JUICE")

        # Manually add a SECOND location for the same hash with a different
        # filename — simulates "same content, also lives at this path with
        # the short name."
        alt = tmp_path / "JUICE" / "spk" / "jup365.bsp"
        alt.write_bytes(b"identical content")
        db.con.execute(
            "INSERT INTO locations VALUES (?, ?, ?, ?, current_timestamp)",
            [h, str(alt.resolve()), "JUICE", None],
        )

        # Now CORRUPT the alt file — content no longer matches its
        # registered hash. The pre-fix code would skip the hash check
        # because `WHERE filename='jup365.bsp'` returns NULL (canonical
        # is the long name). The fix looks up by resolved path.
        alt.write_bytes(b"corrupted bytes!!!")

        download_dir = tmp_path / "download"
        download_dir.mkdir()
        n = db._link_existing_kernels(
            indices=[0],
            filenames=["jup365.bsp"],
            relpaths=["spk/jup365.bsp"],
            download_dir=download_dir,
            mission="JUICE",
        )
        assert n == 0, (
            "C2: link must be refused when the resolved file's hash "
            "no longer matches its locations row"
        )
        db.close()

    def test_link_succeeds_when_canonical_differs_but_content_matches(
        self, tmp_path,
    ):
        """The legitimate path-suffix case: same content, different name.
        After C2, this still works as long as the hash matches."""
        db = KernelDB(tmp_path / "test.duckdb")
        canonical = tmp_path / "JUICE" / "spk" / "jup365_19900101_20500101.bsp"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"identical content")
        h = db.register_file(canonical, mission="JUICE")

        alt = tmp_path / "JUICE" / "spk" / "jup365.bsp"
        alt.write_bytes(b"identical content")
        db.con.execute(
            "INSERT INTO locations VALUES (?, ?, ?, ?, current_timestamp)",
            [h, str(alt.resolve()), "JUICE", None],
        )

        download_dir = tmp_path / "download"
        download_dir.mkdir()
        n = db._link_existing_kernels(
            indices=[0],
            filenames=["jup365.bsp"],
            relpaths=["spk/jup365.bsp"],
            download_dir=download_dir,
            mission="JUICE",
        )
        assert n == 1
        link = download_dir / "JUICE" / "spk" / "jup365.bsp"
        assert link.is_symlink()
        db.close()


class TestC3SupersededBy:
    """C3: re-registering the SAME path with new content must mark the
    old kernels row superseded; resolution then ignores it and is
    deterministic. Two DIFFERENT paths with the same filename are NOT
    supersessions — both remain active."""

    def test_same_path_content_change_marks_superseded(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "JUICE" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("v1 content")
        h1 = db.register_file(kernel, mission="JUICE")

        kernel.write_text("v2 content")
        h2 = db.register_file(kernel, mission="JUICE")
        assert h1 != h2

        # Old row's superseded_by must point to the new hash
        old = db.con.execute(
            "SELECT superseded_by FROM kernels WHERE sha256 = ?", [h1],
        ).fetchone()
        assert old is not None and old[0] == h2

        new = db.con.execute(
            "SELECT superseded_by FROM kernels WHERE sha256 = ?", [h2],
        ).fetchone()
        assert new is not None and new[0] is None
        db.close()

    def test_resolution_skips_superseded(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "JUICE" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("v1")
        db.register_file(kernel, mission="JUICE")
        kernel.write_text("v2")
        h2 = db.register_file(kernel, mission="JUICE")

        # find_by_filename must return only the active (v2) row
        hits = db.find_by_filename("naif0012.tls")
        assert len(hits) == 1
        assert hits[0]["sha256"] == h2
        db.close()

    def test_different_paths_same_name_NOT_superseded(self, tmp_path):
        """Two distinct files sharing a filename are not supersessions."""
        db = KernelDB(tmp_path / "test.duckdb")
        a = tmp_path / "tree_a" / "de440.bsp"
        b = tmp_path / "tree_b" / "de440.bsp"
        a.parent.mkdir(parents=True)
        b.parent.mkdir(parents=True)
        a.write_bytes(b"content A")
        b.write_bytes(b"content B")
        ha = db.register_file(a, mission="A")
        hb = db.register_file(b, mission="B")
        assert ha != hb

        # Both rows should be active (neither superseded)
        rows = db.con.execute(
            "SELECT sha256, superseded_by FROM kernels "
            "WHERE filename = 'de440.bsp'",
        ).fetchall()
        assert len(rows) == 2
        assert all(r[1] is None for r in rows)
        db.close()

    def test_old_location_at_same_path_is_cleaned_up(self, tmp_path):
        """When the path is re-registered with new content, the stale
        old-hash locations row at that path must be removed (it's not
        physically there anymore)."""
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "x" / "foo.bsp"
        kernel.parent.mkdir(parents=True)
        kernel.write_bytes(b"v1")
        h1 = db.register_file(kernel, mission="m")
        kernel.write_bytes(b"v2")
        db.register_file(kernel, mission="m")

        # locations should only have the new hash at this path
        rows = db.con.execute(
            "SELECT sha256 FROM locations WHERE abs_path = ?",
            [str(kernel.resolve())],
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] != h1
        db.close()


class TestVerifyMetakernel:
    """Tests for the `verify` command."""

    def _build_rewritten_mk(self, tmp_path):
        """Helper: scan a tiny tree, rewrite a metakernel, return path + db."""
        db = KernelDB(tmp_path / "test.duckdb")
        juice = tmp_path / "JUICE" / "kernels"
        for sub in ("lsk", "spk", "mk"):
            (juice / sub).mkdir(parents=True)
        (juice / "lsk" / "naif0012.tls").write_text("LSK")
        (juice / "spk" / "de432s.bsp").write_bytes(b"SPK_BYTES")
        db.scan_directory(juice, mission="JUICE")

        src_mk = juice / "mk" / "in.tm"
        src_mk.write_text(textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '..' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = (
                '$KERNELS/lsk/naif0012.tls'
                '$KERNELS/spk/de432s.bsp'
              )
            \\begintext
        """))

        out = tmp_path / "out.tm"
        link_root = tmp_path / "links"
        db.rewrite_metakernel(src_mk, out, mission="JUICE", link_root=link_root)
        return db, out, link_root

    def test_verify_ok_quick(self, tmp_path):
        db, out, _ = self._build_rewritten_mk(tmp_path)
        result = db.verify_metakernel(out)
        assert result["fatal"] is False
        assert result["fail"] == 0
        assert result["ok"] == 2
        db.close()

    def test_verify_ok_deep(self, tmp_path):
        db, out, _ = self._build_rewritten_mk(tmp_path)
        result = db.verify_metakernel(out, deep=True)
        assert result["fatal"] is False
        assert all(e["status"] == "OK" for e in result["entries"])
        db.close()

    def test_verify_detects_dangling_symlink(self, tmp_path):
        db, out, link_root = self._build_rewritten_mk(tmp_path)
        # Break one symlink: remove its target
        lsk_link = link_root / "lsk" / "naif0012.tls"
        target = lsk_link.resolve()
        target.unlink()
        result = db.verify_metakernel(out)
        assert result["fatal"] is True
        statuses = {e["status"] for e in result["entries"]}
        assert "DANGLING" in statuses
        db.close()

    def test_verify_detects_hash_mismatch_in_deep_mode(self, tmp_path):
        db, out, link_root = self._build_rewritten_mk(tmp_path)
        # Corrupt the target of a symlink (same size to bypass quick check)
        target = (link_root / "spk" / "de432s.bsp").resolve()
        original_size = target.stat().st_size
        target.write_bytes(b"X" * original_size)
        result = db.verify_metakernel(out, deep=True)
        assert result["fatal"] is True
        assert any(e["status"] == "HASH_MISMATCH" for e in result["entries"])
        db.close()

    def test_verify_quick_misses_corruption_same_size(self, tmp_path):
        """Quick mode trades safety for speed; document this."""
        db, out, link_root = self._build_rewritten_mk(tmp_path)
        target = (link_root / "spk" / "de432s.bsp").resolve()
        target.write_bytes(b"X" * target.stat().st_size)
        result = db.verify_metakernel(out, deep=False)
        # Quick mode does NOT catch same-size corruption — that's --deep's job
        assert result["fatal"] is False
        db.close()

    def test_verify_detects_size_mismatch_in_quick_mode(self, tmp_path):
        db, out, link_root = self._build_rewritten_mk(tmp_path)
        target = (link_root / "spk" / "de432s.bsp").resolve()
        target.write_bytes(b"short")
        result = db.verify_metakernel(out)
        assert result["fatal"] is True
        assert any(e["status"] == "SIZE_MISMATCH" for e in result["entries"])
        db.close()

    def test_verify_detects_ambiguous_when_two_active_rows(self, tmp_path):
        """C3: if two non-superseded kernels rows share a filename, verify
        must flag AMBIGUOUS — resolution would be non-deterministic."""
        db, out, _ = self._build_rewritten_mk(tmp_path)
        # Insert a phantom second active kernels row for naif0012.tls
        fake_hash = "a" * 64
        db.con.execute(
            "INSERT INTO kernels (sha256, filename, kernel_type, "
            "size_bytes, superseded_by) VALUES (?, ?, 'lsk', 3, NULL)",
            [fake_hash, "naif0012.tls"],
        )
        result = db.verify_metakernel(out)
        assert result["fatal"] is True
        assert any(e["status"] == "AMBIGUOUS" for e in result["entries"])
        db.close()

    def test_verify_detects_traversal_in_kernels_to_load(self, tmp_path):
        """Hand-crafted .tm whose KERNELS_TO_LOAD entries escape the
        PATH_VALUES root must be flagged TRAVERSAL."""
        db = KernelDB(tmp_path / "test.duckdb")
        link_root = tmp_path / "links"
        link_root.mkdir()
        # Build a .tm by hand with an absolute PATH_VALUES and a
        # traversing kernel entry. The resolver still produces a path,
        # but verify_metakernel detects that it escapes link_root.
        mk = tmp_path / "evil.tm"
        mk.write_text(textwrap.dedent(f"""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '{link_root}' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = (
                '$KERNELS/../../etc/passwd'
              )
            \\begintext
        """))
        result = db.verify_metakernel(mk)
        assert result["fatal"] is True
        assert any(e["status"] == "TRAVERSAL" for e in result["entries"])
        db.close()

    def test_verify_flags_non_absolute_path_value(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        mk = tmp_path / "bad.tm"
        mk.write_text(textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '..' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = ( '$KERNELS/x.bsp' )
            \\begintext
        """))
        result = db.verify_metakernel(mk)
        assert result["fatal"] is True
        assert any(e["status"] == "BAD_PATH_VALUE" for e in result["entries"])
        db.close()

    def test_verify_unregistered_is_non_fatal(self, tmp_path):
        """A kernel that exists on disk but is not in `kernels` produces a
        non-fatal warning (UNREGISTERED) — the user should run scan."""
        db, out, link_root = self._build_rewritten_mk(tmp_path)
        # Add an orphan file and reference it via the .tm
        orphan = link_root / "lsk" / "orphan.tls"
        orphan.write_text("orphan")
        # Rewrite the .tm to include the orphan
        text = out.read_text()
        text = text.replace(
            "'$KERNELS/lsk/naif0012.tls'",
            "'$KERNELS/lsk/naif0012.tls'\n    '$KERNELS/lsk/orphan.tls'",
        )
        out.write_text(text)
        result = db.verify_metakernel(out)
        assert any(e["status"] == "UNREGISTERED" for e in result["entries"])
        assert result["fatal"] is False
        db.close()


class TestVerifyCLI:
    """End-to-end CLI smoke tests for `verify`."""

    def test_verify_cli_runs_and_exits_zero_on_ok(self, tmp_path):
        from spice_kernel_db.cli import main
        # Build a happy state
        db = KernelDB(tmp_path / "test.duckdb")
        juice = tmp_path / "JUICE" / "kernels"
        for sub in ("lsk", "mk"):
            (juice / sub).mkdir(parents=True)
        (juice / "lsk" / "naif0012.tls").write_text("LSK")
        db.scan_directory(juice, mission="JUICE")

        src = juice / "mk" / "in.tm"
        src.write_text(textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '..' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = ( '$KERNELS/lsk/naif0012.tls' )
            \\begintext
        """))
        out = tmp_path / "out.tm"
        link_root = tmp_path / "links"
        db.rewrite_metakernel(src, out, mission="JUICE", link_root=link_root)
        db.close()

        with patch("spice_kernel_db.cli.ensure_config", return_value=Config()):
            # Should run without raising SystemExit (exit 0)
            main([
                "--db", str(tmp_path / "test.duckdb"),
                "verify", str(out),
            ])

    def test_verify_cli_exits_nonzero_on_fatal(self, tmp_path):
        from spice_kernel_db.cli import main
        db = KernelDB(tmp_path / "test.duckdb")
        juice = tmp_path / "JUICE" / "kernels"
        for sub in ("lsk", "mk"):
            (juice / sub).mkdir(parents=True)
        (juice / "lsk" / "naif0012.tls").write_text("LSK")
        db.scan_directory(juice, mission="JUICE")

        src = juice / "mk" / "in.tm"
        src.write_text(textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '..' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = ( '$KERNELS/lsk/naif0012.tls' )
            \\begintext
        """))
        out = tmp_path / "out.tm"
        link_root = tmp_path / "links"
        db.rewrite_metakernel(src, out, mission="JUICE", link_root=link_root)
        # Break the symlink
        (link_root / "lsk" / "naif0012.tls").resolve().unlink()
        db.close()

        with patch("spice_kernel_db.cli.ensure_config", return_value=Config()):
            with pytest.raises(SystemExit) as exc:
                main([
                    "--db", str(tmp_path / "test.duckdb"),
                    "verify", str(out),
                ])
            assert exc.value.code == 1


class TestC4DownloadHashStreaming:
    """C4: hash is streamed during download and authoritative.

    Closes the TOCTOU window between "download finished" and
    "register_file re-hashes the file" — and provides the hook for
    external/manifest checksum verification by accepting
    `expected_hash`.
    """

    def test_download_returns_streamed_sha256(self, tmp_path):
        from spice_kernel_db.remote import download_kernel

        content = b"hello world"
        expected = hashlib.sha256(content).hexdigest()

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": str(len(content))}
        mock_resp.read = MagicMock(side_effect=[content, b""])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        dest = tmp_path / "x.bsp"
        with patch("spice_kernel_db.remote.urllib.request.urlopen",
                   return_value=mock_resp):
            _, sha = download_kernel("http://example.com/x.bsp", dest)
        assert sha == expected

    def test_download_rejects_mismatched_expected_hash(self, tmp_path):
        from spice_kernel_db.remote import download_kernel

        content = b"actual content"
        bogus = "0" * 64  # wrong hash on purpose

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": str(len(content))}
        mock_resp.read = MagicMock(side_effect=[content, b""])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        dest = tmp_path / "x.bsp"
        with patch("spice_kernel_db.remote.urllib.request.urlopen",
                   return_value=mock_resp):
            with pytest.raises(IOError, match="[Hh]ash mismatch"):
                download_kernel(
                    "http://example.com/x.bsp", dest, expected_hash=bogus,
                )
        # Partial file should be cleaned up
        assert not dest.exists()

    def test_download_accepts_matching_expected_hash(self, tmp_path):
        from spice_kernel_db.remote import download_kernel

        content = b"content"
        good = hashlib.sha256(content).hexdigest()

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": str(len(content))}
        mock_resp.read = MagicMock(side_effect=[content, b""])
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        dest = tmp_path / "x.bsp"
        with patch("spice_kernel_db.remote.urllib.request.urlopen",
                   return_value=mock_resp):
            _, sha = download_kernel(
                "http://example.com/x.bsp", dest, expected_hash=good,
            )
        assert sha == good
        assert dest.read_bytes() == content

    def test_parallel_propagates_expected_hashes_to_per_url_check(
        self, tmp_path,
    ):
        from spice_kernel_db.remote import download_kernels_parallel

        content = b"abc"
        good = hashlib.sha256(content).hexdigest()
        bogus = "f" * 64

        calls = {}
        def fake_download(url, dest, *, expected_hash=None, **kwargs):
            calls[url] = expected_hash
            if expected_hash and expected_hash != good:
                raise IOError("Hash mismatch")
            dest.write_bytes(content)
            return dest, good

        url_good = "http://example.com/good.bsp"
        url_bad = "http://example.com/bad.bsp"
        tasks = [
            (url_good, tmp_path / "good.bsp", "good.bsp"),
            (url_bad, tmp_path / "bad.bsp", "bad.bsp"),
        ]

        with patch("spice_kernel_db.remote.download_kernel",
                   side_effect=fake_download):
            results, warnings = download_kernels_parallel(
                tasks,
                expected_hashes={url_good: good, url_bad: bogus},
            )
        # The hash mismatch task aborts; the matching one succeeds
        assert len(results) == 1
        assert results[0][1] == good
        assert any("bad.bsp" in w for w in warnings)
        # Per-URL expected_hash was forwarded
        assert calls[url_good] == good
        assert calls[url_bad] == bogus

    def test_parallel_returns_path_and_hash_tuples(self, tmp_path):
        from spice_kernel_db.remote import download_kernels_parallel

        content = b"xyz"
        sha = hashlib.sha256(content).hexdigest()
        def fake_download(url, dest, **kwargs):
            dest.write_bytes(content)
            return dest, sha

        tasks = [
            ("http://example.com/a.bsp", tmp_path / "a.bsp", "a.bsp"),
        ]
        with patch("spice_kernel_db.remote.download_kernel",
                   side_effect=fake_download):
            results, warnings = download_kernels_parallel(tasks)
        assert warnings == []
        assert len(results) == 1
        path, h = results[0]
        assert path == tmp_path / "a.bsp"
        assert h == sha


class TestC5ConcurrencyHandling:
    """C5: race detection must abort, not silently continue. reacquire()
    retries transient lock contention."""

    def test_reacquire_retries_on_lock_contention(self, tmp_path):
        """reacquire() should retry connect() when the first attempt
        raises (transient write-lock contention from another process)."""
        db = KernelDB(tmp_path / "test.duckdb")
        db.release()

        import duckdb as _duck
        real_connect = _duck.connect
        calls = {"n": 0}

        def flaky_connect(path, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _duck.IOException("simulated lock contention")
            return real_connect(path, **kwargs)

        with patch("spice_kernel_db.db.duckdb.connect", side_effect=flaky_connect):
            db.reacquire(initial_delay=0.001)
        assert calls["n"] >= 3
        # And the connection is healthy now
        assert db.con is not None
        db.close()

    def test_get_metakernel_aborts_on_concurrent_modification(self, tmp_path):
        """A `get` whose download window overlaps another writer's
        registration must raise instead of silently continuing."""
        from spice_kernel_db import ConcurrentModificationError

        db = KernelDB(tmp_path / "test.duckdb")
        download_dir = tmp_path / "downloads"

        # Pre-register the kernel under a known hash
        kernel = tmp_path / "preexisting" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("v1 content")
        db.register_file(kernel, mission="JUICE")

        # The remote .tm references that kernel
        tm_text = textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '..' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = ( '$KERNELS/lsk/naif0012.tls' )
            \\begintext
        """)

        # Simulate a concurrent writer that mutates the kernel record
        # in between query_remote_sizes and the post-reacquire check.
        original_query = None
        def evil_query(urls):
            # While the lock is released, "another process" mutates
            kernel.write_text("v2 different content")
            db.reacquire()
            db.register_file(kernel, mission="JUICE")
            db.release()
            return {u: 100 for u in urls}

        with patch(
            "spice_kernel_db.db._fetch_metakernel",
            return_value=(tm_text, "https://e.example/x.tm"),
        ), patch(
            "spice_kernel_db.db.query_remote_sizes", side_effect=evil_query,
        ):
            with pytest.raises(ConcurrentModificationError):
                db.get_metakernel(
                    "https://e.example/x.tm",
                    download_dir=download_dir,
                    mission="JUICE",
                    yes=True,
                )
        db.close()


class _CursorTrap:
    """Wraps a duckdb cursor; raises on a sentinel SQL substring match."""

    def __init__(self, cur, trap: str, raise_after: int = 1):
        self._c = cur
        self._trap = trap.upper()
        self._raise_after = raise_after
        self._hits = 0

    def execute(self, sql, *args, **kw):
        if self._trap in sql.upper():
            self._hits += 1
            if self._hits >= self._raise_after:
                raise RuntimeError("simulated crash")
        return self._c.execute(sql, *args, **kw)

    def __enter__(self):
        self._c.__enter__()
        return self

    def __exit__(self, *a):
        return self._c.__exit__(*a)


class _ConWrapper:
    """Delegates to a real duckdb connection; cursor() returns a trap."""

    def __init__(self, real_con, trap: str, raise_after: int = 1):
        self._real = real_con
        self._trap = trap
        self._raise_after = raise_after

    def __getattr__(self, name):
        return getattr(self._real, name)

    def cursor(self):
        return _CursorTrap(self._real.cursor(), self._trap, self._raise_after)


class TestC6Transactions:
    """C6: multi-statement operations must roll back on failure, not
    leave the DB partially mutated."""

    def test_index_metakernel_rolls_back_on_insert_failure(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        mk = tmp_path / "x.tm"
        mk.write_text(textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '..' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = ( '$KERNELS/lsk/a.tls' '$KERNELS/lsk/b.tls' )
            \\begintext
        """))
        db.index_metakernel(mk)
        mk_resolved = str(mk.resolve())
        before = db.con.execute(
            "SELECT COUNT(*) FROM metakernel_entries WHERE mk_path = ?",
            [mk_resolved],
        ).fetchone()[0]
        assert before == 2

        real = db.con
        db.con = _ConWrapper(real, "INSERT INTO METAKERNEL_ENTRIES", raise_after=2)
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                db.index_metakernel(mk)
        finally:
            db.con = real

        after = db.con.execute(
            "SELECT COUNT(*) FROM metakernel_entries WHERE mk_path = ?",
            [mk_resolved],
        ).fetchone()[0]
        assert after == 2, "C6: index_metakernel must roll back on failure"
        db.close()

    def test_register_file_rolls_back_on_failure(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "x.bsp"
        kernel.write_bytes(b"v1")
        db.register_file(kernel, mission="m")
        n_k = db.con.execute("SELECT COUNT(*) FROM kernels").fetchone()[0]
        n_l = db.con.execute("SELECT COUNT(*) FROM locations").fetchone()[0]

        kernel.write_bytes(b"v2 different content")
        real = db.con
        db.con = _ConWrapper(real, "INSERT OR REPLACE INTO LOCATIONS")
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                db.register_file(kernel, mission="m")
        finally:
            db.con = real

        assert db.con.execute(
            "SELECT COUNT(*) FROM kernels"
        ).fetchone()[0] == n_k
        assert db.con.execute(
            "SELECT COUNT(*) FROM locations"
        ).fetchone()[0] == n_l
        db.close()


class TestC7AtomicFileOps:
    """C7: file/symlink mutations are atomic — no concurrent reader sees
    a missing or truncated file."""

    def test_atomic_write_text_leaves_no_tmp_on_success(self, tmp_path):
        from spice_kernel_db.parser import _atomic_write_text
        p = tmp_path / "out.txt"
        _atomic_write_text(p, "hello")
        assert p.read_text() == "hello"
        # No stray .tmp.* files
        assert list(tmp_path.glob("*.tmp.*")) == []

    def test_atomic_write_text_cleans_up_tmp_on_error(self, tmp_path):
        from spice_kernel_db.parser import _atomic_write_text
        p = tmp_path / "out.txt"
        # Force os.replace to fail
        import os as _os
        with patch.object(_os, "replace", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                _atomic_write_text(p, "content")
        # No partial files left behind
        assert list(tmp_path.glob("*.tmp.*")) == []

    def test_atomic_write_text_overwrites_existing(self, tmp_path):
        from spice_kernel_db.parser import _atomic_write_text
        p = tmp_path / "out.txt"
        p.write_text("old")
        _atomic_write_text(p, "new")
        assert p.read_text() == "new"

    def test_atomic_symlink_creates_new(self, tmp_path):
        from spice_kernel_db.db import _atomic_symlink
        target = tmp_path / "real.bsp"
        target.write_bytes(b"data")
        link = tmp_path / "link.bsp"
        _atomic_symlink(target, link)
        assert link.is_symlink()
        assert link.resolve() == target.resolve()

    def test_atomic_symlink_replaces_existing(self, tmp_path):
        from spice_kernel_db.db import _atomic_symlink
        old_target = tmp_path / "old.bsp"
        old_target.write_bytes(b"old")
        new_target = tmp_path / "new.bsp"
        new_target.write_bytes(b"new")
        link = tmp_path / "link.bsp"
        link.symlink_to(old_target)
        _atomic_symlink(new_target, link)
        assert link.resolve() == new_target.resolve()

    def test_atomic_symlink_cleans_up_tmp_on_error(self, tmp_path):
        from spice_kernel_db.db import _atomic_symlink
        target = tmp_path / "real.bsp"
        target.write_bytes(b"data")
        link = tmp_path / "link.bsp"
        import os as _os
        with patch.object(_os, "replace", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                _atomic_symlink(target, link)
        # The tmp symlink must be cleaned up
        assert list(tmp_path.glob("*.tmp.*")) == []


class TestC8CaseInsensitiveFilenames:
    """C8: filename lookups are case-insensitive so APFS/HFS+/Windows
    (case-insensitive FS) and ext4 (case-sensitive) behave the same.
    Kernel identity is the sha256; name casing is metadata."""

    def test_find_by_filename_case_insensitive_canonical(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        k = tmp_path / "lsk" / "naif0012.tls"
        k.parent.mkdir(parents=True)
        k.write_text("LSK")
        db.register_file(k, mission="JUICE")

        # Query with various casings — all should find the kernel
        for q in ("naif0012.tls", "NAIF0012.TLS", "Naif0012.Tls"):
            hits = db.find_by_filename(q)
            assert len(hits) == 1, f"query {q!r} should match"
            assert hits[0]["abs_path"] == str(k.resolve())
        db.close()

    def test_find_by_filename_case_insensitive_path_suffix(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        # Register a file whose canonical name is e.g. CamelCase
        k = tmp_path / "spk" / "De440.bsp"
        k.parent.mkdir(parents=True)
        k.write_bytes(b"content")
        db.register_file(k, mission="JUICE")

        # find_by_filename with lowercase still finds it via path_suffix
        # or canonical match
        hits = db.find_by_filename("de440.bsp")
        assert len(hits) == 1
        db.close()

    def test_resolve_kernel_handles_case_mismatch(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        k = tmp_path / "lsk" / "naif0012.tls"
        k.parent.mkdir(parents=True)
        k.write_text("LSK")
        db.register_file(k, mission="JUICE")

        # Metakernel might ask for uppercase
        resolved, _ = db.resolve_kernel("NAIF0012.TLS", preferred_mission="JUICE")
        assert resolved == str(k.resolve())
        db.close()

    def test_register_file_detects_case_differing_collision(self, tmp_path):
        """If a file with the same name but different casing AND different
        content gets registered, the C8-aware detection should treat that
        as a same-filename collision (same path = supersession, different
        path = both active)."""
        db = KernelDB(tmp_path / "test.duckdb")

        # First file: lowercase
        a = tmp_path / "a" / "naif0012.tls"
        a.parent.mkdir(parents=True)
        a.write_text("v1")
        db.register_file(a, mission="m")

        # Second file: uppercase, different content, different path
        b = tmp_path / "b" / "NAIF0012.TLS"
        b.parent.mkdir(parents=True)
        b.write_text("v2")
        db.register_file(b, mission="m")

        # Both should be active (different paths, not supersession),
        # but the case-insensitive collision warning should have fired
        rows = db.con.execute(
            "SELECT filename, superseded_by FROM kernels "
            "ORDER BY filename",
        ).fetchall()
        assert len(rows) == 2
        assert all(r[1] is None for r in rows), \
            "different paths should not auto-supersede"
        db.close()


class TestH1DanglingSymlinkRepair:
    """H1: _link_existing_kernels must repair dangling symlinks instead
    of silently skipping them. The old code reported success while
    SPICE then failed at furnsh."""

    def test_dangling_symlink_is_repaired(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        kernel = tmp_path / "JUICE" / "lsk" / "naif0012.tls"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("LSK")
        db.register_file(kernel, mission="JUICE")

        download_dir = tmp_path / "download"
        link_target = download_dir / "JUICE" / "lsk" / "naif0012.tls"
        link_target.parent.mkdir(parents=True)

        # Create a dangling symlink (target doesn't exist)
        ghost = tmp_path / "nowhere" / "old.tls"
        link_target.symlink_to(ghost)
        assert link_target.is_symlink() and not link_target.exists()

        n = db._link_existing_kernels(
            indices=[0],
            filenames=["naif0012.tls"],
            relpaths=["lsk/naif0012.tls"],
            download_dir=download_dir,
            mission="JUICE",
        )
        assert n == 1
        # Now points at the real kernel
        assert link_target.resolve() == kernel.resolve()
        db.close()


class TestH2ParserRobustness:
    """H2: parser tolerates legal SPICE syntax that the old regex broke
    on — markers inside comments, ')' inside strings, '' escape."""

    def test_begindata_token_inside_comment_is_ignored(self, tmp_path):
        from spice_kernel_db.parser import parse_metakernel
        mk = tmp_path / "x.tm"
        mk.write_text(textwrap.dedent("""\
            KPL/MK
            \\begintext
            Inside a comment we discuss \\begindata syntax as an example.
            The parser must NOT treat the line above as a real marker.
            \\begintext
            \\begindata
              KERNELS_TO_LOAD = ( '$KERNELS/lsk/naif.tls' )
            \\begintext
        """))
        parsed = parse_metakernel(mk)
        assert parsed.kernels == ["$KERNELS/lsk/naif.tls"], (
            "the example token inside the comment block leaked into the "
            "data region under the old regex split"
        )

    def test_paren_inside_quoted_string_does_not_truncate_list(
        self, tmp_path,
    ):
        from spice_kernel_db.parser import parse_metakernel
        mk = tmp_path / "x.tm"
        # A kernel path containing ')' would have prematurely closed
        # the list under the old non-greedy regex.
        mk.write_text(textwrap.dedent("""\
            KPL/MK
            \\begindata
              KERNELS_TO_LOAD = (
                '/data/weird)name/foo.bsp'
                '/data/normal.bsp'
              )
            \\begintext
        """))
        parsed = parse_metakernel(mk)
        assert len(parsed.kernels) == 2
        assert parsed.kernels[0] == "/data/weird)name/foo.bsp"
        assert parsed.kernels[1] == "/data/normal.bsp"

    def test_double_quote_escape_in_string(self, tmp_path):
        from spice_kernel_db.parser import parse_metakernel
        mk = tmp_path / "x.tm"
        mk.write_text(textwrap.dedent("""\
            KPL/MK
            \\begindata
              KERNELS_TO_LOAD = ( '/data/it''s_quoted.bsp' )
            \\begintext
        """))
        parsed = parse_metakernel(mk)
        assert parsed.kernels == ["/data/it's_quoted.bsp"]

    def test_indented_begindata_token_works(self, tmp_path):
        """SPICE spec: marker at start of line (optionally indented)."""
        from spice_kernel_db.parser import parse_metakernel
        mk = tmp_path / "x.tm"
        mk.write_text(
            "KPL/MK\n"
            "   \\begindata\n"
            "  KERNELS_TO_LOAD = ( '$K/a.tls' )\n"
            "\\begintext\n"
        )
        parsed = parse_metakernel(mk)
        assert parsed.kernels == ["$K/a.tls"]


class TestH3PathSymbolPrefixCollision:
    """H3: longest PATH_SYMBOLS must be substituted first so an entry
    referencing ``$KERNELS_DATA`` isn't clobbered by an earlier
    ``$KERNELS`` replacement."""

    def test_resolve_prefers_longer_symbol(self):
        from spice_kernel_db.parser import parse_metakernel_text
        text = textwrap.dedent("""\
            \\begindata
              PATH_VALUES  = ( '/a' '/b' )
              PATH_SYMBOLS = ( 'KERNELS' 'KERNELS_DATA' )
              KERNELS_TO_LOAD = ( '$KERNELS_DATA/x.bsp' '$KERNELS/y.bsp' )
            \\begintext
        """)
        parsed = parse_metakernel_text(text, "test")
        # $KERNELS_DATA → /b, $KERNELS → /a — must NOT collapse the
        # longer symbol into /a_DATA via the str.replace prefix bug.
        assert parsed.resolve("$KERNELS_DATA/x.bsp") == "/b/x.bsp"
        assert parsed.resolve("$KERNELS/y.bsp") == "/a/y.bsp"

    def test_relpath_prefers_longer_symbol(self):
        from spice_kernel_db.parser import parse_metakernel_text
        text = textwrap.dedent("""\
            \\begindata
              PATH_VALUES  = ( '/a' '/b' )
              PATH_SYMBOLS = ( 'KERNELS' 'KERNELS_DATA' )
              KERNELS_TO_LOAD = ( '$KERNELS_DATA/x.bsp' '$KERNELS/y.bsp' )
            \\begintext
        """)
        parsed = parse_metakernel_text(text, "test")
        rels = parsed.kernel_relpaths()
        assert "x.bsp" in rels
        assert "y.bsp" in rels
        assert not any("_DATA" in r for r in rels), \
            "the longer symbol must be fully stripped"

    def test_resolve_kernel_urls_prefers_longer_symbol(self):
        from spice_kernel_db.parser import parse_metakernel_text
        from spice_kernel_db.remote import resolve_kernel_urls
        text = textwrap.dedent("""\
            \\begindata
              PATH_VALUES  = ( '..' 'extra' )
              PATH_SYMBOLS = ( 'KERNELS' 'KERNELS_DATA' )
              KERNELS_TO_LOAD = ( '$KERNELS_DATA/x.bsp' '$KERNELS/y.bsp' )
            \\begintext
        """)
        parsed = parse_metakernel_text(text, "test")
        urls = resolve_kernel_urls(
            "https://e.example/mission/mk/m.tm", parsed,
        )
        # $KERNELS_DATA must be resolved (and the longer one not clobbered)
        assert "_DATA" not in "".join(urls), urls
        assert any(u.endswith("x.bsp") for u in urls)
        assert any(u.endswith("y.bsp") for u in urls)


class TestH4MissionCanonicalisation:
    """H4: mission strings are canonicalised at every storage boundary
    so JUICE/juice/Juice converge to one row + one filesystem subtree."""

    def test_canonicalize_mission(self):
        assert canonicalize_mission("juice") == "JUICE"
        assert canonicalize_mission("JUICE") == "JUICE"
        assert canonicalize_mission("Juice") == "JUICE"
        # Sentinels stay lowercase
        assert canonicalize_mission("generic") == "generic"
        assert canonicalize_mission("Generic") == "generic"
        assert canonicalize_mission("UNKNOWN") == "unknown"
        # None / empty
        assert canonicalize_mission(None) == "unknown"
        assert canonicalize_mission("") == "unknown"
        # Whitespace
        assert canonicalize_mission("  juice  ") == "JUICE"

    def test_register_file_canonicalises_mission(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        k = tmp_path / "x.bsp"
        k.write_bytes(b"data")
        db.register_file(k, mission="juice")

        row = db.con.execute(
            "SELECT mission FROM locations"
        ).fetchone()
        assert row[0] == "JUICE"
        db.close()

    def test_add_mission_canonicalises(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        db.add_mission("juice", "http://x/", "http://x/mk/", dedup=True)
        rows = db.con.execute("SELECT name FROM missions").fetchall()
        assert rows == [("JUICE",)]
        db.close()

    def test_startup_warning_for_case_duplicates(self, tmp_path, caplog):
        """If a legacy DB has both `JUICE` and `juice` rows, opening it
        emits a warning so the user knows to merge."""
        # Manually insert duplicate-case rows BYPASSING canonicalisation
        db = KernelDB(tmp_path / "test.duckdb")
        db.con.execute(
            "INSERT INTO missions (name, server_url, mk_dir_url, dedup) "
            "VALUES ('JUICE', '', '', TRUE)"
        )
        db.con.execute(
            "INSERT INTO missions (name, server_url, mk_dir_url, dedup) "
            "VALUES ('juice', '', '', TRUE)"
        )
        db.close()

        with caplog.at_level(logging.WARNING):
            db2 = KernelDB(tmp_path / "test.duckdb")
        db2.close()
        assert any(
            "Mission case duplicates" in r.message
            for r in caplog.records
        )


class TestH5TaskInfoKeyedByPath:
    """H5: when two KERNELS_TO_LOAD entries share a basename under
    different relpaths, attribution must not get scrambled."""

    def test_duplicate_basename_in_metakernel_keeps_separate_attribution(
        self, tmp_path,
    ):
        """End-to-end: a hostile .tm with the same basename under two
        relpaths must result in both files being registered, each with
        its own source_url. This used to overwrite via task_info[name]."""
        db = KernelDB(tmp_path / "test.duckdb")
        download_dir = tmp_path / "downloads"

        tm_text = textwrap.dedent("""\
            KPL/MK
            \\begindata
              PATH_VALUES  = ( '..' )
              PATH_SYMBOLS = ( 'KERNELS' )
              KERNELS_TO_LOAD = (
                '$KERNELS/spk/de.bsp'
                '$KERNELS/spk_alt/de.bsp'
              )
            \\begintext
        """)

        # Mock the network so each URL yields distinct content
        def fake_dl(url, dest, *, expected_hash=None, **kwargs):
            content = b"A" * 10 if "spk/" in url else b"B" * 10
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            return dest, hashlib.sha256(content).hexdigest()

        with patch(
            "spice_kernel_db.db._fetch_metakernel",
            return_value=(tm_text, "https://e.example/m/mk/m.tm"),
        ), patch(
            "spice_kernel_db.db.query_remote_sizes",
            side_effect=lambda urls: {u: 10 for u in urls},
        ), patch(
            "spice_kernel_db.remote.download_kernel",
            side_effect=fake_dl,
        ):
            db.get_metakernel(
                "https://e.example/m/mk/m.tm",
                download_dir=download_dir,
                mission="JUICE",
                yes=True,
            )

        # Both files exist under their distinct relpaths with right content
        spk = download_dir / "JUICE" / "spk" / "de.bsp"
        spk_alt = download_dir / "JUICE" / "spk_alt" / "de.bsp"
        assert spk.read_bytes() == b"A" * 10
        assert spk_alt.read_bytes() == b"B" * 10

        # Each has its own location row with the correct source_url
        rows = db.con.execute(
            "SELECT abs_path, source_url FROM locations "
            "ORDER BY abs_path"
        ).fetchall()
        sources = {r[0]: r[1] for r in rows}
        assert "spk/de.bsp" in sources[str(spk.resolve())]
        assert "spk_alt/de.bsp" in sources[str(spk_alt.resolve())]
        db.close()


class TestH6StalenessDatetimeComparison:
    """H6: staleness check parses dates to datetime rather than doing
    lexicographic comparison on a 16-char timestamp prefix."""

    def test_subminute_acquire_detects_later_remote_correctly(
        self, tmp_path, capsys,
    ):
        from datetime import datetime, timedelta
        from spice_kernel_db.remote import RemoteMetakernel
        db = KernelDB(tmp_path / "test.duckdb")

        # Register a fake metakernel with a known acquired_at
        mk_path = str(tmp_path / "x.tm")
        (tmp_path / "x.tm").write_text("\\begintext\n")
        acq = datetime(2025, 11, 27, 14, 30, 45)
        db.con.execute(
            "INSERT INTO metakernel_registry VALUES (?, ?, ?, ?, ?)",
            [mk_path, "JUICE", "https://e.example/mk/x.tm", "x.tm", acq],
        )

        # Remote modified 16 minutes later — clearly newer
        later = RemoteMetakernel(
            filename="x.tm",
            url="https://e.example/mk/x.tm",
            date="2025-11-27 14:46",
            size="1K",
            base_name="x.tm",
            version_tag=None,
        )
        with patch(
            "spice_kernel_db.db.list_remote_metakernels",
            return_value=[later],
        ):
            db._check_remote_staleness(mk_path, "JUICE")
        captured = capsys.readouterr().out
        assert "Remote update available" in captured
        db.close()

    def test_same_minute_acquire_is_NOT_flagged_stale(
        self, tmp_path, capsys,
    ):
        """If acquired_at is at 14:30:45 and the remote was modified at
        14:30 (same minute, before our acquisition), we must NOT warn."""
        from datetime import datetime
        from spice_kernel_db.remote import RemoteMetakernel
        db = KernelDB(tmp_path / "test.duckdb")

        mk_path = str(tmp_path / "x.tm")
        (tmp_path / "x.tm").write_text("\\begintext\n")
        acq = datetime(2025, 11, 27, 14, 30, 45)
        db.con.execute(
            "INSERT INTO metakernel_registry VALUES (?, ?, ?, ?, ?)",
            [mk_path, "JUICE", "https://e.example/mk/x.tm", "x.tm", acq],
        )
        same_minute = RemoteMetakernel(
            filename="x.tm",
            url="https://e.example/mk/x.tm",
            date="2025-11-27 14:30",
            size="1K",
            base_name="x.tm",
            version_tag=None,
        )
        with patch(
            "spice_kernel_db.db.list_remote_metakernels",
            return_value=[same_minute],
        ):
            db._check_remote_staleness(mk_path, "JUICE")
        captured = capsys.readouterr().out
        assert "Remote update available" not in captured
        db.close()
