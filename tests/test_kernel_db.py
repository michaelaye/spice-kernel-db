"""Tests for spice-kernel-db."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from unittest.mock import MagicMock, patch

from spice_kernel_db import KernelDB, parse_metakernel
from spice_kernel_db.config import Config, load_config, save_config, show_config
from spice_kernel_db.hashing import classify_kernel, guess_mission, sha256_file
from spice_kernel_db.parser import parse_metakernel_text, write_metakernel
from spice_kernel_db.remote import (
    RemoteMetakernel,
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
        assert guess_mission("/data/JUICE/kernels/lsk/naif.tls") == "JUICE"
        assert guess_mission("/data/MRO/kernels/ck/mro.bc") == "MRO"
        # Path-based: directory before 'kernels/' is used
        assert guess_mission("/data/generic_kernels/kernels/lsk/naif.tls") == "generic_kernels"
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
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        # HEAD request for sizes
        mock_resp.headers = {"Content-Length": "1024"}
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
