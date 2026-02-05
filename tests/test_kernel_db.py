"""Tests for spice-kernel-db."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from spice_kernel_db import KernelDB, parse_metakernel
from spice_kernel_db.hashing import classify_kernel, guess_mission, sha256_file
from spice_kernel_db.parser import write_metakernel


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
