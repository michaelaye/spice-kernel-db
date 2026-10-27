"""Metakernels: parse, get, update, verify, server comparison, aliases.

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

    @pytest.fixture(autouse=True)
    def _sizes_unknown(self, remote_sizes):
        """get_metakernel asks the server for kernel sizes; answer "unknown".

        Until #33 these tests reached the real network here and the offline
        guard's exception was swallowed by the size lookup.
        """
        with remote_sizes({}):
            yield

    def _mock_urlopen(self, req_or_url, *args, **kwargs):
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

    @staticmethod
    def _mock_download_kernels_parallel(
        tasks, max_workers=8, total_bytes=None, expected_hashes=None,
    ):
        """Stand-in for the real downloader. Writes fake content to each
        dest path and returns ``(dest, sha256)`` pairs that match the
        on-disk bytes, so ``register_file``'s expected-hash check passes."""
        results: list[tuple[Path, str]] = []
        for url, dest, fname in tasks:
            dest_path = Path(dest)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            content = f"FAKE KERNEL {fname}".encode()
            dest_path.write_bytes(content)
            results.append((dest_path, hashlib.sha256(content).hexdigest()))
        return results, []

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
        """get_metakernel saves the .tm file to disk on confirmed download."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        dl_dir = tmp_path / "downloads"

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("spice_kernel_db.db.download_kernels_parallel",
                   side_effect=self._mock_download_kernels_parallel):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=True,
            )

        mk_file = dl_dir / "JUICE" / "mk" / "test.tm"
        assert mk_file.is_file()
        assert "KERNELS_TO_LOAD" in mk_file.read_text()

    def test_get_defaults_to_configured_kernel_dir(self, populated_db, tmp_path):
        """Without download_dir, get_metakernel uses config.kernel_dir, as the CLI does."""
        from spice_kernel_db.config import Config

        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        kernel_dir = tmp_path / "configured_kernels"

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("spice_kernel_db.db.download_kernels_parallel",
                   side_effect=self._mock_download_kernels_parallel), \
             patch("spice_kernel_db.config.load_config",
                   return_value=Config(kernel_dir=str(kernel_dir))):
            populated_db.get_metakernel(mk_url, mission="JUICE", yes=True)

        assert (kernel_dir / "JUICE" / "mk" / "test.tm").is_file()

    def test_get_registers_in_metakernel_registry(self, populated_db, tmp_path):
        """get_metakernel registers the .tm in metakernel_registry on confirm."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        dl_dir = tmp_path / "downloads"

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("spice_kernel_db.db.download_kernels_parallel",
                   side_effect=self._mock_download_kernels_parallel):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=True,
            )

        row = populated_db.con.execute(
            "SELECT mission, source_url, filename FROM metakernel_registry WHERE filename = 'test.tm'"
        ).fetchone()
        assert row is not None
        assert row[0] == "JUICE"
        assert row[1] == mk_url
        assert row[2] == "test.tm"

    def test_get_indexes_metakernel_entries(self, populated_db, tmp_path):
        """get_metakernel populates metakernel_entries on confirmed download."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        dl_dir = tmp_path / "downloads"

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("spice_kernel_db.db.download_kernels_parallel",
                   side_effect=self._mock_download_kernels_parallel):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=True,
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
             patch("spice_kernel_db.db.download_kernels_parallel",
                   side_effect=self._mock_download_kernels_parallel):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=True,
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
             patch("spice_kernel_db.db.download_kernels_parallel",
                   side_effect=self._mock_download_kernels_parallel):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=True,
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
             patch("spice_kernel_db.db.download_kernels_parallel",
                   side_effect=self._mock_download_kernels_parallel):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=True,
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

    def test_decline_rolls_back_first_time_get(self, populated_db, tmp_path):
        """When the user declines a first-time download, the partial
        registration must be rolled back: no .tm on disk, no row in
        metakernel_registry, no entries in metakernel_entries. Without
        this, `mk` would list a metakernel whose dependencies were
        never downloaded — exactly the bug this prevents."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        dl_dir = tmp_path / "downloads"

        with patch("spice_kernel_db.remote.urllib.request.urlopen",
                   side_effect=self._mock_urlopen), \
             patch("builtins.input", return_value="n"):
            result = populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=False,
            )

        assert result.get("aborted") is True
        # No .tm on disk.
        mk_dest = dl_dir / "JUICE" / "mk" / "test.tm"
        assert not mk_dest.exists()
        # No registry row.
        row = populated_db.con.execute(
            "SELECT COUNT(*) FROM metakernel_registry WHERE filename = 'test.tm'"
        ).fetchone()[0]
        assert row == 0
        # No metakernel_entries rows.
        n_entries = populated_db.con.execute(
            "SELECT COUNT(*) FROM metakernel_entries WHERE mk_path = ?",
            [str(mk_dest)],
        ).fetchone()[0]
        assert n_entries == 0

    def test_decline_preserves_prior_on_update(self, populated_db, tmp_path):
        """When the user declines while *updating* a metakernel that
        already exists locally, the prior on-disk content is restored
        and the registry row is preserved (the prior valid acquisition
        must not be clobbered)."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        dl_dir = tmp_path / "downloads"

        # Initial successful get
        with patch("spice_kernel_db.remote.urllib.request.urlopen",
                   side_effect=self._mock_urlopen), \
             patch("spice_kernel_db.db.download_kernels_parallel",
                   side_effect=self._mock_download_kernels_parallel):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=True,
            )
        mk_dest = dl_dir / "JUICE" / "mk" / "test.tm"
        prior_content = mk_dest.read_bytes()

        # Now re-get and decline. The prior content + registry row stay.
        with patch("spice_kernel_db.remote.urllib.request.urlopen",
                   side_effect=self._mock_urlopen), \
             patch("builtins.input", return_value="n"):
            populated_db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=False,
            )

        assert mk_dest.exists()
        assert mk_dest.read_bytes() == prior_content
        row = populated_db.con.execute(
            "SELECT mission FROM metakernel_registry WHERE filename = 'test.tm'"
        ).fetchone()
        assert row is not None and row[0] == "JUICE"

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


class TestMetakernelListingInfo:
    REMOTE_MK_TEXT = TestGetMetakernel.REMOTE_MK_TEXT

    @pytest.fixture(autouse=True)
    def _sizes_unknown(self, remote_sizes):
        """get_metakernel asks the server for kernel sizes; answer "unknown".

        Until #33 these tests reached the real network here and the offline
        guard's exception was swallowed by the size lookup.
        """
        with remote_sizes({}):
            yield

    def _mock_urlopen(self, req_or_url, *args, **kwargs):
        return TestGetMetakernel._mock_urlopen(self, req_or_url)

    def _get_test_mk(self, db, tmp_path):
        """Helper: get a test metakernel into the DB."""
        mk_url = "https://naif.jpl.nasa.gov/pub/naif/JUICE/kernels/mk/test.tm"
        dl_dir = tmp_path / "downloads"
        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=self._mock_urlopen), \
             patch("spice_kernel_db.db.download_kernels_parallel",
                   side_effect=TestGetMetakernel._mock_download_kernels_parallel):
            db.get_metakernel(
                mk_url, download_dir=dl_dir, mission="JUICE", yes=True,
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

    def test_list_metakernels_show_false_prints_nothing(self, populated_db, tmp_path, capsys):
        self._get_test_mk(populated_db, tmp_path)
        capsys.readouterr()
        results = populated_db.list_metakernels(show=False)
        assert any(r["filename"] == "test.tm" for r in results)
        assert capsys.readouterr().out == ""

    def test_metakernels_covering_filters_by_body_and_time(
        self, populated_db, tmp_path, capsys,
    ):
        """Only metakernels whose SPKs cover the body at `et` come back, quietly."""
        from spice_kernel_db.coverage import CoverageInterval

        self._get_test_mk(populated_db, tmp_path)
        capsys.readouterr()

        def coverage(path, body_id):
            if Path(path).name == "new_kernel.bsp" and body_id == -121:
                return [CoverageInterval(et_start=100.0, et_end=200.0)]
            return []

        with patch("spice_kernel_db.coverage.spk_coverage", side_effect=coverage):
            inside = populated_db.metakernels_covering(-121, et=150.0)
            outside = populated_db.metakernels_covering(-121, et=250.0)
            any_time = populated_db.metakernels_covering(-121)
            other_body = populated_db.metakernels_covering(-28)

        assert [r["filename"] for r in inside] == ["test.tm"]
        assert inside[0]["intervals"] == [(100.0, 200.0)]
        assert outside == []
        assert [r["filename"] for r in any_time] == ["test.tm"]
        assert other_body == []
        assert capsys.readouterr().out == ""

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
        """info_metakernel shows detailed per-kernel info. After a
        successful `get`, every referenced kernel is in the DB — n_missing
        is 0. (Pre-fix, this test asserted n_missing=1 because the prior
        decline-path bug left the .tm registered without downloads.)"""
        self._get_test_mk(populated_db, tmp_path)
        result = populated_db.info_metakernel("test.tm")
        assert result is not None
        assert result["filename"] == "test.tm"
        assert result["mission"] == "JUICE"
        assert result["n_kernels"] == 2
        assert result["n_in_db"] == 2
        assert result["n_missing"] == 0

        # Check per-kernel details
        kernel_names = [k["filename"] for k in result["kernels"]]
        assert "naif0012.tls" in kernel_names
        assert "new_kernel.bsp" in kernel_names

        captured = capsys.readouterr()
        assert "in db" in captured.out

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

    def test_cli_mk_remove_opens_db_writable(self, tmp_path):
        """Regression: `mk --remove <name>` must open the DB read-write.
        `mk` (without --remove) is in the read-only command set; the
        destructive variant has to flip that or DuckDB raises
        InvalidInputException on the DELETE."""
        from spice_kernel_db.cli import main

        db_path = tmp_path / "test.duckdb"
        db = KernelDB(db_path)
        db.con.execute("""
            INSERT INTO metakernel_registry
            (mk_path, filename, mission, source_url, acquired_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            "/tmp/stale.tm", "stale.tm", "JUICE",
            "https://example.com/stale.tm",
        ])
        db.close()

        cfg = Config(
            db_path=str(db_path),
            kernel_dir=str(tmp_path / "kernels"),
        )
        with patch("spice_kernel_db.config.load_config", return_value=cfg):
            main(["mk", "--remove", "stale.tm"])

        # Row is gone — and we did not raise.
        db = KernelDB(db_path, read_only=True)
        try:
            n = db.con.execute(
                "SELECT COUNT(*) FROM metakernel_registry "
                "WHERE filename = 'stale.tm'"
            ).fetchone()[0]
        finally:
            db.close()
        assert n == 0


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


class TestUpdateUnreachableMetakernel:
    """`update` on a metakernel whose source URL now returns 404 (NAIF
    rotated to former_versions/) must raise MetakernelUnreachableError
    rather than crashing with a raw HTTPError stack trace."""

    def test_update_raises_unreachable_on_404(self, tmp_path):
        from spice_kernel_db import MetakernelUnreachableError

        db = KernelDB(tmp_path / "test.duckdb")

        # Seed a metakernel_registry row pointing at a (mocked) dead URL
        mk_path = str(tmp_path / "x.tm")
        (tmp_path / "x.tm").write_text("\\begintext\n")
        url = "https://naif.example/JUICE/kernels/mk/old.tm"
        db.con.execute(
            "INSERT INTO metakernel_registry VALUES "
            "(?, ?, ?, ?, current_timestamp)",
            [mk_path, "JUICE", url, "old.tm"],
        )

        def boom(u):
            raise urllib.error.HTTPError(u, 404, "Not Found", {}, None)

        with patch("spice_kernel_db.db._fetch_metakernel", side_effect=boom):
            with pytest.raises(MetakernelUnreachableError) as exc:
                db.update_metakernel("old.tm")
        assert exc.value.status == 404
        assert exc.value.filename == "old.tm"
        assert exc.value.url == url
        db.close()

    def test_update_re_raises_non_4xx_errors(self, tmp_path):
        """Server 5xx (transient) should not be classified as unreachable —
        the user should retry, not prune."""
        db = KernelDB(tmp_path / "test.duckdb")
        mk_path = str(tmp_path / "x.tm")
        (tmp_path / "x.tm").write_text("\\begintext\n")
        url = "https://naif.example/JUICE/kernels/mk/old.tm"
        db.con.execute(
            "INSERT INTO metakernel_registry VALUES "
            "(?, ?, ?, ?, current_timestamp)",
            [mk_path, "JUICE", url, "old.tm"],
        )

        def boom(u):
            raise urllib.error.HTTPError(u, 503, "Service Unavailable", {}, None)

        with patch("spice_kernel_db.db._fetch_metakernel", side_effect=boom):
            with pytest.raises(urllib.error.HTTPError) as exc:
                db.update_metakernel("old.tm")
        assert exc.value.code == 503
        db.close()


class TestPruneMetakernels:
    """`prune --metakernels` finds registry rows whose source_url is
    permanently dead and removes them. Network errors are NOT treated
    as dead — leaving a row in place is always safer than deleting."""

    def _seed(self, tmp_path, urls: list[tuple[str, str]]) -> tuple[KernelDB, list[Path]]:
        """Helper: insert N registry rows with given (filename, url) pairs.
        Returns (db, list-of-on-disk-paths) for assertions."""
        db = KernelDB(tmp_path / "test.duckdb")
        paths: list[Path] = []
        for fname, url in urls:
            p = tmp_path / "mk" / fname
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\\begintext\n")
            paths.append(p)
            db.con.execute(
                "INSERT INTO metakernel_registry VALUES "
                "(?, ?, ?, ?, current_timestamp)",
                [str(p), "JUICE", url, fname],
            )
        return db, paths

    def test_404_is_classified_dead_and_listed_in_dry_run(self, tmp_path):
        db, paths = self._seed(tmp_path, [
            ("alive.tm", "https://naif.example/alive.tm"),
            ("dead.tm",  "https://naif.example/former_versions/dead.tm"),
        ])

        def head_responses(req, *args, **kwargs):
            url = req.full_url
            if "former_versions" in url:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            return MagicMock(__enter__=MagicMock(return_value=MagicMock()),
                             __exit__=MagicMock(return_value=False))

        with patch("urllib.request.urlopen", side_effect=head_responses):
            dead = db.prune_metakernels(dry_run=True)
        assert len(dead) == 1
        assert dead[0]["filename"] == "dead.tm"
        assert dead[0]["status_code"] == 404
        # Dry run — registry untouched
        n = db.con.execute(
            "SELECT COUNT(*) FROM metakernel_registry"
        ).fetchone()[0]
        assert n == 2
        # File untouched
        assert all(p.exists() for p in paths)
        db.close()

    def test_execute_removes_registry_rows_keeps_files(self, tmp_path):
        db, paths = self._seed(tmp_path, [
            ("alive.tm", "https://naif.example/alive.tm"),
            ("dead.tm",  "https://naif.example/former_versions/dead.tm"),
        ])
        # Also seed an entry to confirm cascading delete
        db.con.execute(
            "INSERT INTO metakernel_entries VALUES (?, 0, ?, ?)",
            [str(paths[1]), "$K/lsk/x.tls", "x.tls"],
        )

        def head_responses(req, *args, **kwargs):
            if "former_versions" in req.full_url:
                raise urllib.error.HTTPError(
                    req.full_url, 404, "Not Found", {}, None,
                )
            return MagicMock(__enter__=MagicMock(return_value=MagicMock()),
                             __exit__=MagicMock(return_value=False))

        with patch("urllib.request.urlopen", side_effect=head_responses):
            db.prune_metakernels(dry_run=False, delete_files=False)

        rows = db.con.execute(
            "SELECT filename FROM metakernel_registry"
        ).fetchall()
        assert rows == [("alive.tm",)]
        n_entries = db.con.execute(
            "SELECT COUNT(*) FROM metakernel_entries WHERE mk_path = ?",
            [str(paths[1])],
        ).fetchone()[0]
        assert n_entries == 0
        # Without --delete-files, the on-disk .tm survives
        assert paths[1].exists()
        db.close()

    def test_delete_files_unlinks_on_disk_tm(self, tmp_path):
        db, paths = self._seed(tmp_path, [
            ("dead.tm", "https://naif.example/former_versions/dead.tm"),
        ])
        def head_responses(req, *args, **kwargs):
            raise urllib.error.HTTPError(
                req.full_url, 404, "Not Found", {}, None,
            )

        with patch("urllib.request.urlopen", side_effect=head_responses):
            db.prune_metakernels(dry_run=False, delete_files=True)
        assert not paths[0].exists()
        db.close()

    def test_network_error_does_NOT_remove_row(self, tmp_path):
        """A timeout or DNS failure must not get the row classified as
        dead — we'd rather leave stale state than delete on a transient."""
        db, paths = self._seed(tmp_path, [
            ("maybe-alive.tm", "https://offline.example/x.tm"),
        ])
        def boom(req, *args, **kwargs):
            raise urllib.error.URLError("name resolution failed")
        with patch("urllib.request.urlopen", side_effect=boom):
            dead = db.prune_metakernels(dry_run=False)
        assert dead == []
        n = db.con.execute(
            "SELECT COUNT(*) FROM metakernel_registry"
        ).fetchone()[0]
        assert n == 1
        assert paths[0].exists()
        db.close()

    def test_5xx_is_not_dead(self, tmp_path):
        db, paths = self._seed(tmp_path, [
            ("transient.tm", "https://flaky.example/x.tm"),
        ])
        def boom(req, *args, **kwargs):
            raise urllib.error.HTTPError(
                req.full_url, 503, "Service Unavailable", {}, None,
            )
        with patch("urllib.request.urlopen", side_effect=boom):
            dead = db.prune_metakernels(dry_run=True)
        assert dead == []
        db.close()


class TestPruneMetakernelsNullSourceUrl:
    """v0.13.2 fix: prune --metakernels missed rows whose source_url
    was NULL (the common case for scan-added metakernels). It must use
    mission.mk_dir_url + filename as a fallback, mirroring update."""

    def test_null_source_url_resolved_via_mission_mk_dir_url(self, tmp_path):
        """A registry row with NULL source_url whose mission has a
        configured mk_dir_url must still get HEAD-probed."""
        db = KernelDB(tmp_path / "test.duckdb")
        db.add_mission(
            "JUICE",
            "https://example.org/",
            "https://example.org/JUICE/kernels/mk/",
            dedup=True,
        )

        # Insert a NULL-source row (mimics what scan_directory writes)
        p = tmp_path / "mk" / "v462.tm"
        p.parent.mkdir(parents=True)
        p.write_text("\\begintext\n")
        db.con.execute(
            "INSERT INTO metakernel_registry (mk_path, mission, source_url, "
            "filename, acquired_at) VALUES (?, 'JUICE', NULL, ?, "
            "current_timestamp)",
            [str(p), "v462.tm"],
        )

        seen_urls: list[str] = []
        def head(req, *args, **kwargs):
            seen_urls.append(req.full_url)
            raise urllib.error.HTTPError(
                req.full_url, 404, "Not Found", {}, None,
            )

        with patch("urllib.request.urlopen", side_effect=head):
            dead = db.prune_metakernels(dry_run=True)

        # URL was derived from mission.mk_dir_url + filename
        assert seen_urls == ["https://example.org/JUICE/kernels/mk/v462.tm"]
        assert len(dead) == 1
        assert dead[0]["filename"] == "v462.tm"
        assert dead[0]["status_code"] == 404
        # Reported source_url is the effective URL we probed
        assert "v462.tm" in dead[0]["source_url"]
        db.close()

    def test_null_source_url_with_no_mission_mk_dir_url_is_skipped(
        self, tmp_path, capsys,
    ):
        """Rows with neither source_url nor a mission mk_dir_url cannot
        be probed and should be noted but not classified as dead."""
        db = KernelDB(tmp_path / "test.duckdb")
        # Mission exists but has no mk_dir_url
        db.con.execute(
            "INSERT INTO missions (name, server_url, mk_dir_url, dedup) "
            "VALUES ('FOO', '', '', TRUE)"
        )
        p = tmp_path / "mk" / "x.tm"
        p.parent.mkdir(parents=True)
        p.write_text("\\begintext\n")
        db.con.execute(
            "INSERT INTO metakernel_registry (mk_path, mission, source_url, "
            "filename, acquired_at) VALUES (?, 'FOO', NULL, ?, "
            "current_timestamp)",
            [str(p), "x.tm"],
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            dead = db.prune_metakernels(dry_run=True)
        # Never called — nothing to probe
        assert mock_urlopen.call_count == 0
        assert dead == []
        captured = capsys.readouterr().out
        assert "no probeable URL" in captured
        db.close()


class TestListMetakernelsAliasAware:
    """v0.13.4 fix: alias rows in metakernel_registry pointed at a
    symlink path; their entries were stored under the resolved target
    path. `list_metakernels` JOINed by mk_path and got n_kernels=0,
    and the identical-to detection (which fingerprints the entry list)
    missed them too. Now aliases inherit their target's count +
    fingerprint."""

    def test_symlink_alias_inherits_target_count(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        mk_dir = tmp_path / "mk"
        mk_dir.mkdir()

        # Real metakernel with two entries
        real = mk_dir / "juice_crema_5_2_v470_20260415_001.tm"
        real.write_text(textwrap.dedent("""\
            KPL/MK
            \\begindata
              KERNELS_TO_LOAD = (
                '$KERNELS/lsk/naif0012.tls'
                '$KERNELS/spk/de432.bsp'
              )
            \\begintext
        """))
        db.index_metakernel(real)
        db.con.execute(
            "INSERT INTO metakernel_registry (mk_path, mission, source_url, "
            "filename, acquired_at) VALUES (?, 'JUICE', 'https://e/x.tm', ?, "
            "current_timestamp)",
            [str(real.resolve()), real.name],
        )

        # Alias symlink — no entries written for this path
        alias = mk_dir / "juice_crema_5_2.tm"
        alias.symlink_to(real.name)  # relative symlink
        db.con.execute(
            "INSERT INTO metakernel_registry (mk_path, mission, source_url, "
            "filename, acquired_at) VALUES (?, 'JUICE', 'https://e/y.tm', ?, "
            "current_timestamp)",
            [str(alias), alias.name],
        )

        rows = db.list_metakernels(mission="JUICE")
        by_name = {r["filename"]: r for r in rows}

        # Both rows now report the same count
        assert by_name["juice_crema_5_2.tm"]["n_kernels"] == 2
        assert by_name["juice_crema_5_2_v470_20260415_001.tm"]["n_kernels"] == 2

        # They're detected as identical content (one of them has
        # identical_to pointing at the other; either direction is fine).
        identicals = [
            (n, r.get("identical_to"))
            for n, r in by_name.items()
            if r.get("identical_to")
        ]
        assert len(identicals) == 1, (
            f"exactly one row should be flagged identical; got {identicals!r}"
        )
        db.close()

    def test_non_alias_symlink_unaffected(self, tmp_path):
        """A symlink pointing at something NOT in the registry should not
        crash the alias-detection branch."""
        db = KernelDB(tmp_path / "test.duckdb")
        mk_dir = tmp_path / "mk"
        mk_dir.mkdir()

        real = mk_dir / "real.tm"
        real.write_text("\\begindata\nKERNELS_TO_LOAD=('a.tls')\n\\begintext\n")
        db.index_metakernel(real)
        db.con.execute(
            "INSERT INTO metakernel_registry (mk_path, mission, source_url, "
            "filename, acquired_at) VALUES (?, 'JUICE', NULL, ?, "
            "current_timestamp)",
            [str(real.resolve()), real.name],
        )

        # Symlink to a file that is NOT registered
        external = tmp_path / "external_target.tm"
        external.write_text("ignored")
        alias = mk_dir / "alias.tm"
        alias.symlink_to(external)
        db.con.execute(
            "INSERT INTO metakernel_registry (mk_path, mission, source_url, "
            "filename, acquired_at) VALUES (?, 'JUICE', NULL, ?, "
            "current_timestamp)",
            [str(alias), alias.name],
        )

        rows = db.list_metakernels(mission="JUICE")
        by_name = {r["filename"]: r for r in rows}
        # Alias keeps its own (zero) count — no target in registry to inherit
        assert by_name["alias.tm"]["n_kernels"] == 0
        # Real is unaffected
        assert by_name["real.tm"]["n_kernels"] == 1
        db.close()


class TestAliasTrail:
    """Workstream B: deduplicated kernels remain followable by every name."""

    def _make_db(self, tmp_path):
        return KernelDB(tmp_path / "alias.duckdb")

    def test_identical_content_records_both_names(self, tmp_path):
        """Two files with identical bytes dedup to one kernels row but both
        names are retained in kernel_aliases and surfaced by aliases()."""
        db = self._make_db(tmp_path)
        try:
            content = b"DEDUP ME - identical bytes"
            a = tmp_path / "predict.bc"
            b = tmp_path / "measured.bc"
            a.write_bytes(content)
            b.write_bytes(content)
            db.register_file(a, mission="TEST")
            db.register_file(b, mission="TEST")

            # one physical kernel, two alias names
            n_kernels = db.con.execute(
                "SELECT count(*) FROM kernels"
            ).fetchone()[0]
            assert n_kernels == 1
            info = db.aliases("measured.bc")
            assert info is not None
            assert set(info["aliases"]) == {"predict.bc", "measured.bc"}
            # both names resolve to the same content
            assert db.aliases("predict.bc")["sha256"] == info["sha256"]
        finally:
            db.close()

    def test_aliases_lookup_by_hash_canonical_and_alias(self, tmp_path):
        db = self._make_db(tmp_path)
        try:
            content = b"trail content"
            for name in ("first.bc", "second.bc"):
                p = tmp_path / name
                p.write_bytes(content)
                db.register_file(p, mission="TEST")
            sha = db.con.execute("SELECT sha256 FROM kernels").fetchone()[0]

            by_hash = db.aliases(sha[:12])
            by_canon = db.aliases("first.bc")
            by_alias = db.aliases("second.bc")
            assert by_hash["sha256"] == sha
            assert by_canon["sha256"] == sha
            assert by_alias["sha256"] == sha
            assert db.aliases("nonexistent.bc") is None
        finally:
            db.close()

    def test_view_reflects_locations_live(self, tmp_path):
        """`kernel_aliases` is a view over `locations`: a second on-disk name
        for the same content appears as an alias with no extra bookkeeping."""
        db = self._make_db(tmp_path)
        try:
            p = tmp_path / "k.bc"
            p.write_bytes(b"x")
            db.register_file(p, mission="TEST")
            sha = db.con.execute("SELECT sha256 FROM kernels").fetchone()[0]
            # Same content recorded at a second path with a different name.
            db.con.execute(
                "INSERT OR IGNORE INTO locations VALUES "
                "(?, ?, 'TEST', NULL, current_timestamp)",
                [sha, str(tmp_path / "k_alias.bc")],
            )
            info = db.aliases("k_alias.bc")
            assert info is not None
            assert {"k.bc", "k_alias.bc"} <= set(info["aliases"])
        finally:
            db.close()

    def test_view_replaces_legacy_table_on_open(self, tmp_path):
        """A DB carrying a 0.17.0-style kernel_aliases *table* is converted to
        the view on the next writable open, without data loss."""
        dbpath = tmp_path / "legacy.duckdb"
        db = KernelDB(dbpath)
        try:
            p = tmp_path / "k.bc"
            p.write_bytes(b"x")
            db.register_file(p, mission="TEST")
        finally:
            db.close()
        # Recreate the old table form in its place.
        import duckdb
        con = duckdb.connect(str(dbpath))
        con.execute("DROP VIEW IF EXISTS kernel_aliases")
        con.execute(
            "CREATE TABLE kernel_aliases "
            "(sha256 VARCHAR, filename VARCHAR, PRIMARY KEY (sha256, filename))"
        )
        con.close()
        db2 = KernelDB(dbpath)  # reopen → table dropped, view defined
        try:
            kind = db2.con.execute(
                "SELECT table_type FROM information_schema.tables "
                "WHERE table_name = 'kernel_aliases'"
            ).fetchone()[0]
            assert kind == "VIEW"
            info = db2.aliases("k.bc")
            assert info is not None and "k.bc" in info["aliases"]
        finally:
            db2.close()


class TestAliasAnnotation:
    """alias_counts_by_name powers the (+N aliases) listing annotation."""

    def test_counts_extra_names_per_filename(self, tmp_path):
        db = KernelDB(tmp_path / "ann.duckdb")
        try:
            # two identical-content files -> shared content, +1 alias each
            for n in ("predict.bc", "measured.bc"):
                p = tmp_path / n
                p.write_bytes(b"SAME BYTES")
                db.register_file(p, mission="TEST")
            # a unique file -> no aliases
            u = tmp_path / "unique.bc"
            u.write_bytes(b"DIFFERENT BYTES")
            db.register_file(u, mission="TEST")

            extra = db.alias_counts_by_name(
                ["predict.bc", "measured.bc", "unique.bc", "absent.bc"]
            )
            assert extra == {"predict.bc": 1, "measured.bc": 1}
            # unique and absent names are omitted (no alias / not in DB)
            assert "unique.bc" not in extra
            assert "absent.bc" not in extra
            assert db.alias_counts_by_name([]) == {}
        finally:
            db.close()


class TestCheckUntrackedMetakernel:
    """Issue #11: `check <name>.tm` raised FileNotFoundError from the parser."""

    def _db(self, tmp_path):
        db = KernelDB(tmp_path / "t.duckdb")
        return db

    def test_untracked_name_raises_a_typed_error_not_filenotfound(self, tmp_path):
        from spice_kernel_db import MetakernelNotTracked

        db = self._db(tmp_path)
        try:
            with pytest.raises(MetakernelNotTracked) as exc:
                db.check_metakernel("never_downloaded.tm")
            assert "not tracked locally" in str(exc.value)
            assert "spice-kernel-db get never_downloaded.tm" in str(exc.value)
        finally:
            db.close()

    def test_a_real_path_still_works(self, tmp_path):
        db = self._db(tmp_path)
        mk = tmp_path / "local.tm"
        mk.write_text(
            "\\begindata\n"
            "PATH_VALUES = ( '/data' )\n"
            "PATH_SYMBOLS = ( 'K' )\n"
            "KERNELS_TO_LOAD = ( '$K/lsk/naif0012.tls' )\n"
            "\\begintext\n"
        )
        try:
            result = db.check_metakernel(mk)
            assert "missing" in result
        finally:
            db.close()

    def test_registered_name_resolves_to_its_path(self, tmp_path):
        """A bare name that IS tracked must resolve, as `resolve` does."""
        db = self._db(tmp_path)
        mk = tmp_path / "tracked.tm"
        mk.write_text(
            "\\begindata\n"
            "PATH_VALUES = ( '/data' )\n"
            "PATH_SYMBOLS = ( 'K' )\n"
            "KERNELS_TO_LOAD = ( '$K/lsk/naif0012.tls' )\n"
            "\\begintext\n"
        )
        try:
            db.con.execute(
                "INSERT INTO metakernel_registry "
                "(mk_path, mission, source_url, filename, acquired_at) "
                "VALUES (?, ?, ?, ?, now())",
                [str(mk), "JUICE", "http://x/tracked.tm", "tracked.tm"],
            )
            assert db.resolve_metakernel_path("tracked.tm") == mk
            result = db.check_metakernel("tracked.tm")
            assert "missing" in result
        finally:
            db.close()

    def test_cli_prints_one_line_and_exits_1(self, tmp_path, capsys):
        from spice_kernel_db.cli import main

        KernelDB(tmp_path / "t.duckdb").close()
        cfg = Config(db_path=str(tmp_path / "t.duckdb"),
                     kernel_dir=str(tmp_path / "k"))
        with patch("spice_kernel_db.config.load_config", return_value=cfg):
            with pytest.raises(SystemExit) as exc:
                main(["check", "never_downloaded.tm"])
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert err.count("\n") == 1, f"expected one line, got: {err!r}"
        assert "not tracked locally" in err
        assert "Traceback" not in err and "FileNotFoundError" not in err


class TestVerifyAgainstServer:
    """Issue #15: check only asked whether a file of each name existed.

    On juice_crema_5_2_v473_20260819_001.tm it reported "Found locally:
    102/102" while 10 of those differed from the server, including a 245 MB
    DSK present locally as 11 MB.
    """

    MK_URL = "https://example.com/JUICE/kernels/mk/juice.tm"

    @pytest.fixture(autouse=True)
    def _server_is_quiet(self, remote_sizes):
        """Sizes unknown, and the server lists no newer metakernel.

        Until #33 these tests reached the real network through both the
        size lookup and check's staleness probe; broad handlers swallowed
        the offline guard's exception.
        """
        with remote_sizes({}), \
             patch("spice_kernel_db.db.list_remote_metakernels",
                   return_value=[]):
            yield

    def _setup(self, tmp_path, local_sizes):
        kdir = tmp_path / "kernels" / "lsk"
        kdir.mkdir(parents=True)
        mk = tmp_path / "juice.tm"
        entries = []
        for name, size in local_sizes.items():
            (kdir / name).write_bytes(b"x" * size)
            entries.append(f"'$K/lsk/{name}'")
        mk.write_text(
            "\\begindata\n"
            f"PATH_VALUES = ( '{tmp_path / 'kernels'}' )\n"
            "PATH_SYMBOLS = ( 'K' )\n"
            f"KERNELS_TO_LOAD = ( {' '.join(entries)} )\n"
            "\\begintext\n"
        )
        db = KernelDB(tmp_path / "t.duckdb")
        db.add_mission("JUICE", "https://example.com/",
                       "https://example.com/mk/", True)
        db.scan_directory(str(tmp_path / "kernels"), mission="JUICE")
        db.con.execute(
            "INSERT INTO metakernel_registry "
            "(mk_path, mission, source_url, filename, acquired_at) "
            "VALUES (?, 'JUICE', ?, 'juice.tm', now())",
            [str(mk), self.MK_URL],
        )
        return db, mk

    def _url(self, name):
        return f"https://example.com/JUICE/kernels/lsk/{name}"

    def _remote_mk(self, names):
        """What the server serves: conventional relative PATH_VALUES."""
        entries = " ".join(f"'$K/lsk/{n}'" for n in names)
        return (
            "\\begindata\n"
            "PATH_VALUES = ( '..' )\n"
            "PATH_SYMBOLS = ( 'K' )\n"
            f"KERNELS_TO_LOAD = ( {entries} )\n"
            "\\begintext\n"
        )

    def test_size_mismatch_is_caught(self, tmp_path):
        """The pck00011 shape: right name, wrong bytes."""
        db, mk = self._setup(tmp_path, {"pck00011.tpc": 133802})
        try:
            with patch("spice_kernel_db.remote.fetch_metakernel",
                       return_value=self._remote_mk(["pck00011.tpc"])), \
                 patch("spice_kernel_db.remote.query_remote_sizes",
                       return_value={self._url("pck00011.tpc"): 131226}):
                r = db.verify_against_server(str(mk), mission="JUICE")
        finally:
            db.close()
        assert r["bad"] == 1
        bad = [e for e in r["entries"] if e["status"] == "SIZE_MISMATCH"]
        assert len(bad) == 1
        assert "133,802" in bad[0]["detail"] and "131,226" in bad[0]["detail"]

    def test_matching_size_is_ok(self, tmp_path):
        db, mk = self._setup(tmp_path, {"naif0012.tls": 5000})
        try:
            with patch("spice_kernel_db.remote.fetch_metakernel",
                       return_value=self._remote_mk(["naif0012.tls"])), \
                 patch("spice_kernel_db.remote.query_remote_sizes",
                       return_value={self._url("naif0012.tls"): 5000}):
                r = db.verify_against_server(str(mk), mission="JUICE")
        finally:
            db.close()
        assert r["bad"] == 0 and r["ok"] == 1

    def test_server_without_a_size_is_unchecked_not_ok(self, tmp_path):
        """A server that will not say must not be counted as verified."""
        db, mk = self._setup(tmp_path, {"naif0012.tls": 5000})
        try:
            with patch("spice_kernel_db.remote.fetch_metakernel",
                       return_value=self._remote_mk(["naif0012.tls"])), \
                 patch("spice_kernel_db.remote.query_remote_sizes",
                       return_value={self._url("naif0012.tls"): None}):
                r = db.verify_against_server(str(mk), mission="JUICE")
        finally:
            db.close()
        assert r["ok"] == 0 and r["bad"] == 0 and r["unchecked"] == 1

    def test_no_source_url_reports_rather_than_claiming_ok(self, tmp_path):
        db, mk = self._setup(tmp_path, {"naif0012.tls": 5000})
        try:
            db.con.execute("UPDATE metakernel_registry SET source_url = NULL")
            r = db.verify_against_server(str(mk), mission="JUICE")
        finally:
            db.close()
        assert r["ok"] == 0
        assert r["entries"][0]["status"] == "NO_REMOTE"

    def test_untracked_metakernel_raises_the_typed_error(self, tmp_path):
        from spice_kernel_db import MetakernelNotTracked
        db = KernelDB(tmp_path / "t.duckdb")
        try:
            with pytest.raises(MetakernelNotTracked):
                db.verify_against_server("nope.tm")
        finally:
            db.close()

    def test_cli_full_exits_1_on_a_mismatch(self, tmp_path, capsys):
        from spice_kernel_db.cli import main

        db, mk = self._setup(tmp_path, {"pck00011.tpc": 133802})
        db.close()
        cfg = Config(db_path=str(tmp_path / "t.duckdb"),
                     kernel_dir=str(tmp_path / "kernels"))
        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch("spice_kernel_db.remote.fetch_metakernel",
                   return_value=self._remote_mk(["pck00011.tpc"])), \
             patch("spice_kernel_db.remote.query_remote_sizes",
                   return_value={self._url("pck00011.tpc"): 131226}):
            with pytest.raises(SystemExit) as exc:
                main(["check", str(mk), "--full"])
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "SIZE_MISMATCH" in out and "pck00011.tpc" in out

    def test_plain_check_still_reports_it_as_found(self, tmp_path, capsys):
        """The bug being fixed: without --full the wrong file counts as present."""
        from spice_kernel_db.cli import main

        db, mk = self._setup(tmp_path, {"pck00011.tpc": 133802})
        db.close()
        cfg = Config(db_path=str(tmp_path / "t.duckdb"),
                     kernel_dir=str(tmp_path / "kernels"))
        with patch("spice_kernel_db.config.load_config", return_value=cfg):
            main(["check", str(mk)])
        out = capsys.readouterr().out
        assert "Found locally:  1/1" in out
        assert "SIZE_MISMATCH" not in out

    def test_hash_and_spice_require_full(self, tmp_path, capsys):
        from spice_kernel_db.cli import main

        db, mk = self._setup(tmp_path, {"naif0012.tls": 10})
        db.close()
        cfg = Config(db_path=str(tmp_path / "t.duckdb"),
                     kernel_dir=str(tmp_path / "kernels"))
        with patch("spice_kernel_db.config.load_config", return_value=cfg):
            with pytest.raises(SystemExit) as exc:
                main(["check", str(mk), "--hash"])
        assert exc.value.code == 2
        assert "require --full" in capsys.readouterr().err
