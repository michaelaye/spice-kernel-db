"""Network layer: listings, URL resolution, probes, downloads.

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

from tests.helpers import _FakeResp


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


class TestBrowseRemoteMetakernels:
    SAMPLE_HTML = TestListRemoteMetakernels.SAMPLE_HTML

    def _mock_urlopen(self, url, *args, **kwargs):
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


class TestProbeMkCandidates:
    def _fake_urlopen(self, ok_urls):
        def _open(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else req
            if url in ok_urls:
                return MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return _open

    def test_preserves_priority_order(self):
        from spice_kernel_db.remote import probe_mk_candidates
        urls = [
            "https://x/A/mk/",
            "https://x/B/mk/",
            "https://x/C/mk/",
        ]
        ok = {"https://x/A/mk/", "https://x/C/mk/"}
        with patch(
            "spice_kernel_db.remote.urllib.request.urlopen",
            side_effect=self._fake_urlopen(ok),
        ):
            result = probe_mk_candidates(urls)
        # Order = input order, only hits returned
        assert result == ["https://x/A/mk/", "https://x/C/mk/"]

    def test_dedups_input(self):
        from spice_kernel_db.remote import probe_mk_candidates
        urls = ["https://x/A/mk/", "https://x/A/mk/", "https://x/B/mk/"]
        with patch(
            "spice_kernel_db.remote.urllib.request.urlopen",
            side_effect=self._fake_urlopen({"https://x/A/mk/"}),
        ):
            result = probe_mk_candidates(urls)
        assert result == ["https://x/A/mk/"]

    def test_empty_input(self):
        from spice_kernel_db.remote import probe_mk_candidates
        assert probe_mk_candidates([]) == []

    def test_all_fail(self):
        from spice_kernel_db.remote import probe_mk_candidates
        with patch(
            "spice_kernel_db.remote.urllib.request.urlopen",
            side_effect=urllib.error.URLError("nope"),
        ):
            assert probe_mk_candidates(["https://x/Z/mk/"]) == []


class TestDiscoverMkUrl:
    def test_registry_before_defaults(self):
        from spice_kernel_db.remote import discover_mk_url, DEFAULT_ALT_MK_PATHS

        reg_first = "https://x/M/data/spice/mk/"
        default_hit = "https://x/M/kernels/mk/"

        def fake(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else req
            if url in (reg_first, default_hit):
                return MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        with patch(
            "spice_kernel_db.remote.urllib.request.urlopen", side_effect=fake,
        ):
            result = discover_mk_url(
                "https://x/", "M", registry_candidates=[reg_first],
            )
        # Registry candidate must appear before the default kernels/mk hit
        assert result.index(reg_first) < result.index(default_hit)
        # Default-path template is probed
        assert "{server}{m}/kernels/mk/".format(server="https://x/", m="M") in result

    def test_default_paths_contains_only_standard_mk(self):
        """An empirical NAIF survey found no alt mk/ paths in use; the default
        list intentionally contains only the standard kernels/mk/ template.
        Alt paths should be added via the curated registry or --mk-dir-url."""
        from spice_kernel_db.remote import DEFAULT_ALT_MK_PATHS
        assert DEFAULT_ALT_MK_PATHS == ("{server}{m}/kernels/mk/",)

    def test_no_hits_returns_empty(self):
        from spice_kernel_db.remote import discover_mk_url
        with patch(
            "spice_kernel_db.remote.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError("u", 404, "x", {}, None),
        ):
            assert discover_mk_url("https://x/", "Z") == []


class TestDownloadWriteSafety:
    """C9: download_kernel must never write *through* a dedup symlink."""

    def test_download_does_not_corrupt_symlink_target(self, tmp_path):
        """Downloading to a path that is a dedup symlink replaces the link
        with a fresh real file and leaves the shared target untouched.

        Mirrors the BepiColombo bug where `hga_scm_*` symlinks pointed at
        `hga_zero…` and a download wrote the wrong kernel into the shared
        file (silent corruption, caught only by register_file's hash check).
        """
        from spice_kernel_db.remote import download_kernel

        ck = tmp_path / "ck"
        ck.mkdir()
        canonical = ck / "canonical.bc"
        canonical.write_bytes(b"CANONICAL CONTENT A")

        # `dest` is a dedup symlink -> canonical (the dangerous setup).
        dest = ck / "alias.bc"
        dest.symlink_to(canonical)

        new_bytes = b"DIFFERENT DOWNLOADED CONTENT B - longer"
        with patch(
            "spice_kernel_db.remote.urllib.request.urlopen",
            return_value=_FakeResp(new_bytes),
        ):
            out, sha = download_kernel("http://example/alias.bc", dest)

        assert out == dest
        assert sha == hashlib.sha256(new_bytes).hexdigest()
        # The shared target must be untouched.
        assert canonical.read_bytes() == b"CANONICAL CONTENT A"
        # `dest` is now its OWN real file (not a symlink) with the new bytes.
        assert not dest.is_symlink()
        assert dest.read_bytes() == new_bytes

    def test_failed_download_leaves_existing_dest_untouched(self, tmp_path):
        """A download that fails its completeness check must not clobber a
        pre-existing file at `dest`, and must leave no temp files behind."""
        from spice_kernel_db.remote import download_kernel

        ck = tmp_path / "ck"
        ck.mkdir()
        dest = ck / "kernel.bc"
        dest.write_bytes(b"PREEXISTING GOOD BYTES")

        # Claim 999 bytes via Content-Length but deliver fewer -> IOError.
        resp = _FakeResp(b"short")
        resp.headers = {"Content-Length": "999"}
        with patch(
            "spice_kernel_db.remote.urllib.request.urlopen",
            return_value=resp,
        ):
            with pytest.raises(IOError):
                download_kernel("http://example/kernel.bc", dest)

        # Existing file survives, and no .tmp litter remains.
        assert dest.read_bytes() == b"PREEXISTING GOOD BYTES"
        assert list(ck.glob(".*tmp")) == []
        assert list(ck.glob("*.tmp")) == []


class TestProbeUrlClassification:
    """probe_url must tell "the server said no" from "the server said nothing".

    Issue #10: _head_ok returned False for both, so `get` printed "Metakernel
    not found" for a metakernel that existed on a slow server.
    """

    def _raise(self, exc):
        def _f(req, timeout=None):
            raise exc
        return _f

    def test_http_404_is_absent(self):
        from spice_kernel_db.remote import ProbeResult, probe_url
        err = urllib.error.HTTPError("http://x/y.tm", 404, "Not Found", {}, None)
        with patch("spice_kernel_db.remote.urllib.request.urlopen",
                   side_effect=self._raise(err)):
            p = probe_url("http://x/y.tm")
        assert p.result is ProbeResult.ABSENT
        assert "404" in p.detail
        assert not p.unreachable

    def test_socket_timeout_is_unreachable(self):
        from spice_kernel_db.remote import ProbeResult, probe_url
        with patch("spice_kernel_db.remote.urllib.request.urlopen",
                   side_effect=self._raise(TimeoutError())):
            p = probe_url("http://x/y.tm", timeout=7.0)
        assert p.result is ProbeResult.UNREACHABLE
        assert "7 s" in p.detail

    def test_urlerror_wrapping_timeout_is_unreachable(self):
        from spice_kernel_db.remote import ProbeResult, probe_url
        err = urllib.error.URLError(TimeoutError("timed out"))
        with patch("spice_kernel_db.remote.urllib.request.urlopen",
                   side_effect=self._raise(err)):
            p = probe_url("http://x/y.tm")
        assert p.result is ProbeResult.UNREACHABLE

    def test_connection_refused_is_unreachable(self):
        from spice_kernel_db.remote import ProbeResult, probe_url
        err = urllib.error.URLError(ConnectionRefusedError("refused"))
        with patch("spice_kernel_db.remote.urllib.request.urlopen",
                   side_effect=self._raise(err)):
            p = probe_url("http://x/y.tm")
        assert p.result is ProbeResult.UNREACHABLE

    def test_head_ok_still_false_for_both(self):
        """The bulk helper keeps its yes/no contract for existing callers."""
        from spice_kernel_db.remote import _head_ok
        for exc in (urllib.error.HTTPError("u", 404, "NF", {}, None),
                    TimeoutError()):
            with patch("spice_kernel_db.remote.urllib.request.urlopen",
                       side_effect=self._raise(exc)):
                assert _head_ok("http://x/y.tm") is False


class TestEveryRequestHasATimeout:
    """Issue #12: four urlopen calls had no timeout and could hang forever."""

    def test_listing_fetch_and_download_pass_a_timeout(self):
        from spice_kernel_db import remote
        seen = []

        class _Resp:
            headers = {"Content-Length": "0"}
            def read(self, *a):
                return b""
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        def _spy(url, *args, **kwargs):
            seen.append(kwargs.get("timeout"))
            return _Resp()

        with patch("spice_kernel_db.remote.urllib.request.urlopen", side_effect=_spy):
            for call in (
                lambda: remote.list_remote_metakernels("http://x/mk/"),
                lambda: remote.list_remote_missions("http://x/"),
                lambda: remote.fetch_metakernel("http://x/a.tm"),
            ):
                try:
                    call()
                except Exception:
                    pass  # parsing an empty body is not what this test checks

        assert seen, "no urlopen calls were made"
        assert all(t is not None for t in seen), (
            f"a request went out with no timeout: {seen}"
        )
        assert all(t == remote.CONNECT_TIMEOUT for t in seen)

    def test_probe_timeout_clears_the_measured_handshake_p90(self):
        """5 s sat below the p90 handshake recorded in remote.py (5.44 s)."""
        from spice_kernel_db.remote import PROBE_TIMEOUT
        assert PROBE_TIMEOUT >= 12.58, "must clear the measured handshake max"
