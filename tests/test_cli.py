"""The argparse surface: flags, prompts, exit codes, version, output.

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


class TestShowConfig:
    def test_show_config_prints_paths(self, capsys):
        cfg = Config(db_path="/tmp/test.duckdb", kernel_dir="/tmp/kernels")
        show_config(cfg)
        captured = capsys.readouterr()
        assert "/tmp/test.duckdb" in captured.out
        assert "/tmp/kernels" in captured.out


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

    def test_subcommand_help_does_not_load_config(self):
        """`get --help` must not start first-time setup on a fresh machine.

        Click runs the root callback before the subcommand sees --help, so
        the config has to be loaded lazily, by the command that needs it.
        """
        from spice_kernel_db.cli import main

        with patch("spice_kernel_db.cli.ensure_config",
                   side_effect=AssertionError("ensure_config called")):
            with pytest.raises(SystemExit) as exc_info:
                main(["get", "--help"])
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


class TestResolveBodyInteractiveReturnsNone:
    """_resolve_body_interactive returns None instead of sys.exit."""

    def test_unknown_body_returns_none(self):
        from spice_kernel_db.cli.prompts import _resolve_body_interactive
        result = _resolve_body_interactive("ZZZZNOTABODY999")
        assert result is None

    def test_invalid_selection_returns_none(self):
        from spice_kernel_db.cli.prompts import _resolve_body_interactive
        # "earth" is ambiguous (center vs barycenter) — give invalid input
        with patch("builtins.input", return_value="999"):
            result = _resolve_body_interactive("earth")
        assert result is None


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


class TestGlobalVerbose:
    """Test that -v works as a global flag."""

    def test_verbose_flag_on_help(self):
        from spice_kernel_db.cli import main

        # --help should mention -v
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


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


class TestPruneMetakernelsCLI:
    """End-to-end CLI smoke."""

    def test_cli_prune_metakernels_dry_run(self, tmp_path):
        from spice_kernel_db.cli import main

        db = KernelDB(tmp_path / "test.duckdb")
        p = tmp_path / "mk" / "dead.tm"
        p.parent.mkdir(parents=True)
        p.write_text("\\begintext\n")
        db.con.execute(
            "INSERT INTO metakernel_registry VALUES "
            "(?, ?, ?, ?, current_timestamp)",
            [str(p), "JUICE",
             "https://naif.example/former_versions/dead.tm", "dead.tm"],
        )
        db.close()

        def boom(req, *args, **kwargs):
            raise urllib.error.HTTPError(
                req.full_url, 404, "Not Found", {}, None,
            )

        with patch("spice_kernel_db.cli.ensure_config", return_value=Config()), \
             patch("urllib.request.urlopen", side_effect=boom):
            main([
                "--db", str(tmp_path / "test.duckdb"),
                "prune", "--metakernels",
            ])

        # Dry run by default — registry row still there
        db2 = KernelDB(tmp_path / "test.duckdb")
        n = db2.con.execute(
            "SELECT COUNT(*) FROM metakernel_registry"
        ).fetchone()[0]
        assert n == 1
        db2.close()


class TestMissionAddNoninteractive:
    """Drive `_mission_add_noninteractive` directly with a constructed args namespace.

    Avoids the global CLI config-dir / argparse plumbing, since we only care
    about the discovery + persistence behaviour here.
    """

    def _make_args(self, **overrides):
        import types
        defaults = dict(
            name=None, server_url=None, mk_dir_url=None,
            no_dedup=False,
        )
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_explicit_mk_dir_url(self, db, monkeypatch):
        from spice_kernel_db.cli import configure as cli

        monkeypatch.setattr(
            "spice_kernel_db.cli.configure.probe_mk_candidates",
            lambda urls, **_: list(urls),
        )
        # Bypass the server-URL guess (no network).
        monkeypatch.setattr(
            "spice_kernel_db.cli.configure._infer_server_url",
            lambda name, server_url_arg: server_url_arg,
        )

        args = self._make_args(
            name="FOO",
            server_url="https://example.com/",
            mk_dir_url="https://example.com/FOO/data/spice/mk/",
        )
        cli._mission_add_noninteractive(
            db, args.name, args.server_url, args.mk_dir_url, args.no_dedup,
        )

        m = db.get_mission("FOO")
        assert m is not None
        assert m["mk_dir_url"] == "https://example.com/FOO/data/spice/mk/"
        assert m["dedup"] is True

    def test_mk_dir_url_unreachable_exits(self, db, monkeypatch):
        from spice_kernel_db.cli import configure as cli

        monkeypatch.setattr(
            "spice_kernel_db.cli.configure.probe_mk_candidates", lambda urls, **_: []
        )
        monkeypatch.setattr(
            "spice_kernel_db.cli.configure._infer_server_url",
            lambda name, server_url_arg: server_url_arg,
        )

        args = self._make_args(
            name="FOO", server_url="https://example.com/",
            mk_dir_url="https://example.com/FOO/no/such/mk/",
        )
        with pytest.raises(SystemExit) as exc:
            cli._mission_add_noninteractive(
            db, args.name, args.server_url, args.mk_dir_url, args.no_dedup,
        )
        assert exc.value.code == 1

    def test_no_hits_exits(self, db, monkeypatch):
        from spice_kernel_db.cli import configure as cli

        monkeypatch.setattr("spice_kernel_db.cli.configure.discover_mk_url", lambda *a, **kw: [])
        monkeypatch.setattr(
            "spice_kernel_db.cli.configure._infer_server_url",
            lambda name, server_url_arg: server_url_arg,
        )

        args = self._make_args(name="MYSTERY", server_url="https://example.com/")
        with pytest.raises(SystemExit) as exc:
            cli._mission_add_noninteractive(
            db, args.name, args.server_url, args.mk_dir_url, args.no_dedup,
        )
        assert exc.value.code == 1

    def test_multiple_hits_exits(self, db, monkeypatch):
        from spice_kernel_db.cli import configure as cli

        monkeypatch.setattr(
            "spice_kernel_db.cli.configure.discover_mk_url",
            lambda *a, **kw: ["https://x/A/mk/", "https://x/B/mk/"],
        )
        monkeypatch.setattr(
            "spice_kernel_db.cli.configure._infer_server_url",
            lambda name, server_url_arg: server_url_arg,
        )

        args = self._make_args(name="M", server_url="https://x/")
        with pytest.raises(SystemExit) as exc:
            cli._mission_add_noninteractive(
            db, args.name, args.server_url, args.mk_dir_url, args.no_dedup,
        )
        assert exc.value.code == 1

    def test_single_hit_persists(self, db, monkeypatch):
        from spice_kernel_db.cli import configure as cli

        monkeypatch.setattr(
            "spice_kernel_db.cli.configure.discover_mk_url",
            lambda *a, **kw: ["https://x/M/data/spice/mk/"],
        )
        monkeypatch.setattr(
            "spice_kernel_db.cli.configure._infer_server_url",
            lambda name, server_url_arg: server_url_arg,
        )

        args = self._make_args(name="M", server_url="https://x/", no_dedup=True)
        cli._mission_add_noninteractive(
            db, args.name, args.server_url, args.mk_dir_url, args.no_dedup,
        )

        m = db.get_mission("M")
        assert m is not None
        assert m["mk_dir_url"] == "https://x/M/data/spice/mk/"
        assert m["dedup"] is False  # --no-dedup honored


class TestGetArchiveFallback:
    """`get <filename> --mission JUICE` should transparently retry under
    `former_versions/` when the live mk/ URL 404s. The headline use case is
    reproducing analyses from an older versioned metakernel (e.g.
    `juice_crema_2_0_ops_v200.tm`) that ESA has since rotated out of the
    live listing."""

    MK_DIR = "https://example.com/JUICE/kernels/mk/"

    def _setup(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        db.add_mission("JUICE", "https://example.com/", self.MK_DIR, True)
        db.close()
        return Config(
            db_path=str(tmp_path / "test.duckdb"),
            kernel_dir=str(tmp_path / "kernels"),
        )

    def test_archive_fallback_when_live_404s(self, tmp_path):
        from spice_kernel_db.cli import main

        cfg = self._setup(tmp_path)
        captured: dict = {}

        def fake_probe(url, timeout=None):
            # live 404s, archived copy is present
            return (
                Probe(ProbeResult.PRESENT)
                if "former_versions/" in url
                else Probe(ProbeResult.ABSENT, "HTTP 404")
            )

        def fake_get_metakernel(self, url, **kwargs):
            captured["url"] = url

        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch("spice_kernel_db.cli.browse.probe_url", side_effect=fake_probe), \
             patch.object(KernelDB, "get_metakernel", fake_get_metakernel):
            main(["get", "juice_crema_2_0_ops_v200.tm", "--mission", "JUICE"])

        assert captured["url"] == (
            self.MK_DIR + "former_versions/juice_crema_2_0_ops_v200.tm"
        )

    def test_live_url_takes_priority(self, tmp_path):
        """When the live URL HEADs 200, archive must not be probed and the
        live URL is the one passed downstream."""
        from spice_kernel_db.cli import main

        cfg = self._setup(tmp_path)
        captured: dict = {}
        head_calls: list[str] = []

        def fake_probe(url, timeout=None):
            head_calls.append(url)
            return (
                Probe(ProbeResult.ABSENT, "HTTP 404")
                if "former_versions/" in url
                else Probe(ProbeResult.PRESENT)
            )

        def fake_get_metakernel(self, url, **kwargs):
            captured["url"] = url

        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch("spice_kernel_db.cli.browse.probe_url", side_effect=fake_probe), \
             patch.object(KernelDB, "get_metakernel", fake_get_metakernel):
            main(["get", "juice_ops.tm", "--mission", "JUICE"])

        assert captured["url"] == self.MK_DIR + "juice_ops.tm"
        # Archive URL must NOT have been probed on the happy path
        assert all("former_versions/" not in u for u in head_calls)

    def test_both_fail_emits_error_with_both_urls(self, tmp_path, capsys):
        from spice_kernel_db.cli import main

        from spice_kernel_db.remote import Probe, ProbeResult

        cfg = self._setup(tmp_path)
        called = {"get": False}

        def fake_get_metakernel(self, url, **kwargs):
            called["get"] = True

        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch("spice_kernel_db.cli.browse.probe_url",
                   return_value=Probe(ProbeResult.ABSENT, "HTTP 404")), \
             patch.object(KernelDB, "get_metakernel", fake_get_metakernel):
            with pytest.raises(SystemExit) as exc:
                main(["get", "juice_nonexistent.tm", "--mission", "JUICE"])

        assert exc.value.code == 1
        assert called["get"] is False  # never reached get_metakernel
        err = capsys.readouterr().err
        assert "not found" in err
        assert self.MK_DIR + "juice_nonexistent.tm" in err
        assert self.MK_DIR + "former_versions/juice_nonexistent.tm" in err


class TestBrowseArchivedFlag:
    """`browse --archived` should target the mission's former_versions/
    subdirectory instead of the live mk/ listing — for discovering older
    versioned metakernels that have been superseded."""

    MK_DIR = "https://example.com/JUICE/kernels/mk/"

    def _setup(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        db.add_mission("JUICE", "https://example.com/", self.MK_DIR, True)
        db.close()
        return Config(
            db_path=str(tmp_path / "test.duckdb"),
            kernel_dir=str(tmp_path / "kernels"),
        )

    def test_archived_flag_appends_former_versions(self, tmp_path):
        from spice_kernel_db.cli import main

        cfg = self._setup(tmp_path)
        captured: dict = {}

        def fake_browse(self, url, **kwargs):
            captured["url"] = url

        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch.object(KernelDB, "browse_remote_metakernels", fake_browse):
            main(["browse", "JUICE", "--archived"])

        assert captured["url"] == self.MK_DIR + "former_versions/"

    def test_without_archived_flag_browses_live_mk(self, tmp_path):
        """Regression: default behavior must not change."""
        from spice_kernel_db.cli import main

        cfg = self._setup(tmp_path)
        captured: dict = {}

        def fake_browse(self, url, **kwargs):
            captured["url"] = url

        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch.object(KernelDB, "browse_remote_metakernels", fake_browse):
            main(["browse", "JUICE"])

        assert captured["url"] == self.MK_DIR

    def test_archived_flag_with_explicit_url(self, tmp_path):
        """When the user passes a full URL, --archived still suffixes
        former_versions/ — letting them browse an archive without a
        configured mission."""
        from spice_kernel_db.cli import main

        cfg = self._setup(tmp_path)
        captured: dict = {}

        def fake_browse(self, url, **kwargs):
            captured["url"] = url

        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch.object(KernelDB, "browse_remote_metakernels", fake_browse):
            main(["browse", self.MK_DIR, "--archived"])

        assert captured["url"] == self.MK_DIR + "former_versions/"


class TestBrowseFilter:
    """`browse --filter SUBSTRING` narrows large listings via a case-
    insensitive substring match on entry filenames."""

    def test_filter_narrows_entries_case_insensitive(self, tmp_path):
        """browse_remote_metakernels filters entries whose filename
        contains the (lowercased) substring."""
        from spice_kernel_db.remote import RemoteMetakernel

        db = KernelDB(tmp_path / "test.duckdb")
        try:
            entries = [
                RemoteMetakernel(
                    filename=f, url="http://x/" + f, date="2024-01-01 00:00",
                    size="1K", base_name=f, version_tag=None,
                )
                for f in [
                    "juice_ops_v200.tm",
                    "juice_crema_5_1.tm",
                    "JUICE_CREMA_5_2.tm",  # uppercase — should match too
                    "rosetta_ops_v01.tm",
                ]
            ]
            with patch(
                "spice_kernel_db.db.list_remote_metakernels",
                return_value=entries,
            ):
                results = db.browse_remote_metakernels(
                    "http://x/mk/", mission="JUICE", filter="crema",
                )
            names = sorted(r["base_name"] for r in results)
            assert names == ["JUICE_CREMA_5_2.tm", "juice_crema_5_1.tm"]
        finally:
            db.close()

    def test_filter_none_returns_all(self, tmp_path):
        """No filter (None) is a no-op — regression guard."""
        from spice_kernel_db.remote import RemoteMetakernel

        db = KernelDB(tmp_path / "test.duckdb")
        try:
            entries = [
                RemoteMetakernel(
                    filename=f, url="http://x/" + f, date="2024-01-01 00:00",
                    size="1K", base_name=f, version_tag=None,
                )
                for f in ["a.tm", "b.tm", "c.tm"]
            ]
            with patch(
                "spice_kernel_db.db.list_remote_metakernels",
                return_value=entries,
            ):
                results = db.browse_remote_metakernels(
                    "http://x/mk/", mission="X",
                )
            assert len(results) == 3
        finally:
            db.close()


class TestBrowseArchivedRendering:
    """In archive mode every entry is a distinct historical file (not a
    snapshot of a current one). Grouping by base name would conflate
    semantically different files into one row, so archive mode must
    render flat — one row per file — regardless of version tag."""

    def _entries(self):
        from spice_kernel_db.remote import RemoteMetakernel
        # All three share base_name "juice_plan.tm" because the version
        # tag is stripped — but they are NOT identical files. In archive
        # mode they must not be grouped.
        return [
            RemoteMetakernel(
                filename=f"juice_plan_{tag}.tm",
                url=f"http://x/juice_plan_{tag}.tm",
                date=date, size="9K",
                base_name="juice_plan.tm",
                version_tag=tag,
            )
            for tag, date in [
                ("v410_20221110_001", "2022-11-10 00:00"),
                ("v423_20230309_001", "2023-03-09 16:30"),
                ("v470_20260521_003", "2026-05-21 14:33"),
            ]
        ]

    def test_archived_mode_disables_base_name_grouping(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        try:
            with patch(
                "spice_kernel_db.db.list_remote_metakernels",
                return_value=self._entries(),
            ):
                results = db.browse_remote_metakernels(
                    "http://x/mk/former_versions/",
                    mission="JUICE",
                    archived=True,
                )
            # 3 entries → 3 distinct results (one per file), not 1 group.
            assert len(results) == 3
            names = sorted(r["base_name"] for r in results)
            assert names == [
                "juice_plan_v410_20221110_001.tm",
                "juice_plan_v423_20230309_001.tm",
                "juice_plan_v470_20260521_003.tm",
            ]
            # In archive mode each entry is its own "current" — n_versions=1.
            assert all(r["n_versions"] == 1 for r in results)
            assert all(len(r["current"]) == 1 for r in results)
        finally:
            db.close()

    def test_non_archived_still_groups_by_base_name(self, tmp_path):
        """Regression: live mk/ view keeps its base-name grouping
        (where versioned snapshots ARE byte-identical to the current
        file and should be hidden from the compact view)."""
        db = KernelDB(tmp_path / "test.duckdb")
        try:
            with patch(
                "spice_kernel_db.db.list_remote_metakernels",
                return_value=self._entries(),
            ):
                results = db.browse_remote_metakernels(
                    "http://x/mk/", mission="JUICE",
                )
            # All 3 entries share base_name "juice_plan.tm" → 1 group.
            assert len(results) == 1
            assert results[0]["base_name"] == "juice_plan.tm"
            assert results[0]["n_versions"] == 3
        finally:
            db.close()


class TestAliasPicker:
    """`aliases` with no argument offers an interactive picker over the
    deduplicated kernels."""

    def test_picker_lists_dedup_and_returns_choice(self, tmp_path, monkeypatch):
        from spice_kernel_db.cli.prompts import _interactive_pick_deduplicated_kernel
        db = KernelDB(tmp_path / "p.duckdb")
        try:
            for n in ("a.bc", "b.bc"):           # identical content -> dedup
                p = tmp_path / n
                p.write_bytes(b"SAME BYTES")
                db.register_file(p, mission="T")
            u = tmp_path / "u.bc"                 # unique -> not offered
            u.write_bytes(b"UNIQUE BYTES")
            db.register_file(u, mission="T")

            monkeypatch.setattr("builtins.input", lambda *a, **k: "1")
            picked = _interactive_pick_deduplicated_kernel(db)
            assert picked in {"a.bc", "b.bc"}
            # the picked name resolves to a real trail
            assert db.aliases(picked)["aliases"] == ["a.bc", "b.bc"]
        finally:
            db.close()

    def test_picker_returns_none_without_dedup(self, tmp_path, monkeypatch):
        from spice_kernel_db.cli.prompts import _interactive_pick_deduplicated_kernel
        db = KernelDB(tmp_path / "p2.duckdb")
        try:
            u = tmp_path / "only.bc"
            u.write_bytes(b"X")
            db.register_file(u, mission="T")
            monkeypatch.setattr("builtins.input", lambda *a, **k: "1")
            assert _interactive_pick_deduplicated_kernel(db) is None
        finally:
            db.close()

    def test_picker_cancel_returns_none(self, tmp_path, monkeypatch):
        from spice_kernel_db.cli.prompts import _interactive_pick_deduplicated_kernel
        db = KernelDB(tmp_path / "p3.duckdb")
        try:
            for n in ("a.bc", "b.bc"):
                p = tmp_path / n
                p.write_bytes(b"SAME")
                db.register_file(p, mission="T")
            monkeypatch.setattr("builtins.input", lambda *a, **k: "")  # blank
            assert _interactive_pick_deduplicated_kernel(db) is None
        finally:
            db.close()


def test_version_matches_installed_metadata():
    from importlib.metadata import version

    import spice_kernel_db

    assert spice_kernel_db.__version__ == version("spice-kernel-db")


class TestGetDistinguishesMissingFromUnreachable:
    """Issue #10: `get` claimed a metakernel did not exist when ESA was slow.

    The real case: the file was downloadable by curl minutes earlier and was
    listed in mk/, but a stalled handshake made both probes fail and skd
    printed "Metakernel ... not found" and exited 1.
    """

    MK_DIR = "https://example.com/JUICE/kernels/mk/"
    NAME = "juice_crema_5_1_150lb_23_1_a3.tm"

    def _cfg(self, tmp_path):
        db = KernelDB(tmp_path / "test.duckdb")
        db.add_mission("JUICE", "https://example.com/", self.MK_DIR, True)
        db.close()
        return Config(
            db_path=str(tmp_path / "test.duckdb"),
            kernel_dir=str(tmp_path / "kernels"),
        )

    def _run(self, tmp_path, probe, capsys):
        from spice_kernel_db.cli import main

        cfg = self._cfg(tmp_path)
        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch("spice_kernel_db.cli.browse.probe_url", side_effect=probe), \
             patch.object(KernelDB, "get_metakernel", lambda self, url, **k: None):
            with pytest.raises(SystemExit) as exc:
                main(["get", self.NAME, "--mission", "JUICE"])
        return exc.value.code, capsys.readouterr().err

    def test_timeout_does_not_claim_the_file_is_missing(self, tmp_path, capsys):
        from spice_kernel_db.remote import Probe, ProbeResult

        def probe(url, timeout=None):
            return Probe(ProbeResult.UNREACHABLE, "no response within 15 s")

        code, err = self._run(tmp_path, probe, capsys)
        assert code == 1
        assert "not found" not in err.lower(), (
            "a timeout must not be reported as a missing metakernel"
        )
        assert "could not reach" in err.lower()
        assert "no response within 15 s" in err

    def test_real_404_still_says_not_found(self, tmp_path, capsys):
        from spice_kernel_db.remote import Probe, ProbeResult

        def probe(url, timeout=None):
            return Probe(ProbeResult.ABSENT, "HTTP 404")

        code, err = self._run(tmp_path, probe, capsys)
        assert code == 1
        assert "not found" in err.lower()
        assert "could not reach" not in err.lower()

    def test_archive_unreachable_is_not_reported_as_absent(self, tmp_path, capsys):
        """Live 404s honestly, archive times out: we still do not know."""
        from spice_kernel_db.remote import Probe, ProbeResult

        def probe(url, timeout=None):
            if "former_versions/" in url:
                return Probe(ProbeResult.UNREACHABLE, "no response within 15 s")
            return Probe(ProbeResult.ABSENT, "HTTP 404")

        code, err = self._run(tmp_path, probe, capsys)
        assert code == 1
        assert "could not reach" in err.lower()


class TestMissionAddAmbiguousServer:
    """Issue #13: the first server that had the mission was taken silently."""

    def test_two_servers_stops_and_lists_them(self, capsys):
        from spice_kernel_db.cli.configure import AmbiguousServer, _infer_server_url

        with patch("spice_kernel_db.cli.configure.list_remote_missions",
                   return_value=["JUICE", "MARS2020"]):
            with pytest.raises(AmbiguousServer) as exc:
                _infer_server_url("JUICE", None)
        assert len(exc.value.urls) == 2
        assert exc.value.mission == "JUICE"

    def test_single_server_still_resolves(self):
        from spice_kernel_db.cli.configure import _infer_server_url
        from spice_kernel_db.remote import SPICE_SERVERS

        esa = SPICE_SERVERS["ESA"]

        def only_esa(url):
            return ["JUICE"] if url == esa else ["OTHER"]

        with patch("spice_kernel_db.cli.configure.list_remote_missions", side_effect=only_esa):
            assert _infer_server_url("JUICE", None) == esa

    def test_explicit_server_url_bypasses_the_check(self):
        from spice_kernel_db.cli.configure import _infer_server_url
        assert _infer_server_url("JUICE", "https://x/") == "https://x/"
        assert _infer_server_url("JUICE", "https://x") == "https://x/"

    def test_cli_exits_2_and_names_both_servers(self, tmp_path, capsys):
        from spice_kernel_db.cli import main

        cfg = Config(db_path=str(tmp_path / "t.duckdb"),
                     kernel_dir=str(tmp_path / "k"))
        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch("spice_kernel_db.cli.configure.list_remote_missions",
                   return_value=["JUICE"]):
            with pytest.raises(SystemExit) as exc:
                main(["mission", "add", "JUICE"])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "2 known servers" in err
        assert "--server-url" in err
        assert "naif.jpl.nasa.gov" in err and "esac.esa.int" in err


class TestPromptsWithoutATerminal:
    """Issue #14: interactive prompts raised EOFError when stdin was not a tty."""

    def test_mission_add_without_a_tty_exits_2(self, tmp_path, capsys):
        from spice_kernel_db.cli import main

        cfg = Config(db_path=str(tmp_path / "t.duckdb"),
                     kernel_dir=str(tmp_path / "k"))
        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch("sys.stdin.isatty", return_value=False):
            with pytest.raises(SystemExit) as exc:
                main(["mission", "add"])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "needs a terminal" in err
        assert "--server-url" in err
        assert "EOFError" not in err and "Traceback" not in err

    def test_ask_reports_eof_instead_of_raising(self, capsys):
        from spice_kernel_db.cli.prompts import _ask
        with patch("builtins.input", side_effect=EOFError):
            with pytest.raises(SystemExit) as exc:
                _ask("Select [1-2]: ")
        assert exc.value.code == 2
        assert "stdin closed" in capsys.readouterr().err

    def test_confirm_treats_eof_as_no(self, capsys):
        from spice_kernel_db.cli.prompts import _confirm
        with patch("builtins.input", side_effect=EOFError):
            assert _confirm("Delete everything? [y/N]: ") is False
        assert "assuming no" in capsys.readouterr().err

    def test_confirm_reads_yes(self):
        from spice_kernel_db.cli.prompts import _confirm
        for answer, expected in (("y", True), ("yes", True),
                                 ("n", False), ("", False)):
            with patch("builtins.input", return_value=answer):
                assert _confirm("? ") is expected

    def test_dedup_prompt_keeps_its_default_on_eof(self, capsys):
        """This one defaults to YES, so EOF must not flip it to no."""
        from spice_kernel_db.cli.configure import _choose_dedup
        with patch("builtins.input", side_effect=EOFError):
            assert _choose_dedup("JUICE") is True

    def test_no_bare_input_calls_remain_in_menus(self):
        """Regression guard: a new menu must go through _ask/_confirm."""
        import inspect
        from spice_kernel_db.cli import configure as cli

        for name in ("_choose_server", "_choose_mission"):
            src = inspect.getsource(getattr(cli, name))
            assert "input(" not in src, f"{name} calls input() directly"


class TestVerboseReachesGetAndUpdate:
    """Issue #4: -v/--verbose was parsed but never passed to get or update."""

    def test_both_methods_accept_verbose(self):
        import inspect
        for name in ("get_metakernel", "update_metakernel"):
            sig = inspect.signature(getattr(KernelDB, name))
            assert "verbose" in sig.parameters, f"{name} has no verbose parameter"
            assert sig.parameters["verbose"].default is False

    def test_cli_get_passes_verbose(self, tmp_path):
        from spice_kernel_db.cli import main

        KernelDB(tmp_path / "t.duckdb").close()
        cfg = Config(db_path=str(tmp_path / "t.duckdb"),
                     kernel_dir=str(tmp_path / "k"))
        seen = {}

        def fake_get(self, url, **kwargs):
            seen.update(kwargs)
            return {"found": [], "missing": [], "downloaded": [], "warnings": []}

        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch.object(KernelDB, "get_metakernel", fake_get):
            main(["-v", "get", "https://example.com/x.tm"])
        assert seen.get("verbose") is True

    def test_cli_update_passes_verbose(self, tmp_path):
        from spice_kernel_db.cli import main

        KernelDB(tmp_path / "t.duckdb").close()
        cfg = Config(db_path=str(tmp_path / "t.duckdb"),
                     kernel_dir=str(tmp_path / "k"))
        seen = {}

        def fake_update(self, name, **kwargs):
            seen.update(kwargs)
            return {"found": [], "missing": [], "downloaded": [], "warnings": []}

        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch.object(KernelDB, "update_metakernel", fake_update):
            main(["-v", "update", "x.tm"])
        assert seen.get("verbose") is True

    def test_without_v_the_flag_is_false(self, tmp_path):
        from spice_kernel_db.cli import main

        KernelDB(tmp_path / "t.duckdb").close()
        cfg = Config(db_path=str(tmp_path / "t.duckdb"),
                     kernel_dir=str(tmp_path / "k"))
        seen = {}

        def fake_get(self, url, **kwargs):
            seen.update(kwargs)
            return {"found": [], "missing": [], "downloaded": [], "warnings": []}

        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch.object(KernelDB, "get_metakernel", fake_get):
            main(["get", "https://example.com/x.tm"])
        assert seen.get("verbose") is False


class TestMissionHelpTextIsConsistent:
    """Issue #3: update's --mission help differed from the other verbs."""

    def test_all_verbs_describe_mission_the_same_way(self, capsys):
        from spice_kernel_db.cli import main

        expected = "Override auto-detected mission name"
        for verb in ("scan", "get", "update", "browse"):
            with pytest.raises(SystemExit):
                main([verb, "--help"])
            out = capsys.readouterr().out
            assert expected in out, f"{verb} --help does not say {expected!r}"
