"""The argparse surface: flags, prompts, exit codes, version, output.

Split out of the former tests/test_kernel_db.py.
"""

from __future__ import annotations
import hashlib
import json
import logging
import shutil
import textwrap
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import spice_kernel_db
from spice_kernel_db import KernelDB, parse_metakernel
from spice_kernel_db.config import Config, load_config, save_config
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
        from spice_kernel_db.cli.configure import _show_config

        cfg = Config(db_path="/tmp/test.duckdb", kernel_dirs=["/tmp/kernels"])
        _show_config(cfg)
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

        with patch("spice_kernel_db.cli.ensure_config_or_setup",
                   side_effect=AssertionError("config loader called")):
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
            kernel_dirs=[str(tmp_path / "kernels")],
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

    def test_invalid_selection_returns_none(self, tty):
        from spice_kernel_db.cli.prompts import _resolve_body_interactive
        # "earth" is ambiguous (center vs barycenter) — give invalid input
        with patch("builtins.input", return_value="999"):
            result = _resolve_body_interactive("earth")
        assert result is None


class TestConfigSetGet:
    """Test config set/get subcommands."""

    def test_config_get(self, tmp_path, capsys):
        from spice_kernel_db.cli import main

        with patch("spice_kernel_db.cli.ensure_config_or_setup", return_value=Config(
            db_path=str(tmp_path / "test.duckdb"),
            kernel_dirs=[str(tmp_path / "kernels")],
        )):
            main(["config", "get", "db_path"])

        captured = capsys.readouterr()
        assert "test.duckdb" in captured.out

    def test_config_set(self, tmp_path, capsys):
        from spice_kernel_db.cli import main

        config = Config(
            db_path=str(tmp_path / "test.duckdb"),
            kernel_dirs=[str(tmp_path / "kernels")],
        )
        with patch("spice_kernel_db.cli.ensure_config_or_setup", return_value=config), \
             patch("spice_kernel_db.config.save_config") as mock_save:
            main(["config", "set", "db_path", "/new/path.duckdb"])

        mock_save.assert_called_once()
        assert config.db_path == "/new/path.duckdb"
        # the roots are a list and have their own verbs; no single-root key (2026-11-21)
        with patch("spice_kernel_db.cli.ensure_config_or_setup", return_value=config), \
             pytest.raises(SystemExit) as exc:
            main(["config", "set", "kernel_dir", "/new/path"])
        assert exc.value.code == 1 and "add-root" in capsys.readouterr().err

    def test_config_get_unknown_key(self, tmp_path):
        from spice_kernel_db.cli import main

        with patch("spice_kernel_db.cli.ensure_config_or_setup", return_value=Config()):
            with pytest.raises(SystemExit):
                main(["config", "get", "nonexistent_key"])


class TestCorruptConfig:
    """Test handling of corrupt config files."""

    def test_corrupt_toml_raises(self, tmp_path):
        """#49: never None, which the next step would read as "not configured" and overwrite."""
        from spice_kernel_db.config import ConfigError, load_config

        corrupt_file = tmp_path / "config.toml"
        corrupt_file.write_text("this is not [valid toml = {{{")

        with patch("spice_kernel_db.config.CONFIG_FILE", corrupt_file), \
             pytest.raises(ConfigError, match="corrupt"):
            load_config()


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

        with patch("spice_kernel_db.cli.ensure_config_or_setup", return_value=Config()):
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

        with patch("spice_kernel_db.cli.ensure_config_or_setup", return_value=Config()):
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
            "INSERT INTO metakernel_registry (mk_path, mission, source_url, filename, acquired_at) VALUES "
            "(?, ?, ?, ?, current_timestamp)",
            [str(p), "JUICE",
             "https://naif.example/former_versions/dead.tm", "dead.tm"],
        )
        db.close()

        def boom(req, *args, **kwargs):
            raise urllib.error.HTTPError(
                req.full_url, 404, "Not Found", {}, None,
            )

        with patch("spice_kernel_db.cli.ensure_config_or_setup", return_value=Config()), \
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
            kernel_dirs=[str(tmp_path / "kernels")],
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
            kernel_dirs=[str(tmp_path / "kernels")],
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

    def test_picker_lists_dedup_and_returns_choice(self, tmp_path, monkeypatch, tty):
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

    def test_picker_cancel_returns_none(self, tmp_path, monkeypatch, tty):
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
            kernel_dirs=[str(tmp_path / "kernels")],
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
                     kernel_dirs=[str(tmp_path / "k")])
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
                     kernel_dirs=[str(tmp_path / "k")])
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
                     kernel_dirs=[str(tmp_path / "k")])
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
                     kernel_dirs=[str(tmp_path / "k")])
        seen = {}

        def fake_update(self, name, **kwargs):
            seen.update(kwargs)
            return {"found": [], "missing": [], "downloaded": [], "warnings": []}

        # #47: a tracked name given to get is re-fetched through update_metakernel
        db = KernelDB(tmp_path / "t.duckdb")
        db.con.execute(
            "INSERT INTO metakernel_registry (mk_path, mission, source_url, filename, acquired_at) VALUES "
            "(?, 'JUICE', 'https://example.com/mk/x.tm', 'x.tm', now())",
            [str(tmp_path / "x.tm")],
        )
        db.close()
        with patch("spice_kernel_db.config.load_config", return_value=cfg), \
             patch.object(KernelDB, "update_metakernel", fake_update):
            main(["-v", "get", "x.tm"])
        assert seen.get("verbose") is True

    def test_without_v_the_flag_is_false(self, tmp_path):
        from spice_kernel_db.cli import main

        KernelDB(tmp_path / "t.duckdb").close()
        cfg = Config(db_path=str(tmp_path / "t.duckdb"),
                     kernel_dirs=[str(tmp_path / "k")])
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
        for verb in ("scan", "get", "import", "browse"):
            with pytest.raises(SystemExit):
                main([verb, "--help"])
            out = capsys.readouterr().out
            assert expected in out, f"{verb} --help does not say {expected!r}"


class TestOneResolver:
    """#45: every command resolves a metakernel reference the same way, so a
    registry filename that `check` accepts works for `list` and `verify` too
    (#41), and a missing argument behaves the same everywhere."""

    SOURCE = "https://example.com/mk/juice_crema_5_2_v473_20260819_001.tm"
    FORMS = ["absolute path", "registry filename", "base name",
             "base name without .tm", "source url"]
    COMMANDS = [("show",), ("check",), ("verify",), ("which", "--metakernel")]

    @pytest.fixture
    def tracked(self, tmp_path):
        """A DB with one tracked JUICE metakernel plus two snapshots of a
        second one, and a ``run`` helper that returns the exit code."""
        from types import SimpleNamespace

        from spice_kernel_db.cli import main

        db_path = tmp_path / "t.duckdb"
        db = KernelDB(db_path)
        kernel = tmp_path / "pck00011.tpc"
        kernel.write_text("REAL PCK CONTENT")
        db.register_file(kernel, mission="JUICE")

        def write_mk(name: str, source_url: str | None = None) -> Path:
            mk = tmp_path / name
            mk.write_text(
                "\\begindata\n"
                f"PATH_VALUES = ( '{tmp_path}' )\n"
                "PATH_SYMBOLS = ( 'K' )\n"
                "KERNELS_TO_LOAD = ( '$K/pck00011.tpc' )\n"
                "\\begintext\n"
            )
            db.con.execute(
                "INSERT INTO metakernel_registry "
                "(mk_path, mission, source_url, filename, acquired_at) "
                "VALUES (?, 'JUICE', ?, ?, now())",
                [str(mk), source_url, name],
            )
            return mk

        mk = write_mk("juice_crema_5_2_v473_20260819_001.tm", self.SOURCE)
        write_mk("juice_crema_5_1_v470_20260101_001.tm")
        write_mk("juice_crema_5_1_v471_20260201_001.tm")
        db.close()

        def run(*argv: str) -> int:
            # check consults the remote mk/ listing when a source URL is
            # known; this test is about resolution, not staleness.
            with patch("spice_kernel_db.cli.ensure_config_or_setup", return_value=Config()), \
                 patch.object(KernelDB, "newer_remote_version", lambda *a, **k: None):
                try:
                    main(["--db", str(db_path), *argv])
                except SystemExit as exc:
                    return exc.code or 0
            return 0

        return SimpleNamespace(path=mk, db_path=db_path, tmp=tmp_path, run=run)

    def ref(self, tracked, form: str) -> str:
        return {
            "absolute path": str(tracked.path),
            "registry filename": "juice_crema_5_2_v473_20260819_001.tm",
            "base name": "juice_crema_5_2.tm",
            "base name without .tm": "juice_crema_5_2",
            "source url": self.SOURCE,
        }[form]

    # C1
    @pytest.mark.parametrize("form", FORMS)
    def test_every_form_resolves_to_the_same_path(self, tracked, form):
        db = KernelDB(tracked.db_path)
        try:
            assert db.resolve_metakernel(self.ref(tracked, form)) == tracked.path
            assert db.resolve_metakernel("JUICE_CREMA_5_2.TM") == tracked.path
            with pytest.raises(spice_kernel_db.MetakernelNotTracked):
                db.resolve_metakernel("juice_crema_5_2.tm", mission="bepi")
        finally:
            db.close()

    # C2
    def test_unknown_and_ambiguous(self, tracked):
        db = KernelDB(tracked.db_path)
        try:
            with pytest.raises(spice_kernel_db.MetakernelNotTracked) as exc:
                db.resolve_metakernel("nope.tm")
            assert "spice-kernel-db get nope.tm" in str(exc.value)
            # A specific version that is not tracked never lands on another one.
            with pytest.raises(spice_kernel_db.MetakernelNotTracked):
                db.resolve_metakernel("juice_crema_5_2_v999_20270101_001.tm")
            with pytest.raises(spice_kernel_db.AmbiguousMetakernel) as exc:
                db.resolve_metakernel("juice_crema_5_1.tm")
            assert sorted(exc.value.candidates) == [
                "juice_crema_5_1_v470_20260101_001.tm",
                "juice_crema_5_1_v471_20260201_001.tm",
            ]
        finally:
            db.close()

    # C3
    @pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
    @pytest.mark.parametrize("form", FORMS)
    def test_commands_accept_every_form(self, tracked, capsys, command, form):
        assert tracked.run(*command, self.ref(tracked, form)) == 0
        err = capsys.readouterr().err
        assert "Traceback" not in err and "not tracked" not in err

    def test_rewrite_accepts_registry_filename(self, tracked, capsys):
        out = tracked.tmp / "local.tm"
        assert tracked.run("rewrite", "juice_crema_5_2.tm", "-o", str(out)) == 0
        assert out.is_file()
        assert "Traceback" not in capsys.readouterr().err

    def test_remove_accepts_base_name(self, tracked):
        assert tracked.run("rm", "juice_crema_5_2.tm") == 0
        assert tracked.run("show", "juice_crema_5_2.tm") == 1
        assert tracked.path.is_file()

    # C4
    @pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
    def test_commands_share_exit_codes(self, tracked, capsys, command):
        assert tracked.run(*command, "nope.tm") == 1
        err = capsys.readouterr().err
        assert "spice-kernel-db get nope.tm" in err and "Traceback" not in err

        assert tracked.run(*command, "juice_crema_5_1.tm") == 2
        err = capsys.readouterr().err
        assert "v470" in err and "v471" in err and "Traceback" not in err

    # C5
    @pytest.mark.parametrize("command", [("show",), ("check",), ("rm",), ("rewrite",)],
                             ids=" ".join)
    def test_missing_argument_without_tty_exits_2(self, tracked, capsys, command):
        assert tracked.run(*command) == 2
        captured = capsys.readouterr()
        assert "juice_crema_5_2" in captured.err and "needs a terminal" in captured.err
        assert "juice_crema_5_2" not in captured.out

    # C6
    def test_missing_argument_at_tty_shows_picker(self, tracked, capsys, tty, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *a, **k: "1")
        assert tracked.run("show") == 0
        out = capsys.readouterr().out
        assert "Local metakernels" in out and "pck00011.tpc" in out


class TestVerbTree:
    """#47: one verb per job in four help panels; every old name is a hidden
    alias that prints its rename note; scan never prompts."""

    NEW_VERBS = {"browse", "get", "update", "ls", "show", "which", "coverage", "stats",
                 "check", "doctor", "scan", "import", "rewrite", "rm", "clean", "dedup",
                 "reset", "mission", "config"}
    OLD_VERBS = {"list", "metakernels", "mk", "resolve", "aliases", "duplicates",
                 "prune", "verify"}

    @pytest.fixture
    def tracked(self, tmp_path):
        """One tracked JUICE metakernel over one registered kernel, a second
        untracked .tm, and a ``run`` helper returning the exit code."""
        from types import SimpleNamespace

        from spice_kernel_db.cli import main

        kdir = tmp_path / "kernels" / "pck"
        kdir.mkdir(parents=True)
        (kdir / "pck00011.tpc").write_text("REAL PCK CONTENT")
        db_path = tmp_path / "t.duckdb"
        db = KernelDB(db_path)
        db.add_mission("JUICE", "https://example.com/", "https://example.com/mk/", True)
        db.scan_directory(str(tmp_path / "kernels"), mission="JUICE")

        def write_mk(name: str) -> Path:
            mk = tmp_path / name
            mk.write_text(
                "\\begindata\n"
                f"PATH_VALUES = ( '{tmp_path / 'kernels'}' )\n"
                "PATH_SYMBOLS = ( 'K' )\n"
                "KERNELS_TO_LOAD = ( '$K/pck/pck00011.tpc' )\n"
                "\\begintext\n"
            )
            return mk

        mk = write_mk("juice_crema_5_2_v473_20260819_001.tm")
        db.index_metakernel(mk)
        db.con.execute(
            "INSERT INTO metakernel_registry "
            "(mk_path, mission, source_url, filename, acquired_at) "
            "VALUES (?, 'JUICE', 'https://example.com/mk/juice_crema_5_2_v473_20260819_001.tm', ?, now())",
            [str(mk), mk.name],
        )
        untracked = write_mk("on_disk.tm")
        db.close()

        def run(*argv: str) -> int:
            with patch("spice_kernel_db.cli.ensure_config_or_setup",
                       return_value=Config(db_path=str(db_path), kernel_dirs=[str(tmp_path / "kernels")])), \
                 patch.object(KernelDB, "newer_remote_version", lambda *a, **k: None):
                try:
                    main(["--db", str(db_path), *argv])
                except SystemExit as exc:
                    return exc.code or 0
            return 0

        return SimpleNamespace(mk=mk, untracked=untracked, db_path=db_path,
                               tmp=tmp_path, kdir=kdir, run=run)

    # C16
    def test_help_lists_the_new_verbs_only(self, tracked, capsys):
        # Click's own command table: Rich draws the help with ASCII boxes
        # on a non-UTF-8 terminal (CI), so the text is not parsed.
        import typer.main

        from spice_kernel_db.cli import app

        commands = typer.main.get_command(app).commands
        assert {n for n, c in commands.items() if not c.hidden} == self.NEW_VERBS
        assert {n for n, c in commands.items() if c.hidden} >= self.OLD_VERBS
        assert tracked.run("--help") == 0
        out = capsys.readouterr().out
        for panel in ("acquire", "inspect", "maintain", "configure"):
            assert panel in out
        assert "browse & acquire" not in out and "transform" not in out

    def test_help_pages_are_consistent(self):
        """Every visible command points onward; metakernel arguments read
        alike; a hidden argument stays out of the usage line."""
        import typer
        import typer.main

        from spice_kernel_db.cli import app

        root = typer.main.get_command(app)
        visible = {n: c for n, c in root.commands.items() if not c.hidden}
        visible.update(
            (f"mission {n}", c) for n, c in visible.pop("mission").commands.items())
        for name, cmd in visible.items():
            assert "See also" in (cmd.epilog or ""), name

        def mk_help(name: str) -> str:
            return next(p.help for p in visible[name].params if p.name == "metakernel")
        assert mk_help("coverage") == mk_help("check") == mk_help("show")

        update = visible["update"]
        # typer.Context, not click: Typer 0.29 vendors Click, and CI has no click.
        with typer.Context(update, info_name="update") as ctx:
            assert update.collect_usage_pieces(ctx) == ["[OPTIONS]"]

    # C17
    @pytest.mark.parametrize("old, new", [
        (("list", "juice_crema_5_2.tm"), ("show", "juice_crema_5_2.tm")),
        (("metakernels",), ("ls",)),
        (("mk", "--mission", "JUICE"), ("ls", "JUICE")),
        (("metakernels", "juice_crema_5_2.tm"), ("show", "juice_crema_5_2.tm")),
        (("resolve", "pck00011.tpc"), ("which", "pck00011.tpc")),
        (("aliases", "pck00011.tpc"), ("which", "pck00011.tpc")),
        (("resolve", "--metakernel", "juice_crema_5_2.tm"),
         ("which", "--metakernel", "juice_crema_5_2.tm")),
        (("duplicates",), ("stats",)),
        (("prune",), ("clean", "--locations")),
        (("prune", "--orphan-symlinks"), ("clean", "--symlinks")),
        (("verify", "juice_crema_5_2.tm"), ("check", "juice_crema_5_2.tm")),
    ], ids=lambda p: " ".join(p))
    def test_old_names_run_with_a_note(self, tracked, capsys, old, new):
        new_rc = tracked.run(*new)
        new_out = capsys.readouterr()
        assert "note:" not in new_out.err
        old_rc = tracked.run(*old)
        old_out = capsys.readouterr()
        assert old_rc == new_rc == 0
        assert f"note: '{old[0]}" in old_out.err and f"is now '{new[0]}" in old_out.err
        assert old_out.out == new_out.out

    def test_old_remove_and_update_forms(self, tracked, capsys):
        seen = {}

        def fake_update(self, name, **kwargs):
            seen["name"] = name
            return {"found": [], "missing": [], "downloaded": [], "warnings": []}

        with patch.object(KernelDB, "update_metakernel", fake_update):
            assert tracked.run("update", "juice_crema_5_2.tm") == 0
        assert seen["name"] == "juice_crema_5_2.tm"
        assert "note: 'update juice_crema_5_2.tm' is now 'get juice_crema_5_2.tm'" in capsys.readouterr().err

        assert tracked.run("mk", "--remove", "juice_crema_5_2.tm") == 0
        assert "note: 'mk --remove' is now 'rm'" in capsys.readouterr().err
        assert tracked.run("show", "juice_crema_5_2.tm") == 1

    # C18
    def test_get_on_a_tracked_name_uses_its_source(self, tracked, capsys):
        seen = {}

        def fake_get(self, url, **kwargs):
            seen["url"] = url
            return {"found": [], "missing": [], "downloaded": [], "warnings": []}

        with patch.object(KernelDB, "get_metakernel", fake_get), \
             patch("spice_kernel_db.cli.browse._locate_remote_metakernel",
                   side_effect=AssertionError("get consulted the mk/ listing for a tracked name")):
            assert tracked.run("get", "juice_crema_5_2") == 0
        assert seen["url"] == "https://example.com/mk/juice_crema_5_2_v473_20260819_001.tm"
        assert "note:" not in capsys.readouterr().err

    def test_get_on_an_imported_metakernel_without_a_source_exits_1(self, tracked, capsys):
        # #47, found while documenting: a LookupError traceback before this.
        db = KernelDB(tracked.db_path)
        db.remove_mission("JUICE")
        db.close()
        assert tracked.run("import", str(tracked.untracked), "--mission", "JUICE") == 0
        assert tracked.run("get", "on_disk.tm") == 1
        err = capsys.readouterr().err
        assert "no source" in err and "Traceback" not in err

    # C19
    def test_update_all_is_dry_run_by_default(self, tracked, capsys, remote_sizes):
        remote_mk = (
            "\\begindata\nPATH_VALUES = ( '..' )\nPATH_SYMBOLS = ( 'K' )\n"
            "KERNELS_TO_LOAD = ( '$K/pck/pck00011.tpc' '$K/pck/new.tpc' )\n\\begintext\n"
        )
        # a second tracked metakernel whose source is gone
        db = KernelDB(tracked.db_path)
        db.con.execute(
            "INSERT INTO metakernel_registry (mk_path, mission, source_url, filename, acquired_at) VALUES "
            "(?, 'JUICE', 'https://example.com/mk/gone.tm', 'gone.tm', now())",
            [str(tracked.tmp / "gone.tm")],
        )
        db.close()

        def fetch(url, *a, **k):
            if url.endswith("gone.tm"):
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            return remote_mk, url

        sizes = {"https://example.com/pck/pck00011.tpc": 16,
                 "https://example.com/pck/new.tpc": 5}
        with patch("spice_kernel_db.db._fetch_metakernel", side_effect=fetch), \
             remote_sizes(sizes), \
             patch("spice_kernel_db.db.download_kernels_parallel",
                   side_effect=AssertionError("update downloaded without --execute")) as dl:
            assert tracked.run("update") == 0
        out = capsys.readouterr().out
        assert "dry run" in out and "gone.tm" in out and "source gone (HTTP 404)" in out
        assert not (tracked.tmp / "kernels" / "JUICE").exists()
        db = KernelDB(tracked.db_path, read_only=True)
        try:
            assert db.con.execute("SELECT count(*) FROM metakernel_registry").fetchone()[0] == 2
        finally:
            db.close()

        with patch("spice_kernel_db.db._fetch_metakernel", side_effect=fetch), \
             remote_sizes(sizes), \
             patch("spice_kernel_db.db.download_kernels_parallel",
                   return_value=([], [])) as dl:
            assert tracked.run("update", "--execute") == 0
        assert dl.call_count == 1
        assert [t[2] for t in dl.call_args.args[0]] == ["new.tpc"]

    # C62, C63, C65
    def test_update_asks_for_sizes_once_a_day(self, tracked, capsys):
        old = "https://example.com/pck/pck00011.tpc"
        new = "https://example.com/pck/new.tpc"
        entries = ["$K/pck/pck00011.tpc"]

        def fetch(url, *a, **k):
            return ("\\begindata\nPATH_VALUES = ( '..' )\nPATH_SYMBOLS = ( 'K' )\n"
                    f"KERNELS_TO_LOAD = ( {' '.join(repr(e) for e in entries)} )\n"
                    "\\begintext\n"), url

        asked: list[list[str]] = []

        def query(urls, **k):
            asked.append(sorted(urls))
            return {u: 16 for u in urls}

        with patch("spice_kernel_db.db._fetch_metakernel", side_effect=fetch), \
             patch("spice_kernel_db.db.query_remote_sizes", side_effect=query), \
             patch("spice_kernel_db.db.download_kernels_parallel", return_value=([], [])):
            assert tracked.run("update") == 0
            assert tracked.run("update") == 0
            assert "1 from today's cache, 0 asked" in capsys.readouterr().out
            entries.append("$K/pck/new.tpc")  # a new version lists one more kernel
            assert tracked.run("update") == 0
            assert tracked.run("update", "--refresh") == 0
            assert tracked.run("get", "juice_crema_5_2", "--yes") == 0
        assert asked == [[old], [new], [new, old], [new, old]]
        cache = json.loads((tracked.tmp / "remote_sizes.json").read_text())
        assert set(cache) == {old, new}

    # C20
    def test_inspect_verbs(self, tracked, capsys):
        assert tracked.run("ls", "juice") == 0
        out = capsys.readouterr().out
        assert "Tracked metakernels" in out and "juice_crema_5_2_v473" in out

        assert tracked.run("show", "juice_crema_5_2") == 0
        out = capsys.readouterr().out
        assert "Source:" in out and "pck00011.tpc" in out and "in db" in out
        assert tracked.run("show", "juice_crema_5_2", "--type", "spk") == 0
        assert "pck00011.tpc" not in capsys.readouterr().out

        assert tracked.run("show", str(tracked.untracked)) == 0
        out = capsys.readouterr().out
        assert "pck00011.tpc" in out and "Not tracked" in out

        alias = tracked.kdir / "pck_alias.tpc"
        alias.write_text("REAL PCK CONTENT")
        db = KernelDB(tracked.db_path)
        db.register_file(alias, mission="JUICE")
        db.close()
        assert tracked.run("which", "pck00011.tpc") == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0] == str(tracked.kdir / "pck00011.tpc")
        assert "pck_alias.tpc" in out and "Known under 2 name(s)" in out
        assert tracked.run("which", "nope.tpc") == 1

        assert tracked.run("stats") == 0
        out = capsys.readouterr().out
        assert "Unique kernels" in out and "Duplicate kernels" in out

    # C21
    def test_scan_never_prompts(self, tracked, capsys, monkeypatch):
        monkeypatch.setattr("builtins.input",
                            lambda *a, **k: pytest.fail("scan read stdin"))
        tree = tracked.tmp / "other" / "MEX" / "kernels" / "lsk"
        tree.mkdir(parents=True)
        (tree / "naif0012.tls").write_text("LSK")
        assert tracked.run("scan", str(tracked.tmp / "other")) == 0
        out = capsys.readouterr().out
        assert "MEX is not configured" in out and "mission add MEX --server-url" in out
        db = KernelDB(tracked.db_path, read_only=True)
        try:
            assert db.get_mission("MEX") is None
        finally:
            db.close()

    # C22
    def test_import(self, tracked, capsys):
        with patch("spice_kernel_db.db._fetch_metakernel",
                   side_effect=AssertionError("import fetched a URL")):
            assert tracked.run("import", str(tracked.untracked), "--mission", "JUICE") == 0
        out = capsys.readouterr().out
        assert "Tracked on_disk.tm (JUICE): 1/1 kernels" in out
        assert tracked.run("check", "on_disk.tm") == 0
        assert tracked.run("ls") == 0
        assert "on_disk.tm" in capsys.readouterr().out

    # C23
    def test_rm_and_clean(self, tracked, capsys):
        gone = tracked.kdir / "gone.tpc"
        gone.write_text("GONE")
        link = tracked.kdir / "dangling.tpc"
        db = KernelDB(tracked.db_path)
        db.register_file(gone, mission="JUICE")
        db.close()
        gone.unlink()
        link.symlink_to(gone)
        # orphan-symlink walk roots at <mk_path>/../.. of registry rows; put
        # a tracked metakernel under <tmp>/kernels/mk so the tree is walked
        mk_dir = tracked.tmp / "kernels" / "mk"
        mk_dir.mkdir()
        shutil.copy(tracked.mk, mk_dir / "walk.tm")
        assert tracked.run("import", str(mk_dir / "walk.tm")) == 0
        capsys.readouterr()

        for target in ("urllib.request.urlopen",):
            with patch(target, side_effect=AssertionError("clean contacted the server")):
                assert tracked.run("clean") == 0
        out = capsys.readouterr().out
        assert "Would remove" in out and str(gone) in out and "dangling.tpc" in out
        assert link.is_symlink()

        assert tracked.run("clean", "--locations", "--execute") == 0
        out = capsys.readouterr().out
        assert "Removed" in out and "dangling.tpc" not in out
        assert link.is_symlink()
        assert tracked.run("clean", "--symlinks", "--execute") == 0
        assert not link.is_symlink()

        assert tracked.run("rm", "walk.tm") == 0
        assert (mk_dir / "walk.tm").is_file()
        assert tracked.run("rm", "walk.tm") == 1
        assert "not tracked" in capsys.readouterr().err


class TestCheckVerdicts:
    """#46: check reports three verdicts with their own exit bits, verify is
    a hidden alias, doctor carries the database-wide report."""

    MK_URL = "https://example.com/JUICE/kernels/mk/juice.tm"

    @pytest.fixture
    def tree(self, tmp_path, remote_sizes):
        from contextlib import ExitStack
        from types import SimpleNamespace

        from spice_kernel_db.cli import main

        kdir = tmp_path / "kernels" / "lsk"
        kdir.mkdir(parents=True)
        sizes = {"a.tls": 10, "b.tls": 20}
        for name, size in sizes.items():
            (kdir / name).write_bytes(b"x" * size)
        mk = tmp_path / "juice.tm"
        mk.write_text(
            "\\begindata\n"
            f"PATH_VALUES = ( '{tmp_path / 'kernels'}' )\n"
            "PATH_SYMBOLS = ( 'K' )\n"
            "KERNELS_TO_LOAD = ( '$K/lsk/a.tls' '$K/lsk/b.tls' )\n"
            "\\begintext\n"
        )
        db_path = tmp_path / "t.duckdb"
        db = KernelDB(db_path)
        db.add_mission("JUICE", "https://example.com/", "https://example.com/mk/", True)
        db.scan_directory(str(tmp_path / "kernels"), mission="JUICE")
        db.con.execute(
            "INSERT INTO metakernel_registry "
            "(mk_path, mission, source_url, filename, acquired_at) "
            "VALUES (?, 'JUICE', ?, 'juice.tm', now())",
            [str(mk), self.MK_URL],
        )
        db.close()

        remote_mk = (
            "\\begindata\nPATH_VALUES = ( '..' )\nPATH_SYMBOLS = ( 'K' )\n"
            "KERNELS_TO_LOAD = ( '$K/lsk/a.tls' '$K/lsk/b.tls' )\n\\begintext\n"
        )

        def url(name: str) -> str:
            return f"https://example.com/JUICE/kernels/lsk/{name}"

        def run(*argv: str, server_sizes: dict | None = None) -> int:
            """Exit code of ``skd --db ... argv``; the server answers only
            when *server_sizes* is given, otherwise any contact is an error."""
            with ExitStack() as stack:
                stack.enter_context(patch("spice_kernel_db.cli.ensure_config_or_setup", return_value=Config()))
                stack.enter_context(patch("spice_kernel_db.remote.fetch_metakernel", return_value=remote_mk))
                if server_sizes is not None:
                    stack.enter_context(remote_sizes({url(n): s for n, s in server_sizes.items()}))
                    stack.enter_context(patch("spice_kernel_db.db.list_remote_metakernels", return_value=[]))
                else:
                    # any contact is a failure: the offline guard in conftest raises,
                    # and so do these in case a name is bound at import
                    for target in ("spice_kernel_db.remote.query_remote_sizes",
                                   "spice_kernel_db.db.query_remote_sizes",
                                   "spice_kernel_db.db.list_remote_metakernels"):
                        stack.enter_context(patch(
                            target, side_effect=AssertionError("check touched the server without --remote")))
                try:
                    main(["--db", str(db_path), *argv])
                except SystemExit as exc:
                    return exc.code or 0
            return 0

        return SimpleNamespace(mk=str(mk), kdir=kdir, sizes=sizes, run=run)

    # C9
    def test_exit_bits(self, tree, capsys):
        assert tree.run("check", tree.mk) == 0
        assert tree.run("check", tree.mk, "--remote", server_sizes=tree.sizes) == 0

        # a size that disagrees with the database: present, does not resolve
        (tree.kdir / "a.tls").write_bytes(b"x" * 11)
        assert tree.run("check", tree.mk) == 2
        assert "SIZE_MISMATCH" in capsys.readouterr().out
        (tree.kdir / "a.tls").write_bytes(b"x" * 10)

        # the server disagrees: bit 4 alone
        assert tree.run("check", tree.mk, "--remote",
                        server_sizes={"a.tls": 99, "b.tls": 20}) == 4

        # b gone from disk: missing (1), and with the server disagreeing on a: 5
        (tree.kdir / "b.tls").unlink()
        assert tree.run("check", tree.mk) == 1
        assert tree.run("check", tree.mk, "--remote",
                        server_sizes={"a.tls": 99, "b.tls": 20}) == 5

    # C10
    def test_three_lines_always(self, tree, capsys):
        (tree.kdir / "b.tls").unlink()
        tree.run("check", tree.mk)
        out = capsys.readouterr().out
        assert "✗ present" in out and "1 missing" in out
        assert "✓ resolves" in out
        assert "– matches remote  not checked" in out

    # C11
    def test_remote_is_opt_in(self, tree, capsys):
        assert tree.run("check", tree.mk, "--hash") == 0      # no server contact
        assert tree.run("check", tree.mk, "--spice") == 2
        assert "needs --remote" in capsys.readouterr().err
        assert tree.run("check", tree.mk, "--full") == 2       # the old flag is gone

    # C12
    def test_verify_alias(self, tree, capsys):
        assert tree.run("verify", tree.mk, "--deep") == 0
        captured = capsys.readouterr()
        assert "'verify' is now 'check'" in captured.err
        assert "✓ present" in captured.out
        tree.run("--help")
        assert "verify" not in capsys.readouterr().out

    # C13
    def test_doctor(self, tree, capsys):
        assert tree.run("doctor") == 0
        out = capsys.readouterr().out
        assert "Poisoned supersession rows: 0" in out and "Kernels: 2" in out
        (tree.kdir / "b.tls").unlink()
        assert tree.run("doctor") == 0                          # a missing file is not corruption
        out = capsys.readouterr().out
        assert "whose file is gone: 1" in out and "b.tls" in out

    def test_json(self, tree, capsys):
        (tree.kdir / "b.tls").unlink()
        assert tree.run("check", tree.mk, "--json") == 1
        doc = json.loads(capsys.readouterr().out)
        assert doc["exit_code"] == 1 and doc["present"]["missing"] == ["$K/lsk/b.tls"]
        assert [v["ok"] for v in doc["verdicts"]] == [False, True, None]
