"""#49: the library is quiet, the config never prompts, the public API is declared."""

from __future__ import annotations

import ast
import io
import logging
import re
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.progress import Progress

import spice_kernel_db
from spice_kernel_db import (
    ConfigError,
    KernelDB,
    add_root,
    check,
    ensure_config,
    get,
    kernel_roots,
    register_metakernel,
    resolve_metakernel,
    scan,
)
from spice_kernel_db import config as config_mod
from spice_kernel_db.config import write_first_config
from spice_kernel_db.remote import query_remote_sizes
from tests.helpers import REMOTE_MK_TEXT, fake_download

MK_URL = "https://example.com/pub/naif/JUICE/kernels/mk/remote_test.tm"


@pytest.fixture
def offline_get(monkeypatch):
    """The three network calls behind get_metakernel, answered locally."""
    monkeypatch.setattr("spice_kernel_db.db._fetch_metakernel", lambda url: (REMOTE_MK_TEXT, url))
    monkeypatch.setattr("spice_kernel_db.db.query_remote_sizes", lambda urls, **kw: {})
    monkeypatch.setattr("spice_kernel_db.db.download_kernels_parallel", fake_download)


@pytest.fixture
def make_db(tmp_path):
    """A KernelDB with one root under tmp_path, closed at teardown."""
    opened: list[KernelDB] = []

    def _make(console: Console | None = None) -> KernelDB:
        db = KernelDB(tmp_path / f"db{len(opened)}.duckdb",
                      kernel_roots=[tmp_path / "store"], console=console)
        opened.append(db)
        return db

    yield _make
    for db in opened:
        db.close()


class TestQuietLibrary:
    def test_library_is_silent(self, tmp_spice_tree, make_db, offline_get, capsys):
        """C40: without a console, nothing reaches stdout, get included."""
        db = make_db()
        db.scan_directory(tmp_spice_tree / "JUICE" / "kernels", mission="JUICE")
        db.scan_directory(tmp_spice_tree / "generic_kernels", mission="generic")
        db.stats()
        db.report_duplicates()
        db.list_metakernels()
        db.info_metakernel("juice_test.tm")
        db.check_report("juice_test.tm")
        db.prune(dry_run=True)
        result = db.get_metakernel(MK_URL, mission="JUICE", yes=True)
        assert len(result["downloaded"]) == 1  # naif0012.tls was already in the tree
        assert capsys.readouterr().out == ""

    def test_console_prints(self, tmp_spice_tree, make_db, offline_get):
        """C41: the same calls on a console print what the CLI printed."""
        buf = io.StringIO()
        db = make_db(Console(file=buf, width=200))
        db.scan_directory(tmp_spice_tree / "JUICE" / "kernels", mission="JUICE")
        db.scan_directory(tmp_spice_tree / "generic_kernels", mission="generic")
        db.stats()
        db.report_duplicates()  # generic and JUICE share files, so it lists [mission] path lines
        db.get_metakernel(MK_URL, mission="JUICE", yes=True)
        out = buf.getvalue()
        assert "Scanned" in out and "kernel files registered" in out  # plain text
        assert "Unique kernels" in out  # a panel
        assert "[JUICE]" in out  # a bracketed mission survives: plain text, not markup
        assert "Metakernel: remote_test.tm" in out and "Downloaded:" in out

    def test_no_console_means_no(self, tmp_path, make_db, offline_get, monkeypatch):
        """C42: no console and no yes: nothing downloaded, nothing left behind."""
        monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("asked stdin"))
        db = make_db()
        with patch.object(KernelDB, "index_metakernel", side_effect=AssertionError("wrote")):
            result = db.get_metakernel(MK_URL, mission="JUICE")
        assert result["aborted"] is True and result["dry_run"] is True
        assert result["downloaded"] == [] and result["missing"]  # the plan is still reported
        assert db.registry_rows() == []
        assert not list((tmp_path / "store").rglob("*.bsp"))

    def test_console_question_takes_a_patched_input(self, make_db, offline_get):
        """C42: the question goes through the console, so a patched input answers it."""
        db = make_db(Console(file=io.StringIO()))
        with patch("builtins.input", return_value="y"):
            result = db.get_metakernel(MK_URL, mission="JUICE")
        assert len(result["downloaded"]) == 2  # an empty database misses both

    def test_closed_stdin_means_no(self, make_db, offline_get):
        """C42: EOF on the question is no, as the CLI's own prompts treat it (#14)."""
        db = make_db(Console(file=io.StringIO()))
        with patch("builtins.input", side_effect=EOFError):
            result = db.get_metakernel(MK_URL, mission="JUICE")
        assert result["aborted"] is True

    def test_progress_disabled_without_console(self):
        """C41: remote progress bars follow the console."""
        with patch("spice_kernel_db.remote.Progress", wraps=Progress) as P:
            query_remote_sizes([])
            assert P.call_args.kwargs["disable"] is True
            query_remote_sizes([], console=Console(file=io.StringIO()))
            assert P.call_args.kwargs["disable"] is False

    def test_warnings_are_logged(self, tmp_path, caplog, capsys):
        """C43: a missing kernel root is a logger warning, not output."""
        db = KernelDB(tmp_path / "q.duckdb", kernel_roots=[tmp_path / "gone"])
        with caplog.at_level(logging.WARNING, logger="spice_kernel_db"):
            db.scan_roots()
        db.close()
        assert "does not exist" in caplog.text
        assert capsys.readouterr().out == ""

    def test_cli_missing_root_goes_to_stderr_once(self, tmp_path, capsys):
        """C43: the CLI prints the library's warnings on stderr as bare lines, one handler per run."""
        from spice_kernel_db.cli import main
        from spice_kernel_db.config import Config, save_config

        (tmp_path / "root").mkdir()
        save_config(Config(db_path=str(tmp_path / "c.duckdb"),
                           kernel_dirs=[str(tmp_path / "root"), str(tmp_path / "gone")]))
        main(["scan"])
        main(["scan"])  # a second run in the same process adds no second line
        out, err = capsys.readouterr()
        assert err.count("does not exist") == 2
        assert err.count("WARNING") == 0 and "does not exist" not in out

    def test_no_output_calls_outside_cli(self):
        """C40/C47: the grep from issues #6 and #49."""
        src = Path(spice_kernel_db.__file__).parent
        hits = []
        for py in src.glob("*.py"):
            for n, line in enumerate(py.read_text().splitlines(), 1):
                code = line.split("#", 1)[0]
                if re.search(r"(?<![\w.])(print|input)\(", code):
                    hits.append(f"{py.name}:{n}: {line.strip()}")
        assert hits == []


class TestConfigErrors:
    def test_corrupt_file_raises(self):
        """C44: a corrupt file is an error, never "not configured"."""
        config_mod.CONFIG_FILE.parent.mkdir(parents=True)
        config_mod.CONFIG_FILE.write_text("[storage\nkernel_dirs = [")
        with pytest.raises(ConfigError, match="corrupt"):
            config_mod.load_config()
        with pytest.raises(ConfigError):
            ensure_config()

    def test_no_file_no_seed_raises(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("prompted"))
        with pytest.raises(ConfigError, match="config --setup"):
            ensure_config()

    def test_seed_writes(self, tmp_path):
        cfg = ensure_config(seed_root=tmp_path / "seed")
        assert cfg.kernel_dirs == [str(tmp_path / "seed")]
        assert config_mod.CONFIG_FILE.is_file()

    def test_cli_corrupt_config_exits_78(self, capsys):
        """C44: a config error never shares an exit code with check's verdict bits."""
        from spice_kernel_db.cli import main

        config_mod.CONFIG_FILE.parent.mkdir(parents=True)
        config_mod.CONFIG_FILE.write_text("[storage\nkernel_dirs = [")
        with pytest.raises(SystemExit) as exc:
            main(["check", "x.tm"])
        assert exc.value.code == 78
        assert "corrupt" in capsys.readouterr().err

    def test_cli_first_run_without_a_terminal_exits_2(self, capsys):
        """C44: the CLI's first-run prompt sits behind the terminal gate (#14)."""
        from spice_kernel_db.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["ls"])
        assert exc.value.code == 2
        assert "config --setup" in capsys.readouterr().err


class TestHostContract:
    """C48: what a host CLI (planetarypy's `plp spice`) mounts and asks."""

    def test_exit_code_mapping(self):
        import typer

        from spice_kernel_db.cli import EXIT_CONFIG, app, exit_code
        from spice_kernel_db.db import (
            AmbiguousKernelRoot,
            AmbiguousMetakernel,
            MetakernelNotTracked,
        )

        assert isinstance(app, typer.Typer)
        assert exit_code(MetakernelNotTracked("x.tm")) == 1
        assert exit_code(AmbiguousMetakernel("x", ["a.tm", "b.tm"])) == 2
        assert exit_code(AmbiguousKernelRoot("JUICE", [Path("/one"), Path("/two")])) == 2
        assert exit_code(ConfigError("corrupt")) == EXIT_CONFIG == 78
        assert exit_code(ValueError("not ours")) is None

    def test_logging_to_stderr_is_scoped_to_the_run(self, capsys):
        from spice_kernel_db.cli import logging_to_stderr

        log = logging.getLogger("spice_kernel_db.db")
        with logging_to_stderr():
            log.warning("inside")
        log.warning("outside")  # Python's last-resort handler prints this one
        err = capsys.readouterr().err
        assert err.count("inside") == 1 and "WARNING" not in err.split("outside")[0]
        assert not logging.getLogger("spice_kernel_db").handlers


class TestPublicApi:
    def test_all_matches_docs(self):
        """C45: the documented list and __all__ are the same names."""
        doc = (Path(__file__).parent.parent / "docs" / "api.qmd").read_text()
        section = doc.split("## Public API", 1)[1].split("\n## ", 1)[0]
        block = section.split("```python", 1)[1].split("```", 1)[0]
        names = {
            alias.name
            for node in ast.parse(block).body if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert names == set(spice_kernel_db.__all__)

    def test_dir_matches_all(self):
        public = {
            n for n in dir(spice_kernel_db)
            if not n.startswith("_")
            and not isinstance(getattr(spice_kernel_db, n), types.ModuleType)
        }
        assert public == set(spice_kernel_db.__all__)

    def test_module_functions(self, tmp_spice_tree, tmp_path, offline_get):
        """C45: the module-level functions work against the configured store."""
        store = tmp_path / "store"
        write_first_config(str(tmp_path / "api.duckdb"), str(store))
        n = scan(tmp_spice_tree / "JUICE" / "kernels", mission="JUICE")
        assert n > 0
        mk = tmp_spice_tree / "JUICE" / "kernels" / "mk" / "juice_test.tm"
        assert isinstance(register_metakernel(mk, mission="JUICE", official=False), dict)
        assert resolve_metakernel("juice_test").resolve() == mk.resolve()
        report = check("juice_test.tm")
        assert isinstance(report["exit_code"], int) and report["verdicts"]
        assert kernel_roots() == [store.resolve()]
        cfg = add_root(tmp_path / "two")
        assert len(cfg.kernel_dirs) == 2
        result = get(MK_URL, mission="JUICE", yes=True, download_dir=store)
        assert len(result["downloaded"]) == 1
        assert scan() > 0  # no directory: every configured root (the store holds the download)
