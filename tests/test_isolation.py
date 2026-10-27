"""The suite cannot reach the developer's real config or database (#28).

These tests deliberately call the real ``save_config`` and ``ensure_config``
with no patching at all. If the autouse ``_isolated_home`` fixture in
``conftest.py`` ever stops working, they write into the pytest temp tree and
still pass -- but the session-scoped ``_real_config_untouched`` guard then
fails on the real file. Together the two pin the behaviour from both sides.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from spice_kernel_db import config
from spice_kernel_db.config import Config, ensure_config, save_config


def _home(tmp_path_factory) -> Path:
    home = Path(os.environ["HOME"])
    assert home.is_relative_to(tmp_path_factory.getbasetemp()), home
    return home


def test_unpatched_save_config_writes_under_a_temporary_home(tmp_path_factory):
    home = _home(tmp_path_factory)
    save_config(Config(db_path="/x/test.duckdb", kernel_dir="/y/kernels"))
    written = home / ".config" / "spice-kernel-db" / "config.toml"
    assert config.CONFIG_FILE == written
    assert written.is_file()
    assert "/y/kernels" in written.read_text()


def test_unpatched_first_run_setup_writes_under_a_temporary_home(tmp_path_factory):
    home = _home(tmp_path_factory)
    assert not config.CONFIG_FILE.exists()
    with patch("builtins.input", return_value=""):
        cfg = ensure_config()
    assert cfg == Config()
    assert config.CONFIG_FILE.is_file()
    # The defaults expand under the temporary HOME, not the real one.
    assert Path(cfg.db_path).expanduser().is_relative_to(home)
