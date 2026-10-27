"""Shared fixtures, and the rule that unit tests do not touch the network.

Why this exists
---------------
The suite already *intended* to be offline: the metakernel fixtures patch
``spice_kernel_db.remote.urllib.request.urlopen``. But the size queries go
through a ``requests`` session instead, which that patch never covered, so a
single test opened six real TLS connections to naif.jpl.nasa.gov. They failed,
the failures were swallowed, and the test passed anyway — paying roughly 0.66 s
of handshake per connection and making every CI run depend on JPL being
reachable.

Blocking the sockets changed no assertion, which is the whole argument: those
connections carried no coverage. So the default is now hard-offline, enforced
rather than hoped for.

What this does NOT mean
-----------------------
This package exists to fetch kernels over the network, so the network genuinely
has to be tested. That is what ``tests/test_network.py`` is for: real requests
to NAIF and ESA, marked ``network``, deselected by default and run on demand:

    pytest -m network

Keeping them apart means a PR that changes a help string does not need JPL to
be up, while the fetching behaviour still gets exercised against real servers.
"""

from __future__ import annotations


import socket

import pytest

from tests._netguard import NetworkUseInOfflineTest

_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "network: test performs real network I/O against NAIF or ESA. "
        "Deselected by default; run with `pytest -m network`.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Deselect network tests unless they were asked for explicitly."""
    if config.getoption("-m"):
        return  # the user said what they wanted; respect it
    skip = pytest.mark.skip(reason="needs -m network (real NAIF/ESA requests)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_network(request):
    """Fail loudly if an offline test opens a socket.

    A guard rather than a convention: the previous leak was invisible precisely
    because the connections failed silently. Localhost is left alone — some
    libraries open loopback sockets for unrelated reasons.
    """
    if "network" in request.keywords:
        yield
        return

    def _blocked(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "::1", "localhost"):
            return _REAL_CONNECT(self, address, *args, **kwargs)
        raise NetworkUseInOfflineTest(
            f"{request.node.nodeid} tried to connect to {address}. "
            "Unit tests must not touch the network: mock it, or mark the test "
            "`@pytest.mark.network` if it genuinely needs a real server."
        )

    def _blocked_ex(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "::1", "localhost"):
            return _REAL_CONNECT_EX(self, address, *args, **kwargs)
        raise NetworkUseInOfflineTest(
            f"{request.node.nodeid} tried to connect to {address}."
        )

    socket.socket.connect = _blocked
    socket.socket.connect_ex = _blocked_ex
    try:
        yield
    finally:
        socket.socket.connect = _REAL_CONNECT
        socket.socket.connect_ex = _REAL_CONNECT_EX


@pytest.fixture
def remote_sizes():
    """Patch query_remote_sizes with deterministic sizes.

    Before the offline guard, these tests opened real connections to NAIF and
    whatever came back was never asserted on, so the size column was untested
    either way. This fixture makes the sizes known so a test can check them.

    Use as::

        def test_x(remote_sizes):
            with remote_sizes({"https://.../naif0012.tls": 5000}):
                ...
    """
    from contextlib import contextmanager
    from unittest.mock import patch

    @contextmanager
    def _patch(sizes: dict[str, int | None]):
        # db.py binds the name at import, so patching the remote module
        # alone never reached get_metakernel (#33).
        with patch("spice_kernel_db.remote.query_remote_sizes",
                   return_value=sizes), \
             patch("spice_kernel_db.db.query_remote_sizes",
                   return_value=sizes) as m:
            yield m

    return _patch


# --- fixtures moved here when test_kernel_db.py was split by subsystem ---
# These need the same imports the old test module had.
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
from tests.helpers import FIXTURE_REGISTRY


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


@pytest.fixture
def fixture_registry(monkeypatch):
    """Point the registry loader at the test fixture (no cache, no bundled lookup)."""
    from spice_kernel_db import registry as reg
    reg.load_registry.cache_clear()
    monkeypatch.setattr(reg, "load_registry", lambda: reg.load_registry_from(FIXTURE_REGISTRY))
    yield reg


# --- user-file isolation (#28) ------------------------------------------------
#
# On 2026-10-24 the suite overwrote ~/.config/spice-kernel-db/config.toml with
# a test's tmp paths. ``test_config_set`` kept the real ``save_config`` away
# only by patching it *by name*; the Typer port moved the import site, the
# patch missed, and the real file was written. Patch targets drift; HOME does
# not. So every test runs with HOME pointed at an empty directory, and the
# session fails if the developer's real config changed anyway.

import os


def _snapshot(path: Path) -> tuple:
    """Content hash plus mtime, or (None,) when the file does not exist."""
    if not path.is_file():
        return (None,)
    return (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)


@pytest.fixture(scope="session", autouse=True)
def _real_config_untouched():
    """Fail the session if any test wrote the developer's real config file.

    Captured before the first test, when HOME is still the real one.
    """
    real = Path(os.environ.get("HOME", str(Path.home())))
    config_file = real / ".config" / "spice-kernel-db" / "config.toml"
    before = _snapshot(config_file)
    yield
    after = _snapshot(config_file)
    assert after == before, (
        f"the test suite modified {config_file} (#28). Some test reached the "
        "real user config: check for a patch target that no longer matches "
        "the import site."
    )


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path_factory):
    """Point HOME, CONFIG_DIR and CONFIG_FILE at an empty directory (#28).

    ``CONFIG_DIR`` / ``CONFIG_FILE`` are expanded at import time, so setting
    HOME alone would not move them; ``DEFAULT_DB_PATH`` and ``LEGACY_DB_PATH``
    are expanded on use, so HOME covers those.
    """
    from spice_kernel_db import config

    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(config, "CONFIG_DIR", home / ".config" / "spice-kernel-db")
    monkeypatch.setattr(config, "CONFIG_FILE", config.CONFIG_DIR / "config.toml")
