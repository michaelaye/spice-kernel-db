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
        with patch("spice_kernel_db.remote.query_remote_sizes",
                   return_value=sizes) as m:
            yield m

    return _patch
