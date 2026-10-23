"""Real requests against NAIF and ESA.

Deselected by default. Run them deliberately::

    pytest -m network
    pytest -m network -k esa          # just the ESA ones

Why these exist
---------------
Everything in test_kernel_db.py is mocked, so nothing proved skd can actually
talk to the archives it exists to talk to. A change that broke real-server
interaction — a listing-regex drift after an Apache upgrade, a redirect that
HEAD stopped following, a timeout too short for ESA's handshake — would pass
the whole suite.

These are the opposite trade: slow, dependent on two external services, and
genuinely checking the thing the package is for. Keeping them behind a marker
means a PR that edits a help string does not need JPL to be up.

What they deliberately do NOT do
--------------------------------
No large downloads. The biggest fetch here is one 131 KB generic PCK. Size
queries and listings are HEAD or small GETs. These are meant to be runnable
often without being a burden on either archive.
"""

from __future__ import annotations

import pytest

from spice_kernel_db.remote import (
    SPICE_SERVERS,
    ProbeResult,
    list_remote_metakernels,
    list_remote_missions,
    probe_url,
    query_remote_sizes,
)

pytestmark = pytest.mark.network

NAIF = SPICE_SERVERS["NASA"]
ESA = SPICE_SERVERS["ESA"]

# A generic kernel: same file for every mission, stable for years, 131 KB.
GENERIC_PCK = NAIF + "generic_kernels/pck/pck00011.tpc"


class TestProbeAgainstRealServers:
    """probe_url's whole purpose is telling 'absent' from 'unreachable' (#10)."""

    def test_existing_file_is_present(self):
        assert probe_url(GENERIC_PCK).result is ProbeResult.PRESENT

    def test_missing_file_is_absent_not_unreachable(self):
        """A 404 from a healthy server must never read as 'cannot reach'."""
        p = probe_url(NAIF + "generic_kernels/pck/does_not_exist_xyz.tpc")
        assert p.result is ProbeResult.ABSENT, (
            f"a real 404 was classified {p.result}: {p.detail}"
        )
        assert "404" in (p.detail or "")

    def test_unroutable_address_is_unreachable_not_absent(self):
        """The failure mode #10 was about: no answer must not mean 'missing'."""
        p = probe_url("https://10.255.255.1/nothing.tm", timeout=3.0)
        assert p.result is ProbeResult.UNREACHABLE
        assert "no response" in (p.detail or "")


class TestListingsParseAgainstLiveApache:
    """The directory regexes are the most fragile thing in the package.

    NASA serves plain-text indexes and ESA serves table-based ones, and both
    are mod_autoindex output that an upgrade can reshape. Mocked fixtures
    freeze whatever the format was on the day they were captured.
    """

    def test_naif_missions_listing(self):
        missions = list_remote_missions(NAIF)
        assert len(missions) > 20, f"only {len(missions)} missions parsed"
        upper = {m.upper() for m in missions}
        assert "JUICE" in upper

    def test_esa_missions_listing(self):
        missions = list_remote_missions(ESA)
        assert len(missions) > 5, f"only {len(missions)} missions parsed"
        upper = {m.upper() for m in missions}
        assert "JUICE" in upper

    def test_esa_metakernel_listing_has_names_dates_and_sizes(self):
        rows = list_remote_metakernels(ESA + "JUICE/kernels/mk/")
        assert rows, "no metakernels parsed from the ESA listing"
        r = rows[0]
        assert r.filename.lower().endswith(".tm")
        assert r.date, "no date parsed"
        assert r.size, "no size parsed"


class TestSizeQueriesAgainstRealServers:
    """query_remote_sizes underpins check --full (#15)."""

    def test_generic_pck_size(self):
        sizes = query_remote_sizes([GENERIC_PCK])
        assert sizes[GENERIC_PCK] == 131226, (
            "pck00011.tpc changed size upstream; if NAIF reissued it, update "
            "this number and check nothing else assumed the old one"
        )

    def test_missing_file_reports_no_size_rather_than_raising(self):
        url = NAIF + "generic_kernels/pck/does_not_exist_xyz.tpc"
        assert query_remote_sizes([url])[url] is None


class TestGenericKernelIsIdenticalAcrossMissions:
    """The premise behind re-scoping #16.

    pck00011.tpc is a generic kernel: NAIF's copy and the one under ESA's
    JUICE tree are the same file. A mission-scoped copy that differs is a
    local problem, not two legitimate variants — which is why the fix belongs
    in `check --full` and in generic-kernel tracking, not in resolve_kernel.
    """

    JUICE_PCK = ESA + "JUICE/kernels/pck/pck00011.tpc"

    def test_naif_and_esa_juice_report_the_same_size(self):
        sizes = query_remote_sizes([GENERIC_PCK, self.JUICE_PCK])
        assert sizes[GENERIC_PCK] == sizes[self.JUICE_PCK], (
            f"generic {sizes[GENERIC_PCK]} vs JUICE {sizes[self.JUICE_PCK]}"
        )

    def test_bepicolombo_variant_is_a_different_file_with_a_different_name(self):
        """It is pck00011_bc_v00.tpc, not a different pck00011.tpc."""
        bc = ESA + "BEPICOLOMBO/kernels/pck/pck00011_bc_v00.tpc"
        sizes = query_remote_sizes([GENERIC_PCK, bc])
        assert sizes[bc] is not None, "the BepiColombo variant moved"
        assert sizes[bc] != sizes[GENERIC_PCK]


class TestEsaHandshakeIsWithinOurTimeout:
    """#12 and #10 were both about ESA's slow TLS handshake.

    The constants in remote.py were chosen from measurements; this fails if
    the server has drifted far enough that they no longer hold, which is the
    signal to re-measure rather than to keep raising them blindly.
    """

    def test_a_probe_completes_inside_the_configured_timeout(self):
        from spice_kernel_db.remote import PROBE_TIMEOUT

        p = probe_url(ESA + "JUICE/kernels/mk/", timeout=PROBE_TIMEOUT)
        assert p.result is not ProbeResult.UNREACHABLE, (
            f"ESA did not answer within PROBE_TIMEOUT={PROBE_TIMEOUT}s "
            f"({p.detail}); re-measure before raising it"
        )
