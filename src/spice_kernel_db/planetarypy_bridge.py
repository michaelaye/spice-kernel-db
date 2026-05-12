"""Optional integration hook for the planetarypy library.

planetarypy already manages PDS-archived SPICE bundles for several NASA
missions. When the [planetarypy] extra is installed and a mission's registry
entry has ``planetarypy = true``, the mission-add flow can offer to delegate
discovery to planetarypy.

This module is intentionally minimal — full delegation is tracked as a
follow-up. The current ``delegate_mission_add`` always returns None, which
causes the CLI to fall back to normal discovery while informing the user.
"""

from __future__ import annotations

import importlib.util

_TRACKING_ISSUE = "https://github.com/michaelaye/spice-kernel-db/issues/2"


def is_available() -> bool:
    """Return True if the optional planetarypy dependency is importable."""
    return importlib.util.find_spec("planetarypy") is not None


def delegate_mission_add(_mission: str, _server_url: str) -> dict | None:
    """Stub delegation hook for planetarypy-managed missions.

    Intended to return ``{'mk_dir_url': ..., 'managed_by': 'planetarypy'}``
    once full integration lands. Currently returns ``None`` to signal that
    callers should fall back to normal discovery.
    """
    return None


def tracking_issue() -> str:
    """URL of the issue tracking full planetarypy integration."""
    return _TRACKING_ISSUE
