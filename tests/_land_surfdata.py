"""Harmonized land surfdata path for tests whose land scheme is a canopy.

The land surface scheme defaults to the two-leaf canopy, which refuses to start
without the per-PFT parameters this file carries.  Resolved against THIS
checkout's data directory, with an environment override for a shared copy.
"""
from __future__ import annotations

import os
import pathlib

SURFDATA = os.environ.get("LEGOESM_TEST_SURFDATA") or str(
    pathlib.Path(__file__).resolve().parents[1]
    / "data" / "legoesm_surfdata_c260716.nc")


def require_surfdata() -> str:
    """Return SURFDATA, or FAIL (never skip) with an actionable message."""
    if not os.path.exists(SURFDATA):
        raise AssertionError(
            f"this test needs the harmonized surfdata and none is at "
            f"{SURFDATA}. The land surface scheme defaults to two-leaf, and a "
            f"canopy run refuses to start without the per-PFT parameters that "
            f"file carries. Stage it or point LEGOESM_TEST_SURFDATA at a copy.")
    return SURFDATA
