"""Veros global 4-degree adapter for the legoESM ``global_overturning`` case.

Wraps :class:`veros.setups.global_4deg.global_4deg.GlobalFourDegreeSetup`.
First-run requires downloading ~50 MB of forcing fields from sid.erda.dk
(``forcing_4deg_global_open_itf.nc`` + ``ecmwf_4deg_monthly_nc4.nc``); the
download is handled inside Veros's asset loader and cached under Veros's
own data directory.
"""

from __future__ import annotations

DEFAULT_RUNLEN_S: float = 90.0 * 86400.0  # 90-day quick comparison


def make_setup(*, runlen_s: float = DEFAULT_RUNLEN_S):
    """Return a fresh GlobalFourDegreeSetup with a shortened ``runlen``."""
    from veros.setups.global_4deg.global_4deg import GlobalFourDegreeSetup

    setup = GlobalFourDegreeSetup()
    setup._legoesm_target_runlen_s = float(runlen_s)
    return setup
