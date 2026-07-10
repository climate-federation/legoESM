"""#871: the MPAS/spectral daily blowup checks were finiteness-ONLY, so a
runaway to 8e8 K stayed 'finite' for 1138 steps (and the T_min=50 floor pinned
its low side).  physical_state_blowup_reason gives every run-time detector the
SAME physical-bounds guard so a runaway aborts at the first daily check.
"""
from __future__ import annotations

from legoesm.driver.diagnostics import physical_state_blowup_reason


def test_healthy_state_passes():
    assert physical_state_blowup_reason(19.0, 210.0, 300.0,
                                        ps_min=6.0e4, ps_max=1.02e5) is None


def test_high_T_runaway_flagged():
    r = physical_state_blowup_reason(20.0, 210.0, 8.0e8)
    assert r is not None and "temperature out of physical bounds" in r


def test_low_T_floor_pinning_flagged():
    # the checkerboard pins even levels at exactly 50 K (< 100) — must trip
    r = physical_state_blowup_reason(20.0, 50.0, 300.0)
    assert r is not None and "min=50.0K" in r


def test_surface_pressure_out_of_bounds_flagged():
    r = physical_state_blowup_reason(5.0, 250.0, 300.0, ps_min=1.0e4, ps_max=1.0e5)
    assert r is not None and "surface pressure out of bounds" in r


def test_no_ps_bounds_when_not_supplied():
    # T in range and no p_s given -> no blowup (daily checks only have T stats)
    assert physical_state_blowup_reason(1.0, 200.0, 320.0) is None
