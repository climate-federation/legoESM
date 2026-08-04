"""``scripts/validate/rcemip_column_budget.py`` — instrument tests.

This validator decides whether an RCE run's column budget closes, so a bug in
IT manufactures a confident wrong physics conclusion.  Each numeric helper is
therefore pinned against a case with a KNOWN analytic answer before the tool is
allowed to be quoted (CLAUDE.md: validate the instrument first).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "validate"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import rcemip_column_budget as rcb  # noqa: E402

from legoesm import constants  # noqa: E402


def test_hydrostatic_pressure_matches_the_isothermal_analytic_solution():
    """For constant T_v and q_v=0 the exact solution is p = p_s exp(-z/H),
    H = R_d T / g.  Anything else means the integrator is wrong.
    """
    T0, p_s = 280.0, 101_480.0
    z_asc = np.linspace(0.0, 20_000.0, 400)
    z = z_asc[::-1]                              # driver stores TOP-DOWN
    T = np.full_like(z, T0)
    qv = np.zeros_like(z)
    p = rcb.hydrostatic_pressure(z, T, qv, p_s)
    H = constants.R_d * T0 / constants.g
    expected = p_s * np.exp(-z / H)              # exact isothermal solution
    np.testing.assert_allclose(p, expected, rtol=2e-4)
    assert np.all(np.diff(p) > 0), "p must increase downward (top-down order)"
    # z=0 is in the grid, so the bottom entry must be p_sfc itself.
    assert p[-1] == pytest.approx(p_s, rel=1e-12)


def test_hydrostatic_pressure_uses_virtual_temperature():
    """Adding vapour lowers density, so p must fall off MORE SLOWLY."""
    z = np.linspace(20_000.0, 0.0, 200)
    T = np.full_like(z, 280.0)
    p_dry = rcb.hydrostatic_pressure(z, T, np.zeros_like(z), 101_480.0)
    p_moist = rcb.hydrostatic_pressure(z, T, np.full_like(z, 0.02), 101_480.0)
    assert np.all(p_moist[:-1] > p_dry[:-1])


def test_layer_thickness_tiles_the_column_without_gap_or_overlap():
    """dz must sum to the top half-level height and every layer be positive."""
    z = np.linspace(30_000.0, 550.0, 30)          # top-down, uniform
    dz = rcb.layer_thickness(z)
    assert dz.shape == z.shape
    assert np.all(dz > 0.0)
    # Uniform full levels starting at 550 m with spacing 1100 m is exactly the
    # legoESM plane-CRM default grid; every layer should then be ~ the spacing,
    # and the lowest one reaches the ground.
    z2 = (np.arange(30) * 1100.0 + 550.0)[::-1]
    dz2 = rcb.layer_thickness(z2)
    np.testing.assert_allclose(dz2, 1100.0, rtol=1e-12)
    assert dz2.sum() == pytest.approx(30 * 1100.0)


def test_layer_mass_integrates_to_the_surface_pressure_over_g():
    """Sum of rho*dz over the column must be p_sfc/g -- the exact mass of a
    hydrostatic column.  This is the control that makes every column integral
    downstream trustworthy."""
    z = np.linspace(30_000.0, 20.0, 600)
    T = np.maximum(290.0 - 0.0065 * z, 200.0)
    qv = np.zeros_like(z)
    p_s = 101_480.0
    p = rcb.hydrostatic_pressure(z, T, qv, p_s)
    dm = rcb.layer_mass(p, z, T, qv)
    total = float(dm.sum())
    assert np.all(dm > 0.0), "every layer mass must be positive"
    np.testing.assert_allclose(total, p_s / constants.g, rtol=0.02)


def test_qv_from_mse_inverts_the_definition():
    z = np.linspace(50.0, 18_000.0, 25)
    T = 300.0 - 0.0067 * z
    qv = 0.018 * np.exp(-z / 4000.0)
    mse_kJ = (constants.c_pd * T + constants.g * z
              + constants.L_v * qv) * 1e-3
    back = rcb.qv_from_mse(mse_kJ.reshape(1, 1, -1), T.reshape(1, 1, -1), z)
    np.testing.assert_allclose(back[0, 0], qv, rtol=1e-12, atol=1e-16)


def test_require_finite_is_fatal_not_a_silent_nan_reduction():
    """nanmean/nansum hiding a broken frame is the documented failure mode."""
    rcb._require_finite("ok", np.ones(4))
    with pytest.raises(SystemExit):
        rcb._require_finite("bad", np.array([1.0, np.nan, 3.0]))
    with pytest.raises(SystemExit):
        rcb._require_finite("inf", np.array([1.0, np.inf]))


def test_analytic_pressure_control_passes_on_the_real_wing_profile():
    """C3 itself: the control must PASS on the profile the IC is built from."""
    rcb.check_analytic_pressure(rcb.WING_P_SFC)


def test_cwv_control_fails_loudly_on_a_wrong_column_integral(tmp_path, capsys):
    """Non-vacuity: C1 must FAIL when the stored CWV disagrees with ours.

    Without this, a passing C1 proves nothing (the delegating-wrapper trap).
    """
    z = np.linspace(20_000.0, 50.0, 20)
    T = np.maximum(300.0 - 0.0067 * z, 200.0)
    qv = 0.018 * np.exp(-z / 4000.0)
    mse_kJ = (constants.c_pd * T + constants.g * z
              + constants.L_v * qv) * 1e-3
    col = lambda a: np.tile(a.reshape(1, 1, -1), (2, 2, 1))  # noqa: E731
    d3 = tmp_path / "snapshots3d"
    d3.mkdir(parents=True)
    np.savez(d3 / "vol_00000100.npz", day=1.0, t_s=86400.0, dx=1000.0,
             Lx=2000.0, Ly=2000.0, z=z, T=col(T), mse=col(mse_kJ),
             cond=col(np.zeros_like(z)), qcloud=col(np.zeros_like(z)),
             w=col(np.zeros_like(z)))
    # Deliberately WRONG stored CWV (10x too big).
    bogus = {1.0: 400.0}
    with pytest.raises(SystemExit):
        rcb.analyse_volumes([d3 / "vol_00000100.npz"], rcb.WING_P_SFC,
                            0.05, bogus)
    # ... and PASSES when the stored CWV is consistent.
    p = rcb.hydrostatic_pressure(z, T, qv, rcb.WING_P_SFC)
    good_cwv = float(np.sum(qv * rcb.layer_mass(p, z, T, qv)))
    rcb.analyse_volumes([d3 / "vol_00000100.npz"], rcb.WING_P_SFC,
                        0.05, {1.0: good_cwv})


def test_c1_refuses_to_report_when_no_day_matches(tmp_path):
    """An unvalidatable integral must not be quoted at all."""
    z = np.linspace(20_000.0, 50.0, 10)
    T = np.maximum(300.0 - 0.0067 * z, 200.0)
    mse_kJ = (constants.c_pd * T + constants.g * z) * 1e-3
    col = lambda a: a.reshape(1, 1, -1)  # noqa: E731
    d3 = tmp_path / "snapshots3d"
    d3.mkdir(parents=True)
    np.savez(d3 / "vol_00000100.npz", day=1.0, t_s=86400.0, dx=1.0, Lx=1.0,
             Ly=1.0, z=z, T=col(T), mse=col(mse_kJ), cond=col(np.zeros_like(z)),
             qcloud=col(np.zeros_like(z)), w=col(np.zeros_like(z)))
    with pytest.raises(SystemExit):
        rcb.analyse_volumes([d3 / "vol_00000100.npz"], rcb.WING_P_SFC, 0.05, {})


# --- adaptive dt (P2) — pure helpers from scripts/run/run_rcemip_plane.py ---

_RUN = Path(__file__).resolve().parents[2] / "scripts" / "run"
if str(_RUN) not in sys.path:
    sys.path.insert(0, str(_RUN))


def _ladder():
    import run_rcemip_plane as rcp
    return rcp.adaptive_dt_ladder


def test_adaptive_dt_halves_above_ca_hi_and_stops_at_the_floor():
    lad = _ladder()
    kw = dict(ca_hi=0.5, ca_lo=0.1, min_factor=0.125)
    dt = 20.0
    for expect in (10.0, 5.0, 2.5, 2.5, 2.5):     # floor = 20*0.125 = 2.5
        dt = lad(dt, 20.0, courant=0.9, **kw)
        assert dt == pytest.approx(expect)


def test_adaptive_dt_restores_toward_base_but_never_past_it():
    lad = _ladder()
    kw = dict(ca_hi=0.5, ca_lo=0.1, min_factor=0.125)
    dt = 2.5
    for expect in (5.0, 10.0, 20.0, 20.0):
        dt = lad(dt, 20.0, courant=0.01, **kw)
        assert dt == pytest.approx(expect)


def test_adaptive_dt_holds_inside_the_dead_band():
    lad = _ladder()
    assert lad(10.0, 20.0, courant=0.3, ca_hi=0.5, ca_lo=0.1,
               min_factor=0.125) == pytest.approx(10.0)


def test_adaptive_dt_rungs_are_exact_powers_of_two_so_jit_cache_hits():
    """Every rung must be bit-exactly reachable from both directions, else a
    'restored' dt is a NEW Python float and forces a fresh compilation."""
    lad = _ladder()
    kw = dict(ca_hi=0.5, ca_lo=0.1, min_factor=0.125)
    down, dt = [], 20.0
    for _ in range(3):
        dt = lad(dt, 20.0, courant=0.9, **kw)
        down.append(dt)
    up = []
    for _ in range(3):
        dt = lad(dt, 20.0, courant=0.01, **kw)
        up.append(dt)
    # coming back up must land on exactly the same float objects' values
    assert up[:-1] == down[-2::-1], (down, up)
    assert len(set(down) | set(up)) <= 4      # bounded retrace count


def test_adaptive_dt_rejects_inverted_or_impossible_thresholds():
    lad = _ladder()
    with pytest.raises(ValueError):
        lad(10.0, 20.0, 0.3, ca_hi=0.1, ca_lo=0.5, min_factor=0.125)
    with pytest.raises(ValueError):
        lad(10.0, 20.0, 0.3, ca_hi=0.5, ca_lo=0.1, min_factor=0.0)
    with pytest.raises(ValueError):
        lad(10.0, 20.0, 0.3, ca_hi=0.5, ca_lo=0.1, min_factor=2.0)


def test_courant_uses_dz_for_w_and_dx_for_horizontal():
    """The F11 CFL error was dividing a VERTICAL velocity by the HORIZONTAL
    spacing. Pin that w goes with dz and u,v go with dx."""
    import run_rcemip_plane as rcp
    import jax.numpy as jnp
    from types import SimpleNamespace as NS
    st = NS(w=NS(data=jnp.array([[[10.0]]])),
            u=NS(data=jnp.array([[[4.0]]])),
            v=NS(data=jnp.array([[[2.0]]])))
    ca_v, ca_h = rcp.courant_numbers(st, dx=4000.0, dz_min=1100.0, dt=20.0)
    assert ca_v == pytest.approx(10.0 * 20.0 / 1100.0)
    assert ca_h == pytest.approx(4.0 * 20.0 / 4000.0)
    # The documented F11 case: w=225, dt=20, dx=4 km, dz=1100 m.
    ca_v2, _ = rcp.courant_numbers(
        NS(w=NS(data=jnp.array([[[225.0]]])), u=NS(data=jnp.zeros((1, 1, 1))),
           v=NS(data=jnp.zeros((1, 1, 1)))),
        dx=4000.0, dz_min=1100.0, dt=20.0)
    assert ca_v2 == pytest.approx(4.09, abs=0.01)   # NOT the 0.225 in F11
