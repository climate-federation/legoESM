"""RAD-7: plane radiation surface boundary = prescribed SST (iter-48).

SAM's radiation uses the (fixed) sea-surface temperature for the surface
longwave emission ``σ·ε·T_sfc⁴``; legoESM's radiation defaults to the
lowest-level AIR temperature ``T[..., -1]``, which drifts from the SST as the
column evolves. ``_plane_radiation_physics_with_sst`` pins the surface boundary
to the SST via the existing ``set_T_sfc_override`` hook. These tests verify the
override actually controls the surface radiative T.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "run"))
import run_rcemip_plane as rcp  # noqa: E402

from legoesm.atmosphere.physics.radiation.integration import (  # noqa: E402
    make_radiation_physics,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (  # noqa: E402
    make_flat_plane_terrain_metric, make_rest_state,
)
from legoesm.grids.plane import create_plane_grid  # noqa: E402
from legoesm.grids.vertical import create_height_coordinate  # noqa: E402


def _state_and_grid():
    grid = create_plane_grid(nx=6, ny=6, nlev=24, dx=1_000.0, dy=1_000.0,
                             dtype=jnp.float64)
    hc = create_height_coordinate(24, H=20_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    st = make_rest_state(grid, hc, dtype=jnp.float64)
    tr = np.zeros((6, 6, 24, 6))
    tr[..., 0] = 0.005   # q_v so the gray scheme has an absorber
    st = st._replace(tracers=st.tracers.replace(data=jnp.asarray(tr)))
    t_air_sfc = float(
        (hc.theta_ref + st.theta_prime.data)[0, 0, -1] * hc.exner_ref[-1])
    return st, grid, hc, tm, t_air_sfc


def _gray_cfg():
    return rcp._build_radiation_config(
        "gray", update_interval_steps=1, clouds=True,
        insolation="rcemip", t_sfc=300.0)


def test_sst_override_equal_to_air_T_is_noop():
    """Pinning the SST to the lowest-level AIR T reproduces the default
    (``T[..., -1]``) radiation exactly — the override only swaps the surface
    boundary value, so SST==air_T changes nothing."""
    st, grid, hc, tm, t_air = _state_and_grid()
    cfg = _gray_cfg()
    rad_default = make_radiation_physics(cfg, model_type="plane")
    rad_pinned = rcp._plane_radiation_physics_with_sst(cfg, grid, t_air)
    d0 = np.asarray(rad_default(st, grid, hc, tm).dtheta_prime_dt.data)
    d1 = np.asarray(rad_pinned(st, grid, hc, tm).dtheta_prime_dt.data)
    np.testing.assert_allclose(d1, d0, rtol=0.0, atol=1e-14)


def test_sst_override_changes_surface_radiation():
    """A SST that differs from the air T changes the near-surface radiative
    tendency — the surface boundary (and hence the LW emission) tracks the SST,
    not the drifting air T (RAD-7)."""
    st, grid, hc, tm, t_air = _state_and_grid()
    cfg = _gray_cfg()
    rad_default = make_radiation_physics(cfg, model_type="plane")
    rad_warm = rcp._plane_radiation_physics_with_sst(cfg, grid, t_air + 5.0)
    d0 = np.asarray(rad_default(st, grid, hc, tm).dtheta_prime_dt.data)
    dw = np.asarray(rad_warm(st, grid, hc, tm).dtheta_prime_dt.data)
    # near-surface tendency must change when the surface T is bumped +5 K
    assert np.abs(dw[..., -3:] - d0[..., -3:]).max() > 1e-7


def test_make_rcemip_physics_pins_sst():
    """Integration: the full RCEMIP physics builder routes radiation through the
    SST-pinned helper (vs the air-T default), so a fixed-SST run's surface
    radiation is SAM-faithful."""
    st, grid, hc, tm, t_air = _state_and_grid()
    cfg = _gray_cfg()
    # build with a SST far from the rest-state air T → radiation must differ
    phys_sst = rcp.make_rcemip_physics(
        grid, hc, tm, cfg, None, dt=5.0, surface_flux=False, T_sfc=290.0)
    # reference: radiation with NO override (default air T)
    rad_default = make_radiation_physics(cfg, model_type="plane")
    d_sst = np.asarray(phys_sst(st, grid, hc, tm).dtheta_prime_dt.data)
    d_def = np.asarray(rad_default(st, grid, hc, tm).dtheta_prime_dt.data)
    assert np.abs(d_sst[..., -3:] - d_def[..., -3:]).max() > 1e-7


def test_gate_insolation_uses_latitude_daily_mean():
    """RAD-8: the GATE latitude daily-mean insolation at 8.5°N (~428 W/m²) is the
    case-appropriate value, distinct from the RCEMIP-protocol insolation. The
    old GATE default (RCEMIP S_0·cosθ ≈ 409.5) was a different case's value."""
    from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
    ins_85 = float(
        daily_mean_insolation(jnp.asarray([np.deg2rad(8.5)]), 80.0).reshape(-1)[0])
    # 8.5°N daily-mean ≈ 428 W/m² (equinox); the equator is higher
    np.testing.assert_allclose(ins_85, 428.5, rtol=2e-2)
    # distinctly above the RCEMIP-effective insolation (551.58·cos42° ≈ 409.5)
    assert ins_85 > 420.0


def test_gate_script_sets_gate_latitude_and_insolation():
    """RAD-8 guard: the GATE driver must place the plane at 8.5°N and use the
    latitude daily-mean insolation ('off'), not the RCEMIP preset."""
    repo = Path(__file__).resolve().parents[2]
    src = (repo / "scripts" / "run" / "run_gate_plane.py").read_text()
    assert "lat0=8.5" in src, "GATE grid must be at 8.5°N (GATE_IDEAL latitude)"
    assert 'insolation="off"' in src, "GATE must use latitude daily-mean insol"


def test_gate_config_resolves_insolation_and_no_coriolis():
    """codex iter-49 G+E: the GATE config (lat0=8.5, insolation='off') must
    RESOLVE the TOA insolation to the 8.5°N daily-mean (~428.5), NOT the
    RCEMIP-effective ~409.5 — and lat0=8.5 must NOT enable Coriolis
    (coriolis_mode='none' ⇒ f=0; lat0 feeds ONLY the radiation)."""
    from legoesm.atmosphere.physics._shared import grid_lat_lon
    from legoesm.atmosphere.physics.radiation.integration import _compute_insolation
    from legoesm.grids.plane import create_plane_grid
    grid = create_plane_grid(nx=8, ny=8, nlev=24, dx=1_000.0, dy=1_000.0,
                             coriolis_mode="none", lat0=8.5, dtype=jnp.float64)
    # E: lat0 does not enable Coriolis
    assert float(grid.f0) == 0.0
    assert float(jnp.max(jnp.abs(jnp.asarray(grid.grid_coriolis)))) == 0.0
    # G: the resolved insolation = 8.5°N daily-mean (same path for gray/rrtmgp)
    lat, lon = grid_lat_lon(grid, (8, 8))
    cfg = rcp._build_radiation_config(
        "gray", update_interval_steps=1, clouds=True,
        insolation="off", t_sfc=300.0)
    out = _compute_insolation(lat, cfg, lon=lon, day_of_year=80.0,
                              seconds_of_day=43200.0)
    ins = float(np.asarray(out[0]).reshape(-1).mean())
    np.testing.assert_allclose(ins, 428.5, rtol=2e-2)
    assert ins > 420.0   # above the RCEMIP-effective ~409.5


def test_sam_faithful_crm_drivers_pin_sst():
    """codex iter-48 F: guard the SAM-faithful CRM drivers against silently
    reverting to the air-T default — they must pin the surface radiative T to
    the SST (run_rcemip_plane via the helper, run_rce_mpi_long via the override
    hook). GATE/RCE/RCE-MPI are the SAM-comparison targets; a regression here
    un-faithfuls the surface LW boundary."""
    repo = Path(__file__).resolve().parents[2]
    plane = (repo / "scripts" / "run" / "run_rcemip_plane.py").read_text()
    # the single-node path routes both builders through the SST helper
    assert "_plane_radiation_physics_with_sst" in plane
    assert plane.count("_plane_radiation_physics_with_sst(") >= 3  # def + 2 uses
    mpi = (repo / "scripts" / "run" / "run_rce_mpi_long.py").read_text()
    assert "set_T_sfc_override" in mpi and "T_SFC_K" in mpi


def test_dx_aware_hyperdiff_scaling():
    """HYPERDIFF (iter-54): K = 1e8·(dx/1000)⁴ — dx=1000 ⇒ 1e8, finer dx scales
    down (biharmonic CFL + 2Δx rate ∝ K/dx⁴, so a fixed K over-damps at finer dx)."""
    assert rcp.dx_aware_hyperdiff(1000.0) == pytest.approx(1.0e8)
    assert rcp.dx_aware_hyperdiff(500.0) == pytest.approx(6.25e6)
    assert rcp.dx_aware_hyperdiff(100.0) == pytest.approx(1.0e4)


def test_cfl_guard_warns_at_fine_dx_not_smoke():
    """codex iter-54 D: cfl_guard warns on the horizontal acoustic CFL at fine
    dx (the fine-dx limiter), but is silent at the dx=1000 m / dt=2 s smoke."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rcp.cfl_guard(2.0, 1000.0, 50.0, label="t")     # smoke config
    assert len(w) == 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rcp.cfl_guard(1.0, 100.0, 50.0, label="t")       # dx=100 m fine config
    assert any("acoustic CFL" in str(x.message) for x in w)
