"""FV3_3D iter 448: NH fast Rayleigh friction (FV3 ``Ray_fast``).

Ports FV3 ``dyn_core.F90:2922-3020`` column Rayleigh damping
at model top.  Applied to u, v, w at end of ``step()``.

Tests
-----

1. ``test_default_rf_tau_zero`` — config defaults preserve
   bit-for-bit baseline.
2. ``test_default_rf_cutoff_3000`` — cutoff defaults to 3000 Pa.
3. ``test_baseline_equals_no_rf`` — explicit rf_tau=0 matches
   default.
4. ``test_rf_damps_top_winds`` — non-zero rf_tau reduces |u|
   at top level.
5. ``test_rf_no_effect_below_cutoff`` — levels with pfull >
   rf_cutoff are bit-for-bit unchanged.
6. ``test_rff_profile_at_cutoff_is_one`` — helper returns
   rff=1.0 at exact cutoff pressure (no damping at boundary).
7. ``test_rff_profile_below_cutoff_lt_one`` — rff < 1.0 at
   p << rf_cutoff (damping active).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.core.fv3_rayleigh_fast import (
    compute_rff_profile, pfull_from_exner,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _build_c8():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=False)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    rng = np.random.default_rng(seed=448)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev + 1))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_p), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d,
                          units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d,
                        units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, height_coord, terrain_metric, state


def test_default_rf_tau_zero():
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.rf_tau_days == 0.0


def test_default_rf_cutoff_3000():
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.rf_cutoff_pa == 3000.0


def test_baseline_equals_no_rf():
    grid, hc, tm, state = _build_c8()
    cfg_d = CDGridCompressibleEulerConfig(n_acoustic_substeps=4)
    cfg_z = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4, rf_tau_days=0.0,
    )
    m_d = CDGridCompressibleEulerModel(grid, hc, tm, cfg_d)
    m_z = CDGridCompressibleEulerModel(grid, hc, tm, cfg_z)
    s_d = m_d.step(state, dt=10.0)
    s_z = m_z.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_d.u.data), np.asarray(s_z.u.data),
        rtol=1e-14, atol=1e-14,
    )


def test_rff_profile_at_cutoff_is_one():
    pfull = jnp.array([3000.0, 5000.0, 10000.0])
    ptop = 100.0
    rff = compute_rff_profile(
        pfull, ptop=ptop, rf_cutoff=3000.0,
        tau_days=10.0, dt=100.0,
    )
    # At exactly rf_cutoff, no damping: rff = 1.
    np.testing.assert_allclose(
        np.asarray(rff[0]), 1.0, rtol=1e-12, atol=1e-12,
    )
    # Levels below cutoff also rff=1 (boundary handled below cutoff).
    np.testing.assert_allclose(
        np.asarray(rff[1:]), 1.0, rtol=1e-12, atol=1e-12,
    )


def test_rff_profile_below_cutoff_lt_one():
    """Levels with pfull << rf_cutoff get damped (rff < 1)."""
    pfull = jnp.array([100.0, 1000.0])
    ptop = 50.0
    rff = compute_rff_profile(
        pfull, ptop=ptop, rf_cutoff=3000.0,
        tau_days=10.0, dt=100.0,
    )
    rff_np = np.asarray(rff)
    assert rff_np[0] < 1.0, f"rff[0]={rff_np[0]} expected < 1"
    assert rff_np[1] < 1.0, f"rff[1]={rff_np[1]} expected < 1"
    # Top level (smaller p) more damped.
    assert rff_np[0] < rff_np[1]


def test_rf_damps_top_winds():
    grid, hc, tm, state = _build_c8()
    cfg_off = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4, rf_tau_days=0.0,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4, rf_tau_days=10.0, rf_cutoff_pa=1.0e5,
        # Use very high cutoff to ensure most levels get damped
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, dt=300.0)
    s_on = m_on.step(state, dt=300.0)
    # Top level |u| should be smaller with RF on.
    u_top_off = float(jnp.mean(jnp.abs(s_off.u.data[..., 0])))
    u_top_on = float(jnp.mean(jnp.abs(s_on.u.data[..., 0])))
    assert u_top_on < u_top_off, (
        f"RF did not damp top |u|: off={u_top_off:.3f}, "
        f"on={u_top_on:.3f}."
    )


def test_rf_no_effect_below_cutoff():
    """Levels with pfull > rf_cutoff are bit-for-bit unchanged."""
    grid, hc, tm, state = _build_c8()
    # rf_cutoff = 100 Pa → only TOP level (p ~ a few hPa) damped;
    # levels k=1..4 have pfull >> 100 Pa → no damping.
    cfg_on = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4, rf_tau_days=10.0, rf_cutoff_pa=100.0,
    )
    cfg_off = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4, rf_tau_days=0.0,
    )
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    s_on = m_on.step(state, dt=300.0)
    s_off = m_off.step(state, dt=300.0)
    # Bottom 2 levels (k=3, 4) high pressure, no RF effect.
    pfull = pfull_from_exner(hc.exner_ref)
    bottom_levs = np.where(np.asarray(pfull) >= 100.0)[0]
    if len(bottom_levs) > 0:
        np.testing.assert_allclose(
            np.asarray(s_on.u.data[..., bottom_levs]),
            np.asarray(s_off.u.data[..., bottom_levs]),
            rtol=1e-14, atol=1e-14,
        )
