"""momentum_only baroclinic tendency: bit-identical du/dv, skips tracer diffusion.

The RK3 momentum sub-stages (``_mom_pert``) freeze T/S/eta and discard the
tracer tendency, so ``tendencies(momentum_only=True)`` skips the (recomputed-
identically) tracer-diffusion stage.  Correctness contract (codex halo-hunt
#3 / task #24):

  * du_dt / dv_dt are BIT-IDENTICAL to the full path — the momentum stages
    never read dT_dt/dS_dt (they read the T/S STATE, which is unchanged).
  * dT_dt / dS_dt DIFFER (the full path's tracer diffusion is skipped) — proves
    the flag is non-vacuous (it actually elides work).

A wrong guard (one that perturbed momentum) would break the first assert; a
no-op guard would break the second.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

_N_LAT, _N_LON, _NZ = 12, 24, 4
_DT = 600.0


def _model_and_state():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid = create_latlon_grid(_N_LAT, _N_LON)
    z_coord = create_ocean_z_star(n_levels=_NZ, H_max=4000.0)
    # Non-trivial lateral + vertical diffusion so the skipped tracer stage is
    # genuinely non-zero (non-vacuity).
    cfg = LatLonCGridOceanConfig(
        momentum_time_integrator="rk3",
        K_h=500.0,
    )
    state = rest_state_latlon_cgrid_ocean(grid, z_coord)
    # Perturb T/S/u/v so du/dv and the tracer-diffusion tendency are non-zero.
    rng = np.random.default_rng(3)
    T = state.T.data + jnp.asarray(0.5 * rng.standard_normal(state.T.data.shape))
    S = state.S.data + jnp.asarray(0.1 * rng.standard_normal(state.S.data.shape))
    u = state.u.data + jnp.asarray(0.05 * rng.standard_normal(state.u.data.shape))
    v = state.v.data + jnp.asarray(0.05 * rng.standard_normal(state.v.data.shape))
    # Perturb directly; the tendency fn applies its own face/land masks, so
    # un-masked perturbation noise is masked identically on both paths.
    state = state._replace(
        T=state.T.replace(data=T),
        S=state.S.replace(data=S),
        u=state.u.replace(data=u),
        v=state.v.replace(data=v),
    )
    return LatLonCGridOceanModel(grid, z_coord, cfg), state


def test_momentum_only_bit_identical_du_dv():
    model, state = _model_and_state()
    full = model.tendencies(state, surface_forcing=None, dt=_DT,
                            momentum_only=False)
    mom = model.tendencies(state, surface_forcing=None, dt=_DT,
                           momentum_only=True)

    du_f = np.asarray(full.du_dt.data)
    du_m = np.asarray(mom.du_dt.data)
    dv_f = np.asarray(full.dv_dt.data)
    dv_m = np.asarray(mom.dv_dt.data)
    # BIT-identical momentum (the whole correctness contract).
    assert np.array_equal(du_f, du_m), (
        f"du_dt not bit-identical: max|diff|={np.max(np.abs(du_f - du_m)):.3e}")
    assert np.array_equal(dv_f, dv_m), (
        f"dv_dt not bit-identical: max|diff|={np.max(np.abs(dv_f - dv_m)):.3e}")

    # NON-VACUITY: the tracer-diffusion tendency IS skipped (so dT_dt differs).
    dT_f = np.asarray(full.dT_dt.data)
    dT_m = np.asarray(mom.dT_dt.data)
    assert not np.array_equal(dT_f, dT_m), (
        "dT_dt identical -> momentum_only did not actually skip tracer "
        "diffusion (vacuous flag)")
    assert float(np.max(np.abs(dT_f))) > 0.0, "full dT_dt is all-zero (test setup)"
