"""SIMTOP TOPMODEL runoff (Niu 2005 / CLM4.5) — sub-grid saturated fraction +
topographic baseflow, exact column-water conservation, and the opt-in dispatch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land.topmodel_runoff import (
    TopmodelConfig,
    water_table_depth,
    saturated_area_fraction,
    baseflow,
    partition_topmodel_runoff,
)

_CFG = TopmodelConfig()
_W_MAX = 150.0
_DT = 1800.0
_KINF = 1.0e-5
_SUCTION = 2.0


def _partition(W, P):
    return partition_topmodel_runoff(
        jnp.asarray(W), jnp.asarray(P), jnp.asarray(1e-5), _DT,
        _W_MAX, _KINF, _SUCTION, _CFG)


def test_water_table_deeper_when_drier():
    z_dry = float(water_table_depth(jnp.asarray(10.0), _W_MAX, _CFG))
    z_wet = float(water_table_depth(jnp.asarray(140.0), _W_MAX, _CFG))
    assert z_dry > z_wet >= 0.0
    assert z_dry <= _CFG.z_wt_max + 1e-9


def test_fsat_and_baseflow_decay_with_depth():
    """Both the saturated fraction and baseflow fall off exponentially as the
    water table deepens (drier soil)."""
    fs = [float(saturated_area_fraction(jnp.asarray(z), _CFG))
          for z in (0.0, 1.0, 3.0)]
    qb = [float(baseflow(jnp.asarray(z), _CFG)) for z in (0.0, 1.0, 3.0)]
    assert fs[0] > fs[1] > fs[2] >= 0.0
    assert qb[0] > qb[1] > qb[2] >= 0.0
    assert fs[0] <= _CFG.f_max + 1e-12


def test_column_water_conserved_exactly():
    """P_input == dW/dt + evap_actual + runoff to machine precision."""
    for W, P in [(30.0, 1e-4), (120.0, 5e-5), (149.0, 2e-4), (2.0, 0.0)]:
        W_new, evap_actual, runoff, r_surf, r_base = _partition(W, P)
        residual = float(P) - (
            float(W_new - W) / _DT + float(evap_actual) + float(runoff))
        assert abs(residual) < 1e-9, f"W={W} P={P}: residual {residual}"
        np.testing.assert_allclose(
            float(runoff), float(r_surf + r_base), rtol=1e-9, atol=1e-12)


def test_wetter_column_runs_off_more():
    """A wetter column (higher f_sat) sheds more surface runoff for the same
    rain — the sub-grid saturation-excess mechanism the bucket lacks."""
    _, _, _, r_surf_wet, _ = _partition(145.0, 1e-4)
    _, _, _, r_surf_dry, _ = _partition(20.0, 1e-4)
    assert float(r_surf_wet) > float(r_surf_dry)


def test_baseflow_drains_unsaturated_column():
    """TOPMODEL produces baseflow even when the column is NOT full (the bucket
    only spills at saturation) — but never drives storage negative."""
    W_new, _, _, _, r_base = _partition(80.0, 0.0)   # no rain, mid-moisture
    assert float(r_base) > 0.0
    assert 0.0 <= float(W_new) <= _W_MAX


def test_storage_bounds_and_nonneg_runoff():
    for W, P in [(0.5, 0.0), (150.0, 1e-3), (75.0, 1e-6)]:
        W_new, _, runoff, r_surf, r_base = _partition(W, P)
        assert 0.0 <= float(W_new) <= _W_MAX
        assert float(runoff) >= 0.0 and float(r_surf) >= 0.0 and float(r_base) >= 0.0


def test_differentiable():
    def loss(P):
        _, _, runoff, _, _ = _partition(80.0, P)
        return jnp.sum(runoff ** 2)
    g = jax.grad(loss)(jnp.asarray(1e-4))
    assert jnp.isfinite(g) and float(g) >= 0.0


def _slab_state(ncol, W_init):
    from legoesm.core.field import Field
    from legoesm.land.state import LandState
    return LandState(
        T_soil=Field(jnp.full(ncol, 285.0), name="T_soil"),
        W_bucket=Field(jnp.full(ncol, W_init), name="W_bucket"),
        snow_depth=Field(jnp.zeros(ncol), name="snow_depth"),
        snow_age=Field(jnp.zeros(ncol), name="snow_age"),
    )


def _forcing(ncol):
    from legoesm.core.coupling_fields import AtmToSurface
    o = jnp.ones(ncol)
    d = dict(sw_down=250.0, lw_down=320.0, precip_total=2e-4, precip_snow=0.0,
             T_lowest=288.0, q_lowest=6e-3, u_lowest=4.0, v_lowest=1.0,
             p_lowest=1e5, p_surface=1.013e5, rho_lowest=1.2, cos_zenith=0.7,
             co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0)
    return AtmToSurface(**{k: v * o for k, v in d.items()})


def test_step_land_dispatch_default_topmodel_and_raise():
    """step_land: default 'bucket' is byte-identical; 'topmodel' routes to the
    SIMTOP path (different runoff); an unknown runoff_scheme raises."""
    from legoesm.land.config import LandConfig
    from legoesm.land.slab_land import step_land

    ncol = 4
    dt = 3600.0
    assert LandConfig().runoff_scheme == "bucket"

    st, forc = _slab_state(ncol, 130.0), _forcing(ncol)
    st_bucket, _, _ = step_land(st, forc, LandConfig(), U_min=1.0, dt=dt)
    st_topo, _, _ = step_land(
        st, forc, LandConfig(runoff_scheme="topmodel"), U_min=1.0, dt=dt)
    # A wet column under rain: TOPMODEL's sub-grid saturated fraction + baseflow
    # drain the bucket differently than the bucket's whole-cell spill, so the
    # updated storage differs -> the dispatch genuinely routes.
    assert not np.allclose(np.asarray(st_bucket.W_bucket.data),
                           np.asarray(st_topo.W_bucket.data))

    with pytest.raises(ValueError, match="runoff_scheme"):
        step_land(st, forc, LandConfig(runoff_scheme="xinanjiang"),
                  U_min=1.0, dt=dt)
