"""P-model state on the slab lane and through the patch mosaic.

Slab: with a P-model switch on and an initialised acclimation state the
two-leaf slab arm runs and advances the state; the switch without state raises
the PR1 loud error; defaults (switches off, state None) leave the new field
None and the step's outputs untouched.  Mosaic: a single unit patch with a
shared acclimation state reproduces the direct two-leaf call; switch-on
without state raises through the mosaic path too.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import LandConfig
from legoesm.land.p_model import PModelConfig, init_pmodel_acclim
from legoesm.land.state import LandState
from legoesm.land.surface_scheme import TwoLeafCanopyConfig


def _forcing(ncol, **overrides):
    ones = jnp.ones(ncol)
    defaults = dict(
        sw_down=400.0, lw_down=300.0, precip_total=1e-5, precip_snow=0.0,
        T_lowest=290.0, q_lowest=6e-3, u_lowest=3.0, v_lowest=1.0,
        p_lowest=1e5, p_surface=1.013e5, rho_lowest=1.2, cos_zenith=0.6,
        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0,
    )
    defaults.update(overrides)
    return AtmToSurface(**{k: v * ones for k, v in defaults.items()})


def _slab_state(ncol, acclim=None, T_init=290.0):
    return LandState(
        T_soil=Field(jnp.full(ncol, T_init), name="T_soil"),
        W_bucket=Field(jnp.full(ncol, 80.0), name="W_bucket"),
        snow_depth=Field(jnp.zeros(ncol), name="snow_depth"),
        snow_age=Field(jnp.zeros(ncol), name="snow_age"),
        TgC=jnp.full(ncol, 17.0),
        pmodel_acclim=acclim,
    )


def _pm_cfg(**kw):
    return LandConfig(surface_scheme=TwoLeafCanopyConfig(
        stomatal_model="medlyn", capacity_scheme="p_model",
        g1_source="p_model", **kw).validate())


def test_slab_two_leaf_pmodel_runs_and_advances_state():
    from legoesm.land.slab_land import step_land
    ncol = 3
    cfg = _pm_cfg()
    acclim = init_pmodel_acclim(
        ncol, t_init_K=290.0, ps_init_pa=101325.0, cfg=cfg.p_model)
    st = _slab_state(ncol, acclim=acclim)
    new_state, resp, _ = step_land(st, _forcing(ncol), cfg, U_min=1.0, dt=1800.0)
    assert new_state.pmodel_acclim is not None
    # daytime step: the gain-weighted means moved toward the forcing
    assert float(new_state.pmodel_acclim.t_mean_K[0]) != pytest.approx(
        float(st.pmodel_acclim.t_mean_K[0]))
    for leaf in new_state.pmodel_acclim:
        assert np.all(np.isfinite(np.asarray(leaf)))
    assert np.all(np.isfinite(np.asarray(resp.T_sfc)))


def test_slab_pmodel_switch_without_state_raises():
    from legoesm.land.slab_land import step_land
    st = _slab_state(2, acclim=None)
    with pytest.raises(ValueError, match="acclimation state"):
        step_land(st, _forcing(2), _pm_cfg(), U_min=1.0, dt=1800.0)


def test_slab_defaults_carry_no_pmodel_state():
    from legoesm.land.slab_land import step_land
    cfg = LandConfig(surface_scheme=TwoLeafCanopyConfig())
    st = _slab_state(2, acclim=None)
    new_state, _, _ = step_land(st, _forcing(2), cfg, U_min=1.0, dt=1800.0)
    assert new_state.pmodel_acclim is None


def test_mosaic_unit_patch_matches_direct_call_with_pmodel():
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.surface_scheme.patch_mosaic import (
        PatchMosaicConfig, PatchSpec, compute_mosaic_canopy_fluxes)
    from legoesm.land.surface_scheme.two_leaf_canopy import (
        compute_two_leaf_canopy_fluxes)

    land_cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(
        stomatal_model="medlyn", capacity_scheme="p_model",
        g1_source="p_model").validate())
    cc = land_cfg.surface_scheme
    acclim = init_pmodel_acclim(
        1, t_init_K=290.0, ps_init_pa=101325.0, cfg=land_cfg.p_model)
    forcing = _forcing(1)
    kw = dict(
        T_soil_top=jnp.full((1,), 289.0), forcing=forcing, canopy_config=cc,
        land_config=land_cfg, canopy_params=None,
        w_frac_rz=jnp.ones((1,)), wind_speed=jnp.full((1,), 3.0),
        wind_dir_x=jnp.ones((1,)), wind_dir_y=jnp.zeros((1,)),
        soil_thermal_fn=lambda G, dt_: jnp.full(jnp.shape(G), 289.0),
        dt=1800.0, pmodel_acclim=acclim)
    direct = compute_two_leaf_canopy_fluxes(**kw)
    mosaic = compute_mosaic_canopy_fluxes(
        mosaic=PatchMosaicConfig(patches=(PatchSpec(frac=1.0),)), **kw)
    np.testing.assert_allclose(
        np.asarray(mosaic.lh_flux), np.asarray(direct.lh_flux), rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(mosaic.gpp), np.asarray(direct.gpp), rtol=1e-6)

    with pytest.raises(ValueError, match="acclimation state"):
        compute_mosaic_canopy_fluxes(
            mosaic=PatchMosaicConfig(patches=(PatchSpec(frac=1.0),)),
            **{**kw, "pmodel_acclim": None})
