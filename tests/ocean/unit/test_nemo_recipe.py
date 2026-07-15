"""Unit tests for the reusable NEMO ocean recipe card."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.fidelity.nemo_recipe import (
    NEMO_BLOCK_MAPPING,
    NEMO_CONSTANTS_CONFIG,
    NEMO_DEFERRED_BLOCKS,
    NEMOModelRecipeConfig,
    build_nemo_eady_recipe,
    build_nemo_recipe,
    build_nemo_rest_recipe,
    nemo_lat_lon_model_config,
)


def test_nemo_model_config_selects_canonical_blocks():
    cfg = nemo_lat_lon_model_config()

    assert cfg.constants == NEMO_CONSTANTS_CONFIG
    assert cfg.g == pytest.approx(constants.g_nemo)
    assert cfg.rho_0 == pytest.approx(constants.rho_ocean)
    assert cfg.constants.c_sw == pytest.approx(constants.c_p_seawater)

    assert cfg.eos == "veros_gsw"
    assert cfg.momentum_advection == "vector_invariant"
    assert cfg.ke_gradient_scheme == "hollingsworth"
    assert cfg.tracer_advection == "ppm_fct"
    assert cfg.pgf_scheme == "smc03"
    assert cfg.barotropic.barotropic_solver == "explicit_substep"
    assert cfg.barotropic.barotropic_time_filter == "cosine"
    assert cfg.momentum_time_integrator == "rk3"
    assert cfg.adaptive_implicit_vertadv is True
    assert cfg.implicit_vertical_mixing is True

    assert cfg.lateral_viscosity.A_h_lat_scaling is True
    assert cfg.lateral_viscosity.A_h_cos_power == 1
    assert cfg.lateral_viscosity.A_h_merid == pytest.approx(0.0)
    assert cfg.lateral_viscosity.A_h_eq_boost == pytest.approx(1.0)
    assert cfg.lateral_viscosity.A_h_eq_sigma_deg == pytest.approx(5.0)
    assert cfg.lateral_viscosity.C_smag_lap == pytest.approx(0.33)
    assert cfg.bottom_drag.bottom_drag_r == pytest.approx(2.5e-4)
    assert cfg.bottom_drag.bottom_drag_bg_velocity == pytest.approx(0.1)
    assert cfg.bottom_drag.bottom_drag_bbl_thickness == pytest.approx(100.0)
    assert cfg.normalize_freshwater is True
    assert cfg.runoff_depth_spread_m == pytest.approx(150.0)

    physics = cfg.physics
    assert physics.constants == NEMO_CONSTANTS_CONFIG
    assert physics.vertical_mixing.scheme == "tke"
    assert physics.lateral_mixing.scheme == "none"
    assert physics.surface_forcing.scheme == "none"
    assert physics.bottom_drag.scheme == "none"
    assert physics.convection.scheme == "none"
    assert physics.shortwave_penetration.scheme == "rgb_chl"
    assert physics.mle is not None

    tke = physics.vertical_mixing.tke
    assert tke is not None
    assert tke.prognostic is True
    assert tke.n2_mode == "adiabatic"
    assert tke.veros_dz_slots is True
    assert tke.positivity == "veros_surface_correction"
    assert tke.kappa_convention == "veros_sqrte"
    assert tke.buoyancy_timing == "post_mixing_veros"
    assert tke.shear_production == "realized_veros"
    assert tke.prandtl_mode == "richardson"
    # NEMO zdftke coefficient parity (TKE_FIDELITY_FINDINGS.md, dump-verified):
    # Ri-Prandtl slope 1/ri_cri = 4.5 (nn_pdl=1), background Kz rn_avm0/rn_avt0.
    assert tke.prandtl_ri_coeff == 4.5
    assert tke.kappaM_min == 1.2e-4
    assert tke.kappaH_min == 1.2e-5

    assert cfg.gm_redi is not None
    assert cfg.gm_redi.slope_scheme == "triads"
    assert cfg.gm_redi.slope_density == "neutral"
    assert cfg.gm_redi.implicit_K33 is True


def test_nemo_model_config_dispatches_upwind3_momentum_variant():
    cfg = nemo_lat_lon_model_config(
        NEMOModelRecipeConfig(momentum_core="flux_form_upwind3")
    )

    assert cfg.momentum_advection == "flux_form"
    assert cfg.momentum_flux_scheme == "upwind3"
    assert cfg.ke_gradient_scheme == "centered"


def test_nemo_model_config_rejects_unknown_high_level_dispatch():
    with pytest.raises(ValueError, match="momentum_core"):
        nemo_lat_lon_model_config(NEMOModelRecipeConfig(momentum_core="bogus"))


def test_nemo_card_builds_valid_latlon_model_and_is_setup_agnostic():
    rest = build_nemo_recipe(n_lat=6, n_lon=8, nlev=3)
    eady = build_nemo_eady_recipe(n_lat=12, n_lon=12, nlev=4)

    LatLonCGridOceanModel(rest.grid, rest.z_coord, rest.model_config)
    LatLonCGridOceanModel(eady.grid, eady.z_coord, eady.model_config)

    assert rest.model_config == eady.model_config
    assert rest.physics_config == rest.model_config.physics
    assert eady.physics_config == eady.model_config.physics
    assert rest.initial_state.T.data.shape != eady.initial_state.T.data.shape


def test_nemo_card_one_step_rest_sanity_is_finite():
    recipe = build_nemo_rest_recipe(n_lat=8, n_lon=12, nlev=4)
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, recipe.model_config)

    new_state = model.step(recipe.initial_state, dt=60.0)

    assert bool(jnp.all(jnp.isfinite(new_state.T.data)))
    assert bool(jnp.all(jnp.isfinite(new_state.u.data)))
    assert bool(jnp.all(jnp.isfinite(new_state.eta.data)))


def test_nemo_iso_lap_card_one_step_is_finite():
    """The nemo_iso_lap card steps finite through the full model — proves the
    card wiring reaches the build-7 operator (kappa_GM=0 guard, active_3d, the
    ramp+shapiro slopes) without error, not just that the config is built."""
    recipe = build_nemo_rest_recipe(
        n_lat=8, n_lon=12, nlev=4,
        cfg=NEMOModelRecipeConfig(lateral_operator="nemo_iso_lap"))
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, recipe.model_config)
    new_state = model.step(recipe.initial_state, dt=60.0)
    assert bool(jnp.all(jnp.isfinite(new_state.T.data)))
    assert bool(jnp.all(jnp.isfinite(new_state.S.data)))
    assert bool(jnp.all(jnp.isfinite(new_state.u.data)))


def test_named_recipe_wrapper_dispatches_and_rejects_unknown_setup():
    rest = build_nemo_rest_recipe(n_lat=6, n_lon=8, nlev=3)
    eady = build_nemo_recipe(setup="eady", n_lat=12, n_lon=12, nlev=4)

    assert rest.model_config == build_nemo_recipe(n_lat=6, n_lon=8, nlev=3).model_config
    assert eady.model_config == rest.model_config
    with pytest.raises(ValueError, match="setup"):
        build_nemo_recipe(setup="bogus")


def test_nemo_mapping_and_deferred_blocks_are_explicit():
    assert any(
        name == "vorticity/PV flux" and "AL81/EEN" in selected
        for name, _, selected in NEMO_BLOCK_MAPPING
    )
    assert any(
        name == "UP3 momentum option" and "upwind3" in selected
        for name, _, selected in NEMO_BLOCK_MAPPING
    )
    assert any(
        name == "meridional viscosity scaling" and "cosine" in selected
        for name, _, selected in NEMO_BLOCK_MAPPING
    )
    assert any(
        name == "equatorial viscosity boost" and "inactive by default" in selected
        for name, _, selected in NEMO_BLOCK_MAPPING
    )
    assert any("leapfrog" in item for item in NEMO_DEFERRED_BLOCKS)


def test_nemo_gyre_momentum_core_selects_ene_c2():
    """A GYRE_BARE-faithful card config selects NEMO's actual GYRE schemes.

    GYRE_BARE runs ln_dynvor_ene (ENE vorticity) + nn_dynkeg=0 (c2 KE) + EOS-80 +
    adcroft PGF — NOT the ORCA-style een/hollingsworth/veros_gsw/smc03 the card
    defaults to. The assembled-single-step audit had to override these by hand;
    this pins that the card CAN express GYRE (momentum_core='vector_invariant_ene'
    + eos/pgf knobs) so the whole GYRE dynamical core is reachable via the card.
    """
    gyre = nemo_lat_lon_model_config(NEMOModelRecipeConfig(
        momentum_core="vector_invariant_ene",
        eos="nemo_eos80",
        pgf_scheme="adcroft",
        lateral_operator="nemo_iso_lap",
    ))
    assert gyre.vorticity_scheme == "ene"
    assert gyre.ke_gradient_scheme == "c2"
    assert gyre.eos == "nemo_eos80"
    assert gyre.pgf_scheme == "adcroft"
    assert gyre.gm_redi.slope_scheme == "nemo_iso_lap"
    # default card stays ORCA-style (een/hollingsworth), unchanged
    d = nemo_lat_lon_model_config()
    assert d.ke_gradient_scheme == "hollingsworth"


def test_nemo_lateral_operator_selects_iso_lap_pure_redi():
    """lateral_operator='nemo_iso_lap' selects NEMO traldf_iso, pure Redi.

    GYRE runs ln_traldf_triad=F (standard rotated-Laplacian) + ln_ldfeiv=F (no
    GM), so the card option must select slope_scheme='nemo_iso_lap', force
    kappa_GM=0 (the operator raises otherwise), and turn on the NEMO ldfslp slope
    fidelity (ML ramp + Shapiro). Default stays the GM-on triad scheme.
    """
    # default: GM-on triads, unchanged
    d = nemo_lat_lon_model_config()
    assert d.gm_redi.slope_scheme == "triads"
    assert d.gm_redi.kappa_GM == 600.0
    # nemo_iso_lap: standard operator, pure Redi, NEMO slope fidelity on
    n = nemo_lat_lon_model_config(
        NEMOModelRecipeConfig(lateral_operator="nemo_iso_lap"))
    assert n.gm_redi.slope_scheme == "nemo_iso_lap"
    assert n.gm_redi.kappa_GM == 0.0
    assert n.gm_redi.nemo_mld_slope_ramp is True
    assert n.gm_redi.nemo_slope_shapiro is True
    # unknown selector raises (dispatch hardening)
    with pytest.raises(ValueError, match="lateral_operator"):
        nemo_lat_lon_model_config(
            NEMOModelRecipeConfig(lateral_operator="bogus"))


def test_nemo_tke_prandtl_bit_reproduces_nemo_pdl():
    """The card's richardson-Prandtl (coeff 4.5) is NEMO nn_pdl=1 exactly.

    NEMO zdftke: pdlr = MAX(0.1, ri_cri/MAX(ri_cri, Ri)) with
    ri_cri = 2/(2 + rn_ediss/rn_ediff) = 2/(2 + 0.7/0.1) = 2/9, so the Prandtl
    number Pr = 1/pdlr = MIN(10, MAX(1, Ri/ri_cri)) = MIN(10, MAX(1, 4.5·Ri)).
    legoESM `_prandtl_number` (richardson) returns MAX(1, MIN(10, coeff·Ri));
    clamp-to-[1,10] is order-independent, so coeff = 1/ri_cri = 4.5 must match
    NEMO's Pr to machine precision across the whole Ri range (incl. the Ri<0
    convective branch where both give Pr=1).
    """
    import numpy as np

    from legoesm.ocean.physics.vertical_mixing.tke import _prandtl_number
    from legoesm.ocean.fidelity.nemo_recipe import _nemo_tke_config

    cfg = _nemo_tke_config()
    assert cfg.prandtl_ri_coeff == 4.5
    ri_cri = 2.0 / (2.0 + 0.7 / 0.1)          # NEMO 2/9
    # _prandtl_number forms Ri = N2 / max(shear_sq, 1e-12); drive Ri via N2 with
    # unit shear (kappaM is unused in the richardson branch).
    Ri = np.linspace(-2.0, 20.0, 2001)
    N2 = jnp.asarray(Ri)
    shear_sq = jnp.ones_like(N2)
    pr_lego = np.asarray(_prandtl_number(N2, shear_sq, jnp.ones_like(N2), cfg))
    pdlr_nemo = np.maximum(0.1, ri_cri / np.maximum(ri_cri, Ri))
    pr_nemo = 1.0 / pdlr_nemo                  # NEMO Pr = 1/pdlr
    np.testing.assert_allclose(pr_lego, pr_nemo, rtol=0, atol=1e-12)


def test_nemo_recipe_is_lazy_registered():
    import legoesm.ocean.fidelity as fidelity

    assert "nemo_recipe" in fidelity.__all__
    assert fidelity.nemo_recipe.nemo_lat_lon_model_config is nemo_lat_lon_model_config
