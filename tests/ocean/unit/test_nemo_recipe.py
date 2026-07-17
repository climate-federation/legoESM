"""Unit tests for the reusable NEMO ocean recipe card."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import pytest
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.fidelity.nemo_recipe import (
    _NEMO_GYRE_DT_S,
    NEMO_BLOCK_MAPPING,
    NEMO_CONSTANTS_CONFIG,
    NEMO_DEFERRED_BLOCKS,
    NEMOModelRecipeConfig,
    _nemo_gyre_vertical_ladder,
    apply_nemo_gyre_surface_forcing,
    build_nemo_eady_recipe,
    build_nemo_gyre_recipe,
    build_nemo_recipe,
    build_nemo_rest_recipe,
    nemo_gyre_initial_T_S,
    nemo_lat_lon_model_config,
)

from legoesm import constants


def test_nemo_model_config_selects_canonical_blocks():
    cfg = nemo_lat_lon_model_config()

    assert cfg.constants == NEMO_CONSTANTS_CONFIG
    assert cfg.g == pytest.approx(constants.g_nemo)
    assert cfg.rho_0 == pytest.approx(constants.rho_ocean_nemo)  # NEMO rau0=1026
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
    from legoesm.ocean.fidelity.nemo_recipe import _nemo_tke_config
    from legoesm.ocean.physics.vertical_mixing.tke import _prandtl_number

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


def test_nemo_tke_deep_floor_is_constant_avtb_not_bryan_lewis():
    """The NEMO recipe TKE floors the quiescent deep at the constant avtb
    (kappaH_min=1.2e-5), NOT the Veros Bryan-Lewis abyssal depth profile.

    NEMO GYRE runs ln_zdfcst=F with a CONSTANT background avtb; the legoESM
    default enable_kappaH_profile=True would floor the deep K_H at the Bryan-Lewis
    profile (~1e-4 at 3000 m, ~8x avtb), masking the independent-kappaH_min
    flooring fix. This guards the abyssal fix at the SHIPPED config (Prandtl + BL
    off), which the isolated tke test can't (it sets BL off by hand).
    """
    import numpy as np
    from legoesm.ocean.fidelity.nemo_recipe import _nemo_tke_config
    from legoesm.ocean.physics.vertical_mixing.tke import compute_K_from_tke

    cfg = _nemo_tke_config()
    assert cfg.enable_kappaH_profile is False
    # Quiescent deep cell (raw K = c_k*l*sqrt(e) = 1e-5 << kappaM_min) at 3000 m,
    # where Bryan-Lewis (if on) would floor K_H ~1e-4. Ri=0.86 -> Pr=3.87.
    e = jnp.array([1.0e-6]); l_k = jnp.array([0.1])
    N2 = jnp.array([0.86]); shear_sq = jnp.array([1.0])
    z_int = jnp.array([-3000.0])
    _K_M, K_H = compute_K_from_tke(
        e, l_k, cfg, N2=N2, shear_sq=shear_sq, z_interface=z_int)
    np.testing.assert_allclose(float(K_H[0]), 1.2e-5, rtol=1e-6)  # = avtb, not BL
    # non-vacuity: the same cell WITH Bryan-Lewis on floors far higher (~1e-4),
    # so BL-off is what makes the deep match NEMO's constant avtb.
    _KM_bl, K_H_bl = compute_K_from_tke(
        e, l_k, cfg._replace(enable_kappaH_profile=True),
        N2=N2, shear_sq=shear_sq, z_interface=z_int)
    assert float(K_H_bl[0]) > 5.0 * 1.2e-5


def test_nemo_gyre_native_builds_valid_model_and_dispatch():
    """The native GYRE_BARE setup builds a valid LatLonCGridOceanModel and is
    reachable through build_nemo_recipe(setup='gyre') (unknown setup still raises)."""
    recipe = build_nemo_gyre_recipe()
    # grid: 32 x 22 T-cells (kpi=30+2, kpj=20+2), 30 wet z-levels.
    assert (recipe.grid.n_lat, recipe.grid.n_lon) == (22, 32)
    assert recipe.z_coord.n_levels == 30
    assert recipe.initial_state.T.data.shape == (22, 32, 30)
    LatLonCGridOceanModel(recipe.grid, recipe.z_coord, recipe.model_config)

    # dispatch parity + unknown setup still hard-errors.
    assert build_nemo_recipe(setup="gyre").model_config == recipe.model_config
    with pytest.raises(ValueError, match="setup"):
        build_nemo_recipe(setup="bogus")


def test_nemo_gyre_coordinate_is_consistent_clean_w_bc():
    """H_max == H_bathy (30 wet levels only) → sigma[nlev] ≈ 0 → w = 0 at the sea
    floor by construction — the clean-w-BC the bridge lacks (FCT2_BOTTOM_DRIFT).

    The bridge builds z_coord.H_max from the FULL 31-level ladder (4601.81 m) but
    H_bathy from 30 wet levels (4300.71 m), leaving sigma_seafloor ≈ 0.065 and a
    spurious w. The native setup has H_max == sum(30 wet e3t) == H_bathy.
    """
    import numpy as np

    recipe = build_nemo_gyre_recipe()
    z = recipe.z_coord
    H_bathy = float(jnp.max(recipe.initial_state.H_bathy.data))
    # consistent coordinate: H_max == wet H_bathy (float32-storage tol on H_bathy).
    assert z.H_max == pytest.approx(4300.710017, abs=1e-3)
    assert z.H_max == pytest.approx(H_bathy, rel=1e-6)

    # sea-floor sigma of diagnose_w's full-cell z* branch:
    # sigma = (z_half_ref[-1] + H_max)/H_max — ~0 for the native (consistent)
    # coordinate, but ~0.065 for the bridge-style inconsistent one.
    sigma_native = (float(z.z_half_ref[-1]) + z.H_max) / z.H_max
    H_max_full = 4601.808643654613          # bridge: 31-level ladder sum
    sigma_bridge = (-4300.710017215397 + H_max_full) / H_max_full
    assert abs(sigma_native) < 1e-6         # clean w-BC by construction
    assert sigma_bridge > 0.06              # non-vacuity: the bug it avoids
    assert abs(sigma_native) < 1e-4 * sigma_bridge

    # And the diagnosed w is exactly 0 at the deepest interface after a step
    # (at rest deta=0; the structural sigma above is the load-bearing guarantee).
    model = LatLonCGridOceanModel(recipe.grid, z, recipe.model_config)
    new = model.step(recipe.initial_state, dt=_NEMO_GYRE_DT_S)
    w = np.asarray(new.w.data)
    assert np.max(np.abs(w[:, :, -1])) == 0.0


def test_nemo_gyre_forced_trajectory_is_finite_and_stable():
    """A runnable forced step (model.step + the post-step thermal applicator +
    the step-level wind) is finite and physically bounded — the whole point of
    the native assembly, exercising the documented run loop VERBATIM."""
    from legoesm.ocean.fidelity.nemo_recipe import nemo_gyre_wind_forcing

    recipe = build_nemo_gyre_recipe()
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, recipe.model_config)
    st = recipe.initial_state
    n_lat, n_lon = st.T.data.shape[0], st.T.data.shape[1]
    for i in range(3):
        t = i * _NEMO_GYRE_DT_S
        st = apply_nemo_gyre_surface_forcing(
            st, recipe.z_coord, _NEMO_GYRE_DT_S, t_seconds=t)
        st = model.step(
            st, dt=_NEMO_GYRE_DT_S,
            surface_forcing=nemo_gyre_wind_forcing(n_lat, n_lon, t_seconds=t))
    for f in (st.T.data, st.S.data, st.u.data, st.v.data, st.eta.data):
        assert bool(jnp.all(jnp.isfinite(f)))
    # gently-forced GYRE spin-up stays laminar (no barotropic blow-up).
    assert float(jnp.max(jnp.abs(st.u.data))) < 1.0
    assert float(jnp.max(jnp.abs(st.eta.data))) < 1.0
    # the thermal applicator actually forces the surface (non-vacuous).
    assert float(jnp.max(jnp.abs(st.T.data - recipe.initial_state.T.data))) > 1e-3
    # the WIND actually forces the momentum (non-vacuous vs a windless run).
    st_nw = recipe.initial_state
    for i in range(3):
        t = i * _NEMO_GYRE_DT_S
        st_nw = apply_nemo_gyre_surface_forcing(
            st_nw, recipe.z_coord, _NEMO_GYRE_DT_S, t_seconds=t)
        st_nw = model.step(st_nw, dt=_NEMO_GYRE_DT_S)
    assert float(jnp.max(jnp.abs(st.u.data - st_nw.u.data))) > 1e-4


def test_nemo_gyre_wind_forcing_sign_chain_end_to_end():
    """The ocean receives EXACTLY NEMO's utau: nemo_gyre_wind returns stress ON
    THE OCEAN, the step-level object carries the ATMOSPHERIC convention (stage
    10b' applies -tau), and the builder negates — so one wind-only step must
    accelerate the top layer with the SIGN of utau (westward south of the
    sign-change latitude, i.e. du<0 where utau<0)."""
    import numpy as np

    from legoesm.ocean.fidelity.nemo_recipe import (
        nemo_gyre_latitudes,
        nemo_gyre_wind,
        nemo_gyre_wind_forcing,
    )

    recipe = build_nemo_gyre_recipe()
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, recipe.model_config)
    st = recipe.initial_state
    n_lat, n_lon = st.T.data.shape[0], st.T.data.shape[1]
    sf = nemo_gyre_wind_forcing(n_lat, n_lon, t_seconds=0.0)
    new = model.step(st, dt=_NEMO_GYRE_DT_S, surface_forcing=sf)
    du_top = np.asarray(new.u.data[:, :, 0] - st.u.data[:, :, 0])

    lat_t = np.asarray(nemo_gyre_latitudes(n_lat))
    utau, _ = nemo_gyre_wind(lat_t, 0.0)
    utau = np.asarray(utau)
    # interior WET rows (skip the 1-cell land rim rows 0 / n_lat-1, where the
    # mask zeroes du) with strongly negative / positive utau: the top-layer du
    # must carry utau's sign.  (The utau sign change sits at 29N ~ the southern
    # wall, so the negative branch lives in the NORTHERN interior rows.)
    interior = np.arange(1, n_lat - 1)
    j_neg = int(interior[np.argmin(utau[interior])])
    j_pos = int(interior[np.argmax(utau[interior])])
    assert utau[j_neg] < -1e-3 and utau[j_pos] > 1e-3
    row_neg = du_top[j_neg, 2:-2]
    row_pos = du_top[j_pos, 2:-2]
    assert float(np.mean(row_neg)) < 0.0, "utau<0 must decelerate u (westward)"
    assert float(np.mean(row_pos)) > 0.0, "utau>0 must accelerate u (eastward)"

    # Non-vacuity of the n_barotropic_substeps=120 override: the card default 30
    # blows up the barotropic external mode on this deep (H~4300 m) grid within a
    # few forced steps — so the =120 choice is provably load-bearing, not cosmetic.
    import dataclasses

    from legoesm.ocean.fidelity.nemo_recipe import _NEMO_GYRE_CARD_CONFIG

    r30 = build_nemo_gyre_recipe(
        cfg=dataclasses.replace(_NEMO_GYRE_CARD_CONFIG, n_barotropic_substeps=30))
    m30 = LatLonCGridOceanModel(r30.grid, r30.z_coord, r30.model_config)
    s30 = r30.initial_state
    for _ in range(3):
        s30 = apply_nemo_gyre_surface_forcing(s30, r30.z_coord, _NEMO_GYRE_DT_S)
        s30 = m30.step(s30, dt=_NEMO_GYRE_DT_S)
    assert not bool(jnp.all(jnp.isfinite(s30.u.data)))   # 30 substeps diverges


def test_nemo_gyre_initial_state_matches_nemo_tanh_profile():
    """The analytic IC equals NEMO usrdef_istate at NEMO's own T-point depths."""
    recipe = build_nemo_gyre_recipe()
    _e3t, gdept = _nemo_gyre_vertical_ladder()
    T = recipe.initial_state.T.data
    S = recipe.initial_state.S.data
    # interior wet column (10, 15); profile is horizontally uniform over ocean.
    for k in (0, 15, 29):
        T_k, S_k = nemo_gyre_initial_T_S(jnp.asarray(gdept[k]))
        assert float(T[10, 15, k]) == pytest.approx(float(T_k), abs=1e-6)
        assert float(S[10, 15, k]) == pytest.approx(float(S_k), abs=1e-6)
    # sanity vs NEMO GYRE profile: warm ~23°C surface, cold ~4°C abyss.
    assert 20.0 < float(T[10, 15, 0]) < 26.0
    assert 3.0 < float(T[10, 15, 29]) < 5.0


def test_nemo_gyre_recipe_selects_gyre_schemes():
    """The recipe's model_config carries NEMO GYRE's actual dynamical-core schemes
    (ENE vorticity, c2 KE, EOS-80, adcroft PGF, nemo_iso_lap pure Redi)."""
    mc = build_nemo_gyre_recipe().model_config
    assert mc.vorticity_scheme == "ene_total"  # NEMO np_CRV combined f+zeta
    assert mc.ke_gradient_scheme == "c2"
    assert mc.eos == "nemo_eos80"
    assert mc.pgf_scheme == "adcroft"
    assert mc.gm_redi.slope_scheme == "nemo_iso_lap"
    assert mc.gm_redi.kappa_GM == 0.0        # ln_ldfeiv=F (no GM bolus)
    assert mc.barotropic.n_barotropic_substeps == 50  # NEMO auto nn_e=50


def test_nemo_gyre_vertical_ladder_matches_nemo_mesh():
    """The analytic MI96 ladder reproduces NEMO's 30 wet e3t summing to H_bathy."""
    import numpy as np

    e3t, gdept = _nemo_gyre_vertical_ladder()
    assert e3t.shape == (30,) and gdept.shape == (30,)
    assert float(np.sum(e3t)) == pytest.approx(4300.710017, abs=1e-4)
    assert np.all(np.diff(gdept) > 0.0)      # strictly increasing T-depths
    assert float(gdept[0]) == pytest.approx(4.975265, abs=1e-4)   # NEMO gdept_1d[0]


def test_nemo_recipe_is_lazy_registered():
    import legoesm.ocean.fidelity as fidelity

    assert "nemo_recipe" in fidelity.__all__
    assert fidelity.nemo_recipe.nemo_lat_lon_model_config is nemo_lat_lon_model_config


def test_surface_stress_implicit_wiring():
    """NEMO dynzdf implicit wind-stress deposition (surface_stress_implicit):
    (a) requires nemo_stage_mean_imposition (init raises without it);
    (b) column-integrated momentum input identical to the explicit kick
    (no double-count through F_slow + the solve deposition);
    (c) non-vacuous (vertical distribution differs);
    (d) card has both flags on."""
    import jax.numpy as jnp
    import numpy as np
    import pytest

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import (
        _NEMO_GYRE_DT_S,
        build_nemo_gyre_recipe,
        nemo_gyre_wind_forcing,
    )

    r = build_nemo_gyre_recipe()
    assert r.model_config.surface_stress_implicit is True
    assert r.model_config.barotropic.nemo_stage_mean_imposition is True

    with pytest.raises(ValueError, match="nemo_stage_mean_imposition"):
        LatLonCGridOceanModel(
            r.grid, r.z_coord,
            r.model_config._replace(barotropic=r.model_config.barotropic
                                    ._replace(nemo_stage_mean_imposition=False)))

    st = r.initial_state
    n_lat, n_lon = st.T.data.shape[0], st.T.data.shape[1]
    sf = nemo_gyre_wind_forcing(n_lat, n_lon, 0.0)
    m_impl = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config)
    mc_expl = r.model_config._replace(
        surface_stress_implicit=False,
        barotropic=r.model_config.barotropic._replace(
            nemo_stage_mean_imposition=False))
    m_expl = LatLonCGridOceanModel(r.grid, r.z_coord, mc_expl)
    s_i = m_impl.step(st, dt=_NEMO_GYRE_DT_S, surface_forcing=sf)
    s_e = m_expl.step(st, dt=_NEMO_GYRE_DT_S, surface_forcing=sf)
    dz = np.asarray(r.z_coord.dz_ref)
    wet = np.asarray(st.u_mask.data) > 0
    Iu_i = np.sum(np.asarray(s_i.u.data) * dz, -1)
    Iu_e = np.sum(np.asarray(s_e.u.data) * dz, -1)
    # same column-integrated momentum input (f32 state => 1e-6 relative)
    ref = float(np.sqrt(np.mean(Iu_e[wet] ** 2)))
    np.testing.assert_allclose(Iu_i[wet], Iu_e[wet], atol=2e-6 * max(ref, 1.0))
    # different vertical distribution (the point of the change)
    assert float(np.max(np.abs(
        np.asarray(s_i.u.data)[..., 0] - np.asarray(s_e.u.data)[..., 0]))) > 1e-4
    for f in (s_i.u.data, s_i.v.data, s_i.T.data):
        assert bool(jnp.all(jnp.isfinite(f)))


def test_nemo_cap_slope_limit_centered_path():
    """NEMO ldfslp steep-slope convention on the centered/nemo_iso_lap path:
    slopes hard-capped at min(rn_slpmax, e3/7e3) with taper == 1 (the flux
    keeps diffusing along the capped direction); dm95 tapers toward zero."""
    import jax.numpy as jnp
    import numpy as np

    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        compute_isopycnal_slopes_latlon_cgrid,
    )

    r = build_nemo_gyre_recipe()
    cfg = r.model_config.gm_redi
    assert cfg.slope_limit == "nemo_cap"
    st = r.initial_state
    # a STEEP horizontal density front (large T gradient) to exceed the cap
    T = st.T.data + 5.0 * jnp.linspace(0, 1, st.T.data.shape[1])[None, :, None]
    from legoesm.ocean.eos import nemo_roquet_eos
    import numpy as _np
    gdept = _np.cumsum(_np.asarray(r.z_coord.dz_ref)) - 0.5 * _np.asarray(
        r.z_coord.dz_ref)
    p3 = jnp.asarray(1026.0 * 9.80665 * gdept)[None, None, :]
    eos_fn = lambda TT, SS, pp: nemo_roquet_eos(TT, SS, pp, rho0=1026.0)
    rho = eos_fn(T, st.S.data, p3)
    J = jnp.ones_like(st.eta.data)
    args = (rho, st.land_mask.data, r.z_coord, J, r.grid)
    # run both limiters on identical inputs
    Sx_c, Sy_c, tap_c = compute_isopycnal_slopes_latlon_cgrid(
        *args, cfg, T=T, S=st.S.data, eos_fn=eos_fn)
    Sx_d, Sy_d, tap_d = compute_isopycnal_slopes_latlon_cgrid(
        *args, cfg._replace(slope_limit="dm95_taper",
                            nemo_mld_slope_ramp=False,
                            nemo_slope_shapiro=False),
        T=T, S=st.S.data, eos_fn=eos_fn)
    dz = np.asarray(r.z_coord.dz_ref)
    cap = np.minimum(cfg.S_max, 0.5 * (dz[:-1] + dz[1:]) / 7.0e3)
    # capped path respects the DOUBLE cap everywhere (shapiro smooths within it)
    assert float(jnp.max(jnp.abs(Sx_c))) <= float(cap.max()) + 1e-8  # f32
    # near-surface interfaces are bound by the e3/7e3 cap, TIGHTER than S_max
    assert cap[0] < cfg.S_max
    # taper is exactly 1 on the capped path (flux never dies at steep fronts)
    np.testing.assert_array_equal(np.asarray(tap_c), 1.0)
    # non-vacuity: the two limiters genuinely differ on this front
    assert float(jnp.max(jnp.abs(Sx_c - Sx_d))) > 0.0


def test_nemo_iso_lap_slope_sign_convention():
    """SIGN GATE (CLAUDE.md sign mandate + the 2026-07-17 winter ttrd_ldf
    certificate): the slope PRODUCER emits S = +dx(rho)/|drho_dz| (GM
    convention, drho_dz floored negative); the nemo_iso_lap OPERATOR was
    certified consuming NEMO-convention slopes slp = -dx(rho)/|drho_dz|
    (ldfslp zau/(zbu<0)). The dispatch must NEGATE. Un-negated, the
    off-diagonal (subduction) fluxes run backward — on NEMO's Jan state the
    200-430 m band read -1.0e-7 K/s vs NEMO's +5.2e-8.

    Gate: dispatch(dT) == operator(-S_produced) exactly, and differs from
    operator(+S_produced) on a front state (non-vacuity)."""
    import jax.numpy as jnp
    import numpy as np

    from legoesm.ocean.eos import nemo_roquet_eos
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        compute_isopycnal_slopes_latlon_cgrid,
        gm_redi_tracer_tendency_latlon,
        nemo_iso_lap_tracer_tendency_latlon_cgrid,
    )

    r = build_nemo_gyre_recipe()
    st = r.initial_state
    cfg = r.model_config.gm_redi
    # meridional front on top of the stable IC stratification
    T = st.T.data + 2.0 * jnp.linspace(0, 1, st.T.data.shape[0])[:, None, None]
    S = st.S.data
    eta = jnp.zeros_like(st.eta.data)
    J = jnp.ones_like(st.eta.data)
    dT_disp, _ = gm_redi_tracer_tendency_latlon(
        T, S, eta, st.H_bathy.data, r.grid, r.z_coord, cfg,
        eos=r.model_config.eos, mask=st.land_mask.data,
        u_mask=st.u_mask.data, v_mask=st.v_mask.data, rho_0=1026.0)

    import numpy as _np
    gdept = _np.cumsum(_np.asarray(r.z_coord.dz_ref)) - 0.5 * _np.asarray(
        r.z_coord.dz_ref)
    p3 = jnp.asarray(1026.0 * 9.80665 * gdept)[None, None, :]
    eos_fn = lambda TT, SS, pp: nemo_roquet_eos(TT, SS, pp, rho0=1026.0)
    rho = eos_fn(T, S, p3)
    S_x, S_y, _ = compute_isopycnal_slopes_latlon_cgrid(
        rho, st.land_mask.data, r.z_coord, J, r.grid, cfg,
        T=T, S=S, eos_fn=eos_fn)
    _ztop = jnp.cumsum(r.z_coord.dz_ref) - r.z_coord.dz_ref
    act = ((st.land_mask.data[:, :, None] > 0.5)
           & (_ztop[None, None, :] < st.H_bathy.data[:, :, None])).astype(T.dtype)

    def op(sx, sy):
        return nemo_iso_lap_tracer_tendency_latlon_cgrid(
            T, sx, sy, st.land_mask.data, st.u_mask.data, st.v_mask.data,
            r.z_coord, J, r.grid, cfg.kappa_Redi, act)

    dT_neg = op(-S_x, -S_y)
    dT_pos = op(S_x, S_y)
    np.testing.assert_allclose(np.asarray(dT_disp), np.asarray(dT_neg), atol=1e-11)  # dispatcher builds rho internally; op-order roundoff
    assert float(jnp.max(jnp.abs(dT_neg - dT_pos))) > 0.0
