"""Direct unit tests for the DINO experiment module.

Exercises the leaf symbols of ``legoesm.ocean.experiments.dino`` —
config defaults, analytical bathymetry, wind-stress interpolation,
T*/S* targets, equatorial 1D profiles, top-layer tendencies, the
Lévy z* helper wrapper, and the lat-lon initial-state construction.
The MPAS path is exercised in a separate test that is skipped when
``scipy`` is unavailable, since regional Voronoi mesh creation depends
on it.

All tests run without JIT and at small problem sizes so the suite
stays cheap.  Production-scale integration of DINO is covered by
``scripts/run/run_dino.py`` and the experiment registry; this file is
the slopbuster-mandated direct-coverage entry point.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.latlon import create_mercator_grid
from legoesm.ocean.experiments import dino
from legoesm.ocean.experiments import AVAILABLE_EXPERIMENTS
from legoesm.ocean.experiments.dino import (
    DINO_L2_RECIPES,
    DINO_RECIPES,
    DINOConfig,
    EXPERIMENT_CONFIG,
    create_dino_z_star,
    create_initial_conditions,
    dino_S_profile_1d,
    dino_S_star,
    dino_T_profile_1d,
    dino_T_star_annual_mean,
    dino_bathymetry,
    dino_config_for_recipe,
    dino_initial_T_S,
    dino_lat_lon_grid,
    dino_lat_lon_initial_state_arrays,
    dino_lat_lon_model_config,
    nemo_faithful_dino_config,
    dino_top_layer_S_tendency,
    dino_top_layer_T_tendency,
    dino_top_layer_u_tendency,
    dino_wind_stress,
)
from legoesm.ocean.init_latlon_cgrid import partial_periodic_seam_wall_latlon
from legoesm.ocean.vertical import create_levy_stretched_z_star


# ---------------------------------------------------------------------
# Config + registry
# ---------------------------------------------------------------------

class TestDINOConfig:
    def test_default_construct(self):
        cfg = DINOConfig()
        assert cfg.n_levels == 36
        assert cfg.H_deep == 4000.0
        assert cfg.H_shallow == 2000.0
        assert cfg.dz_min == 10.0
        # Paper-specific reference constants differ from project canonical
        # values by <0.1% — flagged in the docstring.
        assert cfg.rho_0 == pytest.approx(1026.0)
        assert cfg.c_p == pytest.approx(3991.86)
        # Wind knots: 7 (lat, tau) pairs.
        assert len(cfg.wind_tau_lats_deg) == 7
        assert len(cfg.wind_tau_values) == 7

    def test_mpas_equatorial_visc_boost_propagates(self):
        """The MPAS-only equatorial viscosity boost
        (``mpas_equatorial_visc_boost``) must reach the ``MPASOceanConfig``.
        Without it the forced f→0 equatorial jet runs away on the implicit-CN
        MPAS path (|u| 1.8→7.6 m/s by day 20 → NaN by day 30; fastest edges
        all at |lat|<3°). It mirrors the lat-lon ``A_h_eq_boost`` mechanism.
        ``physics`` is left off so the stub mesh only needs ``areaCell``.
        """
        import dataclasses
        from types import SimpleNamespace
        mesh = SimpleNamespace(areaCell=jnp.full((64,), 1.0e10))  # ~100 km cells
        cfg = DINOConfig()
        assert cfg.mpas_equatorial_visc_boost == pytest.approx(8.0)
        mc, phys = dino.dino_mpas_model_config(mesh, cfg, physics=False)
        assert mc.equatorial_visc_boost == pytest.approx(
            cfg.mpas_equatorial_visc_boost)
        assert phys is None
        # Override (the run_dino --mpas-eq-visc-boost flag) propagates. Use a
        # value distinct from the 8.0 default so the assert is non-vacuous.
        cfg2 = dataclasses.replace(cfg, mpas_equatorial_visc_boost=12.0)
        mc2, _ = dino.dino_mpas_model_config(mesh, cfg2, physics=False)
        assert mc2.equatorial_visc_boost == pytest.approx(12.0)

    def test_mpas_physics_is_wired_into_model_config(self):
        """The MPAS model gates KPP/GM-Redi/convection on
        ``config.physics is not None``; dino_mpas_model_config MUST wire the
        physics into the returned MPASOceanConfig (regression: it used to return
        physics only as the 2nd value, which run_dino drops -> dycore-only MPAS).
        """
        from types import SimpleNamespace
        mesh = SimpleNamespace(areaCell=jnp.full((64,), 1.0e10))
        mc, phys = dino.dino_mpas_model_config(mesh, DINOConfig(), physics=True)
        assert phys is not None
        assert mc.physics is not None, "MPAS physics not wired into the model config"
        assert mc.physics is phys
        assert mc.physics.vertical_mixing.scheme in ("kpp", "tke", "constant")
        # physics=False stays dycore-only.
        mc0, phys0 = dino.dino_mpas_model_config(mesh, DINOConfig(), physics=False)
        assert phys0 is None and mc0.physics is None

    def test_vmix_scheme_default_and_dispatch(self):
        """The shared vertical-mixing helper selects the paper's TKE closure by
        default (with the paper background visc/diff + convective ceiling) and
        KPP on request, and raises on an unknown scheme (dispatch hardening)."""
        import dataclasses
        cfg = DINOConfig()
        # Default is the stable KPP closure (TKE = paper's scheme but unstable
        # in our 1deg DINO past ~day 40 — see DINOConfig.vmix_scheme).
        assert cfg.vmix_scheme == "kpp"
        vm_kpp = dino._dino_vertical_mixing_config(cfg)
        assert vm_kpp.scheme == "kpp"
        assert vm_kpp.kpp.K_bg == pytest.approx(cfg.K_v_bg)
        # The paper-faithful TKE config maps the paper background + ceiling, and
        # prandtl_mode="constant" is required or kappaH_min/kappaM_max are dead.
        cfg_tke = dataclasses.replace(cfg, vmix_scheme="tke")
        vm = dino._dino_vertical_mixing_config(cfg_tke)
        assert vm.scheme == "tke"
        assert vm.tke.prandtl_mode == "constant"
        # MOMENTUM floor = the EFFECTIVE A_v (the SW-corner stabilizer = 5e-4, 4×
        # the paper) so TKE is runnable; the TRACER floor stays at the paper
        # K_v_bg, so the thermocline mixing is unchanged (see A_v_bg_effective).
        assert vm.tke.kappaM_min == pytest.approx(cfg_tke.A_v_bg_effective)
        assert cfg_tke.A_v_bg_effective == pytest.approx(5.0e-4)
        assert vm.tke.kappaH_min == pytest.approx(cfg.K_v_bg)
        assert vm.tke.kappaM_max == pytest.approx(cfg.K_conv)
        assert vm.tke.bg_diff_scale == pytest.approx(0.0)  # constant bg (no Bryan-Lewis)
        # "constant" = background-only (used to unify vmix across grids).
        vm_c = dino._dino_vertical_mixing_config(
            dataclasses.replace(cfg, vmix_scheme="constant"))
        assert vm_c.scheme == "constant"
        # richardson (≈ Oceananigans RiBasedVerticalDiffusivity) and catke
        # (Oceananigans CATKEVerticalDiffusivity) are the L2 Oceananigans-card
        # closures; both wire with the paper background floors.
        vm_ri = dino._dino_vertical_mixing_config(
            dataclasses.replace(cfg, vmix_scheme="richardson"))
        assert vm_ri.scheme == "richardson"
        assert vm_ri.richardson.K_bg == pytest.approx(cfg.K_v_bg)
        assert vm_ri.richardson.A_bg == pytest.approx(cfg.A_v_bg_effective)
        vm_catke = dino._dino_vertical_mixing_config(
            dataclasses.replace(cfg, vmix_scheme="catke"))
        assert vm_catke.scheme == "catke"
        with pytest.raises(ValueError):
            dino._dino_vertical_mixing_config(
                dataclasses.replace(cfg, vmix_scheme="bogus"))

    def test_tke_etau_htau_mode_is_latitude_and_matches_nemo_profile(self):
        """DINO's TKE closure uses the NEMO nn_htau=1 LATITUDE sub-ML
        penetration-depth profile (namelist_ref default, unoverridden by DINO's
        namelist_cfg), not the constant10m (nn_htau=0) profile a prior card
        used — see the etau_htau_mode="latitude" comment in
        _dino_vertical_mixing_config. Pins both the config wiring and the
        exact numeric htau(lat) NEMO formula (zdftke.F90:870):
            htau = max(0.5, min(30, 45*|sin(deg2rad(lat))|))  [m]
        at lat=0 (floor), lat=45 (ceiling), and a mid-latitude value — by
        back-solving nemo_etau_injection() (the real production function),
        not a re-derived formula."""
        import dataclasses
        from legoesm.ocean.physics.vertical_mixing.tke import nemo_etau_injection

        cfg = dataclasses.replace(DINOConfig(), vmix_scheme="tke")
        vm = dino._dino_vertical_mixing_config(cfg)
        assert vm.tke.etau_htau_mode == "latitude"
        assert vm.tke.etau_mode == "below_ml"

        expected_20 = 45.0 * math.sin(math.radians(20.0))
        assert expected_20 == pytest.approx(15.390906449655093, abs=1e-9)

        # Recover the htau the closure actually applies at each latitude by
        # back-solving inj = etau_frac*e_sfc*exp(-depth_w/htau): evaluate the
        # injection at depth_w=0 (isolates etau_frac*e_sfc) and at depth_w=100
        # (adds the exp factor), then invert for htau — exercises the real
        # nemo_etau_injection() call the TKE column uses, no re-derivation.
        e0 = jnp.zeros((1,))
        taum = jnp.asarray([1.0])
        for lat_deg, expected_htau in ((0.0, 0.5), (45.0, 30.0), (20.0, expected_20)):
            lat = jnp.asarray([lat_deg])
            inj0 = nemo_etau_injection(e0, taum, jnp.asarray([0.0]), vm.tke, lat_deg=lat)
            inj100 = nemo_etau_injection(e0, taum, jnp.asarray([100.0]), vm.tke, lat_deg=lat)
            recovered_htau = float(-100.0 / jnp.log(inj100[0, 0] / inj0[0, 0]))
            assert recovered_htau == pytest.approx(expected_htau, rel=0, abs=1e-10), (
                f"lat={lat_deg}: htau={recovered_htau} != expected {expected_htau}")

    def test_tke_prandtl_ri_maps_nemo_nn_pdl(self):
        """DINOConfig.tke_prandtl_ri=True wires the NEMO zdftke nn_pdl=1
        Richardson Prandtl: prandtl_mode='nemo_ri' (Phase-2 #1317 T8 — NEMO's
        EXACT zri=rn2b*avm/(sh2+bshear) form, not Veros's own "richardson"
        Ri=N2/shear_sq, which is missing the avm factor) with coeff=1/ri_cri,
        ri_cri=2/(2+c_eps/c_k)=2/9 -> coeff=4.5. Default False keeps Pr=10
        constant (other recipes byte-identical). Truth-tier: in a strongly
        stratified column (Ri>>ri_cri) the tracer avt drops to 0.1*avm (NEMO
        pdlr floor); in a convecting column (Ri<0) avt tracks avm (Pr=1)."""
        import dataclasses
        from legoesm.ocean.physics.vertical_mixing.tke import (
            _prandtl_number, compute_K_from_tke,
        )
        cfg = DINOConfig()
        assert cfg.tke_prandtl_ri is False
        # Default (off): constant Pr=10.
        vm_off = dino._dino_vertical_mixing_config(
            dataclasses.replace(cfg, vmix_scheme="tke"))
        assert vm_off.tke.prandtl_mode == "constant"
        # On: NEMO Ri-Prandtl.
        vm_on = dino._dino_vertical_mixing_config(
            dataclasses.replace(cfg, vmix_scheme="tke", tke_prandtl_ri=True))
        tke = vm_on.tke
        assert tke.prandtl_mode == "nemo_ri"
        ri_cri = 2.0 / (2.0 + tke.c_eps / tke.c_k)          # NEMO 2/9
        assert tke.prandtl_ri_coeff == pytest.approx(1.0 / ri_cri)
        assert tke.prandtl_ri_coeff == pytest.approx(4.5)
        # pdlr(Ri) curve matches NEMO MAX(0.1, ri_cri/MAX(ri_cri,Ri)) exactly.
        Ri = np.array([-1.0, 0.0, 0.1, ri_cri, 1.0, 5.0, 50.0])
        N2 = jnp.asarray(Ri)                                 # unit shear -> Ri=N2
        Pr = np.asarray(_prandtl_number(N2, jnp.ones_like(N2), jnp.ones_like(N2), tke))
        pdlr_nemo = np.maximum(0.1, ri_cri / np.maximum(ri_cri, Ri))
        np.testing.assert_allclose(1.0 / Pr, pdlr_nemo, rtol=0, atol=1e-12)
        # Strongly stratified interior: avt -> 0.1*avm (pdlr floor). Build K
        # with a large raw K_M (l_k, e sized so K_M >> floors/ceiling irrelevant).
        strat = tke._replace(kappaM_max=1e6, kappaM_min=1e-6, kappaH_min=1e-9)
        e = jnp.full((5,), 1e-2)
        l_k = jnp.full((5,), 10.0)
        big_ri = jnp.full((5,), 100.0)                       # Ri=100 >> ri_cri
        K_M, K_H = compute_K_from_tke(
            e, l_k, strat, N2=big_ri, shear_sq=jnp.ones((5,)))
        np.testing.assert_allclose(np.asarray(K_H) / np.asarray(K_M), 0.1,
                                   rtol=1e-6)
        # Convecting column: Ri<0 -> Pr=1 -> avt tracks avm (before floors).
        _, K_H_conv = compute_K_from_tke(
            e, l_k, strat, N2=jnp.full((5,), -1.0), shear_sq=jnp.ones((5,)))
        np.testing.assert_allclose(np.asarray(K_H_conv) / np.asarray(K_M), 1.0,
                                   rtol=1e-6)

    def test_nemo_dino_kamm_recipe_sets_ri_prandtl(self):
        """The complete NEMO card enables nn_pdl=1; other recipes do not."""
        cfg = dino_config_for_recipe("nemo_dino_kamm")
        assert cfg.tke_prandtl_ri is True
        vm = dino._dino_vertical_mixing_config(cfg)
        assert vm.tke.prandtl_mode == "nemo_ri"
        assert vm.tke.prandtl_ri_coeff == pytest.approx(4.5)
        # Backward-compat: the Veros DINO card keeps constant Pr.
        assert dino_config_for_recipe("veros").tke_prandtl_ri is False

    def test_registry_entry(self):
        assert "dino" in AVAILABLE_EXPERIMENTS
        assert AVAILABLE_EXPERIMENTS["dino"] is EXPERIMENT_CONFIG
        assert EXPERIMENT_CONFIG["name"] == "dino"
        assert EXPERIMENT_CONFIG["config_class"] is DINOConfig
        # Grid support matches the implemented paths.
        gs = EXPERIMENT_CONFIG["grid_support"]
        assert gs["latlon"] is True
        assert gs["mpas"] is True
        assert gs["cubed_sphere"] is False
        assert gs["spectral"] is False


# ---------------------------------------------------------------------
# L2 model recipes (DINO two-level intercomparison)
# ---------------------------------------------------------------------

class TestDINORecipes:
    """The recipe overlay (``DINO_RECIPES`` + ``dino_config_for_recipe``) selects
    each model's canonical blocks as a PURE CONFIG on ``DINOConfig``, threads
    them into the lat-lon model config, and raises on an unknown recipe
    (dispatch hardening — a typo must never silently pick a default)."""

    def test_unknown_recipe_raises(self):
        with pytest.raises(ValueError):
            dino_config_for_recipe("bogus_model")

    def test_catalog_membership(self):
        assert set(DINO_RECIPES) == {
            "legoesm_default", "nemo_paper", "nemo_dino_kamm",
            "nemo_dino_kamm_mlf",
            "veros", "mitgcm", "oceananigans"}
        assert set(DINO_L2_RECIPES) == {"veros", "mitgcm", "oceananigans"}
        assert set(DINO_L2_RECIPES) <= set(DINO_RECIPES)

    def test_legoesm_default_is_identity(self):
        # The identity card = a bare DINOConfig (Wright + KPP), so a no-op run
        # reproduces the production default exactly.
        assert dino_config_for_recipe("legoesm_default") == DINOConfig()

    def test_recipe_overlay_selects_documented_blocks(self):
        # EOS + vertical mixing + tracer advection = the per-model DINO choices.
        nemo = dino_config_for_recipe("nemo_paper")
        assert (nemo.eos, nemo.vmix_scheme, nemo.tracer_advection) == (
            "nemo_seos", "tke", "tvd")
        # nemo_paper must use the geometric S-EOS depth (NEMO gdept) — the DINO
        # tendency certificate showed insitu carries a 10x depth-proportional
        # error in the thermobaric term.  Other recipes keep the insitu default.
        assert nemo.eos_depth == "geometric"
        assert dino_config_for_recipe("legoesm_default").eos_depth == "insitu"
        veros = dino_config_for_recipe("veros")
        assert (veros.eos, veros.vmix_scheme, veros.tracer_advection,
                veros.barotropic_solver) == (
            "veros_nonlin2", "tke", "superbee", "rigid_lid")
        mit = dino_config_for_recipe("mitgcm")
        assert (mit.eos, mit.vmix_scheme, mit.tracer_advection,
                mit.momentum_advection, mit.outer_integrator,
                mit.barotropic_solver) == (
            "unesco80", "kpp", "dst3_multidim", "flux_form", "ab2",
            "implicit_unsplit")
        ocn = dino_config_for_recipe("oceananigans")
        assert (ocn.eos, ocn.vmix_scheme, ocn.tracer_advection,
                ocn.momentum_advection, ocn.outer_integrator) == (
            "veros_gsw", "catke", "weno7", "weno7", "ab2")

    def test_nemo_dino_kamm_card_is_complete(self):
        # The completed NEMO-DINO card (Kamm 2025) sets EVERY NEMO-relevant field
        # explicitly (unlike nemo_paper, which inherits legoESM defaults on ~8).
        # Assert each against its NEMO namelist value so the card can't drift.
        c = dino_config_for_recipe("nemo_dino_kamm")
        nemo = {
            "eos": "nemo_seos", "eos_depth": "geometric",
            "vertical_coordinate": "masked_zco",          # ln_zco_nam, full-step
            "vmix_scheme": "tke", "tke_momentum_visc_bg": 1.2e-4,  # rn_avm0
            "convection_smooth_transition": False, "convection_n2_mode": "adiabatic",
            "convection_n2_threshold": -1e-12,            # zdfevd
            "bottom_drag_scheme": "nemo_quadratic",       # ln_non_lin
            "tracer_advection": "fct2",                   # nn_fct=2
            "gm_kappa_scheme": "treguier", "redi_S_max": 0.01,  # nn_aei=21, rn_slpmax
            "ke_gradient_scheme": "hollingsworth",        # nn_dynkeg=1
            "A_h_eq_boost": 1.0, "A_h_floor": 0.0,        # no legoESM stabilizers
            "barotropic_solver": "explicit_substep",      # ln_dynspg_ts
            "barotropic_time_filter": "nemo_boxcar_centred",  # nn_bt_flt=2
            "forcing_annual_cycle": True, "wind_through_step": True,  # ln_ann_cyc
            "use_gm_redi": True,                          # ln_ldfeiv
            # namdyn_hpg ln_hpg_sco (dynhpg.F90 hpg_sco: qco stretch + zuap
            # slope term — the staircase form stress, #1226) + its trapezoid
            # p' quadrature on the exact gdept ladder.
            "pgf_scheme": "nemo_sco",
            "pgf_quadrature": "nemo_trapezoid",
            # dynzdf composition (#1226): namdrg ref default ln_drgimp=.true.
            # (DINO's &namdrg override sets only ln_non_lin) -- the full
            # NEMO-faithful drag composition is now ON: zdf_drag_in_matrix
            # (implicit diagonal, dynzdf.F90:293-305) + zdf_baroclinic_only
            # (barotropic mean removed from the 3-D solve, dynzdf.F90:147-171)
            # + barotropic_drag_substep (in-subcycle explicit drag + pu_RHSi
            # correction, dyn_drg, dynspg_ts.F90:700-706 + 1584-1642). See
            # the comment at DINO_RECIPES["nemo_dino_kamm"] in dino.py.
            "zdf_drag_in_matrix": True,
            "zdf_baroclinic_only": True,
            "barotropic_drag_substep": True,
            # NEMO dynspg_ts has no eta-diffusion term; alpha=0 is the
            # NEMO-true composition (see dino.py card comment).
            "barotropic_diffusion_alpha": 0.0,
            # Zero-deviation item 2 (#1226): NEMO zhup2_e/zhvp2_e ssh-average
            # face depths (dynspg_ts.F90:568-592), pair-consistent with the
            # tracer continuity; conservation gate
            # test_partial_cells_phase7.py::TestNemoSshAvgFaceDepthGate.
            "barotropic_face_depth": "nemo_ssh_avg",
        }
        for field, want in nemo.items():
            assert getattr(c, field) == want, f"{field}: {getattr(c, field)} != {want}"
        # It must build a model config (the demanding masked_zco + stabilizers-off
        # + boxcar combination is exercised) — the recipe is runnable, not just a
        # dict of values.
        grid = dino_lat_lon_grid(c, n_lon=10)
        dino_lat_lon_model_config(grid, c, physics=True)

    def test_kamm_cards_propagate_zdf_flags_to_model_config(self):
        # #1226: the full NEMO-faithful drag composition (zdf_drag_in_matrix +
        # zdf_baroclinic_only + barotropic_drag_substep) is ON for BOTH kamm
        # cards (MLF inherits from the base nemo_dino_kamm dict — see
        # DINO_RECIPES["nemo_dino_kamm_mlf"]), together with alpha=0 (no NEMO
        # eta-diffusion counterpart) and face_depth="nemo_ssh_avg" (zero-
        # deviation item 2, pair-consistent with the tracer continuity;
        # conservation gate test_partial_cells_phase7.py::
        # TestNemoSshAvgFaceDepthGate — see the dino.py card comment).
        # Assert these propagate through to the model config unchanged on both.
        for recipe in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
            c = dino_config_for_recipe(recipe)
            assert c.zdf_drag_in_matrix is True, recipe
            assert c.zdf_baroclinic_only is True, recipe
            assert c.barotropic_drag_substep is True, recipe
            assert c.barotropic_diffusion_alpha == 0.0, recipe
            assert c.barotropic_face_depth == "nemo_ssh_avg", recipe
            grid = dino_lat_lon_grid(c, n_lon=10)
            mc, _ = dino_lat_lon_model_config(grid, c, physics=True)
            assert mc.zdf_drag_in_matrix is True, recipe
            assert mc.zdf_baroclinic_only is True, recipe
            assert mc.barotropic_drag_substep is True, recipe
            assert mc.barotropic.barotropic_diffusion_alpha == 0.0, recipe
            assert mc.barotropic.barotropic_face_depth == "nemo_ssh_avg", recipe

    def test_zdf_flags_default_false_on_other_recipes(self):
        # Every non-kamm recipe (veros/mitgcm/oceananigans/legoesm_default/
        # nemo_paper) must NOT silently pick up NEMO's dynzdf composition —
        # it is namelist-specific, not a legoESM-wide default.
        for recipe in ("legoesm_default", "nemo_paper", "veros", "mitgcm",
                       "oceananigans"):
            c = dino_config_for_recipe(recipe)
            assert c.zdf_drag_in_matrix is False, recipe
            assert c.zdf_baroclinic_only is False, recipe
            assert c.barotropic_drag_substep is False, recipe
            # #1226: alpha=0 is a kamm-only override; every other recipe
            # keeps the legoESM 2Δx stability-crutch default (0.01).
            assert c.barotropic_diffusion_alpha == 0.01, recipe
            # #1226: nemo_ssh_avg face depths are a kamm-only override; every
            # other recipe keeps the bit-identical min-rule default.
            assert c.barotropic_face_depth == "min_rule", recipe

    def test_barotropic_forcing_centred_mlf_card_only(self):
        # #1226 zero-deviation item 3 (NEMO ln_bt_fw=.FALSE. centred
        # barotropic wind/emp forcing, dynspg_ts.F90:392-421 + the Kbb drag
        # residual :1623-1636): only meaningful under the leapfrog outer
        # integrator, so it lands ONLY on nemo_dino_kamm_mlf -- the FE
        # nemo_dino_kamm card (ln_bt_fw=T forward branch, already uncentred)
        # must stay False.
        mlf = dino_config_for_recipe("nemo_dino_kamm_mlf")
        assert mlf.barotropic_forcing_centred is True
        fe = dino_config_for_recipe("nemo_dino_kamm")
        assert fe.barotropic_forcing_centred is False
        for recipe in ("legoesm_default", "nemo_paper", "veros", "mitgcm",
                       "oceananigans"):
            assert dino_config_for_recipe(recipe).barotropic_forcing_centred is False, recipe
        # Threads into the model config + actually runs (leapfrog + centred
        # forcing + the full NEMO drag composition all together).
        grid = dino_lat_lon_grid(mlf, n_lon=10)
        mc, _ = dino_lat_lon_model_config(grid, mlf, physics=True)
        assert mc.barotropic_forcing_centred is True

    def test_barotropic_een_seed_mlf_card_only(self):
        # #1226 zero-deviation item 4 (NEMO dyn_cor_2D_init(Kmm) EEN
        # coefficient seed, dynspg_ts.F90:355 + :1349-1379): only differs
        # from the legacy window-start freeze under the MLF before-level
        # seed, so "nemo_kmm" lands ONLY on nemo_dino_kamm_mlf; every other
        # recipe keeps the bit-identical "window_start" default.
        mlf = dino_config_for_recipe("nemo_dino_kamm_mlf")
        assert mlf.barotropic_een_seed == "nemo_kmm"
        for recipe in ("nemo_dino_kamm", "legoesm_default", "nemo_paper",
                       "veros", "mitgcm", "oceananigans"):
            assert (dino_config_for_recipe(recipe).barotropic_een_seed
                    == "window_start"), recipe
        grid = dino_lat_lon_grid(mlf, n_lon=10)
        mc, _ = dino_lat_lon_model_config(grid, mlf, physics=True)
        assert mc.barotropic.barotropic_een_seed == "nemo_kmm"

    def test_een_e3f_scheme_mlf_card_only(self):
        # #1226 item 10 (NEMO nn_e3f_typ=1, dynvor.F90::vor_een:733-745 —
        # masked-average e3f, not the min-rule). Only the leapfrog/EEN-total
        # card selects it; every other recipe keeps the bit-identical
        # min-rule default (MITgcm hFacZ convention).
        mlf = dino_config_for_recipe("nemo_dino_kamm_mlf")
        assert mlf.een_e3f_scheme == "nemo_avg"
        for recipe in ("nemo_dino_kamm", "legoesm_default", "nemo_paper",
                       "veros", "mitgcm", "oceananigans"):
            assert dino_config_for_recipe(recipe).een_e3f_scheme == "min", recipe
        grid = dino_lat_lon_grid(mlf, n_lon=10)
        mc, _ = dino_lat_lon_model_config(grid, mlf, physics=True)
        assert mc.een_e3f_scheme == "nemo_avg"

    def test_nemo_paper_convection_is_nemo_hard_switch(self):
        # NEMO zdfevd is a HARD rn2<0 switch on the adiabatic (eosbn2) N^2. The
        # legoESM sigmoid default leaks enhanced mixing into weakly-stable water
        # and over-cools the DINO thermocline ~0.7 C (62-day matched-grid check:
        # T@262m 8.78 vs NEMO 9.50 -> 9.55 with the hard step). The nemo_paper
        # oracle recipe must select the faithful pair; other recipes keep the
        # smooth legoESM default (behavior preservation).
        nemo = dino_config_for_recipe("nemo_paper")
        assert nemo.convection_smooth_transition is False
        assert nemo.convection_n2_mode == "adiabatic"
        assert nemo.convection_n2_threshold == -1e-12   # NEMO zdfevd threshold
        default = dino_config_for_recipe("legoesm_default")
        assert default.convection_smooth_transition is True
        assert default.convection_n2_mode == "insitu"
        assert default.convection_n2_threshold == 0.0
        # The faithful pair must actually THREAD into the built EnhancedDiffusion
        # config (both the lat-lon and MPAS convection builders read the cfg).
        grid = dino_lat_lon_grid(nemo, n_lon=10)
        mc, _ = dino_lat_lon_model_config(grid, nemo, physics=True)
        ed = mc.physics.convection.enhanced_diffusion
        assert ed.smooth_transition is False
        assert ed.n2_mode == "adiabatic"
        assert ed.n2_threshold == -1e-12

    def test_gm_redi_mld_criterion_threads_and_validates(self):
        import dataclasses
        # nemo_dino_kamm selects the NEMO N^2-integral MLD criterion; it must
        # reach the built GMRediConfig.
        kamm = dino_config_for_recipe("nemo_dino_kamm")
        assert kamm.gm_redi_mld_criterion == "n2_integral"
        grid = dino_lat_lon_grid(kamm, n_lon=10)
        mc, _ = dino_lat_lon_model_config(grid, kamm, physics=False)
        assert mc.gm_redi.mld_criterion == "n2_integral"
        # Default recipe keeps the byte-identical pot-density criterion.
        default = dino_config_for_recipe("legoesm_default")
        assert default.gm_redi_mld_criterion == "rho_c"
        # The config builder raises on an unknown criterion (dispatch hardening).
        bad = dataclasses.replace(DINOConfig(), gm_redi_mld_criterion="bogus")
        gridb = dino_lat_lon_grid(bad, n_lon=10)
        with pytest.raises(ValueError, match="gm_redi_mld_criterion"):
            dino_lat_lon_model_config(gridb, bad, physics=False)

    def test_base_override_preserved(self):
        # A recipe overlay keeps the non-scheme setup fields of the base config.
        import dataclasses
        base = dataclasses.replace(DINOConfig(), dt=1800.0)
        cfg = dino_config_for_recipe("mitgcm", base=base)
        assert cfg.dt == pytest.approx(1800.0)     # setup field preserved
        assert cfg.eos == "unesco80"               # recipe field applied

    def test_default_model_config_scheme_identity_unchanged(self):
        # Behavior preservation: a bare DINOConfig still yields the legoESM DINO
        # dycore identity (the newly threaded fields default to the prior values,
        # so existing runs are byte-identical).
        cfg = DINOConfig()
        grid = dino_lat_lon_grid(cfg, n_lon=10)
        mc, _ = dino_lat_lon_model_config(grid, cfg, physics=False)
        assert mc.flat_get("momentum_advection") == "vector_invariant"
        assert mc.flat_get("coriolis_scheme") == "matsuno_split"
        assert mc.flat_get("outer_integrator") == "forward_euler"

    def test_mitgcm_threads_into_model_config(self):
        # The MITgcm card's flux-form / AB2 / unsplit-FS blocks reach the model.
        cfg = dino_config_for_recipe("mitgcm")
        grid = dino_lat_lon_grid(cfg, n_lon=10)
        mc, _ = dino_lat_lon_model_config(grid, cfg, physics=False)
        assert mc.flat_get("momentum_advection") == "flux_form"
        assert mc.flat_get("momentum_flux_scheme") == "centered"
        assert mc.flat_get("coriolis_scheme") == "explicit_ab2"
        assert mc.flat_get("outer_integrator") == "ab2"
        assert mc.flat_get("ab2_scope") == "total"
        assert mc.flat_get("barotropic_solver") == "implicit_unsplit"
        assert mc.flat_get("eos") == "unesco80"

    def test_veros_rigid_lid_forces_ab2_stack(self):
        # The Veros card selects rigid_lid; the builder ALWAYS overlays the
        # coordinated Veros-faithful ab2 stack on that path (ab2 + explicit_ab2 +
        # advective scope), independent of the DINOConfig integrator defaults.
        cfg = dino_config_for_recipe("veros")
        grid = dino_lat_lon_grid(cfg, n_lon=10)
        mc, _ = dino_lat_lon_model_config(grid, cfg, physics=False)
        assert mc.flat_get("barotropic_solver") == "rigid_lid"
        assert mc.flat_get("outer_integrator") == "ab2"
        assert mc.flat_get("coriolis_scheme") == "explicit_ab2"
        assert mc.flat_get("ab2_scope") == "advective"
        assert mc.flat_get("eos") == "veros_nonlin2"
        # rigid_lid force-disables the F_slow AB2 flag (its validation rejects
        # ab2_scope="advective"; the streamfunction projection has no barotropic
        # inertial mode to time-center).
        assert mc.flat_get("barotropic_slow_forcing_ab2") is False

    def test_oceananigans_card_ab2_centers_barotropic_slow_forcing(self):
        # Root-cause guard for the DINO 'oceananigans'-card barotropic blowup
        # (dino_l2_bisect o_ctl: |eta| 6 m by day 15, growth rate ∝ dt, basin-
        # scale off-equatorial quadrupole).  Under coriolis_scheme="explicit_ab2"
        # + barotropic_solver="implicit_cn" the CN predictor gates its FB
        # Coriolis off (_cori_fac=0) and the outer AB2 keeps the barotropic
        # increment un-extrapolated, so WITHOUT barotropic_slow_forcing_ab2 the
        # barotropic-mode Coriolis integrates forward-Euler — unconditionally
        # unstable, |G| = sqrt(1 + (f·dt)²) per step.  The card must carry the
        # Oceananigans-faithful Gᵁ AB2 time-centering (as the validated-stable
        # Silvestri §5 jet stack does).
        cfg = dino_config_for_recipe("oceananigans")
        assert cfg.barotropic_slow_forcing_ab2 is True
        grid = dino_lat_lon_grid(cfg, n_lon=10)
        mc, _ = dino_lat_lon_model_config(grid, cfg, physics=False)
        assert mc.flat_get("coriolis_scheme") == "explicit_ab2"
        assert mc.flat_get("barotropic_solver") == "implicit_cn"
        assert mc.flat_get("outer_integrator") == "ab2"
        assert mc.flat_get("ab2_scope") == "total"
        assert mc.flat_get("barotropic_slow_forcing_ab2") is True


class TestNemoFaithfulGrid:
    """Opt-in NEMO-exact DINO grid (nemo_faithful_grid). Mesh-verified against
    our NEMO 5.0.2 DINO build: 48×195, equator on a T-point, faces [1,49]."""

    def test_default_grid_unchanged(self):
        # Opt-in: a bare config keeps the legoESM [-50,0]/198×50 grid.
        import numpy as np
        assert DINOConfig().nemo_faithful_grid is False
        grid = dino_lat_lon_grid(DINOConfig())
        assert np.asarray(grid.lat).shape == (198,)     # equator on a face
        assert np.asarray(grid.lon).shape == (50,)

    def test_nemo_faithful_matches_nemo_mesh(self):
        # NEMO DINO R1: 48 zonal cells (T-centres [1.5,48.5]), 195 rows with the
        # equator ON a T-point (j=97 = 0.0°) at ±69.151°. The projection is
        # identical; only the half-cell equator staggering + count differ.
        import numpy as np
        cfg = nemo_faithful_dino_config()
        assert cfg.nemo_faithful_grid is True
        assert (cfg.lon_west_deg, cfg.lon_east_deg) == (1.0, 49.0)
        assert cfg.sill_lon_m_deg == 1.0        # sill anchor co-set to west wall
        grid = dino_lat_lon_grid(cfg)
        lat = np.degrees(np.asarray(grid.lat))
        lon = np.degrees(np.asarray(grid.lon))
        assert lat.shape == (195,) and lon.shape == (48,)
        assert lon[0] == pytest.approx(1.5, abs=1e-4)
        assert lon[-1] == pytest.approx(48.5, abs=1e-4)
        assert lat[97] == pytest.approx(0.0, abs=1e-5)     # equator on T-point
        assert abs(lat[0]) == pytest.approx(69.1514, abs=1e-3)

    def test_nemo_faithful_bathymetry_domain_is_wet(self):
        # The co-set lon frame keeps the bathymetry valid (not an all-land
        # domain) — the sill anchor tracks the western wall at 1.0.
        import numpy as np
        from legoesm.ocean.experiments.dino import dino_lat_lon_bowl
        cfg = nemo_faithful_dino_config()
        H = np.asarray(dino_lat_lon_bowl(dino_lat_lon_grid(cfg), cfg))
        assert (H > 1.0).any()                              # not all land
        assert H.max() == pytest.approx(cfg.H_deep, abs=1.0)

    def test_config_helper_preserves_base_recipe(self):
        # run_dino applies `nemo_faithful_dino_config(base=cfg)` AFTER the recipe
        # overlay, so it must keep the recipe's scheme choices while co-setting
        # only the grid/bathymetry lon frame. This covers the flag->config wiring
        # the CLI relies on (the helper's base= path).
        base = dino_config_for_recipe("nemo_paper")
        cfg = nemo_faithful_dino_config(base=base)
        assert cfg.nemo_faithful_grid is True
        assert (cfg.lon_west_deg, cfg.lon_east_deg, cfg.sill_lon_m_deg) == (
            1.0, 49.0, 1.0)
        # recipe fields survive (eos, convection fidelity, barotropic solver)
        assert cfg.eos == "nemo_seos"
        assert cfg.convection_smooth_transition is False
        assert cfg.convection_n2_mode == "adiabatic"

    def test_flag_without_lon_frame_raises(self):
        # Flipping the flag alone (legoESM [-50,0] frame) would put every grid
        # lon outside the basin -> all land; the builder fails loudly instead.
        import dataclasses
        with pytest.raises(ValueError, match="NEMO .1,49. longitude frame"):
            dino_lat_lon_grid(
                dataclasses.replace(DINOConfig(), nemo_faithful_grid=True))


# ---------------------------------------------------------------------
# Bathymetry (paper Appendix A)
# ---------------------------------------------------------------------

class TestDinoBathymetry:
    def test_interior_is_deep(self):
        # Point well inside the basin (no taper, no sill).
        H = float(dino_bathymetry(jnp.array(-25.0), jnp.array(0.0)))
        assert H == pytest.approx(DINOConfig().H_deep, rel=1e-3)

    def test_boundary_is_shallow(self):
        cfg = DINOConfig()
        # Exactly at the western wall.
        H = float(dino_bathymetry(
            jnp.array(cfg.lon_west_deg), jnp.array(0.0),
        ))
        assert H == pytest.approx(cfg.H_shallow, rel=1e-3)

    def test_drake_sill_shallows_basin(self):
        # The Drake sill is a Gaussian ring anchored at the western wall
        # (sill_lon_m_deg) and extending eastward over
        # ``sill_gaussian_width_s`` degrees. At lon=sill_lon_m_deg
        # exactly, ``sill_taper`` is 0 (left edge of smooth-step), so
        # we sample slightly east of the sill anchor where the taper is
        # active. In the channel band the open-flow bathymetry would be
        # H_deep, so any sill effect shoals it below H_deep.
        cfg = DINOConfig()
        lon_inside_sill = cfg.sill_lon_m_deg + 0.5 * cfg.sill_gaussian_width_s
        H_sill_region = float(dino_bathymetry(
            jnp.array(lon_inside_sill),
            jnp.array(cfg.sill_lat_m_deg),
        ))
        H_open_channel = float(dino_bathymetry(
            jnp.array(-25.0),
            jnp.array(cfg.sill_lat_m_deg),
        ))
        assert H_sill_region < H_open_channel
        assert H_sill_region >= cfg.H_sill - 100.0

    def test_shape_broadcast(self):
        lon = jnp.linspace(-50.0, 0.0, 10)
        lat = jnp.linspace(-70.0, 70.0, 12)[:, None]
        H = dino_bathymetry(lon[None, :], lat)
        assert H.shape == (12, 10)
        assert jnp.all(jnp.isfinite(H))


# ---------------------------------------------------------------------
# Wind stress
# ---------------------------------------------------------------------

class TestDinoWindStress:
    def test_hits_knots(self):
        cfg = DINOConfig()
        for lat_knot, tau_knot in zip(cfg.wind_tau_lats_deg, cfg.wind_tau_values):
            tau = float(dino_wind_stress(jnp.array(lat_knot)))
            assert tau == pytest.approx(tau_knot, abs=1e-6)

    def test_outside_range_clips(self):
        cfg = DINOConfig()
        # Beyond the southernmost knot — extrapolation clips to the
        # endpoint per the cubic-smooth-step formulation.
        tau_far_south = float(dino_wind_stress(jnp.array(-89.0)))
        assert tau_far_south == pytest.approx(cfg.wind_tau_values[0], abs=1e-6)
        tau_far_north = float(dino_wind_stress(jnp.array(89.0)))
        assert tau_far_north == pytest.approx(cfg.wind_tau_values[-1], abs=1e-6)


# ---------------------------------------------------------------------
# Restoring targets (paper eqs B1-B2)
# ---------------------------------------------------------------------

class TestRestoringTargets:
    def test_T_star_warmest_at_equator(self):
        cfg = DINOConfig()
        T_eq = float(dino_T_star_annual_mean(jnp.array(0.0)))
        T_north = float(dino_T_star_annual_mean(jnp.array(cfg.lat_max_deg)))
        T_south = float(dino_T_star_annual_mean(jnp.array(-cfg.lat_max_deg)))
        assert T_eq > T_north
        assert T_eq > T_south
        # Equatorial target matches the configured value when lat=0.
        assert T_eq == pytest.approx(cfg.T_star_eq, abs=0.5)

    def test_S_star_dip_at_equator(self):
        cfg = DINOConfig()
        # Equatorial Gaussian dip means S* at the equator is BELOW the
        # cosine-profile peak by approximately ``S_star_eq_dip_amp``.
        S_eq = float(dino_S_star(jnp.array(0.0)))
        S_north = float(dino_S_star(jnp.array(cfg.lat_max_deg)))
        assert S_eq < cfg.S_star_eq  # dip subtracted
        # Northern boundary target ~ S_star_n.
        assert S_north == pytest.approx(cfg.S_star_n, abs=0.5)


# ---------------------------------------------------------------------
# Equatorial 1D T(z), S(z) profiles (paper eq D2-D3)
# ---------------------------------------------------------------------

class TestProfiles1D:
    def test_T_decreasing_with_depth(self):
        z = jnp.linspace(0.0, 4000.0, 41)
        T = dino_T_profile_1d(z)
        # Sea-surface warmer than abyss, abyss roughly 3-5 C.
        T_np = np.asarray(T)
        assert T_np[0] > T_np[-1]
        assert 2.0 < T_np[-1] < 5.5
        assert 20.0 < T_np[0] < 30.0

    def test_S_finite_and_bounded(self):
        z = jnp.linspace(0.0, 4000.0, 41)
        S = np.asarray(dino_S_profile_1d(z))
        assert np.all(np.isfinite(S))
        assert (S > 30.0).all() and (S < 38.0).all()


# ---------------------------------------------------------------------
# Top-layer tendencies (paper eqs 7-9)
# ---------------------------------------------------------------------

class TestTopLayerTendencies:
    def test_T_relaxes_toward_target(self):
        cfg = DINOConfig()
        dz0 = 10.0
        # Cold ocean below warm target: positive tendency.
        dT_dt_warm = float(dino_top_layer_T_tendency(
            T_sfc_C=0.0, T_star=20.0, Q_sr=0.0, dz_0=dz0, cfg=cfg,
        ))
        # Warm ocean below cold target: negative tendency.
        dT_dt_cold = float(dino_top_layer_T_tendency(
            T_sfc_C=20.0, T_star=0.0, Q_sr=0.0, dz_0=dz0, cfg=cfg,
        ))
        assert dT_dt_warm > 0.0
        assert dT_dt_cold < 0.0

    def test_S_relaxes_toward_target(self):
        cfg = DINOConfig()
        dz0 = 10.0
        dS_dt = float(dino_top_layer_S_tendency(
            S_surface=34.0, S_star=36.0, dz_0=dz0, cfg=cfg,
        ))
        assert dS_dt > 0.0  # freshening deficit → S increases

    def test_u_proportional_to_tau(self):
        cfg = DINOConfig()
        dz0 = 10.0
        du_a = float(dino_top_layer_u_tendency(0.1, dz0, cfg))
        du_b = float(dino_top_layer_u_tendency(0.2, dz0, cfg))
        assert du_b == pytest.approx(2.0 * du_a, rel=1e-6)
        # Magnitude check: τ/(ρ·dz) ≈ 0.1 / (1026 · 10) ≈ 9.7e-6 m/s².
        expected = 0.1 / (cfg.rho_0 * dz0)
        assert du_a == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------
# Lévy z* helper (vertical.py) + DINO wrapper
# ---------------------------------------------------------------------

class TestLevyZStar:
    def test_endpoints_and_monotone(self):
        # 36-level DINO grid: dz_min constraint is dz/dk at k=1, which
        # matches dz[0] (forward difference) closely when the stretching
        # parameters keep d²z/dk² small over the top layer. Tight check
        # is therefore valid only at the design n_levels=36.
        n_levels = 36
        zc = create_levy_stretched_z_star(
            n_levels=n_levels, H_max=4000.0, dz_min=10.0,
            k_th=float(n_levels - 1), a_cr=10.5,
        )
        assert zc.n_levels == n_levels
        assert float(zc.z_half_ref[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(zc.z_half_ref[-1]) == pytest.approx(-4000.0, abs=1e-9)
        dz = np.asarray(zc.dz_ref)
        assert (dz > 0).all()
        assert dz[0] == pytest.approx(10.0, rel=0.05)
        assert dz[-1] > dz[0]
        assert dz.sum() == pytest.approx(4000.0, rel=1e-6)

    def test_low_resolution_endpoints(self):
        # At coarse resolutions the forward-difference dz[0] can differ
        # from the analytical dz/dk constraint, but endpoints and total
        # depth must still match exactly.
        zc = create_levy_stretched_z_star(
            n_levels=12, H_max=4000.0, dz_min=10.0,
            k_th=11.0, a_cr=10.5,
        )
        assert zc.n_levels == 12
        assert float(zc.z_half_ref[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(zc.z_half_ref[-1]) == pytest.approx(-4000.0, abs=1e-9)
        dz = np.asarray(zc.dz_ref)
        assert (dz > 0).all()
        assert dz.sum() == pytest.approx(4000.0, rel=1e-6)

    def test_create_dino_z_star_defaults(self):
        cfg = DINOConfig()
        zc = create_dino_z_star(cfg)
        assert zc.n_levels == cfg.n_levels == 36
        assert float(zc.z_half_ref[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(zc.z_half_ref[-1]) == pytest.approx(-cfg.H_deep, abs=1e-9)
        # Top layer ~ dz_min.
        assert float(zc.dz_ref[0]) == pytest.approx(cfg.dz_min, rel=0.05)

    def test_input_validation(self):
        with pytest.raises(ValueError):
            create_levy_stretched_z_star(1, 4000.0, 10.0, 35.0, 10.5)
        with pytest.raises(ValueError):
            create_levy_stretched_z_star(36, 0.0, 10.0, 35.0, 10.5)
        with pytest.raises(ValueError):
            create_levy_stretched_z_star(36, 4000.0, 0.0, 35.0, 10.5)


# ---------------------------------------------------------------------
# Seam-wall helper (lat-lon C-grid)
# ---------------------------------------------------------------------

class TestPartialPeriodicSeamLatLon:
    def test_open_band_stays_ocean(self):
        cfg = DINOConfig()
        grid = create_mercator_grid(
            n_lon=10,
            lat_max_deg=cfg.lat_max_deg,
            lon_west_deg=cfg.lon_west_deg,
            lon_east_deg=cfg.lon_east_deg,
        )
        mask = np.asarray(partial_periodic_seam_wall_latlon(
            grid,
            open_lat_south_deg=cfg.channel_lat_south_deg,
            open_lat_north_deg=cfg.channel_lat_north_deg,
            seam_column_index=0,
        ))
        # Outside the open band, the seam column is land.
        lat_1d_deg = np.degrees(np.asarray(grid.lat))
        for j, lat_deg in enumerate(lat_1d_deg):
            in_band = (
                cfg.channel_lat_south_deg
                <= lat_deg
                <= cfg.channel_lat_north_deg
            )
            if in_band:
                assert mask[j, 0] == 1.0
            else:
                assert mask[j, 0] == 0.0


# ---------------------------------------------------------------------
# Initial-state construction (Mercator path)
# ---------------------------------------------------------------------

class TestLatLonInitialState:
    def test_T_S_field_shapes(self):
        # Use a coarser grid for test speed (4-level z, 6 lon, ~12 lat).
        cfg = DINOConfig()
        grid = dino_lat_lon_grid(cfg, n_lon=6)
        zc = create_levy_stretched_z_star(
            n_levels=4, H_max=cfg.H_deep, dz_min=400.0,
            k_th=3.0, a_cr=2.0,
        )
        T, S, H_bathy, land_mask = dino_lat_lon_initial_state_arrays(
            grid, zc, cfg,
        )
        n_lat, n_lon = grid.n_lat, grid.n_lon
        assert T.shape == (n_lat, n_lon, 4)
        assert S.shape == (n_lat, n_lon, 4)
        assert H_bathy.shape == (n_lat, n_lon)
        assert land_mask.shape == (n_lat, n_lon)
        # Ocean depth on wet cells in [H_shallow, H_deep] (sill carve-out
        # may shoal below H_shallow on a few sill-ring cells — tolerate).
        ocean = np.asarray(land_mask) > 0.5
        Hb = np.asarray(H_bathy)[ocean]
        assert Hb.max() <= cfg.H_deep + 1e-6
        assert Hb.min() > 0.0

    def test_initial_T_S_columns(self):
        cfg = DINOConfig()
        zc = create_levy_stretched_z_star(
            n_levels=8, H_max=cfg.H_deep, dz_min=50.0,
            k_th=7.0, a_cr=2.0,
        )
        # Equator column: should match the 1D profile (uses cfg.lat_max_deg
        # in the meridional weight, so |φ|=0 ⇒ full equatorial profile).
        T_2d, S_2d = dino_initial_T_S(jnp.array(0.0), zc.z_full_ref, cfg)
        # Surface warmer than abyss.
        T_col = np.asarray(T_2d)
        assert T_col.shape == (8,)
        assert T_col[0] > T_col[-1]

    def test_initial_T_S_nemo_case4_bit_exact(self):
        # NEMO usr_def_istate CASE 4 is bit-reproduced by dino_initial_T_S
        # once the meridional-blend anchors are supplied from the model grid
        # (phi_max=MAXVAL(gphit), t_bot/s_bot=MINVAL over the WET field) —
        # the pure (lat,z) formula cannot see the tmask/gphit.
        # NEMO wp is float64; the 1e-12 bit-exact gate needs x64 (float32
        # residual is ~6e-7). Scoped to this test so the rest of the file
        # keeps the suite-default dtype.
        import jax
        prev = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        try:
            self._check_nemo_case4_bit_exact()
        finally:
            jax.config.update("jax_enable_x64", prev)

    def _check_nemo_case4_bit_exact(self):
        cfg = DINOConfig()

        # Reference 1-D depths (positive down) whose DEEPEST level is globally
        # dry — the full-step (ln_zco) staircase that makes MINVAL-over-wet
        # differ from T_1d[-1].
        z_pos = np.array([25.0, 120.0, 500.0, 1500.0, 3000.0, 4200.0])
        z_full_ref = -z_pos                       # legoESM: negative below surface
        lat = np.array([-62.0, -30.0, 0.0, 30.0, 62.0])  # max|lat| = 62 != 70
        # tmask: last level dry everywhere; row |lat|=62 also dry at level 4.
        tmask = np.ones((len(lat), len(z_pos)))
        tmask[:, -1] = 0.0
        tmask[[0, 4], -2] = 0.0

        # --- NEMO case-4 exact transcription (usrdef_istate.F90) ---
        T1d = np.asarray(dino_T_profile_1d(z_pos))
        S1d = np.asarray(dino_S_profile_1d(z_pos))
        T_uni = T1d[None, :] * tmask               # horizontally-uniform, masked
        S_uni = S1d[None, :] * tmask
        phi_max = np.abs(lat).max()                # MAXVAL(gphit)
        t_bot = (T_uni + 100.0 * (1.0 - tmask)).min()   # MINVAL over wet
        s_bot = (S_uni + 100.0 * (1.0 - tmask)).min()
        fac = (phi_max - np.abs(lat))[:, None] / phi_max
        T_nemo = ((T1d[None, :] - t_bot) * fac + t_bot) * tmask
        S_nemo = ((S1d[None, :] - s_bot) * fac + s_bot) * tmask

        # --- legoESM with the NEMO-matched overrides ---
        T_l, S_l = dino_initial_T_S(
            lat, z_full_ref, cfg,
            phi_max_deg=phi_max, t_bot=float(t_bot), s_bot=float(s_bot),
        )
        T_l = np.asarray(T_l) * tmask
        S_l = np.asarray(S_l) * tmask
        assert np.max(np.abs(T_l - T_nemo)) < 1e-12
        assert np.max(np.abs(S_l - S_nemo)) < 1e-12

        # --- default (no overrides) is BYTE-IDENTICAL to the legacy formula ---
        T_def, S_def = dino_initial_T_S(lat, z_full_ref, cfg)
        fac_def = (cfg.lat_max_deg - np.abs(lat))[:, None] / cfg.lat_max_deg
        T_legacy = (T1d[None, :] - T1d[-1]) * fac_def + T1d[-1]
        S_legacy = (S1d[None, :] - S1d[-1]) * fac_def + S1d[-1]
        np.testing.assert_array_equal(np.asarray(T_def), T_legacy)
        np.testing.assert_array_equal(np.asarray(S_def), S_legacy)

    def test_land_mask_override_default_is_byte_identical(self):
        # Default land_mask_override=None reproduces the analytic seam wall
        # exactly (backward-compat for the standalone bowl recipes).
        cfg = DINOConfig()
        grid = dino_lat_lon_grid(cfg, n_lon=6)
        zc = create_levy_stretched_z_star(
            n_levels=4, H_max=cfg.H_deep, dz_min=400.0, k_th=3.0, a_cr=2.0,
        )
        base = dino_lat_lon_initial_state_arrays(grid, zc, cfg)
        seam = dino_lat_lon_initial_state_arrays(
            grid, zc, cfg, land_mask_override=None)
        for a, b in zip(base, seam):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_land_mask_override_reentrant_keeps_seam_wet(self):
        # An i-periodic (ln_Iperio) override marks the whole domain wet ->
        # the westernmost column j-index 0 stays ocean (no seam wall) and
        # its T/S are NOT zeroed (no cold T=0 wall cell).
        cfg = DINOConfig()
        grid = dino_lat_lon_grid(cfg, n_lon=6)
        zc = create_levy_stretched_z_star(
            n_levels=4, H_max=cfg.H_deep, dz_min=400.0, k_th=3.0, a_cr=2.0,
        )
        # Seam wall (default) masks column 0 outside the channel band.
        _, _, _, lm_seam = dino_lat_lon_initial_state_arrays(grid, zc, cfg)
        assert float(np.asarray(lm_seam)[:, 0].sum()) < grid.n_lat  # some land
        # Re-entrant override: everything wet.
        wet = jnp.ones((grid.n_lat, grid.n_lon))
        T, S, _, lm = dino_lat_lon_initial_state_arrays(
            grid, zc, cfg, land_mask_override=wet)
        np.testing.assert_array_equal(np.asarray(lm), np.ones_like(np.asarray(lm)))
        # Column 0 T/S kept (not zeroed) since it is now wet.
        assert np.all(np.asarray(T)[:, 0, 0] != 0.0)

    def test_land_mask_override_reentrant_periodic_u_mask(self):
        # The full state built with a re-entrant override has periodic-
        # consistent zonal face masks (u_mask[:,0] == u_mask[:,-1] wrap) and
        # keeps the N/S walls (ln_Iperio: i-periodic, j-walled).
        from legoesm.ocean.experiments.dino import dino_lat_lon_state
        cfg = DINOConfig()
        grid = dino_lat_lon_grid(cfg, n_lon=6)
        zc = create_levy_stretched_z_star(
            n_levels=4, H_max=cfg.H_deep, dz_min=400.0, k_th=3.0, a_cr=2.0,
        )
        wet = jnp.ones((grid.n_lat, grid.n_lon))
        st = dino_lat_lon_state(grid, zc, cfg, land_mask_override=wet)
        um = np.asarray(st.u_mask.data)
        vm = np.asarray(st.v_mask.data)
        assert um.shape == (grid.n_lat, grid.n_lon + 1)
        # Re-entrant: the west-seam u-face (column 0) is OPEN (all wet) — the
        # meaningful contrast to the default seam wall, where those faces are
        # dry.  (The trailing wrap column mirrors column 0 by construction.)
        st_wall = dino_lat_lon_state(grid, zc, cfg)              # default seam wall
        um_wall = np.asarray(st_wall.u_mask.data)
        assert np.all(um[:, 0] > 0.5)                            # open seam
        assert np.any(um_wall[:, 0] < 0.5)                       # walled seam
        np.testing.assert_array_equal(um[:, 0], um[:, -1])       # wrap convention
        # Interior u-faces all wet (fully re-entrant, no seam wall).
        assert np.all(um > 0.5)
        # N/S boundary v-faces stay walls (not j-periodic).
        assert np.all(vm[0] < 0.5) and np.all(vm[-1] < 0.5)


# ---------------------------------------------------------------------
# create_initial_conditions dispatch
# ---------------------------------------------------------------------

class TestCreateInitialConditionsDispatch:
    def test_rejects_unknown_grid_type(self):
        cfg = DINOConfig()
        grid = dino_lat_lon_grid(cfg, n_lon=4)
        zc = create_levy_stretched_z_star(
            n_levels=4, H_max=cfg.H_deep, dz_min=400.0,
            k_th=3.0, a_cr=2.0,
        )
        with pytest.raises(ValueError, match="Unknown grid_type"):
            create_initial_conditions("cubed_sphere", grid, zc, cfg)


# ---------------------------------------------------------------------
# MPAS partial-periodic seam wall (optional — needs scipy)
# ---------------------------------------------------------------------

scipy = pytest.importorskip("scipy", reason="MPAS Voronoi mesh needs scipy")


class TestMPASSeamWall:
    def test_open_band_keeps_seam_cells_ocean(self):
        from legoesm.grids.voronoi import create_regional_voronoi_mesh
        from legoesm.ocean.init_mpas import partial_periodic_seam_wall_mpas
        cfg = DINOConfig()
        try:
            mesh = create_regional_voronoi_mesh(
                lon_range=(cfg.lon_west_deg, cfg.lon_east_deg),
                lat_range=(-cfg.lat_max_deg, cfg.lat_max_deg),
                resolution_km=200.0,
                periodic_x=True,
            )
        except (ValueError, RuntimeError) as exc:
            pytest.skip(f"Voronoi mesh generation unstable on this "
                        f"platform: {exc}")
        mask = np.asarray(partial_periodic_seam_wall_mpas(
            mesh,
            open_lat_south_deg=cfg.channel_lat_south_deg,
            open_lat_north_deg=cfg.channel_lat_north_deg,
            seam_lon_deg=cfg.lon_west_deg,
        ))
        lat_deg = np.degrees(np.asarray(mesh.latCell))
        lon_deg = (np.degrees(np.asarray(mesh.lonCell)) + 180.0) % 360.0 - 180.0
        # Cells near the seam outside the open band: expect at least
        # one to be land (mask = 0). Inside the band: expect at least
        # one near-seam cell to stay ocean.
        near_seam_dist_deg = (
            (lon_deg - cfg.lon_west_deg) % 360.0
        )
        seam_strip = near_seam_dist_deg < 5.0
        outside_band = (lat_deg < cfg.channel_lat_south_deg) | (
            lat_deg > cfg.channel_lat_north_deg
        )
        inside_band = (lat_deg >= cfg.channel_lat_south_deg) & (
            lat_deg <= cfg.channel_lat_north_deg
        )
        if (seam_strip & outside_band).any():
            assert (mask[seam_strip & outside_band] < 0.5).any()
        if (seam_strip & inside_band).any():
            assert (mask[seam_strip & inside_band] > 0.5).any()


# ---------------------------------------------------------------------
# Rigid-lid island topology (the ACC is the channel island constant)
# ---------------------------------------------------------------------

class TestDINORigidLidIslandTopology:
    """The re-entrant southern channel makes the DINO basin multiply-connected.

    The seam-wall land (column 0) is split by the open channel band into a
    NORTHERN segment and a SOUTHERN segment — two disconnected land masses.
    That is the Veros ACC topology: with two islands the rigid-lid streamfunction
    system has one free island circulation constant, and that constant IS the
    depth-integrated ACC transport.  This is why ``rigid_lid`` is the natural
    barotropic solver here (the ACC is a NULL mode of the free-surface eta solve,
    which is what made implicit_cn / explicit_substep under-/over-shoot + ring in
    the barotropic-solver audit).
    """

    def test_seam_wall_splits_into_two_land_masses(self):
        from legoesm.ocean.dynamics.rigid_lid_islands import _label_islands

        cfg = DINOConfig()
        grid = dino_lat_lon_grid(cfg, n_lon=20)           # small + cheap
        zc = create_levy_stretched_z_star(
            n_levels=4, H_max=cfg.H_deep, dz_min=400.0, k_th=3.0, a_cr=2.0)
        _, _, _, land_mask = dino_lat_lon_initial_state_arrays(grid, zc, cfg)
        cell_land = np.asarray(land_mask) < 0.5

        # All land is the seam column; the channel band rows are open there.
        assert cell_land[:, 1:].sum() == 0, "DINO land must be the seam column only"
        lat = np.degrees(np.asarray(grid.lat))
        chan = (lat > cfg.channel_lat_south_deg) & (lat < cfg.channel_lat_north_deg)
        assert not cell_land[chan, 0].any(), "channel band seam must be open ocean"
        assert cell_land[~chan, 0].any(), "seam outside the channel must be land"

        labels, nisle = _label_islands(cell_land, periodic_x=True)
        # Two land masses (northern + southern seam wall) ⇒ one free ACC constant.
        assert nisle == 2, (
            f"DINO re-entrant channel must yield exactly 2 land masses "
            f"(N + S seam walls) for the ACC island constant; got nisle={nisle}")
        # The two islands are the seam segments north / south of the channel.
        north = labels[(lat > cfg.channel_lat_north_deg), 0]
        south = labels[(lat < cfg.channel_lat_south_deg), 0]
        assert set(np.unique(north[north > 0])).isdisjoint(
            set(np.unique(south[south > 0]))), "N and S walls must be distinct islands"


class TestDINORigidLidFaithfulStack:
    """Selecting rigid_lid auto-applies the Veros-faithful coordinated stack.

    A BARE barotropic_solver="rigid_lid" flip runs away (568 Sv → NaN): the
    bottom-drag depth-mean reaches the streamfunction barotropic balance only via
    the du_diss fold, which is gated on ab2_scope="advective".  So
    dino_lat_lon_model_config pairs rigid_lid with ab2 + explicit_ab2 +
    ab2_scope="advective" + dt_mom_ratio (veros_acc_recipe.py); other solvers keep
    the forward_euler defaults.
    """

    def _model_cfg(self, solver):
        cfg = DINOConfig(barotropic_solver=solver)
        grid = dino_lat_lon_grid(cfg, n_lon=6)
        mcfg, _ = dino.dino_lat_lon_model_config(grid, cfg, physics=True)
        return mcfg

    def test_rigid_lid_applies_faithful_stack(self):
        m = self._model_cfg("rigid_lid")
        assert m.barotropic.barotropic_solver == "rigid_lid"
        assert m.outer_integrator == "ab2"
        assert m.coriolis_scheme == "explicit_ab2"
        assert m.ab2_scope == "advective"
        assert m.dt_mom_ratio == DINOConfig().rigid_lid_dt_mom_ratio == 9.0

    def test_implicit_cn_keeps_forward_euler_defaults(self):
        m = self._model_cfg("implicit_cn")
        assert m.outer_integrator == "forward_euler"
        assert m.coriolis_scheme == "matsuno_split"
        assert m.ab2_scope == "total"
        assert m.dt_mom_ratio == 1.0

    def test_rigid_lid_dt_mom_ratio_is_tunable(self):
        cfg = DINOConfig(barotropic_solver="rigid_lid", rigid_lid_dt_mom_ratio=5.0)
        grid = dino_lat_lon_grid(cfg, n_lon=6)
        mcfg, _ = dino.dino_lat_lon_model_config(grid, cfg, physics=True)
        assert mcfg.dt_mom_ratio == 5.0
# Level-1 exactness: seasonal forcing (usrdef_sbc ln_ann_cyc) + salt flux
# ---------------------------------------------------------------------

class TestSeasonalForcing:
    def test_seasonal_cosine_phases(self):
        from legoesm.ocean.experiments.dino import dino_seasonal_cosines
        day = 86400.0
        # c1 peaks at 21 June (day 171 of the 360-day year), c2 at 21 July.
        c1, c2 = dino_seasonal_cosines(171.0 * 24.0 * 3600.0)
        assert float(c1) == pytest.approx(1.0, abs=1e-12)
        c1_w, _ = dino_seasonal_cosines((171.0 + 180.0) * day)
        assert float(c1_w) == pytest.approx(-1.0, abs=1e-12)
        _, c2_p = dino_seasonal_cosines(201.0 * day)
        assert float(c2_p) == pytest.approx(1.0, abs=1e-12)
        # periodic over the 360-day year
        c1_a, _ = dino_seasonal_cosines(10.0 * day)
        c1_b, _ = dino_seasonal_cosines((10.0 + 360.0) * day)
        assert float(c1_a) == pytest.approx(float(c1_b), abs=1e-12)

    def test_T_star_seasonal_asymmetry_and_mean_consistency(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_T_star_annual_mean, dino_T_star_seasonal,
            dino_seasonal_cosines,
        )
        cfg = DINOConfig()
        day = 86400.0
        lat = jnp.array([-70.0, 0.0, 70.0])
        # boundary swings: north amp 3.0, south amp 0.5 (oracle asymmetry)
        T_jul = dino_T_star_seasonal(lat, 201.0 * day, cfg)   # c2 = +1
        T_jan = dino_T_star_seasonal(lat, 21.0 * day, cfg)    # c2 = -1
        assert float(T_jul[2] - T_jan[2]) == pytest.approx(2 * 3.0, abs=1e-9)
        assert float(T_jan[0] - T_jul[0]) == pytest.approx(2 * 0.5, abs=1e-9)
        # equator: profile=1 -> T* = T_eq, season-independent
        assert float(T_jul[1]) == pytest.approx(float(T_jan[1]), abs=1e-12)
        # zero-phase (c2=0) equals the annual-mean form
        # c2 = 0 at day 201 - 90 = 111
        t0 = 111.0 * day
        _, c2 = dino_seasonal_cosines(t0)
        assert abs(float(c2)) < 1e-9
        np.testing.assert_allclose(
            np.asarray(dino_T_star_seasonal(lat, t0, cfg)),
            np.asarray(dino_T_star_annual_mean(lat, cfg)), rtol=1e-9)

    def test_Q_sr_seasonal_mean_matches_annual_quadrature(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_Q_sr_annual_mean, dino_Q_sr_seasonal,
        )
        cfg = DINOConfig()
        lat = jnp.linspace(-70.0, 70.0, 15)
        # daily samples at t = d days reproduce the annual-mean fn's
        # quadrature EXACTLY (same discrete phases)
        days = np.arange(1, 361, dtype=np.float64)
        acc = np.zeros(15)
        for d in days:
            acc += np.asarray(dino_Q_sr_seasonal(lat, d * 86400.0, cfg))
        np.testing.assert_allclose(
            acc / 360.0, np.asarray(dino_Q_sr_annual_mean(lat, cfg)),
            rtol=1e-10)
        # polar night: high south lat in southern winter (c1=+1) -> 0
        q = dino_Q_sr_seasonal(jnp.array([-70.0]), 171.0 * 86400.0, cfg)
        assert float(q[0]) == 0.0

    def test_apply_seasonal_requires_time_and_changes_forcing(self):
        import dataclasses
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_grid,
            dino_lat_lon_state, dino_lat_lon_surface_forcing_arrays,
            apply_dino_lat_lon_surface_forcing,
        )
        cfg = dataclasses.replace(DINOConfig(), forcing_annual_cycle=True)
        z = create_dino_z_star(cfg)
        g = dino_lat_lon_grid(cfg, n_lon=8)
        st = dino_lat_lon_state(g, z, cfg)
        frc = dino_lat_lon_surface_forcing_arrays(g, cfg)
        with pytest.raises(ValueError, match="t_seconds"):
            apply_dino_lat_lon_surface_forcing(st, frc, z, cfg, 2700.0)
        day = 86400.0
        s_jun = apply_dino_lat_lon_surface_forcing(
            st, frc, z, cfg, 2700.0, t_seconds=171.0 * day)
        s_dec = apply_dino_lat_lon_surface_forcing(
            st, frc, z, cfg, 2700.0, t_seconds=351.0 * day)
        assert float(jnp.max(jnp.abs(s_jun.T.data - s_dec.T.data))) > 0.0
        # flag OFF: t_seconds ignored -> bit-identical to the legacy call
        cfg0 = dataclasses.replace(cfg, forcing_annual_cycle=False)
        a = apply_dino_lat_lon_surface_forcing(
            st, frc, z, cfg0, 2700.0, t_seconds=171.0 * day)
        b = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg0, 2700.0)
        np.testing.assert_array_equal(np.asarray(a.T.data),
                                      np.asarray(b.T.data))

    def test_mpas_apply_rejects_seasonal(self):
        import dataclasses
        from legoesm.ocean.experiments.dino import (
            DINOConfig, apply_dino_mpas_surface_forcing,
        )
        cfg = dataclasses.replace(DINOConfig(), forcing_annual_cycle=True)
        with pytest.raises(NotImplementedError, match="lat-lon"):
            apply_dino_mpas_surface_forcing(None, None, None, cfg, 2700.0)

    def test_salt_and_heat_flux_coefficients_match_nemo(self):
        """A_S == |rn_srp| and A_theta == |rn_trp|; the top-layer salinity
        tendency equals NEMO's sfx/(rho0*e3t) with sfx = srp*(SSS-S*)."""
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_top_layer_S_tendency,
        )
        cfg = DINOConfig()
        assert cfg.A_S == pytest.approx(3.858e-3)     # -rn_srp
        assert cfg.A_theta == pytest.approx(40.0)     # -rn_trp
        S, S_star, dz0 = 35.4, 35.0, 10.0
        ours = float(dino_top_layer_S_tendency(
            jnp.asarray(S), jnp.asarray(S_star), dz0, cfg))
        srp = -3.858e-3
        nemo = srp * (S - S_star) / (cfg.rho_0 * dz0)
        assert ours == pytest.approx(nemo, rel=1e-12)
        assert ours < 0.0    # SSS above target -> freshening flux (sign walk)


class TestWindThroughStep:
    def test_taum_boost_field(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_surface_forcing_arrays,
        )
        cfg = DINOConfig()
        g = dino_lat_lon_grid(cfg, n_lon=8)
        frc = dino_lat_lon_surface_forcing_arrays(g, cfg)
        tau = np.asarray(frc["tau_u_cell_2d"][:, 0])
        taum = np.asarray(frc["taum_2d"][:, 0])
        west = tau > 0.0
        np.testing.assert_allclose(taum[west], 1.3 * np.abs(tau[west]),
                                   rtol=1e-12)
        np.testing.assert_allclose(taum[~west], np.abs(tau[~west]),
                                   rtol=1e-12)
        assert west.any() and (~west).any()

    def test_tke_taum_override_changes_K(self):
        from legoesm.ocean.physics.vertical_mixing.tke import (
            tke_vertical_mixing,
        )
        from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
        nlev = 10
        shape = (4, 3, nlev)
        rng = np.random.default_rng(0)
        T = jnp.asarray(20.0 - 15.0 * np.linspace(0, 1, nlev)[None, None, :]
                        * np.ones(shape))
        S = jnp.full(shape, 35.0)
        u = jnp.zeros(shape); v = jnp.zeros(shape)
        rho = jnp.asarray(1026.0 - 2.0 * np.linspace(0, 1, nlev))[None, None, :] * jnp.ones(shape)
        dz_half = jnp.full(shape[:-1] + (nlev - 1,), 50.0)
        tau = jnp.full(shape[:-1], 0.2)
        out_plain = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, tke_old=None,
            tau_x_surface=tau, tau_y_surface=jnp.zeros_like(tau),
            dt=86400.0, cfg=TKEConfig(), rho_0=1026.0, g=9.80665,
            n_iterations=2)
        out_boost = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, tke_old=None,
            tau_x_surface=tau, tau_y_surface=jnp.zeros_like(tau),
            taum_surface=1.3 * tau,
            dt=86400.0, cfg=TKEConfig(), rho_0=1026.0, g=9.80665,
            n_iterations=2)
        dK = float(jnp.max(jnp.abs(out_boost.K_H - out_plain.K_H)))
        assert dK > 0.0    # boosted taum must strengthen the TKE input

    def test_apply_skips_wind_when_through_step(self):
        import dataclasses
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_grid,
            dino_lat_lon_state, dino_lat_lon_surface_forcing_arrays,
            apply_dino_lat_lon_surface_forcing,
        )
        cfg_off = DINOConfig()
        cfg_on = dataclasses.replace(cfg_off, wind_through_step=True)
        z = create_dino_z_star(cfg_off)
        g = dino_lat_lon_grid(cfg_off, n_lon=8)
        st = dino_lat_lon_state(g, z, cfg_off)
        frc = dino_lat_lon_surface_forcing_arrays(g, cfg_off)
        s_off = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg_off, 2700.0)
        s_on = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg_on, 2700.0)
        # flag ON: the applicator leaves u untouched (the step owns wind)
        np.testing.assert_array_equal(np.asarray(s_on.u.data),
                                      np.asarray(st.u.data))
        assert float(jnp.max(jnp.abs(s_off.u.data - st.u.data))) > 0.0

    def test_step_forcing_carries_wind_and_taum(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_surface_forcing_arrays,
            dino_step_surface_forcing,
        )
        cfg = DINOConfig()
        g = dino_lat_lon_grid(cfg, n_lon=8)
        frc = dino_lat_lon_surface_forcing_arrays(g, cfg)
        sf = dino_step_surface_forcing(frc)
        # atmospheric convention: the core applies -tau (ocean reaction)
        np.testing.assert_array_equal(np.asarray(sf.tau_x),
                                      -np.asarray(frc["tau_u_cell_2d"]))
        np.testing.assert_array_equal(np.asarray(sf.taum),
                                      np.asarray(frc["taum_2d"]))
        assert sf.q_net is None and sf.sw_down is None

    def test_model_step_applies_wind_from_rest(self):
        import dataclasses
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_grid,
            dino_lat_lon_state, dino_lat_lon_model_config,
            dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing,
        )
        cfg = dataclasses.replace(DINOConfig(), wind_through_step=True)
        z = create_dino_z_star(cfg)
        g = dino_lat_lon_grid(cfg, n_lon=10)
        st = dino_lat_lon_state(g, z, cfg)
        model = LatLonCGridOceanModel(
            g, z, dino_lat_lon_model_config(g, cfg, physics=True)[0])
        sf = dino_step_surface_forcing(
            dino_lat_lon_surface_forcing_arrays(g, cfg))
        s1 = model.step(st, dt=cfg.dt, surface_forcing=sf)
        u_top = np.asarray(s1.u.data[..., 0])
        assert np.isfinite(u_top).all()
        # westerlies (tau>0 around -45 lat) accelerate +u at the top layer,
        # with the UNBOOSTED momentum stress: pin the magnitude against the
        # analytic Euler kick tau*dt/(rho0*dz0) (RK3 staging + the implicit
        # friction shave it somewhat; a x1.3 taum leak into momentum or a
        # doubled application would leave the [0.5, 1.1] band).
        lat = np.degrees(np.asarray(g.lat))
        j = int(np.argmin(np.abs(lat - (-45.0))))
        got = u_top[j, 1:-1].mean()
        tau_row = float(np.asarray(
            dino_lat_lon_surface_forcing_arrays(g, cfg)["tau_u_cell_2d"])[j, 0])
        expect = tau_row * cfg.dt / (cfg.rho_0 * float(z.dz_ref[0]))
        assert got > 0.0
        assert 0.5 * expect < got < 1.1 * expect, (got, expect)


# ---------------------------------------------------------------------
# #1492 surface_tendency_placement="leapfrog_rhs" (NEMO tra_sbc Nnn-RHS
# placement vs the legacy "applied_now" pre-step state mutation)
# ---------------------------------------------------------------------

class TestSurfaceTendencyPlacement:
    """Global-closure audit (#1492 0.1) found the legacy placement retains
    only ~0.44 of the applied surface flux per step (the increment lands on
    BOTH sides of the leap-frog combine's ``state_expl.T - state.T`` and
    cancels there, entering only via the Asselin filter's "now" weight;
    closed-form MLF/Asselin recursion match: retention numerically 0.4444
    = 4/9 for gamma=0.1, see mlf_retention_algebra.py. NOTE the label
    "1/(1+2*gamma)" used in an earlier revision is WRONG -- it evaluates to
    0.833 at gamma=0.1; only the NUMBER 4/9 is right, reproduced twice by
    independent instruments on different grids). NEMO's
    ``trasbc.F90`` instead writes into the RHS accumulator BEFORE the
    ``tra_zdf``/leap-frog combine (stpmlf.F90:342 ``tra_sbc(kstp, Nnn, ts,
    Nrhs)``). These tests pin the new "leapfrog_rhs" placement end-to-end
    on the real DINO MLF card.
    """

    def test_default_is_applied_now(self):
        assert DINOConfig().surface_tendency_placement == "applied_now"

    def test_return_rate_does_not_mutate_tracers_and_matches_legacy_delta(self):
        from legoesm.ocean.experiments.dino import (
            create_dino_z_star, dino_lat_lon_grid, dino_lat_lon_state,
            dino_lat_lon_surface_forcing_arrays, dino_config_for_recipe,
        )
        cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
        z = create_dino_z_star(cfg)
        g = dino_lat_lon_grid(cfg, n_lon=8)
        st = dino_lat_lon_state(g, z, cfg)
        frc = dino_lat_lon_surface_forcing_arrays(g, cfg)
        dt = cfg.dt

        # #1492 C2: the placement field is load-bearing, so each route is
        # called with the cfg that DECLARES it. The point of the test is
        # unchanged: the two routes must compute the SAME tendency and differ
        # only in where it is consumed.
        cfg_legacy = dataclasses.replace(
            cfg, surface_tendency_placement="applied_now")
        cfg_rhs = dataclasses.replace(
            cfg, surface_tendency_placement="leapfrog_rhs")
        st_legacy = dino.apply_dino_lat_lon_surface_forcing(
            st, frc, z, cfg_legacy, dt, t_seconds=dt)
        st_rate, (dT_dt, dS_dt) = dino.apply_dino_lat_lon_surface_forcing(
            st, frc, z, cfg_rhs, dt, t_seconds=dt, return_rate=True)
        # T/S UNCHANGED in return_rate mode (only u still gets its wind kick).
        np.testing.assert_array_equal(np.asarray(st_rate.T.data),
                                      np.asarray(st.T.data))
        np.testing.assert_array_equal(np.asarray(st_rate.S.data),
                                      np.asarray(st.S.data))
        # The returned RATE, applied over dt, reproduces the legacy delta
        # to machine precision (same tendency, different consumption site).
        expect_dT = (np.asarray(st_legacy.T.data) - np.asarray(st.T.data)) / dt
        expect_dS = (np.asarray(st_legacy.S.data) - np.asarray(st.S.data)) / dt
        assert float(np.max(np.abs(np.asarray(dT_dt) - expect_dT))) < 1e-10
        assert float(np.max(np.abs(np.asarray(dS_dt) - expect_dS))) < 1e-10

    def test_external_tracer_rate_requires_leapfrog(self):
        """Dispatch hardening: passing external_tracer_rate under any outer
        integrator except leap-frog would be a SILENT no-op (only the MLF
        Nnn pass reads ``_external_tracer_rate``) -- must raise instead."""
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.experiments.dino import (
            create_dino_z_star, dino_lat_lon_grid, dino_lat_lon_state,
            dino_lat_lon_model_config, dino_lat_lon_surface_forcing_arrays,
            dino_config_for_recipe,
        )
        cfg = dino_config_for_recipe("nemo_dino_kamm")  # forward-Euler card
        z = create_dino_z_star(cfg)
        g = dino_lat_lon_grid(cfg, n_lon=8)
        mc, _ = dino_lat_lon_model_config(g, cfg, physics=True)
        assert mc.outer_integrator == "forward_euler"
        model = LatLonCGridOceanModel(g, z, mc)
        st = dino_lat_lon_state(g, z, cfg)
        frc = dino_lat_lon_surface_forcing_arrays(g, cfg)
        # #1492 C2: obtaining a rate requires the cfg to DECLARE the rhs
        # route (the applier now raises on a placement/return_rate mismatch);
        # the point under test is the SEPARATE guard that rejects that rate
        # when the model's outer integrator is not leap-frog.
        cfg_rhs = dataclasses.replace(
            cfg, surface_tendency_placement="leapfrog_rhs")
        _, rate = dino.apply_dino_lat_lon_surface_forcing(
            st, frc, z, cfg_rhs, cfg.dt, t_seconds=cfg.dt, return_rate=True)
        with pytest.raises(ValueError, match="leap-frog"):
            model.step(st, dt=cfg.dt, external_tracer_rate=rate)

    def test_retention_synthetic_violation_both_directions(self):
        """The decisive check (#1492 STEP 2/3): run the SAME few-step DINO
        MLF trajectory under both placements with a real (non-zero)
        restoring tendency injected every step, and confirm:
          - "applied_now" (legacy): retains well BELOW half the applied
            surface heat increment in the global-mean top-layer T (matches
            the 0.1 audit's measured ~0.44 coefficient, well under 0.6).
          - "leapfrog_rhs" (new): retains well ABOVE half -- most of the
            RHS-injected tendency survives into the trajectory, as NEMO's
            placement does.
        A regression here (retention collapsing back toward the legacy
        value under "leapfrog_rhs") means the RHS injection silently
        stopped reaching the leap-frog combine.
        """
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.experiments.dino import (
            dino_lat_lon_grid, dino_lat_lon_vertical, dino_lat_lon_state,
            dino_lat_lon_model_config, dino_lat_lon_surface_forcing_arrays,
            dino_config_for_recipe,
        )
        cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
        g = dino_lat_lon_grid(cfg, n_lon=10)
        z = dino_lat_lon_vertical(g, cfg)  # MLF card needs its matching
                                           # partial-cell/masked-zco coord,
                                           # not the bare z* helper.
        st0 = dino_lat_lon_state(g, z, cfg)
        frc = dino_lat_lon_surface_forcing_arrays(g, cfg)
        mc, _ = dino_lat_lon_model_config(g, cfg, physics=True)
        model = LatLonCGridOceanModel(g, z, mc)
        dt = cfg.dt
        mean_mask = np.asarray(st0.land_mask.data) > 0.5
        n_steps = 6

        def run(placement):
            # #1492 C2: the config field is now LOAD-BEARING -- the applier
            # raises if cfg.surface_tendency_placement disagrees with the
            # return_rate route, so each direction must carry its own cfg.
            cfg_p = dataclasses.replace(cfg,
                                        surface_tendency_placement=placement)
            s = st0
            applied_total = 0.0
            for k in range(n_steps):
                t_next = (k + 1) * dt
                if placement == "leapfrog_rhs":
                    s, (rT, rS) = dino.apply_dino_lat_lon_surface_forcing(
                        s, frc, z, cfg_p, dt, t_seconds=t_next, return_rate=True)
                    applied_total += float(np.mean(
                        np.asarray(rT)[..., 0][mean_mask])) * dt
                    s = model.step(s, dt=dt, external_tracer_rate=(rT, rS))
                else:
                    s_before = s
                    s = dino.apply_dino_lat_lon_surface_forcing(
                        s, frc, z, cfg_p, dt, t_seconds=t_next)
                    applied_total += float(np.mean(
                        (np.asarray(s.T.data) - np.asarray(s_before.T.data)
                         )[..., 0][mean_mask]))
                    s = model.step(s, dt=dt)
            return s, applied_total

        s_now, applied_now = run("applied_now")
        s_rhs, applied_rhs = run("leapfrog_rhs")
        t0_mean = float(np.mean(np.asarray(st0.T.data)[..., 0][mean_mask]))
        d_now = float(np.mean(np.asarray(s_now.T.data)[..., 0][mean_mask])) - t0_mean
        d_rhs = float(np.mean(np.asarray(s_rhs.T.data)[..., 0][mean_mask])) - t0_mean
        assert applied_now > 0.0 and applied_rhs > 0.0
        retention_now = d_now / applied_now
        retention_rhs = d_rhs / applied_rhs
        assert retention_now < 0.6, retention_now
        assert retention_rhs > 0.6, retention_rhs


def test_S_star_boundary_targets_match_oracle():
    """codex catch (dino-s12 #1): rn_sstar_s=35.0, rn_sstar_n=35.1 — the
    DINOConfig defaults had them SWAPPED.  Pin the boundary values (the
    cos-profile term vanishes at |phi|=70 where cos(2*pi*70/140)=-1 makes
    the profile factor 0; the Gaussian dip is negligible there)."""
    from legoesm.ocean.experiments.dino import DINOConfig, dino_S_star
    cfg = DINOConfig()
    assert cfg.S_star_s == 35.0 and cfg.S_star_n == 35.1
    s_s = float(dino_S_star(jnp.asarray(-70.0), cfg))
    s_n = float(dino_S_star(jnp.asarray(70.0), cfg))
    assert s_s == pytest.approx(35.0, abs=1e-6)
    assert s_n == pytest.approx(35.1, abs=1e-6)


def test_mpas_builder_rejects_fct_family_tracer_advection():
    """The r1_exact preset (fct2) must fail LOUDLY at the MPAS config
    builder with actionable guidance, not deep in model construction."""
    import pytest as _pytest

    from legoesm.ocean.experiments.dino import (
        dino_mpas_model_config, dino_r1_exact_config,
    )

    cfg = dino_r1_exact_config()
    with _pytest.raises(ValueError, match="lat-lon only"):
        dino_mpas_model_config(None, cfg, physics=False)


class TestMaskedZco:
    """NEMO ln_zco full-cell masking (zgr_msk_top_bot) for DINO."""

    def _z_and_bowl(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_bowl,
            dino_lat_lon_grid,
        )
        cfg = DINOConfig()
        z = create_dino_z_star(cfg)
        g = dino_lat_lon_grid(cfg, n_lon=12)
        return cfg, z, np.asarray(dino_lat_lon_bowl(g, cfg))

    @staticmethod
    def _mi96_f90(jpk, H, dzmin, kth, acr):
        """INDEPENDENT numpy transliteration of zgr_lib.F90 mi96_1d
        (kkconst=0, ph_co=0 — the 1-D reference ladder DINO's ln_zco
        uses): returns (pdepw_1d[jpk], pdept_1d[jpk]) positive down."""
        import math
        jpkm1 = jpk - 1
        za1 = ((dzmin - H / jpkm1)
               / (math.tanh((1 - kth) / acr)
                  - acr / jpkm1 * (math.log(math.cosh((jpk - kth) / acr))
                                   - math.log(math.cosh((1 - kth) / acr)))))
        za0 = dzmin - za1 * math.tanh((1 - kth) / acr)
        zsur = -za0 - za1 * acr * math.log(math.cosh((1 - kth) / acr))
        w = np.array([zsur + za0 * k
                      + za1 * acr * math.log(math.cosh((k - kth) / acr))
                      for k in range(1, jpk + 1)])
        t = np.array([zsur + za0 * (k + 0.5)
                      + za1 * acr * math.log(math.cosh((k + 0.5 - kth) / acr))
                      for k in range(1, jpk + 1)])
        return w, t

    def test_ladder_matches_f90_and_reference_run(self):
        """The masked-zco ladder must equal the mi96_1d transliteration
        AND the reference run's deptht (first/last wet values hardcoded
        from DINO_1m_grid_T.nc, float32 storage)."""
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_vertical,
        )
        import dataclasses
        cfg = dataclasses.replace(DINOConfig(),
                                  vertical_coordinate="masked_zco")
        g = dino_lat_lon_grid(cfg, n_lon=12)
        coord = dino_lat_lon_vertical(g, cfg)
        assert coord.n_levels == cfg.n_levels - 1     # NEMO jpk dummy level

        w_f90, t_f90 = self._mi96_f90(
            cfg.n_levels, cfg.H_deep, cfg.dz_min, float(cfg.k_th),
            cfg.a_cr)
        np.testing.assert_allclose(
            np.abs(np.asarray(coord.z_half_ref))[1:], w_f90[1:], rtol=1e-9)
        np.testing.assert_allclose(
            np.abs(np.asarray(coord.z_full_ref)), t_f90[:-1], rtol=1e-9)
        # reference-run oracle (deptht, f32): first two + last wet centre
        np.testing.assert_allclose(
            np.abs(np.asarray(coord.z_full_ref))[[0, 1, -1]],
            [5.0335817, 15.322634, 3757.309], rtol=1e-6)

    def test_snap_rule_matches_f90_transliteration(self):
        """k_bot per usrdef_zgr.F90 zgr_msk_top_bot:
        WHERE( pdept(jk) < H .AND. H <= pdept(jk+1) ) k_bot = jk,
        with pdept from the INDEPENDENT mi96 transliteration."""
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_bowl, dino_lat_lon_grid,
            dino_lat_lon_vertical,
        )
        cfg = dataclasses.replace(DINOConfig(),
                                  vertical_coordinate="masked_zco")
        g = dino_lat_lon_grid(cfg, n_lon=12)
        H = np.asarray(dino_lat_lon_bowl(g, cfg))
        coord = dino_lat_lon_vertical(g, cfg)

        _, pdept = self._mi96_f90(
            cfg.n_levels, cfg.H_deep, cfg.dz_min, float(cfg.k_th),
            cfg.a_cr)
        jpkm1 = cfg.n_levels - 1
        k_bot = np.zeros(H.shape, dtype=int)   # 0 = land (k_top=0)
        for jk in range(jpkm1):                # NEMO: 1..jpkm1 wet
            sel = (pdept[jk] < H) & (H <= pdept[jk + 1])
            k_bot[sel] = jk + 1                # NEMO 1-based level count
        np.testing.assert_array_equal(
            np.asarray(coord.bottom_level) + 1, k_bot)

    def test_full_cells_and_snap_depth(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_bowl, dino_lat_lon_grid,
            dino_lat_lon_vertical, dino_masked_zco_coordinate,
        )
        cfg = dataclasses.replace(DINOConfig(),
                                  vertical_coordinate="masked_zco")
        g = dino_lat_lon_grid(cfg, n_lon=12)
        H = np.asarray(dino_lat_lon_bowl(g, cfg))
        coord = dino_lat_lon_vertical(g, cfg)
        H_snap = np.asarray(coord.h_partial).sum(-1)
        h = np.asarray(coord.h_partial)
        dz = np.asarray(coord.dz_ref)
        # every wet cell is a FULL cell; below-bottom cells are zero
        is_full = np.isclose(h, dz[None, None, :], rtol=0, atol=1e-9)
        is_zero = h == 0.0
        assert bool(np.all(is_full | is_zero))
        # and the Jacobian is 1 at eta=0 on wet columns
        from legoesm.ocean.vertical import compute_ocean_jacobian
        J = np.asarray(compute_ocean_jacobian(
            jnp.zeros(H.shape), jnp.asarray(H_snap), coord))
        wet = np.asarray(coord.bottom_level) >= 0
        np.testing.assert_allclose(J[wet], 1.0, rtol=0, atol=1e-12)

    def test_dispatch(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_vertical,
        )
        from legoesm.ocean.vertical import (
            OceanPartialCellCoordinate, OceanZStarCoordinate,
        )
        cfg = DINOConfig()
        g = dino_lat_lon_grid(cfg, n_lon=12)
        assert isinstance(dino_lat_lon_vertical(g, cfg),
                          OceanZStarCoordinate)
        cfg2 = dataclasses.replace(cfg, vertical_coordinate="masked_zco")
        assert isinstance(dino_lat_lon_vertical(g, cfg2),
                          OceanPartialCellCoordinate)
        cfg3 = dataclasses.replace(cfg, vertical_coordinate="sigma")
        with pytest.raises(ValueError, match="vertical_coordinate"):
            dino_lat_lon_vertical(g, cfg3)

    def test_state_H_bathy_matches_coordinate(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_state,
            dino_lat_lon_vertical,
        )
        cfg = dataclasses.replace(DINOConfig(),
                                  vertical_coordinate="masked_zco")
        g = dino_lat_lon_grid(cfg, n_lon=12)
        z = dino_lat_lon_vertical(g, cfg)
        st = dino_lat_lon_state(g, z, cfg)
        # state fields are stored float32 -> f32-appropriate tolerance
        np.testing.assert_allclose(
            np.asarray(st.H_bathy.data),
            np.asarray(z.h_partial).sum(-1), rtol=1e-6)

    def test_mpas_builder_rejects_masked_zco(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_mpas_model_config,
        )
        cfg = dataclasses.replace(DINOConfig(),
                                  vertical_coordinate="masked_zco")
        with pytest.raises(ValueError, match="lat-lon only"):
            dino_mpas_model_config(None, cfg, physics=False)


class TestIsoneutralRediOnly:
    """lateral_tracer_mixing='isoneutral' — NEMO ln_traldf_iso(+msc)."""

    def _preset_model_cfg(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            dino_lat_lon_grid, dino_lat_lon_model_config,
            dino_r1_exact_config,
        )
        cfg = dino_r1_exact_config()
        g = dino_lat_lon_grid(cfg, n_lon=12)
        mc, _ = dino_lat_lon_model_config(g, cfg, physics=True)
        return cfg, g, mc

    def test_preset_builds_redi_only(self):
        cfg, g, mc = self._preset_model_cfg()
        gm = mc.gm_redi
        assert gm is not None
        assert gm.kappa_GM == 0.0
        assert gm.kappa_Redi > 0.0
        assert gm.kappa_redi_lat_scaling is True
        assert gm.S_max == 0.01
        assert gm.slope_density == "neutral"
        assert gm.implicit_K33 is True
        assert not gm.visbeck.enabled and not gm.treguier.enabled
        assert mc.K_h == 0.0                    # no iso-level double-count
        # kappa_Redi equals the legacy K_h coefficient (½·U_T·R·dλ)
        import numpy as _np

        from legoesm import constants as _c
        expect = 0.5 * cfg.U_T * _c.R_earth * float(_np.asarray(g.dlon))
        assert gm.kappa_Redi == pytest.approx(expect, rel=1e-12)

    def test_legacy_default_unchanged(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_model_config,
        )
        cfg = DINOConfig()          # geopotential + use_gm_redi=True
        g = dino_lat_lon_grid(cfg, n_lon=12)
        mc, _ = dino_lat_lon_model_config(g, cfg, physics=True)
        assert mc.K_h > 0.0
        assert mc.gm_redi.visbeck.enabled       # historical adaptive path
        assert mc.gm_redi.kappa_redi_lat_scaling is False

    def test_unknown_mixing_raises(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_model_config,
        )
        cfg = dataclasses.replace(DINOConfig(),
                                  lateral_tracer_mixing="epineutral")
        g = dino_lat_lon_grid(cfg, n_lon=12)
        with pytest.raises(ValueError, match="lateral_tracer_mixing"):
            dino_lat_lon_model_config(g, cfg, physics=True)

    def test_iso_plus_eiv_combined(self):
        # isoneutral + EIV is the FULL NEMO namtra_ldf + namtra_eiv combo
        # (DINO Kamm: static Redi aht=1/2*Ud*e1(phi) + Treguier-21 GM capped
        # at aei0=1/2*Ue*Le) — wired 2026-07-19 (previously raised). Lock:
        # no geopotential K_h, cos-scaled Redi, adaptive kappa enabled.
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_model_config,
        )
        cfg = dataclasses.replace(
            DINOConfig(), lateral_tracer_mixing="isoneutral",
            use_gm_redi=True, gm_kappa_scheme="treguier",
            treguier_aei0=1500.0)
        g = dino_lat_lon_grid(cfg, n_lon=12)
        mc, _ = dino_lat_lon_model_config(g, cfg, physics=True)
        assert mc.K_h == 0.0
        assert mc.gm_redi.kappa_redi_lat_scaling
        assert mc.gm_redi.treguier.enabled
        assert mc.gm_redi.treguier.aei0 == 1500.0

    def test_static_kappa_override_row_scaling(self):
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            static_kappa_redi_override,
        )
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid,
        )
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        g = dino_lat_lon_grid(DINOConfig(), n_lon=12)
        gm_on = GMRediConfig(kappa_Redi=100.0, kappa_redi_lat_scaling=True)
        arr, arr_v = static_kappa_redi_override(gm_on, g)
        assert arr.shape == (g.n_lat, 12)
        assert arr_v.shape == (g.n_lat, 12)
        lat = np.asarray(g.lat)
        # grid.lat is stored float32 -> f32-appropriate tolerance
        np.testing.assert_allclose(
            np.asarray(arr)[:, 0], 100.0 * np.cos(lat), rtol=1e-6)
        # v-face (#1226 tier-2 item 1): NEMO evaluates ahtv INDEPENDENTLY at
        # the v-point (ldftra.F90:325-329 -> ldfc1d_c2d.F90:141-145,
        # ahtv=zUfac*MAX(e1v,e2v)**inn), NOT ahtu broadcast onto the v-face —
        # ground-truth against grid.cos_lat_v at the north-face-of-cell-j
        # convention (matches grid.dx_v[1:,:] used by the operator).
        cos_lat_v = np.asarray(g.cos_lat_v)[1:]
        np.testing.assert_allclose(
            np.asarray(arr_v)[:, 0], 100.0 * cos_lat_v, rtol=1e-6)
        # The two must differ (this is the whole point of the fix) except at
        # the equator-straddling row where cos(lat_T) and cos(lat_v) coincide
        # by symmetry.
        assert not np.allclose(np.asarray(arr)[:, 0], np.asarray(arr_v)[:, 0])
        gm_off = GMRediConfig(kappa_Redi=100.0)
        assert static_kappa_redi_override(gm_off, g) == (None, None)

    def test_static_kappa_override_on_cgrid_geometry(self):
        """cos_lat_v on LatLonCGridGeometry -- the from-rest path.

        Regression for the #1226 tier-2 ahtv fix, which read grid.cos_lat_v
        directly and so crashed EVERY from-rest run while every test passed
        (the tests all used the bridged LatLonGrid, which stores it).
        """
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            static_kappa_redi_override,
        )
        from legoesm.grids.latlon import create_latlon_geometry
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid,
        )
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        g = dino_lat_lon_grid(DINOConfig(), n_lon=12)
        geom = create_latlon_geometry(
            g.n_lat, 12, radius=float(g.radius),
            lat_1d=jnp.asarray(g.lat), lon_1d=jnp.asarray(g.lon),
            lat_face_1d=jnp.asarray(g.lat_v))
        assert geom.cos_lat_v.shape == (g.n_lat + 1,)
        gm_on = GMRediConfig(kappa_Redi=100.0, kappa_redi_lat_scaling=True)
        arr, arr_v = static_kappa_redi_override(gm_on, geom)
        assert arr.shape == (g.n_lat, 12) and arr_v.shape == (g.n_lat, 12)
        # Same quantity as the LatLonGrid answer -- cos at the TRUE v-face
        # latitude.  Agreement to f32 eps, not bit-exact, only because the
        # two cast at different points (grid casts lat_v then cos; the
        # geometry cos's the f64 faces then casts).  A midpoint rebuild
        # would instead be off by 2.5e-4 here -- 3 orders larger.
        ref, ref_v = static_kappa_redi_override(gm_on, g)
        np.testing.assert_allclose(np.asarray(arr_v), np.asarray(ref_v),
                                   rtol=1e-6)
        np.testing.assert_allclose(np.asarray(arr), np.asarray(ref), rtol=1e-6)

    def test_static_kappa_override_refuses_tripole(self):
        """Tripole has no 1-D v-face axis -- refuse, never fabricate ahtv.

        The ndim!=1 guard does not catch tripole (it stores a zonal-mean 1-D
        lat), so without this the consumer would silently build ahtv from a
        half-cell reconstruction of zonal-mean latitudes.
        """
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            static_kappa_redi_override,
        )
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        from legoesm.grids.latlon import create_latlon_geometry
        g = create_latlon_geometry(8, 12)
        tri_like = g._replace(fold=g.fold._replace(is_active=True))
        gm_on = GMRediConfig(kappa_Redi=100.0, kappa_redi_lat_scaling=True)
        with pytest.raises(ValueError, match="no 1-D v-face axis"):
            static_kappa_redi_override(gm_on, tri_like)

    def test_static_kappa_override_is_jit_safe(self):
        """The override runs INSIDE jit -- it must never inspect array values.

        Regression: a NaN-scanning tripole guard raised
        TracerArrayConversionError and killed every production run.
        """
        import jax
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            static_kappa_redi_override,
        )
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        from legoesm.grids.latlon import create_latlon_geometry
        g = create_latlon_geometry(8, 12)
        gm_on = GMRediConfig(kappa_Redi=100.0, kappa_redi_lat_scaling=True)
        # Production shape: the geometry is closed over with STATIC ints but
        # z-star-live ARRAY leaves, so cos_lat_v arrives as a tracer.
        f = jax.jit(lambda cv, lt: static_kappa_redi_override(
            gm_on, g._replace(cos_lat_v=cv, lat=lt)))
        kT, kv = f(jnp.asarray(g.cos_lat_v), jnp.asarray(g.lat))
        assert kT.shape == (8, 12) and kv.shape == (8, 12)

    def test_mpas_builder_rejects_isoneutral(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_mpas_model_config,
        )
        cfg = dataclasses.replace(DINOConfig(),
                                  lateral_tracer_mixing="isoneutral",
                                  use_gm_redi=False)
        with _pytest_raises_valueerror("MPAS DINO path"):
            dino_mpas_model_config(None, cfg, physics=False)

    def test_lat_scaling_rejects_2d_latitudes(self):
        from types import SimpleNamespace

        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            static_kappa_redi_override,
        )
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        gm = GMRediConfig(kappa_Redi=100.0, kappa_redi_lat_scaling=True)
        fake = SimpleNamespace(lat=np.zeros((4, 5)), n_lon=5)
        with pytest.raises(ValueError, match="1-D latitudes"):
            static_kappa_redi_override(gm, fake)


def _pytest_raises_valueerror(match):
    return pytest.raises(ValueError, match=match)


class TestSlopeLimitNemoCap:
    """slope_limit='nemo_cap': slope capped at S_max, taper == 1."""

    def _w_triads(self, slope_limit):
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            _w_triad_slopes_tapers,
        )
        rng = np.random.default_rng(3)
        n_lat, n_lon, nlev = 4, 5, 6
        drho_dx_u = jnp.asarray(rng.normal(size=(n_lat, n_lon + 1, nlev)))
        drho_dy_v = jnp.asarray(rng.normal(size=(n_lat + 1, n_lon, nlev)))
        # weak stratification -> raw slopes far beyond S_max
        drho_dz_w = jnp.full((n_lat, n_lon, nlev - 1), -1e-3)
        return _w_triad_slopes_tapers(
            drho_dx_u, drho_dy_v, drho_dz_w, n_lat, n_lon,
            S_max=0.01, taper_width_frac=0.1, slope_limit=slope_limit)

    def test_cap_bounds_slopes_and_unit_tapers(self):
        out = self._w_triads("nemo_cap")
        slopes, tapers = out[:8], out[8:]
        for sl in slopes:
            assert float(jnp.abs(sl).max()) <= 0.01 + 1e-15
        for tp in tapers:
            np.testing.assert_array_equal(np.asarray(tp), 1.0)

    def test_dm95_tapers_at_steep_slopes(self):
        """dm95 mode: the taper suppresses the flux at steep slopes
        (~0.5 at the in-situ clip point) — the behaviour nemo_cap
        replaces with unit tapers."""
        out = self._w_triads("dm95_taper")
        tapers = out[8:]
        assert float(jnp.stack(tapers).min()) <= 0.55
        assert float(jnp.stack(tapers).max()) < 1.0

    def test_unknown_slope_limit_raises(self):
        with pytest.raises(ValueError, match="slope_limit"):
            self._w_triads("gerdes")

    def test_centered_scheme_accepts_cap(self):
        # nemo_cap is wired for the centered slope path since 2026-07-16
        # (previously it raised; the DM95 taper killed the flux at steep
        # ML-base outcrops where NEMO's cap keeps pumping — plan §G). Lock
        # that centered + nemo_cap runs and returns finite tendencies.
        from legoesm.ocean.experiments.dino import (
            dino_lat_lon_grid, dino_lat_lon_state, dino_lat_lon_vertical,
            dino_r1_exact_config,
        )
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            gm_redi_tracer_tendency_latlon,
        )
        cfg = dino_r1_exact_config()
        g = dino_lat_lon_grid(cfg, n_lon=12)
        z = dino_lat_lon_vertical(g, cfg)
        st = dino_lat_lon_state(g, z, cfg)
        from legoesm.ocean.experiments.dino import dino_lat_lon_model_config
        mc, _ = dino_lat_lon_model_config(g, cfg, physics=True)
        # msc_stabilize (ln_traldf_msc, PR #1233) is nemo_iso_lap-only —
        # clear it when switching to the centered scheme (guarded).
        gm_centered = mc.gm_redi._replace(
            slope_scheme="centered", msc_stabilize=False)
        assert gm_centered.slope_limit == "nemo_cap"
        dT_dt, dS_dt = gm_redi_tracer_tendency_latlon(
            st.T.data, st.S.data, st.eta.data, st.H_bathy.data,
            g, z, gm_centered, eos=mc.eos)
        assert bool(jnp.all(jnp.isfinite(dT_dt)))
        assert bool(jnp.all(jnp.isfinite(dS_dt)))

    def test_preset_selects_cap(self):
        from legoesm.ocean.experiments.dino import (
            dino_lat_lon_grid, dino_lat_lon_model_config,
            dino_r1_exact_config,
        )
        cfg = dino_r1_exact_config()
        assert cfg.redi_slope_limit == "nemo_cap"
        g = dino_lat_lon_grid(cfg, n_lon=12)
        mc, _ = dino_lat_lon_model_config(g, cfg, physics=True)
        assert mc.gm_redi.slope_limit == "nemo_cap"

    def test_mpas_gm_redi_rejects_nemo_cap(self):
        """MPAS centred GM/Redi must raise on slope_limit='nemo_cap'
        rather than silently keep the DM95 taper (codex r7 P1)."""
        from types import SimpleNamespace

        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
            gm_redi_tracer_tendency_mpas,
        )
        cfg = GMRediConfig(slope_limit="nemo_cap")
        with pytest.raises(NotImplementedError, match="slope_limit"):
            # entry-guard fires before any mesh access
            gm_redi_tracer_tendency_mpas(
                None, None, None, None, None, None, cfg)


class TestNemoCentredBarotropic:
    """dynspg_ts ln_bt_fw=F + nn_bt_flt=2 forward-frame reduction."""

    def test_boxcar_window_centred_at_new_time(self):
        from legoesm.ocean.dynamics.barotropic_common import (
            compute_nemo_boxcar_centred_weights,
        )
        n = 12
        w, w_total, w_tr, n_loop = compute_nemo_boxcar_centred_weights(
            n, jnp.float64)
        w = np.asarray(w)
        assert n_loop == w.size
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-14)
        # F90 transliteration (ts_wgt CASE(2)): zwgt1(jn)=1 where |jn-n|/n < 1
        jn = np.arange(1, n_loop + 1, dtype=float)
        expect = (np.abs(jn - n) / n < 1.0).astype(float)
        expect = expect / expect.sum()
        np.testing.assert_allclose(w, expect, rtol=1e-14)
        # centroid at the baroclinic step (tau = 1); symmetric window
        centroid = (w * jn / n).sum()
        np.testing.assert_allclose(centroid, 1.0, rtol=0, atol=0.05)
        # nn_bt_flt=2 boxcar: full width 2n -> loop runs jn=1..2n-1
        assert n_loop == 2 * n - 1

    def test_transport_weights_continuity_telescoping(self):
        """w_transport[j] = sum(w[j:]) / n — the unique choice with
        div(Hu_avg) == (eta_old - eta_avg)/dt (uniform tracer)."""
        from legoesm.ocean.dynamics.barotropic_common import (
            compute_nemo_boxcar_centred_weights,
        )
        n = 9
        w, _, w_tr, n_loop = compute_nemo_boxcar_centred_weights(
            n, jnp.float64)
        w, w_tr = np.asarray(w), np.asarray(w_tr)
        expect = np.array([w[i:].sum() for i in range(n_loop)]) / n
        np.testing.assert_allclose(w_tr, expect, rtol=1e-14)

    def test_auto_substeps_formula(self):
        from legoesm.ocean.dynamics.barotropic_common import (
            nemo_auto_substeps,
        )
        import math

        from legoesm import constants as _c
        dt, H, e1, e2 = 2700.0, 4000.0, 1.1e5, 1.1e5
        inv = 1.0 / e1 ** 2 + 1.0 / e2 ** 2
        n = nemo_auto_substeps(dt, H, inv, float(_c.g), cmax=0.8)
        zcu = math.sqrt(float(_c.g) * H * inv)
        assert n == math.ceil(dt / 0.8 * zcu)
        with pytest.raises(ValueError, match="n="):
            nemo_auto_substeps(1e-6, H, inv, float(_c.g), cmax=0.8)

    def test_mpas_builder_rejects_centred_barotropic(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_mpas_model_config,
        )
        cfg = dataclasses.replace(
            DINOConfig(), barotropic_time_filter="nemo_boxcar_centred")
        with pytest.raises(ValueError, match="barotropic"):
            dino_mpas_model_config(None, cfg, physics=False)
        cfg2 = dataclasses.replace(DINOConfig(), barotropic_auto_cmax=0.8)
        with pytest.raises(ValueError, match="barotropic"):
            dino_mpas_model_config(None, cfg2, physics=False)

    def test_preset_selects_centred_explicit(self):
        from legoesm.ocean.experiments.dino import (
            dino_lat_lon_grid, dino_lat_lon_model_config,
            dino_r1_exact_config,
        )
        import dataclasses

        # The PRESET keeps implicit_cn: the centred window needs the MLF
        # before-state start (job 8826132 NaN; NEMO's own namelist:
        # "model crashes if ln_bt_fw=T"). The blocks remain selectable:
        cfg = dino_r1_exact_config()
        assert cfg.barotropic_solver == "implicit_cn"
        cfg = dataclasses.replace(
            cfg, barotropic_solver="explicit_substep",
            barotropic_time_filter="nemo_boxcar_centred",
            barotropic_auto_cmax=0.8)
        g = dino_lat_lon_grid(cfg, n_lon=12)
        mc, _ = dino_lat_lon_model_config(g, cfg, physics=True)
        assert mc.barotropic.barotropic_solver == "explicit_substep"
        assert mc.barotropic.barotropic_time_filter == "nemo_boxcar_centred"
        # auto count: static int equal to the ln_bt_auto helper's value
        from legoesm.ocean.experiments.dino import _dino_barotropic_substeps
        assert isinstance(mc.barotropic.n_barotropic_substeps, int)
        assert (mc.barotropic.n_barotropic_substeps
                == _dino_barotropic_substeps(g, cfg) >= 2)


# ---------------------------------------------------------------------
# #1226 trasbc.F90:152-153 live top-cell divisor
# (surface_flux_divisor="static"/"nemo_live")
# ---------------------------------------------------------------------

class TestSurfaceFluxDivisor:
    """DINOConfig.surface_flux_divisor: NEMO's trasbc.F90 divides the
    combined non-solar flux by the LIVE e3t(:,:,1,Kmm) = e3t_0*(1+r3t)
    (r3t = ssh/ht_0, domqco.F90:160) -- legoESM's default divides by the
    STATIC dz_ref[0]. "nemo_live" rescales to the live divisor.
    """

    def _fixture(self, eta_value, cfg=None):
        from legoesm.core.field import Field
        from legoesm.ocean.experiments.dino import (
            create_dino_z_star, dino_lat_lon_grid, dino_lat_lon_state,
            dino_lat_lon_surface_forcing_arrays,
        )
        cfg = cfg if cfg is not None else DINOConfig()
        z = create_dino_z_star(cfg)
        g = dino_lat_lon_grid(cfg, n_lon=8)
        st = dino_lat_lon_state(g, z, cfg)
        frc = dino_lat_lon_surface_forcing_arrays(g, cfg)
        # Nonzero eta, masked to zero on land (matches every other eta
        # field in the state — dry columns carry eta=0, r3t=0 there).
        eta = jnp.full_like(st.eta.data, eta_value) * st.land_mask.data
        st = st._replace(eta=Field(
            data=eta, name=st.eta.name, dims=st.eta.dims, units=st.eta.units))
        return g, z, st, frc

    def test_unknown_divisor_raises(self):
        import dataclasses
        from legoesm.ocean.experiments.dino import (
            apply_dino_lat_lon_surface_forcing,
        )
        cfg = dataclasses.replace(DINOConfig(), surface_flux_divisor="bogus")
        g, z, st, frc = self._fixture(5.0, cfg)
        with pytest.raises(ValueError, match="surface_flux_divisor"):
            apply_dino_lat_lon_surface_forcing(st, frc, z, cfg, 2700.0)

    def test_static_is_bit_identical_to_legacy(self):
        """Fallback path (default / no live-divisor opt-in): byte-identical
        to the pre-fix behaviour at every eta, since dz_0_live == dz_0
        exactly when surface_flux_divisor="static" (max|diff|==0.0)."""
        import dataclasses
        from legoesm.ocean.experiments.dino import (
            apply_dino_lat_lon_surface_forcing,
        )
        cfg = DINOConfig()
        assert cfg.surface_flux_divisor == "static"
        g, z, st, frc = self._fixture(7.3, cfg)
        out = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg, 2700.0)

        # Independent pre-fix reconstruction: the static-divisor tendency
        # is dz_0-independent of eta, so re-running at eta=0 on a config
        # that cannot see eta (surface_flux_divisor="static" never reads
        # state.eta) must give the IDENTICAL T/S update.
        g0, z0, st0, frc0 = self._fixture(0.0, cfg)
        out0 = apply_dino_lat_lon_surface_forcing(st0, frc0, z0, cfg, 2700.0)
        max_diff_T = float(jnp.max(jnp.abs(out.T.data - out0.T.data)))
        max_diff_S = float(jnp.max(jnp.abs(out.S.data - out0.S.data)))
        assert max_diff_T == 0.0, max_diff_T
        assert max_diff_S == 0.0, max_diff_S

    def test_nemo_live_matches_independent_transcription(self):
        """Live divisor: assert the top-layer T/S tendency equals an
        INDEPENDENT in-test transcription of trasbc.F90:152-153's single-
        division structure (not a call into the function under test, and
        NOT a post-hoc rescale of the static tendency -- with
        RestoringConfig.implicit=True the denominator is (tau_T + dt),
        which is not simply proportional to 1/dz_0, so the transcription
        must rebuild tau_T/tau_S from the LIVE dz_0 and re-run the SAME
        analytic implicit-Euler formula restoring.py documents, entirely
        without importing restoring.py or tau_from_flux_coefficient).

        PRE-FIX (surface_flux_divisor field did not exist / the applicator
        always divided by the static dz_ref[0]): this test's "nemo_live"
        branch is unreachable with a bare DINOConfig(), and even patched in
        naively as an after-the-fact rescale of the static output this
        assertion measured ~1.2% off (retracted -- see the fix's docstring
        in apply_dino_lat_lon_surface_forcing).
        """
        import dataclasses
        from legoesm.ocean.experiments.dino import (
            apply_dino_lat_lon_surface_forcing,
        )
        from legoesm.ocean.eos import nemo_r3t_stretch

        eta_value = 12.0   # a few metres of ssh -> O(1e-3) r3t on H~2-4 km
        dt = 2700.0
        cfg_static = DINOConfig()
        cfg_live = dataclasses.replace(cfg_static, surface_flux_divisor="nemo_live")

        g, z, st, frc = self._fixture(eta_value, cfg_static)
        out_live = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg_live, dt)

        # Independent transcription of trasbc.F90:152-153 + usrdef_sbc.F90:279
        # + the restoring.py analytic-implicit-Euler algebra (restoring.py's
        # own docstring: dT_dt_eff = (T*-T)/(tau_T+dt)) -- built from scratch
        # here, reading only T*/T/Q_sr/A_theta/dz_0_live/dt, never calling
        # tau_from_flux_coefficient or restoring_surface_forcing.
        dz_0 = float(z.dz_ref[0])
        stretch = np.asarray(nemo_r3t_stretch(z, st.eta.data, st.H_bathy.data))
        dz_0_live = dz_0 * stretch                              # e3t(:,:,1,Kmm)

        rho_0, c_p = cfg_static.rho_0, cfg_static.c_p
        T_top = np.asarray(st.T.data[..., 0])
        S_top = np.asarray(st.S.data[..., 0])
        T_star = np.asarray(frc["T_star_2d"])
        S_star = np.asarray(frc["S_star_2d"])
        Q_sr = np.asarray(frc["Q_sr_2d"])

        tau_T_live = rho_0 * c_p * dz_0_live / cfg_static.A_theta
        tau_S_live = rho_0 * 1.0 * dz_0_live / cfg_static.A_S
        surf_dT = -(T_top - T_star) / (tau_T_live + dt) - Q_sr / (rho_0 * c_p * dz_0_live)
        surf_dS = -(S_top - S_star) / (tau_S_live + dt)

        # Plus the Jerlov SW-penetration tendency's OWN level-0 deposit (eq
        # 10, traqsr.F90 -- a separate NEMO routine, untouched by this fix
        # and still divided by the STATIC dz_ref; the applicator adds it to
        # the SAME top layer as the restoring term). Calling the real
        # (unmodified) shortwave_penetration_tendency here is legitimate --
        # it is a dependency of the function under test, not the function
        # under test itself.
        from legoesm.ocean.physics.shortwave_penetration import (
            shortwave_penetration_tendency, ShortwavePenetrationConfig,
        )
        jacobian = jnp.ones_like(st.eta.data)
        sw_cfg = ShortwavePenetrationConfig(water_type=cfg_static.jerlov_water_type)
        dT_dt_sw_top = np.asarray(shortwave_penetration_tendency(
            sw_down=frc["Q_sr_2d"], z_coord_dz_ref=z.dz_ref,
            z_coord_z_half_ref=z.z_half_ref, jacobian=jacobian, config=sw_cfg,
            rho_0=cfg_static.rho_0, c_sw=cfg_static.c_p,
        )[..., 0])

        expect_dT_top = dt * surf_dT + dt * dT_dt_sw_top
        expect_dS_top = dt * surf_dS

        got_dT_top = np.asarray(out_live.T.data[..., 0] - st.T.data[..., 0])
        got_dS_top = np.asarray(out_live.S.data[..., 0] - st.S.data[..., 0])

        wet = np.asarray(st.land_mask.data) > 0.5
        np.testing.assert_allclose(got_dT_top[wet], expect_dT_top[wet], rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(got_dS_top[wet], expect_dS_top[wet], rtol=1e-10, atol=1e-14)

        # A positive eta -> larger live top-cell thickness -> the SAME
        # surface flux gives a SMALLER tendency magnitude (dilution, not
        # concentration) -- the sign/units check (CLAUDE.md mandatory).
        # Isolate the DIVISOR-AFFECTED piece (restoring + Q_sr-subtraction,
        # i.e. `surf_dT` above) rather than the combined restoring+SW
        # output: the Jerlov SW-penetration deposit at level 0 is IDENTICAL
        # between static/live (out of this fix's scope) and mixing it in
        # would dilute/mask the dilution signal under test.
        dz_0_static = dz_0   # DINOConfig() default: scalar, r3t=0 baseline
        tau_T_static = rho_0 * c_p * dz_0_static / cfg_static.A_theta
        surf_dT_static = (-(T_top - T_star) / (tau_T_static + dt)
                          - Q_sr / (rho_0 * c_p * dz_0_static))
        assert eta_value > 0.0
        assert float(np.mean(stretch[wet])) > 1.0
        mag_static = np.abs(surf_dT_static[wet])
        mag_live = np.abs(surf_dT[wet])
        assert np.all(mag_live <= mag_static + 1e-15), (
            "a larger top cell must DILUTE the flux (smaller |tendency|), "
            "not amplify it")

    def test_below_water_sw_penetration_untouched(self):
        """dT_dt_sw (traqsr.F90, a separate NEMO routine) is out of this
        fix's scope -- the live divisor changes ONLY the level-0 restoring
        term, never the sub-surface Jerlov penetration tendency."""
        import dataclasses
        from legoesm.ocean.experiments.dino import (
            apply_dino_lat_lon_surface_forcing,
        )
        cfg_static = DINOConfig()
        cfg_live = dataclasses.replace(cfg_static, surface_flux_divisor="nemo_live")
        g, z, st, frc = self._fixture(12.0, cfg_static)
        out_static = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg_static, 2700.0)
        out_live = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg_live, 2700.0)
        # Levels 1+ (below the surface restoring layer) must be identical:
        # only the SW penetration tendency reaches them, unaffected by the
        # divisor switch.
        np.testing.assert_array_equal(
            np.asarray(out_static.T.data[..., 1:]),
            np.asarray(out_live.T.data[..., 1:]))


# ---------------------------------------------------------------------
# #1226 c_p truncation (eosbn2.F90:1899 rcp)
# ---------------------------------------------------------------------

class TestCpNemoExact:
    """DINOConfig.c_p paper-Table-1 default (3991.86) truncates NEMO's own
    ``rcp = 3991.86795711963_wp`` (eosbn2.F90:1899, phycst.F90:118 notes
    rho0/rcp are defined in eosbn2, not phycst) at 6 sig figs -- the ENTIRE
    tra_sbc tem residual (#1226 tra_sbc_tem_piece_decompose.py), since
    salinity's own conversion (trasbc.F90:137) has no rcp factor at all.
    """

    def test_default_cp_is_paper_table_value_not_nemo_exact(self):
        # The plain (non-NEMO-card) default stays the paper's own truncated
        # Table 1 value -- unaffected by this fix, exactly like
        # test_default_construct above.
        cfg = DINOConfig()
        assert cfg.c_p == pytest.approx(3991.86)
        NEMO_RCP_EXACT = 3991.86795711963  # eosbn2.F90:1899, verbatim
        assert cfg.c_p != NEMO_RCP_EXACT

    def test_nemo_dino_kamm_recipe_uses_nemo_exact_rcp(self):
        """The NEMO-fidelity card (nemo_dino_kamm / _mlf) must pin c_p to
        NEMO's own eosbn2.F90:1899 rcp EXACTLY, not the paper's truncation --
        same NEMO_CONSTANTS_CONFIG.c_sw value already used by g/omega on
        this card (constants_config.py:56, 67)."""
        NEMO_RCP_EXACT = 3991.86795711963  # eosbn2.F90:1899, verbatim
        for recipe in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
            cfg = dino.dino_config_for_recipe(recipe)
            assert cfg.c_p == NEMO_RCP_EXACT, (recipe, cfg.c_p)

    def test_tra_sbc_tem_reconstruction_bit_identical_with_nemo_exact_cp(self):
        """Independent in-test transcription of trasbc.F90:136,152-153
        (sbc_tsc(jp_tem) = r1_rho0_rcp*qns; pts(Krhs) += zfact*(sbc_tsc_b+
        sbc_tsc)/e3t(Kmm,1)) at a synthetic qns/sbc_hc_b/dz_0: using the
        NEMO card's c_p (NEMO_CONSTANTS_CONFIG.c_sw) reproduces a
        reconstruction built directly from eosbn2.F90's own literal
        bit-for-bit; using the paper's truncated default does NOT (this is
        the PRE-FIX failure -- run with the paper default it does not match
        to machine precision, matching #1226's measured 9.657e-07
        err_norm)."""
        rho_0 = 1026.0
        qns = np.array([37.4, -12.9, 0.0, 121.7])       # synthetic W/m^2
        sbc_hc_b = np.array([1.1e-5, -3.2e-6, 0.0, 4.0e-5])  # synthetic K*m/s
        dz_0 = np.array([10.0, 9.5, 10.0, 8.7])           # synthetic live e3t

        def reconstruct(c_p):
            r1_rho0_rcp = 1.0 / (rho_0 * c_p)
            this_step_rate = r1_rho0_rcp * qns
            return 0.5 * (sbc_hc_b + this_step_rate) / dz_0

        NEMO_RCP_EXACT = 3991.86795711963  # eosbn2.F90:1899, verbatim
        cfg_nemo = dino.dino_config_for_recipe("nemo_dino_kamm_mlf")
        assert cfg_nemo.c_p == NEMO_RCP_EXACT

        recon_nemo_card = reconstruct(cfg_nemo.c_p)
        recon_exact = reconstruct(NEMO_RCP_EXACT)
        np.testing.assert_array_equal(recon_nemo_card, recon_exact)

        recon_paper_default = reconstruct(DINOConfig().c_p)
        max_diff = float(np.max(np.abs(recon_paper_default - recon_exact)))
        assert max_diff > 0.0, (
            "paper Table-1 c_p truncation should NOT reconstruct bit-"
            "identically -- if this is 0.0 the truncation stopped mattering "
            "and the #1226 finding needs re-checking")


# ---------------------------------------------------------------------
# #1226 traqsr.F90:665-712 qsr_2BD live gdepw ladder
# (shortwave_penetration_ladder="static"/"nemo_live")
# ---------------------------------------------------------------------

class TestShortwavePenetrationLadder:
    """DINOConfig.shortwave_penetration_ladder: NEMO's qsr_2BD evaluates the
    two-band absorption profile at the LIVE (z*-stretched) gdepw(Kmm) =
    gdepw_0*(1+r3t) (domzgr_substitute.h90:139, r3t=ssh/ht_0 domqco.F90:160)
    -- legoESM's default uses the STATIC z_coord.z_half_ref. "nemo_live"
    rescales both the interface depths and the layer thickness by the same
    stretch (eos.nemo_r3t_stretch), matching traqsr.F90's single live e3t.
    """

    def _fixture(self, eta_value, cfg=None):
        from legoesm.core.field import Field
        from legoesm.ocean.experiments.dino import (
            create_dino_z_star, dino_lat_lon_grid, dino_lat_lon_state,
            dino_lat_lon_surface_forcing_arrays,
        )
        cfg = cfg if cfg is not None else DINOConfig()
        z = create_dino_z_star(cfg)
        g = dino_lat_lon_grid(cfg, n_lon=8)
        st = dino_lat_lon_state(g, z, cfg)
        frc = dino_lat_lon_surface_forcing_arrays(g, cfg)
        eta = jnp.full_like(st.eta.data, eta_value) * st.land_mask.data
        st = st._replace(eta=Field(
            data=eta, name=st.eta.name, dims=st.eta.dims, units=st.eta.units))
        return g, z, st, frc

    def test_unknown_ladder_raises(self):
        import dataclasses
        from legoesm.ocean.experiments.dino import (
            apply_dino_lat_lon_surface_forcing,
        )
        cfg = dataclasses.replace(DINOConfig(), shortwave_penetration_ladder="bogus")
        g, z, st, frc = self._fixture(5.0, cfg)
        with pytest.raises(ValueError, match="shortwave_penetration_ladder"):
            apply_dino_lat_lon_surface_forcing(st, frc, z, cfg, 2700.0)

    def test_static_is_bit_identical_to_legacy(self):
        """Default (no opt-in): byte-identical to the pre-fix behaviour at
        every eta -- the applicator passes z_half_stretch=None, which
        shortwave_penetration_tendency treats identically to not having the
        new kwarg at all. Isolate the SW-penetration piece (levels 1+, which
        the restoring term never reaches) so this is a clean test of ONLY
        the ladder gate, independent of surface_flux_divisor/eta-dependent
        restoring terms."""
        from legoesm.ocean.experiments.dino import (
            apply_dino_lat_lon_surface_forcing,
        )
        cfg = DINOConfig()
        assert cfg.shortwave_penetration_ladder == "static"
        g, z, st, frc = self._fixture(7.3, cfg)
        out = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg, 2700.0)

        g0, z0, st0, frc0 = self._fixture(0.0, cfg)
        out0 = apply_dino_lat_lon_surface_forcing(st0, frc0, z0, cfg, 2700.0)
        max_diff_T_below = float(jnp.max(jnp.abs(
            out.T.data[..., 1:] - out0.T.data[..., 1:])))
        assert max_diff_T_below == 0.0, max_diff_T_below

    def test_nemo_live_matches_independent_transcription(self):
        """Live ladder: assert the FULL-COLUMN SW penetration tendency
        equals an INDEPENDENT in-test transcription of traqsr.F90's
        qsr_2BD formula (rn_abs*exp(-gdepw*r1_si0) + (1-rn_abs)*
        exp(-gdepw*r1_si1), traqsr.F90:665-712) evaluated at the LIVE
        r3t-stretched gdepw -- built from scratch here, never calling
        shortwave_penetration_tendency.

        PRE-FIX (shortwave_penetration_ladder field did not exist / the
        applicator always used the static z_half_ref): this test's
        "nemo_live" branch is unreachable with a bare DINOConfig(), and the
        static-ladder tendency does not match this transcription once eta
        is nonzero (measured #1226 all-levels err_norm median 2.022e-05).
        """
        from legoesm.ocean.experiments.dino import (
            apply_dino_lat_lon_surface_forcing,
        )
        from legoesm.ocean.eos import nemo_r3t_stretch
        from legoesm.ocean.physics.shortwave_penetration import JERLOV_TYPES
        import dataclasses

        eta_value = 15.0  # a few metres of ssh -> O(1e-3) r3t on H~2-4 km
        dt = 2700.0
        cfg_static = DINOConfig()
        cfg_live = dataclasses.replace(
            cfg_static, shortwave_penetration_ladder="nemo_live")

        g, z, st, frc = self._fixture(eta_value, cfg_static)
        out_live = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg_live, dt)

        # Independent transcription: qsr_2BD's own two-band formula at the
        # LIVE gdepw ladder (traqsr.F90:665-712), reading only z_half_ref,
        # dz_ref, eta, H_bathy, and the Jerlov params -- no call into
        # shortwave_penetration_tendency at all.
        params = JERLOV_TYPES[cfg_static.jerlov_water_type]
        R, zeta1, zeta2 = params.R, params.zeta1, params.zeta2
        stretch = np.asarray(nemo_r3t_stretch(z, st.eta.data, st.H_bathy.data))
        z_half_static = np.asarray(z.z_half_ref)          # (nlev+1,), negative
        z_half_live = z_half_static[None, None, :] * stretch[:, :, None]
        I_half = R * np.exp(z_half_live / zeta1) + (1.0 - R) * np.exp(z_half_live / zeta2)
        frac = I_half[:, :, :-1] - I_half[:, :, 1:]
        frac[:, :, -1] += I_half[:, :, -1]
        dz_live = np.asarray(z.dz_ref)[None, None, :] * stretch[:, :, None]
        Q_sr = np.asarray(frc["Q_sr_2d"])
        expect_dT_dt_sw = Q_sr[:, :, None] * frac / (cfg_static.rho_0 * cfg_static.c_p * dz_live)
        expect_dT_dt_sw = np.where(dz_live > 0.0, expect_dT_dt_sw, 0.0)

        # Independent hand transcription of the applicator's OWN restoring
        # contribution at level 0 (same analytic implicit-Euler algebra as
        # TestSurfaceFluxDivisor.test_nemo_live_matches_independent_
        # transcription above; surface_flux_divisor stays "static" on
        # cfg_live -- an INDEPENDENT gate -- so dz_0 here is the plain
        # static scalar, not the live-divisor array) to isolate the
        # SW-penetration piece under test.
        dz_0 = float(z.dz_ref[0])
        rho_0, c_p = cfg_static.rho_0, cfg_static.c_p
        T_top = np.asarray(st.T.data[..., 0])
        T_star = np.asarray(frc["T_star_2d"])
        tau_T = rho_0 * c_p * dz_0 / cfg_static.A_theta
        surf_dT = -(T_top - T_star) / (tau_T + dt) - Q_sr / (rho_0 * c_p * dz_0)
        expect_dT_top_total = dt * (surf_dT + expect_dT_dt_sw[..., 0])

        wet = np.asarray(st.land_mask.data) > 0.5
        got_dT_top = np.asarray(out_live.T.data[..., 0] - st.T.data[..., 0])
        np.testing.assert_allclose(
            got_dT_top[wet], expect_dT_top_total[wet], rtol=1e-9, atol=1e-14)

        # Levels 1+ are the pure SW-penetration deposit (no restoring
        # contribution reaches them at all).
        got_dT_below = np.asarray(out_live.T.data[..., 1:] - st.T.data[..., 1:])
        expect_dT_below = dt * expect_dT_dt_sw[..., 1:]
        np.testing.assert_allclose(
            got_dT_below[wet], expect_dT_below[wet], rtol=1e-9, atol=1e-14)

    def test_below_water_surface_flux_divisor_untouched(self):
        """shortwave_penetration_ladder is independent of
        surface_flux_divisor -- switching the ladder must NOT change the
        level-0 restoring tendency (only surface_flux_divisor touches it)."""
        import dataclasses
        from legoesm.ocean.experiments.dino import (
            apply_dino_lat_lon_surface_forcing,
        )
        cfg_static = DINOConfig()
        cfg_ladder_live = dataclasses.replace(
            cfg_static, shortwave_penetration_ladder="nemo_live")
        g, z, st, frc = self._fixture(12.0, cfg_static)
        out_static = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg_static, 2700.0)
        out_ladder_live = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg_ladder_live, 2700.0)
        # Isolate the restoring-only piece: subtract each run's OWN
        # level-0 SW deposit (which legitimately differs between the two
        # ladders) rather than asserting level-0 equality directly.
        from legoesm.ocean.physics.shortwave_penetration import (
            shortwave_penetration_tendency, ShortwavePenetrationConfig)
        from legoesm.ocean.eos import nemo_r3t_stretch
        jacobian = jnp.ones_like(st.eta.data)
        sw_cfg = ShortwavePenetrationConfig(water_type=cfg_static.jerlov_water_type)
        dT_dt_sw_static = shortwave_penetration_tendency(
            sw_down=frc["Q_sr_2d"], z_coord_dz_ref=z.dz_ref,
            z_coord_z_half_ref=z.z_half_ref, jacobian=jacobian, config=sw_cfg,
            rho_0=cfg_static.rho_0, c_sw=cfg_static.c_p)
        stretch = nemo_r3t_stretch(z, st.eta.data, st.H_bathy.data)
        dT_dt_sw_live = shortwave_penetration_tendency(
            sw_down=frc["Q_sr_2d"], z_coord_dz_ref=z.dz_ref,
            z_coord_z_half_ref=z.z_half_ref, jacobian=jacobian, config=sw_cfg,
            rho_0=cfg_static.rho_0, c_sw=cfg_static.c_p, z_half_stretch=stretch)
        restoring_static = out_static.T.data[..., 0] - st.T.data[..., 0] - 2700.0 * dT_dt_sw_static[..., 0]
        restoring_ladder_live = out_ladder_live.T.data[..., 0] - st.T.data[..., 0] - 2700.0 * dT_dt_sw_live[..., 0]
        np.testing.assert_allclose(
            np.asarray(restoring_static), np.asarray(restoring_ladder_live),
            rtol=1e-10, atol=1e-14)
