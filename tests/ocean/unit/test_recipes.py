"""Tests for the named recipe catalog (#490).

Covers: registry mechanics, catalog<->factory consistency (a catalog entry must
equal the dycore the source experiment factory actually produces), and the
headline payoff — running ONE experiment setup on a DIFFERENT named recipe.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
from legoesm.ocean.experiments.eady_uniform import (
    EadyUniformConfig,
    eady_uniform_model_config,
)
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig,
    create_eos_config,
    global_overturning_model_config,
    global_overturning_mpas_model_config,
)
from legoesm.ocean.recipes import (
    LATLON_RECIPES,
    MPAS_RECIPES,
    assemble_ocean_config,
    get_recipe,
    list_recipes,
)


class TestRegistry:
    def test_list_recipes(self):
        all_recipes = list_recipes()
        assert "legoesm_linear_v1" in all_recipes
        assert "veros_faithful_v1" in all_recipes
        assert "nemo_dino_v1" in all_recipes
        assert "legoesm_linear_mpas_v1" in all_recipes
        # The proven OMIP NEMO-match recipes (#500) appear in the menu.
        assert "omip_nemo_match_mpas_v1" in all_recipes
        assert "omip_nemo_match_tripole_v1" in all_recipes
        # The MPAS sibling of default_wright_v1, added with the MPAS catalog (#490).
        assert "default_wright_mpas_v1" in all_recipes
        assert list_recipes("mpas") == [
            "default_wright_mpas_v1", "legoesm_linear_mpas_v1",
            "omip_nemo_match_mpas_v1"]
        assert "omip_nemo_match_tripole_v1" in list_recipes("latlon")

    def test_get_recipe_returns_copy(self):
        a = get_recipe("legoesm_linear_v1")
        a["eos"] = "MUTATED"
        assert LATLON_RECIPES["legoesm_linear_v1"]["eos"] == "linear"
        assert get_recipe("legoesm_linear_mpas_v1") == \
            MPAS_RECIPES["legoesm_linear_mpas_v1"]

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown recipe"):
            get_recipe("does_not_exist")

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown recipe kind"):
            get_recipe("legoesm_linear_v1", kind="spectral")
        with pytest.raises(ValueError, match="unknown recipe kind"):
            list_recipes("spectral")

    def test_every_entry_has_eos_and_is_nonvacuous(self):
        for table in (LATLON_RECIPES, MPAS_RECIPES):
            for name, bundle in table.items():
                assert "eos" in bundle, name
                assert len(bundle) >= 6, name

    def test_snapshot_identity_keys_live_in_catalog(self):
        assert LATLON_RECIPES["legoesm_linear_v1"]["n_barotropic_substeps"] == 30
        assert LATLON_RECIPES["nemo_dino_v1"]["A_h_lat_scaling"] is True


class TestCatalogMatchesFactories:
    """A catalog entry must equal the dycore its source factory produces — so the
    named recipe and the experiment never drift apart."""

    def test_legoesm_linear_v1_is_go_default(self):
        mc = global_overturning_model_config(GlobalOverturningConfig())
        for k, v in get_recipe("legoesm_linear_v1").items():
            assert mc.flat_get(k) == v, k

    def test_legoesm_linear_mpas_v1_is_go_mpas_default(self):
        mc = global_overturning_mpas_model_config(GlobalOverturningConfig())
        for k, v in get_recipe("legoesm_linear_mpas_v1", "mpas").items():
            assert getattr(mc, k) == v, k

    def test_eady_weno5_v1_is_eady_default(self):
        mc = eady_uniform_model_config(EadyUniformConfig())
        for k, v in get_recipe("eady_weno5_v1").items():
            assert mc.flat_get(k) == v, k

    def test_veros_faithful_v1_is_acc_dycore(self):
        """Drift guard for veros_faithful_v1 == build_acc_recipe(free-run)."""
        from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_recipe
        mc = build_acc_recipe(with_surface_forcing=True).model_config
        for k, v in get_recipe("veros_faithful_v1").items():
            assert mc.flat_get(k) == v, k

    def test_nemo_dino_v1_is_dino_dycore(self):
        """Drift guard for nemo_dino_v1 == dino_lat_lon_model_config."""
        from legoesm.grids.latlon import create_mercator_grid
        from legoesm.ocean.experiments.dino import DINOConfig, dino_lat_lon_model_config
        grid = create_mercator_grid(n_lon=16, lat_max_deg=70.0,
                                    lon_west_deg=0.0, lon_east_deg=50.0)
        mc, _ = dino_lat_lon_model_config(grid, DINOConfig())
        for k, v in get_recipe("nemo_dino_v1").items():
            assert mc.flat_get(k) == v, k

    def test_omip_nemo_match_mpas_v1_is_factory_dycore(self):
        """Drift guard: catalog == nemo_match_mpas_model_config scheme fields."""
        from legoesm.ocean.fidelity.nemo_match_recipe import (
            nemo_match_mpas_model_config,
        )
        mc = nemo_match_mpas_model_config()
        for k, v in get_recipe("omip_nemo_match_mpas_v1", "mpas").items():
            assert getattr(mc, k) == v, k

    def test_omip_nemo_match_tripole_v1_is_factory_dycore(self):
        """Drift guard: catalog == nemo_match_tripole_model_config scheme fields."""
        from legoesm.ocean.fidelity.nemo_match_recipe import (
            nemo_match_tripole_model_config,
        )
        mc = nemo_match_tripole_model_config()
        for k, v in get_recipe("omip_nemo_match_tripole_v1", "latlon").items():
            assert mc.flat_get(k) == v, k


# The PROVEN OMIP MPAS ico6 dycore — a FROZEN snapshot of the scheme + coefficient
# fields _create_setup("mpas") produced BEFORE the factory rewire (#500).  This is
# the regression net: a silent change to the proven config (which gave SST RMSE
# 0.84 vs NEMO ORCA1) breaks this hard-coded expectation.  ``physics`` is SETUP
# (run-dependent surface forcing + per-run vmix) and excluded from the snapshot.
_FROZEN_MPAS_DYCORE = {
    "A_h": 1.0e5,
    "A_v": 1.0e-4,
    "K_v": 1.0e-5,
    "C_smag_lap": 0.33,
    "K_zeta_bih": 1.0e14,
    "barotropic_solver": "implicit_cn",
    "barotropic_implicit_pcg_tol": 1.0e-10,
    "barotropic_implicit_pcg_maxiter": 300,
    "pgf_scheme": "adcroft",
    "eos": "wright",
    "pv_scheme": "enstrophy",
    "implicit_vertical_mixing": True,
    "normalize_freshwater": True,
    "tracer_advection": "tvd",
    "bottom_drag_r": 1.0e-3,
    "bottom_drag_bbl_thickness": 100.0,
    "bottom_drag_bg_velocity": 0.1,
}

# Same FROZEN snapshot for the PROVEN tripole eORCA025 dycore (SST RMSE 1.15).
_FROZEN_TRIPOLE_DYCORE = {
    "A_h": 1.0e5,
    "A_v": 1.0e-4,
    "K_v": 1.0e-5,
    "B_h": 0.0,
    "C_smag_lap": 0.33,
    "n_barotropic_substeps": 30,
    "barotropic_solver": "implicit_cn",
    "barotropic_implicit_pcg_tol": 1.0e-10,
    "barotropic_implicit_pcg_maxiter": 300,
    "pgf_scheme": "adcroft",
    "eos": "wright",
    "momentum_advection": "vector_invariant",
    "ke_gradient_scheme": "centered",
    "coriolis_scheme": "matsuno_split",
    "outer_integrator": "forward_euler",
    "tracer_time_integrator": "euler",
    "implicit_vertical_mixing": True,
    "tracer_advection": "tvd",
    "bottom_drag_r": 1.0e-3,
    "bottom_drag_bbl_thickness": 100.0,
    "bottom_drag_bg_velocity": 0.1,
    "freshwater_closure": "virtual_salt_flux",
    # OMIP global freshwater correction — _create_setup("tripole") sets it True
    # (run_omip.py); the recipe omitting it silently fell back to the config
    # default False (codex).  MUST stay True.
    "normalize_freshwater": True,
}

# The shared GM/Redi block both proven configs use (frozen, #500).
_FROZEN_GM_REDI = {
    "kappa_GM": 600.0,
    "kappa_Redi": 600.0,
    "S_max": 0.005,
    "slope_scheme": "centered",
}


class TestOMIPNemoMatchFactories:
    """The proven OMIP NEMO-match dycores are production config — a drift would
    silently change a config validated against NEMO ORCA1 climate.  These guards
    pin (a) the factory to its frozen winning coefficients, (b) the factory to
    the PROVEN driver config (_create_setup), and (c) the catalog to the factory
    (in TestCatalogMatchesFactories)."""

    def test_mpas_factory_matches_frozen_coefficients(self):
        """FROZEN-value guard: the MPAS factory reproduces the winning literals."""
        from legoesm.ocean.fidelity.nemo_match_recipe import (
            nemo_match_mpas_model_config,
        )
        mc = nemo_match_mpas_model_config()
        for k, v in _FROZEN_MPAS_DYCORE.items():
            assert getattr(mc, k) == v, k
        assert mc.gm_redi is not None
        assert mc.gm_redi.visbeck.enabled is False
        for k, v in _FROZEN_GM_REDI.items():
            assert getattr(mc.gm_redi, k) == v, k

    def test_tripole_factory_matches_frozen_coefficients(self):
        """FROZEN-value guard: the tripole factory reproduces the winning literals."""
        from legoesm.ocean.fidelity.nemo_match_recipe import (
            nemo_match_tripole_model_config,
        )
        mc = nemo_match_tripole_model_config()
        for k, v in _FROZEN_TRIPOLE_DYCORE.items():
            assert mc.flat_get(k) == v, k
        assert mc.gm_redi is not None
        assert mc.gm_redi.visbeck.enabled is False
        for k, v in _FROZEN_GM_REDI.items():
            assert getattr(mc.gm_redi, k) == v, k

    def test_tripole_recipe_enables_freshwater_normalization(self):
        """Regression (codex fix C): the NEMO-match tripole recipe MUST enable
        the OMIP global freshwater normalization — the recipe config exposes
        normalize_freshwater=True and passes it into LatLonCGridOceanConfig
        (previously silently fell back to the base default False, diverging
        from _create_setup("tripole")'s proven normalize_freshwater=True)."""
        from legoesm.ocean.fidelity.nemo_match_recipe import (
            NEMOMatchTripoleRecipeConfig,
            nemo_match_tripole_model_config,
        )
        assert NEMOMatchTripoleRecipeConfig().normalize_freshwater is True
        mc = nemo_match_tripole_model_config()
        assert mc.flat_get("normalize_freshwater") is True
        # ...and the CATALOG bundles carry it too, so catalog-based assembly
        # cannot silently fall back to the config default False (codex r3).
        assert get_recipe("omip_nemo_match_tripole_v1",
                          "latlon")["normalize_freshwater"] is True
        assert get_recipe("omip_nemo_match_mpas_v1",
                          "mpas")["normalize_freshwater"] is True

    def test_mpas_factory_equals_create_setup_dycore(self):
        """Tie the factory + catalog to the PROVEN driver config: every dycore
        field of _create_setup("mpas") must equal the factory's.  Uses a tiny
        synthetic ico2 Voronoi mesh (162 cells) — no external files."""
        from scripts.run import run_omip as R
        from legoesm.ocean.fidelity.nemo_match_recipe import (
            nemo_match_mpas_model_config,
        )
        from legoesm.ocean.mpas_config import MPASOceanConfig

        _, _, config, _, kind = R._create_setup(
            grid_type="mpas", resolution="ico2", nlev=3, H_max=4000.0,
            physics_preset="none", water_type="jerlov_1",
            forcing_mode="restoring",
        )
        assert kind == "mpas"
        factory_mc = nemo_match_mpas_model_config()
        # Every field EXCEPT physics (SETUP) must match the factory exactly.
        for f in MPASOceanConfig._fields:
            if f == "physics":
                continue
            assert getattr(config, f) == getattr(factory_mc, f), f
        # ...and the run-dependent physics is the proven restoring-mode SETUP.
        assert config.physics is not None
        assert config.physics.surface_forcing.scheme == "combined"
        assert config.physics.convection.scheme == "enhanced_diffusion"
        # The catalog scheme bundle is exactly this config's scheme identity.
        for k, v in get_recipe("omip_nemo_match_mpas_v1", "mpas").items():
            assert getattr(config, k) == v, k

    def test_mpas_create_setup_matches_frozen_snapshot(self):
        """The rewired _create_setup("mpas") config == the pre-rewire FROZEN
        snapshot — proves the factory rewire changed NO dycore field."""
        from scripts.run import run_omip as R

        _, _, config, _, _ = R._create_setup(
            grid_type="mpas", resolution="ico2", nlev=3, H_max=4000.0,
            physics_preset="none", water_type="jerlov_1",
            forcing_mode="restoring",
        )
        for k, v in _FROZEN_MPAS_DYCORE.items():
            assert getattr(config, k) == v, k
        for k, v in _FROZEN_GM_REDI.items():
            assert getattr(config.gm_redi, k) == v, k

    def test_omip_recipes_are_nonvacuous(self):
        """Non-vacuity: the new OMIP recipes differ from the linear defaults."""
        mpas = get_recipe("omip_nemo_match_mpas_v1", "mpas")
        assert mpas != get_recipe("legoesm_linear_mpas_v1", "mpas")
        assert mpas["eos"] == "wright"            # not the linear default
        assert mpas["barotropic_solver"] == "implicit_cn"  # not explicit_substep
        tripole = get_recipe("omip_nemo_match_tripole_v1", "latlon")
        assert tripole != get_recipe("legoesm_linear_v1", "latlon")
        assert tripole["barotropic_solver"] == "implicit_cn"

    def test_factory_default_physics_rejects_unknown_forcing_mode(self):
        """Dispatch hardening: unknown forcing_mode raises (no silent default)."""
        from legoesm.ocean.fidelity import nemo_match_recipe as M
        with pytest.raises(ValueError, match="unknown NEMO-match forcing_mode"):
            M._default_match_physics(forcing_mode="not_a_mode")


class TestReuse:
    """The point: pick a predefined recipe and run it on another setup."""

    def test_go_setup_on_eady_recipe(self):
        """GO setup x eady_weno5_v1 (linear): GO carries the WENO5 dycore."""
        cfg = GlobalOverturningConfig()
        mc = global_overturning_model_config(
            cfg, eos_config=create_eos_config(cfg), recipe="eady_weno5_v1")
        assert mc.momentum_advection == "weno5"      # recipe's scheme identity
        assert mc.pgf_scheme == "smc03"
        assert mc.outer_integrator == "ab2"
        assert mc.lateral_viscosity.A_h == cfg.A_h                      # ...but GO's setup params

    def test_override_beats_recipe(self):
        cfg = GlobalOverturningConfig()
        mc = global_overturning_model_config(
            cfg, eos_config=create_eos_config(cfg),
            recipe="legoesm_linear_v1", barotropic_solver="implicit_cn")
        assert mc.barotropic.barotropic_solver == "implicit_cn"

    def test_default_recipe_is_legoesm_linear(self):
        cfg = GlobalOverturningConfig()
        a = global_overturning_model_config(cfg, eos_config=create_eos_config(cfg))
        b = global_overturning_model_config(
            cfg, eos_config=create_eos_config(cfg), recipe="legoesm_linear_v1")
        for f in ("eos", "outer_integrator", "barotropic_solver", "pgf_scheme"):
            assert a.flat_get(f) == b.flat_get(f)

    def test_unknown_recipe_raises_from_factory(self):
        with pytest.raises(ValueError, match="unknown latlon recipe"):
            global_overturning_model_config(
                GlobalOverturningConfig(), recipe="not_a_recipe")
        with pytest.raises(ValueError, match="unknown mpas recipe"):
            global_overturning_mpas_model_config(
                GlobalOverturningConfig(), recipe="not_a_recipe")


class TestAssembler:
    """The assembler is mechanical: recipe bundle, setup params, overrides."""

    def test_setup_params_and_overrides_layer_on_bundle(self):
        from legoesm.ocean.state import LatLonCGridOceanConfig

        sentinel_eos = object()
        cfg = assemble_ocean_config(
            get_recipe("legoesm_linear_v1"), LatLonCGridOceanConfig,
            eos_linear=sentinel_eos,
            overrides={"pgf_scheme": "smc03"},
            A_h=42.0,
        )
        assert cfg.eos == "linear"
        assert cfg.eos_linear is sentinel_eos
        assert cfg.pgf_scheme == "smc03"
        assert cfg.lateral_viscosity.A_h == 42.0


class TestEosMatching:
    """assemble_ocean_config attaches eos_linear ONLY for a linear recipe, so any
    recipe x setup assembles a coherent config (#490 verified-registry rung 2)."""

    def test_linear_recipe_keeps_eos_linear(self):
        from legoesm.ocean.state import LatLonCGridOceanConfig
        sentinel = object()
        cfg = assemble_ocean_config(
            get_recipe("legoesm_linear_v1"), LatLonCGridOceanConfig,
            eos_linear=sentinel)
        assert cfg.eos == "linear" and cfg.eos_linear is sentinel

    def test_nonlinear_recipe_drops_eos_linear(self):
        from legoesm.ocean.state import LatLonCGridOceanConfig
        cfg = assemble_ocean_config(
            get_recipe("veros_faithful_v1"), LatLonCGridOceanConfig,
            eos_linear=object())   # supplied, but a non-linear recipe drops it
        assert cfg.eos == "veros_nonlin2" and cfg.eos_linear is None

    def test_go_setup_on_veros_recipe_assembles(self):
        """GO (linear setup) x veros_faithful_v1 now ASSEMBLES the Veros dycore
        (eos_linear dropped) instead of a half-configured mix."""
        cfg = GlobalOverturningConfig()
        mc = global_overturning_model_config(
            cfg, eos_config=create_eos_config(cfg), recipe="veros_faithful_v1")
        assert mc.eos == "veros_nonlin2"
        assert mc.eos_linear is None
        assert mc.outer_integrator == "ab2"
        assert mc.lateral_viscosity.A_h == cfg.A_h          # GO's setup params kept


class TestCompatibility:
    """assert_recipe_setup_compatible rejects genuinely-incoherent configs."""

    def test_compat_check_direct(self):
        from types import SimpleNamespace

        from legoesm.ocean.recipes import assert_recipe_setup_compatible
        assert_recipe_setup_compatible(
            SimpleNamespace(eos="linear", eos_linear=object()))       # OK
        assert_recipe_setup_compatible(
            SimpleNamespace(eos="wright", eos_linear=None))           # OK
        with pytest.raises(ValueError, match="no eos_linear"):
            assert_recipe_setup_compatible(
                SimpleNamespace(eos="linear", eos_linear=None))
        with pytest.raises(ValueError, match="non-linear EOS"):
            assert_recipe_setup_compatible(
                SimpleNamespace(eos="veros_nonlin2", eos_linear=object()))
