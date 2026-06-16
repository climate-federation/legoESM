"""Tests for the named recipe catalog (#490).

Covers: registry mechanics, catalog<->factory consistency (a catalog entry must
equal the dycore the source experiment factory actually produces), and the
headline payoff — running ONE experiment setup on a DIFFERENT named recipe.
"""

from __future__ import annotations

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
        assert list_recipes("mpas") == ["legoesm_linear_mpas_v1"]

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
            assert getattr(mc, k) == v, k

    def test_legoesm_linear_mpas_v1_is_go_mpas_default(self):
        mc = global_overturning_mpas_model_config(GlobalOverturningConfig())
        for k, v in get_recipe("legoesm_linear_mpas_v1", "mpas").items():
            assert getattr(mc, k) == v, k

    def test_eady_weno5_v1_is_eady_default(self):
        mc = eady_uniform_model_config(EadyUniformConfig())
        for k, v in get_recipe("eady_weno5_v1").items():
            assert getattr(mc, k) == v, k

    def test_veros_faithful_v1_is_acc_dycore(self):
        """Drift guard for veros_faithful_v1 == build_acc_recipe(free-run)."""
        from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_recipe
        mc = build_acc_recipe(with_surface_forcing=True).model_config
        for k, v in get_recipe("veros_faithful_v1").items():
            assert getattr(mc, k) == v, k

    def test_nemo_dino_v1_is_dino_dycore(self):
        """Drift guard for nemo_dino_v1 == dino_lat_lon_model_config."""
        from legoesm.grids.latlon import create_mercator_grid
        from legoesm.ocean.experiments.dino import DINOConfig, dino_lat_lon_model_config
        grid = create_mercator_grid(n_lon=16, lat_max_deg=70.0,
                                    lon_west_deg=0.0, lon_east_deg=50.0)
        mc, _ = dino_lat_lon_model_config(grid, DINOConfig())
        for k, v in get_recipe("nemo_dino_v1").items():
            assert getattr(mc, k) == v, k


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
        assert mc.A_h == cfg.A_h                      # ...but GO's setup params

    def test_override_beats_recipe(self):
        cfg = GlobalOverturningConfig()
        mc = global_overturning_model_config(
            cfg, eos_config=create_eos_config(cfg),
            recipe="legoesm_linear_v1", barotropic_solver="implicit_cn")
        assert mc.barotropic_solver == "implicit_cn"

    def test_default_recipe_is_legoesm_linear(self):
        cfg = GlobalOverturningConfig()
        a = global_overturning_model_config(cfg, eos_config=create_eos_config(cfg))
        b = global_overturning_model_config(
            cfg, eos_config=create_eos_config(cfg), recipe="legoesm_linear_v1")
        for f in ("eos", "outer_integrator", "barotropic_solver", "pgf_scheme"):
            assert getattr(a, f) == getattr(b, f)

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
        assert cfg.A_h == 42.0


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
        assert mc.A_h == cfg.A_h          # GO's setup params kept


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
