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
    get_recipe,
    list_recipes,
)


class TestRegistry:
    def test_list_recipes(self):
        latlon = list_recipes("latlon")
        assert "legoesm_linear_v1" in latlon
        assert "veros_faithful_v1" in latlon
        assert "nemo_dino_v1" in latlon
        assert list_recipes("mpas") == ["legoesm_linear_mpas_v1"]

    def test_get_recipe_returns_copy(self):
        a = get_recipe("legoesm_linear_v1")
        a["eos"] = "MUTATED"
        assert LATLON_RECIPES["legoesm_linear_v1"]["eos"] == "linear"

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown latlon recipe"):
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
    """The point: pick a predefined recipe and run it on ANY setup. The assembler
    matches the EOS config to the recipe's eos scheme."""

    def test_go_setup_on_eady_recipe(self):
        """GO setup x eady_weno5_v1 (linear): GO carries the WENO5 dycore."""
        cfg = GlobalOverturningConfig()
        mc = global_overturning_model_config(
            cfg, eos_config=create_eos_config(cfg), recipe="eady_weno5_v1")
        assert mc.momentum_advection == "weno5"      # recipe's scheme identity
        assert mc.pgf_scheme == "smc03"
        assert mc.outer_integrator == "ab2"
        assert mc.A_h == cfg.A_h                      # ...but GO's setup params

    def test_go_setup_on_veros_recipe_assembles(self):
        """GO setup x veros_faithful_v1 (non-linear EOS) now ASSEMBLES the Veros
        dycore — the linear eos_linear is dropped (the Veros EOS carries its own
        coefficients), so no mismatch."""
        cfg = GlobalOverturningConfig()
        mc = global_overturning_model_config(
            cfg, eos_config=create_eos_config(cfg), recipe="veros_faithful_v1")
        assert mc.eos == "veros_nonlin2"
        assert mc.eos_linear is None                  # dropped — not the right EOS
        assert mc.outer_integrator == "ab2"
        assert mc.barotropic_solver == "rigid_lid"
        assert mc.A_h == cfg.A_h                       # GO's setup params kept

    def test_go_setup_on_dino_recipe_assembles(self):
        """GO setup x nemo_dino_v1 (eos='wright') assembles; eos_linear dropped."""
        cfg = GlobalOverturningConfig()
        mc = global_overturning_model_config(
            cfg, eos_config=create_eos_config(cfg), recipe="nemo_dino_v1")
        assert mc.eos == "wright"
        assert mc.eos_linear is None
        assert mc.ke_gradient_scheme == "hollingsworth"

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


class TestEosMatching:
    """assemble_ocean_config attaches eos_linear ONLY for a linear recipe, so any
    recipe x setup assembles a coherent config (rung 2)."""

    def test_linear_recipe_keeps_eos_linear(self):
        from legoesm.ocean.recipes import assemble_ocean_config
        from legoesm.ocean.state import LatLonCGridOceanConfig
        sentinel = object()
        cfg = assemble_ocean_config(
            "legoesm_linear_v1", "latlon", LatLonCGridOceanConfig,
            eos_linear=sentinel)
        assert cfg.eos == "linear"
        assert cfg.eos_linear is sentinel

    def test_nonlinear_recipe_drops_eos_linear(self):
        from legoesm.ocean.recipes import assemble_ocean_config
        from legoesm.ocean.state import LatLonCGridOceanConfig
        cfg = assemble_ocean_config(
            "veros_faithful_v1", "latlon", LatLonCGridOceanConfig,
            eos_linear=object())   # supplied, but a non-linear recipe drops it
        assert cfg.eos == "veros_nonlin2"
        assert cfg.eos_linear is None

    def test_overrides_win(self):
        from legoesm.ocean.recipes import assemble_ocean_config
        from legoesm.ocean.state import LatLonCGridOceanConfig
        cfg = assemble_ocean_config(
            "legoesm_linear_v1", "latlon", LatLonCGridOceanConfig,
            eos_linear=object(), overrides={"pgf_scheme": "smc03"}, A_h=42.0)
        assert cfg.pgf_scheme == "smc03"
        assert cfg.A_h == 42.0


class TestCompatibility:
    """The validator still rejects genuinely-incoherent configs (e.g. a forced
    override producing eos='linear' with no coefficients, or a non-linear EOS
    with a contradictory eos_linear)."""

    def test_compat_check_direct(self):
        from types import SimpleNamespace

        from legoesm.ocean.recipes import assert_recipe_setup_compatible
        # linear + eos_linear -> OK
        assert_recipe_setup_compatible(
            SimpleNamespace(eos="linear", eos_linear=object()))
        # non-linear + no eos_linear -> OK (wright/veros need no linear config)
        assert_recipe_setup_compatible(
            SimpleNamespace(eos="wright", eos_linear=None))
        # linear + no eos_linear -> raise (would use default coeffs)
        with pytest.raises(ValueError, match="no eos_linear"):
            assert_recipe_setup_compatible(
                SimpleNamespace(eos="linear", eos_linear=None))
        # non-linear + dangling eos_linear -> raise (contradictory)
        with pytest.raises(ValueError, match="non-linear EOS"):
            assert_recipe_setup_compatible(
                SimpleNamespace(eos="veros_nonlin2", eos_linear=object()))
