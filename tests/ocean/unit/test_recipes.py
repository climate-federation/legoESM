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
    """The point: pick a predefined recipe and run it on a COMPATIBLE setup."""

    def test_go_setup_on_eady_recipe(self):
        """GO setup x eady_weno5_v1 recipe — both linear EOS, so compatible:
        the GO setup now carries the WENO5 dycore, with GO's setup params."""
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


class TestCompatibility:
    """The verified-registry rung: an incompatible recipe x setup must fail loud."""

    def test_go_setup_on_veros_recipe_raises(self):
        """GO supplies a linear EOS config; veros_faithful_v1 is non-linear —
        the assembler must REJECT the mismatch instead of silently ignoring the
        linear config and running the Veros EOS."""
        cfg = GlobalOverturningConfig()
        with pytest.raises(ValueError, match="non-linear EOS"):
            global_overturning_model_config(
                cfg, eos_config=create_eos_config(cfg),
                recipe="veros_faithful_v1")

    def test_nemo_dino_recipe_on_go_raises(self):
        """DINO's recipe is eos='wright' (non-linear) — also incompatible with
        GO's linear setup."""
        cfg = GlobalOverturningConfig()
        with pytest.raises(ValueError, match="non-linear EOS"):
            global_overturning_model_config(
                cfg, eos_config=create_eos_config(cfg), recipe="nemo_dino_v1")

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
        # non-linear + dangling eos_linear -> raise (the GO x veros footgun)
        with pytest.raises(ValueError, match="non-linear EOS"):
            assert_recipe_setup_compatible(
                SimpleNamespace(eos="veros_nonlin2", eos_linear=object()))
