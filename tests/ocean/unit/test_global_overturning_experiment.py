"""Direct unit tests for the global-overturning experiment *setup* helpers.

``test_global_overturning_model_config.py`` already pins the two model-config
factories.  This module covers the rest of
``ocean/experiments/global_overturning.py`` that builds the experiment's
initial condition, forcing, EOS, and result/diagnostic descriptors:

  * ``_add_stratification`` / ``create_initial_conditions`` — exponential T(z)
    profile (warm surface, cold abyss), continent mask applied, finite, right
    shape;
  * ``create_forcings`` — the ``combined`` wind + SST-restoring physics config
    with the config's coefficients threaded through;
  * ``create_eos_config`` — linear T-only buoyancy (``beta_S = 0``);
  * ``create_domain_config`` / ``get_diagnostic_field_specs`` /
    ``get_scalar_units`` — descriptor dictionaries;
  * ``validate_results`` — finiteness + speed sanity gate;
  * unsupported grid type raises (dispatch hardening).

Runs on a small (16x24) lat-lon grid; the MPAS branch and the model-config
factories are exercised elsewhere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig,
    create_domain_config,
    create_eos_config,
    create_forcings,
    create_initial_conditions,
    get_diagnostic_field_specs,
    get_scalar_units,
    validate_results,
)
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(scope="module")
def setup():
    cfg = GlobalOverturningConfig(n_levels=6)
    grid = create_latlon_grid(n_lat=16, n_lon=24)
    z = create_ocean_z_star(n_levels=6, H_max=cfg.H_max)
    return cfg, grid, z


class TestInitialConditions:
    def test_shape_and_finite(self, setup):
        cfg, grid, z = setup
        state = create_initial_conditions("latlon", grid, z, cfg)
        assert state.T.data.shape == (16, 24, 6)
        assert bool(jnp.all(jnp.isfinite(state.T.data)))
        assert bool(jnp.all(jnp.isfinite(state.eta.data)))

    def test_exponential_stratification_warm_surface_cold_deep(self, setup):
        cfg, grid, z = setup
        state = create_initial_conditions("latlon", grid, z, cfg)
        T = state.T.data
        # Surface layer warmer than the abyss, monotone-ish decay with depth.
        surf = float(jnp.mean(T[..., 0]))
        deep = float(jnp.mean(T[..., -1]))
        assert surf > deep
        # Stratification bounded by the config endpoints (deep ~ T_deep_C,
        # surface approaches T_water_init_C).
        assert deep >= cfg.T_deep_C - 0.5
        assert surf <= cfg.T_water_init_C + 0.5
        # Monotone non-increasing column mean with depth.
        col_mean = jnp.mean(T.reshape(-1, T.shape[-1]), axis=0)
        assert bool(jnp.all(jnp.diff(col_mean) <= 1e-6))

    def test_continent_mask_applied(self, setup):
        """The simplified-continent mask makes some cells land (mask < 1)."""
        cfg, grid, z = setup
        state = create_initial_conditions("latlon", grid, z, cfg)
        lm = state.land_mask.data
        # Mixed wet/dry domain (not all ocean, not all land).
        assert float(jnp.min(lm)) < 0.5 < float(jnp.max(lm))

    def test_unknown_grid_type_raises(self, setup):
        cfg, grid, z = setup
        with pytest.raises(NotImplementedError, match="cubed_sphere"):
            create_initial_conditions("cubed_sphere", grid, z, cfg)


class TestForcings:
    def test_combined_scheme_and_threaded_coefficients(self, setup):
        cfg, grid, _ = setup
        phys = create_forcings("latlon", grid, cfg)
        assert phys.surface_forcing.scheme == "combined"
        # Wind + restoring both wired with the config's numbers.
        assert phys.surface_forcing.prescribed.tau_max == cfg.tau_max
        assert phys.surface_forcing.restoring.tau_T == cfg.tau_T_days * 86400.0
        assert phys.surface_forcing.restoring.T_star_eq == cfg.T_star_eq
        # Vertical mixing = constant with the config A_v / K_v.
        assert phys.vertical_mixing.scheme == "constant"
        assert phys.vertical_mixing.constant.A_v == cfg.A_v
        assert phys.vertical_mixing.constant.K_v == cfg.K_v
        # Convection = enhanced_diffusion (statically-unstable adjustment).
        assert phys.convection.scheme == "enhanced_diffusion"

    def test_none_config_defaults(self, setup):
        _, grid, _ = setup
        phys = create_forcings("latlon", grid, None)
        assert phys.surface_forcing.scheme == "combined"


class TestEOSConfig:
    def test_temperature_only_buoyancy(self, setup):
        cfg, _, _ = setup
        eos = create_eos_config(cfg)
        assert eos.alpha_T == cfg.alpha_T
        assert eos.beta_S == 0.0           # T-only buoyancy (Wolfe & Cessi)
        assert eos.T_ref == cfg.T_ref_C


class TestDescriptors:
    def test_domain_config(self, setup):
        cfg, _, _ = setup
        d = create_domain_config(cfg)
        assert d["n_levels"] == cfg.n_levels
        assert d["H_max"] == cfg.H_max
        assert d["eos"] == "linear (T-only)"

    def test_field_specs_and_scalar_units(self):
        specs = get_diagnostic_field_specs()
        assert len(specs) == 3
        assert all(len(s) == 3 for s in specs)     # (name, label, cmap)
        units = get_scalar_units()
        assert units["max_speed"] == "m/s"
        assert set(units) >= {"mean_eta", "max_speed", "mean_T"}


class TestValidateResults:
    def test_healthy_run_passes(self, setup):
        cfg, grid, z = setup
        state = create_initial_conditions("latlon", grid, z, cfg)
        diag = {"max_speed": [0.4, 0.5], "mean_eta": [0.0, 0.01],
                "mean_T": [10.0]}
        ok, note = validate_results(state, diag, cfg)
        assert ok is True
        assert "max_speed=0.5000m/s" in note

    def test_nan_state_fails(self, setup):
        cfg, grid, z = setup
        state = create_initial_conditions("latlon", grid, z, cfg)
        bad = state.T.data.at[0, 0, 0].set(jnp.nan)
        state = state._replace(T=state.T.replace(data=bad))
        ok, note = validate_results(state, {}, cfg)
        assert ok is False
        assert "NaN/Inf" in note

    def test_excessive_speed_fails(self, setup):
        cfg, grid, z = setup
        state = create_initial_conditions("latlon", grid, z, cfg)
        ok, note = validate_results(state, {"max_speed": [7.0]}, cfg)
        assert ok is False
        assert "FAIL" in note
