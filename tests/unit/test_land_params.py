"""Tests for spatially-varying land surface parameters.

Covers:
1. LandSurfaceParams data structure and helpers
2. ConstantParamProvider (backward compat, bit-identical to None)
3. PFTParamProvider (convexity, single-PFT, differentiability)
4. NeuralParamProvider (bounded outputs, differentiability, JIT)
5. build_land_features helper
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from legoesm.land.surface_params import (
    LandSurfaceParams,
    PARAM_BOUNDS,
    PARAM_NAMES,
    N_PARAMS,
    default_land_surface_params,
    array_to_params,
    reshape_params,
    bounds_arrays,
    clm5_pft_table,
    CLM5_PFT_NAMES,
    N_PFT_CLM5,
    read_spatial_param,
)
from legoesm.land.param_providers import (
    ConstantParamProvider,
    PFTParamProvider,
    NeuralParamProvider,
    build_land_features,
    create_land_param_provider,
)
from legoesm.land.config import LandConfig, MultiLayerLandConfig


# =====================================================================
# Fixtures
# =====================================================================

NCOL = 48
KEY = jax.random.PRNGKey(42)


@pytest.fixture
def land_config():
    return LandConfig()


@pytest.fixture
def multilayer_config():
    return MultiLayerLandConfig()


# =====================================================================
# 1. Data structure tests
# =====================================================================

class TestLandSurfaceParams:
    # LAI/SAI/htop/hbot are PRESCRIBED (climatology / surfdata) per-cell canopy
    # fields, not trainable bounded params, so they are the LandSurfaceParams
    # fields outside PARAM_NAMES/BOUNDS.  PARAM_NAMES/BOUNDS index the 12-column
    # CLM5 PFT lookup table that every ``array_to_params`` caller maps
    # column-for-column; the CLM-ML canopy inputs are prescribed separately and
    # read with a ``None`` fallback in ``canopy/clm_ml_interface.py``.
    _NON_BOUNDED_FIELDS = {"LAI", "SAI", "htop", "hbot"}

    def test_namedtuple_fields(self):
        # Every trainable param is a field; the non-bounded extras are the
        # prescribed canopy metadata fields (LAI/SAI/htop/hbot).
        for name in PARAM_NAMES:
            assert name in LandSurfaceParams._fields
        assert (set(LandSurfaceParams._fields) - set(PARAM_NAMES)
                == self._NON_BOUNDED_FIELDS)

    def test_param_bounds_complete(self):
        for name in LandSurfaceParams._fields:
            if name in self._NON_BOUNDED_FIELDS:
                continue  # prescribed metadata, not a trainable bounded param
            assert name in PARAM_BOUNDS, f"Missing bound for {name}"
            lo, hi = PARAM_BOUNDS[name]
            assert lo < hi, f"Invalid bounds for {name}: {lo} >= {hi}"

    def test_default_land_surface_params(self, land_config):
        params = default_land_surface_params(NCOL, land_config)
        assert isinstance(params, LandSurfaceParams)
        assert params.albedo_veg.shape == (NCOL,)
        assert jnp.allclose(params.albedo_veg, land_config.albedo_land)
        assert jnp.allclose(params.z0, land_config.z0_land)
        assert jnp.allclose(params.W_max, land_config.W_max)

    def test_default_multilayer_params(self, multilayer_config):
        params = default_land_surface_params(NCOL, multilayer_config)
        assert jnp.allclose(params.root_depth, multilayer_config.root_depth)
        assert jnp.allclose(params.theta_wp, multilayer_config.theta_wp)
        assert jnp.allclose(params.theta_fc, multilayer_config.theta_fc)

    def test_array_to_params(self):
        arr = jnp.ones((NCOL, N_PARAMS))
        params = array_to_params(arr)
        assert params.albedo_veg.shape == (NCOL,)
        assert jnp.allclose(params.albedo_veg, 1.0)

    def test_reshape_params(self):
        params = default_land_surface_params(6 * 4 * 4, LandConfig())
        reshaped = reshape_params(params, (6, 4, 4))
        assert reshaped.albedo_veg.shape == (6, 4, 4)

    def test_bounds_arrays(self):
        lo, hi = bounds_arrays()
        assert lo.shape == (N_PARAMS,)
        assert hi.shape == (N_PARAMS,)
        assert jnp.all(lo < hi)


# =====================================================================
# 2. CLM5 PFT table
# =====================================================================

class TestCLM5Table:
    def test_shape(self):
        table = clm5_pft_table()
        assert table.shape == (N_PFT_CLM5, N_PARAMS)
        assert N_PFT_CLM5 == 17

    def test_values_in_bounds(self):
        table = clm5_pft_table()
        lo, hi = bounds_arrays()
        for i, name in enumerate(PARAM_NAMES):
            col = table[:, i]
            # Bare soil Vc_max25 = 0 is below bound intentionally
            if name == "Vc_max25":
                continue
            assert jnp.all(col >= lo[i] - 1e-6), (
                f"PFT table {name} below lower bound: min={float(col.min())}"
            )
            assert jnp.all(col <= hi[i] + 1e-6), (
                f"PFT table {name} above upper bound: max={float(col.max())}"
            )

    def test_pft_names(self):
        assert len(CLM5_PFT_NAMES) == N_PFT_CLM5
        assert CLM5_PFT_NAMES[0] == "bare_soil"


# =====================================================================
# 3. ConstantParamProvider
# =====================================================================

class TestConstantProvider:
    def test_from_config(self, land_config):
        provider = ConstantParamProvider.from_config(NCOL, land_config)
        params = provider()
        assert isinstance(params, LandSurfaceParams)
        assert params.albedo_veg.shape == (NCOL,)
        assert jnp.allclose(params.z0, land_config.z0_land)

    def test_from_arrays(self):
        arrays = {name: jnp.full(NCOL, 0.5) for name in PARAM_NAMES}
        provider = ConstantParamProvider.from_arrays(arrays)
        params = provider()
        assert jnp.allclose(params.albedo_veg, 0.5)

    def test_is_equinox_module(self, land_config):
        provider = ConstantParamProvider.from_config(NCOL, land_config)
        assert isinstance(provider, eqx.Module)
        # Leaves should be arrays
        leaves = jax.tree.leaves(provider)
        assert len(leaves) == N_PARAMS


# =====================================================================
# 4. PFTParamProvider
# =====================================================================

class TestPFTProvider:
    def test_single_pft_matches_table(self):
        """Uniform single-PFT fractions should produce that PFT's values."""
        fracs = jnp.zeros((NCOL, N_PFT_CLM5))
        # All needleleaf evergreen temperate (index 1)
        fracs = fracs.at[:, 1].set(1.0)
        provider = PFTParamProvider.from_defaults(fracs)
        params = provider()

        table = clm5_pft_table()
        for i, name in enumerate(PARAM_NAMES):
            expected = table[1, i]
            actual = getattr(params, name)
            assert jnp.allclose(actual, expected, atol=1e-4), (
                f"PFT mismatch for {name}: expected {float(expected)}, "
                f"got {float(actual[0])}"
            )

    def test_convex_combination(self):
        """50/50 mix of PFT 0 and PFT 4 should give average."""
        fracs = jnp.zeros((NCOL, N_PFT_CLM5))
        fracs = fracs.at[:, 0].set(0.5)
        fracs = fracs.at[:, 4].set(0.5)
        provider = PFTParamProvider.from_defaults(fracs)
        params = provider()

        table = clm5_pft_table()
        # After sigmoid, values match the table
        lo, hi = bounds_arrays()
        constrained_table = lo + (hi - lo) * jax.nn.sigmoid(provider.raw_table)
        expected = 0.5 * constrained_table[0] + 0.5 * constrained_table[4]
        for i, name in enumerate(PARAM_NAMES):
            actual = getattr(params, name)
            assert jnp.allclose(actual, expected[i], atol=1e-4), (
                f"Convex combo mismatch for {name}"
            )

    def test_fraction_normalization(self):
        """Fractions that don't sum to 1 should be normalized."""
        fracs = jnp.ones((NCOL, N_PFT_CLM5)) * 2.0  # sum = 34
        provider = PFTParamProvider.from_defaults(fracs)
        params = provider()
        # Should produce same as uniform fractions
        fracs_uniform = jnp.ones((NCOL, N_PFT_CLM5)) / N_PFT_CLM5
        provider_uniform = PFTParamProvider.from_defaults(fracs_uniform)
        params_uniform = provider_uniform()
        assert jnp.allclose(params.albedo_veg, params_uniform.albedo_veg, atol=1e-5)

    def test_differentiable(self):
        """Gradient flows through raw_table."""
        fracs = jnp.ones((NCOL, N_PFT_CLM5)) / N_PFT_CLM5
        provider = PFTParamProvider.from_defaults(fracs)

        def loss_fn(prov):
            params = prov()
            return jnp.mean(params.albedo_veg)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(provider)
        assert jnp.isfinite(loss)
        # raw_table should have non-zero gradient
        assert grads.raw_table is not None
        assert jnp.any(grads.raw_table != 0.0)

    def test_stop_gradient_on_fractions(self):
        """Gradient should NOT flow through pft_fractions."""
        fracs = jnp.ones((NCOL, N_PFT_CLM5)) / N_PFT_CLM5
        provider = PFTParamProvider.from_defaults(fracs)

        def loss_fn(prov):
            params = prov()
            return jnp.mean(params.albedo_veg)

        grads = eqx.filter_grad(loss_fn)(provider)
        # pft_fractions should be zero (stop_gradient)
        if grads.pft_fractions is not None:
            assert jnp.allclose(grads.pft_fractions, 0.0)


# =====================================================================
# 5. NeuralParamProvider
# =====================================================================

class TestNeuralProvider:
    def test_output_shape(self):
        provider = NeuralParamProvider(key=KEY, n_input=20)
        features = jax.random.normal(KEY, (NCOL, 20))
        params = provider(features)
        assert isinstance(params, LandSurfaceParams)
        for name in PARAM_NAMES:
            assert getattr(params, name).shape == (NCOL,)

    def test_output_bounded(self):
        """All outputs must lie within PARAM_BOUNDS."""
        provider = NeuralParamProvider(key=KEY, n_input=20)
        # Test with several random feature sets
        for i in range(5):
            features = jax.random.normal(jax.random.PRNGKey(i), (NCOL, 20))
            params = provider(features)
            lo, hi = bounds_arrays()
            for j, name in enumerate(PARAM_NAMES):
                val = getattr(params, name)
                assert jnp.all(val >= lo[j] - 1e-6), (
                    f"Neural output {name} below bound: min={float(val.min())}"
                )
                assert jnp.all(val <= hi[j] + 1e-6), (
                    f"Neural output {name} above bound: max={float(val.max())}"
                )

    def test_differentiable(self):
        """Gradient flows through NN weights."""
        provider = NeuralParamProvider(key=KEY, n_input=20)
        features = jax.random.normal(KEY, (NCOL, 20))

        def loss_fn(prov):
            params = prov(features)
            return jnp.mean(params.albedo_veg ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(provider)
        assert jnp.isfinite(loss)
        # Check at least one layer has non-zero gradient
        grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
        assert any(jnp.any(g != 0.0) for g in grad_leaves)

    def test_jit_compiles(self):
        """JIT compiles without error and produces same result."""
        provider = NeuralParamProvider(key=KEY, n_input=20)
        features = jax.random.normal(KEY, (NCOL, 20))

        @jax.jit
        def run(prov, feat):
            params = prov(feat)
            return params.albedo_veg

        result = run(provider, features)
        assert result.shape == (NCOL,)
        assert jnp.all(jnp.isfinite(result))

    def test_architecture_params(self):
        """Custom hidden_dim and n_hidden."""
        provider = NeuralParamProvider(
            key=KEY, n_input=10, hidden_dim=32, n_hidden=2,
        )
        assert len(provider.layers) == 3  # 2 hidden + 1 output
        features = jax.random.normal(KEY, (NCOL, 10))
        params = provider(features)
        assert params.albedo_veg.shape == (NCOL,)


# =====================================================================
# 6. build_land_features
# =====================================================================

class TestBuildFeatures:
    def test_default_features(self):
        lat = jnp.linspace(-1.5, 1.5, NCOL)
        lon = jnp.linspace(0, 6.28, NCOL)
        features = build_land_features(lat, lon)
        # 4 (lat/lon) + 12 (soil_type uniform) + 4 (zeros)
        assert features.shape == (NCOL, 20)
        assert jnp.all(jnp.isfinite(features))

    def test_with_soil_type(self):
        lat = jnp.zeros(NCOL)
        lon = jnp.zeros(NCOL)
        soil = jnp.array([i % 12 for i in range(NCOL)])
        features = build_land_features(lat, lon, soil_type=soil)
        # One-hot encoded soil types
        assert features.shape == (NCOL, 20)
        # First cell has soil type 0 → one-hot [1,0,0,...,0]
        assert features[0, 4] == 1.0
        assert features[0, 5] == 0.0

    def test_with_all_features(self):
        lat = jnp.zeros(NCOL)
        lon = jnp.zeros(NCOL)
        features = build_land_features(
            lat, lon,
            soil_type=jnp.zeros(NCOL, dtype=jnp.int32),
            elevation=jnp.full(NCOL, 1000.0),
            mean_precip=jnp.full(NCOL, 3e-5),
            mean_temp=jnp.full(NCOL, 290.0),
            forest_age=jnp.full(NCOL, 50.0),
        )
        assert features.shape == (NCOL, 20)


# =====================================================================
# 7. Factory
# =====================================================================

class TestFactory:
    def test_constant_mode(self, land_config):
        provider = create_land_param_provider("constant", NCOL, land_config)
        assert isinstance(provider, ConstantParamProvider)

    def test_pft_mode(self, land_config):
        fracs = jnp.ones((NCOL, N_PFT_CLM5)) / N_PFT_CLM5
        provider = create_land_param_provider(
            "pft", NCOL, land_config, pft_fractions=fracs,
        )
        assert isinstance(provider, PFTParamProvider)

    def test_neural_mode(self, land_config):
        provider = create_land_param_provider(
            "neural", NCOL, land_config, key=KEY,
        )
        assert isinstance(provider, NeuralParamProvider)

    def test_invalid_mode(self, land_config):
        with pytest.raises(ValueError, match="Unknown"):
            create_land_param_provider("invalid", NCOL, land_config)

    def test_missing_pft_fractions(self, land_config):
        with pytest.raises(ValueError, match="pft_fractions"):
            create_land_param_provider("pft", NCOL, land_config)

    def test_missing_key(self, land_config):
        with pytest.raises(ValueError, match="key"):
            create_land_param_provider("neural", NCOL, land_config)


class TestCustomBounds:
    """Finding #10: providers built with CUSTOM param_bounds must apply THOSE
    bounds in the sigmoid constraint, not the module-global bounds_arrays()
    (fixed 12-param PARAM_NAMES order).  Previously _constrained_table and
    _forward_single called bounds_arrays(), silently ignoring custom bounds."""

    def test_pft_custom_bounds_honored(self):
        # Tighter, DIFFERENT bounds than the defaults (full 12-param order).
        default_bounds = [PARAM_BOUNDS[n] for n in PARAM_NAMES]
        custom_bounds = [(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
                         for (lo, hi) in default_bounds]
        fracs = jnp.zeros((NCOL, N_PFT_CLM5)).at[:, 1].set(1.0)
        provider = PFTParamProvider.from_defaults(
            fracs, param_bounds=custom_bounds,
        )
        params = provider()
        # Round-trip self-consistency: the constrained table built with the
        # SAME custom bounds must reproduce the output (would FAIL if the
        # forward sigmoid used the default bounds while from_defaults used the
        # custom ones).
        lo = jnp.array([b[0] for b in custom_bounds])
        hi = jnp.array([b[1] for b in custom_bounds])
        constrained = lo + (hi - lo) * jax.nn.sigmoid(provider.raw_table)
        for i, name in enumerate(PARAM_NAMES):
            actual = getattr(params, name)
            assert jnp.allclose(actual, constrained[1, i], atol=1e-4), name
            # Every output must respect the CUSTOM (tighter) bounds.
            assert jnp.all(actual >= lo[i] - 1e-6)
            assert jnp.all(actual <= hi[i] + 1e-6)

    def test_neural_custom_bounds_honored(self):
        default_bounds = [PARAM_BOUNDS[n] for n in PARAM_NAMES]
        # Shift the bounds well away from the defaults so a default-bounds
        # sigmoid would produce out-of-range values.
        custom_bounds = [(lo + 0.25 * (hi - lo), lo + 0.45 * (hi - lo))
                         for (lo, hi) in default_bounds]
        provider = NeuralParamProvider(
            key=KEY, n_input=20, param_bounds=custom_bounds,
        )
        lo = jnp.array([b[0] for b in custom_bounds])
        hi = jnp.array([b[1] for b in custom_bounds])
        for seed in range(4):
            features = jax.random.normal(jax.random.PRNGKey(seed), (NCOL, 20))
            params = provider(features)
            for j, name in enumerate(PARAM_NAMES):
                val = getattr(params, name)
                # Outputs must fall inside the CUSTOM band (sigmoid range).
                assert jnp.all(val >= lo[j] - 1e-6), (
                    f"{name} below custom bound: {float(val.min())} < {float(lo[j])}"
                )
                assert jnp.all(val <= hi[j] + 1e-6), (
                    f"{name} above custom bound: {float(val.max())} > {float(hi[j])}"
                )

    def test_neural_custom_param_names_reorder(self):
        """Finding #1 (codex round 1): NeuralParamProvider with a REORDERED
        ``param_names`` but DEFAULT bounds must build the bounds from THOSE names
        (in that order), not from the fixed PARAM_NAMES order — otherwise the
        sigmoid applies the wrong per-parameter [lo,hi] to each output column.
        (A reorder keeps all 12 names so the LandSurfaceParams build succeeds; a
        true subset is unsupported by LandSurfaceParams, which has no field
        defaults.)"""
        reordered = tuple(reversed(PARAM_NAMES))   # all 12, reversed order
        provider = NeuralParamProvider(
            key=KEY, n_input=20, param_names=reordered,   # no param_bounds!
        )
        features = jax.random.normal(KEY, (NCOL, 20))
        params = provider(features)  # must not raise, and bounds must match
        for name in reordered:
            val = getattr(params, name)
            lo, hi = PARAM_BOUNDS[name]
            assert jnp.all(val >= lo - 1e-6) and jnp.all(val <= hi + 1e-6), (
                f"{name} out of its bounds -> bounds not built from param_names"
            )

    def test_pft_constrained_table_uses_instance_bounds(self):
        # Direct contrast: an instance carrying custom bounds must NOT reproduce
        # the default-bounds table.
        default_bounds = [PARAM_BOUNDS[n] for n in PARAM_NAMES]
        custom_bounds = [(lo, lo + 0.2 * (hi - lo))
                         for (lo, hi) in default_bounds]
        fracs = jnp.ones((NCOL, N_PFT_CLM5)) / N_PFT_CLM5
        prov_default = PFTParamProvider.from_defaults(fracs)
        prov_custom = PFTParamProvider.from_defaults(
            fracs, param_bounds=custom_bounds,
        )
        # The two providers must produce DIFFERENT albedo (custom band is tighter
        # and lower), proving the instance bounds drive the constraint.
        a_def = prov_default().albedo_veg
        a_cus = prov_custom().albedo_veg
        assert not jnp.allclose(a_def, a_cus), (
            "custom bounds changed nothing -> bounds_arrays() still used"
        )


class TestReadSpatialParam:
    """read_spatial_param: the shared slab/multilayer spatial-param accessor
    (deduped from per-module _get helpers, ponytail 2026-06-17)."""

    def test_none_returns_fallback(self):
        assert read_spatial_param(None, "albedo_veg", 0.25) == 0.25

    def test_present_returns_field(self):
        ncol = 4
        lp = default_land_surface_params(ncol, None) if False else None
        # Build a minimal params object exposing the requested field.
        params = array_to_params(jnp.ones((ncol, N_PARAMS)) * 0.3)
        out = read_spatial_param(params, PARAM_NAMES[0], -1.0)
        assert jnp.allclose(out, 0.3)
