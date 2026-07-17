"""Unit tests for the SFNO architecture and ML modules."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx

from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.ml.normalization import (
    NormalizationStats,
    normalize,
    denormalize,
    compute_normalization_stats,
    save_normalization_stats,
    load_normalization_stats,
)
from legoesm.ml.spectral_conv import SpectralConv
from legoesm.ml.sfno_block import SFNOBlock
from legoesm.ml.sfno import SFNO, SFNOConfig

# Enable float64 for SHT accuracy
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def grid_t10():
    """Small T10 grid for fast tests."""
    return create_gaussian_grid(n_max=10)


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(42)


# =============================================================================
# Normalization tests
# =============================================================================

class TestNormalization:

    def test_round_trip(self):
        """Normalize then denormalize should recover the original."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        stats = NormalizationStats(
            mean=jnp.array([2.0, 3.0]),
            std=jnp.array([1.0, 1.0]),
        )
        recovered = denormalize(normalize(x, stats), stats)
        np.testing.assert_allclose(recovered, x, atol=1e-6)

    def test_compute_stats(self):
        """Computed stats should give zero-mean, unit-std after normalization."""
        rng = np.random.default_rng(0)
        data = jnp.array(rng.standard_normal((100, 3)))
        stats = compute_normalization_stats(data)
        normed = normalize(data, stats)
        np.testing.assert_allclose(jnp.mean(normed, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(jnp.std(normed, axis=0), 1.0, atol=1e-2)

    def test_eps_floor(self):
        """Channels with zero variance should get eps-floored std."""
        data = jnp.ones((10, 2))
        stats = compute_normalization_stats(data, eps=1e-6)
        assert jnp.all(stats.std >= 1e-6)

    def test_compute_stats_per_channel_and_std_floor(self):
        """Per-channel mean/std correct; a constant channel is std-floored.

        Guards the sfno_full stats path: channel 0 varies (mean/std computable),
        channel 1 is CONSTANT (var 0 -> std must be floored, not 0).
        """
        eps = 1e-6
        # (n_samples, n_spatial, n_channels): ch0 = [1,3] per sample (mean 2,
        # population std 1), ch1 constant 5.0.
        data = jnp.array(
            [[[1.0, 5.0], [3.0, 5.0]],
             [[1.0, 5.0], [3.0, 5.0]]]
        )
        stats = compute_normalization_stats(data, eps=eps)
        np.testing.assert_allclose(stats.mean, [2.0, 5.0], atol=1e-6)
        # ch0 population std = 1.0; ch1 constant -> std floored to eps.
        np.testing.assert_allclose(stats.std[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(stats.std[1], eps, atol=1e-12)
        assert jnp.all(stats.std >= eps)

    def test_sidecar_round_trip(self, tmp_path):
        """save_normalization_stats -> load_normalization_stats preserves values.

        This is the checkpoint-compat mechanism for sfno_full: the eqx
        checkpoint serialises only the SFNO leaves, so norm_stats travel in an
        .npz sidecar. Train + eval must read identical values.
        """
        stats = NormalizationStats(
            mean=jnp.array([1.0, -2.5, 1.0e5, 3.0e-3]),
            std=jnp.array([0.5, 1.0e-6, 900.0, 4.0e-4]),
        )
        path = tmp_path / "norm_stats.npz"
        save_normalization_stats(stats, path)
        assert path.exists()
        loaded = load_normalization_stats(path)
        np.testing.assert_allclose(loaded.mean, stats.mean, rtol=0, atol=0)
        np.testing.assert_allclose(loaded.std, stats.std, rtol=0, atol=0)
        # Round-trips as a working NormalizationStats (normalize/denormalize).
        x = jnp.array([2.0, -2.5, 1.0e5 + 900.0, 3.0e-3])
        rec = denormalize(normalize(x, loaded), loaded)
        np.testing.assert_allclose(rec, x, rtol=1e-6)

    def test_save_rejects_shape_mismatch(self, tmp_path):
        bad = NormalizationStats(mean=jnp.zeros(3), std=jnp.ones(4))
        with pytest.raises(ValueError):
            save_normalization_stats(bad, tmp_path / "bad.npz")

    def test_load_missing_arrays_raises(self, tmp_path):
        path = tmp_path / "wrong.npz"
        np.savez(path, foo=np.zeros(3))
        with pytest.raises(KeyError):
            load_normalization_stats(path)


# =============================================================================
# SpectralConv tests
# =============================================================================

class TestSpectralConv:

    def test_output_shape(self, grid_t10, key):
        """SpectralConv should produce correct output shape."""
        n_sh = grid_t10.n_sh
        conv = SpectralConv(n_sh=n_sh, in_channels=8, out_channels=16, key=key)
        coeffs = jnp.ones((n_sh, 8), dtype=jnp.complex128)
        out = conv(coeffs)
        assert out.shape == (n_sh, 16)
        assert jnp.iscomplexobj(out)

    def test_jit_compatible(self, grid_t10, key):
        """SpectralConv should be JIT-compatible."""
        n_sh = grid_t10.n_sh
        conv = SpectralConv(n_sh=n_sh, in_channels=4, out_channels=4, key=key)
        coeffs = jnp.ones((n_sh, 4), dtype=jnp.complex128)

        @jax.jit
        def apply(conv, x):
            return conv(x)

        out = apply(conv, coeffs)
        assert out.shape == (n_sh, 4)

    def test_weight_shapes(self, key):
        """Weight arrays should have correct shapes."""
        conv = SpectralConv(n_sh=66, in_channels=8, out_channels=16, key=key)
        assert conv.weight_real.shape == (66, 16, 8)
        assert conv.weight_imag.shape == (66, 16, 8)


# =============================================================================
# SFNOBlock tests
# =============================================================================

class TestSFNOBlock:

    def test_output_shape(self, grid_t10, key):
        """SFNOBlock should preserve spatial shape and embed_dim."""
        embed_dim = 16
        block = SFNOBlock(grid=grid_t10, embed_dim=embed_dim, key=key)
        x = jnp.ones(
            (grid_t10.n_lat, grid_t10.n_lon, embed_dim),
            dtype=jnp.float32,
        )
        out = block(x, grid_t10)
        assert out.shape == x.shape

    def test_skip_connection(self, grid_t10, key):
        """Output should differ from input (non-trivial transform) but
        the skip connection means it's not too far from input."""
        embed_dim = 16
        block = SFNOBlock(grid=grid_t10, embed_dim=embed_dim, key=key)
        x = jnp.ones(
            (grid_t10.n_lat, grid_t10.n_lon, embed_dim),
            dtype=jnp.float32,
        )
        out = block(x, grid_t10)
        # Should not be identical (learned transform is non-trivial)
        assert not jnp.allclose(out, x)

    def test_no_nans(self, grid_t10, key):
        """SFNOBlock should not produce NaNs."""
        embed_dim = 16
        block = SFNOBlock(grid=grid_t10, embed_dim=embed_dim, key=key)
        rng = jax.random.PRNGKey(123)
        x = jax.random.normal(
            rng, (grid_t10.n_lat, grid_t10.n_lon, embed_dim),
            dtype=jnp.float32,
        )
        out = block(x, grid_t10)
        assert not jnp.any(jnp.isnan(out))


# =============================================================================
# Full SFNO model tests
# =============================================================================

class TestSFNO:

    def test_forward_pass_shape(self, grid_t10, key):
        """SFNO forward pass should produce correct output shape."""
        config = SFNOConfig(
            in_channels=4, out_channels=4,
            embed_dim=16, n_blocks=2, mlp_expansion=2,
        )
        model = SFNO(config=config, grid=grid_t10, key=key)
        x = jnp.ones(
            (grid_t10.n_lat, grid_t10.n_lon, 4),
            dtype=jnp.float32,
        )
        out = model(x, grid_t10)
        assert out.shape == (grid_t10.n_lat, grid_t10.n_lon, 4)

    def test_residual_prediction(self, grid_t10, key):
        """With residual_prediction=True, output should be close to input
        at initialization (small random weights)."""
        config = SFNOConfig(
            in_channels=4, out_channels=4,
            embed_dim=16, n_blocks=2,
            residual_prediction=True,
        )
        model = SFNO(config=config, grid=grid_t10, key=key)
        x = jnp.ones(
            (grid_t10.n_lat, grid_t10.n_lon, 4),
            dtype=jnp.float32,
        )
        out = model(x, grid_t10)
        # Residual prediction: output = input + small correction
        diff = jnp.mean(jnp.abs(out - x))
        # At init, the correction should be modest (not orders of magnitude)
        assert diff < 10.0

    def test_no_residual_prediction(self, grid_t10, key):
        """Without residual prediction, output can differ significantly."""
        config = SFNOConfig(
            in_channels=4, out_channels=4,
            embed_dim=16, n_blocks=2,
            residual_prediction=False,
        )
        model = SFNO(config=config, grid=grid_t10, key=key)
        x = jnp.ones(
            (grid_t10.n_lat, grid_t10.n_lon, 4),
            dtype=jnp.float32,
        )
        out = model(x, grid_t10)
        assert out.shape == x.shape

    def test_gradient_flow(self, grid_t10, key):
        """Gradients should flow through the full SFNO without NaNs."""
        config = SFNOConfig(
            in_channels=4, out_channels=4,
            embed_dim=16, n_blocks=2,
        )
        model = SFNO(config=config, grid=grid_t10, key=key)
        x = jnp.ones(
            (grid_t10.n_lat, grid_t10.n_lon, 4),
            dtype=jnp.float32,
        )

        def loss_fn(model):
            out = model(x, grid_t10)
            return jnp.mean(out ** 2)

        grads = eqx.filter_grad(loss_fn)(model)

        # Check encoder gradient exists and has no NaNs
        assert grads.encoder.weight is not None
        assert not jnp.any(jnp.isnan(grads.encoder.weight))

        # Check decoder gradient
        assert grads.decoder.weight is not None
        assert not jnp.any(jnp.isnan(grads.decoder.weight))

    def test_different_in_out_channels(self, grid_t10, key):
        """SFNO should work with different input/output channel counts."""
        config = SFNOConfig(
            in_channels=8, out_channels=3,
            embed_dim=16, n_blocks=2,
            residual_prediction=False,
        )
        model = SFNO(config=config, grid=grid_t10, key=key)
        x = jnp.ones(
            (grid_t10.n_lat, grid_t10.n_lon, 8),
            dtype=jnp.float32,
        )
        out = model(x, grid_t10)
        assert out.shape == (grid_t10.n_lat, grid_t10.n_lon, 3)

    def test_no_nans_random_input(self, grid_t10, key):
        """SFNO should not produce NaNs for random inputs."""
        config = SFNOConfig(
            in_channels=4, out_channels=4,
            embed_dim=16, n_blocks=2,
        )
        model = SFNO(config=config, grid=grid_t10, key=key)
        x = jax.random.normal(
            jax.random.PRNGKey(99),
            (grid_t10.n_lat, grid_t10.n_lon, 4),
            dtype=jnp.float32,
        )
        out = model(x, grid_t10)
        assert not jnp.any(jnp.isnan(out))
        assert jnp.all(jnp.isfinite(out))
