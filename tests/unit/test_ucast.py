"""Unit tests for the U-Cast convolutional U-Net backbone.

Exercises :class:`~legoesm.ml.ucast.UCast` (a JAX/Equinox port of the
DhariwalUNet used by Rose-STL-Lab's U-Cast) directly: forward shapes, the
zero-init residual-identity contract, MC-Dropout stochasticity/reproducibility,
the self-attention path, geographic (circular-longitude) padding, GroupNorm
divisibility guarding, and end-to-end differentiability.

A tiny network (2 levels, ``model_channels=8``) on a small grid keeps the
tests fast while still wiring together the stem conv, down/up-sampling blocks,
skip concatenation, attention and the zero-init head.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from legoesm.ml.ucast import UCast, UCastConfig, UNetBlock, GeoConv2d, GroupNorm

# Conv U-Net is happy in float32, but the wider repo enables x64 in many
# fixtures; run these here under x64 to prove the dtype-cast path holds.
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Helpers
# =============================================================================

def _tiny_config(**overrides):
    base = dict(
        in_channels=6, out_channels=6, model_channels=8,
        channel_mult=(1, 2), num_blocks=1, attn_levels=(), dropout=0.3,
    )
    base.update(overrides)
    return UCastConfig(**base)


def _wake_residual_branches(model, *, scale=0.1, seed=5):
    """Make the zero-initialised conv1 / out_conv weights non-zero.

    At initialisation every residual branch and the head are zero, so the
    network is the identity and dropout has no visible effect (correct, but
    untestable).  This emulates a trained network by injecting small random
    weights into the otherwise-zeroed conv1 of every block and the output
    conv, so the dropout signal can propagate.
    """
    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    model = eqx.tree_at(
        lambda m: m.out_conv.weight, model,
        jax.random.normal(k, model.out_conv.weight.shape) * scale,
    )

    def wake(block, bkey):
        if isinstance(block, UNetBlock):
            return eqx.tree_at(
                lambda b: b.conv1.weight, block,
                jax.random.normal(bkey, block.conv1.weight.shape) * scale,
            )
        return block

    keys = jax.random.split(key, len(model.enc) + len(model.dec))
    new_enc = [wake(b, keys[i]) for i, b in enumerate(model.enc)]
    new_dec = [wake(b, keys[len(model.enc) + i]) for i, b in enumerate(model.dec)]
    return eqx.tree_at(lambda m: (m.enc, m.dec), model, (new_enc, new_dec))


# =============================================================================
# Forward shapes / contract
# =============================================================================

class TestForward:

    def test_shapes_and_finite(self):
        m = UCast(_tiny_config(), key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        y = m(x)
        assert y.shape == (16, 32, 6)
        assert jnp.all(jnp.isfinite(y))

    def test_residual_identity_at_init(self):
        """Zero-init head + ``residual_prediction`` ⇒ untrained net is identity."""
        m = UCast(_tiny_config(), key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        np.testing.assert_allclose(np.asarray(m(x)), np.asarray(x), atol=1e-6)

    def test_no_residual_changes_output(self):
        """With ``residual_prediction=False`` the (zero-init) output is ~0."""
        m = UCast(_tiny_config(residual_prediction=False), key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        y = m(x)
        assert float(jnp.max(jnp.abs(y))) < 1e-6

    def test_odd_grid_dimensions(self):
        """Odd spatial sizes survive the down/up-sample + skip-interp path."""
        m = UCast(_tiny_config(), key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (15, 30, 6))
        y = m(x)
        assert y.shape == (15, 30, 6)
        assert jnp.all(jnp.isfinite(y))

    def test_in_out_channels_differ(self):
        m = UCast(_tiny_config(in_channels=6, out_channels=4),
                  key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        y = m(x)
        assert y.shape == (16, 32, 4)


# =============================================================================
# Attention
# =============================================================================

class TestAttention:

    def test_attention_path_runs(self):
        """With ``channels_per_head`` small enough that ``num_heads > 0`` the
        self-attention branch is wired in and stays finite."""
        cfg = _tiny_config(attn_levels=(1,), channels_per_head=4)
        m = UCast(cfg, key=jax.random.PRNGKey(0))
        # At least one block must actually have heads.
        n_attn = sum(
            getattr(b, "num_heads", 0) > 0 for b in (m.enc + m.dec)
        )
        assert n_attn > 0
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        y = m(x)
        assert y.shape == (16, 32, 6)
        assert jnp.all(jnp.isfinite(y))

    def test_large_channels_per_head_disables_attention(self):
        """``num_heads = out // channels_per_head`` floors to 0 when the head
        size exceeds the channel count (matches the reference)."""
        cfg = _tiny_config(attn_levels=(0, 1), channels_per_head=10_000)
        m = UCast(cfg, key=jax.random.PRNGKey(0))
        assert all(getattr(b, "num_heads", 0) == 0 for b in (m.enc + m.dec))

    def test_indivisible_heads_raise(self):
        """out_channels not divisible by num_heads must fail fast (the
        attention reshape would otherwise crash)."""
        from legoesm.ml.ucast import UNetBlock
        # out_channels=10, channels_per_head=4 -> num_heads=2, 10%2==0 ok;
        # out_channels=10, channels_per_head=3 -> num_heads=3, 10%3 != 0 -> raise.
        with pytest.raises(ValueError, match="divisible"):
            UNetBlock(10, 10, attention=True, channels_per_head=3,
                      key=jax.random.PRNGKey(0))


# =============================================================================
# Conditioning channels
# =============================================================================

class TestConditioning:

    def test_condition_concatenated(self):
        """With ``num_conditional_channels > 0`` the model accepts and uses a
        condition tensor (channels added to the stem)."""
        cfg = _tiny_config(in_channels=6, out_channels=6,
                           num_conditional_channels=2)
        m = UCast(cfg, key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        cond = jax.random.normal(jax.random.PRNGKey(2), (16, 32, 2))
        y = m(x, condition=cond)
        assert y.shape == (16, 32, 6)
        assert jnp.all(jnp.isfinite(y))

    def test_missing_condition_raises(self):
        cfg = _tiny_config(num_conditional_channels=2)
        m = UCast(cfg, key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        with pytest.raises(ValueError, match="no condition"):
            m(x)

    def test_unexpected_condition_raises(self):
        m = UCast(_tiny_config(), key=jax.random.PRNGKey(0))  # n_cond=0
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        cond = jax.random.normal(jax.random.PRNGKey(2), (16, 32, 2))
        with pytest.raises(ValueError, match="num_conditional_channels=0"):
            m(x, condition=cond)


# =============================================================================
# MC-Dropout (the probabilistic-forecast mechanism)
# =============================================================================

class TestDropout:

    def test_deterministic_without_key(self):
        """``key=None`` ⇒ inference dropout ⇒ repeatable, no stochasticity."""
        m = _wake_residual_branches(UCast(_tiny_config(), key=jax.random.PRNGKey(0)))
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        np.testing.assert_array_equal(np.asarray(m(x)), np.asarray(m(x)))

    def test_distinct_keys_differ(self):
        """Active MC-Dropout: distinct keys yield distinct members."""
        m = _wake_residual_branches(UCast(_tiny_config(), key=jax.random.PRNGKey(0)))
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        y1 = m(x, key=jax.random.PRNGKey(2))
        y2 = m(x, key=jax.random.PRNGKey(3))
        assert float(jnp.max(jnp.abs(y1 - y2))) > 1e-4

    def test_same_key_reproducible(self):
        m = _wake_residual_branches(UCast(_tiny_config(), key=jax.random.PRNGKey(0)))
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        y1 = m(x, key=jax.random.PRNGKey(2))
        y2 = m(x, key=jax.random.PRNGKey(2))
        np.testing.assert_array_equal(np.asarray(y1), np.asarray(y2))

    def test_dropout_differs_from_deterministic(self):
        m = _wake_residual_branches(UCast(_tiny_config(), key=jax.random.PRNGKey(0)))
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        assert float(jnp.max(jnp.abs(m(x) - m(x, key=jax.random.PRNGKey(2))))) > 1e-4

    def test_vmap_ensemble(self):
        """``jax.vmap`` over per-member keys produces a batched ensemble."""
        m = _wake_residual_branches(UCast(_tiny_config(), key=jax.random.PRNGKey(0)))
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        keys = jax.random.split(jax.random.PRNGKey(9), 4)
        ens = jax.vmap(lambda k: m(x, key=k))(keys)
        assert ens.shape == (4, 16, 32, 6)
        # Members are not all identical.
        assert float(jnp.max(jnp.abs(ens[0] - ens[1]))) > 1e-4


# =============================================================================
# Geographic padding (circular longitude, zero latitude)
# =============================================================================

class TestGeoPadding:

    def test_longitude_is_circular(self):
        """A pure 3x3 GeoConv2d must treat longitude (last axis) periodically:
        a longitudinal roll of the input commutes with the convolution."""
        key = jax.random.PRNGKey(0)
        conv = GeoConv2d(3, 5, 3, init_mode="kaiming_uniform",
                         init_weight=1.0, init_bias=0.1, key=key)
        x = jax.random.normal(jax.random.PRNGKey(1), (3, 8, 12))  # (C,H,W=lon)
        y = conv(x)
        y_roll = conv(jnp.roll(x, 3, axis=-1))
        np.testing.assert_allclose(
            np.asarray(jnp.roll(y, 3, axis=-1)), np.asarray(y_roll), atol=1e-6,
        )

    def test_latitude_not_circular(self):
        """Latitude (axis -2) is zero-padded, so a latitudinal roll does NOT
        commute with the convolution (boundary rows differ)."""
        key = jax.random.PRNGKey(0)
        conv = GeoConv2d(3, 5, 3, init_mode="kaiming_uniform",
                         init_weight=1.0, init_bias=0.1, key=key)
        x = jax.random.normal(jax.random.PRNGKey(1), (3, 8, 12))
        y = conv(x)
        y_roll = conv(jnp.roll(x, 1, axis=-2))
        assert not np.allclose(
            np.asarray(jnp.roll(y, 1, axis=-2)), np.asarray(y_roll), atol=1e-4,
        )


# =============================================================================
# GroupNorm divisibility guard (no silent regrouping)
# =============================================================================

class TestGroupNorm:

    def test_divisible_ok(self):
        gn = GroupNorm(8)  # min(32, 8//4)=2, 8%2==0
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 4, 4))
        assert gn(x).shape == (8, 4, 4)

    def test_indivisible_raises(self):
        # C=14 -> min(32, 14//4)=3, 14%3 != 0 -> must raise, not silently regroup.
        with pytest.raises(ValueError, match="not divisible"):
            GroupNorm(14)


# =============================================================================
# Differentiability
# =============================================================================

class TestDifferentiability:

    def test_grad_wrt_params(self):
        m = UCast(_tiny_config(attn_levels=(1,), channels_per_head=4),
                  key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))

        def loss(net):
            return jnp.mean(net(x) ** 2)

        g = eqx.filter_grad(loss)(m)
        # Head conv is the only non-zero-init weight that directly scales the
        # output residual; its grad must be finite and non-trivial.
        assert jnp.all(jnp.isfinite(g.out_conv.weight))
        assert float(jnp.max(jnp.abs(g.out_conv.weight))) > 0.0
        assert jnp.all(jnp.isfinite(g.enc[0].weight))

    def test_grad_wrt_input(self):
        m = UCast(_tiny_config(), key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (16, 32, 6))
        g = jax.grad(lambda z: jnp.mean(m(z) ** 2))(x)
        assert g.shape == x.shape
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 0.0
