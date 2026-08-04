"""Differentiability tests for the ``packages/ml`` package.

Covers the gradient surface of ``legoesm.ml`` (SFNO / spectral conv /
U-Cast / losses / channel packing / normalization / conservation) and
``legoesm.training`` (dycore training losses, trainable-parameter
transforms, the spec-driven parameter collector, the checkpointed
rollout).  ``legoesm.da`` — which also lives in this package — is
deliberately NOT covered here; it is already exercised by
``tests/unit/test_diff_data_assimilation.py``.

Assertion discipline
--------------------
Every test asserts something that can FAIL for a real defect, not just
``isfinite``:

* analytic VJPs (closed-form gradients checked to tight tolerance),
* exact structural zeros / non-zeros (masked cells, frozen branches),
* round-trip Jacobian identity,
* reachability of every parameter leaf,
* invariance/equivalence (jit vs eager, checkpoint on vs off).

Run with ``JAX_ENABLE_X64=1`` — the SFNO spectral path REQUIRES x64
(``SpectralConv.__init__`` raises otherwise).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# The SFNO spectral weights are float64/complex128 by contract; enable x64
# before any grid or module is built (mirrors tests/unit/test_sfno_pe.py).
jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.field import Field
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate


# ============================================================================
# Shared fixtures / helpers
# ============================================================================

# T5: 10 x 20 grid, 21 SH coefficients — the smallest grid that still
# exercises a non-trivial triangular truncation.
GRID_T5 = create_gaussian_grid(n_max=5)
# T8: 14 x 28, 45 SH coefficients — used where a slightly richer spectrum
# matters (channel packing round trip).
GRID_T8 = create_gaussian_grid(n_max=8)
NLEV = 3
SIGMA = create_sigma_coordinate(n_levels=NLEV)


def _finite_and_nonzero(g, name: str, min_max_abs: float = 0.0):
    """Assert a gradient array is finite and NOT identically zero."""
    g = jnp.asarray(g)
    assert jnp.all(jnp.isfinite(g)), f"{name}: non-finite gradient"
    m = float(jnp.max(jnp.abs(g)))
    assert m > min_max_abs, (
        f"{name}: gradient is identically zero (max|g| = {m:g}) — the "
        f"quantity is unreachable by reverse-mode AD"
    )
    return m


def _has_spatial_structure(g, name: str, rtol: float = 1e-6):
    """Assert a gradient field is not a spatially uniform constant."""
    g = jnp.asarray(g)
    spread = float(jnp.max(g) - jnp.min(g))
    scale = float(jnp.max(jnp.abs(g))) + 1e-300
    assert spread / scale > rtol, (
        f"{name}: gradient is spatially uniform (max-min = {spread:g}, "
        f"scale = {scale:g}) — a structureless gradient usually means the "
        f"spatial operator was bypassed"
    )


def _tree_max_abs(tree):
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    return [float(jnp.max(jnp.abs(l))) for l in leaves]


# ============================================================================
# 1. SpectralConv — the complex-weight spectral operator
# ============================================================================

class TestSpectralConv:
    """``ml/spectral_conv.py``: complex weights stored as (real, imag)."""

    @staticmethod
    def _build(n_sh=6, cin=2, cout=3, seed=0):
        from legoesm.ml.spectral_conv import SpectralConv
        return SpectralConv(n_sh, cin, cout, key=jax.random.PRNGKey(seed))

    @staticmethod
    def _coeffs(n_sh=6, cin=2, seed=1):
        k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
        return (jax.random.normal(k1, (n_sh, cin), dtype=jnp.float64)
                + 1j * jax.random.normal(k2, (n_sh, cin), dtype=jnp.float64))

    def test_weight_real_and_imag_analytic_vjp(self):
        """d|out|^2/dWr = 2 Re(conj(out) c), d/dWi = 2 Im(out conj(c)).

        Both parts must be reached with the CORRECT sign and magnitude.
        A real/imag mix-up (the classic spectral-conv defect) flips the
        imaginary branch's sign and this test fails.
        """
        conv = self._build()
        c = self._coeffs()

        def loss(module):
            out = module(c)
            return jnp.sum(jnp.real(out * jnp.conj(out)))

        g = eqx.filter_grad(loss)(conv)
        out = conv(c)                                  # (n_sh, cout)
        # analytic: broadcast over (n_sh, out, in)
        expect_r = 2.0 * jnp.real(jnp.conj(out)[:, :, None] * c[:, None, :])
        expect_i = 2.0 * jnp.imag(out[:, :, None] * jnp.conj(c)[:, None, :])

        _finite_and_nonzero(g.weight_real, "dL/dweight_real")
        _finite_and_nonzero(g.weight_imag, "dL/dweight_imag")
        np.testing.assert_allclose(
            np.asarray(g.weight_real), np.asarray(expect_r), rtol=1e-11,
            err_msg="real-part VJP does not match the analytic gradient")
        np.testing.assert_allclose(
            np.asarray(g.weight_imag), np.asarray(expect_i), rtol=1e-11,
            err_msg="imag-part VJP does not match the analytic gradient")

    def test_imag_weight_is_not_silently_dropped(self):
        """Perturbing ONLY weight_imag must change the output.

        Guards the failure mode where ``W = weight_real + 1j*weight_imag``
        is built but the imaginary part never influences the result
        (silently-zero gradient on half the parameters).
        """
        conv = self._build()
        c = self._coeffs()
        base = conv(c)
        bumped = eqx.tree_at(
            lambda m: m.weight_imag, conv, conv.weight_imag + 0.1)
        assert float(jnp.max(jnp.abs(bumped(c) - base))) > 1e-3

    def test_gradient_wrt_input_real_and_imag_parts(self):
        """The SH-coefficient input is reachable through BOTH parts."""
        conv = self._build()
        c = self._coeffs()
        cr, ci = jnp.real(c), jnp.imag(c)

        def loss(re, im):
            out = conv(re + 1j * im)
            return jnp.sum(jnp.real(out * jnp.conj(out)))

        g_re, g_im = jax.grad(loss, argnums=(0, 1))(cr, ci)
        _finite_and_nonzero(g_re, "dL/dRe(coeffs)")
        _finite_and_nonzero(g_im, "dL/dIm(coeffs)")

    def test_dtype_is_float64_under_x64(self):
        """Spectral weights (and therefore their cotangents) stay float64."""
        conv = self._build()
        assert conv.weight_real.dtype == jnp.float64
        assert conv.weight_imag.dtype == jnp.float64
        c = self._coeffs()
        g = eqx.filter_grad(
            lambda m: jnp.sum(jnp.abs(m(c)) ** 2))(conv)
        assert g.weight_real.dtype == jnp.float64
        assert g.weight_imag.dtype == jnp.float64


# ============================================================================
# 2. SFNOBlock / SFNO
# ============================================================================

_SFNO_CH = 3


def _tiny_sfno(n_blocks=2, embed_dim=8, checkpoint=False, residual=True,
               out_channels=_SFNO_CH, seed=0):
    from legoesm.ml.sfno import SFNO, SFNOConfig
    cfg = SFNOConfig(
        in_channels=_SFNO_CH, out_channels=out_channels, embed_dim=embed_dim,
        n_blocks=n_blocks, mlp_expansion=2, residual_prediction=residual,
        gradient_checkpoint=checkpoint,
    )
    return SFNO(config=cfg, grid=GRID_T5, key=jax.random.PRNGKey(seed))


def _sfno_input(seed=1, n_ch=_SFNO_CH):
    return jax.random.normal(
        jax.random.PRNGKey(seed), (GRID_T5.n_lat, GRID_T5.n_lon, n_ch),
        dtype=jnp.float64)


class TestSFNOBlock:
    """``ml/sfno_block.py``: LN -> SH analysis -> spectral conv -> synthesis."""

    def test_every_block_parameter_leaf_is_reachable(self):
        from legoesm.ml.sfno_block import SFNOBlock
        block = SFNOBlock(grid=GRID_T5, embed_dim=8, mlp_expansion=2,
                          key=jax.random.PRNGKey(0))
        x = jax.random.normal(
            jax.random.PRNGKey(2), (GRID_T5.n_lat, GRID_T5.n_lon, 8))

        g = eqx.filter_grad(lambda b: jnp.sum(b(x, GRID_T5) ** 2))(block)
        named = {
            "spectral_conv.weight_real": g.spectral_conv.weight_real,
            "spectral_conv.weight_imag": g.spectral_conv.weight_imag,
            "ln.weight": g.ln.weight,
            "ln.bias": g.ln.bias,
            "mlp_linear1.weight": g.mlp_linear1.weight,
            "mlp_linear1.bias": g.mlp_linear1.bias,
            "mlp_linear2.weight": g.mlp_linear2.weight,
            "mlp_linear2.bias": g.mlp_linear2.bias,
        }
        for name, leaf in named.items():
            _finite_and_nonzero(leaf, f"SFNOBlock {name}")

    def test_input_gradient_has_spatial_structure(self):
        """The SH transform couples every point; dL/dx must not be uniform."""
        from legoesm.ml.sfno_block import SFNOBlock
        block = SFNOBlock(grid=GRID_T5, embed_dim=8, mlp_expansion=2,
                          key=jax.random.PRNGKey(0))
        x = jax.random.normal(
            jax.random.PRNGKey(2), (GRID_T5.n_lat, GRID_T5.n_lon, 8))
        g = jax.grad(lambda xx: jnp.sum(block(xx, GRID_T5) ** 2))(x)
        _finite_and_nonzero(g, "SFNOBlock dL/dx")
        _has_spatial_structure(g[..., 0], "SFNOBlock dL/dx[...,0]")

    def test_skip_connection_contributes_identity_to_the_jacobian(self):
        """``return x + residual`` => d(out_c)/d(x_c) has a +1 diagonal.

        Compare the block's per-channel input gradient against the SAME
        block with the residual removed: the difference must be exactly
        the cotangent (the identity path).
        """
        from legoesm.ml.sfno_block import SFNOBlock
        block = SFNOBlock(grid=GRID_T5, embed_dim=8, mlp_expansion=2,
                          key=jax.random.PRNGKey(0))
        x = jax.random.normal(
            jax.random.PRNGKey(2), (GRID_T5.n_lat, GRID_T5.n_lon, 8))
        w = jax.random.normal(jax.random.PRNGKey(3), x.shape)

        g_with = jax.grad(lambda xx: jnp.sum(w * block(xx, GRID_T5)))(x)
        g_without = jax.grad(
            lambda xx: jnp.sum(w * (block(xx, GRID_T5) - xx)))(x)
        np.testing.assert_allclose(
            np.asarray(g_with - g_without), np.asarray(w), rtol=1e-9,
            err_msg="skip connection does not contribute an identity Jacobian")


class TestSFNO:
    """``ml/sfno.py``: encoder -> N blocks -> decoder (+ big residual)."""

    def test_all_parameter_leaves_reachable(self):
        model = _tiny_sfno()
        x = _sfno_input()
        g = eqx.filter_grad(lambda m: jnp.sum(m(x, GRID_T5) ** 2))(model)
        maxima = _tree_max_abs(g)
        assert len(maxima) > 0
        zero = [i for i, m in enumerate(maxima) if m == 0.0]
        assert not zero, (
            f"{len(zero)}/{len(maxima)} SFNO parameter leaves have EXACTLY "
            f"zero gradient — unreachable by AD (indices {zero})")
        assert all(np.isfinite(m) for m in maxima)

    def test_input_gradient_finite_nonzero_structured(self):
        model = _tiny_sfno()
        x = _sfno_input()
        g = jax.grad(lambda xx: jnp.sum(model(xx, GRID_T5) ** 2))(x)
        _finite_and_nonzero(g, "SFNO dL/dx")
        _has_spatial_structure(g[..., 0], "SFNO dL/dx[...,0]")
        assert g.dtype == jnp.float64

    def test_residual_prediction_adds_identity_to_input_jacobian(self):
        """With residual_prediction the input enters BOTH the net and a
        pass-through; the difference of the two input gradients is exactly
        the cotangent restricted to the first ``out_channels``."""
        # ``SFNOConfig`` is an ``eqx.field(static=True)`` (part of the
        # treedef, NOT a leaf) so it cannot be swapped with ``tree_at``.
        # Rebuild instead: ``residual_prediction`` does not participate in
        # weight initialisation, so the same seed gives identical weights.
        res = _tiny_sfno(residual=True, seed=7)
        non = _tiny_sfno(residual=False, seed=7)
        for a, b in zip(_tree_max_abs(res), _tree_max_abs(non)):
            assert a == b, "the two models must share identical weights"
        x = _sfno_input()
        w = jax.random.normal(jax.random.PRNGKey(9), x.shape)
        g_res = jax.grad(lambda xx: jnp.sum(w * res(xx, GRID_T5)))(x)
        g_non = jax.grad(lambda xx: jnp.sum(w * non(xx, GRID_T5)))(x)
        # A missing residual makes this difference 0 instead of w, i.e. a
        # relative error of 1 — the 1e-5 float32 bound is still decisive.
        rel = float(jnp.max(jnp.abs((g_res - g_non) - w))) / float(
            jnp.max(jnp.abs(w)))
        assert rel < 1e-5, (
            f"the big residual skip is missing from the input Jacobian "
            f"(relative mismatch {rel:g})")

    def test_gradient_checkpoint_does_not_change_gradients(self):
        """``gradient_checkpoint`` must only change memory, never values.

        Tolerance note: ``SFNOBlock`` downcasts the post-synthesis branch to
        float32 (``sfno_block.py`` step 4), so the encoder/MLP/decoder adjoint
        is float32 arithmetic even under x64.  Rematerialisation re-emits that
        arithmetic in a different fusion, which is reproducible only to
        float32 epsilon — see
        ``test_non_spectral_branch_runs_in_float32_even_under_x64`` for the
        measured precision boundary.  The check below is still strong: it
        bounds the RELATIVE difference per leaf at 1e-5, far below the O(1)
        relative error a genuinely different adjoint would produce.
        """
        # Rebuild rather than tree_at: SFNOConfig is a static field.
        # ``gradient_checkpoint`` does not affect initialisation, so the same
        # seed yields bit-identical weights.
        plain = _tiny_sfno(checkpoint=False, seed=11)
        ckpt = _tiny_sfno(checkpoint=True, seed=11)
        assert plain.config.gradient_checkpoint is False
        assert ckpt.config.gradient_checkpoint is True
        x = _sfno_input(seed=12)

        def gof(m):
            return eqx.filter_grad(lambda mm: jnp.sum(mm(x, GRID_T5) ** 2))(m)

        g0 = jax.tree_util.tree_leaves(eqx.filter(gof(plain), eqx.is_inexact_array))
        g1 = jax.tree_util.tree_leaves(eqx.filter(gof(ckpt), eqx.is_inexact_array))
        assert len(g0) == len(g1) and len(g0) > 0
        for i, (a, b) in enumerate(zip(g0, g1)):
            scale = float(jnp.max(jnp.abs(a)))
            assert scale > 0.0, f"leaf {i} has an identically-zero gradient"
            rel = float(jnp.max(jnp.abs(a - b))) / scale
            assert rel < 1e-5, (
                f"leaf {i}: checkpointed adjoint differs by rel {rel:g} — far "
                f"beyond the float32 noise floor, so remat changed the VALUE")

    def test_jit_grad_matches_eager_grad(self):
        """See the tolerance note on the checkpoint test: the non-spectral
        branch is float32, so jit fusion reorders it at float32 epsilon."""
        model = _tiny_sfno(seed=13)
        x = _sfno_input(seed=14)
        f = lambda xx: jnp.sum(model(xx, GRID_T5) ** 2)
        g_jit = jax.jit(jax.grad(f))(x)
        g_eager = jax.grad(f)(x)
        rel = float(jnp.max(jnp.abs(g_jit - g_eager))) / float(
            jnp.max(jnp.abs(g_eager)))
        assert rel < 1e-5, f"jit vs eager gradient differ by rel {rel:g}"

    def test_float32_downcast_in_the_block_is_still_present(self):
        """TOLERANCE TRIPWIRE for the two tests above.

        ``SFNOBlock.__call__`` casts UP to float64 for the SH transform and
        then straight back DOWN with a hard-coded ``x.astype(jnp.float32)``
        (step 4 -> 5), regardless of the ambient precision policy.  A
        float64-level fusion difference upstream of that cast can flip the
        float32 rounding, which is why the jit-vs-eager and
        checkpoint-vs-plain gradient checks above are bounded at 1e-5
        relative (MEASURED discrepancy: ~1.1e-7) instead of ~1e-14.

        If this downcast is ever removed, the tolerances above are too loose
        and should be tightened to float64 — so pin the source fact here
        rather than letting a silent precision change go unnoticed.
        """
        import inspect
        from legoesm.ml import sfno_block
        src = inspect.getsource(sfno_block.SFNOBlock.__call__)
        assert "astype(jnp.float32)" in src, (
            "the float32 downcast in SFNOBlock.__call__ is gone — the "
            "jit-vs-eager / checkpoint gradient tolerances above (1e-5) were "
            "chosen for it and should now be tightened toward float64")
        # The spectral weights themselves must stay float64 (x64 contract).
        block = sfno_block.SFNOBlock(
            grid=GRID_T5, embed_dim=8, mlp_expansion=2,
            key=jax.random.PRNGKey(0))
        assert block.spectral_conv.weight_real.dtype == jnp.float64
        assert block.spectral_conv.weight_imag.dtype == jnp.float64

    def test_vmap_over_batch_of_inputs(self):
        model = _tiny_sfno(seed=15)
        xs = jax.random.normal(
            jax.random.PRNGKey(16),
            (3, GRID_T5.n_lat, GRID_T5.n_lon, _SFNO_CH), dtype=jnp.float64)
        gs = jax.vmap(jax.grad(lambda xx: jnp.sum(model(xx, GRID_T5) ** 2)))(xs)
        assert gs.shape == xs.shape
        _finite_and_nonzero(gs, "vmapped SFNO dL/dx")
        # Different members must yield different gradients (not broadcast).
        assert float(jnp.max(jnp.abs(gs[0] - gs[1]))) > 0.0

    @pytest.mark.parametrize("n_steps", [1, 2, 4])
    def test_autoregressive_rollout_gradient(self, n_steps):
        """``lax.scan`` autoregressive rollout: gradient stays finite and
        the sensitivity is non-zero at every horizon."""
        model = _tiny_sfno(n_blocks=1, embed_dim=6, seed=17)

        def rollout(x0):
            def body(c, _):
                return model(c, GRID_T5), None
            xf, _ = jax.lax.scan(body, x0, None, length=n_steps)
            return jnp.sum(xf ** 2)

        x = _sfno_input(seed=18) * 0.1
        g = jax.grad(rollout)(x)
        _finite_and_nonzero(g, f"SFNO {n_steps}-step rollout dL/dx0")


# ============================================================================
# 3. U-Cast (convolutional U-Net emulator)
# ============================================================================

def _tiny_ucast(dropout=0.0, seed=3):
    from legoesm.ml.ucast import UCast, UCastConfig
    cfg = UCastConfig(
        in_channels=3, out_channels=3, model_channels=8, channel_mult=(1, 2),
        num_blocks=1, attn_levels=(1,), channels_per_head=4, dropout=dropout,
    )
    return UCast(cfg, key=jax.random.PRNGKey(seed))


_UCAST_X = jax.random.normal(jax.random.PRNGKey(4), (8, 16, 3), dtype=jnp.float64)


class TestUCast:
    """``ml/ucast.py``: ADM/DhariwalUNet port with zero-initialised heads."""

    def test_zero_init_makes_the_untrained_net_an_exact_identity(self):
        m = _tiny_ucast()
        out = m(_UCAST_X)
        np.testing.assert_allclose(np.asarray(out), np.asarray(_UCAST_X),
                                   rtol=0, atol=0)

    def test_at_init_only_the_output_conv_is_reachable(self):
        """EXPECTED-BY-DESIGN, not a bug: ``out_conv`` is zero-initialised,
        so at step 0 it blocks the adjoint to every upstream leaf.  Pinning
        the exact count catches an accidental change of the init scheme
        (which would silently change training dynamics)."""
        m = _tiny_ucast()
        g = eqx.filter_grad(lambda mm: jnp.sum(mm(_UCAST_X) ** 2))(m)
        maxima = _tree_max_abs(g)
        nonzero_idx = [i for i, v in enumerate(maxima) if v > 0.0]
        assert all(np.isfinite(v) for v in maxima)
        # Exactly the head's weight + bias carry gradient at init.
        head = _tree_max_abs(eqx.filter(g.out_conv, eqx.is_inexact_array))
        assert all(v > 0.0 for v in head), (
            "out_conv (the only non-blocked branch at init) has zero gradient")
        assert len(nonzero_idx) == len(head), (
            f"expected exactly {len(head)} reachable leaves at zero-init "
            f"(the out_conv weight/bias), found {len(nonzero_idx)}")

    def test_every_leaf_reachable_once_the_head_is_non_zero(self):
        """After the head leaves the zero-init plateau, EVERY parameter
        must be reachable — a leaf that is still exactly zero here is a
        genuine dead degree of freedom."""
        m = _tiny_ucast()
        key = jax.random.PRNGKey(21)
        # Perturb every zero-initialised leaf away from the plateau.
        params, static = eqx.partition(m, eqx.is_inexact_array)
        leaves, treedef = jax.tree_util.tree_flatten(params)
        keys = jax.random.split(key, len(leaves))
        leaves = [l + 0.05 * jax.random.normal(k, l.shape, l.dtype)
                  for l, k in zip(leaves, keys)]
        m2 = eqx.combine(jax.tree_util.tree_unflatten(treedef, leaves), static)

        g = eqx.filter_grad(lambda mm: jnp.sum(mm(_UCAST_X) ** 2))(m2)
        maxima = _tree_max_abs(g)
        dead = [i for i, v in enumerate(maxima) if v == 0.0]
        assert not dead, (
            f"{len(dead)}/{len(maxima)} U-Cast leaves are STILL exactly zero "
            f"after perturbing the zero-init layers (indices {dead}) — dead "
            f"degrees of freedom, not a zero-init artefact")
        assert all(np.isfinite(v) for v in maxima)

    def test_input_gradient_is_finite_nonzero_and_structured(self):
        m = _tiny_ucast()
        # Perturb the head so the convolutional branch is live.
        m = eqx.tree_at(lambda mm: mm.out_conv.weight, m,
                        m.out_conv.weight + 0.1)
        g = jax.grad(lambda x: jnp.sum(m(x) ** 2))(_UCAST_X)
        _finite_and_nonzero(g, "UCast dL/dx")
        _has_spatial_structure(g[..., 0], "UCast dL/dx[...,0]")

    def test_dropout_path_is_differentiable(self):
        """MC-dropout (``key`` supplied) must still yield finite, non-zero
        parameter gradients — the Bernoulli mask is constant w.r.t. params."""
        m = _tiny_ucast(dropout=0.25, seed=5)
        m = eqx.tree_at(lambda mm: mm.out_conv.weight, m,
                        m.out_conv.weight + 0.1)
        k = jax.random.PRNGKey(6)
        g = eqx.filter_grad(
            lambda mm: jnp.sum(mm(_UCAST_X, key=k) ** 2))(m)
        _finite_and_nonzero(g.out_conv.weight, "UCast dropout dL/dout_conv.w")


# ============================================================================
# 4. ml/loss.py
# ============================================================================

_W = GRID_T5.weights                      # Gauss-Legendre weights, sum = 2
_NLAT, _NLON = GRID_T5.n_lat, GRID_T5.n_lon


def _pred_target(n_ch=2, seed=30):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    pred = jax.random.normal(k1, (_NLAT, _NLON, n_ch), dtype=jnp.float64)
    tgt = jax.random.normal(k2, (_NLAT, _NLON, n_ch), dtype=jnp.float64)
    return pred, tgt


class TestAreaWeightedMSE:

    def test_analytic_gradient_unmasked(self):
        """dL/dpred = 2 (pred-target) w[lat] * n_lat / (sum(w) * N)."""
        from legoesm.ml.loss import area_weighted_mse
        pred, tgt = _pred_target()
        g = jax.grad(lambda p: area_weighted_mse(p, tgt, _W))(pred)
        n = pred.size
        expect = (2.0 * (pred - tgt) * _W[:, None, None]
                  * _NLAT / jnp.sum(_W) / n)
        _finite_and_nonzero(g, "area_weighted_mse dL/dpred")
        np.testing.assert_allclose(np.asarray(g), np.asarray(expect),
                                   rtol=1e-12)

    def test_latitude_weighting_is_actually_applied(self):
        """The per-latitude gradient ratio must equal the weight ratio.

        A dropped/broadcast-wrong weight would give a uniform ratio of 1.
        """
        from legoesm.ml.loss import area_weighted_mse
        pred = jnp.ones((_NLAT, _NLON, 1))
        tgt = jnp.zeros_like(pred)
        g = jax.grad(lambda p: area_weighted_mse(p, tgt, _W))(pred)
        row = g[:, 0, 0]
        np.testing.assert_allclose(np.asarray(row / row[0]),
                                   np.asarray(_W / _W[0]), rtol=1e-12)
        # Non-uniform weights => non-uniform gradient rows.
        assert float(jnp.max(row) - jnp.min(row)) > 0.0

    def test_masked_gradient_is_exactly_zero_on_masked_cells(self):
        from legoesm.ml.loss import area_weighted_mse
        pred, tgt = _pred_target(seed=31)
        mask = jnp.zeros((_NLAT, _NLON)).at[: _NLAT // 2, :].set(1.0)
        g = jax.grad(lambda p: area_weighted_mse(p, tgt, _W, mask))(pred)
        assert float(jnp.max(jnp.abs(g[_NLAT // 2:]))) == 0.0, (
            "masked-out cells receive gradient — the mask is not applied")
        _finite_and_nonzero(g[: _NLAT // 2], "masked MSE kept-cell gradient")

    def test_masked_region_equals_unmasked_loss_of_that_region(self):
        """The masked normalisation must be Sum(sq w mask)/Sum(w mask)."""
        from legoesm.ml.loss import area_weighted_mse
        pred, tgt = _pred_target(seed=32)
        half = _NLAT // 2
        mask = jnp.zeros((_NLAT, _NLON)).at[:half, :].set(1.0)
        masked = area_weighted_mse(pred, tgt, _W, mask)
        sub = area_weighted_mse(pred[:half], tgt[:half], _W[:half])
        np.testing.assert_allclose(float(masked), float(sub), rtol=1e-12)

    def test_fully_masked_input_has_finite_zero_gradient(self):
        """The tiny-denominator guard must not produce inf/NaN cotangents."""
        from legoesm.ml.loss import area_weighted_mse
        pred, tgt = _pred_target(seed=33)
        mask = jnp.zeros((_NLAT, _NLON))
        g = jax.grad(lambda p: area_weighted_mse(p, tgt, _W, mask))(pred)
        assert jnp.all(jnp.isfinite(g)), "fully-masked MSE gives non-finite grad"
        assert float(jnp.max(jnp.abs(g))) == 0.0


class TestPerVariableMSE:

    def test_channel_weights_are_reachable_by_ad(self):
        """Regression gate for the historical ``_channel_weights`` bug
        (the argument was underscore-prefixed and silently ignored)."""
        from legoesm.ml.loss import per_variable_mse
        pred, tgt = _pred_target(n_ch=3, seed=34)
        cw = jnp.array([1.0, 2.0, 0.5])
        g = jax.grad(
            lambda c: jnp.sum(per_variable_mse(pred, tgt, _W, c)))(cw)
        _finite_and_nonzero(g, "per_variable_mse dL/dchannel_weights")
        assert float(jnp.min(jnp.abs(g))) > 0.0, (
            "at least one channel weight has zero gradient — it is ignored")

    def test_per_channel_gradients_are_independent(self):
        from legoesm.ml.loss import per_variable_mse
        pred, tgt = _pred_target(n_ch=3, seed=35)
        J = jax.jacrev(lambda p: per_variable_mse(p, tgt, _W))(pred)
        # J has shape (n_ch, n_lat, n_lon, n_ch); off-diagonal channel blocks
        # must be exactly zero (channels do not mix).
        for i in range(3):
            for j in range(3):
                blk = J[i, ..., j]
                if i == j:
                    _finite_and_nonzero(blk, f"per_variable_mse d[{i}]/dpred[{j}]")
                else:
                    assert float(jnp.max(jnp.abs(blk))) == 0.0


class TestWeightedMAE:

    def test_analytic_subgradient(self):
        from legoesm.ml.loss import weighted_mae
        pred, tgt = _pred_target(seed=36)
        g = jax.grad(lambda p: weighted_mae(p, tgt, _W))(pred)
        expect = (jnp.sign(pred - tgt) * _W[:, None, None]
                  * _NLAT / jnp.sum(_W) / pred.size)
        np.testing.assert_allclose(np.asarray(g), np.asarray(expect), rtol=1e-12)

    def test_subgradient_at_exact_match_is_plus_one_not_zero(self):
        """DOCUMENTED JAX SEMANTICS (evidence, not a bug): ``jnp.abs`` has
        derivative +1 at 0 in JAX, so the MAE gradient at a perfect fit is
        ``+w``, not 0.  Anyone using MAE as the sole objective must know
        the optimum is not a stationary point of the AD gradient."""
        from legoesm.ml.loss import weighted_mae
        pred = jnp.zeros((_NLAT, _NLON, 1))
        g = jax.grad(lambda p: weighted_mae(p, pred, _W))(pred)
        assert jnp.all(jnp.isfinite(g))
        expect = _W[:, None, None] * _NLAT / jnp.sum(_W) / pred.size
        np.testing.assert_allclose(
            np.asarray(g), np.asarray(jnp.broadcast_to(expect, g.shape)),
            rtol=1e-12)


class TestSpectralLoss:

    def test_zero_error_gives_finite_zero_gradient(self):
        """``jnp.abs`` of a COMPLEX argument is 0/0 at the origin in the
        naive chain rule; JAX defines it to 0 here.  Pin that: a NaN would
        poison ``carry_spectral_loss`` whenever a coefficient matches
        exactly (very common outside the truncation)."""
        from legoesm.ml.loss import spectral_loss
        z = jnp.zeros(_NLAT)

        def f(re):
            return spectral_loss(re + 0j, jnp.zeros_like(re) + 0j)

        g = jax.grad(f)(z)
        assert jnp.all(jnp.isfinite(g)), (
            "spectral_loss gradient is non-finite at pred == target")
        assert float(jnp.max(jnp.abs(g))) == 0.0

    def test_analytic_gradient_and_mixed_zero_pattern(self):
        from legoesm.ml.loss import spectral_loss
        x = jnp.array([0.0, 1.0, 0.0, 2.0, 0.0])

        def f(re):
            return spectral_loss(re + 0j, jnp.zeros_like(re) + 0j)

        g = jax.grad(f)(x)
        np.testing.assert_allclose(np.asarray(g), np.asarray(2.0 * x / x.size),
                                   rtol=1e-12)

    def test_imaginary_part_is_reachable(self):
        from legoesm.ml.loss import spectral_loss
        re = jnp.array([1.0, -2.0, 0.5])
        im = jnp.array([0.3, 0.7, -1.0])
        tgt = jnp.zeros(3) + 0j
        g_im = jax.grad(lambda i: spectral_loss(re + 1j * i, tgt))(im)
        np.testing.assert_allclose(np.asarray(g_im),
                                   np.asarray(2.0 * im / im.size), rtol=1e-12)


class TestCRPS:

    def test_almost_fair_crps_analytic_gradient_two_members(self):
        """L = mean_m|e_m - t| - c*sum_ij|e_i - e_j|, c=(M-1+a)/(2 M^2 (M-1)).

        For M=2 this gives dL/de0 = sign(e0-t)/2 - 2c sign(e0-e1).  A wrong
        finite-ensemble coefficient or a dropped spread term fails here.
        """
        from legoesm.ml.loss import almost_fair_crps
        alpha = 0.95
        ens = jnp.array([[1.0], [2.0]])
        tgt = jnp.array([1.5])
        g = jax.grad(
            lambda e: jnp.sum(almost_fair_crps(e, tgt, alpha=alpha)))(ens)
        c = (2 - 1 + alpha) / (2.0 * 4.0 * 1.0)
        expect = jnp.array([[0.5 * np.sign(-0.5) - 2 * c * np.sign(-1.0)],
                            [0.5 * np.sign(0.5) - 2 * c * np.sign(1.0)]])
        np.testing.assert_allclose(np.asarray(g), np.asarray(expect), rtol=1e-12)
        _finite_and_nonzero(g, "almost_fair_crps dL/densemble")

    def test_identical_members_give_finite_gradient(self):
        """The i == j diagonal of the pairwise term is |0|; JAX's abs
        subgradient makes it finite (no 0/0)."""
        from legoesm.ml.loss import almost_fair_crps
        ens = jnp.stack([jnp.full((3, 1), 1.0), jnp.full((3, 1), 1.0)])
        tgt = jnp.full((3, 1), 1.5)
        g = jax.grad(lambda e: jnp.sum(almost_fair_crps(e, tgt)))(ens)
        assert jnp.all(jnp.isfinite(g))
        np.testing.assert_allclose(np.asarray(g), -0.5 * np.ones_like(g),
                                   rtol=1e-12)

    def test_single_member_falls_back_to_absolute_error(self):
        from legoesm.ml.loss import almost_fair_crps
        ens = jnp.array([[2.0, -1.0]])
        tgt = jnp.array([0.0, 0.0])
        g = jax.grad(lambda e: jnp.sum(almost_fair_crps(e, tgt)))(ens)
        np.testing.assert_allclose(np.asarray(g), np.array([[1.0, -1.0]]))

    # BUG: ``area_weighted_afcrps`` omits the ``n_lat / sum(weights)``
    # resolution-independence correction that ``area_weighted_mse``,
    # (HISTORICAL, fixed in #1413 — kept because it records the measurement.)
    # ``per_variable_mse`` and ``weighted_mae`` all apply (see the explicit
    # "Same resolution-independence correction as area_weighted_mse" comments
    # in ml/loss.py).  It returns ``jnp.mean(crps * w)``, which for a CONSTANT
    # crps field c returns c * sum(w)/n_lat = c * 2/n_lat instead of c.
    # MEASURED at T5 (n_lat = 10, sum(w) = 2.0): returns 0.2 * c, i.e. the
    # value and every gradient through it are too SMALL by a factor of
    # n_lat/2 = 5 (the factor GROWS with resolution: 7 at T8, ~32 at T42).
    # Live impact: this function is the training objective in
    # ml/s2s/sfno_slab/training.py::_s2s_train_step (inside
    # eqx.filter_value_and_grad) and is reported next to area_weighted_mse /
    # weighted_mae in validate_s2s_step, so the three metrics are on
    # different, resolution-dependent scales and the effective S2S learning
    # rate silently changes with grid size.
    # FIXED in #1413 — the strict xfail did its job and flipped; kept as a
    # live regression test.
    def test_afcrps_is_a_resolution_independent_area_mean(self):
        from legoesm.ml.loss import area_weighted_afcrps
        # Two members straddling the target by +/-1 => pointwise CRPS is a
        # spatially CONSTANT field, so the area-weighted mean must be that
        # same constant on any grid.
        tgt = jnp.zeros((_NLAT, _NLON, 1))
        ens = jnp.stack([jnp.full_like(tgt, 1.0), jnp.full_like(tgt, -1.0)])
        got = float(area_weighted_afcrps(ens, tgt, _W))
        alpha = 0.95
        c = (2 - 1 + alpha) / (2.0 * 4.0 * 1.0)
        expect_const = 1.0 - c * 4.0     # obs term - coeff * pairwise sum
        assert abs(got - expect_const) < 1e-10, (
            f"area_weighted_afcrps returned {got:g} for a constant CRPS field "
            f"of {expect_const:g} — off by n_lat/sum(w) = "
            f"{_NLAT / float(jnp.sum(_W)):g}, i.e. the resolution-independence "
            f"correction applied by every other loss in the module is missing")

    def test_afcrps_gradient_is_finite_and_nonzero(self):
        """Separate from the scale bug above: AD reachability itself is OK."""
        from legoesm.ml.loss import area_weighted_afcrps
        tgt = jnp.zeros((_NLAT, _NLON, 1))
        k = jax.random.PRNGKey(37)
        ens = jax.random.normal(k, (3,) + tgt.shape, dtype=jnp.float64)
        g = jax.grad(lambda e: area_weighted_afcrps(e, tgt, _W))(ens)
        _finite_and_nonzero(g, "area_weighted_afcrps dL/densemble")


class TestLatitudeWeightedMetrics:

    def test_rmse_gradient_is_correct_away_from_zero_error(self):
        from legoesm.ml.loss import latitude_weighted_rmse
        pred, tgt = _pred_target(n_ch=1, seed=38)
        pred, tgt = pred[..., 0], tgt[..., 0]
        g = jax.grad(lambda p: latitude_weighted_rmse(p, tgt, _W))(pred)
        _finite_and_nonzero(g, "latitude_weighted_rmse dL/dpred")
        # Analytic: d sqrt(S)/dpred = (1/(2 sqrt(S))) * dS/dpred with
        # S = sum_j(mean_lon (p-t)^2 * w_j)/sum(w).
        rmse = latitude_weighted_rmse(pred, tgt, _W)
        expect = ((pred - tgt) * _W[:, None] / _NLON / jnp.sum(_W)) / rmse
        np.testing.assert_allclose(np.asarray(g), np.asarray(expect), rtol=1e-11)

    def test_rmse_gradient_is_nan_at_zero_error(self):
        """NON-DIFFERENTIABLE AT THE OPTIMUM (documented, NOT a live bug):
        ``sqrt`` has an infinite derivative at 0, so ``d rmse/d pred`` is
        NaN when pred == target exactly.  Evidence that this is currently
        harmless: every call site (``scripts/run/run_aimip.py``,
        ``scripts/run/_aimip_ablation_single.py``) wraps the result in
        ``float(...)`` for scorecard reporting — it is never differentiated.
        This test pins that; if the metric is ever promoted into a training
        objective it must be replaced by the MSE form."""
        from legoesm.ml.loss import latitude_weighted_rmse
        pred = jnp.zeros((_NLAT, _NLON))
        g = jax.grad(lambda p: latitude_weighted_rmse(p, pred, _W))(pred)
        assert jnp.all(jnp.isnan(g))

    def test_bias_gradient_is_exact_and_finite_everywhere(self):
        from legoesm.ml.loss import latitude_weighted_bias
        pred = jnp.zeros((_NLAT, _NLON))
        g = jax.grad(lambda p: latitude_weighted_bias(p, pred, _W))(pred)
        expect = _W[:, None] / _NLON / jnp.sum(_W)
        assert jnp.all(jnp.isfinite(g))
        np.testing.assert_allclose(
            np.asarray(g), np.asarray(jnp.broadcast_to(expect, g.shape)),
            rtol=1e-12)


# ============================================================================
# 5. ml/channel_packing.py — state <-> dense tensor round trip
# ============================================================================

def _pe_state(grid=GRID_T8, with_qv=True):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        isothermal_rest_state_spectral,
    )
    tracers = None
    if with_qv:
        qv = jnp.full((grid.n_lat, grid.n_lon, NLEV), 5.0e-3, dtype=jnp.float64)
        tracers = {"q_v": Field(data=qv, name="q_v",
                                dims=("lat", "lon", "level"), units="kg/kg")}
    return isothermal_rest_state_spectral(grid, SIGMA, tracers=tracers)


def _sw_state(grid=GRID_T8, seed=40):
    from legoesm.atmosphere.dynamics.gcm.spectral_sw import SpectralSWState
    ks = jax.random.split(jax.random.PRNGKey(seed), 8)

    def _c(i):
        return (jax.random.normal(ks[2 * i], (grid.n_sh,), dtype=jnp.float64)
                + 1j * jax.random.normal(ks[2 * i + 1], (grid.n_sh,),
                                         dtype=jnp.float64))
    mk = lambda d, n: Field(data=d, name=n, dims=("sh",), units="")
    return SpectralSWState(vor_hat=mk(_c(0), "vor_hat"),
                           div_hat=mk(_c(1), "div_hat"),
                           phi_hat=mk(_c(2), "phi_hat"),
                           phis_hat=mk(_c(3), "phis_hat"))


class TestChannelPackingPE:

    def test_pack_gradient_reaches_T_hat_and_is_structured(self):
        from legoesm.ml.channel_packing import pack_pe_state, PE3DChannelSpec
        st = _pe_state()
        spec = PE3DChannelSpec(nlev=NLEV)
        Th = st.T_hat.data

        def loss(re, im):
            s = st._replace(T_hat=st.T_hat.replace(data=re + 1j * im))
            return jnp.sum(pack_pe_state(s, GRID_T8)[..., spec.T_slice] ** 2)

        g_re, g_im = jax.grad(loss, argnums=(0, 1))(jnp.real(Th), jnp.imag(Th))
        _finite_and_nonzero(g_re, "pack_pe_state dL/dRe(T_hat)")
        _finite_and_nonzero(g_im, "pack_pe_state dL/dIm(T_hat)")

    def test_pack_gradient_reaches_the_qv_tracer(self):
        """Regression gate: the q channel used to be hard zeros."""
        from legoesm.ml.channel_packing import pack_pe_state, PE3DChannelSpec
        st = _pe_state(with_qv=True)
        spec = PE3DChannelSpec(nlev=NLEV)
        qv = st.tracers["q_v"].data

        def loss(q):
            s = st._replace(tracers={"q_v": st.tracers["q_v"].replace(data=q)})
            return jnp.sum(pack_pe_state(s, GRID_T8)[..., spec.q_slice] ** 2)

        g = jax.grad(loss)(qv)
        _finite_and_nonzero(g, "pack_pe_state dL/dq_v")
        np.testing.assert_allclose(np.asarray(g), np.asarray(2.0 * qv),
                                   rtol=1e-12)

    def test_dry_state_q_channel_is_structurally_zero(self):
        """No q_v tracer => the q channel is ``zeros_like(T)``; the packed
        channel must therefore be exactly zero (not silently reusing T)."""
        from legoesm.ml.channel_packing import pack_pe_state, PE3DChannelSpec
        st = _pe_state(with_qv=False)
        spec = PE3DChannelSpec(nlev=NLEV)
        packed = pack_pe_state(st, GRID_T8)
        assert float(jnp.max(jnp.abs(packed[..., spec.q_slice]))) == 0.0

    def test_unpack_pack_round_trip_jacobian_is_identity_for_T(self):
        """``sh_analysis(sh_synthesis(x)) == x`` within the truncation, so
        the round-trip gradient of a linear functional must return the
        cotangent itself.  Catches a transposed/renormalised transform."""
        from legoesm.ml.channel_packing import pack_pe_state, unpack_pe_output
        st = _pe_state()
        Th = st.T_hat.data
        w = jax.random.normal(jax.random.PRNGKey(41), Th.shape,
                              dtype=jnp.float64)

        def f(re):
            s = st._replace(T_hat=st.T_hat.replace(data=re + 1j * jnp.imag(Th)))
            back = unpack_pe_output(pack_pe_state(s, GRID_T8), s, GRID_T8,
                                    mode="state_update")
            return jnp.sum(w * jnp.real(back.T_hat.data))

        g = jax.grad(f)(jnp.real(Th))
        np.testing.assert_allclose(np.asarray(g), np.asarray(w),
                                   rtol=1e-8, atol=1e-10)

    def test_round_trip_preserves_values(self):
        from legoesm.ml.channel_packing import pack_pe_state, unpack_pe_output
        st = _pe_state()
        back = unpack_pe_output(pack_pe_state(st, GRID_T8), st, GRID_T8,
                                mode="state_update")
        np.testing.assert_allclose(np.asarray(back.T_hat.data),
                                   np.asarray(st.T_hat.data), atol=1e-9)
        np.testing.assert_allclose(
            np.asarray(back.tracers["q_v"].data),
            np.asarray(st.tracers["q_v"].data), atol=1e-12)

    def test_phis_gradient_is_pass_through_in_state_update_zero_in_tendencies(self):
        """Structural branch check: ``mode='tendencies'`` replaces phis_hat
        with ``zeros_like`` (gradient EXACTLY zero); ``mode='state_update'``
        passes it through (gradient == cotangent)."""
        from legoesm.ml.channel_packing import pack_pe_state, unpack_pe_output
        st = _pe_state()
        packed = pack_pe_state(st, GRID_T8)
        ph = st.phis_hat.data
        w = jax.random.normal(jax.random.PRNGKey(42), ph.shape,
                              dtype=jnp.float64)

        def f(re, mode):
            s = st._replace(
                phis_hat=st.phis_hat.replace(data=re + 1j * jnp.imag(ph)))
            out = unpack_pe_output(packed, s, GRID_T8, mode=mode)
            return jnp.sum(w * jnp.real(out.phis_hat.data))

        g_state = jax.grad(lambda r: f(r, "state_update"))(jnp.real(ph))
        g_tend = jax.grad(lambda r: f(r, "tendencies"))(jnp.real(ph))
        np.testing.assert_allclose(np.asarray(g_state), np.asarray(w), rtol=1e-12)
        assert float(jnp.max(jnp.abs(g_tend))) == 0.0, (
            "tendency mode must emit a zero phis tendency (frozen orography)")

    def test_unpack_gradient_reaches_wind_and_qv_channels(self):
        """u/v enter through vor/div; q enters through the tracer dict.
        Each output channel slice must be reachable from the dense tensor."""
        from legoesm.ml.channel_packing import (
            pack_pe_state, unpack_pe_output, PE3DChannelSpec,
        )
        st = _pe_state()
        spec = PE3DChannelSpec(nlev=NLEV)
        packed = pack_pe_state(st, GRID_T8)

        def loss(y):
            out = unpack_pe_output(y, st, GRID_T8, mode="state_update")
            return (jnp.sum(jnp.abs(out.vor_hat.data) ** 2)
                    + jnp.sum(jnp.abs(out.div_hat.data) ** 2)
                    + jnp.sum(jnp.abs(out.T_hat.data) ** 2)
                    + jnp.sum(jnp.abs(out.lnps_hat.data) ** 2)
                    + jnp.sum(out.tracers["q_v"].data ** 2))

        g = jax.grad(loss)(packed + 0.01)
        for name, sl in (("u", spec.u_slice), ("v", spec.v_slice),
                         ("T", spec.T_slice), ("q", spec.q_slice)):
            _finite_and_nonzero(g[..., sl], f"unpack_pe_output dL/d{name}-channels")
        _finite_and_nonzero(g[..., spec.lnps_idx], "unpack_pe_output dL/dlnps")

    def test_unpack_rejects_unknown_mode(self):
        from legoesm.ml.channel_packing import pack_pe_state, unpack_pe_output
        st = _pe_state()
        with pytest.raises(ValueError, match="Unknown unpack mode"):
            unpack_pe_output(pack_pe_state(st, GRID_T8), st, GRID_T8,
                             mode="not_a_mode")


class TestChannelPackingSW:

    def test_pack_gradient_reaches_every_prognostic(self):
        from legoesm.ml.channel_packing import pack_sw_state
        st = _sw_state()

        def loss(vr, dr, pr, sr):
            s = st._replace(
                vor_hat=st.vor_hat.replace(data=vr + 1j * jnp.imag(st.vor_hat.data)),
                div_hat=st.div_hat.replace(data=dr + 1j * jnp.imag(st.div_hat.data)),
                phi_hat=st.phi_hat.replace(data=pr + 1j * jnp.imag(st.phi_hat.data)),
                phis_hat=st.phis_hat.replace(
                    data=sr + 1j * jnp.imag(st.phis_hat.data)),
            )
            return jnp.sum(pack_sw_state(s, GRID_T8) ** 2)

        args = tuple(jnp.real(f.data) for f in
                     (st.vor_hat, st.div_hat, st.phi_hat, st.phis_hat))
        gs = jax.grad(loss, argnums=(0, 1, 2, 3))(*args)
        for name, g in zip(("vor", "div", "phi", "phis"), gs):
            _finite_and_nonzero(g, f"pack_sw_state dL/d{name}_hat")

    def test_unpack_phis_branch_structure(self):
        from legoesm.ml.channel_packing import pack_sw_state, unpack_sw_output
        st = _sw_state()
        packed = pack_sw_state(st, GRID_T8)
        w = jax.random.normal(jax.random.PRNGKey(43), st.phis_hat.data.shape,
                              dtype=jnp.float64)

        def f(re, mode):
            s = st._replace(phis_hat=st.phis_hat.replace(
                data=re + 1j * jnp.imag(st.phis_hat.data)))
            return jnp.sum(w * jnp.real(
                unpack_sw_output(packed, s, GRID_T8, mode=mode).phis_hat.data))

        g_state = jax.grad(lambda r: f(r, "state_update"))(
            jnp.real(st.phis_hat.data))
        g_tend = jax.grad(lambda r: f(r, "tendencies"))(
            jnp.real(st.phis_hat.data))
        np.testing.assert_allclose(np.asarray(g_state), np.asarray(w), rtol=1e-12)
        assert float(jnp.max(jnp.abs(g_tend))) == 0.0

    def test_sw_round_trip_gradient_is_identity(self):
        from legoesm.ml.channel_packing import pack_sw_state, unpack_sw_output
        st = _sw_state()
        vr = jnp.real(st.vor_hat.data)
        w = jax.random.normal(jax.random.PRNGKey(44), vr.shape, dtype=jnp.float64)

        def f(re):
            s = st._replace(vor_hat=st.vor_hat.replace(
                data=re + 1j * jnp.imag(st.vor_hat.data)))
            out = unpack_sw_output(pack_sw_state(s, GRID_T8), s, GRID_T8)
            return jnp.sum(w * jnp.real(out.vor_hat.data))

        np.testing.assert_allclose(np.asarray(jax.grad(f)(vr)), np.asarray(w),
                                   rtol=1e-8, atol=1e-10)


# ============================================================================
# 6. ml/normalization.py
# ============================================================================

class TestNormalization:

    def test_normalize_denormalize_round_trip_gradient_is_one(self):
        from legoesm.ml.normalization import (
            NormalizationStats, normalize, denormalize,
        )
        stats = NormalizationStats(mean=jnp.array([2.0, -1.0]),
                                   std=jnp.array([3.0, 0.5]))
        x = jax.random.normal(jax.random.PRNGKey(50), (4, 2), dtype=jnp.float64)
        w = jax.random.normal(jax.random.PRNGKey(51), (4, 2), dtype=jnp.float64)
        g = jax.grad(
            lambda xx: jnp.sum(w * denormalize(normalize(xx, stats), stats)))(x)
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-12)

    def test_normalize_gradient_scales_as_one_over_std(self):
        from legoesm.ml.normalization import NormalizationStats, normalize
        stats = NormalizationStats(mean=jnp.array([0.0, 0.0]),
                                   std=jnp.array([4.0, 0.25]))
        x = jnp.zeros((3, 2))
        g = jax.grad(lambda xx: jnp.sum(normalize(xx, stats)))(x)
        np.testing.assert_allclose(np.asarray(g[0]), np.array([0.25, 4.0]),
                                   rtol=1e-12)

    def test_constant_channel_gives_finite_gradient(self):
        """Regression gate for the ``0 * inf`` NaN: with var == 0,
        ``d sqrt(var)/dvar`` is infinite, so flooring the STD after the sqrt
        would give a NaN backward pass.  The module floors the VARIANCE
        instead — this pins that the gradient stays finite AND that the
        floored branch contributes exactly zero std-sensitivity."""
        from legoesm.ml.normalization import compute_normalization_stats
        data = jnp.ones((4, 3, 2), dtype=jnp.float64)

        g_std = jax.grad(
            lambda d: jnp.sum(compute_normalization_stats(d).std ** 2))(data)
        assert jnp.all(jnp.isfinite(g_std)), (
            "constant-channel std gradient is non-finite (variance floor "
            "regressed to a post-sqrt std floor)")
        assert float(jnp.max(jnp.abs(g_std))) == 0.0

        g_mean = jax.grad(
            lambda d: jnp.sum(compute_normalization_stats(d).mean ** 2))(data)
        _finite_and_nonzero(g_mean, "compute_normalization_stats dmean/ddata")
        # mean over the leading 2 axes: d(sum mean^2)/d x = 2*mean/(4*3)
        np.testing.assert_allclose(np.asarray(g_mean),
                                   np.full_like(np.asarray(g_mean), 2.0 / 12.0),
                                   rtol=1e-12)

    def test_variable_channel_std_gradient_is_nonzero(self):
        from legoesm.ml.normalization import compute_normalization_stats
        data = jax.random.normal(jax.random.PRNGKey(52), (6, 3, 2),
                                 dtype=jnp.float64)
        g = jax.grad(
            lambda d: jnp.sum(compute_normalization_stats(d).std ** 2))(data)
        _finite_and_nonzero(g, "std gradient on a varying channel")

    def test_weighted_stats_gradient_finite_and_lat_dependent(self):
        """The weight must land on the LATITUDE axis, so the gradient must
        vary across latitude rows by the weight ratio."""
        from legoesm.ml.normalization import compute_normalization_stats
        data = jnp.ones((2, _NLAT, _NLON, 2), dtype=jnp.float64)
        g = jax.grad(
            lambda d: jnp.sum(compute_normalization_stats(
                d, weights=_W).mean ** 2))(data)
        assert jnp.all(jnp.isfinite(g))
        row = g[0, :, 0, 0]
        np.testing.assert_allclose(np.asarray(row / row[0]),
                                   np.asarray(_W / _W[0]), rtol=1e-10)


# ============================================================================
# 7. ml/conservation.py — post-hoc correctors
# ============================================================================

class TestConservationCorrectors:

    def test_dry_air_mass_correction_kills_the_global_mean_sensitivity(self):
        """The corrector enforces a conserved area-weighted mean, so the
        gradient of that mean w.r.t. the *uncorrected* field must be ZERO
        (to round-off).  A dropped/double-counted longitude factor (the
        historical n_lon bug) breaks this exactly."""
        from legoesm.ml.conservation import correct_dry_air_mass
        p_old = jnp.full((_NLAT, _NLON), constants.p_ref)
        p_new = p_old + jax.random.normal(
            jax.random.PRNGKey(60), p_old.shape, dtype=jnp.float64) * 100.0
        w2 = _W[:, None] * jnp.ones((1, _NLON))

        def global_mean(pn):
            return jnp.sum(correct_dry_air_mass(pn, p_old, GRID_T5) * w2)

        g = jax.grad(global_mean)(p_new)
        assert jnp.all(jnp.isfinite(g))
        scale = float(jnp.max(jnp.abs(w2)))
        assert float(jnp.max(jnp.abs(g))) < 1e-12 * scale, (
            f"the mass corrector leaves a global-mean sensitivity of "
            f"{float(jnp.max(jnp.abs(g))):g} (should be ~0)")

    def test_dry_air_mass_local_gradient_is_identity_minus_weight(self):
        """d out_ij / d in_kl = delta - w_k / (sum(w) * n_lon)."""
        from legoesm.ml.conservation import correct_dry_air_mass
        p_old = jnp.full((_NLAT, _NLON), constants.p_ref)
        p_new = p_old + 1.0
        probe = jnp.zeros((_NLAT, _NLON)).at[0, 0].set(1.0)
        g = jax.grad(
            lambda pn: jnp.sum(probe * correct_dry_air_mass(pn, p_old, GRID_T5))
        )(p_new)
        expect = -_W[:, None] / (jnp.sum(_W) * _NLON) * jnp.ones((1, _NLON))
        expect = expect.at[0, 0].add(1.0)
        np.testing.assert_allclose(np.asarray(g), np.asarray(expect), rtol=1e-11)

    def test_moisture_corrector_pins_the_global_integral(self):
        from legoesm.ml.conservation import correct_moisture
        dsig = jnp.asarray(SIGMA.dsigma)
        p_s = jnp.full((_NLAT, _NLON), constants.p_ref)
        q_old = jnp.full((_NLAT, _NLON, NLEV), 5.0e-3)
        q_new = q_old * (1.0 + 0.1 * jax.random.uniform(
            jax.random.PRNGKey(61), q_old.shape, dtype=jnp.float64))

        def global_q(qn):
            qc = correct_moisture(qn, q_old, p_s, dsig, GRID_T5)
            col = jnp.sum(qc * dsig[None, None, :], axis=-1) * p_s
            return jnp.sum(col * _W[:, None])

        g = jax.grad(global_q)(q_new)
        assert jnp.all(jnp.isfinite(g))
        # Global moisture is pinned to the OLD value => zero sensitivity.
        assert float(jnp.max(jnp.abs(g))) < 1e-9 * float(jnp.sum(_W)) * float(
            constants.p_ref), "moisture corrector does not pin the global integral"
        # But the local field still responds.
        g_local = jax.grad(
            lambda qn: jnp.sum(
                correct_moisture(qn, q_old, p_s, dsig, GRID_T5)[0, 0] ** 2)
        )(q_new)
        _finite_and_nonzero(g_local, "correct_moisture local dL/dq_new")

    def test_clip_humidity_gradient_is_exactly_zero_below_zero(self):
        from legoesm.ml.conservation import clip_humidity
        q = jnp.array([-2.0, -1e-12, 1e-12, 3.0])
        g = jax.grad(lambda x: jnp.sum(clip_humidity(x)))(q)
        np.testing.assert_allclose(np.asarray(g), np.array([0.0, 0.0, 1.0, 1.0]))

    def test_ocean_volume_corrector_pins_masked_volume(self):
        from legoesm.ml.conservation import correct_ocean_volume
        mask = jnp.zeros((_NLAT, _NLON)).at[2:, :].set(1.0)
        eta_old = jnp.zeros((_NLAT, _NLON))
        eta_new = jax.random.normal(jax.random.PRNGKey(62), eta_old.shape,
                                    dtype=jnp.float64)
        area = mask * GRID_T5.grid_area

        def vol(e):
            return jnp.sum(correct_ocean_volume(e, eta_old, GRID_T5, mask) * area)

        g = jax.grad(vol)(eta_new)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) < 1e-6 * float(jnp.sum(area))
        # Land cells must not receive any correction sensitivity.
        g_land = jax.grad(
            lambda e: jnp.sum(
                correct_ocean_volume(e, eta_old, GRID_T5, mask)[:2] ** 2)
        )(eta_new)
        assert float(jnp.max(jnp.abs(g_land[2:]))) == 0.0, (
            "ocean cells influence land-cell eta — the mask gate leaked")

    @pytest.mark.parametrize("which", ["heat", "salt"])
    def test_ocean_tracer_correctors_pin_the_volume_integral(self, which):
        from legoesm.ml.conservation import correct_ocean_heat, correct_ocean_salt
        fn = correct_ocean_heat if which == "heat" else correct_ocean_salt
        mask = jnp.ones((_NLAT, _NLON))
        h_old = jnp.full((_NLAT, _NLON, NLEV), 50.0)
        h_new = h_old
        x_old = jnp.full((_NLAT, _NLON, NLEV), 5.0)
        x_new = x_old + jax.random.normal(
            jax.random.PRNGKey(63), x_old.shape, dtype=jnp.float64)
        area = (mask * GRID_T5.grid_area)[..., None]

        def integral(xn):
            xc = fn(xn, x_old, h_new, h_old, GRID_T5, mask)
            return jnp.sum(xc * h_new * area)

        g = jax.grad(integral)(x_new)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) < 1e-6 * float(jnp.sum(area))
        g_local = jax.grad(
            lambda xn: jnp.sum(
                fn(xn, x_old, h_new, h_old, GRID_T5, mask)[0, 0] ** 2))(x_new)
        _finite_and_nonzero(g_local, f"correct_ocean_{which} local gradient")


# ============================================================================
# 8. training/losses.py — dycore training objective
# ============================================================================

def _make_carry(T_val, u_val=0.0, v_val=0.0, q_val=5e-3, ps_val=None,
                flux_val=0.0):
    from legoesm.core.state import HydrostaticState
    from legoesm.driver.compiled_segments import pack_carry
    if ps_val is None:
        ps_val = float(constants.p_ref)
    s3 = (_NLAT, _NLON, NLEV)
    s2 = (_NLAT, _NLON)
    d3 = ("lat", "lon", "lev")
    d2 = ("lat", "lon")
    state = HydrostaticState(
        u=Field(jnp.full(s3, u_val), name="u", dims=d3, units="m/s"),
        v=Field(jnp.full(s3, v_val), name="v", dims=d3, units="m/s"),
        T=Field(jnp.full(s3, T_val), name="T", dims=d3, units="K"),
        p_s=Field(jnp.full(s2, ps_val), name="p_s", dims=d2, units="Pa"),
        phis=Field(jnp.zeros(s2), name="phis", dims=d2, units="m2/s2"),
    )
    return pack_carry(
        state, q_v=jnp.full(s3, q_val), q_c=jnp.zeros(s3), q_r=jnp.zeros(s3),
        held_dT_rad=jnp.zeros(s3),
        held_sw_net_sfc=jnp.full(s2, flux_val),
        held_lw_net_sfc=jnp.full(s2, flux_val),
        held_sw_up_toa=jnp.full(s2, flux_val),
        held_lw_up_toa=jnp.full(s2, flux_val),
        held_sw_down_toa=jnp.zeros(s2), step_index=0,
    )


_SIGMA_FULL = jnp.asarray(SIGMA.sigma_full)


class TestTrainingLosses:

    def test_every_prognostic_branch_is_reachable(self):
        from legoesm.training.losses import carry_mse
        pred = _make_carry(constants.T_freeze + 7.0, u_val=1.0, v_val=-1.0)
        tgt = _make_carry(constants.T_freeze + 9.0, u_val=2.0, v_val=1.0,
                          q_val=6e-3, ps_val=float(constants.p_ref) + 500.0)

        for name in ("T", "u", "v", "q_v", "p_s"):
            g = jax.grad(
                lambda x, n=name: carry_mse(pred._replace(**{n: x}), tgt,
                                            _SIGMA_FULL, lat_weights=_W)
            )(getattr(pred, name))
            _finite_and_nonzero(g, f"carry_mse dL/d{name}")

    def test_scale_normalization_balances_the_variable_branches(self):
        """With ``normalize_by_scale`` the per-variable gradient magnitude,
        measured in units of that variable's own anomaly scale, must be
        commensurate across T / u / q / ps.  Without normalisation the ratio
        spans ~9 orders of magnitude."""
        from legoesm.training.losses import carry_mse, LossConfig
        pred = _make_carry(constants.T_freeze + 7.0, u_val=1.0)
        tgt = _make_carry(constants.T_freeze + 9.0, u_val=2.0, q_val=6e-3,
                          ps_val=float(constants.p_ref) + 500.0)
        cfg = LossConfig()
        scales = {"T": cfg.T_scale, "u": cfg.wind_scale,
                  "q_v": cfg.q_scale, "p_s": cfg.ps_scale}
        mags = {}
        for name, sc in scales.items():
            g = jax.grad(
                lambda x, n=name: carry_mse(pred._replace(**{n: x}), tgt,
                                            _SIGMA_FULL, lat_weights=_W,
                                            config=cfg)
            )(getattr(pred, name))
            mags[name] = float(jnp.max(jnp.abs(g))) * sc
        lo, hi = min(mags.values()), max(mags.values())
        assert lo > 0.0
        assert hi / lo < 1e3, (
            f"scale-normalised gradient magnitudes span {hi/lo:.3g} orders "
            f"(want < 1e3): {mags}")

    def test_bias_terms_are_off_by_default_and_reachable_when_enabled(self):
        """``w_bias_T = 0`` must be a true no-op; turning it on must change
        the gradient (a silently-ignored weight is the failure mode)."""
        from legoesm.training.losses import carry_mse, LossConfig
        pred = _make_carry(constants.T_freeze + 7.0)
        tgt = _make_carry(constants.T_freeze + 9.0)

        def gT(cfg):
            return jax.grad(
                lambda x: carry_mse(pred._replace(T=x), tgt, _SIGMA_FULL,
                                    lat_weights=_W, config=cfg))(pred.T)

        g_off = gT(LossConfig())
        g_on = gT(LossConfig(w_bias_T=5.0))
        assert float(jnp.max(jnp.abs(g_on - g_off))) > 0.0, (
            "w_bias_T has no effect on the gradient — the bias term is dead")
        _finite_and_nonzero(g_on, "carry_mse (bias on) dL/dT")

    def test_radiation_flux_loss_weight_gating_is_structural(self):
        """Each ``held_*`` field must have EXACTLY zero gradient when its
        weight is 0 and a non-zero gradient when it is on."""
        from legoesm.training.losses import radiation_flux_loss, LossConfig
        pred = _make_carry(constants.T_freeze + 7.0, flux_val=10.0)
        tgt = _make_carry(constants.T_freeze + 7.0, flux_val=0.0)
        fields = {
            "held_sw_up_toa": "w_flux_rsut",
            "held_lw_up_toa": "w_flux_olr",
            "held_sw_net_sfc": "w_flux_sfc_sw",
            "held_lw_net_sfc": "w_flux_sfc_lw",
        }
        for fld, wname in fields.items():
            off = jax.grad(
                lambda x, f=fld: radiation_flux_loss(
                    pred._replace(**{f: x}), tgt, lat_weights=_W,
                    config=LossConfig()))(getattr(pred, fld))
            assert float(jnp.max(jnp.abs(off))) == 0.0, (
                f"{fld} receives gradient with all flux weights at 0")
            on = jax.grad(
                lambda x, f=fld, w=wname: radiation_flux_loss(
                    pred._replace(**{f: x}), tgt, lat_weights=_W,
                    config=LossConfig(**{w: 1.0})))(getattr(pred, fld))
            _finite_and_nonzero(on, f"radiation_flux_loss dL/d{fld}")

    def test_flux_terms_enter_carry_mse_only_when_enabled(self):
        from legoesm.training.losses import carry_mse, LossConfig
        pred = _make_carry(constants.T_freeze + 7.0, flux_val=10.0)
        tgt = _make_carry(constants.T_freeze + 7.0, flux_val=0.0)
        base = float(carry_mse(pred, tgt, _SIGMA_FULL, lat_weights=_W))
        withf = float(carry_mse(pred, tgt, _SIGMA_FULL, lat_weights=_W,
                                config=LossConfig(w_flux_olr=1.0)))
        assert withf > base
        g = jax.grad(
            lambda x: carry_mse(pred._replace(held_lw_up_toa=x), tgt,
                                _SIGMA_FULL, lat_weights=_W,
                                config=LossConfig(w_flux_olr=1.0))
        )(pred.held_lw_up_toa)
        _finite_and_nonzero(g, "carry_mse dL/dheld_lw_up_toa (flux on)")

    def test_level_weights_pressure_mode_is_differentiable_in_sigma(self):
        from legoesm.training.losses import level_weights, LossConfig
        sig = jnp.linspace(0.05, 0.975, 8)
        g = jax.grad(
            lambda s: jnp.sum(level_weights(s, config=LossConfig(
                level_weighting="pressure")) ** 2))(sig)
        _finite_and_nonzero(g, "level_weights('pressure') dL/dsigma")

    def test_level_weights_boundary_layer_mode_is_piecewise_constant(self):
        """NON-DIFFERENTIABLE BY CONSTRUCTION (documented): the
        ``boundary_layer`` weighting is ``jnp.where(p > 700 hPa, 2, 1)``, a
        step function, so d/dsigma is exactly 0.  Anyone tuning the vertical
        weighting by gradient must use ``level_weighting='pressure'``."""
        from legoesm.training.losses import level_weights, LossConfig
        sig = jnp.linspace(0.05, 0.975, 8)
        g = jax.grad(
            lambda s: jnp.sum(level_weights(s, config=LossConfig(
                level_weighting="boundary_layer")) ** 2))(sig)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) == 0.0

    def test_carry_spectral_loss_gradient_and_zero_error_behaviour(self):
        from legoesm.training.losses import carry_spectral_loss, LossConfig
        pred = _make_carry(constants.T_freeze + 7.0)
        tgt = _make_carry(constants.T_freeze + 9.0)
        g = jax.grad(
            lambda x: carry_spectral_loss(pred._replace(T=x), tgt, GRID_T5))(pred.T)
        _finite_and_nonzero(g, "carry_spectral_loss dL/dT")
        # Exact match -> finite, zero gradient (no complex-abs 0/0 NaN).
        g0 = jax.grad(
            lambda x: carry_spectral_loss(pred._replace(T=x), pred, GRID_T5))(pred.T)
        assert jnp.all(jnp.isfinite(g0)), (
            "carry_spectral_loss gradient is non-finite at a perfect match")
        assert float(jnp.max(jnp.abs(g0))) < 1e-20

    def test_combined_loss_spectral_term_changes_the_gradient(self):
        from legoesm.training.losses import combined_loss, LossConfig
        pred = _make_carry(constants.T_freeze + 7.0)
        tgt = _make_carry(constants.T_freeze + 9.0)

        def gT(cfg):
            return jax.grad(
                lambda x: combined_loss(pred._replace(T=x), tgt, _SIGMA_FULL,
                                        grid=GRID_T5, config=cfg))(pred.T)

        g0 = gT(LossConfig(spectral_weight=0.0))
        g1 = gT(LossConfig(spectral_weight=1.0))
        _finite_and_nonzero(g0, "combined_loss dL/dT (no spectral)")
        assert float(jnp.max(jnp.abs(g1 - g0))) > 0.0, (
            "spectral_weight has no effect on the gradient")


# ============================================================================
# 9. training/trainable_params.py — constraint transforms
# ============================================================================

class TestTrainableParams:

    def test_sigmoid_to_range_analytic_derivative(self):
        from legoesm.training.trainable_params import sigmoid_to_range
        lo, hi = 0.001, 0.005
        raw = jnp.array([-1.5, 0.0, 2.0])
        g = jax.grad(lambda r: jnp.sum(sigmoid_to_range(r, lo, hi)))(raw)
        s = jax.nn.sigmoid(raw)
        np.testing.assert_allclose(np.asarray(g),
                                   np.asarray((hi - lo) * s * (1 - s)),
                                   rtol=1e-12)
        _finite_and_nonzero(g, "sigmoid_to_range derivative")

    def test_sigmoid_round_trip_derivative_is_unity_in_the_interior(self):
        """``sigmoid_to_range(range_to_sigmoid_array(v))`` must be the
        identity with unit derivative for v strictly inside (lo, hi)."""
        from legoesm.training.trainable_params import (
            sigmoid_to_range, range_to_sigmoid_array,
        )
        lo, hi = 0.4, 0.8
        v = jnp.array([0.45, 0.6, 0.75])
        f = lambda x: sigmoid_to_range(range_to_sigmoid_array(x, lo, hi), lo, hi)
        np.testing.assert_allclose(np.asarray(f(v)), np.asarray(v), rtol=1e-10)
        g = jax.grad(lambda x: jnp.sum(f(x)))(v)
        np.testing.assert_allclose(np.asarray(g), np.ones_like(np.asarray(g)),
                                   rtol=1e-9)

    def test_range_to_sigmoid_clip_kills_the_gradient_out_of_bounds(self):
        """NON-DIFFERENTIABLE BY CONSTRUCTION (documented): the 0.001/0.999
        clamp makes the inverse transform's gradient EXACTLY zero for values
        at or beyond the bounds.  Seeding a trainable at its bound therefore
        freezes it; seeds must lie strictly inside."""
        from legoesm.training.trainable_params import range_to_sigmoid_array
        lo, hi = 0.4, 0.8
        v = jnp.array([lo, hi, 0.6])
        g = jax.grad(lambda x: jnp.sum(range_to_sigmoid_array(x, lo, hi)))(v)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.abs(g[0])) == 0.0 and float(jnp.abs(g[1])) == 0.0
        assert float(jnp.abs(g[2])) > 0.0

    def test_softplus_transform_is_reachable_and_positive(self):
        from legoesm.training.trainable_params import (
            ParamConstraint, TrainablePhysicsParams,
        )
        c = [ParamConstraint("tau_test", 0.0, 1e9, "softplus")]
        p = TrainablePhysicsParams(
            raw_values={"tau_test": jnp.asarray(3.0)}, constraints=c)
        g = eqx.filter_grad(
            lambda pp: jnp.sum(pp.as_dict()["tau_test"] ** 2))(p)
        _finite_and_nonzero(g.raw_values["tau_test"], "softplus dL/draw")
        assert float(p.as_dict()["tau_test"]) > 0.0

    def test_every_default_trainable_is_reachable(self):
        from legoesm.training.trainable_params import TrainablePhysicsParams
        p = TrainablePhysicsParams.from_defaults()
        g = eqx.filter_grad(
            lambda pp: sum(jnp.sum(v ** 2) for v in pp.as_dict().values()))(p)
        for name, leaf in g.raw_values.items():
            _finite_and_nonzero(leaf, f"TrainablePhysicsParams dL/draw[{name}]")

    @pytest.mark.parametrize(
        "conv,rad,turb",
        [("sbm", "gray", "none"), ("dca", "rrtmgp", "none"),
         ("sbm", "gray", "louis"), ("kuo", "none", "none")],
    )
    def test_scheme_filtered_constraints_are_all_reachable(self, conv, rad, turb):
        """Every constraint the scheme filter KEEPS must carry a non-zero
        gradient — the filter exists precisely to avoid dead DOFs."""
        from legoesm.training.trainable_params import (
            TrainablePhysicsParams, trainable_constraints_for_scheme,
        )
        cons = trainable_constraints_for_scheme(conv, rad, turb)
        if not cons:
            pytest.skip("no trainables under this scheme combination")
        p = TrainablePhysicsParams.from_defaults(cons)
        g = eqx.filter_grad(
            lambda pp: sum(jnp.sum(v ** 2) for v in pp.as_dict().values()))(p)
        for name, leaf in g.raw_values.items():
            _finite_and_nonzero(leaf, f"[{conv}/{rad}/{turb}] dL/draw[{name}]")

    def test_constrained_values_stay_inside_bounds_under_large_raw(self):
        """Saturating the sigmoid must not escape [lo, hi].

        PRECISION CAVEAT (measured, low severity — NOT flagged as a bug):
        ``from_defaults`` seeds raw values in the precision policy's compute
        dtype, which is float32 by default, and ``sigmoid_to_range`` casts
        ``lo``/``hi`` to that dtype.  The returned value is therefore the
        float32 REPRESENTATION of the bound, which can sit up to 1 float32
        ulp above it — measured ``albedo_ice = 0.800000011920929`` for a
        declared max of 0.8 (overshoot 1.19e-8).  The bound is compared
        against its own float32 rounding here; a genuine transform escape
        (>= 1 ulp) still fails.  Downstream code that treats a bound as a
        HARD physical limit (a fraction <= 1 feeding ``sqrt(1 - x)``) should
        clamp rather than rely on this transform.
        """
        from legoesm.training.trainable_params import (
            TrainablePhysicsParams, DEFAULT_TRAINABLE,
        )
        p = TrainablePhysicsParams.from_defaults()
        big = {k: jnp.asarray(50.0, dtype=v.dtype)
               for k, v in p.raw_values.items()}
        small = {k: jnp.asarray(-50.0, dtype=v.dtype)
                 for k, v in p.raw_values.items()}
        hi_vals = TrainablePhysicsParams(
            raw_values=big, constraints=p.constraints).as_dict()
        lo_vals = TrainablePhysicsParams(
            raw_values=small, constraints=p.constraints).as_dict()
        for c in DEFAULT_TRAINABLE:
            dt = p.raw_values[c.name].dtype
            lo_b = float(np.asarray(c.min_val, dtype=dt))
            hi_b = float(np.asarray(c.max_val, dtype=dt))
            v_hi, v_lo = float(hi_vals[c.name]), float(lo_vals[c.name])
            assert v_hi <= hi_b, (
                f"{c.name} = {v_hi} escaped its max {hi_b} (dtype {dt})")
            assert v_lo >= lo_b, (
                f"{c.name} = {v_lo} escaped its min {lo_b} (dtype {dt})")
            # Saturation is real: the two extremes must actually differ.
            assert v_hi > v_lo

    def test_to_overrides_preserves_the_traced_leaf(self):
        """Trained values must reach the config as TRACED leaves, not baked
        Python floats (the SegmentForcing doctrine)."""
        from legoesm.training.trainable_params import (
            ParamConstraint, TrainablePhysicsParams,
        )
        c = [ParamConstraint("s.f", 0.1, 0.9, "sigmoid",
                             scheme_key="s", field="f")]
        p = TrainablePhysicsParams(
            raw_values={"s.f": jnp.asarray(0.0)}, constraints=c)

        def loss(pp):
            return jnp.sum(pp.to_overrides()["s"]["f"] ** 2)

        g = eqx.filter_grad(loss)(p)
        _finite_and_nonzero(g.raw_values["s.f"], "to_overrides dL/draw")

    def test_to_segment_kwargs_rejects_scheme_qualified_params(self):
        from legoesm.training.trainable_params import (
            ParamConstraint, TrainablePhysicsParams,
        )
        c = [ParamConstraint("s.f", 0.1, 0.9, "sigmoid",
                             scheme_key="s", field="f")]
        p = TrainablePhysicsParams(
            raw_values={"s.f": jnp.asarray(0.0)}, constraints=c)
        with pytest.raises(ValueError, match="cannot inject scheme-qualified"):
            p.to_segment_kwargs()


# ============================================================================
# 10. training/param_collector.py — spec-driven collection + override splice
# ============================================================================

class TestParamCollector:

    def test_collected_core_params_are_all_reachable(self):
        from legoesm.training.param_collector import build_trainable_params
        p = build_trainable_params(tier="core")
        assert len(p.raw_values) > 0
        g = eqx.filter_grad(
            lambda pp: sum(jnp.sum(v ** 2) for v in pp.as_dict().values()))(p)
        dead = [k for k, v in g.raw_values.items()
                if float(jnp.max(jnp.abs(v))) == 0.0]
        assert not dead, (
            f"{len(dead)}/{len(g.raw_values)} spec-collected tier-1 parameters "
            f"have zero gradient through their constraint transform: {dead[:10]}")
        for k, v in g.raw_values.items():
            assert jnp.all(jnp.isfinite(v)), f"non-finite gradient for {k}"

    def test_extended_tier_is_a_superset_and_still_reachable(self):
        from legoesm.training.param_collector import build_trainable_params
        core = set(build_trainable_params(tier="core").raw_values)
        ext = build_trainable_params(tier="extended")
        assert core <= set(ext.raw_values)
        g = eqx.filter_grad(
            lambda pp: sum(jnp.sum(v ** 2) for v in pp.as_dict().values()))(ext)
        dead = [k for k, v in g.raw_values.items()
                if float(jnp.max(jnp.abs(v))) == 0.0]
        assert not dead, f"unreachable tier<=2 parameters: {dead[:10]}"

    def test_override_splice_keeps_the_gradient_path_into_a_real_config(self):
        """``apply_param_overrides`` must produce a config whose spliced
        field is a TRACED array — a ``_replace`` that dropped the tracer
        (or a copy that re-froze the default) zeroes the gradient."""
        from legoesm.training.param_collector import build_trainable_params
        from legoesm.core.param_overrides import apply_param_overrides
        from legoesm.ice.sea_ice import SeaIceConfig
        p = build_trainable_params(tier="core",
                                   active_scheme_keys={"ice.sea_ice"})
        assert p.raw_values, "expected ice.sea_ice tier-1 parameters"
        base = SeaIceConfig()

        def loss(pp):
            cfg = apply_param_overrides(base, pp.to_overrides()["ice.sea_ice"])
            # A scalar functional of the spliced fields only.
            return sum(jnp.sum(jnp.asarray(getattr(cfg, f)) ** 2)
                       for f in pp.to_overrides()["ice.sea_ice"])

        g = eqx.filter_grad(loss)(p)
        for k, v in g.raw_values.items():
            _finite_and_nonzero(v, f"apply_param_overrides dL/draw[{k}]")

    def test_override_splice_rejects_unknown_fields(self):
        from legoesm.core.param_overrides import apply_param_overrides
        from legoesm.ice.sea_ice import SeaIceConfig
        with pytest.raises(ValueError, match="has no field"):
            apply_param_overrides(SeaIceConfig(),
                                  {"definitely_not_a_field": jnp.asarray(1.0)})


# ============================================================================
# 11. ml/physics/model.py — column MLP parameterization
# ============================================================================

class TestPhysicsParameterizationModel:

    @staticmethod
    def _model(scheme="none", nlev=NLEV, seed=70):
        from legoesm.ml.physics.model import PhysicsParameterizationModel
        return PhysicsParameterizationModel(
            nlev=nlev, hidden_dim=8, n_layers=2,
            microphysics_scheme=scheme, key=jax.random.PRNGKey(seed))

    @pytest.mark.parametrize("scheme", ["none", "kessler", "sundqvist"])
    def test_all_leaves_and_the_input_are_reachable(self, scheme):
        m = self._model(scheme)
        x = jax.random.normal(jax.random.PRNGKey(71), (m.n_input,),
                              dtype=jnp.float64)
        g_par = eqx.filter_grad(lambda mm: jnp.sum(mm(x) ** 2))(m)
        maxima = _tree_max_abs(g_par)
        dead = [i for i, v in enumerate(maxima) if v == 0.0]
        assert not dead, f"[{scheme}] unreachable MLP leaves at {dead}"
        assert all(np.isfinite(v) for v in maxima)
        g_in = jax.grad(lambda xx: jnp.sum(m(xx) ** 2))(x)
        _finite_and_nonzero(g_in, f"[{scheme}] PhysicsParameterizationModel dL/dx")

    def test_output_size_matches_the_head(self):
        from legoesm.ml.physics.model import (
            physics_parameterization_output_size,
        )
        m = self._model("kessler")
        x = jnp.zeros((m.n_input,))
        assert m(x).shape == (
            physics_parameterization_output_size(
                NLEV, microphysics_scheme="kessler"),)

    def test_unknown_microphysics_scheme_raises(self):
        from legoesm.ml.physics.model import PhysicsParameterizationModel
        with pytest.raises(ValueError, match="Unsupported microphysics_scheme"):
            PhysicsParameterizationModel(
                nlev=NLEV, hidden_dim=4, n_layers=1,
                microphysics_scheme="typo", key=jax.random.PRNGKey(0))


# ============================================================================
# 12. training/dycore_rollout.py — checkpointed multi-segment adjoint
# ============================================================================

class TestDifferentiableRollout:
    """The rollout owns a ``lax.scan`` + per-segment ``jax.checkpoint``.

    A cheap analytic segment (T *= decay each segment) makes the exact
    gradient known in closed form, so this validates the ADJOINT, not just
    its finiteness.
    """

    @staticmethod
    def _run(n_days, gradient_checkpoint, decay=0.9):
        from legoesm.training.dycore_rollout import (
            differentiable_rollout, RolloutConfig,
        )
        carry = _make_carry(constants.T_freeze + 7.0)

        def seg(c, n_steps, forcing):
            return c._replace(T=c.T * decay)

        cfg = RolloutConfig(n_days=n_days, segment_steps=1,
                            gradient_checkpoint=gradient_checkpoint,
                            checkpoint_schedule=(
                                "uniform" if gradient_checkpoint else "none"))

        def loss(T0):
            out = differentiable_rollout(carry._replace(T=T0), None, seg, cfg)
            return jnp.sum(out.final_carry.T ** 2)

        return carry.T, loss

    @pytest.mark.parametrize("n_days", [1, 3, 5])
    def test_adjoint_matches_the_closed_form(self, n_days):
        decay = 0.9
        T0, loss = self._run(n_days, True, decay)
        g = jax.grad(loss)(T0)
        expect = 2.0 * (decay ** (2 * n_days)) * T0
        _finite_and_nonzero(g, f"rollout({n_days}d) dL/dT0")
        np.testing.assert_allclose(np.asarray(g), np.asarray(expect), rtol=1e-11)

    def test_checkpointing_does_not_change_the_gradient(self):
        T0, loss_ck = self._run(4, True)
        _, loss_no = self._run(4, False)
        np.testing.assert_allclose(np.asarray(jax.grad(loss_ck)(T0)),
                                   np.asarray(jax.grad(loss_no)(T0)),
                                   rtol=1e-12, atol=1e-14)

    def test_saved_trajectory_is_differentiable_at_every_lead(self):
        """Multi-day supervision differentiates through the STACKED states,
        not just the endpoint; each lead must contribute."""
        from legoesm.training.dycore_rollout import (
            differentiable_rollout, RolloutConfig,
        )
        carry = _make_carry(constants.T_freeze + 7.0)
        decay = 0.8

        def seg(c, n_steps, forcing):
            return c._replace(T=c.T * decay)

        cfg = RolloutConfig(n_days=3, segment_steps=1, gradient_checkpoint=True)

        def loss(T0, lead):
            out = differentiable_rollout(carry._replace(T=T0), None, seg, cfg)
            return jnp.sum(out.saved_states.T[lead] ** 2)

        for lead in range(3):
            g = jax.grad(lambda t, l=lead: loss(t, l))(carry.T)
            expect = 2.0 * (decay ** (2 * (lead + 1))) * carry.T
            np.testing.assert_allclose(np.asarray(g), np.asarray(expect),
                                       rtol=1e-11)

    def test_invalid_schedule_and_storage_combinations_raise(self):
        from legoesm.training.dycore_rollout import (
            differentiable_rollout, RolloutConfig,
        )
        carry = _make_carry(constants.T_freeze + 7.0)
        seg = lambda c, n, f: c
        with pytest.raises(ValueError, match="checkpoint_schedule"):
            differentiable_rollout(
                carry, None, seg,
                RolloutConfig(n_days=1, checkpoint_schedule="binomial"))
        with pytest.raises(ValueError, match="requires gradient"):
            differentiable_rollout(
                carry, None, seg,
                RolloutConfig(n_days=1, gradient_checkpoint=False,
                              storage="host"))


# ============================================================================
# 13. Non-differentiable surfaces in ``ml`` — documented, not forced
# ============================================================================

class TestNonDifferentiableSurfaces:
    """Modules in ``packages/ml`` that legitimately carry NO gradient path.

    Documented with evidence so a future reader does not "fix" them:

    * ``ml/data/era5_loader.py`` — xarray/GCS IO, returns host arrays; the
      network never differentiates w.r.t. the loader.
    * ``ml/s2s/**`` postprocess / evaluation / regrid / cli — numpy +
      xarray scoring and file plumbing (the ONE gradient surface there,
      ``area_weighted_afcrps`` inside ``_s2s_train_step``, is covered by
      :class:`TestCRPS`).
    * ``ml/physics/{data,io,plotting,evaluate}.py`` — dataset assembly,
      checkpoint IO and matplotlib.
    * ``tuning.py`` — pure Python config validation (no jnp at all).
    * ``training/{campaign_*,distributed_*,sweep_planner,column_manifest,
      compare_reanalysis,rda_era5}.py`` — orchestration / IO.
    """

    def test_tuning_module_has_no_jax_surface(self):
        import legoesm.tuning as tuning
        from legoesm.tuning import recommended_params
        assert not hasattr(tuning, "jnp")
        rec = recommended_params(resolution=32, nlev=NLEV)
        assert isinstance(rec, dict) and rec

    def test_era5_loader_returns_host_arrays_not_a_gradient_path(self):
        """The loader is import-only here (no GCS access): assert it exposes
        a plain NamedTuple config and no differentiable entry point."""
        from legoesm.ml.data.era5_loader import ERA5Config
        cfg = ERA5Config()
        assert hasattr(cfg, "_fields")


# ============================================================================
# 14. Remaining training-side gradient surfaces
# ============================================================================

class TestSCMRCEMetrics:
    """``training/scm_rce_metrics.py`` — the AD-safe sqrt used by the
    gradient-based SCM-RCE parameter trainer."""

    def test_safe_sqrt_is_exact_forward_and_finite_at_zero(self):
        from legoesm.training.scm_rce_metrics import safe_sqrt
        x = jnp.array([0.0, 1e-30, 4.0, 9.0])
        np.testing.assert_allclose(np.asarray(safe_sqrt(x)),
                                   np.sqrt(np.asarray(x)), rtol=1e-14)
        g = jax.grad(lambda v: jnp.sum(safe_sqrt(v)))(x)
        assert jnp.all(jnp.isfinite(g)), (
            "safe_sqrt gradient is non-finite — the double-where guard "
            "regressed (a plain jnp.sqrt gives inf at 0 and NaN downstream)")
        assert float(g[0]) == 0.0
        np.testing.assert_allclose(np.asarray(g[2:]),
                                   np.array([0.25, 1.0 / 6.0]), rtol=1e-12)

    def test_plain_sqrt_would_be_nan_here(self):
        """Non-vacuity witness for the guard above."""
        g = jax.grad(lambda v: jnp.sum(jnp.sqrt(v)))(jnp.array([0.0, 4.0]))
        assert not jnp.isfinite(g[0])

    def test_weighted_rmse_gradient_is_finite_at_a_perfect_fit(self):
        from legoesm.training.scm_rce_metrics import weighted_rmse
        w = jnp.full((NLEV,), 1.0 / NLEV)
        g0 = jax.grad(lambda d: weighted_rmse(d, w))(jnp.zeros(NLEV))
        assert jnp.all(jnp.isfinite(g0)) and float(jnp.max(jnp.abs(g0))) == 0.0
        d = jnp.array([1.0, -2.0, 0.5])
        g = jax.grad(lambda x: weighted_rmse(x, w))(d)
        rmse = weighted_rmse(d, w)
        np.testing.assert_allclose(np.asarray(g), np.asarray(w * d / rmse),
                                   rtol=1e-12)

    def test_weighted_std_gradient_is_finite_for_a_uniform_profile(self):
        from legoesm.training.scm_rce_metrics import weighted_std
        w = jnp.full((NLEV,), 1.0 / NLEV)
        g = jax.grad(lambda p: weighted_std(p, w))(jnp.full((NLEV,), 300.0))
        assert jnp.all(jnp.isfinite(g)) and float(jnp.max(jnp.abs(g))) == 0.0
        g2 = jax.grad(lambda p: weighted_std(p, w))(
            jnp.array([290.0, 300.0, 310.0]))
        _finite_and_nonzero(g2, "weighted_std on a varying profile")


class TestAIMIPSpatialField:
    """``training/aimip_spatial.py`` — low-rank trainable lat-lon knobs."""

    @staticmethod
    def _field(transform="log_perturb", init_std=0.1, seed=80):
        from legoesm.training.aimip_spatial import SpatialField
        return SpatialField.from_defaults(
            f_0=2.0e-3, scale=0.5, transform=transform, dtype=jnp.float64,
            init_std=init_std, key=jax.random.PRNGKey(seed))

    @pytest.mark.parametrize("transform", ["log_perturb", "shift"])
    def test_every_basis_coefficient_is_reachable(self, transform):
        f = self._field(transform)
        g = eqx.filter_grad(
            lambda ff: jnp.sum(ff.evaluate(GRID_T5) ** 2))(f)
        _finite_and_nonzero(g.coeffs, f"[{transform}] SpatialField dL/dcoeffs")
        dead = [i for i, v in enumerate(np.asarray(g.coeffs)) if v == 0.0]
        assert not dead, (
            f"[{transform}] basis coefficients {dead} are unreachable — "
            f"those spatial degrees of freedom cannot be trained")

    def test_zero_coefficients_reduce_to_the_scalar_baseline(self):
        f = self._field(init_std=0.0)
        v = f.evaluate(GRID_T5)
        np.testing.assert_allclose(np.asarray(v), np.full_like(np.asarray(v),
                                                               2.0e-3),
                                   rtol=1e-12)

    def test_land_mask_gates_the_gradient_off_ocean_columns(self):
        """Ocean columns fall back to ``f_0``; with a STATIC f_0 they must
        contribute exactly nothing to the coefficient gradient."""
        f = self._field()
        mask = jnp.zeros((GRID_T5.n_lat, GRID_T5.n_lon)).at[:3, :].set(1.0)
        g_ocean = eqx.filter_grad(
            lambda ff: jnp.sum(ff.evaluate(GRID_T5, land_mask=mask)[3:] ** 2))(f)
        assert float(jnp.max(jnp.abs(g_ocean.coeffs))) == 0.0, (
            "ocean (mask == 0) columns still drive the land coefficients")
        g_land = eqx.filter_grad(
            lambda ff: jnp.sum(ff.evaluate(GRID_T5, land_mask=mask)[:3] ** 2))(f)
        _finite_and_nonzero(g_land.coeffs, "SpatialField land-column gradient")

    def test_f_0_override_restores_the_scalar_gradient_path(self):
        """``f_0`` is a STATIC field, so without the override the trained
        scalar leaf is bypassed.  The override must be differentiable."""
        f = self._field()
        g = jax.grad(
            lambda b: jnp.sum(f.evaluate(GRID_T5, f_0_override=b) ** 2))(
                jnp.asarray(2.0e-3))
        _finite_and_nonzero(g, "SpatialField dL/df_0_override")

    def test_unknown_transform_raises(self):
        from legoesm.training.aimip_spatial import SpatialField
        bad = SpatialField(coeffs=jnp.zeros(13), f_0=1.0, scale=0.5,
                           transform="typo", l_max=4, m_max=2)
        with pytest.raises(ValueError, match="Unknown transform"):
            bad.evaluate(GRID_T5)


class TestParameterField:
    """``training/parameter_field.py`` — column diagnoses -> traced field."""

    def test_scatter_is_differentiable_only_at_the_written_columns(self):
        from legoesm.training.parameter_field import scatter_column_field
        shape = (4, 5)
        idx = jnp.array([0, 7, 19])
        vals = jnp.array([1.0, 2.0, 3.0])
        g = jax.grad(
            lambda v: jnp.sum(scatter_column_field(shape, idx, v,
                                                   background=0.5) ** 2))(vals)
        np.testing.assert_allclose(np.asarray(g), np.asarray(2.0 * vals),
                                   rtol=1e-12)
        # Unwritten columns keep the background and carry no value-gradient.
        field = scatter_column_field(shape, idx, vals, background=0.5)
        flat = field.reshape(-1)
        untouched = [i for i in range(20) if i not in (0, 7, 19)]
        assert float(jnp.max(jnp.abs(flat[jnp.array(untouched)] - 0.5))) == 0.0

    def test_invalid_samples_are_structurally_frozen(self):
        """``valid=False`` samples must have EXACTLY zero gradient (they
        keep the background), so a flagged-bad diagnosis cannot train."""
        from legoesm.training.parameter_field import scatter_column_field
        idx = jnp.array([0, 7, 19])
        vals = jnp.array([1.0, 2.0, 3.0])
        valid = jnp.array([True, False, True])
        g = jax.grad(
            lambda v: jnp.sum(scatter_column_field(
                (4, 5), idx, v, background=0.5, valid=valid) ** 2))(vals)
        assert float(jnp.abs(g[1])) == 0.0
        assert float(jnp.abs(g[0])) > 0.0 and float(jnp.abs(g[2])) > 0.0

    def test_kernel_regression_is_differentiable_in_values_and_predictors(self):
        from legoesm.training.parameter_field import environment_kernel_field
        k = jax.random.split(jax.random.PRNGKey(81), 3)
        grid_env = jax.random.normal(k[0], (12, 2), dtype=jnp.float64)
        samp_env = jax.random.normal(k[1], (4, 2), dtype=jnp.float64)
        samp_val = jax.random.normal(k[2], (4,), dtype=jnp.float64)
        ls = jnp.array([1.0, 1.0])

        def loss(v, se, ge):
            return jnp.sum(environment_kernel_field(
                ge, se, v, length_scales=ls) ** 2)

        gv, gs, gg = jax.grad(loss, argnums=(0, 1, 2))(samp_val, samp_env,
                                                       grid_env)
        _finite_and_nonzero(gv, "environment_kernel_field dL/dsample_values")
        _finite_and_nonzero(gs, "environment_kernel_field dL/dsample_env")
        _finite_and_nonzero(gg, "environment_kernel_field dL/dgrid_env")

    def test_kernel_regression_no_neighbour_fallback_is_ad_safe(self):
        """Columns far outside the sampled environment hull fall back to the
        background; the masked normaliser must not leak a 0/0 NaN."""
        from legoesm.training.parameter_field import environment_kernel_field
        grid_env = jnp.array([[0.0, 0.0], [1e6, 1e6]])
        samp_env = jnp.array([[0.0, 0.0]])
        samp_val = jnp.array([2.0])
        ls = jnp.array([1.0, 1.0])
        f = environment_kernel_field(grid_env, samp_env, samp_val,
                                     length_scales=ls, background=-1.0)
        assert jnp.all(jnp.isfinite(f))
        np.testing.assert_allclose(float(f[1]), -1.0, rtol=1e-12)
        g = jax.grad(lambda v: jnp.sum(environment_kernel_field(
            grid_env, samp_env, v, length_scales=ls,
            background=-1.0) ** 2))(samp_val)
        assert jnp.all(jnp.isfinite(g)), (
            "no-neighbour fallback produces a non-finite gradient (0/0 leak)")


class TestGradHorizon:
    """``training/grad_horizon.py`` — adjoint-norm diagnostics."""

    def test_global_grad_norm_matches_optax_global_norm(self):
        import optax
        from legoesm.training.grad_horizon import global_grad_norm
        tree = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([[12.0]])}
        np.testing.assert_allclose(float(global_grad_norm(tree)),
                                   float(optax.global_norm(tree)), rtol=1e-12)

    def test_global_grad_norm_is_complex_safe(self):
        from legoesm.training.grad_horizon import global_grad_norm
        tree = {"z": jnp.array([3.0 + 4.0j])}
        np.testing.assert_allclose(float(global_grad_norm(tree)), 5.0,
                                   rtol=1e-12)

    def test_grad_norm_grows_with_horizon_for_an_amplifying_map(self):
        from legoesm.training.grad_horizon import (
            grad_norm_vs_horizon, estimate_growth_rate,
        )
        lam = 1.3

        def loss_for_horizon(n, x):
            return jnp.sum((lam ** n * x) ** 2)

        x = jnp.ones(4)
        norms = grad_norm_vs_horizon(loss_for_horizon, [1, 2, 4, 8], x)
        assert all(np.isfinite(v) and v > 0 for v in norms.values())
        assert norms[8] > norms[4] > norms[2] > norms[1]
        # d/dx of lam^{2n} |x|^2 => norm ~ lam^{2n}; slope = 2 log(lam).
        np.testing.assert_allclose(estimate_growth_rate(norms),
                                   2.0 * np.log(lam), rtol=1e-8)

    def test_derivative_free_etki_is_not_a_gradient_surface(self):
        """DERIVATIVE-FREE BY DESIGN (documented): ETKI is an ensemble
        Kalman inversion — it approximates the sensitivity from the ensemble
        spread instead of differentiating the forward map, which is exactly
        why it exists alongside the AD trainers.  Its update is a pure array
        transform, so ``jax.grad`` through it is well-defined but physically
        meaningless; assert only that it runs and does not emit NaNs."""
        from legoesm.training.etki import etki_update
        k = jax.random.split(jax.random.PRNGKey(82), 2)
        theta = jax.random.normal(k[0], (6, 3), dtype=jnp.float64)
        g_eval = jax.random.normal(k[1], (6, 4), dtype=jnp.float64)
        y = jnp.zeros(4)
        step = etki_update(theta, g_eval, y)
        assert step.theta.shape == theta.shape
        assert jnp.all(jnp.isfinite(step.theta))
        assert np.isfinite(float(step.misfit))
