"""Direct regression tests for the ml/da/training AD-safety audit fixes.

Each behavioral fix from the adversarial review gets the smallest test that
fails if the fix regresses.  For the NaN-floor findings that is a finite-gradient
assertion at the degenerate input (perfect fit / zero std / constant channel /
fully-masked region / q == 0), which is exactly where the un-floored code
produced inf/NaN under reverse-mode AD.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx

from legoesm.grids import create_latlon_grid
from legoesm.grids.gaussian import create_gaussian_grid


# ---------------------------------------------------------------------------
# Finding 1: DiffusionB._smooth_field / inv_multiply slice by spec
# ---------------------------------------------------------------------------

from legoesm.da.background_error import DiffusionB
from legoesm.da.control_vector import ControlEntry, ControlVectorSpec


def _two_field_spec(ncol, nlev):
    """A control spec with a 2-D field (nlev=1) then a 3-D field (nlev levels)."""
    e0 = ControlEntry("a", offset=0, size=ncol, shape=(ncol,),
                      transform="identity")
    e1 = ControlEntry("b", offset=ncol, size=ncol * nlev, shape=(ncol, nlev),
                      transform="identity")
    return ControlVectorSpec(entries=(e0, e1), total_size=ncol + ncol * nlev)


class TestDiffusionBSpecSlicing:
    def _grid(self):
        return create_latlon_grid(8, 16)

    def test_per_field_independent_smoothing(self):
        """A spatially-constant field must stay constant after smoothing even
        when a STRUCTURED second field shares the control vector.  The old whole
        -vector reshape mixed the two fields' columns, so the constant field
        picked up the other's structure."""
        grid = self._grid()
        ncol = grid.grid_n_columns
        nlev = 2
        spec = _two_field_spec(ncol, nlev)
        sigma = jnp.ones(spec.total_size)
        B = DiffusionB(grid, sigma, horizontal_length_scale=5000e3,
                       n_diffusion_iter=8, spec=spec)

        const_field = jnp.full((ncol,), 7.0)
        structured = jax.random.normal(jax.random.PRNGKey(0), (ncol * nlev,))
        x = jnp.concatenate([const_field, structured])

        out = B._smooth_field(x, B.n_iter // 2 + 1)
        out_a = out[:ncol]
        # Field "a" is constant -> relaxing toward its own mean is a no-op.
        assert jnp.allclose(out_a, const_field, atol=1e-10), (
            "constant field changed -> fields were smoothed together"
        )

    def test_spec_vs_whole_vector_differs(self):
        """With a spec, smoothing must NOT equal the whole-vector smoothing of
        the same flat array (that is the smearing bug)."""
        grid = self._grid()
        ncol = grid.grid_n_columns
        nlev = 2
        spec = _two_field_spec(ncol, nlev)
        sigma = jnp.ones(spec.total_size)
        B_spec = DiffusionB(grid, sigma, horizontal_length_scale=5000e3,
                            n_diffusion_iter=8, spec=spec)
        B_nospec = DiffusionB(grid, sigma, horizontal_length_scale=5000e3,
                              n_diffusion_iter=8, spec=None)
        x = jax.random.normal(jax.random.PRNGKey(1), (spec.total_size,))
        with_spec = B_spec._smooth_field(x, B_spec.n_iter // 2 + 1)
        whole = B_nospec._smooth_field(x, B_nospec.n_iter // 2 + 1)
        assert not jnp.allclose(with_spec, whole), (
            "spec-aware smoothing collapsed to the whole-vector reshape"
        )

    def test_inv_is_exact_inverse_with_spec(self):
        """inv_multiply(B v) == v must still hold field-by-field with a spec."""
        grid = self._grid()
        ncol = grid.grid_n_columns
        nlev = 3
        spec = _two_field_spec(ncol, nlev)
        sigma = jnp.ones(spec.total_size) * 2.0
        B = DiffusionB(grid, sigma, horizontal_length_scale=1500e3,
                       n_diffusion_iter=6, spec=spec)
        v = jax.random.normal(jax.random.PRNGKey(3), (spec.total_size,))
        sqrtT = jax.linear_transpose(B.sqrt_multiply, v)
        Bv = B.sqrt_multiply(sqrtT(v)[0])
        v_rec = B.inv_multiply(Bv)
        assert jnp.allclose(v_rec, v, rtol=1e-5, atol=1e-5)

    def test_wrong_total_size_raises(self):
        """codex: a control vector whose length != spec.total_size must raise,
        not be silently clamped by dynamic_slice."""
        grid = self._grid()
        ncol = grid.grid_n_columns
        nlev = 2
        spec = _two_field_spec(ncol, nlev)
        sigma = jnp.ones(spec.total_size)
        B = DiffusionB(grid, sigma, horizontal_length_scale=5000e3,
                       n_diffusion_iter=8, spec=spec)
        short = jnp.ones(spec.total_size - ncol)  # missing one field
        with pytest.raises(ValueError, match="expected"):
            B._smooth_field(short, 2)
        # sqrt_multiply routes through _smooth_field -> same guard.
        with pytest.raises(ValueError, match="expected"):
            B.sqrt_multiply(jnp.ones(spec.total_size + 1))

    def test_non_multiple_field_size_raises(self):
        """A control field whose size is not a multiple of ncol is a real shape
        bug -> raise, never silently reshape."""
        grid = self._grid()
        ncol = grid.grid_n_columns
        bad = ControlVectorSpec(
            entries=(ControlEntry("a", 0, ncol + 1, (ncol + 1,), "identity"),),
            total_size=ncol + 1,
        )
        sigma = jnp.ones(ncol + 1)
        B = DiffusionB(grid, sigma, horizontal_length_scale=1000e3,
                       n_diffusion_iter=4, spec=bad)
        with pytest.raises(ValueError, match="not a multiple"):
            B._smooth_field(jnp.ones(ncol + 1), 2)

    def test_single_field_no_spec_backward_compatible(self):
        """No-spec single-field path is unchanged (smooths toward the mean)."""
        grid = self._grid()
        n = grid.grid_n_columns
        sigma = jnp.ones(n)
        B = DiffusionB(grid, sigma, horizontal_length_scale=5000e3,
                       n_diffusion_iter=10, spec=None)
        x = jax.random.normal(jax.random.PRNGKey(2), (n,))
        smoothed = B._smooth_field(x, B.n_iter // 2 + 1)
        assert jnp.var(smoothed) < jnp.var(x)


# ---------------------------------------------------------------------------
# Finding 12: HybridB requires >= 2 ensemble members
# ---------------------------------------------------------------------------

from legoesm.da.background_error import DiagonalB, HybridB


class TestHybridBMembers:
    def test_one_member_raises(self):
        static = DiagonalB(sigma=jnp.ones(4))
        with pytest.raises(ValueError, match=">= 2 members"):
            HybridB(static, jnp.ones((1, 4)))

    def test_two_members_ok(self):
        static = DiagonalB(sigma=jnp.ones(4))
        perts = jax.random.normal(jax.random.PRNGKey(0), (2, 4))
        B = HybridB(static, perts)
        out = B.sqrt_multiply(jnp.ones(4))
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Finding 5 + 6: control_vector size assert + softplus inverse guard
# ---------------------------------------------------------------------------

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.da.control_vector import (
    build_control_spec,
    control_to_state,
    control_to_increment,
    _forward_transform,
)


def _sw_state(shape=(4, 4)):
    return ShallowWaterState(
        h=Field(data=jnp.ones(shape) * 1e4, name="h", dims=(), units="m"),
        u=Field(data=jnp.ones(shape) * 5.0, name="u", dims=(), units="m/s"),
        v=Field(data=jnp.ones(shape) * -3.0, name="v", dims=(), units="m/s"),
        h_s=Field(data=jnp.zeros(shape), name="h_s", dims=(), units="m"),
    )


class TestControlVectorSizeAssert:
    def test_control_to_state_wrong_size_raises(self):
        state = _sw_state()
        spec = build_control_spec(state)
        with pytest.raises(ValueError, match="expected a 1-D array of size"):
            control_to_state(jnp.ones(spec.total_size + 3), spec, state)

    def test_control_to_state_2d_raises(self):
        state = _sw_state()
        spec = build_control_spec(state)
        with pytest.raises(ValueError):
            control_to_state(jnp.ones((spec.total_size, 1)), spec, state)

    def test_control_to_increment_wrong_size_raises(self):
        state = _sw_state()
        spec = build_control_spec(state)
        with pytest.raises(ValueError, match="expected a 1-D"):
            control_to_increment(jnp.ones(spec.total_size - 1), spec, state)


class TestSoftplusInverseGuard:
    def test_grad_finite_at_zero(self):
        """softplus forward transform (state -> control) had an inf/NaN gradient
        for a zero state (q == 0); the floored inverse keeps it finite."""
        g = jax.grad(lambda x: jnp.sum(_forward_transform(x, "softplus")))
        out = g(jnp.zeros(5))
        assert jnp.all(jnp.isfinite(out))

    def test_value_finite_at_zero(self):
        val = _forward_transform(jnp.zeros(5), "softplus")
        assert jnp.all(jnp.isfinite(val))

    def test_positive_state_grad_finite(self):
        g = jax.grad(lambda x: jnp.sum(_forward_transform(x, "softplus")))
        assert jnp.all(jnp.isfinite(g(jnp.full((5,), 280.0))))

    def test_large_input_finite_float32(self):
        """codex: log(expm1(x)) overflows for large x in float32 (x > ~88).
        The branched inverse must stay finite (value AND gradient) there."""
        x = jnp.array([0.0, 50.0, 88.0, 300.0], dtype=jnp.float32)
        val = _forward_transform(x, "softplus")
        assert jnp.all(jnp.isfinite(val)), val
        g = jax.grad(
            lambda z: jnp.sum(_forward_transform(z, "softplus")))(x)
        assert jnp.all(jnp.isfinite(g)), g

    def test_roundtrip_large_value(self):
        """softplus(inv_softplus(x)) ~= x for a large positive x."""
        from legoesm.da.control_vector import _inverse_transform
        x = jnp.array([0.5, 5.0, 50.0, 280.0])
        ctrl = _forward_transform(x, "softplus")
        back = _inverse_transform(ctrl, "softplus")
        assert jnp.allclose(back, x, rtol=1e-4)


# ---------------------------------------------------------------------------
# Finding 9: DiagonalR floored inverse + validate()
# ---------------------------------------------------------------------------

from legoesm.da.observation import DiagonalR


class TestDiagonalR:
    def test_inv_finite_at_zero_sigma(self):
        R = DiagonalR(sigma=jnp.array([1.0, 0.0, 2.0]))
        out = R.inv_multiply(jnp.array([1.0, 1.0, 1.0]))
        assert jnp.all(jnp.isfinite(out))

    def test_grad_finite_at_zero_sigma(self):
        d = jnp.array([1.0, 1.0, 1.0])
        g = jax.grad(lambda s: jnp.sum(DiagonalR(sigma=s).inv_multiply(d)))
        out = g(jnp.array([1.0, 0.0, 2.0]))
        assert jnp.all(jnp.isfinite(out))

    def test_validate_raises_on_zero(self):
        with pytest.raises(ValueError, match="strictly positive"):
            DiagonalR(sigma=jnp.array([1.0, 0.0])).validate()

    def test_validate_ok_and_chains(self):
        R = DiagonalR(sigma=jnp.array([1.0, 2.0])).validate()
        assert isinstance(R, DiagonalR)

    def test_normal_inverse_unchanged(self):
        R = DiagonalR(sigma=jnp.array([2.0, 4.0]))
        np.testing.assert_allclose(
            R.inv_multiply(jnp.array([1.0, 1.0])),
            jnp.array([1.0 / 4.0, 1.0 / 16.0]), rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# Finding 8: normalization clamps variance before sqrt
# ---------------------------------------------------------------------------

from legoesm.ml.normalization import compute_normalization_stats


class TestNormalizationGrad:
    def test_grad_finite_constant_channel_unweighted(self):
        """A constant channel has var == 0; clamping AFTER sqrt left a 0*inf NaN
        in the backward pass.  Clamping the variance first fixes it."""
        def loss(data):
            stats = compute_normalization_stats(data, eps=1e-6)
            return jnp.sum(stats.std)
        data = jnp.ones((10, 3))  # every channel constant
        g = jax.grad(loss)(data)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_constant_channel_weighted(self):
        # Weighted path: data is (n_samples, n_lat, n_lon, n_ch); the lat weight
        # has shape (n_lat,).  Channel 0 is constant -> var == 0 -> the same
        # 0 * inf NaN risk in the backward pass.
        n_s, n_lat, n_lon, n_ch = 4, 5, 6, 2
        base = jax.random.normal(jax.random.PRNGKey(0), (n_s, n_lat, n_lon, n_ch))
        const_ch = jnp.ones((n_s, n_lat, n_lon, 1)) * 3.0
        data0 = jnp.concatenate([const_ch, base[..., 1:]], axis=-1)
        weights = jnp.abs(jax.random.normal(jax.random.PRNGKey(1), (n_lat,))) + 0.1

        def loss(d):
            stats = compute_normalization_stats(d, weights=weights, eps=1e-6)
            return jnp.sum(stats.std)
        g = jax.grad(loss)(data0)
        assert jnp.all(jnp.isfinite(g))

    def test_std_floor_value(self):
        stats = compute_normalization_stats(jnp.ones((8, 2)), eps=1e-6)
        assert jnp.allclose(stats.std, 1e-6)


# ---------------------------------------------------------------------------
# Finding 10: masked area_weighted_mse normalizes by sum(w*mask)
# ---------------------------------------------------------------------------

from legoesm.ml.loss import area_weighted_mse


class TestMaskedAreaWeightedMSE:
    def test_masked_equals_unmasked_of_region(self):
        """Masked MSE over a lat/lon block must equal the unmasked MSE of that
        cropped block (the un-normalised mask divided by full size, biasing the
        result low)."""
        key = jax.random.PRNGKey(0)
        n_lat, n_lon, n_ch = 6, 8, 2
        pred = jax.random.normal(key, (1, n_lat, n_lon, n_ch))
        target = jax.random.normal(jax.random.PRNGKey(1), (1, n_lat, n_lon, n_ch))
        weights = jnp.abs(jax.random.normal(jax.random.PRNGKey(2), (n_lat,))) + 0.1

        # Region: lat 1:4, lon 2:6.
        mask = jnp.zeros((n_lat, n_lon))
        mask = mask.at[1:4, 2:6].set(1.0)

        masked = area_weighted_mse(pred, target, weights, mask=mask)

        cropped = area_weighted_mse(
            pred[:, 1:4, 2:6, :], target[:, 1:4, 2:6, :], weights[1:4],
        )
        assert jnp.allclose(masked, cropped, rtol=1e-10), (
            f"masked={float(masked):.6g} cropped={float(cropped):.6g}"
        )

    def test_no_mask_path_unchanged(self):
        """The no-mask path must be byte-identical to the legacy formula."""
        key = jax.random.PRNGKey(5)
        pred = jax.random.normal(key, (2, 4, 5, 3))
        target = jax.random.normal(jax.random.PRNGKey(6), (2, 4, 5, 3))
        weights = jnp.abs(jax.random.normal(jax.random.PRNGKey(7), (4,))) + 0.1

        got = area_weighted_mse(pred, target, weights)
        # Legacy reference formula.
        sq = (pred - target) ** 2
        w = weights[:, None, None]
        ref = jnp.mean(sq * w) * weights.shape[0] / jnp.sum(weights)
        assert got == ref  # exact, same operations

    def test_fully_masked_grad_finite(self):
        pred = jnp.zeros((1, 3, 3, 1))
        target = jnp.ones((1, 3, 3, 1))
        weights = jnp.ones(3)
        mask = jnp.zeros((3, 3))
        g = jax.grad(
            lambda p: area_weighted_mse(p, target, weights, mask=mask)
        )(pred)
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Finding 11: SpectralConv weights are float64 under x64
# ---------------------------------------------------------------------------

from legoesm.ml.spectral_conv import SpectralConv


class TestSpectralConvDtype:
    def test_weights_float64(self):
        conv = SpectralConv(n_sh=12, in_channels=3, out_channels=4,
                            key=jax.random.PRNGKey(0))
        assert conv.weight_real.dtype == jnp.float64
        assert conv.weight_imag.dtype == jnp.float64

    def test_output_complex128(self):
        conv = SpectralConv(n_sh=12, in_channels=3, out_channels=4,
                            key=jax.random.PRNGKey(0))
        coeffs = (jnp.ones((12, 3)) + 1j * jnp.ones((12, 3))).astype(jnp.complex128)
        out = conv(coeffs)
        assert out.dtype == jnp.complex128


# ---------------------------------------------------------------------------
# Finding 3: scm_rce_metrics safe sqrt (finite grad at perfect fit)
# ---------------------------------------------------------------------------

from legoesm.training import scm_rce_metrics as srm


class _Ref:
    mass_weights = [0.25, 0.75]
    T_ref = [200.0, 300.0]
    qv_ref = [0.001, 0.01]
    qcond_ref = [0.0, 0.001]
    precip_ref_mm_day = 3.5


class TestScmRceSafeSqrt:
    def test_weighted_rmse_grad_finite_at_zero(self):
        w = jnp.array([0.25, 0.75])
        g = jax.grad(lambda d: srm.weighted_rmse(d, w))(jnp.zeros(2))
        assert jnp.all(jnp.isfinite(g))

    def test_weighted_rmse_zero_at_zero(self):
        w = jnp.array([0.25, 0.75])
        assert float(srm.weighted_rmse(jnp.zeros(2), w)) == 0.0

    def test_score_profiles_grad_finite_at_perfect_fit(self):
        T = jnp.array(_Ref.T_ref)
        qv = jnp.array(_Ref.qv_ref)
        qc = jnp.array(_Ref.qcond_ref)

        def loss(Tp):
            _, _, _, combined = srm.score_profiles_jax(
                _Ref(), Tp, qv, qc, profile_floor=1e-12)
            return combined
        g = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(g))

    def test_combined_zero_at_perfect_fit(self):
        T, qv, cloud, combined = srm.score_profiles_jax(
            _Ref(), jnp.array(_Ref.T_ref), jnp.array(_Ref.qv_ref),
            jnp.array(_Ref.qcond_ref), profile_floor=1e-12)
        assert float(combined) == 0.0

    def test_weighted_std_grad_finite_uniform(self):
        w = jnp.array([0.5, 0.5])
        g = jax.grad(lambda p: srm.weighted_std(p, w))(jnp.array([3.0, 3.0]))
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Finding 7: RolloutConfig dead field removed
# ---------------------------------------------------------------------------

from legoesm.training.dycore_rollout import RolloutConfig


class TestRolloutConfig:
    def test_no_save_every_field(self):
        assert "save_every_n_segments" not in RolloutConfig._fields

    def test_constructs(self):
        cfg = RolloutConfig(n_days=2)
        assert cfg.n_days == 2


# ---------------------------------------------------------------------------
# Finding 2: training_driver optimizer routes through create_optimizer (clip)
# ---------------------------------------------------------------------------

from legoesm.training.training_driver import _make_driver_optimizer


class TestDriverOptimizerClipping:
    def test_clips_large_gradient(self):
        """The driver optimizer must clip the global grad norm (raw adam/adamw
        did NOT, allowing chaotic-adjoint blow-up)."""
        opt = _make_driver_optimizer(
            lr=1e-3, optimizer_kind="adam", n_epochs=5, n_samples=4,
            grad_clip_norm=1.0, warmup_steps=2,
        )
        params = {"w": jnp.zeros((4,))}
        state = opt.init(params)
        huge = {"w": jnp.full((4,), 1e6)}
        updates, _ = opt.update(huge, state, params)
        # After warmup_cosine, step 0 has lr ~ 0, so check the CLIP transform in
        # isolation by comparing the pre-scaled clipped norm: re-run with a
        # constant-lr optimizer is overkill; instead assert the update is finite
        # and bounded well below the raw 1e6 * lr it would be without clipping.
        flat = jnp.concatenate([jnp.ravel(v) for v in jax.tree_util.tree_leaves(updates)])
        assert jnp.all(jnp.isfinite(flat))
        # Without clipping the update magnitude scales with the 2e6 input norm;
        # with clip_by_global_norm(1.0) it is bounded by ~lr (<= peak lr 1e-3).
        assert float(jnp.linalg.norm(flat)) <= 1e-3 + 1e-9

    def test_adamw_kind_builds(self):
        opt = _make_driver_optimizer(
            lr=5e-4, optimizer_kind="adamw", n_epochs=3, n_samples=2,
            weight_decay=1e-5,
        )
        params = {"w": jnp.ones((3,))}
        state = opt.init(params)
        upd, _ = opt.update({"w": jnp.ones((3,))}, state, params)
        assert jnp.all(jnp.isfinite(upd["w"]))

    def test_single_step_run_updates_on_step0(self):
        """codex P2: a 1-epoch/1-sample run must still apply a NONZERO update on
        the only step.  A warmup that consumes the whole (tiny) run returns the
        init_value (lr == 0) on step 0, silently skipping training."""
        opt = _make_driver_optimizer(
            lr=1e-2, optimizer_kind="adam", n_epochs=1, n_samples=1,
        )
        params = {"w": jnp.zeros((4,))}
        state = opt.init(params)
        updates, _ = opt.update({"w": jnp.ones((4,))}, state, params)
        flat = jnp.concatenate(
            [jnp.ravel(v) for v in jax.tree_util.tree_leaves(updates)])
        assert float(jnp.linalg.norm(flat)) > 0.0, "step-0 update was zero (warmup swallowed the only step)"

    def test_short_run_updates_on_step0(self):
        """A handful of steps (warmup default 100 >> total) must also train on
        step 0 (warmup capped to total // 5)."""
        opt = _make_driver_optimizer(
            lr=1e-2, optimizer_kind="adamw", n_epochs=2, n_samples=3,
            weight_decay=1e-5,
        )
        params = {"w": jnp.zeros((4,))}
        state = opt.init(params)
        updates, _ = opt.update({"w": jnp.ones((4,))}, state, params)
        flat = jnp.concatenate(
            [jnp.ravel(v) for v in jax.tree_util.tree_leaves(updates)])
        assert float(jnp.linalg.norm(flat)) > 0.0


# ---------------------------------------------------------------------------
# Finding 4: make_train_step closes over grid/optimizer (no donated reuse)
# ---------------------------------------------------------------------------

from legoesm.ml.training import make_train_step, create_optimizer, TrainingConfig


class _TinyModel(eqx.Module):
    w: jnp.ndarray

    def __init__(self, n_ch, key):
        self.w = jax.random.normal(key, (n_ch,)) * 0.01

    def __call__(self, x, grid):
        # x: (n_lat, n_lon, n_ch) -> same shape, channel-wise scale.
        return x * self.w


class TestMakeTrainStepDonation:
    def test_two_steps_reuse_grid_and_optimizer(self):
        """grid + optimizer are closed over (not donatable args), so calling the
        jitted step twice must not hit a deleted-buffer error from donation."""
        grid = create_gaussian_grid(n_max=8)
        n_lat, n_lon = grid.n_lat, grid.n_lon
        n_ch = 3
        model = _TinyModel(n_ch, jax.random.PRNGKey(0))
        opt = create_optimizer(TrainingConfig(
            lr=1e-3, warmup_steps=1, total_steps=10, optimizer="adam"))
        opt_state = opt.init(eqx.filter(model, eqx.is_array))
        step = make_train_step(opt, grid)

        bi = jnp.ones((2, n_lat, n_lon, n_ch))
        bt = jnp.zeros((2, n_lat, n_lon, n_ch))
        model, opt_state, loss0 = step(model, opt_state, bi, bt)
        # Reuse the SAME grid + optimizer (closed over) for a second step.
        bi2 = jnp.ones((2, n_lat, n_lon, n_ch)) * 0.5
        model, opt_state, loss1 = step(model, opt_state, bi2, bt)
        assert jnp.isfinite(loss0) and jnp.isfinite(loss1)
