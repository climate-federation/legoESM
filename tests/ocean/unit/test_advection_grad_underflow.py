"""Float32 gradient-underflow safety of the limiter/FCT ratio guards.

The 2026-06-11 ocean differentiability audit found that the eps-floored
ratio guards in the advection family (Zalesak FCT cell ratios, TVD/DST3
slope ratios, thickness divisions) NaN-poison float32 AD — reverse mode
for ``ppm_fct`` (ALL gradients NaN on near-uniform tracers) and forward
mode for ``tvd``/``superbee``/``dst3`` — because the division derivative
forms ``1/den**2``, which underflows to zero for den ~ 1e-30..1e-20 and
yields ``inf * 0 = NaN``. Float32 is the finite-volume default compute,
and near-uniform tracers (uniform initial salinity, mixed layers) are the
ordinary ocean state, so this hit any gradient consumer out of the box.

Fix under test: ``legoesm.core.flux_limiters.grad_safe_ratio`` /
``ratio_grad_floor`` applied at every site. These tests pin:

1. the RAW unguarded pattern still NaNs on the degenerate data
   (non-vacuity: removing the fix from a site is the mutant these
   data would catch),
2. ``grad_safe_ratio`` is primal-bit-identical to plain division and
   gates both operands' derivatives,
3. ``fct_tracer_advection`` reverse+forward gradients are finite on
   near-uniform float32 data,
4. the 2-step lat-lon model rollout with each affected scheme has
   finite reverse AND forward (eager + jit) directional derivatives in
   float32 — the exact configuration that failed before the fix.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.flux_limiters import grad_safe_ratio, ratio_grad_floor

# NOTE: deliberately NOT enabling x64 — the bug class is float32-specific.


def _degenerate_ratio_data(dtype=jnp.float32):
    """(Q, inc) with the corner mix that poisons the unguarded backward:
    inc exactly zero with Q > 0 (huge floored ratio) and tiny nonzero inc.
    """
    Q = jnp.array([0.0, 1e-3, 2.0, 1e-3, 0.0, 5e-4], dtype=dtype)
    inc = jnp.array([0.0, 0.0, 0.0, 1e-25, 1e-25, 3e-7], dtype=dtype)
    return Q, inc


class TestRawPatternNonVacuity:
    """The unguarded formula NaNs on the degenerate data (mutant pin)."""

    def test_raw_zalesak_ratio_reverse_nan_f32(self):
        Q, inc = _degenerate_ratio_data()

        def raw(q, i):
            return jnp.sum(jnp.minimum(1.0, q / jnp.maximum(i, 1e-30)))

        g_inc = jax.grad(raw, argnums=1)(Q, inc)
        assert bool(jnp.any(jnp.isnan(g_inc))), (
            "expected the UNGUARDED ratio backward to NaN on the degenerate "
            "float32 data — if this no longer reproduces, the non-vacuity of "
            "the grad_safe_ratio tests is broken and they must be recalibrated"
        )


class TestGradSafeRatio:
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_primal_bit_identical(self, dtype):
        if dtype == jnp.float64:
            pytest.importorskip("jax")
            if not jax.config.jax_enable_x64:
                pytest.skip("x64 disabled in this process (f32-only module)")
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        num = jax.random.normal(k1, (64,), dtype=dtype)
        den_raw = jax.random.normal(k2, (64,), dtype=dtype)
        # mix in floored cells exactly as call sites produce them
        den_raw = den_raw.at[:8].set(0.0)
        eps = jnp.asarray(1e-30, dtype)
        den = jnp.where(jnp.abs(den_raw) > eps, den_raw, eps)
        ok = jnp.abs(den_raw) > ratio_grad_floor(dtype)
        out = grad_safe_ratio(num, den, ok)
        ref = num / den
        assert np.array_equal(np.asarray(out), np.asarray(ref)), (
            "grad_safe_ratio must be primal-bit-identical to num/den"
        )

    def test_gates_both_operand_gradients(self):
        Q, inc = _degenerate_ratio_data()
        eps = jnp.float32(1e-30)
        floor = ratio_grad_floor(jnp.float32)
        ok = inc > floor

        def safe(q, i):
            return jnp.sum(jnp.minimum(
                1.0, grad_safe_ratio(q, jnp.maximum(i, eps), i > floor)))

        gq, gi = jax.grad(safe, argnums=(0, 1))(Q, inc)
        assert bool(jnp.all(jnp.isfinite(gq))) and bool(jnp.all(jnp.isfinite(gi)))
        ok_np = np.asarray(ok)
        assert np.all(np.asarray(gq)[~ok_np] == 0.0), "gated num grads must be 0"
        assert np.all(np.asarray(gi)[~ok_np] == 0.0), "gated den grads must be 0"
        # the ungated, in-range cell keeps a live, correct gradient
        i_live = int(np.argwhere(ok_np)[0][0])
        q, i = float(Q[i_live]), float(inc[i_live])
        if q / i < 1.0:  # min() passes gradient
            assert np.isclose(float(gq[i_live]), 1.0 / i, rtol=1e-5)

    def test_jvp_finite_and_matches_vjp(self):
        Q, inc = _degenerate_ratio_data()
        eps = jnp.float32(1e-30)
        floor = ratio_grad_floor(jnp.float32)

        def safe(i):
            return jnp.sum(jnp.minimum(
                1.0, grad_safe_ratio(Q, jnp.maximum(i, eps), i > floor)))

        d = jnp.ones_like(inc)
        _, fwd = jax.jvp(safe, (inc,), (d,))
        rev = jnp.vdot(jax.grad(safe)(inc), d)
        assert bool(jnp.isfinite(fwd))
        assert np.isclose(float(fwd), float(rev), rtol=1e-5)


def _near_uniform_fct_inputs():
    """Operator-level near-uniform float32 inputs in the audit's NaN regime:
    a tracer that is exactly uniform over patches (zero anti-diffusive
    fluxes -> inc == 0 with nonzero budgets) and noise-level elsewhere.
    """
    from legoesm.grids.latlon import create_latlon_grid

    grid = create_latlon_grid(12, 24)
    key = jax.random.PRNGKey(3)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tr = 35.0 + 1e-3 * jax.random.normal(k1, (12, 24, 5), dtype=jnp.float32)
    tr = tr.at[:, 6:18, :].set(jnp.float32(35.0))   # exactly uniform patch
    tr = tr.at[..., 2:].set(tr[..., 1:2])           # vertically uniform below
    mfu = 30.0 * jax.random.normal(k2, (12, 25, 5), dtype=jnp.float32)
    mfv = 30.0 * jax.random.normal(k3, (13, 24, 5), dtype=jnp.float32)
    w = 1e-4 * jax.random.normal(k4, (12, 24, 6), dtype=jnp.float32)
    hk = jnp.full((12, 24, 5), 50.0, dtype=jnp.float32)
    return grid, tr, mfu, mfv, w, hk


class TestFCTOperatorGradFinite:
    def test_fct_reverse_and_forward_finite_f32(self):
        from legoesm.ocean.advection import fct_tracer_advection

        grid, tr, mfu, mfv, w, hk = _near_uniform_fct_inputs()

        def loss(t):
            dh, dv = fct_tracer_advection(t, mfu, mfv, w, hk, grid, 600.0)
            return jnp.sum((dh + dv) ** 2)

        g = jax.grad(loss)(tr)
        assert int(jnp.sum(~jnp.isfinite(g))) == 0, "reverse grad not finite"
        d = jnp.ones_like(tr) / tr.size
        _, fwd = jax.jvp(loss, (tr,), (d,))
        assert bool(jnp.isfinite(fwd)), "forward directional derivative NaN"
        fwd_j = jax.jit(lambda t: jax.jvp(loss, (t,), (d,))[1])(tr)
        assert bool(jnp.isfinite(fwd_j)), "jitted forward derivative NaN"


class TestMPASTvdEdges:
    """MPAS sibling of the lat-lon slope-ratio sites (codex-review finding:
    tvd_tracer_to_edges carried the identical unguarded pattern, live on
    the production MPAS/tripole OMIP dispatch)."""

    def test_tvd_tracer_to_edges_jvp_finite_f32(self):
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.ocean.dynamics.advection_mpas import (
            compute_upup_cells,
            tvd_tracer_to_edges,
        )

        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
        upup_pos, upup_neg = compute_upup_cells(mesh)
        nlev = 3
        key = jax.random.PRNGKey(11)
        # near-uniform tracer (the NaN regime) + an exactly-uniform level
        tr = 35.0 + 1e-3 * jax.random.normal(
            key, (mesh.nCells, nlev), dtype=jnp.float32)
        tr = tr.at[:, 1].set(jnp.float32(35.0))
        mf = 30.0 * jax.random.normal(
            jax.random.PRNGKey(12), (mesh.nEdges, nlev), dtype=jnp.float32)

        def loss(t):
            return jnp.sum(tvd_tracer_to_edges(t, mf, mesh,
                                               upup_pos, upup_neg) ** 2)

        g = jax.grad(loss)(tr)
        assert int(jnp.sum(~jnp.isfinite(g))) == 0
        d = jnp.ones_like(tr) / tr.size
        fwd_j = jax.jit(lambda t: jax.jvp(loss, (t,), (d,))[1])(tr)
        assert bool(jnp.isfinite(fwd_j)), "MPAS tvd jitted jvp NaN"


@pytest.mark.parametrize("scheme", ["ppm_fct", "tvd", "superbee", "dst3"])
def test_model_rollout_grads_finite_f32(scheme):
    """2-step lat-lon rollout, float32 (default compute): reverse grad and
    eager+jit forward directional derivatives all finite. Pre-fix: ppm_fct
    reverse was ALL-NaN and tvd/superbee/dst3 forward was NaN (jit).
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(12, 24)
    z_coord = create_ocean_z_star(n_levels=5, H_max=1000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=1000.0, land_lat_threshold=85.0)
    key = jax.random.PRNGKey(0)
    ku, kv = jax.random.split(key)
    lat = jnp.linspace(-1.0, 1.0, 12)[:, None, None]
    lon = jnp.linspace(-1.0, 1.0, 24)[None, :, None]
    lev = jnp.linspace(0.0, 1.0, 5)[None, None, :]
    blob = 1.5 * jnp.exp(-((lat / 0.5) ** 2 + (lon / 0.5) ** 2) - 3.0 * lev)
    state = state._replace(
        T=state.T.replace(data=state.T.data + blob),
        u=state.u.replace(data=state.u.data + 0.03 * jax.random.normal(
            ku, state.u.data.shape)),
        v=state.v.replace(data=state.v.data + 0.03 * jax.random.normal(
            kv, state.v.data.shape)),
    )
    config = LatLonCGridOceanConfig.from_flat(
        A_h=1000.0, K_h=100.0, A_v=1e-3, K_v=1e-4, bottom_drag_r=1e-3,
        tracer_advection=scheme, n_barotropic_substeps=4,
        differentiable_barotropic=True, enable_runtime_checks=False)
    model = LatLonCGridOceanModel(grid, z_coord, config)
    wet3 = (state.land_mask.data[..., None]
            * jnp.ones((1, 1, 5))).astype(state.T.data.dtype)

    def loss(dT0):
        s = state._replace(T=state.T.replace(data=state.T.data + dT0))
        for _ in range(2):
            s = model._step_impl(s, 600.0)
        return (jnp.sum((s.T.data * wet3) ** 2)
                + jnp.sum((s.S.data * wet3) ** 2)
                + 1e4 * (jnp.sum(s.u.data ** 2) + jnp.sum(s.v.data ** 2)))

    zero = jnp.zeros_like(state.T.data)
    val, g = jax.jit(jax.value_and_grad(loss))(zero)
    assert bool(jnp.isfinite(val))
    assert int(jnp.sum(~jnp.isfinite(g))) == 0, (
        f"{scheme}: non-finite reverse gradients "
        f"(NaN={int(jnp.sum(jnp.isnan(g)))})")
    d = (jax.random.normal(jax.random.PRNGKey(42), zero.shape)
         * wet3).astype(zero.dtype)
    d = d / jnp.linalg.norm(d)
    rev = float(jnp.vdot(g, d))
    try:
        _, fwd = jax.jvp(loss, (zero,), (d,))
    except TypeError as exc:
        pytest.fail(
            f"{scheme}: reverse directional derivative={rev:.9e}; "
            f"forward derivative unavailable: {exc}")
    assert bool(jnp.isfinite(fwd)), f"{scheme}: eager forward derivative NaN"
    fwd_j = jax.jit(lambda z: jax.jvp(loss, (z,), (d,))[1])(zero)
    assert bool(jnp.isfinite(fwd_j)), f"{scheme}: jitted forward derivative NaN"
    # reverse and forward agree to float32 noise
    assert np.isclose(rev, float(fwd_j), rtol=1e-3), (
        f"{scheme}: rev={rev:.6e} vs fwd={float(fwd_j):.6e}")
