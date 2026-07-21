"""Truth-tier gates for flux-form mass-conserving cube tracer transport (#771).

The advective form -(u·∇q) does not conserve column water under divergent
flow.  The flux-form step co-transports δp and δp·q with the validated FV3
operator and recovers q = (δp·q)★/δp★, which must be:

1. MASS CONSERVING  — ∑ area·δp·q exactly preserved under a divergent wind.
2. FREE-STREAM PRESERVING — a uniform tracer stays uniform.
3. MONOTONE / POSITIVE — a positive blob never goes negative.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.fv3_sw_core import d2a2c_vect
from legoesm.atmosphere.dynamics.shared.flux_form_tracer_transport import (
    flux_form_tracer_step,
)

jax.config.update("jax_enable_x64", True)


def _divergent_winds(cdgrid, n, nlev):
    """A genuinely DIVERGENT contravariant wind, tiled over ``nlev``.

    Build staggered D-grid winds (u_d at ``(6, n, n+1)``, v_d at
    ``(6, n+1, n)``) whose components vary across the cell so the field carries
    real divergence, then map D-grid → C-grid contravariant with the same
    ``d2a2c_vect`` the mass solver uses.  The winds need not be physical — only
    divergent and correctly staggered — to stress the transport.
    """
    jx = jnp.arange(n + 1) / n            # normalized edge coordinate
    # u varies along x and v along y ⟹ ∇·u ≠ 0.
    u_d = 25.0 * (jnp.sin(2.0 * jnp.pi * jx))[None, None, :] \
        * jnp.ones((6, n, n + 1))
    v_d = 18.0 * (jnp.cos(2.0 * jnp.pi * jx))[None, :, None] \
        * jnp.ones((6, n + 1, n))
    _ua, _va, _uc, _vc, ut2d, vt2d = d2a2c_vect(
        u_d.astype(jnp.float64), v_d.astype(jnp.float64), cdgrid)
    ut = jnp.broadcast_to(ut2d[..., None], (*ut2d.shape, nlev))
    vt = jnp.broadcast_to(vt2d[..., None], (*vt2d.shape, nlev))
    return ut.astype(jnp.float64), vt.astype(jnp.float64)


def _setup(n=12, nlev=4):
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    ut, vt = _divergent_winds(cdgrid, n, nlev)
    # Layer mass: a non-uniform positive δp (varies with face + level so the
    # test is not accidentally trivial).
    lat_c = grid.lat  # (6, n, n)
    base = 900.0 + 50.0 * jnp.cos(lat_c)  # (6, n, n)
    delp = jnp.stack(
        [base * (1.0 + 0.1 * k) for k in range(nlev)], axis=-1
    ).astype(jnp.float64)
    area = cdgrid.base.area
    return grid, cdgrid, ut, vt, delp, area


def _column_water(delp, q, area):
    # ∑ area·δp·q over all faces/cells/levels for one tracer.
    return float(jnp.sum(
        area[..., None] * delp * q, dtype=jnp.float64))


def test_transport_step_4d_matches_vmap():
    """#811: ``transport_step_4d`` (ONE ``pad_halo_4d`` per exchange for all
    levels) is BIT-IDENTICAL to the per-level ``vmap(transport_step(
    mass_target=None))`` on single-rank — a halo is a pure index gather (no
    reduction), so the 4D exchange introduces no fp-associativity change.  This
    is the anchor that certifies the 4D refactor did not perturb the numerics;
    the MPI 1-vs-2-rank test then certifies the single-message correctness."""
    from legoesm.core.fv_tp_2d import transport_step, transport_step_4d
    _grid, cdgrid, ut, vt, delp, _area = _setup(n=12, nlev=5)

    out_4d = transport_step_4d(delp, ut, vt, 600.0, cdgrid, hord=8)
    out_vmap = jax.vmap(
        lambda h, u, v: transport_step(
            h, u, v, 600.0, cdgrid, mass_target=None, hord=8),
        in_axes=(-1, -1, -1), out_axes=-1)(delp, ut, vt)

    assert out_4d.shape == delp.shape
    rel = float(jnp.max(jnp.abs(out_4d - out_vmap))
                / (jnp.max(jnp.abs(out_vmap)) + 1e-30))
    assert rel < 1e-13, (
        f"transport_step_4d not bit-identical to vmap(transport_step): "
        f"rel={rel:.3e}")


def test_d2a2c_vect_4d_matches_vmap():
    """#811: ``d2a2c_vect_4d`` (ONE ``pad_halo_vector_4d`` for all levels) is
    BIT-IDENTICAL to per-level ``vmap(d2a2c_vect)`` on single-rank — the A→C tail
    numerics are SHARED verbatim (the ``global_fields=`` fast path), only the
    vector wind halo is batched into one message."""
    from legoesm.core.fv3_sw_core import d2a2c_vect, d2a2c_vect_4d
    n, nlev = 12, 5
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    jx = jnp.arange(n + 1) / n
    ud = jnp.stack(
        [(20.0 + 3.0 * k) * jnp.sin(2 * jnp.pi * jx)[None, None, :]
         * jnp.ones((6, n, n + 1)) for k in range(nlev)], axis=-1
    ).astype(jnp.float64)                                  # (6, n, n+1, nlev)
    vd = jnp.stack(
        [(15.0 + 2.0 * k) * jnp.cos(2 * jnp.pi * jx)[None, :, None]
         * jnp.ones((6, n + 1, n)) for k in range(nlev)], axis=-1
    ).astype(jnp.float64)                                  # (6, n+1, n, nlev)

    out_4d = d2a2c_vect_4d(ud, vd, cdgrid)
    out_vmap = jax.vmap(
        lambda u, v: d2a2c_vect(u, v, cdgrid),
        in_axes=(-1, -1), out_axes=-1)(ud, vd)
    for a, b, name in zip(out_4d, out_vmap,
                          ("ua", "va", "uc", "vc", "ut", "vt")):
        rel = float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))
        assert rel < 1e-13, f"d2a2c_vect_4d {name} not bit-identical: rel={rel:.3e}"


def test_mass_conservation_under_divergent_wind():
    grid, cdgrid, ut, vt, delp, area = _setup()
    n = grid.lat.shape[1]
    # A localized moisture blob on face 0 (positive, non-uniform).
    q = jnp.zeros((6, n, n, delp.shape[-1], 1), dtype=jnp.float64)
    xx, yy = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
    blob = jnp.exp(-((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n / 4) ** 2)
    q = q.at[0, :, :, :, 0].set(0.02 * blob[:, :, None])

    m0 = _column_water(delp, q[..., 0], area)
    assert m0 > 0.0
    q_new, delp_new = flux_form_tracer_step(q, delp, ut, vt, 600.0, cdgrid)
    # advection must have ACTED (the blob moved / spread), not a no-op.
    assert float(jnp.max(jnp.abs(q_new[..., 0] - q[..., 0]))) > 1e-6
    # column water conserved on the co-transported δ p★ grid, to ~fp accuracy.
    m1 = _column_water(delp_new, q_new[..., 0], area)
    assert abs(m1 - m0) / m0 < 1e-11, f"Δ(∫δp·q)/m0 = {(m1 - m0) / m0:.3e}"


def test_free_stream_preservation():
    grid, cdgrid, ut, vt, delp, area = _setup()
    n = grid.lat.shape[1]
    # Uniform tracer q ≡ 0.5 everywhere.
    q = jnp.full((6, n, n, delp.shape[-1], 1), 0.5, dtype=jnp.float64)
    q_new, _ = flux_form_tracer_step(q, delp, ut, vt, 600.0, cdgrid)
    # A uniform tracer stays uniform under a divergent wind to FP ROUNDOFF (the
    # co-transport of δp makes transport_step's homogeneity cancel the tendency;
    # not bit-exact because the per-field mass targets are summed separately) —
    # the property the advective form + a separate δp evolution wholly lacks.
    assert float(jnp.max(jnp.abs(q_new - 0.5))) < 1e-9


def test_monotone_no_spurious_negatives():
    grid, cdgrid, ut, vt, delp, area = _setup()
    n = grid.lat.shape[1]
    # A sharp positive front (step) — dispersive schemes undershoot to <0 here.
    q = jnp.zeros((6, n, n, delp.shape[-1], 1), dtype=jnp.float64)
    q = q.at[0, : n // 2, :, :, 0].set(0.03)
    q_new, _ = flux_form_tracer_step(q, delp, ut, vt, 900.0, cdgrid)
    assert float(jnp.min(q_new)) > -1e-12, \
        f"spurious negative q = {float(jnp.min(q_new)):.3e}"


def test_multi_tracer_batched_matches_single():
    """Batched (n_tracers>1) transport equals per-tracer transport — the
    tracer-axis vmap must not cross-contaminate."""
    grid, cdgrid, ut, vt, delp, area = _setup(n=12, nlev=3)
    n = grid.lat.shape[1]
    rng = np.random.default_rng(0)
    q2 = jnp.asarray(0.01 * rng.random((6, n, n, delp.shape[-1], 2)))
    q_both, _ = flux_form_tracer_step(q2, delp, ut, vt, 600.0, cdgrid)
    q_a, _ = flux_form_tracer_step(q2[..., :1], delp, ut, vt, 600.0, cdgrid)
    q_b, _ = flux_form_tracer_step(q2[..., 1:], delp, ut, vt, 600.0, cdgrid)
    np.testing.assert_allclose(np.asarray(q_both[..., 0]),
                               np.asarray(q_a[..., 0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(q_both[..., 1]),
                               np.asarray(q_b[..., 0]), rtol=1e-12, atol=1e-12)


def test_differentiable():
    """End-to-end jax.grad must produce finite gradients (the moisture path is
    inside the training segment; transport_step is co-transported twice)."""
    grid, cdgrid, ut, vt, delp, area = _setup(n=12, nlev=3)
    n = grid.lat.shape[1]
    q = jnp.full((6, n, n, delp.shape[-1], 1), 0.01, dtype=jnp.float64)

    def loss(q_in):
        q_out, _ = flux_form_tracer_step(q_in, delp, ut, vt, 600.0, cdgrid)
        return jnp.sum(q_out ** 2)

    g = jax.grad(loss)(q)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0     # non-trivial gradient


def test_shape_validation():
    _, cdgrid, ut, vt, delp, _ = _setup()
    with pytest.raises(ValueError, match="6, n, n, nlev, n_tracers"):
        flux_form_tracer_step(delp, delp, ut, vt, 600.0, cdgrid)  # 4-D q
