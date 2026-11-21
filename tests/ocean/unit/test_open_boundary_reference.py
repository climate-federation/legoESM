"""Width 0 of the smoothed open-boundary radiation is the reference bitwise.

The reference is the colleague's original (pre-smoothing) CROCO / MOM6 radiation,
copied verbatim below from the ENSO fork (commit 463c11fa35, open_boundary_latlon.py)
as the test oracle.
"""
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.ocean.dynamics import open_boundary_latlon as ob  # noqa: E402

_EPS = ob._EPS


# --- reference, verbatim -------------------------------------------------------------------------
def _ref_orlanski(fb_old, f1_old, f1_new, f2_new, data, r_in, r_out, mb=None, m1=None, m2=None):
    """CROCO t3dbc.F / u3dbc.F adaptive oblique radiation with nudging, on
    arrays (..., along) ordered boundary, first, second interior; r_in,
    r_out = dt / tau (dimensionless).  mb, m1, m2 (..., along): wet masks of
    the boundary row and the first and second interior rows.  As CROCO
    multiplies its gradients by the land mask: the tangential gradients are
    zero across a dry neighbour, the normal gradients are zero where the first
    or second interior cell is dry (no radiation: the boundary value is kept
    and nudged with the outflow time scale), and the result is zero on dry
    points.  None: no masking."""
    def tgrad(f, m):
        gb = f - jnp.concatenate([f[..., :1], f[..., :-1]], axis=-1)
        if m is not None:
            gb = gb * m * jnp.concatenate([m[..., :1], m[..., :-1]], axis=-1)
        return jnp.concatenate([gb, gb[..., -1:]], axis=-1)
    gb, g1 = tgrad(fb_old, mb), tgrad(f1_old, m1)
    dft = f1_old - f1_new
    dfx = f1_new - f2_new
    if m1 is not None:          # no radiation from dry interior cells (their values are not water)
        radiate = m1 if m2 is None else m1 * m2
        dft, dfx = dft * radiate, dfx * radiate
    inflow = dfx * dft < 0.0
    dft = jnp.where(inflow, 0.0, dft)
    tau = jnp.where(inflow, r_in, r_out)
    up = dft * (g1[..., :-1] + g1[..., 1:]) > 0.0
    dfy = jnp.where(up, g1[..., :-1], g1[..., 1:])
    cff = jnp.maximum(dfx * dfx + dfy * dfy, _EPS)
    cx = dft * dfx
    cy = jnp.clip(dft * dfy, -cff, cff)
    fb = (cff * fb_old + cx * f1_new - jnp.maximum(cy, 0.0) * gb[..., :-1]
          - jnp.minimum(cy, 0.0) * gb[..., 1:]) / (cff + cx)
    out = (1.0 - tau) * fb + tau * data
    return out if mb is None else out * mb


def _ref_orlanski_1d(fb_new, f1_old, f1_new, f2_new, data, r_in, r_out, rx_max):
    """MOM6 1-D Orlanski radiation with nudging (radiation_open_bdry_conds,
    OBC_RAD_VEL_WT = 1), on arrays (..., along): the boundary value after the
    model step, the first interior old and new, the second interior new.  The
    phase speed rx = min((f1_old - f1_new) / (f1_new - f2_new), rx_max) when
    outward, 0 otherwise; boundary = (fb_new + rx f1_new) / (1 + rx), then
    nudged with weight dt / (tau + dt) (MOM6 gamma_2), tau_in on inflow."""
    dhdt = f1_old - f1_new
    dhdx = f1_new - f2_new
    out = dhdt * dhdx > 0.0
    rx = jnp.where(out, jnp.minimum(dhdt / jnp.where(out, dhdx, 1.0), rx_max), 0.0)
    fb = (fb_new + rx * f1_new) / (1.0 + rx)
    r = jnp.where(out, r_out, r_in)
    g = r / (1.0 + r)
    return (1.0 - g) * fb + g * data


# --------------------------------------------------------------------------------------------------


def _rows(seed, n=5, shape=(4, 32)):
    rng = np.random.default_rng(seed)
    return [jnp.asarray(rng.standard_normal(shape)) for _ in range(n)]


def test_orlanski_width_zero_is_the_reference_bitwise():
    for seed in range(4):
        r = _rows(seed)
        masks = [jnp.asarray((np.random.default_rng(10 + seed + k).random((4, 32)) > 0.2).astype(float))
                 for k in range(3)]
        for rin, rout in ((0.05, 0.001), (0.05, 1e-20)):
            np.testing.assert_array_equal(ob._orlanski(*r, rin, rout, width=0.0),
                                          _ref_orlanski(*r, rin, rout))
            np.testing.assert_array_equal(ob._orlanski(*r, rin, rout, *masks, width=0.0),
                                          _ref_orlanski(*r, rin, rout, *masks))


def test_orlanski_1d_width_zero_is_the_reference_bitwise():
    for seed in range(4):
        r = _rows(seed)
        for rin, rout in ((0.05, 0.001), (0.05, 1e-20)):
            np.testing.assert_array_equal(ob._orlanski_1d(*r, rin, rout, jnp.asarray(1.0), width=0.0),
                                          _ref_orlanski_1d(*r, rin, rout, jnp.asarray(1.0)))


def test_default_width_gradient_is_finite_on_a_flat_interior():
    """Every operand, float64 and float32: a flat column must not turn the gradient NaN."""
    for dt in (jnp.float64, jnp.float32):
        z = jnp.zeros((2, 8), dt)
        args = (z, z, z, z, z)
        for argnum in range(5):
            for f in (lambda *x: ob._orlanski(*x, 0.05, 0.001, width=0.01).sum(),
                      lambda *x: ob._orlanski_1d(*x, 0.05, 0.001, jnp.asarray(1.0, dt), width=0.01).sum()):
                g = jax.grad(f, argnums=argnum)(*args)
                assert np.all(np.isfinite(np.asarray(g))), (dt, argnum)


def test_default_width_gradient_is_finite_in_band_near_rest():
    """Constant rows of size 1e-9 (fluid near rest) sit inside the inflow/outflow band; float32
    gradients through every operand stay finite (no y**-2 in the division derivatives)."""
    for dt in (jnp.float32, jnp.float64):
        x = 1e-9
        args = tuple(jnp.full((2, 8), v, dt) for v in (x, 1.005 * x, x, 0.0, 0.0))
        for argnum in range(5):
            for f in (lambda *a: ob._orlanski(*a, 0.05, 0.001, width=0.01).sum(),
                      lambda *a: ob._orlanski_1d(*a, 0.05, 0.001, jnp.asarray(1.0, dt), width=0.01).sum()):
                g = jax.grad(f, argnums=argnum)(*args)
                assert np.all(np.isfinite(np.asarray(g))), (dt, argnum)


def test_flather_gradient_is_finite_at_dry_points():
    z = jnp.zeros(6, jnp.float32)
    g = jax.grad(lambda e: ob._characteristic(z, z, e, e, e, z, z, jnp.ones(6, jnp.float32) * 5e4, 900.0,
                                              constants.g, z, z, 1.0).sum())(z)
    assert np.all(np.isfinite(np.asarray(g)))


def test_default_width_makes_no_new_extremes():
    """The CROCO update is a convex combination of the old boundary row (cell and its two
    tangential neighbours), the new first interior value and the data; the smoothed
    coefficients keep it so (a sign-flipped split overshoots a boundary plateau)."""
    for seed in range(6):
        fb, f1o, f1n, f2n, d = (np.array(x) for x in _rows(seed))
        fb[:, 10:20] = 1.0                                  # a plateau on the boundary row
        out = np.asarray(ob._orlanski(*(jnp.asarray(x) for x in (fb, f1o, f1n, f2n, d)),
                                      0.05, 0.001, width=0.01))
        pad = np.concatenate([fb[:, :1], fb, fb[:, -1:]], axis=1)
        nb = np.stack([pad[:, :-2], pad[:, 1:-1], pad[:, 2:], f1n, d])
        assert np.all(out >= nb.min(0) - 1e-12) and np.all(out <= nb.max(0) + 1e-12), seed


def test_floored_divisions_have_finite_gradients_on_dry_faces():
    for dt in (jnp.float32, jnp.float64):
        z = jnp.zeros((3, 4), dt)
        g = jax.grad(lambda h: ob._over_floor(h, h.sum(axis=0, keepdims=True)).sum())(z)
        assert np.all(np.isfinite(np.asarray(g)))


def test_hard_width_gradient_is_finite_on_a_flat_interior():
    z = jnp.zeros((2, 8), jnp.float32)
    for argnum in range(5):
        g = jax.grad(lambda *x: ob._orlanski(*x, 0.05, 0.001, width=0.0).sum(), argnums=argnum)(z, z, z, z, z)
        assert np.all(np.isfinite(np.asarray(g))), argnum


def test_flat_point_keeps_the_boundary_value_derivative():
    z = jnp.zeros((2, 8), jnp.float32)
    g = jax.grad(lambda fb: ob._orlanski(fb, z, z, z, z, 0.05, 0.001, width=0.0).sum())(z)
    np.testing.assert_allclose(np.asarray(g), 1.0 - 0.001, rtol=1e-6)      # (1 - r_out) d fb_old
