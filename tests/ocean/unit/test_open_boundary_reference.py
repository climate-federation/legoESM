"""Width 0 of the smoothed open-boundary radiation is the reference bitwise.

The reference is the colleague's original (pre-smoothing) CROCO / MOM6 radiation,
copied verbatim below from the ENSO fork (commit 463c11fa35, open_boundary_latlon.py)
as the test oracle.
"""
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

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
    z = jnp.zeros((2, 8))
    for f in (lambda x: ob._orlanski(x, z, z, z, z, 0.05, 0.001, width=0.01).sum(),
              lambda x: ob._orlanski_1d(x, z, z, z, z, 0.05, 0.001, jnp.asarray(1.0), width=0.01).sum()):
        g = jax.grad(f)(z)
        assert np.all(np.isfinite(np.asarray(g)))
