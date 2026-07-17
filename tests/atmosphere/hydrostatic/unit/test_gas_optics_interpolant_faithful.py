"""Faithfulness pins for the RRTMGP gas-optics interpolation WEIGHTS.

Target: the self-contained interpolant helpers in
``legoesm.atmosphere.physics.radiation.rrtmgp.optics.gas_optics`` —
``_pressure_interpolant`` and ``_mixing_fraction_interpolant``.  These build the
even-grid linear interpolants that ``compute_major_optical_depth`` /
``compute_minor_optical_depth`` use to look up the k-distribution tables in
pressure and binary-species mixing fraction.

Most-trustful source
--------------------
rte-rrtmgp / RRTMGP (Pincus et al. 2019): the pressure interpolation is LINEAR IN
ln(p) on the log-spaced reference-pressure grid (a linear-in-p interpolation would
be a faithfulness bug), and the mixing-fraction interpolation is linear on an even
[0, 1] grid of ``n_mixing_fraction`` points.

Why this file (scope):
``test_rrtmgp_stratosphere.py`` already pins ``_clip_to_table_range`` exactly and
exercises the FULL optical-depth path behaviourally against the real netCDF tables
(finiteness, saturation, AD-safety, energy conservation), but it does NOT pin the
interpolation WEIGHTS.  The relative-abundance (eta) interpolant is
lookup-dependent (needs the gas-optics table) and is out of scope here; the
pressure and mixing-fraction interpolants are self-contained closed forms and are
pinned exactly below.  The
helpers are private, but white-box tests may reach into a module's internals (the
private-import ratchet excludes tests, and the sibling stratosphere test already
imports these symbols).

Certification (test-only):
1. Pressure interpolant is LINEAR IN ln(p): exact interp weights vs an independent
   log-space oracle; the geometric-mean-of-two-nodes canary gives weight 0.5
   (a linear-in-p implementation would not), and on-node gives weight 0/1.
2. The troposphere offset shifts the returned indices (not the weights).
3. Mixing-fraction interpolant is linear on the even [0, 1] grid: exact weights,
   on-node, and differentiability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import gas_optics

jax.config.update("jax_enable_x64", True)


def _a(x):
    return jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))


def _f(x):
    return float(np.asarray(x).reshape(-1)[0])


def _even_grid_interp_oracle(f, f_ref):
    """Independent even-grid linear interpolant: (idx_low, idx_high, w_low, w_high)."""
    f_ref = np.asarray(f_ref)
    delta = f_ref[1] - f_ref[0]
    n = f_ref.shape[0]
    j = int(np.clip(np.floor((f - f_ref[0]) / delta), 0, n - 1))
    j2 = min(j + 1, n - 1)
    lower = f_ref[0] + delta * j
    w_high = abs((f - lower) / delta)
    return j, j2, 1.0 - w_high, w_high


# Log-spaced reference pressure grid: even in ln(p), DESCENDING to match the real
# RRTMGP p_ref ordering (p_ref[-1] ~ 1 Pa) -> negative log-delta, so an
# implementation that assumed a positive log spacing would fail these pins.
_P_REF = jnp.asarray(np.exp(np.linspace(np.log(110000.0), np.log(1.0), 6)))  # Pa, descending


# ---------------------------------------------------------------------------
# 1. Pressure interpolant — linear in ln(p).
# ---------------------------------------------------------------------------
def test_pressure_interpolant_linear_in_log_p_exact_weights():
    # A pressure strictly between two log-nodes: the interpolation weight must be
    # the LOG-space fractional distance, matched to an independent oracle.
    p = _a(5000.0)
    interp = gas_optics._pressure_interpolant(p, _P_REF)
    lp = np.log(5000.0)
    j, j2, w_lo, w_hi = _even_grid_interp_oracle(lp, np.log(np.asarray(_P_REF)))
    assert _f(interp.interp_low.idx) == j
    assert _f(interp.interp_high.idx) == j2
    np.testing.assert_allclose(_f(interp.interp_low.weight), w_lo, rtol=1e-12)
    np.testing.assert_allclose(_f(interp.interp_high.weight), w_hi, rtol=1e-12)


def test_pressure_interpolant_geometric_mean_is_half_weight():
    # DEPARTURE canary (log vs linear): at the GEOMETRIC mean of two adjacent
    # nodes, ln(p) is exactly halfway -> weight 0.5.  A linear-in-p interpolant
    # would give (sqrt(p0*p1) - p0)/(p1 - p0) != 0.5.
    pr = np.asarray(_P_REF)
    p0, p1 = pr[2], pr[3]                       # two adjacent nodes
    p_geo = float(np.sqrt(p0 * p1))
    interp = gas_optics._pressure_interpolant(_a(p_geo), _P_REF)
    np.testing.assert_allclose(_f(interp.interp_high.weight), 0.5, rtol=1e-12)
    np.testing.assert_allclose(_f(interp.interp_low.weight), 0.5, rtol=1e-12)
    # confirm the linear-in-p weight is genuinely different (so the pin discriminates)
    lin_w = (p_geo - float(p0)) / (float(p1) - float(p0))
    assert abs(lin_w - 0.5) > 0.05


def test_pressure_interpolant_on_node():
    # p exactly on a reference node: all interpolation mass sits on that node.
    # (Which endpoint the floor index lands on is fp-dependent for an exactly-on-
    # node value, so pin the representation-INVARIANT facts: the reconstructed
    # ln(p) equals the node, and the two weights are {0, 1}.)
    lpr = np.log(np.asarray(_P_REF))
    p_node = float(np.asarray(_P_REF)[2])
    interp = gas_optics._pressure_interpolant(_a(p_node), _P_REF)
    il, ih = int(_f(interp.interp_low.idx)), int(_f(interp.interp_high.idx))
    recon = (_f(interp.interp_low.weight) * lpr[il]
             + _f(interp.interp_high.weight) * lpr[ih])
    np.testing.assert_allclose(recon, np.log(p_node), rtol=1e-12)
    w = sorted([_f(interp.interp_low.weight), _f(interp.interp_high.weight)])
    np.testing.assert_allclose(w, [0.0, 1.0], atol=1e-9)


def test_pressure_interpolant_beyond_table_edge_clamps_index():
    # p well below the smallest reference pressure (p_ref[-1] ~ 1 Pa): the floor
    # index must CLAMP to the last valid index (5), not run past the table.  With
    # n=6, an unclamped floor would give index 6 (out of bounds); the clamp keeps
    # both endpoints at 5.  (The stratosphere suite exercises p < p_ref[-1].)
    interp = gas_optics._pressure_interpolant(_a(0.01), _P_REF)   # 0.01 Pa << 1 Pa
    assert _f(interp.interp_low.idx) == 5
    assert _f(interp.interp_high.idx) == 5


def test_pressure_interpolant_troposphere_offset_shifts_index():
    # The troposphere offset is added to the indices AFTER the weights are formed:
    # weights identical, indices shifted by the offset.
    p = _a(5000.0)
    base = gas_optics._pressure_interpolant(p, _P_REF)
    off = gas_optics._pressure_interpolant(p, _P_REF, troposphere_offset=jnp.int32(1))
    assert _f(off.interp_low.idx) == _f(base.interp_low.idx) + 1
    assert _f(off.interp_high.idx) == _f(base.interp_high.idx) + 1
    np.testing.assert_allclose(_f(off.interp_high.weight),
                               _f(base.interp_high.weight), rtol=1e-12)
    np.testing.assert_allclose(_f(off.interp_low.weight),
                               _f(base.interp_low.weight), rtol=1e-12)


def test_pressure_interpolant_differentiable():
    # Assert a NONZERO derivative that matches a finite difference -- a
    # stop_gradient on the weight would pass a finiteness-only check but fail here.
    def loss(p):
        interp = gas_optics._pressure_interpolant(p, _P_REF)
        return jnp.sum(interp.interp_high.weight)
    p0 = 5000.0
    g = _f(jax.grad(loss)(_a(p0)))
    fd = (_f(loss(_a(p0 + 1.0))) - _f(loss(_a(p0 - 1.0)))) / 2.0
    assert np.isfinite(g) and abs(g) > 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Mixing-fraction interpolant — linear on the even [0, 1] grid.
# ---------------------------------------------------------------------------
def test_mixing_fraction_interpolant_exact_weights():
    # n=5 -> grid [0, .25, .5, .75, 1], delta 0.25.  eta=0.4 lies in [.25,.5] with
    # w_high = (0.4-0.25)/0.25 = 0.6.
    n = 5
    eta = _a(0.4)
    interp = gas_optics._mixing_fraction_interpolant(eta, n)
    j, j2, w_lo, w_hi = _even_grid_interp_oracle(0.4, np.linspace(0.0, 1.0, n))
    assert _f(interp.interp_low.idx) == j == 1
    assert _f(interp.interp_high.idx) == j2 == 2
    np.testing.assert_allclose(_f(interp.interp_high.weight), 0.6, rtol=1e-12)
    np.testing.assert_allclose(_f(interp.interp_low.weight), 0.4, rtol=1e-12)
    np.testing.assert_allclose(_f(interp.interp_high.weight), w_hi, rtol=1e-12)
    np.testing.assert_allclose(_f(interp.interp_low.weight), w_lo, rtol=1e-12)


def test_mixing_fraction_interpolant_on_node():
    # eta exactly on a grid node (0.5, index 2 for n=5) -> weight fully on it.
    interp = gas_optics._mixing_fraction_interpolant(_a(0.5), 5)
    np.testing.assert_allclose(_f(interp.interp_high.weight), 0.0, atol=1e-12)
    np.testing.assert_allclose(_f(interp.interp_low.weight), 1.0, rtol=1e-12)
    assert _f(interp.interp_low.idx) == 2


def test_mixing_fraction_interpolant_endpoints():
    # eta = 0 -> node 0 (all low weight); eta = 1 -> BOTH endpoints clamp to the
    # last index 4 (an unclamped idx_high would be 5, out of bounds).
    lo = gas_optics._mixing_fraction_interpolant(_a(0.0), 5)
    hi = gas_optics._mixing_fraction_interpolant(_a(1.0), 5)
    assert _f(lo.interp_low.idx) == 0
    np.testing.assert_allclose(_f(lo.interp_low.weight), 1.0, rtol=1e-12)
    assert _f(hi.interp_low.idx) == 4          # clamped to last node
    assert _f(hi.interp_high.idx) == 4         # idx_high clamp (would be 5 unclamped)
    np.testing.assert_allclose(_f(hi.interp_high.weight), 0.0, atol=1e-12)


def test_mixing_fraction_interpolant_differentiable():
    # Nonzero, finite-difference-matched derivative (rejects stop_gradient): the
    # weight w_high = (eta - node)/delta has d/d(eta) = 1/delta = (n-1) = 4 here.
    def loss(eta):
        interp = gas_optics._mixing_fraction_interpolant(eta, 5)
        return jnp.sum(interp.interp_high.weight)
    g = _f(jax.grad(loss)(_a(0.4)))
    assert np.isfinite(g) and abs(g) > 0.0
    np.testing.assert_allclose(g, 4.0, rtol=1e-9)      # 1/delta, delta = 1/(5-1)
