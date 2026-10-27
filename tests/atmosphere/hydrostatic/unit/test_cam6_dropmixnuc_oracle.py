"""CAM6 dropmixnuc port vs the gfortran oracle (verbatim ndrop.F90 minus history I/O).

Cases: ``scripts/validate/cam6_ndrop_dropmixnuc_oracle_cases.txt`` (3 columns x 12
levels, top_lev = 2, dtmicro 600 s and 112.5 s; growing, shrinking, partly-ice and
surface-based clouds; diffusivities above the 100 m^2/s cap).  Expected
``tendnd``: ``..._expected.txt``.  Tolerance fp64 rtol 1e-12 / atol 1e-30.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.microphysics.cam6_dropmixnuc import (  # noqa: E402
    cam6_dropmixnuc, cam6_microp_aero_inputs, cam6_microp_aero_npccn, nsubmix_upper_bound,
)
from legoesm.atmosphere.physics.microphysics.cam6_modal_aerosol import (  # noqa: E402
    MAM4_SPECIES, Cam6ModeProps, voltonumb,
)

_DIR = Path(__file__).resolve().parents[4] / "scripts" / "validate"
NL, NC = 12, 3


def _parse():
    rows = [np.array([float(x) for x in l.split()])
            for l in (_DIR / "cam6_ndrop_dropmixnuc_oracle_cases.txt").read_text().splitlines()]
    it = iter(rows)
    consts = next(it)
    top_lev = int(next(it)[0])
    sig, lo, hi = next(it), next(it), next(it)
    dens, hyg = [], []
    for _ in range(4):
        dens.append(next(it)); hyg.append(next(it))
    props = Cam6ModeProps(sig, lo, hi, voltonumb(lo, sig), voltonumb(hi, sig), tuple(dens), tuple(hyg))
    ncase = int(next(it)[0])
    cases = []
    for _ in range(ncase):
        dt = float(next(it)[0])
        cols = []
        for _c in range(NC):
            f = [next(it) for _ in range(11)]
            num, mass = [], []
            for sp in MAM4_SPECIES:
                num.append(next(it))
                mass.append(np.stack([next(it) for _ in sp]))
            cols.append((f, num, mass))
        cases.append((dt, cols))
    return consts, top_lev, props, cases


def _arrays(cols):
    st = lambda j: jnp.asarray(np.stack([c[0][j] for c in cols]))
    T, pmid, pdel, zm, pint, nc, wsub, cldn, cldo, liq, kvh = (st(j) for j in range(11))
    aer_num = jnp.asarray(np.stack([np.stack([c[1][m] for c in cols]) for m in range(4)]))
    aer_mass = tuple(jnp.asarray(np.stack([c[2][m] for c in cols], axis=1)) for m in range(4))
    return T, pmid, pdel, zm, pint, nc, wsub, cldn, cldo, liq, kvh, aer_num, aer_mass


def test_oracle_constants_are_ours():
    consts, *_ = _parse()
    ours = [np.pi, constants.rho_water, constants.M_water, constants.R_universal, constants.R_v,
            constants.g, constants.L_v, constants.c_pd, constants.epsilon, constants.R_d,
            constants.T_freeze]
    assert list(consts) == [float(c) for c in ours]


@pytest.mark.parametrize("k", [0, 1])
def test_matches_gfortran_oracle(k):
    _, top_lev, props, cases = _parse()
    dt, cols = cases[k]
    T, pmid, pdel, zm, pint, nc, wsub, cldn, cldo, liq, kvh, aer_num, aer_mass = _arrays(cols)
    n_max = nsubmix_upper_bound(T, pmid, pint, 1.0 / pdel, zm, dt, top_lev - 1)
    got = cam6_dropmixnuc(nc, T, pmid, pint, 1.0 / pdel, zm, kvh, wsub, cldn, cldo, liq, dt,
                          aer_num, aer_mass, props, top_lev - 1, n_max)
    exp = np.array([[float(x) for x in l.split()]
                    for l in (_DIR / "cam6_ndrop_dropmixnuc_oracle_expected.txt").read_text().splitlines()
                    if l.strip()])[3 * k:3 * k + 3]
    np.testing.assert_allclose(np.asarray(got)[:, top_lev - 1:], exp, rtol=1e-12, atol=1e-30)
    np.testing.assert_array_equal(np.asarray(got)[:, :top_lev - 1], 0.0)


def test_raises_beyond_static_substep_bound_and_grad_finite():
    """A column needing more mixing sub-steps than the bound raises, eager and
    under jit (user decision 2026-10-26: a clear error, not a silent NaN)."""
    _, top_lev, props, cases = _parse()
    dt, cols = cases[0]
    T, pmid, pdel, zm, pint, nc, wsub, cldn, cldo, liq, kvh, aer_num, aer_mass = _arrays(cols)
    args = (T, pmid, pint, 1.0 / pdel, zm, kvh, wsub, cldn, cldo, liq, dt, aer_num, aer_mass, props,
            top_lev - 1)
    small = lambda x: cam6_dropmixnuc(x, *args, 1)     # bound too small for these columns
    for call in (small, jax.jit(small)):
        with pytest.raises(Exception, match="mixing sub-steps than the bound n_max=1"):
            np.asarray(call(nc))
    n_max = nsubmix_upper_bound(T, pmid, pint, 1.0 / pdel, zm, dt, top_lev - 1)
    g = jax.grad(lambda x: jnp.sum(cam6_dropmixnuc(x, *args, n_max)))(nc)
    assert np.all(np.isfinite(np.asarray(g)))
    f = lambda x: cam6_dropmixnuc(x, *args, n_max)
    np.testing.assert_allclose(np.asarray(jax.jit(f)(nc)), np.asarray(f(nc)), rtol=1e-13, atol=1e-30)


# --- S9a: microp_aero wsub / liquid fraction (cam6_microp_aero_oracle.F90) -----------------

def _mpa():
    rows = [np.array([float(x) for x in l.split()]) for l in
            (_DIR / "cam6_microp_aero_oracle_cases.txt").read_text().splitlines() if l.strip()]
    top_lev = int(rows[0][0])
    wp2, ql, qi = (np.stack(rows[1 + j::3]) for j in range(3))
    exp = np.array([[float(x) for x in l.split()] for l in
                    (_DIR / "cam6_microp_aero_oracle_expected.txt").read_text().splitlines()
                    if l.strip()])
    return top_lev, jnp.asarray(wp2), jnp.asarray(ql), jnp.asarray(qi), exp[0::2], exp[1::2]


def test_microp_aero_inputs_match_gfortran_oracle():
    top_lev, wp2, ql, qi, wsub_f, liq_f = _mpa()
    wsub, liq = cam6_microp_aero_inputs(ql, qi, wp2, top_lev - 1)
    np.testing.assert_allclose(np.asarray(wsub), wsub_f, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(liq), liq_f, rtol=1e-12, atol=0.0)
    # the cases reach the cap, the floor, the interior, qcld == qsmall and both pure phases
    assert (wsub_f == 10.0).any() and (wsub_f[:, top_lev - 1:] == 0.20).any()
    assert ((wsub_f > 0.20) & (wsub_f < 10.0)).any()
    assert ((liq_f == 0.0) & (np.asarray(ql) > 0)).any() and (liq_f == 1.0).any()


def _composition_args():
    _, top_lev, props, cases = _parse()
    dt, cols = cases[0]
    T, pmid, pdel, zm, pint, nc, _w, cldn, cldo, _l, kvh, aer_num, aer_mass = _arrays(cols)
    n_max = nsubmix_upper_bound(T, pmid, pint, 1.0 / pdel, zm, dt, top_lev - 1)
    return (nc, T, pmid, pint, 1.0 / pdel, zm), kvh, cldn, cldo, dt, (aer_num, aer_mass, props,
                                                                      top_lev - 1, n_max)


def test_microp_aero_npccn_is_dropmixnuc_on_the_oracle_inputs():
    top_lev, wp2, ql, qi, wsub_f, liq_f = _mpa()
    st, kvh, cldn, cldo, dt, tail = _composition_args()
    assert tail[3] == top_lev - 1
    got = cam6_microp_aero_npccn(*st, ql, qi, wp2, kvh, cldn, cldo, dt, *tail)
    ref = cam6_dropmixnuc(*st, kvh, jnp.asarray(wsub_f), cldn, cldo, jnp.asarray(liq_f), dt, *tail)
    wsub, liq = cam6_microp_aero_inputs(ql, qi, wp2, top_lev - 1)
    if (np.array_equal(np.asarray(wsub), wsub_f) and np.array_equal(np.asarray(liq), liq_f)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    else:
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-30)
    assert np.isfinite(np.asarray(got)).all() and np.abs(np.asarray(got)).max() > 0.0


def test_microp_aero_npccn_gradient():
    top_lev, wp2, ql, qi, wsub_f, _ = _mpa()
    st, kvh, cldn, cldo, dt, tail = _composition_args()
    f = lambda w: jnp.sum(cam6_microp_aero_npccn(*st, ql, qi, w, kvh, cldn, cldo, dt, *tail))
    g = np.asarray(jax.grad(f)(wp2))
    assert np.isfinite(g).all()
    assert np.abs(g).max() > 0.0, "npccn does not depend on wp2 (detached)"
    assert np.asarray(wp2)[0, 5] == 0.0 and np.asarray(wp2)[0, 6] == 0.0   # tke = 0 interfaces
    gw = np.asarray(jax.grad(lambda w: jnp.sum(cam6_microp_aero_inputs(ql, qi, w, top_lev - 1)[0]))(wp2))
    inner = (wsub_f > 0.20) & (wsub_f < 10.0)
    assert inner.any() and (np.abs(gw[:, :-1]) + np.abs(gw[:, 1:]))[inner].min() > 0.0
