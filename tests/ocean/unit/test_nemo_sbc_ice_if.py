"""NEMO ``sbc_ice_if`` (nn_ice = 1) literal transcription — round 338 / S3.

The reference is an independent pure-Python scalar transcription of the
record build's ppsrc (sbcice_if.f90:154-196, eosbn2.f90:1867-1868), one IEEE
rounding per source operation, with the tie order read from the object code:
MAXSD keeps ``zt_fzp``; MINSD in ``MIN(0., x)`` keeps the literal ``0.``;
``SIGN`` is lib_fortran's ``b >= 0`` form.  Plants show each pin can fail.
"""

from __future__ import annotations

import inspect
import math
from functools import partial

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import legoesm.ocean.bulk_flux_omip as bfo  # noqa: E402
from legoesm.ocean.bulk_flux_omip import nemo_sbc_ice_if  # noqa: E402
from legoesm.ocean.eos import nemo_eos_fzp, nemo_eos_fzp_eos80  # noqa: E402

HLS = 1


def _fzp(s):
    return ((-0.0575 + 1.710523e-3 * math.sqrt(s)) - 2.154996e-4 * s) * s


def _scalar(cover, sss, sst, ts, ff, tmask, smask0, qsr, qns, *,
            aliased=True):
    ny, nx = sss.shape
    fzp = np.empty_like(sss)
    ts_new = np.empty_like(ts)
    for j in range(ny):
        for i in range(nx):
            fzp[j, i] = _fzp(float(sss[j, i])) * float(tmask[j, i])
            a, b = float(ts[j, i]), float(fzp[j, i])
            ts_new[j, i] = a if a > b else b
    fr = fzp.copy()
    qsr_o = np.empty_like(qsr)
    qns_o = np.empty_like(qns)
    ts_read = ts_new if aliased else ts
    for j in range(ny - 2 * HLS):
        for i in range(nx - 2 * HLS):
            J, I = j + HLS, i + HLS
            zt = float(fzp[J, I])
            zfr = float(cover[j, i])
            f = 1.0 if float(sst[J, I]) <= zt else 0.0
            fr[J, I] = f
            qsr_o[j, i] = (1.0 - zfr) * float(qsr[j, i])
            tb = float(ts_read[J, I])
            zqri = -40.0 * (tb - (zt - 1.0))
            x = tb - zt
            zqrj = -40.0 * (x if x < 0.0 else 0.0)
            zqrp = (zfr * ((1.0 - f) * zqri + f * zqrj)) * float(smask0[j, i])
            zqi = -3.0 + (1.0 if float(ff[J, I]) >= 0.0 else -1.0)
            qns_o[j, i] = (((1.0 - zfr) * float(qns[j, i]) + zfr * f * zqi)
                           * float(smask0[j, i]) + zqrp)
    return ts_new, fr, qsr_o, qns_o


def _inputs(seed=338, ny=24, nx=30):
    rng = np.random.default_rng(seed)
    sss = rng.uniform(0.0, 40.0, (ny, nx))
    tmask = (rng.random((ny, nx)) > 0.2).astype(np.float64)
    sss[tmask == 0.0] = 0.0
    sss[0, :5] = -0.0                      # land -0 salinity
    tmask[0, :5] = 0.0
    sss[1, 1], tmask[1, 1] = -0.0, 1.0     # wet -0 salinity -> fzp = +0
    fz = np.array([[_fzp(s) for s in row] for row in sss]) * tmask
    ts = fz + rng.uniform(-1.5, 3.0, (ny, nx))
    ts[tmask == 0.0] = 0.0                 # +0 land T vs -0 fzp: MAXSD tie
    ts[2, 2:9] = fz[2, 2:9]                # exact wet ties
    ts[1, 1] = -0.0                        # -0 vs +0 tie
    ts[3, 3] = np.nan                      # MAXSD returns the second operand
    sst = fz + rng.uniform(-0.5, 0.5, (ny, nx))
    sst[4, 2:9] = fz[4, 2:9]               # sst_m == zt_fzp -> fr_i = 1
    ff = rng.uniform(-1.4e-4, 1.4e-4, (ny, nx))
    ff[5, 1:5] = [0.0, -0.0, 0.0, -0.0]
    ff[6, 1:5] = -0.0
    tmask[6, 1:5] = 1.0                    # -0 Coriolis under model ice
    sss[6, 1:5] = 34.0
    fz = np.array([[_fzp(s) for s in row] for row in sss]) * tmask
    ts[6, 1:5] = fz[6, 1:5] + 0.25
    sst[6, 1:5] = fz[6, 1:5] - 0.1
    ni = (ny - 2 * HLS, nx - 2 * HLS)
    cover = rng.random(ni)
    cover[0, :6] = [0.0, 1.0, -0.0, 1.0, 0.0, 1.0]
    cover[5:7, :] = 1.0
    smask0 = tmask[HLS:-HLS, HLS:-HLS].copy()
    assert int(((ff[1:-1, 1:-1] == 0.0) & np.signbit(ff[1:-1, 1:-1])
                & (sst[1:-1, 1:-1] <= fz[1:-1, 1:-1]) & (cover > 0.0)
                & (smask0 == 1.0)).sum()) >= 4
    qsr = rng.uniform(0.0, 400.0, ni) * smask0
    qns = rng.uniform(-300.0, 100.0, ni) * smask0
    return cover, sss, sst, ts, ff, tmask, smask0, qsr, qns


def _bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def _unequal(got, want):
    return sum(int(np.count_nonzero(_bits(g) != _bits(w)))
               for g, w in zip(got, want))


def _jit():
    return jax.jit(partial(nemo_sbc_ice_if, nn_hls=HLS))


def test_jit_matches_independent_scalar_transcription_bitwise():
    args = _inputs()
    want = _scalar(*args)
    got = _jit()(*args)
    assert _unequal(got, want) == 0
    ts_new = np.asarray(got.ts_surface)
    sss, tmask = args[1], args[5]
    # land S = +0: fzp = -0.0575*0 = -0, so MAXSD(+0 T, -0) keeps -0
    land = np.argwhere((tmask == 0.0) & (sss == 0.0) & ~np.signbit(sss))[0]
    assert np.signbit(ts_new[tuple(land)]) and ts_new[tuple(land)] == 0.0
    # land S = -0: fzp = +0, the tie keeps +0
    assert not np.signbit(ts_new[0, 0])
    assert ts_new[3, 3] == _fzp(sss[3, 3]) * tmask[3, 3]  # NaN T -> zt_fzp
    # halo keeps the freezing point, interior holds the 0/1 fraction
    assert np.array_equal(_bits(np.asarray(got.fr_i)[0, :]),
                          _bits(want[1][0, :]))
    assert set(np.unique(np.asarray(got.fr_i)[1:-1, 1:-1])) <= {0.0, 1.0}


def test_truly_eager_equals_jit_bitwise():
    args = _inputs(seed=7)
    with jax.disable_jit():
        eager = nemo_sbc_ice_if(*args, nn_hls=HLS)
    assert _unequal(eager, _jit()(*args)) == 0


def test_min_branch_is_identically_minus_zero_when_kbb_is_kmm():
    # After the clamp ts >= zt_fzp, so MIN(0., ts - zt) is always the literal
    # 0. and ztrp * 0. = -0: the under-ice damping term never acts.
    args = _inputs(seed=11)
    got = _jit()(*args)
    sss, tmask = args[1], args[5]
    fz = np.array([[_fzp(s) for s in row] for row in sss]) * tmask
    x = np.asarray(got.ts_surface)[1:-1, 1:-1] - fz[1:-1, 1:-1]
    finite = np.isfinite(x)
    assert np.all(x[finite] >= 0.0) and not np.any(np.signbit(x[finite]))


@pytest.mark.parametrize("plant", ["max_tie", "sign", "fzp_teos10",
                                   "unaliased"])
def test_plants_break_the_bitwise_pin(plant, monkeypatch):
    args = _inputs()
    want = _scalar(*args)
    if plant == "max_tie":
        monkeypatch.setattr(bfo, "_fmax", jnp.maximum)
    elif plant == "sign":
        monkeypatch.setattr(bfo, "_sign", jnp.copysign)
    elif plant == "fzp_teos10":
        monkeypatch.setattr(bfo, "nemo_eos_fzp_eos80", nemo_eos_fzp)
    else:
        want = _scalar(*args, aliased=False)
    assert _unequal(_jit()(*args), want) > 0


def test_min_tie_plant_is_unobservable_here(monkeypatch):
    # The MINSD tie order cannot change bits under Kbb = Kmm (see above).
    args = _inputs()
    monkeypatch.setattr(bfo, "_fmin", jnp.minimum)
    assert _unequal(_jit()(*args), _scalar(*args)) == 0


def test_eos80_literal_matches_scalar_and_recipe_copy():
    s = np.concatenate([np.linspace(0.0, 42.0, 4001), [-0.0, 35.0, 34.5]])
    got = np.asarray(jax.jit(nemo_eos_fzp_eos80)(s))
    assert np.array_equal(_bits(got), _bits([_fzp(v) for v in s]))
    recipe = (-0.0575 + 1.710523e-3 * np.sqrt(s) - 2.154996e-4 * s) * s
    assert np.array_equal(_bits(got), _bits(recipe))
    assert np.isnan(float(nemo_eos_fzp_eos80(-1.0)))
    g = jax.grad(lambda v: nemo_eos_fzp_eos80(v))(0.0)
    assert float(g) == -0.0575


def test_no_defaults_for_physical_inputs():
    sig = inspect.signature(nemo_sbc_ice_if)
    assert all(p.default is inspect.Parameter.empty
               for p in sig.parameters.values())


def _fd(fun, x, eps):
    return (fun(x + eps) - fun(x - eps)) / (2.0 * eps)


def test_fd_gradients_open_ice_and_clamp():
    cover, sss, sst, ts, ff, tmask, smask0, qsr, qns = _inputs(seed=5)
    J, I = 10, 12
    tmask[J, I], smask0[J - 1, I - 1] = 1.0, 1.0
    sss[J, I] = 34.0
    cover[J - 1, I - 1] = 0.6
    fz = _fzp(34.0)
    sst[J, I] = fz + 0.3                    # fr_i = 0 (observed ice, no opa ice)
    f = jax.jit(partial(nemo_sbc_ice_if, nn_hls=HLS))

    def qns_of(field, value, index):
        def g(v):
            a = [cover, sss, sst, ts, ff, tmask, smask0, qsr, qns]
            a = [jnp.asarray(x) for x in a]
            a[field] = a[field].at[index].set(v)
            return f(*a).qns[J - 1, I - 1]
        return g

    # open, unclamped surface: dqns/dts = zfr * (1 - fr_i) * ztrp * smask0
    for tsv in (fz + 0.8,):
        g = qns_of(3, tsv, (J, I))
        ad = float(jax.grad(g)(tsv))
        assert ad == pytest.approx(0.6 * -40.0, rel=1e-12)
        assert _fd(g, tsv, 1e-6) == pytest.approx(ad, rel=1e-7)
    # clamped surface (ts below freezing): no gradient to ts, all to S
    g = qns_of(3, fz - 0.5, (J, I))
    assert float(jax.grad(g)(fz - 0.5)) == 0.0
    assert _fd(g, fz - 0.5, 1e-6) == 0.0
    # at the tie the gradient goes to the freezing point (second operand)
    assert float(jax.grad(g)(fz)) == 0.0
    # clamped: zqri = ztrp*(zt - (zt - 1)) is constant, so qns is flat in S
    # and the S sensitivity goes to the clamped temperature instead
    ts[J, I] = fz - 0.5
    gs = qns_of(1, 34.0, (J, I))
    assert float(jax.grad(gs)(34.0)) == 0.0
    assert _fd(gs, 34.0, 1e-6) == 0.0

    def ts_of_s(v):
        a = [jnp.asarray(x) for x in (cover, sss, sst, ts, ff, tmask,
                                      smask0, qsr, qns)]
        a[1] = a[1].at[J, I].set(v)
        return f(*a).ts_surface[J, I]
    adt = float(jax.grad(ts_of_s)(34.0))
    assert adt == pytest.approx(
        -0.0575 + 1.5 * 1.710523e-3 * math.sqrt(34.0) - 2 * 2.154996e-4 * 34.0,
        rel=1e-12)
    assert _fd(ts_of_s, 34.0, 1e-6) == pytest.approx(adt, rel=1e-7)
    # fr_i switch is piecewise constant: no gradient through sst_m
    gm = qns_of(2, float(sst[J, I]), (J, I))
    assert float(jax.grad(gm)(float(sst[J, I]))) == 0.0
    # every input gradient is finite, including land cells (S = 0)
    def total(*a):
        r = f(*a)
        return jnp.nansum(r.qns) + jnp.nansum(r.qsr) + jnp.nansum(
            jnp.where(jnp.isfinite(r.ts_surface), r.ts_surface, 0.0))
    a = [cover, sss, sst, ts, ff, tmask, smask0, qsr, qns]
    a[3] = np.where(np.isnan(a[3]), 0.0, a[3])
    grads = jax.grad(total, argnums=tuple(range(9)))(*map(jnp.asarray, a))
    assert all(np.all(np.isfinite(np.asarray(gr))) for gr in grads)


def test_barrier_plant_breaks_the_pin(monkeypatch):
    # M1: without the per-statement barriers XLA rewrites the arithmetic.
    import legoesm.ocean.eos as eos
    args = _inputs(seed=1, ny=150, nx=182)
    want = _scalar(*args)
    monkeypatch.setattr(bfo, "R", lambda v: v)
    monkeypatch.setattr(eos, "nemo_source_round", lambda v: v)
    assert _unequal(_jit()(*args), want) > 0
