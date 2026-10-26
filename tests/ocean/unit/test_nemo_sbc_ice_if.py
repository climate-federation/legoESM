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
    with np.errstate(invalid="ignore"):          # hardware sqrtsd NaN, as NEMO
        root = float(np.sqrt(np.float64(s)))
    return ((-0.0575 + 1.710523e-3 * root) - 2.154996e-4 * s) * s


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


@pytest.mark.parametrize("barriers", [True, False])
def test_truly_eager_equals_jit_bitwise(barriers, monkeypatch):
    # M2: the unbarriered arm is the guard's plant; it must fail.
    import legoesm.ocean.eos as eos
    if not barriers:
        monkeypatch.setattr(bfo, "R", lambda v: v)
        monkeypatch.setattr(eos, "nemo_source_round", lambda v: v)
    args = _inputs(seed=7)
    with jax.disable_jit():
        eager = nemo_sbc_ice_if(*args, nn_hls=HLS)
    assert (_unequal(eager, _jit()(*args)) == 0) == barriers


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


def _special_grid(subnormals=True):
    """Every (ts, S) pair from a special-value set, one per wet interior cell.

    XLA's CPU backend flushes subnormals to zero; the Fortran build (no
    fast-math) does not, so the scalar reference is compared without them.
    """
    tiny = 5e-324 if subnormals else 1e-300
    sal = [0.0, -0.0, tiny, 1e-300, 34.0, 40.0, -1.0, np.inf, np.nan]
    pairs = []
    for s_v in sal:
        zt = float(nemo_eos_fzp_eos80(s_v))
        tsv = [0.0, -0.0, tiny, -tiny, 1.0, -1.0, -2.5, np.inf, -np.inf,
               np.nan, zt]
        if np.isfinite(zt):
            tsv += [np.nextafter(zt, np.inf), np.nextafter(zt, -np.inf)]
        pairs += [(t_v, s_v) for t_v in tsv if subnormals
                  or not 0.0 < abs(t_v) < np.finfo(np.float64).tiny]
    n = len(pairs)
    ny, nx = 3, n + 2
    sss = np.full((ny, nx), 34.0)
    ts = np.full((ny, nx), 5.0)
    for k, (t_v, s_v) in enumerate(pairs):
        ts[1, k + 1], sss[1, k + 1] = t_v, s_v
    tmask = np.ones((ny, nx))
    sst = np.full((ny, nx), -10.0)          # fr_i = 0 and 1 both appear
    sst[1, 1::2] = 10.0
    ff = np.full((ny, nx), 1e-4)
    ni = (1, n)
    cover = np.full(ni, 0.7)
    qsr = np.full(ni, 200.0)
    qns = np.full(ni, -50.0)
    return cover, sss, sst, ts, ff, tmask, np.ones(ni), qsr, qns


def test_min_tie_cannot_reach_output_checked_proof(monkeypatch):
    # M3 proof.  MINSD(0., x) and jnp.minimum differ only for x = -0 or NaN.
    # x = MAX(ts, zt) - zt with MAXSD keeping zt on ties, so x = -0 needs
    # MAX(...) = -0 and zt = +0, but MAXSD(-0, +0) returns +0: unreachable.
    # NaN x means NaN zt, and then zqri is NaN too; the output NaN is the
    # same word either way.  Checked here on every special-value pair.
    args = _special_grid()
    seen = []
    real_fmin = bfo._fmin

    def recording_fmin(first, second):
        seen.append(np.asarray(first))
        return real_fmin(first, second)

    monkeypatch.setattr(bfo, "_fmin", recording_fmin)
    with jax.disable_jit():
        literal = nemo_sbc_ice_if(*args, nn_hls=HLS)
    (x,) = seen
    assert not np.any((x == 0.0) & np.signbit(x))           # no -0 operand
    assert np.any((x == 0.0) & ~np.signbit(x))              # +0 ties occur
    assert np.any(np.isnan(x))                              # NaN path covered
    monkeypatch.setattr(bfo, "_fmin", jnp.minimum)
    with jax.disable_jit():
        planted = nemo_sbc_ice_if(*args, nn_hls=HLS)
    assert _unequal(literal, planted) == 0
    normal = _special_grid(subnormals=False)
    with jax.disable_jit():
        got = nemo_sbc_ice_if(*normal, nn_hls=HLS)
    assert _unequal(got, _scalar(*normal)) == 0


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


# --- fld_read time blend (fldread.F90:244-246) and the nn_ice = 1 branch ---

from legoesm.ocean.bulk_flux_omip import (  # noqa: E402
    NemoSbcIceIfInputs, nemo_blk_oce_ncar,
)
from legoesm.ocean.forcing.nemo_fld_read import nemo_fld_time_interp  # noqa: E402

_YEAR_S = 365 * 86400


def _tint_scalar(fb, fa, isec, nb, na):
    w = float(isec - nb) / float(na - nb)
    return (1.0 - w) * fb + w * fa


def test_fld_time_interp_matches_scalar_and_moves_a_yearly_field():
    rng = np.random.default_rng(12)
    f = rng.random(200_003)
    nb, na = -_YEAR_S // 2, _YEAR_S // 2      # yearly record mid-times
    blend = jax.jit(nemo_fld_time_interp)
    for isec in (12_345_678, 3 * 86400 + 5400):
        got = np.asarray(blend(f, f, isec, nb, na))
        want = np.array([_tint_scalar(v, v, isec, nb, na) for v in f])
        assert np.array_equal(_bits(got), _bits(want))
        assert np.count_nonzero(_bits(got) != _bits(f)) > 0   # fnow != F
    got = np.asarray(blend(f, f, 0, nb, na))                  # w = 0.5
    assert np.array_equal(_bits(got), _bits(f))
    two = np.asarray(blend(np.zeros(1), np.ones(1), 0, nb, na))
    assert float(two[0]) == 0.5


def _blk_args(shape, seed=3):
    rng = np.random.default_rng(seed)
    def field(lo, hi):
        return jnp.asarray(rng.uniform(lo, hi, shape))
    return (field(-8, 8), field(-8, 8), field(260, 300), field(1e-3, 1e-2),
            field(99000, 102000), field(-1.9, 25), field(-0.3, 0.3),
            field(-0.3, 0.3), field(0, 400), field(250, 400), field(0, 1e-4),
            field(0, 2e-5))


def _ice_case():
    cover, sss, sst, ts, ff, tmask, smask0, qsr, qns = _inputs()
    rng = np.random.default_rng(9)
    after = cover.copy()
    before = np.clip(cover + rng.uniform(-0.1, 0.1, cover.shape), 0.0, 1.0)
    ice = NemoSbcIceIfInputs(before, after, 12_345_678, -_YEAR_S // 2,
                             _YEAR_S // 2, sss, sst, ts, ff, tmask, smask0)
    blk = _blk_args(smask0.shape) + (jnp.asarray(smask0),)
    return blk, ice


def _blk(blk, **kw):
    return nemo_blk_oce_ncar(*blk, 0.066, 1.0, 1.0, 5, **kw)


def test_nn_ice_1_production_call_site_runs_fld_read_and_sbc_ice_if():
    # M4: removing either call from the nn_ice = 1 branch fails this test.
    blk, ice = _ice_case()
    run = jax.jit(lambda b, i: _blk(b, nn_ice=1, ice_if=i))
    got = run(blk, ice)
    base = jax.jit(lambda b: _blk(b, nn_ice=0, ice_if=None))(blk)
    cover = nemo_fld_time_interp(*ice[:5])
    want = jax.jit(partial(nemo_sbc_ice_if, nn_hls=HLS))(
        cover, *ice[5:], base.qsr, base.qns)
    assert _unequal((got.ts_surface, got.fr_i, got.qsr, got.qns), want) == 0
    assert _unequal((got.qns,), (base.qns,)) > 0
    assert _unequal((got.qsr,), (base.qsr,)) > 0
    for name in base._fields[:-2]:
        if name not in ("qsr", "qns"):
            assert _unequal((getattr(got, name),), (getattr(base, name),)) == 0
    assert base.ts_surface is None and base.fr_i is None
    # the cover is the time blend, not either raw slot
    stale = jax.jit(partial(nemo_sbc_ice_if, nn_hls=HLS))(
        ice.fdta_after, *ice[5:], base.qsr, base.qns)
    assert _unequal((got.qns,), (stale.qns,)) > 0


def test_nn_ice_dispatch_refuses():
    blk, ice = _ice_case()
    with pytest.raises(ValueError, match="nn_ice"):
        _blk(blk, nn_ice=2, ice_if=None)
    with pytest.raises(ValueError, match="ice_if"):
        _blk(blk, nn_ice=1, ice_if=None)
    with pytest.raises(ValueError, match="ice_if"):
        _blk(blk, nn_ice=0, ice_if=ice)
    with pytest.raises(ValueError, match="A2D"):
        _blk(blk, nn_ice=1, ice_if=ice._replace(ts_surface=ice.ts_surface[1:]))
