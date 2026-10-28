"""NEMO-literal zdf_iwm (zdfiwm.f90:233-317) vs a scalar transcription."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
    IWMConfig,
    IWMForcing,
    nemo_zdf_iwm,
)

CFG = IWMConfig(enabled=True, form="nemo_literal", mevar=False, tsdiff=True,
                nu_molecular=1.4e-6, k_min=1.4e-7, k_max=1.0e-2,
                require_forcing_maps=True)
RHO0 = 1026.0
NLEV = 30


@pytest.fixture(autouse=True)
def _libm():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _fortran_column(avt, avs, avm, f, rn2, s, ht0, gd, e3, wm, tsdiff=True):
    """One column of zdfiwm.f90:233-317 in Python floats, source order."""
    ecri, ebot, ensq, esho, hbot, hcri = f
    r1_rho0, rnu, r1_6 = 1.0 / RHO0, 1.4e-6, 1.0 / 6.0
    hz = ht0 * s
    if hz != 0.0:
        zfact1 = ecri * r1_rho0 / (1.0 - math.exp(-hz * hcri))
        zfact2 = ebot * (1.0 + hbot / hz) * r1_rho0
    else:
        zfact1 = zfact2 = 0.0
    zfact3 = zfact4 = 0.0
    for k in range(len(rn2)):
        zfact3 = zfact3 + (e3[k + 1] * s) * max(0.0, rn2[k])
        zfact4 = zfact4 + (e3[k + 1] * s) * math.sqrt(max(0.0, rn2[k]))
    if zfact3 != 0.0:
        zfact3 = ensq * r1_rho0 / zfact3
    if zfact4 != 0.0:
        zfact4 = esho * r1_rho0 / zfact4
    out_t, out_s, out_m = list(avt), list(avs), list(avm)
    for k in range(len(rn2)):
        gdn, gup = gd[k + 1] * s, gd[k] * s
        zemx = ((zfact1 * (math.exp((gdn - hz) * hcri) - math.exp((gup - hz) * hcri))
                 + zfact2 * (1.0 / (1.0 + (hz - gdn) / hbot)
                             - 1.0 / (1.0 + (hz - gup) / hbot)))
                * wm[k] / (e3[k + 1] * s)
                + zfact3 * max(0.0, rn2[k])
                + zfact4 * math.sqrt(max(0.0, rn2[k])))
        zreb = zemx / max(1.e-20, rnu * rn2[k])
        zav = zreb * r1_6 * rnu
        zav = min(max(1.4e-7, zav), 1.e-2) * wm[k]
        if tsdiff:
            ratio = (0.505 + 0.495 * math.tanh(
                0.92 * (math.log10(max(1.e-20, zreb * 5.0 * r1_6)) - 0.60))) * wm[k]
        else:
            ratio = 1.0 * wm[k]
        out_s[k] = out_s[k] + zav * ratio
        out_t[k] = out_t[k] + zav
        out_m[k] = out_m[k] + zav
    return out_t, out_s, out_m


def _cases(ncol=300):
    rng = np.random.default_rng(14)
    nif = NLEV - 1
    e3 = np.concatenate([[5.0], np.linspace(10.0, 500.0, NLEV - 1)])
    gd = np.cumsum(e3) - 0.5 * e3[0]
    nbot = rng.integers(0, NLEV, ncol)                     # 0 = land column
    wm = (np.arange(nif)[None, :] < (nbot[:, None] - 1)).astype(np.float64)
    ht0 = np.where(nbot > 0, gd[np.maximum(nbot - 1, 0)] + rng.uniform(0, 5, ncol), 0.0)
    s = 1.0 + rng.normal(0.0, 2e-4, ncol)
    rn2 = rng.normal(1e-5, 2e-5, (ncol, nif))
    rn2[rng.random((ncol, nif)) < 0.05] = 0.0
    rn2[:7] = 0.0                                          # zfact3 == 0 branch
    f = (rng.uniform(0, 0.12, ncol), rng.uniform(0, 0.15, ncol),
         rng.uniform(1e-5, 0.5, ncol), rng.uniform(0, 0.09, ncol),
         rng.uniform(30.0, 3000.0, ncol), 1.0 / rng.uniform(100.0, 3700.0, ncol))
    avt = rng.uniform(1e-10, 1e-3, (ncol, nif)) * wm
    avs = avt * rng.uniform(0.5, 2.0, (ncol, nif))
    avm = rng.uniform(1.4e-6, 1e-2, (ncol, nif)) * wm
    return avt, avs, avm, f, rn2, s, ht0, gd, e3, wm


def _card(case, fn=nemo_zdf_iwm, cfg=CFG):
    avt, avs, avm, f, rn2, s, ht0, gd, e3, wm = case
    forcing = IWMForcing(ebot=jnp.asarray(f[1]), ecri=jnp.asarray(f[0]),
                         ensq=jnp.asarray(f[2]), esho=jnp.asarray(f[3]),
                         hbot=jnp.asarray(f[4]), hcri_inv=jnp.asarray(f[5]))
    return [np.asarray(a) for a in fn(
        jnp.asarray(avt), jnp.asarray(avs), jnp.asarray(avm), forcing,
        jnp.asarray(rn2), jnp.asarray(s), jnp.asarray(ht0), gd, e3,
        jnp.asarray(wm), cfg=cfg, rho_0=RHO0)]


def _reference(case, tsdiff=True):
    avt, avs, avm, f, rn2, s, ht0, gd, e3, wm = case
    out = [np.empty_like(avt) for _ in range(3)]
    for c in range(avt.shape[0]):
        col = _fortran_column(avt[c], avs[c], avm[c], [x[c] for x in f], rn2[c],
                              s[c], ht0[c], gd, e3, wm[c], tsdiff=tsdiff)
        for o, v in zip(out, col):
            o[c] = v
    return out


def _jit_with_constant_maps(avt, avs, avm, forcing, rn2, s, ht0, gd, e3, wm, *, cfg, rho_0):
    """The step's form: the maps are closure constants of the jitted code."""
    return jax.jit(lambda a, b, c, n, t, h, w: nemo_zdf_iwm(
        a, b, c, forcing, n, t, h, gd, e3, w, cfg=cfg, rho_0=rho_0))(
            avt, avs, avm, rn2, s, ht0, wm)


@pytest.mark.parametrize("jit", ["eager", "jit", "jit_constant_maps"])
def test_literal_iwm_is_bit_for_bit_with_the_scalar_transcription(jit):
    case = _cases()
    fn = {"eager": nemo_zdf_iwm,
          "jit": jax.jit(nemo_zdf_iwm, static_argnames=("cfg", "rho_0")),
          # V4 kt 1: XLA folded "/ hbot" into a reciprocal multiply here
          # (41,546 avt cells an ulp off) until hbot was source-rounded
          "jit_constant_maps": _jit_with_constant_maps}[jit]
    got, want = _card(case, fn), _reference(case)
    for name, g, w in zip(("avt", "avs", "avm"), got, want):
        assert np.array_equal(g, w), (name, int(np.sum(g != w)))
    wet = case[-1] > 0
    assert np.sum(got[1][wet] != got[0][wet]) > 1000      # tsdiff makes avs != avt
    assert np.all(got[0][~wet] == case[0][~wet])          # dry: nothing added


def test_tsdiff_false_ratio_is_the_wet_mask():
    case = _cases(60)
    got = _card(case, cfg=CFG._replace(tsdiff=False))
    want = _reference(case, tsdiff=False)
    for g, w in zip(got, want):
        assert np.array_equal(g, w)


def test_literal_form_refuses_mevar_and_generic_form_but_runs_native_policy():
    case = _cases(4)
    with pytest.raises(ValueError, match="ln_mevar"):
        _card(case, cfg=CFG._replace(mevar=True))
    with pytest.raises(ValueError, match="form='nemo_literal'"):
        _card(case, cfg=CFG._replace(form="generic"))
    set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    eager = _card(case)
    compiled = _card(case, jax.jit(
        nemo_zdf_iwm, static_argnames=("cfg", "rho_0")))
    for left, right in zip(eager, compiled, strict=True):
        assert np.isfinite(left).all()
        ulp = np.abs(
            left.view(np.int64) - right.view(np.int64)
        )
        assert int(ulp.max(initial=0)) <= 1


def test_literal_iwm_reverse_gradient_is_finite_where_n2_is_clipped(monkeypatch):
    """sqrt(max(0, N^2)) has an infinite slope at 0; the jitted reverse pass
    through the literal IWM must stay finite there (round 316, rung 10), and
    equal the unguarded sqrt's derivative where N^2 > 0."""
    import legoesm.ocean.physics.vertical_mixing.internal_wave_mixing as iwm_mod

    avt, avs, avm, f, rn2, s, ht0, gd, e3, wm = _cases()
    assert (rn2 <= 0.0).any()
    forcing = IWMForcing(ebot=jnp.asarray(f[1]), ecri=jnp.asarray(f[0]),
                         ensq=jnp.asarray(f[2]), esho=jnp.asarray(f[3]),
                         hbot=jnp.asarray(f[4]), hcri_inv=jnp.asarray(f[5]))

    def loss(n2):
        out = nemo_zdf_iwm(jnp.asarray(avt), jnp.asarray(avs), jnp.asarray(avm),
                           forcing, n2, jnp.asarray(s), jnp.asarray(ht0), gd, e3,
                           jnp.asarray(wm), cfg=CFG, rho_0=RHO0)
        return sum(jnp.sum(x) for x in out)

    g = np.asarray(jax.jit(jax.grad(loss))(jnp.asarray(rn2)))
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)

    # Positive N^2 everywhere: guarded VJP/JVP equal the unguarded jnp.sqrt's.
    n2_wet = jnp.asarray(np.abs(rn2) + 1.0e-6)
    d = jnp.asarray(np.random.default_rng(316).standard_normal(rn2.shape) * 1.0e-7)

    def derivs():
        vjp = jax.jit(jax.grad(loss))(n2_wet)
        jvp = jax.jit(lambda x, v: jax.jvp(loss, (x,), (v,))[1])(n2_wet, d)
        return np.asarray(vjp), float(jvp)

    vjp, jvp = derivs()
    monkeypatch.setattr(iwm_mod, "safe_sqrt", jnp.sqrt)
    vjp_ref, jvp_ref = derivs()
    assert np.all(np.isfinite(vjp_ref)) and np.any(vjp_ref != 0.0) and jvp_ref != 0.0
    # Rounding bound: the two jitted graphs fuse differently (measured 6.8e-16).
    np.testing.assert_allclose(vjp, vjp_ref, rtol=1.0e-14, atol=0.0)
    np.testing.assert_allclose(jvp, jvp_ref, rtol=1.0e-14, atol=0.0)
