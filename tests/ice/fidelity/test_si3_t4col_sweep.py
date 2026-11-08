"""Sweep of the SI3 column port's pre-existing hard switches (votes Z1-Z20).

* The column port reads the SI3 lane's :class:`SI3SwitchConfig` (vote Z18):
  ``SI3_HARD_SWITCHES`` (relative_width 0) is NEMO's hard form everywhere and
  equals passing every per-fix width as 0.
* Numerics guards and roundoff-only zero floors stay hard (votes Z1, Z13)
  because they bind just where their operand is 0 or a roundoff residual:
  checked on every tier-4 fixture (NEMO 5.0.1 SI3 ORCA1, kt=17281) with a
  recording ``jnp.maximum/minimum``, with a control that the recorder flags a
  live-binding floor.  The full record gives the same answer
  (oracle_audit/seaice_fix/sweep/guard_probe.py, floor_probe.py).
* Z8 zh_min, Z4/Z17/Z20 temperature bounds and Z14 live zero floors are C1;
  the melting-point clamps use the one-sided ``min_c1`` that is exact AT the
  bound, so the T_su >= T0 / T_i >= Tmelt branches still fire.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.config import SI3_HARD_SWITCHES, SI3SwitchConfig
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K

from si3_t4col_fixture import dh_args, dt, load, zdf_args

jax.config.update("jax_enable_x64", True)
FIXTURES = ("c2_ti_clamp", "c3_qcn_top_k", "c4_fullmelt", "c5_flood_sst", "c6_melt_order",
            "c7_remap_eps", "c8_sal_himin", "c9_exchange")
HARD = dict(nemo_version="5.0.1", switches=SI3_HARD_SWITCHES)
SMOOTH = dict(nemo_version="5.0.1", switches=SI3SwitchConfig())
ROUNDOFF = 1e-16      # largest guard residual on the full record is 8.3e-17


def _zdf(fx, **kw):
    e_i, e_s, sz, h_i, h_s, t_su, fz = zdf_args(fx)
    return bl._si3_zdf_bl99_step(e_i, e_s, sz, h_i, h_s, t_su, fz, dt(fx), K, **kw)


def _dh(fx, **kw):
    st, zr, fz = dh_args(fx)
    return bl._dh_step_fluxes(st, zr, fz, dt(fx), K, **kw)


def test_hard_switches_equal_every_width_zero():
    """SI3_HARD_SWITCHES == every per-fix width 0.  Bitwise equality with the
    pre-sweep hard code is the literal-row gate (hard rows at this commit vs
    main), not this test."""
    for name in FIXTURES:
        fx = load(name)
        a, b = _zdf(fx, **HARD), _zdf(fx, **HARD, t_clamp_width=0.0)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        (s1, f1) = _dh(fx, **HARD)
        (s2, f2) = _dh(fx, **HARD, full_melt_width=0.0, growth_gate_width=0.0)
        for x, y in zip(tuple(s1) + tuple(f1), tuple(s2) + tuple(f2)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_relative_width_rejects_negative_and_nonfinite():
    fx = load("c9_exchange")
    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="relative_width"):
            _dh(fx, switches=SI3SwitchConfig(relative_width=bad))


class _Recorder:
    """jnp proxy: count, per call line, cells where a nonzero scalar floor wins away from zero."""

    def __init__(self, real):
        self.real, self.hits = real, {}

    def __getattr__(self, name):
        f = getattr(self.real, name)
        if name not in ("maximum", "minimum"):
            return f

        def g(a, b, *args, **kw):
            for x, floor in ((a, b), (b, a)):
                fl = np.asarray(floor)
                if fl.size != 1 or np.asarray(x).size <= 1 or isinstance(floor, jax.core.Tracer):
                    continue
                v = float(fl)
                if v == 0.0 or (name == "maximum") != (v > 0):
                    continue
                xv = np.asarray(x, float)
                binds = xv < v if name == "maximum" else xv > v
                line = sys._getframe(1).f_lineno
                self.hits[line] = self.hits.get(line, 0) + int((binds & (np.abs(xv) > ROUNDOFF)).sum())
            return f(a, b, *args, **kw)
        return g


def _run_all(**kw):
    for name in FIXTURES:
        fx = load(name)
        _zdf(fx, **kw)
        _dh(fx, **kw)
        st, _, fz = dh_args(fx)
        bl._sal_step(st, fz, dt(fx), K, switches=kw["switches"])


def test_hard_guards_bind_only_at_their_zero(monkeypatch):
    rec = _Recorder(bl.jnp)
    monkeypatch.setattr(bl, "jnp", rec)
    with jax.disable_jit():
        _run_all(**HARD)
    src = open(bl.__file__).read().splitlines()
    guards = {ln: n for ln, n in rec.hits.items()
              if any(g in " ".join(src[ln - 1:ln + 3])
                     for g in ("_EPS10", "_EPS20", "1.0e-20", "_P07_CONDUCTIVITY_MIN"))}
    assert guards, "recorder saw no guard call (instrument broken)"
    live = {ln: (n, src[ln - 1].strip()) for ln, n in guards.items() if n}
    assert not live, live


def test_recorder_sees_a_floor_that_binds_on_live_values(monkeypatch):
    """Control: the instrument flags zh_min in its hard form on thin live snow."""
    rec = _Recorder(bl.jnp)
    monkeypatch.setattr(bl, "jnp", rec)
    fx = load("c3_qcn_top_k")
    e_i, e_s, sz, h_i, h_s, t_su, fz = zdf_args(fx)
    monkeypatch.setattr(bl, "max_c1", lambda a, b, s: rec.maximum(a, b))
    with jax.disable_jit():
        bl._radiation(h_i, jnp.full_like(h_s, 4e-4), fz.qtr_ice_top, t_su)
    assert sum(rec.hits.values()) > 0


def test_zh_min_smooth_exact_outside_band_and_differentiable():
    fx = load("c3_qcn_top_k")
    e_i, e_s, sz, h_i, h_s, t_su, fz = zdf_args(fx)
    w = SI3SwitchConfig().relative_width * SI3SwitchConfig().thickness_scale_m

    def qtop(hs, switches):
        r = bl._si3_zdf_bl99_step(e_i, e_s, sz, h_i, jnp.full_like(h_s, hs), t_su, fz, dt(fx), K,
                                  switches=switches)
        return r.qcn_ice_top
    for hs in (0.0, 4e-4, bl._H_MIN, bl._H_MIN + w, 0.05):    # outside the one-sided band: identical
        np.testing.assert_array_equal(np.asarray(qtop(hs, SI3SwitchConfig())),
                                      np.asarray(qtop(hs, SI3_HARD_SWITCHES)))
    mid = bl._H_MIN + 0.5 * w
    assert np.any(np.asarray(qtop(mid, SI3SwitchConfig())) != np.asarray(qtop(mid, SI3_HARD_SWITCHES)))
    g = jax.grad(lambda hs: jnp.sum(qtop(hs, SI3SwitchConfig())))(jnp.asarray(mid))
    eps = 1e-9
    fd = (float(jnp.sum(qtop(mid + eps, SI3SwitchConfig())))
          - float(jnp.sum(qtop(mid - eps, SI3SwitchConfig())))) / (2 * eps)
    assert bool(jnp.isfinite(g))
    np.testing.assert_allclose(float(g), fd, rtol=1e-4)


def test_one_sided_clip_is_exact_at_and_beyond_the_bounds():
    """Z20: min_c1/max_c1 return the bound exactly at and beyond it; width 0 = jnp.clip."""
    T0, w = K.T0, 0.01
    x = jnp.asarray([T0 - 150.0, T0 - 100.0, T0 - 100.0 + w, T0 - 5.0, T0 - w, T0, T0 + 3.0])
    np.testing.assert_array_equal(np.asarray(bl._clip_c1(x, T0 - 100.0, T0, w)),
                                  np.asarray(jnp.clip(x, T0 - 100.0, T0)))
    np.testing.assert_array_equal(np.asarray(bl._clip_c1(x, T0 - 100.0, T0, 0.0)),
                                  np.asarray(jnp.clip(x, T0 - 100.0, T0)))
    y = float(bl._clip_c1(jnp.asarray(T0 - 0.5 * w), T0 - 100.0, T0, w))
    assert T0 - 0.5 * w < y < T0                                     # pulled toward, never above, T0
    g = jax.grad(lambda t: bl._clip_c1(t, T0 - 100.0, T0, w))(jnp.asarray(T0 - 0.5 * w))
    assert 0.0 < float(g) < 1.0


def test_hard_clip_keeps_jnp_clip_order_on_an_inverted_interval():
    """Width 0 is jnp.clip also where 0.75*sss < rn_simin (hi < lo): returns hi."""
    x = jnp.asarray([-1.0, 0.05, 0.5])
    np.testing.assert_array_equal(np.asarray(bl._clip_c1(x, 0.1, 0.0, 0.0)),
                                  np.asarray(jnp.clip(x, 0.1, 0.0)))
    fx = load("c8_sal_himin")
    st, _, fz = dh_args(fx)
    fz0 = fz._replace(sss=jnp.zeros_like(fz.sss))
    S, _, _ = bl._sal_step(st, fz0, dt(fx), K, switches=SI3_HARD_SWITCHES)
    gate = (np.asarray(st.h_ice) > bl._SAL_MIN_THICKNESS)
    assert gate.any(), "no gated cell (vacuous)"
    np.testing.assert_array_equal(np.asarray(S)[gate], 0.0)


def test_smooth_mode_keeps_the_melting_branches_reachable():
    """Columns at the melting point in hard mode reach it exactly in smooth mode (Z12 defect, Z20 fix)."""
    hit = 0
    for name in FIXTURES:
        fx = load(name)
        hard, soft = _zdf(fx, **HARD), _zdf(fx, **SMOOTH)
        at = np.asarray(hard.T_surface) == K.T0
        hit += int(at.sum())
        np.testing.assert_array_equal(np.asarray(soft.T_surface)[at], K.T0)
        melt = K.T0 - K.liquidus_slope * np.asarray(zdf_args(fx)[2])
        ati = np.asarray(hard.T_ice) == melt
        np.testing.assert_array_equal(np.asarray(soft.T_ice)[ati], melt[ati])
    assert hit > 0, "no melting surface in the fixtures (vacuous)"


# Hard MAX(x,0)/MIN(x,0) sites that still bind on live values, with the vote that keeps them.
LIVE_HARD_FLOORS = {
    "overlap = jnp.maximum(": "Z15 interval geometry of the conservative remap",
    "jnp.maximum(0.0, zf / (dE_safe": "Z16 already C1 through the growth gate",
}
# Remaining-energy floors of the take-from-what-is-left cascades: roundoff-only in hard mode;
# smoothed with the cascades (Z5, branch-blend commit).
PENDING_Z5 = "- basal_remove * (-energy_difference"


def test_remaining_hard_zero_floors_bind_only_at_roundoff_or_are_listed(monkeypatch):
    real, hits = bl.jnp, {}

    class P:
        def __getattr__(self, name):
            f = getattr(real, name)
            if name not in ("maximum", "minimum"):
                return f

            def g(a, b, *args, **kw):
                for x, fl in ((a, b), (b, a)):
                    fa = np.asarray(fl)
                    if fa.size != 1 or float(fa) != 0.0 or np.asarray(x).size <= 1:
                        continue
                    xv = np.asarray(x, float)
                    binds = xv < 0 if name == "maximum" else xv > 0
                    live = binds & (np.abs(xv) > 1e-12 * np.abs(xv).max())      # vote Z13
                    ln = sys._getframe(1).f_lineno
                    hits[ln] = hits.get(ln, 0) + int(live.sum())
                return f(a, b, *args, **kw)
            return g
    monkeypatch.setattr(bl, "jnp", P())
    with jax.disable_jit():
        _run_all(**SMOOTH)
    monkeypatch.setattr(bl, "jnp", real)
    src = open(bl.__file__).read().splitlines()
    assert hits, "recorder saw no zero floor (instrument broken)"
    bad = {ln: (n, src[ln - 1].strip()) for ln, n in hits.items()
           if n and not any(k in src[ln - 1] for k in LIVE_HARD_FLOORS)
           and PENDING_Z5 not in " ".join(src[ln - 1:ln + 3])}
    assert not bad, bad


def test_live_floors_are_c1_and_budgets_close_in_smooth_mode():
    """Z14: column mass, enthalpy (dh identity) and salt close in smooth mode on every fixture."""
    for name in FIXTURES:
        fx = load(name)
        st, zr, fz = dh_args(fx)
        out, f = bl._dh_step_fluxes(st, zr, fz, dt(fx), K, **SMOOTH)
        a, a1, d = np.asarray(st.concentration), np.asarray(out.concentration), dt(fx)
        mass = (a1 * (K.rho_ice * out.h_ice + K.rho_snow * out.h_snow)
                - a * (K.rho_ice * st.h_ice + K.rho_snow * st.h_snow)) / d
        wsum = sum(np.asarray(getattr(f, k)) for k in bl.SI3ExchangeFluxes._fields
                   if k.startswith("wfx") and k not in ("wfx_err_sub", "wfx_opw"))
        np.testing.assert_allclose(np.asarray(mass), -wsum, rtol=0, atol=1e-11 * max(1.0, np.abs(mass).max()))
        q = lambda s_: (np.sum(np.asarray(s_.e_ice), -1) * np.asarray(s_.h_ice)  # noqa: E731
                        + np.sum(np.asarray(s_.e_snow), -1) * np.asarray(s_.h_snow)) / 3.0
        g = lambda k: np.asarray(getattr(f, k))  # noqa: E731
        rhs = (g("hfx_thd") + g("hfx_res") + g("hfx_sub") + g("hfx_spr")
               - g("hfx_sum") - g("hfx_bom") - g("hfx_bog") - g("hfx_snw"))
        dq = (a1 * q(out) - a * q(st)) / d
        np.testing.assert_allclose(dq, rhs, rtol=0, atol=1e-11 * max(1.0, np.abs(dq).max()))
        S, _, sf = bl._sal_step(st, fz, d, K, switches=SI3SwitchConfig())
        dsalt = a * K.rho_ice * np.asarray(st.h_ice) * (np.asarray(S) - np.asarray(st.S_bulk)) / d
        ssum = sum(np.asarray(getattr(sf, k)) for k in bl.SI3ExchangeFluxes._fields if k.startswith("sfx"))
        np.testing.assert_allclose(dsalt, -ssum, rtol=0, atol=1e-11 * max(1.0, np.abs(dsalt).max()))
    x = jnp.asarray([-1.0, 0.0, 1e-6, 1e-4, 5.0])
    np.testing.assert_array_equal(np.asarray(bl._pos(x, 0.0)), np.maximum(np.asarray(x), 0.0))
    y = np.asarray(bl._pos(x, 1e-5))
    assert y[0] == 0 and y[1] == 0 and y[3] == 1e-4 and y[4] == 5.0 and 0 < y[2] < 1e-6
    g = jax.grad(lambda t: bl._pos(t, 1e-5))(jnp.asarray(5e-6))
    assert 0 < float(g) <= 4.0 / 3.0 + 1e-12          # positive_part_c1 slope t(4-3t) peaks at 4/3
