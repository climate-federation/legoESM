"""Oracle match: OpenIFS 49r1 subgrid-orography drag (``ifs_sso.ifs_sso_tendencies``).

Fixture ``oracle_fixtures/sso_lott_miller_ifs.npz``: the tier-2 audit's 24
atmosphere columns (16 L32, 5 L137, 3 stress), each with seven SSO parameter
sets, and the outputs of RUNNING OpenIFS 8f6f722f73 arpifs/phys_ec/
{sugwd,gwdrag,gwsetup,gwprofil}.F90 (gfortran 8.5, fp64) in three arms:
``L`` legoESM g/R_d/c_pd, ``N`` 49r1 native constants, ``C`` as L with NAMGWD
GTENLIM = 1e-3 (scripts/validate/oracle/sso_ifs/{build.sh,make_fixture.py}).
SUGWD ran per column on that column's coordinate read as pure sigma.

Error metric (as the IFS convection oracle): per field, max|ours - oracle| /
max|oracle| over all columns and levels.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.ifs_sso import (
    IFSSSOConfig, IFSTOFDConfig, hard_config, hard_tofd_config, ifs_sso_tendencies,
    ifs_tofd_coefficient, sugwd_levels,
)

_FIX = Path(__file__).parent / "oracle_fixtures" / "sso_lott_miller_ifs.npz"
_TOL = 1.0e-12  # oracle bar (max relative error, fp64)
_SMOOTH_GAP = 0.05  # per-column smooth-vs-hard bar (vote V15a; measured max 0.020)
_GROUPS = ("L32", "L137", "STRESS")
_INTS = ("kkcrith", "kcrit", "kkenvh", "kknu", "kknu2")


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(_FIX))


def _consts(fix, arm, g):
    m = re.search(r"CONST MODE RG RD RCPD\s+(\d)\s+(\S+)\s+(\S+)\s+(\S+)", str(fix[f"{arm}_{g}_log"]))
    mode, rg, rd, cpd = int(m.group(1)), *(float(x.replace("D", "E")) for x in m.groups()[1:])
    if arm in ("L", "C"):
        assert mode == 1 and (rg, rd, cpd) == (constants.g, constants.R_d, constants.c_pd)
    return rg, rd, cpd


def _levels(fix, g):
    ph = fix[f"{g}_p_half"]
    out = []
    for row in ph:
        vbh = row / row[-1]
        stpre = 0.5 * (vbh[:-1] + vbh[1:]) * 101325.0
        out.append(sugwd_levels(np.zeros_like(vbh), vbh, stpre))
    return np.array(out)


def _run(fix, arm, g, cfg=None):
    rg, rd, cpd = _consts(fix, arm, g)
    cfg = hard_config() if cfg is None else cfg
    if arm == "C":
        cfg = cfg._replace(gtenlim_m_s2=1.0e-3)
    a = lambda k: jnp.asarray(fix[f"{g}_{k}"], jnp.float64)
    lev = _levels(fix, g)
    return ifs_sso_tendencies(a("p_half"), a("p_full"), a("geo_full"), a("T"), a("u"), a("v"),
                              a("std"), a("gamma"), a("theta"), a("sigma"),
                              lev[:, 0], lev[:, 1], lev[:, 2], cfg, rg, rd, cpd)


def _err(v, ref):
    v = np.asarray(v, np.float64)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(ref))):
        return np.inf
    return float(np.max(np.abs(v - ref)) / max(np.max(np.abs(ref)), 1e-300))


@pytest.mark.parametrize("g", _GROUPS)
def test_sugwd_levels_match_openifs(fix, g):
    lev = _levels(fix, g)
    for j, k in enumerate(("nktopg", "ngwdtop", "ngwdlim")):
        np.testing.assert_array_equal(lev[:, j], fix[f"L_{g}_{k}"])


@pytest.mark.parametrize("arm", ["L", "N", "C"])
@pytest.mark.parametrize("g", _GROUPS)
def test_sso_matches_openifs(fix, arm, g):
    out = _run(fix, arm, g)
    for f in ("soteu", "sotev", "sobeta"):
        e = _err(getattr(out, f), fix[f"{arm}_{g}_{f}"])
        assert e <= _TOL, (f, e)
    ints = np.asarray(out.ints)
    on = fix[f"{g}_std"] > 50.0
    for j, k in enumerate(_INTS):
        np.testing.assert_array_equal(ints[on, j], fix[f"{arm}_{g}_{k}"][on], err_msg=k)


def test_fixture_exercises_every_branch(fix):
    """Each oracle branch fires somewhere (else the match above proves nothing there)."""
    n = {k: 0 for k in ("off", "on_near", "block", "crit", "interp", "clip", "top", "tofd_cut")}
    for g in _GROUPS:
        std = fix[f"{g}_std"]
        off = std <= 50.0
        assert np.all(fix[f"L_{g}_soteu"][off] == 0.0) and np.all(fix[f"L_{g}_sobeta"][off] == 0.0)
        n["off"] += int(off.sum())
        act = (np.abs(fix[f"L_{g}_soteu"]).max(1) > 0) | (fix[f"L_{g}_sobeta"].max(1) > 0)
        n["on_near"] += int(((std > 50.0) & (std < 60.0) & act).sum())
        n["block"] += int((fix[f"L_{g}_sobeta"].max(1) > 0.0).sum())
        n["crit"] += int((fix[f"L_{g}_kcrit"][~off] > 1).sum())
        n["interp"] += int((fix[f"L_{g}_kkcrith"] < fix[f"L_{g}_kkenvh"] - 1)[~off].sum())
        c = np.abs(np.stack([fix[f"C_{g}_soteu"], fix[f"C_{g}_sotev"]]))
        n["clip"] += int((c == 1.0e-3).sum())
        top = fix[f"{g}_p_full"] < 9.9
        n["top"] += int((np.abs(fix[f"L_{g}_soteu"])[top] > 0.0).sum())
        geo = fix[f"{g}_geo_full"] / constants.g
        n["tofd_cut"] += int(((geo > 5000.0) & (fix[f"L_{g}_tofdc"] == 0.0)).sum() > 0)
    assert all(v > 0 for v in n.values()), n


def test_ri_saturation_branch_fires(fix):
    """The ZRIW < GRCRIT (saturated) branch of GWPROFIL is taken somewhere."""
    hits = 0
    for g in _GROUPS:
        sat = np.asarray(_run(fix, "L", g).sat_levels)
        hits += int(sat.sum())
    assert hits > 0


def test_momentum_budget_closes_hard_form(fix):
    """Column wave sink = -dir * [tau_sfc - tau(m+1) * (1 - sum_top dp^2/dp0^2)].

    Below the model-top region the deposited tendency telescopes to the stress
    divergence; the 49r1 top region (p_full < GRFPLM) spreads tau(NGWDTOP+1)
    with weights dp/dp0^2, which leaks the remainder (oracle behaviour)."""
    for g in _GROUPS:
        out = _run(fix, "L", g)
        ph, pf = fix[f"{g}_p_half"], fix[f"{g}_p_full"]
        dp = np.diff(ph, axis=1)
        lev = _levels(fix, g)
        tau = np.asarray(out.tau)
        for i in range(ph.shape[0]):
            top = pf[i] < 9.9
            m = int(top.sum())
            assert np.all(top[:m])                       # contiguous from the model top
            ktop = int(lev[i, 1])
            s = (dp[i, top] ** 2).sum() / (ph[i, ktop] - ph[i, 0]) ** 2 if m else 0.0
            want = -(tau[i, -1] - tau[i, m] + (tau[i, ktop] * s if m else 0.0))
            for f, d in (("soteu", "dir_u"), ("sotev", "dir_v")):
                got = (np.asarray(getattr(out, f))[i] * dp[i]).sum() / constants.g
                np.testing.assert_allclose(got, float(np.asarray(getattr(out, d))[i]) * want,
                                           rtol=1e-10, atol=1e-14)


def test_drag_opposes_low_level_wind(fix):
    for g in _GROUPS:
        out = _run(fix, "L", g, IFSSSOConfig())
        dot = (np.asarray(out.soteu) * np.asarray(out.ulow)[:, None]
               + np.asarray(out.sotev) * np.asarray(out.vlow)[:, None])
        assert np.all(dot <= 1e-18)
        assert np.all(np.asarray(out.sobeta) >= 0.0)


def test_smooth_form_close_to_oracle(fix):
    """The default (smooth) form stays near the hard oracle form, column by column."""
    for g in _GROUPS:
        hard = _run(fix, "L", g)
        soft = _run(fix, "L", g, IFSSSOConfig())
        for f in ("soteu", "sotev", "sobeta"):
            h, s_ = np.asarray(getattr(hard, f)), np.asarray(getattr(soft, f))
            assert np.all(np.isfinite(s_))
            m = np.abs(h).max(1)
            gap = np.abs(s_ - h).max(1) / np.where(m > 0, m, 1.0)
            assert np.all(gap[m > 0] < _SMOOTH_GAP), (g, f, float(gap.max()))
            assert np.all(np.abs(s_[m == 0]).max(initial=0.0) < 1e-12), (g, f)


def test_smooth_clip_keeps_wind_gradient_at_zero(fix):
    """d sotev/dv at v = 0 is not lost by the smooth GTENLIM clip (central FD agrees)."""
    g = "L32"
    i = _drag_column(fix, g)
    rg, rd, cpd = _consts(fix, "C", g)
    lev = _levels(fix, g)
    a = {k: jnp.asarray(fix[f"{g}_{k}"][i:i + 1], jnp.float64)
         for k in ("p_half", "p_full", "geo_full", "T", "u", "std", "sigma")}
    cfg = IFSSSOConfig(gtenlim_m_s2=1.0e-3)

    def f(dv):
        v = jnp.zeros_like(a["u"]) + dv
        o = ifs_sso_tendencies(a["p_half"], a["p_full"], a["geo_full"], a["T"], a["u"], v,
                               a["std"], jnp.asarray([1.0]), jnp.asarray([0.0]), a["sigma"],
                               lev[i:i + 1, 0], lev[i:i + 1, 1], lev[i:i + 1, 2], cfg, rg, rd, cpd)
        return jnp.sum(o.sotev)

    gr = float(jax.grad(f)(0.0))
    fd = (float(f(1e-6)) - float(f(-1e-6))) / 2e-6
    assert fd != 0.0
    assert gr == pytest.approx(fd, rel=1e-3)


def _drag_column(fix, g="L32"):
    """A column with both wave drag above and blocking below (oracle arm L)."""
    wave = np.abs(fix[f"L_{g}_soteu"]).max(1)
    both = (wave > 0.0) & (fix[f"L_{g}_sobeta"].max(1) > 0.0)
    return int(np.argmax(np.where(both, wave, -1.0)))


def _grad_fd(fix, field, i=None, g="L32"):
    """d(sum soteu^2 + sobeta)/d(field[i]) on one column: reverse mode vs central FD."""
    i = _drag_column(fix, g) if i is None else i
    rg, rd, cpd = _consts(fix, "L", g)
    lev = _levels(fix, g)
    a = {k: jnp.asarray(fix[f"{g}_{k}"][i:i + 1], jnp.float64)
         for k in ("p_half", "p_full", "geo_full", "T", "u", "v", "std", "gamma", "theta", "sigma")}

    def f(x):
        b = {**a, field: x}
        o = ifs_sso_tendencies(b["p_half"], b["p_full"], b["geo_full"], b["T"], b["u"], b["v"],
                               b["std"], b["gamma"], b["theta"], b["sigma"],
                               lev[i:i + 1, 0], lev[i:i + 1, 1], lev[i:i + 1, 2],
                               IFSSSOConfig(), rg, rd, cpd)
        return jnp.sum(o.soteu ** 2 + o.sotev ** 2) * 1e6 + jnp.sum(o.sobeta)

    x0 = a[field]
    gr = np.asarray(jax.grad(f)(x0))
    h = 1e-6 * np.maximum(np.abs(np.asarray(x0)), 1.0)
    fd = np.zeros_like(gr)
    for idx in np.ndindex(gr.shape):
        e = np.zeros_like(gr); e[idx] = h[idx]
        fd[idx] = (float(f(x0 + e)) - float(f(x0 - e))) / (2 * h[idx])
    return gr, fd


@pytest.mark.parametrize("field", ["std", "gamma", "theta", "sigma"])
def test_reverse_grad_matches_fd(fix, field):
    gr, fd = _grad_fd(fix, field)
    assert np.all(np.isfinite(gr)) and np.abs(fd).max() > 0.0
    np.testing.assert_allclose(gr, fd, rtol=1e-4, atol=1e-8 * np.abs(fd).max())


def test_reverse_grad_wrt_wind_finite_nonzero(fix):
    gr, fd = _grad_fd(fix, "u")
    assert np.all(np.isfinite(gr)) and np.abs(gr).max() > 0.0
    np.testing.assert_allclose(gr, fd, rtol=1e-3, atol=1e-6 * np.abs(fd).max())


@pytest.mark.parametrize("arm", ["L", "N"])
@pytest.mark.parametrize("g", _GROUPS)
def test_tofd_matches_openifs(fix, arm, g):
    rg, _, _ = _consts(fix, arm, g)
    a = lambda k: jnp.asarray(fix[f"{g}_{k}"], jnp.float64)
    c = ifs_tofd_coefficient(a("geo_full"), a("u"), a("v"), a("sigflt"), hard_tofd_config(), rg)
    ref = fix[f"{arm}_{g}_tofdc"]
    assert (ref > 0).any()
    assert _err(c, ref) <= _TOL


def test_tofd_smooth_form_finite_grad(fix):
    g = "L32"
    a = lambda k: jnp.asarray(fix[f"{g}_{k}"][:7], jnp.float64)
    f = lambda sf, u: jnp.sum(ifs_tofd_coefficient(a("geo_full"), u, a("v"), sf, IFSTOFDConfig()))
    gs, gu = jax.grad(f, argnums=(0, 1))(a("sigflt"), a("u"))
    assert np.all(np.isfinite(gs)) and np.all(np.isfinite(gu)) and np.abs(gu).max() > 0.0
    hard = ifs_tofd_coefficient(a("geo_full"), a("u"), a("v"), a("sigflt"), hard_tofd_config())
    soft = ifs_tofd_coefficient(a("geo_full"), a("u"), a("v"), a("sigflt"), IFSTOFDConfig())
    assert _err(soft, np.asarray(hard)) < 0.05


def test_std_threshold_is_strict(fix):
    """GWDRAG runs only where PHSTD > 50 m: exactly 50 m gives zero (hard form)."""
    g = "L32"
    i = _drag_column(fix, g)
    rg, rd, cpd = _consts(fix, "L", g)
    lev = _levels(fix, g)
    a = {k: jnp.asarray(fix[f"{g}_{k}"][i:i + 1], jnp.float64)
         for k in ("p_half", "p_full", "geo_full", "T", "u", "v", "gamma", "theta", "sigma")}
    o = ifs_sso_tendencies(a["p_half"], a["p_full"], a["geo_full"], a["T"], a["u"], a["v"],
                           jnp.asarray([50.0]), a["gamma"], a["theta"], a["sigma"],
                           lev[i:i + 1, 0], lev[i:i + 1, 1], lev[i:i + 1, 2], hard_config(), rg, rd, cpd)
    assert float(jnp.abs(o.soteu).max() + o.sobeta.max()) == 0.0


@pytest.mark.parametrize("g", _GROUPS)
def test_reverse_grad_finite_on_every_fixture_column(fix, g):
    """Smooth form: d(sum of outputs)/d(SSO params, wind) finite on every column."""
    rg, rd, cpd = _consts(fix, "L", g)
    lev = _levels(fix, g)
    a = {k: jnp.asarray(fix[f"{g}_{k}"], jnp.float64)
         for k in ("p_half", "p_full", "geo_full", "T", "u", "v", "std", "gamma", "theta", "sigma")}

    def f(std, gamma, theta, sigma, u):
        o = ifs_sso_tendencies(a["p_half"], a["p_full"], a["geo_full"], a["T"], u, a["v"],
                               std, gamma, theta, sigma, lev[:, 0], lev[:, 1], lev[:, 2],
                               IFSSSOConfig(), rg, rd, cpd)
        return jnp.sum(o.soteu) * 1e4 + jnp.sum(o.sotev) * 1e4 + jnp.sum(o.sobeta)

    grads = jax.grad(f, argnums=(0, 1, 2, 3, 4))(a["std"], a["gamma"], a["theta"], a["sigma"], a["u"])
    for gr in grads:
        assert np.all(np.isfinite(np.asarray(gr)))
