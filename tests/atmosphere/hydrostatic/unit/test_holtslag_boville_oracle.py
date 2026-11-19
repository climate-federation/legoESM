"""Holtslag-Boville vs the RUNNING oracle (E3SM hb_diff + CAM compute_vdiff).

Fixture ``tests/atmosphere/oracle_fixtures/holtslag_boville.npz`` was generated
by RUNNING E3SM v3.0.1 (d4ca7d0606930f53f30eb7ff7dbb92ec83b83b84)
``hb_diff.F90`` + ``pbl_utils.F90`` (bit-identical to CAM cam_cesm2_1_rel_60)
and CAM cam_cesm2_1_rel_60 (a03b84b7c4e34f965b115686f22a043b85739e56)
``diffusion_solver.F90::compute_vdiff`` set up as ``vertical_diffusion_tend``
does for ``eddy_scheme='HB'``, compiled unmodified (gfortran 8.5.0, fp64) with
``scripts/validate/oracle/holtslag_boville/{build.sh,make_fixture.py}`` on the
21-column tier-2 audit set (metadata in the npz ``meta``).

Metric (the tier-1 audit metric, as tests/atmosphere/test_clubb_core_oracle.py):
per field, ``max |ours - oracle| / max |oracle|``, normalised PER COLUMN for the
diffusivity core (a column with a huge field, e.g. zero-shear Ri ~ 1e40, must
not hide the others) and over the whole fixture for the tendencies (the
tier-2 audit form: the calm column's exact-zero momentum tendency is pure
round-off on both sides); bound 1e-10
in fp64 (user decision 2026-10-19) with every existing sharpness field at its
hard-switch limit.  Surface fluxes are prescribed identically on both sides.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import compute_rho
from legoesm.atmosphere.physics.turbulence import holtslag_boville as HB
from legoesm.atmosphere.physics.turbulence.config import (
    HoltslagBovilleConfig,
    SurfaceLayerConfig,
)
from legoesm.atmosphere.physics.turbulence.holtslag_boville import (
    holtslag_boville_turbulence,
)

jax.config.update("jax_enable_x64", True)

RAIR = constants.R_d
CPAIR = constants.c_pd
NLEV = 48
TOL = 1.0e-10
FIXTURE = (Path(__file__).resolve().parents[2] / "oracle_fixtures"
           / "holtslag_boville.npz")
# Every existing sharpness field at its hard-switch limit (tier-2 audit arm H).
HARD = dict(pbl_crossing_sharpness=1e12, sfc_blend_sharpness=1e12,
            cgs_gate_sharpness=1e12, stable_blend_sharpness=1e12,
            unstable_blend_sharpness=1e20, unstable_kbfs_threshold=1e-15)


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(FIXTURE))


def _err(ours, ref, per_column=False):
    """max|ours-ref| / max|ref| (0 when both are identically 0): over the whole
    fixture, or per column (``per_column``, worst column returned)."""
    ref = np.asarray(ref, np.float64)
    ref = ref.reshape(ref.shape[0], -1)
    ours = np.asarray(ours, np.float64).reshape(ref.shape)
    assert np.all(np.isfinite(ours)) and np.all(np.isfinite(ref))
    ax = 1 if per_column else None
    num, den = np.max(np.abs(ours - ref), axis=ax), np.max(np.abs(ref), axis=ax)
    e = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.where(num > 0, np.inf, 0.0))
    return float(np.max(e))


def _run(fx, cfg_kw, *, pref=True, dtype=jnp.float64, capture=None):
    f = {k: jnp.asarray(fx[f"in_{k}"], dtype) for k in
         ("T", "q", "u", "v", "p_full", "p_half", "zm", "zi")}
    sfc = SurfaceLayerConfig()._replace(
        prescribed_shflx_w_m2=jnp.asarray(fx["in_shflx"], dtype),
        prescribed_lhflx_w_m2=jnp.asarray(fx["in_lhflx"], dtype),
        prescribed_evap_kg_m2_s=jnp.asarray(fx["in_qflx"], dtype),
        prescribed_tau_x_pa=jnp.asarray(fx["in_taux"], dtype),
        prescribed_tau_y_pa=jnp.asarray(fx["in_tauy"], dtype))
    cfg = HoltslagBovilleConfig()._replace(surface=sfc, **cfg_kw)
    rho = compute_rho(f["T"], f["p_full"], q_v=f["q"])
    return holtslag_boville_turbulence(
        f["u"], f["v"], f["T"], f["q"], f["p_full"], f["p_half"], f["zm"],
        f["zi"], f["T"][:, -1], f["q"][:, -1], rho, float(fx["dt"]), cfg,
        pref_mid=jnp.asarray(fx["in_pref_mid"], dtype) if pref else None)


def _captured(fx, monkeypatch, cfg_kw):
    cap = {}
    orig = HB.hb_diffusivities

    def wrap(*a, **k):
        cap["args"] = a
        cap["out"] = orig(*a, **k)
        return cap["out"]
    monkeypatch.setattr(HB, "hb_diffusivities", wrap)
    out = _run(fx, cfg_kw)
    return out, cap["args"], cap["out"]


def test_fixture_provenance(fx):
    meta = json.loads(str(fx["meta"]))
    assert meta["oracle"]["hb_diff"]["sha"] == "d4ca7d0606930f53f30eb7ff7dbb92ec83b83b84"
    assert meta["oracle"]["solver"]["sha"] == "a03b84b7c4e34f965b115686f22a043b85739e56"
    c = meta["constants"]
    assert (c["g"], c["vk"], c["cpair"], c["rair"]) == (
        constants.g, constants.kappa_von_karman, constants.c_pd, constants.R_d)
    assert c["zvir"] == constants.R_v / constants.R_d - 1.0
    assert FIXTURE.stat().st_size < 1_000_000
    # the oracle's top/surface interfaces carry nothing
    for k in ("ora_kvm", "ora_kvh", "ora_cgh", "ora_cgs", "ora_tke"):
        assert np.all(fx[k][:, 0] == 0.0) and np.all(fx[k][:, -1] == 0.0), k


def test_diffusivities_and_tendencies_match_oracle(fx, monkeypatch):
    """compute_hb_diff (ustar, Obukhov length, Ri, pblh, kvm, kvh, cgh, cgs,
    tke, tpert, qpert) and compute_vdiff (u, v, T, q tendencies) at the hard
    limit, every column."""
    out, a, hb = _captured(fx, monkeypatch, HARD)
    dt = float(fx["dt"])
    errs = {
        "ustar": _err(a[8], fx["ora_ustar"][:, None], True),
        "obklen": _err(a[12], fx["ora_obklen"][:, None], True),
        "ri": _err(a[7], fx["ora_ri"], True),
        "pblh": _err(out.h_pbl, fx["ora_pblh"][:, None], True),
        "kvm": _err(hb.Km_half, fx["ora_kvm"][:, 1:-1], True),
        "kvh": _err(hb.Kh_half, fx["ora_kvh"][:, 1:-1], True),
        "kvq": _err(hb.Kh_half, fx["ora_kvq"][:, 1:-1], True),
        "cgh": _err(hb.cgh_half, fx["ora_cgh"][:, 1:-1], True),
        "cgs": _err(hb.cgs_half, fx["ora_cgs"][:, 1:-1], True),
        "tke": _err(hb.tke_half, fx["ora_tke"][:, 1:-1], True),
        "tpert": _err(hb.tpert, fx["ora_tpert"][:, None], True),
        "qpert": _err(hb.qpert, fx["ora_qpert"][:, None], True),
        "du": _err(out.du_dt, (fx["ora_u_out"] - fx["in_u"]) / dt),
        "dv": _err(out.dv_dt, (fx["ora_v_out"] - fx["in_v"]) / dt),
        "dT": _err(out.dT_dt, (fx["ora_s_out"] - (CPAIR * fx["in_T"] + constants.g
                                                 * fx["in_zm"])) / (CPAIR * dt)),
        "dq": _err(out.dq_v_dt, (fx["ora_q_out"] - fx["in_q"]) / dt),
    }
    worst = dict(errs)
    print("HB vs oracle, hard limit, max rel err", worst)
    assert max(worst.values()) <= TOL, worst
    # coverage: unstable outer layer, stable, no-crossing fallback, the
    # first-interior mixing length, and the q-countergradient all exercised
    names = json.loads(str(fx["meta"]))["names"]
    assert np.count_nonzero(fx["ora_cgs"]) > 20
    assert np.any(fx["ora_tpert"] > 0) and np.any(fx["ora_qpert"] > 0)
    assert np.any(fx["ora_obklen"] > 0) and np.any(fx["ora_obklen"] < 0)
    assert fx["ora_kvm"][names.index("top_shear"), 1] == 0.01
    assert fx["ora_pblh"][names.index("tibet_deep_mixed")] > 5000.0


def _budget_residuals(fx, du, dv, dT, dq):
    """Column (water, dry static energy + KE) change minus the surface
    (water, sensible heat) flux, normalised by the largest surface flux."""
    dt = float(fx["dt"])
    dpg = np.diff(fx["in_p_half"], axis=1) / constants.g
    res_q = (np.sum(dq * dpg, axis=1) - fx["in_qflx"]) / np.max(np.abs(fx["in_qflx"]))
    u, v = fx["in_u"], fx["in_v"]
    un, vn = u + dt * du, v + dt * dv
    dke = 0.5 * (un ** 2 + vn ** 2 - u ** 2 - v ** 2) / dt
    res_e = (np.sum((CPAIR * dT + dke) * dpg, axis=1) - fx["in_shflx"]) / np.max(
        np.abs(fx["in_shflx"]))
    return res_q, res_e


def test_column_water_and_energy_budgets_close_as_the_oracle(fx):
    """Column water changes by the surface water flux and column dry static
    energy + KE by the surface heat flux (KE dissipation returns the kinetic
    energy removed by diffusion and surface drag as heat) -- with the same
    residual as the oracle's own budget, to the oracle-match tolerance."""
    dt = float(fx["dt"])
    out = _run(fx, {})
    ours = _budget_residuals(fx, *(np.asarray(x) for x in
                                   (out.du_dt, out.dv_dt, out.dT_dt, out.dq_v_dt)))
    ora = _budget_residuals(
        fx, (fx["ora_u_out"] - fx["in_u"]) / dt, (fx["ora_v_out"] - fx["in_v"]) / dt,
        (fx["ora_s_out"] - (CPAIR * fx["in_T"] + constants.g * fx["in_zm"])) / (CPAIR * dt),
        (fx["ora_q_out"] - fx["in_q"]) / dt)
    print("budget residual / max sfc flux: water ours %.2e oracle %.2e; "
          "energy ours %.2e oracle %.2e" % (np.abs(ours[0]).max(), np.abs(ora[0]).max(),
                                             np.abs(ours[1]).max(), np.abs(ora[1]).max()))
    assert np.abs(ours[0] - ora[0]).max() <= TOL
    assert np.abs(ours[1] - ora[1]).max() <= TOL


# Columns whose wind has NO vertical shear: in fp32 the oracle's 1e-36 floor
# on the squared wind difference underflows (s2 ~ 1e-41 flushes to 0), so
# Ri = n2/s2 is inf/NaN and the gradient is not finite.  Pre-existing (same on
# the pre-port code); the fix needs a guard choice that is pending (PR ask).
_ZERO_SHEAR = ("calm_zero_flux", "tibet_deep_mixed")


def _tendency_grads(fx, dtype):
    f = {k: jnp.asarray(fx[f"in_{k}"], dtype) for k in ("T", "q", "u")}

    def loss(T, q, u):
        fx2 = dict(fx)
        fx2.update(in_T=T, in_q=q, in_u=u)
        o = _run(fx2, {}, dtype=dtype)
        return (jnp.sum(o.dT_dt ** 2) * 1e8 + jnp.sum(o.dq_v_dt ** 2) * 1e14
                + jnp.sum(o.du_dt ** 2) * 1e6 + jnp.sum(o.h_pbl))
    return jax.grad(loss, argnums=(0, 1, 2))(f["T"], f["q"], f["u"])


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_grad_finite_on_fixture_columns(fx, dtype):
    """jax.grad of a scalar of the tendencies w.r.t. T, q, u: finite and
    nonzero (fp32: on every column with vertical wind shear)."""
    names = json.loads(str(fx["meta"]))["names"]
    keep = np.array([dtype == jnp.float64 or n not in _ZERO_SHEAR for n in names])
    for gr in _tendency_grads(fx, dtype):
        assert gr.dtype == dtype
        assert bool(jnp.all(jnp.isfinite(gr[keep])))
        assert float(jnp.max(jnp.abs(gr[keep]))) > 0.0


@pytest.mark.xfail(strict=True, reason="fp32 zero-shear Ri underflow; guard pending (PR ask)")
def test_grad_finite_fp32_zero_shear_columns(fx):
    names = json.loads(str(fx["meta"]))["names"]
    sel = np.array([n in _ZERO_SHEAR for n in names])
    for gr in _tendency_grads(fx, jnp.float32):
        assert bool(jnp.all(jnp.isfinite(gr[sel])))


def test_ustar_gradient_exact_at_weak_stress(fx):
    """d ustar / d taux equals the analytic 0.5*ustar*taux/|tau|^2 of the
    oracle's sqrt(|tau| rrho), including stresses whose |tau|^2 < 1e-6."""
    def ustar(taux):
        fx2 = dict(fx)
        fx2.update(in_taux=taux, in_tauy=jnp.zeros_like(taux))
        return _run(fx2, {}).ustar
    taux = jnp.full(fx["in_taux"].shape, 2.0e-4)
    us = ustar(taux)
    assert bool(jnp.all(us > 0.01))           # above the ustar_min floor
    g = jax.vmap(jax.grad(lambda t, i: ustar(t)[i]), in_axes=(None, 0))(
        taux, jnp.arange(taux.shape[0]))
    analytic = 0.5 * us / taux
    assert _err(jnp.diagonal(g), analytic, True) <= TOL


def test_eager_jit_parity(fx):
    eager = _run(fx, {})
    jitted = jax.jit(lambda: _run(fx, {}))()
    for a, b in zip(eager[:10], jitted[:10]):
        assert _err(np.asarray(b), np.asarray(a)) <= TOL


def _build_column(kind):
    sig_half = np.linspace(0.01, 1.0, NLEV + 1)
    p_s = 1.0e5
    p_half = sig_half * p_s
    p_mid = 0.5 * (p_half[:-1] + p_half[1:])
    z_half = -8000.0 * np.log(np.clip(sig_half, 1e-6, None))
    z_mid = 0.5 * (z_half[:-1] + z_half[1:])
    exner = (p_mid / 1e5) ** (RAIR / CPAIR)

    if kind == "convective":
        theta = np.where(z_mid < 1000.0, 301.0,
                         np.where(z_mid < 1100.0, 301.0 + 0.03 * (z_mid - 1000.0),
                                  304.0 + 0.003 * (z_mid - 1100.0)))
        q = np.full(NLEV, 0.006)
        u = np.full(NLEV, 5.0); v = np.zeros(NLEV)
        shflx, qflx, taux, tauy = 150.0, 5.0e-5, 0.04, 0.0
    elif kind == "stable":
        theta = 265.0 + 0.01 * z_mid
        q = np.full(NLEV, 0.001)
        u = np.full(NLEV, 8.0); v = np.zeros(NLEV)
        shflx, qflx, taux, tauy = -20.0, 0.0, 0.06, 0.0
    elif kind == "neutral":
        theta = np.full(NLEV, 290.0)
        q = np.full(NLEV, 0.004)
        u = np.full(NLEV, 10.0); v = np.zeros(NLEV)
        shflx, qflx, taux, tauy = 0.0, 0.0, 0.08, 0.0
    else:
        raise ValueError(kind)
    T = theta * exner
    return dict(p_mid=p_mid, p_half=p_half, z_mid=z_mid, z_half=z_half,
                theta=theta, T=T, q=q, u=u, v=v,
                shflx=shflx, qflx=qflx, taux=taux, tauy=tauy)



def test_full_scheme_runs_and_is_finite():
    """End-to-end scheme (bulk-flux closure + diffusion) stays finite."""
    col = _build_column("convective")
    ncol = 1
    args = [
        jnp.array(col["u"][None, :]), jnp.array(col["v"][None, :]),
        jnp.array(col["T"][None, :]), jnp.array(col["q"][None, :]),
        jnp.array(col["p_mid"][None, :]), jnp.array(col["p_half"][None, :]),
        jnp.array(col["z_mid"][None, :]), jnp.array(col["z_half"][None, :]),
        jnp.array([col["T"][-1] + 4.0]), jnp.array([col["q"][-1] + 0.01]),
        jnp.array((col["p_mid"] / (RAIR * col["T"]))[None, :]),
    ]
    out = holtslag_boville_turbulence(*args, 300.0, HoltslagBovilleConfig())
    for arr in (out.du_dt, out.dv_dt, out.dT_dt, out.dq_v_dt, out.Km, out.Kh, out.h_pbl):
        assert jnp.all(jnp.isfinite(arr))
    assert 100.0 < float(out.h_pbl[0]) < 4000.0  # physical convective PBL


def test_grad_finite_through_full_scheme():
    """jax.grad of a scalar loss is finite for all stratification inputs."""
    col = _build_column("stable")  # stable -> exercises the cube-root / sqrt edges
    base = [
        jnp.array(col["u"][None, :]), jnp.array(col["v"][None, :]),
        jnp.array(col["T"][None, :]), jnp.array(col["q"][None, :]),
        jnp.array(col["p_mid"][None, :]), jnp.array(col["p_half"][None, :]),
        jnp.array(col["z_mid"][None, :]), jnp.array(col["z_half"][None, :]),
        jnp.array([col["T"][-1] - 2.0]), jnp.array([col["q"][-1]]),
        jnp.array((col["p_mid"] / (RAIR * col["T"]))[None, :]),
    ]

    def loss(T):
        a = list(base); a[2] = T
        o = holtslag_boville_turbulence(*a, 300.0, HoltslagBovilleConfig())
        return jnp.sum(o.dT_dt ** 2) + jnp.sum(o.h_pbl)

    g = jax.grad(loss)(base[2])
    assert jnp.all(jnp.isfinite(g))


def test_grad_matches_finite_difference():
    """AD gradient w.r.t. surface T matches a centered finite difference.

    Verifies gradient CORRECTNESS (not just finiteness) where the scheme has
    a broad, non-trivial sensitivity: the surface temperature drives the
    bulk fluxes -> kbfs -> the whole convective K-profile + PBL height.
    """
    col = _build_column("convective")
    base = [
        jnp.array(col["u"][None, :]), jnp.array(col["v"][None, :]),
        jnp.array(col["T"][None, :]), jnp.array(col["q"][None, :]),
        jnp.array(col["p_mid"][None, :]), jnp.array(col["p_half"][None, :]),
        jnp.array(col["z_mid"][None, :]), jnp.array(col["z_half"][None, :]),
        jnp.array([col["T"][-1] + 4.0]), jnp.array([col["q"][-1] + 0.01]),
        jnp.array((col["p_mid"] / (RAIR * col["T"]))[None, :]),
    ]

    def loss(T_sfc):
        a = list(base); a[8] = T_sfc
        o = holtslag_boville_turbulence(*a, 300.0, HoltslagBovilleConfig())
        return jnp.sum(o.dT_dt ** 2) * 1e6 + jnp.sum(o.h_pbl)

    T_sfc = base[8]
    g_ad = float(jax.grad(loss)(T_sfc)[0])
    eps = 1e-3
    g_fd = float((loss(T_sfc + eps) - loss(T_sfc - eps)) / (2 * eps))
    rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-8)
    assert rel < 1e-4, f"AD grad {g_ad:.6e} vs FD {g_fd:.6e} (rel {rel:.2e})"


def test_float32_prognostics_with_float64_column_inputs(fx):
    """The coupled lane under x64 hands float64 pressures/heights/surface fields
    next to float32 winds/T/q (#2110); the solve must follow the winds' dtype
    instead of raising on mixed dtypes."""
    f32 = {k: jnp.asarray(fx[f"in_{k}"], jnp.float32) for k in ("T", "q", "u", "v")}
    f64 = {k: jnp.asarray(fx[f"in_{k}"], jnp.float64)
           for k in ("p_full", "p_half", "zm", "zi")}
    T64, q64 = f32["T"].astype(jnp.float64), f32["q"].astype(jnp.float64)
    rho = compute_rho(T64, f64["p_full"], q_v=q64)
    out = holtslag_boville_turbulence(
        f32["u"], f32["v"], f32["T"], f32["q"], f64["p_full"], f64["p_half"],
        f64["zm"], f64["zi"], T64[:, -1], q64[:, -1], rho, float(fx["dt"]),
        HoltslagBovilleConfig(),
        pref_mid=jnp.asarray(fx["in_pref_mid"], jnp.float64))
    for arr in (out.du_dt, out.dv_dt, out.dT_dt, out.dq_v_dt):
        assert arr.dtype == jnp.float32
    for arr in (out.du_dt, out.dv_dt, out.dT_dt, out.dq_v_dt):
        assert jnp.all(jnp.isfinite(arr))
