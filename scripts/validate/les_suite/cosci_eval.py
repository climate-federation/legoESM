"""Experimenter for the turbulence co-scientist: run one candidate closure (or one
baseline closure) through the evaluation stages and write one JSON of numbers.

Stages (``--stage``):

* ``E0`` contract + conservation checks on test columns, jit parity, and the
  gradient of the planned E2 loss w.r.t. every parameter (inert = exactly zero).
* ``L``  idealized limits, no LES needed: neutral surface layer (κ, Pr_t), z-less
  stable scaling (K(z) ratio, Pr_t vs Ri), free-convection mixed layer (Deardorff
  lapse rate, entrainment ratio), neutral Ekman turning. Each metric is first run on
  a known answer (instrument check) and on the listed baselines.
* ``E1`` a priori: the closure's fluxes at every LES frame's mean state vs the LES
  resolved+SGS fluxes.
* ``E2`` a posteriori: SCM from LES t=0 vs the LES trajectory, all output frames,
  θ_l/q_t/u/v and heat/moisture/momentum fluxes weighted equally, regimes weighted
  equally; parameters AD-tuned on the training cases only (Adam, the LES-suite AD
  tuner's budget), scored on training and held-out cases.

A candidate owns either scalars or momentum; ``--baseline`` names the roster closure
that supplies the other part (and the surface stress). ``--baseline-only`` scores a
roster closure instead (default, LES-tuned, and AD-tuned arms) for comparison.

Usage::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/validate/les_suite/cosci_eval.py \\
        --candidate scripts/validate/les_suite/cosci_candidates/hb_equivalent.py \\
        --baseline holtslag_boville --heat-form theta \\
        --stage E2 --artifacts cbl.npz --out e2.json
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from legoesm.atmosphere.les_suite import (  # noqa: E402
    get_case,
    list_cases,
    register_default_catalog,
)
from legoesm.atmosphere.les_suite.bridge import (  # noqa: E402
    load_artifact,
    total_turbulent_flux,
)
from legoesm.atmosphere.les_suite.candidate import column_inputs, load_candidate  # noqa: E402
from legoesm.atmosphere.les_suite.candidate_eval import (  # noqa: E402
    AD_LR,
    AD_STEPS,
    MOMFLUX_FLOOR_USTAR2_FRAC,
    SCM_DT_S,
    SCM_NLEV,
    Fluxes,
    adam_descend_normalized,
    e0_column_checks,
    e2_nsteps,
    e2_score,
    ekman_metrics,
    free_convection_metrics,
    gradient_activity,
    momentum_flux_floor,
    neutral_surface_layer_metrics,
    rollout,
    stable_metrics,
    synthetic_artifact,
)
from legoesm.atmosphere.les_suite.scm_coupling import interp_profile  # noqa: E402
from legoesm.atmosphere.les_suite.scm_runner import (  # noqa: E402
    build_cbl_scm_from_artifact,
    scm_physics_config,
)
from legoesm.atmosphere.les_suite.score import (  # noqa: E402
    PROFILE_FLOOR_FLUX,
    PROFILE_FLOOR_QFLUX,
    normalized_rmse_series,
)
from legoesm.atmosphere.physics.combined import make_physics  # noqa: E402
from legoesm.atmosphere.physics.turbulence.les_tuned import (  # noqa: E402
    LESTunedInapplicableError,
    apply_les_tuned_turbulence,
)
from legoesm.core.profile_metrics import layer_weights_from_heights  # noqa: E402
from legoesm.training.inert_params import assert_no_inert  # noqa: E402

from legoesm import constants  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts" / "run"))
from tune_scm_to_les import (  # noqa: E402
    _base_turbulence,
    _scheme_tunable_metas,
    apply_overrides_to_base,
)

#: Held-out case families (user decision 2026-10-15), matched as substrings of the
#: artifact case name; the classification of every artifact is written to the output.
HELD_OUT_FAMILIES = ("ayotte", "sullivan_patton", "gabls3", "gabls4", "ekman", "rico")

# --- L-stage idealized setups (numbers fixed here and recorded in every output) ---
_L_THETA0 = 300.0
_L_Z0 = 0.1                # m, the tuner base surface roughness
_L_USTAR = 0.3             # m/s, neutral surface-layer log profile
_L_NEUTRAL_WTH = 1.0e-5    # K m/s, a near-zero heat flux so K_h is measurable
_L_SL_ZMAX = 100.0         # m, interfaces treated as surface layer
_L_STABLE_S = 0.01         # 1/s, uniform shear
_L_STABLE_RI = (0.05, 0.1, 0.25, 0.5, 1.0)
_L_STABLE_WTH = -1.0e-3    # K m/s
_L_STABLE_HEIGHTS = (50.0, 100.0, 200.0, 400.0)
_L_CBL_Q0 = 0.06           # K m/s (Nieuwstadt)
_L_CBL_ZI0 = 800.0         # m
_L_CBL_GAMMA = 0.008       # K/m above z_i0
_L_CBL_HOURS = 2.0
_L_EKMAN_G = 10.0          # m/s
_L_EKMAN_F = 1.0e-4        # 1/s
_L_EKMAN_PERIODS = 3       # inertial periods integrated; the last one is averaged


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _regime(case_name: str) -> str:
    if not list_cases():
        register_default_catalog()
    try:
        return get_case(case_name).regime
    except (KeyError, ValueError):
        return "unregistered:" + case_name


def _held_out(case_name: str) -> bool:
    low = case_name.lower()
    return any(f in low for f in HELD_OUT_FAMILIES)


# --- arms ---------------------------------------------------------------------------

class Arm:
    """A closure configuration under evaluation: the baseline turbulence config plus,
    for a candidate arm, the candidate and its parameters."""

    def __init__(self, base_turb, candidate=None, heat_form=None, ke_heating=None):
        self.base_turb = base_turb
        self.candidate = candidate
        self.heat_form = heat_form
        self.ke_heating = ke_heating

    def prepare(self, artifact, *, turb=None, nlev, dt):
        """Build the SCM for ``artifact`` once (concrete config): ``(scm, physics,
        u_geo, v_geo, f_c)`` on the column, reusable across parameter values."""
        turb = self.base_turb if turb is None else turb
        scm, _grid = build_cbl_scm_from_artifact(artifact, turb, les_tuned=False,
                                                nlev=nlev, dt=dt)
        forcing = scm.forcing_arg
        f_c = 0.0 if forcing is None else float(forcing.f_c)

        def _geo(fn):  # the profile the SCM forcing applies, surface-first
            return None if fn is None else jnp.asarray(fn(scm.t_seconds))[::-1]
        ug = None if forcing is None else _geo(forcing.u_geo)
        vg = None if forcing is None else _geo(forcing.v_geo)
        return scm, scm_physics_config(artifact, turb), ug, vg, f_c

    def run_prepared(self, prep, nsteps, params=None):
        scm, phys, ug, vg, f_c = prep
        return rollout(scm, phys, nsteps, candidate=self.candidate, params=params,
                       heat_form=self.heat_form, ke_heating=self.ke_heating,
                       f_c=f_c, u_geo=ug, v_geo=vg)

    def run_with_turb(self, prep, artifact, nsteps, turb):
        """BASELINE rollout of a column prepared with the concrete base, with its
        physics rebuilt from ``turb`` (which may be traced), so it traces under
        ``jit``. State, grid, forcing and the initial physics state are the
        prepared ones: they do not depend on any tier-1 tunable (they do depend on
        floors such as ``tke_min``, which are not tuned). Equality with a full
        rebuild at perturbed values is tested."""
        if self.candidate is not None:
            raise ValueError("run_with_turb is for baseline arms (no candidate)")
        scm, _, ug, vg, f_c = prep
        cfg = scm_physics_config(artifact, turb)
        col = copy.copy(scm)
        col.physics_fn = make_physics(cfg, model_type="hydrostatic", dt=scm.dt)
        return rollout(col, cfg, nsteps, f_c=f_c, u_geo=ug, v_geo=vg)

    def run(self, artifact, nsteps, *, params=None, turb=None, nlev, dt):
        return self.run_prepared(self.prepare(artifact, turb=turb, nlev=nlev, dt=dt),
                                 nsteps, params)


def candidate_case_loss(arm, artifact, *, nlev, dt):
    """The E2 loss of one case as a jitted function of the candidate parameters: the
    SCM is built once and the rollout compiled once, so every tuning step reuses it."""
    prep = arm.prepare(artifact, nlev=nlev, dt=dt)
    n = e2_nsteps(artifact, dt)

    @jax.jit
    def loss(params):
        rec, _ = arm.run_prepared(prep, n, params)
        return e2_score(artifact, rec, dt).combined
    return loss


def _nsteps(artifact, dt) -> int:
    return int(round(float(jnp.asarray(artifact.times_s)[-1]) / dt))


# --- L stage ---------------------------------------------------------------------------

def _l_neutral(arm, nlev, dt):
    z = jnp.arange(10.0, 2000.0, 20.0)
    kap = constants.kappa_von_karman
    tstar = -_L_NEUTRAL_WTH / _L_USTAR
    art = synthetic_artifact(
        "L_neutral_sl", z, _L_THETA0 + tstar / kap * jnp.log(z / _L_Z0),
        _L_USTAR / kap * jnp.log(z / _L_Z0), jnp.zeros_like(z),
        w_theta_s=_L_NEUTRAL_WTH, hours=1.0, nframes=2)
    rec, _ = arm.run(art, 1, params=arm_params(arm), nlev=nlev, dt=dt)
    return rec, neutral_surface_layer_metrics(rec.at(0), rec.fluxes_of_step(0),
                                              z_max=_L_SL_ZMAX)


def _l_stable(arm, nlev, dt):
    z = jnp.arange(10.0, 1000.0, 20.0)
    out = {}
    for ri in _L_STABLE_RI:
        gamma = ri * _L_THETA0 * _L_STABLE_S ** 2 / constants.g
        art = synthetic_artifact(
            f"L_stable_ri{ri:g}", z, _L_THETA0 + gamma * z, 1.0 + _L_STABLE_S * z,
            jnp.zeros_like(z), w_theta_s=_L_STABLE_WTH, hours=1.0, nframes=2)
        rec, _ = arm.run(art, 1, params=arm_params(arm), nlev=nlev, dt=dt)
        out[f"Ri={ri:g}"] = stable_metrics(rec.at(0), rec.fluxes_of_step(0),
                                           _L_STABLE_HEIGHTS)
    return out


def _l_cbl(arm, nlev, dt):
    z = jnp.arange(10.0, 2000.0, 20.0)
    theta = _L_THETA0 + _L_CBL_GAMMA * jnp.maximum(z - _L_CBL_ZI0, 0.0)
    art = synthetic_artifact("L_free_convection", z, theta, jnp.zeros_like(z),
                             jnp.zeros_like(z), w_theta_s=_L_CBL_Q0,
                             hours=_L_CBL_HOURS, nframes=13)
    n = _nsteps(art, dt)
    rec, _ = arm.run(art, n + 1, params=arm_params(arm), nlev=nlev, dt=dt)
    col = rec.at(n)  # profile and flux of the same column, at the end time
    return free_convection_metrics(col.theta_l, col.z_full, rec.fluxes_of_step(n).wtheta,
                                   col.z_half, q0=_L_CBL_Q0, theta0=_L_THETA0)


def _l_ekman(arm, nlev, dt):
    z = jnp.arange(10.0, 3000.0, 20.0)
    hours = _L_EKMAN_PERIODS * 2.0 * math.pi / _L_EKMAN_F / 3600.0
    art = synthetic_artifact("L_neutral_ekman", z, jnp.full_like(z, _L_THETA0),
                             jnp.full_like(z, _L_EKMAN_G), jnp.zeros_like(z),
                             w_theta_s=0.0, hours=hours, nframes=2, f_c=_L_EKMAN_F,
                             u_geo=_L_EKMAN_G, v_geo=0.0)
    n = int(round(hours * 3600.0 / dt))
    rec, _ = arm.run(art, n, params=arm_params(arm), nlev=nlev, dt=dt)
    last = int(round(2.0 * math.pi / _L_EKMAN_F / dt))
    mean = lambda a: jnp.mean(a[-last:], axis=0)  # noqa: E731
    return ekman_metrics(mean(rec.inputs.u), mean(rec.inputs.v), mean(rec.fluxes.uw),
                         mean(rec.fluxes.vw), mean(rec.inputs.z_half), ug=_L_EKMAN_G, vg=0.0)


def arm_params(arm):
    return None if arm.candidate is None else arm.candidate.default_params()


def instrument_checks(artifacts) -> dict:
    """Each L metric on an input with a known answer."""
    out = {}
    z = jnp.arange(10.0, 400.0, 20.0)
    zh = jnp.concatenate([jnp.zeros(1), 0.5 * (z[1:] + z[:-1]), z[-1:] + 10.0])
    kap, ust, prt = constants.kappa_von_karman, _L_USTAR, 0.85
    u = ust / kap * jnp.log(z / _L_Z0)
    th = _L_THETA0 + 0.001 * jnp.log(z / _L_Z0)
    g_u = (u[1:] - u[:-1]) / (z[1:] - z[:-1])
    g_t = (th[1:] - th[:-1]) / (z[1:] - z[:-1])
    km = kap * ust * zh[1:-1]
    uw = jnp.concatenate([jnp.asarray([-ust ** 2]), -km * g_u, jnp.zeros(1)])
    wt = jnp.concatenate([jnp.zeros(1), -(km / prt) * g_t, jnp.zeros(1)])
    one = jnp.ones_like(z)
    inp = column_inputs(z_full=z, z_half=zh, rho_full=one, rho_half=jnp.ones_like(zh),
                        mass=one, exner=one, exner_half=jnp.ones_like(zh), p_full=one,
                        p_half=jnp.ones_like(zh), theta_l=th, q_t=0 * z, u=u, v=0 * z,
                        w_theta_s=0.0, w_qt_s=0.0, uw_s=-ust ** 2, vw_s=0.0, dt=1.0)
    fl = Fluxes(wtheta=wt, wqt=0 * zh, uw=uw, vw=0 * zh)
    sl = neutral_surface_layer_metrics(inp, fl, z_max=_L_SL_ZMAX)
    out["neutral_sl_known"] = {"expected_kappa": kap, "expected_Pr_t": prt, **sl}
    known_k = (("z_less", lambda zz: 5.0 + 0 * zz), ("z_squared", lambda zz: 1e-3 * zz ** 2))
    for label, kfun in known_k:
        kk = kfun(zh[1:-1])
        fl2 = Fluxes(wtheta=jnp.concatenate([jnp.zeros(1), -kk * g_t, jnp.zeros(1)]),
                     wqt=0 * zh, uw=jnp.concatenate([jnp.zeros(1), -kk * g_u, jnp.zeros(1)]),
                     vw=0 * zh)
        sm = stable_metrics(inp, fl2, _L_STABLE_HEIGHTS)
        exp = 1.0 if label == "z_less" else (sm["z"][-1] / sm["z"][0]) ** 2
        out[f"stable_{label}_known"] = {"expected_ratio": exp,
                                        "K_m_ratio": sm["K_m_ratio_top_bottom"]}
    # analytic constant-K Ekman spiral (no-slip): alpha(z1) and the 5 % stress depth
    k_e, f, g_e = 5.0, _L_EKMAN_F, _L_EKMAN_G
    d = math.sqrt(2.0 * k_e / f)
    ze = jnp.arange(10.0, 3000.0, 20.0)
    zeh = jnp.concatenate([jnp.zeros(1), 0.5 * (ze[1:] + ze[:-1]), ze[-1:] + 10.0])
    zz = ze / d
    ue = g_e * (1.0 - jnp.exp(-zz) * jnp.cos(zz))
    ve = g_e * jnp.exp(-zz) * jnp.sin(zz)
    zh_ = zeh / d
    dudz = g_e / d * jnp.exp(-zh_) * (jnp.cos(zh_) + jnp.sin(zh_))
    dvdz = g_e / d * jnp.exp(-zh_) * (jnp.cos(zh_) - jnp.sin(zh_))
    em = ekman_metrics(ue, ve, -k_e * dudz, -k_e * dvdz, zeh, ug=g_e, vg=0.0)
    a1 = math.degrees(math.atan2(float(ve[0]), float(ue[0])))
    out["ekman_analytic_known"] = {"expected_turning_deg": a1,
                                   "expected_h_m": d * math.log(20.0), **em}
    # free-convection metric on a column with known z_i, entrainment ratio and lapse
    zf = jnp.arange(10.0, 2000.0, 20.0)
    zi_k, ratio_k, lapse_k = 1010.0, -0.2, -0.5e-3   # m, 1, K/m
    wtf = jnp.where(zf <= zi_k, _L_CBL_Q0 * (1.0 + (ratio_k - 1.0) * zf / zi_k),
                    jnp.minimum(0.0, ratio_k * _L_CBL_Q0 * (1.0 - (zf - zi_k) / 200.0)))
    fc_k = free_convection_metrics(_L_THETA0 + lapse_k * zf, zf, wtf, zf, q0=_L_CBL_Q0,
                                   theta0=_L_THETA0)
    out["free_convection_known"] = {
        "expected_z_i": zi_k, "expected_entrainment_ratio": ratio_k,
        "expected_lapse_K_per_km": 1e3 * lapse_k,
        "expected_w_star": (constants.g * _L_CBL_Q0 * zi_k / _L_THETA0) ** (1.0 / 3.0),
        **fc_k}
    # the same metric applied to LES truth (a reported number, not a check),
    # next to the gate-0 record for reading side by side
    for path, art in artifacts:
        if art.case_name != "cbl_nieuwstadt":
            continue
        wt_les = total_turbulent_flux(art.wtheta_resolved, art.wtheta_sgs)[-1]
        zl = jnp.asarray(art.heights_m)
        fc = free_convection_metrics(jnp.asarray(art.theta)[-1], zl, wt_les, zl,
                                     q0=float(jnp.asarray(art.w_theta_s)[-1]),
                                     theta0=_L_THETA0)
        g0 = _REPO / "docs/atmosphere/les_suite/gate0_nieuwstadt_result.json"
        out[f"free_convection_on_les:{Path(path).name}"] = {
            **fc, "gate0_record": (json.loads(g0.read_text()).get("diagnostics", {})
                                   if g0.exists() else f"missing: {g0}")}
    return out


def stage_l(arms: dict, artifacts, nlev, dt) -> dict:
    res = {"instrument": instrument_checks(artifacts), "arms": {}}
    for name, arm in arms.items():
        _rec, sl = _l_neutral(arm, nlev, dt)
        res["arms"][name] = {"neutral_surface_layer": sl,
                             "stable": _l_stable(arm, nlev, dt),
                             "free_convection": _l_cbl(arm, nlev, dt),
                             "neutral_ekman": _l_ekman(arm, nlev, dt)}
    return res


# --- E1 -------------------------------------------------------------------------------

#: Artifact channels indexed by output time: 2-D (nt, nz) profile series, and these
#: 1-D (nt,) surface series. Selected by NAME, never by matching a dimension length.
_TIME_SERIES_1D = ("w_theta_s", "w_qv_s", "T_s", "tau_x_s", "tau_y_s")
_TIME_INTEGRALS = ("w_theta_s_cum", "tau_x_s_cum", "tau_y_s_cum")
_STATIC_FIELDS = ("case_name", "sgs", "heights_m", "times_s", "f_c", "prescribe",
                  "les_setup", "u_geo", "v_geo", "subsidence_w", "theta_adv", "qv_adv")


def _frame_artifact(art, k):
    """A two-frame reference whose t=0 is frame ``k`` (the a-priori state); the surface
    series hold their frame-``k`` value and the flux integral is rebuilt to match."""
    sel = jnp.asarray([k, k])
    fields = {}
    for f in dataclasses.fields(art):
        val = getattr(art, f.name)
        if val is None or f.name in _STATIC_FIELDS:
            continue
        if f.name in _TIME_INTEGRALS:
            fields[f.name] = None
        elif f.name in _TIME_SERIES_1D or jnp.asarray(val).ndim == 2:
            fields[f.name] = jnp.asarray(val)[sel]
        else:
            raise ValueError(f"_frame_artifact: unclassified artifact field {f.name!r}")
    fields["times_s"] = jnp.asarray([0.0, 3600.0])
    if art.w_theta_s_cum is not None:
        fields["w_theta_s_cum"] = fields["times_s"] * jnp.asarray(art.w_theta_s)[k]
    return dataclasses.replace(art, **fields)


def stage_e1(arms: dict, artifacts, nlev, dt) -> dict:
    res = {}
    for path, art in artifacts:
        z = jnp.asarray(art.heights_m)
        w = layer_weights_from_heights(z)
        nt = int(jnp.asarray(art.times_s).shape[0])
        truth = {"wtheta": total_turbulent_flux(art.wtheta_resolved, art.wtheta_sgs)[1:]}
        momflux_floor = momentum_flux_floor(art)
        floors = {"wtheta": PROFILE_FLOOR_FLUX, "wqt": PROFILE_FLOOR_QFLUX,
                  "uw": momflux_floor, "vw": momflux_floor}
        if art.is_moist:
            truth["wqt"] = total_turbulent_flux(art.wqt_resolved, art.wqt_sgs)[1:]
        if momflux_floor is not None:
            truth["uw"] = total_turbulent_flux(art.uw_resolved, art.uw_sgs)[1:]
            truth["vw"] = total_turbulent_flux(art.vw_resolved, art.vw_sgs)[1:]
        per_arm = {}
        for name, arm in arms.items():
            model = {k: [] for k in truth}
            for k in range(1, nt):
                rec, _ = arm.run(_frame_artifact(art, k), 1, params=arm_params(arm),
                                 nlev=nlev, dt=dt)
                fl = rec.fluxes_of_step(0)
                for key in truth:
                    model[key].append(interp_profile(getattr(fl, key), rec.at(0).z_half, z))
            per = {k: float(normalized_rmse_series(jnp.stack(model[k]), truth[k], w,
                                                   floors[k])) for k in truth}
            per["combined"] = math.sqrt(sum(v ** 2 for v in per.values()) / len(per))
            per_arm[name] = per
        res[Path(path).name] = per_arm
    return res


# --- E2 -------------------------------------------------------------------------------

def _case_scores(arm, artifacts, nlev, dt, params=None, turb=None):
    out = {}
    for path, art in artifacts:
        n = e2_nsteps(art, dt)
        rec, _ = arm.run(art, n, params=params, turb=turb, nlev=nlev, dt=dt)
        s = e2_score(art, rec, dt)
        out[Path(path).name] = {"per_var": {k: float(v) for k, v in s.per_var.items()},
                                "combined": float(s.combined)}
    return out


def tune_active(loss_of, lo, hi, x0, names, *, steps, lr):
    """AD-tune under the NO-INERT rule. The first gradient of the training loss is
    measured: a parameter with an exactly-zero gradient is inactive on these cases,
    so it is FROZEN at its default and removed from the optimised set (the
    ``_inactive_keys`` pattern, decided by measurement as in
    ``legoesm.training.inert_params``); every remaining parameter must then pass
    ``assert_no_inert`` (a non-finite gradient refuses the run).

    Returns ``(x_best, record)`` with ``x_best`` the FULL parameter vector."""
    g = jax.grad(loss_of)(x0)
    active = [i for i in range(len(names)) if float(g[i]) != 0.0]
    frozen = [names[i] for i in range(len(names)) if i not in active]
    try:
        assert_no_inert({f"{i}:{names[i]}": g[i] for i in active})
    except ValueError as err:
        raise SystemExit(f"E2 tuning refused: {err}") from err
    record = {"frozen_inactive": frozen,
              "init_grad": {n: float(g[i]) for i, n in enumerate(names)}}
    if not active:
        return x0, {**record, "not_tuned": "every parameter is inactive on the "
                                           "training cases"}
    idx = jnp.asarray(active)
    best, best_loss, d_loss, hist = adam_descend_normalized(
        lambda xa: loss_of(x0.at[idx].set(xa)), lo[idx], hi[idx], x0[idx],
        steps=steps, lr=lr)
    return x0.at[idx].set(best), {**record, "train_loss_default": d_loss,
                                  "train_loss_best": best_loss, "history": hist}


def baseline_loss(arm, base, fields, train, *, nlev, dt):
    """Regime-mean E2 training loss of a roster closure as a function of its tunable
    values, compiled ONCE: the SCM build and rollout are traced with the parameter
    vector, so every tuning step reuses one executable instead of re-lowering the
    rollout (which exhausted XLA's CPU JIT on CLUBB)."""
    preps = {path: arm.prepare(art, nlev=nlev, dt=dt) for path, art in train}

    @jax.jit
    def loss_of(x):
        turb = apply_overrides_to_base(base, {f: x[i] for i, f in enumerate(fields)})

        def per_case(path, art):
            rec, _ = arm.run_with_turb(preps[path], art, e2_nsteps(art, dt), turb)
            return e2_score(art, rec, dt).combined
        return _regime_mean_loss(per_case, train)
    return loss_of


def les_tuned_arm(base):
    """The LES-tuned config of ``base``, or ``(None, reason)`` when the tuned
    coefficients do not apply to this base (e.g. CLUBB coefficients fit on the
    prognostic path vs the tuner's diagnostic base) -- recorded, not fatal."""
    try:
        tuned = apply_les_tuned_turbulence(base)
    except LESTunedInapplicableError as err:
        return None, f"not applicable: {err}"
    return (tuned, None) if tuned != base else (None, "no LES-tuned entry")


def _regime_mean_loss(scores_fn, train):
    """Regimes weighted equally: mean over regimes of the mean over their cases."""
    by_reg: dict = {}
    for path, art in train:
        by_reg.setdefault(_regime(art.case_name), []).append(scores_fn(path, art))
    return sum(sum(v) / len(v) for v in by_reg.values()) / len(by_reg)


def stage_e2(cand_arm, base_names, artifacts, nlev, dt, steps, lr,
             tune: bool) -> dict:
    train = [(p, a) for p, a in artifacts if not _held_out(a.case_name)]
    res = {"split": {Path(p).name: {"held_out": _held_out(a.case_name),
                                    "regime": _regime(a.case_name)}
                     for p, a in artifacts}, "arms": {}}

    if cand_arm is not None:
        cand = cand_arm.candidate
        names, lo, hi = cand.bounds()
        x0 = jnp.asarray([float(cand.params[k][0]) for k in names])
        to_params = lambda x: {k: x[i] for i, k in enumerate(names)}  # noqa: E731
        entry = {"default": _case_scores(cand_arm, artifacts, nlev, dt,
                                         params=cand.default_params())}
        base_rec_diff = {}
        for path, art in artifacts:
            n = e2_nsteps(art, dt)
            rc, _ = cand_arm.run(art, n, params=cand.default_params(), nlev=nlev, dt=dt)
            rb, _ = Arm(cand_arm.base_turb).run(art, n, nlev=nlev, dt=dt)
            base_rec_diff[Path(path).name] = {
                k: float(jnp.max(jnp.abs(getattr(rc.inputs, k) - getattr(rb.inputs, k))))
                for k in ("theta_l", "q_t", "u", "v")}
        entry["max_abs_diff_vs_baseline_rollout"] = base_rec_diff
        if tune and train and names:
            case_loss = {path: candidate_case_loss(cand_arm, art, nlev=nlev, dt=dt)
                         for path, art in train}

            def loss_of(x):
                p = to_params(x)
                return _regime_mean_loss(lambda path, art: case_loss[path](p), train)
            best, record = tune_active(loss_of, lo, hi, x0, names, steps=steps, lr=lr)
            tuned = to_params(best)
            entry["ad"] = {"params": {k: float(v) for k, v in tuned.items()}, **record,
                           "scores": _case_scores(cand_arm, artifacts, nlev, dt,
                                                  params=tuned)}
        res["arms"]["candidate:" + cand.name] = entry
    for name in base_names:
        base, key = _base_turbulence(name)
        arm = Arm(base)
        entry = {"default": _case_scores(arm, artifacts, nlev, dt)}
        tuned_turb, why_not = les_tuned_arm(base)
        entry["les_tuned"] = ({"not_scored": why_not} if tuned_turb is None else
                              _case_scores(arm, artifacts, nlev, dt, turb=tuned_turb))
        if tune and train:
            metas = _scheme_tunable_metas(key, (1,))
            fields = [m.field for m in metas]
            lo = jnp.asarray([float(m.bounds[0]) for m in metas])
            hi = jnp.asarray([float(m.bounds[1]) for m in metas])
            x0 = jnp.asarray([float(m.default) for m in metas])

            loss_of = baseline_loss(arm, base, fields, train, nlev=nlev, dt=dt)
            best, record = tune_active(loss_of, lo, hi, x0, fields, steps=steps, lr=lr)
            ov = {f: float(best[i]) for i, f in enumerate(fields)}
            entry["ad"] = {"params": ov, **record,
                           "scores": _case_scores(arm, artifacts, nlev, dt,
                                                  turb=apply_overrides_to_base(base, ov))}
        res["arms"]["baseline:" + name] = entry
    return res


# --- E0 -------------------------------------------------------------------------------

def stage_e0(cand_arm, artifacts, nlev, dt) -> dict:
    cand = cand_arm.candidate
    params = cand.default_params()
    cols = {}
    _rec, _ = _l_neutral(Arm(cand_arm.base_turb), nlev, dt)
    cols["neutral_surface_layer"] = _rec.at(0)
    z = jnp.arange(10.0, 1000.0, 20.0)
    gamma = 0.25 * _L_THETA0 * _L_STABLE_S ** 2 / constants.g
    st = synthetic_artifact("E0_stable", z, _L_THETA0 + gamma * z, 1.0 + _L_STABLE_S * z,
                            jnp.zeros_like(z), w_theta_s=_L_STABLE_WTH, hours=1.0, nframes=2)
    cols["stable_ri0.25"] = Arm(cand_arm.base_turb).run(st, 1, nlev=nlev, dt=dt)[0].at(0)
    for path, art in artifacts:
        cols["t0:" + Path(path).name] = Arm(cand_arm.base_turb).run(
            art, 1, nlev=nlev, dt=dt)[0].at(0)
    res = {"columns": {k: e0_column_checks(cand, params, v, heat_form=cand_arm.heat_form,
                                           ke_heating=cand_arm.ke_heating)
                       for k, v in cols.items()}}
    train = [(p, a) for p, a in artifacts if not _held_out(a.case_name)]
    if train and cand.params:
        case_loss = {path: candidate_case_loss(cand_arm, art, nlev=nlev, dt=dt)
                     for path, art in train}

        def loss(p):
            return _regime_mean_loss(lambda path, art: case_loss[path](p), train)
        res["gradient_activity_on_E2_loss"] = gradient_activity(loss, params)
    else:
        res["gradient_activity_on_E2_loss"] = (
            "not run: needs >=1 training artifact and >=1 parameter")
    return res


# --- main -----------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("E0", "L", "E1", "E2"))
    p.add_argument("--candidate", type=Path, default=None)
    p.add_argument("--baseline", default=None,
                   help="roster closure supplying the non-owned part (required with --candidate)")
    p.add_argument("--heat-form", choices=("theta", "enthalpy"), default=None,
                   help="scalar candidates only (required): how the heat flux is applied")
    p.add_argument("--ke-heating", choices=("on", "off"), default=None,
                   help="momentum candidates only (required): KE dissipation as heat")
    p.add_argument("--baseline-only", nargs="*", default=[],
                   help="roster closures to score (default/LES-tuned/AD-tuned arms)")
    p.add_argument("--artifacts", nargs="*", type=Path, default=[])
    p.add_argument("--ad-steps", type=int, default=AD_STEPS)
    p.add_argument("--ad-lr", type=float, default=AD_LR)
    p.add_argument("--nlev", type=int, default=SCM_NLEV)
    p.add_argument("--dt", type=float, default=SCM_DT_S)
    p.add_argument("--no-tune", action="store_true", help="E2 without AD tuning")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.candidate is None and not args.baseline_only:
        p.error("give --candidate and/or --baseline-only")
    if args.candidate is not None and args.baseline is None:
        p.error("--candidate needs --baseline (no default)")
    if args.stage == "E0" and args.candidate is None:
        p.error("--stage E0 checks a candidate; give --candidate")
    return args


def check_kind_choices(kind: str, heat_form, ke_heating) -> None:
    """Each candidate kind states exactly the choice that acts on it: a scalar
    candidate its heat form, a momentum candidate its dissipation heating."""
    if kind == "scalar" and (heat_form is None or ke_heating is not None):
        raise SystemExit("a scalar candidate needs --heat-form and takes no --ke-heating")
    if kind == "momentum" and (ke_heating is None or heat_form is not None):
        raise SystemExit("a momentum candidate needs --ke-heating and takes no --heat-form")


def main(argv=None) -> int:
    args = parse_args(argv)
    if not jax.config.read("jax_enable_x64"):
        raise SystemExit("run with JAX_ENABLE_X64=1 (conservation checks need float64)")
    artifacts = [(str(p), load_artifact(p)) for p in args.artifacts]
    cand_arm = None
    arms = {}
    if args.candidate is not None:
        cand = load_candidate(args.candidate)
        check_kind_choices(cand.kind, args.heat_form, args.ke_heating)
        base, _key = _base_turbulence(args.baseline)
        cand_arm = Arm(base, cand, args.heat_form,
                       None if args.ke_heating is None else args.ke_heating == "on")
        arms["candidate:" + cand.name] = cand_arm
    for name in args.baseline_only:
        arms["baseline:" + name] = Arm(_base_turbulence(name)[0])
    if args.stage == "E0":
        result = stage_e0(cand_arm, artifacts, args.nlev, args.dt)
    elif args.stage == "L":
        result = stage_l(arms, artifacts, args.nlev, args.dt)
    elif args.stage == "E1":
        result = stage_e1(arms, artifacts, args.nlev, args.dt)
    else:
        result = stage_e2(cand_arm, args.baseline_only, artifacts, args.nlev, args.dt,
                          args.ad_steps, args.ad_lr, not args.no_tune)
    record = {
        "stage": args.stage, "result": result,
        "provenance": {
            "git_sha": _git_sha(), "argv": sys.argv[1:] if argv is None else argv,
            "candidate": None if args.candidate is None else {
                "path": str(args.candidate), "sha256": _sha256(args.candidate),
                "kind": cand_arm.candidate.kind, "params": cand_arm.candidate.params},
            "baseline": args.baseline, "heat_form": args.heat_form,
            "ke_heating": args.ke_heating,
            "momflux_floor_ustar2_frac": MOMFLUX_FLOOR_USTAR2_FRAC,
            "nlev": args.nlev, "dt": args.dt, "ad_steps": args.ad_steps, "ad_lr": args.ad_lr,
            "baseline_only": list(args.baseline_only),
            "held_out_families": HELD_OUT_FAMILIES,
            "l_stage_constants": {k: v for k, v in globals().items()
                                  if k.startswith("_L_")},
            "artifacts": {p: _sha256(p) for p, _ in artifacts},
            "jax": jax.__version__, "x64": bool(jax.config.read("jax_enable_x64")),
            "platform": jax.default_backend(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, default=float))
    print(f"[cosci_eval] stage={args.stage} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
