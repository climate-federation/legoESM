#!/usr/bin/env python
"""Single-column-model (SCM) microphysics validation on DYCOMS-II + BOMEX.

Drives EVERY switchable warm/ice microphysics scheme in a single column set up
from the canonical DYCOMS-II RF01 (stratocumulus, Stevens et al. 2005 MWR 133)
and BOMEX (shallow cumulus, Siebesma et al. 2003 JAS 60) thermodynamic
soundings + their large-scale forcing, and checks, for each scheme × case ×
precision (float64 and float32):

  * STABILITY — no NaN/Inf; T, q_v, q_c stay in physical bounds over the run.
  * PHYSICAL CONSISTENCY — total water (q_v + condensate + accumulated surface
    precip) conserved against the large-scale source/sink; cloud water forms in
    the saturated layer; liquid-water path in the literature order of magnitude
    (DYCOMS ~40-90 g/m², BOMEX ~3-12 g/m²); precip non-negative.
  * PRECISION — float64 and float32 both run and agree to a loose tolerance.

This is NOT a turbulent SCM — it does not parameterize the boundary-layer
fluxes that maintain the LES cloud; it exercises the MICROPHYSICS on the
case's realistic cloudy column + large-scale forcing, which is what stresses
each scheme's condensation / autoconversion / accretion / sedimentation /
ice-phase paths for stability and conservation. The cloud layer is initialized
saturated (DYCOMS) or seeded with the case's in-cloud condensate (BOMEX) so the
schemes have condensate to process.

Run:  python scripts/validate/scm_microphysics_cases.py [--scheme NAME] [--case dycoms|bomex]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.dynamics.les.spectral_les_moist import (
    make_anelastic_reference,
)
from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
from legoesm.atmosphere.physics.microphysics.integration import (
    get_microphysics_fn,
    min_tracer_slots,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio


# Schemes with a self-contained warm/ice column tendency (ml_emulator needs a
# trained net; "none" is a no-op — both excluded).
SCHEMES = ("kessler", "sundqvist", "seifert_beheng", "morrison",
           "thompson", "p3", "sdm", "fast_sbm")

# Literature liquid-water-path ranges [g/m²].
#  * DYCOMS is OVERCAST (cloud fraction ≈ 1), so a single saturated column ≈
#    the grid-mean LES value — directly comparable to Stevens et al. 2005
#    (~40-80 g/m²); the band is widened for the schemes' spread.
#  * BOMEX is BROKEN cumulus (cloud fraction ≈ 0.1); a single column cannot
#    represent that fraction, so the SCM tests the IN-CLOUD column. Siebesma
#    2003 in-cloud LWP is ~10-80 g/m² (grid-mean ~5-10 = fraction × in-cloud).
LWP_REF = {"dycoms": (20.0, 120.0), "bomex": (5.0, 90.0)}
# Physically-representative initial in-cloud liquid for the BOMEX cumulus
# (Siebesma 2003: q_c ~ 1e-4 kg/kg near cloud base) over a thin near-base
# layer → in-cloud LWP ~20 g/m² to start; the schemes then process it.
_BOMEX_QC_SEED = 1.0e-4
_BOMEX_CLOUD = (520.0, 700.0)


def _dycoms_column(nz=60, ztop=1500.0):
    """Canonical DYCOMS-II RF01 column (Stevens 2005): θ_l, q_t, winds."""
    z_f = np.linspace(0.0, ztop, nz + 1)
    z_c = 0.5 * (z_f[:-1] + z_f[1:])
    z_i = 840.0
    thl = np.where(z_c < z_i, 289.0, 297.5 + (z_c - z_i) ** (1.0 / 3.0))
    qt = np.where(z_c < z_i, 9.0e-3, 1.5e-3)
    p_sfc = 1017.8e2
    return z_c, z_f, thl, qt, p_sfc, dict(
        Nc=140.0e6, shf=15.0, lhf=115.0, div=3.75e-6, z_i=z_i,
        kappa_rad=85.0, F0=70.0, F1=22.0)


def _bomex_column(nz=75, ztop=3000.0):
    """Canonical BOMEX column (Siebesma 2003): θ_l, q_t piecewise-linear."""
    z_f = np.linspace(0.0, ztop, nz + 1)
    z_c = 0.5 * (z_f[:-1] + z_f[1:])

    def pw(z, pts):
        zs, vs = zip(*pts)
        return np.interp(z, zs, vs)

    thl = pw(z_c, [(0, 298.7), (520, 298.7), (1480, 302.4),
                   (2000, 308.2), (3000, 311.85)])
    qt = pw(z_c, [(0, 17.0e-3), (520, 16.3e-3), (1480, 10.7e-3),
                  (2000, 4.2e-3), (3000, 3.0e-3)])
    p_sfc = 1015.0e2
    return z_c, z_f, thl, qt, p_sfc, dict(
        shf_K_ms=8.0e-3, qf_ms=5.2e-5, w_sub_top=-0.65e-2, z_sub=1500.0,
        cool_K_day=-2.0)


def _saturation_adjust(thl, qt, exner, p, dtype):
    """(θ_l, q_t) → (θ, q_v, q_c) by a damped fixed-point saturation
    adjustment (all-or-nothing: condense the supersaturated excess)."""
    th = jnp.asarray(thl, dtype)        # first guess θ ≈ θ_l (q_c=0)
    qt = jnp.asarray(qt, dtype)
    exner = jnp.asarray(exner, dtype)
    p = jnp.asarray(p, dtype)
    Lv_over_cp = constants.L_v / constants.c_pd
    for _ in range(20):
        T = th * exner
        qsat = saturation_mixing_ratio(T, p)
        qc = jnp.maximum(qt - qsat, 0.0)
        th_new = jnp.asarray(thl, dtype) + (Lv_over_cp * qc) / exner
        th = th + 0.5 * (th_new - th)
    T = th * exner
    qc = jnp.maximum(qt - saturation_mixing_ratio(T, p), 0.0)
    qv = qt - qc
    return th, qv, qc


def _scheme_config(scheme, env):
    cfg = MicrophysicsConfig(scheme=scheme)
    return cfg


def run_case(scheme, case, dtype, n_steps=None, verbose=False):
    """Run one scheme on one case at one precision; return a result dict."""
    jax.config.update("jax_enable_x64", dtype == jnp.float64)
    base_case = case.replace("_cold", "")
    # Cold variants shift θ_l down so the cloud is supercooled (mixed-phase) —
    # exercises freezing / ice / melt in the ice-capable schemes that the warm
    # cases leave untested. q_t is scaled DOWN with Clausius-Clapeyron so the
    # column's relative humidity is PRESERVED (colder air holds far less water;
    # keeping the warm q_t at −20 °C would force ~10× supersaturation and an
    # unphysical 20 g/kg of condensate — a malformed test, not a cold cloud).
    cold_shift = 18.0 if case.endswith("_cold") else 0.0
    if base_case == "dycoms":
        z_c, z_f, thl, qt, p_sfc, env = _dycoms_column()
        dt, hours = 1.0, 1.0
    else:
        z_c, z_f, thl, qt, p_sfc, env = _bomex_column()
        dt, hours = 1.0, 1.0
    if cold_shift > 0.0:
        # Clausius-Clapeyron: qsat ∝ exp(−L_v/R_v · 1/T). Scale q_t by the
        # saturation ratio between the shifted and original cloud-layer
        # temperature so RH is unchanged.
        T_warm = thl   # ≈ T near the surface for the scaling reference
        T_cold = thl - cold_shift
        ratio = np.exp(-constants.L_v / constants.R_v
                       * (1.0 / T_cold - 1.0 / T_warm))
        qt = qt * ratio
        thl = thl - cold_shift
    if n_steps is None:
        n_steps = int(hours * 3600 / dt)
    nz = z_c.shape[0]
    dz = float(z_f[1] - z_f[0])

    ref = make_anelastic_reference(z_c, z_f, p_sfc, thl,
                                   qv_prof=np.minimum(qt, 0.02), dtype=dtype)
    exner = np.asarray(ref.exner_c)
    p_c = np.asarray(ref.p_c)
    th, qv, qc = _saturation_adjust(thl, qt, exner, p_c, dtype)
    # BOMEX is sub-saturated (cumulus condensate comes from updrafts the SCM
    # has no dynamics for). To exercise the microphysics on a SUSTAINED cloud
    # (rather than have it instantly evaporate in the sub-saturated mean — the
    # correct but uninformative SCM behaviour), the thin near-cloud-base layer
    # is brought to SATURATION (RH=1, a cumulus-core column) and seeded with
    # the case's in-cloud liquid (≈20 g/m²). The schemes then process a
    # persistent cloud; total water is augmented (the post-seed column is the
    # conservation baseline).
    if base_case == "bomex":
        lo, hi = _BOMEX_CLOUD
        seed = ((z_c >= lo) & (z_c <= hi))
        qsat = saturation_mixing_ratio(th * jnp.asarray(exner, dtype),
                                       jnp.asarray(p_c, dtype))
        qv = jnp.where(jnp.asarray(seed), qsat, qv)
        qc = qc + jnp.where(jnp.asarray(seed),
                            jnp.asarray(_BOMEX_QC_SEED, dtype), 0.0)

    # Top-down single column (ncol=1, nz).
    flip = lambda a: jnp.asarray(a, dtype)[::-1][None, :]
    T_col = flip(th * exner)
    qv_col = flip(qv)
    qc_col = flip(qc)
    p_full = flip(p_c)
    p_half = jnp.asarray(np.asarray(ref.p_f)[::-1][None, :], dtype)
    rho = flip(np.asarray(ref.rho_c))
    dz_col = jnp.full((1, nz), dz, dtype)

    scheme_name, micro_fn, scheme_cfg = get_microphysics_fn(
        _scheme_config(scheme, env))
    nslots = max(min_tracer_slots(scheme_name), 9)
    Nc0 = env.get("Nc", 1.0e8)

    # State: q_v, q_c, q_r, q_i, q_s, q_g, N_c, N_r, N_i.
    state = dict(
        T=T_col, q_v=qv_col, q_c=qc_col,
        q_r=jnp.zeros((1, nz), dtype), q_i=jnp.zeros((1, nz), dtype),
        q_s=jnp.zeros((1, nz), dtype), q_g=jnp.zeros((1, nz), dtype),
        N_c=jnp.where(qc_col > 1e-6, Nc0, 0.0).astype(dtype),
        N_r=jnp.zeros((1, nz), dtype), N_i=jnp.zeros((1, nz), dtype))

    # Large-scale forcing (per step): radiative cooling drives DYCOMS
    # condensation; both get a uniform clear-sky-ish cooling so the cloud is
    # sustained over the run (simple, case-representative).
    cool = (-1.0 / 86400.0) if base_case == "bomex" else (-2.0 / 86400.0)
    cool_K_s = cool  # K/s applied to T

    def micro_step(st):
        hyd = HydrometeorState(
            q_c=st["q_c"], q_r=st["q_r"], q_i=st["q_i"], q_s=st["q_s"],
            q_g=st["q_g"], N_c=st["N_c"], N_r=st["N_r"], N_i=st["N_i"])
        out = micro_fn(st["T"], st["q_v"], hyd, p_full, p_half, rho, dz_col,
                       dt, scheme_cfg)
        return out

    micro_jit = jax.jit(micro_step)

    # Codex methodology review: the float32 run must be GENUINELY float32
    # (toggling jax_enable_x64 alone is not proof) — assert the scheme output
    # dtype before trusting the precision result.
    out0 = micro_jit(state)
    for fld in (out0.dT_dt, out0.dq_v_dt, out0.dq_c_dt, out0.precipitation):
        if fld.dtype != dtype:
            raise RuntimeError(
                f"precision leak: scheme output dtype {fld.dtype} != {dtype} "
                f"({scheme}/{case}) — the float32 run is not genuine")

    _QWATER = ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g")
    _QNUM = ("N_c", "N_r", "N_i")
    col_int = lambda q: float(jnp.sum(q * rho * dz_col))   # kg/m² (×area=1)
    lwp0 = (col_int(state["q_c"]) + col_int(state["q_r"])) * 1000.0  # g/m²
    water0 = sum(col_int(state[k]) for k in _QWATER)
    accum_precip = 0.0
    nan_step = -1
    clamp_fired = False        # did any scheme drive a WATER field negative?
    neg_min = 0.0              # most-negative pre-clamp WATER excursion (kg/kg)
    max_qc = 0.0               # condensate growth bound
    precip_neg = False
    # Pre-clamp negativity tolerance, scaled PER FIELD CLASS (codex caught the
    # harness applying a water-scaled tol to number fields): a number
    # concentration ~1e6-1e8 /m³ dipping to −1e-2 /m³ is ~1e-9 relative —
    # physically zero, the dycore floors it harmlessly — and must NOT flag as
    # instability. Only a meaningful negative in a WATER mass (kg/kg ~1e-3)
    # signals a real conservation/stability failure.
    qv_scale = max(float(jnp.max(state["q_v"])), 1.0e-6)
    neg_tol_water = -1.0e-6 * qv_scale          # ~ −1.7e-9 kg/kg

    for n in range(n_steps):
        out = micro_jit(state)
        # Check EVERY scheme output finite (not just T/q_v) — a NaN in any
        # hydrometeor or number tendency is an instability.
        all_tend = (out.dT_dt, out.dq_v_dt, out.dq_c_dt, out.dq_r_dt,
                    out.dq_i_dt, out.dq_s_dt, out.dq_g_dt, out.dN_c_dt,
                    out.dN_r_dt, out.dN_i_dt, out.precipitation)
        if not all(bool(jnp.all(jnp.isfinite(a))) for a in all_tend):
            nan_step = n
            break
        if bool(jnp.any(out.precipitation < 0.0)):
            precip_neg = True
        # RAW (pre-clamp) update — inspect for negatives BEFORE flooring so a
        # non-conserving / unstable scheme cannot hide behind the floor.
        raw = {"T": state["T"] + dt * (out.dT_dt + cool_K_s)}
        raw["q_v"] = state["q_v"] + dt * out.dq_v_dt
        raw["q_c"] = state["q_c"] + dt * out.dq_c_dt
        raw["q_r"] = state["q_r"] + dt * out.dq_r_dt
        raw["q_i"] = state["q_i"] + dt * out.dq_i_dt
        raw["q_s"] = state["q_s"] + dt * out.dq_s_dt
        raw["q_g"] = state["q_g"] + dt * out.dq_g_dt
        raw["N_c"] = state["N_c"] + dt * out.dN_c_dt
        raw["N_r"] = state["N_r"] + dt * out.dN_r_dt
        raw["N_i"] = state["N_i"] + dt * out.dN_i_dt
        # WATER fields: a meaningful negative (below the kg/kg tolerance) is a
        # real conservation/stability failure → flag.
        water_min = min(float(jnp.min(raw[k])) for k in _QWATER)
        neg_min = min(neg_min, water_min)
        if water_min < neg_tol_water:
            clamp_fired = True
        # NUMBER fields: a concentration ~1e6-1e8 /m³ dipping a hair below 0
        # (e.g. −1e-2 /m³ from a number-bookkeeping delta) is physically zero
        # and floored harmlessly. Flag only a sign error that is BOTH below an
        # absolute −1 /m³ floor AND large relative to the field — i.e. a real
        # blow-up, not roundoff.
        for k in _QNUM:
            mn = float(jnp.min(raw[k]))
            scale = max(float(jnp.max(jnp.abs(raw[k]))), 1.0)
            if mn < -1.0 and mn < -1.0e-3 * scale:
                clamp_fired = True
        # Commit with a positivity floor (only after recording the excursion).
        for k in _QWATER + _QNUM:
            state[k] = jnp.maximum(raw[k], 0.0)
        state["T"] = raw["T"]
        accum_precip += float(jnp.sum(out.precipitation)) * dt   # kg/m²
        max_qc = max(max_qc, float(jnp.max(state["q_c"])))
        if not bool(jnp.all(jnp.isfinite(state["T"]))):
            nan_step = n
            break

    lwp1 = (col_int(state["q_c"]) + col_int(state["q_r"])) * 1000.0
    water1 = sum(col_int(state[k]) for k in _QWATER)
    water_resid = abs((water1 + accum_precip) - water0) / max(water0, 1e-12)
    Tmin = float(jnp.min(state["T"])) if nan_step < 0 else float("nan")
    Tmax = float(jnp.max(state["T"])) if nan_step < 0 else float("nan")
    lwp_ok = LWP_REF[base_case][0] <= lwp1 <= LWP_REF[base_case][1]
    # STABLE = finite all-fields + T bounds (both ends) + NO clamp fired
    # (water stayed non-negative without help) + non-negative precip +
    # bounded condensate. Conservation residual ≤ 1e-3 (clamps proven not to
    # have masked it, since clamp_fired would already fail).
    stable = (nan_step < 0 and 150.0 < Tmin < 340.0 and Tmax < 340.0
              and not clamp_fired and not precip_neg
              and max_qc < 0.05 and water_resid < 1.0e-3)
    return dict(
        scheme=scheme, case=case, dtype=str(dtype.__name__),
        stable=stable, nan_step=nan_step, lwp0=lwp0, lwp1=lwp1,
        precip_mm=accum_precip, water_resid=water_resid, Tmin=Tmin, Tmax=Tmax,
        clamp_fired=clamp_fired, neg_min=neg_min, max_qc=max_qc,
        precip_neg=precip_neg, lwp_ok=lwp_ok)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheme", default=None, choices=SCHEMES)
    ap.add_argument("--case", default=None,
                    choices=("dycoms", "bomex", "dycoms_cold", "bomex_cold"))
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args(argv)

    schemes = (args.scheme,) if args.scheme else SCHEMES
    # Warm cases (DYCOMS, BOMEX) + cold mixed-phase variants (profile shifted
    # below freezing) so the ICE-capable schemes (morrison/thompson/p3/
    # fast_sbm/seifert) actually exercise their freezing/ice/melt paths — the
    # warm cases alone leave those branches untested (codex coverage finding).
    cases = (args.case,) if args.case else \
        ("dycoms", "bomex", "dycoms_cold", "bomex_cold")
    print(f"{'scheme':15s} {'case':12s} {'prec':4s} {'STABLE':8s} "
          f"{'LWP':8s} {'precip':8s} {'water_res':10s} {'clamp':6s} "
          f"{'qcmax':8s} {'Tmin':6s} {'Tmax':6s} {'LWPok':6s}")
    rows = []
    for scheme in schemes:
        for case in cases:
            res = {}
            for dtype in (jnp.float64, jnp.float32):
                try:
                    r = run_case(scheme, case, dtype, n_steps=args.steps)
                    res[dtype] = r
                except Exception as exc:   # noqa: BLE001
                    print(f"{scheme:15s} {case:12s} {dtype.__name__[-2:]:4s} "
                          f"ERROR: {type(exc).__name__}: {str(exc)[:50]}")
                    rows.append(dict(scheme=scheme, case=case,
                                     dtype=dtype.__name__, error=str(exc)))
                    res[dtype] = None
            for dtype in (jnp.float64, jnp.float32):
                r = res.get(dtype)
                if r is None:
                    continue
                flag = "OK" if r["stable"] else "**UNSTABLE"
                cf = "FIRED" if r["clamp_fired"] else "-"
                print(f"{scheme:15s} {case:12s} {r['dtype'][-2:]:4s} "
                      f"{flag:8s} {r['lwp1']:8.2f} {r['precip_mm']:8.4f} "
                      f"{r['water_resid']:10.2e} {cf:6s} {r['max_qc']:8.2e} "
                      f"{r['Tmin']:6.1f} {r['Tmax']:6.1f} "
                      f"{('y' if r['lwp_ok'] else 'n'):6s}")
                rows.append(r)
            # float32-vs-float64 agreement (a genuine precision validation,
            # not just 'both ran').
            r64, r32 = res.get(jnp.float64), res.get(jnp.float32)
            if r64 and r32 and r64["stable"] and r32["stable"]:
                base = max(abs(r64["lwp1"]), 1.0)
                rel = abs(r32["lwp1"] - r64["lwp1"]) / base
                if rel > 0.15:
                    print(f"{'':15s} {case:12s} f32-vs-f64 LWP DIVERGE "
                          f"rel={rel:.2%}")
    # Summary line.
    good = [r for r in rows if "error" not in r and r.get("stable")]
    bad = [r for r in rows if "error" in r or not r.get("stable", False)]
    print(f"\nSUMMARY: {len(good)} stable / {len(rows)} runs; "
          f"{len(bad)} flagged.")
    if bad:
        for r in bad:
            tag = r.get("error", "unstable")[:50]
            print(f"  FLAG {r['scheme']}/{r['case']}/{r.get('dtype')}: {tag}")
    return rows


if __name__ == "__main__":
    main()
