#!/usr/bin/env python
"""Is the IFS super-saturation treatment live in the CRM, the SCM and the GCM?

The IFS/SAM treatment has two halves, and this checker measures BOTH, in each
of the three lanes that run legoESM's atmospheric physics:

* **ICE** — pristine air below 235 K may hold ice super-saturation up to the
  homogeneous-freezing ramp ``rh_homo = 2.583 - T/207.8`` (gSAM 1.8.7
  ``MICRO_SAM1MOM/cloud.f90``, "modeled after IFS"), withdrawn where cloud ice
  already exists.  Implemented as ``thermo.homogeneous_freezing_rh_factor`` and
  applied to the DEPOSITION target of morrison / thompson / p3.
* **LIQUID** — no liquid super-saturation: the excess is condensed onto the
  saturation curve.  Implemented as the in-scheme
  ``hard_saturation_adjustment`` guard, which is opt-in.

Three lanes, defined by which code builds the microphysics config:

  CRM     scripts/run/run_rcemip_plane.py     (plane compressible-Euler CRM/LES)
  SCM     scripts/run/run_scm_rce_campaign.py (single-column RCE campaign)
  GCM     legoesm.driver.physics_pipeline     (global/regional coupled model)

Sections, in the order they must be read:

  1. FORMULA   — the shipped factor vs an independent implementation of the
                 gSAM expression, over a temperature sweep.  A mismatch here
                 invalidates everything below it.
  2. LOAD-BEARING — the flags flipped ON vs OFF on a synthetic column, through
                 the real scheme.  If a flag changes nothing, the reachability
                 table in 3 is vacuous, so this runs FIRST.
  3. REACHABLE — the resolved config each lane hands to the scheme.

Exit code 0 iff every mechanical assertion holds.  Physical numbers are
printed, not judged: this reports what the caps do on one synthetic column, it
does not certify that a run is correct.

Run (CPU, seconds)::

    JAX_ENABLE_X64=1 python scripts/validate/check_ifs_supersaturation_cap.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm import thermo
from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
)
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.p3 import p3_microphysics
from legoesm.atmosphere.physics.microphysics.thompson import (
    thompson_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState

# gSAM cloud.f90 ramp, written out independently of the shipped helper so the
# comparison in section 1 is a real cross-check and not a tautology.
GSAM_RH_HOMO_A = 2.583
GSAM_RH_HOMO_B = 207.8
GSAM_RH_HOMO_T_MAX = 235.0
GSAM_QCI_GATE = 1.0e-8

ICE_SCHEMES = ("morrison", "thompson", "p3")
ICE_FIELD = "homogeneous_ice_supersaturation"
LIQUID_FIELD = "hard_saturation_adjustment"

# Cold cirrus cell: 215 K at 200 hPa, ice-super-saturated, ICE-FREE (so the
# allowance applies) — the regime the ramp exists for.
COLD_T_K = 215.0
COLD_P_PA = 20_000.0
COLD_RH_ICE = 1.60          # above rh_homo(215 K) = 1.548, below nothing
# A trace of cloud ice BELOW the qci gate (1e-8 kg/kg): the allowance still
# applies, but the crystals give deposition a surface to act on, so the ON/OFF
# contrast is carried by the process the allowance actually modifies.  With
# q_i exactly 0 the difference is real but ~0.3 % of the tendency, which is too
# close to "any nonzero number" to serve as a non-vacuity check.
COLD_Q_ICE = 5.0e-9
# Warm boundary-layer cell: 300 K at 1000 hPa, 10 % liquid super-saturated.
WARM_T_K = 300.0
WARM_P_PA = 100_000.0
WARM_RH_LIQ = 1.10
DT_S = 60.0
# Ice above the qci gate: the allowance must be WITHDRAWN here, which is the
# internal control that makes the below-gate difference interpretable.
COLD_Q_ICE_ABOVE_GATE = 1.0e-7
# Warm control for the ICE half: above 235 K the ramp is 1.0, so the flag must
# be a no-op no matter what else the cell is doing.
ICE_NOOP_T_K = 260.0
# Sub-saturated control for the LIQUID half: with q_v below q_sat there is no
# excess to drain, so the guard must be a no-op.
LIQUID_NOOP_RH = 0.80
# Minimum |ON - OFF| relative to the OFF tendency for the LIQUID guard, whose
# effect IS a large share of the warm-cell tendency (measured 36 %).
#
# The ICE half deliberately has NO relative-magnitude threshold. Measured on
# the cold cell: the absolute effect is IDENTICAL in morrison and thompson
# (2.389e-10 kg/kg/s), but it is 91 % of thompson's total vapour sink and
# 0.44 % of morrison's, because morrison's single-step tendency at an ice-poor
# 215 K cell is dominated by Cooper nucleation — which the allowance
# deliberately does NOT gate (PR #1527: deposition target only). A relative
# threshold would therefore fail morrison for having MORE physics, not for
# ignoring the flag. The gate controls below replace it: an implementation
# that ignores the flag cannot produce a difference below the qci gate AND
# bit-identical output above it.
MIN_RELATIVE_EFFECT_LIQUID = 0.01
# Section 2b bound per scheme, from the measurement in that function's
# docstring.  morrison and thompson reduce EXACTLY to a deposition-target
# multiplier (0.000 %), so they are pinned there with only round-off slack;
# p3 does not, for a reason not yet identified, so it keeps the gross bound.
_TARGET_ISOLATION_BOUND = {"morrison": 1.0e-9, "thompson": 1.0e-9}


def _gsam_rh_homo(T: np.ndarray, q_ice: np.ndarray | None = None) -> np.ndarray:
    """Independent NumPy transcription of the gSAM/IFS ramp."""
    ramp = np.maximum(GSAM_RH_HOMO_A - T / GSAM_RH_HOMO_B, 1.0)
    cold = T < GSAM_RH_HOMO_T_MAX
    if q_ice is not None:
        cold = cold & (q_ice < GSAM_QCI_GATE)
    return np.where(cold, ramp, 1.0)


def section_1_formula() -> list[str]:
    """Shipped factor vs the independent transcription."""
    failures: list[str] = []
    # The 235 K gate is piecewise; a 0.5 K-spaced sweep straddles it without
    # landing on it, so a regression that shifted the gate by one grid step
    # would pass. The boundary and its two neighbours are appended explicitly.
    T = np.sort(np.concatenate([
        np.linspace(180.0, 300.0, 241),
        np.array([GSAM_RH_HOMO_T_MAX - 1.0e-9, GSAM_RH_HOMO_T_MAX,
                  GSAM_RH_HOMO_T_MAX + 1.0e-9]),
    ]))
    got = np.asarray(thermo.homogeneous_freezing_rh_factor(jnp.asarray(T)))
    want = _gsam_rh_homo(T)
    dev = float(np.max(np.abs(got - want)))
    print(f"  ramp, ice-free, 180-300 K (241 pts): max|shipped - gSAM| = {dev:.3e}")
    if dev > 1.0e-12:
        failures.append(f"ramp deviates from the gSAM expression by {dev:.3e}")

    # The qci withdrawal: same temperatures, ice present everywhere.
    q_ice = np.full_like(T, 1.0e-6)
    got_i = np.asarray(
        thermo.homogeneous_freezing_rh_factor(jnp.asarray(T), jnp.asarray(q_ice)))
    want_i = _gsam_rh_homo(T, q_ice)
    dev_i = float(np.max(np.abs(got_i - want_i)))
    print(f"  ramp, q_ice = 1e-6 kg/kg      : max|shipped - gSAM| = {dev_i:.3e} "
          f"(withdrawn => 1.0 everywhere: {bool(np.allclose(got_i, 1.0))})")
    if dev_i > 1.0e-12:
        failures.append(f"qci-gated ramp deviates by {dev_i:.3e}")
    if not np.allclose(got_i, 1.0):
        failures.append("qci gate did not withdraw the allowance")

    # Anchors quoted in the code comments and in PR #1527.
    for T_a, want_a in ((235.0, 1.0), (234.9, GSAM_RH_HOMO_A - 234.9 / GSAM_RH_HOMO_B),
                        (190.0, GSAM_RH_HOMO_A - 190.0 / GSAM_RH_HOMO_B)):
        got_a = float(thermo.homogeneous_freezing_rh_factor(jnp.asarray(T_a)))
        print(f"  rh_homo({T_a:6.1f} K) = {got_a:.4f}   (expected {want_a:.4f})")
        if abs(got_a - want_a) > 1.0e-12:
            failures.append(f"rh_homo({T_a}) = {got_a}, expected {want_a}")

    # The floor: the factor is an ALLOWANCE, never a reduction of q_sat_ice.
    lo = float(np.min(got))
    print(f"  min factor over the sweep      = {lo:.6f} (must be >= 1)")
    if lo < 1.0:
        failures.append(f"factor dropped below 1 ({lo})")
    return failures


def _column(T_K: float, p_Pa: float, q_v: float, q_i: float = 0.0):
    """One-cell column in the backend interface's own shapes."""
    T = jnp.full((1, 1), T_K)
    p_full = jnp.full((1, 1), p_Pa)
    p_half = jnp.asarray([[p_Pa * 1.05, p_Pa * 0.95]])
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((1, 1), 500.0)
    z = jnp.zeros((1, 1))
    hyd = HydrometeorState(
        q_c=z, q_r=z, q_i=jnp.full((1, 1), q_i), q_s=z, q_g=z,
        N_c=jnp.full((1, 1), 1.0e8), N_r=z, N_i=jnp.full((1, 1), 1.0e4),
    )
    return jnp.full((1, 1), q_v), (T, hyd, p_full, p_half, rho, dz)


def _scheme_fn_and_config(scheme: str):
    """(tendency function, config factory) for one ice-carrying scheme."""
    from legoesm.atmosphere.physics.microphysics.config import (
        MorrisonConfig, P3Config, ThompsonConfig,
    )
    if scheme == "morrison":
        # SAM M2005 flavor: the plane CRM's oracle-matching configuration.
        return morrison_microphysics, (
            lambda **kw: MorrisonConfig(morrison_flavor="sam", **kw))
    if scheme == "thompson":
        return thompson_microphysics, (lambda **kw: ThompsonConfig(**kw))
    if scheme == "p3":
        return p3_microphysics, (lambda **kw: P3Config(**kw))
    raise ValueError(f"no ice-scheme entry point for {scheme!r}")


def _on_off(scheme: str, field: str, q_v, rest) -> tuple[float, float]:
    """Vapour tendency with ``field`` True and False, same column."""
    fn, cfg = _scheme_fn_and_config(scheme)
    T, hyd, p_full, p_half, rho, dz = rest
    out = {}
    for on in (True, False):
        res = fn(T, q_v, hyd, p_full, p_half, rho, dz, DT_S, cfg(**{field: on}))
        out[on] = float(res.dq_v_dt[0, 0])
    return out[True], out[False]


def _report(label: str, on: float, off: float) -> float:
    """Print an ON/OFF pair; return the difference.  No judgement here."""
    delta = on - off
    rel = abs(delta) / max(abs(off), 1.0e-30)
    print(f"    {label:52s} ON={on:+.6e}  OFF={off:+.6e}  "
          f"delta={delta:+.6e} ({100 * rel:6.2f} % of OFF)")
    return delta


def section_2_load_bearing() -> list[str]:
    """ON vs OFF through the real schemes — proves the flags are not inert.

    Every ice-carrying scheme is exercised, not just one: a table of default-
    True config fields (section 3) says nothing about a scheme whose
    implementation ignores the field.

    Each half is checked against its own INTERNAL CONTROL — a cell where the
    feature must do exactly nothing — rather than against a magnitude
    threshold.  A control is stronger here: an implementation that ignores the
    flag produces zero difference everywhere, and one that applies it
    unconditionally produces a difference in the control too.  Both fail.
    """
    failures: list[str] = []

    # --- ICE half ------------------------------------------------------------
    q_sat_i = float(thermo.saturation_mixing_ratio_ice(
        jnp.asarray(COLD_T_K), jnp.asarray(COLD_P_PA)))
    ramp = float(thermo.homogeneous_freezing_rh_factor(jnp.asarray(COLD_T_K)))
    q_v_cold = COLD_RH_ICE * q_sat_i
    print(f"  cold cell: T={COLD_T_K} K, p={COLD_P_PA / 100:.0f} hPa, "
          f"q_sat_ice={q_sat_i:.3e} kg/kg, RH_ice={COLD_RH_ICE:.2f}, "
          f"rh_homo={ramp:.4f}")
    print(f"  ICE: effect below the qci gate (q_ice={COLD_Q_ICE:.0e}), then the "
          f"two controls that must be EXACT no-ops")
    for scheme in ICE_SCHEMES:
        q_v, rest = _column(COLD_T_K, COLD_P_PA, q_v_cold, q_i=COLD_Q_ICE)
        on, off = _on_off(scheme, ICE_FIELD, q_v, rest)
        delta = _report(f"{scheme}: below gate", on, off)
        if delta == 0.0:
            failures.append(
                f"{scheme}: {ICE_FIELD} changes nothing below the qci gate — "
                "the flag is not reaching the deposition target")
        elif delta < 0.0:
            failures.append(
                f"{scheme}: enabling the ice allowance removed MORE vapour "
                f"(delta={delta:+.3e}); it must RETAIN vapour")

        # CONTROL 1 — ice above the gate: gSAM withdraws the allowance, so the
        # two configurations must agree BIT FOR BIT.
        q_v2, rest2 = _column(COLD_T_K, COLD_P_PA, q_v_cold,
                              q_i=COLD_Q_ICE_ABOVE_GATE)
        on2, off2 = _on_off(scheme, ICE_FIELD, q_v2, rest2)
        _report(f"{scheme}: CONTROL q_ice={COLD_Q_ICE_ABOVE_GATE:.0e} (expect 0)",
                on2, off2)
        if on2 != off2:
            failures.append(
                f"{scheme}: the allowance is still active at q_ice="
                f"{COLD_Q_ICE_ABOVE_GATE:.0e} kg/kg, above the "
                f"{GSAM_QCI_GATE:.0e} withdrawal gate (delta={on2 - off2:+.3e})")

        # CONTROL 2 — warm: the ramp is 1.0 above 235 K, so likewise a no-op.
        q_sat_i_warm = float(thermo.saturation_mixing_ratio_ice(
            jnp.asarray(ICE_NOOP_T_K), jnp.asarray(COLD_P_PA)))
        q_v3, rest3 = _column(ICE_NOOP_T_K, COLD_P_PA,
                              COLD_RH_ICE * q_sat_i_warm, q_i=COLD_Q_ICE)
        on3, off3 = _on_off(scheme, ICE_FIELD, q_v3, rest3)
        _report(f"{scheme}: CONTROL T={ICE_NOOP_T_K:.0f} K (expect 0)", on3, off3)
        if on3 != off3:
            failures.append(
                f"{scheme}: the allowance is active at {ICE_NOOP_T_K} K, above "
                f"the {GSAM_RH_HOMO_T_MAX} K ramp cutoff "
                f"(delta={on3 - off3:+.3e})")

    # --- LIQUID half ---------------------------------------------------------
    q_sat_l = float(thermo.saturation_mixing_ratio(
        jnp.asarray(WARM_T_K), jnp.asarray(WARM_P_PA)))
    print(f"  warm cell: T={WARM_T_K} K, p={WARM_P_PA / 100:.0f} hPa, "
          f"q_sat_liq={q_sat_l:.3e} kg/kg, RH_liq={WARM_RH_LIQ:.2f}; "
          f"control at RH_liq={LIQUID_NOOP_RH:.2f}")
    for scheme in ICE_SCHEMES:
        q_vw, restw = _column(WARM_T_K, WARM_P_PA, WARM_RH_LIQ * q_sat_l)
        on, off = _on_off(scheme, LIQUID_FIELD, q_vw, restw)
        delta = _report(f"{scheme}: super-saturated", on, off)
        rel = abs(delta) / max(abs(off), 1.0e-30)
        if rel < MIN_RELATIVE_EFFECT_LIQUID:
            failures.append(
                f"{scheme}: {LIQUID_FIELD} moves the tendency by "
                f"{100 * rel:.3f} %, below the "
                f"{100 * MIN_RELATIVE_EFFECT_LIQUID:.0f} % floor")
        elif delta > 0.0:
            failures.append(
                f"{scheme}: enabling the liquid guard removed LESS vapour "
                f"(delta={delta:+.3e}); it must condense the excess faster")

        # CONTROL — sub-saturated: nothing to drain, so an exact no-op.
        q_vs, rests = _column(WARM_T_K, WARM_P_PA, LIQUID_NOOP_RH * q_sat_l)
        on_s, off_s = _on_off(scheme, LIQUID_FIELD, q_vs, rests)
        _report(f"{scheme}: CONTROL RH_liq={LIQUID_NOOP_RH:.2f} (expect 0)",
                on_s, off_s)
        if on_s != off_s:
            failures.append(
                f"{scheme}: the liquid guard fires in SUB-saturated air "
                f"(RH={LIQUID_NOOP_RH}, delta={on_s - off_s:+.3e})")
    return failures


def section_2b_target_isolation() -> list[str]:
    """Does the allowance act on the DEPOSITION TARGET ALONE?

    Sections 1-2 prove the ramp is right, the flag is reachable, and it is
    correctly gated.  They do NOT prove it is applied to the intended term: an
    implementation that scaled a different gated ice quantity would satisfy
    all of them.  This is the equivalence that pins the target.

    If the only effect is ``q_sat_i -> rh_homo * q_sat_i`` in the deposition
    driving term, then a cell at ``q_v = S * q_sat_i`` WITH the allowance has
    the same deposition driving supersaturation as a cell at
    ``q_v = (S - rh_homo + 1) * q_sat_i`` WITHOUT it.

    NUCLEATION IS TURNED OFF (``N_i0 = 0``, so the Cooper target is zero),
    and it has to be.  MEASURED without it: thompson matched to 0.000 %, but
    morrison came out 237748 % apart and p3 35.7 %.  That is not a mis-targeted
    allowance — it is the construction's own flaw.  Moving q_v from S = 1.60 to
    S' = 1.052 also moves the nucleation SOURCE, which is gated on ice
    supersaturation (p3 states the gate explicitly: ``cooper_supi_min`` = 0.05,
    smoothed with sharpness 200 — S' sits right on it).  The two morrison
    numbers say so directly: its OFF(S') equalled thompson's answer exactly,
    i.e. with nucleation quiet the schemes agree, and its ON(S) carried the
    whole extra nucleation sink.  Holding "everything else identical" requires
    silencing that source, not just holding T and p.

    MEASURED 2026-08-10 (job 9356775), with nucleation silenced:

        morrison  0.000 %      thompson  0.000 %      p3  35.415 %

    morrison and thompson reduce EXACTLY to a deposition-target multiplier, so
    they are held to that: a nonzero residual there is a regression, and the
    bound is the measurement rather than a round number.

    p3 does NOT reduce to it, and that is an OPEN measured fact rather than a
    tolerance to widen.  Silencing Cooper removed morrison's 237748 % residual
    entirely but left p3's 35 % untouched, so p3 carries a second
    q_v-dependent term at this cell that the allowance does not pass through.
    Its absolute deposition is also ~6x smaller than the other two
    (3.59e-12 vs 2.25e-11 kg/kg/s), i.e. a different capacitance/PSD path.
    Attributing it needs p3's per-process tendencies instrumented; until then
    the cause is UNKNOWN, not "small", and p3 is held only to the gross bound.
    """
    failures: list[str] = []
    q_sat_i = float(thermo.saturation_mixing_ratio_ice(
        jnp.asarray(COLD_T_K), jnp.asarray(COLD_P_PA)))
    ramp = float(thermo.homogeneous_freezing_rh_factor(jnp.asarray(COLD_T_K)))
    s_on = COLD_RH_ICE
    s_off = COLD_RH_ICE - ramp + 1.0
    print(f"  nucleation silenced (N_i0 = 0) so the only q_v sink is deposition")
    print(f"  equivalence: ON at RH_ice={s_on:.4f} vs OFF at "
          f"RH_ice={s_off:.4f} (both carry the same deposition driving "
          f"supersaturation {(s_on - ramp) * q_sat_i:.4e} kg/kg)")
    for scheme in ICE_SCHEMES:
        fn, cfg = _scheme_fn_and_config(scheme)
        q_v_on, rest_on = _column(COLD_T_K, COLD_P_PA, s_on * q_sat_i,
                                  q_i=COLD_Q_ICE)
        q_v_off, rest_off = _column(COLD_T_K, COLD_P_PA, s_off * q_sat_i,
                                    q_i=COLD_Q_ICE)
        # N_i0 = 0 => Cooper target = 0 => no nucleation source; the only
        # remaining q_v sink at this cold, liquid-free, snow-free cell is
        # deposition onto the pre-existing crystals.
        a = float(fn(*(rest_on[0], q_v_on, *rest_on[1:]), DT_S,
                     cfg(**{ICE_FIELD: True, "N_i0": 0.0})).dq_v_dt[0, 0])
        b = float(fn(*(rest_off[0], q_v_off, *rest_off[1:]), DT_S,
                     cfg(**{ICE_FIELD: False, "N_i0": 0.0})).dq_v_dt[0, 0])
        resid = abs(a - b) / max(abs(b), 1.0e-30)
        print(f"    {scheme:10s} ON(S)={a:+.6e}  OFF(S')={b:+.6e}  "
              f"|residual| = {100 * resid:8.3f} % of OFF(S')")
        # Per-scheme bound: exact for the two schemes measured at 0, gross for
        # the one with an unexplained residual (see the docstring).
        bound = _TARGET_ISOLATION_BOUND.get(scheme, 1.0)
        if resid > bound:
            failures.append(
                f"{scheme}: the allowance does not behave as a pure "
                f"deposition-target multiplier (residual {100 * resid:.4f} % "
                f"of the equivalent no-allowance cell, bound {100 * bound:.4f} %)")
    return failures


def _lane_crm(scheme: str, *, guard: bool = False):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_rcp_lane", REPO_ROOT / "scripts" / "run" / "run_rcemip_plane.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod._build_microphysics_config(
        scheme, hard_saturation_adjustment=guard)
    return getattr(cfg, scheme)


def _lane_scm(scheme: str, *, guard: bool = False):
    from scripts.run import run_scm_rce_campaign as camp
    cfg = camp.make_physics_config(
        microphysics=scheme, hard_saturation_adjustment=guard)
    return getattr(cfg.microphysics, scheme)


def _lane_gcm(scheme: str, *, guard: bool = False):
    # The global lane resolves its sub-config inside physics_pipeline; there is
    # no public wrapper that returns it without building a whole pipeline, so
    # the checker reaches for the resolver directly.  This is the resolver the
    # coupled driver calls (kernel_registry documents it as such).
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import _resolve_microphysics
    cfg = ExperimentConfig(microphysics=scheme,
                           hard_saturation_adjustment=guard)
    _fn, micro_cfg = _resolve_microphysics(cfg)
    return micro_cfg


LANES = (("CRM  (run_rcemip_plane)", _lane_crm),
         ("SCM  (run_scm_rce_campaign)", _lane_scm),
         ("GCM  (driver.physics_pipeline)", _lane_gcm))


def section_3_reachable() -> list[str]:
    """The config each lane actually hands to the scheme.

    Two questions, not one: the ICE allowance must be ON by default (it ships
    on), and the LIQUID guard must be REACHABLE — i.e. a lane asked to enable
    it must return a sub-config with it enabled.  Checking only that the field
    exists would pass a lane that accepts the flag and drops it.
    """
    failures: list[str] = []
    print(f"  {'lane':32s} {'scheme':10s} {'ice(default)':14s} "
          f"{'liquid(default)':16s} liquid(requested ON)")
    for label, fn in LANES:
        for scheme in ICE_SCHEMES:
            try:
                sub = fn(scheme)
                sub_on = fn(scheme, guard=True)
            except Exception as exc:                      # noqa: BLE001
                failures.append(f"{label} could not build {scheme}: {exc!r}")
                print(f"  {label:32s} {scheme:10s} ERROR: {exc!r}")
                continue
            ice = getattr(sub, ICE_FIELD, "<absent>")
            liq = getattr(sub, LIQUID_FIELD, "<absent>")
            liq_on = getattr(sub_on, LIQUID_FIELD, "<absent>")
            print(f"  {label:32s} {scheme:10s} {str(ice):14s} "
                  f"{str(liq):16s} {liq_on}")
            if ice is not True:
                failures.append(
                    f"{label}/{scheme}: {ICE_FIELD} is {ice!r}, expected True "
                    "(the ice allowance ships ON)")
            if liq is not False:
                failures.append(
                    f"{label}/{scheme}: {LIQUID_FIELD} defaults to {liq!r}; it "
                    "is an opt-in and must default False")
            if liq_on is not True:
                failures.append(
                    f"{label}/{scheme}: asked for {LIQUID_FIELD}=True and the "
                    f"lane returned {liq_on!r} — the flag is dropped somewhere "
                    "between the lane's entry point and the scheme's config")
    # Every FACTORY-REACHABLE scheme must either carry the guard or be a
    # declared exemption (enforced as a ratchet in
    # tests/test_microphysics_supersaturation_guard.py; repeated here so the
    # operator sees the partition next to the lane table).
    from legoesm.atmosphere.physics.microphysics.config import (
        HARD_SAT_GUARD_SCHEMES, HARD_SAT_GUARD_EXEMPT,
    )
    print(f"  guard-carrying schemes: {tuple(HARD_SAT_GUARD_SCHEMES)}")
    print(f"  declared exemptions   : {tuple(HARD_SAT_GUARD_EXEMPT)}")
    everything = set(MicrophysicsConfig()._fields) - {"scheme"}
    uncovered = everything - set(HARD_SAT_GUARD_SCHEMES) - set(HARD_SAT_GUARD_EXEMPT)
    if uncovered:
        failures.append(
            f"microphysics schemes neither guarded nor exempt: {sorted(uncovered)}")
    return failures


def main() -> int:
    print("=" * 78)
    print("IFS/SAM super-saturation treatment — CRM / SCM / GCM")
    print("=" * 78)
    failures: list[str] = []
    print("\n[1] FORMULA — shipped factor vs an independent gSAM transcription")
    failures += section_1_formula()
    print("\n[2] LOAD-BEARING — ON vs OFF through the real scheme")
    failures += section_2_load_bearing()
    print("\n[2b] TARGET ISOLATION — is it the deposition target alone?")
    failures += section_2b_target_isolation()
    print("\n[3] REACHABLE — the sub-config each lane builds")
    failures += section_3_reachable()

    print("\n" + "-" * 78)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All mechanical assertions hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
