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
# Minimum |ON - OFF| that counts as load-bearing, as a FRACTION of the OFF
# tendency.  A merely-nonzero difference can be floating-point noise from a
# reordered expression; the allowance changes the deposition driving
# supersaturation by ~90 % in this cell, so anything below a percent means the
# flag is not reaching the process it is supposed to govern.
MIN_RELATIVE_EFFECT = 0.01


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
    T = np.linspace(180.0, 300.0, 241)
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


def _report_effect(label: str, on: float, off: float, *, expect_retains: bool,
                   failures: list[str]) -> None:
    """Print an ON/OFF pair and judge only its MECHANICAL properties."""
    delta = on - off
    rel = abs(delta) / max(abs(off), 1.0e-30)
    print(f"    {label:34s} ON={on:+.6e}  OFF={off:+.6e}  "
          f"delta={delta:+.6e} ({100 * rel:6.2f} % of OFF)")
    if rel < MIN_RELATIVE_EFFECT:
        failures.append(
            f"{label}: |ON-OFF| is {100 * rel:.3f} % of the tendency, below the "
            f"{100 * MIN_RELATIVE_EFFECT:.0f} % floor — the flag is not reaching "
            "the process it governs (or this column is the wrong regime)")
        return
    if expect_retains and delta <= 0.0:
        failures.append(
            f"{label}: enabling the ice allowance removed MORE vapour "
            f"(delta={delta:+.3e}); it must RETAIN vapour")
    if (not expect_retains) and delta >= 0.0:
        failures.append(
            f"{label}: enabling the liquid guard removed LESS vapour "
            f"(delta={delta:+.3e}); it must condense the excess faster")


def section_2_load_bearing() -> list[str]:
    """ON vs OFF through the real schemes — proves the flags are not inert.

    Every ice-carrying scheme is exercised, not just one: a table of default-
    True config fields (section 3) says nothing about a scheme whose
    implementation ignores the field.
    """
    failures: list[str] = []

    # --- ICE half: one cold cell, all three ice schemes ----------------------
    q_sat_i = float(thermo.saturation_mixing_ratio_ice(
        jnp.asarray(COLD_T_K), jnp.asarray(COLD_P_PA)))
    q_v, rest = _column(COLD_T_K, COLD_P_PA, COLD_RH_ICE * q_sat_i,
                        q_i=COLD_Q_ICE)
    ramp = float(thermo.homogeneous_freezing_rh_factor(jnp.asarray(COLD_T_K)))
    print(f"  cold cell: T={COLD_T_K} K, p={COLD_P_PA / 100:.0f} hPa, "
          f"q_sat_ice={q_sat_i:.3e} kg/kg, RH_ice={COLD_RH_ICE:.2f}, "
          f"rh_homo={ramp:.4f}, q_ice={COLD_Q_ICE:.1e} kg/kg (below the "
          f"{GSAM_QCI_GATE:.0e} gate, so the allowance still applies)")
    for scheme in ICE_SCHEMES:
        on, off = _on_off(scheme, ICE_FIELD, q_v, rest)
        _report_effect(f"{scheme}: {ICE_FIELD}", on, off,
                       expect_retains=True, failures=failures)

    # --- LIQUID half: one warm cell, every scheme that carries the guard -----
    q_sat_l = float(thermo.saturation_mixing_ratio(
        jnp.asarray(WARM_T_K), jnp.asarray(WARM_P_PA)))
    q_vw, restw = _column(WARM_T_K, WARM_P_PA, WARM_RH_LIQ * q_sat_l)
    print(f"  warm cell: T={WARM_T_K} K, p={WARM_P_PA / 100:.0f} hPa, "
          f"q_sat_liq={q_sat_l:.3e} kg/kg, RH_liq={WARM_RH_LIQ:.2f}")
    for scheme in ICE_SCHEMES:
        on, off = _on_off(scheme, LIQUID_FIELD, q_vw, restw)
        _report_effect(f"{scheme}: {LIQUID_FIELD}", on, off,
                       expect_retains=False, failures=failures)
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
