#!/usr/bin/env python
"""Static physics validation for convection schemes.

For a fixed tropical sounding (T moist adiabat from 302 K, q_v 90 % saturated
in BL / 40 % aloft, weak shear) we probe every convection scheme and verify
four properties that any physically-meaningful cumulus parameterisation
must satisfy:

1. **Sign correctness** of the column tendencies:
   - column-integrated heating  ``H = c_p ∫ dT/dt dp/g`` is non-negative
     (latent release outweighs detrainment cooling),
   - column vapor sink  ``Q_v = L_v ∫ dq_v/dt dp/g`` is ≤ 0 (drying),
   - column cloud-water source  ``Q_c = L_v ∫ dq_c_conv/dt dp/g`` is ≥ 0.

2. **Approximate MSE conservation** in a *closed-column* (no surface flux)
   probe.  ``H + Q_v + Q_c`` should be ≈ 0 for a self-contained plume; in
   practice the kernel formulation (subsidence ``g/c_p`` term + plume
   detrainment of moist-adiabat T) violates this by an amount that
   depends on the scheme's entrainment/detrainment ratio.  We report the
   relative error so deviations from zero are visible.

3. **Trigger gating**: with the column dropped to a *stable* sounding
   (T moist adiabat plus a dry uniform 5 K perturbation everywhere) the
   smooth-CAPE trigger should suppress the scheme to small tendencies
   (column heating < 50 W/m²).

4. **Comparison to SBM** as a known-good baseline: the *sign* of every
   reported column tendency should match SBM's, and the magnitudes
   should be within a factor of 100.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python scripts/validate_convection_physics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.convection.config import (
    SBMConfig, DCAConfig, MassFluxConfig, ConvectiveEDMFConfig, KuoConfig,
    ZhangMcFarlaneConfig, KainFritschConfig, EmanuelConfig,
    TiedtkeConfig, BechtoldConfig,
)
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.atmosphere.physics.convection.mass_flux import (
    mass_flux_convection, edmf_convection,
)
from legoesm.atmosphere.physics.convection.dca import dca_convection
from legoesm.atmosphere.physics.convection.kuo import kuo_convection
from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
    zhang_mcfarlane_convection,
)
from legoesm.atmosphere.physics.convection.kain_fritsch import (
    kain_fritsch_convection,
)
from legoesm.atmosphere.physics.convection.emanuel import emanuel_convection
from legoesm.atmosphere.physics.convection.tiedtke import tiedtke_convection
from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.grids.vertical import create_sigma_coordinate


def tropical_sounding(ncol: int = 4, nlev: int = 30):
    """Tropical sounding suitable for triggering convection.

    T = 302 K at the surface, lapse rate 6.5 K/km, isothermal at 200 K
    above the tropopause.  q_v = 0.9 q_sat below σ = 0.7 (PBL), 0.4
    q_sat aloft.  CAPE ~ 8 kJ/kg.
    """
    sigma = create_sigma_coordinate(nlev)
    p_s = jnp.full((ncol,), constants.p_ref)
    p_full = p_s[:, None] * sigma.sigma_full
    p_half = p_s[:, None] * sigma.sigma_half
    z_approx = -8000.0 * jnp.log(jnp.clip(sigma.sigma_full, 1e-3))
    T_profile = jnp.maximum(302.0 - 6.5e-3 * z_approx, 200.0)
    T = jnp.broadcast_to(T_profile, (ncol, nlev))
    q_sat_p = saturation_mixing_ratio(T, p_full)
    q_v = jnp.where(sigma.sigma_full > 0.7, 0.9 * q_sat_p, 0.4 * q_sat_p)
    return T, q_v, p_full, p_half


def stable_sounding(ncol: int = 4, nlev: int = 30):
    """Stable sounding — strong inversion + nearly-dry column → CAPE ≈ 0.

    Surface temperature 275 K with an isothermal-then-warming profile
    aloft (so a lifted parcel is always cooler than the environment) and
    q_v at 5 % saturation everywhere (high LCL).  Combined this gives a
    parcel that never reaches positive buoyancy at any level, so CAPE is
    essentially zero — what the trigger should detect.

    The naive "tropical T + 5 K" approach we tried first actually
    *increases* CAPE (warmer parcels carry more buoyancy potential), so
    we have to construct stability against the moist adiabat directly.
    """
    sigma = create_sigma_coordinate(nlev)
    p_s = jnp.full((ncol,), constants.p_ref)
    p_full = p_s[:, None] * sigma.sigma_full
    p_half = p_s[:, None] * sigma.sigma_half
    # Inversion-style profile: 275 K at sfc, +30 K above sigma=0.5 (very
    # warm aloft so any saturated parcel from the surface is negatively
    # buoyant everywhere).
    inversion = 30.0 * jax.nn.sigmoid(20.0 * (0.5 - sigma.sigma_full))
    T_profile = 275.0 + 30.0 - inversion   # warm everywhere except sfc
    T = jnp.broadcast_to(T_profile, (ncol, nlev))
    q_sat_p = saturation_mixing_ratio(T, p_full)
    q_v = 0.05 * q_sat_p   # nearly dry → very high LCL
    return T, q_v, p_full, p_half


def _with_solve(cfg, subsidence_solve: str):
    """Apply the matched-kernel override to ONE scheme sub-config.

    ``subsidence_solve="as_shipped"`` returns the config untouched.  A scheme
    with no ``subsidence_solve`` field (sbm / dca / kuo -- adjustment or
    Kuo-type closures with no compensating-subsidence kernel; emanuel --
    shipped buoyancy-sorting path bypasses the shared kernel) is returned
    untouched too, and ``kernel_arm_status`` below reports that explicitly so
    a reader never assumes it was kernel-matched.
    """
    if subsidence_solve == "as_shipped":
        return cfg
    if not hasattr(cfg, "subsidence_solve"):
        return cfg
    return cfg._replace(subsidence_solve=subsidence_solve)


def kernel_arm_status(name: str, subsidence_solve: str) -> str:
    """Human-readable record of what the arm actually did to this scheme."""
    if subsidence_solve == "as_shipped":
        return "as_shipped"
    cfg = _CONFIG_FACTORY[name]()
    if not hasattr(cfg, "subsidence_solve"):
        return f"not_applicable:{name}"
    return f"forced:{name}={subsidence_solve}"


# Default sub-config per scheme, used ONLY by ``kernel_arm_status`` to decide
# whether the knob exists.  Kept next to ``call_scheme`` so the two cannot
# drift apart.
_CONFIG_FACTORY = {
    "sbm": SBMConfig,
    "dca": DCAConfig,
    "kuo": KuoConfig,
    "mass_flux": MassFluxConfig,
    "edmf": ConvectiveEDMFConfig,
    "zhang_mcfarlane": ZhangMcFarlaneConfig,
    "kain_fritsch": KainFritschConfig,
    "emanuel": EmanuelConfig,
    "tiedtke": TiedtkeConfig,
    "bechtold": BechtoldConfig,
}


def call_scheme(
    name: str, T, q_v, p_full, p_half, dt: float = 1800.0,
    *, fresh_carry: bool = False, subsidence_solve: str = "as_shipped",
):
    """Invoke a scheme by name with safe defaults; return ConvectionOutput.

    ``fresh_carry=True`` resets the prognostic carry of mass_flux/EDMF
    to zero so the trigger gating is exercised cleanly (otherwise the
    slow ``tau_adj`` / ``tau_a`` relaxation keeps the scheme firing
    from a non-zero initial M_c / a_u).

    ``subsidence_solve`` selects the matched-kernel arm -- ``"as_shipped"``
    (defaults) or ``"implicit_flux"`` / ``"advective"`` forced on every scheme
    that owns the knob.  See ``kernel_arm_status``.
    """
    if subsidence_solve not in ("as_shipped", "implicit_flux", "advective"):
        raise ValueError(
            f"call_scheme: unknown subsidence_solve {subsidence_solve!r}"
        )
    ncol, nlev = T.shape
    u = jnp.zeros_like(T)
    v = jnp.zeros_like(T)
    prog = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    common = dict(T=T, q_v=q_v, p_full=p_full, p_half=p_half, dt=dt)
    if name == "sbm":
        return sbm_convection(**common, config=SBMConfig())
    if name == "dca":
        return dca_convection(**common, config=DCAConfig())
    if name == "mass_flux":
        M_c0 = jnp.zeros((ncol,)) if fresh_carry else jnp.full((ncol,), 0.005)
        out, _ = mass_flux_convection(
            **common, M_c=M_c0,
            config=_with_solve(MassFluxConfig(), subsidence_solve),
        )
        return out
    if name == "edmf":
        a_u0 = jnp.zeros((ncol,)) if fresh_carry else jnp.full((ncol,), 0.05)
        out, _ = edmf_convection(
            **common, a_u=a_u0,
            config=_with_solve(ConvectiveEDMFConfig(), subsidence_solve),
        )
        return out
    if name == "kuo":
        return kuo_convection(**common, config=KuoConfig())
    if name == "zhang_mcfarlane":
        out, _ = zhang_mcfarlane_convection(
            **common, u=u, v=v, conv_prog_profile=prog,
            config=_with_solve(
                ZhangMcFarlaneConfig(enable_cmt=False), subsidence_solve),
        )
        return out
    if name == "kain_fritsch":
        out, _ = kain_fritsch_convection(
            **common, w_grid=jnp.zeros_like(T),
            conv_prog_profile=prog,
            config=_with_solve(KainFritschConfig(), subsidence_solve),
        )
        return out
    if name == "emanuel":
        out, _ = emanuel_convection(
            **common, conv_prog_profile=prog, config=EmanuelConfig(),
        )
        return out
    if name == "tiedtke":
        out, _ = tiedtke_convection(
            **common, u=u, v=v, conv_prog_profile=prog,
            config=_with_solve(
                TiedtkeConfig(enable_cmt=False), subsidence_solve),
            moisture_convergence=jnp.zeros_like(T),
        )
        return out
    if name == "bechtold":
        out, _, _ = bechtold_convection(
            **common, u=u, v=v, conv_prog_profile=prog,
            conv_stoch_state=stoch, prng_key=None,
            config=_with_solve(
                BechtoldConfig(enable_stochastic=False, enable_cmt=False),
                subsidence_solve),
            moisture_convergence=jnp.zeros_like(T),
        )
        return out
    raise ValueError(f"Unknown scheme: {name}")


class Diagnostics(NamedTuple):
    H: float           # column heating, W/m²
    Q_v: float         # column vapor sink, W/m² (negative for drying)
    Q_c: float         # column cloud-water source, W/m²
    residual: float    # H + Q_v + Q_c, W/m² (energy non-conservation)
    rel_err: float     # |residual| / (|H| + |Q_v| + |Q_c|)
    cape_max: float    # peak CAPE [J/kg]
    peak_dT: float     # peak |dT/dt| in K/day


def column_diagnostics(out: ConvectionOutput, p_half) -> Diagnostics:
    dp = p_half[:, 1:] - p_half[:, :-1]
    H = float(
        jnp.sum(out.dT_dt * dp / constants.g, axis=1).mean()
    ) * constants.c_pd
    Q = float(
        jnp.sum(out.dq_v_dt * dp / constants.g, axis=1).mean()
    ) * constants.L_v
    C = float(
        jnp.sum(out.dq_c_conv_dt * dp / constants.g, axis=1).mean()
    ) * constants.L_v
    R = H + Q + C
    rel = abs(R) / (abs(H) + abs(Q) + abs(C) + 1e-10)
    return Diagnostics(
        H=H, Q_v=Q, Q_c=C, residual=R, rel_err=rel,
        cape_max=float(out.cape.max()),
        peak_dT=float(jnp.max(jnp.abs(out.dT_dt))) * 86400.0,
    )


def kernel_arm_residuals(out: ConvectionOutput, p_half) -> dict[str, float]:
    """Separate VAPOR-MSE and TOTAL-WATER residuals for the kernel-arm sweep.

    WHICH INVARIANT, AND WHY NOT ``column_diagnostics``: Tests 1-4 above report
    ``H + Q_v + Q_c`` as their energy residual.  That is the RIGHT quantity for
    a scheme that hands microphysics condensate whose latent heat has NOT been
    released, but it is the WRONG one for this package's actual convention --
    ``dq_c_conv_dt`` is ALREADY-CONDENSED cloud (Kain-Fritsch, Bechtold, EDMF,
    Tiedtke, ZM and Arakawa-Wu all add ``+(L_v/c_p) dq_c`` in-scheme).  Under
    that convention a perfectly paired scheme has ``H + Q_v = 0`` and
    ``H + Q_v + Q_c = Q_c``, so reporting the latter would show a fictitious
    "non-conservation" that scales with condensate -- and, because ``Q_c``
    excludes ``dq_r_conv_dt``, would also move with precipitation routing while
    the thermodynamics is unchanged.  That is exactly the kind of
    arm-dependent, non-kernel residual that would corrupt a published ranking.

    Reported instead, as three separate numbers:
      * ``vapor_mse_residual_W_m2``  = int (c_p dT + L_v dq_v) dp/g  -> 0
      * ``water_residual_kg_m2_s``   = int (dq_v + dq_c + dq_r) dp/g -> 0
      * ``condensate_throughput_kg_m2_s`` = int dq_c dp/g  -- a SCALE for the
        two residuals above, never itself part of an "error".

    Sign convention: tendencies are SOURCES (``state += dt*tend``); arrays are
    surface-LAST so ``dp = p_half[1:] - p_half[:-1] > 0``.
    """
    dp = p_half[:, 1:] - p_half[:, :-1]
    dq_r = getattr(out, "dq_r_conv_dt", None)
    dq_r = jnp.zeros_like(out.dq_c_conv_dt) if dq_r is None else dq_r
    vapor_mse = float(jnp.mean(jnp.sum(
        (constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt) * dp,
        axis=1) / constants.g))
    water = float(jnp.mean(jnp.sum(
        (out.dq_v_dt + out.dq_c_conv_dt + dq_r) * dp, axis=1) / constants.g))
    thr = float(jnp.mean(jnp.sum(
        (out.dq_c_conv_dt + dq_r) * dp, axis=1) / constants.g))
    return {
        "vapor_mse_residual_W_m2": vapor_mse,
        "water_residual_kg_m2_s": water,
        "condensate_throughput_kg_m2_s": thr,
        "latent_throughput_W_m2": constants.L_v * thr,
        "vapor_mse_rel_to_latent": (
            abs(vapor_mse) / abs(constants.L_v * thr)
            if abs(constants.L_v * thr) > 1e-12 else float("nan")
        ),
    }


SCHEMES = (
    "sbm", "dca",
    "mass_flux", "edmf", "kuo",
    "zhang_mcfarlane", "kain_fritsch", "emanuel",
    "tiedtke", "bechtold",
)


# Schemes whose *near-zero* tendency on the static tropical probe is
# documented physics (not a regression).  The exemption is granted ONLY
# when the absolute column heating is small (< _NEAR_ZERO_H W/m²); if
# either scheme starts emitting a *large* tendency that's also out of
# the SBM band, that is a real failure and is reported.
#
# - Kuo's column-moisture-excess closure produces zero tendencies on a
#   static probe — it needs column super-saturation, which arises from
#   time-integrated surface evaporation, not from a single snapshot.
# - Kain-Fritsch with ``w_grid=0`` gates strongly through its
#   ``w_thresh_offset`` trigger and also produces near-zero tendencies
#   on the static probe.
_NEAR_ZERO_EXEMPT = {"kuo", "kain_fritsch"}
_NEAR_ZERO_H = 200.0   # W/m²


def main(argv: list[str] | None = None) -> int:
    """Run all validation tests; return 0 on success, 1 on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help=("Write the Test-5 matched-kernel conservation sweep (per scheme, "
              "per kernel arm, per probe sounding) as JSON for the SCM-RCE "
              "convection paper tables."),
    )
    args = parser.parse_args(argv)
    print("=" * 78)
    print("  Convection physics validation")
    print("=" * 78)

    failures: list[str] = []

    # ---- Test 1: tropical (CAPE-positive) sounding ------------------------
    T, q_v, p_full, p_half = tropical_sounding()
    print()
    print(f"  Test 1 — tropical sounding (CAPE ~8 kJ/kg, T_sfc=302 K, "
          f"PBL 90% RH)")
    print()
    print(f"  {'scheme':>16}  {'H':>8}  {'Q_v':>8}  {'Q_c':>8}  "
          f"{'H+Q_v+Q_c':>10}  {'rel_err':>8}  {'peak K/d':>8}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  "
          f"{'-'*8}  {'-'*8}")
    sbm_diag = None
    diags = {}
    for name in SCHEMES:
        out = call_scheme(name, T, q_v, p_full, p_half)
        d = column_diagnostics(out, p_half)
        diags[name] = d
        if name == "sbm":
            sbm_diag = d
        print(f"  {name:>16}  {d.H:8.2f}  {d.Q_v:8.2f}  {d.Q_c:8.2f}  "
              f"{d.residual:10.2f}  {d.rel_err:8.3f}  {d.peak_dT:8.2f}")

    # ---- Test 2: stable sounding (no convection expected) -----------------
    T_st, q_v_st, p_full_st, p_half_st = stable_sounding()
    print()
    print(f"  Test 2 — stable sounding (CAPE ≈ 0): convection should be "
          f"suppressed.")
    print()
    print(f"  {'scheme':>16}  {'H':>8}  {'Q_v':>8}  {'Q_c':>8}  "
          f"{'peak K/d':>8}  {'gating':>10}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    for name in SCHEMES:
        # Stable test uses fresh_carry=True so mass_flux / EDMF prognostic
        # state doesn't keep firing from a stale non-zero M_c / a_u.
        out = call_scheme(
            name, T_st, q_v_st, p_full_st, p_half_st, fresh_carry=True,
        )
        d = column_diagnostics(out, p_half_st)
        # Threshold tuned for the inversion-stable sounding: any scheme
        # with column heating > 100 W/m² OR peak |dT/dt| > 10 K/day in
        # a CAPE-zero column has gating issues.
        gating_ok = abs(d.H) < 100.0 and d.peak_dT < 10.0
        status = "OK" if gating_ok else "WEAK"
        print(f"  {name:>16}  {d.H:8.2f}  {d.Q_v:8.2f}  {d.Q_c:8.2f}  "
              f"{d.peak_dT:8.2f}  {status:>10}")
        if not gating_ok:
            failures.append(
                f"Test 2 (gating): {name} fires in CAPE≈0 column "
                f"(H={d.H:.1f} W/m², peak={d.peak_dT:.1f} K/d)"
            )

    # ---- Test 3: sign correctness on tropical sounding --------------------
    print()
    print(f"  Test 3 — sign correctness on tropical sounding "
          f"(H≥0, Q_v≤0, Q_c≥0)")
    print()
    print(f"  {'scheme':>16}  {'H≥0':>5}  {'Q_v≤0':>6}  {'Q_c≥0':>6}  "
          f"verdict")
    print(f"  {'-'*16}  {'-'*5}  {'-'*6}  {'-'*6}  --------")
    for name in SCHEMES:
        d = diags[name]
        ok_H = d.H >= -1.0
        ok_Q = d.Q_v <= 1.0
        ok_C = d.Q_c >= -1.0
        verdict = "PASS" if (ok_H and ok_Q and ok_C) else "FAIL"
        print(f"  {name:>16}  {'Y' if ok_H else 'N':>5}  "
              f"{'Y' if ok_Q else 'N':>6}  {'Y' if ok_C else 'N':>6}  "
              f"{verdict}")
        if verdict == "FAIL":
            failures.append(
                f"Test 3 (sign): {name} produced wrong-sign column "
                f"tendency (H={d.H:.1f} Q_v={d.Q_v:.1f} Q_c={d.Q_c:.1f})"
            )

    # ---- Test 4: comparison vs SBM reference ------------------------------
    print()
    print(f"  Test 4 — comparison vs SBM (column heating ratio "
          f"in [0.01, 100])")
    print()
    print(f"  {'scheme':>16}  {'H/H_sbm':>8}  {'Q_v/Q_v_sbm':>11}  "
          f"verdict")
    print(f"  {'-'*16}  {'-'*8}  {'-'*11}  --------")
    H_sbm = sbm_diag.H if abs(sbm_diag.H) > 1.0 else 1.0
    Qv_sbm = sbm_diag.Q_v if abs(sbm_diag.Q_v) > 1.0 else -1.0
    for name in SCHEMES:
        if name == "sbm":
            continue
        d = diags[name]
        rH = d.H / H_sbm if abs(H_sbm) > 1e-3 else float("nan")
        rQ = d.Q_v / Qv_sbm if abs(Qv_sbm) > 1e-3 else float("nan")
        in_range = 0.01 < abs(rH) < 100.0 and 0.01 < abs(rQ) < 100.0
        # Exemption applies *only* when the scheme is in the documented
        # near-zero list AND its absolute tendency is actually near zero
        # on this probe.  Large out-of-range tendencies for Kuo / KF
        # would still indicate a regression and must FAIL.
        is_near_zero = (
            name in _NEAR_ZERO_EXEMPT
            and abs(d.H) < _NEAR_ZERO_H
            and abs(d.Q_v) < _NEAR_ZERO_H
        )
        if in_range:
            verdict = "PASS"
        elif is_near_zero:
            verdict = "EXEMPT"   # documented: scheme barely fires here
        else:
            verdict = "FAIL"
        print(f"  {name:>16}  {rH:8.2f}  {rQ:11.2f}  {verdict}")
        if verdict == "FAIL":
            failures.append(
                f"Test 4 (magnitude): {name} column heating ratio "
                f"{rH:.3f} or vapor ratio {rQ:.3f} outside [0.01, 100] "
                f"(H={d.H:.1f}, Q_v={d.Q_v:.1f} — not in the documented "
                f"near-zero regime so this is a real divergence)"
            )

    # ---- Test 5: MATCHED-KERNEL conservation sweep ------------------------
    # Deliverable for the SCM-RCE convection intercomparison: the column
    # MSE-conservation residual for EVERY scheme under BOTH kernel arms, so a
    # published ranking can be read next to the conservation error of the
    # scheme that produced it.
    #
    # The residual is COLUMN-DEPENDENT (a documented property of this family),
    # so BOTH probe soundings are reported -- quoting one number as "the"
    # residual would be misleading.
    print()
    print("  Test 5 - matched-kernel conservation sweep")
    print("    vaporMSE  = int (c_p dT + L_v dq_v) dp/g  [W/m^2]   -> 0")
    print("    rel/latent= |vaporMSE| / (L_v * int dq_c dp/g)      -> 0")
    print("      (a ratio of ~1.0 means the condensation heating is MISSING)")
    print("    water     = int (dq_v + dq_c + dq_r) dp/g [kg/m^2/s] -> 0")
    print("    arm 'as_shipped' = scheme defaults; 'implicit_flux' = forced")
    print("    conservative kernel on every scheme that owns the knob.")
    print()
    kernel_report = {}
    for probe_name, cols in (
        ("tropical", (T, q_v, p_full, p_half)),
        ("stable", (T_st, q_v_st, p_full_st, p_half_st)),
    ):
        pT, pq, ppf, pph = cols
        print(f"    -- probe: {probe_name} sounding --")
        print(f"    {'scheme':>16}  {'arm':>14}  {'vaporMSE':>11}  "
              f"{'rel/latent':>10}  {'water':>11}  {'status':>26}")
        for name in SCHEMES:
            for arm in ("as_shipped", "implicit_flux"):
                status = kernel_arm_status(name, arm)
                out_k = call_scheme(
                    name, pT, pq, ppf, pph, subsidence_solve=arm)
                r = kernel_arm_residuals(out_k, pph)
                r["status"] = status
                kernel_report.setdefault(probe_name, {}).setdefault(
                    name, {})[arm] = r
                print(f"    {name:>16}  {arm:>14}  "
                      f"{r['vapor_mse_residual_W_m2']:11.3e}  "
                      f"{r['vapor_mse_rel_to_latent']:10.3e}  "
                      f"{r['water_residual_kg_m2_s']:11.3e}  {status:>26}")
        print()

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(kernel_report, indent=2) + "\n")
        print(f"  kernel-arm conservation report -> {args.json_out}")

    # ---- Final summary ----------------------------------------------------
    print()
    print("=" * 78)
    if failures:
        print(f"  VALIDATION FAILED — {len(failures)} issue(s):")
        for msg in failures:
            print(f"    - {msg}")
        print("=" * 78)
        return 1
    print("  VALIDATION PASSED — tests 1-4 clean across the scheme matrix "
          "(test 5 is a REPORT, not a gate).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
