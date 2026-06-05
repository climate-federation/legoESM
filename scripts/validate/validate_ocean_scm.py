#!/usr/bin/env python
"""Ocean single-column-model (SCM) vertical-physics validation harness (#335).

Runs idealized SCM cases against analytic / reference ORACLES and reports
the agreement.  This is the scaffold the #335 discussion calls for: a single
place to assess ocean vertical physics (convection, vertical mixing) against
known answers, with a case registry that LES / GOTM oracles can be added to.

Currently implemented
---------------------
* ``convective_deepening`` — free-convective mixed-layer deepening into a
  linear stratification under constant surface cooling.  Analytic oracle:
  energy conservation (``rho_0 c_sw integral(dT) dz == Q0 t``) and the
  ``sqrt(t)`` encroachment deepening law
  (``h(t) = sqrt(2 Q0 t / (rho_0 c_sw Gamma))``).

Planned ORACLES (see issue #335)
--------------------------------
* GOTM (https://github.com/gotm-model/code) column solutions.
* Wagner et al. (2025) LES suite (Zenodo 10.5281/zenodo.20057136).
Register them as new ``Case`` entries returning observed-vs-oracle profiles;
the reporting / tolerance machinery here is reused unchanged.

Usage
-----
    JAX_ENABLE_X64=1 .venv/bin/python scripts/validate/validate_ocean_scm.py
    JAX_ENABLE_X64=1 .venv/bin/python scripts/validate/validate_ocean_scm.py \
        --case convective_deepening --plot

Exit status is non-zero if any case is outside tolerance (CI-friendly).
"""

from __future__ import annotations

import argparse
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.scm import OceanColumnModel
from legoesm.ocean.scm_forcing import OceanSCMForcing
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.eos import rho_0, c_sw


def _mld(T: np.ndarray, zcen: np.ndarray, threshold: float = 0.1) -> float:
    """Mixed-layer depth: deepest cell within ``threshold`` K of the surface."""
    mask = np.abs(T - T[0]) < threshold
    return float(zcen[mask].max())


def run_convective_deepening(
    *,
    nlev: int = 30,
    H_max: float = 400.0,
    dz_surface: float = 6.0,
    dz_deep: float = 14.0,
    gamma: float = 0.01,
    T_s0: float = 20.0,
    Q0: float = 250.0,
    dt: float = 3600.0,
    days=(3.0, 6.0, 12.0),
) -> dict:
    """Run the free-convective deepening case and compare to the oracle.

    Returns a dict with per-time MLD, heat-conservation error, and the
    sqrt(t)-scaling diagnostic.
    """
    cfg = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="constant"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
    )
    probe = OceanColumnModel.create(
        nlev=nlev, dt=dt, T_profile=jnp.full((nlev,), T_s0), S_profile=35.0,
        physics_config=cfg, H_max=H_max, dz_surface=dz_surface, dz_deep=dz_deep,
    )
    dz = np.asarray(probe.z_coord.dz_ref)
    zcen = np.cumsum(dz) - 0.5 * dz
    T_init = jnp.asarray(T_s0 - gamma * zcen)
    T_init_np = np.asarray(T_init)
    forcing = OceanSCMForcing(q_net=lambda t: -Q0)

    rows = []
    for d in days:
        model = OceanColumnModel.create(
            nlev=nlev, dt=dt, T_profile=T_init, S_profile=35.0,
            physics_config=cfg, H_max=H_max, dz_surface=dz_surface,
            dz_deep=dz_deep, forcing=forcing, implicit_vertical_mixing=True,
        )
        nsteps = int(round(d * 86400.0 / dt))
        final, _ = model.run(nsteps, save_every=nsteps)
        T = np.asarray(final.T.data).reshape(-1)
        t = nsteps * dt
        heat_removed = rho_0 * c_sw * float(np.sum(dz * (T_init_np - T)))
        heat_err = abs(heat_removed - Q0 * t) / (Q0 * t)
        h_num = _mld(T, zcen)
        h_slab = math.sqrt(2.0 * Q0 * t / (rho_0 * c_sw * gamma))
        rows.append(dict(days=d, t=t, mld=h_num, h_slab=h_slab,
                         heat_err=heat_err, T=T))

    # sqrt(t) scaling: fit log(MLD) vs log(t); slope should be ~0.5.
    logt = np.log(np.array([r["t"] for r in rows]))
    logh = np.log(np.array([r["mld"] for r in rows]))
    slope = float(np.polyfit(logt, logh, 1)[0])
    return dict(name="convective_deepening", rows=rows, slope=slope, zcen=zcen)


def report(result: dict, tol_heat=5e-3, tol_slope=0.08,
           amp_lo=1.2, amp_hi=1.8, plot=False) -> bool:
    print(f"\n=== {result['name']} ===")
    print(f"{'days':>6} {'MLD[m]':>9} {'slab[m]':>9} {'MLD/slab':>9} {'heat_err':>10}")
    ok = True
    for r in result["rows"]:
        amp = r["mld"] / r["h_slab"]
        flags = []
        if r["heat_err"] >= tol_heat:
            flags.append("HEAT")
            ok = False
        if not (amp_lo < amp < amp_hi):
            flags.append("AMP")
            ok = False
        flag = ("  <-- " + ",".join(flags)) if flags else ""
        print(f"{r['days']:6.1f} {r['mld']:9.1f} {r['h_slab']:9.1f} "
              f"{amp:9.3f} {r['heat_err']:10.2e}{flag}")
    slope_err = abs(result["slope"] - 0.5)
    slope_ok = slope_err < tol_slope
    ok = ok and slope_ok
    print(f"sqrt(t) deepening: d(log MLD)/d(log t) = {result['slope']:.3f} "
          f"(oracle 0.5, |err|={slope_err:.3f}) "
          f"{'OK' if slope_ok else '<-- SCALING'}")

    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from pathlib import Path
            out = Path("results/ocean_scm")
            out.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            zc = result["zcen"]
            for r in result["rows"]:
                ax[0].plot(r["T"], -zc, label=f"{r['days']:.0f} d")
            ax[0].set(xlabel="T [degC]", ylabel="z [m]",
                      title="Convective deepening")
            ax[0].legend()
            t = np.array([r["t"] for r in result["rows"]]) / 86400.0
            ax[1].plot(t, [r["mld"] for r in result["rows"]], "o-", label="SCM MLD")
            ax[1].plot(t, [r["h_slab"] for r in result["rows"]], "k--",
                       label="slab oracle")
            ax[1].set(xlabel="time [d]", ylabel="MLD [m]",
                      title=f"slope={result['slope']:.3f} (oracle 0.5)")
            ax[1].legend()
            fig.tight_layout()
            p = out / "convective_deepening.png"
            fig.savefig(p, dpi=110)
            print(f"saved {p}")
        except Exception as e:  # plotting is best-effort
            print(f"(plot skipped: {e})")
    return ok


_CASES = {"convective_deepening": run_convective_deepening}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=sorted(_CASES), default="convective_deepening")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    result = _CASES[args.case]()
    ok = report(result, plot=args.plot)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
