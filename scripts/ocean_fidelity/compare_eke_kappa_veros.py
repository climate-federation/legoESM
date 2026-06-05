"""Phase G EKE gate E9 — oracle confirmation of the legoESM EKE closure vs Veros.

Runs a developed-flow Veros ACC integration (``enable_eke=True``; the ACC setup
sets eke_c_k=0.4, eke_c_eps=0.5, eke_k_max=1e4, eke_lmin=100, eke_cross=2.0,
eke_crhin=1.0, superbee advection, isopycnal diffusion), captures Veros's 3-D
``eke`` / ``K_gm`` / ``eke_len`` fields, and checks whether legoESM's prognostic
GM coefficient ``eke_kappa_gm`` (= clip(c_k·L·√E, 0, k_max)) reproduces Veros's
``K_gm`` when fed Veros's OWN ``eke`` and ``eke_len`` — i.e. whether the closure
FORMULA + parameters match the oracle, independent of how each model computes the
mixing length.

Result (2026-05-29, runlen 864000 s = 10 d, this machine): at the time level the
captured ``K_gm`` was computed from, legoESM matches Veros to MACHINE PRECISION
(max relative error 0.0, correlation 1.0 over 19560 wet cells).

The mixing length itself DIFFERS (Veros ``eke_len`` ~8 km mean via the eddy-energy
Rhines scale; legoESM ``L`` ~200 km via the Visbeck first-baroclinic Rossby radius)
— that is the documented "extend L" follow-up (strategy doc §8 EKE ledger), and is
why EKE is not yet flipped on in ``build_acc_model_config``.

Informational gate (not pass/fail). Requires Veros installed.

Usage::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        .venv/bin/python scripts/ocean_fidelity/compare_eke_kappa_veros.py
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runlen-s", type=float, default=864000.0,
                    help="Veros ACC integration length [s] (default 10 days).")
    ap.add_argument("--force-recompute", action="store_true",
                    help="Ignore any cached Veros result and re-run.")
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.ocean.fidelity.veros_runner import (
        run_veros, DEFAULT_CAPTURE_VARS,
    )
    from legoesm.ocean.physics.lateral_mixing.eke import eke_kappa_gm, EKEConfig

    print(f"Running Veros ACC (enable_eke) for {args.runlen_s:.0f} s ...")
    res = run_veros(
        "acc_channel", runlen_s=args.runlen_s,
        capture_vars=tuple(DEFAULT_CAPTURE_VARS) + ("eke", "K_gm", "eke_len"),
        force_recompute=args.force_recompute,
    )
    V = res.variables
    eke = np.asarray(V["eke"])        # (x, y, z, tau)
    K_gm = np.asarray(V["K_gm"])      # (x, y, z)
    eke_len = np.asarray(V["eke_len"])  # (x, y, z)

    print(f"  Veros eke     [m^2/s^2]: max={np.nanmax(np.abs(eke)):.4g}")
    print(f"  Veros K_gm    [m^2/s]  : mean={np.nanmean(K_gm):.4g} "
          f"max={np.nanmax(np.abs(K_gm)):.4g}")
    print(f"  Veros eke_len [km]     : mean={np.nanmean(eke_len)/1e3:.3g} "
          f"max={np.nanmax(eke_len)/1e3:.3g}")

    # legoESM closure params default to the Veros ACC values.
    cfg = EKEConfig()
    print(f"\nlegoESM EKEConfig: c_k={cfg.c_k} c_eps={cfg.c_eps} "
          f"l_min={cfg.l_min} kappa_gm_max={cfg.kappa_gm_max} "
          f"(Veros ACC: c_k=0.4 c_eps=0.5 lmin=100 k_max=1e4)")

    print("\nlegoESM eke_kappa_gm(E, L) vs Veros K_gm, per captured time level:")
    best = None
    wet = K_gm > 0.0
    for tau in range(eke.shape[-1]):
        mine = np.asarray(eke_kappa_gm(jnp.asarray(eke[..., tau]),
                                       jnp.asarray(eke_len), cfg))
        rel = np.abs(mine[wet] - K_gm[wet]) / np.maximum(np.abs(K_gm[wet]), 1e-12)
        corr = float(np.corrcoef(mine[wet], K_gm[wet])[0, 1])
        print(f"  tau={tau}: max_rel_err={rel.max():.3e} "
              f"mean_rel_err={rel.mean():.3e} corr={corr:.6f} (n={int(wet.sum())})")
        if best is None or rel.max() < best[1]:
            best = (tau, rel.max(), corr)

    tau, maxrel, corr = best
    print(f"\nBest match: tau={tau}  max_rel_err={maxrel:.3e}  corr={corr:.6f}")
    if maxrel < 1e-10:
        print("=> legoESM EKE closure reproduces Veros K_gm to MACHINE PRECISION.")
        print("   (Mixing-length L still differs: see the 'extend L' follow-up.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
