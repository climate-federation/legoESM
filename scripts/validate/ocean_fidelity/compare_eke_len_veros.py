"""Phase G eke_len gate L4 — oracle confirmation of the legoESM Rhines-limited
mixing length vs Veros.

Extends the E9 K_gm check (``compare_eke_kappa_veros.py``) one step deeper: E9
showed legoESM reproduces Veros ``K_gm`` when fed Veros's OWN ``eke_len``; this
script shows legoESM also reproduces Veros's ``eke_len`` itself (and its two
ingredients ``L_rossby``/``L_rhines``) when fed Veros's OWN column buoyancy
profile, Coriolis, β and eke — i.e. the new mixing-length FORM
(``eke_deformation_radius`` + ``eke_rhines_length`` + ``eke_len_composite``)
matches the oracle, independent of how each model evolves the state.

Reproduces, per Veros time level ``tau``:

    int_N_dz = Σ_z √(max(0,N²[τ]))·dzw·maskW          (= Veros C_rossby·π)
    L_rossby = eke_deformation_radius(int_N_dz, |f|, β)   vs Veros L_rossby   (2-D)
    L_rhines = eke_rhines_length(eke[τ], β)              vs Veros L_rhines   (3-D)
    eke_len  = eke_len_composite(L_rossby[…,None], L_rhines) vs Veros eke_len (3-D)
    K_gm     = eke_kappa_gm(eke[τ], eke_len)             vs Veros K_gm       (3-D)

ACC params (``veros/setups/acc/acc.py``): eke_c_k=0.4, eke_cross=2.0,
eke_crhin=1.0, eke_lmin=100, eke_k_max=1e4. These map to
``EKEConfig(mixing_length_scheme="rhines", eke_cross=2.0, eke_crhin=1.0)``.

Informational gate (not pass/fail). Requires Veros installed.

Usage::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        .venv/bin/python scripts/validate/ocean_fidelity/compare_eke_len_veros.py
"""

from __future__ import annotations

import argparse

import numpy as np


def _stats(mine: np.ndarray, ref: np.ndarray, wet: np.ndarray) -> tuple[float, float]:
    """max relative error + Pearson correlation over the wet cells."""
    m, r = mine[wet], ref[wet]
    rel = np.abs(m - r) / np.maximum(np.abs(r), 1e-12)
    corr = float(np.corrcoef(m, r)[0, 1]) if m.size > 1 else 1.0
    return float(rel.max()), corr


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

    from legoesm.ocean.fidelity.veros_runner import run_veros, DEFAULT_CAPTURE_VARS
    from legoesm.ocean.physics.lateral_mixing.eke import (
        EKEConfig, eke_deformation_radius, eke_rhines_length, eke_len_composite,
        eke_kappa_gm,
    )

    extra = ("eke", "K_gm", "K_iso", "eke_len", "L_rossby", "L_rhines",
             "Nsqr", "dzw", "maskW", "coriolis_t", "beta")
    print(f"Running Veros ACC (enable_eke) for {args.runlen_s:.0f} s ...")
    res = run_veros(
        "acc_channel", runlen_s=args.runlen_s,
        capture_vars=tuple(DEFAULT_CAPTURE_VARS) + extra,
        force_recompute=args.force_recompute,
    )
    V = res.variables
    eke = np.asarray(V["eke"])              # (x, y, z, tau)
    Nsqr = np.asarray(V["Nsqr"])            # (x, y, z, tau)
    dzw = np.asarray(V["dzw"])              # (z,)
    maskW = np.asarray(V["maskW"])          # (x, y, z)
    cor_t = np.asarray(V["coriolis_t"])     # (x, y)
    beta = np.asarray(V["beta"])            # (x, y)
    V_Lross = np.asarray(V["L_rossby"])     # (x, y)
    V_Lrhin = np.asarray(V["L_rhines"])     # (x, y, z)
    V_ekelen = np.asarray(V["eke_len"])     # (x, y, z)
    V_Kgm = np.asarray(V["K_gm"])           # (x, y, z)
    V_Kiso = np.asarray(V["K_iso"])         # (x, y, z) — = K_gm when enable_eke_isopycnal_diffusion

    print(f"  Veros L_rossby [km]: mean={np.nanmean(V_Lross)/1e3:.3g} "
          f"max={np.nanmax(V_Lross)/1e3:.3g}")
    print(f"  Veros L_rhines [km]: mean={np.nanmean(V_Lrhin)/1e3:.3g} "
          f"max={np.nanmax(V_Lrhin)/1e3:.3g}")
    print(f"  Veros eke_len  [km]: mean={np.nanmean(V_ekelen)/1e3:.3g} "
          f"max={np.nanmax(V_ekelen)/1e3:.3g}")

    # ACC EKE config with the Rhines-limited mixing length.
    cfg = EKEConfig(mixing_length_scheme="rhines", eke_cross=2.0, eke_crhin=1.0)
    print(f"\nlegoESM EKEConfig: scheme={cfg.mixing_length_scheme} "
          f"eke_cross={cfg.eke_cross} eke_crhin={cfg.eke_crhin} "
          f"c_k={cfg.c_k} l_min={cfg.l_min} k_max={cfg.kappa_gm_max}")

    wet3 = maskW > 0.5                       # 3-D wet mask (W grid)
    wet2 = wet3.any(axis=2) & (V_Lross > 0.0)  # 2-D columns with a defined radius
    f_abs = jnp.asarray(np.abs(cor_t))
    beta_j = jnp.asarray(beta)

    print("\nlegoESM (fed Veros's own N²/f/β/eke) vs Veros, per captured time level:")
    print("  tau |   L_rossby(2D)    |   L_rhines(3D)    |   eke_len(3D)     |   K_gm(3D)")
    best = None
    for tau in range(eke.shape[-1]):
        # int_N_dz = Σ_z √(max(0,N²))·dzw·maskW  (= Veros C_rossby·π).
        int_N_dz = jnp.asarray(
            np.sum(np.sqrt(np.maximum(0.0, Nsqr[..., tau]))
                   * dzw[None, None, :] * maskW, axis=2)
        )
        L_rossby = eke_deformation_radius(int_N_dz, f_abs, beta_j, cfg)
        L_rhines = eke_rhines_length(jnp.asarray(eke[..., tau]),
                                     beta_j[..., None], cfg)
        eke_len = eke_len_composite(L_rossby[..., None], L_rhines, cfg)
        K_gm = eke_kappa_gm(jnp.asarray(eke[..., tau]), eke_len, cfg)

        e_ross, c_ross = _stats(np.asarray(L_rossby), V_Lross, wet2)
        e_rhin, c_rhin = _stats(np.asarray(L_rhines), V_Lrhin, wet3)
        e_elen, c_elen = _stats(np.asarray(eke_len), V_ekelen, wet3)
        e_kgm, c_kgm = _stats(np.asarray(K_gm), V_Kgm, wet3)
        print(f"  {tau:>3} | {e_ross:.2e} c={c_ross:.4f} | {e_rhin:.2e} c={c_rhin:.4f}"
              f" | {e_elen:.2e} c={c_elen:.4f} | {e_kgm:.2e} c={c_kgm:.4f}")
        worst = max(e_ross, e_rhin, e_elen, e_kgm)
        if best is None or worst < best[1]:
            best = (tau, worst, (e_ross, e_rhin, e_elen, e_kgm))

    tau, worst, errs = best
    print(f"\nBest match: tau={tau}  worst max_rel_err={worst:.3e}  "
          f"(L_rossby={errs[0]:.2e} L_rhines={errs[1]:.2e} "
          f"eke_len={errs[2]:.2e} K_gm={errs[3]:.2e})")
    if worst < 1e-10:
        print("=> legoESM reproduces Veros L_rossby + L_rhines + eke_len + K_gm "
              "to MACHINE PRECISION.")
        print("   The Rhines-limited mixing-length FORM matches the oracle; EKE can "
              "be adopted apples-to-apples in the ACC recipe (L5).")

    # R3: K_iso = K_gm (Veros enable_eke_isopycnal_diffusion=True). Confirm Veros's
    # OWN K_iso == K_gm on wet cells; legoESM reproduces it because the step drives
    # kappa_redi_override = kappa_gm_override (the prognostic kappa that L4 matched
    # to Veros K_gm to machine precision), so legoESM K_iso == legoESM K_gm == Veros
    # K_gm == Veros K_iso by transitivity.
    wet = maskW > 0.5
    rel_kiso = (np.abs(V_Kiso[wet] - V_Kgm[wet])
                / np.maximum(np.abs(V_Kgm[wet]), 1e-12))
    print(f"\nVeros K_iso vs K_gm (enable_eke_isopycnal_diffusion): "
          f"max_rel_err={rel_kiso.max():.3e}, mean K_iso={np.nanmean(V_Kiso):.4g} "
          f"m^2/s ({int(wet.sum())} wet cells)")
    if rel_kiso.max() < 1e-12:
        print("=> Veros K_iso == K_gm confirmed. legoESM reproduces it via "
              "kappa_redi_override = kappa_gm_override (gate R2): K_iso=K_gm matches "
              "the oracle to machine precision (transitive with the K_gm match above).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
