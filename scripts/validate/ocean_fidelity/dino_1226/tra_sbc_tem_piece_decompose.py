#!/usr/bin/env python
"""#1226 Part 2: which PIECE of tra_sbc's tem tendency owns the 9.657e-07
err_norm residual that survives the #1226 live-divisor fix (commit
fbe04d93a / 5e9b0eb87), when sal clears to 0.0 under the SAME fix?

THE WALK (task brief): trasbc.F90's own MLF-averaged Krhs increment for
temperature has additive structure at k=0 (restoring.py:150-165, reused by
production dino.py:3406-3420 and the just-fixed measure_tra_sbc):

    dT_dt_top = -(T_Kbb - T*)/tau_T_eff        <- PIECE A: T-restoring
                - Q_sr/(rho_0*c_p*dz_0_2d)     <- PIECE B: Q_sr subtraction

Salinity has ONLY piece A's counterpart (S-restoring; RestoringConfig has no
Q_sr-subtraction analogue for salinity -- restoring.py:165 only touches
surf_dT; trasbc.F90:137 ``sbc_tsc(jp_sal) = r1_rho0*sfx`` has NO ``rcp``
factor at all). This asymmetry is exactly what localises the residual below.

RETRACTION (recorded per Rule 1e, logged before any number below): a first
pass decomposed tem into "PIECE A (T-restoring) vs PIECE B (Q_sr
subtraction)" this-step-only rates and found BOTH pieces reproduced NEMO's
own dumps to ~1e-15/1e-16 -- but that test used ``cfg.c_p`` on BOTH the
"legoESM" and "NEMO-reconstructed" sides of the comparison (the NEMO-side
reconstruction used ``r1_rho0_rcp = 1/(cfg.rho_0*cfg.c_p)`` to convert
NEMO's OWN dumped ``qns``/``qsr`` into a rate), so the comparison was
SELF-CONSISTENT rather than a real test against NEMO's true constant --
exactly the failure mode Rule 1e warns about (a passing self-check that
tests the wrong thing). Reading ``eosbn2.F90:1899`` (NOT ``phycst.F90`` --
that module's own comment at :118 says "reference density and heat capacity
now defined in eosbn2.f90") found NEMO's ``rcp = 3991.86795711963_wp``
EXACTLY, vs ``DINOConfig.c_p = 3991.86`` (dino.py:92) -- a 6-sig-fig
TRUNCATION, relative error 1.9933e-06. Reconstructing NEMO's own tem dump
from NEMO's own ``qns``/``sbc_hc_b`` (bit-exact NEMO arithmetic, no legoESM
physics involved at all) using ``cfg.c_p`` gives err_norm median 9.6668e-07;
using NEMO's EXACT ``rcp`` gives 0.0 (bit-identical) -- CONFIRMED below.
Salinity's own reconstruction (same restart/dump chain, no ``rcp`` anywhere)
is ALREADY 0.0 with ``cfg.c_p`` -- exactly consistent with the truncation
being T-only.

NEMO SIDE (read, not assumed):
  * ``phycst.F90:118``: "reference density and heat capacity now defined in
    eosbn2.f90" -- rho0/rcp are NOT namelist constants, they are HARDCODED
    Fortran literals in the EOS module.
  * ``eosbn2.F90:1898-1899``: ``rho0 = 1026._wp`` (matches
    ``DINOConfig.rho_0=1026.0`` exactly) ; ``rcp = 3991.86795711963_wp``
    (DINOConfig.c_p=3991.86 truncates this).
  * ``usrdef_sbc.F90:254-260`` (nn_forcingtype=4 CASE(4) ELSE branch,
    confirmed active): ``qtot = rn_trp*(T_Kbb - T*)`` [W/m^2]
    (``rn_trp=-40=-A_theta``).
  * ``usrdef_sbc.F90:279``: ``qns = (qtot - zqsr_dayMean) * tmask`` [W/m^2]
    -- so ``qtot_nemo = qns_nemo + qsr_nemo`` is EXACT ARITHMETIC on NEMO's
    own dumped ``sbc_dump_qns.bin``/``sbc_dump_qsr.bin``
    (``cancelling_rows_per_element.py`` ``measure_sbc`` already loads both)
    -- no re-derivation, no independent reconstruction of qtot from T/T*.
  * ``trasbc.F90:136-137``: ``sbc_tsc(jp_tem) = r1_rho0_rcp*qns`` (has rcp)
    vs ``sbc_tsc(jp_sal) = r1_rho0*sfx`` (NO rcp) -- the structural reason
    salinity CANNOT show this defect while temperature does.
  * ``trasbc.F90:150-153``: ``pts(Krhs) += zfact*(sbc_tsc_b+sbc_tsc)
    /e3t(:,:,1,Kmm)`` -- the MLF average + live-divisor, verified exact by
    Part 1 (sal clears to 0.0 there too), confirming the divisor is NOT
    where the tem-only residual lives.

THIS SCRIPT IS READ-ONLY on packages/src -- pure measurement, one new probe,
reuses coverage_rows_measure.py's build_state()/dump-loaders (Rule 0: read
the oracle; do not re-derive its formula).

Run::

    CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \\
      .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/tra_sbc_tem_piece_decompose.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import numpy as np
import netCDF4 as nc
import jax.numpy as jnp

_HERE = os.path.dirname(__file__)


def _load_sibling(name: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_HERE, name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_cov = _load_sibling("coverage_rows_measure.py", "_coverage_rows_measure")

RUN_DIR = _cov.RUN_DIR
RESTART = _cov.RESTART
KT_DUMP = _cov.KT_DUMP
_load_haloed = _cov._load_haloed
per_element_stats = _cov.per_element_stats

from legoesm.ocean.experiments.dino import dino_Q_sr_seasonal
from legoesm.ocean.eos import nemo_r3t_stretch
from legoesm.ocean.fidelity.time_levels import time_level_for_dump
from legoesm.ocean.physics.shortwave_penetration import JERLOV_TYPES


def main() -> int:
    st = _cov.build_state()
    br, cfg = st["br"], st["cfg"]
    tmask2d = st["tmask2d"]
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    n_lat, n_lon = tmask2d.shape

    print("\n" + "=" * 78)
    print("PART 2: tra_sbc tem residual isolation (rho0/rcp constant vs T/S asymmetry)")
    print("=" * 78)

    # ---- production dz_0_2d (identical to the just-fixed measure_tra_sbc) --
    divisor = getattr(cfg, "surface_flux_divisor", "static")
    dz_0 = float(br.z_coord.dz_ref[0])
    print(f"  cfg.surface_flux_divisor = {divisor!r}")
    if divisor == "nemo_live":
        stretch = np.asarray(nemo_r3t_stretch(br.z_coord, br.state.eta.data,
                                                br.state.H_bathy.data))
        dz_0_2d = dz_0 * stretch
    elif divisor == "static":
        dz_0_2d = np.broadcast_to(np.float64(dz_0), tmask2d.shape)
    else:
        raise ValueError(f"unknown surface_flux_divisor {divisor!r}")

    t_seconds = KT_DUMP * cfg.dt
    lat_deg_1d = np.degrees(np.asarray(br.geometry.lat))
    Q_sr_1d = np.asarray(dino_Q_sr_seasonal(jnp.asarray(lat_deg_1d), t_seconds, cfg))
    Q_sr_2d = np.broadcast_to(Q_sr_1d[:, None], (n_lat, n_lon))

    # NEMO side: qns_nemo is NEMO's own dumped, ALREADY-COMBINED non-solar
    # flux (usrdef_sbc.F90:279 qns=(qtot-qsr)*tmask) -- used directly below,
    # no re-derivation of qtot from T/T* needed for this reconstruction.
    qns_nemo = _load_haloed(os.path.join(RUN_DIR, "sbc_dump_qns.bin"), jpi, jpj, hls)[..., 0]
    print(f"  time_level_for_dump('sbc_dump_qns.bin') = "
          f"{time_level_for_dump('sbc_dump_qns.bin')!r}  "
          "(usrdef_sbc.F90:421-423 reads ts(...,Kbb) for qns/qtot)")

    # NEMO's own MLF-average + live-divisor machinery (trasbc.F90:118-153):
    # reconstruct NEMO's OWN dumped tem Krhs PURELY from NEMO's OWN dumped/
    # restart fields (qns_nemo, sbc_hc_b) and NEMO's own documented formula --
    # NO legoESM physics involved on either side of this specific check. If
    # this reconstruction does not reproduce tem_nemo bit-for-bit, the gap
    # lives in a CONSTANT used to build the reconstruction, not in legoESM's
    # restoring/Q_sr formulas (which never enter this test at all).
    with nc.Dataset(os.path.join(RUN_DIR, RESTART)) as r:
        sbc_hc_b = np.asarray(r["sbc_hc_b"][0]).squeeze()
    tem_nemo = _load_haloed(os.path.join(RUN_DIR, "stp_dump_14_trasbc_tem.bin"),
                             jpi, jpj, hls)[..., 0]
    sal_nemo = _load_haloed(os.path.join(RUN_DIR, "stp_dump_14_trasbc_sal.bin"),
                             jpi, jpj, hls)[..., 0]

    print("\n" + "-" * 78)
    print("PURE-NEMO-ARITHMETIC RECONSTRUCTION: cfg.c_p (truncated) vs NEMO's exact rcp")
    print("-" * 78)
    print("  phycst.F90:118 'reference density and heat capacity now defined "
          "in eosbn2.f90' -> eosbn2.F90:1898-1899: rho0=1026._wp (EXACT match "
          f"to cfg.rho_0={cfg.rho_0}), rcp=3991.86795711963_wp vs "
          f"cfg.c_p={cfg.c_p} (TRUNCATED at 6 sig figs).")
    NEMO_RCP_EXACT = 3991.86795711963   # eosbn2.F90:1899, verbatim
    rel_cp_trunc = abs(cfg.c_p / NEMO_RCP_EXACT - 1.0)
    print(f"  relative truncation |cfg.c_p/rcp_exact - 1| = {rel_cp_trunc:.6e}")

    def _reconstruct_tem(rcp_value):
        r1_rho0_rcp = 1.0 / (cfg.rho_0 * rcp_value)
        this_step_rate = r1_rho0_rcp * qns_nemo   # trasbc.F90:136 sbc_tsc(jp_tem)
        return 0.5 * (sbc_hc_b + this_step_rate) / dz_0_2d   # trasbc.F90:152-153

    tem_recon_lego_cp = _reconstruct_tem(cfg.c_p)
    tem_recon_exact_rcp = _reconstruct_tem(NEMO_RCP_EXACT)
    r_tem_lego_cp = per_element_stats("tem RECONSTRUCTION [cfg.c_p, truncated]",
                                       tem_recon_lego_cp, tem_nemo, tmask2d,
                                       sign_changing=True)
    r_tem_exact = per_element_stats("tem RECONSTRUCTION [NEMO exact rcp]",
                                     tem_recon_exact_rcp, tem_nemo, tmask2d,
                                     sign_changing=True)

    # sal reconstruction has NO c_p/rcp anywhere (trasbc.F90:137 sbc_tsc(jp_sal)
    # = r1_rho0*sfx -- r1_rho0 only) -- STRUCTURALLY cannot show this defect.
    # A self-check, not a new claim: confirms 0.0 independent of this script's
    # own path (Part 1's fidelity_bar_gate already established this).
    with nc.Dataset(os.path.join(RUN_DIR, RESTART)) as r:
        sbc_sc_b = np.asarray(r["sbc_sc_b"][0]).squeeze()
    sfx_nemo = _load_haloed(os.path.join(RUN_DIR, "sbc_dump_sfx.bin"), jpi, jpj, hls)[..., 0]
    sal_recon = 0.5 * (sbc_sc_b + sfx_nemo / cfg.rho_0) / dz_0_2d
    d_sal_recon = float(np.max(np.abs((sal_recon - sal_nemo)[tmask2d])))
    print(f"  [self-check, structural] sal reconstruction (no rcp anywhere) "
          f"vs sal_nemo dump: max|diff|={d_sal_recon:.3e} (want ~roundoff -- "
          "proves salinity is STRUCTURALLY immune to a c_p/rcp defect, "
          "matching Part 1's independently-established sal err_norm=0.0)")
    assert d_sal_recon < 1e-18, "sal reconstruction should be exact to roundoff (no rcp term)"

    print(f"\n  VERDICT: tem reconstruction err_norm median "
          f"[cfg.c_p]={r_tem_lego_cp['bar_metric']:.3e}  "
          f"[NEMO exact rcp]={r_tem_exact['bar_metric']:.3e}")
    if r_tem_exact["bar_metric"] < 1e-9 and r_tem_lego_cp["bar_metric"] > 1e-9:
        print("  CONFIRMED: swapping cfg.c_p for NEMO's exact rcp (eosbn2.F90:"
              "1899) makes the PURE-NEMO-ARITHMETIC tem reconstruction "
              "bit-identical to NEMO's own dump. The 9.657e-07 err_norm "
              "residual is the c_p 6-sig-fig TRUNCATION -- not a restoring, "
              "Q_sr-subtraction, or divisor defect. This explains the T/S "
              "asymmetry exactly: sal's own conversion (trasbc.F90:137) has "
              "no rcp factor and is structurally immune (verified 0.0 above).")
    else:
        print("  NOT CONFIRMED -- the c_p truncation does not explain the "
              "residual by this test; see the numbers above for what remains.")

    # =========================================================================
    # PART 2b: independent test of the STATIC-vs-LIVE-gdepw hypothesis on the
    # tra_qsr row itself (already a MEASURED DEBT row: k=0 corr 0.99999996 /
    # ratio 0.99996836). If legoESM's shortwave_penetration_tendency's use of
    # the STATIC z_coord.z_half_ref (vs NEMO's LIVE gdepw(Kmm) in qsr_2BD,
    # traqsr.F90:671,679-680) is the tra_qsr row's cause, replacing z_half_ref
    # with the LIVE r3t-stretched interface depths (same nemo_r3t_stretch
    # helper Part 1 reused -- eos.py:640) should measurably shrink tra_qsr's
    # own residual. This is a DIAGNOSTIC A/B on the EXISTING tra_qsr row, not
    # a production change (packages/ read-only per task rules).
    # =========================================================================
    print("\n" + "=" * 78)
    print("PART 2b: tra_qsr STATIC vs LIVE gdepw A/B (independent test)")
    print("=" * 78)
    print("  NEMO qsr_2BD (traqsr.F90:665-712, MLF #else branch, ln_qsr_2bd=T "
          "confirmed namelist_cfg:180): zatt(k+1) = "
          "[rn_abs*exp(-gdepw(k+1,Kmm)*r1_si0) + (1-rn_abs)*exp(-gdepw(k+1,Kmm)"
          "*r1_si1)]*r1_rho0_rcp -- gdepw(Kmm) is the LIVE (z*-stretched) "
          "w-level depth at the CURRENT step, not the static gdepw_1d.")
    print(f"  rn_abs=0.58 rn_si0=0.35 rn_si1=23.0 (namelist_ref:429-436) == "
          f"JERLOV_TYPES['I'] R/zeta1/zeta2 exactly "
          f"(cfg.jerlov_water_type={cfg.jerlov_water_type!r})")

    tem_after_sbc = _load_haloed(os.path.join(RUN_DIR, "stp_dump_14_trasbc_tem.bin"), jpi, jpj, hls)
    tem_after_qsr = _load_haloed(os.path.join(RUN_DIR, "stp_dump_17_traqsr_tem.bin"), jpi, jpj, hls)
    tem_qsr_only_nemo = tem_after_qsr - tem_after_sbc   # bracketed increment (full column)

    params = JERLOV_TYPES[cfg.jerlov_water_type]
    R, zeta1, zeta2 = params.R, params.zeta1, params.zeta2

    # STATIC (production path, dino.py:3430-3436 / shortwave_penetration.py):
    z_half_static = np.asarray(br.z_coord.z_half_ref)     # (nlev+1,), negative, SAME at every column
    I_half_static = R * np.exp(z_half_static / zeta1) + (1.0 - R) * np.exp(z_half_static / zeta2)
    frac_static = I_half_static[:-1] - I_half_static[1:]
    frac_static[-1] += I_half_static[-1]
    dz_actual = np.asarray(br.z_coord.dz_ref)             # jacobian=1 in the production call (dino.py:3428)
    dT_dt_static = Q_sr_2d[:, :, None] * frac_static[None, None, :] / (cfg.rho_0 * cfg.c_p * dz_actual[None, None, :])

    # LIVE: replace z_half_ref with the r3t-stretched interface ladder, PER
    # COLUMN (2D eta/H_bathy -> per-cell stretch), reusing nemo_r3t_stretch
    # (no re-derivation of r3t -- same helper Part 1 used for the divisor).
    stretch_col = np.asarray(nemo_r3t_stretch(br.z_coord, br.state.eta.data,
                                                br.state.H_bathy.data))   # (n_lat, n_lon)
    z_half_live = z_half_static[None, None, :] * stretch_col[:, :, None]  # (n_lat, n_lon, nlev+1)
    I_half_live = R * np.exp(z_half_live / zeta1) + (1.0 - R) * np.exp(z_half_live / zeta2)
    frac_live = I_half_live[:, :, :-1] - I_half_live[:, :, 1:]
    frac_live[:, :, -1] += I_half_live[:, :, -1]
    dz_live_col = dz_actual[None, None, :] * stretch_col[:, :, None]      # live top-cell-consistent thickness
    dT_dt_live = Q_sr_2d[:, :, None] * frac_live / (cfg.rho_0 * cfg.c_p * dz_live_col)

    nk_dump = tem_qsr_only_nemo.shape[-1]
    dT_dt_static = dT_dt_static[..., :nk_dump]
    dT_dt_live = dT_dt_live[..., :nk_dump]
    mask3 = np.broadcast_to(tmask2d[:, :, None], dT_dt_static.shape)

    print("\n  --- VARIANT STATIC (current production shortwave_penetration_tendency) ---")
    r_static_all = per_element_stats("tra_qsr [STATIC gdepw], all levels",
                                      dT_dt_static, tem_qsr_only_nemo, mask3,
                                      sign_changing=False)
    r_static_k0 = per_element_stats("tra_qsr [STATIC gdepw], k=0",
                                     dT_dt_static[..., 0], tem_qsr_only_nemo[..., 0],
                                     tmask2d, sign_changing=False)

    print("\n  --- VARIANT LIVE (per-column r3t-stretched gdepw) ---")
    r_live_all = per_element_stats("tra_qsr [LIVE gdepw], all levels",
                                    dT_dt_live, tem_qsr_only_nemo, mask3,
                                    sign_changing=False)
    r_live_k0 = per_element_stats("tra_qsr [LIVE gdepw], k=0",
                                   dT_dt_live[..., 0], tem_qsr_only_nemo[..., 0],
                                   tmask2d, sign_changing=False)

    # SELF-CHECK: forcing r3t->0 (stretch_col->1 everywhere) must make LIVE
    # bit-identical to STATIC -- proves the ONLY difference is the stretch.
    stretch_forced_one = np.ones_like(stretch_col)
    z_half_live0 = z_half_static[None, None, :] * stretch_forced_one[:, :, None]
    I_half_live0 = R * np.exp(z_half_live0 / zeta1) + (1.0 - R) * np.exp(z_half_live0 / zeta2)
    frac_live0 = I_half_live0[:, :, :-1] - I_half_live0[:, :, 1:]
    frac_live0[:, :, -1] += I_half_live0[:, :, -1]
    dz_live0 = dz_actual[None, None, :] * stretch_forced_one[:, :, None]
    dT_dt_live0 = Q_sr_2d[:, :, None] * frac_live0 / (cfg.rho_0 * cfg.c_p * dz_live0)
    d_bitcheck = float(np.max(np.abs(dT_dt_live0[..., :nk_dump] - dT_dt_static)))
    print(f"\n  [self-check] forcing stretch->1 in the LIVE branch vs STATIC: "
          f"max|diff|={d_bitcheck:.3e}  (want 0.0 -- proves the ONLY "
          "difference between STATIC/LIVE is the r3t stretch, not some other "
          "silently-differing path)")
    assert d_bitcheck < 1e-13, (
        "LIVE (stretch forced to 1) and STATIC differ by more than the "
        "stretch -- controlled-comparison premise VIOLATED")

    print(f"\n  VERDICT (tra_qsr STATIC-vs-LIVE gdepw): "
          f"all-levels err_norm median STATIC={r_static_all['bar_metric']:.3e} "
          f"-> LIVE={r_live_all['bar_metric']:.3e}   "
          f"k=0-only median STATIC={r_static_k0['bar_metric']:.3e} "
          f"-> LIVE={r_live_k0['bar_metric']:.3e}")
    improved = r_live_all["bar_metric"] < r_static_all["bar_metric"]
    print(f"  LIVE gdepw {'IMPROVES' if improved else 'does NOT improve'} "
          "tra_qsr's own residual vs STATIC.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
