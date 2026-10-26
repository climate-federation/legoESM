#!/usr/bin/env python
"""#1226 y20 spike: print legoESM's LIVE vertical diffusivity on the spike
column and set it beside NEMO's own restart ``avt_k``/``avm_k``.

WHY: ``spike_column_profile.py`` showed the spike is a COLUMN REDISTRIBUTION
between the mixed layer (k=0..9) and the ML-base cell (k=10) of column
(j=158, i=42), in which legoESM applies only ~26-29% of NEMO's total tracer
tendency.  The only reservoir of that size in the column is the BEFORE-level
inversion at k=10 (NEMO restart 230400: tb(k10)=14.599077 against
tb(k9)=14.506905 -- 0.092 K WARMER below, strongly unstable at Nbb, while the
NOW level is marginally stable).  NEMO's EVD trigger is two-armed --

    zdfevd.F90:92-94   IF( MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12 ) &
                          p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)

with rn2 from ``bn2(ts(Nnn),...)`` and rn2b from ``bn2(ts(Nbb),...)``
(MY_SRC/stpmlf.F90:200-201), rn_evd=100, nn_evdm=1 (ocean.output:753-755) --
so at this cell NEMO fires EVD through the Nbb ARM.

legoESM's DINO card ALREADY selects that faithful trigger (instantiated and
printed below: ``convection_two_level_trigger=True``,
``convection_evd_n2_time_level='nemo_now_before'``, ``K_conv=100.0``,
``evd_on_momentum=True``, ``convection_n2_threshold=-1e-12``), so the trigger
CONFIGURATION is not the DIFF.  This probe asks the only remaining
mechanical question: does the diffusivity legoESM ACTUALLY hands the implicit
solve reach 100 m2/s at that interface?

It measures nothing else, tests no hypothesis, and runs ONE production step.

RESULT (2026-08-18, fp64, LEGOESM_NEMO_E3T=both, day-0 gate 0.000e+00)
---------------------------------------------------------------------
legoESM live K_v on this column, interfaces 6,7,8,9:

    100.0   100.0   100.0   2.2349e-02      (interface i couples cells i,i+1)

NEMO's OWN post-EVD diffusivity for the SAME step (RUN_SEQDUMP_Y20_1R/
dump_avt.bin, written after zdf_evd) at W-levels 5..12:

    100  100  100  100  100  100  100  1.5763e-05

NEMO W-level jk sits above 1-based cell jk, i.e. it couples 0-based cells
jk-2 and jk-1 -> legoESM interface jk-2.  So:

    NEMO EVD-active interfaces  4,5,6,7,8,9,10
    legoESM EVD-active          4,5,6,7,8       (interface 9 = 2.23e-02)

=> legoESM's convective/EVD region stops ONE INTERFACE SHALLOWER than
   NEMO's.  NEMO homogenises cell 10 into the mixed layer; legoESM leaves
   it out, so cell 10 keeps its BEFORE-level warm anomaly (tb=14.599077 vs
   tb(k9)=14.506905) and ends 5.71e-02 K WARMER than NEMO while the layer
   above ends 8.4e-03 K COOLER.  Sign, location and magnitude all match,
   and the spike is a closed redistribution: cell-10 gain 1.0654 K.m vs
   layer 0-9 loss 1.0661 K.m (ratio 0.9993), as a conservative mixing
   operator must be.

WHICH ARM: at that interface NEMO's own dumps give rn2 = -7.19e-12 and
rn2b = -4.0665e-09 (tke_dump_rn2.bin / tke_dump_rn2b.bin, threshold
-1.e-12).  The NOW arm clears the threshold by only 7x; the BEFORE arm by
4000x.  legoESM therefore fails the Nbb arm of MIN(rn2, rn2b) at this
interface even though the card selects evd_n2_time_level="nemo_now_before".

CROSS-RUN CONTROL: the EVD dumps come from RUN_SEQDUMP_Y20_1R (1 rank) and
the state/restarts from RUN_TWIN_STEP1 (16 ranks), both continuing the same
230400 restart.  tke_dump_avt_final.bin[158,42,10] = 2.252785e-02 equals
RUN_TWIN_STEP1 restart 230401 avt_k[158,42,10] = 2.252785e-02 to all
printed digits -- the two runs agree, so the dumps are usable here.

RETRACTED, in this same file: the docstring above proposed the TKE inverse-
Prandtl factor as the mechanism, from the step-ENTRY restart carry
(avt_k/avm_k = 0.1627 at that cell).  NEMO's live tke_dump_pdlr.bin is
exactly 1.0 at every level of this column down to W11, so the Prandtl
branch is NOT involved.  The mechanism is the EVD extent.

INSTRUMENT BUG FIXED, recorded so the number is not re-quoted: the first
run of this probe and of spike_column_profile.py printed a continuity
control of 2.257e+00 @(0,0,0).  That was np.abs(np.where(mask, d, -inf)),
which turns the masked sentinel into +inf and puts the argmax on land.
Both files now take |.| BEFORE masking.  The column data and the
5.710106e-02 spike value were never affected.

Run:
CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  python scripts/validate/ocean_fidelity/dino_1226/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/spike_kv_column.py
"""
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_OCEAN_FIDELITY = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _SCRIPTS_OCEAN_FIDELITY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DT = 2700.0
J, I = 158, 42
_N = {"calls": 0}


def _install(jnp, jax):
    """Wrap the live K-profile call and print the spike column.

    HOOK POINT: the model imports ``compute_vertical_K_profiles`` from the
    PACKAGE (`ocean_model_latlon_cgrid.py:5572`), not from `k_profiles`, so
    BOTH bindings are patched -- patching only the defining module leaves the
    package alias untouched and the hook never fires (measured 0 calls in the
    carry-injection probe when that mistake was made).
    """
    import legoesm.ocean.physics.vertical_mixing as vmix_pkg
    import legoesm.ocean.physics.vertical_mixing.k_profiles as kp

    orig = kp.compute_vertical_K_profiles

    def patched(*a, **kw):
        out = orig(*a, **kw)
        _N["calls"] += 1
        K_v, A_v = out[0], out[1]
        jax.debug.print(
            "  [K-call] K_v(j=158,i=42, interfaces 6..13) = {k}",
            k=K_v[J, I, 6:14])
        jax.debug.print(
            "           A_v(same)                        = {a}",
            a=A_v[J, I, 6:14])
        return out

    kp.compute_vertical_K_profiles = patched
    vmix_pkg.compute_vertical_K_profiles = patched


def main() -> int:
    import jax
    import jax.numpy as jnp
    import multistep_replay as mr
    from legoesm.core.precision import get_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing,
        dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays,
        dino_step_surface_forcing,
    )

    if not mr.have_step1_artifacts():
        print("SKIP: oracle artifacts not present")
        return 0
    print(f"PRECISION control dtype = {get_policy().control}")
    print(f"LEGOESM_NEMO_E3T = {os.environ.get('LEGOESM_NEMO_E3T')!r}")

    g, br, cfg, st0 = mr.build_replay_ic()
    # RULE 10: the EVD switches this probe's framing rests on, instantiated.
    for f in ("convection_two_level_trigger", "convection_evd_n2_time_level",
              "convection_n2_threshold", "convection_n2_mode", "K_conv",
              "evd_on_momentum", "n2_before_advection", "tke_n2_time_level",
              "outer_integrator", "surface_tendency_placement"):
        print(f"  CFG {f} = {getattr(cfg, f, '<absent>')!r}")

    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    _wind = bool(getattr(cfg, "wind_through_step", False))
    sf = dino_step_surface_forcing(forcing) if _wind else None
    print(f"  FORCING wind_through_step={_wind}")

    _install(jnp, jax)
    if getattr(cfg, "surface_tendency_placement", None) == "leapfrog_rhs":
        st, rate = apply_dino_lat_lon_surface_forcing(
            st0, forcing, br.z_coord, cfg, DT, t_seconds=DT, return_rate=True)
    else:
        st = apply_dino_lat_lon_surface_forcing(
            st0, forcing, br.z_coord, cfg, DT, t_seconds=DT)
        rate = None
    st1 = model.step(st, DT, surface_forcing=sf, external_tracer_rate=rate)

    ns = mr.nemo_now_state_at(mr.IC_STEP + 1)
    tmask3 = np.asarray(g.tmask) > 0.5
    d = np.asarray(st1.T.data) - ns.T
    _abs = np.abs(d)
    a = np.where(tmask3, _abs, -np.inf)
    idx = np.unravel_index(int(np.argmax(a)), a.shape)
    print(f"\nCONTINUITY CONTROL: dT max={float(_abs[idx]):.6e} "
          f"@{tuple(int(v) for v in idx)}  (expect 5.710106e-02 @ (158,42,10))")
    if _N["calls"] == 0:
        raise SystemExit("K hook never fired -- wrong hook point")
    print(f"K-profile calls = {_N['calls']}")

    from multistep_replay import RUN_TWIN_STEP1
    from rebuild_nemo_restart import rebuild
    for kt in (mr.IC_STEP, mr.IC_STEP + 1):
        raw = rebuild(f"{RUN_TWIN_STEP1}/DINO_{kt:08d}_restart_*.nc",
                      ["avt_k", "avm_k"])
        avt = np.moveaxis(np.asarray(raw["avt_k"], dtype=np.float64), 0, -1)
        avm = np.moveaxis(np.asarray(raw["avm_k"], dtype=np.float64), 0, -1)
        print(f"  NEMO restart kt={kt} (PRE-evd, zdfphy.F90:284-286) "
              f"avt_k(j,i,W-levels 7..14) = "
              f"{np.array2string(avt[J, I, 7:15], precision=4)}")
        print(f"  NEMO restart kt={kt}                              "
              f"avm_k(same)               = "
              f"{np.array2string(avm[J, I, 7:15], precision=4)}")
    print("  index map: NEMO W-level k <-> legoESM interface k-1 "
          "(interface i couples cells i and i+1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
