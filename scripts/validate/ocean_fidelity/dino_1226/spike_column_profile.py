#!/usr/bin/env python
"""#1226 y20 per-step injection: CHARACTERISE the spike before hypothesising.

Prints, for the ONE column the surviving per-step injection peaks on
(j=158, i=42), the full vertical profile of

  * NEMO's own one-step change   T(Nnn,kt) - T(Nnn,kt-1)          [oracle]
  * legoESM's one-step change    T_after   - T_entry              [ours]
  * the difference               T_lego(after) - T_NEMO(Nnn,kt)   [the spike]
  * the entry stratification T, S and the water-mass slope dT/dS
  * legoESM's vertical diffusivity K_v on that column

so the operator CLASS can be named from the shape instead of guessed.  No
hypothesis is tested here and no arm is run -- ONE production step, the same
continuity control every other y20 probe uses (dT max 5.710106e-02 at
(158,42,10) under the production ``leapfrog_rhs`` placement; the 5.7305e-02
anchor recorded elsewhere is the ``applied_now`` placement of
``carry_injection_discriminator``, k33_fold_toggle.py:136).

Run:
CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  python scripts/validate/ocean_fidelity/dino_1226/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/spike_column_profile.py
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
CELL = (158, 42, 10)


def main() -> int:
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
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    _wind = bool(getattr(cfg, "wind_through_step", False))
    sf = dino_step_surface_forcing(forcing) if _wind else None
    print(f"  FORCING: wind_through_step={_wind} tau_x range="
          f"[{float(np.asarray(sf.tau_x).min()):.4f},"
          f"{float(np.asarray(sf.tau_x).max()):.4f}]" if _wind
          else "  FORCING: wind OFF")
    _placement = getattr(cfg, "surface_tendency_placement", None)
    print(f"  CFG surface_tendency_placement = {_placement!r}   "
          f"tracer_advection = {getattr(cfg, 'tracer_advection', None)!r}   "
          f"outer_integrator = {getattr(cfg, 'outer_integrator', None)!r}")

    if _placement == "leapfrog_rhs":
        st, rate = apply_dino_lat_lon_surface_forcing(
            st0, forcing, br.z_coord, cfg, DT, t_seconds=DT, return_rate=True)
    else:
        st = apply_dino_lat_lon_surface_forcing(
            st0, forcing, br.z_coord, cfg, DT, t_seconds=DT)
        rate = None
    st1 = model.step(st, DT, surface_forcing=sf, external_tracer_rate=rate)

    kt = mr.IC_STEP + 1
    ns = mr.nemo_now_state_at(kt)
    tmask3 = np.asarray(g.tmask) > 0.5
    dT_all = np.asarray(st1.T.data) - ns.T
    _abs = np.abs(dT_all)
    a = np.where(tmask3, _abs, -np.inf)
    idx = np.unravel_index(int(np.argmax(a)), a.shape)
    print(f"\nCONTINUITY CONTROL: dT max = {float(_abs[idx]):.6e} "
          f"@ {tuple(int(v) for v in idx)}  (expect 5.710106e-02 @ "
          f"(158, 42, 10))")

    j, i, _ = CELL
    T0 = np.asarray(st0.T.data)[j, i]
    S0 = np.asarray(st0.S.data)[j, i]
    Tb0 = np.asarray(st0.T_before.data)[j, i]
    T1 = np.asarray(st1.T.data)[j, i]
    S1 = np.asarray(st1.S.data)[j, i]
    nT0 = mr.nemo_now_state_at(mr.IC_STEP).T[j, i]
    nS0 = mr.nemo_now_state_at(mr.IC_STEP).S[j, i]
    nT1 = ns.T[j, i]
    nS1 = ns.S[j, i]
    wet = tmask3[j, i]
    nk = int(wet.sum())
    print(f"\ncolumn (j={j}, i={i}): {nk} wet levels of {wet.size}")
    print("  k |    T_entry |  dT_NEMO   |  dT_lego   |  SPIKE     |"
          "  dS_NEMO   |  dS_lego   |  SPIKE_S   | T-Tbb(entry)")
    for k in range(min(nk, 20)):
        print(f" {k:3d} | {T0[k]:10.6f} | {nT1[k]-nT0[k]: .3e} | "
              f"{T1[k]-T0[k]: .3e} | {T1[k]-nT1[k]: .3e} | "
              f"{nS1[k]-nS0[k]: .3e} | {S1[k]-S0[k]: .3e} | "
              f"{S1[k]-nS1[k]: .3e} | {T0[k]-Tb0[k]: .3e}")
    print("\n  entry stratification and water-mass slope:")
    for k in range(min(nk - 1, 19)):
        dT = T0[k] - T0[k + 1]
        dS = S0[k] - S0[k + 1]
        sl = dT / dS if abs(dS) > 1e-12 else float("nan")
        print(f"   k{k:2d}->k{k+1:<2d}  dT={dT: .6e}  dS={dS: .6e}  "
              f"dT/dS={sl: .4f}")
    sT = float(T1[CELL[2]] - nT1[CELL[2]])
    sS = float(S1[CELL[2]] - nS1[CELL[2]])
    print(f"\n  SPIKE VECTOR at k={CELL[2]}: dT={sT:.6e} dS={sS:.6e} "
          f"ratio={sT / sS:.4f} K/PSU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
