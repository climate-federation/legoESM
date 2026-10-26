#!/usr/bin/env python
"""#1226 tracer-injection OWNERSHIP toggle: the isoneutral K33 vertical fold.

CONTEXT / PREMISE CORRECTION (Rule 10, Rule 1e)
-----------------------------------------------
The commissioning task asked to *turn ON* ``implicit_K33``/``msc_stabilize``
as a probe arm, on the stated premise that the production
``nemo_dino_kamm_mlf`` card resolves them ``False`` (K33 fully EXPLICIT in the
RHS).  Instantiating the resolved config REFUTES that premise:

    dino_lat_lon_model_config(nemo_dino_kamm_mlf).gm_redi
        implicit_K33  = True     (dino.py:2877)
        msc_stabilize = True     (dino.py:2898)
        slope_scheme  = nemo_iso_lap
        slope_positions = nemo_native

i.e. the NEMO ``tra_zdf`` fold the hypothesis says is MISSING is already
SELECTED in production, and the injection spike is measured WITH it on.  A
"turn it ON" arm is therefore vacuous (ON -> ON).  ``deep_box_heat_budget.py``
:91-93 independently records the same ("this recipe sets implicit_K33=True
... Production was always correct"); the 1c3dfe19b commit's printed
``implicit_K33=False`` is unsupported by any code in the lane that reads that
field and is RETRACTED here.

THE EXPERIMENT ACTUALLY RUN (the meaningful direction)
-----------------------------------------------------
Ownership test (Rule 4): remove the fold and see whether the spike survives.
One variable off production:

  ON  (production)  gm_redi.implicit_K33=True,  msc_stabilize=True
  OFF (ablation)    gm_redi.implicit_K33=False, msc_stabilize=False

Both models step ONE step from the SAME bridged day-0 state (kt 230400->401),
wind-on, fp64, identical reductions.  Prediction under the (now-inverted)
hypothesis that the fold OWNS the spike: OFF makes the single-cell spike at
(158,42,10) GROW (>2x) or blow up.  If OFF leaves it ~unchanged (<2x), the fold
does NOT own the injection and the "K33 fold" attribution is REFUTED.

Continuity control FIRST (Rule 1e): the ON arm MUST reproduce the recorded
base spike max 5.7305e-2 K at cell (158,42,10) before the OFF arm is read.

Usage
-----
    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/run_fp64.py \\
      scripts/validate/ocean_fidelity/dino_1226/k33_fold_toggle.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_OCEAN_FIDELITY = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _SCRIPTS_OCEAN_FIDELITY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DT = 2700.0
# Recorded base spike (wind-on, applied_now, this state) — continuity anchor.
_ANCHOR_MAX = 5.7305e-2
_ANCHOR_CELL = (158, 42, 10)


def _stats(diff, mask):
    m = np.broadcast_to(mask, diff.shape)
    a = np.abs(diff)
    idx = np.unravel_index(int(np.argmax(np.where(m, a, -np.inf))), a.shape)
    sel = a[m]
    return {
        "max": float(a[idx]),
        "argmax": tuple(int(v) for v in idx),
        "at_anchor": float(a[_ANCHOR_CELL]),
        "p99.9": float(np.percentile(sel, 99.9)),
        "p99": float(np.percentile(sel, 99.0)),
        "p50": float(np.percentile(sel, 50.0)),
    }


def main() -> int:
    import jax
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
        print("SKIP: RUN_TRAJ/RUN_TWIN_STEP1 oracle artifacts not present")
        return 0

    print(f"PRECISION control dtype = {get_policy().control}")
    print(f"LEGOESM_NEMO_E3T = {os.environ.get('LEGOESM_NEMO_E3T')!r}")

    g, br, cfg, st0 = mr.build_replay_ic()  # day-0 gate prints above
    kt = mr.IC_STEP + 1

    mc_on, _ = dino_lat_lon_model_config(br.geometry, cfg)
    gr = mc_on.gm_redi
    print(f"  RESOLVED gm_redi: implicit_K33={gr.implicit_K33} "
          f"msc_stabilize={gr.msc_stabilize} slope_scheme={gr.slope_scheme!r} "
          f"slope_positions={gr.slope_positions!r}")
    if not (gr.implicit_K33 and gr.msc_stabilize):
        raise SystemExit(
            "PREMISE-CHECK: expected production fold ON; got "
            f"implicit_K33={gr.implicit_K33} msc={gr.msc_stabilize}")
    mc_off = mc_on._replace(
        gm_redi=gr._replace(implicit_K33=False, msc_stabilize=False))

    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    _wind = bool(getattr(cfg, "wind_through_step", False))
    sf = dino_step_surface_forcing(forcing) if _wind else None
    _tau = np.asarray(sf.tau_x) if _wind else np.zeros(1)
    print(f"  FORCING: wind_through_step={_wind} "
          f"surface_stress_implicit={getattr(cfg,'surface_stress_implicit',None)} "
          f"tau_x[Pa] range=[{float(_tau.min()):.4f},{float(_tau.max()):.4f}]")
    for f in ("t_depth_ref", "dz_ref", "z_full_ref", "z_half_ref"):
        if hasattr(br.z_coord, f):
            print(f"  GEOMETRY dtype {f} = "
                  f"{np.asarray(getattr(br.z_coord, f)).dtype}")
    print(f"  STATE dtypes T={st0.T.data.dtype} S={st0.S.data.dtype} "
          f"u={st0.u.data.dtype} eta={st0.eta.data.dtype}")

    ns = mr.nemo_now_state_at(kt)
    tmask3 = np.asarray(g.tmask) > 0.5

    # The recorded anchor (5.7305e-2) is the discriminator's ``base`` arm, which
    # forces applied_now (carry_injection_discriminator.py:282); production cfg
    # is leapfrog_rhs.  Match the base arm: one variable (the K33 fold) moves
    # between ON/OFF, placement held at applied_now for BOTH.
    import dataclasses
    cfg_an = dataclasses.replace(cfg, surface_tendency_placement="applied_now")

    def _step(mc, tag):
        jax.clear_caches()
        model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
        st = apply_dino_lat_lon_surface_forcing(
            st0, forcing, br.z_coord, cfg_an, DT, t_seconds=DT)
        st = model.step(st, DT, surface_forcing=sf, external_tracer_rate=None)
        dT = np.asarray(st.T.data) - ns.T
        dS = np.asarray(st.S.data) - ns.S
        sT, sS = _stats(dT, tmask3), _stats(dS, tmask3)
        print(f"\n[{tag}] implicit_K33={mc.gm_redi.implicit_K33} "
              f"msc={mc.gm_redi.msc_stabilize}")
        print(f"  dT max {sT['max']:.4e} @ {sT['argmax']}  "
              f"@anchor{_ANCHOR_CELL} {sT['at_anchor']:.4e}  "
              f"p99.9 {sT['p99.9']:.4e}  p50 {sT['p50']:.4e}")
        print(f"  dS max {sS['max']:.4e} @ {sS['argmax']}  "
              f"@anchor{_ANCHOR_CELL} {sS['at_anchor']:.4e}  "
              f"p99.9 {sS['p99.9']:.4e}  p50 {sS['p50']:.4e}")
        return sT, sS

    on_T, on_S = _step(mc_on, "ON  (production)")

    # Continuity control FIRST (Rule 1e).
    rel = on_T["at_anchor"] / _ANCHOR_MAX
    print(f"\n  CONTINUITY control: ON @anchor {on_T['at_anchor']:.4e} vs "
          f"recorded {_ANCHOR_MAX:.4e}  ratio {rel:.4f}  "
          f"{'OK' if 0.98 < rel < 1.02 else '*** ANCHOR MISMATCH ***'}")

    off_T, off_S = _step(mc_off, "OFF (K33 fold removed)")

    print("\n=== OWNERSHIP VERDICT (OFF/ON; >2x at spike = fold owns it) ===")
    for name, on, off in (("dT", on_T, off_T), ("dS", on_S, off_S)):
        r_anchor = off["at_anchor"] / max(on["at_anchor"], 1e-30)
        r_max = off["max"] / max(on["max"], 1e-30)
        r_p999 = off["p99.9"] / max(on["p99.9"], 1e-30)
        r_p50 = off["p50"] / max(on["p50"], 1e-30)
        print(f"  {name}: @anchor {r_anchor:7.3f}x  max {r_max:7.3f}x  "
              f"p99.9 {r_p999:7.3f}x  p50(bulk) {r_p50:7.3f}x")
    print("\n  Interpretation: OFF/ON ~1x at the spike => the K33 fold does NOT"
          " own the injection (REFUTES the 1c3dfe19b attribution).\n"
          "  OFF/ON >>1x or blow-up => the fold is stabilizing the spike"
          " (fold owns it); a p50(bulk) move is a Rule-8 signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
