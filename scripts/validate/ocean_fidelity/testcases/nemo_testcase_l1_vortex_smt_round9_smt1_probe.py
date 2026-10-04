#!/usr/bin/env python
"""Round 220 / VORTEX_SMT round 9 -- the SMT-1 rung's NEMO-side instrument.

Three questions, each answered from NEMO's own admitted records and never
from a hand prediction:

1. **Sanity.**  max |ssh| and max |u| at kt=1 and kt=10, and the NaN count.
2. **Does enhanced vertical diffusion ever fire?**  zdfevd.F90:93 writes
   ``p_avt = rn_evd * wmask`` wherever ``MIN( rn2, rn2b ) <= -1.e-12``.  The
   probe forms N^2 at every wet interface of every recorded step with the
   CARD's own equation of state (the deck's S-EOS, decision 69) and counts
   the interfaces that clear that threshold.  Both arms of NEMO's MIN are
   the step-entry tracer under RK3 (stprk3.F90 computes rn2b on Nbb and
   copies rn2 = rn2b before any stage), so one arm per recorded step is the
   whole trigger.
3. **What did the module move on NEMO's side?**  the record-to-record
   difference between the SMT-1 run and the SMT-0 (round 3) run, per field
   per step.  That difference IS the module's signature, independent of
   legoESM.

Usage:
  nemo_testcase_l1_vortex_smt_round9_smt1_probe.py --smt1 <dir> --smt0 <dir>
      [--output probe.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nemo_testcase_phase3_trajectory_gate import read_entry  # noqa: E402

CASE = "VORTEX_SMT1_VEC-zps"
# zdfevd.F90:93 -- the literal threshold, not a rounded one.
EVD_N2_THRESHOLD = -1.0e-12


def _evd_flag(card, T, S):
    """zdfevd's trigger mask on one recorded state, through the model's own
    helper (no second formula: ``convective_K_A_flag`` is the function the
    implicit solve calls, threaded exactly as ``k_profiles`` threads it)."""
    import jax.numpy as jnp
    from legoesm.ocean.physics.convection.enhanced_diffusion import (
        convective_K_A_flag,
    )
    from legoesm.ocean.physics.vertical_mixing.k_profiles import _compute_rho
    from legoesm.ocean.eos import nemo_bn2_live_geometry
    from legoesm.ocean.vertical import compute_ocean_jacobian

    recipe = card.recipe
    z = recipe.z_coord
    st = recipe.initial_state
    st = st._replace(T=st.T.replace(data=jnp.asarray(T)),
                     S=st.S.replace(data=jnp.asarray(S)))
    cc = recipe.model_config.physics.constants
    cfg = recipe.model_config.physics.convection.enhanced_diffusion
    jac = compute_ocean_jacobian(st.eta.data, st.H_bathy.data, z)
    rho = _compute_rho(st, z, jac, eos_fn=None, g=cc.g, rho0=cc.rho_0)
    t_depth, w_depth, e3w = nemo_bn2_live_geometry(
        z, st.eta.data, st.H_bathy.data)
    _, _, flag = convective_K_A_flag(
        rho, z.dz_ref, jac, cfg, T=st.T.data, S=st.S.data,
        t_depth=t_depth, w_depth=w_depth, e3w_int=e3w,
        g=cc.g, rho_ref=cc.rho_0)
    return np.asarray(flag)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smt1", type=Path, required=True)
    ap.add_argument("--smt0", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    card = build_nemo_testcase_card(CASE)

    out: dict = {"case": CASE, "evd_threshold": EVD_N2_THRESHOLD,
                 "smt1_dir": str(args.smt1), "smt0_dir": str(args.smt0),
                 "steps": {}}
    total_fired = 0
    for kt in range(1, args.steps + 1):
        name = f"oracle_entry_kt{kt:08d}.bin"
        a = read_entry(args.smt1 / name, CASE)
        b = read_entry(args.smt0 / name, CASE)
        flag = _evd_flag(card, a["T"], a["S"])
        fired = int(np.count_nonzero(flag))
        total_fired += fired
        row = {
            "nan": int(sum(int(np.isnan(a[f]).sum())
                           for f in ("T", "S", "u", "v", "ssh"))),
            "max_abs_ssh_m": float(np.max(np.abs(a["ssh"]))),
            "max_abs_u_m_s": float(np.max(np.abs(a["u"]))),
            "max_abs_T_K": float(np.max(np.abs(a["T"]))),
            "evd_fired_cells": fired,
            "vs_smt0": {f: float(np.max(np.abs(a[f] - b[f])))
                        for f in ("T", "S", "u", "v", "ssh")},
        }
        out["steps"][str(kt)] = row
    out["evd_fired_cells_total"] = total_fired
    out["verdict_evd_inert"] = total_fired == 0
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
