#!/usr/bin/env python3
"""M-01 follow-up: which ``stp_MLF`` composition is closer to NEMO ITSELF.

``mlf_step_mechanism_ab.py`` (commit 80eaf9b33) measured ``_leapfrog_step`` vs
``_nemo_mlf_step`` against EACH OTHER, one step at a time, both re-based on the
certified arm. That never touched NEMO. This script runs TWO INDEPENDENT
compounding trajectories -- arm A on ``_leapfrog_step``, arm B on
``_nemo_mlf_step`` -- from the SAME bridged day-180 state, and compares BOTH
against NEMO's own per-step restarts at kt=5761..5764 (steps 1-4), which is
the finest exact NEMO ground truth this environment has past day 180 (see
``docs/ocean/fidelity/dino_mlf_composition_ab_preregister.md`` -- day 185/
step 160 itself has no NEMO reference anywhere in the oracle tree; that gap is
preregistered, not discovered here).

Both arms use the SAME ``mc`` (the certified card's resolved config,
``outer_integrator="leapfrog"``, ``implicit_vmix_e3t_now_divisor=False`` --
UNCHANGED) and call the two ``_step_impl``-composition methods directly,
bypassing ``model.step()``'s dispatch -- the same device commit 80eaf9b33
already used to avoid the divisor confound (``outer_integrator="nemo_mlf"``
hard-requires ``implicit_vmix_e3t_now_divisor=True`` at construction,
``ocean_model_latlon_cgrid.py::_validate_config``).

Run (fp64, CPU):

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both python \
        mlf_composition_ab_vs_nemo.py --steps 160
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import numpy as np

import kamm_twin_90d as twin

DINO_CFG = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO"
STEP1_DIR = os.environ.get("DINO_NEMO_RUN_D180_STEP1_1R",
                           f"{DINO_CFG}/RUN_D180_STEP1_1R")

FIELDS = ("T", "S", "u", "v", "eta")


def field_move(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def vs_nemo(st, ns, tmask3, umask3, vmask3, wet2) -> dict:
    """Worst |legoESM - NEMO restart| per field, wet cells only.

    Same indexing convention as ``kamm_twin_90d.verify_day0_matches_restart``
    (u/v face-shifted by one to match NEMO's un/vn convention), extended from
    surface-only to the FULL 3-D field.
    """
    t_l = np.asarray(st.T.data, dtype=np.float64)
    s_l = np.asarray(st.S.data, dtype=np.float64)
    u_l = np.asarray(st.u.data, dtype=np.float64)[:, 1:, :]
    v_l = np.asarray(st.v.data, dtype=np.float64)[1:, :, :]
    eta_l = np.asarray(st.eta.data, dtype=np.float64)
    return {
        "T": float(np.max(np.abs(t_l[tmask3] - ns.T[tmask3]))),
        "S": float(np.max(np.abs(s_l[tmask3] - ns.S[tmask3]))),
        "u": float(np.max(np.abs(u_l[umask3] - ns.u[umask3]))),
        "v": float(np.max(np.abs(v_l[vmask3] - ns.v[vmask3]))),
        "eta": float(np.max(np.abs(eta_l[wet2] - ns.ssh[wet2]))),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--recipe", default="nemo_dino_kamm_mlf")
    p.add_argument("--steps", type=int, default=160, help="160 = 5 days at dt=2700s")
    p.add_argument("--run-traj", default=twin.RUN_TRAJ)
    p.add_argument("--run-stepdump", default=twin.RUN_STEPDUMP)
    p.add_argument("--restart-file", default=twin.RESTART_FILE)
    p.add_argument("--step1-dir", default=STEP1_DIR)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)

    if os.environ.get("JAX_ENABLE_X64") != "1":
        raise SystemExit("run with JAX_ENABLE_X64=1 (Rule 1c: oracle work is fp64)")
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    br, cfg, mc, model, forcing, sf, st0 = twin._build_twin_state(
        args.recipe, args.run_traj, args.run_stepdump,
        bridge_tke=True, use_gm_redi=None, restart_file=args.restart_file)
    if mc.outer_integrator != "leapfrog":
        raise SystemExit(f"{args.recipe} does not resolve to the certified "
                          f"leapfrog composition (got {mc.outer_integrator!r})")
    if getattr(mc, "implicit_vmix_e3t_now_divisor", False):
        raise SystemExit("certified card unexpectedly sets "
                          "implicit_vmix_e3t_now_divisor=True -- the "
                          "one-variable divisor-hold assumption is violated")
    print(f"CERTIFIED CONFIG: outer_integrator={mc.outer_integrator} "
          f"implicit_vmix_e3t_now_divisor={mc.implicit_vmix_e3t_now_divisor} "
          "(unchanged for BOTH arms)", flush=True)

    from legoesm.ocean.fidelity.precision_gate import require_fp64
    require_fp64(br.geometry, st0, context="mlf_composition_ab_vs_nemo")

    from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
    g = read_nemo_mesh_mask(f"{args.run_traj}/mesh_mask.nc", nn_hls=0)
    tmask3 = np.asarray(g.tmask) > 0.5
    umask3 = np.asarray(g.umask) > 0.5
    vmask3 = np.asarray(g.vmask) > 0.5
    wet2 = np.asarray(br.land_mask.data) > 0.5

    dt = twin.DT
    placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    t0_sec = twin.seasonal_t0_seconds(f"{args.run_stepdump}/{args.restart_file}")

    from legoesm.ocean.experiments.dino import apply_dino_lat_lon_surface_forcing

    lf = jax.jit(lambda s, e: model._leapfrog_step(
        s, dt, surface_forcing=sf, external_tracer_rate=e))
    mlf = jax.jit(lambda s, e: model._nemo_mlf_step(
        s, dt, surface_forcing=sf, external_tracer_rate=e))

    def forced(st, k):
        t_sec = t0_sec + (k + 1) * dt
        if placement == "leapfrog_rhs":
            return apply_dino_lat_lon_surface_forcing(
                st, forcing, br.z_coord, cfg, dt, t_seconds=t_sec,
                return_rate=True)
        return apply_dino_lat_lon_surface_forcing(
            st, forcing, br.z_coord, cfg, dt, t_seconds=t_sec), None

    # Determinism control: same arm, same input, twice -> must be exactly 0.0.
    stf0, ext0 = forced(st0, 0)
    a1 = lf(stf0, ext0)
    a2 = lf(stf0, ext0)
    for name in FIELDS:
        d = field_move(getattr(a1, name).data, getattr(a2, name).data)
        if d != 0.0:
            raise SystemExit(f"determinism control failed: leapfrog vs "
                              f"leapfrog moved {name} by {d:.3e}")
    print("determinism control: leapfrog vs leapfrog = 0.0 on all fields",
          flush=True)

    st_lf = st0
    st_mlf = st0
    arm_vs_arm_T = []
    vs_nemo_rows = []
    for k in range(args.steps):
        stf_lf, ext_lf = forced(st_lf, k)
        stf_mlf, ext_mlf = forced(st_mlf, k)
        st_lf = lf(stf_lf, ext_lf)
        st_mlf = mlf(stf_mlf, ext_mlf)

        row_aa = {name: field_move(getattr(st_lf, name).data,
                                   getattr(st_mlf, name).data)
                  for name in FIELDS}
        arm_vs_arm_T.append(row_aa["T"])
        kt = 5760 + (k + 1)
        step_report = {"step": k + 1, "kt": kt, "arm_vs_arm": row_aa}

        if (k + 1) in (1, 2, 3, 4):
            restart_path = f"{args.step1_dir}/DINO_{kt:08d}_restart.nc"
            ns = read_nemo_restart(restart_path, nn_hls=0)
            d_lf = vs_nemo(st_lf, ns, tmask3, umask3, vmask3, wet2)
            d_mlf = vs_nemo(st_mlf, ns, tmask3, umask3, vmask3, wet2)
            step_report["leapfrog_vs_nemo"] = d_lf
            step_report["nemo_mlf_vs_nemo"] = d_mlf
            vs_nemo_rows.append(step_report)
            print(f"step {k+1} (kt={kt}) vs NEMO restart {restart_path}:")
            for name in FIELDS:
                print(f"    {name:4s} leapfrog={d_lf[name]:.6e}  "
                      f"nemo_mlf={d_mlf[name]:.6e}  "
                      f"arm_vs_arm={row_aa[name]:.6e}")
        else:
            if (k + 1) % 20 == 0 or (k + 1) == args.steps:
                print(f"step {k+1}: arm_vs_arm T={row_aa['T']:.6e}", flush=True)

    # Day-5 (step args.steps) arm-vs-arm, no NEMO anchor at this step.
    final_aa = {name: field_move(getattr(st_lf, name).data,
                                 getattr(st_mlf, name).data)
                for name in FIELDS}

    # SCALE-COMPATIBLE check, at step 4 (the last exact NEMO anchor).
    step4 = next(r for r in vs_nemo_rows if r["step"] == 4)
    scale = {}
    for name in FIELDS:
        d_lf = step4["leapfrog_vs_nemo"][name]
        d_mlf = step4["nemo_mlf_vs_nemo"][name]
        closer, farther = min(d_lf, d_mlf), max(d_lf, d_mlf)
        scale[name] = {
            "leapfrog_vs_nemo": d_lf, "nemo_mlf_vs_nemo": d_mlf,
            "closer_arm": "nemo_mlf" if d_mlf <= d_lf else "leapfrog",
            "ratio_farther_over_closer": (farther / closer) if closer > 0 else float("inf"),
            "scale_compatible": bool(closer < 0.5 * farther),
        }

    report = {
        "recipe": args.recipe, "dt": dt, "steps": args.steps,
        "producer_git_sha": _git_sha(),
        "certified_outer_integrator": mc.outer_integrator,
        "certified_implicit_vmix_e3t_now_divisor":
            bool(mc.implicit_vmix_e3t_now_divisor),
        "vs_nemo_steps_1_4": vs_nemo_rows,
        "arm_vs_arm_T_curve": arm_vs_arm_T,
        "day5_arm_vs_arm_no_nemo_anchor": final_aa,
        "scale_compatible_at_step4": scale,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


def _git_sha() -> str:
    from legoesm.ocean.fidelity.provenance import git_sha
    try:
        return git_sha(allow_dirty=True)
    except RuntimeError:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
