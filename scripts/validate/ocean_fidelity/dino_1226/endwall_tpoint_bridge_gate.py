#!/usr/bin/env python3
"""One-step CPU gate for the DINO prior-stress T-point bridge correction.

Registered in ``PREREG_endwall_wind_placement.md`` at commit af3acf4a0.
No GPU integration or model-physics edit is performed here.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parents[3]
for _path in (str(_DIR), str(_DIR.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")
os.environ.setdefault("DINO_1226_IC_STEP", "5760")

import endwall_wind_placement as ewp  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import kamm_twin_90d as kamm  # noqa: E402
import numpy as np  # noqa: E402
import post_tendency_stage_birth as ptsb  # noqa: E402
from legoesm.grids.operators_latlon_cgrid import (  # noqa: E402
    interp_cell_to_uface,
)
from legoesm.ocean.experiments.dino import (  # noqa: E402
    apply_dino_lat_lon_surface_forcing,
)
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask  # noqa: E402

PREREG_COMMIT = "af3acf4a0"
RECEIVER = (1, 49)
RDT = 2.0 * kamm.DT
REMOVAL_BAR = 0.999999
RESTORE_RELATIVE_BAR = 1.0e-6


def _git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", "-C", str(_ROOT), *args], text=True).strip()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_exact(a, b) -> bool:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    return len(leaves_a) == len(leaves_b) and all(
        np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(leaves_a, leaves_b))


def _state_other_leaves_exact(a, b) -> tuple[bool, int]:
    checked = 0
    for name in a._fields:
        if name in ("tau_x_prev", "tau_y_prev"):
            continue
        checked += len(jax.tree_util.tree_leaves(getattr(a, name)))
        if not _tree_exact(getattr(a, name), getattr(b, name)):
            return False, checked
    return True, checked


def _material_support(values: np.ndarray, wet: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(values[wet])))
    if peak == 0.0:
        return np.zeros_like(wet)
    return wet & (np.abs(values) >= 0.01 * peak)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def _removal_fraction(legacy_excess: float, candidate_excess: float) -> float:
    return 1.0 - abs(candidate_excess) / abs(legacy_excess)


def _restore_relative_error(
        legacy_excess: float, candidate_excess: float) -> float:
    return abs(candidate_excess - legacy_excess) / abs(legacy_excess)


def _direct(cap: dict, rho0: float) -> np.ndarray:
    tau = np.asarray(cap["tau_i_u"], dtype=np.float64)
    dz = np.asarray(cap["dz0_u"], dtype=np.float64)
    out = np.zeros_like(tau)
    np.divide(RDT * tau, rho0 * dz, out=out, where=dz > 0.0)
    return out[:, 1:]


def _prepare(state, forcing, z_coord, cfg, first_time: float):
    placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    if placement == "leapfrog_rhs":
        return apply_dino_lat_lon_surface_forcing(
            state, forcing, z_coord, cfg, kamm.DT,
            t_seconds=first_time, return_rate=True)
    return (apply_dino_lat_lon_surface_forcing(
        state, forcing, z_coord, cfg, kamm.DT,
        t_seconds=first_time), None)


def main() -> int:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.precision_gate import require_fp64

    set_policy(PrecisionPolicy.fp64())
    sha = _git(["rev-parse", "HEAD"])
    dirty = _git(["status", "--porcelain", "--untracked-files=no"])
    if dirty:
        raise SystemExit("REFUSING dirty tracked tree before registered CPU gate")

    restart = Path(kamm.RUN_STEPDUMP) / kamm.RESTART_FILE
    mesh = Path(kamm.RUN_TRAJ) / "mesh_mask.nc"
    input_paths = {
        "restart": restart.resolve(),
        "mesh_mask": mesh.resolve(),
        "nemo_zdf_pre": Path(ptsb.dump_lane.dump_path(
            "zdf_dump_u1_prestress.bin")),
        "nemo_zdf_post": Path(ptsb.dump_lane.dump_path(
            "zdf_dump_u1_poststress.bin")),
        "nemo_fslow": Path(ptsb.dump_lane.dump_path(
            "wnd_dump_zu_frc_inc.bin")),
        "probe": Path(__file__).resolve(),
        "harness": Path(kamm.__file__).resolve(),
        "prereg": _DIR / "PREREG_endwall_wind_placement.md",
    }
    hashes = {name: _sha256(path) for name, path in input_paths.items()}
    print("RETRACTION=CONFIRMED_NAMED_AND_APPLIED_DIFFER_2.0966766111")
    print("RETRACTION=receiving_face_coastal_REFUTED_did_not_trace_the_donor")
    print("PROVENANCE " + json.dumps({
        "git_sha": sha,
        "git_dirty": False,
        "prereg_commit": PREREG_COMMIT,
        "inputs": {name: str(path) for name, path in input_paths.items()},
        "input_sha256": hashes,
        "flags": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
            "LEGOESM_NEMO_E3T": os.environ.get("LEGOESM_NEMO_E3T"),
        },
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, sort_keys=True))

    common = dict(
        bridge_before=True, bridge_tke=False,
        surface_stress_implicit=False, e3t_mode="both")
    legacy = kamm._build_twin_state(
        "nemo_dino_kamm_mlf", kamm.RUN_TRAJ, kamm.RUN_STEPDUMP,
        bridge_before_stress_tpoint=False, **common)
    corrected = kamm._build_twin_state(
        "nemo_dino_kamm_mlf", kamm.RUN_TRAJ, kamm.RUN_STEPDUMP,
        bridge_before_stress_tpoint=True, **common)
    br_l, cfg_l, mc_l, _, forcing_l, sf_l, state_l = legacy
    br_t, cfg_t, mc_t, model_t, forcing_t, sf_t, state_t = corrected
    require_fp64(br_t.geometry, br_t.z_coord, state_t,
                 context="endwall T-point bridge CPU gate")

    other_exact, checked_leaves = _state_other_leaves_exact(state_l, state_t)
    stress_changed = not (
        np.array_equal(np.asarray(state_l.tau_x_prev),
                       np.asarray(state_t.tau_x_prev))
        and np.array_equal(np.asarray(state_l.tau_y_prev),
                           np.asarray(state_t.tau_y_prev)))
    build_context_exact = all((
        _tree_exact(br_l.geometry, br_t.geometry),
        _tree_exact(br_l.z_coord, br_t.z_coord),
        _tree_exact(cfg_l, cfg_t),
        _tree_exact(mc_l, mc_t),
        _tree_exact(forcing_l, forcing_t),
        _tree_exact(sf_l, sf_t),
    ))

    # A planted prognostic edit proves the other-leaf identity control fires.
    eta_plant = state_t.eta.replace(data=jnp.asarray(
        np.asarray(state_t.eta.data).copy()).at[1, 1].add(1.0e-6))
    planted_prog = state_t._replace(eta=eta_plant)
    prog_plant_fires = not _state_other_leaves_exact(state_l, planted_prog)[0]
    print("CONTROL identical bridge: "
          f"other_state_leaves_exact={other_exact} checked={checked_leaves} "
          f"stress_changed={stress_changed} "
          f"build_context_exact={build_context_exact} "
          f"prognostic_plant_fires={prog_plant_fires}")
    if not (other_exact and stress_changed and build_context_exact
            and prog_plant_fires):
        raise SystemExit("identical-bridge control failed")

    prior_time = kamm._restart_elapsed_seconds(str(restart))
    first_time = prior_time + kamm.DT
    state_l, ext_l = _prepare(
        state_l, forcing_l, br_l.z_coord, cfg_l, first_time)
    state_t, ext_t = _prepare(
        state_t, forcing_t, br_t.z_coord, cfg_t, first_time)
    prepared_other_exact, _ = _state_other_leaves_exact(state_l, state_t)
    if not prepared_other_exact or not _tree_exact(ext_l, ext_t):
        raise SystemExit("surface-forcing preparation changed a non-stress input")

    # Planted stagger swap: put the legacy U-as-T carry back onto the corrected
    # bridge. All other state leaves remain those of the corrected arm.
    state_p = state_t._replace(
        tau_x_prev=state_l.tau_x_prev, tau_y_prev=state_l.tau_y_prev)
    plant_other_exact, _ = _state_other_leaves_exact(state_p, state_t)
    if not plant_other_exact:
        raise SystemExit("planted stagger swap changed a non-stress state leaf")

    cap_zero = ewp._run_arm(
        model_t, state_t, sf_t, external_rate=ext_t, scale=0.0)
    cap_legacy = ewp._run_arm(
        model_t, state_l, sf_l, external_rate=ext_l, scale=1.0)
    cap_tpoint = ewp._run_arm(
        model_t, state_t, sf_t, external_rate=ext_t, scale=1.0)
    cap_plant = ewp._run_arm(
        model_t, state_p, sf_t, external_rate=ext_t, scale=1.0)

    rho0 = float(mc_t.constants.rho_0)
    direct_l = _direct(cap_legacy, rho0)
    direct_t = _direct(cap_tpoint, rho0)
    direct_p = _direct(cap_plant, rho0)
    fslow_l = (np.asarray(cap_legacy["F_slow_u"])
               - np.asarray(cap_zero["F_slow_u"]))[:, 1:]
    fslow_t = (np.asarray(cap_tpoint["F_slow_u"])
               - np.asarray(cap_zero["F_slow_u"]))[:, 1:]
    fslow_p = (np.asarray(cap_plant["F_slow_u"])
               - np.asarray(cap_zero["F_slow_u"]))[:, 1:]

    nemo_direct = (ptsb._load("zdf_dump_u1_poststress.bin")
                   - ptsb._load("zdf_dump_u1_prestress.bin"))
    nemo_fslow = ptsb._load("wnd_dump_zu_frc_inc.bin")
    grid = read_nemo_mesh_mask(str(mesh), nn_hls=0)
    wet = np.asarray(grid.umask, dtype=bool)[..., 0]

    prior_legacy_ocean = -np.asarray(state_l.tau_x_prev, dtype=np.float64)
    prior_t_ocean = -np.asarray(state_t.tau_x_prev, dtype=np.float64)
    bad_prior_face = np.asarray(
        interp_cell_to_uface(prior_legacy_ocean), dtype=np.float64)[:, 1:]
    good_prior_face = np.asarray(
        interp_cell_to_uface(prior_t_ocean), dtype=np.float64)[:, 1:]
    predicted_tau = 0.5 * (bad_prior_face - good_prior_face)
    dz = np.asarray(cap_tpoint["dz0_u"], dtype=np.float64)[:, 1:]
    predicted_direct = np.zeros_like(predicted_tau)
    np.divide(RDT * predicted_tau, rho0 * dz,
              out=predicted_direct, where=dz > 0.0)

    direct_correction = direct_l - direct_t
    fslow_correction = fslow_l - fslow_t
    predicted_support = _material_support(predicted_direct, wet)
    direct_support = _material_support(direct_correction, wet)
    fslow_support = _material_support(fslow_correction, wet)
    direct_jaccard = _jaccard(predicted_support, direct_support)
    fslow_jaccard = _jaccard(predicted_support, fslow_support)

    legacy_excess = float(direct_l[RECEIVER] - nemo_direct[RECEIVER])
    corrected_excess = float(direct_t[RECEIVER] - nemo_direct[RECEIVER])
    planted_excess = float(direct_p[RECEIVER] - nemo_direct[RECEIVER])
    removal_fraction = _removal_fraction(legacy_excess, corrected_excess)
    planted_restore_relative_error = _restore_relative_error(
        legacy_excess, planted_excess)
    removal_ok = removal_fraction >= REMOVAL_BAR
    planted_restore_ok = planted_restore_relative_error <= RESTORE_RELATIVE_BAR
    # Reachability: leaving the corrected carry in place must fail the restore
    # condition, and scoring the legacy carry as "corrected" must fail removal.
    unplanted_restore_error = _restore_relative_error(
        legacy_excess, corrected_excess)
    restoration_refute_reachable = unplanted_restore_error > RESTORE_RELATIVE_BAR
    legacy_as_candidate_removal = _removal_fraction(
        legacy_excess, legacy_excess)
    removal_refute_reachable = legacy_as_candidate_removal < REMOVAL_BAR

    # The support reducer itself exercises both exact decision extremes.
    synthetic_a = np.zeros_like(wet)
    synthetic_b = np.zeros_like(wet)
    synthetic_c = np.zeros_like(wet)
    synthetic_a.flat[0] = True
    synthetic_b.flat[0] = True
    synthetic_c.flat[1] = True
    support_controls_fire = (
        _jaccard(synthetic_a, synthetic_b) == 1.0
        and _jaccard(synthetic_a, synthetic_c) == 0.0)
    print("CONTROL registered bars: "
          f"removal_refute_reachable={removal_refute_reachable} "
          f"legacy_as_candidate_removal={legacy_as_candidate_removal:.17g} "
          f"restoration_refute_reachable={restoration_refute_reachable} "
          f"unplanted_restore_error={unplanted_restore_error:.17g} "
          f"support_controls_fire={support_controls_fire}")
    if not (removal_refute_reachable and restoration_refute_reachable
            and support_controls_fire):
        raise SystemExit("registered decision control failed")

    result = {
        "provenance": {
            "git_sha": sha,
            "prereg_commit": PREREG_COMMIT,
            "prior_reconstruction_seconds": prior_time,
            "first_step_seconds": first_time,
            "bridge_before_stress_stagger": "T",
            "bridge_before_stress_sha256": kamm._stress_content_sha256(
                state_t.tau_x_prev, state_t.tau_y_prev),
        },
        "receiver": {"j": RECEIVER[0], "i": RECEIVER[1]},
        "direct_receiver": {
            "nemo": float(nemo_direct[RECEIVER]),
            "legacy": float(direct_l[RECEIVER]),
            "tpoint": float(direct_t[RECEIVER]),
            "planted_legacy_stagger": float(direct_p[RECEIVER]),
            "legacy_excess": legacy_excess,
            "tpoint_excess": corrected_excess,
            "planted_excess": planted_excess,
            "legacy_minus_tpoint": float(direct_correction[RECEIVER]),
            "donor_prediction": float(predicted_direct[RECEIVER]),
            "removal_fraction": removal_fraction,
            "removal_bar": REMOVAL_BAR,
        },
        "fslow_receiver": {
            "nemo": float(nemo_fslow[RECEIVER]),
            "legacy": float(fslow_l[RECEIVER]),
            "tpoint": float(fslow_t[RECEIVER]),
            "planted_legacy_stagger": float(fslow_p[RECEIVER]),
            "legacy_excess": float(fslow_l[RECEIVER] - nemo_fslow[RECEIVER]),
            "tpoint_excess": float(fslow_t[RECEIVER] - nemo_fslow[RECEIVER]),
            "legacy_minus_tpoint": float(fslow_correction[RECEIVER]),
        },
        "support": {
            "predicted_count": int(predicted_support.sum()),
            "direct_count": int(direct_support.sum()),
            "fslow_count": int(fslow_support.sum()),
            "predicted_direct_jaccard": direct_jaccard,
            "predicted_fslow_jaccard": fslow_jaccard,
        },
        "planted_stagger_swap": {
            "restore_relative_error": planted_restore_relative_error,
            "restore_relative_bar": RESTORE_RELATIVE_BAR,
            "other_state_leaves_exact": plant_other_exact,
        },
        "controls": {
            "other_state_leaves_exact": other_exact,
            "build_context_exact": build_context_exact,
            "prognostic_plant_fires": prog_plant_fires,
            "removal_refute_reachable": removal_refute_reachable,
            "legacy_as_candidate_removal": legacy_as_candidate_removal,
            "restoration_refute_reachable": restoration_refute_reachable,
            "unplanted_restore_error": unplanted_restore_error,
            "support_controls_fire": support_controls_fire,
        },
    }
    valid = removal_ok and planted_restore_ok
    result["classification"] = (
        "CONFIRMED_TPOINT_BRIDGE_CPU_GATE" if valid
        else "REFUTED_OR_UNRESOLVED_TPOINT_BRIDGE_CPU_GATE")
    print("RESULT " + json.dumps(result, indent=2, sort_keys=True))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
