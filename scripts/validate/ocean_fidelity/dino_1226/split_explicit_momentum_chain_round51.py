#!/usr/bin/env python3
"""Score the retained free-surface-filter chain in source order."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

import kamm_twin_90d as twin
import split_explicit_momentum_chain_round47 as r47
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_io import read_nemo_restart_before
from legoesm.ocean.vertical import nemo_qco_live_face_geometry_from_operands


ROUND50_SHA = "9d17606b1dfe4190456df58cb1a0e4aaac6e6456c86849fb93dbfc39f3b7dc80"
EXPECTED = {
    "DINO_00005760_restart.nc": "0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e",
    "mesh_mask.nc": "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
    "atf_dump_ssh_before.bin": "a54fd5c0c71ae08acba5dec5a94aacffc9c0a31ee64d1c595f2b3443a38ff184",
    "atf_dump_ssh_after.bin": "ea3e234c9c8cbcaaea915f3e3e35ff3c18fab74cec3667e407e95654886eaf02",
    "spg_dump_pssh_final.bin": "7db66cadeb8d04522885a766fd1dd77e010a9adcaa5e8c07ce832a5339928c70",
    "sshnxt_dump_ssh_after.bin": "4d0ba9b100ba12084029ba611b4f03a5ef9f96f2b1f47ba7033e41e70ac0abae",
    "seq_dump_r3t_f_kt00005761.bin": "45caf97d057958c507de9a8ab9a2dfe83001cd43fda3c8e3400d52e0ae0cba48",
    "seq_dump_r3u_f_kt00005761.bin": "999d1c66413e81e6928f344c07979b8ef67a02ea3a4454ee2663ca9299c18d23",
    "seq_dump_r3v_f_kt00005761.bin": "18fc7c0ff506b6b0131adf39e823aac1f9a40034aa1c70693c9de2f8da38989c",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load2(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != 203 * 56:
        raise SystemExit(f"{path}: expected full-halo (203,56) stream")
    return raw.reshape(203, 56)[2:-2, 2:-2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--round50", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if r47.r46.r45.r44._tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round50.resolve()) != ROUND50_SHA:
        raise SystemExit("official round-50 receipt changed")
    prior = json.loads(args.round50.read_text())
    if (prior.get("session_id") != session or prior.get("disposition")
            != "ROW6_MLF_BARO_CORR_AT_BAR_UPSTREAM_TARGET_EXACT"):
        raise SystemExit("round 50 does not release the filter chain")
    run = args.run_stepdump.resolve()
    for name, expected in EXPECTED.items():
        if not (run / name).is_file() or _sha(run / name) != expected:
            raise SystemExit(f"retained free-surface input changed: {name}")

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    _, _, cfg, model, _, _, _ = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()), str(run),
        bridge_before=True, restart_file="DINO_00005760_restart.nc")
    gamma = np.float64(cfg.asselin_gamma)
    before = np.asarray(read_nemo_restart_before(
        run / "DINO_00005760_restart.nc", nn_hls=0).ssh, dtype=np.float64)
    now = _load2(run / "atf_dump_ssh_before.bin")
    kaa = _load2(run / "spg_dump_pssh_final.bin")
    stale_kaa = _load2(run / "sshnxt_dump_ssh_after.bin")
    after = _load2(run / "atf_dump_ssh_after.bin")
    eta_f = np.asarray(jnp.asarray(now) + gamma * (
        jnp.asarray(before) - 2.0 * jnp.asarray(now) + jnp.asarray(kaa)))
    eta_stale = np.asarray(jnp.asarray(now) + gamma * (
        jnp.asarray(before) - 2.0 * jnp.asarray(now) + jnp.asarray(stale_kaa)))

    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        tmask3 = np.moveaxis(np.asarray(ds["tmask"][0]), 0, -1)[..., :35]
        umask3 = np.moveaxis(np.asarray(ds["umask"][0]), 0, -1)[..., :35]
        vmask3 = np.moveaxis(np.asarray(ds["vmask"][0]), 0, -1)[..., :35]
    masks = {"ssh_atf": tmask3[..., 0] > .5,
             "r3t_f": tmask3[..., 0] > .5,
             "r3u_f": umask3[..., 0] > .5,
             "r3v_f": vmask3[..., 0] > .5}
    zc = model.z_coord
    e3t0 = jnp.asarray(zc.nemo_e3t_0[..., :35], dtype=jnp.float64)
    tmask = jnp.asarray(tmask3, dtype=jnp.float64)
    ht0 = jnp.sum(e3t0 * tmask, axis=-1)
    wet_t = tmask[..., 0]
    r1_ht0 = wet_t / (ht0 + 1.0 - wet_t)
    r3t = np.asarray(jnp.asarray(eta_f) * r1_ht0)
    geom = nemo_qco_live_face_geometry_from_operands(
        jnp.asarray(eta_f), e3t0, e3t0,
        jnp.asarray(umask3), jnp.asarray(vmask3),
        zc.nemo_hu_0, zc.nemo_hv_0,
        zc.nemo_e1e2t, zc.nemo_e1e2u, zc.nemo_e1e2v)
    candidates = {
        "ssh_atf": eta_f,
        "r3t_f": r3t,
        "r3u_f": np.asarray(geom.r3u),
        "r3v_f": np.asarray(geom.r3v),
    }
    oracle = {
        "ssh_atf": after,
        "r3t_f": _load2(run / "seq_dump_r3t_f_kt00005761.bin"),
        "r3u_f": _load2(run / "seq_dump_r3u_f_kt00005761.bin"),
        "r3v_f": _load2(run / "seq_dump_r3v_f_kt00005761.bin"),
    }
    rows = {
        name: r47._metric(candidates[name], oracle[name], masks[name])
        for name in candidates
    }
    controls = {
        "identity_rows_at_bar": all(
            r47._metric(oracle[n], oracle[n], masks[n])["gate_status"]
            == "AT BAR" for n in oracle),
        "point_plants_fire": all(
            r47._plant(oracle[n], masks[n]) for n in oracle),
        "roll_plants_fire": all(
            r47._metric(np.roll(oracle[n], 1, axis=1), oracle[n], masks[n])[
                "gate_status"] != "AT BAR" for n in oracle),
        "stale_pre_spg_kaa_fails_ssh": r47._metric(
            eta_stale, after, masks["ssh_atf"])["gate_status"] != "AT BAR",
        "active_populations_match": (
            int(masks["r3t_f"].sum()) == 9920
            and int(masks["r3u_f"].sum()) == 9758
            and int(masks["r3v_f"].sum()) == 9868),
        "all_finite": all(np.all(np.isfinite(v[masks[n]]))
                          for n, v in candidates.items()),
    }
    valid = all(controls.values())
    order = ("ssh_atf", "r3t_f", "r3u_f", "r3v_f")
    first = next((name for name in order
                  if rows[name]["gate_status"] != "AT BAR"), None)
    disposition = ("FREE_SURFACE_FILTER_AT_BAR" if valid and first is None
                   else "INVALID" if not valid
                   else f"FREE_SURFACE_FILTER_DIVERGED_{first.upper()}")
    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round51-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "chain": "free_surface_filter",
        "rows": rows,
        "controls": controls,
        "stale_kaa_arm": r47._metric(eta_stale, after, masks["ssh_atf"]),
        "bindings": {
            "round50": _sha(args.round50.resolve()),
            **{name: _sha(run / name) for name in EXPECTED},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round51.md"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
            "nemo_sshwzv": _sha(nemo / "cfgs/DINO/MY_SRC/sshwzv.F90"),
            "nemo_domqco": _sha(nemo / "src/OCE/DOM/domqco.F90"),
        },
        "disposition": disposition,
        "first_diverged_row": first,
        "ordered_next": "momentum_rhs_tail" if disposition
                        == "FREE_SURFACE_FILTER_AT_BAR" else first,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition} first_diverged_row={first}")
    for name in order:
        print(name, rows[name]["gate_status"],
              rows[name]["normalized_rms_error"],
              rows[name]["per_element_max_error_over_nemo_rms"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
