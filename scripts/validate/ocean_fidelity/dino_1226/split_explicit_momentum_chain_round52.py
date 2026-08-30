#!/usr/bin/env python3
"""Score finalize_lbc, Kmm rewrite, and momentum Asselin tail."""
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

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

import split_explicit_momentum_chain_round47 as r47
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_restart,
    read_nemo_restart_before,
)


ROUND51_SHA = "aaa492d7107ba00096d1d93c9240127856365dae71f56fbe65f41f611e1146c1"
EXPECTED = {
    "DINO_00005760_restart.nc": "0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e",
    "mesh_mask.nc": "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
    "baro_dump_u_before_kt00005761.bin": "dd6aa8b6f3794a5aa3b3bffa2c1f50548792b9adaff1d0ff76e5e29ba42ff8d0",
    "baro_dump_v_before_kt00005761.bin": "ac510a24ae30eab9298cbd57972022dc92c4a0c6e8faf4a196ec5f800ff049df",
    "baro_dump_u_after_kt00005761.bin": "cd9434ab613ee6e2fed0dd5245383d50b365c2347d4c5cc334e262221e673163",
    "baro_dump_v_after_kt00005761.bin": "068b26d53d3bcb88d2853ae62ddb9ebfdc9d063d7241ec747ada8dedbdae8b1f",
    "seq_dump_postlbc_u_aaa_kt00005761.bin": "52a7955d10390464958a85e43593d8f719bff9b9d3a38313aebe3a3e5d48a8c8",
    "seq_dump_postlbc_v_aaa_kt00005761.bin": "7994a578b975d93081679037f7e70d56d6bfbf025097f371c7e28bd6208f1af8",
    "atf_dump_uu_before.bin": "b54ab4ebc1cf106bcf02fcca43a844397c3fd3c498b3e454435b43fea68f652e",
    "atf_dump_vv_before.bin": "d944e7d6dd3c0224b1fbf30a4cd4e7581a4e5daba9d287cc133d2343c269cd3a",
    "atf_dump_uu_after.bin": "f3218ae8a7a4c1aa45c236c5ae850215081355932563dafdf67e01109576759f",
    "atf_dump_vv_after.bin": "28e328347522a4db90b2f6e540465943ba51c0f97b5e9ced1e7b23c123de008c",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _filter(now, before, after, gamma):
    return np.asarray(jnp.asarray(now) + gamma * (
        jnp.asarray(before) - 2.0 * jnp.asarray(now) + jnp.asarray(after)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round51", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if r47.r46.r45.r44._tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round51.resolve()) != ROUND51_SHA:
        raise SystemExit("official round-51 receipt changed")
    prior = json.loads(args.round51.read_text())
    if (prior.get("session_id") != session
            or prior.get("disposition") != "FREE_SURFACE_FILTER_AT_BAR"):
        raise SystemExit("round 51 does not release the momentum tail")
    run = args.run.resolve()
    for name, expected in EXPECTED.items():
        if not (run / name).is_file() or _sha(run / name) != expected:
            raise SystemExit(f"retained momentum-tail input changed: {name}")

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    current = read_nemo_restart(run / "DINO_00005760_restart.nc", nn_hls=0)
    before = read_nemo_restart_before(
        run / "DINO_00005760_restart.nc", nn_hls=0)
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        masks = {
            "u": np.moveaxis(np.asarray(ds["umask"][0]), 0, -1)[..., :35] > .5,
            "v": np.moveaxis(np.asarray(ds["vmask"][0]), 0, -1)[..., :35] > .5,
        }
    arrays = {}
    for c, dump_c in (("u", "u"), ("v", "v")):
        arrays[c] = {
            "baro_before": r47._load3(
                run / f"baro_dump_{dump_c}_before_kt00005761.bin"),
            "baro_after": r47._load3(
                run / f"baro_dump_{dump_c}_after_kt00005761.bin"),
            "postlbc": r47._load3(
                run / f"seq_dump_postlbc_{dump_c}_aaa_kt00005761.bin"),
            "kmm_nemo": r47._load3(
                run / f"atf_dump_{'uu' if c == 'u' else 'vv'}_before.bin"),
            "filtered": r47._load3(
                run / f"atf_dump_{'uu' if c == 'u' else 'vv'}_after.bin"),
            "kmm_production": np.asarray(getattr(current, c))[..., :35],
            "kbb": np.asarray(getattr(before, c))[..., :35],
        }
    gamma = np.float64(0.1)
    rows = {}
    oracle_kmm_filter = {}
    stale_kaa_filter = {}
    for c in ("u", "v"):
        a = arrays[c]
        rows[f"finalize_lbc_{c}"] = r47._metric(
            a["baro_after"], a["postlbc"], masks[c])
        rows[f"kmm_rewrite_{c}"] = r47._metric(
            a["kmm_production"], a["kmm_nemo"], masks[c])
        rows[f"dyn_atf_{c}"] = r47._metric(
            _filter(a["kmm_production"], a["kbb"], a["postlbc"], gamma),
            a["filtered"], masks[c])
        oracle_kmm_filter[c] = r47._metric(
            _filter(a["kmm_nemo"], a["kbb"], a["postlbc"], gamma),
            a["filtered"], masks[c])
        stale_kaa_filter[c] = r47._metric(
            _filter(a["kmm_nemo"], a["kbb"], a["baro_before"], gamma),
            a["filtered"], masks[c])

    controls = {
        "identity_rows_at_bar": all(
            r47._metric(arrays[c]["filtered"], arrays[c]["filtered"], masks[c])[
                "gate_status"] == "AT BAR" for c in ("u", "v")),
        "point_plants_fire": all(
            r47._plant(arrays[c]["filtered"], masks[c]) for c in ("u", "v")),
        "roll_plants_fire": all(
            r47._metric(np.roll(arrays[c]["filtered"], 1, axis=1),
                        arrays[c]["filtered"], masks[c])["gate_status"]
            != "AT BAR" for c in ("u", "v")),
        "oracle_kmm_filters_at_bar": all(
            oracle_kmm_filter[c]["gate_status"] == "AT BAR" for c in ("u", "v")),
        "stale_pre_reconcile_kaa_filters_fail": all(
            stale_kaa_filter[c]["gate_status"] != "AT BAR" for c in ("u", "v")),
        "active_populations_match": (
            int(masks["u"][..., 0].sum()) == 9758
            and int(masks["v"][..., 0].sum()) == 9868),
        "all_finite": all(np.all(np.isfinite(value[masks[c]]))
                          for c in ("u", "v")
                          for value in arrays[c].values()),
    }
    valid = all(controls.values())
    order = ("finalize_lbc_u", "finalize_lbc_v", "kmm_rewrite_u",
             "kmm_rewrite_v", "dyn_atf_u", "dyn_atf_v")
    first = next((name for name in order
                  if rows[name]["gate_status"] != "AT BAR"), None)
    disposition = ("MOMENTUM_TAIL_AT_BAR" if valid and first is None
                   else "INVALID" if not valid
                   else f"MOMENTUM_TAIL_DIVERGED_{first.upper()}")
    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round52-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "chain": "momentum_rhs_tail",
        "rows": rows,
        "oracle_kmm_filter": oracle_kmm_filter,
        "stale_kaa_filter": stale_kaa_filter,
        "controls": controls,
        "bindings": {
            "round51": _sha(args.round51.resolve()),
            **{name: _sha(run / name) for name in EXPECTED},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round52.md"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
            "nemo_dynatf": _sha(nemo / "cfgs/DINO/MY_SRC/dynatf_qco.F90"),
        },
        "disposition": disposition,
        "first_diverged_row": first,
        "ordered_next": "tracer_tail" if disposition
                        == "MOMENTUM_TAIL_AT_BAR" else first,
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
