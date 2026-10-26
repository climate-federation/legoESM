#!/usr/bin/env python3
"""Existing-dump peel for row-1.2's vertical seed association."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import jax
from jax import config as jax_config
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np

jax_config.update("jax_enable_x64", True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_dump(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != 203 * 56:
        raise ValueError(f"unexpected full-halo dump size {raw.size}: {path}")
    return raw.reshape(203, 56)[2:-2, 2:-2]


def left_sum(terms: np.ndarray) -> np.ndarray:
    out = terms[0].copy()
    for k in range(1, terms.shape[0]):
        out = out + terms[k]
    return out


def reverse_sum(terms: np.ndarray) -> np.ndarray:
    out = terms[-1].copy()
    for k in range(terms.shape[0] - 2, -1, -1):
        out = out + terms[k]
    return out


def metric(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict:
    c, o = candidate[mask], oracle[mask]
    rms = float(np.sqrt(np.mean(o * o)))
    diff = c - o
    return {
        "n": int(c.size),
        "normalized_rms_error": float(np.sqrt(np.mean(diff * diff)) / rms),
        "max_error_over_nemo_rms": float(np.max(np.abs(diff)) / rms),
        "bit_mismatch_count": int(np.count_nonzero(c.view(np.uint64) != o.view(np.uint64))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--recurrence", type=Path, required=True)
    ap.add_argument("--prior", type=Path)
    ap.add_argument("--ssh-variable", choices=("sshb", "sshn"), default="sshb")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True).strip():
        raise SystemExit("tracked tree must be clean")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")

    restart = args.run / "DINO_00005760_restart.nc"
    mesh = args.run / "mesh_mask.nc"
    with Dataset(restart) as d:
        ssh = np.asarray(d[args.ssh_variable][0], dtype=np.float64)
        vel = {"u": np.asarray(d["ub"][0, :35], dtype=np.float64),
               "v": np.asarray(d["vb"][0, :35], dtype=np.float64)}
    with Dataset(mesh) as d:
        e3 = {"u": np.asarray(d["e3u_0"][0, :35], dtype=np.float64),
              "v": np.asarray(d["e3v_0"][0, :35], dtype=np.float64)}
        masks = {"u": np.asarray(d["umask"][0, :35]) > 0,
                 "v": np.asarray(d["vmask"][0, :35]) > 0}
        e1t = np.asarray(d["e1t"][0], dtype=np.float64)
        e2t = np.asarray(d["e2t"][0], dtype=np.float64)
        face_area = {
            "u": np.asarray(d["e1u"][0] * d["e2u"][0], dtype=np.float64),
            "v": np.asarray(d["e1v"][0] * d["e2v"][0], dtype=np.float64),
        }
    area_t = e1t * e2t
    hu0 = {q: left_sum(e3[q] * masks[q]) for q in ("u", "v")}
    wet2 = {q: masks[q][0] & (hu0[q] > 0) for q in ("u", "v")}
    if (int(wet2["u"].sum()), int(wet2["v"].sum())) != (9758, 9868):
        raise SystemExit("registered wet populations changed")

    r3 = {
        "u": 0.5 * (area_t * ssh + np.roll(area_t * ssh, -1, axis=1))
             / hu0["u"] / face_area["u"],
        "v": 0.5 * (area_t * ssh + np.roll(area_t * ssh, -1, axis=0))
             / hu0["v"] / face_area["v"],
    }
    oracle = {q: load_dump(args.run / f"spg_dump_{q}n_e_init.bin")
              for q in ("u", "v")}
    scores: dict[str, dict[str, dict]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for q in ("u", "v"):
        m3 = masks[q].astype(np.float64)
        stretch = 1.0 + r3[q][None, ...] * m3
        live_w = (e3[q] * stretch) * m3
        live_terms = (live_w * vel[q]) * m3
        static_terms = ((e3[q] * vel[q]) * m3)
        denom_live = (1.0 / hu0[q]) / (1.0 + r3[q])
        literal = left_sum(live_terms) * denom_live
        static_left = left_sum(static_terms) * (1.0 / hu0[q])
        vector = np.sum(live_terms, axis=0) * denom_live
        reverse = reverse_sum(live_terms) * denom_live
        # The production family is a fused stacked XLA reduction. This arm uses
        # the exact raw inputs and CPU/fp64 backend, without reimplementing a
        # physical operand.
        pair = jnp.sum(jnp.stack([
            jnp.asarray(static_terms), jnp.asarray(e3[q] * m3)], axis=-1),
            axis=0)
        fused = np.asarray(pair[..., 0] / pair[..., 1])
        arrays[q] = {"literal_live_left": literal, "static_left": static_left,
                     "vector_live": vector, "reverse_live": reverse,
                     "production_fused": fused}
        for arm, arr in arrays[q].items():
            scores.setdefault(arm, {})[q] = metric(arr, oracle[q], wet2[q])

    bar = 1.0e-15
    literal_pass = all(scores["literal_live_left"][q]["normalized_rms_error"] <= bar
                       and scores["literal_live_left"][q]["max_error_over_nemo_rms"] <= bar
                       for q in ("u", "v"))
    controls_above = all(any(scores[a][q]["max_error_over_nemo_rms"] > bar
                             for a in ("vector_live", "reverse_live", "production_fused"))
                         for q in ("u", "v"))
    if not literal_pass:
        disposition = "INPUT_OR_TIME_LEVEL_OPEN"
    elif controls_above:
        disposition = "OWNED_BY_LEFT_ACCUMULATION"
    else:
        disposition = "ASSOCIATION_NOT_DISCRIMINATING"
    if (disposition == "OWNED_BY_LEFT_ACCUMULATION"
            and args.ssh_variable == "sshb" and args.prior is not None):
        disposition = "OWNED_BY_BEFORE_THICKNESS_TIME_LEVEL_AND_LEFT_ACCUMULATION"
    planted = {q: arrays[q]["literal_live_left"].copy() for q in ("u", "v")}
    for q in ("u", "v"):
        idx = np.unravel_index(np.argmax(np.abs(oracle[q]) * wet2[q]), oracle[q].shape)
        planted[q][idx] = np.nextafter(planted[q][idx], np.inf)
    controls = {
        "one_ulp_perturbation_detected": all(
            metric(planted[q], arrays[q]["literal_live_left"], wet2[q])["bit_mismatch_count"] == 1
            for q in ("u", "v")),
        "reverse_order_distinct": any(
            np.any(arrays[q]["reverse_live"][wet2[q]].view(np.uint64)
                   != arrays[q]["literal_live_left"][wet2[q]].view(np.uint64))
            for q in ("u", "v")),
    }
    if not all(controls.values()):
        raise SystemExit("red control failed")
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round18-v1",
        "session_id": session, "git_commit": commit,
        "ssh_variable": args.ssh_variable,
        "backend": jax.default_backend(), "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "bar": bar, "disposition": disposition, "scores": scores,
        "controls": controls,
        "input_sha256": {p.name: sha256(p) for p in (
            restart, mesh, args.run / "spg_dump_un_e_init.bin",
            args.run / "spg_dump_vn_e_init.bin", args.recurrence,
            *(() if args.prior is None else (args.prior,)))},
        "script_sha256": sha256(Path(__file__)),
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"ROUND17 disposition={disposition} output={args.output} sha256={sha256(args.output)}")
    for arm in scores:
        print(arm, scores[arm])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
