#!/usr/bin/env python3
"""Offline 2x2 association peel of NEMO's retained un_adv trajectory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import netCDF4
import numpy as np


SESSION = "01a04e34-d1fb-73e0-b25a-177641f0a246"
EXPECTED = {
    "substep_dump.bin": "39a2b3f6464758233d75eea1112aebb73b0f21383761e43d2e2d765a94b42e32",
    "mesh_mask.nc": "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
    "spg_dump_un_adv_final.bin": "4d8e7a6445ba465c8229954b443c467862381409805163f2045f5854017917ab",
}
ROUND59_SHA = "85ea27cce4804d98f281940fe472e798d9fa64c741c23bb55e3fca40ee9ca677"
BAR = 1.0e-15
HALO = 2
ZA = (1.781105, -1.06221, 0.281105)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked(root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, text=True).strip()


def _trajectory(path: Path):
    with path.open("rb") as stream:
        jpi, jpj, n_loop = (
            int(value) for value in np.fromfile(stream, dtype="<i4", count=3))
        n2 = jpi * jpj
        names = ("sshn_e", "ssha_e", "zsshp2_e", "un_e", "vn_e",
                 "ua_e", "va_e")
        rows = {}
        for _ in range(n_loop):
            jn = int(np.fromfile(stream, dtype="<i4", count=1)[0])
            rows[jn] = {
                name: np.fromfile(stream, dtype="<f8", count=n2)
                .reshape(jpj, jpi)[HALO:-HALO, HALO:-HALO]
                for name in names
            }
        if stream.read(1):
            raise SystemExit("substep trajectory has trailing bytes")
    return jpi, jpj, n_loop, rows


def _weights(n_loop: int):
    # dynspg_ts::ts_wgt, DINO nn_e=23, ln_bt_fw=F, nn_bt_flt=2.
    nn_e = 23
    primary = np.zeros(3 * nn_e, dtype=np.float64)
    kpit = 0
    for jn in range(1, 3 * nn_e + 1):
        if abs(float(jn - 2 * nn_e)) / float(nn_e) < 1.0:
            primary[jn - 1] = 1.0
            kpit = jn
    raw = np.zeros(kpit, dtype=np.float64)
    for jn in range(kpit):
        for ji in range(jn, kpit):
            raw[jn] = raw[jn] + primary[ji]
    if kpit != n_loop:
        raise SystemExit(f"weight loop {kpit} != dump loop {n_loop}")
    return raw, np.sum(raw, dtype=np.float64)


def _metric(candidate, oracle, wet):
    reference_rms = float(np.sqrt(np.mean(np.square(oracle[wet]))))
    error = np.abs(candidate - oracle) / max(reference_rms, 1.0e-300)
    wet_error = error[wet]
    return {
        "bar": BAR,
        "reference_rms": reference_rms,
        "max_normalized_error": float(np.max(wet_error)),
        "n_diverged_faces": int(np.sum(wet_error > BAR)),
        "n_wet_faces": int(np.sum(wet)),
        "n_nonfinite": int(np.sum(~np.isfinite(candidate[wet]))),
        "pass": bool(np.all(wet_error <= BAR) and np.all(np.isfinite(wet_error))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--round59", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    if os.environ.get("CODEX_SESSION_ID") != SESSION:
        raise SystemExit("registered CODEX_SESSION_ID required")
    if _tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round59.resolve()) != ROUND59_SHA:
        raise SystemExit("official round-59 held receipt changed")
    prior = json.loads(args.round59.read_text())
    if (prior.get("disposition") != "TRACER_ENTRY_DIVERGED_8.3"
            or prior["rows"][0]["metrics"]["n_diverged_columns"] != 104):
        raise SystemExit("round 59 does not admit the residual peel")
    run = args.run_dir.resolve()
    for name, expected in EXPECTED.items():
        if not (run / name).is_file() or _sha(run / name) != expected:
            raise SystemExit(f"retained input changed: {name}")

    jpi, jpj, n_loop, rows = _trajectory(run / "substep_dump.bin")
    if (jpi, jpj, n_loop) != (56, 203, 68):
        raise SystemExit(f"unexpected trajectory shape {(jpi, jpj, n_loop)}")
    with netCDF4.Dataset(run / "mesh_mask.nc") as dataset:
        def read(name):
            return np.asarray(dataset[name][0], dtype=np.float64)

        e3u0 = read("e3u_0")
        umask3 = read("umask")
        e1t = read("e1t")
        e2t = read("e2t")
        e1u = read("e1u")
        e2u = read("e2u")
    wet = umask3[0] > 0.5
    hu0 = np.sum(e3u0 * umask3, axis=0, dtype=np.float64)
    area_t = e1t * e2t
    r1_area_u = 1.0 / (e1u * e2u)
    r1_e2u = 1.0 / e2u
    raw, divisor = _weights(n_loop)
    normalized = raw / divisor
    oracle_full = np.fromfile(
        run / "spg_dump_un_adv_final.bin", dtype="<f8").reshape(jpj, jpi)
    oracle = oracle_full[HALO:-HALO, HALO:-HALO]

    arms = {
        "raw_metric": np.zeros_like(oracle),
        "raw_cancelled": np.zeros_like(oracle),
        "normalized_metric": np.zeros_like(oracle),
        "normalized_cancelled": np.zeros_like(oracle),
    }
    literal_fluxes = []
    for jn in range(1, n_loop + 1):
        un = rows[jn]["un_e"]
        unm1 = rows.get(jn - 1, rows[jn])["un_e"]
        unm2 = rows.get(jn - 2, rows[jn])["un_e"]
        ssh = rows[jn]["sshn_e"]
        sshm1 = rows.get(jn - 1, rows[jn])["sshn_e"]
        sshm2 = rows.get(jn - 2, rows[jn])["sshn_e"]
        if jn < 3:
            ua_mid = un
            ssh_mid = ssh
        else:
            ua_mid = ((ZA[0] * un + ZA[1] * unm1) + ZA[2] * unm2)
            ssh_mid = ((ZA[0] * ssh + ZA[1] * sshm1) + ZA[2] * sshm2)
        east_area = np.roll(area_t, -1, axis=1)
        east_ssh = np.roll(ssh_mid, -1, axis=1)
        area_ssh_sum = area_t * ssh_mid + east_area * east_ssh
        depth = hu0 + (((0.5 * r1_area_u) * area_ssh_sum) * wet)
        zh_u = (e2u * ua_mid) * depth
        metric_flux = zh_u * r1_e2u
        cancelled_flux = depth * ua_mid
        literal_fluxes.append(metric_flux)
        arms["raw_metric"] = (
            arms["raw_metric"] + (raw[jn - 1] * zh_u) * r1_e2u)
        arms["raw_cancelled"] = (
            arms["raw_cancelled"] + raw[jn - 1] * cancelled_flux)
        arms["normalized_metric"] = (
            arms["normalized_metric"]
            + (normalized[jn - 1] * zh_u) * r1_e2u)
        arms["normalized_cancelled"] = (
            arms["normalized_cancelled"]
            + normalized[jn - 1] * cancelled_flux)
    arms["raw_metric"] = arms["raw_metric"] / divisor
    arms["raw_cancelled"] = arms["raw_cancelled"] / divisor
    metrics = {name: _metric(value, oracle, wet)
               for name, value in arms.items()}

    identity = _metric(oracle, oracle, wet)["pass"]
    point = np.array(oracle, copy=True)
    point_idx = tuple(int(value) for value in np.argwhere(wet)[0])
    point[point_idx] = point[point_idx] + 4.0 * BAR * metrics[
        "raw_metric"]["reference_rms"]
    rolled = np.zeros_like(oracle)
    for weight, flux in zip(raw, literal_fluxes[1:] + literal_fluxes[:1]):
        rolled = rolled + (weight * (e2u * flux)) * r1_e2u
    rolled = rolled / divisor
    controls = {
        "identity_at_bar": identity,
        "wet_point_plant_red": not _metric(point, oracle, wet)["pass"],
        "substep_roll_plant_red": not _metric(rolled, oracle, wet)["pass"],
        "cancelled_form_plant_red": not metrics[
            "normalized_cancelled"]["pass"],
        "finite_reconstruction": all(
            row["n_nonfinite"] == 0 for row in metrics.values()),
        "raw_weight_sum_2070": bool(divisor == 2070.0),
    }
    valid = all(controls.values())
    literal_at_bar = metrics["raw_metric"]["pass"]
    nonliteral_red = any(
        not metrics[name]["pass"] for name in arms if name != "raw_metric")
    if not valid:
        disposition = "INVALID_CONTROL"
    elif literal_at_bar and nonliteral_red:
        disposition = "ROW8_3_LITERAL_ACCUMULATOR_AT_BAR_UPSTREAM_OPERANDS_OPEN"
    else:
        disposition = "OPEN_UNRESOLVED_LITERAL_OR_RECONSTRUCTION"
    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round60-v1",
        "session_id": SESSION,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "disposition": disposition,
        "metrics": metrics,
        "controls": controls,
        "bindings": {
            "round59": _sha(args.round59.resolve()),
            **{name: _sha(run / name) for name in EXPECTED},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(
                root / "docs/ocean/fidelity" /
                "PREREG_split_explicit_momentum_chain_round60.md"),
            "nemo_dynspg_ts": _sha(
                nemo / "cfgs/DINO/MY_SRC/dynspg_ts.F90"),
        },
        "ordered_next": (
            "capture_production_Hu_avg" if disposition != "INVALID_CONTROL"
            else "repair_controls"),
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for name, metric in metrics.items():
        print(name, metric["pass"], metric["n_diverged_faces"],
              metric["max_normalized_error"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
