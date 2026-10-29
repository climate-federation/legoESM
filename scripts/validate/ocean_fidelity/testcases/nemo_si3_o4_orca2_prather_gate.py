#!/usr/bin/env python3
"""Fail-closed reader and 0-ULP production scorer for SI3 rung O4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.si3_orca2_state import (
    SI3AllPratherMoments,
    SI3ORCA2AdvectionGrid,
    SI3ORCA2StateConfig,
    SI3PratherMoments,
    si3_dyn_adv_pra_orca2,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o1_orca2_sas_gate import max_ulp_diff

MAGIC = b"O4PRA001"
HEADER_BYTES = 8 + 4 + 16 * 4 + 4 * 8
STEPS = (1, 2)

GRID = (
    "e1u", "e2v", "r1_e1u", "r1_e2v", "e2u", "e1v", "e1e2t",
    "r1_e1e2t", "tmask", "umask", "vmask", "u_ice", "v_ice",
)
STATE = (
    "ato_i", "h_i", "h_s", "h_ip", "v_i", "v_s", "sv_i", "oa_i",
    "a_i", "a_ip", "v_ip", "v_il", "e_s", "e_i", "szv_i", "wfx_res",
    "hfx_res", "sfx_res", "diag_adv_mass", "diag_adv_salt", "diag_adv_heat",
)
FAMILIES = ("ice", "sn", "a", "age", "sal", "ap", "vp", "vl", "c0", "e", "si")
COMPONENTS = ("sx", "sy", "sxx", "syy", "sxy")
MOMENTS = tuple(f"{family}.{component}" for family in FAMILIES for component in COMPONENTS)
KINDS = {1: GRID, 2: STATE, 3: MOMENTS, 4: (), 5: STATE, 6: MOMENTS}

# Used by the shared static read-set gate.  Entry kinds are 1--4; kinds 5--6
# are outputs and therefore cannot justify a production read.
PROJECTIONS = {(kind, name): name.replace(".", "_") for kind in KINDS for name in KINDS[kind]}
REPO_ROOT = Path(__file__).resolve().parents[4]


def _refuse(message: str) -> ValueError:
    return ValueError(f"STATUS PLANT-FIRED: {message}")


def _shape(kind: int, name: str, meta: dict[str, int]) -> tuple[int, ...]:
    full = (meta["jpi"], meta["jpj"])
    interior = (meta["nie0"] - meta["nis0"] + 1, meta["nje0"] - meta["njs0"] + 1)
    if kind == 1 or name == "ato_i":
        return full
    if name in {"wfx_res", "hfx_res", "sfx_res", "diag_adv_mass", "diag_adv_salt", "diag_adv_heat"}:
        return interior
    family = name.split(".", 1)[0]
    if family == "c0" or name == "e_s":
        return full + (meta["nlay_s"], meta["jpl"])
    if family in {"e", "si"} or name in {"e_i", "szv_i"}:
        return full + (meta["nlay_i"], meta["jpl"])
    return full + (meta["jpl"],)


def read_record(path: str | Path, plant: str | None = None) -> dict[str, Any]:
    """Read one O4 stream record and reject every undeclared byte or extent."""
    p = Path(path)
    raw = bytearray(p.read_bytes())
    if plant == "header" and raw:
        raw[0] ^= 1
    elif plant == "truncation" and raw:
        del raw[-1]
    elif plant == "step" and len(raw) >= 20:
        raw[16:20] = np.asarray([99], dtype="<i4").tobytes()
    if len(raw) < HEADER_BYTES or bytes(raw[:8]) != MAGIC:
        raise _refuse(f"bad/truncated O4 header: {p}")
    version = int(np.frombuffer(raw, dtype="<i4", count=1, offset=8)[0])
    if version != 1:
        raise _refuse(f"unsupported O4 version {version}")
    m = np.frombuffer(raw, dtype="<i4", count=16, offset=12).astype(np.int64)
    keys = ("kind", "kt", "rank", "jpi", "jpj", "jpl", "nlay_i", "nlay_s", "nn_hls",
            "nis0", "nie0", "njs0", "nje0", "nfields", "icycle", "sweep")
    meta = {key: int(value) for key, value in zip(keys, m, strict=True)}
    if meta["kind"] not in KINDS or meta["nfields"] != len(KINDS[meta["kind"]]):
        raise _refuse(f"unknown kind/count in {p}")
    if meta["kt"] not in STEPS or meta["sweep"] != (meta["kt"] - 1) % 2:
        raise _refuse(f"wrong step/sweep metadata in {p}")
    if min(meta[k] for k in ("jpi", "jpj", "jpl", "nlay_i", "nlay_s")) < 1:
        raise _refuse(f"non-positive O4 extent in {p}")
    if (meta["nie0"] - meta["nis0"] + 1 != meta["jpi"] - 2 * meta["nn_hls"] or
            meta["nje0"] - meta["njs0"] + 1 != meta["jpj"] - 2 * meta["nn_hls"]):
        raise _refuse(f"interior/full extent inconsistency in {p}")
    scalars = np.frombuffer(raw, dtype="<f8", count=4, offset=76).copy()
    if not np.all(np.isfinite(scalars)):
        raise _refuse(f"non-finite header scalar in {p}")
    offset = HEADER_BYTES
    fields: dict[str, np.ndarray] = {}
    for name in KINDS[meta["kind"]]:
        shape = _shape(meta["kind"], name, meta)
        count = int(np.prod(shape, dtype=np.int64))
        stop = offset + count * 8
        if stop > len(raw):
            raise _refuse(f"truncated field {name} in {p}")
        fields[name] = np.frombuffer(raw, dtype="<f8", count=count, offset=offset).copy().reshape(shape, order="F")
        offset = stop
    if offset != len(raw):
        raise _refuse(f"undeclared trailing bytes in {p}")
    return {"meta": meta, "scalars": scalars, "fields": fields, "sha256": hashlib.sha256(raw).hexdigest()}


def load_deck(root: str | Path, plant: str | None = None) -> dict[tuple[int, int], dict[str, Any]]:
    root = Path(root)
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for step in STEPS:
        for kind in KINDS:
            path = root / f"o4_kind{kind:02d}_step{step:04d}_rank0000.bin"
            if not path.is_file():
                raise _refuse(f"missing declared O4 record {path.name}")
            records[(step, kind)] = read_record(path, plant if (step, kind) == (1, 1) else None)
    extra = set(root.glob("o4_kind*_step*_rank*.bin")) - {
        root / f"o4_kind{kind:02d}_step{step:04d}_rank0000.bin" for step in STEPS for kind in KINDS
    }
    if extra:
        raise _refuse(f"undeclared O4 records: {sorted(p.name for p in extra)}")
    base = records[(1, 1)]["meta"]
    invariant = ("rank", "jpi", "jpj", "jpl", "nlay_i", "nlay_s", "nn_hls", "nis0", "nie0", "njs0", "nje0")
    for rec in records.values():
        if any(rec["meta"][key] != base[key] for key in invariant):
            raise _refuse("cross-record O4 metadata mismatch")
    return records


def _moments(fields: dict[str, np.ndarray]) -> SI3AllPratherMoments:
    return SI3AllPratherMoments(*(
        SI3PratherMoments(*(jnp.asarray(fields[f"{family}.{c}"]) for c in COMPONENTS))
        for family in FAMILIES
    ))


def _flatten_moments(value: SI3AllPratherMoments) -> dict[str, np.ndarray]:
    return {f"{family}.{component}": np.asarray(getattr(getattr(value, family), component))
            for family in FAMILIES for component in COMPONENTS}


def _pad_interior(a: np.ndarray, meta: dict[str, int]) -> np.ndarray:
    h = meta["nn_hls"]
    out = np.zeros((meta["jpi"], meta["jpj"]), dtype=np.float64)
    out[h:meta["jpi"] - h, h:meta["jpj"] - h] = a
    return out


def _crop_interior(a: np.ndarray, meta: dict[str, int]) -> np.ndarray:
    h = meta["nn_hls"]
    return a[h:meta["jpi"] - h, h:meta["jpj"] - h]


def score_step(records: dict[tuple[int, int], dict[str, Any]], step: int, plant: str | None = None) -> dict[str, Any]:
    """Replay a recorded entry through the production O4 path and demand 0 ULP."""
    grid_f = records[(step, 1)]["fields"]
    entry = records[(step, 2)]["fields"]
    moment_entry = records[(step, 3)]["fields"]
    cfl = records[(step, 4)]
    expected = dict(records[(step, 5)]["fields"])
    meta = records[(step, 1)]["meta"]
    cfg = SI3ORCA2StateConfig.for_orca2_sas_ice(
        jpl=meta["jpl"], nlay_i=meta["nlay_i"], nlay_s=meta["nlay_s"],
        nn_hls=meta["nn_hls"], nn_icesal=4, ln_icedyn=True, ln_icethd=False,
        ln_dynadv_pra=True, rDt_ice=float(cfl["scalars"][0]),
    )
    grid = SI3ORCA2AdvectionGrid(*(jnp.asarray(grid_f[name]) for name in GRID[:11]))
    residual_names = ("wfx_res", "hfx_res", "sfx_res", "diag_adv_mass", "diag_adv_salt", "diag_adv_heat")
    kwargs = {name: jnp.asarray(_pad_interior(entry[name], meta)) for name in residual_names}
    hooks: dict[str, Any] = {}
    if plant == "invert-sweep":
        hooks["_plant_invert_sweep_order"] = True
    elif plant == "wrong-fold-sign":
        hooks["_plant_wrong_nfd_sign"] = "sx"
    elif plant == "omit-subcycle-lbc":
        hooks["_plant_omit_subcycle_lbc"] = True
    out = si3_dyn_adv_pra_orca2(
        kt=step, u_ice=jnp.asarray(grid_f["u_ice"]), v_ice=jnp.asarray(grid_f["v_ice"]),
        h_i=jnp.asarray(entry["h_i"]), h_s=jnp.asarray(entry["h_s"]), h_ip=jnp.asarray(entry["h_ip"]),
        ato_i=jnp.asarray(entry["ato_i"]), v_i=jnp.asarray(entry["v_i"]), v_s=jnp.asarray(entry["v_s"]),
        sv_i=jnp.asarray(entry["sv_i"]), oa_i=jnp.asarray(entry["oa_i"]), a_i=jnp.asarray(entry["a_i"]),
        a_ip=jnp.asarray(entry["a_ip"]), v_ip=jnp.asarray(entry["v_ip"]), v_il=jnp.asarray(entry["v_il"]),
        e_s=jnp.asarray(entry["e_s"]), e_i=jnp.asarray(entry["e_i"]), szv_i=jnp.asarray(entry["szv_i"]),
        prather=_moments(moment_entry), grid=grid, config=cfg, hi_max_top=float(cfl["scalars"][1]),
        cflice_delayed=jnp.asarray(cfl["scalars"][3]), **kwargs, **hooks,
    )
    state_names = STATE[:1] + STATE[4:15]
    actual = {name: np.asarray(value) for name, value in zip(state_names, out[:12], strict=True)}
    actual.update({name: _crop_interior(np.asarray(value), meta) for name, value in zip(residual_names, out[13:19], strict=True)})
    actual.update({"h_i": entry["h_i"], "h_s": entry["h_s"], "h_ip": entry["h_ip"]})
    actual_moments = _flatten_moments(out[12])
    if plant == "one-bit":
        expected["v_i"] = expected["v_i"].copy()
        expected["v_i"].flat[0] = np.nextafter(expected["v_i"].flat[0], np.inf)
    failures: list[str] = []
    max_ulp = 0
    for name, got in actual.items():
        ulp = max_ulp_diff(got, expected[name], strict_signed_zero=True)
        max_ulp = max(max_ulp, ulp)
        if ulp:
            failures.append(f"{name}={ulp}")
    expected_moments = records[(step, 6)]["fields"]
    seam = np.zeros((meta["jpi"], meta["jpj"]), dtype=bool)
    seam[:meta["nn_hls"], :] = True; seam[-meta["nn_hls"]:, :] = True
    fold = np.zeros_like(seam); fold[:, -2 * meta["nn_hls"]:] = True
    wet = np.asarray(grid_f["tmask"]) > 0
    if not np.any(seam & wet) or not np.any(fold & wet):
        raise _refuse("empty wet seam/fold score region")
    regional_max = {"seam": 0, "fold": 0}
    for name, got in actual_moments.items():
        ulp = max_ulp_diff(got, expected_moments[name], strict_signed_zero=True)
        max_ulp = max(max_ulp, ulp)
        if ulp:
            failures.append(f"{name}={ulp}")
        regional_max["seam"] = max(regional_max["seam"], max_ulp_diff(got, expected_moments[name], seam & wet, strict_signed_zero=True))
        regional_max["fold"] = max(regional_max["fold"], max_ulp_diff(got, expected_moments[name], fold & wet, strict_signed_zero=True))
    if int(np.asarray(out[21])) != cfl["meta"]["icycle"]:
        failures.append("icycle")
    if max_ulp_diff(np.asarray(out[19]), np.asarray(cfl["scalars"][3]), strict_signed_zero=True):
        failures.append("delayed-cfl")
    if max_ulp_diff(np.asarray(out[20]), np.asarray(cfl["scalars"][2]), strict_signed_zero=True):
        failures.append("current-cfl")
    if failures:
        raise _refuse(f"O4 step {step} is not 0 ULP: {', '.join(failures[:12])}")
    return {"step": step, "max_ulp": max_ulp, **regional_max, "sweep": "x-y" if step == 1 else "y-x"}


def score_deck(root: str | Path, plant: str | None = None) -> dict[str, Any]:
    set_policy(PrecisionPolicy.fp64())
    jax.config.update("jax_enable_x64", True)
    records = load_deck(root, plant if plant in {"header", "truncation", "step"} else None)
    scores = [score_step(records, step, plant) for step in STEPS]
    return {"status": "PASS", "criterion": "every exit, seam, and fold cell == 0 ULP", "scores": scores}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, raw: str, label: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise _refuse(f"{label} path is not relative: {raw}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise _refuse(f"{label} path escapes evidence: {raw}") from exc
    return resolved


def admit_evidence(root: str | Path, expect_commit: str | None = None,
                   plant: str | None = None) -> dict[str, Any]:
    """Admit a completed reference/record pair against its pre-run manifest."""
    root = Path(root)
    manifest_path = root / "manifest/run_manifest.json"
    if not manifest_path.is_file():
        raise _refuse("missing pre-run manifest")
    manifest = json.loads(manifest_path.read_text())
    required = {"producer_commit", "binary_sha256", "binary_paths", "source_sha256",
                "deck_sha256", "input_sha256", "planned_records", "steps", "ranks"}
    if required - set(manifest):
        raise _refuse(f"manifest missing keys {sorted(required-set(manifest))}")
    producer = str(manifest["producer_commit"])
    if plant == "producer":
        producer = "0" * 40
    if expect_commit is not None and producer != expect_commit:
        raise _refuse(f"producer commit mismatch: {producer} != {expect_commit}")
    try:
        subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{producer}^{{commit}}"],
                       check=True, capture_output=True)
    except subprocess.SubprocessError as exc:
        raise _refuse(f"producer commit is unavailable: {producer}") from exc
    for label, raw in manifest["binary_paths"].items():
        path = _relative(root, raw, f"binary {label}")
        if not path.is_file():
            raise _refuse(f"missing retained {label} binary")
        want = manifest["binary_sha256"].get(label)
        if plant == "binary" and label == sorted(manifest["binary_paths"])[0]:
            want = "0" * 64
        if _sha(path) != want:
            raise _refuse(f"retained {label} binary digest mismatch")
    fixture = bool(manifest.get("fixture", False)) and os.environ.get("O4_DRIVER_FIXTURE_ACTIVE") == "1"
    for raw, want in manifest["source_sha256"].items():
        if Path(raw).is_absolute() or ".." in Path(raw).parts:
            raise _refuse(f"source path is not repository-relative: {raw}")
        if fixture:
            got = _sha(REPO_ROOT / raw)
        else:
            try:
                blob = subprocess.check_output(["git", "-C", str(REPO_ROOT), "show", f"{producer}:{raw}"])
            except subprocess.SubprocessError as exc:
                raise _refuse(f"source is not pinned by producer commit: {raw}") from exc
            got = hashlib.sha256(blob).hexdigest()
        if got != want:
            raise _refuse(f"producer source digest mismatch: {raw}")
    for section in ("deck_sha256", "input_sha256"):
        for raw, want in manifest[section].items():
            path = _relative(root, raw, section)
            if not path.is_file() or _sha(path) != want:
                raise _refuse(f"{section} digest mismatch: {raw}")
    if plant == "missing-complete" or not (root / "RUN_COMPLETE").is_file():
        raise _refuse("missing acquisition RUN_COMPLETE")
    for lane in ("reference_2step", "record_2step"):
        run = root / lane
        for name in ("ocean.output", "run.user.stdout.log", "run.user.stderr.log",
                     "run.user.time.log", "RUN_COMPLETE"):
            if not (run / name).is_file():
                raise _refuse(f"missing {lane}/{name}")
        joined = (run / "ocean.output").read_text(errors="replace") + "\n" + \
                 (run / "run.user.stdout.log").read_text(errors="replace") + "\n" + \
                 (run / "run.user.stderr.log").read_text(errors="replace")
        if plant == "error-log" and lane == "record_2step":
            joined += "\nE R R O R planted"
        if __import__("re").search(r"E\s+R\s+R\s+O\s+R|ABORT|STOP\s+[1-9]", joined, __import__("re").I):
            raise _refuse(f"fatal NEMO token in {lane}")
        if "STOP 0" not in joined or "MPIRUN_RC=0" not in (run / "run.user.time.log").read_text():
            raise _refuse(f"{lane} lacks STOP 0 / successful launcher status")
    ref_restarts = sorted((root / "reference_2step").glob("ORCA2_SAS_00000002_restart*.nc"))
    rec_restarts = sorted((root / "record_2step").glob("ORCA2_SAS_00000002_restart*.nc"))
    if not ref_restarts or [p.name for p in ref_restarts] != [p.name for p in rec_restarts]:
        raise _refuse("reference/record restart sets differ")
    for ref, rec in zip(ref_restarts, rec_restarts, strict=True):
        if plant == "restart" and rec.name.endswith("restart_ice.nc"):
            raise _refuse("observer changed planted ice restart")
        if ref.read_bytes() != rec.read_bytes():
            raise _refuse(f"observer changed restart bytes: {ref.name}")
    records = root / "record_2step/o4_prather_records"
    expected = set(manifest["planned_records"])
    actual = {p.name for p in records.glob("*.bin")}
    if plant == "missing-frame" and actual:
        actual.remove(sorted(actual)[0])
    if actual != expected:
        raise _refuse(f"record set mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    score_plant = "one-bit" if plant == "wet-1ulp" else None
    score = score_deck(records, score_plant)
    checksums = root / "manifest/artifacts.sha256"
    if not checksums.is_file():
        raise _refuse("missing artifact checksum list")
    for line in checksums.read_text().splitlines():
        want, raw = line.split(None, 1)
        path = _relative(root, raw.strip(), "checksum")
        if not path.is_file() or _sha(path) != ("0" * 64 if plant == "checksum" else want):
            raise _refuse(f"artifact checksum mismatch: {raw.strip()}")
    return {"status": "PASS_O4_ADMISSION", "producer_commit": producer,
            "records": len(actual), "restarts": [p.name for p in ref_restarts], "score": score}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, nargs="?")
    parser.add_argument("--admit-evidence", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=("header", "truncation", "step", "one-bit", "invert-sweep", "wrong-fold-sign", "omit-subcycle-lbc",
                                             "producer", "binary", "checksum", "missing-complete", "error-log", "restart", "missing-frame", "wet-1ulp"))
    args = parser.parse_args()
    if args.admit_evidence:
        result = admit_evidence(args.admit_evidence, args.expect_commit, args.plant)
    elif args.records:
        result = score_deck(args.records, args.plant)
    else:
        parser.error("records or --admit-evidence is required")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
