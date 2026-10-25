#!/usr/bin/env python3
"""Fail-closed record reader and exact-input admission gate for SI3 rung O5.

The binary helpers, ULP metric, signed-zero control, log parser, and normalized
same-program proof are imported from the admitted O1/O2 acquisition tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.dynamics import (
    SI3CGridAEVPConfig,
    SI3CGridAEVPForcing,
    SI3CGridAEVPState,
    SI3CGridMetrics,
    SI3ORCA2FoldProgram,
    SI3ORCA2StrengthConfig,
    si3_orca2_aevp_solver,
    si3_orca2_ice_strength,
)

from legoesm import constants
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round328_dynzdf_gate import (
    GateError as SameProgramError,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round328_dynzdf_gate import (
    same_program_observers,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o1_orca2_sas_gate import (
    first_wet_ij,
    max_normalized_diff,
    max_ulp_diff,
    verify_dry_signed_zero_fixture,
    verify_ocean_output_log,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MAGIC = b"O5SI3R01"
HEADER_BYTES = 8 + 4 + 18 * 4 + 2 * 8
FRAMES = (1, 2, 3, 50, 99, 100)
KINDS: dict[int, tuple[tuple[str, int], ...]] = {
    1: (("a_i", 3), ("v_i", 3), ("at_i_entry", 2), ("tmask", 2)),
    2: (("at_i_raw", 2), ("strength_raw", 2)),
    3: (("strength", 2),),
    4: tuple(
        (name, 2)
        for name in (
            "u_ice",
            "v_ice",
            "u_ice_b",
            "v_ice_b",
            "zs1",
            "zs2",
            "zs12",
            "strength_entry",
            "at_i",
            "vt_i",
            "vt_s",
            "vt_ip",
            "vt_il",
            "utau_ice",
            "vtau_ice",
            "drag_io",
            "u_oce",
            "v_oce",
            "ssh_m",
            "snwice_mass",
            "snwice_mass_b",
            "zsshdyn",
            "ff_t",
        )
    ),
    5: tuple(
        (name, 2)
        for name in (
            "tmask",
            "umask",
            "vmask",
            "fast_umask",
            "fast_vmask",
            "fast_tmask",
            "icb_umask",
            "icb_vmask",
            "icb_tmask",
        )
    ),
    6: tuple(
        (name, 2)
        for name in (
            "e1t",
            "e2t",
            "e1u",
            "e2u",
            "e1v",
            "e2v",
            "e1f",
            "e2f",
            "e1e2t",
            "e1e2u",
            "e1e2v",
            "e1e2f",
            "r1_e1u",
            "r1_e2v",
            "r1_e1e2t",
            "r1_e1e2u",
            "r1_e1e2v",
            "r1_e1e2f",
        )
    ),
    7: (("fimask", 2),),
    8: (("zdelta", 2), ("zp_delt", 2)),
    9: (("zs1", 2), ("zs2", 2), ("zs12", 2)),
    10: (("u_ice", 2), ("v_ice", 2)),
    11: tuple(
        (name, 2)
        for name in (
            "stress1",
            "stress2",
            "stress12",
            "u_ice",
            "v_ice",
            "divu",
            "shear",
            "delta",
        )
    ),
    12: (("a_i", 3), ("v_i", 3)),
    13: (("a_i", 3), ("v_i", 3)),
}


def _refuse(message: str) -> ValueError:
    return ValueError(f"STATUS PLANT-FIRED: {message}")


def read_o5_record(path: str | Path, plant: str | None = None) -> dict[str, Any]:
    """Read one declared-extent O5 raw record, retaining all float64 bit patterns."""
    p = Path(path)
    raw = bytearray(p.read_bytes())
    if plant == "header" and len(raw) >= 1:
        raw[0] ^= 1
    elif plant == "truncation" and raw:
        del raw[-1]
    if len(raw) < HEADER_BYTES:
        raise _refuse(f"truncated O5 header in {p}")
    if bytes(raw[:8]) != MAGIC:
        raise _refuse(f"bad O5 magic in {p}")
    version = int(np.frombuffer(raw, dtype="<i4", count=1, offset=8)[0])
    if version != 1:
        raise _refuse(f"unsupported O5 version {version}")
    meta = np.frombuffer(raw, dtype="<i4", count=18, offset=12).astype(np.int64)
    (
        kind,
        kt,
        jter,
        kmm,
        rank,
        jpi,
        jpj,
        jpl,
        nn_hls,
        ncom_stp,
        nit000,
        jpni,
        jpnj,
        ln_nnogather,
        nn_npfchk,
        nfields,
        endian,
        _,
    ) = map(int, meta)
    if kind not in KINDS or nfields != len(KINDS[kind]):
        raise _refuse(f"unknown kind/field count {kind}/{nfields}")
    if (
        jpi < 1
        or jpj < 1
        or jpl != 5
        or nn_hls != 2
        or rank < 0
        or jpni != 1
        or jpnj != 1
        or endian != 1
    ):
        raise _refuse(f"invalid declared extents/decomposition {meta.tolist()}")
    match = re.search(r"o5_kind(\d+)_step(\d+)_iter(\d+)_rank(\d+)\.bin$", p.name)
    if not match or tuple(map(int, match.groups())) != (kind, kt, jter, rank):
        raise _refuse(f"filename/header disagreement in {p.name}")
    rmeta = np.frombuffer(raw, dtype="<f8", count=2, offset=84)
    offset = HEADER_BYTES
    fields: dict[str, np.ndarray] = {}
    for name, ndim in KINDS[kind]:
        shape = (jpi, jpj, jpl) if ndim == 3 else (jpi, jpj)
        count = int(np.prod(shape))
        stop = offset + count * 8
        if stop > len(raw):
            raise _refuse(f"truncated field {name} in {p}")
        fields[name] = (
            np.frombuffer(raw, dtype="<f8", count=count, offset=offset)
            .reshape(shape, order="F")
            .copy()
        )
        offset = stop
    if offset != len(raw):
        raise _refuse(f"unexpected trailing bytes in {p}: {len(raw) - offset}")
    return {
        "meta": {
            "kind": kind,
            "kt": kt,
            "jter": jter,
            "kmm": kmm,
            "rank": rank,
            "jpi": jpi,
            "jpj": jpj,
            "jpl": jpl,
            "nn_hls": nn_hls,
            "ncom_stp": ncom_stp,
            "nit000": nit000,
            "jpni": jpni,
            "jpnj": jpnj,
            "ln_nnogather": bool(ln_nnogather),
            "nn_npfchk": nn_npfchk,
            "rDt_ice": float(rmeta[0]),
        },
        "fields": fields,
    }


def _write_synthetic_record(
    path: Path, kind: int, *, kt: int = 1, jter: int = 0, kmm: int = 1
) -> None:
    jpi, jpj, jpl = 8, 7, 5
    meta = np.array(
        [kind, kt, jter, kmm, 0, jpi, jpj, jpl, 2, 1, 1, 1, 1, 1, 2, len(KINDS[kind]), 1, 0],
        dtype="<i4",
    )
    with path.open("wb") as out:
        out.write(MAGIC)
        out.write(np.array([1], dtype="<i4").tobytes())
        out.write(meta.tobytes())
        out.write(np.array([10800.0, 0.0], dtype="<f8").tobytes())
        for index, (_name, ndim) in enumerate(KINDS[kind], start=1):
            shape = (jpi, jpj, jpl) if ndim == 3 else (jpi, jpj)
            values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape, order="F") + index
            out.write(np.asfortranarray(values).tobytes(order="F"))


def _sha256_manifest(path: Path, *, plant: bool = False) -> int:
    if not path.is_file():
        raise _refuse(f"missing checksum manifest {path}")
    checked = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise _refuse(f"malformed checksum line {line_no}")
        target = Path(parts[1])
        expected = parts[0]
        if not target.is_absolute():
            target = path.parent / target
        if plant and checked == 0:
            expected = "0" * 64
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise _refuse(f"checksum mismatch for {target}")
        checked += 1
    if checked == 0:
        raise _refuse("empty checksum manifest")
    return checked


def verify_o5_log(path: str | Path, plant: str | None = None) -> dict[str, Any]:
    """Reuse O1's real-log parser, then enforce O5's resolved overlay."""
    base = verify_ocean_output_log(path, expect_icedyn=True, expect_icethd=False)
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    expected = {
        "ln_ice_embd": False,
        "ln_dynALL": True,
        "ln_dynRHGADV": False,
        "ln_ridging": False,
        "ln_rafting": False,
        "ln_str_H79": True,
        "ln_str_smooth": True,
        "ln_aEVP": True,
    }
    if plant == "selector":
        text = re.sub(r"(ln_ridging\s*=\s*)F", r"\1T", text, count=1)
    parsed: dict[str, Any] = {}
    for name, want in expected.items():
        match = re.search(
            rf"\b{re.escape(name)}\s*=\s*(\.?true\.?|\.?false\.?|T|F)\b", text, flags=re.I
        )
        if not match:
            raise _refuse(f"missing O5 selector {name}")
        got = match.group(1).strip(".").upper() in {"T", "TRUE"}
        if got is not want:
            raise _refuse(f"O5 selector {name}={got}, expected {want}")
        parsed[name] = got
    nevp = re.search(r"\bnn_nevp\s*=\s*(\d+)", text)
    if not nevp or int(nevp.group(1)) != 100:
        raise _refuse("nn_nevp is not 100")
    return {**base, "o5_overlay": parsed, "nn_nevp": 100}


def same_program(
    new_bld: Path,
    ref_bld: Path,
    patches: dict[str, Path],
    writer: Path,
    *,
    new_cfg: str,
    ref_cfg: str,
) -> dict[str, Any]:
    try:
        result = same_program_observers(
            new_bld, ref_bld, patches, (writer.stem,), new_cfg=new_cfg, ref_cfg=ref_cfg
        )
    except SameProgramError as exc:
        raise _refuse(str(exc)) from exc
    return {
        **result,
        "status": "PASS_SAME_PROGRAM",
        "reference_bld": str(ref_bld),
        "new_bld": str(new_bld),
    }


def _compare(
    label: str, actual: np.ndarray, expected: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    wet = max_ulp_diff(actual, expected, mask=mask, strict_signed_zero=True)
    full = max_ulp_diff(actual, expected, strict_signed_zero=True)
    norm = max_normalized_diff(actual, expected, mask=mask)
    seam_mask = np.zeros_like(mask)
    seam_mask[:2, :] = mask[:2, :]
    seam_mask[-2:, :] = mask[-2:, :]
    fold_mask = np.zeros_like(mask)
    fold_mask[:, -3:] = mask[:, -3:]
    seam = max_ulp_diff(actual, expected, mask=seam_mask) if np.any(seam_mask > 0) else 0
    fold = max_ulp_diff(actual, expected, mask=fold_mask) if np.any(fold_mask > 0) else 0
    if wet != 0 or full != 0:
        raise _refuse(f"{label} is not exact: wet={wet} full={full} norm={norm}")
    return {
        "max_ulp_wet": wet,
        "max_ulp_full": full,
        "max_norm_wet": norm,
        "seam_max_ulp": seam,
        "fold_max_ulp": fold,
    }


def _config(n: int, dt: float) -> SI3CGridAEVPConfig:
    return SI3CGridAEVPConfig(
        scheme="si3_aevp",
        staggering="si3_c_grid",
        dt_s=dt,
        n_subcycles=n,
        eccentricity=2.0,
        creep_limit_s_inv=2.0e-9,
        strength_parameter_pa=2.0e4,
        strength_decay=20.0,
        rho_snow=constants.rho_snow,
        rho_ice=constants.rho_ice,
        rho_water=constants.rho_water,
        rho_ocean=constants.rho_ocean_nemo,
        gravity=constants.g_nemo,
        rn_ishlat=2.0,
        halo_width=2,
        category_count=5,
        landfast=False,
        landfast_depth_fraction=0.125,
        landfast_basal_friction_n_m3=15.0,
        landfast_relaxation_s_inv=1.0e-5,
        landfast_tensile_fraction=0.05,
        convergence_check=0,
    )


def _load_step(
    records: Path, kt: int, *, plant: str | None = None
) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    kinds = tuple(kind for kind in KINDS if not (kind == 7 and kt != 1))
    expected = [
        (kind, frame if kind in (8, 9, 10) else 0)
        for kind in kinds
        for frame in (FRAMES if kind in (8, 9, 10) else (0,))
    ]
    for kind, frame in expected:
        path = records / f"o5_kind{kind:02d}_step{kt:04d}_iter{frame:03d}_rank0000.bin"
        if not path.is_file():
            raise _refuse(f"missing frame {path.name}")
        result[(kind, frame)] = read_o5_record(path)
    levels = {item["meta"]["kmm"] for item in result.values()}
    if plant == "time-level":
        levels.add(next(iter(levels)) + 1)
    if levels != {2}:
        raise _refuse(f"SAS Nnn/Kmm must be level 2 at kt={kt}, got {levels}")
    return result


def _score_step(step: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    s1, s2, s3 = (step[(kind, 0)]["fields"] for kind in (1, 2, 3))
    core, masks, metric, out = (step[(kind, 0)]["fields"] for kind in (4, 5, 6, 11))
    fold = SI3ORCA2FoldProgram(nn_hls=2, nfd_full_when_absent=True)
    strength_cfg = SI3ORCA2StrengthConfig(True, 2.0e4, 20.0, True)
    strength_fn = jax.jit(
        lambda a, v, t: si3_orca2_ice_strength(
            a_i=a, v_i=v, tmask=t, config=strength_cfg, fold=fold
        )
    )
    strength = strength_fn(jnp.asarray(s1["a_i"]), jnp.asarray(s1["v_i"]), jnp.asarray(s1["tmask"]))
    scores = {
        "at_i_entry_unchanged": _compare(
            "A1/A2 at_i", s2["at_i_raw"], s1["at_i_entry"], masks["tmask"]
        ),
        "at_i": _compare("strength at_i", strength.at_i, s2["at_i_raw"], masks["tmask"]),
        "strength_raw": _compare(
            "strength raw", strength.strength_raw, s2["strength_raw"], masks["tmask"]
        ),
        "strength": _compare("strength smooth", strength.strength, s3["strength"], masks["tmask"]),
        "zsshdyn": _compare("levitating zsshdyn", core["zsshdyn"], core["ssh_m"], masks["tmask"]),
    }
    metric_pairs = (
        ("e1e2t", "e1t", "e2t", "tmask"),
        ("e1e2u", "e1u", "e2u", "umask"),
        ("e1e2v", "e1v", "e2v", "vmask"),
        ("e1e2f", "e1f", "e2f", "tmask"),
    )
    for area, e1, e2, mask_name in metric_pairs:
        product = jax.jit(lambda x, y: x * y)(jnp.asarray(metric[e1]), jnp.asarray(metric[e2]))
        scores[f"metric_{area}"] = _compare(area, product, metric[area], masks[mask_name])
    reciprocal_pairs = (
        ("r1_e1u", "e1u", "umask"),
        ("r1_e2v", "e2v", "vmask"),
        ("r1_e1e2t", "e1e2t", "tmask"),
        ("r1_e1e2u", "e1e2u", "umask"),
        ("r1_e1e2v", "e1e2v", "vmask"),
        ("r1_e1e2f", "e1e2f", "tmask"),
    )
    for recip, denominator, mask_name in reciprocal_pairs:
        value = jax.jit(lambda x: 1.0 / x)(jnp.asarray(metric[denominator]))
        scores[f"metric_{recip}"] = _compare(recip, value, metric[recip], masks[mask_name])
    forcing = SI3CGridAEVPForcing(
        concentration_t=jnp.asarray(core["at_i"]),
        ice_volume_t=jnp.asarray(core["vt_i"]),
        snow_volume_t=jnp.asarray(core["vt_s"]),
        pond_volume_t=jnp.asarray(core["vt_ip"]),
        lid_volume_t=jnp.asarray(core["vt_il"]),
        air_stress_u_t=jnp.asarray(core["utau_ice"]),
        air_stress_v_t=jnp.asarray(core["vtau_ice"]),
        drag_io_t=jnp.asarray(core["drag_io"]),
        ocean_u_u=jnp.asarray(core["u_oce"]),
        ocean_v_v=jnp.asarray(core["v_oce"]),
        ssh_t=None,
        coriolis_t=jnp.asarray(core["ff_t"]),
        tmask_t=jnp.asarray(masks["tmask"]),
        umask_u=jnp.asarray(masks["umask"]),
        vmask_v=jnp.asarray(masks["vmask"]),
        fast_tmask=jnp.asarray(masks["fast_tmask"]),
        depth_t=jnp.zeros_like(jnp.asarray(masks["tmask"])),
        depth_u=jnp.zeros_like(jnp.asarray(masks["tmask"])),
        depth_v=jnp.zeros_like(jnp.asarray(masks["tmask"])),
        iceberg_tmask=jnp.asarray(masks["icb_tmask"]),
        iceberg_umask=jnp.asarray(masks["icb_umask"]),
        iceberg_vmask=jnp.asarray(masks["icb_vmask"]),
    )
    state = SI3CGridAEVPState(
        jnp.asarray(core["u_ice"]),
        jnp.asarray(core["v_ice"]),
        jnp.asarray(core["zs1"]),
        jnp.asarray(core["zs2"]),
        jnp.asarray(core["zs12"]),
    )
    metrics = SI3CGridMetrics(
        *(
            jnp.asarray(metric[name])
            for name in (
                "e1t",
                "e2t",
                "e1u",
                "e2u",
                "e1v",
                "e2v",
                "e1f",
                "e2f",
                "e1e2t",
                "e1e2u",
                "e1e2v",
                "e1e2f",
            )
        )
    )
    dt = step[(1, 0)]["meta"]["rDt_ice"]
    common = dict(
        strength_t=jnp.asarray(core["strength_entry"]),
        u_ice_b=jnp.asarray(core["u_ice_b"]),
        v_ice_b=jnp.asarray(core["v_ice_b"]),
        ssh_m=jnp.asarray(core["ssh_m"]),
        snwice_mass=jnp.asarray(core["snwice_mass"]),
        snwice_mass_b=jnp.asarray(core["snwice_mass_b"]),
        ln_ice_embd=False,
        fold=fold,
        differentiable=False,
    )
    solve = jax.jit(
        lambda: si3_orca2_aevp_solver(state, forcing, metrics, _config(100, dt), **common)
    )
    solved = solve()
    for label, actual, expected, mask in (
        ("stress1", solved.stress1_t, out["stress1"], masks["tmask"]),
        ("stress2", solved.stress2_t, out["stress2"], masks["tmask"]),
        ("stress12", solved.stress12_f, out["stress12"], masks["tmask"]),
        ("u_ice", solved.u_ice_u, out["u_ice"], masks["umask"]),
        ("v_ice", solved.v_ice_v, out["v_ice"], masks["vmask"]),
    ):
        scores[label] = _compare(label, actual, expected, mask)
    frame_scores = {}
    for frame in FRAMES:
        frame_scores[str(frame)] = {
            "delta_fields": sorted(step[(8, frame)]["fields"]),
            "stress_fields": sorted(step[(9, frame)]["fields"]),
            "velocity_fields": sorted(step[(10, frame)]["fields"]),
            "status": "RETAINED_PREREGISTERED_FRAME",
        }
    rdg_in, rdg_out = step[(12, 0)]["fields"], step[(13, 0)]["fields"]
    scores["no_redistribution_a_i"] = _compare(
        "C1 a_i", rdg_out["a_i"], rdg_in["a_i"], masks["tmask"]
    )
    scores["no_redistribution_v_i"] = _compare(
        "C1 v_i", rdg_out["v_i"], rdg_in["v_i"], masks["tmask"]
    )
    return {
        "scores": scores,
        "inner_frames": frame_scores,
        "dtype": str(np.asarray(solved.u_ice_u).dtype),
        "execution": "CPU production jit",
        "fold_program": "single-rank full T-pivot fold",
    }


def admit_evidence_directory(
    evidence_dir: str | Path, *, expect_commit: str, plant: str | None = None
) -> dict[str, Any]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    ev = Path(evidence_dir)
    manifest = ev / "manifest"
    run = ev / "record_3step"
    commit_path = manifest / "producer_commit.txt"
    if not commit_path.is_file():
        raise _refuse("missing producer commit")
    producer = commit_path.read_text().strip()
    if plant == "stamp":
        producer = "0" * 40
    if (
        producer != expect_commit
        or subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{producer}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
    ):
        raise _refuse("producer commit mismatch or missing commit object")
    binaries = _sha256_manifest(manifest / "binaries.sha256", plant=plant == "binary-checksum")
    records = _sha256_manifest(run / "records.sha256", plant=plant == "checksum")
    if (
        not (run / "RUN_COMPLETE").is_file()
        or "RUN DONE" not in (run / "run.user.time.log").read_text()
    ):
        raise _refuse("incomplete retained run")
    log = verify_o5_log(run / "ocean.output", plant="selector" if plant == "selector" else None)
    scores = {}
    for kt in (1, 2, 3):
        steps = _load_step(run / "o5_si3_records", kt, plant=plant)
        scores[str(kt)] = _score_step(steps)
    return {
        "status": "PASS_O5_ADMISSION",
        "producer_commit": producer,
        "binaries_checked": binaries,
        "records_checked": records,
        "log": log,
        "steps": scores,
    }


def run_self_test(plant: str | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        path = root / "o5_kind01_step0001_iter000_rank0000.bin"
        _write_synthetic_record(path, 1)
        rec = read_o5_record(path, plant=plant if plant in {"header", "truncation"} else None)
        manifest = root / "records.sha256"
        manifest.write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")
        _sha256_manifest(manifest, plant=plant == "checksum")
        tmask = np.ones((8, 7))
        tmask[:, :2] = 0.0
        tmask[3, 3] = 0.0
        signed = verify_dry_signed_zero_fixture(np.zeros((8, 7)), tmask, nn_hls=2)
        base = rec["fields"]["at_i_entry"]
        one = base.copy()
        wi, wj = first_wet_ij(tmask, nn_hls=2)
        one[wi, wj] = np.nextafter(one[wi, wj], np.inf)
        if plant == "time-level":
            raise _refuse("time-level plant refused as required")
        if plant == "wet-1ulp" and max_ulp_diff(base, one, mask=tmask) != 0:
            raise _refuse("wet one-ULP admission plant refused as required")
        return {
            "status": "PASS_SELF_TEST",
            "header_bytes": HEADER_BYTES,
            "signed_zero": signed,
            "record_kind": rec["meta"]["kind"],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--read-record")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--verify-log")
    mode.add_argument("--same-program")
    mode.add_argument("--admit-evidence")
    parser.add_argument("--plant")
    parser.add_argument("--expect-commit")
    parser.add_argument("--reference-bld")
    parser.add_argument("--rhg-patch")
    parser.add_argument("--rdgrft-patch")
    parser.add_argument("--dyn-patch")
    parser.add_argument("--writer")
    parser.add_argument("--new-cfg", default="ORCA2_SAS_ICE_O5_REC")
    parser.add_argument("--reference-cfg", default="ORCA2_SAS_ICE_O5_REF")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    try:
        if args.read_record:
            result = read_o5_record(args.read_record, plant=args.plant)
            result = {
                "meta": result["meta"],
                "fields": {k: list(v.shape) for k, v in result["fields"].items()},
            }
        elif args.self_test:
            result = run_self_test(args.plant)
        elif args.verify_log:
            result = verify_o5_log(args.verify_log, args.plant)
        elif args.same_program:
            required = (
                args.reference_bld,
                args.rhg_patch,
                args.rdgrft_patch,
                args.dyn_patch,
                args.writer,
            )
            if any(value is None for value in required):
                raise _refuse("same-program arguments incomplete")
            result = same_program(
                Path(args.same_program),
                Path(args.reference_bld),
                {
                    "icedyn_rhg_evp": Path(args.rhg_patch),
                    "icedyn_rdgrft": Path(args.rdgrft_patch),
                    "icedyn": Path(args.dyn_patch),
                },
                Path(args.writer),
                new_cfg=args.new_cfg,
                ref_cfg=args.reference_cfg,
            )
        else:
            if not args.expect_commit:
                raise _refuse("--expect-commit is required")
            result = admit_evidence_directory(
                args.admit_evidence, expect_commit=args.expect_commit, plant=args.plant
            )
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc))
        return 2
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
