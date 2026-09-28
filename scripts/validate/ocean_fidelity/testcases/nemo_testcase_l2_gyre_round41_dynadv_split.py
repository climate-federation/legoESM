#!/usr/bin/env python3
"""Round-41 GYRE stage-3 ``dyn_keg`` / ``dyn_zad`` given-input gate.

The record is inadmissible until two independent source-order replays update
``before_keg -> after_keg -> after_zad`` with zero unequal cells.  Scientific
rows then execute legoESM's own C2 KE-gradient and shared NEMO-advective ZAD
paths on the dumped NEMO operands.  Exactness and the operator-relative
``1e-15`` bar are separate; only zero unequal cells can be DISCHARGED.

The model-side one-sided halo convention is explicit: the last owned column is
substituted at the x seam, while the missing y row is zero.  These are the
native GYRE card conventions consumed by the shared model paths below.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import require, sha256
from nemo_testcase_l2_gyre_round40_stage3_operators import (
    TERMS_RECORD,
    read_stage3_terms,
)

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round41/oracle_dynadv_split")
RECORD = "oracle_dynadv_split_kt00000001_s3.bin"
ROUND40_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round40_oracle_stage3_terms")
MAGIC = "NEMO_L2_ADVSP_1"
BAR = 1.0e-15
DIMS = (36, 26, 31)
HEADER_FIELDS = (
    "version", "kt", "kstg", "Kbb", "Kmm", "Krhs", "Kaa", "nn_dynkeg",
    "jpi", "jpj", "jpk", "jpkm1", "ntsi", "ntei", "ntsj", "ntej", "bits",
)
EXPECTED = (
    "before_keg_u", "before_keg_v", "after_keg_u", "after_keg_v",
    "after_zad_u", "after_zad_v", "uu_Kmm", "vv_Kmm", "ww",
    "wsd_effective", "ln_vortex_force", "e3t_Kmm", "e3u_Kmm", "e3v_Kmm",
    "e3w_Kmm", "e3t_0", "e3u_0", "e3v_0", "e3w_0", "e1e2t", "e1e2u",
    "e1e2v", "r1_e1u", "r1_e2v", "r1_e1e2u", "r1_e1e2v", "tmask",
    "umask", "vmask", "wmask",
)
SCALARS = {"ln_vortex_force"}
TWO_D = {"e1e2t", "e1e2u", "e1e2v", "r1_e1u", "r1_e2v",
         "r1_e1e2u", "r1_e1e2v"}


def _xy(raw: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return raw.reshape((nx, ny), order="F").T.copy()


def _xyz(raw: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return raw.reshape((nx, ny, nz), order="F").transpose(1, 0, 2).copy()


def read_split(path: Path, *, plant_header: bool = False,
               expect_kt: int = 1) -> dict:
    """Read the named/ranked stream fail-closed through physical EOF."""
    with path.open("rb") as handle:
        magic_raw = handle.read(16)
        require(len(magic_raw) == 16, f"{path}: short magic")
        magic = magic_raw.decode("ascii").rstrip()
        raw_header = handle.read(4 * len(HEADER_FIELDS))
        require(len(raw_header) == 4 * len(HEADER_FIELDS), f"{path}: short header")
        header_values = list(struct.unpack(f"={len(HEADER_FIELDS)}i", raw_header))
        if plant_header:
            header_values[2] = 2
        header = dict(zip(HEADER_FIELDS, header_values, strict=True))
        require(magic == MAGIC, f"{path}: bad magic {magic!r}")
        require((header["version"], header["kt"], header["kstg"],
                 header["nn_dynkeg"]) == (1, expect_kt, 3, 0),
                f"{path}: wrong stage/branch header {header}")
        require(header["Kbb"] == header["Kmm"]
                and header["Krhs"] == header["Kaa"]
                and all(header[name] in (1, 2, 3)
                        for name in ("Kbb", "Kmm", "Krhs", "Kaa")),
                f"{path}: invalid RK level tuple {header}")
        require((header["jpi"], header["jpj"], header["jpk"], header["jpkm1"],
                 header["bits"]) == (*DIMS, 30, 64),
                f"{path}: wrong extents/dtype {header}")
        require((header["ntsi"], header["ntei"], header["ntsj"], header["ntej"])
                == (3, 34, 3, 24), f"{path}: wrong owned bounds {header}")
        arrays: dict[str, np.ndarray | float] = {}
        order: list[str] = []
        while True:
            raw_name = handle.read(16)
            if not raw_name:
                break
            require(len(raw_name) == 16, f"{path}: short array name")
            name = raw_name.decode("ascii").rstrip()
            require(name not in arrays, f"{path}: duplicate {name!r}")
            raw_shape = handle.read(16)
            require(len(raw_shape) == 16, f"{path}: short shape for {name}")
            rank, n1, n2, n3 = struct.unpack("=4i", raw_shape)
            require(rank in (0, 2, 3), f"{path}: bad rank {rank} for {name}")
            count = 1 if rank == 0 else n1 * n2 * (n3 if rank == 3 else 1)
            raw = np.frombuffer(handle.read(8 * count), dtype=np.float64)
            require(raw.size == count, f"{path}: short payload for {name}")
            require(np.isfinite(raw).all(), f"{path}: non-finite {name}")
            if rank == 0:
                arrays[name] = float(raw[0])
            elif rank == 2:
                require((n1, n2, n3) == (*DIMS[:2], 1),
                        f"{path}: wrong 2-D shape for {name}")
                arrays[name] = _xy(raw, n1, n2)
            else:
                require((n1, n2, n3) == DIMS,
                        f"{path}: wrong 3-D shape for {name}")
                arrays[name] = _xyz(raw, n1, n2, n3)
            order.append(name)
    require(tuple(order) == EXPECTED,
            f"{path}: field contract differs: got {order}, expected {list(EXPECTED)}")
    require(arrays["ln_vortex_force"] == 0.0,
            f"{path}: wsd arm is live; this gate transcribes the ww-only arm")
    require(np.count_nonzero(arrays["wsd_effective"]) == 0,
            f"{path}: effective wsd is not zero")
    return {"header": header, "arrays": arrays}


def _keg_replay(a: dict) -> tuple[np.ndarray, np.ndarray]:
    """Literal ``dynkeg.F90:117-130`` accumulator replay."""
    u = a["uu_Kmm"]
    v = a["vv_Kmm"]
    out_u = np.array(a["before_keg_u"], copy=True)
    out_v = np.array(a["before_keg_v"], copy=True)
    nx, ny, nz = DIMS
    # Fortran header values are 1-based inclusive.
    i0, i1 = 2, 33  # Fortran ntsi=3, ntei=34
    j0, j1 = 2, 23  # Fortran ntsj=3, ntej=24
    for k in range(nz - 1):
        zh = np.zeros((ny, nx), dtype=np.float64)
        for j in range(j0, j1 + 2):
            for i in range(i0, i1 + 2):
                zu = np.float64(u[j, i - 1, k] * u[j, i - 1, k])
                zu = np.float64(zu + u[j, i, k] * u[j, i, k])
                zv = np.float64(v[j - 1, i, k] * v[j - 1, i, k])
                zv = np.float64(zv + v[j, i, k] * v[j, i, k])
                zh[j, i] = np.float64(np.float64(0.25) * np.float64(zv + zu))
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                grad_u = np.float64(zh[j, i + 1] - zh[j, i])
                grad_u = np.float64(grad_u * a["r1_e1u"][j, i])
                out_u[j, i, k] = np.float64(out_u[j, i, k] - grad_u)
                grad_v = np.float64(zh[j + 1, i] - zh[j, i])
                grad_v = np.float64(grad_v * a["r1_e2v"][j, i])
                out_v[j, i, k] = np.float64(out_v[j, i, k] - grad_v)
    return out_u, out_v


def _zad_replay(a: dict) -> tuple[np.ndarray, np.ndarray]:
    """Literal active ``dynzad.F90:102-137`` accumulator replay."""
    u = a["uu_Kmm"]
    v = a["vv_Kmm"]
    ww = a["ww"]
    area = a["e1e2t"]
    out_u = np.array(a["after_keg_u"], copy=True)
    out_v = np.array(a["after_keg_v"], copy=True)
    i0, i1, j0, j1 = 2, 33, 2, 23
    carry_u = np.zeros(DIMS[:2][::-1], dtype=np.float64)
    carry_v = np.zeros(DIMS[:2][::-1], dtype=np.float64)
    for k in range(DIMS[2] - 2):
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                zwf = np.float64(area[j, i] * ww[j, i, k + 1])
                zwfi = np.float64(area[j, i + 1] * ww[j, i + 1, k + 1])
                zwfj = np.float64(area[j + 1, i] * ww[j + 1, i, k + 1])
                flux_u = np.float64(zwfi + zwf)
                flux_v = np.float64(zwfj + zwf)
                dz_u = np.float64(u[j, i, k] - u[j, i, k + 1])
                dz_v = np.float64(v[j, i, k] - v[j, i, k + 1])
                next_u = np.float64(flux_u * dz_u)
                next_v = np.float64(flux_v * dz_v)
                scale_u = np.float64(np.float64(0.25) * a["r1_e1e2u"][j, i])
                scale_u = np.float64(scale_u / a["e3u_Kmm"][j, i, k])
                term_u = np.float64(scale_u * np.float64(carry_u[j, i] + next_u))
                out_u[j, i, k] = np.float64(out_u[j, i, k] - term_u)
                scale_v = np.float64(np.float64(0.25) * a["r1_e1e2v"][j, i])
                scale_v = np.float64(scale_v / a["e3v_Kmm"][j, i, k])
                term_v = np.float64(scale_v * np.float64(carry_v[j, i] + next_v))
                out_v[j, i, k] = np.float64(out_v[j, i, k] - term_v)
                carry_u[j, i] = next_u
                carry_v[j, i] = next_v
    k = DIMS[2] - 2
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            scale_u = np.float64(np.float64(0.25) * a["r1_e1e2u"][j, i])
            scale_u = np.float64(scale_u / a["e3u_Kmm"][j, i, k])
            out_u[j, i, k] = np.float64(
                out_u[j, i, k] - np.float64(scale_u * carry_u[j, i]))
            scale_v = np.float64(np.float64(0.25) * a["r1_e1e2v"][j, i])
            scale_v = np.float64(scale_v / a["e3v_Kmm"][j, i, k])
            out_v[j, i, k] = np.float64(
                out_v[j, i, k] - np.float64(scale_v * carry_v[j, i]))
    return out_u, out_v


def _owned(a: np.ndarray, nlev: int = 30) -> np.ndarray:
    return np.asarray(a)[2:-2, 2:-2, :nlev]


def _owned2(a: np.ndarray) -> np.ndarray:
    return np.asarray(a)[2:-2, 2:-2]


def _model_terms(a: dict) -> dict[str, np.ndarray | str]:
    """Execute the shared legoESM KEG and ZAD paths on NEMO operands."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        interp_cell_to_uface,
        interp_cell_to_vface,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        _bc_ke_and_pressure_gradients,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.vertical import nemo_advective_vertical_momentum_advection

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    grid, zc, state = card.recipe.grid, card.recipe.z_coord, card.recipe.initial_state

    u_native, v_native = _owned(a["uu_Kmm"]), _owned(a["vv_Kmm"])
    u = np.concatenate([u_native[:, -1:, :], u_native], axis=1)
    v = np.concatenate([np.zeros_like(v_native[:1]), v_native], axis=0)
    cell_shape = u_native.shape
    zeros = jnp.zeros(cell_shape, dtype=jnp.float64)
    eta = jnp.asarray(state.eta.data, dtype=jnp.float64)
    bathymetry = jnp.asarray(state.H_bathy.data, dtype=jnp.float64)
    mask = jnp.asarray(state.land_mask.data, dtype=jnp.float64)

    before_u = jnp.asarray(_owned(a["before_keg_u"]))
    before_v = jnp.asarray(_owned(a["before_keg_v"]))

    def keg_fn(ju, jv, accumulator_u, accumulator_v):
        rows = _bc_ke_and_pressure_gradients(
            ju, jv, zeros, zeros, grid, cfg, zc, eta, bathymetry,
            cfg.g, mask, legacy_hpg_algebraic=False)
        dku, dkv = rows[0], rows[2]
        after_u = jax.lax.optimization_barrier(accumulator_u + (-dku[:, 1:, :]))
        after_v = jax.lax.optimization_barrier(accumulator_v + (-dkv[1:, :, :]))
        return dku, dkv, after_u, after_v

    dke_u, dke_v, after_keg_u, after_keg_v = jax.jit(keg_fn)(
        jnp.asarray(u), jnp.asarray(v), before_u, before_v)
    dke_u = np.asarray(dke_u)[:, 1:, :]
    dke_v = np.asarray(dke_v)[1:, :, :]

    ww = jnp.asarray(_owned(a["ww"], nlev=31))
    area_t = jnp.asarray(_owned2(a["e1e2t"]))
    area_w = jax.lax.optimization_barrier(area_t[..., None] * ww)
    w_u = jax.lax.optimization_barrier(interp_cell_to_uface(area_w))
    w_v = jax.lax.optimization_barrier(interp_cell_to_vface(area_w, grid=grid))
    hu_n, hv_n = _owned(a["e3u_Kmm"]), _owned(a["e3v_Kmm"])
    au_n, av_n = _owned2(a["e1e2u"]), _owned2(a["e1e2v"])
    um_n, vm_n = _owned(a["umask"]), _owned(a["vmask"])
    hu = np.concatenate([hu_n[:, -1:, :], hu_n], axis=1)
    hv = np.concatenate([np.ones_like(hv_n[:1]), hv_n], axis=0)
    au = np.concatenate([au_n[:, -1:], au_n], axis=1)
    av = np.concatenate([np.ones_like(av_n[:1]), av_n], axis=0)
    um = np.concatenate([um_n[:, -1:, :], um_n], axis=1)
    vm = np.concatenate([np.zeros_like(vm_n[:1]), vm_n], axis=0)

    nemo_after_keg_u = jnp.asarray(_owned(a["after_keg_u"]))
    nemo_after_keg_v = jnp.asarray(_owned(a["after_keg_v"]))

    def zad_fn(ju, jv, accumulator_u, accumulator_v):
        zu = nemo_advective_vertical_momentum_advection(
            ju, w_u, jnp.asarray(hu), jnp.asarray(au)[..., None],
            face_active=jnp.asarray(um), bottom_face_mask_mode="nemo_faithful")
        zv = nemo_advective_vertical_momentum_advection(
            jv, w_v, jnp.asarray(hv), jnp.asarray(av)[..., None],
            face_active=jnp.asarray(vm), bottom_face_mask_mode="nemo_faithful")
        after_u = jax.lax.optimization_barrier(accumulator_u + zu[:, 1:, :])
        after_v = jax.lax.optimization_barrier(accumulator_v + zv[1:, :, :])
        return zu, zv, after_u, after_v

    zad_u, zad_v, after_zad_u, after_zad_v = jax.jit(zad_fn)(
        jnp.asarray(u), jnp.asarray(v), nemo_after_keg_u, nemo_after_keg_v)
    return {
        "keg_u": -dke_u,
        "keg_v": -dke_v,
        "zad_u": np.asarray(zad_u)[:, 1:, :],
        "zad_v": np.asarray(zad_v)[1:, :, :],
        "after_keg_u": np.asarray(after_keg_u),
        "after_keg_v": np.asarray(after_keg_v),
        "after_zad_u": np.asarray(after_zad_u),
        "after_zad_v": np.asarray(after_zad_v),
        "backend": jax.default_backend(),
    }


def _row(name: str, reference: np.ndarray, candidate: np.ndarray,
         mask: np.ndarray, *, operator_reference: np.ndarray) -> dict:
    ref, got, active = map(np.asarray, (reference, candidate, mask))
    require(ref.shape == got.shape == active.shape, f"{name}: shape mismatch")
    require(active.any(), f"{name}: empty mask")
    absolute = float(np.max(np.abs(got[active] - ref[active])))
    refmax = float(np.max(np.abs(np.asarray(operator_reference)[active])))
    relative = absolute / refmax if refmax else (0.0 if absolute == 0 else float("inf"))
    return {
        "name": name,
        "n": int(active.sum()),
        "n_unequal": int(np.count_nonzero(
            got[active].view(np.uint64) != ref[active].view(np.uint64))),
        "absolute_max": absolute,
        "reference_max_abs": refmax,
        "relative_max_abs": relative,
        "bar": BAR,
        "exact": bool(np.array_equal(got[active], ref[active])),
        "comparison": "post-statement accumulator; bar relative to operator contribution",
        "status": "AT-BAR" if relative <= BAR else "DEBT",
    }


def run(record: Path, *, round40_root: Path, expect_commit: str,
        plant: str | None = None, validate_only: bool = False) -> dict:
    stamp = worktree_stamp()
    expected = expect_commit.lower()
    actual = str(stamp["commit"]).lower()
    if plant == "stamp":
        expected = "0" * 40
    require(len(expected) == 40 and all(c in "0123456789abcdef" for c in expected),
            f"expected commit must be a full hexadecimal SHA: {expected!r}")
    require(actual == expected,
            f"commit stamp mismatch: report {actual}, expected {expected}")
    rec = read_split(record, plant_header=plant == "header")
    a = rec["arrays"]
    keg_u, keg_v = _keg_replay(a)
    if plant == "calibration":
        keg_u[3, 3, 0] = np.nextafter(keg_u[3, 3, 0], np.inf)
    zad_u, zad_v = _zad_replay(a)
    calibration = {
        "keg_u": int(np.count_nonzero(keg_u != a["after_keg_u"])),
        "keg_v": int(np.count_nonzero(keg_v != a["after_keg_v"])),
        "zad_u": int(np.count_nonzero(zad_u != a["after_zad_u"])),
        "zad_v": int(np.count_nonzero(zad_v != a["after_zad_v"])),
    }
    require(all(v == 0 for v in calibration.values()),
            f"source replay calibration failed: {calibration}")

    parent = read_stage3_terms(round40_root / TERMS_RECORD)["arrays"]
    closure = {}
    for face in ("u", "v"):
        candidate = np.array(_owned(a[f"after_zad_{face}"], nlev=31), copy=True)
        if plant == "closure" and face == "u":
            candidate[0, 0, 0] = np.nextafter(candidate[0, 0, 0], np.inf)
        closure[face] = int(np.count_nonzero(candidate != parent[f"after_adv_{face}"]))
    require(all(v == 0 for v in closure.values()),
            f"round40 dyn_adv closure failed: {closure}")

    common = {
        "format": "nemo-testcase-l2-gyre-round41-dynadv-split-v1",
        "worktree": stamp,
        "record": str(record),
        "record_sha256": sha256(record),
        "round40_record_sha256": sha256(round40_root / TERMS_RECORD),
        "bar": BAR,
        "calibration_cells_unequal": calibration,
        "round40_closure_cells_unequal": closure,
        "planted_control": plant,
    }
    if validate_only:
        return {**common, "mode": "validate-only", "status": "PASS", "rows": []}

    model = _model_terms(a)
    rows = []
    for face in ("u", "v"):
        mask = _owned(a[f"{face}mask"]) > 0.5
        before = _owned(a["before_keg_" + face])
        nemo_after_keg = _owned(a["after_keg_" + face])
        nemo_after_zad = _owned(a["after_zad_" + face])
        candidate_keg = np.asarray(model["after_keg_" + face])
        candidate_zad = np.asarray(model["after_zad_" + face])
        if plant == "keg" and face == "u":
            candidate_keg = candidate_keg.copy()
            candidate_keg[tuple(np.argwhere(mask)[0])] += 1.0
        if plant == "zad" and face == "u":
            candidate_zad = candidate_zad.copy()
            candidate_zad[tuple(np.argwhere(mask)[0])] += 1.0
        rows.append(_row(
            f"GYRE-zco.kt1.stage3.keg.{face}", nemo_after_keg,
            candidate_keg, mask, operator_reference=nemo_after_keg - before))
        rows.append(_row(
            f"GYRE-zco.kt1.stage3.zad.{face}", nemo_after_zad,
            candidate_zad, mask,
            operator_reference=nemo_after_zad - nemo_after_keg))
    first = next((op for op in ("keg", "zad")
                  if any(r["status"] == "DEBT" and f".{op}." in r["name"]
                         for r in rows)), None)
    keg_debt = any(r["status"] == "DEBT" and ".keg." in r["name"]
                   for r in rows)
    zad_debt = any(r["status"] == "DEBT" and ".zad." in r["name"]
                   for r in rows)
    prediction_verdict = "CONFIRMED" if keg_debt and not zad_debt else "REFUTED"
    status = "AT-BAR" if first is None else "DEBT"
    if plant in {"keg", "zad"}:
        planted = next(r for r in rows
                       if r["name"] == f"GYRE-zco.kt1.stage3.{plant}.u")
        require(status == "DEBT" and planted["absolute_max"] > 0.5,
                f"{plant} plant did not move its own row by the planted unit")
    return {**common,
        "mode": "score",
        "execution_regime": "production_jit",
        "precision_policy": "fp64/libm",
        "jax_backend": model["backend"],
        "rows": rows,
        "first_operator_over_bar": first,
        "preregistered_prediction_verdict": prediction_verdict,
        "status": status,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / RECORD)
    parser.add_argument("--round40-root", type=Path, default=ROUND40_ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--plant", choices=("header", "calibration", "closure",
                                             "keg", "zad", "stamp"))
    args = parser.parse_args(argv)
    report = run(args.record, round40_root=args.round40_root,
                 expect_commit=args.expect_commit, plant=args.plant,
                 validate_only=args.validate_only)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    for row in report["rows"]:
        print(f"{row['status']:<7} {row['name']:<38} "
              f"unequal {row['n_unequal']}/{row['n']} "
              f"max {row['absolute_max']:.17g} rel {row['relative_max_abs']:.6g}")
    print("FIRST-OPERATOR-OVER-BAR", report.get("first_operator_over_bar"))
    print("PREREGISTERED-VERDICT",
          report.get("preregistered_prediction_verdict", "UNMEASURED"))
    print("STATUS", report["status"])
    if args.plant:
        return 1
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
