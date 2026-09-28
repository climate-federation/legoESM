#!/usr/bin/env python3
"""Walk OVERFLOW kt=3 stage-2 UP3 to its first non-bit statement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import nemo_testcase_l1_overflow_round56_up3_operands_gate as r56_gate
import numpy as np
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _up3_reconstruct
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l1_overflow_round50_pair_gate import GateError, require
from nemo_testcase_phase3_trajectory_gate import expected_masks

RECORD_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round56/acquisition/oracle_overflow_up3_operands"
)
RECORD = "oracle_r56_up3_kt00000003_s2.bin"
PARENT = "oracle_r50_momentum_kt00000003_s2.bin"
PRODUCER_COMMIT = "561483aeed99f3a678a879d31856929f41789afa"
COMPILED = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/"
    "OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90"
)
SOURCE_ORDER = (
    "u.horizontal_curvature",
    "u.velocity_pair",
    "u.selected_curvature",
    "u.nemo_t_face_flux_replay",
    "u.t_face_flux",
)


def _bits(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    require(array.dtype == np.dtype("float64"), "comparison value is not fp64")
    return array.view(np.uint64)


def _score(name: str, oracle, candidate, *, plant: bool = False) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    require(oracle.shape == candidate.shape, f"{name}: shape drift")
    require(oracle.size > 0, f"{name}: empty score domain")
    require(np.isfinite(oracle).all() and np.isfinite(candidate).all(),
            f"{name}: non-finite operand")
    baseline = _bits(candidate) != _bits(oracle)
    tested = candidate.copy()
    planted_index = None
    if plant:
        equal = np.flatnonzero(~baseline)
        require(equal.size > 0, f"{name}: no equal value for plant")
        planted_index = int(equal[0])
        flat = tested.reshape(-1)
        flat[planted_index] = np.nextafter(flat[planted_index], np.float64(np.inf))
        require(flat[planted_index] != candidate.reshape(-1)[planted_index],
                f"{name}: plant did not move")
    unequal = _bits(tested) != _bits(oracle)
    absolute = np.abs(tested - oracle)
    nonzero_reference = oracle != 0.0
    scored_nonzero = unequal & nonzero_reference
    if scored_nonzero.any():
        spacing = np.abs(np.spacing(oracle[scored_nonzero]))
        require(np.all(spacing > 0.0), f"{name}: invalid fp64 spacing")
        relative_max = float(np.max(
            absolute[scored_nonzero] / np.abs(oracle[scored_nonzero])))
        row_scale_ulp_max = float(np.max(absolute[scored_nonzero] / spacing))
    else:
        relative_max = None
        row_scale_ulp_max = None
    return {
        "name": name,
        "status": "BIT_EXACT" if not unequal.any() else "NON_BIT",
        "exact": not bool(unequal.any()),
        "n": int(oracle.size),
        "baseline_n_unequal": int(np.count_nonzero(baseline)),
        "n_unequal": int(np.count_nonzero(unequal)),
        "absolute_max": float(np.max(absolute)),
        "relative_max_nonzero_reference": relative_max,
        "row_scale_ulp_max_nonzero_reference": row_scale_ulp_max,
        "zero_reference_to_nonzero": int(np.count_nonzero(
            unequal & ~nonzero_reference & (tested != 0.0))),
        "signed_zero_only": int(np.count_nonzero(
            unequal & ~nonzero_reference & (tested == 0.0))),
        "plant": plant,
        "planted_flat_index": planted_index,
    }


@jax.jit
def _production_t_flux(q0, q1, far_pos, adv_pos, adv_neg, far_neg, selector):
    """Current legoESM T-face flux before its exact factor-of-four mapping."""
    q_average = 0.5 * (q0 + q1)
    reconstructed = _up3_reconstruct(
        far_pos, adv_pos, adv_neg, far_neg, selector)
    return q_average * reconstructed


def _embed_parent_mask(
    parent_mask: np.ndarray, full_shape: tuple[int, int, int], origin: tuple[int, int]
) -> np.ndarray:
    """Embed the Python (y,x,z) parent mask in the Fortran (x,y,z) record."""
    parent = np.asarray(parent_mask, dtype=bool).transpose(1, 0, 2)
    full = np.zeros(full_shape, dtype=bool)
    oi, oj = origin[0] - 1, origin[1] - 1
    require(oi >= 0 and oj >= 0, "record origin is not one-based positive")
    require(oi + parent.shape[0] <= full_shape[0], "parent x extent exceeds record")
    require(oj + parent.shape[1] <= full_shape[1], "parent y extent exceeds record")
    require(parent.shape[2] == full_shape[2], "parent/record vertical extent drift")
    full[oi : oi + parent.shape[0], oj : oj + parent.shape[1], :] = parent
    return full


def _analyze_u(
    fields: dict[str, np.ndarray],
    parent_active_u: np.ndarray,
    origin: tuple[int, int],
    *,
    plant: str | None,
    production_flux_fn=None,
) -> dict:
    kbb = np.asarray(fields["kbb_u"], dtype=np.float64)
    kmm = np.asarray(fields["kmm_u"], dtype=np.float64)
    transport = np.asarray(fields["transport_u"], dtype=np.float64)
    require(kbb.shape == kmm.shape == transport.shape, "UP3 U input shape drift")
    active = _embed_parent_mask(parent_active_u, kbb.shape, origin)

    # A flux stored at i+1 contributes to active RHS cells i and i+1.
    pair_domain = np.zeros_like(active)
    pair_domain[:-1] = active[:-1] | active[1:]
    pair_domain[0] = False
    pair_domain[-2:] = False
    coords = np.argwhere(pair_domain)
    require(coords.size > 0, "no active U face-flux domain")
    i, j, k = coords[:, 0], coords[:, 1], coords[:, 2]

    curvature = np.zeros_like(kbb)
    centre = kbb[1:-1]
    curvature[1:-1] = (
        (kbb[2:] - centre) + (kbb[:-2] - centre)
    ) * active[1:-1]
    curvature_domain = np.zeros_like(active)
    curvature_domain[i, j, k] = True
    curvature_domain[i + 1, j, k] = True
    rows = [
        _score(
            SOURCE_ORDER[0],
            fields["curv_uu"][curvature_domain],
            curvature[curvature_domain],
        )
    ]

    pair = kmm[i, j, k] + kmm[i + 1, j, k]
    rows.append(_score(SOURCE_ORDER[1], fields["pair_u_t"][i, j, k], pair))
    selected = np.where(pair > 0.0, curvature[i, j, k], curvature[i + 1, j, k])
    rows.append(
        _score(SOURCE_ORDER[2], fields["selected_u_t"][i, j, k], selected)
    )

    production_args = (
        jnp.asarray(transport[i, j, k]),
        jnp.asarray(transport[i + 1, j, k]),
        jnp.asarray(kmm[i - 1, j, k]),
        jnp.asarray(kmm[i, j, k]),
        jnp.asarray(kmm[i + 1, j, k]),
        jnp.asarray(kmm[i + 2, j, k]),
        jnp.asarray(pair),
    )
    if production_flux_fn is None:
        production_value = _production_t_flux(*production_args)
    else:
        production_value = production_flux_fn(
            *production_args,
            jnp.asarray(active[i, j, k], dtype=kbb.dtype),
            jnp.asarray(active[i + 1, j, k], dtype=kbb.dtype),
        )
    production = np.asarray(
        production_value,
        dtype=np.float64,
    )
    # NEMO stores Qsum*(pair-gamma*curvature); legoESM stores
    # (Qsum/2)*reconstruction, where reconstruction is half the bracket.
    production_in_nemo_units = np.float64(4.0) * production
    oracle_flux = fields["flux_u_t"][i, j, k]
    nemo_replay = (
        (transport[i, j, k] + transport[i + 1, j, k])
        * (pair - np.float64(1.0 / 3.0) * selected)
    )
    rows.append(_score(SOURCE_ORDER[3], oracle_flux, nemo_replay))
    flux_row = _score(
        SOURCE_ORDER[4],
        # The writer receives zFu_t(ji+1,jj) but stores the scalar under its
        # loop coordinate (ji,jj).  Its self-describing field is therefore
        # loop-aligned, not native zFu_t-array-aligned.
        oracle_flux,
        production_in_nemo_units,
        plant=plant == "face_flux",
    )
    rows.append(flux_row)

    first = next((row for row in rows if row["baseline_n_unequal"]), None)
    p2 = all(row["baseline_n_unequal"] == 0 for row in rows[:3])
    p3 = p2 and flux_row["baseline_n_unequal"] > 0
    if plant == "face_flux":
        require(
            flux_row["n_unequal"] == flux_row["baseline_n_unequal"] + 1,
            "face-flux plant did not add exactly one refusal",
        )
        status = "PLANTED_REFUSAL"
    elif p3:
        status = "FIRST_NON_BIT_NAMED"
    elif first is not None:
        status = "EARLIER_NON_BIT_NAMED"
    else:
        status = "CONTINUE_REQUIRED"

    if first is not None:
        unequal = (
            _bits(production_in_nemo_units)
            != _bits(oracle_flux)
        ) if first["name"] == SOURCE_ORDER[4] else None
        if unequal is not None:
            at = int(np.flatnonzero(unequal)[0])
            first["first_record_index_0based"] = [
                int(i[at]), int(j[at]), int(k[at])]
            first["first_fortran_index_1based"] = [
                int(i[at] + 1), int(j[at] + 1), int(k[at] + 1)]

    flux_unequal = _bits(production_in_nemo_units) != _bits(oracle_flux)
    boundary = flux_unequal & (~active[i, j, k] | ~active[i + 1, j, k])
    interior = flux_unequal & active[i, j, k] & active[i + 1, j, k]
    require(int(np.count_nonzero(boundary | interior))
            == flux_row["baseline_n_unequal"],
            "face-flux mismatch partition does not close")

    def _partition_score(name: str, domain: np.ndarray) -> dict:
        if np.any(domain):
            return _score(name, oracle_flux[domain], production_in_nemo_units[domain])
        return {
            "name": name,
            "status": "BIT_EXACT",
            "exact": True,
            "n": 0,
            "baseline_n_unequal": 0,
            "n_unequal": 0,
            "absolute_max": 0.0,
            "relative_max_nonzero_reference": None,
            "row_scale_ulp_max_nonzero_reference": None,
            "zero_reference_to_nonzero": 0,
            "signed_zero_only": 0,
            "plant": False,
            "planted_flat_index": None,
        }

    mismatch_partition = {
        "boundary_masked_stencil": _partition_score(
            "u.t_face_flux.boundary_masked_stencil",
            boundary,
        ),
        "wet_interior_association": _partition_score(
            "u.t_face_flux.wet_interior_association",
            interior,
        ),
    }

    return {
        "status": status,
        "plant": plant,
        "source_order": list(SOURCE_ORDER),
        "rows": rows,
        "face_flux_mismatch_partition": mismatch_partition,
        "first_non_bit_statement": first,
        "active_parent_u": int(np.count_nonzero(parent_active_u)),
        "contributing_t_faces": int(coords.shape[0]),
        "R57-P2": "CONFIRMED" if p2 else "REFUTED",
        "R57-P3": "CONFIRMED" if p3 else "REFUTED",
        "R57-P4": "CONFIRMED" if plant == "face_flux" else "NOT_RUN",
    }


def run(root: Path, expect_commit: str, plant: str | None) -> dict:
    stamp = worktree_stamp()
    require(stamp["clean"], "analysis worktree is dirty")
    require(stamp["commit"] == expect_commit,
            f"analysis commit {stamp['commit']} != {expect_commit}")
    require(jax.default_backend() == "cpu", "round-57 gate is CPU-only")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")

    source = COMPILED.read_text()
    for sentinel in (
        "zlu_uu(ji,jj) = (  ( puu(ji+1,jj  ,jk,Kbb) - puu(ji  ,jj  ,jk,Kbb) )",
        "IF( zui > 0 ) THEN   ;   zl_u = zlu_uu(ji  ,jj)",
        "zFu_t(ji+1,jj  ) = (  zFu(ji,jj) + zFu(ji+1,jj  )  ) * ( zui - gamma1 * zl_u )",
    ):
        require(sentinel in source, f"compiled UP3 statement drift: {sentinel}")

    record = root / RECORD
    parent = root / PARENT
    admission = r56_gate.admit(record, parent, PRODUCER_COMMIT, None)
    require(admission["status"] == "AT_BAR", "round-56 record is not admitted")
    parsed = r56_gate.read_up3_record(record)
    card = build_nemo_testcase_card("OVERFLOW-zps")
    active_u = np.asarray(expected_masks(card)["u"], dtype=bool)
    analysis = _analyze_u(
        parsed["fields"], active_u, tuple(parsed["header"]["origin"]), plant=plant)
    analysis.update(
        {
            "format": "nemo-testcase-l1-overflow-round57-up3-statement-v1",
            "case": "OVERFLOW-zps",
            "kt": 3,
            "stage": 2,
            "claim_label": "given NEMO's recorded operands",
            "worktree": stamp,
            "precision": "cpu-fp64-libm-production-jit",
            "record_root": str(root),
            "record_sha256": parsed["sha256"],
            "record_admission": admission["status"],
            "compiled_source": str(COMPILED),
            "R57-P1": "CONFIRMED",
            "R57-P5": "UNMEASURED_UNTIL_FINAL_DIFF",
        }
    )
    return analysis


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=RECORD_ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=("face_flux",))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.record_dir, args.expect_commit, args.plant)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(rendered, end="")
        if args.plant:
            return 2
        return 0 if report["status"] in {
            "FIRST_NON_BIT_NAMED", "EARLIER_NON_BIT_NAMED"} else 1
    except (GateError, r56_gate.GateError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "REFUSE", "reason": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
