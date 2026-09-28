#!/usr/bin/env python3
"""Walk the held source-order UP3 arm from kt=3 stage 2 to kt=4 entry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l1_overflow_round50_pair_gate import admit, require
from nemo_testcase_l1_overflow_round51_pair_gate import (
    ORACLE_ROOT,
    PRODUCER_COMMIT,
    _nemo_owned,
    _physical_levels,
    _record_pair,
    _score,
    _u,
)
from nemo_testcase_phase3_trajectory_gate import (
    expected_masks,
    lego_fields,
    read_entry,
)


FORMAT = "nemo-testcase-l1-overflow-round60-downstream-walk-v1"
SOURCE_ORDER = (
    "kt3.entry.u",
    "s2.after_adv.u",
    "s2.pre_zdf.u",
    "s2.raw_kaa.u",
    "s2.postbar_kaa.u",
    "s3.after_adv.u",
    "s3.after_ldf.u",
    "s3.pre_zdf.u",
    "s3.raw_kaa.u",
    "s3.postbar_kaa.u",
    "kt4.entry.u",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _active(value, mask) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(value.shape == mask.shape, f"active-value shape drift {value.shape}/{mask.shape}")
    return np.ascontiguousarray(value[mask])


def _observer(card, state, **hook_values) -> dict[str, np.ndarray]:
    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hook_values),
    )
    observed = model.step(state, dt=card.dt_s)
    fields = {
        "u": np.asarray(observed.u.data),
        "T": np.asarray(observed.T.data),
        "S": np.asarray(observed.S.data),
        "ssh": np.asarray(observed.eta.data),
    }
    # Five private hooks otherwise retain five XLA executables and can hit the
    # per-process compiler-map limit.  Values are materialized before clearing.
    del observed, model
    jax.clear_caches()
    return fields


def _physical_u(value, mask):
    return _physical_levels(_u(value), mask)


def _collect(card, state, ordinary_after, momentum, masks) -> tuple[dict, list[dict]]:
    """Return production-bound source frames and their NEMO-relative rows."""
    s2_rhs = _observer(card, state, expose_stage2_momentum_rhs=True)
    s2_raw = _observer(card, state, expose_stage2_raw_momentum=True)
    s2_postbar = _observer(card, state, expose_momentum_stage=2)
    s3_pre_ldf = _observer(card, state, expose_stage3_momentum_rhs="pre_ldf")
    s3_post_ldf = _observer(card, state, expose_stage3_momentum_rhs="post_ldf")

    u_mask = np.asarray(masks["u"], dtype=bool)
    entry = _physical_u(state.u.data, u_mask)
    final = _physical_u(ordinary_after.u.data, u_mask)
    values = {
        "kt3.entry.u": entry,
        "s2.after_adv.u": _physical_u(s2_rhs["u"], u_mask),
        # Compiled stprk3_stg:374-429 executes no stage-2 statement between
        # after_adv and pre_zdf on this deck.  One production observation is
        # intentionally bound to both source labels; the NEMO record must
        # independently prove its two stored payloads are also bit-identical.
        "s2.pre_zdf.u": _physical_u(s2_rhs["u"], u_mask),
        "s2.raw_kaa.u": _physical_u(s2_raw["u"], u_mask),
        "s2.postbar_kaa.u": _physical_u(s2_postbar["u"], u_mask),
        "s3.after_adv.u": _physical_u(s3_pre_ldf["u"], u_mask),
        "s3.after_ldf.u": _physical_u(s3_post_ldf["u"], u_mask),
        "s3.pre_zdf.u": _physical_u(s3_post_ldf["u"], u_mask),
        # There is no existing narrow production observer after dyn_zdf and
        # before the barotropic replacement.  Keep it explicit, never infer it.
        "s3.raw_kaa.u": None,
        "s3.postbar_kaa.u": final,
        "kt4.entry.u": final,
    }
    oracle_entry3 = read_entry(ORACLE_ROOT / "oracle_step_entry_kt00000003.bin", "OVERFLOW-zps")
    oracle_entry4 = read_entry(ORACLE_ROOT / "oracle_step_entry_kt00000004.bin", "OVERFLOW-zps")
    references = {
        "kt3.entry.u": oracle_entry3["u"],
        "s2.after_adv.u": _nemo_owned(momentum[2]["after_adv_u"]),
        "s2.pre_zdf.u": _nemo_owned(momentum[2]["pre_zdf_u"]),
        "s2.raw_kaa.u": _nemo_owned(momentum[2]["raw_kaa_u"]),
        "s2.postbar_kaa.u": _nemo_owned(momentum[2]["postbar_kaa_u"]),
        "s3.after_adv.u": _nemo_owned(momentum[3]["after_adv_u"]),
        "s3.after_ldf.u": _nemo_owned(momentum[3]["after_ldf_u"]),
        "s3.pre_zdf.u": _nemo_owned(momentum[3]["pre_zdf_u"]),
        "s3.raw_kaa.u": _nemo_owned(momentum[3]["raw_kaa_u"]),
        "s3.postbar_kaa.u": _nemo_owned(momentum[3]["postbar_kaa_u"]),
        "kt4.entry.u": oracle_entry4["u"],
    }
    rows = []
    arrays = {}
    for name in SOURCE_ORDER:
        reference = _physical_levels(references[name], u_mask)
        value = values[name]
        arrays[f"oracle::{name}"] = _active(reference, u_mask)
        if value is None:
            rows.append({
                "name": name,
                "status": "UNMEASURED_NO_PRODUCTION_OBSERVER",
                "exact": None,
                "n": int(u_mask.sum()),
                "n_unequal": None,
                "reason": "no narrow write-only observer between dyn_zdf and barotropic replacement",
            })
            continue
        row = _score(name, reference, value, u_mask)
        rows.append(row)
        arrays[name] = _active(value, u_mask)

    require(np.array_equal(arrays["s2.after_adv.u"], arrays["s2.pre_zdf.u"]),
            "legoESM stage-2 after_adv/pre_zdf alias drift")
    require(np.array_equal(
        arrays["oracle::s2.after_adv.u"], arrays["oracle::s2.pre_zdf.u"]),
        "NEMO stage-2 after_adv/pre_zdf payloads differ")
    require(np.array_equal(arrays["s3.after_ldf.u"], arrays["s3.pre_zdf.u"]),
            "legoESM stage-3 after_ldf/pre_zdf alias drift")
    require(np.array_equal(
        arrays["oracle::s3.after_ldf.u"], arrays["oracle::s3.pre_zdf.u"]),
        "NEMO stage-3 after_ldf/pre_zdf payloads differ")

    ordinary_fields = lego_fields(ordinary_after)
    observer_noninterference = []
    for label, observed in (
        ("s2_rhs", s2_rhs), ("s2_raw", s2_raw), ("s2_postbar", s2_postbar),
        ("s3_pre_ldf", s3_pre_ldf), ("s3_post_ldf", s3_post_ldf),
    ):
        for field in ("T", "S", "ssh"):
            mask = masks[field]
            row = _score(
                f"observer.{label}.{field}", ordinary_fields[field],
                observed[field], mask,
            )
            require(row["exact"], f"observer {label} perturbed {field}")
            observer_noninterference.append(row)
    return arrays, rows + observer_noninterference


def _write_sidecar(output: Path, arrays: dict[str, np.ndarray]) -> dict:
    sidecar = output.with_suffix(".arrays.npz")
    np.savez_compressed(sidecar, **arrays)
    return {
        "path": str(sidecar),
        "sha256": _sha256(sidecar),
        "fields": list(arrays),
    }


def _read_sidecar(report: dict) -> dict[str, np.ndarray]:
    meta = report["sidecar"]
    path = Path(meta["path"])
    require(path.is_file(), f"missing sidecar {path}")
    require(_sha256(path) == meta["sha256"], f"sidecar hash drift: {path}")
    with np.load(path) as stored:
        require(stored.files == meta["fields"], "sidecar field order drift")
        return {name: np.asarray(stored[name]) for name in stored.files}


def classify_direction(base, candidate, oracle) -> dict:
    """Exact per-cell movement plus deterministic aggregate direction."""
    base = np.asarray(base, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    require(base.shape == candidate.shape == oracle.shape, "direction shape drift")
    base_error = np.abs(base - oracle)
    candidate_error = np.abs(candidate - oracle)
    moved = base.view(np.uint64) != candidate.view(np.uint64)
    toward = moved & (candidate_error < base_error)
    away = moved & (candidate_error > base_error)
    neutral = moved & ~(toward | away)
    base_linf = float(np.max(base_error))
    candidate_linf = float(np.max(candidate_error))
    base_l2 = float(np.linalg.norm(base_error))
    candidate_l2 = float(np.linalg.norm(candidate_error))
    if candidate_linf < base_linf and candidate_l2 < base_l2:
        direction = "TOWARD"
    elif candidate_linf > base_linf and candidate_l2 > base_l2:
        direction = "AWAY"
    elif not np.any(moved):
        direction = "UNCHANGED"
    else:
        direction = "MIXED"
    return {
        "direction": direction,
        "n": int(base.size),
        "n_moved": int(np.count_nonzero(moved)),
        "n_toward": int(np.count_nonzero(toward)),
        "n_away": int(np.count_nonzero(away)),
        "n_neutral": int(np.count_nonzero(neutral)),
        "base_linf": base_linf,
        "candidate_linf": candidate_linf,
        "base_l2": base_l2,
        "candidate_l2": candidate_l2,
        "max_abs_move": float(np.max(np.abs(candidate - base))),
    }


def compare(reference_path: Path, candidate_report: dict) -> dict:
    reference_report = json.loads(reference_path.read_text())
    require(reference_report["format"] == FORMAT, "reference format drift")
    base = _read_sidecar(reference_report)
    candidate = _read_sidecar(candidate_report)
    require(base.keys() == candidate.keys(), "sidecar key drift")
    rows = []
    for name in SOURCE_ORDER:
        if name not in base:
            rows.append({"name": name, "direction": "UNMEASURED"})
            continue
        rows.append({
            "name": name,
            **classify_direction(base[name], candidate[name], base[f"oracle::{name}"]),
        })
    first_moved = next((row for row in rows if row.get("n_moved", 0)), None)
    # A downstream compensator is named only after the UP3 boundary itself.
    # It must reverse a strictly TOWARD aggregate to strictly AWAY, or vice
    # versa. MIXED never becomes an attribution by judgment.
    measured = [row for row in rows if row["direction"] != "UNMEASURED"]
    after_adv_index = next(
        index for index, row in enumerate(measured)
        if row["name"] == "s2.after_adv.u")
    initial_direction = measured[after_adv_index]["direction"]
    opposite = {"TOWARD": "AWAY", "AWAY": "TOWARD"}.get(initial_direction)
    reversal = next((
        row for row in measured[after_adv_index + 1:]
        if opposite is not None and row["direction"] == opposite
    ), None)
    return {
        "status": "MEASURED",
        "reference": str(reference_path),
        "reference_commit": reference_report["worktree"]["commit"],
        "candidate_commit": candidate_report["worktree"]["commit"],
        "first_moved_boundary": first_moved,
        "after_adv_direction": initial_direction,
        "first_direction_reversal": reversal,
        "compensating_owner": None if reversal is None else reversal["name"],
        "rows": rows,
    }


def run(output: Path, expect_commit: str, reference: Path | None, plant: bool) -> dict:
    stamp = worktree_stamp()
    require(stamp["clean"], "producer worktree is dirty")
    require(stamp["commit"] == expect_commit,
            f"producer commit mismatch: {stamp['commit']} != {expect_commit}")
    require(jax.default_backend() == "cpu", "round-60 gate is CPU-only")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")

    admission = admit(ORACLE_ROOT, PRODUCER_COMMIT, None)
    require(admission["status"] == "AT_BAR", "round-50 record not admitted")
    momentum, _ = _record_pair(ORACLE_ROOT)
    card = build_nemo_testcase_card("OVERFLOW-zps")
    cfg = card.recipe.model_config
    require((cfg.outer_integrator, cfg.momentum_time_integrator,
             cfg.momentum_advection, cfg.momentum_flux_scheme, cfg.pgf_scheme)
            == ("forward_euler", "rk3_ws", "flux_form", "nemo_up3", "nemo_sco"),
            "resolved OVERFLOW program drift")
    require(str(card.recipe.initial_state.u.data.dtype) == "float64",
            "state dtype is not float64")
    masks = expected_masks(card)
    ordinary = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    state = card.recipe.initial_state
    for _ in range(2):
        state = ordinary.step(state, dt=card.dt_s)
    ordinary_after = ordinary.step(state, dt=card.dt_s)
    arrays, rows = _collect(card, state, ordinary_after, momentum, masks)

    if plant:
        name = "kt3.entry.u"
        planted = arrays[name].copy()
        planted[0] = np.nextafter(planted[0], np.float64(np.inf))
        require(planted[0] != arrays[name][0], "entry plant did not move")
        clean = arrays[name].view(np.uint64) != arrays[f"oracle::{name}"].view(np.uint64)
        red = planted.view(np.uint64) != arrays[f"oracle::{name}"].view(np.uint64)
        require(int(np.count_nonzero(red)) == int(np.count_nonzero(clean)) + 1,
                "entry plant did not add exactly one refusal")
        arrays[name] = planted

    report = {
        "format": FORMAT,
        "status": "PLANTED_REFUSAL" if plant else "WALK_MEASURED",
        "case": "OVERFLOW-zps",
        "kt": 3,
        "claim_label": "independent",
        "precision": "cpu-fp64-libm-production-jit",
        "worktree": stamp,
        "record_admission": {
            "status": admission["status"],
            "producer_commit": admission["producer_commit"],
            "records": len(admission["records"]),
        },
        "compiled_source_order": list(SOURCE_ORDER),
        "rows": rows,
        "plant": plant,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report["sidecar"] = _write_sidecar(output, arrays)
    if reference is not None:
        report["comparison"] = compare(reference, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--plant-entry", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.output, args.expect_commit, args.reference, args.plant_entry)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.write_text(rendered)
        print(rendered, end="")
        return 2 if args.plant_entry else 0
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        rendered = json.dumps({"status": "REFUSE", "reason": str(error)}, indent=2) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(rendered, end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
