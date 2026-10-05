#!/usr/bin/env python3
"""Close OVERFLOW's stage-3 post-dyn_zdf, pre-barotropic raw-Kaa boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.fidelity.provenance import worktree_stamp
import nemo_testcase_l1_overflow_round60_downstream_walk_gate as R60


FORMAT = "nemo-testcase-l1-overflow-round61-raw-kaa-v1"
SOURCE_ORDER = R60.SOURCE_ORDER


def _collect(card, state, ordinary_after, momentum, masks,
             round60_report: Path, entry_input: Path):
    """Extend the hash-bound round-60 sidecar with one freshly measured frame."""
    prior = json.loads(round60_report.read_text())
    R60.require(prior["format"] == R60.FORMAT, "round-60 report format drift")
    if prior.get("controlled_entry") is not None:
        R60.require(
            prior["controlled_entry"]["sha256"] == R60._sha256(entry_input),
            "round-60 controlled-entry hash drift")
    arrays = R60._read_sidecar(prior)
    rows = list(prior["rows"])
    u_mask = np.asarray(masks["u"], dtype=bool)
    current_entry = R60._active(R60._physical_u(state.u.data, u_mask), u_mask)
    current_final = R60._active(
        R60._physical_u(ordinary_after.u.data, u_mask), u_mask)
    R60.require(
        np.array_equal(current_entry, arrays["kt3.entry.u"]),
        "current controlled entry differs from round-60 sidecar")
    R60.require(
        np.array_equal(current_final, arrays["s3.postbar_kaa.u"]),
        "current postbar U differs from round-60 sidecar")
    R60.require(
        np.array_equal(current_final, arrays["kt4.entry.u"]),
        "current kt4 entry differs from round-60 sidecar")

    observed = R60._observer(
        card, state, expose_stage3_raw_momentum=True)
    raw = R60._physical_u(observed["u"], u_mask)
    reference = R60._physical_levels(
        R60._nemo_owned(momentum[3]["raw_kaa_u"]), u_mask)
    raw_row = R60._score("s3.raw_kaa.u", reference, raw, u_mask)
    arrays["s3.raw_kaa.u"] = R60._active(raw, u_mask)

    replaced = False
    for index, row in enumerate(rows):
        if row["name"] == "s3.raw_kaa.u":
            R60.require(
                row["status"] == "UNMEASURED_NO_PRODUCTION_OBSERVER",
                "round-60 raw-Kaa placeholder drift")
            rows[index] = raw_row
            replaced = True
            break
    R60.require(replaced, "round-60 raw-Kaa placeholder is missing")

    ordinary = R60.lego_fields(ordinary_after)
    for field in ("T", "S", "ssh"):
        row = R60._score(
            f"observer.s3_raw.{field}", ordinary[field], observed[field],
            masks[field])
        R60.require(row["exact"], f"raw-Kaa observer perturbed {field}")
        rows.append(row)
    final_u = R60._active(
        R60._physical_u(ordinary_after.u.data, u_mask), u_mask)
    R60.require(
        np.any(final_u.view(np.uint64) != arrays["s3.raw_kaa.u"].view(np.uint64)),
        "raw-Kaa observer is vacuous: it equals postbar U on every active cell")
    return arrays, rows


def compare(reference_path: Path, candidate_report: dict) -> dict:
    """Classify every now-measured boundary under round 60's frozen rule."""
    reference_report = json.loads(reference_path.read_text())
    R60.require(reference_report["format"] == FORMAT, "reference format drift")
    base = R60._read_sidecar(reference_report)
    candidate = R60._read_sidecar(candidate_report)
    R60.require(base.keys() == candidate.keys(), "sidecar key drift")
    rows = [
        {
            "name": name,
            **R60.classify_direction(
                base[name], candidate[name], base[f"oracle::{name}"])
        }
        for name in SOURCE_ORDER
    ]
    first_moved = next((row for row in rows if row["n_moved"]), None)
    after_adv_index = next(
        index for index, row in enumerate(rows)
        if row["name"] == "s2.after_adv.u")
    initial_direction = rows[after_adv_index]["direction"]
    direction_change = next((
        row for row in rows[after_adv_index + 1:]
        if row["direction"] != initial_direction
    ), None)
    opposite = {"TOWARD": "AWAY", "AWAY": "TOWARD"}.get(initial_direction)
    reversal = next((
        row for row in rows[after_adv_index + 1:]
        if opposite is not None and row["direction"] == opposite
    ), None)
    return {
        "status": "MEASURED",
        "reference": str(reference_path),
        "reference_commit": reference_report["worktree"]["commit"],
        "candidate_commit": candidate_report["worktree"]["commit"],
        "first_moved_boundary": first_moved,
        "after_adv_direction": initial_direction,
        "first_direction_change": direction_change,
        "first_direction_reversal": reversal,
        "compensating_owner": None if reversal is None else reversal["name"],
        "rows": rows,
    }


def plant_sidecar(reference_path: Path, output: Path,
                  expect_commit: str) -> dict:
    """Add one ULP to one oracle-exact active U value without integrating."""
    stamp = worktree_stamp()
    R60.require(stamp["clean"], "producer worktree is dirty")
    R60.require(
        stamp["commit"] == expect_commit,
        f"producer commit mismatch: {stamp['commit']} != {expect_commit}")
    reference = json.loads(reference_path.read_text())
    R60.require(reference["format"] == FORMAT, "reference format drift")
    arrays = R60._read_sidecar(reference)
    name = "kt3.entry.u"
    candidate = arrays[name].copy()
    oracle = arrays[f"oracle::{name}"]
    clean = candidate.view(np.uint64) == oracle.view(np.uint64)
    R60.require(bool(np.any(clean)), "no oracle-exact active U cell for plant")
    index = int(np.flatnonzero(clean)[0])
    before = int(np.count_nonzero(~clean))
    candidate[index] = np.nextafter(candidate[index], np.float64(np.inf))
    after = int(np.count_nonzero(
        candidate.view(np.uint64) != oracle.view(np.uint64)))
    R60.require(after == before + 1, "plant did not add exactly one refusal")
    arrays[name] = candidate
    report = {
        "format": FORMAT,
        "status": "PLANTED_REFUSAL",
        "worktree": stamp,
        "source_report": str(reference_path),
        "plant": {
            "name": name, "active_flat_index": index,
            "before_unequal": before, "after_unequal": after,
            "delta_unequal": after - before,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report["sidecar"] = R60._write_sidecar(output, arrays)
    return report


def run(output: Path, expect_commit: str, reference: Path | None,
        plant: bool, entry_input: Path, round60_report: Path) -> dict:
    stamp = worktree_stamp()
    R60.require(stamp["clean"], "producer worktree is dirty")
    R60.require(
        stamp["commit"] == expect_commit,
        f"producer commit mismatch: {stamp['commit']} != {expect_commit}")
    R60.require(jax.default_backend() == "cpu", "round-61 gate is CPU-only")
    R60.require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    R60.require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    R60.require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
        "precision policy is not fp64/libm")

    admission = R60.admit(R60.ORACLE_ROOT, R60.PRODUCER_COMMIT, None)
    R60.require(admission["status"] == "AT_BAR", "round-50 record not admitted")
    momentum, _ = R60._record_pair(R60.ORACLE_ROOT)
    card = build_nemo_testcase_card("OVERFLOW-zps")
    cfg = card.recipe.model_config
    R60.require(
        (cfg.outer_integrator, cfg.momentum_time_integrator,
         cfg.momentum_advection, cfg.momentum_flux_scheme, cfg.pgf_scheme)
        == ("forward_euler", "rk3_ws", "flux_form", "nemo_up3", "nemo_sco"),
        "resolved OVERFLOW program drift")
    R60.require(
        str(card.recipe.initial_state.u.data.dtype) == "float64",
        "state dtype is not float64")
    state = R60._read_entry_state(entry_input, card.recipe.initial_state)
    masks = R60.expected_masks(card)
    ordinary = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    ordinary_after = ordinary.step(state, dt=card.dt_s)
    print("ROUND61_KT4_ENTRY_READY", file=sys.stderr, flush=True)
    arrays, rows = _collect(
        card, state, ordinary_after, momentum, masks,
        round60_report, entry_input)

    if plant:
        name = "kt3.entry.u"
        planted = arrays[name].copy()
        oracle = arrays[f"oracle::{name}"]
        clean = planted.view(np.uint64) == oracle.view(np.uint64)
        R60.require(bool(np.any(clean)), "no exact active-U cell for plant")
        index = int(np.flatnonzero(clean)[0])
        before = int(np.count_nonzero(~clean))
        planted[index] = np.nextafter(planted[index], np.float64(np.inf))
        after = int(np.count_nonzero(
            planted.view(np.uint64) != oracle.view(np.uint64)))
        R60.require(after == before + 1, "plant did not add one refusal")
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
        "controlled_entry": {
            "path": str(entry_input), "sha256": R60._sha256(entry_input)},
        "round60_source": {
            "path": str(round60_report),
            "sha256": R60._sha256(round60_report),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report["sidecar"] = R60._write_sidecar(output, arrays)
    if reference is not None:
        report["comparison"] = compare(reference, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry-input", type=Path)
    parser.add_argument("--round60-report", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--plant-entry", action="store_true")
    parser.add_argument("--plant-sidecar", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.plant_sidecar is not None:
            report = plant_sidecar(
                args.plant_sidecar, args.output, args.expect_commit)
        else:
            R60.require(args.entry_input is not None, "--entry-input is required")
            R60.require(
                args.round60_report is not None,
                "--round60-report is required")
            report = run(
                args.output, args.expect_commit, args.reference,
                args.plant_entry, args.entry_input, args.round60_report)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.write_text(rendered)
        print(rendered, end="")
        return 2 if (args.plant_entry or args.plant_sidecar is not None) else 0
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        rendered = json.dumps(
            {"status": "REFUSE", "reason": str(error)}, indent=2) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(rendered, end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
