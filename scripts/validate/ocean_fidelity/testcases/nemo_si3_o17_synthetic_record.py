#!/usr/bin/env python3
"""Writer-shaped synthetic O17 record from the model, scored by the real scorer.

Preflight evidence that the scorer can pass a record laid out exactly as the
O17 writer lays it out (owned ``A2D(0)`` extents where NEMO allocates them,
source-derived by ``nemo_si3_o17_record_budget.record_layout``) and that it
fails a one-ULP and a signed-zero change.  Values of every field the scorer
reads or compares come from the model; writer fields the scorer never reads
are written as +0 so the file still has the writer's exact size and order.
No NEMO is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import struct
import time
from pathlib import Path

import jax
import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4.nemo_acquisition_extents import DeckDimensions
from scripts.validate.ocean_fidelity.testcases import nemo_si3_o17_composition_scorer as scorer
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_ad_probe import DEFAULT_MASTER, _case
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_record_budget import record_layout

MASKS = ("fast_tmask", "fast_umask", "fast_vmask", "icb_tmask", "icb_umask", "icb_vmask")


def _fit(value, shape: tuple[int, ...], full: tuple[int, int], h: int) -> np.ndarray:
    """Model array -> the writer's extent (owned slice or zero-padded full)."""
    value = np.asarray(value, dtype=np.float64)
    if value.shape == shape:
        return value
    owned = (full[0] - 2 * h, full[1] - 2 * h)
    if value.shape[:2] == full and shape[:2] == owned and value.shape[2:] == shape[2:]:
        return value[h:-h, h:-h, ...]
    if value.shape[:2] == owned and shape[:2] == full and value.shape[2:] == shape[2:]:
        out = np.zeros(shape, dtype=np.float64)
        out[h:-h, h:-h, ...] = value
        return out
    raise ValueError(f"model value {value.shape} cannot take writer extent {shape}")


def write_record(path: Path, kind: int, kt: int, rank: int, grid: tuple[int, ...],
                 fields: list[tuple[str, np.ndarray]]) -> None:
    """Byte layout of o17_master_writer.F90:30-114 (stream, native endian, Fortran order)."""
    with path.open("wb") as stream:
        stream.write(b"O17S0001" + struct.pack("=i", 1) + struct.pack("=8i", kind, kt, rank, *grid))
        for name, value in fields:
            dims = list(value.shape) + [0] * (4 - value.ndim)
            stream.write(name.ljust(32).encode("ascii") + struct.pack("=i4i", value.ndim, *dims))
            stream.write(np.asarray(value, dtype="=f8").tobytes(order="F"))
        stream.write(b"__END__".ljust(32) + struct.pack("=i4i", -1, 0, 0, 0, 0))


def _records(values: dict[int, dict[str, object]], kt: int, rank: int, layout, full, h, card,
             conditional_present) -> dict[int, dict]:
    out = {}
    for kind, fields in layout.items():
        row = {}
        for name, shape, conditional in fields:
            if conditional and not conditional_present(kind, name):
                continue
            source = values.get(kind, {})
            if name == "cflice_delayed":
                row[name] = np.asarray(source[name], dtype=np.float64).reshape(())
            elif name in source:
                row[name] = _fit(source[name], shape, full, h)
            else:
                row[name] = np.zeros(shape, dtype=np.float64)  # never read by the scorer
        out[kind] = {"meta": {"kind": kind, "kt": kt, "rank": rank, "jpi": full[0], "jpj": full[1],
                              "jpl": card.state.jpl, "nlay_i": card.state.nlay_i,
                              "nlay_s": card.state.nlay_s}, "fields": row}
    return out


def generate(root: Path, steps: int, entry, forcing, card, runner, master: Path = DEFAULT_MASTER):
    """Write ``steps`` model steps from ``entry`` as writer records under ``root/record_100step``."""
    full, h = tuple(entry.ice.u_ice.shape), int(card.state.nn_hls)
    layout = record_layout(deck=DeckDimensions(full[0], full[1], 31, h))
    run = root / "record_100step"
    records_dir = run / "o17_master_records"
    records_dir.mkdir(parents=True)
    for name in ("output.namelist.ice", "output.namelist.dyn"):  # real admitted NEMO output
        shutil.copyfile(master / "record_100step" / name, run / name)
    current = entry
    for kt in range(1, steps + 1):
        values = scorer.record_inputs(current, forcing._replace(kt=kt))
        has_cfl = current.ice.cflice_delayed is not None

        def present(kind, name, kt=kt, has_cfl=has_cfl):
            if name in MASKS:  # o17_dyn_masks runs inside ice_dyn, after kind 22
                return not (kt == 1 and kind == 22)
            return has_cfl if kind == 22 else name in values.get(kind, {})

        draft = _records(values, kt, 0, layout, full, h, card, present)
        step_entry, step_forcing = scorer.entry_state(draft, card), scorer.live_forcing(draft, card)
        out, trace = runner(step_entry, step_forcing)
        for kind, name, value in scorer.scored_fields(out, trace):
            values.setdefault(kind, {})[name] = value
        if out.ice.cflice_delayed is not None:
            for kind in layout:
                if kind != 22:
                    values.setdefault(kind, {})["cflice_delayed"] = out.ice.cflice_delayed
        final = _records(values, kt, 0, layout, full, h, card, present)
        for kind, record in final.items():
            name = f"o17_kind{kind:02d}_{scorer.KINDS[kind]}_step{kt:04d}_rank0000.bin"
            write_record(records_dir / name, kind, kt, 0,
                         (full[0], full[1], card.state.jpl, card.state.nlay_i, card.state.nlay_s),
                         [(n, v) for n, v in record["fields"].items()])
        current = out


def memoized(runner):
    """Re-use a model result for bit-identical inputs (plants change only records)."""
    cache: dict[str, object] = {}

    def run(entry, forcing):
        # si3_o17_step overwrites aevp.(u_ice_u, v_ice_v) with ice.(u_ice, v_ice)
        # (si3_o17.py:619) and stress.(u_ice, v_ice) with final.state.(u_ice, v_ice)
        # (si3_o17.py:1124-1128) before reading either slot.
        norm_entry = entry._replace(
            aevp=entry.aevp._replace(u_ice_u=entry.ice.u_ice, v_ice_v=entry.ice.v_ice)
        )
        zero = np.zeros_like(np.asarray(forcing.sss_m))
        norm_forcing = forcing._replace(
            stress=forcing.stress._replace(u_ice=zero, v_ice=zero)
        )
        digest = hashlib.sha256(str(forcing.kt).encode())
        for leaf in jax.tree_util.tree_leaves((norm_entry, tuple(norm_forcing[1:]))):
            array = np.ascontiguousarray(np.asarray(leaf))
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        key = digest.hexdigest()
        if key not in cache:
            cache[key] = runner(entry, forcing)
        return cache[key]
    return run


def _plant(path: Path, name: str, mutate) -> dict:
    record = scorer.read_record(path)
    value = record["fields"][name]
    index = mutate(value)
    write_record(path, record["meta"]["kind"], record["meta"]["kt"], record["meta"]["rank"],
                 tuple(record["meta"][k] for k in ("jpi", "jpj", "jpl", "nlay_i", "nlay_s")),
                 list(record["fields"].items()))
    return {"file": path.name, "field": name, "index": [int(i) for i in index]}


def _verify_parallel_ranges(identical_dir: Path, serial_report: dict, steps: int, kwargs: dict) -> dict:
    """Prove per-step split-then-merge equals serial per-step and split free-run/missing/dup refuse."""
    per_step_labels = list(scorer.PER_STEP_LABELS)
    range_reports = [
        scorer.score(identical_dir, per_step_labels, step_range=(kt, kt), **kwargs)
        for kt in range(1, steps + 1)
    ]
    merged = scorer.merge_reports(range_reports, expect_steps=steps)
    serial_per_step = scorer.score(identical_dir, per_step_labels, **kwargs)
    equal_ignoring_timing = (
        scorer.strip_timing(merged) == scorer.strip_timing(serial_per_step)
        and scorer.strip_timing(merged)["labels"]["GIVEN-NEMO-ENTRY"]
        == scorer.strip_timing(serial_report)["labels"]["GIVEN-NEMO-ENTRY"]
    )

    split_free_run_refused = True
    for call in (
        lambda: scorer.free_run(identical_dir, step_range=(2, steps), **kwargs),
        lambda: scorer.score(identical_dir, list(scorer.VALID_LABELS), step_range=(2, steps), **kwargs),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            split_free_run_refused = False

    missing_refused = False
    try:
        scorer.merge_reports(range_reports[:-1], expect_steps=steps)
    except ValueError:
        missing_refused = True

    duplicate_refused = False
    try:
        scorer.merge_reports([*range_reports, range_reports[-1]], expect_steps=steps)
    except ValueError:
        duplicate_refused = True

    return {
        "split_merge_equals_serial": equal_ignoring_timing,
        "split_free_run_refused": split_free_run_refused,
        "missing_step_refused": missing_refused,
        "duplicated_step_refused": duplicate_refused,
        "merged_status": merged["status"],
    }


def _verify_free_run(out: Path, identical_dir: Path, steps: int, h: int, kwargs: dict) -> dict:
    """Prove free run passes on the synthetic record and detects the carried-field plant."""
    free_identical = scorer.free_run(identical_dir, **kwargs)

    def perturb_prev_v_i(kt: int, rank: int, out_state):
        del rank
        if kt != 1:
            return out_state
        v_i = np.asarray(out_state.ice.v_i).copy()
        v_i[h, h, 0] = np.nextafter(v_i[h, h, 0], np.inf)
        return out_state._replace(ice=out_state.ice._replace(v_i=jax.numpy.asarray(v_i)))

    original_carry = scorer.carry_step_state

    def planted_carry(prev_out, records, card):
        rec_entry = scorer.entry_state(records, card)
        return prev_out._replace(ice=prev_out.ice._replace(v_i=rec_entry.ice.v_i))

    scorer.carry_step_state = planted_carry
    try:
        planted = scorer.free_run(identical_dir, post_step_hook=perturb_prev_v_i, **kwargs)
    finally:
        scorer.carry_step_state = original_carry

    (out / "free_run.json").write_text(json.dumps(free_identical, indent=1, sort_keys=True) + "\n")
    (out / "free_run_carried_plant.json").write_text(
        json.dumps(planted, indent=1, sort_keys=True) + "\n"
    )

    step2 = planted["per_step"]["2:0"]
    plant_detected = (
        planted["status"] == "FAIL"
        and step2["carry_chain_intact"] is False
        and "carried_entry.ice.v_i" in step2["failed_fields"]
        and planted["first_difference"] is not None
        and planted["first_difference"]["step"] == 2
        and planted["first_difference"]["field"] == "carried_entry.ice.v_i"
        and planted["first_difference"]["different"] == 1
    )
    return {
        "identical_status": free_identical["status"],
        "identical_different": free_identical["different"],
        "identical_first_difference": free_identical["first_difference"],
        "carried_plant_status": planted["status"],
        "carried_plant_detected": plant_detected,
        "carried_plant_first_difference": planted["first_difference"],
    }


def preflight(out: Path, steps: int, shape: tuple[int, int], execution: str,
              plants: bool = True) -> dict:
    """Identical record PASS; one-ULP t_si and signed-zero plants FAIL.

    ``plants=False`` scores only the identical record with the unmemoized
    production runner, which is the scorer's own memory footprint per step.
    """
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    entry, forcing, card = _case(DEFAULT_MASTER, shape)
    runner = scorer.make_runner(card, execution)
    shared = memoized(runner) if plants else runner
    generate(out / "identical", steps, entry, forcing, card, shared)
    kwargs = dict(execution=execution, expect_steps=steps, card=card, runner=shared)
    reports = {"identical": scorer.score(out / "identical", list(scorer.VALID_LABELS), **kwargs)}
    h = int(card.state.nn_hls)
    last = f"step{steps:04d}_rank0000.bin"
    for plant in ("one_ulp_t_si", "signed_zero") if plants else ():
        shutil.copytree(out / "identical", out / plant)
        exit_path = out / plant / "record_100step/o17_master_records" / f"o17_kind28_step_exit_{last}"
        if plant == "one_ulp_t_si":
            def mutate(value):  # t_si is A2D(0) (ice.F90:451): every element is owned
                value.flat[value.size // 2] = np.nextafter(value.flat[value.size // 2], np.inf)
                return np.unravel_index(value.size // 2, value.shape)
            reports[plant + "_plant"] = _plant(exit_path, "t_si", mutate)
        else:
            def mutate(value):
                zeros = np.argwhere((value == 0.0) & ~np.signbit(value))
                if not len(zeros):
                    raise ValueError("no +0 in the planted field")
                value[tuple(zeros[0])] = -0.0
                return zeros[0]
            reports[plant + "_plant"] = _plant(exit_path, "wfx_bog", mutate)
        reports[plant] = scorer.score(out / plant, list(scorer.VALID_LABELS), **kwargs)
    def fired(name: str, field: str) -> bool:
        given = reports[name]["labels"]["GIVEN-NEMO-ENTRY"]
        return reports[name]["status"] == "FAIL" and field in given["steps"][f"{steps}:0"]["failed_fields"]

    parallel_summary = (
        _verify_parallel_ranges(out / "identical", reports["identical"], steps, kwargs)
        if plants else None
    )
    free_summary = (
        _verify_free_run(out, out / "identical", steps, h, kwargs) if plants else None
    )
    ok = reports["identical"]["status"] == "PASS" and (not plants or (
        fired("one_ulp_t_si", "step_exit.t_si")
        and fired("signed_zero", "step_exit.wfx_bog")
        and parallel_summary["split_merge_equals_serial"]
        and parallel_summary["split_free_run_refused"]
        and parallel_summary["missing_step_refused"]
        and parallel_summary["duplicated_step_refused"]
        and free_summary["identical_status"] == "PASS"
        and free_summary["identical_different"] == 0
        and free_summary["carried_plant_detected"]
    ))
    summary = {
        "status": "PASS_O17_SYNTHETIC_RECORD_SCORER" if ok else "FAIL_O17_SYNTHETIC_RECORD_SCORER",
        "steps": steps, "grid": list(shape), "owned": [shape[0] - 2 * h, shape[1] - 2 * h],
        "execution": execution, "wall_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if parallel_summary is not None:
        summary["parallel_ranges"] = parallel_summary
    if free_summary is not None:
        summary["free_run"] = free_summary
    for name in ("identical", "one_ulp_t_si", "signed_zero") if plants else ("identical",):
        rep = reports[name]
        summary[name] = {"status": rep["status"],
                         "labels": {k: {"status": v["status"], "different": v["different"],
                                        "nonfinite": v["nonfinite"], "max_relative": v["max_relative"],
                                        "failed_fields": sorted({f for row in v["steps"].values()
                                                                 for f in row["failed_fields"]})}
                                    for k, v in rep["labels"].items()},
                         "scorer_rss_kib_after_each_step": rep["scorer_rss_kib_after_each_step"]}
        (out / f"{name}.json").write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    for name in ("one_ulp_t_si_plant", "signed_zero_plant") if plants else ():
        summary[name] = reports[name]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--grid", type=int, nargs=2, default=(184, 152))
    parser.add_argument("--execution", choices=("jit", "eager"), default="eager")
    parser.add_argument("--identical-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 2:
        raise SystemExit("REFUSE: at least two steps")
    summary = preflight(args.out.resolve(), args.steps, tuple(args.grid), args.execution,
                        plants=not args.identical_only)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
