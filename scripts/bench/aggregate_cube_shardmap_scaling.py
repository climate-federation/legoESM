#!/usr/bin/env python
"""Aggregate per-N Blocker-1 cube shard_map timing POINTS into a cross-node
strong-scaling verdict.

Each point comes from ``bench_cube_shardmap_halo.py --point`` (one launch = one
global device count). A single multi-controller launch fixes its device count and
holds only its own shard, so it cannot compute a speed-up curve or reference a
single-device baseline; this aggregator does that ACROSS the per-N launches and
applies the decisive gates:

* **baseline present** — an ``n_devices == 1`` point must exist (the speed-up
  denominator).
* **ppermute proof** — every ``n_devices > 1`` point lowered a
  ``collective_permute`` (``spmd_hlo_collective is True``), so the measured
  executable is the bandwidth-optimal ppermute path, not a replicated all_gather
  fallback. ``None`` (probe unsupported) or ``False`` both fail.
* **config consistency** — all points share ``n_grid`` / ``n_lev`` / ``precision``
  (else the curve compares different problems); a mismatch fails loudly.
* **efficiency** — (decisive, ``--gate-efficiency``) efficiency at the largest
  count ≥ ``--pass-efficiency`` AND the largest count is > 1, i.e. the GSPMD cube
  halo cleared the ``mpi4jax`` np6 anti-scale cliff.

Emits ``scaling.csv`` + ``report.md`` + ``verdict.json``; exits nonzero if an
enforced gate fails. Pure Python — no JAX, no model import.

Usage::

    python scripts/bench/aggregate_cube_shardmap_scaling.py \
        --root results/blocker1_xnode --gate-efficiency --pass-efficiency 0.6 \
        --output-dir results/blocker1_xnode/agg
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path

logger = logging.getLogger("aggregate_cube_shardmap_scaling")

_REQUIRED_KEYS = ("mode", "n_devices", "ms_per_step", "spmd_hlo_collective", "config")
# backend included: mixing a cpu baseline with cuda points would give a garbage
# (false-superlinear) curve, so it must break config-consistency like n_grid does.
_CONFIG_KEYS = ("n_grid", "n_lev", "precision", "backend")


def _collect_paths(root: Path | None, inputs: list[Path] | None) -> list[Path]:
    """Resolve the set of point ``results.json`` files to aggregate."""
    if inputs:
        paths = list(inputs)
    elif root is not None:
        paths = sorted(root.glob("*/results.json"))
    else:
        raise ValueError("provide --root DIR or --inputs a.json b.json ...")
    if not paths:
        raise ValueError(f"no point results.json found (root={root}, inputs={inputs})")
    return paths


def _load_points(paths: list[Path]) -> list[dict]:
    """Load + validate each point JSON. Raises on malformed / non-point files.

    Validation is SEMANTIC, not just key-presence: a negative/zero/NaN
    ``ms_per_step``, a non-integer or non-positive ``n_devices``, or a
    ``spmd_hlo_collective`` that is not exactly ``True``/``False``/``None`` are
    rejected up front so the gate math can never run on garbage (a negative
    baseline could otherwise produce a passing non-decisive verdict).
    """
    points = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        missing = [k for k in _REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"{p}: not a point results.json (missing {missing})")
        if data["mode"] != "point":
            raise ValueError(f"{p}: mode={data['mode']!r}, expected 'point'")
        if not isinstance(data["config"], dict):
            raise ValueError(f"{p}: config must be an object, got "
                             f"{type(data['config']).__name__}")
        cfg_missing = [k for k in _CONFIG_KEYS if k not in data["config"]]
        if cfg_missing:
            raise ValueError(f"{p}: config missing {cfg_missing}")

        n = data["n_devices"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"{p}: n_devices must be a positive integer, got {n!r}")
        ms = data["ms_per_step"]
        if isinstance(ms, bool) or not isinstance(ms, (int, float)) \
                or not math.isfinite(ms) or ms <= 0:
            raise ValueError(f"{p}: ms_per_step must be a finite positive number, "
                             f"got {ms!r}")
        coll = data["spmd_hlo_collective"]
        # Identity, not equality: `1 in (True, False, None)` is True in Python
        # (1 == True), so a stray JSON 1/0 must be rejected by `is` checks.
        if coll is not True and coll is not False and coll is not None:
            raise ValueError(f"{p}: spmd_hlo_collective must be true/false/null, "
                             f"got {coll!r}")
        pc = data.get("process_count")
        if pc is not None and (isinstance(pc, bool) or not isinstance(pc, int) or pc < 1):
            raise ValueError(f"{p}: process_count must be a positive integer, got {pc!r}")

        data["_path"] = str(p)
        points.append(data)
    return points


def aggregate(points: list[dict], pass_efficiency: float,
              gate_efficiency: bool, required_counts: list[int] | None = None) -> dict:
    """Build the scaling curve + gate verdict from the loaded points.

    ``required_counts`` (e.g. ``[1, 2, 3, 6]``) are device counts that MUST be
    present — the decisive claim is about a specific cross-node count (np6), so a
    ``PASS`` computed only from single-node points (np≤3, because np6 was missing
    or dropped) is rejected rather than silently certifying the cliff cleared.
    """
    if not points:
        raise ValueError("no points to aggregate")

    # --- Config consistency (fail loud, do not average different problems) ---
    configs = {tuple(pt["config"][k] for k in _CONFIG_KEYS) for pt in points}
    config_pass = len(configs) == 1
    config_reason = None
    if not config_pass:
        config_reason = (
            f"points disagree on {_CONFIG_KEYS}: "
            + "; ".join(sorted(str(c) for c in configs)))

    # --- Unique device counts (a duplicated count is ambiguous) -------------
    counts = [pt["n_devices"] for pt in points]
    dups = sorted({c for c in counts if counts.count(c) > 1})
    if dups:
        raise ValueError(f"duplicate n_devices across points: {dups} "
                         f"(each device count must be a single launch)")

    by_count = {pt["n_devices"]: pt for pt in points}
    baseline_present = 1 in by_count
    base_ms = by_count[1]["ms_per_step"] if baseline_present else None

    rows = []
    for n in sorted(by_count):
        pt = by_count[n]
        ms = pt["ms_per_step"]
        speedup = (base_ms / ms) if (baseline_present and ms > 0) else None
        efficiency = (speedup / n) if speedup is not None else None
        rows.append({
            "n_devices": n,
            "ms_per_step": ms,
            "speedup": speedup,
            "efficiency": efficiency,
            "spmd_hlo_collective": pt["spmd_hlo_collective"],
            "process_count": pt.get("process_count"),
        })

    # --- ppermute proof: every >1-device point lowered collective_permute ---
    multi = [r for r in rows if r["n_devices"] > 1]
    ppermute_pass = all(r["spmd_hlo_collective"] is True for r in multi) if multi else True

    largest = max(by_count)
    largest_row = next(r for r in rows if r["n_devices"] == largest)
    eff_at_largest = largest_row["efficiency"]

    # --- Efficiency gate (decisive only) -----------------------------------
    efficiency_pass = (not gate_efficiency) or (
        largest > 1 and eff_at_largest is not None
        and eff_at_largest >= pass_efficiency)

    # --- Required-count gate: the decisive claim names a specific count -----
    required = sorted(set(required_counts or []))
    missing_required = [c for c in required if c not in by_count]
    required_present = not missing_required

    # --- A verdict needs at least one >1-device point (else nothing scaled) --
    has_multi_point = bool(multi)

    reasons = []
    if not baseline_present:
        reasons.append("no single-device (n_devices=1) baseline point — cannot "
                       "compute speed-up; add an N=1 launch.")
    if not has_multi_point:
        reasons.append("no point with n_devices > 1 — nothing to scale; the "
                       "verdict would be vacuous.")
    if gate_efficiency and largest <= 1:
        reasons.append("decisive aggregation needs a point with n_devices > 1 "
                       "(only a baseline was provided).")
    if missing_required:
        reasons.append(f"required device counts absent: {missing_required} "
                       f"(present: {sorted(by_count)}); the decisive claim names "
                       f"these counts, so a PASS from a subset is not accepted.")
    if config_reason:
        reasons.append(config_reason)

    gates = {"baseline_present": baseline_present, "config_consistent": config_pass,
             "has_multi_point": has_multi_point, "required_counts_present": required_present,
             "ppermute": ppermute_pass, "efficiency": efficiency_pass}
    enforced = ["baseline_present", "config_consistent", "has_multi_point"]
    if required:
        enforced.append("required_counts_present")
    if multi:
        enforced.append("ppermute")
    if gate_efficiency:
        enforced.append("efficiency")
    overall_pass = all(gates[g] for g in enforced) and not reasons

    return {
        "n_points": len(points),
        "device_counts": sorted(by_count),
        "baseline_ms_per_step": base_ms,
        "curve": rows,
        "efficiency_at_largest": eff_at_largest,
        "largest_n_devices": largest,
        "pass_efficiency": pass_efficiency,
        "gate_efficiency": gate_efficiency,
        "required_counts": required,
        "missing_required_counts": missing_required,
        "gates": gates,
        "enforced_gates": enforced,
        "reasons": reasons,
        "overall_pass": overall_pass,
    }


def _fmt(v, spec: str) -> str:
    return format(v, spec) if isinstance(v, (int, float)) else str(v)


def _write(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verdict.json").write_text(json.dumps(result, indent=2))

    with (out_dir / "scaling.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_devices", "ms_per_step", "speedup", "efficiency",
                    "spmd_hlo_collective", "process_count"])
        for r in result["curve"]:
            w.writerow([r["n_devices"], r["ms_per_step"], r["speedup"],
                        r["efficiency"], r["spmd_hlo_collective"],
                        r["process_count"]])

    req = result["required_counts"]
    req_str = (f" required={req} missing={result['missing_required_counts']}"
               if req else "")
    lines = ["# Blocker-1 cross-node cube shard_map — aggregated scaling", "",
             f"points={result['n_points']} counts={result['device_counts']}"
             f"{req_str} decisive={result['gate_efficiency']}", "",
             "| n_devices | ms/step | speedup | efficiency | ppermute | procs |",
             "|---|---|---|---|---|---|"]
    for r in result["curve"]:
        lines.append(
            f"| {r['n_devices']} | {_fmt(r['ms_per_step'], '.3f')} | "
            f"{_fmt(r['speedup'], '.2f')} | {_fmt(r['efficiency'], '.2f')} | "
            f"{r['spmd_hlo_collective']} | {r['process_count']} |")
    lines += ["",
              f"efficiency at largest ({result['largest_n_devices']} dev) = "
              f"{_fmt(result['efficiency_at_largest'], '.2f')} "
              f"(threshold {result['pass_efficiency']:.2f}, "
              f"enforced={result['gate_efficiency']})", ""]
    for g, ok in result["gates"].items():
        mark = "PASS" if ok else "FAIL"
        enf = " (enforced)" if g in result["enforced_gates"] else ""
        lines.append(f"- {g}: {mark}{enf}")
    for reason in result["reasons"]:
        lines.append(f"- REASON: {reason}")
    lines += ["",
              f"OVERALL: {'PASS' if result['overall_pass'] else 'FAIL'} "
              f"(enforced: {', '.join(result['enforced_gates'])})"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", type=Path,
                   help="Directory whose */results.json are the per-N points.")
    g.add_argument("--inputs", type=Path, nargs="+",
                   help="Explicit list of point results.json files.")
    p.add_argument("--pass-efficiency", dest="pass_efficiency", type=float, default=0.6,
                   help="Efficiency threshold at the largest device count.")
    p.add_argument("--gate-efficiency", dest="gate_efficiency", action="store_true",
                   help="Make the efficiency threshold a hard exit gate (decisive).")
    p.add_argument("--require-counts", dest="require_counts", default=None,
                   help="Comma list of device counts that MUST be present (e.g. "
                        "'1,2,3,6'); a decisive PASS from a subset is rejected.")
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    required = ([int(c) for c in args.require_counts.split(",") if c.strip()]
                if args.require_counts else None)
    paths = _collect_paths(args.root, args.inputs)
    points = _load_points(paths)
    result = aggregate(points, args.pass_efficiency, args.gate_efficiency, required)
    _write(result, args.output_dir)
    print(json.dumps({"overall_pass": result["overall_pass"],
                      "gates": result["gates"],
                      "enforced": result["enforced_gates"],
                      "reasons": result["reasons"],
                      "efficiency_at_largest": result["efficiency_at_largest"],
                      "output_dir": str(args.output_dir)}, indent=2))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
