"""Decompose fp32/fp64 step time into memory + compute share via two-precision fit.

Model:
    T(prec) = a · dtype_bytes + c
where
    a [s/byte]            ≈ (passes/cell·lev × N_cells · N_lev) / BW
    c [s, dtype-invariant]  = compute time per step (FLOPs, dispatch)

Solving across paired (fp32, fp64) measurements:
    a = (T_fp64 − T_fp32) / (8 − 4) = (T_fp64 − T_fp32) / 4
    c = 2·T_fp32 − T_fp64

This is NOT a measurement. It is a 2-point linear fit that assumes:
  (i) the same JAX/XLA graph differs only in dtype between the runs;
  (ii) HBM-traffic scales 2× between fp32 and fp64 (true if memory pattern
       identical and no precision-dependent algorithmic divergence);
  (iii) compute time is dtype-invariant (only true if FLOPs are not the
       bottleneck — consumer GPUs penalise fp64 ALU so this is an
       upper-bound on the compute share).

Outputs:
  ratio        = T_fp64 / T_fp32   (≈2 → bandwidth-bound; <1 → mixed-dtype anomaly)
  a*BW         = inferred passes·cell·lev /step
  mem_share    = a × 8 / T_fp64   (fraction of fp64 step that is bandwidth)
  eff_HBM_GB_s = a × BW           (this IS independent of BW choice —
                                   only assumed for converting a into passes)

Usage:
    PYTHONPATH=. .venv/bin/python scripts/analyze_gpu_scaling.py \\
        --fp64-csv 'results/scaling_gpu_baseline_cs/strong_scaling.csv' \\
        --fp32-csv 'results/scaling_gpu_baseline_cs_fp32/strong_scaling.csv' \\
        --label cubed-sphere [--peak-bw 1.79e12]
"""
from __future__ import annotations

import argparse
import csv
import glob

# RTX 5090 default; override via --peak-bw.
PEAK_BW_DEFAULT = 1.79e12  # bytes/s

REQUIRED_COLS = ("resolution", "n_levels", "total_cells", "time_per_step_ms")


def _read_csv(pattern: str) -> list[dict]:
    rows = []
    files = glob.glob(pattern)
    if not files:
        print(f"ERROR: no files match {pattern!r}")
        return rows
    for p in files:
        with open(p, newline="") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames is None:
                continue
            missing = [c for c in REQUIRED_COLS if c not in rd.fieldnames]
            if missing:
                print(f"WARN: {p} missing columns {missing}; skipping")
                continue
            for ix, r in enumerate(rd):
                # Validate numerics
                try:
                    t_ms = float(r["time_per_step_ms"])
                    tc = float(r["total_cells"])
                    if not (t_ms > 0 and tc > 0):
                        print(f"WARN: {p} row {ix} non-positive — skipping")
                        continue
                except (TypeError, ValueError):
                    print(f"WARN: {p} row {ix} bad numerics — skipping")
                    continue
                r["_src"] = p
                rows.append(r)
    return rows


def _key(r) -> tuple:
    # Strict pairing — includes mode, physics, n_gpus so runs from different
    # configs are not silently paired.
    return (
        str(r.get("resolution", "?")),
        int(r["n_levels"]),
        r.get("mode", ""),
        r.get("physics_level", ""),
        int(r.get("n_gpus", 1) or 1),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp64-csv", required=True)
    p.add_argument("--fp32-csv", required=True)
    p.add_argument("--label", default="grid")
    p.add_argument("--peak-bw", type=float, default=PEAK_BW_DEFAULT,
                   help="Peak HBM bandwidth in bytes/s (default RTX 5090 1.79e12)")
    args = p.parse_args()

    fp64 = _read_csv(args.fp64_csv)
    fp32 = _read_csv(args.fp32_csv)
    if not fp64 or not fp32:
        print("missing input rows; exiting")
        return 1

    by_key_64 = {_key(r): r for r in fp64}
    by_key_32 = {_key(r): r for r in fp32}

    # Duplicate detection
    if len(by_key_64) != len(fp64):
        print("WARN: duplicate keys in fp64 CSV — later rows overwrote earlier")
    if len(by_key_32) != len(fp32):
        print("WARN: duplicate keys in fp32 CSV — later rows overwrote earlier")

    common = sorted(set(by_key_64).intersection(by_key_32))
    if not common:
        print("no common (res, lev, mode, physics, n_gpus) rows in fp64/fp32 inputs")
        return 1

    print(f"\n{args.label} — 2-precision linear decomposition (peak HBM "
          f"= {args.peak_bw/1e12:.2f} TB/s)")
    print("=" * 110)
    print(f"{'res':>6} {'cells':>10} {'fp64 ms':>9} {'fp32 ms':>9} "
          f"{'ratio':>6} {'mem ms':>8} {'cmp ms':>8} {'mem%':>5} "
          f"{'B/cell·lev':>12} {'mem-bw%':>8}  classification")
    print("-" * 130)

    for k in common:
        r64 = by_key_64[k]
        r32 = by_key_32[k]
        res = r64["resolution"]
        ncells = int(r64["total_cells"])
        nlev = int(r64["n_levels"])
        t64 = float(r64["time_per_step_ms"]) * 1e-3
        t32 = float(r32["time_per_step_ms"]) * 1e-3
        ratio = t64 / t32

        # Classify regime by ratio.
        if ratio < 1.0:
            classify = "FP32-ANOMALY (mixed dtype?)"
        elif ratio < 1.5:
            classify = "compute/dispatch dominated"
        elif ratio < 2.05:
            classify = "memory-bound"
        elif ratio < 2.5:
            classify = "mostly memory + small fp64 ALU pen."
        else:
            classify = "fp64 ALU penalty significant"

        # 2-precision linear fit (ASSUMES compute time dtype-invariant):
        #   T(prec) = a · dtype_bytes + c
        # Solve: a = (t64 - t32) / 4 ; c = 2·t32 - t64.
        # ONLY meaningful when ratio ∈ [1.5, 2.1] (memory-bound regime).
        # For ratio > 2.1 the fp64 ALU penalty makes the assumption invalid:
        # c < 0 and the decomposition reports non-physical numbers.
        a = (t64 - t32) / 4.0  # s/byte if assumption holds
        c = 2.0 * t32 - t64
        decomp_valid = (1.5 <= ratio <= 2.5) and a > 0
        if decomp_valid:
            mem_t64 = 8 * a
            cmp_t = max(c, 0.0)
            mem_pct = min(mem_t64 / t64 * 100, 100.0)
            b_per_cell_lev = (8 * a) * args.peak_bw / (ncells * nlev)
            eff_bw_pct = mem_pct  # fraction of step that is at peak BW
            print(f"{res:>6} {ncells:>10,} {t64*1000:>9.2f} {t32*1000:>9.2f} "
                  f"{ratio:>6.2f} {mem_t64*1000:>8.2f} {cmp_t*1000:>8.2f} "
                  f"{mem_pct:>4.0f}% {b_per_cell_lev:>10.0f}   "
                  f"{eff_bw_pct:>6.0f}%  {classify}")
        else:
            # Skip decomposition; still report ratio.
            print(f"{res:>6} {ncells:>10,} {t64*1000:>9.2f} {t32*1000:>9.2f} "
                  f"{ratio:>6.2f} {'—':>8} {'—':>8} {'—':>5} {'—':>12}   "
                  f"{'—':>7}  {classify}")

    print()
    print("Classification:")
    print("  ratio <1.0 : fp32 SLOWER than fp64 — mixed-dtype/anomalous graph")
    print("  ratio 1.0-1.5: compute / dispatch / overhead floor dominates")
    print("  ratio 1.5-2.05: bandwidth-bound (fit valid)")
    print("  ratio 2.05-2.5: bandwidth + small fp64 ALU penalty (fit valid)")
    print("  ratio >2.5: fp64 ALU penalty large — 2-precision fit unreliable")
    print()
    print("Fit assumes T(prec) = a × dtype_bytes + c (compute dtype-invariant).")
    print("This breaks on consumer GPUs (RTX 50-series fp64 ALU = 1/64 fp32 nominal).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
