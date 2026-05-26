#!/usr/bin/env bash
# Run strong + weak scaling sweep across 1/2/4 ranks for the plane CRM
# step_halo path. Writes one CSV row per (mode, n_ranks) into
# results/dd_scaling/scaling_sweep.csv and prints the parallel
# efficiency table.
#
# Usage:
#   scripts/run_dd_scaling_sweep.sh                      # default
#   NX=64 NY=64 TIME_STEPS=30 scripts/run_dd_scaling_sweep.sh
#
# Env vars:
#   NX, NY            global grid for strong; per-rank grid for weak (default 24)
#   NLEV              vertical levels (default 20)
#   DT                outer dt [s] (default 1.0)
#   N_ACOUSTIC        acoustic substeps (default 12)
#   WARMUP            warmup steps before timing (default 3)
#   TIME_STEPS        steps inside the timed window (default 20)
#   RANKS             space-separated rank list (default "1 2 4")
#   OUTPUT            output dir (default results/dd_scaling)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NX="${NX:-24}"
NY="${NY:-24}"
NLEV="${NLEV:-20}"
DT="${DT:-1.0}"
N_ACOUSTIC="${N_ACOUSTIC:-12}"
WARMUP="${WARMUP:-3}"
TIME_STEPS="${TIME_STEPS:-20}"
RANKS="${RANKS:-1 2 4}"
OUTPUT="${OUTPUT:-results/dd_scaling}"
PYBIN="${PYBIN:-.venv/bin/python}"

mkdir -p "$OUTPUT"
CSV="$OUTPUT/scaling_sweep.csv"
rm -f "$CSV"

echo "DD scaling sweep: ranks=$RANKS nx=$NX ny=$NY nlev=$NLEV dt=$DT"
echo "  warmup=$WARMUP time_steps=$TIME_STEPS  -> $CSV"

for MODE in strong weak; do
    for N in $RANKS; do
        echo "--- $MODE n_ranks=$N ---"
        JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 mpirun -np "$N" "$PYBIN" \
            "$REPO_ROOT/scripts/bench_plane_crm_dd_scaling.py" \
            --mode "$MODE" \
            --nx "$NX" --ny "$NY" --nlev "$NLEV" \
            --dt "$DT" --n-acoustic-substeps "$N_ACOUSTIC" \
            --warmup-steps "$WARMUP" --time-steps "$TIME_STEPS" \
            --output "$CSV"
    done
done

echo ""
echo "=== Scaling sweep complete ==="
echo "CSV: $CSV"
"$PYBIN" - <<'PY'
import csv
from pathlib import Path
csv_path = Path("results/dd_scaling/scaling_sweep.csv")
rows = list(csv.DictReader(open(csv_path)))
by_mode = {}
for row in rows:
    by_mode.setdefault(row["mode"], []).append(row)
print()
for mode, mode_rows in by_mode.items():
    print(f"=== {mode} scaling ===")
    print(f"{'ranks':>6} {'wall/step [s]':>14} {'steps/s':>10} {'efficiency':>12}")
    mode_rows.sort(key=lambda r: int(r["n_ranks"]))
    baseline_wall = float(mode_rows[0]["wall_per_step_s"])
    for r in mode_rows:
        n = int(r["n_ranks"])
        wall = float(r["wall_per_step_s"])
        rate = float(r["steps_per_s"])
        if mode == "strong":
            eff = baseline_wall / (n * wall) if n > 0 else 0.0
        else:  # weak
            eff = baseline_wall / wall
        print(f"{n:6d} {wall:14.5f} {rate:10.3f} {eff:12.3f}")
PY
