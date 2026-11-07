#!/bin/bash
#SBATCH --job-name=scoretint
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scoretint_%j.out
#
# SCORE THE MASK + IC-BLEND RUN (trp_orca1faithtint_d30) AGAINST NEMO beside a
# partner run chosen by BASE: trp_orca1faithmask_d30 (mask only, days 5-15;
# the day-paired difference isolates the initial-condition blend alone) or
# trp_orca1faith_d30 (pre-mask baseline, any day). Built by cp -f from
# _score_orca1faithmask.sbatch; only names and paths changed.
#
# Built from the working _score_orca1faith_d30.sbatch (cp -f, then the paths,
# labels and the oracle record changed; nothing else). Both snapshots are
# scored against the SAME oracle record: GATEWAY 5-day mean DAY/5-1 (0-based), so a
# day-10 pair reads record 1 (days 5-10) and the day-30 pair reads record 5.
#
# This IS a one-variable comparison: trp_orca1z0_d30 differs from
# trp_orca1faith_d30 in exactly --tke-surface-bc-level nemo_z0 (card
# a811a3e2f enforces code identity with baseline commit 3c37fea09), so a
# difference between the two columns IS attributable to the surface-TKE
# placement. The pre-registered VERDICT, however, is scored by
# _probe_vertical_momentum.sbatch / _probe_euc_profiles.sbatch on the day-30
# snapshot; these maps are for visual inspection and the regional scorecard.
#
# CAVEAT carried from the baseline card: 207 cells (0.317% of our wet points)
# are wet for us and dry for the oracle -- excluded from any same-domain
# statement.
#
# Usage: sbatch --export=ALL,DAY=<5-multiple>[,BASE=trp_orca1faithmask_d30] _score_orca1faithtint.sbatch
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
DAY=$(( 10#${DAY:-10} ))   # 10#: a zero-padded DAY would parse as octal
# 0-based GATEWAY record k = mean over run days 5k..5k+5 (time_counter_bounds,
# measured 2026-09-15), so the record that ENDS at snapshot day D is D/5-1 --
# the convention of every day-30 scorecard (rec 5 = days 25-30) and of the
# sector maps.  D/5 (used until 2026-09-15) was the record STARTING at D.
REC=$(( DAY / 5 - 1 ))
[ $(( (REC + 1) * 5 )) -eq "$DAY" ] && [ "$REC" -ge 0 ] || { echo "FATAL: DAY=$DAY is not a positive multiple of 5 -- no matching 5-day oracle record"; exit 1; }
TAG=$(printf 'day%04d' "$DAY")
# The card writes snapshot_final.npz at its target day (30), not a
# day-numbered file; every other 5-day snapshot is day-numbered.
NEW=$R/trp_orca1faithtint_d30/snapshot_$TAG.npz
[ -r "$NEW" ] || { [ "$DAY" -eq 30 ] && NEW=$R/trp_orca1faithtint_d30/snapshot_final.npz; }
BASE=${BASE:-trp_orca1faith_d30}
OLD=$R/$BASE/snapshot_$TAG.npz
[ -r "$OLD" ] || { [ "$DAY" -eq 30 ] && OLD=$R/$BASE/snapshot_final.npz; }
OUTDIR=$R/score_orca1faithtint_d${DAY}_vs_${BASE#trp_}

for f in "$NEW" "$OLD" "$GW"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
echo "=== inputs (mtimes prove which state is being scored); oracle record $REC (0-based) = days $((DAY-5))-$DAY ==="
ls -l --time-style=+%Y-%m-%d_%H:%M "$NEW" "$OLD"
$PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
  --tripole "$NEW" \
  --mpas    "$OLD" \
  --label-tripole tint --label-mpas ${BASE#trp_} \
  --nemo-gridt "$GW" --nemo-time-idx "$REC" --res-deg 1.0 \
  --out-dir "$OUTDIR" 2>&1 | tail -40
echo "=== report.json headline numbers ==="
OUTDIR="$OUTDIR" $PY - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ["OUTDIR"]) / "report.json"
if not p.exists():
    raise SystemExit(f"no report at {p}")
r = json.load(open(p))
for fld in ("SST", "SSS", "MLD"):
    blk = r["fields"].get(fld, {})
    for side, v in blk.items():
        if side == "unit" or not isinstance(v, dict):
            continue
        g = v.get("global")
        if g:
            print(f"  {fld:4s} {side:22s} bias {g['bias']:+.4f}  "
                  f"rmse {g['rmse']:.4f}")
PY
echo "=== figures written ==="
ls -1 "$OUTDIR"/*.png 2>/dev/null | head
echo DONE
