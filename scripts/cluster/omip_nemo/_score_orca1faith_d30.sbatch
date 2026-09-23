#!/bin/bash
#SBATCH --job-name=scorefaith
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scorefaith_%j.out
#
# SCORE THE ADOPTED-CARD BASELINE AGAINST NEMO.
#
# Both runs are scored against the SAME oracle record (GATEWAY rec 5, day 30),
# which is the only comparison the directive asks for: "compare to NEMO".
#
# The pre-adoption baseline is included as the SECOND model purely so the two
# NEMO-relative skills appear side by side. IT IS NOT AN ATTRIBUTION. Five
# things moved together when the card was adopted (four sets plus
# chlorophyll), so NO SINGLE LEVER may be credited for any difference between
# the two columns, and neither column licenses a statement about which change
# did what. A single-lever claim needs its own one-variable arm.
#
# CAVEAT to carry into the report: 207 cells (0.317% of our wet points) are
# wet for us and dry for the oracle -- measured, one-directional, 187 of them
# northern mid-latitude. They are not a comparison and should be excluded
# from any same-domain statement.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
# The card writes snapshot_final.npz at its target day, not a day-numbered
# file (leg 1 did the same at day 15), so day0030 usually does not exist.
NEW=$R/trp_orca1faith_d30/snapshot_day0030.npz
[ -r "$NEW" ] || NEW=$R/trp_orca1faith_d30/snapshot_final.npz
OLD=$R/trp_base2_d30/snapshot_day0030.npz

for f in "$NEW" "$OLD" "$GW"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
echo "=== inputs (mtimes prove which state is being scored) ==="
ls -l --time-style=+%Y-%m-%d_%H:%M "$NEW" "$OLD"

$PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
  --tripole "$NEW" \
  --mpas    "$OLD" \
  --label-tripole orca1faith --label-mpas pre_adoption \
  --nemo-gridt "$GW" --nemo-time-idx 5 --res-deg 1.0 \
  --out-dir $R/score_orca1faith_d30 2>&1 | tail -40

echo "=== report.json headline numbers ==="
$PY - <<'PY'
import json, pathlib
p = pathlib.Path("/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/"
                 "score_orca1faith_d30/report.json")
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
ls -1 /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/score_orca1faith_d30/*.png 2>/dev/null | head
echo DONE
