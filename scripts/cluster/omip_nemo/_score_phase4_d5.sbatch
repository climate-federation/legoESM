#!/bin/bash
#SBATCH --job-name=score_ph4
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score_ph4_%j.out
# Score each 6-hourly phase of day 4-5 of trp_phase4_d5 against NEMO record 0
# (days 1-5 mean) through the SAME three-way scorer and the SAME cell set as
# the day-30 cards (the MPAS/FESOM d30 snapshots supplied as --also-mask),
# with the pinned control's own day-5 00 UTC snapshot in the second slot.
# Pre-registration is in _trp_phase4_d5.sbatch.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
# The scorer refuses to pair snapshots from different days (a guard this
# campaign earned), so each phase is scored with ITSELF in the second slot and
# without the d30 --also-mask files; the cell set is therefore the phase's own
# wet set, identical across the five rows below, which is what the phase
# comparison needs. The control's day-5 00 UTC snapshot is scored the same way.
score() {  # $1 = snapshot, $2 = label
  OUT=$R/score_phase4_d5/$2; mkdir -p $OUT
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole "$1" --mpas "$1" --label-tripole "$2" --label-mpas dup \
    --nemo-gridt $N --nemo-time-idx 0 --out-dir $OUT > $OUT/log.txt 2>&1 \
    || echo "scorer failed for $2: $(tail -2 $OUT/log.txt)"
}
for tag in 0004.250 0004.500 0004.750 0005; do
  S=$R/trp_phase4_d5/snapshot_day$tag.npz
  [ -s "$S" ] || { echo "MISSING $S"; continue; }
  score "$S" ph_$tag
done
score $R/trp_pinned_d30/snapshot_day0005.npz ctl_d5_00utc
$PY - <<'PYEOF'
import json,glob,os
R="/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/score_phase4_d5"
print(f"{'phase (UTC)':14s} {'nino3 SST bias':>15s} {'eq_pac SST bias':>16s} {'tropics bias':>13s} {'global rmse':>12s} {'n cells':>8s}")
for d in sorted(glob.glob(R+"/ph_*"))+sorted(glob.glob(R+"/ctl_*")):
    p=d+"/report.json"
    if not os.path.exists(p): print(d, "no report"); continue
    r=json.load(open(p)); a=[k for k in r["fields"]["SST"] if k.endswith("_vs_nemo") and not k.startswith("dup")][0]
    f=r["fields"]["SST"][a]
    print(f"{os.path.basename(d):14s} {f['nino3']['bias']:+15.3f} {f['eq_pacific']['bias']:+16.3f} {f['bands']['tropics_23S_23N']['bias']:+13.3f} {f['global']['rmse']:12.3f} {r['n_common_cells']:8d}")
PYEOF
echo "=== DONE ==="
