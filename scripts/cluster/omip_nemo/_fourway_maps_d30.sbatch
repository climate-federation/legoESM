#!/bin/bash
#SBATCH --job-name=4waymaps
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fourway_%j.out
#
# THE VERDICT, AS A PICTURE: our three grids against the oracle at day 30.
#
# The enhanced-diffusion arm was REFUTED today -- surface-to-20 m contrast 3.43
# against the control's 3.49, where confirming needed 1.5 and the oracle sits at
# 0.76. The arm is NOT a fourth panel here, and deliberately so: at six
# hundredths of a kelvin from its control it would be visually identical, and a
# panel nobody can tell apart invites the reader to see a difference that the
# numbers say is absent. The arm's result is a number, not a map.
#
# REUSED, NOT REBUILT: scripts/plot/plot_fourway_grids_nemo.py already renders
# exactly this -- one figure per field, all grids beside NEMO.
#
# THE ONE TRAP THIS CARD EXISTS TO AVOID: the plotter's --nemo-time-idx
# defaults to -1, the LAST record, which is day 90 of the oracle's archive.
# Omitting it silently compares our day 30 against the oracle's day 90 and has
# already produced one phantom regression in this campaign. Day 30 is record 5,
# by the convention REC = DAY/5 - 1, verified against the file's own time
# bounds: record 5 spans 2000-01-02 to 2000-01-07, i.e. model days 25-30 from
# an oracle start of 1999-12-08. Passed explicitly below.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
OUT=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/fourway_d30

TRP=$R/trp_pinned_d30/snapshot_final.npz
MPA=$R/mpas_orca1faithtint_d30/snapshot_day0030.npz
FES=$R/fesom_orca1faithtint_d30/snapshot_day0030.npz
for f in "$TRP" "$MPA" "$FES" "$GT"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

# Same stamp discipline as the scoring card: a lane that stamps its day must
# agree with the mark, and one that stamps nothing says so rather than being
# quietly trusted.
$PY - "$TRP" "$MPA" "$FES" <<'STAMP' || { echo "FATAL: a lane is not at day 30"; exit 1; }
import sys
import numpy as np
for path in sys.argv[1:]:
    d = np.load(path, allow_pickle=True)
    if "time_days" not in d.files:
        print(f"  {path.split('/')[-2]}: NO STAMP (unverified)")
        continue
    got = float(np.asarray(d["time_days"]).ravel()[0])
    print(f"  {path.split('/')[-2]}: stamp {got:g}")
    if abs(got - 30.0) > 1e-6:
        sys.exit(1)
STAMP

mkdir -p "$OUT"
$PY -u scripts/plot/plot_fourway_grids_nemo.py \
  --tripole "$TRP" --mpas "$MPA" --fesom "$FES" \
  --nemo-gridt "$GT" --nemo-time-idx 5 \
  --out-dir "$OUT"
rc=$?
echo "=== card exit: $rc ==="
ls -la "$OUT"/*.png 2>/dev/null | tail -20
exit $rc
