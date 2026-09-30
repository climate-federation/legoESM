#!/bin/bash
#SBATCH --job-name=nancpu
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nancpu_%j.out
#
# The GPU bisection arms are stuck in the queue. This asks a CHEAPER question
# that does not need a GPU at all: does the NaN appear IMMEDIATELY, or does it
# develop over days?
#
# PRE-REGISTERED:
#   NaN within the first day  => a broken TERM, not a slow instability. Bisect
#       on CPU from here, no GPU needed, and look at what the adopted set
#       changes structurally rather than at timestep/stability.
#   CLEAN after a day         => a slower instability; the GPU arms are the
#       right instrument and this says nothing more.
# Either way this costs no GPU and queues immediately.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
OUT=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/nancpu_fullcard
echo "HEAD: $(git log --oneline -1)"

$PY -u scripts/run/run_omip_core2.py \
  --grid tripole \
  --mesh /burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc \
  --nlev 75 --dt 150 --years 0.00274 \
  --woa-init \
  --forcing-path /burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/core2_nyf_mod \
  --prognostic-sea-ice --ice-freeze-scheme unesco \
  --gm-treguier --gm-kappa-min 0 --gm-slope-scheme nemo_iso_lap \
  --gm-slope-positions nemo_native --gm-bolus-advection through_fct \
  --gm-msc-stabilize --redi-coefficient nemo21 --redi-aht0 900 \
  --ice-exchange nemo_si3 --ice-snow bulk --ice-snow-flooding on \
  --forcing-time-interp linear --chl-remap nemo_scrip \
  --output "$OUT" --snapshot-every-days 1 2>&1 | tail -30
echo "=== exit $? ==="

$PY -u - "$OUT" <<'PY'
import numpy as np, glob, os, sys
snaps = sorted(glob.glob(os.path.join(sys.argv[1], "snapshot_day*.npz")))
if not snaps:
    print("NO SNAPSHOT -- died before writing one"); raise SystemExit(1)
d = np.load(snaps[-1])
bad = {k: int(np.isnan(np.asarray(d[k], float)).sum())
       for k in ("T", "S", "u", "v") if k in d}
print(f"\n{os.path.basename(snaps[-1])} NaN counts: {bad}")
print("VERDICT:", "NaN WITHIN A DAY -- a broken term, not slow instability"
      if sum(bad.values()) else "clean after a day -- slower instability")
PY
echo DONE
