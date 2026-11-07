#!/bin/bash
#SBATCH --job-name=fesom_b1_5d
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesomb1_%j.out
#
# GLM B1 review: 10 unforced steps see nothing — 5 unforced days expose
# baroclinic adjustment, z-star PG-error spurious currents on the real
# bathymetry, polar static-instability overturning from the PHC winter
# state, and SSH/tracer conservation drift with NO forcing to hide behind.
# PRE-REGISTERED: (a) |u|max stays bounded (< ~1 m/s; an unforced ocean
# from rest+PHC has no energy source); (b) global-mean SST/SSS drift over
# 5 days < 0.05 C / 0.05 PSU (conservation; zstar should be ~exact);
# (c) SSH tendency DECAYS after the initial adjustment (GLM: a spurious
# steady SSH tendency = mesh-metric inconsistency, not physics).
set -uo pipefail
PIN=${1:?usage: sbatch _fesom_b1_unforced_5d.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:/burg-archive/glab/users/pg2328/fesom_jax_repo"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
MI=$ROOT/data/fesom2_core2/mesh_ic/core2_mesh_ic
JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid fesom \
  --fesom-mesh-dir "$MI/mesh_core2" --fesom-ic-dir "$MI/ic_core2" \
  --no-emp --dt 1800 --years 0.0137 --snapshot-every-days 1 \
  --output $ROOT/results/omip_nemo/fesom_b1_unforced5d 2>&1 | tail -10
echo "=== pre-registered checks ==="
$PY - <<'PY'
import glob
import numpy as np
snaps = sorted(glob.glob(
    "/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/"
    "fesom_b1_unforced5d/snapshot_day*.npz"))
prev_eta = None
for f in snaps:
    z = np.load(f)
    wet = z["land_mask"] > 0.5
    sst = float(np.mean(z["T"][wet, 0]))
    sss = float(np.mean(z["S"][wet, 0]))
    eta = z["eta"]
    deta = (float(np.mean(np.abs(eta - prev_eta)))
            if prev_eta is not None else float("nan"))
    prev_eta = eta
    print(f"{f.rsplit('/',1)[-1]}: SST {sst:.4f} SSS {sss:.4f} "
          f"|deta|/day {deta:.2e} etamax {np.abs(eta).max():.3f}")
PY
echo DONE
