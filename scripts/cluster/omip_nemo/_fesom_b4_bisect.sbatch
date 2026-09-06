#!/bin/bash
#SBATCH --job-name=fesom_b4_bisect
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesomb4bis_%j.out
#
# B4 card (same as _fesom_b4_d30.sbatch) with DAILY snapshots so the day of
# the non-finite state is visible, and env toggles to drop ONE B4 lever:
#   sbatch _fesom_b4_bisect.sbatch <tag> <pinned-sha>   (env: ICE=0 SSS=0 RUNOFF=0)
# 9599860 (full B4, 5-day ramp) reported "non-finite temperature after 1442
# steps" with no earlier check; overflow in T_sfc**4 says SST diverged first.
set -uo pipefail
TAG=${1:?tag}; PIN=${2:?pinned-sha}
ICE=${ICE:-1}; SSS=${SSS:-1}; RUNOFF=${RUNOFF:-1}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)  arm=$TAG ICE=$ICE SSS=$SSS RUNOFF=$RUNOFF"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:/burg-archive/glab/users/pg2328/fesom_jax_repo:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
MI=$ROOT/data/fesom2_core2/mesh_ic/core2_mesh_ic
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
EXTRA=()
[ "$ICE" = 1 ] && EXTRA+=(--prognostic-sea-ice --prognostic-ice-dynamics free_drift
  --ice-init $NEMOIN/Ice_initialization.nc --ice-ocean-heat-coeff 70)
[ "$SSS" = 1 ] && EXTRA+=(--sss-restore --sss-restore-channel water_flux
  --sss-restore-normalization live_s --sss-restore-tau-days 45.5
  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc)
[ "$RUNOFF" = 1 ] && EXTRA+=(--runoff)
OUT=$ROOT/results/omip_nemo/fesom_b4_bis_$TAG
JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid fesom \
  --fesom-mesh-dir "$MI/mesh_core2" --fesom-ic-dir "$MI/ic_core2" \
  --woa-init \
  --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
    $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --dm2dc --sw-rgb-chl "${EXTRA[@]}" \
  --forcing-ramp-days 5 \
  --dt 1800 --years 0.0823288 --snapshot-every-days 1 \
  --output $OUT 2>&1 | tail -14
echo "=== run rc=$? ==="
$PY - "$OUT" <<'PY'
import glob, sys
import numpy as np
for f in sorted(glob.glob(sys.argv[1] + "/snapshot_day*.npz")):
    z = np.load(f); T = z["T"]; wet = z["land_mask"] > 0.5
    ok = np.isfinite(T[wet]).all()
    print(f.rsplit('/',1)[-1], "finite" if ok else "NaN",
          f"SSTrange[{np.nanmin(T[wet,0]):.2f},{np.nanmax(T[wet,0]):.2f}]",
          f"Tmax_all={np.nanmax(np.abs(T[wet])):.1f}")
PY
echo DONE
