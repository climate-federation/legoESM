#!/bin/bash
#SBATCH --job-name=mpas_qlead15
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mpas_qlead15_%j.out
# A/B vs the 180d fullharm arm (ice_skin lead freeze, flood-fill-contaminated
# coastal IC): SAME card, 15 days, now with the NEMO lead heat budget
# (--ice-lead-freeze-source nemo_qlead, the driver default) and the masked
# NEMO-IC regrid.  Two variables vs the control by necessity (both are bug
# fixes); the qlead A/B alone is the in-job ice_skin twin below when ARM=both.
# PRE-REGISTERED: Arctic conc>0.95 mean supercooling deficit at day 15:
# control 0.346 K (mpas_fullharm_180d snapshot_day0015) -> nemo_qlead < 0.05 K.
# Ice concentration under the pack must not collapse (control: A>=0.99 cell
# count 2720 -> 756 in 5 days).
set -uo pipefail
PIN=${1:?usage: sbatch _mpas_qlead_15d.sbatch <pinned-sha> [nemo_qlead|ice_skin]}
SRC=${2:-nemo_qlead}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)  lead_freeze_source=$SRC"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/mpas_${SRC}_15d
JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid mpas \
  --mpas-level 7 --mpas-lloyd 20 --mpas-vmix tke \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
  --tke-kappa-convention veros_sqrte \
  --dt 150 --nemo-vertical --partial-cell \
  --mle --mle-ce 0.06 \
  --dm2dc --sw-rgb-chl \
  --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
  --prognostic-ice-dynamics free_drift --ice-lead-freeze-source $SRC \
  --years 0.0411 --snapshot-every-days 5 \
  --diag-every-days 5 --tracer-advection superbee --runoff-depth-nemo-ini \
  --river-mouth-restoring-gate --bottom-drag-scheme nemo_quadratic \
  --ice-ocean-heat-coeff 70 \
  --woa-init \
  --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
    $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --ice-init $NEMOIN/Ice_initialization.nc \
  --output $OUT 2>&1 | grep -E "heat-icecover\]   arctic|\[done\]|Traceback|Error|steps this leg|nemo-tmask" | head -80
echo "=== run rc=${PIPESTATUS[0]}"
$PY - "$OUT" <<'PY'
import sys, glob, numpy as np
for f in sorted(glob.glob(sys.argv[1] + "/snapshot_day*.npz")):
    z = np.load(f); T = z["T"]; wet = z["land_mask"] > 0.5; ic = z["ice_concentration"]
    lat = z["lat_T"]; lat = np.degrees(lat) if lat.max() < 7 else lat
    s = T[wet][:, 0]; c = ic[wet]; la = lat[wet]; d = np.maximum(-1.8 - s, 0)
    m = (c > 0.95) & (la > 60)
    print(f.rsplit('/', 1)[-1], f"Arctic conc>0.95 n={m.sum()} frac<-1.8={np.mean(s[m] < -1.8):.3f} "
          f"mean deficit={d[m].mean():.3f} K  SSTmin={s.min():.2f}  n(A>=0.99,Arctic)={((c >= 0.99) & (la > 60)).sum()}")
PY
echo DONE
