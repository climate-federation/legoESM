#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=trpsbc_gmtreg2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trpsbc_ab_%j.out
# TEST ARM for the GM coefficient: NEMO ldf_eiv (Treguier, nn_aei_ijk_t=21)
# with a 200 m^2/s equatorial floor, INSTEAD of the tripole default Visbeck.
#
# CONTROL / BASELINE for this arm is the EXISTING run
#   results/omip_nemo/nemolev_trpsbc_ndir_mxl3_prog/snapshot_day0030.npz
# (same tree, same d30, same --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3
# --tke-prognostic).  The ONLY difference below is
#   --gm-treguier --gm-kappa-min 200
# so any d30 delta is attributable to the GM coefficient alone.
#
# NOTE: this file previously described itself as a surface-TKE-BC A/B whose
# "ONLY variable is --tke-surface-bc ($SBC)".  That was WRONG on two counts:
# $SBC was never interpolated into the run command (the BC is hardcoded on the
# ship line, $SBC only labelled the outputs), and the arm additionally turned on
# --gm-treguier.  Scoring it against the pre-TKE trp5 baseline would move >=2
# variables at once -- a confound, not a result.  If you want the TKE-BC A/B,
# run BOTH arms by looping $SBC and leave the GM flags at their defaults.
set +e
# Labels the output directories only; the BC itself is on the ship line below
# and both must be edited together.
SBC=nemo_dirichlet
ROOT=/burg/glab/users/pg2328/legoESM; cd $ROOT
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1
RUN=$N/cfgs/ORCA1/EXP00/RUN_REF; INP=$N/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields
INITD=$N/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
SSSCLIM=$INP/sss_climatology_for_restoring.nc
ICEIC=$N/cfgs/ORCA1/INPUTS/Ice_initialization.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
echo "=== tree (main) ==="; git -C $ROOT log --oneline -1; git -C $ROOT branch --show-current
echo "=== ARM: --gm-treguier --gm-kappa-min 200 (baseline: ndir_mxl3_prog) ==="
GA="--grid tripole --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 \
  --min-levels 2 --C-smag-lap 3.0 --smag-cfl-safety 0.125 --ew-cyclic-overlap"
# d30 A/B: --years 0.0834 (~30.4 d); the first 30 d of the trajectory are
# identical to a longer run, so this is the trp5 d30 physics reached faster.
COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal \
  --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.0834 --snapshot-every-days 30 --diag-every-days 5"
LEVERS="--tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate"
P45="--bottom-drag-scheme nemo_quadratic \
  --iwm --iwm-forcing-file $INP/zdfiwm_forcing_TRA.nc --dm2dc \
  --isf --isf-forcing-file $INP/runoff-icb_DaiTrenberth_Depoorter.nc --bbl-adv"
IC="--woa-init --woa-t $WTIC --woa-s $WSIC \
  --nemo-monthly-init $INITD/woce_temp_monthly_init_4p2.nc $INITD/woce_salt_monthly_init_4p2.nc \
  --sss-restore-file $SSSCLIM --tripole-vmix tke --ice-init $ICEIC"
OUT=$ROOT/results/omip_nemo/nemolev_trpsbc_gmtreg2
# A STALE snapshot in $OUT would be scored as if this job produced it (set +e
# means a crashed run still reaches the scorer). Refuse to reuse it.
if [ -e "$OUT" ]; then
  mv "$OUT" "$OUT.stale.$SLURM_JOB_ID" || { echo "[guard] cannot clear $OUT"; exit 1; }
  echo "[guard] moved pre-existing $OUT aside (stale-snapshot guard)"
fi
mkdir -p $OUT
# Provenance: the arms are only comparable if both came from THIS tree.
git -C $ROOT rev-parse HEAD > $OUT/commit.sha
git -C $ROOT status --porcelain > $OUT/dirty.txt
echo "[provenance] commit $(cat $OUT/commit.sha) dirty_files=$(wc -l < $OUT/dirty.txt)"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $LEVERS $P45 $IC --tke-surface-bc $SBC --tke-mxl-choice 3 --tke-prognostic --gm-treguier --gm-kappa-min 200 --sw-rgb-chl --output $OUT 2>&1 | tail -70
RUN_RC=${PIPESTATUS[0]}; echo "run_rc=$RUN_RC"; date
if [ "$RUN_RC" -ne 0 ]; then
  echo "[guard] test arm FAILED (rc=$RUN_RC) -> refusing to score. Any"
  echo "        snapshot below would be partial/stale, and scoring it would"
  echo "        report a GM effect that no completed run produced."
  exit $RUN_RC
fi
SNAP=$OUT/snapshot_day0030.npz
# CONTROL arm: the pre-existing no-GM-flag run at the SAME d30, same tree, same
# TKE card.  Scored HERE, by the SAME harness invocation as the test arm, so
# the two numbers are comparable; a metric lifted from an older log was
# computed by a possibly-different protocol and is NOT a valid control.
BASE=$ROOT/results/omip_nemo/nemolev_trpsbc_ndir_mxl3_prog/snapshot_day0030.npz
score_arm () {   # $1 = snapshot, $2 = label
  if [ ! -f "$1" ]; then echo "[score] MISSING $2 snapshot: $1"; return 1; fi
  echo "=== SCORE $2 d30 vs NEMO m1 ==="
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$1" \
    --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label "$2" \
    --output-dir $ROOT/results/omip_nemo/compare_${2}_m01 2>&1 | tail -40
  $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$1" "$GT" "$2 d30" 1 2>&1 | tail -12
}
if [ -f "$SNAP" ]; then
  score_arm "$BASE" "ctrl_ndir_mxl3_prog"     # control: --gm-treguier OFF
  score_arm "$SNAP" "test_gmtreg_kmin200"     # test:    --gm-treguier ON
  echo "[compare] the ONLY difference between the two arms above is"
  echo "          --gm-treguier --gm-kappa-min 200. If the control snapshot is"
  echo "          missing, RE-RUN it on this tree before claiming a direction."
else
  echo "[run] no day-30 snapshot -> check $OUT for blowup."
fi
echo "=== DONE ==="; date
