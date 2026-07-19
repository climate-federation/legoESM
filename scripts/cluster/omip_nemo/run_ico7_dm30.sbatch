#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ico7_dm30
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ico7_dm30_%j.out
# ico7 MPAS + NEW per-cell NEMO ln_rnf_depth_ini runoff-depth map (--runoff-depth-nemo-ini)
# with a SHALLOW cap (--runoff-dep-max 30) -> Siberian-shelf freshwater retention.
# Otherwise IDENTICAL to run_ico7_ewfix (which used flat --runoff-depth-spread-m 150).
# A/B: does the per-cell shallow map cut MPAS Arctic SSS (+3.69 salty) toward NEMO?
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/nemo-gaps-p4p5; cd $WT
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=/burg-archive/glab/users/pg2328/legoESM
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
git -C $WT log --oneline -1
GA="--grid mpas --mpas-level 7 --mpas-lloyd 20 --dt 150"
COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv \
  --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
  --prognostic-ice-dynamics free_drift --years 0.3 --snapshot-every-days 15 --diag-every-days 5"
LEVERS="--tracer-advection superbee --runoff-depth-nemo-ini --runoff-dep-max 30"
IC="--woa-init --woa-t $WTIC --woa-s $WSIC"
OUT=$ROOT/results/omip_nemo/nemolev_ico7_dm30
echo "=== RUN ico7_dm30 (MPAS + per-cell nemo-ini runoff, dep_max=30) ==="
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py $GA $COMMON $LEVERS $IC --output $OUT 2>&1 | tail -60
echo "run_rc=${PIPESTATUS[0]}"; date
SNAP=$OUT/snapshot_day0090.npz
if [ -f "$SNAP" ]; then
  echo "=== SCORE + Arctic regions (vs NEMO m3) ==="
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" \
    --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful --grid-label ico7_dm30 \
    --output-dir $ROOT/results/omip_nemo/compare_ico7_dm30_m03 2>&1 | tail -18
  $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$SNAP" "$GT" "MPAS ico7_dm30 (nemo-ini dep30)" 3 2>/dev/null
fi
echo "=== DONE ==="; date
