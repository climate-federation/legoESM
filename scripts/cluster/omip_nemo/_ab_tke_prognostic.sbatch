#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=trpsbc_prog
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trpsbc_ab_%j.out
# A/B: NEMO Dirichlet vs Veros-flux surface-TKE BC on the tripole zdftke path.
# CONTROLLED: identical trp5 ship command, SAME tree (main, /burg), d30; the
# ONLY variable is --tke-surface-bc ($SBC).  Hypothesis: the summer-hemisphere
# surface-intensified warm (SST +0.53, SHmid +1.07 / antarctic +0.97 / tropics
# +0.71 at d30 vs NEMO; profile probe = ML heat wedge) is under-mixed by the
# Veros flux (|tau|/rho0)^{3/2} BC; NEMO's linear Dirichlet en(1)=rn_ebb|tau|/
# rho0 injects more surface TKE under moderate summer winds -> cools the ML.
# Compare the two arms at d30 (NEMO month=1, seasonally matched to a Jan d30).
set +e
SBC=nemo_dirichlet
ROOT=/burg/glab/users/pg2328/legoESM; cd $ROOT
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
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
echo "=== ARM: --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 ==="
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
OUT=$ROOT/results/omip_nemo/nemolev_trpsbc_ndir_mxl3_prog
mkdir -p $OUT
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $LEVERS $P45 $IC --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic --sw-rgb-chl --output $OUT 2>&1 | tail -70
echo "run_rc=${PIPESTATUS[0]}"; date
SNAP=$OUT/snapshot_day0030.npz
if [ -f "$SNAP" ]; then
  echo "=== SCORE $SBC d30 vs NEMO m1 ==="
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label "trpsbc_ndir_mxl3_prog" \
    --output-dir $ROOT/results/omip_nemo/compare_trpsbc_ndir_mxl3_prog_m01 2>&1 | tail -40
  $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$SNAP" "$GT" "trpsbc_ndir_mxl3_prog d30" 1 2>&1 | tail -12
else
  echo "[run] no day-30 snapshot -> check $OUT for blowup."
fi
echo "=== DONE $SBC ==="; date
