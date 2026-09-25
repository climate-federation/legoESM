#!/bin/bash
#SBATCH --job-name=spmd2gpu
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=5:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/spmd2gpu_%j.out
#
# MULTI-PROCESS DISCRIMINATOR (2026-08-26): the single-process 8-fake-device
# CPU run of this exact card is CLEAN through 30 steps, so the eORCA025
# step-23 NaN is either multi-process-specific or 1/4-degree-mesh-specific.
# This is the SAME 1-degree card under jax.distributed with 2 processes x
# 1 GPU on one node (362 rows, no pad needed at 2 bands): NaN => the
# multi-process runtime (collectives / gather addressability) is the
# defect, at 1 degree, debuggable on one node.  Clean => 1/4-mesh-specific.
# (Original CPU-repro header follows.)
# goes NaN by step <=23 (v exactly zero, u/T NaN).  The CPU equivalence gates
# all pass, but they are synthetic small grids, single process, NO south pad,
# and no host-loop BCs.  This runs the FULL faithful 1-degree card through the
# SAME driver lane (--n-gpus 8, global-in/global-out, host BCs, prognostic
# ice) on 8 fake CPU devices in ONE process, on the REAL eORCA1 mesh
# (362 rows -> south-pad +6 -> 368 = 8x46), ~30 steps at per-step diag.
#   NaN <= 30 steps  => the defect is in the banding/pad/host-BC lane itself,
#                       reproduced offline and debuggable here.
#   clean            => the defect is specific to MULTI-PROCESS execution
#                       (jax.distributed collectives / gather addressability),
#                       and the hunt moves there.
set -uo pipefail
PIN=${1:?usage: sbatch _spmd8_1deg_cpu_repro.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/spmd2gpu_1deg_repro

srun --mpi=pmix \
  env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
$PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --n-gpus 2 --distributed --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.000143 --diag-every-days 0.0018 \
  --tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate \
  --bottom-drag-scheme nemo_quadratic \
  --iwm --iwm-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
  --dm2dc --isf --isf-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
  --bbl-adv --woa-init \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
    $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --tripole-vmix tke --ice-init $NEMOIN/Ice_initialization.nc \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
  --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff 70 \
  --output $OUT 2>&1 | tail -60
rc=$?
echo "=== rc=$rc ==="
echo "=== diag rows (NaN check) ==="
cat $OUT/diag_timeseries.csv 2>/dev/null | head -40
echo DONE
