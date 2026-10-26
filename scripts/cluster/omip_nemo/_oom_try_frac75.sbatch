#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=trp_frac75
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx8000
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trp_frac75_%j.out
# Antarctic under-melt A/B (2026-07-28): the trpsbc_win_d90 config VERBATIM
# (from its run_manifest command_line — the controlled-comparison baseline)
# with ONE change: --ice-ocean-heat-coeff 70 (vs default 20) + 62-day window
# (covers the SH melt season; control = trpsbc_win_d90's own d60 snapshot).
set +e
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date
git -C $ROOT log --oneline -1
# GPU-contention guard (2026-07-30): slurm does NOT mask devices on these
# dual-GPU nodes (nvidia-smi shows BOTH), so JAX grabs GPU 0 — which another
# job may already own; XLA's 0.95 preallocation then fails with
# "Failed to load in-memory CUBIN ... CUDA_ERROR_OUT_OF_MEMORY".  Pick the
# GPU with the most FREE memory and pin it explicitly.
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
  --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
echo "[gpu] pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 --C-smag-lap 3.0 --smag-cfl-safety 0.125 --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.02 --snapshot-every-days 30 --diag-every-days 5 --tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate --bottom-drag-scheme nemo_quadratic --iwm --iwm-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc --dm2dc --isf --isf-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc --bbl-adv --woa-init --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc --nemo-monthly-init /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc --sss-restore-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc --tripole-vmix tke --ice-init /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/Ice_initialization.nc --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic --sw-rgb-chl --output /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/nemolev_trp_frac75_d7 2>&1 | tail -30
echo "run_rc=${PIPESTATUS[0]}"; date
echo "=== DONE ==="
