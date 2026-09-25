#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=e025nan
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/e025smk_%j.out
#
# FEASIBILITY SMOKE (user-approved, 2026-08-24): does the FULL faithful 1-degree
# card RUN at 1/4-degree (eORCA025) L75 fp64 on 4 GPUs (2 nodes x 2) under the
# lat-band SPMD ocean step, finite, without OOM?  Both reviewers NO-GO'd a naive
# 1-GPU 30-day run: L75 full-physics needs ~75 GB fp64 (>1 GPU), and dt=75 with
# the full card (TKE/MLE/ice/BBL/ISF host-step interactions) is unproven -- RK3
# only ever protected the stripped L20 config.  This smoke is the GATE before any
# 150-180 day EUC run is even designed.
#
# GATE (all must hold):
#   - reaches [done] (or the smoke projection) with NO [ABORT] and finite state
#   - per-GPU memory fits (report the last nvidia-smi sample)
#   - max|u|,max|v| stay physical (< a few m/s) through the smoke -- the 1/4-deg
#     cold start has blown up before (day 2) without RK3, and the full card at
#     dt=75 is untested
# On PASS: design the 150-180 day EUC run (140W section vs TAO, subsurface-T
# tripwire, --nemo-monthly-init IC, pinned forcing) as a SEPARATE decision.
# On FAIL: report where it broke (OOM vs blowup vs a physics flag unsupported
# under --distributed) and bisect dt 75->37.5 or the offending flag, NOT the
# science.
set +e
MAIN=/burg-archive/glab/users/pg2328/legoESM
cd "$MAIN"
# Run the CURRENT merged branch's packages (SPMD + full card are on it now, no
# worktree needed).  Prepend so the branch code wins over any editable install.
export PYTHONPATH="$(ls -d $MAIN/packages/*/ | sed 's:/$::' | tr '\n' ':')$MAIN"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
FORCING=/burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/core2_nyf_mod
OUT=$MAIN/results/omip_nemo/e025_fullcard_smoke
LOG=$MAIN/omip_nemo/e025smk_${SLURM_JOB_ID}_8g_nandbg.log
echo "HEAD: $(git rev-parse --short HEAD)  $(git log --oneline -1)"
echo "=== ALLOC ==="; date; echo "nodes: $SLURM_JOB_NODELIST"
scontrol show hostnames "$SLURM_JOB_NODELIST"
[ -d "$FORCING" ] && echo "forcing cache: $FORCING (pinned)" || echo "WARN: forcing cache $FORCING missing -> default resolver"

# srun: 1 process/GPU; jax.distributed bootstraps from PMIx; local GPU via
# SLURM_LOCALID (see run_eorca025_4gpu_multinode.sbatch for the topology notes).
# Full faithful card + --n-gpus 8 --distributed, dt=75, RK3, ~2.2 sim-days.
# --constraint=a100: run 9478088's allocation MIXED RTX8000 + A100 nodes.
# Mixed archs poisoned the shared compile cache (9493399 died loading a
# wrong-arch CUBIN: CUDA_ERROR_INVALID_SOURCE) and is the prime suspect for
# 9478088's silent first-step deadlock.  RTX8000 fp64 is 1/32-rate anyway.
#
# Hang forensics (run 9478088 froze at the FIRST sharded step: all 8 ranks
# 0% CPU / 0% GPU for 4+ h — a collective bootstrap deadlock, not compile):
#  - sitecustomize (PYTHONPATH-injected) arms an hourly all-thread Python
#    stack dump + SIGUSR1 dump-on-demand, so a silent hang self-reports.
#  - NCCL_DEBUG shows whether the inter-node NCCL bootstrap ever completes.
export PYTHONPATH="$MAIN/scripts/tmp/_faultdump:$PYTHONPATH"
srun --mpi=pmix \
  env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
      FAULTDUMP_EVERY_S=3600 \
      NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV \
      TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_MEM_FRACTION=.92 \
      JAX_COMPILATION_CACHE_DIR=/burg-archive/glab/users/pg2328/legoESM/.jax_cache_e025 \
  $PY "$MAIN/scripts/run/run_omip_core2.py" --grid tripole \
    --mesh data/grids/eORCA025_mesh_mask.nc --n-gpus 8 --distributed \
    --barotropic-solver implicit_cn \
    --dt 75 --min-levels 2 --nemo-vertical --balanced-init \
    --C-smag-lap 3.0 --smag-cfl-safety 0.125 \
    --ew-cyclic-overlap --partial-cell --pgf-scheme smc03 \
    --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
    --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
    --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
    --years 0.004 --diag-every-days 0.02 \
    --tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate \
    --bottom-drag-scheme nemo_quadratic \
    --iwm --iwm-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
    --dm2dc --isf --isf-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
    --woa-init \
    --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
    --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
    --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
      $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
    --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
    --tripole-vmix tke --ice-init $NEMOIN/Ice_initialization.nc \
    --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
    --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff 70 \
    --forcing-path $FORCING \
    --output ${OUT}_8g_nandbg 2>&1 | tee "$LOG" | tail -120
rc=${PIPESTATUS[0]}
echo "e025smk_rc=${rc}"
echo "=== smoke verdict ==="
if [ "$rc" -ne 0 ]; then
  echo "[SMOKE-FAIL] srun rc=${rc} (OOM? unsupported flag under --distributed? check $LOG)"
elif grep -q "\[ABORT\]" "$LOG"; then
  echo "[SMOKE-FAIL] non-finite [ABORT] -- cold-start blowup at 1/4 deg full card"; rc=1
elif grep -qE "\[done\]|\[smoke\]" "$LOG"; then
  echo "[SMOKE-OK] finite to end; full L75 card FITS + RUNS at 1/4 deg on 4 GPUs"
else
  echo "[SMOKE-INCOMPLETE] no done/abort marker (check $LOG)"; rc=1
fi
echo "=== max|u|,|v| trajectory (blowup check) ==="
grep -oE "max_abs_u[^,]*,[^,]*max_abs_v[^,}]*" "$LOG" | tail -6
echo "=== GPU MEM (last sample, this node) ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>&1 | head -3
echo "=== DONE ==="; exit ${rc}
