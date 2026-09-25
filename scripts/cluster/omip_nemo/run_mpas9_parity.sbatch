#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=mpas9_parity
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|a40"
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mpas9_parity_%j.out
# MPAS9 = mpas8_corr + the 2026-07-18 cross-grid PARITY fixes, from the
# vfs-tke-fix tree (the wave's ONE pinned tree):
#   * --tracer-advection superbee NOW HONORED (was silently tvd — call-site
#     wiring bug fixed in this tree)
#   * --pgf-scheme smc03: wiring fixed BUT MEMORY-INFEASIBLE at ico7 L75
#     (per-column reconstruction builds >33 GiB intermediates; OOM on 48 GB
#     at 95% pool). MPAS stays on its adcroft default — DECLARED capability
#     gap until a chunked smc03 lands.
#   * --bottom-drag-scheme nemo_quadratic (mpas8 sbatch had dropped it)
# Everything else identical to mpas8_corr (KPP 0.15, ice-init, sn_sss,
# runoff-depth map, river gate, woa-annual IC — structured-only monthly
# loader gap unchanged).  KPP grid → TKE eice N/A; the KPP under-ice lever
# is still open.  Score day-90 vs NEMO m3; parity-gate the manifest against
# trp5_eice90 before quoting any cross-grid number.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/vfs-tke-fix; cd $WT
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=/burg-archive/glab/users/pg2328/legoESM
ROOT2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1
RUN=$ROOT2/cfgs/ORCA1/EXP00/RUN_REF
INP=$ROOT2/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
SSSCLIM=$INP/sss_climatology_for_restoring.nc
ICEIC=$ROOT2/cfgs/ORCA1/INPUTS/Ice_initialization.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
echo "=== worktree (fix tree) ==="; git -C $WT log --oneline -1
GA="--grid mpas --mpas-level 7 --mpas-lloyd 20 --dt 150"
COMMON="--nemo-vertical --partial-cell --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.3 --snapshot-every-days 30 --diag-every-days 5"
LEVERS="--tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate"
PARITY="--bottom-drag-scheme nemo_quadratic"
IC="--woa-init --woa-t $WTIC --woa-s $WSIC --sss-restore-file $SSSCLIM --ice-init $ICEIC"
LABEL=mpas9_parity
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
echo "=== RUN $LABEL (mpas8 + parity: superbee/smc03 honored + nemo_quadratic drag) ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $LEVERS $PARITY $IC --kpp-ri-crit 0.15 --output $OUT 2>&1 | tail -60
echo "run_rc=${PIPESTATUS[0]}"; date
SNAP=$OUT/snapshot_day0090.npz
if [ -f "$SNAP" ]; then
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GT" --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label "$LABEL" \
    --output-dir $ROOT/results/omip_nemo/compare_${LABEL}_m03 2>&1 | tail -30
  $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$SNAP" "$GT" "mpas9_parity d90" 3
  echo "=== PARITY GATE vs trp5 (first-minutes manifests) ==="
  $PY scripts/validate/omip_config_parity.py \
    $OUT/run_manifest.json \
    $ROOT/results/omip_nemo/nemolev_trp5_eice90/run_manifest.json | tail -12
fi
echo "=== DONE ==="; date
