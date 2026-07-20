#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=trp6_kpp
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trp6_kpp_%j.out
# TRP6 = the GRID-vs-SCHEME decomposition run (2x2 completion).
# Config is BYTE-IDENTICAL to trp5_eice90 EXCEPT the vertical-mixing closure:
#   trp5:  --tripole-vmix tke   (ORCA1 zdftke card, eice=3 default)
#   trp6:  --tripole-vmix kpp --kpp-ri-crit 0.15 --kpp-eice 3
# (KPP params matched to mpas9/mpas10/ll: Ri_crit 0.15, under-ice eice=3.)
# Two controlled comparisons from this ONE run (day-30 vs NEMO m1):
#   * SCHEME effect, grid fixed:  trp5 d30 (TKE-eice) vs trp6 d30 (KPP-eice)
#     -- same tripole grid, same forcing/IC/levers, ONLY the closure differs.
#   * GRID effect, scheme fixed:  trp6 (tripole-KPP-eice) vs mpas10
#     (MPAS-KPP-eice) -- same closure+params; residual declared diffs are the
#     known capability gaps (iwm/dm2dc/isf/bbl/sw-rgb-chl/pgf-smc03), gate-
#     classified.  Together these decompose the tripole<->MPAS Arctic MLD gap
#     (+6.7 vs +62.6) into scheme vs grid, which was previously confounded.
# 30-day probe: the halocline/MLD signature is formed by day 30.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/vfs-tke-fix; cd $WT
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=/burg-archive/glab/users/pg2328/legoESM
ROOT2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1
RUN=$ROOT2/cfgs/ORCA1/EXP00/RUN_REF
INP=$ROOT2/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields
INITD=$ROOT2/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
SSSCLIM=$INP/sss_climatology_for_restoring.nc
ICEIC=$ROOT2/cfgs/ORCA1/INPUTS/Ice_initialization.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
echo "=== worktree (fix branch) ==="; git -C $WT log --oneline -1
# --mesh must be the RELATIVE path (resolved via the worktree's data symlink):
# build_tripole's mesh-consistency guard compares path STRINGS and the grid
# side uses the relative default (absolute $ROOT/... form fails the guard).
GA="--grid tripole --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 \
  --min-levels 2 --C-smag-lap 3.0 --smag-cfl-safety 0.125 --ew-cyclic-overlap"
COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal \
  --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.085 --snapshot-every-days 30 --diag-every-days 5"
LEVERS="--tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate"
P45="--bottom-drag-scheme nemo_quadratic \
  --iwm --iwm-forcing-file $INP/zdfiwm_forcing_TRA.nc \
  --dm2dc \
  --isf --isf-forcing-file $INP/runoff-icb_DaiTrenberth_Depoorter.nc \
  --bbl-adv"
IC="--woa-init --woa-t $WTIC --woa-s $WSIC \
  --nemo-monthly-init $INITD/woce_temp_monthly_init_4p2.nc $INITD/woce_salt_monthly_init_4p2.nc \
  --sss-restore-file $SSSCLIM \
  --tripole-vmix kpp --kpp-ri-crit 0.15 --kpp-eice 3 --ice-init $ICEIC"
LABEL=trp6_kpp
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
MPAS10=$ROOT/results/omip_nemo/nemolev_mpas10_kppeice
TRP5=$ROOT/results/omip_nemo/nemolev_trp5_eice90
echo "=== RUN $LABEL (trp5 config, closure swapped tke->kpp[Ri 0.15, eice3]) ==="
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $LEVERS $P45 $IC --sw-rgb-chl --output $OUT 2>&1 | tail -80
echo "run_rc=${PIPESTATUS[0]}"; date
SNAP=$OUT/snapshot_day0030.npz
if [ -f "$SNAP" ]; then
  echo "=== $LABEL day-30 vs NEMO m1 ==="
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label "$LABEL" \
    --output-dir $ROOT/results/omip_nemo/compare_${LABEL}_m01 2>&1 | tail -25
  echo "=== $LABEL Arctic-SSS regions d30 (tripole-KPP-eice) ==="
  $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$SNAP" "$GT" "trp6_kpp d30" 1
  echo "=== SCHEME A/B (grid fixed): trp5 d30 (TKE-eice, same everything else) ==="
  [ -f "$TRP5/snapshot_day0030.npz" ] && \
    $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$TRP5/snapshot_day0030.npz" "$GT" "trp5_eice90 d30 (TKE)" 1
  echo "=== GRID A/B (scheme fixed): mpas10 d30 (MPAS-KPP-eice) ==="
  [ -f "$MPAS10/snapshot_day0030.npz" ] && \
    $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$MPAS10/snapshot_day0030.npz" "$GT" "mpas10_kppeice d30" 1
  echo "=== PARITY GATE trp6 vs mpas10 (same-scheme pair must be gate-clean) ==="
  [ -f "$MPAS10/run_manifest.json" ] && \
    $PY scripts/validate/omip_config_parity.py \
      $OUT/run_manifest.json $MPAS10/run_manifest.json | tail -12
else
  echo "[run] no day-30 snapshot -> check $OUT for blowup."
fi
echo "=== DONE ==="; date
