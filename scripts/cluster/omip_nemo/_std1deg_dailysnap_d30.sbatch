#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=std_dsnap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/stddsnap_%j.out
#
# THE PROTOCOL-MISMATCH DISCRIMINATOR (codex HIGH + GLM top pick, 2026-08-25).
# Both reviewers left one suspect standing for the tropical-Pacific shallow
# MLD "bias" (-5 to -7 m): our 00Z INSTANTANEOUS snapshot samples the Pacific
# in local daytime (diurnal shoaling) while NEMO's mldr10_1 is a 5-DAY MEAN
# of an online diagnostic.  The collocation probe already refuted the
# barrier-layer (corr(dMLD,dSSS) ~ -0.1) and local-entrainment
# (corr(dMLD,dSST) ~ 0.0) explanations.
#
# ONE variable vs the standard card: --snapshot-every-days 1 (30 daily
# snapshots instead of 15-day cadence).  Then MLD is computed per snapshot
# over days 26-30 and AVERAGED (mean-of-diagnostic) before scoring vs
# GATEWAY rec 5 — the protocol-matched comparison.
# PRE-REGISTERED: tropical-Pacific mean-MLD bias |b| < 2 m => the "-5 to -7 m
# Pacific MLD bias" was the measurement protocol, CLOSED as a model issue;
# bias <= -5 m persisting => REAL model bias, reopen (mixing/forcing).
# The NW-Pacific box number is read the same way (+30 m expected to shrink
# if 5-day means under-sample storm deepening).
set -uo pipefail
PIN=${1:?usage: sbatch _std1deg_dailysnap_d30.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/nemolev_trp_std1deg_dailysnap_d30

echo "=== gate ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_core2_corrected_cache_default.py 2>&1 | tail -2
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.0823288 --snapshot-every-days 1 --diag-every-days 5 \
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
  --gateway-transports --mle --mle-ce 0.06 \
  --output $OUT 2>&1 | tail -15
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }

export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
SNAPS=""
for d in 0026 0027 0028 0029 0030; do
  [ -f $OUT/snapshot_day$d.npz ] || { echo "FATAL: snapshot_day$d missing"; exit 1; }
  SNAPS="$SNAPS --legoesm-snapshot $OUT/snapshot_day$d.npz"
done
echo "=== 5-day-MEAN MLD (d26-30) vs GATEWAY rec 5 (protocol matched) ==="
JAX_ENABLE_X64=1 $PY scripts/validate/ocean_fidelity/mld_basin_vs_nemo.py \
  $SNAPS --nemo-gridt "$GW_T" --nemo-time-idx 5 \
  --png $ROOT/results/omip_nemo/mld_pacific_meand30.png
echo "=== control: SAME run, day-30 INSTANTANEOUS only (isolates the protocol) ==="
JAX_ENABLE_X64=1 $PY scripts/validate/ocean_fidelity/mld_basin_vs_nemo.py \
  --legoesm-snapshot $OUT/snapshot_day0030.npz --nemo-gridt "$GW_T" --nemo-time-idx 5
echo DONE
