#!/bin/bash
#SBATCH --job-name=mpasbase2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:55:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mpasbase2_%j.out
# NEW PRODUCTION BASELINE, MPAS (user decisions 2026-09-06): the unified MPAS
# card on the merged code with the explicit vorticity filter OFF (--K-zeta-bih
# 0; the recipe default is now 0 and a nonzero value is gated at the first
# step) and the virtual closure's temperature twin (code).  30 days in one
# leg on short (0.53 steps/s -> ~9.5 h); restart-save kept as insurance.
# PRE-REGISTERED (day 30 vs GATEWAY rec 5, vs mpas_unified_180d): SST rmse
#   within 0.05 of 0.50; nino3 <= +1.08; no polar regression; max|u| < 2.5.
set -uo pipefail
PIN=${1:?usage: sbatch _mpas_base2_d30.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/mpas_base2_d30

echo "=== gate: corrected forcing is the default ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_core2_corrected_cache_default.py 2>&1 | tail -2
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid mpas \
  --mpas-level 7 --mpas-lloyd 20 --mpas-vmix tke --K-zeta-bih 0 \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
  --tke-kappa-convention veros_sqrte \
  --dt 150 --nemo-vertical --partial-cell \
  --mle --mle-ce 0.06 \
  --dm2dc --sw-rgb-chl \
  --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
  --prognostic-ice-dynamics free_drift --ice-lead-freeze-source nemo_qlead \
  --isf --isf-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
  --iwm --iwm-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
  --bbl-adv --years 0.0822 --snapshot-every-days 15 --restart-save $OUT/restart.npz \
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
  --output $OUT 2>&1 | tail -25
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }

export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

echo "=== matched-day scores vs GATEWAY (rec 5/11/17 = d26-30/56-60/86-90) ==="
for pair in "0030 5"; do
  set -- $pair
  SNAP=$OUT/snapshot_day$1.npz
  [ -f "$SNAP" ] || { echo "no day $1 snapshot"; continue; }
  echo "### day $1 vs GATEWAY rec $2 ###"
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GW_T" --nemo-time-idx "$2" --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label mpas_base2_d$1 \
    --output-dir $ROOT/results/omip_nemo/compare_mpasbase2_d$1 2>&1 | tail -38
done

echo "=== OUR nino3 SST trajectory (plateau test, no oracle) ==="
$PY - "$OUT" <<'PY'
import sys, glob, numpy as np
out = sys.argv[1]
prev = None
for d in range(15, 181, 15):
    snap = f"{out}/snapshot_day{d:04d}.npz"
    if not glob.os.path.exists(snap):
        continue
    z = np.load(snap)
    lat = z["lat_T"]; lon = z["lon_T"] % 360
    T0 = z["T"][..., 0]; wet = z["land_mask"] > 0.5
    box = (np.abs(lat) <= 5) & (lon >= 210) & (lon <= 270) & wet
    # codex 9562869: MPAS SCVT cells are quasi-uniform area; uniform weights,
    # not cos(lat), when the snapshot carries no cell_area.
    area = z["cell_area"] if "cell_area" in z else np.ones_like(lat)
    w = area[box]; v = T0[box]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    sst = float((v[ok] * w[ok]).sum() / w[ok].sum())
    dd = "" if prev is None else f"  d{sst - prev:+.3f}"
    print(f"day {d:3d}  nino3 SST {sst:.3f} C{dd}")
    prev = sst
print("\nPLATEAU (|d| shrinking, <0.3 K over d90->180) => bounded residual; "
      "LINEAR growth => implementation drift remains.")
PY
echo DONE
