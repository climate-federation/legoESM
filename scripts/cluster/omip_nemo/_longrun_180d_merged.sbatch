#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=long180
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/long180_%j.out
#
# DOES THE EQUATORIAL COLD TONGUE EQUILIBRATE OR KEEP DRIFTING?
# The closure-code hunt is closed: our TKE closure is faithful and self-
# consistent (codex review + a self-consistency check, l_impl/l_buoy=1.0-1.1 at
# every depth).  The residual nino3 +1.42 K at day 30 is NOT a code bug; it is
# the self-reinforcing weak-EUC / high-Ri loop, and the open question is whether
# it is a still-growing drift (implementation) or a bounded 1-degree-resolution
# residual that equilibrates like FESOM did (+0.55/+0.20/+0.17 at d10/20/30).
#
# This extends the production run on the MERGED #1645 card to 180 days.  The
# TEST is the SHAPE of our own nino3 SST trajectory, which needs no oracle:
# a plateau after ~day 40-60 => bounded resolution residual (the canonical
# 1-deg warm bias takes years; 30 days is a transient); continued linear growth
# => an implementation drift still to be found.  Where NEMO's matched cold start
# reaches (GATEWAY 5-day means, days 1-90), we also score the bias directly:
# rec 5 = d26-30, rec 11 = d56-60, rec 17 = d86-90.  Beyond day 90 there is no
# matched NEMO cold start, so only OUR trajectory shape is read (not a bias).
#
# PRE-REGISTERED: read our nino3 (5S-5N,150W-90W) SST at days 30/60/90/120/150/
# 180 from the snapshots.
#   EQUILIBRATES  if the day-90->180 change is < 0.3 K AND < the day-30->90
#                 change  => bounded residual, resolution/spin-up, closure OK.
#   STILL DRIFTS  if it keeps growing >= linearly past day 90 => implementation
#                 defect remains; re-open with the trajectory's spatial pattern.
# No --kprofile-snapshots (the closure is settled; save the extra tendency
# evals).  Everything else byte-identical to the day-30 merged run.
set -uo pipefail
PIN=${1:?usage: sbatch _longrun_180d_merged.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/nemolev_trp_long180

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
  --years 0.4932 --snapshot-every-days 15 --diag-every-days 5 \
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
  --output $OUT 2>&1 | tail -25
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }

export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

echo "=== nino3 trajectory (matched NEMO to day 90; our shape beyond) ==="
for pair in "0030 5" "0060 11" "0090 17"; do
  set -- $pair
  SNAP=$OUT/snapshot_day$1.npz
  [ -f "$SNAP" ] || { echo "no day $1 snapshot"; continue; }
  echo "### day $1 vs GATEWAY rec $2 ###"
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GW_T" --nemo-time-idx "$2" --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label long180_d$1 \
    --output-dir $ROOT/results/omip_nemo/compare_long180_d$1 2>&1 \
    | grep -E "nino3_5S5N|\[SST\].*rmse" | head -3
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
    area = z["cell_area"] if "cell_area" in z else np.cos(np.deg2rad(lat))
    w = area[box]; v = T0[box]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    sst = float((v[ok] * w[ok]).sum() / w[ok].sum())
    dd = "" if prev is None else f"  d{sst - prev:+.3f}"
    print(f"day {d:3d}  nino3 SST {sst:.3f} C{dd}")
    prev = sst
print("\nPLATEAU (|d| shrinking, <0.3 K over d90->180) => bounded residual, "
      "closure OK; LINEAR growth => implementation drift remains.")
PY
echo DONE
