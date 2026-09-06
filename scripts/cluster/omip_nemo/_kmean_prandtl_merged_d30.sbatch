#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=kmean
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/kmean_%j.out
#
# THE DECISIVE MEASUREMENT: our equatorial Prandtl vs NEMO, done RIGHT.
#
# A single instantaneous snapshot of a TKE diffusivity cannot be compared to
# NEMO's 5-day-mean avm/avt -- proven this session, the same arm's surface avt
# ratio swung x0.49 (day 10) to x0.00 (day 30), which is intermittency, not a
# bias.  This arm dumps K EVERY DAY (--snapshot-every-days 1 --kprofile-
# snapshots) and averages days 26-30 offline, matching NEMO grid_W record 5
# (days 26-30).  That is the first apples-to-apples K comparison in the
# campaign, and it settles the question every remaining lever hinges on:
#   RATIO defect  (our avm ~ NEMO, avt << NEMO)  -> fix the Prandtl partition
#   STARVATION    (both << NEMO)                 -> fix TKE production / the
#                                                   missing equatorial convection
#
# SECOND PURPOSE: this is the FIRST cold-tongue read on the MERGED card, which
# now carries Dhruv's #1645 NEMO-faithful bottom-drag time level + control
# volume.  Production mixing length (nn_mxl 3) -- the mxl 3-vs-2 A/B this
# session REFUTED any cold-tongue effect (0.03 C), so nn_mxl stays at the
# production value and is not the variable here.
#
# PRE-REGISTERED, cold tongue 200-260E |lat|<=2, 5-day-mean K (days 26-30) vs
# NEMO rec 5, surface 5-65 m band:
#   NEMO target: avm 1.68e-3, avt 1.57e-3 m2/s, Pr 1.21, conv-frac ~0.44.
#   RATIO defect   CONFIRMED if avm/NEMO in 0.7-1.5x AND avt/NEMO <= 0.5x
#   STARVATION     CONFIRMED if BOTH avm/NEMO <= 0.5x AND avt/NEMO <= 0.5x
#   NEITHER        both within 0.5-2x -> the closure K is right, the cold
#                  tongue is not a vertical-mixing-magnitude defect
#   Report conv-frac either way: ours << 0.44 means we systematically fail to
#   convect where NEMO does, a SEPARATE (regime, not magnitude) finding.
# Also re-scores SST/nino3 on the merged card so #1645's effect on the tongue
# is measured (mxl3 pre-merge was +1.42 at day 30; a move is Dhruv's).
set -uo pipefail
PIN=${1:?usage: sbatch _kmean_prandtl_merged_d30.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/nemolev_trp_kmean_d30

echo "=== gate ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/unit/test_snapshot_kprofiles.py \
  tests/ocean/unit/test_equatorial_diffusivity_oracle.py 2>&1 | tail -2
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.0822 --snapshot-every-days 1 --diag-every-days 5 \
  --kprofile-snapshots \
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
GW_W=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc

echo "=== 5-DAY-MEAN K (days 26-30), the apples-to-apples comparison ==="
# Average the daily K dumps over the NEMO record's own window, write one
# synthetic snapshot, and run the probe on it.  This is the fix for the
# instantaneous-vs-mean mismatch that made every prior K number noise.
$PY - "$OUT" <<'PY'
import sys, glob, numpy as np
out = sys.argv[1]
days = [26, 27, 28, 29, 30]
snaps = [f"{out}/snapshot_day{d:04d}.npz" for d in days]
snaps = [s for s in snaps if glob.os.path.exists(s)]
if len(snaps) < 3:
    raise SystemExit(f"FAIL: only {len(snaps)} of days 26-30 present; "
                     "cannot form a 5-day mean")
Ks_M, Ks_H = [], []
base = None
for s in snaps:
    z = np.load(s)
    Ks_M.append(np.asarray(z["K_M_diag"]))
    Ks_H.append(np.asarray(z["K_H_diag"]))
    if base is None:
        base = {k: np.asarray(z[k]) for k in
                ("T", "S", "lat_T", "lon_T", "land_mask", "z_center_ref",
                 "z_interface_ref")}
base["K_M_diag"] = np.nanmean(np.stack(Ks_M), axis=0)
base["K_H_diag"] = np.nanmean(np.stack(Ks_H), axis=0)
np.savez(f"{out}/snapshot_kmean_d26_30.npz", **base)
print(f"wrote 5-day-mean K from {len(snaps)} days: {[s.split('day')[-1] for s in snaps]}")
PY
$PY -u scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
  --nemo-gridw "$GW_W" --rec 5 \
  --legoesm-snapshot $OUT/snapshot_kmean_d26_30.npz 2>&1 | tail -90
echo "=== kmean eqdiff rc=$?"

echo "=== SST/nino3 on the merged #1645 card (day 30) ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot $OUT/snapshot_day0030.npz \
  --nemo-gridt "$GW_T" --nemo-time-idx 5 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label kmean_d30 \
  --output-dir $ROOT/results/omip_nemo/compare_kmean_d30 2>&1 | tail -32
echo DONE
