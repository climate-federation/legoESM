#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=trpevd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:55:00
# 11:55, not more: the short partition's own ceiling is 12 h, and a job asking
# for more sits in PartitionTimeLimit forever.  The measured steady rate with
# the sampler on is 0.47 steps/s, so a 15-day leg is ~5.1 h.
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trpevd_%j.out
# EVD ARM (one variable vs trp_base2): NEMO's enhanced vertical diffusion,
# ORCA1 &namzdf ln_zdfevd=T rn_evd=100 nn_evdm=0 -> --convection
# enhanced_diffusion, K_conv 100 (tracers only; nu_conv stays 0), K_bg 0,
# trigger on NEMO's bn2 (TEOS-10), hard N2 <= -1e-12 (zdfevd.F90:26).
# NEMO's MIN(rn2, rn2b) needs a before-state; this card runs the rk3 outer
# integrator (as ORCA1 does under key_RK3), so the trigger is now-only.
# --evd-occupancy-every-hours 1: the fraction of wet columns whose first
# interior interface carries K_conv (cold-tongue box + global), hourly, mean
# per csv row -- the firing diagnostic both claim reviewers required.
# Everything else byte-identical to _trp_base2_d30.sbatch.  Cost of the hourly
# occupancy sample: one extra K-profile evaluation (~a model step) per hour,
# ~720 over 30 days, ~4% of the leg walltime; the K read is the solve's own
# (diagnose_vertical_K), a proxy only in that it is re-evaluated at the
# sample time rather than captured inside the step.
# PRE-REGISTERED (day 30 AND day 15, judged against trp_base2 at the SAME 00Z
# snapshot sampling; NEMO's 5-day means are the reference, never the gate):
#   PRIMARY  220-240E |lat|<=2 T(0.5 m)-T(20 m) <= 1.9 K at day 30 (base2 3.80;
#     NEMO 0.76). Day 15 (base2 2.50) is reported as the second matched point:
#     a 5-day mean endpoint would need daily snapshots the CONTROL does not
#     have, so two matched snapshots stand in for it and single-snapshot
#     synoptic noise is a caveat on the number, not a movable gate.
#   MIDDLE OUTCOME, registered in advance: a lens landing in 1.9-3.0 K with
#     occupancy > 0.3 = MECHANISM ENGAGED but incomplete. 00Z is ~15-17 h local
#     here, so the residual contains the day's radiative restratification, which
#     a hard-N^2 trigger cannot erode by design. That cell is NOT a licence to
#     tune K_conv: inspect the diurnal rebuild (hourly occupancy vs the lens)
#     first, and treat any K_conv change as a new, separately asked arm.
#   SECONDARY nino3 bias (base2 +1.12, expect <= +0.6); global SST rmse <= 0.52
#     (base2 0.50) as a do-no-harm FLAG, not a veto (inside 30-day noise).
#   DESCRIPTIVE, not gated: Arctic/Antarctic MLD bias (base2 +13.5 m Arctic),
#     Arctic SSS bias (base2 +0.04), and the sea-ice trio in diag_timeseries
#     (area, mean concentration, max thickness) -- under-ice brine convection
#     now fires 100 m2/s on tracers, so the ice column is the one harm channel
#     this arm could touch without the SST metrics seeing it.
#   DESCRIPTIVE, matched to the control's own bars: Antarctic SST bias
#     (base2 +0.22) and Antarctic MLD bias (base2 +5.0 m) -- Southern Ocean
#     deep convection is where EVD's largest global effect lives.
#   REFUTE: lens > 3 K at day 30 while the NIGHT-TIME box occupancy (local
#     solar hour 18-06 in evd_occupancy.csv, not the 5-day row mean, which
#     cannot separate "every night" from "weakly all day") exceeds 0.3.
#   NO-OP (plumbing, do not score): evd_top_occ_eq < 0.02 averaged over days
#     5-30 -> the trigger never fired; fix the wiring, do not draw physics.
#   Hourly sampling under-resolves sub-hourly firing, so the occupancy is a
#   LOWER bound -- which is the safe direction for both the 0.3 and 0.02 bars.
#   The occupancy columns measure the TRIGGER (NEMO bn2 <= -1e-12 at the top
#   interfaces), which under the hard trigger IS the firing fraction and costs
#   a top-levels slice; measuring the delivered K hourly is not affordable
#   (diagnose_vertical_K has no jit of its own: 42 s per call, ~17 steps).
#   GLM's consistency identity therefore reads the SNAPSHOT K dumps
#   (--kprofile-snapshots) against the trigger duty cycle, not an hourly K.
set -uo pipefail
PIN=${1:?usage: sbatch _trp_evd_d30.sbatch <pinned-sha> [leg: 1|2]}
LEG=${2:-1}
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/trp_evd_d30

echo "=== gate ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_core2_corrected_cache_default.py 2>&1 | tail -2
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 \
  --A-h 2e4 --A-h-profile-file $NEMOIN/orca1_inputs/data_repository/input_fields/eddy_viscosity_3D.nc \
  --visc-schedule '0:1e4:3.0,8:1e4:1.5,16:1e4:0.5' \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --ice-lead-freeze-source nemo_qlead \
  --years $([ "$LEG" = 1 ] && echo 0.0411 || echo 0.0822) --snapshot-every-days 15 --diag-every-days 5 \
  --restart-save $OUT/restart_leg$LEG.npz $([ "$LEG" = 2 ] && echo "--restart-from $OUT/restart_leg1.npz") \
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
  --gateway-transports --mle --mle-ce 0.06 --kprofile-snapshots \
  --convection enhanced_diffusion --convection-K-conv 100 --convection-K-bg 0 \
  --convection-n2-mode nemo_bn2 --convection-n2-eos teos10 \
  --convection-trigger hard --convection-n2-threshold=-1e-12 \
  --evd-occupancy-every-hours 1 \
  --output $OUT 2>&1 | tail -25
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }
# The trigger the model RESOLVED, not the flags typed (codex: occupancy alone
# cannot tell the hard bn2 trigger from the legacy smooth one -- both fire at
# night).  A wrong resolved config is a FATAL, not a scored arm.
$PY - $OUT/run_manifest.json <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
ed = m["config"]["resolved_config"]["runtime_config"]["physics"]["convection"]["enhanced_diffusion"]
want = {"K_conv": 100.0, "K_bg": 0.0, "nu_conv": 0.0, "n2_mode": "nemo_bn2",
        "n2_eos_form": "teos10", "smooth_transition": False,
        "n2_threshold": -1e-12, "two_level_trigger": False}
bad = {k: (ed.get(k), v) for k, v in want.items() if ed.get(k) != v}
sch = m["config"]["resolved_config"]["runtime_config"]["physics"]["convection"]["scheme"]
print("resolved convection:", sch, {k: ed.get(k) for k in want})
if sch != "enhanced_diffusion" or bad:
    raise SystemExit(f"FATAL: resolved EVD config is not the NEMO trigger: {bad}")
PY
[ $? -eq 0 ] || exit 1

export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

[ "$LEG" = 2 ] || { cp -f $OUT/snapshot_final.npz $OUT/snapshot_day0015.npz; echo "leg 1 done (day 15); state copied to snapshot_day0015.npz; scoring runs in leg 2"; exit 0; }
echo "=== nino3 trajectory (matched NEMO to day 90; our shape beyond) ==="
for pair in "0030 5"; do
  set -- $pair
  SNAP=$OUT/snapshot_day$1.npz
  [ -f "$SNAP" ] || { echo "no day $1 snapshot"; continue; }
  echo "### day $1 vs GATEWAY rec $2 ###"
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GW_T" --nemo-time-idx "$2" --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label trp_evd_d$1 \
    --output-dir $ROOT/results/omip_nemo/compare_trpevd_d$1 2>&1 \
    | grep -E "nino3_5S5N|\[SST\].*rmse" | head -3
done

echo "=== undercurrent / thermocline vs GATEWAY (same instrument as job 9648737) ==="
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot $OUT/snapshot_day0030.npz --nemo-gridt "$GW_T" --nemo-t-recs 5:6 \
  --label trp_evd_d30 --nemo-wfile $GW/ORCA1_5d_20000101_20000331_grid_W.nc --nemo-w-var wo --nemo-w-recs 5:6 \
  --nemo-ufile $GW/ORCA1_5d_20000101_20000331_grid_U.nc --meridional-lon 240,260 \
  --nemo-vfile $GW/ORCA1_5d_20000101_20000331_grid_V.nc 2>&1 | grep -vE "Warning|warn" | tail -120

echo "=== THE PRIMARY ENDPOINT: near-surface lens, evd arm vs base2, same 00Z sampling ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py \
  --snapshot $ROOT/results/omip_nemo/trp_base2_d30/snapshot_day0030.npz --label base2 \
  --snapshot $OUT/snapshot_day0030.npz --label evd \
  --snapshot-early $ROOT/results/omip_nemo/trp_base2_d30/snapshot_day0015.npz \
  --snapshot-early $OUT/snapshot_day0015.npz \
  --nemo-gridt "$GW_T" --nemo-rec 5 --nemo-rec-early 2 --surface-profile 2>&1 | sed -n '/=== Near-surface/,$p'
echo "=== night-vs-day firing (local solar hour at 230E; the 0.3 REFUTE bar reads the night rows) ==="
$PY - $OUT/evd_occupancy.csv <<'PY2'
import sys, numpy as np
try:
    d = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
except Exception as e:
    raise SystemExit(f"no per-sample occupancy record: {e}")
h, occ, occ3 = d["local_hour_box"], d["occ_eq"], d["occ3_eq"]
night = (h >= 18.0) | (h < 6.0)
for tag, sel in (("night 18-06", night), ("day 06-18", ~night),
                 ("all", np.ones_like(night, bool))):
    if sel.any():
        print(f"{tag:>12}: n={int(sel.sum()):5d}  occ(top iface) {occ[sel].mean():.3f}"
              f"  occ(top 3) {occ3[sel].mean():.3f}  max {occ[sel].max():.3f}")
PY2

echo "=== firing occupancy + sea ice: each row = mean of hourly samples since the"
echo "    previous row (5 days); the REFUTE bar reads the day-30 row, the NO-OP bar"
echo "    the mean of the day 5-30 rows; ice columns are descriptive ==="
head -1 $OUT/diag_timeseries.csv
tail -n 8 $OUT/diag_timeseries.csv
echo "--- the same columns from the control, for the ice/occupancy comparison ---"
head -1 $ROOT/results/omip_nemo/trp_base2_d30/diag_timeseries.csv
tail -n 3 $ROOT/results/omip_nemo/trp_base2_d30/diag_timeseries.csv
echo DONE
