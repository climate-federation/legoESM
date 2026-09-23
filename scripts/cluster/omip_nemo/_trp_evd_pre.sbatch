#!/bin/bash
#SBATCH --job-name=evdpre
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:55:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/evdpre_%j.out
#
# PREFLIGHT for _trp_evd_d30.sbatch (Claude review 2026-09-07): the resolved
# EVD config and the firing diagnostic are checked on a HANDFUL OF STEPS, so a
# mis-resolved trigger or a dead diagnostic costs minutes instead of the 7-hour
# leg it would otherwise abort at the end of.  Same command as the arm, tiny
# --years, its own output directory, occupancy sampled every ~2.5 min.
set -uo pipefail
PIN=${1:?usage: sbatch _trp_evd_pre.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
PRE=$ROOT/results/omip_nemo/trp_evd_preflight
mkdir -p $PRE
LEG=1
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
  --years ${YEARS:-0.0013} --snapshot-every-days 1 --diag-every-days ${DIAGD:-0.05} \
   \
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
  --evd-occupancy-every-hours ${OCCH:-1} \
  --output $PRE 2>&1 | tail -25
run_rc=$?
echo "=== preflight run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: preflight run failed rc=$run_rc"; exit 1; }

echo "=== 1. RESOLVED config (must be the NEMO trigger, not the legacy smooth one) ==="
$PY - $PRE/run_manifest.json <<'PY2'
import json, sys
m = json.load(open(sys.argv[1]))
ed = m["config"]["resolved_config"]["runtime_config"]["physics"]["convection"]["enhanced_diffusion"]
sch = m["config"]["resolved_config"]["runtime_config"]["physics"]["convection"]["scheme"]
want = {"K_conv": 100.0, "K_bg": 0.0, "nu_conv": 0.0, "n2_mode": "nemo_bn2",
        "n2_eos_form": "teos10", "smooth_transition": False,
        "n2_threshold": -1e-12, "two_level_trigger": False}
print("scheme:", sch, {k: ed.get(k) for k in want})
bad = {k: (ed.get(k), v) for k, v in want.items() if ed.get(k) != v}
if sch != "enhanced_diffusion" or bad:
    raise SystemExit(f"FATAL: resolved EVD config is wrong: {bad}")
print("resolved config OK")
PY2
[ $? -eq 0 ] || exit 1

echo "=== 2. the firing diagnostic produced samples (a dead instrument is not a null result) ==="
$PY - $PRE/evd_occupancy.csv <<'PY2'
import sys, numpy as np
d = np.genfromtxt(sys.argv[1], delimiter=",", names=True, ndmin=1)
n = d["occ_eq"].size
print(f"samples={n}  occ_eq mean {d['occ_eq'].mean():.4f} max {d['occ_eq'].max():.4f}"
      f"  occ3 mean {d['occ3_eq'].mean():.4f}  local hours "
      f"{d['local_hour_box'].min():.2f}-{d['local_hour_box'].max():.2f}")
if n == 0:
    raise SystemExit("FATAL: the occupancy sampler wrote no rows")
PY2
[ $? -eq 0 ] || exit 1

echo "=== 3. STEADY-STATE throughput with the sampler ON (the arm's walltime rests on this) ==="
echo "    The csv rate is CUMULATIVE and so is swamped by the one-time compile;"
echo "    the incremental rate between two late rows is the number that matters."
$PY - $PRE/diag_timeseries.csv <<'PY2'
import sys, numpy as np
d = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
st, r = np.atleast_1d(d["step"]), np.atleast_1d(d["steps_per_s"])
ok = (st > 0) & (r > 0)
st, r = st[ok], r[ok]
if st.size < 2:
    raise SystemExit("FATAL: need >= 2 diagnostic rows to separate compile from stepping")
t = st / r                                    # wall seconds since the leg start
inc = np.diff(st) / np.diff(t)                # rate between consecutive rows
print("cumulative rate (compile included):", np.round(r, 3))
print("incremental rate per interval      :", np.round(inc, 3))
steady = float(np.median(inc[len(inc) // 2:])) # the late half, after compile
print(f"STEADY steps/s = {steady:.3f}  -> 15-day leg = {8640 / steady / 3600:.1f} h "
      f"(limit 13 h); control without the sampler ran 0.42-0.44")
if steady < 0.35:
    raise SystemExit(f"FATAL: steady {steady:.3f} < 0.35 steps/s -- raise "
                     "--evd-occupancy-every-hours before launching the arm")
print("throughput OK")
PY2
[ $? -eq 0 ] || exit 1
echo DONE
