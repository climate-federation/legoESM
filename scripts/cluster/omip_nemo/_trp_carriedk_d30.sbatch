#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=trpcarrk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=9:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trpcarrk_%j.out
# ONE VARIABLE against trp_base2_d30: the TKE COEFFICIENT LIFETIME.
#   --tke-preclosure-coeff-source  current_subiteration (card default)
#                               -> carried_previous_step  (NEMO's avm_k/avt_k)
# Byte-identical to _trp_base2_d30.sbatch in every other flag, same two 15-day
# legs, same day-30 vs GATEWAY rec 5 scoring, so the comparison is controlled.
#
# WHAT THE FLAG CHANGES. NEMO's zdf_tke consumes the SAVE'd avm_k/avt_k pair in
# zdf_sh2, the Prandtl ratio, the TKE matrix and the RHS, and overwrites that
# pair only AFTER the solve (zdftke.F90).  Our default recomputes the pair
# inside the same call and evaluates shear production against it
# (tke.py:2746, 3086-3088: _K_M_pre is the carried value only under this
# option).  Codex named the consequence: a low recomputed coefficient makes low
# production, which makes a low coefficient.  The option is implemented, cold-
# start seeded from zdf_phy_init's background-times-wmask construction,
# restart-bridged, unit-tested, and selected by the DINO NEMO-oracle preset --
# it was simply never wired into orca1_zdftke_config.  User approved this arm
# 2026-09-10.
#
# WHY IT IS THE REMAINING CANDIDATE. The equatorial collapse is a PRODUCTION
# deficit, not a clamp: the floors are faithful and inactive (energy 283x above
# its floor, length 124x above its own), the backgrounds match by construction,
# and the Prandtl form is faithful.  Four hypotheses have been refuted by their
# own pre-registered arms -- enhanced vertical diffusion, the mixing length
# (the oracle's setting is the SHORTER one), the shear discretisation (1.22x
# where 10x was required), and advection (our 0-25 m budget closes and is
# 1.25-1.29x the oracle's, i.e. stronger).  Energy and length collapse together
# through the buoyancy length sqrt(2e)/N, which is why no single physics knob
# has moved it; this flag acts on the loop itself rather than on one term.
#
# PRE-REGISTERED, day 30 vs GATEWAY rec 5, written BEFORE the run:
#   PRIMARY   20-60 m tracer diffusivity rises from 2.60e-07 by at least 10x.
#             Below 2x = the coefficient lifetime is NOT the operator, refuted,
#             and the equatorial closure needs a different explanation.
#   CO-GATE   20-60 m Prandtl number falls from 11.4 toward the oracle's 1.2.
#             A primary hit WITHOUT this is suspect: deeper mixing lowers the
#             stratification, which lowers Ri and unclamps the tracer limiter,
#             so the gate can be passed with no real gain in turbulence.
#   SECONDARY undercurrent core at 220E rises from 0.20 m/s toward 0.55;
#             T(0.5)-T(20) in the cold tongue falls from 3.80 K.
#   GUARDRAIL SST rmse must not degrade by more than 0.05 from 0.51, nino3
#             stays <= +1.10, no polar SST regression beyond 0.1, max|u| < 2.5.
#             A primary hit bought with a guardrail breach is NOT an adoption.
#
# SCOPE, stated so the three-grid ask is not quietly broken: the carried
# memory lives on LatLonCGridOceanState only, so this row is tripole-today.
# MPAS and FESOM need the state fields before their cards can agree.
# 30 days as two 15-day legs (0.42 h/day; restart-save/from); leg 1 COPIES its
# final state to snapshot_day0015.npz (a symlink got overwritten by leg 2 once).
set -uo pipefail
PIN=${1:?usage: sbatch _trp_carriedk_d30.sbatch <pinned-sha> [leg: 1|2]}
LEG=${2:-1}
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/trp_carriedk_d30

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
  --tke-preclosure-coeff-source carried_previous_step \
  --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff 70 \
  --gateway-transports --mle --mle-ce 0.06 --kprofile-snapshots \
  --output $OUT 2>&1 | tail -25
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }

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
    --sss-faithful --grid-label trp_carriedk_d$1 \
    --output-dir $ROOT/results/omip_nemo/compare_trpcarriedk_d$1 2>&1 \
    | grep -E "nino3_5S5N|\[SST\].*rmse" | head -3
done

echo "=== undercurrent / thermocline vs GATEWAY (same instrument as job 9648737) ==="
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot $OUT/snapshot_day0030.npz --nemo-gridt "$GW_T" --nemo-t-recs 5:6 \
  --label trp_carriedk_d30 --nemo-wfile $GW/ORCA1_5d_20000101_20000331_grid_W.nc --nemo-w-var wo --nemo-w-recs 5:6 \
  --nemo-ufile $GW/ORCA1_5d_20000101_20000331_grid_U.nc --meridional-lon 240,260 \
  --nemo-vfile $GW/ORCA1_5d_20000101_20000331_grid_V.nc 2>&1 | grep -vE "Warning|warn" | tail -120
echo DONE
