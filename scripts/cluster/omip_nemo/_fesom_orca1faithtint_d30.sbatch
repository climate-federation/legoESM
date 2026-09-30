#!/bin/bash
#SBATCH --job-name=fesomharm
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --constraint="a40|rtx8000"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesomharm_%j.out
#
# FESOM2, HARMONIZED TO THE TRIPOLE PRODUCTION CARD (user, 2026-09-16:
# "FESOM and MPAS should have same physics as tripolar").
#
# Built by copying _fesom_base2_d30.sbatch and adding every production-card
# flag the FESOM lane wires.  "Wires" is not a judgement here: the lane
# carries an explicit allowlist (_FESOM_WIRED_DESTS) and hard-rejects any
# user-set selector outside it, so the harmonized subset is exactly the
# intersection of the production card with that set.
#
# ADDED vs _fesom_base2_d30:
#   --nemo-init-tint      NEMO's 1-Jan IC = 0.5*Dec + 0.5*Jan.  This flag was
#                         already threaded into build_fesom_ocean but was MISSING
#                         from the allowlist until ea58d2473, so it was
#                         unreachable on this lane; it is the lever that took the
#                         tripole's day-30 MLD rmse from 21.0 to 16.4 m.
#   --ice-exchange nemo_si3          the adopted SI3 exchange set.
#   --forcing-time-interp linear     NEMO's fldread ln_tint on the forcing too.
#   --woa-t/--woa-s                  the production card's WOCE files, explicitly.
#   --snapshot-every-days 5          the tripole's cadence, so the runs pair
#                                    day-for-day.
#
# NOT ADDED, AND THESE ARE THE REMAINING PHYSICS GAPS vs the tripole -- every
# one of them is rejected by the stage-B4 allowlist, i.e. the lane does not
# wire it, so closing any of them is lane work, not a card edit:
#   GM/Redi (the whole Treguier + nemo21 set), --ice-freeze-scheme,
#   --ice-snow/--ice-snow-flooding, --geothermal, --isf, --bbl-adv,
#   --bottom-drag-scheme, --tracer-advection, --runoff-depth-nemo-ini,
#   --mle, --min-levels, --pgf-scheme, --momentum-rk3,
#   --adaptive-implicit-vertadv, the lateral-viscosity set.
# And one more that the ALLOWLIST LETS THROUGH but the code refuses at use:
#   --chl-remap nemo_scrip. It is in _FESOM_WIRED_DESTS, so the stage gate
#   accepts it, and the first version of this card carried it -- the run died
#   at 4 minutes (job 9815674, and MPAS the same way in 9815673) because the
#   oracle's SCRIP weights encode the eORCA1 DESTINATION grid and the loader
#   refuses any target that is not (332, 362). It is tripole-only in effect.
#   Neither reviewer caught it; the driver's own guard did, which is the
#   argument for guards that name the shape they need. Chlorophyll therefore
#   reaches FESOM through the inverse-distance remap.
#
# Three further differences are structural rather than missing:
#   * dt stays 1800 s (tripole and MPAS run 150 s).  --dt IS allowlisted, so
#     this is a COST choice, not a wiring limit: 150 s would be a 12x longer
#     run and does not fit the partition.
#   * SSS restoring stays on the water_flux channel with live_s normalization
#     -- the tracer channel cannot write into the read-only FESOM state facade,
#     and the driver refuses it.  The tripole uses the tracer channel.
#   * --nemo-domain-mask is a no-op here: the wet domain is the mesh's.
#
# PRE-REGISTERED (day 30 vs GATEWAY record 5, against fesom_base2_d30's
# SST 0.5277 / SSS 0.5078 / MLD 27.1172 on the three-grid cell set):
#   CONFIRMS harmonization helps if SSS rmse drops below 0.45 (the IC blend is
#   a salinity lever first) and MLD rmse below 25 m; REFUTES it if both stay
#   within 0.02 / 1 m of base2.
set -uo pipefail
PIN=${1:?usage: sbatch _fesom_orca1faithtint_d30.sbatch <pinned-sha>}
ROOT=${ROOT:-/burg-archive/glab/users/pg2328/_omip_run3}; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
echo "fesom_jax: $(git -C /burg-archive/glab/users/pg2328/fesom_jax_repo log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')/burg-archive/glab/users/pg2328/fesom_jax_repo:$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=${OUT:-/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/fesom_orca1faithtint_d30}
echo "OUT=$OUT"

# Ginsburg's cgroup does not hide the other GPUs on a shared node, so JAX takes
# device 0 whether or not another job is already on it (this is how the MPAS arm
# 9815701 ran out of memory beside another tenant). Pick the emptiest one first.
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | tr '\n' ' ' | sed 's/^/[gpu] free MiB by index: /'; echo
export CUDA_VISIBLE_DEVICES=$(
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
echo "[gpu] using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid fesom \
  --fesom-mesh-dir "$ROOT/data/fesom2_core2/mesh_nemo75_30m" \
  --woa-init \
  --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
    $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --nemo-init-tint \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --dm2dc --sw-rgb-chl \
  --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --ice-lead-freeze-source nemo_qlead \
  --fesom-vmix legoesm_tke --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 \
  --tke-prognostic --tke-kappa-convention veros_sqrte \
  --ice-init $NEMOIN/Ice_initialization.nc --ice-ocean-heat-scheme nemo_ustar \
  --sss-restore --sss-restore-channel water_flux \
  --sss-restore-normalization live_s \
  --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --runoff --river-mouth-restoring-gate \
  --iwm --iwm-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
  --forcing-ramp-days 0 `# user 2026-09-27: no ramp, as on tripole/MPAS and in NEMO` \
  `# ---- the adopted sets the FESOM lane wires ----` \
  --ice-exchange nemo_si3 \
  --forcing-time-interp linear \
  --dt 1800 --years ${YEARS:-0.0822} --snapshot-every-days 5 ${EXTRA:-} \
  --state-accumulate --mld-accumulate `# 5-day window means, as NEMO writes and the tripole card scores` \
  --output $OUT 2>&1 | tail -20
run_rc=${PIPESTATUS[0]}
echo "=== run rc=$run_rc"
[ "$run_rc" -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }
echo "OUT=$OUT"
echo DONE
