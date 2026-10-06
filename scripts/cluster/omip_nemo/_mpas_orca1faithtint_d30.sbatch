#!/bin/bash
#SBATCH --job-name=mpasharm
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:55:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mpasharm_%j.out
#
# MPAS, HARMONIZED TO THE TRIPOLE PRODUCTION CARD (user, 2026-09-16:
# "FESOM and MPAS should have same physics as tripolar").
#
# Built by copying _mpas_base2_d30.sbatch and adding every flag the
# production tripole card carries THAT THE MPAS LANE ACTUALLY THREADS.
# The test for "actually threads" was the driver call site, not the help
# text: a flag argparse accepts but the MPAS builder never receives is a
# silent drop, and adding it here would make the card lie.
#
# ADDED vs _mpas_base2_d30 (each verified wired for --grid mpas):
#   --nemo-init-tint       NEMO's 1-Jan IC = 0.5*Dec + 0.5*Jan (grid-agnostic,
#                          applied after the build at load_nemo_monthly_init_ts).
#                          This is the lever that took the tripole's day-30 MLD
#                          rmse from 21.0 to 16.4 m, and MPAS never had it.
#   --nemo-domain-mask     wet domain = the oracle's domain_cfg (read_mesh_mask_bathy
#                          takes nemo_domain_cfg on this lane too).
#   --pgf-scheme smc03     the tripole runs smc03; the driver default is adcroft,
#                          so base2 was running a DIFFERENT pressure gradient.
#   --ice-freeze-scheme unesco, --ice-exchange nemo_si3, --ice-snow bulk,
#   --ice-snow-flooding on the adopted SI3 exchange set (resolved grid-agnostically
#                          in main() and applied to the ice config both lanes use).
#   --forcing-time-interp linear  (host forcing loop).
#   --ew-cyclic-overlap    codex 9815604 caught this one: it is NOT a structured-
#                          grid seam flag. It runs the tripole's own
#                          _ew_overlap_fill on the eORCA1 SOURCE mask and
#                          bathymetry BEFORE they are regridded onto the Voronoi
#                          cells, so without it the MPAS run inherits a different
#                          coastline and depth field than the tripole from the
#                          same file.
#   snapshot/diag cadence  5 d / 1 d, the tripole's, so the two runs can be
#                          paired day-for-day.
#
# NOT ADDED, AND THESE ARE THE REMAINING PHYSICS GAPS vs the tripole:
#   * the GM/Redi Treguier set (--gm-treguier, --gm-kappa-min, --gm-slope-scheme,
#     --gm-slope-positions, --gm-bolus-advection, --gm-msc-stabilize,
#     --redi-coefficient nemo21, --redi-aht0 900) is HARD-REFUSED off the tripole
#     lane -- the override lives inside build_tripole. MPAS therefore runs its own
#     default GM/Redi, not NEMO's. Closing this is a code port, not a card edit.
#   * the lateral-viscosity set. --A-h and --C-smag-lap ARE threaded into
#     build_mpas_ocean (codex 9815604), but their companions are not: the
#     tripole's A_h is a 3-D eddy-viscosity FIELD (--A-h-profile-file) ramped by
#     --visc-schedule (x3 for 8 days, x1.5 to day 16, x0.5 after), and
#     --smag-cfl-safety has no MPAS path. Its effective viscosity is therefore
#     nowhere equal to the scalar 2e4, so carrying "--A-h 2e4 --C-smag-lap 3.0"
#     across would be a NEW viscosity choice dressed as harmonization, on a mesh
#     whose defaults were tuned for it. Left at the MPAS defaults, deliberately.
#   * --min-levels, --momentum-rk3, --adaptive-implicit-vertadv: accepted by
#     argparse but never passed to build_mpas_ocean -> silent drops. Omitted
#     deliberately.
#   * --chl-remap nemo_scrip. Carried in the first version of this card and it
#     killed the run at 4 minutes (job 9815673): the oracle's SCRIP weights
#     encode the eORCA1 DESTINATION grid, so the loader refuses any target that
#     is not (332, 362) -- it is tripole-only in effect, whatever the flag gate
#     accepts. Neither reviewer caught it; the driver's own guard did. The
#     chlorophyll field therefore reaches MPAS through the inverse-distance
#     remap, which is a real and unclosable difference from the tripole until
#     weights are generated for the Voronoi mesh.
#
# PRE-REGISTERED (day 30 vs GATEWAY record 5, against mpas_base2_d30's
# SST 0.4523 / SSS 0.3246 / MLD 23.4561 on the three-grid cell set):
#   CONFIRMS harmonization helps if MLD rmse drops below 20 m (the tripole's
#   IC blend alone bought 4.6 m there); REFUTES it if MLD rmse stays within
#   1 m of 23.5 or any field degrades by more than its own 0.05.
set -uo pipefail
PIN=${1:?usage: sbatch _mpas_orca1faithtint_d30.sbatch <pinned-sha>}
ROOT=${ROOT:-/burg-archive/glab/users/pg2328/_omip_run3}; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
# --years is the ABSOLUTE target, so both legs ask for the full DAYS and leg 2
# simply resumes from leg 1's restart if the 12 h partition cap bites (base2
# ran 30 days in ~9.5 h, and this card adds daily diagnostics).
DAYS=${DAYS:-30}
LEG=${LEG:-1}
YEARS=$($PY -c "print(${DAYS}/365.0)")
OUT=${OUT:-/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/mpas_orca1faithtint_d${DAYS}}
echo "DAYS=$DAYS LEG=$LEG YEARS=$YEARS OUT=$OUT"

# Ginsburg's cgroup does not hide the other GPUs on a shared node, so JAX takes
# device 0 whether or not another job is already on it. Job 9815701 died that
# way: it compiled beside a 2-hour tenant on g098 and asked for 33 GiB that were
# not there. Pick the emptiest visible GPU before JAX initialises.
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | tr '\n' ' ' | sed 's/^/[gpu] free MiB by index: /'; echo
export CUDA_VISIBLE_DEVICES=$(
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
echo "[gpu] using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# One MPAS step at ico7 x 75 levels peaks at 38 GiB after rematerialization, and
# JAX claims only 75% of the 45 GiB card by default, so the step does not fit in
# the pool even on an EMPTY GPU (job 9815772: both devices idle, still out of
# memory asking for 33.34 GiB). Raising the fraction to 0.95 made it WORSE
# (9815918): the pool then leaves nothing outside itself and CUDA cannot even
# load the compiled kernel ("Failed to load in-memory CUBIN"). On-demand
# allocation avoids both walls. Neither setting changes any physics.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PREALLOCATE:-false}
echo "[gpu] XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"

echo "=== gate: corrected forcing is the default ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_core2_corrected_cache_default.py 2>&1 | tail -2
rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] || { echo "FATAL: gate rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid mpas \
  --mpas-level 7 --mpas-lloyd 20 --mpas-vmix tke --K-zeta-bih 0 \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
  --tke-kappa-convention veros_sqrte \
  --dt 150 --nemo-vertical --partial-cell --pgf-scheme ${PGF:-smc03} \
  --ew-cyclic-overlap \
  --mle --mle-ce 0.06 \
  --dm2dc --sw-rgb-chl \
  --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
  --prognostic-ice-dynamics free_drift --ice-lead-freeze-source nemo_qlead \
  --isf --isf-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
  --iwm --iwm-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
  --iwm-n2-mode nemo_bn2 --iwm-n2-eos-form teos10 `# user 2026-10-06: NEMO zdfiwm reads bn2 (zdfiwm.F90:185-211), stated explicitly` \
  --bbl-adv --years $YEARS --snapshot-every-days 5 --diag-every-days 1 \
  --restart-save $OUT/restart_leg$LEG.npz \
  $([ "$LEG" = 2 ] && echo "--restart-from $OUT/restart_leg1.npz") \
  --tracer-advection superbee --runoff-depth-nemo-ini \
  --river-mouth-restoring-gate --bottom-drag-scheme nemo_quadratic \
  --ice-ocean-heat-coeff 70 \
  --woa-init \
  --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
    $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --nemo-init-tint \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --ice-init $NEMOIN/Ice_initialization.nc \
  `# ---- the adopted sets the MPAS lane threads ----` \
  --ice-freeze-scheme unesco \
  --ice-exchange nemo_si3 --ice-snow bulk --ice-snow-flooding on \
  --forcing-time-interp linear \
  --nemo-domain-mask \
  --output $OUT 2>&1 | tail -30
run_rc=${PIPESTATUS[0]}
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }
echo "OUT=$OUT"
echo DONE
