#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=kappa90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/kappa90_%j.out
#
# ============================================================================
# NEMO's K-from-TKE AMPLITUDE at day 90 — the largest single closure defect.
# ============================================================================
#
# THE DEFECT.  NEMO tke_avn: zsqen = SQRT(en); zav = rn_ediff*zmxlm*zsqen;
# p_avm = MAX(zav, avmb)*wmask — the square root takes `en`, NOT `2*en`; the
# factor 2 lives in the LENGTH, zmxlm = SQRT(2*en/rn2) at zdftke.F90:651.
# (NEMO's own header comment writes that floor as `avtb`; the CODE uses `avmb`
# — codex 9400815 #1.)  legoESM's TKEConfig default `gaspar_sqrt2e` applies
# sqrt(2*e) in the AMPLITUDE as well, and the tke_mxl_choice 3/4 branch already
# builds its length from sqrt(2)*sqrt(e)/N (tke.py:698-702).  So every tripole
# and MPAS arm to date has DOUBLE-COUNTED the sqrt(2) on that branch.
#
# ALREADY MEASURED, on NEMO's own state, no integration (Stage A, 39ce0701c):
#     gaspar_sqrt2e (every prior arm)  K_H 4.048e-2 = 4.81x NEMO avt 8.410e-3
#     veros_sqrte   (NEMO's form)      K_H 2.730e-2 = 3.25x
# a 1.48x reduction against the 1.414 the algebra predicts.  The measurement
# was made last session and the CARD WAS NEVER FIXED; this run is the first at
# NEMO's amplitude.
#
# WHY IT SHOULD MOVE THE SOUTHERN OCEAN, and why that is a HYPOTHESIS and not
# a corollary.  At fixed e/l_k/N2/shear the pin divides K by sqrt(2); less
# tracer diffusivity means less entrainment at the mixed-layer base, and the
# day-90 Southern Ocean is 19 m TOO DEEP (mlefix_d90 89.6 m vs NEMO 70.4 m),
# i.e. we over-mix — the same direction as the Arctic halocline erosion.  BUT
# the prognostic TKE solve is COUPLED (codex 9400815 #3): lower K_M cuts shear
# production and reinforces the reduction, while lower K_H cuts stable
# buoyancy destruction and offsets it, and both feed back through N2, l_k and
# the Prandtl number.  The realized change in K need not be 1/1.414 and could
# locally reverse.  THIS RUN IS THE FALSIFIER; the algebra is not.
# It is still a much larger lever than
# the nn_mxl dissipation-length choice, which a same-day offline probe
# (job 9398998) showed is worth only ~1-3 m: nn_mxl already separates the two
# settings by a factor 2.7 in the top 20 m at DAY 30, where the measured
# mixed-layer difference is 0.9 m.  Arm 9398236 was cancelled for this one.
#
# CONTROL / ONE VARIABLE.  Command below is nemolev_trp_mlefix_d90's manifest
# command_line VERBATIM plus `--tke-kappa-convention veros_sqrte` (which is now
# also the card default; passed explicitly so the manifest records it) and a
# new --output.  The code revision differs from the control by this session's
# commit, but every other change in it is inert unless its flag is passed, so
# the pair is effectively one-variable — the same standing the MLE A/B had.
#
# PRE-REGISTERED METRIC: MLD bias, band antarctic_S_of_45S, day-90 snapshot vs
# NEMO month 3, compare_omip_nemo.py --res-deg 1.0 — the IDENTICAL protocol
# that produced the mlefix_d90 baseline of +19.2 m (corr 0.257).
#     CONFIRMS : bias <= +8 m
#     REFUTES  : bias >= +16 m  (the amplitude is not the Southern Ocean lever
#                either, and the next suspect is NEMO nn_mle=1's use of hmlef
#                rather than the turbulent hmld for the MLE streamfunction)
#     INDETERMINATE: +8 to +16 m — report the number, claim no direction.
#
# PRE-REGISTERED RISK, and it must be reported whatever the Southern Ocean
# does.  Cutting K by ~1.41x EVERYWHERE may reintroduce the tropical warm SST
# that switching to prognostic TKE removed (+0.55 -> +0.12 C in the d30 A/B).
# Watch, against the mlefix_d90 baseline: global SST rmse (0.63 C at m03),
# tropical SST bias, and tropics/NH-midlat MLD.  A Southern Ocean fix bought
# with a tropical regression is a trade, not a win, and gets reported as one.
# NEMO's amplitude is still NEMO's amplitude — if the tropics degrade, the
# compensating error is elsewhere and this arm has located it.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM
cd $ROOT || exit 1

# ---- preflight: refuse to run a 30 h job against an unpinned tree ----------
echo "HEAD:    $(git -C $ROOT rev-parse HEAD)"
echo "SUBJECT: $(git -C $ROOT log --oneline -1)"
git -C $ROOT status --porcelain -- scripts/run/run_omip_core2.py \
    packages/ocean/legoesm/ocean/physics/vertical_mixing/ > /tmp/kappa90_dirty.$$
if [ -s /tmp/kappa90_dirty.$$ ]; then
  echo "FATAL: model/driver paths are dirty at launch. Offending paths:"
  cat /tmp/kappa90_dirty.$$; rm -f /tmp/kappa90_dirty.$$; exit 1
fi
rm -f /tmp/kappa90_dirty.$$

export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
  --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
IN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS

XLA_PYTHON_CLIENT_PREALLOCATE=false LEGOESM_JIT_CACHE_DIR="" \
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  --grid tripole --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 --ew-cyclic-overlap --nemo-vertical \
  --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --momentum-rk3 \
  --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
  --prognostic-ice-dynamics free_drift --years 0.2466 --snapshot-every-days 30 \
  --diag-every-days 5 --tracer-advection superbee --runoff-depth-nemo-ini \
  --river-mouth-restoring-gate --bottom-drag-scheme nemo_quadratic \
  --iwm --iwm-forcing-file $IN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
  --dm2dc --isf --isf-forcing-file $IN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
  --bbl-adv --woa-init \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --nemo-monthly-init \
    $IN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
    $IN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --sss-restore-file $IN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --tripole-vmix tke --ice-init $IN/Ice_initialization.nc \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
  --tke-kappa-convention veros_sqrte \
  --sw-rgb-chl --ice-ocean-heat-coeff 70 --gateway-transports \
  --output $ROOT/results/omip_nemo/nemolev_trp_kappanemo_d90 \
  --mle --mle-ce 0.06
rc=$?
echo "=== run_omip_core2 rc=$rc ==="
exit $rc
