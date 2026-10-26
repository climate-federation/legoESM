#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=nnmxl2_90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nnmxl2_90_%j.out
#
# ============================================================================
# NEMO nn_mxl=2 AT DAY 90 -- the Southern-Ocean autumn over-deepening arm.
# ============================================================================
#
# WHY THIS RUN EXISTS (fidelity, not tuning).  ORCA1's namelist_cfg sets
# `nn_mxl = 2` (EXP00/namelist_cfg:444, verified this session).  Our ORCA1
# zdftke card (run_omip_core2.orca1_zdftke_config) hardcodes
# `tke_mxl_choice=3` = NEMO nn_mxl=3, and EVERY d90 tripole arm to date ran
# choice 3.  `--tke-mxl-choice 4` is the choice that maps to the oracle's
# actual nn_mxl=2.  So this is running our model at the setting NEMO runs,
# which needs no hypothesis to justify.
#
# WHAT IT SHOULD FIX.  choice 3 and 4 share the lup/ldown sweeps and the SAME
# eddy-coefficient length l_k = min(lup,ldn); they differ ONLY in the
# dissipation length (tke.py:786-800): choice 3 uses l_eps = sqrt(lup*ldn),
# choice 4 uses l_eps = min(lup,ldn).  min <= sqrt(product), so choice 4
# dissipates MORE (eps = c_eps*e^{3/2}/l_eps) -> less TKE -> shallower ML.
#
# WHY IT IS SEASONALLY SELECTIVE, which is the whole point.  In a stably
# stratified summer boundary layer lup ~= ldn, so min ~= sqrt and the two
# choices coincide -- and the MEASURED day-30 pair says exactly that:
#     compare_nemolev_trp_mxl3ctl_d30_m01 (choice 3)  SO MLD 42.0  bias +2.3
#     compare_nemolev_trp_mxl4_d30_m01    (choice 4)  SO MLD 41.1  bias +1.4
# a 0.9 m separation.  In autumn convection lup is bounded by the distance to
# the surface while ldn runs to the pycnocline, so lup << ldn and the two
# lengths diverge by a factor of a few.  That is the regime the d90 arms sit
# in, and it is where every d90 arm is too deep:
#     compare_gwcorr_d90_m03 (MLE off)  SO MLD 93.9  NEMO 70.4  bias +23.5
#     compare_mlefix_d90_m03 (MLE on)   SO MLD 89.6  NEMO 70.4  bias +19.2
#     compare_mle_d90_m03    (+ real fw) SO MLD 85.3 NEMO 70.4  bias +14.9
# Fox-Kemper MLE moves it the right way but recovers only ~18% of the excess.
#
# CONTROL / ONE VARIABLE.  Command below is nemolev_trp_mlefix_d90's manifest
# command_line VERBATIM with `--tke-mxl-choice 3` -> `4` and a new --output.
# Nothing else changes.  mlefix_d90 (not mle_d90) is the base because mle_d90
# pairs `--freshwater-closure real_freshwater` with `--sss-restore`, which
# real_freshwater_restoring_conflict() REJECTS at HEAD unless
# `--sss-restore-channel water_flux` is also passed -- that would make this a
# three-variable change.  The water_flux closure gets its own later arm.
#
# PRE-REGISTERED METRIC: MLD bias, band antarctic_S_of_45S, day-90 snapshot vs
# NEMO month 3, scored with compare_omip_nemo.py at --res-deg 1.0 -- the
# IDENTICAL protocol that produced the mlefix_d90 baseline of +19.2 m.
#     CONFIRMS : bias <= +9 m   (more than half the excess removed)
#     REFUTES  : bias >= +16 m  (less than ~17% removed; nn_mxl is not the
#                Southern-Ocean autumn lever and the next suspect is the MLE
#                mixed-layer-depth choice, NEMO nn_mle=1 hmlef vs hmld)
#     INDETERMINATE: +9 to +16 m -- report the number, claim no direction.
# SECONDARY, no pre-registration, reported only: SO MLD spatial correlation
# (mlefix_d90 baseline 0.257; a mean-bias-only fix leaves it near 0.257, a
# structural fix should raise it toward the day-30 value of ~0.65), the
# day-30 snapshot bands (must NOT degrade from the measured +1.4 m), and
# global SST/SSS.
#
# CAVEAT TO CARRY WITH EVERY NUMBER (GLM-5.2, and it is right): both sides
# compare an INSTANTANEOUS snapshot to a NEMO MONTHLY MEAN.  That is a real
# protocol asymmetry, but it is identical in the baseline and in this arm, so
# the A/B separation is unaffected; only the absolute bias carries it.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM
cd $ROOT || exit 1

# ---- preflight: pin, and refuse to run against an unpinned tree ------------
echo "HEAD:    $(git -C $ROOT rev-parse HEAD)"
echo "SUBJECT: $(git -C $ROOT log --oneline -1)"
git -C $ROOT status --porcelain -- scripts/run/run_omip_core2.py \
    packages/ocean/legoesm/ocean/physics/vertical_mixing/ > /tmp/nnmxl2_dirty.$$
if [ -s /tmp/nnmxl2_dirty.$$ ]; then
  echo "FATAL: model/driver paths are dirty at launch -- a 30 h run on an"
  echo "unpinned tree is not reproducible.  Offending paths:"
  cat /tmp/nnmxl2_dirty.$$
  rm -f /tmp/nnmxl2_dirty.$$
  exit 1
fi
rm -f /tmp/nnmxl2_dirty.$$

export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
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
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 4 --tke-prognostic \
  --sw-rgb-chl --ice-ocean-heat-coeff 70 --gateway-transports \
  --output $ROOT/results/omip_nemo/nemolev_trp_nnmxl2_d90 \
  --mle --mle-ce 0.06
rc=$?
echo "=== run_omip_core2 rc=$rc ==="
exit $rc
