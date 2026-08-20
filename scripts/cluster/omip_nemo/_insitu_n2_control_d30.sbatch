#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=insn2_30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/insn2_30_%j.out
#
# ============================================================================
# THE MATCHED insitu-N2 CONTROL for arm 9409057. Cloned from that arm's OWN
# sbatch so the ENVIRONMENT matches too, not just the command line -- the
# first two attempts (9412628, 9412863) were rebuilt from the manifest's
# recorded command_line and died in ~10 minutes with CUDA_ERROR_OUT_OF_MEMORY
# because they lacked this launcher's most-free-GPU selection and its
# XLA_PYTHON_CLIENT_PREALLOCATE=false. The launcher environment is part of
# the run; a control reconstructed from the command line alone is not a
# clone.
#
# WHY THIS RUN EXISTS AT ALL. The signed arm's header names job 9405182
# (nemolev_trp_alphatke_d90) as its control and states that n2_mode is the
# only behavioural difference. That is FALSE, checked field by field: 9405182
# also runs dissipation_discretization='backward_euler' where this card runs
# 'nemo_1p5_split'. Every other day-30 arm on disk is further away still,
# differing in alpha_tke AND kappa_convention AND n2_mode. There was no
# one-variable control anywhere, so the signed arm could not be attributed.
#
# THE ONE VARIABLE, verified from the two run manifests rather than asserted:
# 3 of 556 resolved-config leaves differ, and they are n2_mode, output_path
# and the config hash.
#
# PRE-REGISTERED, Southern Ocean mean MLD bias at day 30, signed minus this:
#   CONFIRMED : signed lands -8 to +8 m.
#   PARTIAL   : -15 to -8 m.
#   OVERSHOOT : beyond +8 m.
#   REFUTED   : still below -15 m.
# COUNTER-PREDICTION already on record: calm-column Arctic K went 1.37x ->
# 1.80x NEMO, so Arctic SSS should get slightly WORSE. If it IMPROVES,
# calm-column K is not the controlling term for the halocline.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM
cd $ROOT || exit 1

# ---- preflight: refuse to run a 30 h job against an unpinned tree ----------
echo "HEAD:    $(git -C $ROOT rev-parse HEAD)"
echo "SUBJECT: $(git -C $ROOT log --oneline -1)"
git -C $ROOT status --porcelain -- scripts/run/run_omip_core2.py \
    packages/ocean/legoesm/ocean/physics/vertical_mixing/ > /tmp/sgn30_dirty.$$
if [ -s /tmp/sgn30_dirty.$$ ]; then
  echo "FATAL: model/driver paths are dirty at launch. Offending paths:"
  cat /tmp/sgn30_dirty.$$; rm -f /tmp/sgn30_dirty.$$; exit 1
fi
rm -f /tmp/sgn30_dirty.$$

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
  --prognostic-ice-dynamics free_drift --years 0.0822 --snapshot-every-days 10 \
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
  --tke-n2-mode insitu \
  --output $ROOT/results/omip_nemo/nemolev_trp_insitun2_d30 \
  --mle --mle-ce 0.06
rc=$?
echo "=== run_omip_core2 rc=$rc ==="
exit $rc
