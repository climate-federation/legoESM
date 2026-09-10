#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=bclv30A
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/bclevel30_%j.out
#
# #1690 ACCEPTANCE A/B, arm A: tke_surface_bc_level = interior_pinned (control).
#
# ONE VARIABLE between the arms: --tke-surface-bc-level (absent = card default
# interior_pinned vs nemo_z0). Everything else is the canonical ORCA1 tripole
# card VERBATIM (from _ab_alphatke_d90), cut to d30 (--years 0.0822),
# snapshot at d30.
#
# CONTEXT. PR #1704 fixed the nemo_z0 face metric (was half e3t(1), doubling
# the surface coupling — measured ~2x near-surface TKE). The card had REVERTED
# nemo_z0 for exactly that defect; the mode is now selectable via the new
# flag, default unchanged. This pair is the measurement the owner asked for
# before deciding the card default (2026-09-04).
#
# PRE-REGISTERED READING (GLM's falsifier, recorded before the runs):
#   * PRIMARY: day-30 MLD vs the de Boyer Montegut (2022) climatology
#     (scripts/validate/compare_mld_dbm.py, mld_dr003_ref10m.nc staged).
#     nemo_z0 NOT WORSE globally + subpolar => fix acceptable for the card;
#     MLD bias WORSENS materially => the card default stays and the issue
#     records why.
#   * SECONDARY (direction, no gate): near-surface (5-26 m) equatorial shear
#     should INCREASE under nemo_z0 (the DINO EUC signature, PR #1689:
#     0.71 -> 0.97 of NEMO at d90).
#   * Expected NULL: gate-level global metrics barely move (issue's own
#     expectation; the change lives in the top 1-2 cells).
# d30 caveat: MLD in January is NH-winter-deepening dominated; the SH-winter
# map needs longer. Read the d30 pair directionally, both arms same window.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM_wt1690
cd $ROOT || exit 1

# ---- preflight: refuse to run a 30 h job against an unpinned tree ----------
echo "HEAD:    $(git -C $CODE_ROOT rev-parse HEAD)"
echo "SUBJECT: $(git -C $CODE_ROOT log --oneline -1)"
git -C $CODE_ROOT status --porcelain -- scripts/run/run_omip_core2.py \
    packages/ocean/legoesm/ocean/physics/vertical_mixing/ > /tmp/alpha90_dirty.$$
if [ -s /tmp/alpha90_dirty.$$ ]; then
  echo "FATAL: model/driver paths are dirty at launch. Offending paths:"
  cat /tmp/alpha90_dirty.$$; rm -f /tmp/alpha90_dirty.$$; exit 1
fi
rm -f /tmp/alpha90_dirty.$$

# data/ is gitignored so it lives ONLY in the main checkout, and the driver
# REQUIRES the mesh path to be the same string for the grid build and the
# mask/bathy read (it raises "tripole mesh mismatch" otherwise, job 9692794/5).
# So: run with CWD = the data root and the mesh path RELATIVE, while the CODE
# comes from this worktree via PYTHONPATH + an absolute script path. First
# attempt died on a relative path from the worktree (9657825/6); second on an
# absolute one that tripped the consistency guard.
DATA_ROOT=/burg-archive/glab/users/pg2328/legoESM
CODE_ROOT=$ROOT
[ -e "$DATA_ROOT/data/grids/eORCA1.2_mesh_mask.nc" ] || {
  echo "FATAL: mesh missing under DATA_ROOT=$DATA_ROOT"; exit 3; }
cd "$DATA_ROOT" || exit 2
export PYTHONPATH="$(ls -d $CODE_ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$CODE_ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
  --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
IN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS

XLA_PYTHON_CLIENT_PREALLOCATE=false LEGOESM_JIT_CACHE_DIR="" \
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u $CODE_ROOT/scripts/run/run_omip_core2.py \
  --grid tripole --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 --ew-cyclic-overlap --nemo-vertical \
  --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --momentum-rk3 \
  --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
  --prognostic-ice-dynamics free_drift --years 0.0822 --snapshot-every-days 30 \
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
  --output $CODE_ROOT/results/omip_nemo/nemolev_trp_bclevel_d30_pinned \
  --mle --mle-ce 0.06
rc=$?
echo "=== run_omip_core2 rc=$rc ==="
exit $rc
