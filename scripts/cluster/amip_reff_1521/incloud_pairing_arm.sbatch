#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=icpair1521
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1521/icpair_%j.out
#
# #1521: how much shortwave does the droplet-number / condensate pairing
# error actually cost?
#
# THE CLAIM UNDER TEST (reviewed by GLM before this card was written):
# the PSD radius goes as (q_c/N_c)^(1/3), so the two must share a footing.
# compute_cloud_properties reconstructs in-cloud condensate ONLY when the
# droplet number is dead. Production sets aerosol_ccn, whose AOD-derived
# number is an AMBIENT (in-cloud) concentration but is ~1e8 and so takes the
# grid-mean branch. The radius is therefore too small by cf^(1/3).
#
# PRE-REGISTERED READING, written before the run:
#   * the number this produces is the rsut delta of the `in-cloud` arm
#     against the `PSD(ckpt)` baseline, W/m2, same checkpoint and same
#     8-point diurnal quadrature as the published sweep (job 9650438), so
#     the rows are directly comparable to it.
#   * CONFIRMS a material contribution: |delta| >= 2 W/m2.
#   * REFUTES it as a magnitude lever: |delta| < 1 W/m2. The pairing would
#     still be a correctness defect worth fixing, but it would stop being an
#     answer to the +36.6 W/m2 and I would say so.
#   * BETWEEN 1 and 2: real but minor; report as such, do not round up.
#
# WHY A CHEAPER TEST CANNOT ANSWER IT: the radius-to-flux mapping is
# nonlinear and column-dependent (the sweep itself measured d(rsut)/d(r_eff)
# falling from -3.48 to -2.32 W/m2/um between 8 and 14 um), and GLM's
# specific warning is that a global-mean cube root MISSTATES the answer
# because cloud fraction is bimodal. Only the kernel, per column, settles it.
#
# INSTRUMENT CONTROL, printed by the probe itself: where the layer is
# overcast the correction must be exactly the identity. A ratio that is not
# 1.000000 there means the arm is measuring its own arithmetic and the
# number must be discarded.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM_wt1521b
cd $ROOT || exit 1
echo "HEAD:    $(git -C $ROOT rev-parse HEAD)"
echo "SUBJECT: $(git -C $ROOT log --oneline -1)"
git -C $ROOT status --porcelain -- scripts/validate/amip_bias/reff_rsut_sensitivity.py \
    packages/atmosphere/legoesm/atmosphere/physics/clouds/ > /tmp/icpair_dirty.$$
if [ -s /tmp/icpair_dirty.$$ ]; then
  echo "FATAL: probe or cloud physics dirty at launch:"; cat /tmp/icpair_dirty.$$
  rm -f /tmp/icpair_dirty.$$; exit 1
fi
rm -f /tmp/icpair_dirty.$$

PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
# Never override SLURM's device: two co-scheduled jobs on one node both see
# every GPU here and a free-memory tie-break sends both to the same one
# (#1690 arms 9692961/2, control died OOM).
echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

CKPT=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1354/energy_12d_res5_intervalmean/checkpoint_day_0012.npz
# Venue is a knob only because the GPU partition can be blocked for hours by
# this account's own long ocean runs, and the measurement is a DIFFERENCE
# between arms computed identically -- so it does not care which backend it
# runs on. The probe reprints the whole published table, so a CPU run also
# re-derives the GPU rows and doubles as a backend cross-check.
JAX_PLATFORMS=${ICPAIR_PLATFORM:-cuda} JAX_ENABLE_X64=1 $PY -u \
  $ROOT/scripts/validate/amip_bias/reff_rsut_sensitivity.py "$CKPT" \
  --mesh-level 5 --n-times 8
echo "rc=$?"
