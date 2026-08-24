#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --job-name=crm_pilot
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/crm_pilot_%j.out
set +e
REPO=/burg-archive/glab/users/pg2328/legoESM
WT=/burg-archive/glab/users/pg2328/_wt_crmpilot_$SLURM_JOB_ID
cd "$REPO"
git worktree add --detach "$WT" origin/scm-rce-convection-campaign >/dev/null 2>&1
cd "$WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')

# 1 sim-day = 4320 steps at dt=20. Attempt 2 died at step 360 (2 h), so 1 day
# is ample to discriminate. ONE variable changes per arm.
#   A: reproduce attempt 1 short  (filter 0.0,  noise 0.5)
#   B: reproduce attempt 2        (filter 0.35, noise 0.2)
#   C: filter alone               (filter 0.35, noise 0.0)
#   D: noise alone, lowered       (filter 0.0,  noise 0.2)
#   E: Wing standard noise        (filter 0.0,  noise 0.1)
#   F: gentler filter             (filter 0.15, noise 0.2)
run_arm () {
  NAME=$1; NU=$2; AMP=$3
  echo "############ ARM $NAME  si_w_filter_nu=$NU  theta_noise_amp=$AMP"
  JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_rcemip_plane.py \
    --nx 128 --ny 128 --nlev 30 --dx 2000 --dt 20.0 \
    --steps 4320 \
    --semi-implicit --n-acoustic-substeps 12 --off-centering 0.2 \
    --smag-cs 0.2 --hyperdiff 5e6 \
    --radiation rrtmgp --radiation-interval 90 \
    --microphysics kessler --precision float64 \
    --theta-noise-amp $AMP --seed-kind band_noise --seed-kmax 6 \
    --si-w-filter-nu $NU \
    --print-every 360 \
    --output $WT/_pilot_$NAME 2>&1 | grep -E "^ *[0-9]+ |NON-FINITE" | tail -8
  echo "arm $NAME rc=${PIPESTATUS[0]}"
}
run_arm A 0.0  0.5
run_arm B 0.35 0.2
run_arm C 0.35 0.0
run_arm D 0.0  0.2
run_arm E 0.0  0.1
run_arm F 0.15 0.2
cd "$REPO"; git worktree remove --force "$WT" 2>/dev/null
echo ALLDONE
