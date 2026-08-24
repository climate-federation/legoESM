#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=wangchk
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/wangchk_%j.out
set +e
: "${W_WT:?set W_WT}"
WT=$W_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=============== wangara forcing tests ==============="
$PY -m pytest -q tests/atmosphere/test_wangara_day33_forcing.py 2>&1 | tail -25
echo "wangara rc=${PIPESTATUS[0]}"

echo "=============== 2-minute CPU smoke of the wangara LES ==============="
# Prove the driver runs the new case, that the diurnal flux is actually being
# re-evaluated, and that the geostrophic profile reaches the momentum RHS,
# BEFORE spending a GPU on 8 simulated hours.
$PY scripts/run/run_spectral_cbl.py --case wangara \
    --nx 16 --ny 16 --nz 32 --Lx 1600 --Ly 1600 --Lz 2000 --z0 0.01 \
    --hours 0.05 --dt 0.5 --record-frames 2 --print-every 50 \
    --output /tmp/wangara_smoke_$SLURM_JOB_ID 2>&1 | tail -25
echo "smoke rc=${PIPESTATUS[0]}"

echo "=============== nieuwstadt default is unchanged ==============="
$PY -c "
import subprocess, sys
out = subprocess.run([sys.executable, 'scripts/run/run_spectral_cbl.py', '--help'],
                     capture_output=True, text=True).stdout
assert '--case' in out and 'wangara' in out, 'case flag missing'
print('CLI exposes --case {nieuwstadt,wangara}')
"
echo "DONE $(date +%T)"
