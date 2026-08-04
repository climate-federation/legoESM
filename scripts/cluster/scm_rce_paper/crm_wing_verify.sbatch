#!/bin/bash
# Verification pass for the Wing-2018 seed + RCEMIP reference fetch + the
# column-budget validator.  CPU only, no model integration.
#
# Runs from its OWN pinned worktree so an edit in a shared tree cannot change
# what this job executed (the anti-confound rule).  Echoes the pinned SHA first.
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=crm_wingverify
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/crm_wingverify_%j.out
set +e

BRANCH="${BRANCH:-crm/rcemip-resolved-convection}"
WT="${WT:-/burg-archive/glab/users/pg2328/_wt_crmverify_$SLURM_JOB_ID}"
SRC=/burg-archive/glab/users/pg2328/legoESM
RUN80=/burg-archive/glab/users/pg2328/legoESM/results/rcemip1_n128_ocean

git -C "$SRC" worktree add --detach "$WT" "origin/$BRANCH" >/dev/null 2>&1
cd "$WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"
echo "branch: origin/$BRANCH"
echo "host: $(hostname)"
echo "worktree: $WT"

export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo
echo "################ 1. UNIT TESTS (x64 -- numerics rule) ################"
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu $PY -m pytest -q \
  tests/unit/test_sam_case_setup.py \
  tests/unit/test_fetch_rcemip_reference.py \
  tests/unit/test_rcemip_column_budget.py \
  2>&1 | tail -30
echo "tests rc=$?"

echo
echo "############ 2. NON-VACUITY: seed dispatch must REJECT junk ##########"
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu $PY - <<'PYEOF' 2>&1 | tail -20
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("scripts/run").resolve()))
import jax.numpy as jnp
from legoesm.atmosphere.dynamics.crm.sam_case_setup import (
    wing2018_seed_amplitudes, wing2018_thermal_noise_seed)
print("wing amps (protocol):", wing2018_seed_amplitudes())
s = wing2018_thermal_noise_seed(8, 8, 12, rng_seed=0)
import numpy as np
s = np.asarray(s)
print("seeded levels (nonzero):", [k for k in range(12) if np.any(s[..., k] != 0)])
print("per-layer max|theta'| lowest->up:",
      [float(np.abs(s[..., -1 - j]).max()) for j in range(5)])
import run_rcemip_plane as rcp
try:
    rcp.parse_args(["--seed-kind", "garbage"])
    print("DISPATCH-GUARD FAIL: junk seed-kind accepted")
except SystemExit:
    print("DISPATCH-GUARD OK: argparse rejects --seed-kind garbage")
try:
    wing2018_seed_amplitudes(amp_lowest=0.0)
    print("ZERO-AMP GUARD FAIL: silent no-op seed accepted")
except ValueError as e:
    print("ZERO-AMP GUARD OK:", str(e).split(".")[0])
PYEOF

echo
echo "########## 3. COLUMN BUDGET of the existing 80-day run ###############"
echo "  (run 9285110: 128x128 dx=2km nlev=30 UNIFORM dz=1100m, rrtmgp+kessler,"
echo "   --theta-noise-amp 0.5 --seed-kind band_noise, 80 sim-days)"
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu $PY scripts/validate/rcemip_column_budget.py \
  --run-dir "$RUN80" --sfc-stride 16 2>&1 | tail -80
echo "budget rc=$?"

echo
echo "########## 4. RCEMIP community reference: SIZE CHECK #################"
$PY scripts/data/fetch_rcemip_reference.py --check-size \
  --model SAM_CRM --case RCE_small300 2>&1 | tail -20

echo
echo "########## 5. RCEMIP community reference: FETCH + CONVERT ############"
$PY scripts/data/fetch_rcemip_reference.py \
  --model SAM_CRM --case RCE_small300 \
  --cache /burg-archive/glab/users/pg2328/legoESM/data/rcemip_raw \
  --dest /burg-archive/glab/users/pg2328/legoESM/results/rcemip_ref_sam300 \
  --last-days 25 --n-snapshots 10 2>&1 | tail -30
echo "fetch rc=$?"

echo
echo "--- reference dir contents ---"
ls -la /burg-archive/glab/users/pg2328/legoESM/results/rcemip_ref_sam300/snapshots3d/ 2>&1 | tail -12
echo "vol count: $(ls /burg-archive/glab/users/pg2328/legoESM/results/rcemip_ref_sam300/snapshots3d/vol_*.npz 2>/dev/null | wc -l)"

git -C "$SRC" worktree remove --force "$WT" >/dev/null 2>&1
echo "ALLDONE"
