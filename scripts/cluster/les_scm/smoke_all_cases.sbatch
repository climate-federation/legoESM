#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --job-name=les_smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_smoke_%j.out
set +e
# ---------------------------------------------------------------------------
# SMOKE for the LES -> SCM turbulence-tuning campaign.
#
# Purpose is NOT science. Two things only, both cheap:
#   1. Prove every LES driver still runs after the les_record flux change and
#      that the per-frame profile npz actually contains wth (all cases) and
#      qv/wqv (moist cases). Without this the production runs would burn their
#      GPU-hours producing a reference with no heat flux in it.
#   2. MEASURE seconds-per-simulated-hour per case at a known resolution, so
#      the production walltimes are sized from measurement rather than from a
#      template guess.
#
# Resolutions here are deliberately far below the published case setups; no
# number produced by this job is a physics result and nothing may cite it.
# ---------------------------------------------------------------------------
WT=/burg/glab/users/pg2328/les_scm_wt
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"; nvidia-smi -L
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
OUT=/burg-archive/glab/users/pg2328/legoESM/results/les_smoke
rm -rf $OUT; mkdir -p $OUT

run_case () {   # name, seconds-budget-label, command...
  local name=$1; shift
  echo "=============================================================="
  echo "CASE $name  start=$(date +%T)"
  local t0=$SECONDS
  "$@"
  local rc=$?
  echo "CASE $name  rc=$rc  wall=$((SECONDS - t0))s"
  return $rc
}

# --- moist cases (gSAM decks) ---------------------------------------------
run_case bomex $PY scripts/run/run_bomex_les.py \
  --nx 32 --ny 32 --nz 40 --Lx 3200 --Ly 3200 --Lz 3000 \
  --hours 0.25 --dt 1.0 --record-frames 3 --print-every 200 \
  --output $OUT/bomex

run_case rico $PY scripts/run/run_rico_les.py \
  --nx 32 --ny 32 --nz 40 --Lx 3200 --Ly 3200 --Lz 4000 \
  --hours 0.25 --dt 2.0 --record-frames 3 --print-every 200 \
  --output $OUT/rico

run_case dycoms $PY scripts/run/run_dycoms_les.py \
  --nx 32 --ny 32 --nz 64 --Lx 1120 --Ly 1120 --Lz 1500 \
  --hours 0.25 --dt 0.5 --record-frames 3 --print-every 400 \
  --output $OUT/dycoms

# --- dry PBL cases (analytic ICs) -----------------------------------------
run_case gabls1 $PY scripts/run/run_spectral_sbl.py \
  --nx 32 --ny 32 --nz 48 --Lx 400 --Ly 400 --Lz 400 \
  --hours 0.25 --dt 0.1 --record-frames 3 --print-every 2000 \
  --output $OUT/gabls1

run_case wangara $PY scripts/run/run_spectral_cbl.py \
  --nx 32 --ny 32 --nz 48 --Lx 3200 --Ly 3200 --Lz 1600 \
  --hours 0.25 --dt 0.5 --record-frames 3 --print-every 1000 \
  --output $OUT/wangara

# --- the actual assertion --------------------------------------------------
echo "=============================================================="
echo "VERIFY profile contents"
$PY - <<'PYEOF'
import sys
from pathlib import Path
import numpy as np

OUT = Path("/burg-archive/glab/users/pg2328/legoESM/results/les_smoke")
DRY = {"z", "theta", "wth", "uw", "vw", "tke", "u", "v"}
MOIST = DRY | {"qv", "wqv", "qc", "wqc", "cloud_frac"}
EXPECT = {"bomex": MOIST, "rico": MOIST, "dycoms": MOIST,
          "gabls1": DRY, "wangara": DRY}

bad = []
for case, want in EXPECT.items():
    frames = sorted((OUT / case / "profiles").glob("prof_*.npz"))
    if not frames:
        bad.append(f"{case}: NO profile frames written")
        continue
    with np.load(frames[-1], allow_pickle=True) as f:
        have = set(f.files)
        missing = want - have
        if missing:
            bad.append(f"{case}: missing {sorted(missing)}")
            continue
        wth = f["wth"]
        # A finite, not-identically-zero heat flux is the whole point.
        if not np.all(np.isfinite(wth)):
            bad.append(f"{case}: wth has non-finite values")
        elif np.allclose(wth, 0.0):
            bad.append(f"{case}: wth is identically zero (no resolved flux)")
        else:
            extra = f" wqv|max|={np.abs(f['wqv']).max():.3e}" if "wqv" in have else ""
            print(f"  {case:8s} OK  frames={len(frames)} "
                  f"nz={wth.size} wth|max|={np.abs(wth).max():.4e}{extra}")

if bad:
    print("SMOKE FAILED:")
    for b in bad:
        print("   " + b)
    sys.exit(1)
print("SMOKE PASSED: every case wrote finite, nonzero resolved fluxes")
PYEOF
echo "verify rc=$?"
echo "DONE $(date +%T)"
