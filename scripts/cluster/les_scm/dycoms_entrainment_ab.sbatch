#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --job-name=dycoms_ab
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/dycoms_ab_%j.out
set +e
# ---------------------------------------------------------------------------
# DYCOMS-II RF01 stratocumulus burn-off: controlled A/B.
#
# THE OBSERVATION. The cloud starts CORRECT and erodes:
#   0.56 h  cc=0.99  LWP=37.7   (ref cc~1.0, LWP 50-80)
#   1.11 h  cc=0.71  LWP=18.9
#   4.00 h  cc=0.36  LWP= 8.9
# while the 0.25 h smoke (nx=32, nz=64, Lx=1120) ends at LWP=47.1, cc=1.00 --
# i.e. "it worked before" is a 15-minute run, not a 4-hour one. No code change
# explains it: the only candidate (2c9ba9b61, weno5->van_leer on the tracer
# path) bites solely when scalar_advection is weno5, and DYCOMS uses van_leer.
# So this is physics: excessive cloud-top entrainment, the known RF01 failure
# mode (Stevens et al. 2005 found most LES entrain too much).
#
# WHAT EACH ARM ANSWERS (one variable each, all 2 h -- the failure is already
# unambiguous by 1.1 h, so 2 h is enough and 4 h would be waste):
#   A  smoke geometry, LONG   -> is it TIME or RESOLUTION/DOMAIN? If the coarse
#                               config also burns off, resolution is exonerated.
#   B  production geometry    -> the baseline being reproduced.
#   C  B minus Lilly SGS      -> CONTROL. Must be WORSE than B; if it is not,
#                               the stratification suppression is not doing
#                               anything and that is the bug.
#   D  finer inversion        -> dz 7.8 m -> 5.2 m (the RF01 spec is 5 m). Tests
#                               numerical entrainment across a razor inversion.
#                               Also tests the "nz<=200 compile cliff" claim in
#                               run_dycoms_les.py:92, which is prose.
#
# CONFIRMS a fix: LWP >= 40 g/m2 AND cloud cover >= 0.9 at 2 h.
# REFUTES:       LWP < 20 g/m2 OR cover < 0.6.
# No cheaper offline test exists -- cloud-top entrainment is LES dynamics.
# ---------------------------------------------------------------------------
WT=/burg/glab/users/pg2328/les_scm_wt
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"; nvidia-smi -L
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
OUT=/burg-archive/glab/users/pg2328/legoESM/results/dycoms_ab

arm () {   # label, then driver args
  local label=$1; shift
  echo "=============================================================="
  echo "ARM $label  $*"
  local t0=$SECONDS
  rm -rf $OUT/$label; mkdir -p $OUT/$label
  $PY scripts/run/run_dycoms_les.py --hours 2 --record-frames 12 \
      --print-every 4000 --output $OUT/$label "$@" 2>&1 \
    | grep -E "FINAL|BLOWUP|Error|h max\|w\|" | tail -6
  echo "ARM $label rc=$? wall=$((SECONDS - t0))s"
}

# A: smoke geometry, run LONG -> separates time from resolution/domain
arm A_coarse_long   --nx 32 --ny 32 --nz 64  --Lx 1120 --Ly 1120 --Lz 1500 --dt 0.5
# B: production geometry -> reproduce the baseline failure
arm B_baseline      --nx 96 --ny 96 --nz 192 --Lx 3360 --Ly 3360 --Lz 1500 --dt 0.5
# C: control on the Lilly stable-stratification SGS suppression
arm C_no_lilly      --nx 96 --ny 96 --nz 192 --Lx 3360 --Ly 3360 --Lz 1500 --dt 0.5 \
                    --no-sgs-buoyancy
# D: finer inversion, dz 7.8 -> 5.2 m (RF01 spec is 5 m)
arm D_nz288         --nx 96 --ny 96 --nz 288 --Lx 3360 --Ly 3360 --Lz 1500 --dt 0.35

echo "=============================================================="
echo "SUMMARY (ref: LWP 50-80 g/m2, cloud cover ~1.0, z_i 840-870 m)"
$PY - "$OUT" <<'PYEOF2'
import sys, glob
from pathlib import Path
import numpy as np
out = Path(sys.argv[1])
print(f"{'arm':16s} {'LWP g/m2':>9s} {'cover':>7s} {'z_i m':>7s}   verdict")
for a in ("A_coarse_long", "B_baseline", "C_no_lilly", "D_nz288"):
    f = out / a / "dycoms_les_final.npz"
    if not f.exists():
        print(f"{a:16s} {'-':>9s} {'-':>7s} {'-':>7s}   NO OUTPUT"); continue
    with np.load(f, allow_pickle=True) as d:
        lwp = float(d["lwp_timemean"]) if "lwp_timemean" in d.files else float(d["lwp"])
        cc = float(d["cloud_cover_timemean"]) if "cloud_cover_timemean" in d.files else float(d["cloud_cover"])
        zi = float(d["zi"]) if "zi" in d.files else float("nan")
    ok = "SOLVED" if (lwp >= 40.0 and cc >= 0.9) else (
         "burned off" if (lwp < 20.0 or cc < 0.6) else "partial")
    print(f"{a:16s} {lwp:9.1f} {cc:7.3f} {zi:7.0f}   {ok}")
PYEOF2
echo "DONE $(date +%T)"
