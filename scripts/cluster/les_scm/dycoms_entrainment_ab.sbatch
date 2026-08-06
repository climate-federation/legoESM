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
# Overridable so the job can be pinned to the SAME worktree a review saw.
# It previously hardcoded a different tree than the reviewed commit, so an
# arm could test other LES/SGS code while being cited as evidence.
WT=${AB_WT:-/burg/glab/users/pg2328/les_scm_wt}
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
  # 4 h, so the driver's second-half mean IS the published hours 2-4 window.
  # The previous round ran 2 h and therefore averaged hours 1-2 while comparing
  # against a 2-4 h target.
  # PIPESTATUS, not $?: piping through grep|tail records TAIL's status, so an
  # OOM, traceback or blow-up reported rc=0 and a green Slurm job.
  $PY scripts/run/run_dycoms_les.py --hours 4 --record-frames 24 \
      --print-every 8000 --output $OUT/$label "$@" 2>&1 \
    | grep -E "FINAL|BLOWUP|Error" | tail -3
  local rc=${PIPESTATUS[0]}
  echo "ARM $label rc=$rc wall=$((SECONDS - t0))s"
  [ "$rc" -ne 0 ] && echo "ARM $label FAILED -- do not cite this arm"
}

# Geometry and timestep are IDENTICAL in every arm. The previous round changed
# nx/ny/nz with Lx/Ly in one arm and nz with dt in another, so neither could be
# attributed. The theta budget showed the dominant warming is UNRESOLVED
# transport (+0.187 K/h, 123% of the total, against radiation reaching the
# mixed layer at only -0.013 K/h), so every arm here varies exactly one
# unresolved-transport setting.
G="--nx 96 --ny 96 --nz 192 --Lx 3360 --Ly 3360 --Lz 1500 --dt 0.5"

# ARM LABELS MUST NAME WHAT THE FLAGS ACTUALLY DO. The previous two SGS arms
# tested nothing they claimed:
#   "R_smagorinsky" passed --sgs-model smagorinsky WITHOUT --static. The dynamic
#     path ignores --sgs-model (run_dycoms_les.py:110-113), so that arm was a
#     bit-identical DUPLICATE of R_baseline reported as an SGS-model effect.
#   "R_static_cs" passed --static --cs 0.18, but 0.18 is already the --cs
#     DEFAULT and --sgs-model still defaulted to vreman. So it varied only
#     --static, and was a dynamic-vs-static test mislabelled as a Cs test.
# Each arm below differs from the one named in its comment by ONE flag.
arm R_baseline    $G                                     # dynamic Vreman (default)
arm R_static_vre  $G --static                            # vs baseline: dynamic -> static
arm R_static_smag $G --static --sgs-model smagorinsky    # vs R_static_vre: Vreman -> Smagorinsky
arm R_smag_cs10   $G --static --sgs-model smagorinsky --cs 0.10  # vs R_static_smag: Cs 0.18 -> 0.10
arm R_weno5       $G --scalar-advection weno5
arm R_nufloor     $G --nu-floor 0.5

echo "=============================================================="
echo "SUMMARY  ref: LWP 50-80 g/m2, cover ~1.0, z_i 840-870 m (hours 2-4)"
$PY /burg-archive/glab/users/pg2328/legoESM/scripts/validate/dycoms_theta_budget.py \
    $OUT/R_baseline 2>/dev/null | tail -3
for a in R_baseline R_static_vre R_static_smag R_smag_cs10 R_weno5 R_nufloor; do
  echo "--- $a"
  $PY /burg-archive/glab/users/pg2328/legoESM/scripts/validate/dycoms_theta_budget.py \
      $OUT/$a 2>/dev/null | tail -2
done
echo "DONE $(date +%T)"
