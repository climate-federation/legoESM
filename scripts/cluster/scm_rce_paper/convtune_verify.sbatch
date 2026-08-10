#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=convtune_vfy
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/convtune_vfy_%j.out
set +e
# ---------------------------------------------------------------------------
# Tree verification for the merged convection-tuning campaign branch.
#
# Answers three questions BEFORE any production arm is submitted:
#   1. does the merged tree import and do its own gates pass?
#   2. is the IFS supersaturation cap actually reachable in all three lanes
#      (CRM plane driver / SCM campaign / global atmospheric model)?
#   3. what does one SCM-RCE evaluation cost in wall time and peak RSS, with
#      the JIT-cache fix in place?  (Jobs 9331806/9331807 died at rc=134 with
#      "LLVM ERROR: Unable to allocate section memory" after ~40 evals.)
#
# Nothing here is a production measurement; it sizes the production arms.
# ---------------------------------------------------------------------------
WT=/burg-archive/glab/users/pg2328/_wt_convtune_pin
# PIN is exported by the launch command (sbatch --export=ALL,PIN=<sha>), not
# hardcoded: the SHA of a commit cannot be known inside the commit itself,
# and a self-derived $(git rev-parse HEAD) would make the check vacuous.
PIN=${PIN:?export PIN=<sha> so the job asserts the tree it was reviewed at}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=$REPO/results/rcemip_ref_sam300
OUT=$REPO/results/scm_rce_convtune/_verify

cd $WT || exit 90
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)  date: $(date -Is)"

# --- preflight: the tree under test IS the tree that was reviewed ----------
HEAD_SHA=$(git rev-parse HEAD)
if [ "$HEAD_SHA" != "$PIN" ]; then
  echo "PREFLIGHT FAIL: worktree at $HEAD_SHA, expected $PIN"; exit 91
fi
for f in scripts/run/run_scm_rce_convection_intercomparison.py \
         scripts/run/run_scm_rce_campaign.py \
         scripts/run/run_rcemip_plane.py \
         packages/core/legoesm/thermo.py ; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "PREFLIGHT FAIL: $f not tracked"; exit 92; }
  [ -f "$f" ] || { echo "PREFLIGHT FAIL: $f missing"; exit 93; }
done
[ -d "$REF/snapshots3d" ] || { echo "PREFLIGHT FAIL: reference $REF has no snapshots3d"; exit 94; }
echo "REFERENCE=$REF  ($(ls $REF/snapshots3d | wc -l) volume files, $(ls $REF/snapshots | wc -l) surface files)"
head -1 $REF/SOURCE.txt

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
# Persistent JIT cache OFF: the campaign's own logs showed XLA:CPU AOT results
# being loaded on a node whose CPU features did not match the compiling node
# ("could lead to execution errors such as SIGILL").  A shared cache across
# heterogeneous burst nodes is not worth that risk here.
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo; echo "############ 1. unit gates ############"
JAX_ENABLE_X64=1 $PY -m pytest -q -p no:cacheprovider \
  tests/unit/test_rcemip_wing_ic.py \
  tests/unit/test_homogeneous_freezing_rh.py \
  tests/unit/test_rcemip_plane_hard_sat_drain.py \
  tests/unit/test_run_rcemip_plane_hard_sat_cli.py \
  tests/unit/test_run_rcemip_plane_footguns.py \
  tests/unit/test_scm_rce_convection_intercomparison_cli.py \
  tests/test_microphysics_supersaturation_guard.py \
  tests/unit/test_supersaturation_guard_behaviour.py \
  tests/unit/test_check_ifs_supersaturation_cap.py \
  2>&1 | tail -25
echo "GATES_RC=${PIPESTATUS[0]}"

echo; echo "############ 2. IFS cap reachability in all three lanes ############"
JAX_ENABLE_X64=1 $PY $WT/scripts/validate/check_ifs_supersaturation_cap.py 2>&1 | tail -60
echo "CAP_RC=${PIPESTATUS[0]}"

echo; echo "############ 3. cost probe: 1 scheme, short SCM, morrison + guard ############"
mkdir -p $OUT
/usr/bin/time -v env JAX_ENABLE_X64=1 $PY \
  $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
  --reference-dir $REF \
  --outdir $OUT \
  --schemes bechtold \
  --subsidence-solve implicit_flux \
  --microphysics morrison \
  --hard-saturation-adjustment \
  --days 2 \
  --dt 600 \
  --analysis-days 1 \
  --last-reference-files 5 \
  --radiation rrtmgp \
  --tune-evals 3 \
  --tune-seed 20260810 \
  --force \
  --no-merge 2>&1 | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader" | tail -40
echo "COST_RC=${PIPESTATUS[0]}"
echo ALLDONE
