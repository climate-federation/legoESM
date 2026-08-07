#!/bin/bash
#SBATCH --job-name=gsam_rce_snd
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/gsam_rcemip/probe_%j.log
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
# Measure the gSAM RCEMIP1 RCE300 sounding against our analytic Wing IC.
# Runs from a PINNED worktree of the pushed branch so a concurrent edit in the
# main checkout cannot change what this job read (anti-confound rule).
set -euo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
BRANCH="${BRANCH:-feat/rcemip-gsam-oracle-sounding}"
WT=/burg-archive/glab/users/pg2328/legoESM_wt/gsam_rce_${SLURM_JOB_ID}
GSAM_ROOT=/burg/glab/users/pg2328/gsam_oracle/gSAM1.8.8
SND="$GSAM_ROOT/CASES/RCEMIP1/snd_rcemip_300s6.11.2"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

mkdir -p "$(dirname "$WT")" "$REPO/results/gsam_rcemip"
git -C "$REPO" worktree add --detach "$WT" "origin/$BRANCH"
echo "PINNED sha $(git -C "$WT" rev-parse HEAD)  branch=$BRANCH"
trap 'git -C "$REPO" worktree remove --force "$WT" || true' EXIT

# scripts/tmp/ is gitignored (throwaway probes are never tracked), so the probe
# cannot come from the pinned tree.  Copy it in and record its SHA256 so the
# exact instrument that produced these numbers is identifiable afterwards.
mkdir -p "$WT/scripts/tmp"
cp "$REPO/scripts/tmp/_probe_gsam_rcemip_sounding.py" "$WT/scripts/tmp/"
echo "PROBE sha256 $(sha256sum "$WT/scripts/tmp/_probe_gsam_rcemip_sounding.py")"

cd "$WT"
_pp="$WT"
for _d in "$WT"/packages/*/; do
  [ -d "${_d}legoesm" ] && _pp="${_pp}:${_d}"
done
export PYTHONPATH="${_pp}:${WT}/src:${PYTHONPATH:-}"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export LEGOESM_GSAM_ROOT="$GSAM_ROOT"

echo "=== PROBE ==="
"$PY" scripts/tmp/_probe_gsam_rcemip_sounding.py --snd "$SND"
echo "=== PROBE DONE rc=$? ==="
