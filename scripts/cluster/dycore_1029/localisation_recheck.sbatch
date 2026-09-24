#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=hs1029loc2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/loc2_%j.out
#
# #1029 -- does the repaired localisation say anything, on a real blow-up?
#
# Twelve arms reported "at blowup: max_wind_k=0, lat=-88.75, lon=0" -- the same
# triple every time, at every mountain height and both failure timescales.
# That is argmax on a wholly non-finite field returning index 0, not a place.
# The guard now refuses to localise a field with nothing finite in it, and the
# series additionally records WHERE the state first stopped being finite and
# HOW MUCH of it went.
#
# READING RULE, pre-registered:
#   * "non-finite at blowup" now lists the three max_wind_* location keys
#       -> the guard fires, the fabricated coordinate is gone.  Required.
#   * first_nonfinite_* finite and n_nonfinite_wind small
#       -> the burst IS localisable and that coordinate is the real one; read
#          it against the three suspects (ridge at 30N/270E, |lat|>80 polar,
#          model top).
#   * first_nonfinite_* also non-finite and n_nonfinite_wind = every cell
#       -> the state is wholly gone by the time the check fires, and NO
#          per-step diagnostic can localise this; the next move is a finer
#          blow-up check interval, not a better statistic.
# The last outcome is the most likely one and it is still worth having on the
# record, because it retires per-step localisation as an approach.
#
# Two seeds, one from each end of the measured survival distribution, so the
# answer is not read off a single draw.
# COST: two idealized arms, minutes each.  No offline test can produce a
# blow-up state's geometry without integrating to the blow-up.
set -uo pipefail

ROOT=/burg-archive/glab/users/pg2328/_sweep0923_wt
cd "$ROOT" || exit 2
OUT="/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/loc2_$SLURM_JOB_ID"
mkdir -p "$OUT"
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/ac5006/legoESM/.venv/bin/python
[ -x "$PY" ] || { echo "FATAL: interpreter $PY missing"; exit 2; }
echo "sha=$(git -C "$ROOT" rev-parse --short HEAD)$([ -n "$(git -C "$ROOT" status --porcelain)" ] && echo -dirty)  $(date -u)"

for s in 65537 42; do
  echo "=== seed $s (measured survival: 11.1 d and 54.6 d) $(date -u) ==="
  LEGOESM_TOPO_SEED=$s LEGOESM_TOPO_H0=2000 "$PY" -u \
      scripts/matrix/run_atmosphere_test_matrix.py \
      --only hydro --test "=held_suarez_topo" --grid latlon \
      --no-cross-grid-plots -o "$OUT/s$s" 2>&1 | grep -E "BLOWUP|XFAIL|XPASS"
  r=$(find "$OUT/s$s" -name results.txt | head -1)
  [ -n "$r" ] && grep '^notes:' "$r"
done
echo DONE $(date -u)
