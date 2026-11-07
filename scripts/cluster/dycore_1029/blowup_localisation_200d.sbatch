#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=hs1029loc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/loc_%j.out
#
# #1029 -- WHERE does the lat-lon held_suarez_topo jet runaway happen?
#
# The thread records the magnitude (131 m/s -> NaN) and the step, never the
# LOCATION, so the three live suspects are indistinguishable from the record.
# The lat-lon Held-Suarez series now logs the argmax of the wind speed in
# degrees and level index at every diagnostic sample AND at the blow-up step
# itself (`scalars_at_blowup`), which is the whole point of this run.
#
# THE MOUNTAIN: DCMIP 2012 2-0-0 ridged cosine bell, centred at
# lon 270 E / lat 30 N, envelope radius 3*pi/4 rad (~7000 km) with a ~1300 km
# ridged fine structure, peak 2000 m.
#
# PRE-REGISTERED READING RULE (write it down before the run, not after):
#   * argmax near 30 N / 270 E, in the lower half of the column
#       => terrain-following pressure-gradient error over the steep ridge.
#          CONFIRMS the generator suspect (hybrid PGF hybrid_factor vs the
#          Simmons-Burridge geopotential, #1078 / A_half).
#   * argmax at |lat| > 80, anywhere in lon
#       => a polar-filter / metric-singularity problem that the mountain only
#          triggers.  The PGF work would be aimed at the wrong place.
#   * argmax at k near the model top, away from the ridge
#       => lid / sponge / gravity-wave reflection.
#   * argmax wandering between samples with no fixed site
#       => not a localized defect; treat the runaway as a global mode.
# A run that survives 200 d instead is XPASS and is itself reportable.
#
# CONTROL (one variable = the topography): the flat held_suarez arm on the
# same grid/coord/length, which passes, so any location claim is read against
# where the healthy jet's maximum sits.
#
# COST: ~10 min on one A100 for both arms (the reproducer NaNs in ~70 s of
# wall on the original card; the sponge-mitigated tree reaches ~day 96).
# No cheaper offline test exists -- the quantity wanted is the location of a
# maximum in a trajectory that has to be integrated to produce it.
set -uo pipefail

ROOT=/burg-archive/glab/users/pg2328/legoESM
cd "$ROOT" || exit 2
OUT="${OUT:-$ROOT/results/dycore_1029/localisation}"
mkdir -p "$OUT"

export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/ac5006/legoESM/.venv/bin/python

echo "sha=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo -dirty)  host=$(hostname)  $(date -u)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"

run_arm () {   # $1 = label, $2 = case
    echo "=== arm $1 (case=$2) $(date -u) ==="
    "$PY" -u scripts/matrix/run_atmosphere_test_matrix.py \
        --only hydro --test "=$2" --grid latlon \
        --no-cross-grid-plots -o "$OUT/$1" \
        || echo "[arm $1] matrix exited nonzero -- see $OUT/$1/summary.json"
}

run_arm topo held_suarez_topo
run_arm flat held_suarez

echo "=== blow-up location (the deliverable) ==="
for a in topo flat; do
  echo "--- $a"
  "$PY" - "$OUT/$a" <<'PYEOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
for f in sorted(root.rglob("summary.json")):
    d = json.loads(f.read_text())
    print(f.relative_to(root), json.dumps(d, indent=1)[:1500])
for f in sorted(root.rglob("mean_timeseries.csv")):
    lines = f.read_text().splitlines()
    print(f.relative_to(root), "HEAD:", lines[0])
    for ln in lines[-6:]:
        print("   ", ln)
PYEOF
done
echo "DONE $(date -u)"
