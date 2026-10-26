#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=padfix
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/padfix_%j.out
#
# Gate for the eORCA025 south-pad fix: the pad missed cos_lat_v (a v-face
# profile added to the geometry after the pad was written), which made the
# SPMD band stack ragged on the padded 1/4-degree grid.  Three parts:
#   1. the new tripwire tests (generic stagger audit + the exact band-stack
#      repro + seam_wall_rows growth) PASS on the fixed code;
#   2. NON-VACUITY: the PRE-fix pad (extracted from git, imported standalone)
#      demonstrably produces the ragged cos_lat_v the crash reported;
#   3. the whole existing tripole-internals suite still passes (the pad's
#      bit-exactness contract untouched).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
SCRATCH=${TMPDIR:-/tmp}/padfix_$$

echo "=== 1. new tripwire tests + full tripole-internals suite ==="
$PY -m pytest -q tests/grids/test_tripole_internals.py 2>&1 | tail -90
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: tests failed rc=$rc"; exit 1; }

echo "=== 2. NON-VACUITY: the pre-fix pad reproduces the ragged cos_lat_v ==="
mkdir -p "$SCRATCH"
git show 810fe93c4:packages/core/legoesm/grids/tripole.py > "$SCRATCH/tripole_prefix.py"
$PY - "$SCRATCH/tripole_prefix.py" <<'PY'
import importlib.util
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("tripole_prefix", sys.argv[1])
old = importlib.util.module_from_spec(spec)
sys.modules["tripole_prefix"] = old
spec.loader.exec_module(old)

n_lat, n_pad = 15, 5
g = old.create_synthetic_tripole(n_lat=n_lat, n_lon=24)
gp = old.pad_tripole_grid_south(g, n_pad)
got = tuple(np.shape(gp.cos_lat_v))
want_broken = (n_lat + 1,)              # the OLD pad leaves it un-grown
want_fixed = (n_lat + n_pad + 1,)
print(f"pre-fix cos_lat_v shape: {got}  (fixed code gives {want_fixed})")
if got == want_fixed:
    raise SystemExit("FAIL: pre-fix pad already grows cos_lat_v -- the new "
                     "tests could not have failed before the fix (vacuous)")
if got != want_broken:
    raise SystemExit(f"FAIL: unexpected pre-fix shape {got}")
print("NON-VACUITY OK: the pre-fix pad is demonstrably broken exactly as "
      "the 1/4-degree crash reported")
PY
rc=$?; rm -rf "$SCRATCH"
[ $rc -eq 0 ] || { echo "FATAL: non-vacuity check failed rc=$rc"; exit 1; }
echo "=== ALL GATES PASS ==="
echo DONE
