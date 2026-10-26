#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=nemogw
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nemogw_%j.out
# Score the ORACLE's own Arctic 66N gateway transports, so our -0.69 Sv Pacific
# sector is measured against NEMO rather than against observations.
#
# WHY THIS EXISTS: the production baseline's Pacific sector runs OUT of the
# Arctic at -0.69 Sv where observations put +0.8 to +1.0 Sv IN.  The model-side
# diagnostic was validated first (sign read from the code; the exactly-zero
# Siberian sector confirmed to be Siberian land, not a masking defect), so the
# defect is in the circulation -- but "wrong" had only ever been said against
# observations.  This supplies the oracle number.
#
# HOW IT IS CONSTRUCTED, because the obvious construction is wrong: a constant
# -j row sum is NOT a 66N section on eORCA1, which is a tripole.  The new
# `--arctic-gateways` mode instead feeds NEMO's uo/vo/e3u/e3v straight into the
# MODEL's OWN region/face/section machinery, so both sides are scored on an
# identical section and no numerics are duplicated.  The only translation is
# the C-grid index shift, which is not a fresh derivation -- it is the same
# shift ocean/vertical.py already applies to these very metrics.
#
# MATCHED WINDOW, held fixed on both sides: oracle record 5 (0-based) is run
# days 25-30.  Our matched value is the day-25 -> day-30 increment of the
# cumulative series, (-5.9752e9 + 4.4481e9)/2880 = -0.5303 Sv -- NOT the
# -0.6916 Sv whole-run mean, which spans a different window.
#
# PRE-REGISTERED, written before the run:
#   PRIMARY  NEMO's bering_pacific is POSITIVE (into the Arctic).  That makes
#            the reversal a real model defect and sizes it.
#   REFUTE   NEMO is also negative, i.e. the oracle does not push water north
#            across 66N in this window either -- then our sign is not the
#            defect it appears to be and the whole lead is dead.
#   GUARD    The staggering self-check must pass and the four sectors must net
#            to a small residual.  A large net means the section is not closed
#            and NO transport from this run may be quoted.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
MESH=/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc

echo "=== gate: the new core's unit tests ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_nemo_transports.py 2>&1 | tail -15
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || { echo "FATAL: unit gate rc=$rc — not running the measurement"; exit 1; }

echo "=== oracle Arctic gateways, record 5 (days 25-30) ==="
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -u scripts/validate/nemo_transports.py \
  --arctic-gateways --time-idx 5 \
  --grid-u $GW/ORCA1_5d_20000101_20000331_grid_U.nc \
  --grid-v $GW/ORCA1_5d_20000101_20000331_grid_V.nc \
  --mesh $MESH 2>&1 | grep -vE "^\s*$" | tail -30
echo "=== ours, SAME window (day 25 -> 30 increment of the cumulative series) ==="
echo "    bering_pacific -0.5303 Sv   davis_caa -2.2404 Sv   atlantic_nordic +2.9418 Sv"
echo DONE
