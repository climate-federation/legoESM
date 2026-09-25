#!/bin/bash
#SBATCH --job-name=nyfwind
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nyfwind_%j.out
#
# A CACHE IN WHICH ONLY THE WIND CHANGED.
#
# Switching to the fields NEMO reads took global SST error from 0.97 to 0.81
# and the cold-tongue bias from +2.50 to +1.49, and that was read as evidence
# that our equatorial trades had been 33 percent too weak.  The adversarial
# review refused the attribution: the corrected cache also changes shortwave
# by -5.4 W/m2, air temperature by +1.2 K, humidity by -3 percent and
# precipitation by +15 percent.  Any of those could carry a tropical SST
# improvement, and the undercurrent -- which the wind story predicts should
# have responded -- did not.
#
# This builds the cache that separates them: corrected U_10_MOD/V_10_MOD,
# raw everything else.  It is NOT what NEMO reads, so it is an attribution
# instrument and must never be used to score fidelity.
#
# GATE.  Two numbers decide whether the build did what it claims, and the job
# fails rather than writing a cache nobody checked:
#   nino3 mean |U10| must be ~6.28 m/s  (the corrected value; raw is 4.74)
#   global mean shortwave must be ~165.4 W/m2 (the RAW value; corrected 160.0)
# Together they say wind corrected, radiation not.
set -uo pipefail
R=/burg-archive/glab/users/pg2328/legoESM; cd $R || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$R/src:$(ls -d $R/packages/*/ | sed 's:/$::' | tr '\n' ':')$R"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
IN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS

echo "=== gate: variant separation ==="
$PY -m pytest -q tests/unit/test_build_core2_nyf_variants.py 2>&1 | tail -3
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: variant tests failed rc=$rc"; exit 1; }

echo "=== CORE-II inputs: $IN"
[ -r "$IN/u_10.15JUNE2009.nc" ] || { echo "FATAL: no CORE-II winds under $IN"; exit 1; }

OUT=/burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/core2_nyf_windsonly/nyf.zarr
$PY -u scripts/data/build_core2_nyf_zarr.py --inputs-dir "$IN" \
  --wind-variant windsonly --out "$OUT" 2>&1 | tail -30
rc=$?; echo "=== build rc=$rc"; [ $rc -eq 0 ] || exit 1

# GATE: run the relative check, which compares the SAME statistic across all
# three caches.  The first version of this gate compared against absolute
# values measured elsewhere with a different reduction (source-grid records
# 100-119, and an unweighted 365-record mean) and failed a cache that was
# correct.  Comparing across reductions is exactly the error a gate exists to
# catch, so the check now only ever compares like with like.
bash scripts/cluster/omip_nemo/_verify_nyf_variants.sbatch
rc=$?
echo "=== verify rc=$rc ==="
[ $rc -eq 0 ] || { echo "FATAL: the winds-only cache is not winds-only"; exit 1; }
echo DONE
