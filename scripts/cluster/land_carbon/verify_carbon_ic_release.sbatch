#!/bin/bash
# Reproduce the published land-carbon IC's headline numbers from the RELEASE
# ARTIFACTS -- the 1:1 replication check for `land-carbon-ic-v1`.
#
# This does NOT rebuild the IC (that is the 7 h Stage-2 chain).  It verifies that
# the artifacts you downloaded are the ones the numbers were computed from, and
# that THIS checkout still reproduces those numbers and still ingests the file:
#
#   1. sha256 of every downloaded asset vs the release's SHA256SUMS
#   2. the area-weighted global scorecard, recomputed here from the finidat
#   3. the strict whole-grid loader accepts it (the coupled path)
#   4. a seeded-vs-cold single-point LMIP pair (the run_lmip path), incl. the
#      permafrost-phi restart round-trip
#
#   mkdir -p scripts/tmp data/land_carbon_ic
#   gh release download land-carbon-ic-v1 -R climate-federation/legoESM \
#       -D data/land_carbon_ic
#   sbatch scripts/cluster/land_carbon/verify_carbon_ic_release.sbatch
#
# Submit from the repo root (WORKDIR defaults to SLURM_SUBMIT_DIR); override
# LEGOESM_WORKDIR / LEGOESM_PYTHON / CIC_RELEASE_DIR.
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carbon_ic_verify
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=scripts/tmp/%x_%j.out

set -uo pipefail
WORKDIR="${LEGOESM_WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
test -d "$WORKDIR/packages" || {
    echo "WORKDIR=$WORKDIR is not a legoESM checkout (no packages/); submit from the repo root or set LEGOESM_WORKDIR"; exit 2; }
cd "$WORKDIR"

export PYTHONPATH=$(ls -d "$WORKDIR"/packages/*/ | sed 's:/$::' | tr '\n' ':'):$WORKDIR
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
PYJ="${LEGOESM_PYTHON:-/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python}"
REL="${CIC_RELEASE_DIR:-$WORKDIR/data/land_carbon_ic}"
IC="$REL/global_carbon_ic.npz"

echo "=== node: $(hostname)  time: $(date)  verify land-carbon-ic-v1 ==="
echo "provenance: $WORKDIR @ $(git rev-parse --short HEAD 2>/dev/null || echo NO-GIT) dirty=$(git status --porcelain 2>/dev/null | wc -l) files"
test -f "$IC" || { echo "MISSING $IC -- gh release download land-carbon-ic-v1 -R climate-federation/legoESM -D $REL"; exit 2; }
RC_ALL=0

echo "### 1. checksums vs the release SHA256SUMS"
if [ -f "$REL/SHA256SUMS" ]; then
    ( cd "$REL" && sha256sum -c --ignore-missing SHA256SUMS ) || RC_ALL=1
else
    echo "  (no SHA256SUMS in $REL -- skipping; download it to pin identity)"
fi

echo "### 2. global scorecard recomputed from the finidat"
$PYJ - "$IC" <<'PY' || RC_ALL=1
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
lat, m = d["lat"], d["land_mask"].astype(bool)
w = np.cos(np.deg2rad(lat))
soc = (d["C_som_active"] + d["C_som_slow"] + d["C_som_passive"]) / 1000.0
bio = (d["C_lab"] + d["C_fol"] + d["C_root"] + d["C_wood"]) / 1000.0
awt = lambda x, s: float(np.sum(x[s] * w[s]) / np.sum(w[s]))
got = {"n_land": int(m.sum()), "soc": awt(soc, m), "bio": awt(bio, m)}
for lo, hi, name in [(66, 90, "arctic"), (50, 66, "boreal"), (-23, 23, "tropics")]:
    s = m & (lat >= lo) & (lat < hi)
    got[name] = awt(soc, s)
# Published values (docs/land/global_carbon_ic_spinup.md).  A mismatch means the
# file is NOT the one the documentation describes.
exp = {"n_land": 4914, "soc": 3.470, "bio": 7.880,
       "arctic": 0.61, "boreal": 6.66, "tropics": 2.52}
ok = True
for k, e in exp.items():
    g = got[k]
    hit = (g == e) if k == "n_land" else abs(g - e) <= 0.005 * max(1.0, abs(e))
    ok &= hit
    print(f"  {k:8s} got {g:10.3f}  expected {e:10.3f}  {'OK' if hit else 'MISMATCH'}")
print("scorecard:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY

echo "### 3. the STRICT whole-grid loader (coupled path) ingests it"
$PYJ - "$IC" <<'PY' || RC_ALL=1
import sys
from legoesm.land.carbon.global_init import load_finidat_carbon_ic
carbon, phi = load_finidat_carbon_ic(sys.argv[1], expect_ncol=13824)
assert carbon is not None, "not detected as a carbon finidat"
assert phi is not None, "permafrost phi missing"
print(f"  pools {carbon.C_som_slow.shape} phi {phi.shape}  OK")
PY

echo "### 4. single-point LMIP: seeded vs cold, and the phi restart round-trip"
OUT="$WORKDIR/scripts/tmp/carbon_ic_verify_$SLURM_JOB_ID"
$PYJ scripts/run/run_lmip.py --lat 40.0 --lon -105.0 --days 2 --dt 1800 \
    --carbon-scheme differland --carbon-ic "$IC" --output "$OUT/seeded" 2>&1 | tail -12
$PYJ scripts/run/run_lmip.py --lat 40.0 --lon -105.0 --days 2 --dt 1800 \
    --carbon-scheme differland --output "$OUT/cold" 2>&1 | tail -4
$PYJ scripts/run/run_lmip.py --lat 40.0 --lon -105.0 --days 1 --dt 1800 \
    --carbon-scheme differland \
    --restart-from "$OUT/seeded/restart_final.npz" --output "$OUT/resumed" 2>&1 | tail -4
$PYJ - "$OUT" "$IC" <<'PY' || RC_ALL=1
import sys, numpy as np
out, ic = sys.argv[1], sys.argv[2]
som = lambda p: sum(float(np.asarray(np.load(p)[f"carbon_C_som_{s}"]).reshape(-1)[0])
                    for s in ("active", "slow", "passive"))
seeded, cold = som(f"{out}/seeded/restart_final.npz"), som(f"{out}/cold/restart_final.npz")
r = np.load(f"{out}/resumed/restart_final.npz")
# The seed must (a) differ from a cold start and (b) sit at the FILE's value --
# a seed that merely "runs" proves nothing (a control that perturbs nothing).
d = np.load(ic, allow_pickle=True)
lat, lon, mask = d["lat"], d["lon"], d["land_mask"].astype(bool)
p2 = np.deg2rad(40.0); dl = np.deg2rad(((lon + 105.0 + 180.0) % 360.0) - 180.0)
h = (np.sin((np.deg2rad(lat) - p2) / 2) ** 2
     + np.cos(np.deg2rad(lat)) * np.cos(p2) * np.sin(dl / 2) ** 2)
dist = np.where(mask, np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))), np.inf)
i = int(np.argmin(dist))
file_som = float(d["C_som_active"][i] + d["C_som_slow"][i] + d["C_som_passive"][i])
print(f"  nearest land cell {i} at {dist[i]:.3f} deg; file SOM {file_som:.1f} gC/m2")
print(f"  seeded {seeded:.1f}  cold {cold:.1f}  resumed {som(f'{out}/resumed/restart_final.npz'):.1f}")
ok = (abs(seeded - file_som) < 0.05 * file_som and abs(seeded - cold) > 0.1 * cold
      and "soil_frozen_fraction" in r.files)
print("lmip seeding:", "PASS" if ok else "FAIL",
      "" if "soil_frozen_fraction" in r.files else "(phi did NOT survive the restart)")
sys.exit(0 if ok else 1)
PY

echo "VERIFY_EXIT=$RC_ALL  done: $(date)"
exit $RC_ALL
