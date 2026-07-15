#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carbon_cal_validate
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=results/carbon_calibration/validate_%j.out
# =============================================================================
# Stage-B (B2) carbon calibration -- CHEAP CPU validation before the GPU run.
#   1. B1 SOC-observation unit tests + the new cover-weight tests
#   2. the B2 trainer unit tests (incl. a --quick dry-run FULL pipeline on CPU)
#   3. the B1 differentiability gradient smoke (proves jax.grad still flows)
#   4. a metadata probe of the CLM5 surfdata ORGANIC/DZSOI columns
# Run:  mkdir -p results/carbon_calibration && sbatch scripts/cluster/carbon_calibration/validate_carbon_trainer.sbatch
# =============================================================================
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/lc_rebase
cd "$REPO" || { echo "cannot cd $REPO"; exit 2; }
mkdir -p results/carbon_calibration

PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO:$(ls -d "$REPO"/packages/*/ | sed 's:/$::' | tr '\n' ':')${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
CLM5=/burg-archive/glab/users/pg2328/legoESM/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc

echo "=== node $(hostname) | python $($PY --version 2>&1) ==="
echo "=== [1/4] SOC-observation unit tests ==="
$PY -m pytest tests/land/unit/test_soc_observations.py -q; R1=$?

echo "=== [2/4] B2 trainer unit tests (incl --quick dry-run full pipeline) ==="
$PY -m pytest tests/land/unit/test_train_carbon_params.py -q -s; R2=$?

echo "=== [3/4] B1 differentiability gradient smoke ==="
$PY -m pytest tests/land/validation/test_carbon_calibration_gradient.py -q -s; R3=$?

echo "=== [4/4] CLM5 surfdata ORGANIC/DZSOI probe ==="
$PY - "$CLM5" <<'PY'
import sys, numpy as np, xarray as xr
f = sys.argv[1]
ds = xr.open_dataset(f, decode_times=False)
print("dims:", dict(ds.dims))
for v in ("ORGANIC", "DZSOI", "ZSOI", "PCT_SAND", "PCT_CLAY"):
    if v in ds:
        a = ds[v]
        print(f"  {v}: dims={a.dims} shape={a.shape} units={a.attrs.get('units','?')}")
    else:
        print(f"  {v}: ABSENT")
if "ORGANIC" in ds:
    o = np.asarray(ds["ORGANIC"].values, float)
    col = np.nansum(o, axis=0)                     # crude per-cell sum over layers
    print(f"  ORGANIC per-layer range [{np.nanmin(o):.3g}, {np.nanmax(o):.3g}]"
          f"  layer0 nonzero cells={int(np.sum(o[0] > 0))}")
ds.close()
PY
R4=$?

echo "=== RESULTS: soc_unit=$R1 trainer_unit=$R2 grad_smoke=$R3 organic_probe=$R4 ==="
if [ "$R1" -eq 0 ] && [ "$R2" -eq 0 ] && [ "$R3" -eq 0 ] && [ "$R4" -eq 0 ]; then
    echo "VALIDATE_OK: all green -- safe to launch the GPU calibration."
else
    echo "VALIDATE_FAIL: fix before the GPU run."
fi
