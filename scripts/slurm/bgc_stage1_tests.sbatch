#!/bin/bash
#SBATCH --job-name=bgc-s1-tests
#SBATCH --partition=compute
#SBATCH --account=bb1596
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=bgc-s1-tests-%j.out
#
# MARBL port, stage 1: unit tests + repo survey + data survey (the dev session cannot
# run shell commands).  Submit from the repo root:
#     sbatch scripts/slurm/bgc_stage1_tests.sbatch
# Output: ./bgc-s1-tests-<jobid>.out

set -uo pipefail
module load git
# Work from the repo root regardless of where sbatch was called.
REPO=$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel) || { echo "not inside the repo"; exit 1; }
cd "$REPO"

# Use this repo's src, not an installed copy.
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PY=/work/bb1596/b384036/b384036/envs/climateval_gpu/bin/python
PYTEST="$PY -m pytest -q --no-header -p no:cacheprovider"

echo "=== host / env ==="
hostname; date
git rev-parse --abbrev-ref HEAD; git log --oneline -3; git status --short
$PY -c "import jax; print('jax', jax.__version__, jax.devices())"
$PY - <<'EOF'
import importlib
for m in ("xarray", "netCDF4", "h5netcdf", "scipy", "cfgrib", "eccodes", "zarr", "pandas", "optax", "equinox"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:10s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:10s} MISSING ({type(e).__name__})")
EOF

echo
echo "=== survey: KPP call sites / removed fields ==="
grep -rn "kpp_vertical_mixing(\|h_bl_prev\|c_b\b\|_boundary_layer_depth" --include="*.py" src scripts tests || true
echo
echo "=== survey: old BGC schemes (kept until stage 2; callers listed for the later port) ==="
grep -rn "npzd_v2\|marbl_cesm\|marbl_lite\|par_profile_v2\|init_bgc_from_woa\|_WOA_BGC_PATHS\|nudge_nutrients\|NO3_target\|O2_init_surf\|theta_min_d" \
    --include="*.py" src scripts tests || true
echo
echo "=== survey: BGC entry points outside the package ==="
grep -rn "BiogeoConfig(\|step_ocean_biogeochemistry\|init_biogeo_state\|OceanBiogeoState(" \
    --include="*.py" src scripts tests | grep -v "/biogeochemistry/" || true
echo
echo "=== survey: possibly unused imports in edited files ==="
grep -n "^from pathlib import Path\|Path" src/legoesm/ocean/init_woa.py | head -5 || true
grep -n "constants\.\|^import jax$" src/legoesm/ocean/physics/vertical_mixing/integration.py | head -5 || true

echo
echo "=== data survey ==="
for d in /work/bb1596/b384036/data/bgc_obs /work/bb1596/b384036/data/bgc_obs/woa23 \
         /work/bb1596/b384036/data/bgc_obs/glodapv2 \
         /pool/data/ERA5 /pool/data/ERA5/E5 /pool/data/ERA5/E5/sf /pool/data/ERA5/E5/sf/an \
         /pool/data/ERA5/E5/sf/an/1M /pool/data/ERA5/E5/sf/fc /pool/data/ERA5/E5/sf/fc/1M \
         /pool/data/ICDC/ocean /pool/data/ICDC/atmosphere; do
    echo "--- $d"
    ls "$d" 2>&1 | head -40
done
echo "--- ERA5 monthly sample file (first match)"
find /pool/data/ERA5/E5/sf -maxdepth 4 -path "*1M*" -name "*2010*" 2>/dev/null | head -20
echo "--- bgc_obs file list / sizes"
find /work/bb1596/b384036/data/bgc_obs -type f -exec ls -la {} \; 2>/dev/null | head -100

echo
echo "=== tests: new (KPP, MARBL, column) ==="
$PYTEST tests/ocean/unit/test_kpp_boundary_layer.py
RC1=$?
$PYTEST tests/ocean/unit/test_marbl.py
RC2=$?
$PYTEST tests/ocean/unit/test_ocean_column.py
RC3=$?

echo
echo "=== tests: existing KPP / vertical mixing ==="
KPP_TESTS=$(grep -rlE "kpp|KPPConfig|vertical_mixing|k_profiles" tests --include="test_*.py" \
            | grep -v test_kpp_boundary_layer | sort -u)
echo "$KPP_TESTS"
$PYTEST $KPP_TESTS
RC4=$?

echo
echo "=== tests: existing ocean BGC + anything importing the BGC package ==="
BGC_TESTS=$(grep -rlE "biogeochemistry|run_omip" tests --include="test_*.py" \
            | grep -v "test_marbl.py\|test_ocean_column.py" | sort -u)
echo "$BGC_TESTS"
$PYTEST $BGC_TESTS
RC5=$?

echo
echo "=== import smoke for edited scripts ==="
$PY -c "import ast,sys; [ast.parse(open(f).read(), f) for f in ['scripts/run_omip.py']]; print('run_omip.py parses')"

echo
echo "=== summary: kpp_new=$RC1 marbl=$RC2 column=$RC3 kpp_existing=$RC4 bgc_existing=$RC5 ==="
date
