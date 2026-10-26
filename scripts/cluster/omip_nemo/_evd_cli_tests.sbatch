#!/bin/bash
#SBATCH --job-name=evdcli
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/evdcli_%j.out
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1) (+ uncommitted)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
$PY -m pytest -q tests/unit/test_run_omip_core2_evd_trigger_cli.py tests/unit/test_run_omip_core2_kappa_convention_cli.py tests/unit/test_run_omip_core2_freshwater_closure_cli.py tests/ocean/unit/test_evd_lens_trigger_probe.py 2>&1 | tail -15
$PY -c "import ast,sys; ast.parse(open('scripts/run/run_omip_core2.py').read()); print('syntax ok')"
echo DONE
