#!/bin/bash
#SBATCH --job-name=evdlens
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/evdlens_%j.out
# Offline falsifier (GLM/codex claim review 2026-09-06): would NEMO's EVD
# trigger fire on our lens columns after a night of cooling?
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=$ROOT/results/omip_nemo
$PY -m pytest -q tests/ocean/unit/test_evd_lens_trigger_probe.py 2>&1 | tail -2
for s in trp_base2_d30/snapshot_day0030.npz trp_base2_d30/snapshot_day0015.npz mpas_base2_d30/snapshot_day0030.npz; do
  for box in "220 240" "240 260" "150 170"; do
    set -- $box
    echo "############ $s box $1-$2E"
    $PY -u scripts/validate/ocean_fidelity/evd_lens_trigger_probe.py --snapshot $R/$s --lon-lo $1 --lon-hi $2 2>&1 | grep -v Warning
    echo "### rc=$?"
  done
done
echo DONE
