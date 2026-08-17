#!/bin/bash
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/slurm_logs/%x-%j.out
# Canopy conductance sweep against ERA5 evaporation, offline, on the AMIP mesh.
# One year (1978) per candidate; the coupled arm showed wet-land LE ~25-55 %
# low, so the sweep raises conductance: g1 (stomatal slope) and vc_max25.
set -euo pipefail
REPO=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_amip3
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
PKG="${REPO}/packages"
export PYTHONPATH="${REPO}/src:${PKG}/core:${PKG}/atmosphere:${PKG}/coupler:${PKG}/tools:${PKG}/ml:${PKG}/ocean:${PKG}/land:${PKG}/ice"
export JAX_ENABLE_X64=1
cd "${REPO}"
BASE=config/lmip/amip_mpas4_spinup.yaml
OUT=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/canopy_cal
mkdir -p "${OUT}"
run_one () {  # name g1 vcmax
  local name="$1" g1="$2" vc="$3"
  local cfg="${OUT}/${name}.yaml"
  ${PY} - "$BASE" "$cfg" "$g1" "$vc" <<'PYEOF'
import sys, yaml
base, out, g1, vc = sys.argv[1:5]
c = yaml.safe_load(open(base))
c["forcing"]["year_start"] = 1978; c["forcing"]["year_end"] = 1978
c["time"]["n_steps"] = 8760
if g1 != "null": c["physics"]["g1"] = float(g1)
if vc != "null": c["physics"]["vc_max25"] = float(vc)
yaml.safe_dump(c, open(out, "w"))
PYEOF
  ${PY} -u scripts/run/run_lmip_biophys.py --config "${cfg}" --output "${OUT}/${name}" 2>&1 | tail -1
}
run_one default null null
run_one g1_9    9.0  null
run_one g1_12  12.0  null
run_one vc_90  null  90.0
run_one vc_120 null 120.0
run_one g1_9_vc90 9.0 90.0
echo SWEEP_DONE
