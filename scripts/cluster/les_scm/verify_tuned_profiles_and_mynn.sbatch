#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=les_verify
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_verify_%j.out
set +e
# Two jobs in one CPU slot, both cheap, both prerequisites for the full
# all-case campaign:
#
#  1. UNIT GATE for the tuned-profile capture. The driver now writes
#     scm_<scheme>_tuned_<var> alongside the default family and the per-case
#     plotter overlays the two. Neither needs a GPU to test.
#
#  2. LOCALIZE mynn25's non-finite rollout. It is the ONE closure the campaign
#     cannot rank (excluded on cbl in every run so far, with and without
#     --surface-flux-to-closure), so "tune all turbulence parameterizations"
#     is not satisfied until it either runs or is understood. The probe reports
#     the step, the field and the level; it prints no verdict.
#
# Every other scheme is swept on cbl too, as the CONTROL: if several closures
# die at the same step the cause is the case setup, not mynn25.
: "${V_WT:?set V_WT to the pinned worktree}"
WT=$V_WT
cd "$WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export PYTHONPATH=$WT/src:$(ls -d "$WT"/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=============== 1. unit gate ==============="
$PY -m pytest -q \
    tests/unit/test_plot_scm_les_turbulence_case_select.py \
    tests/unit/test_diag_scheme_blowup.py \
    tests/atmosphere/hydrostatic/test_run_scm_les_turbulence_tuning.py \
    2>&1 | tail -30
echo "unit rc=${PIPESTATUS[0]}"

echo "=============== 2. mynn25, every case ==============="
# Steps chosen to overrun the longest case at its own dt; the probe stops at
# the first non-finite value, so an early death costs nothing.
for c in cbl gabls1 ekman wangara bomex rico astex dycoms; do
  echo "--------- mynn25 / $c ---------"
  $PY scripts/validate/diag_scheme_blowup.py --case "$c" --scheme mynn25 \
      --nlev 48 --steps 2000 --print-every 200 2>&1 | tail -25
  echo "rc=${PIPESTATUS[0]}"
done

echo "=============== 3. control: every scheme on cbl ==============="
# If mynn25 is alone in dying here, the defect is in mynn25. If it has company,
# the defect is in the cbl arm.
for s in smagorinsky louis tke mynn25 clubb_lite clubb holtslag_boville ysu edmf; do
  echo "--------- $s / cbl ---------"
  $PY scripts/validate/diag_scheme_blowup.py --case cbl --scheme "$s" \
      --nlev 48 --steps 2000 --print-every 400 2>&1 | tail -12
  echo "rc=${PIPESTATUS[0]}"
done
echo "DONE $(date +%T)"
