#!/bin/bash
#SBATCH --job-name=tkeseed
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/tkeseed_%j.out
#
# THE PRE-REGISTERED PER-TERM BUDGET AT NEMO'S SEED (6641554e1, "NEXT").
#
# Our TKE equation, one 150 s step on NEMO's frozen day-365 restart state seeded
# from NEMO's own en, returns its OWN per-term contributions (TKEStepBudget:
# production, Langmuir, buoyancy, dissipation, vertical transport incl. the
# surface-face flux, pin, floor, etau) that sum to e_new - e_old to round-off
# (tests/ocean/unit/test_tke_step_budget.py). Beside them, NEMO's own terms from
# the same restart with zdftke.F90's formulas: eps = rn_ediss*dissl*en, B =
# avt_k*rn2, P = avm_k*S2, transport = the zzd_up/zzd_lw operator on en with
# avm_k and the 2e-5 face floor. NEMO's closure residual is bounded by its
# day-365 -> day-366 en change (restart 8784, rebuilt here if absent).
#
# Read at 2.1, 3.3 and 4.5 m, median of per-column ratios, the 208 calm columns
# 220-240E |lat|<=2. Branches (verbatim from 6641554e1):
#   * our transport INTO the 2-5 m band < 0.5x NEMO's while sinks match within
#     30% -> the vertical TKE transport (face coefficient or metric) is the
#     defect; veros_dz_slots and the 2e-5 face floor are the two candidate rows;
#   * our dissipation > 2x NEMO's at matched energy -> the dissipation length
#     near the surface (mixing-length construction / anchor);
#   * our Langmuir source < 0.5x NEMO's -> the h_lc depth on this state
#     [NOT SCOREABLE HERE: NEMO's LC is not in the restart; the table prints
#     OUR Langmuir term so its size against eps can be read];
#   * all terms within 30% -> the 3.7x is not local; a per-step trace of the
#     2.1 m interface over the first 20 steps is next;
#   * NEMO's own budget does not close to within 3x its daily change -> the
#     frozen-state premise fails at 2-5 m and nothing above is read.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
echo "HEAD: $(git -C $M log --oneline -1)"
echo "dirty: $(git -C $M status --porcelain | wc -l)"
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
D=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
NEXT=$D/restart_8784_en.npz
if [ ! -s "$NEXT" ]; then
  echo "=== rebuilding en from the +24 h restart tiles ==="
  $PY -u scripts/validate/ocean_fidelity/rebuild_nemo_restart.py \
    --pattern "$D/ORCA1_00008784_restart_oce_*.nc" --fields en --out "$NEXT" || exit 1
fi
DT=${DT:-150}
OUT=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/eq_avm_seed_budget_dt${DT}
mkdir -p "$OUT"
$PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
  --tfile $D/ORCA1_1h_20010101_20010101_trd1h_T.nc \
  --ufile $D/ORCA1_1h_20010101_20010101_trd1h_U.nc \
  --vfile $D/ORCA1_1h_20010101_20010101_trd1h_V.nc \
  --rec 1 --skip-maps --mxl-choice 3 --iwm-backgrounds \
  --restart-npz $D/restart_8760_full.npz --restart-next-npz "$NEXT" \
  --mode-a-iterations 1 --mode-a-dt $DT --mode-a-budget \
  --output-dir "$OUT"
rc=$?
echo "=== card exit: $rc ==="
exit $rc
