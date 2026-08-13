#!/bin/bash
#SBATCH --job-name=zstep
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/zstep_%j.out
#
# ============================================================================
# IS THE RESIDUAL EXCESS OUR MIXING LENGTH, OR OUR TKE EVOLUTION?
# ============================================================================
#
# Job 9403798 measured, seeded with NEMO's own `en` and one 3600 s step:
#     Arctic calm    3.09x NEMO avt   corr 0.743
#     Antarctic calm 1.71x            corr 0.723
# That number cannot say whether our LENGTH is wrong before the step, or the
# lengths agree and our TKE evolves differently during it.
#
# This removes the step.  NEMO's restart holds, at ONE instant (step 8760),
# `en`, `avm_k`, `avt_k`, `dissl` AND `tn/sn/un/vn`.  NEMO's coefficient is
# avm = rn_ediff*zmxlm*sqrt(en); ours is K_M = c_k*l_k*sqrt(e), same
# c_k = rn_ediff = 0.1.  Fed the SAME en, the K ratio IS the length ratio.
#
# Stage 1 re-rebuilds the restart WITH the zdftke coefficients — the existing
# restart_8760_global.npz was built with the rebuilder's default field list,
# which keeps only tn/sn/un/vn/sshn/en and drops avm_k/avt_k/dissl.
#
# PRE-REGISTERED, Arctic band, volume-weighted length ratio l_k(ours)/zmxlm(NEMO)
# over interfaces where NEMO's avm_k is off its avmb floor:
#   LENGTH IS THE DEFECT   : ratio >= 2.5  -- our mixing length is ~3x NEMO's
#       before a single step, and the sweeps / surface anchor / N2 seed are
#       where to look.  The TKE evolution is then exonerated.
#   LENGTH IS FINE         : ratio <= 1.3  -- lengths agree, so the 3.09x is
#       built during the step, and the suspects become the shear production
#       (we run tke_shear_production="squared_centered"; NEMO's face-native
#       zdf_sh2 form with its coastal doubling is IMPLEMENTED but the ORCA1
#       card never selects it) and the buoyancy sink.
#   MIXED / INDETERMINATE  : 1.3 to 2.5 -- report both, claim no single cause.
#
# The probe refuses to print a number until three controls pass: its own K_M
# must equal c_k*l_k*sqrt(en) on unclamped interfaces; NEMO's inversion is
# restricted to interfaces off the avmb floor and the excluded fraction is
# printed; and the inverted zmxlm is cross-checked against NEMO's stored
# `dissl`, which ORCA1's nn_mxl=2 makes equal to zmxlm.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TRD2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
MESH=$REPO/data/grids/eORCA1.2_mesh_mask.nc
OUT=$REPO/results/omip_nemo/zero_step_closure
FULL=$TRD2/restart_8760_full.npz
echo "HEAD: $(git -C $REPO log --oneline -1)"
NFAIL=0
run () { echo; echo "########## $1"; shift; "$@"; local rc=$?; echo "=== rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1)); }

run "rebuild the step-8760 restart WITH the zdftke coefficients" \
  $PY -u scripts/validate/ocean_fidelity/rebuild_nemo_restart.py \
    --pattern "$TRD2/ORCA1_00008760_restart_oce_*.nc" \
    --fields tn sn un vn sshn en avm_k avt_k dissl \
    --out $FULL

run "zero-step closure: our l_k and K_M vs NEMO's own zmxlm and avm_k" \
  $PY -u scripts/validate/ocean_fidelity/nemo_zero_step_closure.py \
    --restart-npz $FULL --mesh-mask "$MESH" \
    --out-json $OUT/step8760.json

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
