#!/bin/bash
#SBATCH --job-name=nino3pr
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nino3pr_%j.out
#
# THE 2.3x IS NOT IN THE CLOSURE AMPLITUDE OR THE MIXING LENGTH.
#
# Job 9417282 (zero-step closure, nino3 box, NEMO's OWN restart en handed to
# our kernel, scored against NEMO's own avm_k):
#     nino3   K_M/avm_k 0.999    l_k/zmxlm 1.070    e_grow 1.032
# i.e. handed NEMO's TKE, our MOMENTUM diffusivity reproduces NEMO's to 0.1%
# in the cold tongue, our mixing length is within 7%, and one 3600 s step
# barely moves our TKE.  Meanwhile Stage A scores K_H against avt at 2.311
# there.  A closure that is right on momentum and 2.3x on tracer has a
# TRACER/MOMENTUM SPLIT problem, not an energy problem.
#
# THE COROBORATING NULL. The --prandtl-mode nemo_ri arm came back BIT-IDENTICAL
# to the card's "richardson" (4.3459e-03 both, ratio 2.297 both).  Two
# different Prandtl formulas producing identical numbers to five digits is not
# a physics coincidence; either both saturate on the same clamp, or the
# Prandtl reduction is not reaching K_H on this path at all.
#
# WHAT THIS RUN MEASURES.  NEMO writes avm and avt at the SAME record, so its
# own Prandtl number is avm/avt with nothing inferred, and ours is K_M/K_H.
# The new prandtl_split_report prints both beside K_M/avm and K_H/avt, with
# NEMO's avtb/avmb floors excluded (on a floored interface avm/avt is a ratio
# of two constants and says nothing about nn_pdl) and the pointwise
# factorisation asserted before any row is printed.
#
#   Pr_ours ~ 1.00 while Pr_nemo ~ 2-3   -> the Prandtl reduction is INERT on
#        this path; that is the whole 2.3x and it is a code defect, not a
#        tuning gap.  Next step is to find where K_H loses the division.
#   Pr_ours ~ Pr_nemo, K_M/avm ~ 2.3     -> the split is fine and the Mode-B
#        equilibrium TKE is the problem (the zero-step arm handed our kernel
#        NEMO's en; Stage A lets it find its own).  Different fix entirely.
#   Pr_ours ~ Pr_nemo, K_M/avm ~ 1.0     -> the factorisation control would
#        have failed; the run aborts rather than printing that.
#
# One arm, the production card (mxl=3), same tfile/rec/protocol as the 2.311
# baseline.  Nothing here runs the model.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM; cd $REPO || exit 1
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
echo "=== git SHA: $(git -C $REPO rev-parse HEAD) ==="
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
D=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
$PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
  --tfile $D/ORCA1_1h_20010101_20010101_trd1h_T.nc \
  --ufile $D/ORCA1_1h_20010101_20010101_trd1h_U.nc \
  --vfile $D/ORCA1_1h_20010101_20010101_trd1h_V.nc \
  --rec 1 --skip-maps --mxl-choice 3 \
  --output-dir $REPO/results/omip_nemo/nino3_prandtl_split
echo "=== rc=$? ==="
