#!/bin/bash
#SBATCH --job-name=nino3len
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=0:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nino3len_%j.out
#
# THE QUESTION.  Stage A measured K_H(ours)/avt(NEMO) = 2.311 in the nino3
# calm columns, and five one-variable ablations (prandtl_mode, tke_mxl_choice,
# etau_mode, lc, plus the misconceived bg_diff_scale arm) account for ~1% of
# it.  So the excess is in the core closure K = c_k * l * sqrt(e).  Which
# factor?
#
# THE DISCRIMINATOR, and why it needs no run.  At the restart instant NEMO
# hands us BOTH its TKE (`en`) and its momentum diffusivity (`avm_k`), so
# NEMO's own length inverts exactly: zmxlm = avm_k / (c_k*sqrt(en)).  The
# zero-step table drives OUR mixing length with NEMO's OWN en, so
#
#     K0_rat  ==  l_k(ours) / zmxlm(NEMO)      -- a PURE LENGTH ratio,
#
# with the TKE amplitude divided out by construction (the probe's own C1
# asserts K_M == c_k*l_k*sqrt(en) before any of this is read).  The one-step
# table then reports e_grow = our TKE after 3600 s over the en we were handed.
#
#   LENGTH  : nino3 K0_rat ~ 2.3   -> the mixing length is the whole story;
#             next probe splits lup from ldn against NEMO's dissl.
#   AMPLITUDE: nino3 K0_rat ~ 1.0  -> the length is right and our TKE level is
#             ~5x too high in the cold tongue (K ~ sqrt(e)), so the source
#             terms / dissipation balance is where the 2.3x lives.
#   MIXED   : anything between -- report the split, claim neither alone.
#
# Everything here is a re-read of an EXISTING restart; the only change is that
# the tables now carry the lon-boxed nino3/nino4/eqpac regions beside the
# latitude bands.  The 23S-23N band average is what hid a +3.2 C cold-tongue
# SST bias behind a +0.63 band mean, and it would hide this too.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
echo "=== git SHA: $(git -C $WT rev-parse HEAD) ==="
T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -u \
  scripts/validate/ocean_fidelity/nemo_zero_step_closure.py \
  --restart-npz $T/restart_8760_full.npz \
  --mesh-mask $WT/data/grids/eORCA1.2_mesh_mask.nc \
  --trd-tfile $T/ORCA1_1h_20010101_20010101_trd1h_T.nc --trd-rec 0 \
  --out-json $WT/results/omip_nemo/zero_step_closure/nino3_length_split.json
echo "=== rc=$? ==="
