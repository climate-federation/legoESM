#!/bin/bash
#SBATCH --job-name=cdx_emin
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_emin_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/legoESM
LOG=$WT/omip_nemo/cdx_emin_body_${SLURM_JOB_ID}.txt
codex exec --sandbox read-only -C $WT \
"Adversarially review commit 134cff43 (git show 134cff43): under --iwm, orca1_zdftke_config
now sets tke_background=1e-10 and mxl_min=1e-3 (from 1e-6 and the 1e-8 default), to match
NEMO zdftke.F90:840-843 which forces rn_emin=1e-10 and rmxl_min=1e-3 when ln_zdfiwm=T.
ATTACK, numbered, severity-tagged:
1. Is the claim TRUE that NEMO forces rn_emin=1e-10 and rmxl_min=1e-3 under ln_zdfiwm?
   Open /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/src/OCE/ZDF/zdftke.F90 around
   line 840 and the ORCA1 namelist_cfg (ln_zdfiwm value). Quote the lines.
2. Does the en floor (rn_emin) actually feed the diffusivity avm_k? Confirm the order in
   zdftke.F90: en=MAX(en,rn_emin) at ~:469 inside tke_tke, THEN tke_avn computes avm/avt from
   en (:182-184). If NEMO floors en BEFORE avn, our tke_background IS the right analogue.
3. In OUR code (packages/.../vertical_mixing/tke.py + config.py), is tke_background used as the
   en floor the same way (MAX(en,tke_background)) and is mxl_min the length floor NEMO's rmxl_min
   maps to? Any place a 1e-10 floor could underflow, divide-by-zero (sqrt(en), 1/N, l_k), or make
   the quasi-steady Mode-B TKE solve non-finite? Check compute_mixing_lengths and compute_K_from_tke
   for en->0 behavior.
4. Is the non-IWM branch truly unchanged (rn_emin=1e-6, mxl_min=1e-8 = prior default)? Confirm no
   other caller of orca1_zdftke_config or TKEConfig relies on the old iwm-branch tke_background.
5. Any test that pins tke_background=1e-6 or mxl_min=1e-8 for the iwm path that will now break, and
   should it be updated (real) or does it guard something (keep)?
GO/NO-GO on the fix being faithful and safe." > "$LOG" 2>&1
echo "rc=$?"; grep -qiE "Failed to refresh token|refresh token was already used" "$LOG" && echo "AUTH FAIL"
cat "$LOG"; echo DONE
