#!/bin/bash
#SBATCH --job-name=cdxnino3
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdxnino3_%j.out
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "HEAD: $(git -C $WT log --oneline -1)"
codex exec --sandbox read-only -C $WT \
  "ADVERSARIAL REVIEW OF A DIAGNOSTIC INSTRUMENT, not of model code. Your job
is to break the INFERENCE this instrument is about to support, before I quote
a number from it. A wrong instrument here manufactures a confident wrong
physics conclusion and buys a code change on a false mechanism.

THE DIFF: git diff HEAD~1..HEAD (three files).
  scripts/validate/ocean_fidelity/nemo_zero_step_closure.py
  scripts/validate/ocean_fidelity/global_tracer_content.py  (load_mesh_longitude)
  scripts/cluster/omip_nemo/_nino3_length_split.sbatch

THE STANDING MEASUREMENT I AM TRYING TO EXPLAIN. compare_tendencies_nemo.py
Stage A scores our TKE closure diffusivity K_H against NEMO's avt on a saved
ORCA1 state.  In the nino3 calm columns (no EVD) the ratio is 2.311; nino4 is
1.370.  Five one-variable TKEConfig ablations move it by ~1% between them
(prandtl_mode 0.0000%, tke_mxl_choice 0.6%, etau_mode -0.5%, lc -0.7%; a
sixth, bg_diff_scale, was a misconceived control and proves nothing).  So I
claim the excess is in the core closure K = c_k * l * sqrt(e), and I want to
know WHICH FACTOR.

THE CLAIM I NEED YOU TO ATTACK. In nemo_zero_step_closure.py the zero-step
table drives OUR mixing length with NEMO's OWN restart `en`, and NEMO's own
length inverts as zmxlm = avm_k/(c_k*sqrt(en)) (zdftke tke_avn). I claim
therefore that

    K0_rat = <K_M ours> / <avm_k NEMO>  IS a pure MIXING-LENGTH ratio,

with the TKE amplitude divided out by construction, so reading it in the nino3
box splits the 2.311 into a length factor and an amplitude factor.

QUESTIONS. Quote file:line for every answer; say REFUTED or SUPPORTED.

 1. IS THE RATIO ACTUALLY PURE? K0_rat is a ratio of two VOLUME-WEIGHTED MEANS,
    not a mean of the pointwise ratio l_k/zmxlm. Those are equal only if the
    weights are uncorrelated with the ratio. Does the code print both (there is
    an lk/zmxlm column and an lk/zmxlm median)? Under what correlation would
    the volume-mean ratio and the pointwise median diverge enough to change a
    'length vs amplitude' verdict, and does the nino3 box (strong shallow
    thermocline, thin upper cells) hit that case? If it does, say what to
    report instead.

 2. WHAT CONTAMINATES THE LENGTH SIDE THAT IS NOT LENGTH? Our l_k is seeded by
    N2 computed with OUR EOS (S-EOS or the insitu form, --n2-for-length) on
    NEMO's T/S, while ORCA1 runs TEOS-10. In the equatorial thermocline N2 is
    large and sharply peaked. Bound the error that puts into l_k. If an N2
    difference alone could produce a factor of ~2 in l_k there, then this probe
    does NOT separate 'our length algorithm' from 'our N2' and I must say so.

 3. IS THE TARGET THE SAME PHYSICS ON BOTH SIDES? Stage A scores K_H against
    `avt` (tracer, carries the Prandtl number and NEMO's EVD); the zero-step
    table scores K_M against `avm_k` (momentum, nn_evdm=0). Also the two use
    different NEMO instants: Stage A a 1-hour output record, the closure a
    restart at step 8760. Am I entitled to divide 2.311 by a K0_rat measured on
    a different variable at a different instant? If not, what is the smallest
    change that makes the two comparable -- and is there a K_H-vs-avt zero-step
    variant already in this file I should use instead?

 4. THE NEW REGION MASKS. _region_masks builds latitude bands and lon boxes
    once for both tables. Check: (a) the dateline OR branch for nino4;
    (b) load_mesh_longitude slices glamt on _NATIVE_J/_NATIVE_I, the same frame
    as gphit -- confirm that is the frame `lat` is already on and that the
    restart/mesh shape assert would catch a mismatch; (c) the C0 guard raises
    under 100 columns -- is 100 a floor that can fire on a legitimate mesh, and
    is there a case where a box is non-empty in COLUMNS but empty after the
    `ok` (avm_k off its floor) mask, so it still vanishes from the table
    silently? That last one is the failure the guard was meant to stop.

 5. WHAT WOULD MAKE THE ANSWER UNINTERPRETABLE EITHER WAY? Name the single
    measurement that, if it came back a particular way, would mean neither
    'length' nor 'amplitude' is the right frame for the nino3 excess -- e.g. a
    clamp binding, the surface anchor dominating, or the box being mostly
    interfaces where avm_k sits on its avmb floor and is therefore excluded.
    Tell me what fraction of nino3 interfaces I should expect to survive `ok`,
    and what to do if it is small."
rc=$?
echo "=== CODEX RC=$rc ==="
