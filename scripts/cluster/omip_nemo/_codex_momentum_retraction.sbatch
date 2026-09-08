#!/bin/bash
#SBATCH --job-name=cdxmomret
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdxmomret_%j.out
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "HEAD: $(git -C $WT log --oneline -1)"
codex exec --sandbox read-only -C $WT \
  "YOUR LAST REVIEW (9417284) MADE ME RETRACT TWO CLAIMS. This one asks you to
break the REPLACEMENT claim before it becomes the new campaign direction.
Attack it; do not confirm it.

WHAT YOU ALREADY REFUTED, and I accepted:
 * that K_M/avm_k = 0.999 in nino3 licenses 'the closure amplitude and length
   are right' -- it sits on 1.78% of the box by VOLUME.
 * that 2.311 (K_H vs avt, Stage-A state, calm mask, rms) could be divided by a
   zero-step K_M/avm_k (different variable, state, mask AND reduction).
Both are retracted in the git log. Coverage is now printed per box; --n2-eos-form
now reaches compute_buoyancy_frequency_nemo_bn2; a box emptied by a mask is fatal
rather than silently dropped.

THE NEW CLAIM, from prandtl_split_report in compare_tendencies_nemo.py -- ONE
state, ONE calm mask, the SAME rms reduction that produced the 2.311:

    nino3/calm  whole column   K_H/avt 2.311   K_M/avm 0.635
    nino3/calm  above 300 m    K_H/avt 1.054   K_M/avm 0.473
    eqpac/calm  whole column   K_H/avt 2.500   K_M/avm 0.807
    eqpac/calm  above 300 m    K_H/avt 0.999   K_M/avm 0.472
    our Prandtl on its 10.0 clamp on 64% of nino3 interfaces above 300 m,
    NEMO's median Prandtl there 2.14.

I now claim: (A) the 2.3x over-mixing is a DEEP signal and there is NO tracer
over-mixing in the equatorial thermocline; (B) the real equatorial defect is
that our MOMENTUM viscosity is ~2x too small above 300 m; (C) PLAUSIBLE only --
one mechanism, our vertical shear being too small, predicts both (low shear ->
low TKE production -> K_M small; low shear -> Ri large -> Pr pinned at 10),
and the card's tke_shear_production='squared_centered' vs NEMO's face-native
zdfsh2 is the documented candidate.

QUESTIONS. file:line for everything; say REFUTED or SUPPORTED per item.

 1. IS (A) AN ARTIFACT OF rms ON A HEAVY TAIL? The whole-column rms ratio is
    dominated by whichever field has the largest values. Below 300 m NEMO's avt
    is near its background while ours may not be. Show whether the 2.311 is (i)
    a genuine deep over-mixing, or (ii) a few outlier interfaces, or (iii) an
    artifact of my floor detection keeping 97.6% of interfaces when it should
    have excluded the background. Note the detected 'avtb' is 1.401e-07 while
    the namelist rn_avt0 is 1.2e-5 -- explain that discrepancy, because if my
    floor mask is wrong then BOTH the whole-column and the shallow rows are.

 2. DOES ln_zdfiwm BREAK THE DEEP COMPARISON OUTRIGHT? ORCA1 sets
    ln_zdfiwm=.true. Our TKE closure does not model internal-wave mixing at
    all. Where does zdf_iwm deposit, how large is it against avt at 1000-4000 m
    in the equatorial Pacific, and does it make the deep K_H/avt ratio
    meaningless in EITHER direction? If NEMO's deep avt is LARGER because of
    IWM, my 2.3x deep excess is an UNDERSTATEMENT, which changes the story
    again -- work out the sign.

 3. IS (B) MEASURING WHAT I THINK? K_M comes from tke_vertical_mixing on
    NEMO's T/S/u/v with Mode-B (tke_old=None, n_iterations=3) -- an
    equilibrium, not NEMO's prognostic en. NEMO's avm at the same record is a
    TIME MEAN over the hour (cell_methods='time: mean'). Is a 0.47 ratio
    attributable to the closure, or could the Mode-B equilibrium alone produce
    it? What is the cheapest control that separates 'our K_M is too small' from
    'Mode-B is not NEMO's en'? The Stage-A2 Mode-A path in the same file feeds
    NEMO's own restart en -- can it be pointed at K_M/avm above 300 m instead?

 4. THE SHEAR MECHANISM (C). Read the squared_centered shear in
    packages/ocean/legoesm/ocean/physics/vertical_mixing/ and NEMO's
    zdfsh2.F90. In the equatorial upper ocean specifically -- NOT near coasts,
    where the (2-umask*umask) doubling lives -- would our form give a SMALLER
    shear than NEMO's, and by roughly how much? If the answer is 'no
    difference away from coasts', mechanism (C) is refuted and I need another
    one for a factor of 2. Also: run_stage_a interpolates uoce/voce to T
    points before differencing vertically. Does that horizontal averaging
    reduce the vertical shear of the EUC, and is that a defect of the PROBE
    rather than of the model?

 5. WHAT WOULD YOU MEASURE NEXT, given that the thermocline tracer diffusivity
    is now RIGHT and the 26 m too-deep Z20 therefore has to come from
    somewhere else? Name the one offline measurement on the existing NEMO
    output that best discriminates equatorial upwelling / wind stress /
    momentum viscosity, and say what value would confirm and what would refute."
rc=$?
echo "=== CODEX RC=$rc ==="
