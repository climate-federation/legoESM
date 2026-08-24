#!/bin/bash
#SBATCH --job-name=cdx_fwdiag
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_fwdiag_%j.out

# Mandatory adversarial review of the freshwater/ice DIAGNOSTIC tooling. These
# instruments decide what we believe about the Southern-Ocean melt rate, so a
# bug here manufactures a confident wrong physics conclusion (CLAUDE.md:
# "run the review on the DIAGNOSTIC TOOLING, not just the model code").
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json   "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Adversarially review commits eaeb9aa1a, 19bb4d81d and the file
scripts/validate/ocean_fidelity/so_freshwater_budget.py in a JAX ocean model.
Scope: those diffs plus enough surrounding code to judge them. These are
DIAGNOSTIC instruments; a silent bug here produces a confident wrong physics
conclusion, which is worse than a crash.

WHAT THEY CLAIM. We measure the Antarctic (south of 45S) surface freshwater
budget against NEMO ORCA1 and conclude our sea ice melts about 3.2x too fast:
52-63e-6 kg/m2/s sustained over simulated days 2-5, against NEMO's entire
runoff+ice term of 16.67e-6. On that basis we are about to change the
ice-ocean heat exchange coefficient on the production card.

ATTACK THESE SPECIFICALLY:
1. The melt number is computed OUTSIDE these files as the day-to-day
   difference of an area-weighted mean of ice_concentration*ice_thickness over
   lat<-45 with land_mask>0.5, times 1000 kg/m3 / 86400 s. Is that actually a
   freshwater flux per unit OCEAN area comparable to NEMO's empmr residual?
   Consider: cos(lat) weighting on a curvilinear tripole where cell area is
   NOT proportional to cos(lat); ice volume per grid-cell area vs per
   ice-covered area; whether ice EXPORT across 45S or ice ADVECTION
   contaminates a quantity we attribute to melt; and whether snow (ht_s)
   belongs in the water budget.
2. so_freshwater_budget.py's sign control (lines ~105-118). Does the control
   actually discriminate the two conventions, or would it pass under BOTH?
   The subtropical band is evaporative in either storage sign only if the
   negation is applied consistently -- check that the control tests the RAW
   array, not an already-negated one, and that a band with mixed sign cannot
   accidentally satisfy it.
3. The like-for-like comparison in the same file: our P-E is rebuilt at ONE
   INSTANT from record idx, NEMO's is a 5-DAY MEAN. The docstring calls a
   10-20 percent difference inside that mismatch. Is that defensible for a
   band where the synoptic freshwater flux swings sign? Does the record
   selection (argmin |t - day*86400|) actually land inside the NEMO window
   the --nemo-recs slice covers, or can they silently disagree?
4. The per-band [fwbudget-bands] print added in eaeb9aa1a to
   scripts/run/run_omip_core2.py. Verify the stride arithmetic
   (max(1, round(86400/dt))) emits exactly once per simulated day for dt=150,
   that the printed terms use the SAME sign convention as the header claims,
   and that reading these fields does not force a device sync that changes
   what the run computes or silently retraces the step function.
5. _ice_heatcoeff_probe.sbatch (19bb4d81d): is it genuinely ONE VARIABLE
   against the production card, or does any other flag differ? Diff its
   argv against _fwband_probe.sbatch and say exactly which flags differ.
   Its header pre-registers 'coeff 20 melt ~= 16-20e-6 => the 70 is the
   cause'. Is that decision rule falsifiable, or could BOTH branches be
   consistent with the coefficient being irrelevant?

For each finding give severity (CRITICAL/MAJOR/MINOR), file:line, why it is
wrong, and the smallest correct fix. If a claim above is unsupported by the
code, say so plainly -- refuting our conclusion is the most valuable output.
Do not restate what the code does; only defects." 2>&1
rc=$?
echo "=== codex exit rc=$rc ==="
