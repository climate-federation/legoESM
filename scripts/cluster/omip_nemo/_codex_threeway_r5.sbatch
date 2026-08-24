#!/bin/bash
#SBATCH --job-name=codex_3way5
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_3way_r5_%j.out

# Round 5: verify the round-4 fixes (commit d6e8a1063 and its parent) and
# review the NEW probe global_tracer_content.py that produced the MLE
# salt-conservation verdict.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Two-part review.

PART A: verify round-4 fixes in
scripts/validate/ocean_fidelity/compare_three_way_nemo.py and
tests/validate/test_compare_three_way_nemo.py (commits 9a3eda39c, d6e8a1063).
R4-1: --smooth-radius-deg 0 previously bypassed validation via truthiness.
  Claimed fix: _validate_args called immediately after parse_args, rejects 0,
  all use sites compare 'is not None', functional tests call the validator
  with 0/-5/270/nan/inf.  Verify no remaining truthiness use of
  smooth_radius_deg anywhere in the script.
R4-2: near_land screening caveat now in report.json (near_land_caveat) and
  --help no longer says 'real land/sea classification'.  Verify.
R4-3: main-path tripwires: _validate_args before mkdir; 'ocean =
  build_ocean_mask(coverage' string test.  Would each actually FAIL on the
  revert it targets?

PART B: adversarial review of the NEW probe
scripts/validate/ocean_fidelity/global_tracer_content.py and
tests/validate/test_global_tracer_content.py.  This probe just produced a
load-bearing verdict: the MLE arm's total salt is conserved (drift -1.3e-5
relative over 30 d) while its SURFACE freshens by 4.5 psu, so the defect was
classified 'vertical redistribution, not a conservation bug'.  Attack that
verdict's instrument:
 - native frame slicing [0:331, 1:361] of a (332,362) eORCA1 array: is the
   north-fold ghost row really row 331 and are BOTH overlap columns 0 and 361
   (not 360/361)?  If the frame is off by one, what does it do to the number?
 - level-last transpose: T is (332,362,75), mesh e3t_0 is (75,332,362).
 - z-star dilation (H+eta)/H applied uniformly per column: is that the right
   volume correction for this model's z-star, and does using e3t_0 (reference
   partial-cell thickness) with snapshot eta double- or under-count?
 - tmask from the mesh vs the snapshot's own wet definition (land_mask,
   |z|<H_bathy): could a mismatch hide a real salt leak in cells one mask
   counts and the other does not?
 - is a -1.3e-5 relative drift over 30 days actually NEGLIGIBLE for this
   verdict, or is it 100x the control's drift (+4e-7) and itself evidence of
   a small non-conservation in the MLE operator?  Quantify what -1.3e-5 of
   4.67e19 psu m3 means in psu applied to the 0-50m bin.
 - the depth-bin attribution uses gdept_1d, ignoring partial cells: can that
   move salt between bins spuriously?
Severity-tagged findings, then a verdict table."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
