#!/bin/bash
#SBATCH --job-name=cdx_4c
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_4c_%j.out

# Adversarial review of (a) how the corrected-forcing arm's result is being
# read, and (b) the design of the arm that is about to spend ~30 GPU-hours on
# the fourth cell of the factorial.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json   "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Adversarially review an ocean-model fidelity campaign. This is a JAX ocean
model being matched term by term against NEMO ORCA1 forced with CORE-II normal
year. Do NOT summarise what the code does. Find defects in the CLAIMS, in the
INSTRUMENT, and in the DESIGN of the run about to be launched. Rank findings
CRITICAL / MAJOR / MINOR and say for each what measurement would settle it.

=== PART 1: how a finished arm's result is being read ===

Two runs, both tripole eORCA1 L75, dt=150 s, cold start from the WOCE January
monthly IC, scored against NEMO ORCA1's own output on the shared mesh:

  control  scripts/cluster/omip_nemo/  (arm 'signedn2_d30', raw CORE-II winds)
  arm      scripts/cluster/omip_nemo/_ab_modforcing_d30.sbatch (job 9450607),
           which passes --forcing-path .../core2_nyf_mod and is otherwise
           claimed byte-identical to the control.

MATCHED day-20 comparison (same instrument, same NEMO month, same boxes):
  nino3 SST bias    +2.50 -> +1.49
  nino3.4 SST bias  +2.04 -> +1.16
  tropics SST bias  +0.48 -> +0.08
  SH midlat         +0.81 -> +0.38
  NH midlat         -0.59 -> -0.75
  global SST rmse   0.966 -> 0.808

Day 30 for the arm: global SST rmse 0.928, bias +0.052, nino3 bias +2.05,
NH_midlat bias -1.02, Arctic SSS rmse 2.01, Arctic MLD bias +18.8 m.

The pre-registration in the arm's own header required, at day 30: nino3 below
+2.0 to CONFIRM; and separately the Equatorial Undercurrent core at 200E to
reach at least 0.42 m/s, because the undercurrent is the claimed mechanism.
MEASURED EUC core speed (max over 0-400 m, |lat|<=1 box mean):
  lon 200E: ours 0.116 m/s at 133 m, NEMO 0.496 m/s at 133 m
  lon 220E: ours 0.125, NEMO 0.546
  lon 240E: ours 0.134, NEMO 0.260
On the RAW forcing the same instrument reported roughly 0.34 at 200E, i.e. the
undercurrent appears to have got WEAKER under stronger trades.

QUESTIONS FOR YOU, and be hostile:
 1. Was that arm actually one-variable? Read _ab_modforcing_d30.sbatch and the
    control's flags and say what else differs. The run log
    omip_nemo/modfrc_9450607.out does NOT echo the resolved forcing path, so
    say how you would prove from the artifacts which cache it used.
 2. Is comparing the arm's EUC number against a remembered 0.34 legitimate?
    Check whether both numbers come from the same instrument invocation, the
    same day, the same depth range, and the same reduction. If they do not,
    the 'EUC got weaker' claim is a confound, not a result.
 3. Stronger easterly trades weakening the equatorial undercurrent at day 30
    of a cold start: is that physically expected (spin-up, Yoshida jet, the
    surface-intensified westward branch masking the subsurface core) or is it
    a defect signature? Name the diagnostic that discriminates.
 4. The pre-registration is being read as 'SST CONFIRMED, EUC REFUTED, so the
    mechanism story is half wrong but the fix is kept because it is a
    correctness fix, not a tuning knob.' Attack that reading.

=== PART 2: the run about to be launched ===

scripts/cluster/omip_nemo/_ab_ahprof_smag1_modfrc_d90.sbatch completes a 2x2
factorial: corrected winds x ORCA1's prescribed lateral-viscosity file
(nn_ahm_ijk_t=-30 reading eddy_viscosity_3D.nc, 20000 m2/s midlatitude ramping
to 1000 at the equator) with Smagorinsky C reduced 3.0 -> 1.0. The other three
cells are signedn2_d30, modfrc_d30 and _ab_ahprof_smag1_d90.sbatch (job
9444637, which ran 90 days stably and got EUC 0.926 at 240E against NEMO
1.018).

 5. Is this actually a factorial? The 'viscosity' axis changes TWO things at
    once, the A_h field and the Smagorinsky coefficient. Say whether that
    breaks the one-variable rule and, if so, which of the four cells becomes
    uninterpretable.
 6. Are the pre-registered thresholds falsifiable, or has each been set where
    it cannot fail? Attack each threshold.
 7. Cost check: this is ~30 GPU-hours. Name any cheaper offline test on the
    EXISTING day-30 and day-90 snapshots of the three finished cells that
    would answer the same question, and say why it would or would not.
 8. Stability: the Smagorinsky-off twin of the viscosity axis grew equatorial
    max|u| monotonically to 6.21 m/s by day 60. The Smag=1.0 twin did not.
    Adding a 33 percent stronger trade wind on top: what is the failure mode
    to watch, and is a max|u| tripwire under 2.5 m/s the right instrument?

=== PART 3: the code diff ===

git diff HEAD in this repo adds a provenance print to
packages/ocean/legoesm/ocean/forcing/core2.py (which cache resolved, plus a
nino3-box mean wind speed that identifies it) and a new test
tests/ocean/unit/test_core2_corrected_cache_default.py. Review both. In
particular: does the new _nino3_wind_speed helper do a correct area-weighted
box mean on this grid, does it handle a longitude convention other than
0-360, and can the new tests fail if the behaviour they assert is removed?
Also check the earlier commit 24a25b286 that made the corrected cache the
default: it shipped with no test and a silent-fallback path.
" 2>&1 | tail -220
echo "=== codex rc=$? ==="
echo DONE
