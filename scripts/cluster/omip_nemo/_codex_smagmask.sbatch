#!/bin/bash
#SBATCH --job-name=cdx_smag
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_smag_%j.out

# Adversarial review of the claim that Smagorinsky masks the prescribed
# equatorial viscosity. This conclusion redirects the whole viscosity campaign
# and a 44h GPU arm is already running on it, so it gets attacked BEFORE more
# is built on top.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json   "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Adversarially review the file
scripts/validate/ocean_fidelity/smag_vs_prescribed_viscosity.py and the
conclusion drawn from it in commit ad04916ba (git show ad04916ba). This is a
JAX ocean model matching NEMO ORCA1.

THE CLAIM. Our equatorial cold tongue is too warm (nino3 bias +3.1 C). We
replaced a uniform lateral viscosity A_h=1e5 with NEMO's prescribed 3-D field
(~5000 m2/s at the equator) and day-30 nino3 moved only 0.07 C. The probe runs
the model's own smagorinsky_viscosity_cgrid on each arm's day-30 snapshot and
reports, area-weighted over |lat|<1: A_smag 5.763e4 vs prescribed 5079, ratio
11.35. We conclude the Smagorinsky closure supplies the viscosity the
prescribed field gives up, that lowering A_h therefore cannot work, and that
the real lever is C_smag. A 44-hour arm at C_smag=1.0 is running on this.

ATTACK THESE SPECIFICALLY:
1. Is A_smag actually ADDED to the prescribed background in the momentum
   equation, or does the code take a MAX, or apply them in different operators
   (e.g. one on the stress tensor and one as a separate divergence), or apply
   Smagorinsky only where some flag is set? Read ocean_pe_latlon_cgrid.py and
   latlon_cgrid_operators.py and say exactly how the two combine. If they are
   NOT simply additive, the 'total viscosity' arithmetic in the commit message
   (1e5+3.04e4 vs 5079+5.76e4) is wrong and so is the conclusion.
2. STAGGERING AND REDUCTION. A_smag is computed at cell centres from a strain
   rate whose shear component is averaged from vertices. The prescribed field
   ahmt_3d/ahmf_3d are NEMO's T- and F-point viscosities. Is the probe
   comparing quantities at the SAME point? It picks ahmt_3d if present. Is
   that the one our A_h background actually corresponds to, given our code may
   apply its background at faces? An apples-to-oranges point comparison would
   inflate or deflate the ratio.
3. LEVEL. The probe reports level 0 (surface) only. The Equatorial
   Undercurrent core is at 50-150 m. Does the conclusion survive at the EUC
   core depth, and does the prescribed NEMO field vary with depth there in a
   way that changes the ratio? A surface-only ratio used to justify a
   claim about the undercurrent is a possible depth mismatch.
4. THE SELF-COMPENSATION CLAIM. We say lowering A_h sharpens the shear which
   raises A_smag from 3.04e4 (control) to 5.76e4 (ramp arm). But those are two
   DIFFERENT RUNS at day 30 that also differ in their whole evolution. Is
   attributing the A_smag increase to the background reduction justified, or
   could it be any other divergence between the runs? Name the measurement
   that would settle it.
5. THE C^2 EXTRAPOLATION. We argue C=3.0/sqrt(11.35)=0.89 brings A_smag down
   to the prescribed level, ignoring that |D| itself responds to the change.
   Given the feedback we just claimed exists, is that extrapolation
   self-contradictory? What would it predict wrongly?
6. Does the probe's grid alignment (align_output_to_mesh in
   so_freshwater_budget.py, tolerance 1e-6 deg) actually guarantee the
   prescribed field and the model field refer to the same cells, including
   near the tripole fold and the cyclic seam where the halo was dropped?

For each finding give severity (CRITICAL/MAJOR/MINOR), file:line, why it is
wrong, and the smallest correct fix. Refuting the conclusion is the most
valuable output -- a 44h GPU arm is running on it. Do not restate what the
code does; only defects." 2>&1
rc=$?
echo "=== codex exit rc=$rc ==="
