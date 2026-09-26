#!/bin/bash
#SBATCH --job-name=cdx_kprof
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_kprof_%j.out
#
# Adversarial review of the instrument BEFORE two GPU arms are judged with it.
# The instrument decides what we believe, so a defect here manufactures a
# confident wrong physics conclusion.
#
# AUTH: do NOT copy ~/.codex/auth.json into a scratch CODEX_HOME.  Codex
# rotates its OAuth refresh token and the copy is invalidated the moment
# anything refreshes the real one, which exits 0 having reviewed nothing.
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/legoESM
LOG=$WT/omip_nemo/cdx_kprof_body_${SLURM_JOB_ID}.txt
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
"Adversarially review a DIAGNOSTIC added to a JAX ocean model that is being
matched, term by term, against NEMO ORCA1.  Find defects.  Do not summarise
what the code does.

WHAT WAS ADDED (commit d2d30f93e, on branch fix/omip-gm-treguier-one-variable):

1. scripts/run/run_omip_core2.py: a --kprofile-snapshots flag.  When set, at
   every snapshot the driver calls model.tendencies(state, surface_forcing=sf,
   dt=dt) and stores the returned K_v and A_v in the snapshot .npz as
   K_H_diag / K_M_diag.  See _kprofiles() and the two _save_snapshot call
   sites in the OMIP forcing loop.

2. scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py: an
   optional --legoesm-snapshot arm that reduces those arrays in the same box
   (|lat|<=2, 200-260E), the same 5-65 m window and the same per-column median
   as the NEMO side, and prints the ratio.

3. tests/unit/test_snapshot_kprofiles.py and
   tests/ocean/unit/test_equatorial_diffusivity_oracle.py.

THE CLAIM THIS INSTRUMENT IS MEANT TO SETTLE.  NEMO's published grid_W avm/avt
in the equatorial cold tongue give a closure-active (5-65 m) Prandtl median of
1.211 and avt median 1.57e-3 m2/s.  The standing hypothesis is that our TKE
closure mixes momentum too well and heat too poorly (undercurrent 4x too weak
with the right pressure gradient; thermocline at the right depth but 2.3x too
sharp; cold tongue warming when NEMO and an independent model both cool).

QUESTIONS I MOST WANT ATTACKED, in this order:

A. IS K_v/A_v ON THE TENDENCY THE SAME QUANTITY AS NEMO'S avt/avm?  Check
   where the latlon C-grid tendency sets K_v/A_v
   (packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py) and what
   the TKE path in
   packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py returns.
   Specifically: are these the diffusivities the implicit vertical solves
   actually consume, or a pre-limiter / pre-background / pre-convection value?
   Does the run's card add anything AFTER this point -- internal-wave mixing
   (--iwm), double diffusion, convective enhancement, an additive molecular
   background -- that NEMO's published avt would include and ours would not?
   A missing additive term would make our numbers look too small and would
   CONFIRM the hypothesis for the wrong reason.

B. STAGGERING AND LEVEL CONVENTION.  Ours are asserted to live at interfaces
   (nlev-1) and are labelled with midpoints of z_center_ref; NEMO's are on
   depthw.  Is that the same set of depths, or off by half a cell?  A previous
   error in this campaign compared a face maximum against a cell-centre mean
   and called the agreement 1.2%.

C. IS THE DIAGNOSTIC EVALUATED AT THE STATE IT CLAIMS?  It runs AFTER the
   step loop for the final snapshot and inside the loop for the periodic ones,
   using the last `sf` built.  Is `sf` at that point the forcing for the step
   just taken or the next one, and does the closure's K depend on the CARRIED
   TKE (so that a fresh tendency call reproduces what the step used) or on
   something advanced during the step (so that it does not)?

D. CAN THE TESTS FAIL?  Each new test should be shown to fail if the feature
   is removed.  Point at any that would pass against a broken implementation,
   and at any assertion that pins a convention the code merely happens to
   have.

E. THE PRE-REGISTRATION in scripts/cluster/omip_nemo/_ab_mxl_nemo_faithful_d30.sbatch
   claims to isolate ONE variable (--tke-mxl-choice 3 vs 4, said to be NEMO's
   nn_mxl=3 vs nn_mxl=2).  Verify from
   packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py that choice 4
   is nn_mxl=2 and that nothing else changes with it.  Also state whether the
   CONFIRMED/REFUTED thresholds could both be satisfied, or neither.

Answer with a numbered list of concrete defects, each with file:line and what
it would make us wrongly conclude.  If a question above has no defect, say so
in one line and move on." > "$LOG" 2>&1
rc=$?
echo "=== codex rc=$rc ==="
# The prompt itself contains none of these tokens, so a hit is the tool's own.
grep -qiE "Failed to refresh token|refresh token was already used|not logged in" "$LOG" \
  && echo "FATAL: codex auth failed -- this review DID NOT RUN"
cat "$LOG"
echo DONE
