#!/bin/bash
#SBATCH --job-name=codex_mlefix
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_mlefix_%j.out

# Mandatory adversarial review of the MLE vertical-branch fix (physics change,
# both grids) before the A/B relaunch.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Adversarial review of the Fox-Kemper MLE vertical-branch fix, commits
465823b83 (fix) and 12aad60bf (test rewrite).  Files:
packages/ocean/legoesm/ocean/physics/lateral_mixing/mle_latlon_cgrid.py,
packages/ocean/legoesm/ocean/physics/lateral_mixing/mle_mpas.py,
tests/ocean/unit/test_mle.py, tests/ocean/unit/test_mle_mpas.py.
Oracle: /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/src/OCE/TRA/tramle.F90
(zu_mle = dk[psi_uw], zv_mle = dk[psi_vw], zw_mle = -di[psi_uw]-dj[psi_vw],
handed to the advection scheme as one divergence-free transport).

Background: the horizontal-only operator conserved the global tracer sum but
pumped tracer at transport-convergence cells (-54 psu river-plume extremes in
30 d of the d90 A/B).  The fix reconstructs W from continuity on the MASKED
horizontal transports: W(k)-W(k+1) = -div_h(k), W=0 at the sea floor, surface
interface hard-zeroed, centered tracer at interior W-interfaces.

Attack list:
1. SIGN/ORIENTATION: verify W positive-toward-surface is consistent with the
   flux difference Fz[k]-Fz[k+1] used in the cell divergence, on BOTH grids.
   An inverted W doubles the spurious term instead of cancelling it -- state
   the discrete identity that proves the 3-D divergence vanishes per cell for
   a uniform tracer, and check the code implements exactly it.
2. The surface hard-zero: W(0) is set to 0 after the reverse cumsum.  Its
   pre-zero residual is claimed to be column-telescoping roundoff.  Is there
   any configuration (masked step faces, partial cells, MLD deeper than the
   column, mu clipped at gdepw/H>1) where W(0) is NOT roundoff -- i.e. the
   column sum of div_h is materially nonzero -- so zeroing it silently
   deletes real transport and breaks per-cell continuity?  Quote lines.
3. Centered T at interior interfaces concatenated with zero end columns:
   check the interface/level indexing is not off by one (T_w[k] must sit
   between levels k-1 and k where W[k] lives).
4. jnp.flip(cumsum(flip)) under reverse-mode AD and under jit: any
   performance or correctness landmine vs a direct cumsum from the bottom?
5. Dry cells: FzT at interfaces adjacent to dry cells -- can a nonzero W
   cross a bottom step into a dry cell's interface on the partial-cell
   ladder?  Ours multiplies horizontal transports by masks BEFORE building W;
   is the resulting W guaranteed zero at and below the first dry interface?
6. The test rewrite: are the retired expectations really wrong under the
   correct operator (two-column analytic argument in the commit message), or
   did the fix change behaviour the tests were right to pin?  Is the new
   stratification metric (mean surface-minus-lower-ML tendency > 0) a sound
   invariant of the FK closed cell with centered fluxes?
7. NEMO comparison: NEMO hands the MLE transport to its ADVECTION scheme
   (FCT) rather than using standalone centered fluxes.  Name the practical
   consequences of our standalone centered choice (dispersion/overshoots at
   fronts?) and whether any NEMO-side limiter exists that we silently lack.
Severity-tagged findings + verdict table."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
