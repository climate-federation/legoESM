#!/bin/bash
#SBATCH --job-name=codex_wf2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_wf2_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "ROUND 2 review of the SSS-restoring water-flux routing, now at dcdc71dcd.
Round 1 (job 9387241) raised three items, all applied:
  (a) RED: water_flux under the DEFAULT virtual_salt_flux closure double-applied
      restoring (volume via eta/z-star AND the closure's virtual-salt tendency,
      because virtual_salt_flux(freshwater,...) builds its net internally and
      net_freshwater_flux sums `restoring`).  FIX: water_flux now REQUIRES
      --freshwater-closure real_freshwater, and
      real_freshwater_restoring_conflict() exempts exactly water_flux + live_s.
  (b) YELLOW: fw is None with no P-E/runoff/ice, so fw._replace crashed.  FIX:
      build a zero FreshwaterForcing as the carrier.
  (c) guard ORDER now most-specific-first.

Commits under review: e2cbf3a43 (hoist), 8a628d17b (channel), fa12f4ac0
(closure requirement + exemption), dcdc71dcd (fw=None).
NEMO source: /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/src/OCE

Severity-tagged, verdict per item:
 1. Are (a), (b), (c) actually fixed, or fixed in appearance only?  For (a) in
    particular: with real_freshwater selected, trace fw.restoring to EVERY
    consumer and confirm it is applied exactly ONCE.
 2. The exemption in real_freshwater_restoring_conflict is claimed NARROW.
    Enumerate the argument combinations and confirm no other path lifts it.
 3. Items from round 1 you did NOT get to, which I still need answered:
    - global freshwater normalization: with restoring in the water budget, is
      normalize_freshwater now normalizing the RESTORING flux?  NEMO does not
      normalize its erp.  If ours does, name the drift it introduces.
    - the heat term vs existing precip/evap heat-content handling: is
      q_net += F*c_p*sst double-counting anything already applied?
    - z-star: under this channel salinity changes ONLY by dilution.  Does our
      z-star conserve h*S as that requires, and does the top-cell thickness
      used match the one the flux assumed?
 4. The exclusivity test is still a source-string assertion.  Propose a
    concrete stronger test that does NOT need a real eORCA1 mesh.
 5. Anything in these four commits that would produce a WRONG NUMBER rather
    than an error.  That is the class that matters here.
 6. STEEL-MAN: strongest argument this routing will not reduce the common-mode
    Arctic SSS bias (+1.35 tripolar / +1.42 MPAS / +1.48 FESOM2 psu at d30),
    and the measurement that discriminates it from my hypothesis."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
