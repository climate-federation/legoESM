#!/bin/bash
#SBATCH --job-name=codex_wf
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_wf_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Adversarial review of commits e2cbf3a43 (hoist the SSS-restoring input
assembly above the ocean step) and 8a628d17b (--sss-restore-channel
water_flux, NEMO nn_sssr=2 routing).  This is a CONSERVATION-RELEVANT surface
flux change, so weight budget closure and sign errors above style.

NEMO source: /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/src/OCE

Severity-tagged, verdict per item:
 1. THE HOIST. e2cbf3a43 moved _sss_ice / _R_gate / _sss_tgt_step above
    _ocean_step, claiming value-identity because the ice is updated earlier in
    the same iteration, the runoff already fed fw, and the target selection is
    pure.  Verify each.  In particular: does _ocean_step (or any of its SPMD /
    persistent-sharded variants) mutate ice_state, land_mask, or _R?  Is
    ice_state at that point the PRE- or POST-ice-step object, and does moving
    the read change which one is seen?
 2. DOUBLE APPLICATION.  Under water_flux the flux enters fw.restoring AND the
    post-step applier is skipped.  Trace every OTHER consumer of fw.restoring
    and of net_freshwater_flux on this path.  Is there a site that ALREADY
    applies fw.restoring (making the new routing a second application), or one
    that now sees a nonzero restoring for the first time and mis-handles it?
    Note run_omip_core2.py has a line that strips restoring for the KPP
    buoyancy sum -- check I did not break its intent.
 3. SIGN AND UNITS end to end: freshwater_flux [kg/m2/s, +into ocean] ->
    fw.restoring -> net_freshwater_flux -> freshwater_eta_tendency (/rho_0)
    -> eta.  Does a too-salty cell end up FRESHENED and its sea level RAISED?
    Compare against NEMO, where erp is +upward, emp = emp + erp, and
    pssh = pssh - rDt*(emp/rho0 + hdiv).
 4. THE HEAT TERM.  q_net += freshwater_flux*c_p_seawater*sst_C.  Confirm our
    q_net is positive INTO the ocean on this path, that adding to it does not
    double-count a heat-content term the surface-forcing code already applies
    (search for existing precip/evap heat-content handling), and that the
    Celsius reference is right.
 5. CONSERVATION.  With restoring in the water budget, is the global freshwater
    normalization (normalize_freshwater / --no-normalize-freshwater) now
    normalizing the RESTORING flux too?  Should it?  NEMO does not normalize
    its erp.  Name the measurement that would show a spurious global drift.
 6. TRACER SIDE.  Under water_flux the salinity now changes ONLY through
    dilution by the added volume.  Is our z-star dilution actually conserving
    h*S the way that requires, and does the top-cell thickness used match the
    one the flux assumed?
 7. TEST QUALITY.  Are the 7 new tests non-vacuous?  The exclusivity test is a
    source-string assertion -- propose a stronger check that does not need a
    real eORCA1 mesh.
 8. STEEL-MAN: give the strongest argument that this routing will NOT reduce
    the common-mode Arctic SSS bias (+1.35 tripolar / +1.42 MPAS / +1.48
    FESOM2 psu vs NEMO at d30), and name the measurement that discriminates
    your argument from mine."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
