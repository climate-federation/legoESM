#!/bin/bash
#SBATCH --job-name=codex_gmop
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/codex_gmop_%j.out

set -u
WT=/burg-archive/glab/users/pg2328/legoESM
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="

codex exec --sandbox read-only -C $WT \
  "Adversarial review of commit 49aa73766 ('expose NEMO's lateral-mixing
operator to the CORE-II driver') and the claims in its message.

THE CLAIM.  NEMO ORCA1 runs ln_traldf_lap/ln_traldf_iso=.true.,
ln_traldf_triad=.false., ln_traldf_msc=.true., and adds the eddy-induced
transport to the advecting velocity (LDF/ldftra.F90 ldf_eiv_trp, called by
traadv.F90) so the bolus rides the monotone FCT limiter.  Our OMIP tripole runs
slope_scheme='centered' + gm_bolus_advection='centred' (an unlimited standalone
centred flux, no akz).  The faithful forms exist in the model but had zero
references in scripts/run/run_omip_core2.py.
NEMO source: /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/src/OCE
NEMO namelist: /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/namelist_cfg

Review, severity-tagged, verdict per item:
 1. Verify the namelist and Fortran claims by reading them.  In particular: is
    'nemo_iso_lap' in our GMRediConfig actually the analogue of NEMO's
    ln_traldf_iso standard operator, and does it really carry the
    ln_traldf_msc akz term?  Read
    packages/ocean/legoesm/ocean/physics/lateral_mixing/ and the K33/akz path
    in ocean_model_latlon_cgrid.py.  Quote lines.
 2. Is the through_fct gate really (nemo_iso_lap AND through_fct)?  If a user
    passes --gm-slope-scheme nemo_iso_lap WITHOUT --gm-bolus-advection
    through_fct, what runs -- akz operator + centred bolus?  Is that a
    coherent, physically meaningful combination, or a trap that should also
    be guarded?
 3. The new override does _gm_base = _ovr.get('gm_redi', config.flat_get('gm_redi')).
    Is flat_get the right accessor here, and is `config` at that point the
    object whose gm_redi actually reaches the model?  Could the override be
    silently dropped or double-applied?
 4. Does the change compose correctly with --gm-treguier (the branch above it)
    and with --no-gm-redi?  Check ordering: the operator block runs AFTER the
    if/elif chain -- can _ovr['gm_redi'] be None there and slip past the guard?
 5. The commit claims 'no existing run changes' because both flags default to
    None.  Verify.
 6. Are the 8 new tests non-vacuous?  Specifically
    test_the_pair_the_model_honors_is_the_pair_we_expose -- does it actually
    fail if someone renames the model-side gate, or is it asserting only on
    the driver's own constants?
 7. What is the STRONGEST argument that switching the OMIP tripole to
    nemo_iso_lap + through_fct will NOT improve fidelity to NEMO, despite the
    namelist match?  Name the measurement that would settle it."
rc=$?
echo "=== CODEX RC=$rc ==="
exit $rc
