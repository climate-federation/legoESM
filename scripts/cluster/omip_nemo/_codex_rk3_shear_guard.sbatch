#!/bin/bash
#SBATCH --job-name=cdxrk3
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdxrk3_%j.out
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "HEAD: $(git -C $WT log --oneline -1)"
NFAIL=0
run () { echo; echo "########## $1"; shift; "$@"; local rc=$?; echo "=== rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1)); }

run "vertical-mixing + latlon-cgrid suites (the two files changed)" \
  $PY -m pytest -q tests/ocean/unit -k "tke or vertical_mixing or shear or latlon_cgrid" 2>&1 | tail -25

run "card + CLI suites" \
  $PY -m pytest -q tests/unit/test_run_omip_core2_kappa_convention_cli.py \
      tests/ocean/unit/test_tripole_vmix_cli.py tests/ocean/unit/test_mpas_tke.py

echo; echo "########## CODEX ADVERSARIAL REVIEW"
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME: $(ls -A $CODEX_HOME) ==="

codex exec --sandbox read-only -C $WT \
  "REFUTE THIS IF IT IS WRONG. I removed a deliberate safety guard.

WHAT I CHANGED (git diff HEAD):
 (a) ocean_model_latlon_cgrid.py -- DELETED the ValueError that refused
     tke_shear_production in ('nemo_burchard','nemo_face_native') unless
     outer_integrator was in ('leapfrog','nemo_mlf').
 (b) k_profiles.py -- when there is NO before state and the shear form is
     face_native/burchard, feed the NOW faces as BOTH the now and before
     factors instead of leaving them None.

MY ARGUMENT: NEMO's RK3 step calls
    stprk3.F90:165   CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )
passing Nbb as BOTH Kbb and Kmm -- and the leap-frog form ( Nbb, Nnn ) is the
commented-out line 164 directly above. So inside zdf_sh2 the product
uu(Kmm)*uu(Kbb) is a square on ONE time level and no before state is
consulted. Our ORCA1 oracle is a key_RK3 build. Hence the guard was blocking
the configuration that is FAITHFUL to our oracle, and now x now is not an
approximation here but exactly what NEMO does.

QUESTIONS -- quote sources:
 1. Verify stprk3.F90:164-165. Is line 165 really live and 164 commented? Is
    zdf_phy's Kmm argument forwarded unchanged to zdf_sh2 (zdfphy.F90:268)?
    Does anything else in the RK3 path re-set Kmm before zdf_sh2 runs?
 2. Is our ORCA1 oracle ACTUALLY key_RK3? Check the build/cpp keys for the
    RUN_TRD2/RUN_REF configuration, not just a comment claiming it.
 3. Is my k_profiles edit correct? I reused the `_stag` shape test to decide
    the state is face-staggered. Does that match the checks the leap-frog
    branch does above it (it raises on a pre-centred state)? Should the
    no-before branch raise the same way rather than silently doing nothing
    when the state is NOT staggered? Right now it silently leaves the faces
    None in that case, which will hit the kernel's own requirement error --
    is that acceptable or a silent-no-op?
 4. WHAT DID THE GUARD PROTECT THAT I HAVE NOW EXPOSED? Name any path where
    face_native under forward_euler/ab2/rk3 is genuinely wrong or unstable --
    e.g. an AB2 outer step where 'now' is not the level zdf_sh2 should see,
    or an SSP-RK3 substep where state.u is mid-stage rather than Nbb.
    THIS IS THE QUESTION I MOST WANT ANSWERED: is 'now' in our RK3 the same
    thing as NEMO's Nbb, or is it a mid-stage value?
 5. I deliberately did NOT touch the sibling guard on
    tke_n2_time_level='nemo_before', arguing rn2b is a genuinely different
    TRACER time level that RK3 does not collapse. Check that against
    stprk3/zdftke -- is rn2b also equal to rn2 under RK3, in which case I
    should have relaxed both, or is my distinction right?
 6. Anything else I have over-read."
rc=$?
echo "=== CODEX RC=$rc ==="
echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
