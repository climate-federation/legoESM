#!/bin/bash
#SBATCH --job-name=kappacdx
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/kappacdx_%j.out
#
# Tests + MANDATORY codex adversarial review for the K-from-TKE amplitude fix
# (orca1_zdftke_config now pins kappa_convention="veros_sqrte").
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM
cd $WT || exit 1
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "HEAD: $(git -C $WT log --oneline -1)"
NFAIL=0
run () { echo; echo "########## $1"; shift; "$@"; local rc=$?; echo "=== rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1)); }

run "new kappa-convention CLI + card tests" \
  $PY -m pytest -q tests/unit/test_run_omip_core2_kappa_convention_cli.py

run "the suites that consume the card I just changed" \
  $PY -m pytest -q tests/unit/test_run_omip_core2_kpp_override.py \
      tests/unit/test_run_omip_core2_freshwater_closure_cli.py \
      tests/unit/test_omip_config_parity.py \
      tests/test_dispatch_hardening.py tests/test_validate_strict_coverage.py

run "TKE closure unit tests (the amplitude branch lives here)" \
  $PY -m pytest -q tests/ocean/unit -k "tke or vertical_mixing or kappa" 2>&1 | tail -30

echo; echo "########## CODEX ADVERSARIAL REVIEW"
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME=$CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="

codex exec --sandbox read-only -C $WT \
  "CHALLENGE THIS FIX. I need it refuted if it is wrong, not confirmed.

WHAT I CHANGED (uncommitted or just-committed, see git diff/log):
\`orca1_zdftke_config\` in scripts/run/run_omip_core2.py now pins
\`kappa_convention=\"veros_sqrte\"\`, where the TKEConfig default is
\`gaspar_sqrt2e\`.  I also added a \`--tke-kappa-convention\` CLI lever
threaded through build_tripole_vmix_config / build_tripole /
_validate_tke_card_grid, and tests/unit/test_run_omip_core2_kappa_convention_cli.py.

MY CLAIM, which is what I want attacked:
NEMO computes avm = MAX( avtb, rn_ediff * zmxlm * en^(1/2) ) -- see
src/OCE/ZDF/zdftke.F90 line 150 and the tke_avn header at 553 -- so the sqrt
takes en, not 2*en, and the factor 2 lives in the length
zmxlm = SQRT(2*en/rn2) at line 651.  Our compute_K_from_tke (packages/ocean/
legoesm/ocean/physics/vertical_mixing/tke.py around 1607) under
'gaspar_sqrt2e' computes c_k*l_k*sqrt(2*max(e,tke_background)), and the
tke_mxl_choice 3/4 branch (same file, around 700) builds its length from
sqrt(2)*sqrt(e)/N.  So the default double-counts sqrt(2) on the branch this
card selects, and 'veros_sqrte' (c_k*l_k*sqrt(max(0,e))) is NEMO's form.

QUESTIONS. Quote both sources for each.
 1. Read zdftke.F90 tke_avn IN FULL. Is avm really rn_ediff*zmxlm*SQRT(en)?
    Is there any other factor -- a 2, a sqrt(2), an e3w division, a pdlr --
    between en and avm that I have missed? Does avt = pdlr*avm or something
    else? Check nn_pdl=1.
 2. Is 'veros_sqrte' EXACTLY NEMO's amplitude, or only approximately? NEMO
    floors en at rn_emin=1e-6 BEFORE the sqrt; veros_sqrte does
    sqrt(max(0,e)) and returns 0 for e<=0. Where in the legoESM code is the
    carried TKE floored, and can K reach 0 on this path in a way NEMO cannot?
    If our TKE can be negative (the 'negative-TKE debt' the code mentions),
    does veros_sqrte introduce a discontinuity or an AD problem that
    gaspar_sqrt2e did not have?
 3. THE SIGN OF THE CONSEQUENCE. I am claiming this fix REDUCES our vertical
    diffusivity by ~1.41x and therefore SHALLOWS our mixed layer. Our
    Southern Ocean mixed layer at day 90 is 19 m TOO DEEP (89.6 m vs NEMO
    70.4 m) and our Arctic halocline is destroyed by over-mixing. Is the sign
    right? Is there any feedback in the prognostic TKE solve (smaller K ->
    less shear production? -> different equilibrium en) that could make the
    net effect on K, or on the mixed layer, go the OTHER way or be much
    smaller than 1.41x?
 4. IS THE CARD THE RIGHT PLACE? kappa_convention is a TKEConfig field with a
    global default. Changing the ORCA1 card changes tripole AND mpas (both
    reach build_tripole_vmix_config). Does anything else in the repo depend
    on the ORCA1 card carrying gaspar_sqrt2e -- a committed baseline, a
    bit-identity test, a scorecard threshold, the DINO recipe? Grep for it.
    Would any test go red for the right reason or the wrong reason?
 5. THE TEST. tests/unit/test_run_omip_core2_kappa_convention_cli.py asserts
    the legacy/NEMO ratio is exactly sqrt(2) on a synthetic column. Is that
    fixture actually free of the kappaM_min floor and kappaM_max ceiling, and
    of the Prandtl chain, so the assertion cannot pass for the wrong reason?
    Does it fail if the pin is removed? Name anything vacuous.
 6. WHAT I HAVE OVER-READ. The Stage A measurement I am citing (4.81x with
    gaspar vs 3.25x with veros_sqrte, Arctic calm columns, commit 39ce0701c)
    compares our TKE-only K_H against NEMO's history avt, which also carries
    EVD and zdfiwm. Does that contamination change the DIRECTION of my
    conclusion, or only its magnitude? And a 3.25x residual remains after
    this fix -- what is the single most likely cause of THAT, given the
    correlation stays at 0.63 in every configuration?"
rc=$?
echo "=== CODEX RC=$rc ==="
echo; echo "=== ALL DONE (failed test stages: $NFAIL) ==="
exit $((NFAIL > 0))
