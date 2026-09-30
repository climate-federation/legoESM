#!/bin/bash
#SBATCH --job-name=basecdx
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/basecdx_%j.out
#
# Card audit + tests + MANDATORY codex adversarial review for alpha_tke=1.0.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM
cd $WT || exit 1
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "HEAD: $(git -C $WT log --oneline -1)"
NFAIL=0
run () { echo; echo "########## $1"; shift; "$@"; local rc=$?; echo "=== rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1)); }

run "FIELD-BY-FIELD CARD AUDIT (the systematic finding)" \
  $PY -u scripts/validate/ocean_fidelity/audit_orca1_tke_card.py --iwm \
     --out-json $WT/results/omip_nemo/card_audit/orca1_tke_iwm.json

run "card pins + CLI tests" \
  $PY -m pytest -q tests/unit/test_run_omip_core2_kappa_convention_cli.py \
      tests/unit/test_run_omip_core2_kpp_override.py \
      tests/unit/test_omip_config_parity.py

run "params reachability + dispatch ratchets" \
  $PY -m pytest -q tests/unit/test_params_reachability_audit.py \
      tests/test_dispatch_hardening.py tests/test_param_specs.py 2>&1 | tail -20

echo; echo "########## CODEX ADVERSARIAL REVIEW"
export CODEX_HOME=$(mktemp -d /tmp/codex_home_XXXX)
cp /burg-archive/home/pg2328/.codex/auth.json  "$CODEX_HOME/" 2>/dev/null || true
cp /burg-archive/home/pg2328/.codex/config.toml "$CODEX_HOME/" 2>/dev/null || true
echo "=== CODEX_HOME contents: $(ls -A $CODEX_HOME) ==="

codex exec --sandbox read-only -C $WT \
  "ATTACK THIS NEW BASELINE. Refute it if it is wrong.

CHANGE: \`orca1_zdftke_config\` now pins FIVE more NEMO options together --
tke_mxl_choice 3->4, veros_dz_slots=True, n2_mode='nemo_bn2',
dissipation_discretization='nemo_1p5_split', tke_surface_bc_level='nemo_z0'
-- on top of kappa_convention='veros_sqrte' and alpha_tke=1.0 from earlier
today. It is labelled a NEW BASELINE, not five one-variable edits.

EVIDENCE (job 9405195; our TKE kernel fed NEMO's OWN restart state and en, one
3600 s step, scored against NEMO's avm because ORCA1 sets nn_evdm=0 so EVD
never touches momentum; Antarctic band, 491743 interfaces on every row):
    card as it was (alpha=30 + gaspar)  ~2.6
    + kappa fix                          1.846
    + alpha_tke=1                        0.634   (OVERSHOOTS low)
    + these five                         0.854   Antarctic, 0.930 Arctic
  individually off the alpha-fixed card: diss split 0.739, bn2 0.808,
  veros_dz_slots 0.627.

QUESTIONS. Quote sources for each.
 1. Is each of the five ACTUALLY what ORCA1 runs? Check nn_mxl=2 -> choice 4
    against zdftke.F90's SELECT CASE; check that veros_dz_slots really maps
    dz_cell->e3t and dz_half->e3w in the branch the tripole takes; check that
    'nemo_bn2' is NEMO's rn2 and not a different N2; check the 1.5/0.5 split
    line by line; check nemo_z0 against zdftke.F90:264,403-410.
 2. WHICH OF THE FIVE NEEDS INPUTS THE TRIPOLE PATH MIGHT NOT SUPPLY AT
    RUNTIME? I verified k_profiles.py:811-815 threads dz_surface when
    veros_dz_slots OR nemo_z0 is set, and that it threads the gdept/gdepw
    ladders for nemo_bn2. Is that ACTUALLY reached on --grid tripole
    --tripole-vmix tke --partial-cell, and on --grid mpas which shares this
    card? A 30-hour run that dies at minute two is the failure I most want to
    avoid. Name any path where one of these raises or silently no-ops.
 3. COMBINATION HAZARDS. Do any two of these interact badly -- e.g.
    veros_dz_slots changing the metric slots that nemo_z0's virtual surface
    row assumes, or nemo_1p5_split with the implicit-linearized buoyancy sink
    driving the diagonal non-positive? Is there any regime (deep convection,
    N2<0, a 1-level column, a dry column) where the combination is unstable
    when each alone is fine?
 4. IS 0.854/0.930 ACTUALLY GOOD? The zero-step reference on the same
    interfaces is 0.861 and NEMO's own avm rises 5% over the hour. So we now
    end slightly BELOW NEMO having started slightly below. Is landing at 0.85
    meaningfully different from the 0.861 we started at -- i.e. did the step
    become right, or did it just stop being wrong by cancellation?
 5. WHAT DOES THIS BASELINE BREAK? It changes MPAS too. Check DINO recipes,
    committed scorecards, bit-identity tests, the trainable-parameter
    collector, and anything asserting tke_mxl_choice==3.
 6. WHAT AM I OVER-READING? One hour, one January restart, one band.
    Name what this cannot support -- especially any claim about the day-90
    Southern Ocean mixed layer or the Arctic halocline."
rc=$?
echo "=== CODEX RC=$rc ==="
echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
