#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=eman_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/eman_cdx_%j.out
set +e
WT=${WT:?}; PIN=${PIN:?}
for _CH in "${SLURM_TMPDIR:-}" "/tmp" "/burg-archive/glab/users/pg2328/_codexhome"; do
  [ -n "$_CH" ] || continue
  CANDIDATE="$_CH/codex_home_${SLURM_JOB_ID:-$$}"
  mkdir -p "$CANDIDATE" 2>/dev/null || continue
  cp /burg-archive/home/pg2328/.codex/auth.json "$CANDIDATE/" 2>/dev/null || continue
  [ -f "$CANDIDATE/auth.json" ] || continue
  export CODEX_HOME="$CANDIDATE"; break
done
[ -n "${CODEX_HOME:-}" ] || { echo "FATAL: no usable CODEX_HOME"; exit 90; }
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version || exit 91
cd $WT || exit 92
HEAD_SHA=$(git rev-parse HEAD); echo "PINNED sha: $HEAD_SHA"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 93; }
git diff --quiet && git diff --cached --quiet || {
  echo "PREFLIGHT FAIL: DIRTY TREE at $HEAD_SHA — uncommitted edits would run"
  git status --porcelain | head -20
  exit 94
}

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
FIND THE BUG in this Emanuel (1991) convection implementation. Do not review
style. The evidence below says something is wrong; your job is to locate it in
the code.

FILES:
  packages/atmosphere/legoesm/atmosphere/physics/convection/emanuel.py
  packages/atmosphere/legoesm/atmosphere/physics/convection/_emanuel_mixing.py
  packages/atmosphere/legoesm/atmosphere/physics/convection/_emanuel_downdraft.py
  packages/atmosphere/legoesm/atmosphere/physics/convection/config.py  (EmanuelConfig + its __param_spec__)
  packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py  (how it is called)

ORACLE (read it, this is the reference implementation):
  .physics-validator/emanuel/  -- convect43c.f, Emanuel's CONVECT v4.3c

MEASURED EVIDENCE, all from a single-column RCEMIP1 radiative-convective
equilibrium (fixed SST 300 K, 100 days, dt 600 s, RRTMGP, Morrison
microphysics, last 5 days scored against a SAM CRM reference):

1. A derivative-free search spent 240 evaluations over emanuel's 25 tunable
   parameters, sampling each at the 25 % and 75 % points of its declared range
   and then jointly at random, and NOT ONE candidate beat the defaults. Every
   other scheme improved: dca 6.06 -> 2.13 changing ONE parameter, mass_flux
   7.63 -> 2.52, edmf 7.88 -> 3.23, kain_fritsch 13.64 -> 8.95.
2. emanuel's tuned water budget is P - E = -2.13 mm/day: the column accumulates
   2.13 mm/day of water and is NOT in equilibrium after 100 days. Every other
   scheme closes to |P-E| <= 0.02.
3. Its precipitation is 0.87 mm/day against the CRM's 2.4.
4. Its temperature error over the scored domain got slightly WORSE under
   tuning (7.17 -> 7.30 K) while its humidity error improved slightly.
5. A previous campaign found the same pattern (P 0.869 vs E 2.138 mm/day).

QUESTIONS, each with file:line:

A. Is there a code path by which emanuel's tunable parameters do NOT affect the
   tendencies -- a value read from a module constant instead of the config, a
   branch that ships OFF so the parameters behind it are inert, a cfg field
   shadowed by a local, a parameter consumed only inside a `jnp.where` whose
   predicate is always False in a deep tropical column? Enumerate every one of
   the 25 spec'd parameters and say where it is CONSUMED, or that it is not.
B. The water budget. Trace every water source and sink emanuel returns and
   compare with the oracle: precipitation (PRECIP), the unsaturated downdraft's
   evaporation, the detrained condensate, and what the scheme does with water
   it removes from the updraft. A scheme that removes vapour from the column
   without returning it as precipitation or condensate is the obvious candidate
   for a 2.13 mm/day imbalance -- confirm or refute that specifically.
C. Precipitation of 0.87 against 2.4 mm/day at RCE. In the oracle, what sets
   the conversion of cloud water to precipitation, and is that path faithfully
   reproduced? Check the auto-conversion threshold and the fall speed handling.
D. `enable_unsaturated_downdraft` ships OFF. With it off, what happens to the
   water that the oracle's downdraft would have re-evaporated? Is it dropped,
   or conserved another way?
E. Anything in these three files that silently produces a WRONG NUMBER rather
   than an error, especially a sign, an index-base (Fortran 1-based vs Python
   0-based), a level-ordering assumption (this repo's arrays run TOP-TO-BOTTOM
   with the surface at index -1), or a unit.

Answer as a numbered list of concrete defects with file:line and the one-line
fix. If the parameters ARE all reachable and the budget DOES close, say so
explicitly and name the evidence that convinced you -- an honest "no bug found
in these files" is a valid answer, but then say where else the 2.13 mm/day
could come from.
PROMPT
)" 2>&1
CODEX_RC=$?
echo "CODEX_RC=$CODEX_RC"
[ $CODEX_RC -eq 0 ] || echo "REVIEW DID NOT RUN"
echo ALLDONE
