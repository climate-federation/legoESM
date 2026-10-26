#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_efix_fb
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_efix_fb_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha (base): $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of an UNCOMMITTED working-tree diff (`git diff` in this
repo -- do not look for a commit, it has not landed yet).

CONTEXT. FV3 duo GSPMD dycore. Three earlier per-face `for t in range(6)`
Python loop sites were already replaced with `jax.vmap` face-batching
(dual-reviewed, merged) because under GSPMD face-sharding every device was
computing all six faces and paying a masked-sum all-reduce per traced [t]
read (measured: this was the dominant scaling cost, ~3.6s/step). This diff
does the SAME thing to the last remaining site: the ENERGY FIXER's
`close_out_pt` call inside `fv_dynamics_step`
(packages/core/legoesm/core/fv3_dynamics.py, ~line 966), which only
executes when `consv_te>0`. Also touches
tests/grids/test_fv3_dynamics.py (a new non-vacuity test).

ATTACK IN THIS ORDER, cite file:line, output BLOCKER/MAJOR/MINOR with the
concrete failure scenario and smallest fix, or say where you found nothing.

1. `dtmp` (from `energy_fixer_dtmp_jax`, computed just above the diff) is a
   GLOBAL scalar reduced over ALL SIX faces -- face-invariant by
   construction. The new vmapped branch CLOSES OVER it (does not pass it
   through in_axes) while the old loop branch also used the same `dtmp` for
   every face iteration. Confirm these are actually equivalent under jit
   with GSPMD face-sharding -- does closing over a replicated/global scalar
   inside a `jax.vmap` whose mapped axis IS the sharded face axis cause any
   problem (extra collective, wrong value per shard, anything)?
2. `in_axes=(0, 0, 0)` where the THIRD positional arg is a Python LIST of
   `nq` per-face tracer arrays (`[qq[i] for i in range(nq)]`, each shape
   `(6, ...)`). Confirm this is valid jax.vmap semantics -- does in_axes=0
   broadcast across every element of that list/pytree correctly, giving
   `_close_one` a list of shape-`(...)` slices per mapped call? Any static
   vs traced nq subtlety?
3. Is the new `batched` branch BITWISE-equivalent to the old loop branch
   for every one of pt/pkz/q and for every value nq can take (0, 1, many)?
4. Does the new test `test_batched_matches_loop_consv_fixer` actually
   EXERCISE this branch (consv_te=1.0, k_split=1 so last_step + fixer
   fires) and is it non-vacuous (would it fail if the vmap were wrong,
   e.g. dtmp accidentally per-face instead of global, or the wrong axis
   mapped)?
5. Any regression risk to the consv_te=0 (fixer OFF) certified path, or to
   the `batched=False` default (RULE 3: certified loop path stays default
   everywhere)?
6. Any comment/docstring added that is not true.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_EFIX_FB_DONE
