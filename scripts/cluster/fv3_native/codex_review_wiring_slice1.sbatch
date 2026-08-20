#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_wire
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_wire_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of the commit that wires the certified FV3 duo-grid
JAX dycore into the model as a selectable discretization ("fv3_duo").
Scope: the most recent commit whose subject starts "feat(fv3): the
duo-grid JAX dycore is callable as a model" (git log --oneline -5;
diff it). The dycore itself (fv3_dynamics.py and below) is certified
against the pinned Fortran and is NOT re-reviewed here; the wiring is.

ATTACK IN THIS ORDER, file:line each. A wiring bug that runs the wrong
physics or silently accepts an unsupported config is worse than a
crash.

1. THE REFUSAL WALL (component_factory.py). Slice 1 supports ONLY dry,
   physics-off, fp64, single-process, km in {5,10}. Enumerate every
   config field that could smuggle physics or moisture past the checks
   (radiation/convection/microphysics/turbulence/gwd are checked -- what
   about surface fluxes, land, ocean coupling, sea ice, chemistry,
   aerosols, nudging, forcing decks, tracers beyond the sphum
   passenger, precision-adjacent flags like mixed precision?). For each
   hole: what config reaches fv_dynamics_step and what happens -- does
   the lane's own zvir/consv_te guard catch it, or does something run
   silently wrong?
2. THE DRIVER LANE (model_driver.py _run_fv3_duo). Dispatch order: it
   runs BEFORE the mpas/spectral branches keyed on grid_type -- can a
   config with discretization=fv3_duo plus a conflicting grid_type
   select this lane with a grid it cannot use, or vice versa? The dt
   CFL clamp 1920->600 s: where does it come from, is it derived or
   hardcoded, and is the user told? The days convention: total days --
   consistent with the other lanes' meaning? The snapshot: does the
   blowup guard (finite + 400 m/s envelope) TERMINATE or just warn, and
   is a guard-tripped run distinguishable from a completed one in the
   manifest? Restart refusal: can a restart-configured run reach the IC
   rebuild and silently restart-from-scratch instead of refusing?
3. THE MODEL WRAPPER (fv3_duo_dynamics.py). check_nsplt_schedule runs
   eagerly per step -- does that force a device sync per step, and is
   that acceptable or should it be documented as the cost of the C5
   guard? The bundled pytree: is its structure stable across steps
   (scan/jit cache safety), and does step() ever rebuild the jitted
   callable? km/kord refusals at construction vs at step -- anything
   only discovered N steps in?
4. THE REGISTRY EDITS (dynamics/__init__.py, supported_matrix,
   grids/factory). Is the lazy-import entry correct (a typo there fails
   only on first resolution)? Does _AXIS_TO_SOLVER's NH-first ordering
   change any existing reverse lookup? Does create_fv3_duo_grid leak
   NumPy-lane objects into the JAX bundle that a jit boundary would
   choke on later?
5. THE TESTS (test_fv3_duo_dynamics.py, test_run_amip_cli.py addition).
   For each refusal test: does it assert the MESSAGE or just the
   exception type (a different error on the same path passes a
   type-only test)? Does the ModelDriver smoke assert anything about
   the PHYSICS of the result (a wind envelope, a mass invariant)
   beyond "completed + snapshot exists"? The deprecation-count 17->19
   absorption: is the note honest about absorbing a pre-existing red?
6. ANY COMMENT OR DOCSTRING not true of the code, especially the
   snapshot-schema claim and the "one compiled program" claim for the
   NH carry.

Output BLOCKER / MAJOR / MINOR with the concrete failure each enables.
Say explicitly where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"
echo CODEX_WIRE_REVIEW_DONE
