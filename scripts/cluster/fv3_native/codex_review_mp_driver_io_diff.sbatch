#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_mpio_diff
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_mpio_diff_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha (base): $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of an UNCOMMITTED working-tree diff implementing
multi-process SPMD checkpoint/restart/snapshot I/O for the FV3 duo GSPMD
dycore. Run `git diff -- packages/coupler/legoesm/driver/model_driver.py
packages/coupler/legoesm/driver/component_factory.py
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/fv3_duo_dynamics.py`
to see it -- it has NOT landed yet.

THIS IS ROUND 2. An earlier design-only review (no code) found 4 BLOCKERs
and 5 MAJORs; this diff is the fix. Verify EACH one is actually fixed, not
just claimed fixed:

BLOCKER 1 (deadlock): rank 0 load failure before broadcast could hang the
other ranks forever. Fixed by `_spmd_barrier_on_root_error` called BEFORE
`_broadcast_fv3_duo_bundle` runs -- check `_load_fv3_duo_checkpoint`.

BLOCKER 2 (broadcast needs known shape/dtype on every rank first):
`_broadcast_fv3_duo_bundle` broadcasts (step,day,nq) first, then sizes
every leaf against a template built from a throwaway
`dcmip16_initial_state()` call (identical on every rank by construction) --
verify this actually gives every non-root rank a correctly-shaped/dtyped
placeholder BEFORE each `broadcast_one_to_all` call, and that the leaf
NAME SET (state_*/press_*/q_*/omga/nh_*) is genuinely derivable from the
model/config alone without touching the file on non-root ranks.

BLOCKER 3 (fresh IC never globally sharded): `_fv3_duo_reshard_bundle_global`
is now called on BOTH the fresh-IC and the restart bundle in `_run_fv3_duo`
right after either is obtained, before the step loop. Verify it is not
missed on any code path, and that `jax.make_array_from_callback` there is
called correctly (shape/sharding/callback) against `self.model.step_out_shardings`
(now stored as a model attribute in fv3_duo_dynamics.py -- check that too).

BLOCKER 4 (Held-Suarez unsafe under MP): a new refusal in `_run_fv3_duo`
raises NotImplementedError when `_is_spmd_multiprocess() and
cfg.held_suarez_forcing`. Verify this actually blocks every path that could
reach `_fv3_duo_apply_held_suarez` under multi-process SPMD, and that the
refusal condition for the OTHER (non-SPMD multi-process) case was also
fixed to be unambiguous (the earlier review flagged: "refuse multiprocess
runs unless _is_spmd_multiprocess() is true -- not refuse when it is
true").

MAJOR 1 (status file race): `_fv3_duo_write_status` -- verify it is now
root-gated + rendezvous'd under multi-process, and that the single-process
path is BYTE-IDENTICAL to before (an early return of the original
one-liner).

MAJOR 2 (snapshot error handling too narrow): `_fv3_duo_snapshot` --
verify the ENTIRE root-only section (path prep, np.savez, blowup logging)
is inside one try/except before the barrier, that blowup/finiteness is
computed identically on every rank from GATHERED data (a pure function, no
extra reduction), and that "write even when blown up" is preserved (the
snapshot must still be written on a blown-up state, not skipped).

MAJOR 3 (load validation too shallow): does the multi-process path
actually validate leaf shapes/keys with the same rigor as single-process,
or could a corrupt/truncated checkpoint silently pass? (Note the design
intentionally reuses `_decode_fv3_duo_checkpoint_arrays`'s FULL existing
validation on rank 0 -- verify that reuse is real, not bypassed.)

MAJOR 4 (sharding object drift): `fv3_duo_dynamics.py` now stores
`self.step_out_shardings` / `self.step_spmd_mesh` -- verify the loader
actually uses `self.model.step_out_shardings` (the stored object) and
never reconstructs an independent NamedSharding/Mesh.

MAJOR 5 (final-state digest not multiprocess-safe): NOT touched by this
diff -- the design review claims `_record_final_state_digest` (around
line 4221) is ALREADY safe under multi-process because it (a) already
early-returns on non-root via `self._mpi_rank != 0`, and (b) already wraps
its digest computation in a broad try/except that logs+continues on
failure (a non-fully-addressable jax.Array raises inside `pytree_state_digest`,
which the except catches). VERIFY this claim directly by reading that
function -- is it actually already safe, or does the earlier MAJOR still
apply? Say explicitly which.

ALSO CHECK, independent of the above checklist:
1. `component_factory.py`'s fv3_duo SPMD branch: does `jax.devices()` vs
   `jax.local_devices()` selection, and the divisibility/1-device
   validation, actually work correctly for both single- and
   multi-process launches (no regression to the existing single-process
   error messages/behavior)?
2. Any place where the diff changed SINGLE-PROCESS behavior when it
   claims not to (the design intent throughout is "single-process arm is
   the original code, byte-identical, only guarded behind an `if not
   self._is_spmd_multiprocess()` early return").
3. `_fv3_duo_checkpoint_write` refactor in `_fv3_duo_save_checkpoint`:
   confirm the extracted root-write body is IDENTICAL to the original
   inline code (same schema stamps, same fsync/replace/fsync-dir
   ordering), and that the multi-process except-block's tmp-file cleanup
   computes the SAME path the write function used (same os.getpid()).
4. Any remaining hole in the two-stage broadcast protocol: NaN/Inf
   payload preservation through `broadcast_one_to_all`, dtype
   preservation, an off-by-one in the q_i name range, or the nh
   present/absent branch disagreeing across ranks.
5. Any comment/docstring added that is not true.

Output BLOCKER/MAJOR/MINOR with file:line, the concrete failure scenario,
and the smallest fix. For each of the 4 BLOCKERs + 5 MAJORs above, say
explicitly FIXED / NOT FIXED / PARTIALLY FIXED.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_MPIO_DIFF_DONE
