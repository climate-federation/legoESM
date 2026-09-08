#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_mpio_dsn
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_mpio_design_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL DESIGN REVIEW -- no code has been written for this yet, review
the PLAN only. Read-only, do not edit anything.

CONTEXT. The FV3 duo GSPMD dycore's model/step lane is already proven
correct across multi-node multi-GPU (ring exchange + face-batched phases,
bitwise parity vs single-device -- read
scripts/validate/fv3_native/spmd_multiprocess_parity.py, especially its
`_global_arrays` helper around line 100: the proven
`jax.make_array_from_callback(h.shape, shard, lambda idx: h[idx])` idiom).
UNDESIGNED: the DRIVER's checkpoint/restart/snapshot I/O for a real
multi-PROCESS run. Today it's refused in
packages/coupler/legoesm/driver/model_driver.py::_run_fv3_duo (~line 7808):
`if (self._mpi_world_size or 1) > 1: raise NotImplementedError(...)`.

Read the existing SINGLE-process code this plan must generalize:
`_fv3_duo_snapshot`, `_fv3_duo_flatten_bundle`, `_fv3_duo_save_checkpoint`,
`_load_fv3_duo_checkpoint` in model_driver.py (roughly lines 7980-8230).
It does `np.asarray(v)` / `jnp.asarray(a)` directly on bundle leaves --
correct only when every leaf is fully host-addressable (true in
single-process multi-GPU; false under real multi-process SPMD, where each
process is host-addressable for only ITS OWN shards of a global jax.Array).

Read the EXISTING helpers this plan proposes to REUSE (grep them, do not
assume from the names): `_is_spmd_multiprocess()` (~line 6010),
`_spmd_barrier_on_root_error()` (~line 6019, rendezvous-on-root-only-write
failure), `_gather_spmd_tree_to_host()` (~line 6050, tree_map
process_allgather+tiled=True over a whole pytree, no-op on
fully-addressable leaves). Also read component_factory.py ~lines 795-824
(fv3_duo's SPMD branch -- currently builds its mesh over
`jax.local_devices()` ONLY, single-process).

THE PLAN (evaluate exactly this; tell me what's wrong or missing, do not
just re-derive your own from scratch):

SAVE (_fv3_duo_snapshot, _fv3_duo_save_checkpoint), called every step from
_run_fv3_duo's loop:
1. Before either runs: `bundle_host = self._gather_spmd_tree_to_host(bundle)`
   -- a COLLECTIVE every process calls identically.
2. Blowup/finiteness checks run on EVERY process against bundle_host (pure
   function of bit-identical input -> identical decision every rank, no
   further reduction needed for the check itself).
3. The actual np.savez write is gated `if jax.process_index() == 0:` only.
4. Wrap the rank-0 write in try/except, then call the EXISTING
   `self._spmd_barrier_on_root_error(err)` (every process calls it) right
   after.
5. Full-state-replication host memory cost of the allgather is ACCEPTED as
   bounded at this lane's C48-C384 fp64 sizes (few GB); explicitly not
   engineered further (no per-leaf streaming write) unless state size
   grows -- stated as a residual, not silently dropped.
6. component_factory.py's fv3_duo branch already asserts `6 %
   len(devs) == 0` at construction, so every rank always owns >=1 whole
   face -- no empty/zero-size shard can reach the allgather.

LOAD (_load_fv3_duo_checkpoint):
1. Single-process (jax.process_count()==1): UNCHANGED, current
   jnp.asarray(a) path stays bit-for-bit.
2. Multi-process: rank 0 ONLY reads the npz off disk (single FS reader,
   avoids N-way NFS reads and FS-visibility-semantics reliance). Rank 0
   runs the EXISTING full validation/decode (schema/km/resolution/
   hydrostatic/dt guards, unchanged) producing host numpy leaves + Python
   scalars (step/day).
3. Every leaf's numpy array plus step/day is broadcast to all ranks via
   `jax.experimental.multihost_utils.broadcast_one_to_all` -- this
   collective IS the write-then-read barrier (no separate
   sync_global_devices needed), symmetric with the save path's
   rank-0-only I/O.
4. Every rank reconstructs its OWN addressable shards of the properly
   sharded jax.Array per leaf via the SAME
   `jax.make_array_from_callback(h.shape, shard, lambda idx: h[idx])`
   idiom already proven in spmd_multiprocess_parity.py, against the
   model's step_out_shardings/step_spmd_mesh (needs exposing on
   self.model for the loader to reuse -- small factory-side plumbing).

DRIVER REFUSAL: replace the blanket `self._mpi_world_size > 1` raise with
the narrower `self._is_spmd_multiprocess()`-gated check -- still refuse
real multi-process MPI mode (unimplemented, orthogonal), permit
multi-process when distributed_mode == 'spmd'.

FACTORY: component_factory.py's fv3_duo SPMD branch must build its mesh
over the GLOBAL jax.devices() (not jax.local_devices()) when
jax.process_count() > 1, matching spmd_multiprocess_parity.py's own
already-proven pattern.

ASK EXPLICITLY: (a) any hole in the SAVE-side barrier/error-handling
reuse; (b) any hole in the LOAD-side broadcast-then-reconstruct plan (does
make_array_from_callback correctness depend on anything beyond "identical
host array on every process" this plan doesn't guarantee -- ordering,
dtype, contiguity, NaN handling in the broadcast); (c) whether reusing
_gather_spmd_tree_to_host/_spmd_barrier_on_root_error AS-IS is safe for
THIS call site, or whether their docstrings (written for other call sites)
hide an assumption that does not hold here; (d) anything about restart
round-trip determinism (bitwise) this plan does not cover; (e) any
blocker to lifting the _run_fv3_duo refusal exactly as proposed.

Output BLOCKER/MAJOR/MINOR with file:line, the concrete failure scenario,
and the smallest fix. Say where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_MPIO_DESIGN_DONE
