#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_auto
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_auto_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"
codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of an UNCOMMITTED diff (git diff
packages/coupler/legoesm/driver/component_factory.py) that makes the
fv3_duo dycore AUTO-ADAPT its execution layout to the visible device
count, replacing the old `if config.distributed: shard else: single`
gate.

INTENT (user directive): a single-process run should use whatever devices
are visible with NO --distributed flag: 1 device -> single-device loop;
2/3/6 local devices -> face-shard + face-batch; a count that does not
divide 6 (4,5,7...) -> LOUD fallback to single-device. Multi-process
(process_count>1) STAYS explicit behind --distributed (it changes the
driver's checkpoint/snapshot I/O and the refusal gates key off
config.distributed).

CLAIMS TO VERIFY (the physics/answer must not silently change):
1. Is face-sharding + face-batching really (near-)bitwise the
   single-device loop, so auto-selecting it on device count is a
   PERFORMANCE choice not a SCIENTIFIC one? (grep: spmd_multiprocess_parity
   asserts 0.0 bitwise at 2/3/6; step_face_batched batched==loop tests.)
2. Does the multiprocess-without-distributed REFUSAL actually fire before
   any sharding is built (so a multi-process launch can never silently
   run the single-process I/O path across ranks)?
3. Single-process AUTO-SHARDED (config.distributed=False, process_count=1):
   trace it into the driver model_driver.py::_run_fv3_duo -- do ALL its
   gates pass? `(self._mpi_world_size or 1) > 1` (should be False),
   `_is_spmd_multiprocess()` (False -> HS guard skipped, single-process
   I/O path used), and does `_fv3_duo_reshard_bundle_global` (no-op when
   not _is_spmd_multiprocess) leave a fully-addressable bundle that
   model.step then GSPMD-shards correctly? Is np.asarray on a
   single-process FACE-SHARDED leaf (snapshot/checkpoint) fully
   addressable (all 6 shards on this one process's local devices)?
4. Does EXPLICIT --distributed with an unshardable count (1, or 4/5/7)
   still raise loudly (not silently run single-device claiming
   distribution)?
5. The auto path with n in {4,5,7...} falls back to single-device with a
   logger.warning -- is that a SILENT behaviour substitution (forbidden)
   or an acceptable LOUD adapt? Does it change the ANSWER (single-device
   is the reference, so no)?
6. Is the resolved layout LOGGED in every branch (so the choice appears
   in the run record, per the no-hidden-choices rule)?
7. NEW RISK: CPU with xla_force_host_platform_device_count set to 6 (some
   tests set it) would now AUTO-SHARD over 6 CPU devices instead of
   running single-device. Is that a problem for existing fv3_duo tests
   that construct the model on CPU? grep tests for that flag + fv3_duo.
8. Any config field added (would need __param_spec__/validate)? (should be
   none -- reads device count, not config.)

Output BLOCKER/MAJOR/MINOR + file:line, concrete failure scenario, or
CLEAN.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_AUTO_DONE
