#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_scale2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_scale2_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"
codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
CONFIRMING REVIEW round 2 of scripts/bench/bench_fv3_duo_spmd_scaling.py
(read the WHOLE current file -- it was rewritten to fix your round-1
findings). Verify each round-1 issue is now closed, FIXED/NOT FIXED:
1. BLOCKER only-one-leaf-blocked -> _block_all() blocks jax.tree_util.tree_leaves(bundle) after warmup and each timed step.
2. BLOCKER rank-0-only critical path -> sharded loop: sync_global_devices barrier before t0, perf_counter, step, _block_all, process_allgather each rank's local, take max, median of per-sample maxima.
3. BLOCKER arms not isolated -> now TWO modes (--mode sharded | single) run as SEPARATE srun launches in the sbatch (sharded = 6 tasks; single = 1 task 1 GPU, no jax.distributed). Confirm the single arm truly runs no collectives and is a separate process.
4. MAJOR broad-except-as-OOM -> single arm's except now names the exception type + exits 2 (FAILED), not a generic OOM skip.
5. MAJOR topology not enforced -> sharded mode asserts device_count==6, process_count==6, local_device_count==1.
6. MAJOR no non-vacuity -> _assert_face_sharded after warmup AND after timing (is_fully_replicated False AND devices_indices_map == requested face sharding, per raw leaf).
7. MINOR time.time -> perf_counter; MINOR warmup 2->3.
ALSO look for NEW bugs the rewrite introduced: does process_allgather(jax.numpy.asarray(local)) inside the timed loop but after `local` is captured leak cost into the measurement? Is the barrier/allgather sequence identical on every rank (no divergence -> no hang)? Does _sha() import failure degrade gracefully? Is the sbatch's second srun (--nodes=1 --ntasks=1 --gres=gpu:1) correct for an isolated single-GPU run within a 3-node allocation? Output BLOCKER/MAJOR/MINOR + file:line, or CLEAN.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_SCALE2_DONE
