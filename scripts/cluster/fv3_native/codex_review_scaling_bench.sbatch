#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_scale
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_scale_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of a JAX multi-process GPU SCALING BENCHMARK instrument:
scripts/bench/bench_fv3_duo_spmd_scaling.py (read the whole file). It is
about to decide whether the FV3 duo dycore's 6-GPU (3 nodes x 2, 1
face/device, GSPMD face sharding + shard_map ring halo + face-batched
phases) step is FASTER or SLOWER than the same step on a single GPU. An
instrument bug here manufactures a confident-wrong scaling conclusion, so
attack the MEASUREMENT validity, not just crashes.

Context you can verify in the repo: model.step returns a bundle dict
{state, press, q, omga, nh}; the sharded model's outputs are pinned
face-sharded via step_out_shardings; the sharded step issues cross-device
collectives (ring ppermute + border all_gather); the single-device arm
(FV3DuoDynamicsModel(grid, cfg) with no sharding) runs on this process's
own local card with a fully-addressable host IC (no collectives), run
redundantly on all 6 processes; only rank 0 prints. Compare against the
proven timing pattern in scripts/validate/fv3_native/spmd_face_shard_parity.py
(warmup + timed loop) -- is this bench's pattern consistent with it?

ATTACK IN THIS ORDER, cite file:line, output BLOCKER/MAJOR/MINOR with the
concrete way it distorts the FASTER/SLOWER verdict and the smallest fix:

1. _time_steps blocks only `b["state"]["pt"]` (ONE leaf) per step. The
   step is async-dispatched and returns other leaves (press/q/omga/nh)
   possibly still in flight. Does timing only pt under-count the true
   step wall, and could it bias the sharded arm vs the single arm
   DIFFERENTLY (collective ordering)? What should it block on
   (jax.block_until_ready over the whole pytree / tree_leaves)?
2. Both arms are timed in the SAME process on the SAME physical GPU,
   sharded arm FIRST then single-device -- warm/fragmented memory,
   allocator state. Does the ordering bias the second (single-device)
   arm? Isolate?
3. The sharded step is a COLLECTIVE; each of 6 processes times
   independently, only rank 0 prints. Does rank 0's per-step wall
   validly measure the collective's cost or hide stragglers on other
   ranks? Should it barrier + take the max across processes?
4. n_warmup=2: enough to pay XLA compile AND cublas/cudnn autotune
   (algorithm selection can happen over the first few RUNS)? Could
   residual autotune inflate early timed samples (median over 20
   mitigates, but note it)?
5. time.time() vs time.perf_counter() around async dispatch + a
   single-leaf block: does the Python->XLA dispatch overhead sit inside
   the timed region fairly for both arms, and is the wall clock choice
   adequate?
6. Any path where the reported speedup is WRONG IN SIGN -- the single
   arm secretly paying a distributed cost, or the sharded arm's
   collectives partially hidden by the single-leaf block so the sharded
   number reads artificially LOW (making it look faster than it is)?
7. --skip-single try/except swallows ALL exceptions as "OOM" -- could a
   real bug (wrong sharding, shape error) be silently reported as
   "single-device SKIPPED/OOM" and hide a broken measurement?
8. Anything else making the number not mean "6-GPU-sharded ms/step vs
   1-GPU ms/step for the same physics."

Rank findings by how much each could flip or distort the verdict.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_SCALE_DONE
