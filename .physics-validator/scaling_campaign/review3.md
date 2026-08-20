Not ready to sign off: 8 findings. I did not modify files.

1. **High — federation gate is not reliable on PALS or nested Slurm.**  
   [run_cpu_mpi_scaling.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/bench/run_cpu_mpi_scaling.py:154) treats `PALS_LOCAL_SIZE` as launcher world size. It is per-node, so a 24-rank / 4-ranks-per-node run can accept a one-node partial federation as “4 of 4,” masking 20 unfederated processes. It also ignores `SLURM_STEP_NUM_TASKS`, so `srun -n1` inside a larger allocation can be mistaken for the allocation-wide rank count.  
   **Verdict:** conditional refute. Using `jax.process_index()` is correct after federation, and the gate detects unfederated processes when a true global OMPI/PMI size is available; it does not meet that guarantee generally. Use the project’s canonical launcher-size resolver, prefer step size, and never use `PALS_LOCAL_SIZE` as a global count.

2. **High — geometry broadcasting can hide genuine configuration divergence.**  
   [_replicated_put](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:518) unconditionally makes process 0 authoritative for geometry array values. A differing mask or derived grid array on another host is silently replaced rather than detected; static geometry/config fields are not covered either.  
   **Verdict:** refute the “safe convergence” implication. Add a pre-broadcast cross-host manifest/fingerprint check for dimensions, scalar grid metadata, array names/shapes/dtypes, precision, solver/model config, and input identity. Allow only intentional ULP-level drift in derived arrays after that validation.

3. **Medium — the documented direct multicontroller launch still reintroduces GPU visibility pinning.**  
   [bench_cube_tiled_step_scaling.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/bench/bench_cube_tiled_step_scaling.py:39) recommends `--gpus-per-task=1`. That commonly leaves each task seeing only CUDA ordinal 0, while JAX selects the `SLURM_LOCALID`-th local device; local IDs 1–3 then fail.  
   **Verdict:** active route-B campaign lanes are correct: the updated sbatches use visible node GPUs plus `--gpu-bind=none`, with no manual `CUDA_VISIBLE_DEVICES` pin. Update this stale recipe to match them.

4. **Medium — cube tiled results still collide across jobs.**  
   [cube_tiled_step.sbatch](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/cluster/scaling_levante/cube_tiled_step.sbatch:41) defaults every job to the same JSONL path and appends twice.  
   **Verdict:** the requested `OUTDIR_j${SLURM_JOB_ID}` change is confirmed for the main Levante sbatches, but this independently launched tiled benchmark remains collision-prone. Give its default result file a job-ID suffix too.

5. **Low — preflight is harmless to lane-E exit status, but invalid as lane-E validation.**  
   [gpu_multinode_scaling.sbatch](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:130) always launches `NP_LL` tasks and swallows failure. With only lane E enabled it can test too few local IDs, or request more nodes than E requires and emit a discarded failure.  
   **Verdict:** harmless in the narrow sense that it does not change `rc_all`; not a meaningful preflight for E. Size it from the active lanes (or explicitly test all local IDs once per allocated node).

6. **Medium — the stated `cons-rtol=1e-5, rc=0` ladder gate is not established by the checked-in protocol.**  
   The conservation code correctly fails a violation, but the timed LL384 commands in [gpu_scaling.sbatch](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/cluster/scaling_levante/gpu_scaling.sbatch:127) do not pass `--check-conservation` or `--cons-rtol`; only a small smoke command does.  
   **Verdict:** cannot confirm the reported arms were gated at `1e-5` from this campaign definition. Put the gate on every reported arm and retain command/result receipts.

7. **High — “14–18 µs per-allreduce latency” is an invalid attribution.**  
   The 123 vs 63 count is arithmetically correct, but `single_reduce` is a different Chronopoulos–Gear solver, not a reduction-count-only treatment: it changes vector work, matvec ordering, collective payload/dependencies, numerics, and compiler fusion opportunities. See [barotropic_common.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:638).  
   **Verdict:** refute. At most this is an *effective time per eliminated reduction-batch opportunity* for this A/B executable. A dependency-matched allreduce microbenchmark or profiler trace is required for a latency claim.

8. **High — the 24-GPU tiled vs 6-GPU face-lane conclusion is not a fair strong-scaling comparison.**  
   [bench_cube_tiled_step_scaling.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/bench/bench_cube_tiled_step_scaling.py:157) is a tiled, held-Suarez, blocked-loop benchmark; the face lane is a distinct face-sharded path with different initialization/configuration and harness. [run_cpu_mpi_scaling.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/bench/run_cpu_mpi_scaling.py:560) already contains the more comparable tiled builder.  
   **Verdict:** refute the causal “slower because 9.2k columns/GPU is below the floor” claim. The column counts are factual, but the timing comparison is cross-path descriptive only. Compare 6 and 24 GPUs using the same tiled path, configuration, IC, and timing protocol.

Requested behavior checks:

- **CS-SPMD gate:** single-process/no-launcher works; true global launcher size catches unfederated singleton processes. Finding 1 prevents sign-off for PALS/nested-Slurm cases.
- **Finiteness gate:** confirmed. `jnp.all(jnp.isfinite(...))` produces a globally reduced replicated scalar for sharded leaves, so each controller can fetch it; single-process works; the existing non-finite path still returns **6**.
- **Geometry dtype/shape:** confirmed, conditional on a consistent `jax_enable_x64=True` configuration. `np.asarray → broadcast_one_to_all → jnp.asarray` preserves `float64` and shape. At one process it performs no broadcast, though it is not a literal zero-copy no-op.
- **vmix-f32/Amdahl claim:** partially confirmed. f32 is faster at nd1 and nd4 while having lower efficiency, which is consistent with a smaller parallel component and similar fixed cost. It is not proof of Amdahl causality. “+12% at nd4” is also not universal: ICN is 9.1% by `T64/T32−1` (8.3% time reduction); wide is 11.7% (10.5% time reduction).

Focused checks passed: syntax checks, shell parsing, 4 cube benchmark tests, and 6 parallel ocean tests.