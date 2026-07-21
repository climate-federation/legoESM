# Lat-band SPMD scaling — lat-lon C-grid hydrostatic atm step (A1)

Scaling of `make_sharded_atm_latlon_step` / `run_atm_latlon_spmd` (the
single-process multi-device lat-band decomposition of the lat-lon C-grid
hydrostatic dycore). Bench: `scripts/bench/bench_atm_latlon_spmd_scaling.py`;
plot: `scripts/plot/plot_atm_latlon_spmd_scaling.py`.

All numbers are steady-state min ms/step (compile excluded); JAX_ENABLE_X64.

## Perf prerequisite (jit-cache)

The bench surfaced a real perf bug: a bare `shard_map` is **not**
compilation-cached, so the un-jitted band step recompiled on **every** call —
~142 s/step at 32×64×10 nd=2 (job 8560671), and the whole equivalence gate ran
~65 min. Fixed by passing `dt` as a traced operand + wrapping the shard_map in
`jax.jit` built once (commit `fa2ce32b6`): first call compiles, the rest hit the
cache (probe 8561202: `[3079, 1.4, 1.2, 1.1, 1.1]` ms). The full 17-test SPMD
suite then runs in 129 s (was ~65 min); equivalence unchanged.

## GPU (Ginsburg, 2 GPU/node) — job 8561259

| mode | grid | nd=1 | nd=2 | result |
|------|------|------|------|--------|
| strong | 128×256×30 (fixed) | 6.09 ms | 5.56 ms | **1.10× speedup** |
| weak   | 64 lat/dev | 3.51 ms (64×256×30) | 5.51 ms (128×256×30) | **0.64 efficiency** |

Modest, sub-linear: at this size the per-GPU work is small and the cross-band
`ppermute` halo (over the node's PCIe interconnect) dominates the gain. This
matches the prior Ginsburg 2-GPU practical-limit findings for the ocean / BCW
campaigns — the lat-band atm SPMD is correct and positive-scaling, but the
2-GPU PCIe fabric caps strong scaling. Larger per-device grids (more compute per
halo byte) and >2 devices are the levers for better efficiency.

> **Update:** the multi-node `jax.distributed` path IS wired now —
> `bench_atm_latlon_spmd_scaling.py --multicontroller` (route-B, native NCCL
> ppermute, no mpi4jax). Derecho/Levante job lanes exist; production-scale
> numbers are the remaining measurement gap. See
> `docs/performance/scaling/SCALING_STATUS_AUDIT.md`.

## CPU virtual devices (characterization, NOT speedup) — job 8561216

| mode | grid | nd=1 | nd=2 | nd=4 |
|------|------|------|------|------|
| strong | 64×128×20 | 9.85 ms | 11.05 ms (0.89×) | 10.85 ms (0.91×) |

`--xla_force_host_platform_device_count=N` puts all "devices" on the same
physical CPU (threads sharing cores) → no real parallelism; the ~10% slowdown is
the (small) `ppermute` halo overhead. Useful only to confirm correctness at
device scale + bound the comm cost, not as a speedup.
