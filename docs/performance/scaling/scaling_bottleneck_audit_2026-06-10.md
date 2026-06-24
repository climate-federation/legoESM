# legoESM scaling bottleneck audit — 2026-06-10

Two read-only audits (atmosphere, ocean) + two codex scope/design reviews.
Context: Ginsburg RTX 8000 (2 GPU/node, SYS/PCIe no NVLink), 32-core CPU
nodes. Fixes already landed this campaign: multiface ppermute halo (killed
all_gather full-cube replication), SLURM_LOCALID GPU self-blinding, latlon
cut-face parity, ico sharded physics_fn. In flight: ocean distributed PCG,
AMIP segment-sharding flattened-carry fix.

## Ranked NEW levers

### Ocean
| # | Lever | weak/strong | mechanism | status |
|---|---|---|---|---|
| O1 | **Distributed PCG barotropic** | weak (fatal→fixed) | explicit_substep n_substeps∝resolution; implicit_cn CG deadlocks under MPI. Fixed-M PCG + batched-allreduce dots + custom_linear_solve AD. Helmholtz+Jacobi already MPI-correct. | **IMPLEMENTING** (codex: needs-change, core sound) |
| O2 | eta-floor redistribution | weak+strong | clamp_and_redistribute n_iter=3 allreduces ×2/substep ×n_substeps. O1 deletes the substep loop → mostly subsumed. | queue (post-O1) |
| O4 | latlon 4D batched halo | weak+strong | ~27 per-field sendrecv/tendency ×RK3 (~80/step momentum); no pad_halo_4d analog for latlon; async_halo.py unwired. | queue (review next) |
| O6 | land-aware partitioning | weak (realistic) | make_latlon_band_layout = equal lat rows, not equal wet-cells; continents idle some ranks. Invisible in aquaplanet. | queue |
| O5 | stacked multi-tracer advection | weak | T,S advected in Python loop, separate halos; N tracers→N× msgs; ×3 RK3. BGC scales linearly. | queue |
| O7 | MPAS ocean MPI | cliff | no distributed factory; single-device only. | document/defer |
| — | vertical Thomas solve | none | z never sharded; confirmed safe. | no action |

### Atmosphere
| # | Lever | weak/strong | mechanism | status |
|---|---|---|---|---|
| A1 | **MPI cubed-sphere replicated dynamics** | both (catastrophic) | every MPI rank computes all 6 faces; only physics scatters. mpi4jax pad_halo_mpi exchanges full 6-face arrays. Dynamics never scales under MPI. | **SCOPED: moderate–multi-week** (see below) |
| A2 | MPAS stacked_meshes replicated | weak (memory) | sharded_dynamics.py:1787 replicates all-device mesh metrics per device → O(N_global) mem. | queue (focused) |
| A3 | spectral single-device | cliff | global SH transform = full-latitude sum; no distributed transform; level-mesh "embarrassingly parallel" docstring misleading (is_distributed=False hardcoded). | **document** |
| A4 | cubed-sphere {1,2,3,6}-only | strong cliff | 4/5/8/12 → NotImplementedError or idle device. | queue |
| A5 | per-stage zero_mean_tendency | weak (latency) | 3 allreduces/step when zero_mean on + fix_mass off; half-gated already. | config default fix |
| A7 | latlon polar filter imbalance | weak | rfft over full lon row only on poleward bands → straggler ranks. | queue |
| A8 | jnp.min(area_corner) in JIT | minor + landmine | static metric recomputed per stage; silent-wrong if face-sharded later. | cheap fix |

## Scope verdict — A1 (codex)
Real fix = **pure-JAX-`distributed` SPMD** for cubed-sphere (global face mesh
from `jax.devices()` after `jax.distributed.initialize()`, `make_sharded_step`
+ ppermute halo + JAX-native reductions across nodes), **NOT** mixing mpi4jax
and jax.distributed collectives inside one step. Crux risk (#5): mpi4jax
(`MPI.COMM_WORLD`) + jax.distributed (XLA runtime) can coexist at process
level (repo inits MPI then optionally jax.distributed, distributed.py:138),
but mixing both collective systems inside one jitted/scanned timestep needs
real deadlock/order/AD testing — not safe by construction. This is likely
WHY MPI dynamics is replicated today.
- mesh.py:207 builds the mesh from LOCAL devices → wrong for multi-controller
  (needs global `jax.make_mesh` + dense process_id).
- model_driver.py:1763 activates SPMD halo only in the NON-distributed branch;
  distributed branch stays on replicated mpi4jax dynamics.
Effort: **moderate-to-multi-week**. On Ginsburg 2 GPU/node: 1 process/node
using both local GPUs (on-node SPMD ALREADY WORKS — C192 f64 eff 0.73), then
jax.distributed across nodes. **Recommendation: on-node multi-GPU SPMD is the
achieved win; multi-node GPU SPMD is a scoped future effort (decision point,
not autostart). Document MPI cubed-sphere as replicated-dynamics now.**

## Sequencing
1. O1 distributed PCG (in flight) — biggest ocean weak-scaling win.
2. AMIP segment-sharding fix (in flight) — unblocks on-node multi-GPU AMIP.
3. O4 latlon batched halo + A2/A8/A5/A3 focused fixes (short, low-risk).
4. O6 land-aware partitioning — realistic-ocean weak scaling.
5. A1 multi-node SPMD — scoped, surfaced as a decision (multi-week).
