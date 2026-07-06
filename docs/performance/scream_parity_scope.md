# SCREAM-parity scope: can legoESM reach cloud-resolving GPU throughput?

> **Status: scoping (2026-07-05).** Verdict: per-GPU kernel speed is reachable
> within ~2–3× of SCREAM; the *actual* blocker is multi-node scaling, which the
> current `mpi4jax` transport structurally cannot deliver. Feasibility collapses
> to **one decisive experiment** (Blocker 1). Chasing SCREAM's 27k-GPU exascale
> throughput is the wrong goal — it sacrifices the differentiability moat. Target
> **differentiable cloud-resolving at O(1k) GPU**, not throughput parity.

Companion runbook: `REAL_HARDWARE_SCALING.md`. Decisive-experiment harness:
`scripts/bench/bench_cube_shardmap_halo.py` (+ `scripts/cluster/scaling_derecho/`).

---

## 1. The comparison, honestly

SCREAM (Donahue et al. 2024, JAMES `10.1029/2024MS004314`, Gordon Bell finalist):

| | SCREAM v1 | legoESM (as benched) |
|---|---|---|
| Resolution | **3.25 km** (ne1024pg2), cloud-resolving | 28–156 km |
| Levels | 128 | 26 (bench default) |
| Physics | full CRM: P3 micro + SHOC turb + RRTMGP rad, NH HEVI-IMEX | **dynamical core only** (see caveats) |
| Scale | **~27,000 MI250 GCDs** / ~8,192 Frontier nodes | **≤16 A100, single node** |
| Throughput | **1.26 SYPD** (≈419 sim-days/day) | dyn-only Mcells/s (below) |

**Two honesty caveats that must not be lost:**

1. **The 678 Mcells/s single-GPU headline is a thermally-throttled RTX 5090
   *laptop*, not an A100** (`docs/performance/scaling/SCALING_SUMMARY.md:7,29`;
   LL128/L26/fp32, Held-Suarez). The A100 figure (~1.2–1.5 Gcells/s if
   bandwidth-bound) is an *extrapolation*, not a measurement.
2. **Every plotted scaling curve is dynamics-only** — argparse default
   `--physics none`, dry Jablonowski-Williamson baroclinic wave
   (`scripts/bench/run_levante_gpu_scaling.py:2612`). Radiation has never been on
   a scaling ladder; the full-physics tier (`rrtmg_full`) cannot run multi-node
   at all today (`run_levante_gpu_scaling.py:261-266`, AMIP-segment, single-node,
   cube/latlon only).

**Consequence:** a SYPD head-to-head is meaningless (10–50× resolution gap,
different physics load), and the Mcells/s per-GPU numbers compare a *dynamics-only*
kernel to SCREAM's *full-CRM* load. No "we're faster" claim is defensible.
legoESM's genuine edge is **end-to-end reverse-mode differentiability** (SCREAM
has none) and **single-GPU accessibility**, not throughput supremacy.

---

## 2. Feasibility verdict per axis

| Axis | Verdict |
|---|---|
| Per-GPU kernel speed vs Kokkos | Reachable, ~2–3× XLA tax, closeable. |
| Timestep algorithm (few-km dt) | Narrow gap — NH + implicit primitives already exist; integrator architecture is the work. |
| Physics cost at few-km | Mostly a *wiring* problem; bulk schemes already GPU-ready. |
| **Multi-node scaling to O(10³–10⁴)** | **Currently NO.** `mpi4jax` transport cannot get there. This is the gate. |

---

## 3. The four concrete blockers

### Blocker 1 — Communication layer (THE wall) — 6–12 mo, HIGH risk

Two disjoint comm families:
- **(A) `mpi4jax` point-to-point** — lat-lon-band, plane-CRM, spectral-FFT,
  Voronoi. **Blocking** (no `Isend`/`Irecv` ⇒ zero comm/compute overlap:
  `halo_exchange.py:42`, `async_halo.py:10`), **host-staged by default**
  (`reductions.py:250-279`), on the **legacy XLA custom-call slow path**
  (`scripts/cluster/scaling_derecho/README.md:266`).
- **(B) GSPMD `ppermute` over NCCL/ICI** — cube-SPMD + lat-lon-band-SPMD, wired
  through `jax.distributed` (`early_init.py`, `distributed.py`). This is the path
  that can scale.

Structural ceilings:
- Cube SPMD mesh bounded by **face-count `6·kt²`**; tiled table builder **hangs at
  kt=4 / np96** (`cubesphere_exchange.py:1078`). Validated only to np24 (kt=2),
  np54 (kt=3).
- Cube-panel cross-node `ppermute` schedule **anti-scales on TCP** (107→206 ms/step
  np1→6, `coupler/.../compiled_segments.py:484`; `sharded_dynamics.py:761`).
- `mpi4jax` Voronoi tag ceiling already hit and worked around with rank-free tags
  (`halo_exchange_voronoi.py:191-202`, #684) — a symptom of the transport not
  being built for scale.

**Work:** (a) reductions → native `psum` (`reductions.py`); (b) migrate each
grid's `sendrecv`/`alltoall` → `shard_map` `collective_permute`
(`latlon_mpi.py`, `plane_mpi.py`, `voronoi_mpi.py`, `distributed_fft.py`),
reusing the `latlon_spmd.py` pattern; (c) generalize the cube tiled schedule past
kt=3 to an intra-face n×n nearest-neighbor decomposition
(`cubesphere_exchange.py`); (d) gain real async overlap once on NCCL.

**Risk:** AD through `collective_permute` must stay VJP-correct (CLAUDE.md MPI-AD
doctrine); Voronoi ragged `shard_map` is unproven.

### Blocker 2 — IMEX-ARK integrator (dt at few-km) — 3–5 mo, MEDIUM risk

NH compressible-Euler already exists on **5 grids** (shared core
`compressible_euler.py`), with an **opt-in vertically-implicit acoustic solve**
(`semi_implicit_acoustic`, `compressible_euler.py:126,876-1039`) and a
Klemp-Wilhelmson buoyancy Schur-fold. The batched tridiagonal solver is AD-safe
(`packages/core/legoesm/timestepping/tridiagonal.py:436-437`, JVP+transpose).

Gap (architecture, not primitives):
- No monolithic **IMEX / ARK** integrator anywhere (repo-wide grep clean). Current
  scheme is Klemp-Skamarock split-explicit with `n_acoustic_substeps=6`
  (`split_explicit.py`).
- Vertical-implicit is **w-only scalar** tridiag; a true HEVI stage needs the
  coupled `(w, ρ', θ')` column block solved implicitly at the full model dt.
- Horizontal acoustic is still substepped, so dt is horizontal-acoustic-bound via
  `n_acoustic_substeps` rather than chosen as a first-class dt
  (`cfl_diagnostic.py:71-75`).

**Work:** ARS/ARK additive-RK driver (explicit tableau = horizontal+advective,
implicit = vertical column); coupled block-tridiag (extend the `implicit_buoyancy`
template); static horizontal-acoustic CFL dt picker (`column_sound_speed_upper_bound`
already computes the number, `cfl_diagnostic.py:45`). Reuse the SSP stage machinery
and the closest existing prototype `plane_acoustic_substeps_si_horizontal`
(`compressible_euler_plane.py:2521`).

### Blocker 3 — Radiation wiring + full-physics on the distributed path — flips: days; distributed: 2–3 mo; LOW–MED risk

- The **g-point-parallel** radiation path exists (`rrtmgp/rte/two_stream.py:298-323`,
  `jax.vmap` over g-point blocks) but **production defaults to the sequential
  checkpointed scan, ~6–26× slower** (`radiation/rrtmgp/config.py:242` and
  `coupler/.../driver/config.py:328`, both `gpoint_batch_size=0`). Only the forward
  bench opts into 32.
- Radiation is called **every step** (`rad_update_steps=1`); sub-cycling machinery
  exists but is off by default.
- The radiative tier `rrtmg_full` runs through the AMIP **segment** path with **no
  MPI step**, cube/latlon single-node only — so the multi-GPU ladder can only run
  `none` or Kessler-`moist`.
- Bulk ice micro (Morrison/Thompson/P3) + Smagorinsky SGS + Thomas implicit vertical
  diffusion are **already clean `(ncol,nlev)` GPU kernels, no host callbacks or
  scatters**. `fast_sbm`, `sdm`, `mc3d` are the real cost cliffs — keep them out of
  the hot path (use the shipped ML surrogates when fidelity demands).

**Work:** flip `gpoint_batch_size>0` default; enable rad sub-cycling; wire the
full-physics pipeline onto the MPI/SPMD step via the `create_column_mesh` seam
(`combined.py:140-146`); ship a CRM config = bulk ice micro + Smagorinsky +
convection off.

### Blocker 4 — Per-GPU kernel pass-count — 1–2 mo, LOW risk

- SSP-RK3 evaluates the tendency **3× / step** and XLA inlines three copies of the
  graph (`timestepping/dispatch.py:42-47`); a bit-verified scan-folded variant
  (`ssp_rk3_step_scan`) exists but is not the default.
- The 2D SW ∇⁴ calls `laplacian_compact` **4× with 4 un-reused halo exchanges**
  (`operators.py:224`); the 3D path already has the `padded=`/`inner_lap=` reuse
  hooks (`operators_3d.py:189,205-217`), the 2D path does not.
- Cube div-damp/boundary use ~12 `.at[].set` scatters/substep. Net ≈ 24–42
  full-field HBM sweeps/timestep — the dominant efficiency leak is **pass count**,
  not per-kernel bandwidth.
- fp32 is end-to-end on cube/latlon/ico; spectral is complex128/CPU by design.

**Work:** ship `ssp_rk3_step_scan` as default; add halo-reuse hooks + u+v stacking
to the 2D `laplacian_compact` (4 halos → 1); fold the standalone div-damp
divergence into the mass-flux divergence; drop the fp64 conservation-accumulator
from the innermost fp32 SW step.

---

## 4. Dependency-ordered plan

1. **Weeks (cheap, de-risking):** Blocker 4 kernel wins + Blocker 3 config flips +
   **measure a real A100 baseline** (current per-GPU number is a throttled laptop).
2. **Now, in parallel — the gate:** run the Blocker 1 decisive experiment
   (`scripts/bench/bench_cube_shardmap_halo.py`, 2-node NCCL). If the GSPMD
   `shard_map` cube halo cannot hold efficiency past the `mpi4jax` np6 anti-scale
   cliff, the parity answer is **no** and the rest is moot.
3. **If Blocker 1 clears:** full Blocker 1 migration (6–12 mo) ∥ Blocker 2 IMEX
   (3–5 mo) ∥ Blocker 3 distributed full-physics (2–3 mo).

**GPU-count arithmetic (why scaling, not GPU count, is the gate).** 1.26 SYPD at
3.25 km ≈ 460 sim-s/wall-s. Explicit hydrostatic dt ≈ 6 s ⇒ ~77 steps/s ⇒
3.2×10⁹ cells × 77 ≈ **250,000 Mcells/s** required. At ~400 Mcells/s/GPU sustained
that is **~625 GPUs *if scaling were ideal*** for dynamics+light physics; full-CRM
physics pushes toward SCREAM's O(10⁴). The GPU count is *available* — holding
≥60% efficiency to ≥1000 GPUs multi-node is the open question.

**Realistic target:** differentiable, cloud-resolving-capable, ~1 SYPD on **O(1k)
GPU**, in **12–24 focused months**, *conditional on Blocker 1 clearing its gate.*

---

## 5. The decisive experiment (Blocker 1)

`scripts/bench/bench_cube_shardmap_halo.py` — drives the **production** SPMD cube
halo (`set_halo_backend("spmd")` + `activate_spmd_halo_backend` + `pad_halo`), not
a bespoke copy, so the result speaks for the model.

Measures, at fixed global cube size, sweeping device count (1 → 6 → tiled 24/54):
- **Correctness gate:** SPMD exchange bit-matches a single-device replicated
  reference (the halo must be exact).
- **AD gate:** `jax.grad` through the SPMD halo matches the replicated reference
  (`collective_permute` VJP correctness — CLAUDE.md MPI-AD doctrine).
- **Scaling metric:** per-exchange latency + a halo+∇² micro-step; strong-scaling
  speedup and efficiency vs baseline.

**Pass condition:** GSPMD `shard_map` cube halo holds **≥60% strong-scaling
efficiency at np16 across 2 nodes**, clearing the `mpi4jax` np6 anti-scale cliff.
Local (macOS/CPU) run validates correctness + AD + plumbing on
`--xla_force_host_platform_device_count`; the real GPU numbers come from the
Derecho 2-node NCCL job.

**Status (2026-07-05).** Harness `scripts/bench/bench_cube_shardmap_halo.py`
(+ `tests/bench/test_bench_cube_shardmap_halo.py`) built and codex-clean (4-round
adversarial review). Validated on 6 emulated CPU devices:
- Correctness Δ ≤ 2.8e-10 (fp64) — SPMD halo moves exactly the right data.
- AD: nonuniform-cotangent VJP through the ppermute halo matches the
  single-device reference to **1.7e-16**, with a lowered-HLO `collective_permute`
  assertion so a silent local fallback cannot pass the gate.
- CPU efficiency **anti-scales at np6** (proxy) — the harness faithfully
  reproduces the cliff on a slow "fabric," which is exactly what the NCCL run
  must overturn.
Gates are vacuous-pass-proof: decisive mode rejects single-device, requires the
AD gate to run, and gates the *timed* executable (not just the AD path) on a
proven ppermute collective.

**Single-node Derecho lane shipped:**
`scripts/cluster/scaling_derecho/blocker1_cube_shardmap.pbs` runs the real 4×A100
NVLink measurement (counts 1→3, correctness + AD on GPU).

**Cross-node lane shipped (the actual cliff crossing).** np6 spans 2 Derecho nodes
(4 GPU/node), which a single-controller sweep cannot express — a multi-controller
job fixes the device count for the whole run. Implemented as:
- `bench_cube_shardmap_halo.py --point` — multi-controller TIMING-POINT mode: one
  `mpiexec -n N` launch (one process/GPU, federated via
  `initialize_jax_distributed_multiprocess`) emits one point at global count N; the
  mesh is built over `jax.devices()[:N]` (NOT `local_devices()`), barrier + max-
  over-ranks timing, ppermute proven in the timed HLO, degrade-guarded.
- `scripts/bench/aggregate_cube_shardmap_scaling.py` — combines per-N points into a
  speed-up/efficiency curve and gates: baseline-present, ppermute-on-every-N>1,
  config-consistent (incl. backend), `--require-counts` present (no vacuous PASS
  from a subset), `has_multi_point`, and (decisive) efficiency ≥ threshold.
- `scripts/cluster/scaling_derecho/blocker1_cube_shardmap_xnode.pbs` — 2-node,
  `for N in 1 2 3 6: mpiexec -n N --point` then aggregate with `--gate-efficiency
  --require-counts`; the job exit code IS the decisive verdict.

Validated locally end-to-end (per-N `--point` launches → aggregate → gates;
codex-clean 4 rounds + code-reviewer). Only real cross-process NCCL/gloo federation
is cluster-only (macOS is gloo-blocked); the single-process points, the aggregation
logic, and every gate are covered by tests. **Open action:** `qsub
blocker1_cube_shardmap_xnode.pbs` on Derecho — that is the measurement that answers
the SCREAM-parity feasibility question.

---

## 6. Strategic note

Beating SCREAM at its own game (27k-GPU Kokkos exascale throughput) is a 5+
person-year systems chase against a Gordon-Bell team, and it *costs the reason
legoESM exists*: SCREAM cannot differentiate; legoESM can. The winning claim is
**"first differentiable few-km global model,"** not "faster than SCREAM." Scope
the throughput work only as far as it serves that claim — cloud-resolving-capable
at O(1k) GPU with gradients intact.
