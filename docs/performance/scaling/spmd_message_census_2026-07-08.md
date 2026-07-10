# SPMD per-step message census + remaining strong-scaling levers (2026-07-08)

Context: the first full Derecho route-B campaign (native NCCL over Slingshot,
`gpu_multinode_scaling.pbs` lanes + the CPU/GPU sweeps) produced the
CPU-vs-GPU strong-scaling plots showing:

- **cubed-sphere ≤6 GPU face-shard anti-scales** (6-GPU Mcells/s ≤ 1-GPU at
  C48/C96-class; a 2-GPU dip below 1-GPU everywhere),
- **latlon 78 km plateaus at the 4→8 GPU node crossing**, 156 km anti-scales,
- **icosahedral scales best** (28 km still rising at 16 A100, eff ≈ 0.38–0.47),
- coarse grids flat everywhere — the known **per-device saturation floor**
  (keep ≥ ~30k columns/GPU; not a defect).

Since the transport is already overlappable (route-B ppermute, not route-A
mpi4jax), the remaining strong-scaling loss at small tiles is **per-step
message count × per-message latency**. This note pins the actual counts.

## Method

Optimized-HLO census on CPU virtual devices (message *count* is a static
schedule — resolution-independent — so a C24/L8 or 64×128/L8 compile gives
the production count). One-off probes (untracked scratch, `scripts/tmp/`);
the method is three lines and the cube tiled bench already records it
per-row as `hlo_collective_permutes`:

```python
# XLA_FLAGS=--xla_force_host_platform_device_count=N  JAX_PLATFORMS=cpu
hlo = jit_step.lower(state, dt).compile().as_text()
n_cp = hlo.count("collective-permute-start") + hlo.count("collective-permute(")
```

Counted alongside `all-gather` / `all-reduce` occurrences.

## Results

### Cube cs-spmd full PE step (C24/L8 shape; counts are shape-independent)

| devices | collective-permutes / step | all-reduce / step |
|---|---|---|
| 2 | 15 | 1 |
| 3 | 26 | 1 |
| 6 | 46 | 1 |

The single all-reduce is the conservation fixer (fine). The 46 CPs at 6
devices = ~11.5 exchange points × 4 ppermute rounds (the edge-coloring
matching floor: each face has 4 neighbors, so 4 rounds is the minimum for
pairwise ppermute).

**Packing is already at floor here** — no code lever left at the call-site
level:

- `explicit_pad_halo_vector_4d` packs (u,v) into ONE collective
  (concat along the level axis, rotate-and-exchange once).
- The dycore's stage-level packed halo (`packed_pad_halo_4d`,
  iter-58/60/61) merges {ζ, B, 1/T, ln_ps (+hybrid_factor, +div_v)} into one
  exchange per stage.
- `pad_halo_4d` batches all vertical levels per message by construction.

At C48/L26 a 1-A100 step is ~2.6 ms; 46 sequential small CPs at
~30–80 µs NCCL p2p latency ≈ 1.5–3.5 ms — that IS the anti-scaling of the
coarse cube curves, and the 2-GPU dip (15 CPs but 3 faces/shard + sync per
CP on a halved compute slice).

### Atm latlon SPMD step (64×128/L8, 4 devices)

| arm | collective-permutes / step |
|---|---|
| baseline | 41 |
| `LEGOESM_LATLON_SPMD_FUSED_HALO=1` | **29 (−29%)** |

The fused multi-pad (audit item 7, shipped opt-in for the ocean with a
25%-fewer-ppermutes receipt) also covers the atm step via
`pad_with_pole_bc_lat_multi` — same bit-identical packing, same flag.
Remaining 29 CPs live inside shared per-operator pads
(`operators_latlon_cgrid.py`, PPM h2 pads); merging those needs the
iter-58/60-style "operators accept pre-padded fields" surgery — a separate
project, only worth it if the GPU A/B on the flag shows the count term
dominating.

## MEASURED lane-T verdicts (Derecho, 2026-07-09, 8×A100 route-B NCCL,
## same-allocation A/B — latlon LL512-class atm, ocean LL288-class)

| arm | latlon ms/step | latlon SYPD | ocean ms/step | verdict |
|---|---|---|---|---|
| base | 6.48 | 25.4 | 33.09 | control |
| fused (`LEGOESM_LATLON_SPMD_FUSED_HALO=1`) | 7.36 | 22.3 | 32.72 | **latlon −12 % — default stays OFF**; ocean +1 % (noise) |
| xla (CP-combine 32 MiB + pipelined p2p) | 7.22 | 22.8 | — | **−10 % — not recommended as-is**; split the two flags in a follow-up arm before discarding |
| pgle (`JAX_ENABLE_PGLE=true`) | **5.97** | **27.5** | — | **+8.5 % — the winner**; recommend per-run on route-B latlon lanes |

Readings:
1. **Fused multi-pad loses at this size/count**: −29 % messages, but each
   message ~4× larger plus the pack/unpack concats — at LL512/np8 the
   per-message latency saved is smaller than the copy overhead added.
   The flag stays opt-in (it may still win at higher rank counts /
   smaller per-rank tiles where latency dominates — re-A/B there before
   discarding).
2. **PGLE's profile-guided re-scheduling is the real overlap win** —
   +8.5 % without touching the model.  Keep it per-run opt-in
   (recompiles after the profiling runs; AOT-incompatible), and wire it
   into the production route-B job env for latlon-class lanes.
3. **The combined xla arm hurt**; pipelined-p2p and the CP-combiner need
   separate arms to attribute (follow-up lane-T variant).
4. Caveats: one size per component, one repeat — treat sub-5 % deltas as
   noise; the cube base/xla arms and np16/24 rungs are still pending.

## Consequences — what to run next on Derecho/Levante

The count floor being (near-)reached in code moves the lever to the GPU
runtime, in this order (now wired as **lane T** in
`gpu_multinode_scaling.pbs` / the Levante twin, outputs under `_ab_tuning/`
which the aggregator deliberately skips):

1. **Fused-halo flag A/B** (`LEGOESM_LATLON_SPMD_FUSED_HALO=1`) on the
   latlon atm + ocean lanes — flip the default per the audit contract only
   with this receipt.
2. **XLA collective combining + pipelined p2p**
   (`--xla_gpu_collective_permute_combine_threshold_bytes`,
   `--xla_gpu_enable_pipelined_p2p`): the cube's 4 rounds/exchange are
   data-independent — the GPU CollectivePermute combiner can merge
   same-round CPs the SPMD partitioner emits separately; pipelining
   overlaps them with compute. Code-side round count cannot go below 4
   (matching floor), so this is where the cube ≤6-GPU curves have their
   remaining headroom.
3. **PGLE** (`JAX_ENABLE_PGLE=true`, profiling runs=3): profile-guided
   latency estimates re-schedule collectives; not defaulted (recompiles
   mid-job, AOT-incompatible).
4. `NCCL_NCHANNELS_PER_NET_PEER` sweep (4 → 8/16) if send/recv-bound after
   1–3.

Structural levers beyond that (separate projects, already ranked in
`derecho_levante_sota_review_2026-07.md` §4): cube np>6 sub-face tiled
production assembly (d_sw1/d_sw5/d_sw6) — THE cube lever past 6 GPUs; the
per-operator latlon pad merge above; per-device floor discipline for
production configs (≥ ~30k columns/GPU).

## Plot-pipeline fix shipped with this note

The SPMD bench lanes (`bench_atm_latlon_spmd_scaling`,
`bench_mpas_spmd_scaling`, `bench_ocean_latlon_spmd_scaling`,
`bench_cube_tiled_step_scaling`) now emit flat `sypd` / `mcells_per_s` /
`grid_type` / `resolution` fields (`metadata.tidy_throughput_fields`,
canonical 365.25-day formula), and `aggregate_bcw_scaling.py` ingests their
`.jsonl` records (n_devices ladder axis, component=ocean kept,
virtual-CPU-device proxy rows skipped). This closes the empty
latlon SYPD panels in `plot_cpu_vs_gpu_scaling.py` — those lanes were
invisible to the aggregator before.
