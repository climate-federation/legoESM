# CRM GPU large-N throughput cliff — investigation (2026-06-09)

The plane CRM (compressible-Euler) fp32 GPU throughput drops ~1.45× once the
domain grows past ~1.1 M cells. This note records what is **measured**, what was
**wrong in the first draft** (caught by codex adversarial review), and the
**actual** tractable lever.

> **Correction (codex review).** The first version of this note claimed the
> acoustic substep loop couples horizontally and that temporal-blocking tiling
> needs an `n_substeps`-wide halo (→ net-negative). That is **wrong** for the
> benchmarked configuration: the bench runs the **vertical-only / column-local**
> semi-implicit acoustic substeps, which have **no horizontal coupling**. The
> fat-halo model does not apply, and cheap horizontal tiling is *plausible*, not
> net-negative. The `>100 % HBM` figure was also overstated as proof of L2
> residency — it is a *modeled* upper-bound traffic estimate, not measured bytes.
> Conclusions revised below.

**Host:** 1× RTX 5090 Laptop GPU (24 GB). fp32, nlev=30, dx=2000 m, dt=2 s,
`scripts/bench/bench_crm_gpu_scaling.py` (default `semi_implicit_acoustic=True`,
`substep_horizontal_acoustic=False` ⇒ column-local acoustic).

## What is measured (robust, 2 passes, <2 % spread)

| nx  | cells   | step (ms) | throughput |
|----:|--------:|----------:|-----------:|
| 128 | 491,520 | 1.85–1.87 | 262–266 Mc/s |
| 192 |1,105,920| 4.13–4.15 | 267–268 Mc/s |
| 256 |1,966,080| 10.73–10.79 | 182–183 Mc/s |

- **Real, reproducible 1.45× per-cell throughput cliff** between N192 (1.1 M
  cells, fast) and N256 (1.97 M cells, slow). Directly timed (Mc/s, ms) — not
  model-dependent. The threshold (1.1 M–2 M cells) is consistent with the working
  set spilling a cache level on this GPU.
- (The bench's `≈% HBM` column is a 6–20 bytes/cell/pass *model*, not measured
  HBM bytes; treat it as indicative only — do not read "114 %" as "served from
  L2".)

## Mechanism

The full step is SSP-RK3 split-explicit: per RK stage, one **slow tendency**
(horizontal: pressure gradient, divergence, advection, del4 hyperdiffusion —
needs ±2-ish halo) followed by **12 acoustic substeps**. In the benchmarked
config those substeps are **column-local** (`plane_acoustic_substeps_semi_implicit`
→ `_semi_implicit_acoustic_column_kernel`; `u`,`v` pass through unchanged) — each
horizontal column is independent. So the 36 substep passes per step (3 stages ×
12) carry **no horizontal coupling**; the only horizontal stencils are the 3
slow-tendency passes.

## Levers

**XLA flags (zero-code) — no effect.** latency-hiding scheduler + autotune L4 +
while-loop double buffering: 184 vs 182 Mc/s. XLA does not tile a stencil loop to
keep a block cache-resident.

**Horizontal tiling — TESTED, and it does NOT help the production config.** A
faithful PoC (`scripts/tmp/_poc_acoustic_tiling.py`, since removed) ran the real
acoustic substep kernel full-field vs 2×2-tiled on N256:

| config | full | 2×2 tiled | result |
|--------|-----:|----------:|--------|
| vertical-only (`semi_implicit`, **dry**) | 2.91 ms | 1.75 ms | **1.66× (bit-identical)** |

So tiling the *vertical-only, dry* acoustic loop is a real win (anti-DCE checked).
**But that is not the production configuration**, per a second codex review:

1. **Production uses `substep_horizontal_acoustic=True`** (all runners:
   `run_rcemip_plane`, `run_les_plane`, `run_gate_plane`, `run_lba_plane`) →
   `plane_acoustic_substeps_si_horizontal`, whose substeps **DO couple
   horizontally** (per-substep pressure-gradient + divergence, ±1). Tiling that
   needs a halo of width `n_substeps` (=12) → a 152² tile that itself spills →
   **net-negative** (the original analysis, correct *for this mode*). The 1.66×
   only exists in the vertical-only mode production does not run.
2. **Moist `b_moist` is not column-local.** `_acoustic_moist_buoyancy_w` subtracts
   the *horizontal mean* of qv/qcond/θ′ (the SAM closure — same coupling that
   forced the MPI `acoustic_moist_global_mean` opt-in). The PoC's bit-identity
   held only because it was dry (n_tracers=0); with real moisture each tile would
   use its slab-local mean → wrong physics unless the global b_moist is precomputed
   and sliced.
3. The PoC's isolated 1.66× would be **Amdahl-limited** in the full SSP-RK step
   anyway (slow tendency, RK combination, reassembly remain), and the bench's
   bandwidth model / "L2-resident" label is indicative, not measured.

⇒ **Intra-kernel horizontal tiling is not a production CRM GPU lever.** The
production acoustic substeps couple horizontally (fat-halo → net-negative) and the
moist buoyancy couples through a planar mean. The win exists only for a dry
vertical-only mode that production does not use.

(Aside worth fixing separately: `bench_crm_gpu_scaling.py` runs the *vertical-only*
acoustic mode, so its absolute numbers under-represent the production
`si_horizontal` cost.)

## The lever that remains

- **Multi-device domain decomposition** keeps each device's tile below the cache
  cliff (~1.1 M cells) regardless of acoustic mode — the certified `step_halo` DD
  provides it, gated on ≥2 devices (this host has 1 GPU + CPU-only MPI). Note
  `step_halo` currently rejects `substep_horizontal_acoustic` under multi-rank, so
  DD large-N today implies the vertical-only acoustic mode.
- **Single-GPU large-N in the production `si_horizontal` mode**: no clean JAX
  lever (fat-halo tiling net-negative); would need a Pallas on-chip-blocked
  acoustic kernel — large, GPU-specific, deferred.

Bottom line (after two codex reviews): the 1.45× large-N cache cliff is real and
measured; horizontal tiling recovers it **only** in the dry vertical-only acoustic
mode (PoC: 1.66×), which production does not use. For the production
horizontally-coupled + moist config, intra-kernel tiling does not pay; the lever is
multi-device decomposition (≥2 GPUs).
