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

**Horizontal tiling — the right, still-UNTESTED lever.** Because the dominant
(36-pass) acoustic loop is column-local, the domain can be split into horizontal
tiles that each fit cache, with **no acoustic-substep halo at all**. Only the 3
slow-tendency passes need a small (±2–4) halo. A `scan`/`vmap` over ~128²–192²
tiles — slow tendency on tile+halo, substeps on the tile interior — should keep
each tile cache-resident and recover much of the 1.45× with only a few-percent
halo overhead (NOT the ~1.4× fat-halo penalty the first draft wrongly assumed).
This is plausible but **not yet implemented or measured** — the honest status.

**Pallas — not required to first try.** A custom on-chip-blocked kernel would help
further, but the column-local structure means a *plain-JAX* tiled step is worth
trying first; "needs Pallas" is **not** established.

## Caveats / honest limits

- The cheap-tiling win is a hypothesis backed by the column-local structure, not a
  measurement. Next step: implement a tiled step, validate **bit-identical** vs
  the untiled `step()` (never regress), measure N256 tiled vs untiled with
  `--repeat`, codex-review.
- If `substep_horizontal_acoustic=True` (horizontal-acoustic mode) were used
  instead, the substeps DO couple ±1/substep and the fat-halo analysis would
  re-apply — but `step_halo` (DD) rejects that mode, and the bench does not use
  it.
- Single-GPU only matters when a single device must hold > ~1.1 M cells; multi-GPU
  / MPI decomposition that keeps each device's tile under that threshold sidesteps
  the cliff for the column-local mode (gated on ≥2 devices; this host has 1 GPU +
  CPU-only MPI).

Bottom line: the 1.45× large-N cliff is real and measured; the realistic
first-line fix is a **plain-JAX horizontal tile loop** (cheap because the acoustic
substeps are column-local), to be implemented + validated + benchmarked next — the
first draft's "fat-halo net-negative / needs Pallas" verdict was wrong (codex).
