# Distance to the theoretical limit — panels a / c / e (2026-08-16)

Scope: the three panels the campaign is actively pushing — atmosphere
lat-lon GPU (a), MPAS icosahedral GPU (c), ocean lat-lon GPU (e).
Every number below is tagged CONFIRMED (receipted job id in the source
table / memory) or PLAUSIBLE (inferred, needs the named receipt).
"Theoretical limit" per panel = single-device compute of the same
per-device workload + unavoidable (unhidden) wire time, given the
measured fact that **halo collectives on this stack overlap nothing**
(2026-08-13 instrument: N same-channel rounds serialise at runtime in
NCCL; the interior/rim overlap pattern is refuted — cut rounds or
bytes, do not try to hide them).

## 0. Cross-cutting gate — verify the interconnect FIRST (cheap, blocks everything)

The running 192-GPU lat-lon job (26979367) prints the early-init
warning: *multi-node launch with no NCCL net plugin visible —
cross-node collectives will likely run on TCP sockets*.

- The warning is a heuristic (`early_init.py`: it only checks for
  `libnccl-net*`/`NCCL_NET_PLUGIN`). On Levante's Mellanox IB, NCCL's
  **built-in NET/IB verbs transport needs no plugin** — the plugin
  (aws-ofi-nccl) is for libfabric fabrics (Slingshot/EFA). So the
  warning is PLAUSIBLY a false alarm here — but nobody has receipted
  it on this cluster.
- **Receipt**: one 2-node arm with `NCCL_DEBUG=INFO`, grep
  `NET/IB` vs `NET/Socket`. Confirms: `Using network IB`. Refutes:
  `Using network Socket` — in which case every multi-NODE number in
  the figure (>4 GPUs on Levante) is transport-degraded and this is
  the single biggest lever in the whole campaign (sockets vs HDR200
  is ~an order of magnitude in cross-node bandwidth). Cost: minutes
  on gpu-devel. Scope: the debug line names the TRANSPORT only — a
  bandwidth claim needs per-link padded bytes and active pair counts
  on top.
- Consistency check available offline: the MPAS wire term (1.85 ms
  for the step's halo payload @64) can be converted to an implied
  bandwidth and compared against HDR200 (~25 GB/s/dir/node) vs TCP
  (~1-3 GB/s). If implied bandwidth is single-digit GB/s per node,
  the warning is real.

## c. MPAS icosahedral GPU — 35 % of the step is unexplained; splitting it is the work

CONFIRMED budget @ s9/64 GPUs, wide halo (6.82 ms/step):

| term | ms | status |
|---|---|---|
| local compute | 2.00 | CONFIRMED (single-GPU control, same per-device load) |
| halo wire (bytes) | 1.85 | CONFIRMED (ballast A/B, job 26919124) |
| halo latency | 0.57 | fitted (11 rounds × 52 µs) |
| unexplained | 2.40 | 35 % of step; skew REFUTED (all ranks within 24 µs, receipt e96c3d231); "inside the collectives" |

Floor with today's code structure = 2.00 + 1.85 + 0.57 ≈ 4.4 ms ⇒
**gap 2.4 ms (35 %)**. The 2.40 ms is UNEXPLAINED, not "recoverable" —
whether any of it can be removed is exactly what the receipts below
decide. Best case (all of it removed, 2× compute from the reorder
lever, one 52 µs round left): 1.00 + 1.85 + 0.05 ≈ 2.9 ms — treat
that as the optimistic bound, not a target. (codex r1: earlier
2.6-ms/"~55 % recoverable" arithmetic retracted.)

Ranked levers:

1. **Split the 2.40 ms "inside the collectives"** (biggest known
   unknown). GLM r1 mechanism read: the ballast's own receipt BOUNDS
   byte-coupled congestion small — 4× bytes ran only ~12 % over
   linear, so a congestion term proportional to bytes would have
   added ~+7 ms, not +0.36. The residual is therefore predominantly
   BYTE-INDEPENDENT: leading suspects are (i) pack/unpack
   gather-scatter kernels running at the same ~6×-off-roofline
   gather bandwidth as the compute (invisible to the wire budget),
   (ii) XLA pre/post-collective layout ops (flatten/pad/concat D2D
   copies, including the per-round-MAX zero-padding = fake bytes),
   (iii) dispatch gaps/event syncs between the 11 serialised rounds.
   Caveats GLM itself flags: the dense ballast bypasses the gather
   path, and 4× message size can flip the NCCL protocol (LL→Simple)
   — both confound the bound. Discriminators, cheapest first:
   (a) identity-permutation ppermute at identical shapes — splits
   pack/index cost from transport, needs no profiler (important:
   nsys does NOT record these collectives on this stack, receipt
   2026-08-13); (b) nsys bucketing of the comm window for the PACK
   and memcpy kernels only (those DO appear). CAVEAT (codex r1): a
   bare stride-k sweep does NOT discriminate — stride changes fill
   count AND fetched depth together, and only k ∈ {1, evals−1,
   evals} are correctness-sound; any sweep must record per-arm
   padded bytes + schedule and equalize bytes via ballast.
2. **Few-collective halo (`ragged_all_to_all`) at 64+**: currently a
   RECEIPTED 1.22× LOSS at s9/64 (unpruned zero-size slices), and it
   is 11 rounds → 2 collectives (cells + edges), not 1. Demoted as a
   direct lever — BUT it has a second life the reviewers jointly
   expose: overlap is refuted only for MANY same-channel rounds; a
   SINGLE collective was measured 79 % hidden with the
   latency-hiding scheduler on. Collapsing to 2 collectives is the
   precondition for hiding the 1.85 ms wire term under the 2.0 ms
   compute. Order: fix slice pruning → re-receipt ragged cost alone
   → then A/B latency-hiding overlap on top. Ceiling if both land:
   most of wire + latency disappears.
3. **Compute is ~6× off the HBM roofline** (CONFIRMED single-GPU nsys,
   job 26920190: not launch-bound, 32 kernels/step, real kernel time).
   GLM r1 sizing of the sub-levers (PLAUSIBLE until receipted):
   (i) LAYOUT — level-contiguous columns so each indexed gather moves
   a whole 104 B column, all fields batched through ONE shared-index
   gather: 2.5–4×, the biggest single lever; (ii) horizontal FUSION
   of the 32 kernels (XLA will not fuse across gather/scatter
   boundaries): 2–3×; (iii) RCM/Hilbert local reorder: only
   1.2–1.6× — with contiguous columns coalescing is already decent;
   its real value is contiguous halo PACK, i.e. it attacks the
   2.40 ms residual too. Net plausible: 2.0 → 0.6–0.9 ms. Receipts:
   single-GPU A/Bs, no cluster time.
4. Bytes: per-round-MAX padding (`sharded_dynamics.py`) means byte
   cuts only pay off at the round's fattest pair. Pricing: the
   0.37 step-%/byte-% figure is the 32-GPU EMPIRICAL ratio
   (10.5/28); the wire-linear price implied by the 64-GPU budget is
   0.27 (1.85/6.82) — quote whichever matches the device count.
   METIS −28 % padded bytes is config-only and already receipted
   (−10.5 % @32); default-flip still blocked on the
   disconnected-components question. Note (codex r1): Vizing bounds
   only the CURRENT ownership graph — an ownership that lowers
   max_degree can still cut rounds; the degree-prototype probe
   reported a 9.4 % padded-byte cut. Low expected value, not
   refuted.
5. **bf16 halo payload** (GLM r1 addition): halves the 1.85 ms wire
   term; does NOT touch the byte-independent residual. Costs:
   reproducibility across device counts, AD/conservation impact on
   halo-adjacent gradients, and it breaks bit-parity gates —
   needs its own truth-tier validation before any ladder point.
   Park behind the layout/overlap levers.

## a. Atmosphere lat-lon GPU — collective COUNT is the wall; two structural exits exist

State (CONFIRMED): packed stage-entry exchange 25 → 13
collectives/step, −11.8 % @128, bit-identical, opt-in env. LL2304@192
and LL2880@192 arms in flight (job 26979367). CPU/gloo @128: packed
exchange |effect| < 3 % — compute-dominated there, so panel-a levers
are GPU-lane levers.

Known blockers and exits:

1. **Bucket E (13 → 7 collectives): count alone is NOT the prize.**
   A production A/B/A2 already removed 6 of 13 collectives and bought
   only 1.7 % — marginal collective cost is ~18 µs; bytes/wait
   dominate (codex r1, campaign log). The recompute exit stays
   blocked for the right reason (recomputed ghost is ~1 ULP off —
   copying is exact, re-deriving is not). The copy exit (carry
   derived `div_dp` ghost rows as extra payload) is therefore
   PLAUSIBLE only if it also cuts WAIT, and it ADDS bytes — price it
   with a payload-aware receipt (record per-arm bytes; interleaved
   arms, ≥40 steps) before building. No longer "predicted to win".
2. **1-D lat bands thin out at high device counts**: LL2304@192 = 12
   rows/band; halo (2 rows/side under packed exchange) is then a
   third of the band — surface/volume is the scaling killer past
   ~192. A 2-D SPMD tile pad body exists
   (`latlon_spmd.py::make_latlon_2d_pad_body`) but the GPU atm bench
   is hard-wired 1-D (CONFIRMED, codex r1). Corrections to the
   earlier claim (codex r1): panel d's CPU 0.83 @512 is the SEPARATE
   MPI wall-pole lane, NOT this SPMD body — it is no performance
   receipt for the GPU path; and in the SPMD body the pole-fold
   collective executes on EVERY tile (both `where` branches run), so
   fold traffic does not localise to polar tiles. Receipt order:
   microbench the 2-D pad body alone @16-64 first (prices fold +
   corner traffic); only then wire the GPU lane. The √-bytes claim
   is unpriced until then.
3. Harness: the in-flight 192 arms run **12 timed steps**; the
   campaign's own lesson is 12 steps = 7.7 % spread on identical
   arms, 60 steps = 0.3 %. Fine for "does it run at 192 / rough
   GC/s", NOT fine as the paper point or for any A/B delta claim at
   the −5..−12 % scale of these levers. Re-receipt the keepers at
   ≥40 steps.

## e. Ocean lat-lon GPU — first find out what the 15.81 ms IS

CORRECTION (codex r1, CRITICAL): the 15.81 ms @128 receipt runs the
production **implicit** barotropic solver (`implicit_cn`, the bench
default) — there IS no substep loop in that number, so the wide-halo
substep lever cannot improve it. The bench refuses `--wide-halo`
unless `--baro-solver explicit_substep` is selected. The earlier
"barotropic substep loop is the un-attacked term" framing was wrong
for this panel's measured point.

1. **Budget-split the implicit-CN step first.** The 15.81 ms @128 has
   no compute/wire/latency split like panel c's. The implicit solver's
   iterative pressure solve carries its own collectives (halo +
   global reductions per iteration) — that, not a substep loop, is
   the comm-cadence suspect at 128. Instruments: single-device
   control (same per-device load) + a ported ballast knob (bench-side
   change). Rank panel-e levers from that split, not by analogy to
   MPAS.
2. **Wide-halo explicit-substep lane: separate ladder, own baseline.**
   `barotropic_substeps_wide_halo_latlon_cgrid` exists (opt-in;
   guards: explicit solver + local clamp mandatory, no EEN
   precompute, no drag-substep, no NEMO AB3/AM4 filters or MLF
   before-state, 1-D bands only, chunk×reach ≤ min band height). A
   fair A/B is explicit-substep-baseline vs explicit-substep+wide,
   both arms clamp-pinned — NEVER vs the implicit 15.81 ms point.
   Harness note: the parity gate caps runs at 8 steps — run parity
   as a separate smoke, the timed A/B at ≥40 steps. PLAUSIBLE win on
   the explicit lane; whether the explicit lane can beat implicit-CN
   at 128 at all is itself the first question the pair of ladders
   answers. Soundness conditions (GLM r1) the implementation must
   hold and the A/B must gate on: halo width = chunk × substep
   stencil reach PLUS the averaging-filter radius; the strip
   recompute must be the SAME compiled arithmetic as the owner's
   (non-identical op order seeds seam discontinuities that propagate
   at gravity-wave speed); never hold stale halos across a chunk —
   recompute them; chunk static at trace time.
3. 2-D decomposition for the ocean is a BUILD, not a guard-lift
   (codex r1): the production ocean SPMD wrapper is wholly 1-D
   band-based — no 2-D state/geometry/sharding factory exists. Park
   until the panel-a 2-D microbench prices the pattern.

## What NOT to build (measured refutations, still binding)

- Interior/rim comm-compute overlap with the MANY-round halo (any
  panel): refuted 2026-08-13 — rounds serialise in NCCL regardless
  of schedule windows. (GLM r1 re-proposed it; the 2026-08-13
  measurement stands. The surviving form is overlap of a
  SINGLE/few-collective exchange — see the ragged item.)
- REVIEWER DISAGREEMENT log: GLM prices collective count as
  second-order (consistent with the 1.7 % lat-lon receipt) and byte
  serialisation as binding; codex demands byte-equalized arms before
  any round-count claim. Both are honoured above: no round-count
  lever proceeds without a bytes-pinned receipt.
- Byte-aware MPAS ownership (dissolving thin contacts): wrong end of
  the per-round-MAX padding distribution — receipted 0.0 %. Scope
  note (codex r1): Vizing-optimality binds only the CURRENT graph;
  an ownership lowering max_degree itself is unrefuted, merely
  low-value (degree prototype: 9.4 % padded-byte cut).
- Python-order reordering of ppermutes: byte-identical XLA module.
- Ragged one-collective halo at 64+ as-is: receipted 1.22× LOSS at
  s9/64 (unpruned zero-size slices; and it is 2 collectives, not 1).

## Recommended order (each step gated on its receipt)

1. NCCL transport receipt (§0) — minutes, gates every multi-NODE
   point. `NCCL_DEBUG=INFO` names the transport only; a bandwidth
   claim additionally needs per-link padded bytes + active pairs.
2. MPAS 2.40 ms split (§c1) — identity-permutation ppermute probe
   first (splits pack cost from transport, no profiler needed), then
   bytes-equalized arms; NOT a bare stride sweep.
3. MPAS layout + fusion, single-GPU (§c3) — parallel track, no
   cluster time; biggest plausible levers on the 6×-off-roofline
   compute (level-contiguous columns, batched shared-index gathers,
   horizontal fusion); RCM reorder third within this track.
4. Ocean implicit-CN budget split (§e1); explicit-substep wide-halo
   ladder as its own pair (§e2).
5. Lat-lon bucket-E-by-copy only WITH payload-aware pricing (§a1) —
   count-only justification is refuted.
6. Lat-lon 2-D pad-body microbench (§a2) — the ≥256-device enabler
   for the paper's right edge; wire the GPU lane only if the
   microbench pays. (GLM r1 dissent: at ≤64 ranks 2-D raises
   perimeter/area and helps only latency-bound regimes — consistent
   with parking it behind everything else and aiming it at ≥192.)
7. Ragged slice-pruning fix → few-collective exchange → overlap A/B
   (§c2) — the only path on this stack where the wire term can be
   HIDDEN rather than shrunk.
