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

## 0. Cross-cutting gate — RESOLVED: transport is InfiniBand (CONFIRMED)

**Receipt landed 2026-08-16, job 26996570 (2 vader nodes, 4 ranks,
`NCCL_DEBUG=INFO`): all four ranks print `NCCL INFO Using network IB`.**
The early-init sockets warning is a false alarm on Levante — NCCL's
builtin IB verbs transport needs no plugin. Every multi-node figure
point stands as measured. Follow-up worth one line: the warning's
heuristic could check for a usable verbs device before claiming
"likely TCP sockets". Original gate reasoning kept below for the
record.

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

## c. MPAS icosahedral GPU — every term of the step is now named; comm overhead is the target

CONFIRMED budget @ s9/64 GPUs, wide halo (6.820 ms/step) — three-arm
split receipt (job 26942819: interleaved full / no-comm / no-staging
arms, 60 steps, spreads 0.3–0.6 %). Three decimals because the
rounded values do not visibly sum:

| term | ms | how |
|---|---|---|
| no-halo program (owned+ghost compute, masking, machinery) | 3.460 | MEASURED (no-staging arm) |
| halo staging (gather/concat/scatter) | 0.315 | MEASURED (no-comm − no-staging) |
| halo wire (payload slope) | 1.815 | MEASURED (ballast N=2 delta; N=4 from job 26919124 was super-linear 3.36 vs 3.00, so this is a payload SLOPE, not pure wire) |
| in-collective wait / per-op overhead | 1.230 | arithmetic residual (full − no-comm − 1.815; exact) |

The former "2.40 unexplained" resolves into the 1.230 ms residual
(rank skew refuted at <24 µs) plus ~1.46 ms by which the no-halo
program exceeds the 2.00 ms single-GPU control. Denominator resolved
(job 26998727, production builders WITH production's nlev size
weights, s9 lloyd-0 reorder-for-64 sfc): **the wide depth-9 fill is
14 rounds** (max_degree 13), so the residual prices at ~88 µs/round.
The earlier "11 collectives" figure's mesh-state provenance is
unresolved — do not reuse it. Two new facts from the same receipt:
(a) the wide schedule uses 14 rounds against a degree bound of 13 —
one recolouring round may be recoverable (~88 µs); (b) per-round-MAX
padding ships ~2.0× the true cell halo (36,953 padded vs 18,377
actual rows), so exact per-pair sizing — what the ragged path does —
would roughly HALVE the 1.815 ms payload term. That upgrades the
ragged-after-pruning lever from latency play to latency+bytes play.

Offline row-count receipt (job 26997346, production partition
builders): sfc at wide depth 9 computes 59,338 cell rows vs 40,961
owned (×1.449; edges ×1.429), pricing ghost+padding rows at
**+0.86–0.90 ms of the 1.46** — an upper-bound PROXY (the 49 ns/row
rate embeds the control's fixed costs and cell/edge mix; codex r2 +
GLM r2 agree). Falsifier, single GPU, cheap: run padded shapes
{40,961; ~50k; 59,338} same-program and fit the slope — the slope is
the price. Masking zeroes OUTPUTS, not work (confirmed in source), so
these rows are genuinely computed. Remainder after the proxy:
~0.6 ms masking/machinery, bounded not attributed.

Partitioner check (job 26997437, same caveats): metis ×1.342 cell
rows (−0.21 ms) at 16 vs 14 probe rounds. With the 88–112 µs/round
range this is anywhere from a small metis GAIN (~+0.04 ms net) to a
wash — NOT decided offline; and the offline round counts are not
production's. GPU timing of a metis wide-halo arm decides it; metis's
receipted value meanwhile stays the byte cut (−10.5 % @32).
GLM's r1 pack-cost hypothesis is REFUTED at 64: staging is 0.315 ms
(~5 % of step), not the residual.

**GLM r2 mechanism read of the comm stack (PLAUSIBLE, cheap to
falsify): the exchange moves ~6 GB/s effective against ~25 GB/s
available, so the "wire" slope itself is partly per-op overhead —
wire and wait would then compress TOGETHER under a fused/multi-channel
exchange.** Falsification ladder, env-vars before code, one variable
per arm: `NCCL_MAX_NCHANNELS` (4→8), `NCCL_PROTO` (LL128 vs Simple),
then round fusion. Caveat from the campaign's own receipt: full
CUDA-graph capture of collectives was already measured HARMFUL
(cmdbuf 1.268×, #1575) — graph capture is NOT on this ladder.

LADDER RESULT — **CONFIRMED (jobs 26999539 + 27002564)**: the
multi-channel-p2p env pair (`NCCL_MIN/MAX_NCHANNELS=8` +
`NCCL_P2P_NET_CHUNKSIZE=131072`) is a large, replicated, env-only
win at s9@64 wide halo:
- mcp2p step 5.87/5.88/5.91 ms across THREE arms in TWO jobs on
  different node sets (spread 0.04 ms) vs same-job base 7.18/7.26
  and best-ever base 6.75 → **−0.9 to −1.3 ms per step (−13 to
  −18 %)**. Two-point fit (26999539): intercept −0.53, slope −0.32 —
  it compresses BOTH per-op overhead and payload cost, as GLM r2
  predicted.
- mcp2p is also far LESS node-set-sensitive than base (base wanders
  6.75–7.26 across days/jobs; mcp2p does not) — consistent with the
  variable part of the step being the serialised per-op overhead.
- LL128 protocol: within noise (expected 0.1–0.3 ms win absent).
- Validity-log arms kept failing (NCCL_DEBUG_FILE runs); the knobs'
  effect is evidenced by the replicated delta itself — an inert arm
  cannot produce −1.3 ms.
SECOND-LANE RECEIPT (job 27004277, LL2304@32 packed exchange,
palindrome pairs): base 10.521 ms (spread 0.038) vs mcp2p 9.884
(spread 0.006) — **−6.1 %, CONFIRM band: the pair TRANSFERS to the
lat-lon lane** (panels a and, by the same exchange machinery, e).
NEXT (decision + receipts): (1) adopt the two vars in the production
GPU launch env — USER DECISION, it changes every multi-GPU run;
(2) receipt at 128+ devices (the 192 arms in queue predate the vars);
(3) rerun the three-arm split under mcp2p, then re-rank the
fused-exchange build against the smaller comm stack.
Ops note, 2026-08-16 evening: three consecutive jobs lost to sick
nodes/stragglers (l50003 CUDA fault, l50139 hang, l50027/l50009
stragglers) — exclude-lists are load-bearing today.

Floor reading from the split: the step is 3.460 compute-side + 3.360
comm-side (0.315 staging + 1.815 payload slope + 1.230 residual).
Optimistic bound if the named levers all land (fusion/layout 2× on
the compute side, fused few-collective exchange compressing
wire+wait): ~2 ms — treat as bound, not target. (codex r1: earlier
2.6-ms/"~55 % recoverable" arithmetic retracted; superseded again by
the 26942819 split.)

Ranked levers:

1. ~~Split the 2.40 ms~~ **DONE — receipt 26942819** (the
   identity-permutation instrument both reviews asked for already
   existed as the no-comm/no-staging knobs and had been run at 64).
   Result above. What remains actionable from the split:
   (i) **comm overhead 1.230 ms residual (+ part of the 1.815
   slope)** — attacked first by the NCCL env-var ladder (GLM r2,
   costs one small A/B job), then by the fused few-collective
   exchange (item 2), NOT by repartitioning (skew refuted);
   (ii) **ghost+padding rows, 0.86–0.90 ms proxy** — single-GPU
   shape-slope falsifier first; then the stride-k diff (only k ∈
   {1, evals−1, evals} sound, per-arm padded bytes + schedule
   recorded) and/or padding size-classes instead of one global max;
   (iii) **~0.6 ms masking/machinery** — HLO census of the
   no-staging arm (single-node, cheap) before any lever is named.
2a. **2026-08-18 receipts (jobs 27039649 + 27040145), read before
   touching the ragged/wide levers:**
   (i) the WIDE-halo step is only PHYSICALLY equivalent to the exact
   step (documented outer-ring recompute, band 1e-2 — its unit test),
   so the bench parity gate (exact-exchange tolerances, u x64 1e-5)
   fails wide configs BY CONTRACT (u 2.66e-3 at s6@8, x64-confirmed,
   narrow clean at 1e-9). Not a regression; do not parity-gate wide
   arms with exact tolerances.
   (ii) ragged and coloured ppermute produce IDENTICAL fields under
   wide at s6@8 — the ragged machinery moves the same rows multi-node;
   its correctness gap is closed by a NARROW multi-host parity gate.
   (iii) XLA's multi-host ragged decomposer flag HANGS compile at 2
   nodes (arms b/d) — REFUTED as a lever on this jaxlib; the one-shot
   kernel flag compiles and runs (arm c) and stays on the ladder
   (GLM predicts inert cross-node; the ladder decides).
2. **Few-collective halo (`ragged_all_to_all`) at 64+**: currently a
   RECEIPTED 1.22× LOSS at s9/64 (unpruned zero-size slices), and it
   is 11 rounds → 2 collectives (cells + edges), not 1. Demoted as a
   direct lever — BUT it has a second life the reviewers jointly
   expose: overlap is refuted only for MANY same-channel rounds; a
   SINGLE collective was measured 79 % hidden with the
   latency-hiding scheduler on. Collapsing to 2 collectives is the
   precondition for hiding the 1.815 ms payload term under the 2.0 ms
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
   its real value is contiguous halo PACK (staging is only 0.315 ms,
   so that side prize is small). Net plausible: 2.0 → 0.6–0.9 ms. Receipts:
   single-GPU A/Bs, no cluster time.
4. Bytes: per-round-MAX padding (`sharded_dynamics.py`) means byte
   cuts only pay off at the round's fattest pair. Pricing: the
   0.37 step-%/byte-% figure is the 32-GPU EMPIRICAL ratio
   (10.5/28); the wire-linear price implied by the 64-GPU budget is
   0.27 (1.815/6.82) — quote whichever matches the device count.
   METIS −28 % padded bytes is config-only and already receipted
   (−10.5 % @32); default-flip still blocked on the
   disconnected-components question. Note (codex r1): Vizing bounds
   only the CURRENT ownership graph — an ownership that lowers
   max_degree can still cut rounds; the degree-prototype probe
   reported a 9.4 % padded-byte cut. Low expected value, not
   refuted.
5. **bf16 halo payload** (GLM r1 addition): halves the 1.815 ms payload
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
2. ~~MPAS 2.40 ms split~~ DONE (receipt 26942819; §c1). Next in its
   place: NCCL env-var ladder A/B (channels/proto) on the comm residual,
   the single-GPU ghost-row slope fit, and the HLO census of the
   ~0.6 ms masking/machinery — all cheap.
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
