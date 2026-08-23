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
2b. **2026-08-18 afternoon receipts — the ragged lever LANDED, the
   overlap lever DIED, both cleanly:**
   (i) RAGGED WIN at 64 (job 27040575, narrow parity green, 60-step
   arms): ragged wide 6.25/6.29 ms vs coloured controls 6.79/6.61
   (mean 6.70, spread 2.7%) — ratio 0.933. The 1.220x loss is GONE on
   the current stack with no XLA flags; the one-shot flag is inert
   (0.939). Replicated off-arms 6.17/6.28 in 27041201 (best 6.225
   mean). The zero-size-slice pathology no longer binds at 64.
   (ii) OVERLAP-BY-SCHEDULER REFUTED with mechanism (jobs 27041261 +
   27041201): the latency-hiding scheduler DOES lower both ragged
   collectives as async start/done pairs (HLO dump receipt,
   is_sync:false) — the flag is not inert — yet the step gets 25-31%
   SLOWER and wildly noisy (median 7.8-8.5, per-step 5.4-15.7). With
   the wide fill at step start gating all compute there is NO
   independent work to hide behind (GLM r3 predicted exactly this);
   the async machinery only buys collisions. Scheduler flags are OFF
   the table for this program shape.
   (iii) Surviving overlap path = RESTRUCTURE: interior/rim split
   (compute owned-interior tendencies while the exchange flies, then
   the rim), or cells/edges fills placed at their first consumers.
   Code change in the sharded step, not an env var. Unpriced.
   (iv) mcp2p on ragged WITH overlap flags: no help (7.81/8.49 vs
   7.83/7.78). WITHOUT overlap (job 27041921): **NEW LANE BEST
   5.70/5.75 ms** vs same-job ragged base 6.26/6.17 (ratio 0.921,
   spread 1.4%) — the pair transfers to the 2-collective exchange.
   s9@64 progression, all receipted: 9.60 baseline -> 8.40
   size-colouring -> 6.98 wide -> 5.87 coloured+mcp2p -> 5.73
   ragged+mcp2p (cumulative -40%). On the figure as step 4.
2c. **Interior/rim split — the surviving wire-hiding design (2026-08-18,
   unbuilt, dual design review pending).** Preconditions now all
   receipted: async lowering of the 2 ragged collectives works (HLO
   27041261); overlap fails ONLY because the fill gates every consumer
   (27041201); single-collective overlap hides 79% when independent
   work exists (microbench). Design: make the INTERIOR tendency
   independent of the fill at the DEPENDENCY level — no manual sync:
   (i) compute the full-field tendency from the PRE-fill local buffer
   (interior cells correct by construction: their whole stencil is
   owned; rim cells garbage); (ii) after the ragged fill lands,
   recompute ONLY the rim band (cells within stencil reach of a ghost)
   via the ring-distance indices the wide-halo machinery already
   carries, and scatter into the tendency; (iii) XLA's latency-hiding
   then hoists (i) between ragged-start and ragged-done on its own —
   the mechanism the A/B proved functional. Cost: rim recompute is a
   subset gather-compute (~surface/volume fraction, at s9@64 wide the
   ghost fraction is ~0.45 of rows — the win shrinks as rim grows, so
   price at NARROW depth too, where rim is ~10-15%). Risks (for
   review): masked-garbage contamination via reductions (any global
   sum before the rim patch must mask rim rows); double-compute
   determinism (rim rows computed twice must take the SECOND value
   bitwise); AD through the scatter (VJP of a scatter-overwrite is
   well-defined but must be tested with check_grads); pytree/shape
   stability of the rim index sets (static, from the partition build).
   Compute cost bound: interior pass over all rows + rim pass over rim
   rows = 1 + rim_frac of today's compute; wins iff hidden wire >
   rim_frac * compute.
   DESIGN REVIEW VERDICTS (2026-08-18, both reviewers):
   - GLM: mechanism sound (shared reads don't serialize; SSA), rim
     subset recompute is the right shape (break-even rim ~40-60%),
     scatter-overwrite with unique_indices is AD-exact (grads flow
     through the SECOND value by construction), ghosts must be
     STALE-FINITE never NaN (0*NaN backward hazard), receipts =
     bitwise-vs-unsplit + NaN-poison forward. Prefer the LAYOUT-SPLIT
     formulation (interior-prefix/rim-suffix packed arrays — no
     scatter at all) if reindexing is acceptable.
   - codex: the win is NOT obtainable by scattering rows from a second
     full call to the tendency function (it always evaluates full
     arrays) — a SUBSET/rim execution path in the RHS, or a compact
     static rim submesh fed to the unchanged RHS, is REQUIRED. Rim
     membership must come from the ACTUAL dependency graph (PV via
     vertices + kiteAreas, APVM, del4, edgesOnEdge tangential paths),
     not cellsOnCell hops alone — the committed `_build_rim_rings`
     (cell-hop BFS) is a conservative approximation whose width must
     be VERIFIED by the ghost-poison test before any receipt. Change
     surface: `_ragged_halo_fill` (factor pre-fill buffers),
     `_make_local_wide_step` (launch fill -> interior work -> rim
     patch of all four tendency channels), new static rim plan
     threaded through `make_voronoi_sharded_step`. Seven existing
     gates must stay green (wide equivalence, AD, native parity,
     SPMD-vs-serial, ragged schedule, multicontroller parity) + a new
     GPU split-vs-unsplit integration gate (CPU cannot run the ragged
     collective).
   STATUS: spec complete, build NOT started — this is the next major
   work item; est. multi-session. The rim ring builder + tests are
   committed (65eaea84d).
   BUILD ROUTE (2026-08-18, reuse discovery): the compact rim submesh
   needs NO new partition logic — feed the existing partition builder a
   synthetic two-rank ownership (device d's rim cells owned by rank 0,
   everything else rank 1, halo_depth = RHS stencil radius) and its
   closure IS the rim closure; `build_local_mesh` then emits the
   remapped compact mesh the unchanged RHS runs on (external refs -1,
   already masked by the TRiSK operators). Remaining new code: the
   static gather map (device-local buffer -> submesh order), the rim
   scatter of the four tendency channels, per-device padding to the max
   rim size, and the poison-verified stencil width. Phase 1 = builder +
   CPU test (operator on submesh == operator on full mesh at rim rows).
   PHASE-1 CODE REVIEW (codex, 2026-08-18): landed (a90a68db6) but DO
   NOT WIRE until three blockers clear:
   (i) edge SCATTER set must come from `edge_rim` (predicate
   `0 <= edge_rim <= width` — cut edges are 0 under the min rule)
   restricted to device-owned shard rows; the synthetic partition's
   lower-cell edge-ownership rule is unrelated to production rows and
   can miss / mis-scatter / double-patch. Use rim_part ONLY for the
   submesh.
   (ii) RESOLVED 2026-08-18 (test_rim_plan_closure.py): the FULL
   tendency (energy PV via vertices + kites, APVM, del4 1e16) on the
   submesh matches the global tendency at every rim entity to the f64
   floor (1.39e-10, plateau across depths 2-4) at closure depth 2,
   while depth 1 fails at 1.06e-3 — seven orders of separation, gate
   provably non-vacuous. codex's partially-closed-vertex hazard does
   not bite this RHS config at depth >= 2 (a real gap would keep
   improving with depth; a plateau is closure). Gate must be re-run if
   the RHS grows a new operator.
   (iii) per-device `partition_voronoi_mesh` is a Python-loop setup
   wall at s9/64 (~billions of iterations incl. rank-1's whole-mesh
   comm schedule that the plan never uses) — replace with a
   vectorized, comm-free compact-closure builder before any
   production-scale receipt.
   GLM review landed 2026-08-18 evening (r4) — the WIRING CHECKLIST,
   each item mechanical:
   (1) INTERIOR-EDGE LEAK (the big one): interior-classified edges with
   wide tangential (edgesOnEdge) stencils read unfilled halo in the
   pre-fill pass and are never overwritten; the static closure gate is
   structurally blind (compares rim/scatter rows only). Fixes: edge rim
   from edge-graph BFS or an overhang margin on the predicate, AND the
   wiring receipt must be a BITWISE diff of ALL owned rows vs the
   unsplit step (not rim rows only). rim FAR-edge exact-cover tripwire
   added to the builder (this commit).
   (2) padded-submesh runs need where-based masking (multiplicative
   mask x NaN garbage leaks through row reductions) and a NaN-canary
   bitwise assertion; mask BEFORE indexing (negative take wraps).
   (3) buffer discipline: gather must read the post-wait receive
   buffer (double-buffer if reused); no per-device reduction between
   interior pass and rim overwrite (diagnostics/CFL stats included);
   the tendency buffer must never be read by the interior pass.
   (4) branch-dependent stencils (limiters, APVM upwind selection)
   mean smooth-IC gates under-cover: production runs need a masked-
   read assertion (any -1/external read on a non-pad row = hard error)
   rather than outcome-only gates.
   (5) sizing holds (submesh 0.2-1 ms vs 2-3 ms window) IFF the
   submesh RHS is one pre-compiled static-shape call; budget against
   the MIN rank window, not the mean.
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
2b. **The 192-rank hang RESOLVED to a thin-band trigger (2026-08-18,
   job 27036060): LL2880@192 (15-row bands) runs clean — 6.74 ms,
   63.98 GC/s, the lane's best throughput — while LL2304@192 (12-row
   bands) deadlocks NCCL channel setup on the first call.** Seven
   candidates refuted along the way (latency-hiding scheduler,
   fused-halo path, XLA comm-splitting, CUMEM allocator, mcp2p pair,
   P2P transport, eager connect); rank count refuted by this receipt
   and by 16-row bands at 144 working. The same 12-row program runs
   fine on CPU virtual devices, so the comm pattern is legal —
   NCCL-specific init behaviour under the thin-band halo/pole-window
   graph. PRACTICAL RULE until root-caused in the pad-window code:
   keep bands ≥ ~15 rows (n_lat/n_dev ≥ 15); the preflight should
   refuse thinner. Root-cause status (2026-08-18, job
   27037112): the compiled program is STRUCTURALLY IDENTICAL at 12-
   and 15-row bands — 25 collective-permutes + 1 all-reduce, same
   source-target pairs, and the halo messages are the SAME SIZE in
   both configs (only the local tile height differs). So the model
   code is exonerated; the deadlock lives in the GPU runtime stack's
   communicator initialization and its trigger is UNKNOWN (honest
   label: cause unknown; every falsifiable candidate we could name is
   refuted). Mitigation shipped: preflight guard (bands ≥ 15 rows,
   escape hatch env) + the LL2880 working point. Revisit only on a
   jaxlib/NCCL upgrade.
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

## What FESOM2's scaling imports to (user question 2026-08-17)

The FESOM2 curve on the figure is the native FORTRAN reference, not a
JAX result. Its scaling mechanisms map onto levers this assessment
already carries — ranked by our own receipts:

1. **Exact per-pair message sizes** (FESOM's precomputed exchange
   lists) → our per-round-MAX padding ships 2.0× the true halo
   (receipt 26998727); the import is the ragged path after its
   zero-slice pruning fix. Both MPAS and ocean lanes.
2. **Node-aware partitioning + rank placement** (hierarchical METIS,
   intra-node neighbours preferred) → unmeasured here; offline
   receipt cheap (extend the partition scorer with an intra-node pair
   fraction).
3. **Rank-local mesh setup** (no global mesh per rank, ever) → we
   OOM-patched the symptom (int32 maps); the full import removes the
   setup wall entirely for >200-rank MPAS.
4. **Non-blocking overlap of halo exchange with interior compute** →
   refuted on the NCCL stack for many-round exchanges; becomes viable
   again exactly when lever 1 collapses the exchange to 1-2
   collectives.
5. Their per-op comm cost is intrinsically small (persistent MPI) →
   our equivalent was the 1.230 ms per-op residual, already cut by
   the mcp2p env pair (−13..−18 %).

Note the control: our CPU-MPI lanes already scale FESOM-like (lat-lon
2-D eff 0.83 @512), so the gap is GPU-lane NCCL behaviour, not model
structure — consistent with every receipt above.

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

---

# Supersedes sections a and c — measured 2026-08-19

Everything below is a receipted arm from one day of measurement. Where it
contradicts sections a or c above, this section is the record; the older
text is kept for provenance, not for citation.

## The two atmosphere lanes are limited by opposite things

Paired arms, every halo collective deleted and the otherwise identical
program re-timed, which is the only way to split the step on this stack —
the profiler does not record these collectives, and a capture attempted
on 2026-08-19 returned identical row counts for two runs three times
apart in length, i.e. truncated and unusable (job 27072274, discarded).

lat-lon 2048x4096 L26, one allocation, arms pinned to the same nodes
(job 27071069):

| devices | step | local | communication | share | perfect |
|---|---|---|---|---|---|
| 32 | 6.972 | 6.308 | 0.664 | 9.5% | 6.296 |
| 64 | 4.728 | 3.368 | 1.360 | 28.8% | 3.148 |
| 128 | 3.950 | 2.285 | 1.665 | 42.2% | 1.574 |

icosahedral subdivision 9, deep + ragged halo (jobs 27068830/32/33):

| devices | step | local | communication | share |
|---|---|---|---|---|
| 8 | 23.135 | 23.055 | 0.080 | 0.3% |
| 16 | 19.230 | 19.120 | 0.110 | 0.6% |
| 32 | 7.230 | 7.190 | 0.040 | 0.6% |

## a. lat-lon — bandwidth, and the boundary that never shrinks

CONFIRMED. The halo moves 18.515 MB per device per step in 13 messages,
counted from the compiled program at production resolution. Against the
measured communication that is about 11 GB/s per device at 128 devices on
a node link of roughly 25 GB/s shared by four devices — within about a
factor of two of the fabric.

CONFIRMED, with the packing control the first attempt lacked (job
27073582, 64 devices): doubling the payload costs +1.554 ms, of which
+0.105 is the device-side packing the multiplier itself adds, so the extra
bytes cost +1.450 on the wire. Of 1.464 ms of communication, 1.450 is
payload and 0.014 is per-operation — about one microsecond per message.

RETRACTED then RESTORED. The first payload arm had no packing control and
its raw delta exceeded the whole communication term, which cannot license
"per-operation cost is zero"; it licenses only "packing exceeds the
per-operation term". The controlled arm above restores the conclusion.

The structural cause: a device owns a latitude BAND spanning the full
longitude circle, so its halo is two rows of 4,096 columns at any device
count. Adding devices shrinks the work and leaves the boundary alone.

CONFIRMED by compiled-program census, 16 devices, 4096 longitudes, 26
levels: latitude bands move 18.515 MB in 13 collectives; a 2x8 tiling
moves 3.483 MB in 108. The byte cut is 5.3x, the geometric prediction, and
there is NO all-gather in the tiled program — the mechanism review's
objection that the tiled pole fold would gather the whole circle on every
tile is refuted by the artifact. Six of the 108 touch a full longitude
extent.

At one microsecond per message, 108 messages cost about 0.1 ms against a
payload cut of 1.450 to 0.27, so the tiled lane is predicted at roughly
3.75 ms against 4.841 at 64 devices. That A/B is the open item.

Also CONFIRMED and unresolved: at 128 devices the LOCAL work is 2.285 ms
against 1.574 perfect, 45% above, while at 32 devices it is 0.2% above.
Tiling does not address that term — per-device cell count is unchanged —
and the mechanism review attributes it to a fixed per-step kernel-count
floor plus reductions whose latency grows with rank count.

## c. icosahedral — not communication, and not the mesh either

CONFIRMED: communication is 0.04–0.11 ms of the step at 8, 16 and 32
devices, and doubling the wire payload costs nothing measurable. Below 64
devices this lane does not communicate.

REFUTED, each with its own receipt:
- ghost-ring size and exchange round count as the cause of the 8-to-16
  stall — both on trend at every device count, offline census job 27068906
  (space-filling curve) and 27070623 (graph partitioner);
- partition quality — the graph partitioner wins 19.3% at 8 devices and
  only 5.5% at 16, and with it the step is 18.10 ms at 8 devices and
  18.47 at 16, so doubling the devices buys nothing at all (jobs
  27070258/59);
- host dispatch — removing every per-step synchronisation changes the
  stalled size by 1% (job 27072488);
- the deep halo's own trade — it still wins 15.5% at 8 devices and 6.1%
  at 16 where communication measures zero, so its gain is compilation and
  fusion across the step body, not messages avoided (jobs 27069554/55).

CONFIRMED cause, single GPU, no devices and no collectives involved:

| cells | 13 levels | 26 levels | 52 levels |
|---|---|---|---|
| 40,962 | 1.630 | 2.000 | 2.830 |
| 163,842 | 7.500 | 20.245 | 13.820 |
| 655,362 | 33.250 | 50.170 | 51.640 |
| 2,621,442 | 121.425 | 173.750 | 214.530 |

At 163,842 cells, 52 levels runs 32% FASTER than 26 despite twice the
work; everywhere else more levels cost more. As per-cell excess over
neighbouring mesh sizes: 1.15x at 13 levels, 2.53x at 26, 1.22x at 52.
Production runs 26 levels, and at 16 devices each device holds 163,841
cells — the exact cell of that table. That is the stall.

The graph partitioner's win is separable and real: 4–7% fewer cells to
compute, and 18% cheaper per cell at 8 devices, which is memory locality
on a step that runs about six times off a bandwidth roofline. It is a
configuration flip. It does NOT survive into the trapped shape, where the
graph arm is 2% worse per cell.

Open: whether the trap is a cache straddle (the live set at 26 levels is
about 33 MB against a 40 MB last-level cache — broad hump predicted) or
stride alignment and address aliasing (104-byte rows at 26 levels are not
16-byte aligned; 208 at 52 are — sharp spike predicted). The level scan
that separates them by response shape is running. The fix follows from the
shape and must be verified as a band of good paddings, never a single
lucky value.

## What NOT to do, updated

- Do not cite "collective COUNT is the wall" for lat-lon. Per-operation
  cost is about one microsecond per message; the wall is bytes.
- Do not tune the icosahedral lane's communication below 64 devices. It
  does not have any.
- Do not tune the icosahedral partition to fix the 16-device stall. Two
  partitioners and an offline census say it is not the partition.
- Do not quote the padded-byte verdict string in the partition A/B
  harness. It was written for a bytes hypothesis, and at 8 devices there
  are no bytes to save.

## Addendum, later on 2026-08-19 — the icosahedral cause, and three tiled blockers

### The icosahedral stall is a level-count tax on the unstructured kernels

One GPU, no devices, no collectives, float32, 163,842 cells, 60 timed
steps per arm, two replicates, spreads at or under 1% apart from a single
4.8% outlier. Cost per cell per level, picoseconds:

  13 3520 | 16 1547 | 18 2318 | 20 1524 | 21 2578 | 22 4910 | 24 4147
  25 4224 | 26 4755 | 27 3861 | 28 4058 | 30 4591 | 31 3741 | 32 1629
  34 2949 | 36 1706 | 40 1664 | 52 1621

Cheap: 16, 20, 32, 36, 40, 52. Expensive: everything from 22 to 31, plus
18, 21 and 34. The step is 22.6 ms at 30 levels and 8.5 ms at 32 — more
work, a third of the time. Production runs 26, at 3.1x the cheapest.

That IS the 8-to-16 device stall: at 16 devices each device holds 163,841
cells at 26 levels, and the parallel step (18.9 ms) matches the
single-GPU step at that shape (20.2 ms).

Two explanations were tested and both REFUTED by the table:
- byte alignment of a cell's column — 24 and 28 levels give 96- and
  112-byte strides, both multiples of the 16-byte vector width, and both
  are expensive;
- cache capacity — 52 levels holds the largest live set measured and is
  among the cheapest.
An earlier commit asserted the alignment rule from five samples and was
retracted when the finer scan arrived. The cause is UNKNOWN.

It is NOT a compiler or hardware property. The lat-lon core over the same
level counts on one GPU is flat: 851 to 954 picoseconds per column per
level, 1.12x end to end, 26 levels at 1.04x the cheapest, every arm to
0.0% (job 27078027). So it belongs to the unstructured core's kernels —
which points at the neighbour gathers over irregular connectivity, the one
structure the lat-lon core does not have.

Shipped: an advisory raised from the unstructured model's constructor (NOT
from the shared vertical-coordinate factory, which serves both lanes)
carrying the measured table, disclaiming a cause, and saying counts absent
from the table are unmeasured rather than cheap.

### Three blockers cleared on the tiled lat-lon lane, none of them visible in tests

The tiled step, its state layout and its pole masks already existed and
were gated by single-process tests. Getting it onto the cluster took
three fixes, and each failure mode was invisible to those tests:

1. The initial state was built globally and handed to the tiled sharder,
   which XLA services with an all-gather: 105 GiB per device at 2048x4096
   on 64 devices. Fixed by making the existing shard-local builder take
   either mesh — the only band-specific thing in it was the partition
   spec.
2. The benchmark rejected legal tiled runs, because it checked that the
   latitude rows divide by the DEVICE count, which is the band rule.
3. The tiled factory defaults to SHARDED geometry stacks, and a
   multi-process program may not close over a sharded global array. Every
   tiled arm died on "Closing over jax.Array that spans non-addressable
   devices ... float32[4,8,512,512]". The band lane has always defaulted
   to replicated stacks; the bench now asks for the same.

Compiled-program census, unchanged by any of the above: latitude bands
move 18.515 MB per device per step in 13 collectives, a 2x8 tiling moves
3.483 MB in 108, and there is no all-gather in the tiled program.

### Standing prediction for the tiled A/B

Communication at 64 devices is 1.464 ms, of which 1.450 is payload and
0.014 per-operation across 13 messages — about one microsecond each,
measured with the packing control. At that price 108 messages cost about
0.1 ms and the payload falls 5.3x to 0.27, so the tiled step should land
near 3.75 ms against 4.841 for bands. If it lands at or above 4.8, the
per-message cost does not extrapolate from 13 messages to 108 and the
next move is packing the tiled exchanges the way the band lane packs its
own, not abandoning tiling.
