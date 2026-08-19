# Lat-lon SPMD collective-permute packing — roadmap and receipts

Working note for the 2026-08-11 packing sequence on the sharded lat-lon
atmosphere band step (`make_sharded_atm_latlon_step`, 1-D latitude
decomposition, SSP-RK3). Every number below is a measured census on CPU
virtual devices (optimized-HLO `collective-permute` count, nd=8) unless
labelled a projection.

## Where the count went

| step | CPs/step | status |
|---|---|---|
| baseline (fused multi-pad + XLA overlap, the production deck) | 29 | — |
| **C** — band geometry pads precomputed | 25 | MERGED (#1581) |
| **A** — packed stage-entry exchange (opt-in) | 13 | MERGED (#1582) |
| **E** — locally recomputed mass-flux ghost row | 7 | BLOCKED, see below |

Structure of the 13: 3 RK stages x (1 packed entry epoch + 1 mass-flux
epoch) x 2 directions = 12, plus the single north-only
`reconstruct_vface_lower` exchange, which cannot join an epoch because
its output feeds KE/B *upstream* of the entry pads.

## Bucket E — measured, works, blocked on reproducibility

The idea (FV3's `whalo` pattern): give the entry epoch enough extra
depth that the mass-flux/`sigma_dot` ghost row becomes locally
computable, removing the second per-stage epoch.

The halo arithmetic checks out and the census hits the target:

* `sigma_dot`/`mass_flux` are vertical closures over `cumsum(div_dp)`
  (`grids/vertical.py:2554-2644`), elementwise in the horizontal — a
  ghost row needs only `div_dp` (+ `p_s` on the sigma lane) at that row.
* `div_dp` at the ghost row needs `dp_u*u` there (lon-only interp, so
  halo 1 suffices) and the meridional flux at the face beyond it, i.e.
  `dp` at halo 2 — or, cheaper, exchange the interior meridional
  mass-flux rows `(dp_v*v)[1:-1]` at halo 1, which lands exactly the
  neighbour's next face at both ends.
* Payload growth: +17% sigma / +14% hybrid on the entry buffer, roughly
  break-even against the `nlev+1` mass-flux exchange it removes.
* No cyclic dependency: `dp` never reads `sigma_dot`
  (`primitive_eq_latlon_cgrid.py:390,571`).
* Census with the wide path on: **7 CPs/step**, as designed.

It is blocked on a numerics detail, not a stencil error. At nd=4/8 the
recomputed ghost row differs from the exchanged one by ~1 ULP
(max abs 4.34e-19 on `u` after 2 steps, x64, only at rows adjacent to a
band cut; `dp_v` ghost 3.64e-12 on values ~2.5e4). Operand transport is
exact (`dp`, `u`, `interp_cell_to_uface(dp)` ghosts all differ by 0.0).
The decisive control: writing the *identical expression text*
`0.5*(pad[:-1] + pad[1:])` matches at exactly 0.0 (CSE unifies the
instruction), while four mathematically identical but differently
written forms all land on the same 3.64e-12. The neighbour's `div_dp`
row is the output of a fused instruction over its full row range
(an FMA contraction site); a locally written two-row equivalent is a
different instruction and the compiler may contract it differently.

This is the difference between *copying* a value (bucket A, exact) and
*re-deriving* one (bucket E).

Two ways to close it, both of which perturb something:

1. Build `dp_v`/`div_dp` on the extended row range in **every** env
   state so all lanes share one instruction — changes today's
   production numbers by ~1 ULP per stage.
2. Keep the recompute and move the acceptance bar from bit-exact to
   `allclose` with a stated tolerance plus a conservation receipt.

## Decision (2026-08-11): sequence E behind A's timing receipt

Neither option is taken yet, on purpose. Bucket A's **GPU wall-clock**
A/B (`scripts/cluster/scaling_levante/atm_ll_packed_exchange_ab.sbatch`,
LL2048@64, gate: CONFIRM <= 0.85, REFUTE >= 0.98) has not run — the
queue is saturated. If the 25 -> 13 count cut does not convert to wall
time, a further 13 -> 7 cut is worth nothing, and shipping a
numerics-perturbing path to buy it would be a pure loss. Measure first.

If A confirms, revisit E with option 2 (opt-in `LEGOESM_LATLON_PACKED_
EXCHANGE=2`, `allclose` bar with the tolerance stated at the gate, plus
mass/energy conservation receipts and the codex+physics-validator chain
that any numerics change requires).

Nothing from bucket E is in the tree; the working copy was reverted.
