# ORCA2 round 316 — gradient-only NaN guards, forward bits unchanged

Base `27dc0be613c2` (round 315). Prereg `6fc5bc677e78` (frozen before any
measurement). Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round316/`.
CPU, fp64, x64, production JIT, libm policy.

## Verdict
- Every guard and FIX 2 landed. Forward values are bit-identical on all 13
  cards checked, after every group: every state leaf's bytes at every step
  (NaN payloads and signed zeros included).
- Seven ORCA2 pairs: all 1,360 ladder rows identical to the shipped rows.
  GYRE ten-step archive and day-30 snapshots byte-identical.
- Gradient results: see "Gradients" (filled as the runs finish).

## What changed (one commit per group)
| group | commit | statement | guard |
|---|---|---|---|
| A | `fae1fe1a42b9` | `ocean_model_latlon_cgrid.py:1936`, `ocean_model_latlon_cgrid.py:1949` (stage-1/2 tracer flux divergence) | divide by 1 where `e3t(Kmm) == 0` |
| A | same | `ocean_model_latlon_cgrid.py:2573-2575` and the inactive alternate combine, masked at `ocean_model_latlon_cgrid.py:2581` | `q_after` = 1 on the lanes that where discards (same mask) |
| B | `9900ba02269a` | `implicit_solver.py:363-382` (and the other two divisions in the same literal solve) | divide by 1 where the pivot is exactly 0 |
| B | same | `ocean_model_latlon_cgrid.py:13455` (top-face surface stress, U and V) | divide by 1 where `e3 * rho0 == 0` |
| C | `99f37662a54c` | `internal_wave_mixing.py:444`, `tke.py:1082`, `tke.py:3647` | the shared `safe_sqrt` (double where) |
| C | same | `shortwave_penetration.py:560` | divide by 1 below the blue level, where the where discards it |
| FIX 2 | `984cbbbb6e72` | `ocean_pe_latlon_cgrid.py:4294-4298`, `ocean_pe_latlon_cgrid.py:4375-4379` | the five `float()` casts dropped |

Guards keyed on "divisor == 0" change only lanes whose forward value was
inf/NaN; guards keyed on a mask change only lanes that mask discards. Note:
`internal_wave_mixing.py` and `tke.py` were byte-identical to the vmix lane's
V4 copies; they now differ by these lines (a future V4 re-import must keep them).

## Forward bits (instrument `nemo_testcase_l4_orca2_round316_forward_bits.py`)
SHA-256 of every state leaf's raw bytes at every kt (10 steps; OMT-1 8),
production `model.step`, base vs each group cumulatively
(`fp_<group>/`, `forward_bits_summary.txt`).
- A, B, C, D (= FIX 2) vs base: BIT-IDENTICAL on rung 0, OMT-1..4, rung 10,
  GYRE, VORTEX zco flux/vector, VORTEX seamount flux/vector, OVERFLOW,
  LOCK_EXCHANGE — 52 of 52 comparisons, 0 leaf-steps moved.
- The changed code runs where expected: step jaxpr changed on all 13 cards
  for A and B, on GYRE and rung 10 only for C, on none for FIX 2 (so FIX 2
  is identical by construction).
- Controls: base re-run twice is identical (rung 0, rung 10, GYRE); one
  planted hash change is reported MOVED (exit 1).

## Required gates on forward values
- Seven pairs (round-313 gate, shipped arm, tip `7a1825f4d6f2`): rung 0,
  OMT-1..4 (independent), rung 10 (independent, given-NEMO-entry): 200/160/
  200/200/200/200/200 rows, all IDENTICAL to round 313's shipped rows (no
  model change between round 313 and this base); every pair
  `PASS_R276_SOURCE_LADDER` (`ladder_vs_r313.txt`).
- GYRE ten-step: 210 residual arrays byte-equal base vs tip; archive SHA-256
  `3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b` at
  base, tip and round 309; status DEBT both (unchanged).
- GYRE day 30: 30/30 daily snapshots byte-identical base vs tip; day-30
  SHA-256 `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`
  (= round 277's).
- VORTEX x4, OVERFLOW, LOCK_EXCHANGE: bit-identical (above).
- DINO month gate: runs at landing (operator).

## Non-vacuity (tests at `c33a28b1f827`, jit, x64)
Each test is the jitted reverse gradient through the guarded statement with
traced inputs; each FAILS with its fix reverted and passes restored
(`nonvacuity_reasons.log`):
- dry-cell tracer divergence: non-finite gradient entries with the `e3t`
  guard reverted;
- literal vertical solve with a zero-pivot column;
- literal internal-wave mixing where N^2 is clipped to 0;
- literal TKE mixing length at zero TKE;
- drag rate: `jax.grad` over all five coefficients; reverted, the cast
  raises `ConcretizationTypeError`.
Not unit-tested (closures inside the step): the `q_after` combine, the
surface-stress division, the dissipation-length root, the shortwave divisor;
those are checked by the one-step walker below.

## Gradients
PENDING — filled from `jobs/` when the walker, probe, fused and flip runs
finish.

## Infrastructure
The first branch-flip counter stored every predicate and every intermediate:
15 copies reached ~110 GB each, the machine hit 1006/1006 GB, seven were
OOM-killed and I stopped the rest (my own PIDs). Rewritten to
evaluate both sides in lockstep and free values after last use; re-run on
OMT-2 only.

## Gates
PENDING.

codex review: operator
