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
- Rung 0 / OMT-4 / OMT-1: the one-step reverse gradient has 0 NaN sources
  in every mode; a fused 10-step `jit(grad)` on rung 0 is finite and equals
  forward mode to 3.5e-15.
- Rung 10: one source survived group C (shortwave divisor on dry levels
  above the blue level); fixed in C2, nothing hides behind it.
- FD misses on the T loss track the convective-adjustment (EVD) on/off
  switch: CONFIRMED on OMT-2.

## What changed (one commit per group)
| group | commit | statement | guard |
|---|---|---|---|
| A | `fae1fe1a42b9` | `ocean_model_latlon_cgrid.py:1936`, `ocean_model_latlon_cgrid.py:1949` (stage-1/2 tracer flux divergence) | divide by 1 where `e3t(Kmm) == 0` |
| A | same | `ocean_model_latlon_cgrid.py:2573-2575` and the inactive alternate combine, masked at `ocean_model_latlon_cgrid.py:2581` | `q_after` = 1 on the lanes that where discards (same mask) |
| B | `9900ba02269a` | `implicit_solver.py:363-382` (and the other two divisions in the same literal solve) | divide by 1 where the pivot is exactly 0 |
| B | same | `ocean_model_latlon_cgrid.py:13455` (top-face surface stress, U and V) | divide by 1 where `e3 * rho0 == 0` |
| C | `99f37662a54c` | `internal_wave_mixing.py:444`, `tke.py:1082`, `tke.py:3670` | the shared `safe_sqrt` (double where) |
| C, C2 | same, `C2` below | `shortwave_penetration.py:647` | divide by 1 below the blue level (discarded) and, C2, on zero-thickness dry levels |
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

## C2 (`2f1b26df90a2`)
The walker found group C's shortwave guard incomplete: 12,518 rung-10 dry
levels above the blue level still divide by a zero thickness. C2 also
divides by 1 there. Forward bits re-proven: 13/13 cards BIT-IDENTICAL vs base
(jaxpr changed on rung 10 only); rung-10 ladder at C2 identical on both
labels (200/200 each); GYRE ten-step archive `3b3d865cf5c82d38` again.

## Gradients (one-step walker = round 315's, `jobs/walk_*.json`)
| card | loss / mode | sources | dT0 . T_dir0 |
|---|---|---|---|
| rung 0 (tip `7a1825f4d6f2`) | T chained, fused, raw and sanitized | 0 | -4.2080326158e-2 (round 315 forward mode -4.2080326157569e-2) |
| rung 0 | SSH chained raw/sanitized, fused raw | 0 | 4.4076248945e-4 |
| OMT-1 | all 8 modes | 0 | T -4.2176898682e-2; SSH 4.4051036285e-4 |
| rung 10 (before C2) | T chained raw | 2 (shortwave div + its mul) | NaN |
| rung 10 (before C2) | T chained sanitized | the same 2 only | -4.1788247936e-2, 0 non-finite leaves |

The walker's control (raw walk vs compiled pullback): same finite pattern,
max relative difference 6.3e-11 (rung 0 T), 4.8e-13 (OMT-1 T). The rung-0
walker was stopped after 7 of 8 modes and the rung-10 walker was OOM-killed
after 3 T modes (each held ~300 GB).

Fused multi-step (`nemo_testcase_l4_orca2_round316_fused_grad.py`, one
`jit(grad)`: step 1 + checkpointed scan over steps 2..10): rung 0, T loss,
N = 10: finite on every cell, |dL/dT0| = 1.9340e-2, reverse . T_dir0 =
-4.200802232317228e-2 vs forward mode -4.200802232317213e-2 (3.5e-15).
Compile+run: reverse 2,300 s, forward 939 s.

PENDING at commit time (detached, results land in `jobs/`): round 315's
full probe on rung 0 and OMT-1..4 (tip `7a1825f4d6f2`; the C2 line does not
run on these cards) and rung 10 (C2); the fused rung-0 SSH loss; fused rung
10 at C2 (its forward-mode value answers whether rung 10's forward-mode NaN
was these sources).

## Branch flips (OMT-2, N = 1, `jobs/flips_omt2_*.json`)
Lockstep interpreter over one step, both sides of x +- eps T_dir0. Control:
interpreter vs compiled step, max abs difference <= 5.5e-14.

| eps | T FD miss | EVD trigger flips | face-min thickness flips | other |
|---|---|---|---|---|
| 1e-3 | 1.17e-2 | 6,474 + 432 coefficient | 3,188 | 0 |
| 1e-4 | 1.53e-3 | 1,065 + 71 coefficient | 114 | 0 |
| 1e-5 | 1.96e-4 | 120 + 8 | 0 | 0 |
| 1e-6 | 3.15e-5 | 30 + 2 | 0 | 0 |
| 1e-7 | 7.79e-7 | 0 | 0 | 0 |

- CONFIRMED: the miss falls with the flip count and passes 1e-6 at the
  first eps with no flips. The flips are the convective-adjustment trigger
  (N^2 below threshold switches the diffusivity to its convective value,
  three comparisons per interface) and its coefficient composition.
- Excluded: zero crossings inside the source-rounding helper's IEEE
  identity (529,568 .. 91 lanes); they are not a branch of the function.
- SSH misses (2.8e-8 at 1e-3, 1.6e-6 / 2.5e-6 at 1e-6 / 1e-7) are the
  rounding floor, not flips.
- OMT-1 and OMT-3 not measured (the first counter ran out of memory).

## Infrastructure
The first branch-flip counter stored every predicate and every intermediate:
15 copies reached ~110 GB each, the machine hit 1006/1006 GB, seven were
OOM-killed and I stopped the rest (my own PIDs). Rewritten to
evaluate both sides in lockstep and free values after last use; re-run on
OMT-2 only.

## Gates (x64, CPU)
- Citation gate on this receipt: PASS (`citation_gate_receipt.json`).
- Default citation-gate test after the re-anchor: 17 passed.
- Private-import ratchet: 6 passed.
- land.sh T4 files + round-315/316 instrument tests + the four extended
  test files, run file by file (machine load): citation gate 17, ratchet 6,
  TKE terms 70, recipe 25, freshwater 16, barotropic state 8, ORCA2
  parallel receipt 2, round-315 probe 10, round-316 instruments 3, IWM 6,
  implicit literal 14, dynzdf composition 21 — all passed, 0 failed
  (`gates/`). Two files exceeded the 10-minute foreground limit under load
  with 0 failures so far (stage face-mask rank; WS tracer RK3, 18 passed
  before the cut); full runs detached to `gates/*_full.log`. land.sh runs
  both again at landing.

## Review resume: tests strengthened (`29bbd4fe2`, no model change)
Reviews: Claude LAND; codex HOLD on tests only (implementation clean). The
four gradient tests checked finiteness only, so a guard that stopped the
gradient would pass. Each now also compares the guarded VJP and JVP (jit,
traced inputs, x64) on wet lanes with an unguarded reference:

| test | unguarded reference | bound |
|---|---|---|
| TKE raw mixing length (en > 0) | plain `sqrt` formula, inline | exact |
| literal IWM (N^2 > 0 everywhere) | `safe_sqrt` swapped for `jnp.sqrt` | 1e-14 rel (measured 6.8e-16) |
| stage tracer RHS (wet cells) | analytic d(rhs)/d(e3t) = -rhs/e3t | 1e-13 rel |
| literal tridiagonal solve (wet column) | dense solve of the same matrix | 1e-12 rel |

The fused-gradient parser test now runs the entry point: missing or unknown
`--rung` exits 2 before any run; a valid rung reaches the run with the
default steps and losses.

Plants (`test_plants/plants.log`, `plant_reasons.log`): `stop_gradient` in
the `safe_sqrt` body (TKE and IWM fail), on the tracer divisor guard, on the
solver pivot guard, and `--rung` made optional — each fails on its equality
or exit-code line; restored, 5/5 pass; tree clean after each revert.

Gates after the change: TKE terms 70, IWM 6, implicit literal 14, round-316
instruments 3, ratchet 6, default citation gate 17 — all passed. Tracer RK3
file: 20/37 passed, 0 failed in two 10-minute slices under machine load;
the edited test passes alone; the full file passed 37/37 (44 min) before
this change, which touches only that test's body.

codex review: operator
