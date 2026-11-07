# Preregistration — NEMO testcase L2 GYRE round 130

Date: 2026-09-20

Incoming lane tip: `a18ba326ec94ab4afbac28de681e69e714621ec8`

This document is frozen before any Round-130 local proof, ladder, month, or
year candidate measurement. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round130/`.

Round 129 refuted the run-to-run-spread explanation: legoESM's day-240 member
spread is only `1.27e-8` of the `1.6446741930292448e-2 K` seed-0 gap. This
round therefore performs the operator-directed year-scored re-evaluation of
the held, locally NEMO-exact patches. It changes no configuration choice, no
stabilizer, and no NEMO source. Every arm starts from the incoming tip and
contains exactly one registered held patch.

## P0 — immutable control and instrument calibration

The immutable year control is
`phase3/year_equivalence/gyre/lego_seed0_year`, produced at clean commit
`4d250301588d3ed0ad83fb20d6bf520e175d576e`. Before any candidate is admitted,
the incoming tip is run independently with the same existing harness:

`nemo_testcase_l2_gyre_year_fromrest.py --member 0 --days 360 --snap-steps 6 --tag year`

The manifest must name GYRE-zco, seed 0, fp64/libm, 2,160 steps, six-step
snapshots and all eight registered days. Its T3D RMS against
`phase3/year_owners/nemo_seed0` must reproduce the immutable control at all
registered days `30,60,90,120,180,240,300,360`; in particular:

| day | frozen T3D RMS (K) |
|---:|---:|
| 30 | `6.890484901489568e-5` |
| 60 | `1.9329973681936875e-4` |
| 90 | `1.8645021144913585e-3` |
| 120 | `1.0501256819510476e-3` |
| 180 | `3.580551011866709e-3` |
| 240 | `1.6446741930292448e-2` |
| 300 | `1.3597404177319843e-2` |
| 360 | `1.1223573910167267e-2` |

Frozen prediction: every row reproduces to the precision stored in the
immutable scorer artifact. Any mismatch stops candidate ranking as an
instrument failure. A baseline-value plant must print `STATUS PLANT-FIRED`
and exit nonzero.

## P1 — complete held-patch registry

The eight individually scored arms are fixed below. `git apply --3way` may be
used only to replay the preserved patch onto the current tip; the resulting
diff must contain no unregistered hunk.

| id | preserved manifest | statement whose local proof is repeated |
|---|---|---|
| `r62_coeff` | `nemo_testcase_l2_gyre_round62_held_tracer_coefficient.patch` | stage-3 `tra_zdf` matrix/coefficient associations |
| `r88_kaa` | `nemo_testcase_l2_gyre_round88_held_kaa_wzv_bundle.patch` | pre-solve Kaa SSH plus W/WZV state and restart carriage |
| `r89_assign` | `nemo_testcase_l2_gyre_round89_held_source_rounded_kaa_bundle.patch` | source-rounded RK3 assignment boundary with the R88 state |
| `r97_rhs` | `nemo_testcase_l2_gyre_round97_held_full_stage1_rhs.patch` | stage-1 consumed full momentum RHS |
| `r99_wclock` | `nemo_testcase_l2_gyre_round99_held_stage1_w_ratio_clock_full_rhs.patch` | source-rounded r3t ratio, W recurrence, clock and full RHS |
| `r105_shear` | `nemo_testcase_l2_gyre_round105_held_shear_routing_split.patch` | step-entry free-surface route into `zdf_sh2` only |
| `r109_handoff` | `nemo_testcase_l2_gyre_round109_held_vector_stage1_handoff.patch` | vector-stage projection/recombination handoff |
| `r112_fct` | `nemo_testcase_l2_gyre_round112_fct_metric_upstream_held.patch` | two-step FCT metric transports and divided concentration RHS |

The following shelf files are excluded before measurement: the LDF patch is
already landed by Round 110; the parallel TKE K_H and Round-60 association
proofs fail or lack the required production-step JIT exactness; Round-47 ZAD
and Round-51 raw histories were superseded by the landed Round-85 momentum
bundle; Round-62 content materialization is explicitly refuted; the
preliminary Round-99 W patch is superseded by `r99_wclock`. These are not
silently missing candidate arms.

## P2 — local admission, source and plants

Each arm is admitted to a 360-day run only if its existing committed walk is
re-run at the current tip with NEMO's recorded stage entry and shows the
registered statement bit-exact in both production-step JIT and production
eager execution. Where the historical proof has an isolated-JIT cross-check,
that row is retained but cannot substitute for production execution. The
existing one-ULP plant must alter a consumed wet value, print
`STATUS PLANT-FIRED`, and exit nonzero. A stale or non-applying patch, a
non-exact production row, or a plant that exits zero is recorded and skipped.

The local source expectations are taken from the record's compiled branches:

* `R46KT2/trazdf.f90:461-477,523-528,545-578` forms the vertical matrix,
  content RHS and solve used by `r62_coeff`;
* `R64KRHS/stprk3.f90:188-226`, `stp2d.f90:141-213`, and
  `sshwzv.f90:293-300` define the Kaa, complete RHS and W/WZV program used by
  `r88_kaa`, `r89_assign`, `r97_rhs`, and `r99_wclock`;
* `R101TKEW/stprk3.f90:167-168`, `zdfphy.f90:319-320`, and
  `zdfsh2.f90:102,107` bind both shear-divisor slots to step entry for
  `r105_shear` while the tracer solver retains its stage-3 field;
* `R111FCTW/stprk3_stg.f90:295-296,326-346,860` constructs and passes metric
  transports, and `traadv_fct.f90:503-623` performs the two-step upstream
  writes used by `r112_fct`.

Every citation is re-opened from compiled `BLD/ppsrc/nemo` before the receipt
maps it. Citation ranges in this preregistration are source expectations, not
substitutes for the receipt citation map.

Frozen prediction: all eight patches still apply. The named row of each arm
is bit-exact under production JIT and eager, and each plant fires. A failure is
`REFUTED`, retained, and removes only that arm from year scoring.

## P3 — one-patch year and short-trajectory arms

Every admitted patch is applied ALONE to an incoming-tip scratch tree. Its
clean candidate commit is stamped in every artifact. The unchanged harness
runs member 0 for 360 days with exactly the P0 flags. No candidate may reuse a
different patch's run, mutable state, or control restart.

Each arm also runs the canonical kt=1..10 ladder and the 30-day member. The
registered table contains local proof, day-30/day-240/day-360 T3D RMS,
day-240 change from P0, first-over-bar before/after, and the count and identity
of kt1 AT-BAR rows lost. Every moved ladder, month, and year row is retained in
the machine-readable artifacts. A registry plant removes one candidate or
one registered year day and must print `STATUS PLANT-FIRED` and exit nonzero.

Frozen directional predictions, used only as falsifiable forecasts:

* `r105_shear` is predicted to be the best single arm but to improve day 240
  by less than one percent, extrapolating its tiny admitted day-30 direction;
* `r88_kaa`, `r89_assign`, `r109_handoff`, and `r112_fct` are predicted to
  worsen day 240 because their short trajectories previously worsened;
* `r62_coeff`, `r97_rhs`, and `r99_wclock` are predicted to move day 240 by
  less than one percent in either direction.

The measured ranking, not these predictions, determines the verdict. Exact
equality is neither improvement nor worsening.

## P4 — landing and pair rule

Singles are ranked by `P0 day240 - candidate day240`, descending. Ties are
broken by day-360 improvement, then day-30 improvement, then the P1 registry
order. The best single can land only when all Decision-43 and Decision-45
conditions hold mechanically:

1. day-30 T3D RMS decreases;
2. the first-over-bar row is not earlier;
3. no kt1 row that was AT-BAR leaves the bar;
4. every moved short and month row is registered;
5. every executing shared card is measured; and
6. day-240 and day-360 T3D RMS do not worsen, with all eight year rows
   registered before/after.

If no single passes but at least two singles strictly improve day 240, the
only pair attempted is the top two day-240 improvers under the frozen ordering.
Each must retain its local proof, the pair is run through the same full year,
and the year response is explicitly bisected against both single arms. No
other post-hoc combination is allowed. If fewer than two improve day 240,
there is no pair arm.

The executing-card set is derived from resolved recipes. GYRE-zco and generic
NEMO-GYRE are measured where their route executes; a DINO route that executes
the changed statement is measured, not inferred. LOCK_EXCHANGE and OVERFLOW
are reported from the derived execution result. ORCA2 remains
`UNMEASURED-WITH-SPEC`: run its native certified ladder and year member before
transferring a landing claim.

## P5 — fail-closed outcome

If a candidate passes P4 it lands in the one shared implementation and its
360-day output becomes the new immutable year before arm. If every candidate
fails, production remains unchanged and the receipt is `HELD`. In that case
the family with the largest absolute measured day-240 response is the only
admissible sensitivity lead; it may name a first non-bit operand only if the
one-family substitution was actually carried through the full year. An
operator inferred from a 10-step walk alone may not be promoted to a day-240
owner.

A separate read-only Codex review must try to refute registry completeness,
local-production provenance, one-variable isolation, ranking, the pair rule,
and any landing verdict. The receipt citation gate and its shifted-citation
plant must run. No physics is committed before the review and gates permit it.
