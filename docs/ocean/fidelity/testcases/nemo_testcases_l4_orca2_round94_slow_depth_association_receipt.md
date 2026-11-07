# ORCA2 round 94 — rung-0 slow-depth association is named and held

Date: 2026-10-02. Base `cffea640d`; final tree recorded below. Scope is ocean
only. Every rung-0 number in this receipt is **independent**. No number here is
mixed with the shipped-card, given-NEMO-entry claim.

## Verdict

**HELD.** The existing two-rank slow-boundary record is sound. Production's
first active non-bit statement is NEMO's vector-invariant vertical average,
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:206-219`. A source-associated
one-variable arm closes both depth rows, and then every remaining forcing row
through `ssh_rhs` is bit-exact. The next non-bit boundary is the split-explicit
output. The arm is not landed: it moves GYRE and violates the immutable 2-ULP
bar in 49 of 70 certified rows. The final package tree therefore restores the
base implementation; the exact candidate is retained as the committed held
patch `nemo_testcase_l4_orca2_round94_slow_depth_held.patch`.

No selector, carried state, stabilizer, sea-ice field, or
`unmeasured_features` tuple changed.

## Existing-record admission

The round-93 checker had treated the record-wide 94x152 domain as every
field's dimensions. The record is actually self-describing: `depth_u` is the
first field and declares 90x148. The repaired parser consumes each field's
name, rank and dimensions, derives its payload length, checks a unique exact
14-field registry, finite owned values, centred owned coverage and exact EOF.
It does not predict a field order, byte count, or header tuple.

Admission of the existing target succeeds without a NEMO rerun:

- both rank records are 1,538,832 bytes and cover the 180x148 global owned
  domain exactly once;
- all 14 fields are present on both ranks with their own declared shapes;
- all 20 terminal restart shards remain byte-identical to the admitted
  round-92 parent;
- header, duplicate-name, field-dimension, truncation, swapped-rank and
  restart-byte plants all fire.

The decisive transcript is
`round93/acquisition/orca2_rung0_slow_ranked_10step_np2/round93_slow_admission.json`.
The existing target is admitted; no acquisition is needed.

## Frozen predictions

| Prediction | Verdict | Measurement |
|---|---|---|
| R94-P1 record sound | **CONFIRMED** | Both ranks parse, cover once, end at EOF, and preserve all terminal restarts. |
| R94-P2 depth forcing bit-exact | **REFUTED** | Production U: 7,082/15,789 active cells, max `3.3881317890172014e-21 m s-2`, rms `1.6418954692527803e-22`; V: 7,124/15,875, max `5.082197683525802e-21`. |
| R94-P3 drag first | **REFUTED** | Depth owns the walk; after the source arm, drag is exact. |
| R94-P4 observer passive | **CONFIRMED** | Observed versus ordinary T/S/u/v/ssh: zero unequal bits. |

The layout, record-bit and trace-bit plants each exit 2 with
`STATUS PLANT-FIRED`. Failed predictions are retained, not rewritten.

## First statement and one-variable walk

The compiled program left-accumulates
`SUM(e3*rhs*mask) * r1_h` before drag. Drag and wind follow in
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:231-250`; the final slow
forcing is dumped at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:285`; the sea-surface RHS is
formed at `ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:290-312`; and the
split-explicit solver/output follows at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:317-326`.

The rank-0 operand walk proves U thickness, RHS, three-dimensional mask and
reciprocal depth are individually bit-exact. A literal NumPy replay is exact.
The ordinary production JIT reduction is not: rank-0 U differs on 3,896/7,890
active cells. A static-loop-only JIT arm also remains non-bit. Materializing
each source multiply, add and final multiply makes U and V both 0 unequal.
That is the discriminator: XLA reassociation of an algebraically equivalent
reduction, not an input field.

The candidate measurement at clean commit `1f1b3654e` gives:

| boundary | unequal active cells | max absolute | verdict |
|---|---:|---:|---|
| depth U/V | 0 / 0 | 0 / 0 | AT-BAR |
| drag U/V | 0 / 0 | 0 / 0 | AT-BAR |
| wind U/V | 0 / 0 | 0 / 0 | AT-BAR |
| final U/V | 0 / 0 | 0 / 0 | AT-BAR |
| ssh RHS | 0 | 0 | AT-BAR |
| ssh after split-explicit | 16,433/16,433 | `0.3685315364572615 m` | first non-bit after arm |
| barotropic U/V after | 15,789 / 15,875 | `0.0642170631553079 / 0.0326373983355282 m s-1` | downstream |

The `ssh_after` rms is `0.02496560526651195 m`; its argmax is `[j=143,i=124]`,
legoESM `0.3590303046374879 m`, NEMO `-0.009501231819773564 m`. This advances
the next source walk to the internals of `dyn_spg_ts`; note B27's
`e3f_0vor` Coriolis denominator is the first registered one-variable candidate
when that source-order walk reaches `dyn_cor_2D`.

The known V fold-row operand debt (13 thickness and 668 mask/RHS cells; 35
literal-result cells) is later than the U statement and remains reported, not
folded into this owner.

## Shared-card landing gate

The GYRE gate was run at the untouched base `cffea640d` and at candidate
`1f1b3654e`, both CPU/fp64, ten steps, 70 rows. The first-over-bar result stays
kt=3 for T/S/u/v/ssh and there are no row-status changes, but 58 rows move.
The offline oracle-relative comparator returns **FAIL**:

- 49 rows contain at least one cell worse than the 2-ULP bar;
- 109,681 cells exceed that bar;
- the largest worsening is 61,826.25 row-scale oracle ULP in
  `GYRE-zco.kt9.before.v`;
- the base/tip residual SHA-256 values are respectively
  `58c6297cee40b39c2c5889b5338bb84dcbb8c30fefe6a39dd80a12aa9241aa76`
  and `ea6a714024ea5945fe092f903cf7772f0706f0412f41f056bfe7fd2045d21a92`.

This is a mechanical landing refusal, not a judgment call. The candidate was
reverted in `bfde20c99`; the final package files equal the base files. The
shipped-card ORCA2 ten-step process also terminated silently without a JSON or
refusal line. No result is claimed from that process, and it does not weaken
the already-dispositive GYRE refusal.

## Gates, tests, and review

The default citation gate passes with zero unmapped citations, failures, or
map-audit failures; shifting a real mapped citation by two lines makes it fail
with `SYMBOL-NOT-AT-LINE` and exit 1. Focused parser/walk tests pass 8/8.

The prescribed single `tests/ocean/fidelity -n 12` run reached 96%, displayed
six failures, and then reproduced the established silent-tail termination:
the pytest process disappeared without a summary or `lastfailed` cache. This
is the same visible count and failure mode recorded by round 93. No new
failure is assigned without an ID, and this receipt does not represent the
wide battery as PASS.

The required separate `codex exec --sandbox read-only` review failed before
reading the diff: `failed to initialize in-process app-server client:
Read-only file system`. Verdict: **independent review unavailable in-sandbox**.
The complete transcript is `round94/codex_review.log`.

## OPEN

1. Walk the rung-0 split-explicit solver from its admitted inputs to the first
   non-bit internal substep, in compiled order. Reuse the VORTEX per-substep
   record pattern if the existing terminal record cannot discriminate it.
2. At `dyn_cor_2D`, measure NEMO's four-cell masked `e3f_0vor` denominator as
   note B27 requires.
3. Treat the held source-associated slow-depth statement as a cancelling-pair
   item on GYRE: locate the later shared statement that currently compensates
   its reduction-order error before reconsidering a landing.
4. Re-run the shipped-card ten-step ladder only after diagnosing its silent
   no-artifact termination. It is unverified in this round.
5. The rung-0 package card, its given-entry/independent ten-step ladders and
   independent month remain subsequent hierarchy work.
