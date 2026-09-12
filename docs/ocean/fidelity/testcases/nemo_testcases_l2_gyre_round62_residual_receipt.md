# NEMO testcase L2 GYRE — round 62 residual receipt

Date: 2026-09-12. Parent production tip: `b6027cebae17`. CPU,
`JAX_ENABLE_X64=1`. Predictions were frozen in the three round-62 PREREGs.
Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round62/`.

## 1. Month re-ranking

`month_step_gap.json` is clean-stamped `4189e2c68`; the day-30 decomposition
has the same stamp. Temperature first becomes non-bit after step 1
(`2.044e-15 K` RMS, `6,858` cells), but remains AT-BAR. Its first over-bar
departure is after step 2: `4.227e-6 K` RMS, `1.627511418e-4 K` max. Selected
RMS points are step 4 `5.912e-4`, step 10 `1.198710372e-4`, step 59
`1.927e-3`, and day 30 `1.239756827e-2 K`. Thus step 10 contains only
`0.9669%` of the eventual RMS magnitude; `99.033%` grows later. P1 is
**REFUTED** (step 1 was not bit-exact); P3 is **CONFIRMED**.

At day 30, S is the leading dimensionless field (`7.494%` of its NEMO
signal), not T (`1.056%`), so combined P2 is **REFUTED**. Temperature error is
concentrated in `0--100 m` (`77.188%` of squared error) and the west third
(`50.986%`). The peak is `+0.423232 K` at `(j=19,i=25,k=11)`, `208.50 m`,
`44.50 N`; it is not the bulk depth-band owner.

## 2. kt=3 tracer operator ladder

`zdf_operand_ladder_v3.json` is clean-stamped `84cff05d8`. NEMO zeros the
surface coefficient, builds the implicit matrix, performs LU, writes the
content RHS, and back-substitutes at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:450`,
`:461-477`, `:523-528`, `:545-564`, and `:573-578`.

Baseline kt=3 T is `1.627511418e-4 K` max. Single substitutions give:

| arm | T max K |
|---|---:|
| effective K / EVD interfaces | `1.627503129e-4` / `1.627511418e-4` |
| e3t(Kaa) / e3w(Kmm) | `1.627508435e-4` / `1.627511418e-4` |
| content surface / interior / bottom | `1.341055937e-4` / `1.649584869e-4` / `1.627511418e-4` |
| complete recorded content | `1.002651828e-8` |
| content + K + e3t + e3w | `3.310773877e-11` |
| recorded solved column | exact (`0`) |

The result refutes the proposed non-K solve-operand owner: no single row,
thickness, EVD interface, or K statement reaches the bar. The first measured
owner is the *complete incoming content RHS* (`1.679392692e-3` content units),
upstream of `tra_zdf`; the round-38 record contains no decomposition of that
quantity. Direct replay with recorded inputs is bit-exact. A separate
coefficient-expression candidate makes the all-recorded matrix/sweep exact,
but does not address the live `1.6e-4 K` owner.

Two preregistered content-association candidates were **REFUTED** and reverted
in production. No statement takes live kt=3 to bar, so no physics statement is
eligible to land and kt=3/day-30 before/after remain unchanged:
`1.627511418e-4 K` max / `1.239756827e-2 K` RMS.

## 3. kt=1 closure walk

`kt1_closure_walk_v4.json` is clean-stamped `9ea51581b6`. At kt=2 entry the
carried rows are already non-bit: en `654`, avm `5,159`, avt `4,905`, dissl
`15,387`, rn2/rn2b `13,299` cells. With all recorded operands substituted,
the production TKE matrix lower/diagonal/upper is exact, but the RHS first
differs in `238/17,400` cells, max `1.665334537e-16`. That is the earliest
resolved non-bit statement: NEMO's full production/stratification/dissipation
RHS at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:394-433`.
The source-written reconstruction retains exactly those 238 differences, so
the record cannot distinguish a subterm. NEMO then sweeps/floors en at
`:466-483`; downstream Prandtl, avt derivation, and pre-EVD copy are exact.
No single substitution makes K_H exact; baseline remains `5,310` cells,
`5.636255351e-13` max.

## 4. Rule 12, review, and disposition

| card | moved production rows | disposition |
|---|---:|---|
| GYRE kt=1..10 / days 1..30 | 0 | recorded baselines unchanged; no candidate eligible |
| LOCK_EXCHANGE / OVERFLOW | 0 | constant mixing per `round33_lock_zdf_matrix/namelist_cfg:131` and `round33_overflow_zdf_matrix/namelist_cfg:128` |
| DINO | 0 | shared statement not changed; separate branch not run |
| ORCA2 | 0 | UNMEASURED-with-spec |

Pre/post self-review found no production delta relative to `b6027cebae17`.
Required independent GLM and Codex reviews could not run (missing credential;
network denied), so the diagnostics are explicitly **UNREVIEWED**. This blocks
shipping physics, not reporting falsified hypotheses.

Validation at clean stamp `06b00873b`: citation gate PASS; its shifted-line
plant exits nonzero; the production operand-ULP plant detects exactly one
changed cell (`2.842e-14`) and exits nonzero; 50 focused tracer/TKE/citation/
stamp tests pass.

## 5. ASKED / UNASKED and open questions

| item | status |
|---|---|
| ASKED | none |
| UNASKED | none; no configuration choice was made |

Open: which upstream stage-3 tracer operator creates the content mismatch;
which RHS subterm creates the 238-cell TKE association residual; ORCA2 remains
UNMEASURED-with-spec.
