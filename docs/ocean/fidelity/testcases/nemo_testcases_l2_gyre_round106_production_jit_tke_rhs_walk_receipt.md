# NEMO-testcases L2 GYRE round 106 receipt: production-JIT TKE RHS walk

Date: 2026-09-17

Incoming tip: `27ffba40e75b0927a313273e3a7822298ce9764b`

Round status: **HELD — no physics or configuration landed**

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round106/`

## Round 106 — compiled-source basis

The first remaining non-bit statement in the authorized vertical-physics
subwalk is still the TKE right-hand-side assignment in the record-producing
compiled branch,
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:439-442`.
Its three matrix predecessors at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:434-436` are BIT.
NEMO separately assigns `zfact3 = 0.5_wp * rn_ediss` at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:261` before the RHS
uses `zfact3 * dissl * en`.

This round names no different NEMO statement. It establishes that the RHS
output is BIT in eager execution, non-bit under an isolated JIT, and more
widely non-bit inside the full production step. Changing the algebraic
dissipation tree to the compiled `zfact3` tree is inert. The surviving owner
is therefore compiled JAX/XLA evaluation context, not a different NEMO source
association.

The round followed the frozen preregistration
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round106.md`. Its P9
addendum was committed after the first materialization family was refuted and
before either corrected dissipation-association arm was implemented or run.

## Outcome first

Round 105's held shear-routing patch was reapplied exactly in a clean
measurement commit. It made the consumed `p_sh2` operand BIT in production
JIT, reproducing round 105. The RHS nevertheless remained 11,024 of 20,416
cells unequal at `5.551115123125783e-17`. Seven one-boundary materialization
arms, an all-boundaries arm, the direct NEMO dissipation tree, and the
materialized NEMO dissipation tree all remained non-bit.

The decisive execution discriminator is:

| execution boundary | expression | unequal / 20,416 | max abs | class |
|---|---|---:|---:|---|
| isolated-closure eager | current model tree | 0 | `0.0` | BIT |
| isolated-closure JIT | current model tree | 5,117 | `5.551115123125783e-17` | AT-BAR |
| isolated-closure eager | compiled NEMO tree | 0 | `0.0` | BIT |
| isolated-closure JIT | compiled NEMO tree | 5,117 | `5.551115123125783e-17` | AT-BAR |
| production step JIT | current model tree | 11,024 | `5.551115123125783e-17` | AT-BAR |

Thus isolated JIT already creates 5,117 unequal cells, and full-step fusion
adds 5,907 more. Both source trees give the same isolated counts. This is
positive evidence for XLA fusion/context and direct evidence against the
association hypothesis. It is not a claim that the present
`nemo_source_round` identity is sufficient: the materialization family below
proves that it is not.

No arithmetic arm closes the production row, the global first owned output
remains kt1 stage-1 U, and the round-105 routing already failed Rule 12.
Therefore there is no candidate eligible for the 954-row trajectory gate and
no downstream physics may land.

## Provenance and exact measurement arms

All scientific JSON artifacts carry a clean worktree stamp. The principal
commits are:

- `3838e6f2cfea4f49f9008d0597d3b04fde3751bd`: default-instrument baseline;
- `87c2f5d6cddd4a1440d6774963e7d7b9facea353`: exact reapplication of the
  held round-105 routing and the first source-order arms;
- `b284d62c2c310e21c0ff05a7af59d3c6f0b05948`: corrected NEMO dissipation
  association arms;
- `721bc58b32ae7de0ce0e59f96c1b661cf2c84142`: labeled eager / isolated-JIT /
  production discriminator and production-JIT RHS plant;
- `45b9fd471e0248cc7d1b626756a5a4593e6440c2`: restored production and the
  canonical clean stage twin.

The first attempted neutral run used the old unbounded return path and was
interrupted before it emitted a report. Its zero-byte
`instrument_default_jit.log` is retained. The committed bounded path changes
only when the gate returns; it still drives `self._step_jitted` for the actual
measurement.

At the default-instrument commit, before reapplying the routing, the bounded
production row reproduced round 105's restored baseline exactly:
`p_sh2` 17,400 unequal at `3.811744924985501e-14`; `en_rhs` 11,993 unequal at
`5.488912518947231e-10`; `zd_up`, `zd_lw`, and `zdiag` BIT. This establishes
that adding the selector did not move the default production path.

With the held routing applied, P1 also reproduced exactly:

| row | eager | production JIT |
|---|---:|---:|
| `zd_up` | 0 / `0.0` | 0 / `0.0` |
| `zd_lw` | 0 / `0.0` | 0 / `0.0` |
| `zdiag` | 0 / `0.0` | 0 / `0.0` |
| consumed `p_sh2` | 0 / `0.0` | 0 / `0.0` |
| `en_rhs` | 0 / `0.0` | 11,024 / `5.551115123125783e-17` |

Canonical artifacts are `routing_baseline_eager.json` and
`routing_baseline_jit.json`.

## Frozen source-order predictions: results

Counts below are production-step JIT `en_rhs` scores. Every run retained BIT
matrix predecessors and a BIT `p_sh2` operand.

| arm | actual boundary | unequal | max abs | verdict |
|---|---|---:|---:|---|
| baseline | none | 11,024 | `5.551115123125783e-17` | reproduced |
| `p_avt_rn2` | materialize `p_avt*rn2` | 11,024 | `5.551115123125783e-17` | inert |
| initial `zfact3_dissl` label | actually materialized `0.5*(rn_ediss*dissl)` | 11,024 | `5.551115123125783e-17` | inert; label defect retained |
| `dissipation_product` | materialize complete existing dissipation product | 10,402 | `1.1102230246251565e-16` | changed but worsened max |
| `after_stratification` | materialize `p_sh2-p_avt*rn2` | 11,024 | `5.551115123125783e-17` | inert |
| `parenthesized_sum` | materialize complete source sum | 11,024 | `5.551115123125783e-17` | inert |
| `dt_product` | materialize `rn_Dt*sum` | 11,024 | `5.551115123125783e-17` | inert |
| `masked_increment` | materialize scaled masked increment | 11,024 | `5.551115123125783e-17` | inert |
| `all` | every initially listed boundary | 10,402 | `1.1102230246251565e-16` | non-bit |
| `nemo_dissipation_tree` | `(0.5*rn_ediss)*dissl*en` | 11,024 | `5.551115123125783e-17` | association inert |
| materialized NEMO tree | materialize scalar and both products | 10,402 | `1.1102230246251565e-16` | non-bit |

P2's prediction that only `masked_increment` would close the row is
**REFUTED**. P3's prediction that the all-boundaries arm would close it is
**REFUTED**. P9's prediction that the corrected NEMO association, alone or
with its multiplication boundaries materialized, would close it is
**REFUTED**.

The initial arm called `zfact3_dissl` was an instrument defect, not a source
finding: it inherited legoESM's precombined `diss_rate` and therefore did not
implement its name. The defect is stated here rather than silently rewriting
the frozen evidence. The source was reread, the P9 addendum was committed,
and the corrected arms were run separately.

## Plant

No arm became globally BIT, so P4's proposed “new exact row goes 0 to 1”
antecedent was not reached. The new production-JIT RHS plant was nevertheless
run against the clean baseline to prove the scorer is non-vacuous. It advanced
one finite, previously equal reference cell at `[0,0,0]`; the row changed
from 11,024 to exactly 11,025 unequal cells, the report named
`GYRE-zco.kt2.tke_matrix.production_step.en_rhs`, printed
`STATUS PLANT-FIRED`, and exited nonzero. Artifact: `rhs_plant_jit.json`.

## Decision-41 stage tables after restoring production

The held routing and its regression test were removed. Its manifest again
applies cleanly. The consolidated stage twin then passed at clean commit
`45b9fd471e0248cc7d1b626756a5a4593e6440c2`. Cells are reported as
**BIT / AT-BAR / DEBT**.

Given NEMO's recorded entry:

| kt | stage 1 | external step | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 1 | `13 / 4 / 0` | `11 / 0 / 0` | `14 / 2 / 1` | `11 / 1 / 5` |
| 2 | `10 / 4 / 3` | `0 / 0 / 11` | `11 / 3 / 3` | `10 / 2 / 5` |

Model-chained entry:

| kt | stage 1 | external step | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 1 | `8 / 4 / 5` | `10 / 1 / 0` | `5 / 5 / 7` | `3 / 5 / 9` |
| 2 | `3 / 0 / 14` | `0 / 0 / 11` | `0 / 0 / 17` | `0 / 0 / 17` |

These tables are identical to round 105. The first owned non-bit output is
still kt1 stage-1 U: 7,620 unequal cells, maximum
`5.421010862427522e-20`, AT-BAR. V remains downstream in the same stage.
Artifact: `stage_twin_restored.json`.

## Rule 12, magnitude ordering, and trajectory headline

There is **no round-106 trajectory candidate**. Running a scratch trajectory
for a private measurement selector would not be a Rule-12 landing test, and
the only physics arm in the tree was the already-held round-105 routing. It
was restored before the canonical stage run. Therefore no 954-row comparison
or days 1-30 candidate arm was generated in this round.

The immutable before arm remains
`phase3/merge_main_2026-09-17/after/{ladder.json,day_gap.json}`:

| required headline | immutable value |
|---|---:|
| kt2 T | `1.4210854715202004e-14` |
| kt2 S | `2.1316282072803006e-14` |
| kt2 U | `2.7377110452773967e-12` |
| kt2 V | `3.2849219221489645e-12` |
| kt3 T | `1.627497246303733e-4` K |
| kt3 S | `6.327735185607253e-6` |
| day-30 T RMS | `1.2397011296352737e-2` K |

Those values are reported, not remeasured. The landed default path is
numerically unchanged by construction and by the restored stage twin. This
round's one-ULP RHS residue is not a newly demonstrated owner of kt3 T or the
day-30 gap. The much larger shear operand removed by the held routing remains
the local magnitude owner, but round 105's 59 violating rows and 203,983
violating cells still veto that routing. Aggregate improvement never
overrides Rule 12.

## Other configurations

- **DINO:** an eventual unconditional TKE source-rounding change would share
  this statement and remains a production-step risk. No such change landed.
  The selector is private, defaults empty, and cannot be constructed by a
  card. The focused suite includes all 128 tests in
  `tests/ocean/unit/test_dino_experiment.py`; they pass. This is not promoted
  to a DINO trajectory claim.
- **LOCK_EXCHANGE and OVERFLOW:** no public arithmetic changed. Their inherited
  card audit says they do not instantiate this literal prognostic-TKE RHS;
  no new trajectory claim is made.
- **ORCA2:** remains **UNMEASURED-WITH-SPEC**. Before any eventual arithmetic
  candidate is promoted, run a one-step before/after state comparison through
  ORCA2's production closure. No unmeasured tank is called a pass.

## Review, citations, and tests

The required command was run as a separate read-only Codex pass. It could not
initialize its in-process client under the sandbox. The required verbatim
fallback verdict is:

> independent review unavailable in-sandbox

The tool's terminal line was:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

There is no fabricated `SHIP` or `DO NOT SHIP` verdict. Full log:
`round106/codex_review.log`.

The focused suite command covered the consolidated stage gate tests, citation
gate tests and controls, carried-TKE coefficient tests, and DINO experiment
tests. Exact summary:

> 225 passed, 9 warnings in 142.23s

The warnings are the pre-existing JAX float64-to-float32 scatter future
warnings in nine DINO tests. No known-red exception was needed by this suite.

The receipt citation gate must pass for every mapped citation, and its planted
shift of the round-106 RHS citation must fail with a nonzero exit. The
authoritative post-receipt artifacts are `citation_gate_final.json` and
`citation_gate_final_plant.json`; they are produced after committing this
receipt so their worktree stamps bind the receipt itself.

## Preregistered verdict ledger

| prediction | result |
|---|---|
| P1: routing baseline reproduces eager BIT and JIT 11,024 / one-ULP RHS | **CONFIRMED** |
| P2: `masked_increment` alone closes production JIT | **REFUTED**; unchanged |
| P3: every listed boundary materialized closes production JIT | **REFUTED**; 10,402 remain |
| P4: first newly exact row plant goes 0 to 1 | **NOT REACHED**; no exact arm; non-vacuous 11,024-to-11,025 plant passed |
| P5: attribution can land only after stage order and Rule 12 | **CONFIRMED HOLD**; kt1 stage-1 U remains earlier |
| P6: tank risk remains explicit | **CONFIRMED**; no tank trajectory claim |
| P7: default instrumentation is neutral | **CONFIRMED** by restored baseline and stage twin |
| P9: corrected NEMO dissipation association closes JIT | **REFUTED** in both direct and materialized forms |

## What landed

Only measurement infrastructure landed:

1. a private, default-empty RHS boundary selector threaded through the existing
   consolidated production-step gate;
2. a bounded one-production-step return path for RHS arms;
3. explicit isolated-eager, isolated-JIT, and production-step labels;
4. a nonzero RHS ULP plant;
5. exact compiled-source citation mappings and rigid re-anchors forced by the
   consolidated gate extension.

No public configuration, physics default, carried field, restart schema,
threshold, stabilizer, NEMO source, year harness, reconciliation gate,
freshwater pair, or #1484 guard changed. The round-105 routing remains held in
its existing manifest.

## OPEN — round 107

1. Stay at the same compiled RHS assignment and the same recorded-entry
   production step. Do not promote the held routing and do not walk a
   downstream statement.
2. Preregister and score the RHS **intermediates themselves** in compiled
   order, not only the final RHS: `p_avt*rn2`, `zfact3*dissl`, the complete
   dissipation product, the shear-minus-stratification subtotal, the complete
   parenthesized sum, the timestep product, the masked increment, and the
   final accumulation. Run isolated eager and isolated JIT with one returned
   intermediate per arm so an extra returned tuple cannot materialize all
   boundaries at once. Name the first intermediate that becomes non-bit.
3. Inspect optimized HLO/LLVM for that first arm versus eager semantics. The
   next candidate must enforce IEEE binary64 source-operation rounding, not
   add a physical stabilizer NEMO lacks. A full-precision `reduce_precision`
   discriminator is permitted only after preregistering and proving its
   primal identity on finite, subnormal, signed-zero, infinity, and NaN inputs
   and finite reverse-mode behavior on physical inputs.
4. Any candidate must close the production-step RHS, not merely isolated JIT.
   Then re-evaluate the held shear routing only in the same-stage measurement
   arm. It still cannot land while kt1 stage-1 U is the earlier owned non-bit
   output. If that stage-order condition ever closes, run the full 954-row
   Rule-12 ladder and days 1-30 before promotion.
5. DINO shares the literal TKE statement. Any real arithmetic change requires
   a DINO production-step before/after row. ORCA2 remains
   UNMEASURED-WITH-SPEC; LOCK_EXCHANGE and OVERFLOW must be re-audited if the
   instantiated statement changes.

No NEMO acquisition and no user configuration decision are needed for this
OPEN work.
