# NEMO-testcases L2 GYRE round 107 receipt: production-JIT rn2 owner

Date: 2026-09-18

Incoming tip: `a89ab761ca98920016f4eee2f0ecae3ccd781dad`

Round status: **HELD — no physics or configuration landed**

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round107/`

## Round 107 — compiled-source basis

The record-producing program assigns the TKE scalar coefficient at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:261`, writes the three
matrix rows at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:434-436`, and then
evaluates the right-hand side at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:439-442`.  The latter
contains, in source order, the shear input, the `p_avt * rn2`
stratification product, the dissipation product, the timestep multiplication,
the wet mask, and the accumulation into `en`.

The compiled stage program calls `eos_rab` and `bn2` from the step-entry
tracer slot, copies `rn2b` into `rn2`, and then calls vertical physics with
both formal time levels bound to the step-entry slot at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/stprk3.f90:159-168`.  The
executed `bn2` loop constructs `zrw`, `zaw`, and `zbw`, then writes the
buoyancy-frequency field from the temperature and salinity differences,
free-surface thickness divisor, and wet mask at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1609-1618`.

This round's first returned non-bit value in the authorized RHS walk is the
stratification product: 16,137 of 20,416 words differ, maximum
`4.367513634279986e-22`, through the production JIT.  A one-variable replay
closes the attribution: the production `p_avt` operand is BIT, whereas the
production `rn2` operand differs in 16,232 words at
`4.438452643612534e-19`; multiplying only that measured `rn2` by NEMO's
recorded `p_avt` reproduces all 16,137 product differences and their exact
maximum.  The multiplication is therefore **not** named as the owning
arithmetic statement.  The first owned boundary is its upstream `rn2`
operand, whose compiled producer is `bn2`.  This round does not claim which
statement inside `bn2` is first non-bit; that is the round-108 walk.

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round107.md` at commit
`0b7bdb2ff6c9eba505e7a20ef44c07f2d9bf17b7`.

## Outcome first

P2's proposed boundary is **REFUTED**.  It predicted that the first non-bit
intermediate would be the shear-minus-stratification subtotal.  Instead the
first production-JIT extraction, `p_avt * rn2`, is already non-bit.  Both
operands and the product remain BIT in the isolated eager and isolated-JIT
closures, so this is specifically a full-production-step result.

The operand discriminator then showed that the result is inherited from
upstream `rn2`, not created by the product association.  In production-eager
execution the same operand and product have only five unequal IEEE words,
all `+0.0` versus `-0.0` with numeric maximum `0.0`, while the final RHS is
BIT.  The 16,232 numerical `rn2` differences are therefore production-JIT
only.

P3's proposed FMA mechanism is also **REFUTED**.  The matched production-step
module contains no `llvm.fma`, `fmuladd`, or `contract` marker in any optimized
LLVM file.  The source stack frame for the model expression resolves only to
an `abs` operation in optimized HLO; it does not expose a separately
identifiable multiply.  This is evidence against the preregistered fused
multiply-subtract explanation, not a positive claim about the compiler's
unmapped rewrite.  The exact compiler rewrite remains unmeasured.

No source-rounding candidate was built.  P4's antecedent was false, the
global first owned stage output is still kt1 stage-1 U, and the held routing
needed to expose the one-ULP RHS residue already fails Rule 12.  No physics
change was eligible for trajectory measurement or landing.

## Provenance and fail-closed corrections

All interpreted scientific JSON artifacts carry clean worktree stamps.  The
principal commits are:

- `eeb38dfa8ad0d1d78eb197705fd450c0cd491da7`: private one-intermediate
  production-step instrument and default-selector baseline;
- `ca4c55bd45e87f12f666dd7490a795ab976968e4`: exact temporary reapplication
  of the held round-105 shear routing;
- `96c56a1f8f20dd15be278b09abf6e5dac461d8ce`: corrected state-reference
  production closure and the eight primary arms;
- `6dd06b8c620a9c49c74538333665541325315828`: bit-strict new-row
  classification, intermediate plant, and compiler dump;
- `6eb3c96f8ad5c222d51b20d2a2888ae10434ea70`: post-hoc operand selectors;
- `ffcd91158d5888088708c1bbdf9937350d4ea3e7`: one-variable downstream
  product replay;
- `a6ae4789e3a0f351de73892129cceff6507881ec`: restored production and the
  canonical stage twin;
- `cd4f29fd881791ece1c41768e6dbae089c12d592`: regenerated bit-strict eager
  product artifact with the held routing temporarily applied;
- `e673e4483f36a4f8078fa4ad4a904899b80ea86e`: restored landed production
  after that measurement.

Three refusals/corrections are retained rather than hidden:

1. The first production-JIT product attempt failed before JSON emission
   because the selector was absent from the state-reference closure.  Its
   traceback remains in `rhs_p_avt_rn2_production_jit.log`; the corrected
   closure was committed before the authoritative arm ran.
2. The first eager artifact counted five unequal words but called the row BIT
   because its numeric maximum was zero.  The scorer was made bit-strict, and
   `rhs_p_avt_rn2_production_eager_bitstrict_v2.json` now calls the five
   signed-zero differences AT-BAR.
3. A shell wrapper initially accepted any exit code 1 as a final-RHS plant
   firing; a mistyped commit correctly caused a stamp refusal and emitted no
   JSON.  The refusal is preserved in `rhs_final_plant_restored.log`.  The
   corrected wrapper required exit 1, an emitted report, and the exact
   `PLANT-FIRED` status before accepting
   `rhs_final_plant_restored_v2.json`.

After the instrumentation moved lines in two already-cited files, every
affected citation was re-anchored mechanically.  All 36 moved entries had
identical first/last deltas and unchanged pinned extents; the global map audit
then had zero failures.

## Eight-intermediate walk

Each arm returns exactly one selected intermediate plus the final RHS.  The
reference expression rebuilt from admitted recorded operands is BIT in every
arm.  All isolated-eager intermediate rows and final-RHS rows are BIT.
Counts below are unequal words out of 20,416; `0*` means five signed-zero word
differences with numeric maximum zero, not BIT.

| returned intermediate | production JIT intermediate | isolated JIT intermediate | production eager intermediate | production-JIT final RHS |
|---|---:|---:|---:|---:|
| `p_avt_rn2` | 16,137 / `4.367513634279986e-22` | 0 / `0.0` | 5 / `0.0` (`0*`) | 11,024 / `5.551115123125783e-17` |
| `zfact3_dissl` | 0 / `0.0` | 0 / `0.0` | 0 / `0.0` | 11,024 / `5.551115123125783e-17` |
| `dissipation_product` | 0 / `0.0` | 0 / `0.0` | 0 / `0.0` | 10,402 / `1.1102230246251565e-16` |
| `after_stratification` | 15,997 / `4.367513634279986e-22` | 4,186 / `1.3234889800848443e-23` | 0 / `0.0` | 11,024 / `5.551115123125783e-17` |
| `parenthesized_sum` | 12,131 / `3.3881317890172014e-21` | 6,326 / `3.3881317890172014e-21` | 0 / `0.0` | 11,024 / `5.551115123125783e-17` |
| `dt_product` | 11,743 / `5.551115123125783e-17` | 5,632 / `5.551115123125783e-17` | 0 / `0.0` | 11,024 / `5.551115123125783e-17` |
| `masked_increment` | 11,743 / `5.551115123125783e-17` | 5,632 / `5.551115123125783e-17` | 0 / `0.0` | 11,024 / `5.551115123125783e-17` |
| `final_accumulation` | 11,024 / `5.551115123125783e-17` | 5,117 / `5.551115123125783e-17` | 0 / `0.0` | 11,024 / `5.551115123125783e-17` |

The returned value is observability instrumentation, and it can change XLA's
optimization graph.  This is measured, not assumed away: returning the full
dissipation product changes the final production row from 11,024 cells at one
ULP to 10,402 cells at two ULP, and returning the product changes the isolated
final count from 5,117 to 4,934.  Consequently, final-row count changes are
not used for attribution.  Attribution rests on the selected value itself and
the one-variable operand replay below.

The default-empty selector is neutral.  Before temporarily applying the held
routing, `instrument_default_production_jit.json` reproduces the incoming
production boundary: `p_sh2` is 17,400 cells non-bit at
`3.811744924985501e-14`, the matrix rows are BIT, and the RHS is 11,993 cells
non-bit at `5.488912518947231e-10`.  With the held routing, `p_sh2` and all
three matrix rows are BIT and the final-accumulation arm reproduces 11,024
one-ULP RHS differences.  P1 is confirmed.

## Operand attribution

The operand arms were post-hoc discriminators, clearly labeled as such.  Each
returns one production operand.  The downstream replay multiplies that
selected operand by the other NEMO-recorded operand in source order; it does
not claim to reproduce the production-JIT multiplication.

| selected production operand | operand unequal / max | downstream product replay unequal / max | verdict |
|---|---:|---:|---|
| `p_avt` under production JIT | 0 / `0.0` | 0 / `0.0` | operand and product BIT |
| `rn2` under production JIT | 16,232 / `4.438452643612534e-19` | 16,137 / `4.367513634279986e-22` | reproduces the complete product row |
| `rn2` under production eager | 5 / `0.0` | 5 / `0.0` | signed-zero words only; final RHS BIT |

The product count and maximum in the JIT `rn2` replay exactly equal the
independently measured production-JIT product row.  This closes the local
budget without dividing by a rounded scale factor: no product mismatch
remains for `p_avt` or multiplication association to own.

## Compiler discriminator

The external XLA dump is under `xla_p_avt_rn2_production/`; its 455 MB are not
committed.  The matched module is `module_0389.jit__step_jitted`.
`compiler_ir_audit.txt` records SHA-256
`095e244716f3f8aedbf6c4eb14355be4c24c91ba5e5b39f69fa3cab5e6cec5de`
for the before-optimization HLO and
`5eb1b74131df050a4010495e4e360e90b287186499f1c2b00aa5ef252cb7a1e2`
for the optimized HLO.  Zero optimized LLVM files contain an FMA,
`fmuladd`, or contract marker.  Because the source stack map exposes only an
`abs` operation at that frame, the dump cannot name the exact rewrite.

P3 is therefore refuted as preregistered, and P4 was not run: the measured
first boundary was upstream `rn2`, not the predicted contracted subtraction.
No `reduce_precision` primitive or other rounding candidate was introduced.

## Plants

The new intermediate plant advanced one finite nonzero reference value at
index `[1,1,24]` by one ULP.  The product row moved from 16,137 to exactly
16,138 unequal words, printed `STATUS PLANT-FIRED`, and exited 1.  Artifact:
`rhs_p_avt_rn2_plant.json`.

The established final-RHS plant ran on restored landed production.  It moved
the RHS row from 11,993 to exactly 11,994 unequal cells at index `[0,0,0]`,
printed `STATUS PLANT-FIRED`, and exited 1.  The accepting wrapper also
verified the JSON status and plant name.  Artifact:
`rhs_final_plant_restored_v2.json`.

## Decision-41 stage tables

After removing the held routing, the consolidated stage twin passed at clean
commit `a6ae4789e3a0f351de73892129cceff6507881ec`.  The only files changed after
that stamp and before this receipt were citation-map documentation; the
production diff is empty.  Cells are **BIT / AT-BAR / DEBT**.

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

The first owned non-bit output remains kt1 stage-1 U: 7,620 unequal cells,
maximum `5.421010862427522e-20`, AT-BAR.  Artifact:
`stage_twin_restored.json`.  These tables are identical to round 106.

## Rule 12, magnitude ordering, and trajectory headline

There is **no round-107 trajectory candidate**.  The private selectors do not
alter the default path, and the only physics used for discrimination was the
already-held round-105 shear routing.  It was removed after measurement and
its manifest applies cleanly.  That routing remains vetoed by its 59 Rule-12
violating rows / 203,983 violating cells.  A downstream one-ULP TKE residue
cannot bypass the earlier kt1 stage-1 U output.

The immutable before arm remains
`phase3/merge_main_2026-09-17/after/{ladder.json,day_gap.json}`.  The required
headlines are reported, not remeasured:

| required headline | before | round-107 after |
|---|---:|---:|
| kt2 T | `1.4210854715202004e-14` | NOT RUN — no eligible candidate |
| kt2 S | `2.1316282072803006e-14` | NOT RUN — no eligible candidate |
| kt2 U | `2.7377110452773967e-12` | NOT RUN — no eligible candidate |
| kt2 V | `3.2849219221489645e-12` | NOT RUN — no eligible candidate |
| kt3 T | `1.627497246303733e-4` K | NOT RUN — no eligible candidate |
| kt3 S | `6.327735185607253e-6` | NOT RUN — no eligible candidate |
| day-30 T RMS | `1.2397011296352737e-2` K | NOT RUN — no eligible candidate |

The newly isolated `rn2` boundary is many orders smaller than the kt3 T and
day-30 gaps and has not been shown to own either.  Magnitude ordering therefore
does not convert this attribution result into a landing candidate.

## Other configurations

- **DINO:** no arithmetic or card changed.  An eventual unconditional `bn2`
  correction would be shared and requires a DINO production-step before/after
  row.  The focused DINO unit suite result is recorded below; it is not a
  trajectory claim.
- **LOCK_EXCHANGE and OVERFLOW:** no public TKE or buoyancy-frequency
  arithmetic changed, so there is no round-107 candidate to score.  No pass is
  claimed from absence of execution.
- **ORCA2:** remains **UNMEASURED-WITH-SPEC**.  Before any `bn2` candidate is
  promoted, run a one-step production-closure before/after state comparison
  and register every moved row.

## Review, citations, and tests

The required separate read-only Codex command attempted to refute the `rn2`
attribution, observer-effect handling, stage-order hold, and Rule-12 table.  It
could not initialize its in-process client in the read-only sandbox.  The
required verbatim fallback verdict is:

> independent review unavailable in-sandbox

The tool's terminal line was:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

No `SHIP` or `DO NOT SHIP` verdict is fabricated.  Full log:
`codex_review.log`.

The first focused run exposed the stale historical citations caused by this
round's line insertions; its exact summary was:

> 1 failed, 242 passed, 9 warnings in 143.83s (0:02:23)

The one failure was the aggregate receipt citation control.  All moved map
keys and every live prose occurrence were then shifted rigidly, with both
endpoints moving together and extents unchanged.  The isolated citation suite
subsequently reported:

> 16 passed in 1.80s

The full focused suite was then rerun on the clean committed repair.  It
covered the round-107 discriminator tests, modified round-103 replay tests,
the consolidated stage-gate tests, all citation controls, carried-TKE tests,
and the complete DINO experiment file.  Its authoritative terminal summary
was:

> 243 passed, 9 warnings in 142.93s (0:02:22)

The nine warnings are the existing JAX float64-to-float32 scatter future
warnings in DINO tests; there were no test failures and no known-red exception
was needed by this focused suite.  Logs: `focused_tests.log`,
`citation_tests_after_reanchor_v2.log`, and `focused_tests_final.log`.

The authoritative post-receipt citation artifacts are
`citation_gate_final.json` and `citation_gate_final_plant.json`.  The clean
gate must map every compiled-source citation above; the planted shift of
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1609-1618` must emit
FAIL and exit nonzero.

## Preregistered verdict ledger

| prediction | result |
|---|---|
| P1: held routing reproduces BIT matrix/shear rows, eager BIT RHS, isolated-JIT 5,117, production-JIT 11,024 | **CONFIRMED** on the uninstrumented final-accumulation boundary; returned-product observability changes the isolated final count to 4,934 |
| P2: first non-bit intermediate is `after_stratification` | **REFUTED**; the returned product is already 16,137 cells non-bit in production JIT, inherited entirely from `rn2` |
| P3: optimized code exposes a fused/reassociated multiply-subtract | **REFUTED**; matched LLVM has zero FMA/contract markers and the source stack map exposes no standalone multiply |
| P4: full-precision source-round candidate if P2 matches | **NOT RUN**; antecedent false |
| P5: intermediate and final-RHS plants fire and exit nonzero | **CONFIRMED**; 16,137→16,138 and 11,993→11,994 |
| P6: no landing while stage order and Rule 12 veto it | **CONFIRMED HOLD** |
| P7: other-configuration risk remains explicit | **CONFIRMED**; no tank trajectory pass claimed; ORCA2 remains unmeasured-with-spec |
| P8: default-empty instrumentation is neutral | **CONFIRMED** by the incoming production boundary and restored stage twin |

## What landed

Only measurement infrastructure landed:

1. a private, default-empty selector that returns one RHS intermediate through
   the existing production-step gate;
2. bit-strict classification for the new rows, including signed zero;
3. a nonzero one-ULP intermediate plant;
4. post-hoc single-operand extraction and a clearly labeled recorded-order
   product replay;
5. direct tests for selector validation, one-return semantics, bit-strict
   scoring, replay labeling, and plant behavior;
6. compiled-source citation mappings and rigid re-anchors.

No public configuration, physics default, carried field, restart schema,
threshold, stabilizer, NEMO source, year harness, reconciliation gate,
freshwater pair, or #1484 guard changed.  The round-105 routing remains held
only in its existing manifest.

## OPEN — round 108

1. Stay at the first owned boundary named here: the production-JIT `rn2`
   input generated by compiled `bn2`.  Do not attempt a source-rounding patch
   in the downstream TKE RHS and do not walk a later operator.
2. Preregister and extend the **existing consolidated stage gate**, not a new
   harness, to walk the compiled `bn2` loop in statement order under NEMO's
   recorded step-entry tracers, expansion coefficients, free-surface field,
   thickness, and mask.  Return one value per arm through the production step:
   `zrw`, `zaw`, `zbw`, the temperature contribution, the salinity
   contribution, their difference, the gravity product, the thickness
   division, and the masked `rn2` output.  Score production JIT, production
   eager, isolated JIT, and isolated eager; plants must fire under production
   JIT.
3. First reproduce this round's 16,232-cell / `4.438452643612534e-19`
   production-JIT `rn2` boundary and the five signed-zero eager words.  Name
   the first internal non-bit value.  Treat returned-value observer effects
   as measured rows, not as candidate benefit.
4. Read the optimized production module only after the first internal boundary
   is measured.  A rounding primitive is not a candidate unless it makes that
   production-step row BIT with NEMO's own inputs and has preregistered special
   value and reverse-mode controls.
5. Stage order remains binding: kt1 stage-1 U is still the earlier owned
   output.  Any `bn2` change stays held until that output closes and the full
   954-row ladder plus days 1-30 passes against the immutable before arm.
6. DINO shares buoyancy-frequency production.  Any real arithmetic candidate
   requires a DINO production-step before/after row.  ORCA2 remains
   UNMEASURED-WITH-SPEC; re-audit LOCK_EXCHANGE and OVERFLOW before promotion.

No NEMO acquisition and no user configuration or carried-state decision are
needed for this OPEN work.
