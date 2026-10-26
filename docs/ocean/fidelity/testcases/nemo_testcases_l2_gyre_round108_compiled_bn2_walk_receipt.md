# NEMO-testcases L2 GYRE round 108 receipt: compiled `bn2` walk

Date: 2026-09-18

Incoming tip: `a91c48e11c2fae2d6d482dfceba3e7097cbede69`

Round status: **HELD — diagnostic instrumentation landed; no physics or
configuration changed**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round108/`

## Round 108 — compiled-source basis

The record-producing GYRE card selects TEOS-10 and disables S-EOS at
`GYRE_OMIP_L2_P3_SM_R101TKEW/EXP00/namelist_cfg:126-128`.  The compiled
stage calls `eos_rab` and `bn2` from the step-entry tracer slot, copies
`rn2b` to `rn2`, and then invokes vertical physics at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/stprk3.f90:159-168`.

The executed expansion-coefficient branch is the TEOS-10/EOS-80 case,
including its depth and salinity transforms, ALP/BET Horner programs, mask,
and beta division, at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1257-1310`.
The following compiled `bn2` loop is
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1609-1618`.
Within that loop the depth weight is written at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1610-1611`,
the thermal and haline coefficients at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1613` and
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1614`, and the
temperature contribution, salinity contribution, difference, gravity
product, thickness division, and wet-mask multiplication at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1616`,
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1617`,
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1616-1617`, and
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1616-1618`.

The first returned non-bit value is the compiled depth-weight statement:
14,353 of 20,416 words differ under the production JIT, maximum
`1.2212453270876722e-15`.  Production eager is BIT.  Isolated JIT on the
replay-certified operands has exactly the same 14,353-word result, and the
direct production-versus-isolated-JIT comparison is BIT over all 20,416
words.  The result therefore belongs to JIT evaluation of this statement,
not to an inherited production-only geometry operand.

This does **not** make the statement a correction candidate.  Carrying its
measured output through the remaining source program explains 2,425 final
unequal words, while the production output has 7,094 unequal words and a
6,860-word residual against that one-variable replay.  Those counts overlap
and must not be added.  The statement is first in compiled order, but it is
not the sole owner of the final `rn2` row.

## Outcome first

The preregistered P2 prediction is **REFUTED**.  It predicted that `zrw` would
be BIT and that `zaw` would be the first JIT-only difference.  Instead `zrw`
is already DEBT under both the production and isolated JITs and is BIT under
both eager modes.

P3 is also **REFUTED as a complete final-output ownership claim**.  The
measured `zrw` discrepancy propagates to the final magnitude, but a separate
6,860-word JIT residual remains.  The walk names the first non-bit statement;
it does not claim that this statement alone owns the final field.

P4's proposed source-tree mechanism is **REFUTED**.  The optimized production
module fuses the three depth products, two subtractions, and division into one
vectorized kernel, but the emitted LLVM retains separate `fmul`, `fsub`, and
`fdiv` instructions with no FMA, `fmuladd`, `contract`, `fast`, or reassociate
marker.  That is evidence for a JIT/backend boundary and against the proposed
association rewrite.  It is not evidence for a safe source-rounding patch.

No arithmetic candidate was preregistered or built.  The Decision-41 first
owned stage output remains kt1 stage-1 U, upstream of this diagnostic walk.
No trajectory arm was eligible, no physics changed, and the landing verdict
is **HELD**.

## Provenance and fail-closed corrections

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round108.md` at commit
`94f66b07bfee88f4c1ee4f0cc38f379ba457e01b`.  Principal implementation and
measurement commits are:

- `bcf725065`: add the one-output walk to the consolidated stage gate;
- `595acedb1`: correct the replay to the configured TEOS-10 branch;
- `8f2b781b8`: bridge raw recorded tracers and separate the model-upstream
  control;
- `2c270eb9a`: carry measured `zrw` through the remaining source statements;
- `ea158c892`: compare production and isolated-JIT values directly;
- `2fc98b42a`, `7e4d88b76`, and `60463caad`: rigid citation re-anchoring and
  compiled-source map coverage.

Two failed attempts are retained, not rewritten as measurements:

1. The initial gate encoded the preregistration's mistaken S-EOS reading.  It
   checked the instantiated card before a numerical row, refused with
   `round-108 bn2 replay requires the compiled GYRE S-EOS arm`, and emitted no
   JSON.  Artifact: `bn2_masked_rn2_production_jit.log`.
2. The corrected TEOS-10 replay initially used legoESM's below-seafloor-
   extrapolated tracers instead of the promised raw NEMO entry.  The
   NEMO-from-NEMO replay was numerically exact but moved five dry signed-zero
   words and refused before production scoring.  Artifact:
   `bn2_masked_rn2_production_jit_v2.log`.  The corrected arm passes raw
   recorded T/S and replay-certified alpha/beta; the ordinary model-upstream
   arm remains a separate reported control.

The nine primary JIT/eager artifacts are stamped at clean commit
`8f2b781b8a2c94da2997f07b7d17a03f22c1f63f`; the downstream propagation is
stamped at `2c270eb9a9e84ccfa3fa170f8f5dd7e7523030bb`; and the final ownership
artifact is stamped at `ea158c8927c0a77bd8c9c5a2c4aced650cbce4f3`.

## Input bridge and reproduced boundary

Before interpreting an internal row, every arm rebuilds the configured
TEOS-10 alpha/beta program and complete masked `rn2` from NEMO's admitted
raw step-entry T/S, masks, free surface, reference geometry, and constants.
The NEMO-from-NEMO replay is BIT, and reconstructed live `e3w` is BIT against
the round-59 operand record.

The ordinary model-upstream control reproduces round 107 exactly:

| execution | final unequal words | maximum absolute error | verdict |
|---|---:|---:|---|
| production JIT | 16,232 | `4.438452643612534e-19` | P1 confirmed |
| production eager | 5 signed-zero words | `0.0` | P1 confirmed, not BIT |

The given-NEMO-input arm isolates the local program.  Its final production
JIT row is 7,094 words at `4.0657581468206416e-20`; its production-eager row
is BIT.  The model-upstream minus given-input difference is inherited from
upstream tracer/alpha/beta handling and is not assigned to this `bn2` walk.

## Nine-value compiled-order walk

Counts are unequal IEEE words out of 20,416.  Each arm returns one selected
value.  `PJ` is the full production step under JIT with NEMO inputs; `PE` is
the production closure with JIT disabled; `IJ` and `IE` are isolated-closure
JIT and eager; `CJ` is the ordinary model-upstream production JIT.  The
isolated labels are discriminators, never substitutes for production proof.

| selected value | PJ unequal / max | PE unequal / max | IJ unequal / max | IE unequal / max | CJ unequal / max |
|---|---:|---:|---:|---:|---:|
| `zrw` | 14,353 / `1.2212453270876722e-15` | 0 / `0.0` | 14,353 / `1.2212453270876722e-15` | 0 / `0.0` | 14,353 / `1.2212453270876722e-15` |
| `zaw` | 4,558 / `5.421010862427522e-20` | 0 / `0.0` | 4,558 / `5.421010862427522e-20` | 0 / `0.0` | 18,700 / `6.415488874077793e-5` |
| `zbw` | 4,527 / `1.0842021724855044e-19` | 0 / `0.0` | 4,527 / `1.0842021724855044e-19` | 0 / `0.0` | 19,976 / `7.952878502919095e-4` |
| temperature contribution | 4,167 / `1.0842021724855044e-19` | 0 / `0.0` | 4,167 / `1.0842021724855044e-19` | 0 / `0.0` | 15,534 / `1.0842021724855044e-18` |
| salinity contribution | 4,081 / `5.421010862427522e-20` | 0 / `0.0` | 4,081 / `5.421010862427522e-20` | 0 / `0.0` | 16,921 / `4.743384504624082e-18` |
| contribution difference | 8,129 / `1.6263032587282567e-19` | 0 / `0.0` | 8,129 / `1.6263032587282567e-19` | 0 / `0.0` | 16,522 / `4.7704895589362195e-18` |
| gravity product | 7,689 / `1.734723475976807e-18` | 0 / `0.0` | 7,689 / `1.734723475976807e-18` | 0 / `0.0` | 16,428 / `4.683753385137379e-17` |
| thickness division | 7,094 / `4.0657581468206416e-20` | 0 / `0.0` | 7,094 / `4.0657581468206416e-20` | 0 / `0.0` | 16,232 / `4.438452643612534e-19` |
| masked final `rn2` | 7,094 / `4.0657581468206416e-20` | 0 / `0.0` | 7,094 / `4.0657581468206416e-20` | 0 / `0.0` | 16,232 / `4.438452643612534e-19` |

Returning an intermediate changes the compilation graph.  The final-output
row ranges from 4,949 to 7,127 unequal words across the given-input selector
arms even though every selected row above is stable between production and
isolated JIT.  Final-output count variation is therefore recorded observer
effect, not candidate benefit and not used to order statements.

## First-statement propagation and ownership

The final ownership artifact
`bn2_zrw_production_jit_ownership.json` records these independent checks:

| comparison | unequal words | maximum | meaning |
|---|---:|---:|---|
| production `zrw` vs NEMO | 14,353 | `1.2212453270876722e-15` | first non-bit selected value |
| isolated-JIT `zrw` vs NEMO | 14,353 | `1.2212453270876722e-15` | JIT-only local reproduction |
| production vs isolated-JIT `zrw` | 0 | `0.0` | production geometry adds no value difference |
| measured `zrw` carried through source-order eager remainder vs NEMO final | 2,425 | `4.0657581468206416e-20` | `zrw` contributes at the final magnitude |
| production final vs that one-variable replay | 6,860 | `4.0657581468206416e-20` | independent later JIT residual remains |
| production final vs NEMO | 7,094 | `4.0657581468206416e-20` | complete given-input output |

The eager controls are all BIT for the given-input arm, including both
propagation rows.  In the ordinary model-upstream eager arm, `zrw` and its
source-order propagation are BIT while the final/residual rows contain only
the five inherited signed-zero words.

This satisfies the ordered walk: the compiled `zrw` statement is the first
non-bit statement.  It refutes a stronger single-owner interpretation and
leaves later JIT-only boundaries open.  Because the first statement's emitted
tree already matches the compiled arithmetic and no candidate was frozen,
the round stops rather than guessing a rounding primitive.

## Optimized-code discriminator

The external compiler dump is under `xla_zrw_production/` and is not
committed.  The matched given-input production module is
`module_0389.jit__step_jitted`.  SHA-256 values are:

- before-optimization HLO:
  `737c30ca535667bb7cfa3adda929fc589e7bacd3e586d90f23af330250d75251`;
- optimized HLO:
  `4cb79c7be74d47e46d4d89f16261f740f67c6638aa8ebc5a0b6932a5cac4f721`;
- unoptimized matched LLVM kernel:
  `15f53ab21e1be487c9d54161c951d7e9e9092343a7a99c7b2ba2ae586982cfc4`;
- optimized matched LLVM kernel:
  `82ccc4e24176a5e1372d879c80def4c99a7b7284476768f03fb6aa08797ce69b`.

The optimized HLO's `subtract_divide_fusion.8` contains three multiplies,
two subtractions, and one division in source order.  The optimized LLVM
vector body repeats three vector `fmul`, two vector `fsub`, and one vector
`fdiv`; its scalar tail does the same.  Across the matched optimized file
there are 25 multiply, 16 subtract, and 8 divide instructions and zero FMA,
`fmuladd`, contract, fast, or reassociate markers.  P4's association mechanism
is therefore refuted.  The remaining explanation is only an inference:
NEMO/NumPy eager and XLA's fused SIMD kernel differ despite the same visible
tree.  This round does not promote that inference to a correction.

## Plant

The production-JIT ULP plant advanced one finite nonzero, previously equal
`zrw` reference at index `[0,0,0]`, baseline
`0.5031944354208339`, by one ULP.  The row moved from exactly 14,353 to
14,354 unequal words, the report named
`GYRE-zco.kt2.bn2.production_step.zrw`, the log printed
`STATUS PLANT-FIRED`, and the process exited 1.  Artifact:
`bn2_zrw_production_jit_plant.json`.  P5 is confirmed.

## Decision-41 stage tables

The consolidated stage twin passed at clean commit
`7e4d88b7611ac792629cf9caa88f5849d77e8a17`.  Both complete scientific row
arrays are equal to round 107, not merely their counts.  Cells below are
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

The first owned non-bit stage output remains kt1 stage-1 U: 7,620 unequal
cells, maximum `5.421010862427522e-20`, AT-BAR.  Artifact:
`stage_twin_final.json`.  The later `ea158c892` commit changes only the gate's
diagnostic comparison and its direct unit test; model production code is
unchanged after the clean stage-twin stamp.

## Rule 12, magnitude, and required headline

There is no round-108 trajectory candidate.  The private selectors default
empty, and the restored stage twin proves that the committed instrumentation
moves no stage row.  Running the 954-row ladder or 30-day integration on a
private returned-value observer would not be a candidate comparison, so both
are correctly not run.

The immutable before arm remains
`phase3/merge_main_2026-09-17/after/{ladder.json,day_gap.json}`:

| required headline | before | round-108 after |
|---|---:|---:|
| kt2 T | `1.4210854715202004e-14` | NOT RUN — no candidate |
| kt2 S | `2.1316282072803006e-14` | NOT RUN — no candidate |
| kt2 U | `2.7377110452773967e-12` | NOT RUN — no candidate |
| kt2 V | `3.2849219221489645e-12` | NOT RUN — no candidate |
| kt3 T | `1.627497246303733e-4` K | NOT RUN — no candidate |
| kt3 S | `6.327735185607253e-6` | NOT RUN — no candidate |
| day-30 T RMS | `1.2397011296352737e-2` K | NOT RUN — no candidate |

The local final `rn2` error, `4.0657581468206416e-20`, is many orders below
the kt3 T and day-30 T gaps and is not claimed to own either magnitude.

## Other configurations

DINO executes the same buoyancy-frequency routine twice per leapfrog step,
as shown by its compiled calls at
`DINO/BLD/ppsrc/nemo/stpmlf.f90:187-193`.  This round changes no default
arithmetic, so there is no before/after DINO candidate row.  Any future
`zrw` correction requires a DINO production-step comparison before reliance.

LOCK_EXCHANGE and OVERFLOW were not reclassified as passes: no instantiated
statement changed, and no promotion was attempted.  They must be re-audited
for execution if a real candidate is proposed.  ORCA2 remains
**UNMEASURED-WITH-SPEC**: instantiate its resolved card in fp64, run one
production step before/after from the same entry, and bit-score its
buoyancy-frequency and consumed TKE outputs before promotion.

## Frozen prediction ledger

| prediction | verdict | evidence |
|---|---|---|
| P1 reproduced boundary and replay | **CONFIRMED** | chained JIT is 16,232 / `4.438452643612534e-19`; eager is five signed-zero words; NEMO replay is BIT |
| P2 first non-bit at `zaw` | **REFUTED** | `zrw` is already 14,353 / `1.2212453270876722e-15` under both JIT modes |
| P3 first statement wholly explains final | **REFUTED** | 2,425 propagated words plus a 6,860-word production residual; counts overlap |
| P4 fused/reassociated source tree | **REFUTED** | fusion exists, but matched LLVM preserves `fmul`/`fsub`/`fdiv` and has no contraction/fast marker |
| P5 production plant | **CONFIRMED** | exactly one added unequal word, nonzero baseline, nonzero exit, `PLANT-FIRED` |
| P6 no landing before stage order and proof | **CONFIRMED** | no candidate; kt1 stage-1 U remains first owned output |
| P7 configuration scope | **CONFIRMED as scope, not as pass** | DINO compiled execution cited; tanks not re-audited; ORCA2 spec retained |
| P8 default instrumentation neutrality | **CONFIRMED** | complete final stage arrays equal round 107 |

## Verification and independent review

Focused and full-suite terminal summaries, the independent Codex verdict,
and the final citation-gate results are added below after their clean runs.

## OPEN — round 109

1. Return to Decision-41 execution order.  The first owned stage output is
   still kt1 stage-1 U; do not continue downstream TKE/`bn2` work and do not
   build a `zrw` rounding candidate while that upstream row remains non-bit.
2. Reuse the consolidated stage-1 instruments and the admitted round-46,
   round-98, and round-99 records.  Individual HPG, LDF, vorticity, KEG, and
   ZAD rows are already BIT given NEMO's entry.  Resume at the first
   post-operator W/transport/RK boundary that can feed the kt1 stage-1 U
   output; compare production JIT, production eager, and isolated JIT, with a
   production plant.  Do not repeat the generic full-RHS split already vetoed
   in rounds 97/99.
3. Treat round 108's `zrw` result as a bounded diagnostic: first non-bit under
   JIT, but not sole owner of final `rn2`, and no visible source association
   defect.  If a later authorized round returns here, first discriminate the
   6,860-word downstream residual one statement at a time; no rounding
   primitive is eligible without a frozen special-value and reverse-mode
   proof through the production step.
4. A future candidate lands only after its first owned stage row closes and
   the full 954-row Rule-12 ladder plus days 1-30 passes against the immutable
   before arm.  Re-audit both tanks, measure DINO if the shared statement
   changes, and retain the ORCA2 one-step spec.

No NEMO acquisition, user configuration decision, or carried-state decision
is requested.
