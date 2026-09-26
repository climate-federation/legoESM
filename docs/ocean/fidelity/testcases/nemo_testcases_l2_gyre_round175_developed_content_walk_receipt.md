# Round 175 receipt — developed temperature-content producer walk

**Status: HELD.**  No production physics, configuration, carried state,
restart schema, card default, or trajectory changed.  At the developed
step-1081 state, the before-content family is bit exact.  The first non-bit
input to NEMO's content expression is inherited `T(Krhs)`: 18,000/18,000
active cells differ under both the production JIT step and the complete eager
step.  Replacing only that operand carries essentially the whole content
residual; replacing only live `e3t(Kmm)` carries eleven orders of magnitude
less RMS.  NEMO's content expression itself is bit exact given NEMO operands,
so this round found no statement that can land.

Preregistration: `PREREG_nemo_testcases_l2_gyre_round175.md`, commit
`a822f2d82`.  Final instrument commit: `a24f763e97869af7b2c7164f21c079ac27644fb7`.
Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round175/`.

## Compiled statement

The record-producing build executes the surface form at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:549-552` and the
interior form at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:562-567`.  In both,
the complete right-hand side is, in compiled association,

`e3t(Kbb) * T(Kbb) + p2dt * e3t(Kmm) * T(Krhs)`.

The cited lines also show that the result is immediately consumed by the
forward recurrence at line 567.  The committed citation map pins the enclosing
compiled range
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:549-567` to both the
second-recurrence comment and the final `pt(Kaa)` assignment.  No dead branch
or source-tree surrogate is cited.

## Admission and execution context

`daily_record_audit_v11.json` admits all 360 Round-132 daily boundaries and
all 12 overlaps with `year_fromrest/nemo_seed0`.  The earlier
`daily_record_audit_v10.log` is retained: it correctly refused a mistaken
`year_owners/nemo_seed0` control path rather than silently accepting it.

The walk extends `nemo_testcase_l2_gyre_year_owners.py`; it does not create a
second harness.  It drives the real production step from NEMO's admitted
step-1080 restart.  `production_step_jit` calls `self._step_jitted`;
`production_eager` executes the same full step with JIT disabled.  Observation
uses the existing `tracer_process_trace` and `vertical_solve_trace` products;
there is no callback or added returned field.  Each mode compares its returned
state with the corresponding unobserved run: **0 unequal state bytes**.

`T(Krhs)` is reconstructed as the model's directly observed accumulated
content divided by the model's directly observed `p2dt*e3t(Kmm)`.  It is
therefore an equivalent consumed operand, not a newly returned internal field.
The authoritative content and reciprocal-substitution rows use the direct
production traces.  This limitation is why Round 176 must walk the directly
observed process boundaries before naming an operator statement.

## Registered walk

All rows score the 18,000 active cells.  Values are cells unequal, maximum
absolute difference, and RMS respectively.

| row | production JIT | complete eager | verdict |
|---|---:|---:|---|
| `T(Kbb)` | 0; 0; 0 | 0; 0; 0 | BIT |
| `e3t(Kbb)` | 0; 0; 0 | 0; 0; 0 | BIT |
| before content | 0; 0; 0 | 0; 0; 0 | BIT |
| equivalent `T(Krhs)` | 18,000; 2.4737100947103865e-08 K; 7.401204554995205e-10 K | 18,000; 2.473710094784989e-08 K; 7.401204555027710e-10 K | first non-bit |
| `e3t(Kmm)` | 18,000; 1.9667822925839573e-10 m; 4.079831732798634e-11 m | same | non-bit, downstream-secondary |
| accumulated `Krhs` content | 18,000; 3.769802907240938e-02 K m; 7.046353169355859e-04 K m | 18,000; 3.769802907354625e-02 K m; 7.046353169465208e-04 K m | non-bit |
| complete content | 18,000; 3.7698029072316785e-02 K m; 7.046353169351283e-04 K m | 18,000; 3.7698029073453654e-02 K m; 7.046353169460630e-04 K m | non-bit |

The literal NEMO reconstruction is 0/18,000 unequal in both modes.  Rebuilding
the model content from the two direct model traces is also 0/18,000 unequal,
which closes the observer's arithmetic accounting.

### Reciprocal operand substitutions

| one model operand, all other inputs NEMO | JIT RMS / max (K m) | eager RMS / max (K m) |
|---|---:|---:|
| `T(Krhs)` only | 7.046353169351340e-04 / 3.769802907237474e-02 | 7.046353169460688e-04 / 3.769802907351161e-02 |
| `e3t(Kmm)` only | 7.954944128515275e-15 / 9.411915691259765e-14 | same |

Thus `T(Krhs)` carries 0.99999999999936 of the measured JIT accumulated-row
RMS, while the thickness-only RMS is about 8.9e10 times smaller.  This is a
measured one-variable attribution of this boundary, not a claim that the
upstream `T(Krhs)` operator has yet been identified.

## Frozen predictions and controls

| preregistered prediction | result |
|---|---|
| Round-125 literal reconstruction stays bit exact | CONFIRMED, 0/18,000 |
| observers move zero state bytes | CONFIRMED in JIT and eager |
| before family BIT; accumulated family first non-bit with at least 1,000 cells | CONFIRMED; 18,000 cells |
| complete content max at least 1e-6 K m | CONFIRMED; 3.7698e-02 K m |
| JIT and eager name the same first row | CONFIRMED; `T(Krhs)` |
| first-wet `T(Krhs)` ULP moves accumulated and complete rows | **REFUTED** |

The last prediction failed in two useful ways and is not rewritten away.  The
first wet value was rounded out even in the product (`content_krhs_ulp_plant.log`).
Searching for a surviving product ULP found cell `[1,1,1]`: it moves exactly
one accumulated-content cell, but the subsequent addition rounds it out of the
complete RHS (`content_krhs_ulp_plant_final.log`).  There is no wet-cell
`T(Krhs)` one-ULP perturbation that survives both boundaries.  The final
fail-closed controls therefore separate the two statements:

* `developed-content-krhs-ulp` moves one accumulated-content cell and exits 1
  with `STATUS PLANT-FIRED`;
* `developed-content-rhs-ulp` moves one complete-RHS cell and exits 1 with
  `STATUS PLANT-FIRED`.

Earlier callback-based observers are also retained as refuted instrument
attempts: the narrowed callback still moved 3,864 returned-state bytes.  It
was discarded, not waived.  The final existing-trace observer moves zero.

## Landing and certified trajectory

There is no production candidate: the cited NEMO expression is already bit
exact given NEMO operands.  Accordingly the Decision 43/45/55/59 trajectory,
year, cards, DINO, and tanks gates are not re-run; nothing they score changed.
The immutable Round-163 arm remains authoritative:

| headline | unchanged value |
|---|---:|
| first over bar | kt=3 |
| kt2 U max abs | 8.326672684688674e-17 m s-1 (AT-BAR) |
| kt2 V max abs | 9.714451465470120e-17 m s-1 (AT-BAR) |
| kt3 T max abs | 4.940071072212504e-07 K |
| kt3 S max abs | 4.0086298724872904e-08 psu |
| day-30 T3D RMS | 6.572574374770603e-05 K |
| day-240 T3D RMS | 1.644836070117868e-02 K |
| day-360 T3D RMS | 1.122566001855131e-02 K |

No GYRE, generic, DINO, tank, ORCA2, or MPAS card executes changed production
code because there is no changed production code.  No configuration or
carried-state decision is requested.

## Independent review

The required separate review was attempted with `codex exec --sandbox
read-only`.  It was unavailable in this sandbox.  Its complete verdict is
quoted verbatim:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)  
> Reading additional input from stdin...  
> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

This is **independent review unavailable in-sandbox**, not a SHIP verdict.
Because the round lands no physics and its claims are controlled by committed
tests and nonzero plants, the measurement and HELD conclusion remain usable.

## OPEN — round 176

Stay at developed step 1081 and walk the producer of accumulated `T(Krhs)` in
NEMO's compiled order using the admitted Round-123 process-budget record and
the existing production process/FCT traces: advection, surface boundary,
shortwave, and lateral diffusion.  Register each directly observed boundary's
contribution to the `7.046353169355859e-04 K m` JIT content RMS, under full
production JIT and complete eager execution with a production plant.  Name the
first non-bit compiled statement only if its direct operand is captured; if a
boundary is inherited, continue upstream.  No acquisition is presently
needed.

