# Preregistration: NEMO-testcases L2 GYRE round 85 momentum bundle

Date: 2026-09-13. Frozen at legoESM `b8059db7f608` after reading the Round
83--84 receipts, Decisions 37--38, the held Round 47/79/82 evidence, the
immutable Decision-36 ladder/day arms, and the executing compiled sources,
but before reapplying a held production candidate or running a new scientific
comparison.

## Magnitude rank and bundle authorization

Round 84 makes HPG bit-exact, places the first non-bit cumulative boundary at
LDF (`2.5292467120726215e-14` U / `3.502735092670824e-14` V), and places the
first magnitude-bearing boundary at the ZAD addition
(`1.9220297482797664e-09` U / `1.966061294804274e-09` V incremental maxima).
LDF stays explicit fidelity debt, but it is roughly five orders below ZAD and
does not outrank the ZAD walk under the user's magnitude rule.

Decision 38 authorizes one preregistered candidate made from separately
proved NEMO-exact statements. This round bundles exactly:

1. the already-present Decision-37 absolute barotropic-history pair, including
   the loud restart migration/failure contract from Round 79;
2. Round 82's tracer-cell bottom-drag materialization and window-entry U-face
   inverse-depth association; and
3. Round 47's stage-specific ZAD operand association for RK stages 1--3.

No TKE, tracer-content, tracer-coefficient, LDF, HPG, freshwater,
reconciliation, or #1484 candidate is included. If this three-member bundle
fails the certified ladder, Decision 38 permits a ladder-only bisection into
`histories + drag` and `ZAD`; no post-hoc member substitution is allowed.

## Compiled statements and local-exactness admission

The acquired GYRE program stores the tracer-cell drag result before the face
average (`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/zdfdrg.f90:229-236`),
forms the window-entry inverse depths in their own assignments
(`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:357-375`), and
consumes coefficient times current velocity times that inverse in written
order (`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:701-707`).

The same executing external-mode program consumes current/b/bb absolute
histories in the AB3 midpoint (`dynspg_ts.f90:511-520`), rotates them after
each substep (`dynspg_ts.f90:834-846`), and reads/writes the six absolute
arrays (`dynspg_ts.f90:1049-1077`). Decision 37 already authorized the paired
carried-state semantics and loud migration failure.

The admitted Round-64 compiled ZAD initializes its carried workspace, forms
the W transport and Kmm velocity-difference association, updates every
interior level, and then updates the bottom level
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzad.f90:102-138`). The Round-47
literal WZV/KEG/ZAD replays were bit-exact at kt=1/2 and stages 1/2/3.

Before the bundle may reach the ladder, current-tip reproof must independently
retain: absolute-history midpoint rows bit-exact on the admitted Round-77/81
record; both drag coefficients and entry inverse depths bit-exact on the
Round-81 record; and every literal ZAD replay bit-exact on the Round-64 record.
Each proof must retain its existing non-vacuity plant. Any unequal local row,
record-admission failure, or non-firing plant rejects the member and stops the
bundle before trajectory scoring.

## Frozen trajectory predictions and falsifiers

The immutable before arm is
`decision36_nemo_face_shear/after_kt1_10.json`. Its registered rows are:

- kt2 U `2.7478404751243857e-12`, V `3.305560306813421e-12`;
- kt3 T `1.6275114177588534e-04 K`, S `6.327755180279837e-06 g/kg`; and
- day-30 T RMS `1.2397568272314757e-02 K`.

The bundle is predicted to reproduce the held ZAD movement at the first
over-bar boundary: kt2 U `2.7377110452773967e-12` and V
`3.284922138989399e-12`, both toward the bar. It is also predicted to expose
the previously measured compensating-error cost at kt3: T approximately
`3.722344652268031e-04 K` and S approximately
`3.683245441046983e-05 g/kg`. Those kt3 values are predictions, not permission
to omit their moved-cell census or the day-30 measurement. Day-30 direction is
not predicted because no historical production-member arm contains this exact
three-member bundle.

The bundle passes the short Rule-12 ladder only if every local proof is exact,
all kt1 rows remain bit-identical, no AT-BAR row leaves the bar, first-over-bar
is not earlier than kt2 U/V, both kt2 U and V move at the bit level toward the
bar, and every moved/worsened row is registered. The numeric predictions are
REFUTED if their values do not reproduce; the candidate is rejected if either
kt2 row is unchanged/worse, any AT-BAR row becomes DEBT, or first-over-bar is
earlier. A passing bundle proceeds to the fixed days 1--30 run even when later
rows worsen, exactly as Decision 38 specifies.

If the bundle fails, the two permitted halves use the same immutable before
arm and the same falsifiers. A half that cannot independently move both kt2
U/V toward the bar is not eligible. No result from one half may be combined
arithmetically with the other.

## Rule-12 cards

| lane | frozen Round-85 disposition |
|---|---|
| GYRE kt=1..10 | Measure the complete three-member bundle, or the two preregistered halves after failure; preserve all AT-BAR rows, forbid an earlier first-over-bar, and register every changed row |
| GYRE days 1..30 | Run the production member only after a ladder pass; score every day and T/S/U/V/SSH row against `decision36_nemo_face_shear/after_day_gap.json` and the unchanged NEMO root |
| LOCK_EXCHANGE-zco | The shared split-explicit histories/drag statements execute, while flux-UP3 does not execute ZAD; after a GYRE pass, measure the recorded kt=1..10 rows and preserve every AT-BAR row |
| OVERFLOW-zps | Its boxcar path does not execute cross-window absolute histories and flux-UP3 does not execute ZAD; after a GYRE pass, measure or prove the shared drag statement inert on the recorded state |
| DINO | **SHARED-STATEMENT RISK:** DINO does not execute WS-RK3 histories/ZAD, but it calls the shared drag producer; 96--98% regional cancellation forbids neutrality inference |
| ORCA2 | **UNMEASURED-WITH-SPEC:** independently align T/S/U/V/SSH, native masks, six histories, tracer/face drag coefficients, entry inverse depth, W/ZAD operands and cumulative momentum boundaries through kt=1..10 in fp64; reject any unequal given-input replay, AT-BAR loss, or earlier first-over-bar |

No configuration/default, selector, coefficient, timestep, stabilizer, year
harness, reconciliation gate, freshwater pair, #1484 guard, held manifest,
NEMO source, or NEMO executable changes. The only carried-state choice is the
already-authorized Decision-37 absolute-history representation.
