# Round 179 receipt — developed `ldf_slp` causal-input record

**Status: STOPPED_FOR_RECORD.**  Round 178 named `uslp` as the first returned
non-bit developed tracer-LDF row, but its admitted record lacks the causal
inputs and ordered intermediates before that write.  Round 179 preregistered,
built, and syntax-proved one passive NEMO acquisition for those rows.  NEMO
was not run in the sandbox, no scientific row was measured, and no production
physics, configuration, carried state, restart schema, card default, or
trajectory changed.

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round179.md`,
commits `bf96b0ceb` and `b00b97780`.  Acquisition implementation commit:
`1f9e755e1`.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round179/`.

## First open boundary and source order

The active compiled step computes `rhd` and calls
`ldf_slp(kstp,rhd,rn2b,Nbb,Nbb)` at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/stprk3.f90:173-180`.
The routine first reads `nmln`, live depth, and the mixed-layer inverse depths
at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:156-180`,
then forms the descending horizontal and vertical density gradients at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:183-213`.
The U/V gradient, limiter, mixed-layer recurrence, raw slope, and Shapiro
write follow at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:222-268`.

Round 177 records only the returned slopes and downstream tensor/flux rows.
It has none of `prd`, `pn2`, `nmln`, `zgru`, `zdzr`, the limiter arms, or the
mixed-layer recurrence.  Therefore Round 179 does not attribute the existing
`uslp` difference to an unobserved statement.

## Acquisition contract

The new target is `GYRE_OMIP_L2_P3_SM_R179SLPWALK`; its output root is
`round179/oracle_slope_walk`.  The run script clones `GYRE_PISCES`, copies the
admitted Round-132 daily-restart EXP00, MY_SRC, and preprocessor card file by
file, and adds only `ldfslp_round179.patch`.  The source patch removes zero
lines and writes one step-1081 record after `lbc_lnk`.

The fixed 7,324,076-byte record contains:

* 31 three-dimensional fp64 rows: direct density/buoyancy/masks, live face
  thicknesses, horizontal and vertical density gradients, raw and limited
  denominators, integer selectors represented exactly in fp64, live depths,
  raw U/V slopes, recurrence values before and after each descending level,
  and filtered `uslp`/`vslp`;
* 17 two-dimensional fp64 rows: QCO ratios, mixed-layer depths and inverses,
  metrics, `nmln`/bottom indices, and the U/V mixed-layer indices; and
* three 31-level reference ladders.

The header literally registers `Kbb=1`, `Kmm=1`, dimensions `36x26x31`,
`jpkm1=30`, counts `31/17/3`, and schema version 179.  Admission is fail
closed on magic, header, byte count, finite values, the 18,000-cell census,
integer-valued selectors, non-vacuous fields, current-commit stamp, STOP 0,
and both restart comparisons.  The candidate step-1080 and step-1440 restarts
must each be byte-identical to the uninstrumented Round-132 daily restart.

The binding in-run calibration reconstructs `uslp` and `vslp` from the new
record's own raw slopes and masks using the compiled Shapiro association at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:262-268`.
The Round-177 returned slopes are also compared, but that cross-build result
is informational under the developed-record admission policy; it cannot
refuse an otherwise restart-identical record.

## Frozen predictions and verdicts

No causal record exists yet, so scientific predictions remain **UNMEASURED**:

| preregistered prediction | Round-179 verdict |
|---|---|
| first non-bit causal row is developed `nmln` | UNMEASURED — record required |
| `prd` is non-bit | UNMEASURED — record required |
| `pn2` is non-bit | UNMEASURED — record required |
| production JIT and eager name the same causal boundary | UNMEASURED — Round 180 |

The mechanical predictions are **CONFIRMED**: the source card applies with
zero removals, the GYRE preprocessor state compiles it, and the layout plant
removes the registered post-recurrence row, prints
`STATUS PLANT-FIRED: source-layout`, and exits 69.

## Trajectory and Rule-12 disposition

There is no candidate patch, so Decisions 43/45/55/59 and the Rule-12 ladder
do not run.  The unchanged execution census is explicit:

| card or gate | executes future slope statement | Round-179 measurement |
|---|---:|---|
| GYRE NEMO-identity | yes | no production difference; acquisition only |
| DINO | yes, shared-risk | UNMEASURED until a candidate exists |
| generic NEMO-GYRE | recipe-dependent | no production difference |
| LOCK_EXCHANGE / OVERFLOW | no changed statement | tanks not run |
| ORCA2 ocean-only | possible shared operator | UNMEASURED-with-spec; no candidate |

The immutable Round-163 trajectory remains: first over bar kt=3; kt2 U/V
`8.330180021824198e-17 / 9.714451465470120e-17`; kt3 T/S
`4.9400710722e-7 / 4.0086298725e-8`; day-30 T RMS
`6.57257437477e-5 K`; day-240 `1.64483607012e-2 K`; and day-360
`1.12256600186e-2 K`.  These values are carried context, not remeasurement.

## Independent review

The required read-only `codex exec` pass was attempted against the committed
diff.  It produced no scientific verdict because the in-process app-server
could not initialise in the read-only sandbox.  Its terminal disposition is
quoted verbatim:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
>
> codex_exit_code=1

Therefore: **independent review unavailable in-sandbox**.  This is not a SHIP,
HOLD, or DO NOT SHIP verdict.

## Mechanical gates and evidence

The source preflight reports:

> `SYNTAX_PROOF_PASS ldfslp.f90`
>
> `ROUND179_SLOPE_WALK_PREFLIGHT_READY`

The record-magic plant is implemented but cannot execute until the operator
produces the record.  Round 180 must require its nonzero `STATUS PLANT-FIRED`
result before parsing scientific rows.

| artifact | SHA-256 |
|---|---|
| `preflight.log` | `23fff4b4d369ff207fd1820cb732f679f85e2a7a6b6462d926092dde0ae12872` |
| `layout_plant.log` | `831c68020af81934d5c23f7fa3d311cc859ede03422abc1b86cbbaaadecad29d` |
| `codex_review.log` | `626acf42958926c40c748abdf9a45bd51d2418c34d7c471ca5022ecf845f6573` |

Choices made: no scientific or configuration choice.  The end step and
restart cadence reproduce the admitted Round-132 comparison window and exist
only to prove observer passivity.

## OPEN — Round 180

The operator runs
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round179_slope_walk/run.sh`.
Round 180 first runs `--admit-existing` and `--plant-admission`.  If both
restarts are byte-identical and the in-run Shapiro reconstruction is exact,
extend `developed_tracer_ldf_statement_walk` itself to parse and score every
new row under the complete production JIT step and complete eager step.  Name
the first non-bit causal operand or statement before `uslp`, retain all
refuted preregistered predictions, and fire a production-step one-ULP plant on
that boundary.  Do not change the live-thickness route or downstream tensor
statements while this earlier slope boundary remains open.  A one-variable
NEMO-exact candidate is eligible only after the full trajectory/card/tank gate.
