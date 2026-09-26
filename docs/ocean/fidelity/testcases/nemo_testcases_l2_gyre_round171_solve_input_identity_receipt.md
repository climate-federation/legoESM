# Round 171 receipt — developed tracer-solve input identity

**Status: HELD.**  No production physics, configuration, carried state,
restart schema, card default, or immutable before arm changed.  The required
day-240 ranking of `e3t(Kaa)` versus the already-formed tracer content is
withheld: feeding the production-jitted solve its own `e3t(Kaa)` through the
new private slot changes the step-1081 state by `7,538` bytes.  The moved
certified cells are T `3,813`, S `3,691`, U `0`, V `0`, and SSH `0`.

The identity failure reproduces when the e3t slot is isolated; the content
slot is not needed to trigger it.  Therefore no directed e3t/content arm is a
controlled comparison, no day-240 magnitude owner is named, and no candidate
is eligible for a Rule-12 or Decision-43/45/55/59 landing gate.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round171/`.  The frozen
preregistration is commit `c1d1f329e`; the first-class fail-closed result is at
clean commit `324c0d816`.

The immutable production headline remains Round 163: kt2 T/S/U/V AT-BAR,
first-over-bar kt3, day-30 T RMS `6.572574374770603e-05` K, day-240
`1.644836070117868e-02` K, and day-360 `1.122566001855131e-02` K.  This round
runs no accepted from-rest candidate, so none of those rows moves.

## Compiled statements and intended comparison

In the record build's executing adaptive-implicit branch, NEMO forms the two
off-diagonal terms from `e3w(Kmm)`, then consumes `e3t(Kaa)` in the diagonal at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:468-474`.
It forms the before-content plus now-tendency RHS at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:549-567` and consumes
the matrix/content in the backward recurrence at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:577-582`.

The existing private vertical-solve tuple was extended rather than adding a
second harness.  Its fifth slot replaces only the tracer matrix cell weight;
its sixth replaces only temperature content.  Both execute through
`LatLonCGridOceanModel.step -> self._step_jitted`.  Ordinary calls leave the
tuple absent, so no card executes either diagnostic slot.

## Admission and calibration

The Round-132 daily record is admitted with its 360-file SHA stamp: 360 of 360
daily boundaries present, steps 6 through 2160, with all 12 monthly overlaps
bit-identical.  The Round-123 process and Round-125 vertical records both pass
their existing admissions at the measurement commit.  The NEMO literal
rebuild at step 1081 remains BIT for every recorded boundary:

| boundary | cells unequal |
|---|---:|
| formed coefficient `zwt_mix` | `0` |
| lower `zwi` | `0` |
| diagonal `zwd` | `0` |
| upper `zws` | `0` |
| LU diagonal `zwt_lu` | `0` |
| temperature RHS `rhs_T` | `0` |
| forward recurrence `fwd_T` | `0` |
| solved temperature `sol_T` | `0` |

The first audit attempt omitted the acquisition stamp.  It was preserved as
`daily_record_audit_missing_stamp.*`; the developed gate refused it before
model execution.  The corrected audit names and verifies the stamp before any
scientific arm runs.

## Frozen predictions and verdicts

| preregistered item | verdict | evidence |
|---|---|---|
| free and exact-K/e3w baselines reproduce Round 169 | **UNMEASURED** | identity check runs first and refuses the instrument |
| e3t identity arm moves zero state bytes | **REFUTED** | `7,538` bytes move at step 1081 |
| content identity arm moves zero state bytes | **UNMEASURED** | fail-closed stop after the earlier e3t identity failure |
| directed e3t/content endpoints are bit-exact | **WITHHELD** | an identity-valued input already changes the compiled step |
| neither input removes half the `1.241262968697578e-03` K remainder | **UNMEASURED** | no controlled day-240 trajectory exists |
| e3t removes more day-240 RMS than content | **UNMEASURED** | no ranking exists |
| both scale plants reach day-240 temperature and exit nonzero | **UNMEASURED** | plants are downstream of the failed identity prerequisite |

The original combined fifth-plus-sixth-slot identity run also moved `7,538`
bytes.  That result did not identify which slot owned the movement, so it was
not accepted as attribution.  The slots were then isolated before rerunning.
The isolated e3t slot reproduces the same byte count and field split:

| field | wet cells moved |
|---|---:|
| T | `3,813` |
| S | `3,691` |
| U | `0` |
| V | `0` |
| SSH | `0` |

This is a production-JIT result, not an isolated-closure result.  The values
fed through the fifth slot are a copied view of the model's own live
`e3t_after`; the only difference is materialising it as a full-step input.
That changes XLA fusion/rounding of the tracer solve, exactly the failure mode
notes L/L-amend require the identity arm to detect.  The gate writes
`sensitivity: null`, prints `STATUS REFUTED`, and exits `1`; it cannot emit a
ranking after this failure.

## Scope and landing verdict

The package diff adds only private diagnostic tuple slots guarded by
`_vertical_K_test_override is not None`.  GYRE production, generic GYRE,
DINO, LOCK_EXCHANGE, OVERFLOW, and ORCA2 pass `None`; no card, recipe, default,
or physical statement is changed.  The scientific comparison is refused
before a candidate exists, so the 70-row ladder, month/year trajectories,
DINO shared-statement row, tanks, generic card, and ORCA2 spec are not invoked
and have zero registered movement this round.

No first non-bit NEMO statement is named.  The compiled matrix and content
statements above remain the intended boundaries, but their day-240 sensitivity
has not been measured by a passive production closure.  Calling either an
owner would violate the preregistration and the production-JIT rule.

## Review, citations, and tests

The required separate review command was run with `codex exec --sandbox
read-only`.  Its exact disposition is:

> independent review unavailable in-sandbox

The terminal error was `failed to initialize in-process app-server client:
Read-only file system`; it emitted no SHIP or DO NOT SHIP verdict.

Test and citation results are recorded at the final committed tip below.

## OPEN — round 172

1. Keep Round 163 as the immutable production arm.  Do not cite or land an
   e3t/content sensitivity from Round 171; both directed values are absent.
2. Replace the tuple-arity seam with one fixed-shape production-closure input
   contract whose disabled and identity forms compile to the same graph.
   Its first gate is byte equality against the ordinary production step at
   step 1081 and over steps 1081--1440.  Any moved byte repeats this round's
   refutation and forbids substitution.
3. Only after that identity gate is BIT may the frozen e3t-versus-content
   ranking be rerun, with separate consumed-endpoint plants.  If no fixed-shape
   seam can be passive under the full step, switch to a preregistered NEMO-side
   paired sensitivity record rather than an isolated JAX closure.
4. No NEMO acquisition, configuration decision, or carried-state change is
   requested yet.
