# NEMO-testcases L2 GYRE round 114 receipt: U transport operands

Date: 2026-09-18

Incoming lane tip: `d03b47da8`

Preregistration commit: `21f7f602c`

Authoritative measurement commit: `e7ec8266f`

Round status: **HELD — no physics or configuration changed; the first
non-bit U factor is the Kmm `1+r3u` face-geometry factor, while replacing
only recorded `e3u(Kmm)` slightly worsens local kt3 T and does not identify
whether half-step SSH or the geometry statement owns the error**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round114/`

## Outcome first

Round 114 reproduces the Round-113 complete-transport endpoint, separates U,
V, and W at their production-jitted FCT consumer, and then walks the U
producer factors in compiled order.  The result is a cancelling transport
triplet, not an individually attributable U, V, or W owner:

| production-JIT arm | T `adv_up1` max | local kt3 T max | ratio to baseline kt3 T |
|---|---:|---:|---:|
| ordinary live triplet | `6.545655547452618e-11 s-1` | `8.600420500215478e-7 K` | 1.000 |
| NEMO-directed U only | `1.2933522527552834e-9 s-1` | `1.854705174153537e-5 K` | 21.565x worse |
| NEMO-directed V only | `2.0849204878401915e-9 s-1` | `2.943318141390705e-5 K` | 34.223x worse |
| NEMO W only | `2.0937752209624375e-9 s-1` | `3.0150479254587026e-5 K` | 35.057x worse |
| complete NEMO-directed U/V/W | `1.632720658269341e-16 s-1` | `5.763797261693071e-8 K` | 14.922x better |

The U/V consumer arms are called “NEMO-directed,” not BIT substitutions:
the current metric-free representation divides recorded `zFu/zFv` by the
metric and production rematerializes them.  U therefore retains 1,999 unequal
faces at `3.637978807091713e-12`, V retains 1,953 at
`7.275957614183426e-12`, and W is BIT.  This is the already-held Round-112
association debt.  It does not invalidate the sign of the single-family
result, but it forbids an individual ownership claim.  The complete triplet
exactly reproduces Round 113's direct and kt3 endpoints.

Inside U, static `e2u` is BIT on all 580 consumed columns.  The first non-bit
subexpression is the Kmm `1+r3u` factor: 580/580 columns differ, maximum
`7.552691805301492e-11`.  The resulting recorded `e3u(Kmm)` differs in all
17,400 consumed faces, maximum `2.271167431899812e-8 m`.  Later inputs are
also non-bit: `uu(Kmm)` is 17,400/17,400 at
`7.453550767597909e-6 m s-1`; final `un_adv` is 580/580 at
`1.2029895814569258e-4 m2 s-1`; carried `uu_b(Kmm)` is 580/580 at
`5.039835017785037e-8 m s-1`; and `zFu` is 17,400/17,400 at
`8.109319272585253 m3 s-1`.

Replacing only the directly recorded `e3u(Kmm)` makes that row BIT but does
not improve a downstream result:

| row | ordinary | `e3u(Kmm)` only | disposition |
|---|---:|---:|---|
| U `zFu` max | `8.109319272585253` | `8.109319313995456` | worse by `4.1410203266423196e-8` |
| T `adv_up1` max | `6.545655547452618e-11` | `6.545655698139779e-11` | worse by `1.5068716131654003e-18` |
| local kt3 T max | `8.600420500215478e-7 K` | `8.600420713378298e-7 K` | worse by `2.1316282072803006e-14 K` |
| local kt3 S max | `6.979443156751586e-8 psu` | same | no maximum change |

This is a diagnostic oracle substitution, not an implementation candidate.
It neither supplies independent model execution nor distinguishes whether the
non-bit face ratio is caused by the half-step SSH handed to the routine or by
the routine's arithmetic.  No Decision-43 ladder/month arm was therefore
created, and the held Round-112 FCT patch was not applied.

## Prediction ledger

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round114.md`.

| prediction | disposition | evidence |
|---|---|---|
| Round-46/77 and Round-111 records admit without inferred fields | **CONFIRMED** | All registered digests, producers, headers, extents, and admissions pass; six owned stage fields are BIT across the Round-46 and Round-77 copies. |
| Each U/V/W single arm changes T `adv_up1`, no single arm closes both tracers | **CONFIRMED** | All three hashes move; every T and S `adv_up1` row retains 18,000 unequal wet cells. |
| U alone reduces kt3 T by less than 50% | **REFUTED** | It does not reduce kt3 T; it worsens the maximum 21.565x. |
| Complete triplet reproduces Round 113 | **CONFIRMED** | T `adv_up1` is exactly `1.632720658269341e-16`; local kt3 T is exactly `5.763797261693071e-8 K`. |
| `e2u` is BIT and the first non-bit family is Kmm `r3u/e3u` geometry | **CONFIRMED** | `e2u` is 0/580 unequal; `1+r3u` is 580/580 and `e3u(Kmm)` is 17,400/17,400 unequal. |
| Thickness-only substitution improves `zFu` and T `adv_up1` | **REFUTED** | Both maxima worsen by the values in the outcome table. |
| Thickness-only substitution changes kt3 T by less than 10% | **CONFIRMED in magnitude, wrong in sign** | The ratio is `1.0000000247851626`; the maximum worsens rather than improves. |
| Round is diagnostic and no candidate advances | **CONFIRMED** | No production statement, public option, or carried state changed. |

The preregistration's “inherited upstream” expectation is **not established**.
The first boundary is established; its owner remains the discriminating
measurement in OPEN.  This receipt does not silently promote correlation
with Round 113's non-bit Kmm tracer thickness into an SSH verdict.

## Resolved configuration and record admission

Before any authoritative v2 measurement, the gate printed and fail-closed on
this resolved configuration:

```text
ROUND114 RESOLVED CONFIG {"adaptive_implicit_vertadv": false, "barotropic_solver": "explicit_substep", "card": "GYRE-zco", "gm_redi_enabled": true, "nemo_stage_mean_imposition": true, "outer_integrator": "forward_euler", "tracer_advection": "fct2", "tracer_time_integrator": "rk3_ws"}
```

The authoritative artifact is
`round114/u_transport_walk/u_transport_walk_v2.json`, stamped at clean commit
`e7ec8266fefa2c8ca136e0a7740bf5ca2a83dd57`.  It admits:

| record/control | admitted value |
|---|---|
| Round-77 producer | `6c0fe440340c1c1c7ad16d8cbf547d93264bee26` |
| Round-77 admission SHA-256 | `cddd662d6380d7aa053bb4cfc20fbcb481a9da43009200dcbaa359e8c4e2f11d` |
| Round-77 kt2/stage-3 record SHA-256 | `11da7f370ec495181adc80c7fd9daf3300b245ea8db646085122a8f46b2007e8` |
| Round-46 baseline stage record SHA-256 | `a26121247da49d421da6ce8489627dd0b8302d10bdcd4fd1becf7752eb88524f` |
| Round-77 kt2 advective-mean SHA-256 | `412f917a8652002beef2db22af2ef1a916171173b10fc0fb14b62fd6077d95e2` |
| inherited admission census | 45 byte-identical, 24 classified-changed, 281 admitted halo values, verdict PASS |

The directly consumed owned `e2u`, `e3u_Kmm`, `r3u_Kmm`, `u_Kmm`, `umask`,
and `uu_b_Kmm` fields are BIT between the Round-46 record and its admitted
Round-77 descendant.  The compiled stage writer stores `uu(Kmm)`, `r3u(Kmm)`,
and the literal expanded `e3u(Kmm)` at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:525-547`.
The separate final external mean comes from the admitted stream whose
compiled accumulator and write are
`GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90:588-606`.
The Round-111 `zFu` target remains the passive record admitted in Round 111.

The first attempted command used an incorrectly expanded measurement commit
and refused before execution with `commit stamp mismatch`; that refusal is
preserved in `u_transport_walk/u_transport_walk.log`.  The first correctly
stamped run and plant are retained, but v2 is authoritative because it adds
the pre-run configuration print.  All numeric mode, split, factor, kt3, and
hash rows are bit-identical between the two correct runs; only the ownership
label was sharpened from the expanded `e3u` result to its earlier `1+r3u`
factor.

## Compiled statement and first non-bit boundary

The acquired build initializes `n_baro_upd` to the hybrid arm at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:52-54`.
That active branch forms `zub` from final `un_adv`, inverse Kmm depth, and
carried `uu_b(Kmm)` at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:281-292`.
The next written U statement is

`e2u * (e3u_3d * (1+r3u(Kmm)*umask)) * (uu(Kmm)+zub*umask)`

at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:295-296`.
The same triplet derives W at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:326-346`
and reaches tracer advection at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:860`.

Within that written U expression, `e2u` is BIT and `1+r3u(Kmm)*umask` is the
first non-bit value.  NEMO's compiled geometry routine obtains `r3u` from
the supplied SSH at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domqco.f90:266-268`.
For stage 3, Kmm is the preceding stage's N+1/2 slot; the compiled stage
schedule and half-level assignments are at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:198-242`.

Those citations name the next discriminating boundary; they do not yet say
which side is wrong.  Round 115 must drive the compiled geometry statement
from the directly recorded NEMO half-step SSH and static metric/depth inputs.
If its `r3u` is BIT, the own-chain half-step SSH owns the difference.  If it
is still non-bit, the geometry transcription owns it.  No downstream U/V/W
or FCT patch may be inferred before that test.

## Production mode table and plant

Rows below are consumed cells unequal / maximum absolute difference:

| U boundary | production JIT | production eager | isolated-closure JIT |
|---|---:|---:|---:|
| `e2u` | 0/580 / `0` | same | same |
| `1+r3u(Kmm)` | 580/580 / `7.552691805301492e-11` | same | same |
| `e3u(Kmm)` | 17,400/17,400 / `2.271167431899812e-8` | same | same |
| `uu(Kmm)` | 17,400/17,400 / `7.453550767597909e-6` | `7.453550731688265e-6` | production-JIT value |
| `un_adv` | 580/580 / `1.2029895814569258e-4` | `1.2029895814613667e-4` | production-JIT value |
| `uu_b(Kmm)` | 580/580 / `5.039835017785037e-8` | `5.039835017806721e-8` | production-JIT value |
| `zFu` | 17,400/17,400 / `8.109319272585253` | `8.10931923351336` | production-JIT value |

The isolated closure is explicitly not labelled production.  Its `zub`,
corrected U, and `zFu` arrays are nevertheless BIT against the values captured
inside the complete production-JIT step, so this particular statement shows
no isolated-versus-production fusion split.  Eager remains secondary.

The authoritative plant artifact is
`round114/u_transport_walk/u_transport_walk_v2_plant.json`.  It changes the
recorded `e3u_Kmm[1,2,0]` word from bits `4621821095688030040` to
`4621821095688030041`.  The consumed `e3u`, materialized `zFu`, and T
`adv_up1` hashes all change.  The gate prints `STATUS PLANT-FIRED`; the
wrapper and `u_transport_walk_v2_plant.exit` both report exit `1`.

## Decision-43 headline, cards, and scope

No production physics changed, so no trajectory arm exists.  The immutable
Round-110 landing remains before and after:

| required row | before | after | disposition |
|---|---:|---:|---|
| kt2 T | `1.4210854715202004e-14` | same | AT-BAR |
| kt2 S | `2.1316282072803006e-14` | same | AT-BAR |
| kt2 U | `2.7377110452773967e-12` | same | first-over-bar DEBT |
| kt2 V | `3.2849219221489645e-12` | same | first-over-bar DEBT |
| kt3 T | `8.600419718618468e-7` | same | DEBT |
| kt3 S | `6.979441735666114e-8` | same | DEBT |
| day-30 T RMS | `6.890484901489568e-5 K` | same | no production candidate |

GYRE-zco and the generic NEMO-GYRE recipe share the WS stage transport
implementation.  They cannot move because this round changes only a private
diagnostic gate.  The DINO recipes use their Euler tracer lane and likewise
cannot move; no DINO number is claimed.  LOCK_EXCHANGE and OVERFLOW are not
used to generalize the GYRE operand result.  Any future production change
must derive its executing-card set from resolved recipes and measure every
executing certified card.

ORCA2 remains **UNMEASURED-WITH-SPEC**: resolve its integrator and active
transport statement, acquire or admit its half-step SSH/r3 and U/V/W
operands, run this factor table under its production closure, and then run
its certified trajectory.  No GYRE result is transferred to it.

## Review, citations, and tests

Independent review verdict: **independent review unavailable in-sandbox**.
The required command was attempted with `codex exec --sandbox read-only` and
exited 1 before a reviewer started.  Its verbatim terminal finding was:

```text
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

It emitted no `DO NOT SHIP` verdict.  The complete log is
`round114/review/codex_review.log`; the unavailable-review fallback is not an
author self-review presented as independent evidence.

The citation gate was run from `## Outcome first` at clean commit
`2b83e6119c55a544962f62abb97ecc9a7943f3b4`.  It found all nine citations,
no unmapped citation, no failing map entry, and reported `PASS`.  Its shifted
plant moved the compiled geometry range cited above by two lines, identified
the real first endpoint at line 266 rather than planted line 268, reported
`SYMBOL-NOT-AT-LINE`, and exited 1.  The first audit had separately caught an
ambiguous repeated `SELECT CASE` anchor and a writer endpoint one line short;
both were repaired by pinning the exact source occurrences without widening
or weakening either range.

Focused CPU summaries, all with `JAX_ENABLE_X64=1`, were:

```text
31 passed in 48.80s
119 passed in 373.93s (0:06:13)
```

The first line is both modified gate-test modules.  The second is the exact
inherited four-file push gate: receipt citations, TKE NEMO terms, NEMO recipe,
and real freshwater closure.

The required one-piece
`-n 12 tests/ocean/fidelity tests/ocean/unit` run was attempted once and
collected 8,181 cases.  It reached 96%, but ten workers aborted in JAX
compilation and xdist repeatedly replaced them.  After the tenth dead worker,
the controller made no progress and emitted neither a terminal summary nor a
JUnit file; one interrupt was required, and the command exited 130.  Therefore
there is no summary line to quote and no mechanically complete failing-set
diff for that corrupted run; this receipt does not invent either one.  Its
last usable progress text was:

```text
........................................................................ [ 96%]
......................
```

Nine named crash nodes could be recovered from the interleaved traces; the
tenth trace was truncated before its test frame.  Replaying all nine named
nodes in one fresh serial process expanded to twelve parameterized cases and
reported:

```text
4 failed, 8 passed in 437.52s (0:07:17)
```

All eight non-advection crash nodes pass.  The four failures are exactly the
`ppm_fct`, `tvd`, `superbee`, and `dst3` parameters of
`test_model_rollout_grads_finite_f32`; all four IDs occur verbatim in the
pinned 87-ID incoming baseline.  Thus the recovery failing-set diff contains
zero new IDs.  Round 114 changes no production package file, and every changed
executable test/gate path is covered by the 31/31 focused pass.  Complete
ordinary, plant, push-gate, interrupted-run, and serial-replay evidence is
under `round114/{citations,tests}/`.

## ASKED / UNASKED

ASKED and completed: frozen preregistration; admitted-record reuse; resolved
configuration print; production-JIT U/V/W split; compiled-order U factor
table; production eager and isolated-JIT labels; first non-bit boundary;
one-factor full-step substitution; direct `zFu`, `adv_up1`, and kt3 results;
production-JIT ULP plant; compiled-source citations; explicit DINO/card risk;
ORCA2 specification; and separate read-only review attempt.

UNASKED and not done: no NEMO source was modified; `makenemo` and `mpirun`
were not run; no oracle was acquired; no production physics, configuration,
default, coefficient, stabilizer, carried state, restart schema, year harness,
reconciliation gate, freshwater pair, or #1484 guard changed; no held patch
was applied; no ladder or month run was spent on a non-candidate.

## OPEN for round 115

1. Stay at the first record-backed non-bit U factor.  Extend the same
   production stage gate; do not create another harness.  At kt2 stage 2,
   drive the compiled `r3u` geometry statement at
   `GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domqco.f90:266-268`
   from NEMO's recorded N+1 SSH and static `e1e2t`, `r1_hu_0`, and
   `r1_e1e2u`, under production JIT with eager and isolated labels.
2. First reproduce the ordinary 580/580 `1+r3u` row and its
   `7.552691805301492e-11` maximum.  Then score input SSH and each written
   multiply/add boundary.  The discriminating result is: BIT geometry given
   NEMO SSH means own-chain half-step SSH owns; non-BIT geometry given NEMO
   SSH means the geometry transcription owns.
3. Reuse the Round-46 kt2 stage-2/stage-3 records and existing admitted r3
   tools.  Request acquisition only if a directly consumed operand is truly
   absent; never invert `e3u` or `zFu` to manufacture it.
4. Keep the U/V/W cancellation explicit.  Do not call U, V, or W an
   individual magnitude owner from the single-family arms, and do not revive
   the held Round-112 metric-FCT patch.
5. Only an implementable statement with a production-JIT local proof advances
   to Decision 43: same-base ladder/month, no earlier first-over-bar, no kt1
   AT-BAR loss, every moved row registered, and every recipe-derived executing
   card measured.  Keep ORCA2 UNMEASURED until its specification above runs.
