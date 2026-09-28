# NEMO-testcases L2 GYRE round 116 receipt: external-mode walk

Date: 2026-09-19

Incoming lane tip: `bb9c586891df9cb089be3675b3a1de38e160ee2f`

Preregistration commits:
`5bce1e568c62fc923aabff786f992f3b7921c142` and the premeasurement control
correction `2dde339b98a751756113f949e31e8e929a4ca40a`

Authoritative measurement commit:
`81bffa076f6a79711bc66463c6c70dd1d0b7ead4`

Round status: **HELD — no physics or configuration changed; the production-JIT
external loop is bit-exact through its substep-1 drag trend and first differs
in the imported frozen slow-U forcing, while replacing only final SSH with
NEMO's value worsens the cancelling stage-3 tracer state**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round116/`

## Outcome first

The complete production-JIT external-step walk names the first non-bit
boundary at substep 1 as `slow_u`: all 580 wet U cells differ, with maximum
`1.0529650291768787e-11 m s-2`.  Its V twin is 570/570 at
`1.0765559917925099e-11 m s-2`.  Every prior substep-1 boundary is BIT,
including histories, midpoint velocity/SSH, face depth, transport,
continuity, backward-interpolated SSH, pressure gradient, Coriolis, drag
coefficient, inverse depth, and the completed `trd_u`/`trd_v`.

This is an imported boundary, not an external-loop arithmetic owner.  The
compiled program first copies `Ue_rhs`/`Ve_rhs` and the drag coefficients at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:286-291`, then
constructs and removes the Kmm two-dimensional Coriolis trend at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:298-321`.
This round scores the completed frozen `zu_frc`/`zv_frc` result, not its
unrecorded current-tip `Ue_rhs`/`Ve_rhs` input separately.  Ownership therefore
moves to that producer walk; changing the external update would be
post-boundary and is forbidden by the evidence.

The completed weighted N+1 SSH reproduces Round 115 exactly: 600/600 wet
cells differ from NEMO, maximum `7.072560112143626e-7 m`.  Replacing only that
SSH at the stage handoff moves every U/V/W transport family and both tracers,
but closes none.  Local kt3 T worsens from `8.600420500215478e-7` to
`1.07454589937106e-6 K` (1.24941x); local kt3 S worsens from
`6.979443156751586e-8` to `1.720615962597094e-6` (24.6526x).  This is a
measured cancellation, not evidence to fix or preserve the live SSH error.
There is no source-exact candidate and no Decision-43 ladder/month arm.

## Frozen prediction ledger

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round116.md`.

| frozen prediction or falsifier | disposition | evidence |
|---|---|---|
| final production-JIT SSH is 600/600 at `7.072560112143626e-7 m` | **CONFIRMED** | The authoritative row reproduces count and maximum exactly. |
| eager final SSH is 600/600 and within `1e-12 m` of JIT | **CONFIRMED** | Eager maximum is `7.072560112154468e-7 m`, differing from the JIT maximum by `1.0842e-18 m`. |
| production JIT first differs at substep-1 `slow_u` or `slow_v`; U maximum lies in `[1e-12,2e-11]` | **CONFIRMED** | First is `slow_u`, 580/580 at `1.0529650291768787e-11`. |
| every substep-1 row through `trd_u`/`trd_v` is BIT | **CONFIRMED** | All 38 earlier registered boundaries are BIT. |
| exact SSH substitution moves U/V/W; none becomes BIT | **CONFIRMED** | All five registered stage outputs move; all ordinary and directed rows remain non-BIT. |
| directed local kt3 T remains non-BIT and changes by less than 50% | **CONFIRMED** | It remains non-BIT and worsens 24.9411%. |
| directed local kt3 S remains non-BIT and changes by less than 50% | **REFUTED** | It remains non-BIT but worsens 24.6526x, far beyond 50%.  The failed prediction is retained rather than relabelled. |
| the history ULP becomes the first production-JIT mismatch | **CONFIRMED** | One planted `u_b` word becomes the first row, 1/580 at `6.776263578034403e-21`; the process exits 1. |
| the handoff SSH ULP reaches the shared QCO ratio | **CONFIRMED** | One SSH word and one derived `r3u` word move; the process exits 1. |
| no external-loop candidate is eligible when slow forcing is imported non-BIT | **CONFIRMED** | No production file or configuration changed; ladder/month were not run. |

## Instrument correction and record admission

The first implementation used the existing early-return substep hook and
produced provisional artifacts `production_jit.json` and
`production_eager.json`.  That is not the complete production closure under
the Round-102 JIT discipline.  A fail-closed comparison against the normal
full-stage hook then refused with:

```text
GATE FAILED: external-only and ordinary full-step handoffs differ
```

The refusal is preserved in `production_jit_v3.log`; the preceding v2 attempt
refused only because its expected commit was abbreviated.  No number from an
early-return artifact governs this receipt.

The corrected instrument monkeypatches only the diagnostic call boundary,
captures the real substep trace with `jax.debug.callback` inside the complete
production step, returns the ordinary two-value barotropic result to the
caller, and then compares the complete returned state against an
uninstrumented production step.  The final JIT observer and the separate
live-stage observer are each BIT over 23 leaves and 198,956 cells.  Their five
handoff fields (SSH, external U/V, and U/V transport means) are also BIT.
Thus the measurement is through the production step, not an isolated JIT of
the local equations.  The v4 and v5 science payloads compare byte-for-byte
after removing only their commit stamps and the added verdict field.

The admitted Round-81 stream is producer
`295a42edc9d9f45707a5349097e7a0183f57463c`, SHA-256
`566f3f240d05ce526fc9005c2b4a5112e4e4d90484056d6ec6b8a445e8c6abd6`,
and 10,439,164 bytes.  Admission requires 46/70 inherited records byte
identical, all 24 classified changes, 264 admitted differences, exact
Round-77 U-midpoint identity, fp64 headers and physical EOF.  The writer opens
the stream and writes its native masks at
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:468-477`;
writes histories, extrapolation coefficients, midpoint fields and face depth
at
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:522-572`;
writes flux, divergence and continuity SSH at
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:603-615`;
writes backward interpolation, pressure, Coriolis, drag/inverse depth and
frozen forcing at
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:670-713`;
and writes exchanged velocities plus the post-swap histories at
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:814-849`.

A direct compiled-source diff against the current Round-111 build is
`current_vs_record_dynspg.diff`, SHA-256
`44a1952806d9f51e9839a1207ab04124de437f153faf32567ebb08bb390e9654`.
Every difference over the scored loop is an added declaration, allocation,
condition widened solely around a `WRITE`, or additive record write; the
computational statements are unchanged.

## Compiled execution order

The current compiled loop sets the midpoint coefficients, extrapolates U/V
and SSH, constructs face depth, and forms transports at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:456-531`.
It forms divergence and continuity SSH at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:542-553`,
then backward-interpolates SSH and forms the surface pressure gradients at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:593-607`.
The executing Coriolis call is
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:611`;
the non-wetting drag additions are
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:627-630`;
and the active vector update consumes pressure, completed trend and frozen
forcing in the written association at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:657-667`.
Depth refresh and the exchanged exit follow at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:701-730`.
Finally, the program swaps histories and accumulates weighted U/V/SSH at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:749-780`,
then divides the completed averages at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:797-804`.
These are the compiled statements whose order defines the table below.

## Production walk

The authoritative artifact is `external_walk/production_jit_v5.json`,
SHA-256
`189e10202781fe8efd8b9ebf1edf3ec97c95f0671d3fcd1a9a0709305d8921e2`.
It stamps a clean tree at the measurement commit, CPU, fp64, x64, libm and
production JIT.  The JSON registers all 2,200 cells of the 50-by-44 boundary
table; “cell” here means a table cell containing its own native wet-cell
census, not one ocean grid cell.

At substep 1, the following 38 boundaries are BIT: all three midpoint
coefficients; U/V/SSH entry, one-back and two-back histories; midpoint U/V/SSH;
U/V face depth and transport; SSH forcing, divergence and continuity result;
all four backward coefficients; backward SSH; U/V pressure and Coriolis;
U/V drag coefficient and inverse depth; completed U/V trend; and swapped SSH.
The six non-BIT rows are:

| substep-1 boundary | cells unequal / wet | maximum absolute difference |
|---|---:|---:|
| `slow_u` | 580/580 | `1.0529650291768787e-11` |
| `slow_v` | 570/570 | `1.0765559917925099e-11` |
| `u_exit` | 580/580 | `3.032539284029834e-9` |
| `v_exit` | 570/570 | `3.1004812563623226e-9` |
| `swap_u` | 580/580 | `3.032539284029834e-9` |
| `swap_v` | 570/570 | `3.1004812563623226e-9` |

After substep 1, the prior non-BIT exit is the next substep's first input.
The complete first-non-BIT registry is:

| substep | first boundary | cells unequal / wet | maximum absolute difference |
|---:|---|---:|---:|
| 1 | `slow_u` | 580/580 | `1.0529650291768787e-11` |
| 2 | `u_entry` | 580/580 | `3.032539284029834e-9` |
| 3 | `u_entry` | 580/580 | `5.1234040916522955e-9` |
| 4 | `u_entry` | 580/580 | `7.042448258948059e-9` |
| 5 | `u_entry` | 580/580 | `8.027638608314771e-9` |
| 6 | `u_entry` | 580/580 | `8.990975144050042e-9` |
| 7 | `u_entry` | 580/580 | `9.56159137669841e-9` |
| 8 | `u_entry` | 580/580 | `1.0334965474946857e-8` |
| 9 | `u_entry` | 580/580 | `1.0652836969482753e-8` |
| 10 | `u_entry` | 580/580 | `1.100192403463536e-8` |
| 11 | `u_entry` | 580/580 | `1.1340429315336194e-8` |
| 12 | `u_entry` | 580/580 | `1.2468780637598254e-8` |
| 13 | `u_entry` | 580/580 | `1.3594215753900876e-8` |
| 14 | `u_entry` | 580/580 | `1.478137886352579e-8` |
| 15 | `u_entry` | 580/580 | `1.598914640136206e-8` |
| 16 | `u_entry` | 580/580 | `1.7197046992163678e-8` |
| 17 | `u_entry` | 580/580 | `1.841638078175405e-8` |
| 18 | `u_entry` | 580/580 | `1.96648547947818e-8` |
| 19 | `u_entry` | 580/580 | `2.0963398349475512e-8` |
| 20 | `u_entry` | 580/580 | `2.232916088357531e-8` |
| 21 | `u_entry` | 580/580 | `2.376797089009711e-8` |
| 22 | `u_entry` | 580/580 | `2.5270416928609672e-8` |
| 23 | `u_entry` | 580/580 | `2.6813222158209027e-8` |
| 24 | `u_entry` | 580/580 | `2.8364362343092937e-8` |
| 25 | `u_entry` | 580/580 | `2.9889094105978176e-8` |
| 26 | `u_entry` | 580/580 | `3.135484326317146e-8` |
| 27 | `u_entry` | 580/580 | `3.273446987875784e-8` |
| 28 | `u_entry` | 580/580 | `3.400832563704801e-8` |
| 29 | `u_entry` | 580/580 | `3.516543087744763e-8` |
| 30 | `u_entry` | 580/580 | `3.620370986421456e-8` |
| 31 | `u_entry` | 580/580 | `3.712911238965694e-8` |
| 32 | `u_entry` | 580/580 | `3.795367755309292e-8` |
| 33 | `u_entry` | 580/580 | `3.869290159889266e-8` |
| 34 | `u_entry` | 580/580 | `3.9362943034639344e-8` |
| 35 | `u_entry` | 580/580 | `3.9978220448943363e-8` |
| 36 | `u_entry` | 580/580 | `4.054988266683195e-8` |
| 37 | `u_entry` | 580/580 | `4.1085470640542764e-8` |
| 38 | `u_entry` | 580/580 | `4.158983517950007e-8` |
| 39 | `u_entry` | 580/580 | `4.206704813799426e-8` |
| 40 | `u_entry` | 580/580 | `4.252273811018725e-8` |
| 41 | `u_entry` | 580/580 | `4.296611509451136e-8` |
| 42 | `u_entry` | 580/580 | `4.3411008306191816e-8` |
| 43 | `u_entry` | 580/580 | `4.38755380045364e-8` |
| 44 | `u_entry` | 580/580 | `4.4380491473223344e-8` |
| 45 | `u_entry` | 580/580 | `4.4946911812240864e-8` |
| 46 | `u_entry` | 580/580 | `4.5593655659736596e-8` |
| 47 | `u_entry` | 580/580 | `4.633562249579774e-8` |
| 48 | `u_entry` | 580/580 | `4.718302986761024e-8` |
| 49 | `u_entry` | 580/580 | `4.8141662788376074e-8` |
| 50 | `u_entry` | 580/580 | `4.921366864257901e-8` |

Production eager is secondary.  Its artifact
`external_walk/production_eager_v5.json`, SHA-256
`f8ac32e1bfedce02721a732284b550f5e0dd26fb42b8ffd96fe8fbdf96635d1a`,
first differs at substep-1 `ssh_forcing`: 193/600 at
`6.617444900424222e-24`.  The completed continuity result moves only 6/600
at `5.421010862427522e-20`, while the slow-U maximum is
`1.0529650291765478e-11`.  This eager-only ordering does not override the JIT
owner.

The separately labelled isolated-closure JIT is also non-production.  It
differs in 1,595 continuity-SSH cells at `4.336808689942018e-19`, 14,235
U-midpoint cells at `2.168404344971009e-19`, and 25 completed U-trend cells at
`3.308722450212111e-24`, among its twelve non-BIT rows.  These differences
demonstrate why neither eager nor an isolated JIT may certify the production
boundary.

## Record-directed SSH magnitude

The diagnostic intervention changes exactly one handoff field.  Its directed
SSH equals the independent NEMO N+1 entry bit-for-bit, and U/V external
velocities plus both transport averages equal the ordinary arm bit-for-bit.
The identity guard covers all five fields.  Rows below are cells unequal / wet
and maximum absolute difference against the direct NEMO stage record or
next-step entry.

| stage-3 output | ordinary production JIT | NEMO-SSH-directed | directed minus ordinary |
|---|---:|---:|---:|
| `zFu` | 17,400/17,400 / `8.109319272585253` | 17,400/17,400 / `8.10944682781701` | 17,400/17,400 / `5.230136721365852e-4` |
| `zFv` | 17,100/17,100 / `10.850152134285963` | 17,100/17,100 / `10.85052704601003` | 17,100/17,100 / `1.3274354187160498e-3` |
| `zFw` | 17,996/18,000 / `23.68020002427511` | 18,000/18,000 / `23.53577163242153` | 18,000/18,000 / `0.5518561485699536` |
| local kt3 T | 17,999/18,000 / `8.600420500215478e-7 K` | 17,995/18,000 / `1.07454589937106e-6 K` | 17,880/18,000 / `1.0837906074812054e-6 K` |
| local kt3 S | 17,265/18,000 / `6.979443156751586e-8` | 17,009/18,000 / `1.720615962597094e-6` | 16,655/18,000 / `1.7205889264459984e-6` |

The substitution makes U and V transport maxima worse by factors
1.00001573 and 1.00003455, improves the W maximum by only 0.6099%, worsens T
by 24.94%, and worsens S by 2,365%.  The three transport families respond
differently and the tracers strongly cancel the SSH error.  Per the Round-114
warning, this one-variable oracle arm is magnitude evidence only.  It neither
attributes the SSH mismatch nor creates a candidate.

## Plants and non-vacuity

The production-JIT history plant artifact is
`external_walk/history_ulp_plant_v5.json`, SHA-256
`2a10bf8e765d07ea01fd5cfdbc7d10b49b62ca4ad1200c15afa3b230989e3f8d`.
It changes `u_b[0,1,1]` by one ULP, makes that exact row the first mismatch
(1/580, `6.776263578034403e-21`), prints
`ROUND116 HISTORY-ULP PLANT STATUS PLANT-FIRED`, and exits 1.

The handoff plant artifact is
`external_walk/handoff_ssh_ulp_plant_v5.json`, SHA-256
`779eeb9585cffbbe594306f756a2d1cf01dad8a6841fa561f3be7c8b055dc2f2`.
It changes recorded `ssh[1,1]` bits
`4558473732904528572 -> 4558473732904528573`, moves exactly one SSH cell by
`1.0842021724855044e-19` and exactly one directly derived QCO U ratio by
`2.6469779601696886e-23`, prints
`ROUND116 HANDOFF-SSH-ULP PLANT STATUS PLANT-FIRED`, and exits 1.  The plant
does not require an unrelated later tracer rounding to preserve the bit.

Six focused unit controls cover compiled order, complete record mapping,
drag-face association, isolated-boundary separation, the real QCO handoff,
and one-ULP pytree identity.  Removing either observer-identity guard makes
the science gate refuse before emitting an artifact.

## Decision-43 trajectory and cross-card scope

No production statement changed, so there is no candidate and no moved
trajectory row.  The incoming production values remain both before and after:

| required headline | before | after | disposition |
|---|---:|---:|---|
| kt2 T | `1.4210854715202004e-14 K` | same | AT-BAR, unchanged |
| kt2 S | `2.1316282072803006e-14` | same | AT-BAR, unchanged |
| kt2 U | `2.7377110452773967e-12 m s-1` | same | first-over-bar DEBT, unchanged |
| kt2 V | `3.2849219221489645e-12 m s-1` | same | first-over-bar DEBT, unchanged |
| kt3 T | `8.600419718618468e-7 K` | same | DEBT, unchanged |
| kt3 S | `6.979441735666114e-8` | same | DEBT, unchanged |
| day-30 T RMS | `6.890484901489568e-5 K` | same | no candidate; month not rerun |

Decision 43's month-decrease, first-over-bar, kt1, moved-row registry and
recipe-derived executing-card criteria are not invoked by an empty production
diff.  The diagnostic SSH arm above is not called an after arm.

GYRE-zco and the generic NEMO-GYRE recipe execute the external solver.  DINO
shares its implementation through a different time integration and carries
the known 96--98% regional cancellation risk.  This round does not claim a
DINO number: diagnostics cannot move DINO bits, and the next shared production
candidate must measure the DINO card before landing.  LOCK_EXCHANGE and
OVERFLOW likewise cannot move because no shared production statement changed.

ORCA2 remains **UNMEASURED-WITH-SPEC**: independently record and admit all
substep U/V/SSH histories, midpoint coefficients and fields, face depth,
transports, continuity, backward pressure, Coriolis, drag, frozen forcing,
exchanged exits, weighted final U/V/SSH, next-step entry, and stage-3 U/V/W/T/S
on native masks; drive the production JIT and eager controls; require the
plants to fire; then run its certified short and long trajectories for any
shared candidate.

No restart/checkpoint representation, configuration, selector, coefficient,
timestep, stabilizer, carried state, year harness, reconciliation gate,
freshwater pair, #1484 guard, held manifest, or NEMO source changed.

## Review, citations, and tests

The required adversarial command was run from the provisional receipt commit
with `codex exec --sandbox read-only -C` and a prompt that tried to refute the
production-step interception, record/current-source association, compiled
order, first-non-bit boundary, diagnostic substitution, cancellation verdict,
Rule-12/Decision-43 disposition, and OPEN handoff.  It exited 1 before a
reviewer started, so it emitted no `SHIP`, `HOLD`, or `DO NOT SHIP` verdict.
Its verbatim terminal finding was:

```text
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Accordingly, **independent review unavailable in-sandbox**.  This is not
presented as approval.  The complete attempted-review log is
`round116/review/codex_review.log`.

The clean-tree receipt citation gate reports **PASS** for all 16/16 compiled
citations, zero unmapped citations, zero cited-row failures, zero full-map
audit failures, and all built-in plants firing.  Its separate shifted-line
plant moves the current-build frozen-forcing citation, reports
`SYMBOL-NOT-AT-LINE`, and exits 1.  Ordinary and plant artifacts are under
`round116/citations/`.

Focused CPU/fp64 suites reported exactly:

```text
25 passed in 2.71s
119 passed in 458.38s (0:07:38)
```

The first is the extended external-step gate.  The second is the inherited
four-file push gate: receipt citations, TKE NEMO terms, NEMO recipes, and real
freshwater closure.

The required one-piece
`-n 12 tests/ocean/fidelity tests/ocean/unit` run was attempted exactly once.
It collected 8,187 cases and reached 95%, but nine workers aborted in JAX
compilation and xdist repeatedly replaced them.  After the ninth dead worker,
the controller made no progress for more than eight minutes and emitted
neither a terminal summary nor a JUnit file.  One interrupt closed that stale
controller; the execution session reported exit 1.  There is therefore no
full-suite summary line to quote and no mechanically complete failing-set
diff; this receipt does not invent either one.  Its last progress line was:

```text
tests/ocean/unit/test_nemo_ws_stage_mean_weights.py::test_staircase_card_distinguishes_reference_from_live_stage_mean_weights[OVERFLOW-zps]
```

All nine crash traces named their active node.  A fresh serial replay of those
nine exact node IDs reported:

```text
9 passed in 70.76s (0:01:10)
```

The incomplete interleaved run had recorded 202 unique ordinary failed IDs
before the stall: 79 occur in the pinned 87-ID incoming baseline and 123 do
not.  Those counts are retained under `round116/tests/`, but are explicitly a
resource-contaminated partial observation rather than a failing-set verdict.
None of the 123 is in a Round-116-changed test: the diff from incoming commit
`bb9c586891df9cb089be3675b3a1de38e160ee2f` changes no production package and
no ocean unit test, while every changed executable validation/test path is in
the 25/25 focused pass.  The nine compiler-crash nodes also pass serially.
Thus the available recovery evidence contains no Round-116-owned regression;
it does not mislabel the incomplete full-tree run as green.  Complete focused,
push-gate, interrupted-run, ID-diff, and crash-replay evidence is under
`round116/tests/`.

## ASKED / UNASKED

ASKED and completed: preregistration before measurement; reuse and extension
of the admitted Round-81 gate rather than a second harness; exact record
admission; compiled-source diff; 50-substep source-order walk; full weighted
N+1 SSH alignment; production JIT, production eager and isolated labels;
full-step observer non-interference; two nonzero plants; complete U/V/W
stage-3 and local kt3 T/S one-variable magnitude table; explicit failed S
prediction; first upstream boundary; baseline kt2 U/V, kt3 T/S and day-30 T;
DINO risk; and ORCA2 acquisition specification.

UNASKED and not done: NEMO source was not modified; `makenemo` and `mpirun`
were not run; no oracle was acquired; no production physics, configuration,
default, coefficient, stabilizer, carried state, restart schema, year harness,
reconciliation gate, freshwater pair, #1484 guard, or held patch changed; no
early-return trace is called production; no ladder/month run was spent on a
non-candidate.

## OPEN for round 117

1. Stay on the frozen slow-forcing boundary.  Reuse and extend the admitted
   Round-83 producer gate; do not build another harness and do not return to
   the exonerated external-loop update or QCO geometry.
2. At the current post-Round-110 tip, split the frozen forcing in compiled
   order under the complete production JIT: the incoming `Ue_rhs`/`Ve_rhs`,
   the Kmm `dyn_cor_2D` result, and the final subtract-and-mask result.  Score
   U and V separately against the record and retain eager/isolated labels.
   The first imported non-BIT operand moves the walk upstream; a non-BIT
   result is a candidate only when every direct input is BIT.
3. If `Ue_rhs`/`Ve_rhs` is the first input, rerun the admitted cumulative RHS
   order at the current tip (`after_hpg`, `after_ldf`, `after_vor`,
   `after_keg`, `after_zad`, `after_adv`) before reusing any Round-84/85 owner
   label.  Quantify its contribution through final external SSH and local
   kt3 T/S so magnitude, not one last bit, ranks the next action.
4. The prior evidence predicts LDF remains the first bit boundary, but that is
   a hypothesis until the current-tip production-JIT producer walk reproduces
   it.  Apply Decision 43 only to a current-tip source-exact candidate and
   measure DINO if the statement is shared.
5. No NEMO acquisition is requested: the Round-64, Round-81 and next-step
   records already contain the required producer, external-step and handoff
   boundaries.  Request a new record only if the current-tip split proves a
   directly consumed operand absent.

ACQUISITION_NEEDED: NONE

DECISION_NEEDED: NONE
