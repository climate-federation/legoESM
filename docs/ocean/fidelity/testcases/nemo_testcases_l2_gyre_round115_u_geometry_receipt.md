# NEMO-testcases L2 GYRE round 115 receipt: U geometry boundary

Date: 2026-09-19

Incoming lane tip: `da07184ff`

Preregistration commit: `1b45f2ccce6100a89d4967102e159efd42a9c5fb`

Authoritative measurement commit: `5f8192ef913428489fbe9c53d395b473568c3973`

Round status: **HELD — no physics or configuration changed; the U-face QCO
geometry statement and the hybrid half-step interpolation are bit-exact when
driven by NEMO's recorded inputs, and the first non-bit boundary is the
external-mode N+1 SSH handed to them**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round115/`

## Outcome first

Round 115 closes the discriminator left by Round 114.  The ordinary
production-step row is reproduced exactly: all 580 consumed U columns differ
in `1+r3u(Kmm)`, with maximum `7.552691805301492e-11`.  However, NEMO's
recorded full-step SSH makes every written U-geometry boundary, the directly
recorded full-step `r3u`, and the directly recorded hybrid half-step `r3u`
bit-exact in the production JIT.  The same result holds in production eager
and separately labelled isolated-closure JIT.

The first live non-bit input is instead the external-mode N+1 SSH: 600/600 wet
T cells differ, with production-JIT maximum `7.072560112143626e-7 m` and RMS
`1.1825507396187026e-7 m`.  Its derived half-step SSH is also 600/600 non-bit,
with exactly half the full-step maximum to the reported precision.  The
current geometry of the live full SSH consequently differs in 580/580 U
columns by `1.5105378299507882e-10`, and its live half ratio differs in
580/580 by `7.552689149747324e-11`.

Therefore the first non-bit compiled boundary named by this round is the N+1
SSH returned by the split-explicit external-mode call, not an arithmetic
statement in the QCO geometry routine.  The external call is the next source
walk; this round does not infer which statement *inside* that solver first
creates the SSH difference.  No implementable QCO correction exists, the held
Round-112 metric-FCT patch was not applied, and no Decision-43 ladder/month arm
was spent.

## Frozen prediction ledger

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round115.md`.  It was reused
verbatim from the operator-interrupted clone
`/tmp/autopilot-work-QLVniqA4`: this lane fetched its single committed
preregistration with `--ff-only`; the interrupted clone's uncommitted draft
was not adopted.

| frozen prediction or falsifier | disposition | evidence |
|---|---|---|
| ordinary `1+r3u(Kmm)` is 580/580 at `7.552691805301492e-11` | **CONFIRMED** | The production-JIT and eager ordinary rows reproduce both values exactly. |
| static `e1e2t`, `r1_hu_0`, and `r1_e1e2u` are BIT | **CONFIRMED** | 0/600, 0/580, and 0/580 unequal respectively in production JIT and eager. |
| own-chain full or half SSH is first non-bit, maximum between `1e-8` and `1e-5 m` | **CONFIRMED** | Full-step maximum is `7.072560112143626e-7 m`; half-step maximum is `3.536280056071813e-7 m`. |
| every compiled-order U trace boundary is BIT given NEMO inputs | **CONFIRMED** | Each of seven rows is 0/580 in all three execution labels. |
| recorded full-step `r3u` is BIT given NEMO SSH | **CONFIRMED** | 0/580 unequal in production JIT, eager, and isolated JIT. |
| compiled hybrid half-step interpolation is BIT | **CONFIRMED** | Both `r3u` and its production `1+r3u` are 0/580 unequal. |
| QCO or interpolation non-bit would trigger a frozen candidate addendum | **NOT TRIGGERED** | Both candidate-owning statements are BIT; ownership moved upstream. |
| diagnostic round changes no production statement | **CONFIRMED** | Only the existing gate, its tests, the citation map, preregistration, and this receipt changed. |

Every preregistered falsifier was tested; none fired.  In particular, this is
not a post-hoc relabelling of Round 114's half-step ratio: both the full-step
SSH and full-step ratio are now directly scored.

## Record admission and configuration

The authoritative ordinary artifact is `u_geometry_walk_v2.json`.  It stamps
a clean tree at `5f8192ef913428489fbe9c53d395b473568c3973`, fp64, and this
resolved recipe before measuring:

```text
ROUND115 RESOLVED CONFIG {"adaptive_implicit_vertadv": false, "barotropic_solver": "explicit_substep", "card": "GYRE-zco", "gm_redi_enabled": true, "nemo_stage_mean_imposition": true, "outer_integrator": "forward_euler", "tracer_advection": "fct2", "tracer_time_integrator": "rk3_ws"}
```

The gate admits the Round-46 kt2/stage-3 record only with producer
`715c9865008e7e419004ad1e5e643e8b71257c00`, SHA-256
`a26121247da49d421da6ce8489627dd0b8302d10bdcd4fd1becf7752eb88524f`,
the exact 64-bit header, kt=2, stage=3, and the frozen owned extents.  It reads
full-step and half-step SSH and `r3u` directly; it does not invert a downstream
transport to manufacture an input.

The first post-preregistration attempt refused before a scientific result with
`GATE FAILED: field/mask shape mismatch`: the admitted T mask still had its
vertical axis.  That log is preserved as `u_geometry_walk.log`.  The fix takes
the directly recorded surface mask and adds explicit cell/native-U/redundant-U
extent guards.  The authoritative v2 ordinary and plant measurements were
then made from the clean fix commit above.  No failed-run number is cited as
science.

## Compiled source and boundary ownership

The record producer's compiled step calls the single external-mode program
before stage 1 at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:188-201`.
Its compiled two-dimensional caller names the result as N+1 SSH and passes
the output slot to `dyn_spg_ts` at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:301-308`.
Stage 1 then saves that N+1 field as `ssha` and calls the QCO geometry routine
from it at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:138-178`.

The recorded build's U statement is the written left product plus right
product, then multiplication by `0.5`, `r1_hu_0`, and `r1_e1e2u` at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domqco.f90:266-268`.
The current Round-111 compiled branch contains the same executing U statement
at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domqco.f90:266-268`.
Thus the direct record and the currently transcribed target are each cited
from their own compiled branch.

The source-replayed reference depth is not a layer-reduction shortcut.  The
record build initializes and accumulates the reference U depth in ascending
level order at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domain.f90:193-200`,
then writes the masked reciprocal at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domain.f90:212-215`.
The horizontal routine constructs the cell/face areas at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domhgr.f90:155-160`
and the face-area reciprocal at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domhgr.f90:170-171`.

After the full-step ratio is written, the hybrid stage-1 branch forms its
one-third ratio at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:188-190`.
The executing hybrid stage-2 arm constructs the N+1/2 SSH and U ratio from
the before and saved full-step slots at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:228-235`;
stage 3 identifies Kmm as that N+1/2 slot at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:240`.
These associations are why exact full geometry plus exact half interpolation
exonerates the QCO and interpolation statements and moves the first boundary
to the earlier external result.

## Production modes and exactness table

Rows are consumed cells unequal / maximum absolute difference.  “Source
trace” means the production callback evaluates the real shared geometry from
NEMO's recorded operands inside the full production step and compares each
boundary with an independent scalar binary64 replay.  It does not substitute
the result into the returned model state.  Eager is secondary; the isolated
closure is explicitly not called production.

| input or direct output | production-step JIT | production eager | isolated-closure JIT |
|---|---:|---:|---:|
| static `e1e2t` | 0/600 / `0` | 0/600 / `0` | source replay input |
| static `r1_hu_0` | 0/580 / `0` | 0/580 / `0` | source replay input |
| static `r1_e1e2u` | 0/580 / `0` | 0/580 / `0` | source replay input |
| live full-step SSH | 600/600 / `7.072560112143626e-7 m` | 600/600 / `7.072560112154468e-7 m` | not a production observation |
| live half-step SSH | 600/600 / `3.536280056071813e-7 m` | 600/600 / `3.5362800560783184e-7 m` | not a production observation |
| live full-step `r3u` | 580/580 / `1.5105378299507882e-10` | 580/580 / `1.5105378299529058e-10` | not a production observation |
| live half-step `r3u` | 580/580 / `7.552689149747324e-11` | 580/580 / `7.552689149757912e-11` | not a production observation |
| geometry from recorded full SSH | 0/580 / `0` | 0/580 / `0` | 0/580 / `0` |
| recorded hybrid half interpolation | 0/580 / `0` | 0/580 / `0` | 0/580 / `0` |
| production `1+` of recorded half ratio | 0/580 / `0` | 0/580 / `0` | not emitted |
| ordinary live `1+r3u(Kmm)` | 580/580 / `7.552691805301492e-11` | same | not emitted |

Every source-trace boundary is BIT:

| written boundary given NEMO inputs | production-step JIT | production eager | isolated-closure JIT |
|---|---:|---:|---:|
| west `e1e2t*ssh` product | 0/580 / `0` | 0/580 / `0` | 0/580 / `0` |
| east `e1e2t*ssh` product | 0/580 / `0` | 0/580 / `0` | 0/580 / `0` |
| parenthesized sum | 0/580 / `0` | 0/580 / `0` | 0/580 / `0` |
| multiplication by `0.5` | 0/580 / `0` | 0/580 / `0` | 0/580 / `0` |
| multiplication by `r1_hu_0` | 0/580 / `0` | 0/580 / `0` | 0/580 / `0` |
| multiplication by `r1_e1e2u` (`r3u`) | 0/580 / `0` | 0/580 / `0` | 0/580 / `0` |
| `1+r3u` | 0/580 / `0` | 0/580 / `0` | 0/580 / `0` |

Production JIT governs the verdict.  Its agreement with eager and isolated
JIT is reported to close the Round-102/103 fusion caveat; neither secondary
mode is used to promote an eager-only claim.

## Plant and non-vacuity

The authoritative plant artifact is `u_geometry_walk_v2_plant.json`.  It
advances recorded `ssh_n1[1,1]` by exactly one ULP, bits
`4558473732904528572 -> 4558473732904528573`, and identifies native U face
`[1,1]` as the propagated target.  A written product hash changes; the
record-directed full ratio changes in one cell by
`2.6469779601696886e-23`; and the hybrid half ratio changes in one cell by
`1.3234889800848443e-23`.  The process prints `STATUS PLANT-FIRED` and exits
1.  The plant artifact's temporary `qco_u_geometry_statement` owner label is
not a scientific verdict: that run deliberately perturbs the oracle input and
is accepted only when it becomes non-bit.

The three focused unit controls independently test the full trace, propagation
through production-style JIT, and rejection of distinct duplicate executions.
Removing the propagation or extent guard makes those controls fail rather
than silently passing.

## Decision-43 trajectory, cards, and scope

No production statement changed, so there is no candidate and no moved
trajectory row to register.  The immutable Round-110 landing remains both
before and after:

| required headline | before | after | disposition |
|---|---:|---:|---|
| kt2 T | `1.4210854715202004e-14 K` | same | AT-BAR, unchanged |
| kt2 S | `2.1316282072803006e-14 psu` | same | AT-BAR, unchanged |
| kt2 U | `2.7377110452773967e-12 m s-1` | same | first-over-bar DEBT, unchanged |
| kt2 V | `3.2849219221489645e-12 m s-1` | same | first-over-bar DEBT, unchanged |
| kt3 T | `8.600419718618468e-7 K` | same | DEBT, unchanged |
| kt3 S | `6.979441735666114e-8 psu` | same | DEBT, unchanged |
| day-30 T RMS | `6.890484901489568e-5 K` | same | no candidate; month not rerun |

Decision 43's decrease, first-over-bar, kt1, moved-row-registry, and executing-
card criteria are **not invoked**, rather than declared passed by an empty
comparison.  The measured result forbids a QCO candidate and points to an
earlier producer walk.

GYRE-zco and the generic NEMO-GYRE recipe execute the shared QCO geometry.
DINO also shares that implementation through a different stage/tracer lane.
None can move because this round changes only a diagnostic gate.  No DINO
number is claimed, and LOCK_EXCHANGE/OVERFLOW are not used to generalize a
GYRE operand result.

ORCA2 remains **UNMEASURED-WITH-SPEC**: admit independently generated ORCA2
full- and half-step SSH, QCO face ratios, masks, reference depth, and metric
operands; run this table through its production closure on native staggering;
then run its certified kt1..10 and long-horizon trajectory.  Any future shared
production change must derive and measure its executing-card set.

## Review, citations, and tests

The required adversarial command was run from the provisional receipt commit
with `codex exec --sandbox read-only -C` and a prompt that tried to refute the
record/current-source association, production-JIT label, source associations,
ULP propagation, upstream-owner conclusion, no-candidate trajectory table,
and OPEN handoff.  It exited 1 before a reviewer started, so it emitted no
`SHIP`, `HOLD`, or `DO NOT SHIP` verdict.  Its verbatim terminal finding was:

```text
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Accordingly, **independent review unavailable in-sandbox**.  This is not
presented as approval.  The complete attempted-review log is
`round115/review/codex_review.log`.

The clean-tree receipt citation gate reports **PASS** for all 13/13 compiled
citations, zero unmapped citations, zero cited-row failures, zero full-map
audit failures, and all built-in plants firing.  Its separate shifted-line
plant moves the record-build U-geometry citation by two lines, reports
`SYMBOL-NOT-AT-LINE`, and exits 1.  Ordinary and plant artifacts are under
`round115/citations/`.

Focused CPU/fp64 suites reported exactly:

```text
34 passed in 64.74s (0:01:04)
119 passed in 405.35s (0:06:45)
```

The first is the modified production gate plus receipt-citation tests.  The
second is the inherited four-file push gate: receipt citations, TKE NEMO
terms, NEMO recipes, and real freshwater closure.

The required one-piece
`-n 12 tests/ocean/fidelity tests/ocean/unit` run was attempted exactly once.
It collected 8,184 cases and reached 95%, but nine workers aborted in JAX
compilation and xdist repeatedly replaced them.  Pytest emitted no terminal
summary and no JUnit file; its last progress line was:

```text
.........................................................s.............. [ 95%]
```

A read-only process check then found no pytest or worker process, while the
controller shell remained open.  One interrupt closed that stale shell and
the command exited 130.  There is therefore no full-suite summary line to
quote and no complete failing-set diff; this receipt does not invent either.
Two of the nine crash traces ended before identifying a test frame.

Seven crash-node functions were recoverable.  The first replay command had an
incorrectly unqualified `test_shapes_preserved` node and correctly refused
with:

```text
no tests ran in 0.47s
```

The corrected fresh serial replay expanded to ten cases and reported:

```text
5 failed, 5 passed in 253.96s (0:04:13)
```

All five failures are exact members of the pinned 87-ID incoming baseline:
the four `test_model_rollout_grads_finite_f32` parameters (`ppm_fct`, `tvd`,
`superbee`, and `dst3`) and
`TestEKEBudgetCoupling::test_signed_iso_sink_gets_raw_kappa_split_call`.
The mechanical replay diff reports zero new IDs.  The recovered leapfrog,
ocean-shape, stage-1 transport, mass-flux-store, and stage-face-mask nodes all
pass.  This recovery does not convert the corrupted one-piece run into a
complete suite; it only classifies every crash node whose trace named a test.
Complete logs and the baseline diff are under `round115/tests/`.

## ASKED / UNASKED

ASKED and completed: frozen preregistration reuse; existing stage-gate
extension rather than a second harness; exact Round-46 record admission;
resolved configuration print; ordinary-row reproduction; compiled-order U
geometry trace; direct full/half SSH and ratio scores; production JIT, eager,
and isolated labels; production-JIT one-ULP plant; record-build and current-
build compiled citations; first upstream boundary; explicit U/V/W
cancellation retention; Decision-43 disposition; DINO/card risk; and ORCA2
specification.

UNASKED and not done: NEMO source was not modified; `makenemo` and `mpirun`
were not run; no oracle was acquired; no production physics, configuration,
default, coefficient, stabilizer, carried state, restart schema, year harness,
reconciliation gate, freshwater pair, or #1484 guard changed; no held patch
was applied; no scratch toggle was promoted; no ladder or month run was spent
on a non-candidate.

## OPEN for round 116

1. Move upstream to the external-mode N+1 SSH boundary; do not continue the
   now-exonerated QCO arithmetic and do not revive the held Round-112 FCT
   patch.  Reuse and extend the admitted Round-81 external-step gate/record,
   not a new harness.
2. First reproduce this round's 600/600 full-step SSH row and
   `7.072560112143626e-7 m` maximum at the stage handoff.  Then align the
   external record's final post-swap SSH with the model's actual N+1 output
   and walk its 50 substeps in compiled order under the production JIT:
   entry/current histories, U/V/SSH midpoint, transports/divergence,
   continuity SSH, backward interpolation, pressure, Coriolis/drag, slow
   forcing, update, exchange, and swap.  Preserve production eager and
   isolated labels but do not let them govern.
3. Re-evaluate the external walk at the current post-Round-110 tip: the
   histories/drag/ZAD bundle has landed since the older Round-81/82 diagnosis,
   so prior owner labels are hypotheses, not current measurements.  The first
   non-bit imported operand owns the next producer walk; the first non-bit
   result with exact inputs is a candidate statement.
4. Quantify magnitude before proposing a fix: register the effect of the
   record-directed full SSH on the complete U/V/W stage-3 transport and local
   kt3 T/S, retaining the Round-114 cancelling-triplet warning.  Only an
   implementable production-JIT statement advances to same-base Decision-43
   ladder/month and recipe-derived executing-card measurements.
5. The compiled outer boundary remains the N+1 external call at
   `GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stp2d.f90:301-308`.
   No acquisition is requested because the admitted Round-81 record already
   contains every external substep and final U/V/SSH swap; request a new NEMO
   record only if a directly consumed current boundary is demonstrably absent.
