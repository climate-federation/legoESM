# NEMO testcase L2 GYRE phase 3 — round 119 production-association receipt

Date: 2026-09-19

Status: **HELD — diagnostic gate landed; no physics or configuration changed.**

## Verdict

The preregistered production-JIT closure is **REFUTED** at its first new
falsifier.  Given bit-identical HPG, LDF, VOR, KEG and ZAD term arrays, the
full-step NEMO-order arm and the separately-jitted NEMO-order reconstruction
are BIT through HPG, LDF and VOR.  At the KEG write, production JIT differs in
exactly **1 / 17,400 U cells**, maximum
`4.1359030627651384e-25 m s-2`; V stays BIT.  The same U word survives ZAD and
ADV.  Production eager is BIT at every boundary.  Therefore the isolated
closure is not production certification, the association owner is withheld,
and no candidate is eligible for the ladder or month run.

The larger facts reproduce.  Each production arm closes bit-for-bit to its own
live total.  Production-JIT NEMO order versus ordinary model order is
6,882/17,400 U and 6,566/17,100 V cells at
`8.470329472543003e-22 m s-2`, exactly the Round-118 residual.  Eager gives
different counts, 6,859 U and 6,569 V, at the same maximum.  This attributes
the differing census to the complete compiled closure, but the one-cell KEG
falsifier forbids claiming that the isolated reconstruction is its bit-exact
surrogate.

Magnitude remains elsewhere: the refreshed ZAD operand table names W first
and largest, 18,000/18,000 cells at `7.946658315637966e-7 m s-1`; Kmm U/V,
the materialized face thicknesses, T area and both reciprocal face areas are
BIT.  This round does not spend a Decision-43 trajectory on a last-bit
association diagnostic.

## Frozen provenance

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round119.md`, SHA-256
`16fe3ab7f0676f9d3169c6ffb307718b675873e622f79e40609a6135682d0f80`,
committed before measurement as
`f76fb816d2b381a43ed0d04afeb9ef339c3e459b`.  The clean measurement commit is
`21c27d97f445750729ea504cfb4c0ee07f9b8179`.  Its private diagnostic reuses
`nemo_testcase_l2_gyre_round83_slow_forcing_walk.py`; no second harness was
created.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round119/association/`:

| artifact | role | SHA-256 |
|---|---|---|
| `production_jit.json` | authoritative complete production-step comparison | `b6bcef658343cc2cc4519fa9cf217b7f81506efb38baf0b25505464d32ac0cff` |
| `production_eager.json` | separately labelled eager control | `6c71dffe96f834a43ab0b8c40bdc8531d632ee9cfe485df30c917a50412165ff` |
| `hpg_ulp_plant.json` | production-closure causal control | `7fb58fc3da6408f72b099fdfac59ec6fdb040abc67e32e458eb4277f590b03e7` |
| `production_jit.log` | JIT gate log | `a1c0319a305ee14b9ca0422fb1f9843455de37ad1b5c436a0dc2525cbe07713c` |
| `production_eager.log` | eager gate log | `81da600cc2b32c78be046a2d5d508e38a3861e60c30700052a0c4626145d4abb` |
| `hpg_ulp_plant.log` | plant log, including named nonzero verdict | `3e1c328fc7cd236123c5e912bd718cb85f052df895dfe359cbbe0c3ee016509a` |

The inherited Round-64 stage record, Round-81 external-step record and
Round-117 direct pre-loop record all passed their existing admissions before
these rows were evaluated.  The Round-119 gate also retained Round 118's
known direct-boundary result: incoming differs on 580/570 U/V faces,
pre-loop Coriolis differs on 457/425 faces only at about `1e-23`, and the
written subtract is BIT given those operands.  Consequently the inherited
`producer_split.prediction_confirmed=false` is not a new Round-119 regression;
it is Round 118's retained refutation of its original BIT-Coriolis prediction.

## Compiled program that actually ran

This receipt binds every claim to the exact Round-117 record producer.  Its
live namelist selects vector form and C2 KEG at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/EXP00/namelist_cfg:161-162`, and the completed
run reports those values at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/EXP00/ocean.output:780-781`.  The compiled
initializer maps that selection to `np_VEC_c2` and accepts only C2/HW KEG at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynadv.f90:339-345`; this is an
executed branch, not a dead-arm quotation.

The stage-1 program calls HPG, LDF, VOR, KEG and ZAD into the same `Krhs`, in
that order, at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:147-179`.
The LDF loop's in-place plus-association is explicit at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynldf_lev.f90:121-141`.
For the selected C2 arm, KE and the two in-place RHS writes occupy
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynkeg.f90:117-131`; the first
production-only non-BIT boundary is specifically the U/V `Krhs = Krhs -
gradient*metric` pair at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynkeg.f90:129-130`.

This names the first measured statement.  It does **not** say NEMO's KEG
arithmetic is wrong or that legoESM's isolated KEG term is an owner.  The NEMO
oracle residual is already `2.529e-14`/`3.503e-14` before KEG, whereas the new
production-versus-isolated discrepancy is a single `4.136e-25` U word.  The
next measurement must distinguish materialization/masking from full-step XLA
fusion at that exact write.

## Production/eager/isolated tables

All rows below use the same admitted kt2 stage-1 entry.  “Production source”
is the private complete-step arm.  “Isolated” is the separately-jitted term
accumulator and is never labelled production.

### Raw term identity between production arms

| term | U unequal / wet | V unequal / wet | maximum |
|---|---:|---:|---:|
| HPG | 0 / 17,400 | 0 / 17,100 | 0 |
| LDF | 0 / 17,400 | 0 / 17,100 | 0 |
| VOR | 0 / 17,400 | 0 / 17,100 | 0 |
| KEG | 0 / 17,400 | 0 / 17,100 | 0 |
| ZAD | 0 / 17,400 | 0 / 17,100 | 0 |

### Production source order versus isolated source order

| boundary | production JIT U | production JIT V | production eager U/V |
|---|---:|---:|---:|
| after HPG | BIT | BIT | BIT / BIT |
| after LDF | BIT | BIT | BIT / BIT |
| after VOR | BIT | BIT | BIT / BIT |
| after KEG | 1 / 17,400, `4.1359030627651384e-25` | BIT | BIT / BIT |
| after ZAD | 1 / 17,400, `4.1359030627651384e-25` | BIT | BIT / BIT |
| after ADV | 1 / 17,400, `4.1359030627651384e-25` | BIT | BIT / BIT |

The ordinary actual accumulator closes to the ordinary live RHS with zero
unequal cells.  The production-source accumulator also closes to its own live
RHS with zero unequal cells.  Thus the row is neither a WRITE-only trace error
nor a final-total proxy; it appears inside the full production step.

### Production source order versus NEMO

| boundary | U maximum | V maximum | disposition |
|---|---:|---:|---|
| after HPG | 0 | 0 | BIT |
| after LDF | `2.5292467120726215e-14` | `3.502735092670824e-14` | first oracle non-BIT boundary |
| after VOR | `2.5292467332484452e-14` | `3.502735092670824e-14` | LDF debt retained |
| after KEG | `2.5292467332484452e-14` | `3.502735092670824e-14` | LDF debt retained; one-cell production/isolated split is far smaller |
| after ZAD/ADV | `1.9220276136604423e-9` | `1.966059508480186e-9` | ZAD dominates |

### Refreshed ZAD inputs

| operand | unequal / wet | maximum | disposition |
|---|---:|---:|---|
| Kmm U | 0 / 17,400 | 0 | BIT |
| Kmm V | 0 / 17,100 | 0 | BIT |
| W | 18,000 / 18,000 | `7.946658315637966e-7` | first and largest |
| r3u | 580 / 580 | `1.1010194822576465e-16` | non-BIT representation; materialized thickness is BIT |
| r3v | 570 / 570 | `1.107889978251063e-16` | non-BIT representation; materialized thickness is BIT |
| U face thickness | 0 / 17,400 | 0 | BIT consumed divisor |
| V face thickness | 0 / 17,100 | 0 | BIT consumed divisor |
| T area | 0 / 600 | 0 | BIT |
| reciprocal U area | 0 / 580 | 0 | BIT |
| reciprocal V area | 0 / 570 | 0 | BIT |

The current cumulative incremental maxima also reproduce ZAD as the largest
boundary: `1.9220297482797664e-9` U and `1.966061294804274e-9` V.  The earlier
Round-86 controlled substitution remains the magnitude attribution; Round 119
refreshes its live operands and does not relabel a raw-input comparison as a
new substitution proof.

## Prediction disposition and control

| preregistered statement | verdict | evidence |
|---|---|---|
| ordinary final closes its live total | CONFIRMED | BIT U/V |
| raw operands are identical across full-step arms | CONFIRMED | all 10 term/face rows BIT |
| production source order equals isolated source order at every boundary | **REFUTED** | first failure after KEG, one U cell at `4.136e-25` |
| association-arm final reproduces 6,882/6,566 at `8.470e-22` | CONFIRMED under production JIT | exact registered counts and maximum |
| HPG BIT, LDF first oracle debt, ZAD largest increment | CONFIRMED | complete boundary table above |
| W remains first/largest ZAD input | CONFIRMED | complete operand table above |

The production HPG-ULP plant changed exactly one U raw HPG word at native
location `[1,1,2]`, maximum `2.0679515313825692e-25`; LDF/VOR/KEG/ZAD raw
terms remained BIT.  The word moved every HPG/LDF/VOR/KEG/ZAD/ADV boundary
(one cell each), and the command exited nonzero while printing verbatim:

`ROUND119 ASSOCIATION-HPG-ULP STATUS PLANT-FIRED`

## Landing gate and blast radius

No source-exact production candidate exists after the falsifier, so Decision
43's ladder/month gate is **NOT RUN**.  No moved-row table can honestly be
constructed for an unrun candidate.  The immutable production anchors remain:

| headline | unchanged inherited value |
|---|---:|
| kt2 T | `1.4210854715202004e-14 K` |
| kt2 S | `2.1316282072803006e-14 PSU` |
| kt2 U | `2.7377110452773967e-12 m s-1` |
| kt2 V | `3.2849219221489645e-12 m s-1` |
| kt3 T | `8.600419718618468e-7 K` |
| kt3 S | `6.979441735666114e-8 PSU` |
| day-30 T rms | `6.890484901489568e-5 K` |

The new arm is a private `_NEMOWSRK3TestHooks` value with a false default; no
public recipe or configuration field can select it.  Consequently GYRE,
generic NEMO-GYRE, DINO, LOCK_EXCHANGE and OVERFLOW execute unchanged
production in this round.  DINO nevertheless shares the underlying momentum
tendency/KEG statement and remains **AT RISK** for any future production
landing; its severe row cancellation forbids inferring a DINO verdict from
GYRE.  Tanks do not execute the diagnostic arm.  ORCA2 remains
**UNMEASURED-WITH-SPEC**: record its native stage-entry operator terms, execute
ordinary/source-order full-step JIT plus eager and the propagating plant, then
run the card's certified trajectory before any shared association landing.

## Verification

Focused test before measurement:

- `tests/ocean/fidelity/test_nemo_testcase_l2_gyre_round83_slow_forcing_walk.py`:
  `9 passed in 1.07s`.

Final focused set (Round-51 live operands, the extended Round-83 gate,
Round-117 pre-loop gate, citation-gate tests, DINO momentum diagnostics and
both NEMO-recipe suites):

- `89 passed in 331.48s (0:05:31)`.

The clean-tree receipt citation gate found all seven citations, no failures,
no unmapped citations and no map-audit failures: `status=PASS`.  Its complete
self-test table fired, including the unplanted passing controls.  The explicit
shifted-citation plant moved the first endpoint of the mapped KEG range; it
exited nonzero with `status=FAIL` and
`SYMBOL-NOT-AT-LINE`.  The gate and plant JSON/log SHA-256 values are
`a1856ab8ddcf222b251f6067b20b08c2208cfa43d539514ed6f80f575fbe9d33`
and `0d568c3725a63c35c13121a50480f2b873bd342eb68577305bc65fb10a3a011a`,
respectively.

The mandated combined `tests/ocean/fidelity tests/ocean/unit` run collected
8,203 tests and reached 99%, but eight xdist workers aborted during concurrent
JAX compilation (nine `Fatal Python error: Aborted` records) and the surviving
pool ceased emitting output.  It was interrupted after a documented
20-minute no-output bound.  Pytest therefore printed **no terminal summary
line**; claiming one would be false.  The preserved partial log SHA-256 is
`bc0ca4b3195498f30a234bedd8532c6782b49c4bcbbf98de539c48af8a501b41`.
Following the repository instruction to split compiler-heavy suites, the
separate fidelity-tree run collected 1,384 tests and reached 98% without a
worker crash, but its successive long-tail cases exceeded the same round CPU
bound; it too was interrupted and printed **no terminal summary line**.  Its
partial log SHA-256 is
`d2ca723a7d5c8d9fa70130d93a01c56033137a931b2b19571d4d489dc913eec4`.

All seven unique crash sites were then replayed serially.  The first command
used two unqualified method node IDs and accurately ended with
`no tests ran in 0.41s`; collection was corrected rather than hidden.  The
correct replay included the complete 15-case mass-flux parametrization and
ended:

- `21 passed in 549.36s (0:09:09)`.

Its log SHA-256 is
`4efa8553a4634e0075292d115e6d9b39620332615234fd648000146c8b045580`.
Finally, the frozen 87-node pre-existing-red list was replayed directly.  Its
literal terminal summary was:

- `71 failed, 10 warnings, 16 errors in 145.31s (0:02:25)`.

The 87 observed failed/error node IDs and the frozen baseline sort to
byte-identical files (shared SHA-256
`5c1fe77c7c00939050c91e507a288882bbd57f6115907155afecf8ff3f18d97f`);
both `new_vs_baseline.txt` and `missing_vs_baseline.txt` are empty.  Thus no
failure observed in the focused set, serial crash replay or direct baseline
comparison is new to this round.  Because the two broad runs did not print a
terminal summary, this receipt does **not** overstate that comparison as a
complete 8,203-node failure-set proof; the partial-run limitation is retained
as verification debt, not converted into a green claim.

## Independent review

The required separate command was run with `codex exec --sandbox read-only -C
/tmp/autopilot-work-1649114084` and the adversarial prompt covered the diff,
compiled branch, production labels, plant, blast radius and the no-trajectory
verdict.  The reviewer could not initialize in the mandated sandbox.  Its
verbatim terminal verdict was:

`Error: failed to initialize in-process app-server client: Read-only file system (os error 30)`

Independent review is therefore unavailable in-sandbox; there is no fabricated
SHIP verdict.  The preserved log is `round119/review/codex_review.log`, SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN — exact handoff to round 120

1. Stay in the shared Round-83 gate and at the Round-119 production arm.  Do
   not create a new harness and do not spend a trajectory yet.
2. At the single production-JIT U `after_keg` mismatch, register the exact
   native index, input/output bit patterns and ULP direction.  Compare the
   full-step internal unmasked KEG addend with the WRITE-only face-masked term
   used by the isolated reconstruction.  Use one-variable production arms to
   discriminate mask materialization from XLA fusion/association; eager and
   isolated-JIT stay separately labelled, and a production plant must move the
   targeted KEG boundary.
3. The exact compiled statement to close is the C2 U `Krhs` update at
   `GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynkeg.f90:129-130`.
   Do not call the association source-exact until production JIT is BIT at
   every boundary with identical operands.
4. Once that discriminator closes, return immediately to magnitude: W is the
   current first/largest ZAD operand and ZAD carries about `1.9e-9`, fifteen
   orders above the KEG production/isolated word.  Re-enter W at its current-tip
   producer; do not revive the held Round-88 Kaa/WZV patch without a fresh
   production-JIT stage proof and a Decision-43 day-30 improvement.
5. Any production KEG/W change is shared with DINO and must measure DINO before
   landing.  Keep ORCA2 `UNMEASURED-WITH-SPEC` until its native record exists.

No NEMO acquisition and no user configuration/carried-state decision are
needed for this handoff.
