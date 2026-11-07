# NEMO testcase L2 GYRE phase 3 — round 121 W magnitude intervention receipt

Date: 2026-09-19

Status: **HELD — the W-only local discriminator closes, but its required
production plant is inert; no trajectory or physics claim lands.**

## Verdict

Replacing only the stage-1 W field at model step `kt=2` with the admitted
NEMO field removes the measured W-owned ZAD discrepancy under the complete
production JIT.  The ordinary W row reproduces at 18,000/18,000 unequal cells
and `7.946658315637966e-7 m s-1`; the ordinary incremental ZAD discrepancies
reproduce at `1.9220297482797664e-9 m s-2` U and
`1.966061294804274e-9 m s-2` V.  With NEMO W, the consumed W is BIT and the
incremental residual falls to `1.0587911840678754e-22` U and
`2.117582368135751e-22 m s-2` V.  Every registered non-W operand and pre-ZAD
boundary stays BIT.  The complete returned state moves in 95,811/198,956
words, maximum `5.00766287514498e-6`, so this is a live production-step
intervention rather than an isolated arithmetic result.

That strong local result does **not** establish W as a month or year owner.
The frozen control required a one-ULP W perturbation to move a raw ZAD word
and a returned-state word.  The implemented arm changed the first
lexicographic eligible W word by
`6.617444900424222e-24`, but both raw ZAD arrays and the full returned state
remained BIT.  The command exited nonzero and printed
`ROUND121 ZAD-W-ULP STATUS PLANT-INERT`.  P2 is therefore **REFUTED** and the
preregistered P3 condition was not met.  The kt=1..10, days 1--30 and 360-day
intervention trajectories were correctly withheld; the day-240 contribution
of W remains **UNMEASURED**.

The preregistration said the selector would find the first eligible word
*whose* ULP propagates.  The implementation stopped at the first eligible
word instead of exhaustively establishing that no later word propagates.
Accordingly, `PLANT-INERT` is a fail-closed control result and a selector
defect, not evidence that every possible one-ULP W plant is inert.  This
limitation is part of the handoff below; it is not hidden by immediately
changing the plant scale.

No shared physics, configuration, carried state, restart schema, year
harness, reconciliation gate, freshwater pair or `#1484` guard changed.

## Frozen provenance

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round121.md`, SHA-256
`6df34269a035bc75f25df0646a6004aafa3d5e6648134ff755c3df153b42f322`,
committed before measurement as
`0f4bc3ab4`.  The private hook, existing-gate extension and focused controls
were committed as `a65af1fe9`; the corrected pre-ZAD registry was committed
as `49cf7016f48e13860c8ba00d4cead968cb827d4f`.  Every authoritative science
artifact is stamped with that clean full commit.  The admitted Round-64 stage
record SHA-256 is
`64c42dbecee9e82639384dfa084899d799ae215a27b44ffaf055c488e0ae2642`.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round121/`:

| artifact | role | SHA-256 |
|---|---|---|
| `production_jit_final.json` | authoritative production-JIT W intervention | `538f07eee460d38dd3016c698f735e58777bc5573671c4c1d1dce7c0f8d7e221` |
| `production_eager.json` | separately labelled complete-step eager control | `191faa615e99e2cb69943b03f2a5abf18b8055091769df913930f841d7136f8b` |
| `zad_w_ulp_plant.json` | nonzero inert production-JIT plant | `45675c67b234050a35f62a2dae056fde850eee9834b530d0da58775ba0de6b1e` |
| `production_jit_final.log` | named JIT confirmation | `e29bcc8cd353c5972c0f44f586bda5cd94e0205fe2354a04ffe84462f5d08a49` |
| `production_eager.log` | named eager measurement | `f5bc75662305494fe16c780b70fd02a80c2570e2e8b96752d649323ba0e7c832` |
| `zad_w_ulp_plant.log` | named nonzero `PLANT-INERT` verdict | `6c58acfa741a17015cdf2f225472c3d8b1a74345cbf4d3ff8c430a48da568acf` |
| `codex_review.log` | required separate review attempt | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |

The first scientific command used an abbreviated expected commit and the
fail-closed worktree stamp refused it with `GATE FAILED: Round-117 commit
stamp mismatch`; retained `production_jit.log` has SHA-256
`76e677e5c52774c97cf8dbf1127c9172a0f0465d04e1fe3a7d5dbfdc5e490d0d`.
The first full-stamp run then exposed a gate-classification defect: it treated
the ordinary normal-order `after_ldf` cumulative boundary as pre-ZAD and
therefore printed `ROUND121 PRODUCER REFUTED` despite the W arm itself
passing.  That attempt is retained as `production_jit_attempt2.json` and
`.log`, SHA-256
`4e235b002d0f9f93ef37b994203165d4e7ce28158feb820435b56d18b83999d2`
and `46cb0233f44c327177b12095a07830f809ae8a6a058a8af7edc6b30f30fd3821`.
The corrected registry admits only the actual pre-ZAD HPG/VOR/KEG boundaries
and retains raw LDF as the independent identity check; no numerical value was
discarded or overwritten.

## Compiled program that actually ran

The admitted record's compiled stage calls HPG, LDF and VOR, constructs
`r3t(Kaa)` and `ww`, and only then calls KEG and ZAD into `Krhs` at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:141-176`.
The selected QCO W producer performs its bottom-up recurrence, including the
free-surface-volume term, at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90:293-300`.
The compiled ZAD operator reads `ww` to form the three transports, builds the
vertical shear products and updates the U/V `Krhs` fields at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzad.f90:102-138`.

Those citations establish the executed boundary and statement order.  This
round does not claim the QCO recurrence is itself the first wrong W-producer
statement: the required trajectory attribution stopped at its inert plant.
At the resolution measured here, the first and largest non-BIT ZAD input is
the consumed W boundary.  Naming an upstream W statement requires the future
compiled-order producer walk after a propagating control exists.

## One-variable production result

The private hook replaces only W at the stage-1 ZAD call boundary during the
model's `kt=2` step.  It preserves the live velocities, U/V thicknesses,
masks, metrics, HPG, LDF, VOR, KEG and ZAD implementation.  It defaults to
`None`; no recipe or card can select it.  The trajectory proxy, which was not
run, delegates all numerical stepping and output to the existing committed
from-rest harness and would apply the override on the second step only.

Authoritative production-JIT rows:

| row | ordinary versus NEMO | NEMO-W arm / arm effect |
|---|---:|---:|
| consumed W | 18,000/18,000; `7.946658315637966e-7` | **BIT** to NEMO |
| incremental ZAD U residual | `1.9220297482797664e-9` | 32 cells; `1.0587911840678754e-22` |
| incremental ZAD V residual | `1.966061294804274e-9` | 18 cells; `2.117582368135751e-22` |
| raw ZAD U, directed versus ordinary | — | 17,400 cells; `1.9220297482797598e-9` |
| raw ZAD V, directed versus ordinary | — | 17,100 cells; `1.9660612948042607e-9` |
| non-W raw operands | — | all BIT |
| non-W ZAD inputs | — | all BIT |
| pre-ZAD HPG/VOR/KEG boundaries | — | all BIT |
| complete returned state | — | 95,811/198,956; `5.00766287514498e-6` |

The separately labelled complete-step eager control reproduces the ordinary
W and ZAD rows.  Its NEMO-W incremental residual is 30 U cells at
`2.117582368135751e-22` and 10 V cells at
`5.293955920339377e-23`; the returned state moves in 95,846/198,956 words at
`5.007662828640513e-6`.  Thus eager independently supports the direction of
the result, but it is not substituted for the production-JIT proof.  The
eager artifact's top-level `prediction_confirmed=false` belongs to a legacy
producer-association classification in the extended gate; its dedicated
`round121_w_intervention.prediction_confirmed=true` is the Round-121 result.

## Plant and prediction disposition

The implemented plant chose active W index `[1,1,0]` and moved it one
representable fp64 word toward positive infinity.  The consumed-W check sees
exactly one changed cell at `6.617444900424222e-24`; all non-W raw operands
remain BIT.  The raw ZAD U/V checks see zero changed cells, and the
returned-state check sees zero changed words.  This is not a vacuous zero
perturbation, but it is swallowed by downstream rounding before the
registered ZAD boundary.  Because the selector did not search later eligible
words, this result rejects the implemented control without proving the
stronger all-words claim frozen in P2.

| preregistered statement | verdict | evidence |
|---|---|---|
| admitted record, ordinary W count/max and ordinary incremental ZAD maxima reproduce | CONFIRMED | exact frozen values under production JIT |
| directed consumed W becomes BIT | CONFIRMED | 0/18,000 unequal |
| every registered non-W input and pre-ZAD boundary stays BIT | CONFIRMED | all emitted rows BIT |
| directed incremental residual is below `8.470329472543003e-22` | CONFIRMED | maxima `1.059e-22` U, `2.118e-22` V |
| complete returned state changes | CONFIRMED | 95,811 words, `5.008e-6` maximum |
| selector finds the first propagating one-ULP W word | **REFUTED / gate defect** | implementation tried only first eligible word |
| implemented one-ULP W plant changes raw ZAD and returned state | **REFUTED** | both registered downstream rows BIT; named nonzero exit |
| P2 passes and authorizes P3 | **REFUTED** | frozen plant conjunct failed |
| W changes day 240 by less than 10% | **UNMEASURED** | P3 was conditional on P2; no trajectory was run |

## Trajectory, landing and blast radius

Because P3 was not authorized, there is no candidate after arm.  The required
headlines are reported without inventing one:

| headline | immutable before | Round-121 candidate after |
|---|---:|---:|
| kt2 T | `1.4210854715202004e-14 K` | NOT RUN — plant stop |
| kt2 S | `2.1316282072803006e-14 PSU` | NOT RUN — plant stop |
| kt2 U | `2.7377110452773967e-12 m s-1` | NOT RUN — plant stop |
| kt2 V | `3.2849219221489645e-12 m s-1` | NOT RUN — plant stop |
| kt3 T | `8.600419718618468e-7 K` | NOT RUN — plant stop |
| kt3 S | `6.979441735666114e-8 PSU` | NOT RUN — plant stop |
| day-30 T3D rms | `6.890484901489568e-5 K` | NOT RUN — plant stop |
| day-240 T3D rms | `1.6446741930292448e-2 K` | NOT RUN — plant stop |
| day-360 T3D rms | `1.1223573910167267e-2 K` | NOT RUN — plant stop |

Decision 43 and Decision 45 are therefore not invoked as a landing
comparison.  No kt=1 row moved, no first-over-bar classification changed,
and the immutable before arms remain unchanged.  The only moved numerical
rows are the complete-step local rows registered above.  DINO shares the W
and ZAD statements and remains explicitly **AT RISK** for any future shared W
producer fix, but it does not execute the private default-off hook.  The
generic GYRE recipe likewise cannot select it.  LOCK_EXCHANGE and OVERFLOW do
not execute the intervention.  ORCA2 remains **UNMEASURED-WITH-SPEC**: record
its compiled-card W producer inputs, W/ZAD boundary and step/year entries;
prove the production-JIT statement with a propagating control; then run its
certified ladder, month and registered year rows before a shared landing.

## Verification

The focused extended Round-83/Round-121 gate suite ended literally:

- `14 passed in 0.56s`.

It covers the private-hook default, recipe/config exclusion, second-step-only
trajectory routing, argument validation and the pre-existing Round-83/117/120
contracts.  Its log SHA-256 is
`4518aa25854960ecc8a41b792af3a2eea10eb3524b71bbfa4e7950499864adbc`.

The mandatory combined `tests/ocean/fidelity` plus `tests/ocean/unit` command
created 12 workers and collected 8,213 tests.  Repeated JAX compiler aborts
caused xdist to replace workers; the run reached 98%, then spent eight silent
minutes in the known legacy phase-3 stage sweep.  It was interrupted at the
round CPU bound and printed **no terminal summary line**, so it is not called
green.  Its literal progress stream contains 39 raw `FAILED`/`ERROR` node
markers: five occur in the frozen 87-node list and 34 do not.  That is not a
terminal failing set—worker aborts, rescheduling and the interruption prevent
pytest from producing one—so the raw-marker difference is disclosed but not
misrepresented as 34 code regressions.  The full log SHA-256 is
`47d10f24dbc7ab0fed726bde22d81deb8d21753d7aee62353eb8815ce162db41`.

The sole raw failure on the directly changed NEMO-WS production path was a
fatal abort while JAX compiled
`test_nemo_ws_exposed_tracer_stage_carries_the_stage_ssh`.  Its isolated CPU
replay ended literally:

- `1 passed in 98.29s (0:01:38)`.

The replay log SHA-256 is
`e5591df8ab48cbcb4addd23bfeee39c53dd467f575422f8444990dc31a73efac`.
Together with the exact ordinary GYRE reproduction and the focused default-
off tests, no changed-path regression remains demonstrated by the stressed
full-tree attempt.

The receipt-citation unit suite ended literally:

- `16 passed in 1.87s`.

Its log SHA-256 is
`b24a1cbd08b27e20462e95e94082dda11c20649ee3b4691b89d463394d675e23`.
The receipt citation command found all three compiled-source citations, no
unmapped citations, no failures and no map-audit failures: `status=PASS`.
The deliberately shifted stage-program citation exited nonzero with
`status=FAIL` and `SYMBOL-NOT-AT-LINE`.  The gate and shifted-plant JSON
SHA-256 values are
`a3eed5cac2d316320e86a1bab1433ac581ac4bb3d1c1c028649396ce9fb092df`
and
`e2d5408523c514d497e9a4db0602beadbe91b6ce70d2a7ac4483b8cf22b5d2a5`.

## Independent review

The required separate command was run with `codex exec --sandbox read-only -C
/tmp/autopilot-work-sIkn2QvP` and an adversarial prompt covering the complete
diff, one-variable claim, plant failure, trajectory stop, compiled citations
and blast radius.  It could not initialize in the mandated sandbox.  Its
verbatim terminal result was:

`Error: failed to initialize in-process app-server client: Read-only file system (os error 30)`

independent review unavailable in-sandbox; no SHIP verdict is fabricated.

## OPEN — exact handoff to round 122

1. Do not call W a day-240 owner and do not run or land the existing
   trajectory proxy from this round's failed P2.  The local W-to-ZAD closure
   is real; the month/year attribution remains unmeasured.
2. First repair the selector promised by Round 121 without changing the
   frozen one-ULP scale: search eligible words in lexicographic order through
   the complete production JIT and either return the first single word that
   moves raw ZAD and returned state, or emit a mechanically checked exhaustive
   no-propagating-word result.  Preserve index `[1,1,0]` as the inert attempt.
   Only if the exhaustive result is empty may a new preregistration choose a
   larger, scale-aware displacement.  Continue to compare identical complete-
   production graphs and keep eager separate.
3. Only after a control exits nonzero with `STATUS PLANT-FIRED`, run the
   same one-time `kt=2` W intervention through the full kt=1..10 ladder,
   days 1--30 and the admitted 360-day member.  Register every row and rank
   day 240 first, then day 360 and day 30, against the unchanged immutable
   before arms.
4. If the controlled W intervention is below the frozen 10% day-240
   materiality threshold, return to the registered year-owner decomposition
   rather than optimizing W.  If it is material, start a new preregistered
   compiled-order walk at W's QCO producer and name its first production-JIT
   non-BIT statement before proposing shared physics.
5. Any shared producer candidate must measure DINO and every recipe-derived
   executing card, plus the Decision-43 ladder/month and Decision-45 year.
   Keep ORCA2 `UNMEASURED-WITH-SPEC` until its native record and certified
   trajectory exist.

No NEMO acquisition and no user configuration or carried-state decision are
needed for this handoff.
