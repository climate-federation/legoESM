# NEMO-testcases L2 GYRE round 102 receipt: production TKE statement walk

Date: 2026-09-16. Branch `fidelity/nemo-testcases-l2-gyre-codex2`.

## Verdict

**STOPPED_FOR_DECISION; no production physics or configuration landed.** The
operator-produced Round-101 record is valid. Its apparent duplicate failure
was confined to 600 unconsumed `jpk` RHS sentinel cells: all 21,120 RHS cells
that the compiled solve can consume are BIT, as are the complete entry and
post-sweep duplicates. The existing consolidated stage gate then walked the
TKE closure through the full production `step` / `_step_jitted` path.

With NEMO's recorded closure memory and stress modulus, the production trace
is BIT at entry and after the boundary block, then first becomes non-bit after
Langmuir: 3,223 of 21,120 cells differ, with maximum absolute difference
`2.8334868416759059e-5`. A diagnostic-only production arm which changes only
the already-existing Langmuir evaluation selector from `vectorized` to
`nemo_literal` is BIT through that boundary. The first non-bit model block is
therefore the configured vectorized replacement of NEMO's ordered `zpelc`
initialization and vertical recurrence. The literal arm next first differs in
the assembled RHS, so that is the next compiled-order walk only after this
configuration question is settled.

**AMENDMENT, 2026-09-17, round 103: the attribution in the paragraph above is
REFUTED and must not be read as standing.** The claim that the configured
vectorized replacement of NEMO's ordered `zpelc` initialization and vertical
recurrence is the first non-bit block was tested by the operator with a
one-variable-at-a-time isolation run through this round's own gate, with both
endpoints reproduced exactly (3,223 cells and 0). Seven differences separate
the two Langmuir arms, not the one named here. The ordered recurrence is
MEASURABLY INERT: adding it alone to the compact arm leaves all 3,223 cells
unequal at the same maximum. The measured owners are the mixing-layer index
and the missing surface mask, which are MUTUALLY REDUNDANT - jointly they own
2,912 cells and the entire magnitude, but either one alone fully cures it, so
"X alone fixes it" was never evidence that X is the owner. The 311-cell
remainder splits between the exponent form (153) and the update association
(210); 2,912 + 311 = 3,223 and the account closes. Round 103 did not walk that
recurrence, and no future round should.

A second correction to this receipt's framing, from the same review: selecting
the literal arm buys ATTRIBUTION, not accuracy at the consumed output. The
post-sweep row is 979 cells on BOTH arms and the literal arm is marginally
worse in the last bits.

Selecting a different evaluation is a configuration choice. The preregistered
round explicitly forbade changing the card without the user, so no numerical
candidate, 954-row ladder, or 30-day arm was run. The requested decision is:
**may the GYRE NEMO-identity card select
`tke_langmuir_evaluation="nemo_literal"`? Pick: YES.** If yes, the next round
must preregister that candidate, require the production stage boundary to
close, and apply Rule 12 before landing it.

The ordered campaign remains in **kt=1 stage 1**. The accepted TKE record is a
kt2 discriminator because that is the independently duplicated record
immediately preceding the kt3 magnitude target; it does not authorize a
downstream-stage landing or release the held kt1-stage1 W owner. No NEMO
source or executable was modified or run by the agent. No shared card,
coefficient, timestep, stabilizer, carried state, restart schema, year
harness, reconciliation gate, freshwater pair, or #1484 guard changed.

## Registration and implementation commits

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round102.md`, first committed
as `e338697551a49a86fa117262e0621c1f6f63f482` before admission or scientific
measurement. Its committed addenda preserve the two refuted boundary
predictions, register stress attribution, freeze the Langmuir discriminator,
register the required bottom-level operand routing, and invalidate then repair
the initially misdirected production plant.

The incoming production tip was
`29af6665f77ca8b958af33c91eaa08d1e0aa010f`. The analysis and instrument work
was committed in this order:

- `718ea95661529049a37fbb065318bd0ef08a4fcd` binds duplicate admission to the
  compiled-consumed TKE levels;
- `ecb516423e310a207831af716b116a8ed0d2d97e` through
  `162b1e8ebae1256c83b7fa0a23e58a62a1918835` extend the existing consolidated
  gate, expose statement boundaries inside the full production step, orient
  the record, drive it from NEMO's entry, and split compiler-heavy modes;
- `26ed4a46a74141fc03142d965a5063b8ac626f47` through
  `ef7e11b75ce9057d1d146585315f4057b654da99` preregister and measure the
  boundary and live-surface-operand decompositions;
- `ff93de1353b9e210def3857389f2f93592e323cc` through
  `9e1495d4eb59fb7250c94a68495c9a183de95273` preregister and run the literal
  discriminator through production and route the existing bottom-level
  operand to the dormant literal consumer; and
- `da09a140e9c12fe337e30c0253ab606b5b876890` through
  `05b1f97030197257b4eaae2aec2c69c95280db64` preregister and repair the
  production ULP control and disambiguate a source-order test.

The current configured `vectorized` arm is unchanged bit-for-bit. The only
runtime support change outside the gate is dormant operand plumbing needed by
the diagnostic literal arm; focused tests prove that unrelated configurations
still reject an unused bottom-level operand.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round102/`.

## Compiled-source basis

Every Fortran citation below is to the record producer's compiled
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo` branch, except the explicitly
named independent Round-59 duplicate writer.

The new writer copies the complete live `jpk` RHS image at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/l2_r101_tke_walk.f90:96-104` and
writes all five full arrays at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/l2_r101_tke_walk.f90:116-128`.
The independent Round-59 writer instead zero-initializes the record arrays and
copies RHS only through `jpkm1` at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:162-190`. That
explains, but does not excuse, the stored-sentinel difference: the binding
domain comes from the compiled solve below.

The compiled driver brackets `tke_tke` with the Round-101 recorder and runs
`tke_avn` afterward at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:196-207`. Inside
`tke_tke`, the surface scalar is formed at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:257` and consumes
`taum` in the surface assignment at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:285`. The active bottom
block and its assignment are at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:299-307`, followed by
the first new boundary callback at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:324`.

On this compiled card the no-Stokes-drift arm first associates `zWlc2` from
the same stress modulus at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:353`. It then evaluates
`zpelc(1)` and the source-ordered vertical recurrence at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:361-365`, before the
reverse crossing search, depth, and in-place TKE source at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:369-388`. The second
new record boundary is immediately after this block at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:395`.

The matrix and RHS loop explicitly covers levels `2:jpkm1` and states that
level `jpk` is outside the solve at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:399-443`; its boundary
callback is at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:473`. The following
three recurrences and terminal floor/mask also stop at `jpkm1`, and the final
callback follows them, at
`GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:475-495`. These are the
executing statements and record sites used below; no inactive source arm is
used to name the result.

## Record admission and the Round-101 exit-66 diagnosis

The existing record was admitted without rebuilding or rerunning NEMO.
`tke_record_duplicate_audit.json` reports PASS at clean commit
`718ea95661529049a37fbb065318bd0ef08a4fcd`; its SHA-256 is
`f671b269b9f20a93b24023ff45556daad74218934f4be4c9b47da7c233cb3a95`.
The fixed physical EOF is 873,028 bytes. The producer commit is the immutable
incoming `29af6665f77ca8b958af33c91eaa08d1e0aa010f`; the NEMO target is
`GYRE_OMIP_L2_P3_SM_R101TKEW`, the executable digest is
`3590e854f46b430f594e33ff854381d7fa439b77d0cf9e06d68b22c68913e913`,
and the record digest is
`f8f0bb10f41f48d25fce039b7322b98a5a0ea73837b450c01a7201675fabf9dd`.
The independent Round-59 operand record digest is
`b8f3bead4f30b78257153a0c40c5dd77656aee227b1a94a46625a5604faaca5d`.

The exact duplicate rows are:

| field/domain | cells | unequal | max abs | admission |
|---|---:|---:|---:|---|
| entry, complete stored array | 21,824 | 0 | 0 | binding BIT |
| RHS, complete stored array | 21,824 | 600 | `1.0e-6` | reported DEBT |
| RHS, compiled-consumed `1:jpkm1` | 21,120 | 0 | 0 | binding BIT |
| RHS, unconsumed `jpk` sentinel | 704 | 600 | `1.0e-6` | non-binding DEBT |
| post-sweep, complete stored array | 21,824 | 0 | 0 | binding BIT |

Thus the operator's exit 66 came from an over-broad duplicate assertion, not
from a scientific mismatch in any compiled-consumed value. The executable
gate now reports the sentinel and cannot label the complete RHS BIT.

The standard twin admission also reports PASS with unchanged restart and mesh
digests, no violation, and only the registered additive record. Its artifact
is `tke_record_twin_admission.json`, SHA-256
`51ecf1f504bed669360c672bc84abcf715ea7ad0a4be103a6469bb1d17205197`.
Header, truncation, producer-stamp, record-ULP, and commit-stamp controls each
exit 1. The separate consumed-field twin plant also exits 1. No acquired field
was read scientifically before these checks passed.

## Consolidated production stage twin

The existing Round-46/51 consolidated gate—not a second harness—was extended
and run through the real `LatLonCGridOceanModel.step` and `_step_jitted`
production path. To stay under the known per-process LLVM specialization
limit, the ordinary stage table and TKE statement walk are separate modes of
that one gate. The attempted combined process exhausted the compiler after
the ninth specialization before producing science; this operational attempt
is preserved as `stage_twin_production_tke_preclosure.log` and was replaced by
the preregistered split modes.

The complete row-per-output artifact is `stage_twin_production_tke.json`,
SHA-256
`6b2ba1b6e0d5b1c607f1b633a96376d055d574f0b343500ffba7c0a312fad700`,
stamped from clean commit `cbe4703d2bdb4544d606558bbafd320ca2b7bcba`.
`B / A / D` below means BIT / AT-BAR-but-not-BIT / DEBT.

Given NEMO's recorded stage entry:

| kt | external | stage 1 | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 1 | 11 / 0 / 0 | 13 / 4 / 0 | 14 / 2 / 1 | 11 / 1 / 5 |
| 2 | 0 / 0 / 11 | 10 / 4 / 3 | 11 / 3 / 3 | 10 / 2 / 5 |

Model-chained entry:

| kt | external | stage 1 | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 1 | 10 / 1 / 0 | 8 / 4 / 5 | 5 / 5 / 7 | 3 / 5 / 9 |
| 2 | 0 / 0 / 11 | 3 / 0 / 14 | 0 / 0 / 17 | 0 / 0 / 17 |

For the first stage in campaign order, the full kt1-stage1 table is:

| output | given NEMO entry: class / unequal / max abs | chained: class / unequal / max abs |
|---|---|---|
| T | B / 0 / 0 | B / 0 / 0 |
| S | B / 0 / 0 | B / 0 / 0 |
| U | A / 7,620 / `5.421010862427522e-20` | same |
| V | A / 8,460 / `5.421010862427522e-20` | same |
| SSH | B / 0 / 0 | B / 0 / 0 |
| e3t(Kmm) | B / 0 / 0 | B / 0 / 0 |
| e3u(Kmm) | B / 0 / 0 | B / 0 / 0 |
| e3v(Kmm) | B / 0 / 0 | B / 0 / 0 |
| W | A / 10,548 / `1.6543612251060553e-23` | same |
| TKE en | B / 0 / 0 | D / 914 / `5.536774594960583e-3` |
| TKE avm | B / 0 / 0 | D / 5,714 / `8.176154756539789e-2` |
| TKE avt | B / 0 / 0 | D / 5,714 / `8.176154756539789e-3` |
| TKE dissl | B / 0 / 0 | D / 17,400 / `6.398461778125713e-3` |
| TKE surface avm | B / 0 / 0 | D / 571 / `8.994045779657744e-3` |
| zFu | B / 0 / 0 | B / 0 / 0 |
| zFv | B / 0 / 0 | B / 0 / 0 |
| zFw | A / 10,232 / `1.9895196601282805e-13` | same |

The artifact's mechanically reported first non-BIT output is kt1-stage1 U,
AT-BAR in 7,620 cells, but the consumed momentum RHS is already non-bit; it is
inherited rather than a new statement owner. The campaign's first owned stage
remains kt1 stage 1 and W remains its unresolved/held owner. This round's kt2
TKE record diagnoses the magnitude-ranked compensating closure family without
advancing the stage order.

## Note-L production-fusion discriminator

The held Round-89 assignment-boundary arithmetic was rerun with all three
required execution labels. `production statement` below means the exact
assignment transcription evaluated from operands captured inside the full
production step; `production vs NEMO` also contains upstream operand error and
is not used to judge the statement itself.

| row | isolated eager | isolated JIT | production statement |
|---|---:|---:|---:|
| kt1 s1 U | 0 / 0 | 0 / 0 | 0 / 0 |
| kt1 s1 V | 0 / 0 | 0 / 0 | 0 / 0 |
| kt2 s1 U | 0 / 0 | 2,894 / `6.938893903907228e-18` | 1,436 / `6.938893903907228e-18` |
| kt2 s1 V | 0 / 0 | 2,812 / `1.3877787807814457e-17` | 3,499 / `1.3877787807814457e-17` |

Each cell is `unequal / max abs`. This **confirms** the operator's JIT/fusion
hypothesis for the selected held member: eager equality is not production
equality, and even an isolated JIT has different fusion behavior from the
full step. From Round 102 onward, local-exactness and stage-twin claims are
binding only through the production step; isolated eager and isolated JIT may
be reported alongside but never alone. Production controls must fire there.

## Production TKE statement walk

The binding artifact is
`tke_production_walk_literal_discriminator_routed.json`, SHA-256
`b9a3b5f40f2e6a72ed27c839f6043742210e53014884f05eed45b52feb04fac7`,
stamped from clean commit `9e1495d4eb59fb7250c94a68495c9a183de95273`.
All rows cover every owned cell without a wet-mask exception or numeric
tolerance. The entry domain is the model-represented NEMO levels `2:jpkm1`;
the compiled surface assignment overwrites level 1 before later consumption,
so entry level 1 remains explicitly UNMEASURED-WITH-SPEC. Every later row
covers all 32 x 22 x 30 levels `1:jpkm1`.

| production arm and boundary | unequal | max abs | class |
|---|---:|---:|---|
| model forcing: entry | 0 | 0 | BIT |
| model forcing: after boundaries | 260 | `1.734723475976807e-18` | AT-BAR |
| model forcing: after Langmuir | 3,549 | `2.8334868416759073e-5` | DEBT |
| model forcing: RHS | 15,165 | `2.8334868416759073e-5` | DEBT |
| model forcing: post-sweep | 1,239 | `6.8096880131290893e-12` | DEBT |
| recorded NEMO `taum`: entry | 0 | 0 | BIT |
| recorded NEMO `taum`: after boundaries | 0 | 0 | BIT |
| recorded NEMO `taum`: after Langmuir | 3,223 | `2.8334868416759059e-5` | DEBT |
| recorded NEMO `taum`: RHS | 14,905 | `2.8334868416759059e-5` | DEBT |
| recorded NEMO `taum`: post-sweep | 979 | `6.8096880131290893e-12` | DEBT |
| diagnostic literal: entry | 0 | 0 | BIT |
| diagnostic literal: after boundaries | 0 | 0 | BIT |
| diagnostic literal: after Langmuir | 0 | 0 | BIT |
| diagnostic literal: RHS | 11,993 | `5.488912518947231e-10` | DEBT |
| diagnostic literal: post-sweep | 979 | `6.809688229969524e-12` | DEBT |

### Prediction and attribution ledger

1. The original preregistered prediction—BIT through the boundary block under
   the model forcing, then first non-bit after Langmuir—is **REFUTED**. The
   boundary row already had 260 unequal surface cells.
2. The preregistered bottom-assignment prediction is **REFUTED**. All 260
   inequalities are at the surface; the NEMO-changed bottom set is empty and
   the unchanged interior is BIT.
3. The stress-operand hypothesis is **CONFIRMED**. Production `taum` differs
   from the recorded operand in 303 of 704 cells, max
   `2.7755575615628914e-17`; scalar replay from recorded `taum` is BIT, and
   substituting only recorded `taum` makes the surface assignment BIT through
   the full step. Therefore that surface miss is inherited upstream, not owned
   by the TKE assignment.
4. With the inherited stress removed, the registered Langmuir prediction is
   **CONFIRMED**. The configured vectorized arm first differs after Langmuir;
   the diagnostic literal arm remains BIT through that boundary. Since the
   only changed static selector replaces the vectorized accumulation with the
   compiled ordered `zpelc` program, that recurrence is the first named
   non-bit block.

The literal diagnostic initially refused before executing science because
the existing bottom-level coordinate was routed only for the independent
bottom-TKE-boundary option. That pre-science refusal neither confirmed nor
refuted the prediction. A committed addendum registered routing the same
operand when literal Langmuir consumes it, with tests for both consumers and
the unused-operand refusal; the rerun then produced the binding table above.

## Plants

The final production ULP report is
`tke_production_ulp_plant_fixed.json`, SHA-256
`b05ac182e3e825fe970a9195eddfa047e57c8382626b95f088778cb8618d59c1`,
stamped from clean commit `8748cea62f8809648091abae08b0dbe6605de75b`.
It exits 1 after changing exactly one bitwise cell in
`GYRE-zco.kt2.tke_statement.production_step.en_after_boundaries`: the clean
row has zero unequal cells and the planted row has exactly one, at index
`[0,0,0]`.

The earlier file `tke_production_ulp_plant.json` is explicitly **INVALID** and
is retained as failure evidence. Its obsolete selector produced
`plant_target=null` and changed no cell even though the generic command exit
was nonzero. The preregistered repair changed the selector to the binding
recorded arm and added a fail-closed non-null assertion before the valid rerun.
No receipt claim uses the invalid artifact.

## Rule 12, magnitude, and testcase dispositions

No card or numerical implementation candidate was authorized, so running the
954-row ladder or 30-day member would have been an unregistered configuration
experiment. The immutable Round-96/97 before arm therefore remains the only
trajectory arm; “unchanged” below means no candidate was created, not a new
trajectory measurement.

| headline | immutable before arm | Round 102 disposition |
|---|---:|---|
| kt2 T RMS | `1.4210854715202004e-14` | unchanged; AT-BAR |
| kt2 S RMS | `2.1316282072803006e-14` | unchanged; AT-BAR |
| kt2 U RMS | `2.7377110452773967e-12` | unchanged; first-over-bar |
| kt2 V RMS | `3.284922138989399e-12` | unchanged; first-over-bar |
| kt3 T RMS | `1.627497246303733e-4` | unchanged; magnitude target |
| kt3 S RMS | `6.327735185607253e-6` | unchanged |
| day-30 T RMS | `1.2397011295506804e-2 K` | unchanged; magnitude target |

| lane | disposition |
|---|---|
| GYRE stage twin | PASS instrument; first owned stage remains kt1 stage 1, with W unresolved/held; the kt2 TKE discriminator cannot advance stage order |
| GYRE kt=1--10 | no authorized candidate; no 954-row score; first-over-bar remains kt2 U/V in the immutable arm |
| GYRE days 1--30 | no authorized candidate; immutable day-30 T RMS retained |
| LOCK_EXCHANGE-zco | no shared numerical/card change landed; no new tank-fidelity claim |
| OVERFLOW-zps | no shared numerical/card change landed; no new tank-fidelity claim and partial-cell behavior remains in spec |
| DINO | **SHARED-STATEMENT RISK:** DINO executes the shared TKE program; selecting literal Langmuir would require its own registered production-stage and trajectory assessment |
| ORCA2 | **UNMEASURED-WITH-SPEC:** resolve its integrator and score the same production TKE boundaries and selector before any shared landing |

No AT-BAR row can have left the bar and first-over-bar cannot have moved,
because no candidate arm exists. This is not an improvement claim.

## Independent review and verification

The required separate review command was attempted twice with the full diff,
stage tables, TKE walk, source citations, and Rule-12/decision challenge. Both
read-only invocations failed before a reviewer model started. The terminal
verdict is therefore **independent review unavailable in-sandbox**;
unavailability is not approval. The required verbatim result is:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only
> file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file
> system (os error 30)

The complete log is `codex_review_unavailable.txt`, SHA-256
`f0ffddf7919e317a9151a43c6a4fb34cf70b9d356f2665467a3a2240a168a8d8`.
No review returned DO NOT SHIP; equally, no review returned SHIP. This round
lands only the instrument, fail-closed controls, dormant operand plumbing,
tests, and receipt—not a physics/configuration candidate.

The final focused CPU/fp64 suite covers the consolidated production gate,
record reader and plants, production TKE trace, bottom-level routing, and TKE
physics. It passes **138/138 tests in 71.24 s**. The JUnit artifact is
`focused_pytest_final.xml`, SHA-256
`9a978efdc427f9cd4fe25a765bba824d24b6a0a8236afbb87bcce97e2661a626`.
An earlier combined run had one source-text assertion failure because it found
the newly added diagnostic call before the chained call it intended to check;
the production behavior was not failing. That artifact is retained as
`focused_pytest.xml`; the assertion was narrowed to the chained function and
the complete suite passed above. The two operator-declared pre-existing red
tests were not chased.

The clean-tree citation gate at commit
`0197e03d190b5f4de95293034c5f1b9b3d28f137` finds and maps all 15/15
compiled-source citations, with no unmapped citation, failed anchor, global
map-audit failure, or self-test failure. Its initial JSON SHA-256 is
`da4a13b28a9080d280ce664160dc6db30742517d124ea403f8ccafc0c9af139c`.
Shifting the ordered-`zpelc` citation by two lines exits 1 with
`SYMBOL-NOT-AT-LINE`; the plant JSON SHA-256 is
`7c3a270fa0675dee0f44ab765197a1a0ef761c4b67f7e749276e6368eb86c809`.
The Round-102 production trace shifted 27 older local Python anchors, so the
global map and the nine still-live master-receipt citations were mechanically
rebased rather than exempted. The citation suite then passes **16/16 tests in
1.69 s**; its JUnit SHA-256 is
`6f00813640c9b80a3508b5866b83d6e0d1eb9cea77e173f893bb3c92524d411a`.
A final clean-commit gate rerun overwrites `citation_gate.json` after this
receipt commit so the evidence stamp names the delivered commit.

GitHub issue #1455 is unavailable through this clone's local-only remote, so
no issue post is claimed.

## OPEN — round 103

1. Obtain the user's configuration decision: may the GYRE NEMO-identity card
   select `tke_langmuir_evaluation="nemo_literal"`? The evidence-backed pick
   is **YES** because the full production step becomes BIT through the named
   compiled Langmuir boundary with NEMO's recorded inputs.
2. If YES, preregister the exact card candidate before applying it. Re-run the
   production TKE stage boundary with its production ULP plant, then the full
   954-row kt1--10 ladder. Land only if the stage condition closes, no AT-BAR
   row leaves the bar, first-over-bar is not earlier, and every moved row is
   registered. Only then run and score days 1--30 against the immutable
   Round-96/97 before arm. Measure or prove nonexecution in both tanks; retain
   explicit DINO risk and ORCA2 UNMEASURED-WITH-SPEC.
3. If the user says NO, keep the vectorized arm and walk its ordered `zpelc`
   operands/associations within the same production step to construct a
   non-configuration implementation candidate; do not advance stage order.
4. After an accepted literal selection—or after that recurrence is otherwise
   made BIT—the next measured first non-bit boundary is the assembled RHS:
   11,993 unequal cells, max `5.488912518947231e-10`. Continue there in
   compiled statement order only after the upstream decision and Rule-12
   disposition are complete.

No NEMO acquisition is needed for the next round: the admitted record already
contains every required boundary and the next RHS image.
