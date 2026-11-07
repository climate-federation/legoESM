# NEMO-testcases L2 GYRE round 101 receipt: TKE statement-boundary record

Date: 2026-09-16. Branch `fidelity/nemo-testcases-l2-gyre-codex2`.

## Verdict

**STOPPED_FOR_RECORD; no production physics landed.** Round 101 completed the
Round-100 per-row attribution and recertified the unchanged production stage
tables. The full-RHS-only Round-97 arm and the full-RHS/W/ratio/clock Round-99
arm move the same 85/954 rows, violate the same 56 rows, change no class, and
retain kt2 U/V as first-over-bar. The later W members change metrics in only
21 rows, all at kt7--10; they neither cause nor compensate the Rule-12 veto.

The next magnitude-ranked same-stage family is the already non-bit chained
TKE state. The existing admitted NEMO record jumps from TKE entry to the
assembled RHS, so it cannot distinguish the surface/bottom boundary block
from the active Langmuir assignment. Naming either statement now would be a
post-hoc guess. This round therefore preregistered and shipped a new additive,
WRITE-only NEMO acquisition card for the two absent boundaries. The agent did
not run `makenemo` or `mpirun`.

The first owned stage remains **kt=1 stage 1**. The raw output table's first
non-bit row is U, but its consumed momentum RHS is already non-bit, so that U
row is inherited rather than a new owner. W remains the first owned output;
its already-named ratio/clock/full-RHS correction remains held by Rule 12.
The first non-bit statement in the compensating TKE walk is **not yet named**:
the exact unresolved interval is now mechanically bounded and requires the
requested record.

No production package, configuration, timestep, coefficient, stabilizer,
carried state, restart schema, year harness, reconciliation gate, freshwater
pair, #1484 guard, NEMO source tree, or NEMO executable changed.

## Registration and commits

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round101.md`, committed as
`40db8aa9a73341306d4f9f5f56f227bfbd766223` before any Round-101 measurement
or instrument implementation. The fixed recorder, exact-EOF reader, admission
controls, and tests were committed as
`09f940520763962ce5f1202db49066e0ba2c0068`. The digest-pinned post-hoc
attribution probe was committed before its first execution as
`37c5ef5585adf335f07c2185c051da2666d9a72b`; its final clean-worktree stamp
for both clean and planted reports is commit
`3c94d68c38deacc04e00f8d16004069590157bdb`. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round101/`.

## Registered Round-97 versus Round-99 attribution

The canonical artifact is `rule12_r97_r99_attribution.json`, SHA-256
`71723bd13364b3630413173bab5da4e45ba622ecad8dea87fca1ed741ab5ff35`.
The probe digest-pins both independent input reports, requires the same unique
ordered set of 954 rows, and emits every differing metric for every differing
row.

| property | Round 97 full RHS | Round 99 full RHS + W/ratio/clock |
|---|---:|---:|
| Rule-12 verdict | FAIL | FAIL |
| certified rows | 954 | 954 |
| moved rows | 85 | 85 |
| violating rows | 56 | 56 |
| class changes | 0 | 0 |
| first-over-bar | kt2 U/V | kt2 U/V |

The moved-row sets and violating-row sets are exactly identical. Of the 954
row metric dictionaries, 21 differ: one at kt7, six at kt8, seven at kt9 and
seven at kt10. By field they are S 3, T 3, SSH 3, U 2, V 3, `uu_b` 3 and
`vv_b` 4. There is no differing row at kt1--6. This is post-hoc attribution,
not a substitute for either arm's preregistered Rule-12 verdict.

The row-drop and emptied-metric plants both exit 1. Their SHA-256 values are
`2aab5af9f6215158c5d2af26adefc0183e3e7ea1fd22df2628ea98b48a6d192d`
and `a73d0b50f5e9465c04429a7d633277e95e47c2bab18aed71a1822d70e6f813df`.

## Production stage-twin tables

The unchanged production tree was freshly run through the consolidated
production-step stage gate. The row-per-output-field artifact is
`stage_twin_baseline.json`, SHA-256
`9f59e60aa1e2257f027a2b5d1f056c918ad88d124983e8ce9122b126bf216ead`;
it is stamped at clean commit `37c5ef5585adf335f07c2185c051da2666d9a72b` and reports PASS. `B`, `A`, and
`D` below are BIT, AT-BAR-not-BIT, and DEBT.

Given NEMO's recorded entry, each cell is `B / A / D` row counts:

| kt | external | stage 1 | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 1 | 11 / 0 / 0 | 13 / 4 / 0 | 14 / 2 / 1 | 11 / 1 / 5 |
| 2 | 0 / 0 / 11 | 10 / 4 / 3 | 11 / 3 / 3 | 10 / 2 / 5 |

Model-chained entry:

| kt | external | stage 1 | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 1 | 10 / 1 / 0 | 8 / 4 / 5 | 5 / 5 / 7 | 3 / 5 / 9 |
| 2 | 0 / 0 / 11 | 3 / 0 / 14 | 0 / 0 / 17 | 0 / 0 / 17 |

The complete 17-field kt1-stage1 rows, the first stage in the walk, are:

| field | given NEMO entry | model chained |
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

The consumed kt1-stage1 momentum RHS is DEBT before the assignment: U has
17,400 unequal cells, V has 17,100, and both maxima are
`2.0121494123449567e-8`. Thus the U/V output rounding rows do not establish a
new owned statement. Conversely, all five TKE rows are BIT given NEMO entry
and DEBT only in the chain. Decision 41 classifies those closure residuals as
inherited; this round seeks their first upstream statement boundary only as a
possible measured compensator for the already-vetoed full-RHS candidate. It
does not promote TKE to an owner or a landing candidate.

## Missing record and acquisition contract

The admitted Round-59 stream contains TKE entry, assembled pre-sweep RHS and
post-sweep output, but no image after boundary assignment or after Langmuir.
Round 101 adds exactly those two images while repeating the three existing
boundaries for independent bitwise identity checks.

The new record is
`oracle_tke_statement_walk_kt00000002.bin`: 16-byte magic, thirteen native
32-bit header integers, then five `32*22*31` binary64 arrays in fixed order:
`en_entry`, `en_after_boundaries`, `en_after_langmuir`, `rhs_pre_sweep`, and
`en_post_sweep`. Its only accepted physical EOF is 873,028 bytes. The
consolidated gate refuses bad magic, clock/slot/domain/bounds/dtype headers,
truncation, trailing bytes, NaN/Inf, wrong producer/digest/name stamps, or a
one-ULP change in a duplicated consumed field. Entry, RHS and output must be
BIT against the independent Round-59 record.

The record is deliberately kt2 because that is the only accepted TKE operand
stream with independent entry/RHS/output duplicates and it directly precedes
the kt3 magnitude target. It is a discriminator only: it cannot authorize a
kt2 landing, relabel an inherited closure row as owned, or advance the ordered
stage campaign past the still-held kt1-stage1 W boundary.

The user-executed `run.sh` uses new target
`GYRE_OMIP_L2_P3_SM_R101TKEW`, clones `GYRE_PISCES`, copies the complete
Round-59 source card file by file, applies additive patches only, preserves
the ten-step namelist, proves both modified Fortran units with
`gfortran -fsyntax-only`, records the binary and producer commit, and performs
twin admission. `--admit-existing` never invokes `makenemo` or `mpirun`.
Every explicit or unexpected nonzero acquisition exit prints `REFUSE:`.

The clean committed preflight exited 0 and has SHA-256
`1e0af27d6fb1178632321ee15dea7e284d45eb00dd53d6b43c8739a2b1842783`.
The source-layout plant removes the post-Langmuir callback, prints
`REFUSE: source-layout plant removed the post-Langmuir boundary`, and exits
69; its log SHA-256 is
`83bfe35f44028c34b87114b0e030a07d0845621a111f09bc0e5797649ae2fc75`.
Header, truncation, stamp, one-ULP, byte-size and twin-consumption plants are
embedded fail-closed controls to run after the record exists.

## Frozen prediction status and magnitude

The preregistered prediction is **NOT YET MEASURED**: production is expected
to be BIT through `en_after_boundaries` and first non-bit after Langmuir. The
registered falsifiers remain live: an earlier boundary miss, a later RHS or
recurrence miss, or five BIT boundaries all refute that prediction. No
statement is named from the missing values.

No candidate exists, so the certified ladder and 30-day member were not
rerun. The immutable production values remain:

| headline | Round-96/97 before arm | Round 101 |
|---|---:|---:|
| kt2 T RMS | `1.4210854715202004e-14` | unchanged |
| kt2 S RMS | `2.1316282072803006e-14` | unchanged |
| kt2 U RMS | `2.7377110452773967e-12` | unchanged; first-over-bar |
| kt2 V RMS | `3.284922138989399e-12` | unchanged; first-over-bar |
| kt3 T RMS | `1.627497246303733e-4` | unchanged; magnitude target |
| kt3 S RMS | `6.327735185607253e-6` | unchanged |
| day-30 T RMS | `1.2397011295506804e-2 K` | unchanged; magnitude target |

No AT-BAR row can leave the bar and first-over-bar cannot move because no
shared numerical implementation changed. This is not an improvement claim.

## Testcase dispositions

| lane | disposition |
|---|---|
| GYRE stage twin | PASS on unchanged production; first owned stage remains kt1 stage1; TKE internal statement is unresolved pending the fixed record |
| GYRE kt=1--10 | no candidate; immutable first-over-bar remains kt2 U/V; per-row attribution proves full RHS alone already owns the 85-row/56-row veto |
| GYRE days 1--30 | no candidate; immutable day-30 T RMS retained |
| LOCK_EXCHANGE-zco | recorder is NEMO-only and no shared numerical statement changed; no new tank-fidelity claim |
| OVERFLOW-zps | recorder is NEMO-only and no shared numerical statement changed; partial-cell construction retained |
| DINO | **SHARED-STATEMENT RISK:** DINO executes the shared TKE program; no trajectory-neutrality claim, and the regional-cancellation warning remains in force |
| ORCA2 | **UNMEASURED-WITH-SPEC:** resolve its integrator, then record its native production-step TKE boundaries and inputs with the same stamp, EOF and one-ULP controls |

## Compiled-source basis

The exact Round-59 record-producing driver calls vertical physics before the
external mode and RK stages at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:168`. The resolved TKE
arm first computes shear and then calls `zdf_tke` at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90:317-337`.

Inside the compiled closure, `tke_tke` executes before `tke_avn` and the
existing Round-59 recorder closes only after both at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:190-200`. The first
routine writes its surface values at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:277-281` and its active
bottom-friction value at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:292-300`.

The active Langmuir arm culminates in the in-place `en` assignment at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:318-380`. The following
block forms the inverse Prandtl field, matrix and complete budget RHS at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:394-433`; the compiled
recurrences and terminal floor/mask are at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:466-483`. The requested
boundaries surround these executing statements; no dead source arm is cited.

## Review and verification

The required separate read-only Codex review was invoked against every change
since incoming commit `569599e61db0`, the receipt, the stage table and the
Rule-12 attribution, with an explicit request to return SHIP, HOLD or DO NOT
SHIP and to challenge the kt2-record scope. No reviewer model started. Its
verbatim terminal result is:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only
> file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file
> system (os error 30)

Therefore **independent review unavailable in-sandbox**; unavailability is not
approval. No physics diff is being shipped, and the round remains stopped for
the independently produced record. The review log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The focused final-tree suite passed **72 tests in 5.54 s**. It covers the
consolidated production-stage gate and new exact-EOF TKE reader, the Round-59
TKE operand machinery, twin admission, citation extraction/map auditing, and
all planted unit controls. Its log SHA-256 is
`83aa5d5f2e23b07ee17ac4b181a75b82ab1c2ccce30e8ec8c6faa37ed46d9064`.
`bash -n` also passes for the acquisition script; its clean preflight is the
gfortran proof recorded above.

The clean-tree citation gate finds and maps all 8/8 compiled-source citations,
with no unmapped citation, failed anchor, global map-audit failure or self-test
failure. Its JSON SHA-256 is
`92556b1b5cc60d566f536d8fa980bffd13dce15024b7b02fb1d0149a44cd3230`.
Shifting the Langmuir citation by two lines exits 1 with
`SYMBOL-NOT-AT-LINE`; the plant JSON SHA-256 is
`783182e7c4aa61a5b13000b2175a2472da1c8c03ce4977ccf4a323b6e02e0b16`.

GitHub issue #1455 is unavailable through this clone's local-only remote, so
no issue-update claim is made. No configuration or carried-state decision is
needed.

## OPEN — round 102

1. The operator runs the requested Round-101 acquisition. Admit an existing
   run with the same `run.sh`; do not rebuild an unchanged binary. Require the
   fixed 873,028-byte EOF, producer/binary stamps, three Round-59 duplicate
   rows, all record plants, the unchanged final restart and mesh, and standard
   twin admission before reading a new boundary scientifically.
2. Extend the existing consolidated stage gate, not a second harness, to
   expose the model's matching boundary values through the full production
   step. Keep the three labels `isolated-closure eager`, `isolated-closure
   JIT`, and `production step`; only the last may name a statement, and its
   one-ULP plant must fire through production.
3. Apply the frozen decision tree in compiled order: boundary block,
   Langmuir, RHS, recurrences. Name the first non-bit statement only after the
   preceding production boundary is BIT. Preserve any refuted prediction.
4. A numerical candidate is eligible only if the relevant production stage
   output closes and the complete 954-row Rule-12 gate passes. Do not bundle
   across stages, and do not advance beyond kt1 stage1 while its owned W row
   remains unresolved/held.
5. Keep kt3 T `1.627497246303733e-4 K` and day-30 T RMS
   `1.2397011295506804e-2 K` visible. Do not infer TKE ownership of either
   magnitude without the production-stage and trajectory measurements.
