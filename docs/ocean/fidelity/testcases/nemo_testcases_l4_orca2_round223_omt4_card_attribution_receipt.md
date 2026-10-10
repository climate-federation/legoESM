# ORCA2 round 223 — OMT-4 card and tracer-advection attribution

Date: 2026-10-10. Frozen base: `272ff3437`. Preregistration commit:
`808b1ac2b`. Status: **LANDED** (gate-local record and OMT-4 card; no
production physics change). Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round223/`.

This round changes no `packages/` file, shipped rung-0/rung-10 card, carried
state, stabiliser, sea-ice selector, or `unmeasured_features` tuple. Every
trajectory number is labelled either **independent** or **given NEMO's
entry**.

## Existing record: admitted, not rerun

The operator's smoke, uninstrumented ten-step calibration, both rank-complete
P3 twins, and 96-step month all completed. The first admission refused the
step-95 restart because both rank shards report `kt=0`. Inspection after the
preregistration showed that both files have zero time records and zero payload
values for every scored field. They are terminally truncated file definitions,
not step-95 states.

The record build opens a restart named from the current `nitrst` one step
before its scheduled write at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/restart.f90:94-146`. A write closes
the file and advances the fixed-size list cursor only when `kt == nitrst` at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/restart.f90:153-201`. With ten list
slots ending at 95 and `nn_itend=96`, the cursor stays at 95; the terminal-open
arm reopens that filename at step 96 with no matching write. The time-step
payload is itself conditional through `iom_rstput` at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/daymod.f90:405-417`.

The repaired checker reuses round 186/187's committed self-describing NetCDF
classifier. It admits valid rank-complete month restarts 10,20,...,90 and
records both step-95 shards as
`TERMINAL_REOPEN_TRUNCATED_COMPLETED_RESTART`. Admission also parses 80 frames
per twin, makes 400 array-equality comparisons, and checks four kt=10 terminal
restarts byte-for-byte. Status:
`PASS_R222_OMT4_ENTRY_STAGE_AND_MONTH_RECORD`. NEMO was not rebuilt or rerun.

## Exact OMT-3 -> OMT-4 edge

The gate-local card changes one and only one model-config field:
`tracer_advection = none -> fct2`; every other OMT-3 field is structurally
equal. The compiled deck reads and validates the FCT selection at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:592-602` and
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:630-647`. In the live RK3
program, stages 1--2 take the centred branch and stage 3 dispatches FCT at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:497-540`. A planted
`tracer_advection=none` edge refuses.

## Baseline ladders

Both labels begin from bit-identical T, S, u, v, and ssh arrays and complete
40 checkpoints / 200 scored rows on CPU/JIT/fp64/libm. Their numerical scores
are identical. The first non-bit checkpoint remains kt=1 stage-1 T; this is
carried debt, not a claim that FCT executes before that boundary.

| labelled state | row | rms | maximum |
|---|---|---:|---:|
| independent | kt1 stage1 T | 1.1375785057897583e-05 K | 0.0013606315900794863 K |
| independent | kt1 stage1 ssh | 0.006238271974498368 m | 0.1310917645484672 m |
| independent | kt10 stage3 T | 0.0021565586189475658 K | 0.9614413148261045 K |
| independent | kt10 stage3 S | 0.00105615847955814 PSU | 0.21974903918725985 PSU |
| independent | kt10 stage3 u | 0.0029816064573501967 m/s | 0.28996722106074313 m/s |
| independent | kt10 stage3 v | 0.003312208051058466 m/s | 1.1967701542015403 m/s |
| independent | kt10 stage3 ssh | 0.024352931961325045 m | 0.3949686188367629 m |
| given NEMO's entry | kt1 stage1 T | 1.1375785057897583e-05 K | 0.0013606315900794863 K |
| given NEMO's entry | kt1 stage1 ssh | 0.006238271974498368 m | 0.1310917645484672 m |
| given NEMO's entry | kt10 stage3 T | 0.0021565586189475658 K | 0.9614413148261045 K |
| given NEMO's entry | kt10 stage3 S | 0.00105615847955814 PSU | 0.21974903918725985 PSU |
| given NEMO's entry | kt10 stage3 u | 0.0029816064573501967 m/s | 0.28996722106074313 m/s |
| given NEMO's entry | kt10 stage3 v | 0.003312208051058466 m/s | 1.1967701542015403 m/s |
| given NEMO's entry | kt10 stage3 ssh | 0.024352931961325045 m | 0.3949686188367629 m |

The admitted NEMO month completes step 96 and retains scored restarts through
step 90; step 95 is unavailable only because of the terminal overwrite above.
No legoESM month is claimed in this card-landing round.

## Complete vector-unit arm: tracer advection owns the interaction

The complete round-217 unit (Ve_rhs/ssvmask fold pair, seven-array
association, and materialised V transport) is applied atomically. For both
labels, kt=1--7 complete and kt=8 refuses before producing its first
checkpoint:

`raw-mesh e3w_int must contain only finite values > 0`.

OMT-2 (+drag) and OMT-3 (+momentum LDF) completed the same arm through kt=10;
the OMT-4 baseline also completes. OMT-4 is therefore the first rung on which
the unit recreates the registered kt=8 live-W-thickness refusal. This
**CONFIRMS** tracer advection as the module carrying the compensating
interaction and exonerates drag and momentum LDF. It does not identify an FCT
statement yet and does not authorize a partial landing. Because the candidate
does not complete, no Decision-96 toward/away vote is manufactured.

Artifacts and SHA-256:

- `omt4_before.json`: `73892897d08e287590435b741e230d341552768364313e450ea100783e66d3c8`
- `omt4_atomic.log`: `973bbd48ba821edff92502b57124cf1d6d560d6dd4acccef930d646bc29f90af`
- `omt4_atomic_given.log`: `8ffca39902e4b0eb621b4de9a9bd3004c98ce82c8a836a1a1fcf4436efefde9a`
- admitted record JSON: `874b2e688e5ebd2fbf6f72c9de279eb11ec552164bba41f170528ada6742eb8e`

## Preregistration dispositions

| ID | disposition |
|---|---|
| R223-P1 | **CONFIRMED**: both step-95 shards have `kt=0` and zero scored payloads, exactly the terminal-reopen signature. |
| R223-P2 | **CONFIRMED**: the existing record admits with 80 frames/twin, 400 equal comparisons, and valid month steps 10--90; NEMO was not rerun. |
| R223-P3 | **CONFIRMED**: the card has one config delta and both labelled baselines complete kt=1..10. |
| R223-P4 | **CONFIRMED**: the atomic unit recreates the live-W refusal during kt=8 for both labels. |
| R223-P5 | **CONFIRMED**: the terminal-overwrite and card-edge planted violations fire; inherited acquisition plants remain enforced. |

## Validation and review

The focused acquisition/card/citation run collected ten tests: nine passed,
then the new citation test caught two non-unique anchors. After pinning those
anchors, the failed ID passed in isolation, so the final tree is 10/10 for the
focused set. This run also executes the new terminal-overwrite violation and
proves it raises rather than classifying a plausible empty file as data. The
combined log SHA-256 is
`7550115b2d93b9bc177e76ef11ba68c797adfe6ade01606cd5237ea0759a0bd6`.

On clean commit `1b93c1c33`, the round citation gate passes six citations and
the cumulative default gate passes 274; both have zero failures, unmapped
citations, or map-audit failures. Shifting the mapped restart range by two
lines makes the round gate fail with `SYMBOL-NOT-AT-LINE` and exit 1. The
round/default/plant JSON SHA-256 values are
`377e051476139e3f46deb3670b019073d19b1bbe99a3939c130322dca1593fa1`,
`0c048355bd6c5dde24403ba1d1034b7cc20571f062a67857c1e87d23e06929f0`,
and `47091ec415321490d00e904f345b44a8ec63ce6ad0b2867cd4697385240000c7`.

Independent review was attempted with `codex exec --sandbox read-only` and
exited 1 before reading the diff: `failed to initialize in-process app-server
client: Read-only file system (os error 30)`. **Independent review unavailable
in-sandbox**; this is not a PASS. The review log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The prescribed `tests/ocean/fidelity -n 12` battery collected 3,083 tests and
reached 99%. It recorded 3,062 passed, seven skipped, and four registered
pre-existing failures: the GYRE round-129 spread-floor record stamp,
allow-dirty scope, worktree-stamp ratchet, and SI3 scalar-math provenance
gate. The remaining ten tests were unclassified when every real pytest process
disappeared without a terminal summary; the idle wrapper was interrupted and
the battery was not relaunched. It is not called PASS. Log SHA-256:
`cb79480fdfdbdd20ce1ccf642361b3a411d19933c3f25d8d8cbfadae50316636`.

No `packages/` file changed, so GYRE, DINO, lock-exchange, overflow, and the
shipped ORCA2 cards cannot execute a changed statement. Their trajectories are
unchanged by construction; this round lands only the record interpretation
and gate-local OMT-4 card.

## OPEN

Before OMT-5, walk the OMT-4 tracer-advection interaction offline from passive
completed states. In compiled order, separate the centred stage-1/2 path from
the live stage-3 FCT path, then find the first statement whose atomic-unit
output departs the OMT-4 NEMO record and precedes kt=8's invalid W thickness.
The dispatcher is cited above; no in-executable observer is permitted. The
complete cancelling unit remains atomic and private until that partner is
named and a full Decision-96 census completes. Sea ice and the card's
`unmeasured_features` tuple remain untouched.

ASKED choices: Decisions 103, 109, and standing Decision 96. UNASKED choices:
empty.
