# ORCA2 round 200 — OMT-0 record-contract recovery

Date: 2026-10-09. Incoming tip: `cd9c3d234`. Preregistration:
`f87effef5`. Recovery implementation: `7015b2a44`; citation repair:
`250b4beef`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round200/`.

## Result

**STOPPED_FOR_RECORD.** Round 199's two-step OMT-0 smoke is valid and is
re-admitted by content, but its longer acquisition contract was impossible.
The corrected fail-closed launcher is committed and preflight-clean. It needs
the operator to run it before any OMT-0 card, ladder or first-non-bit claim is
made.

Every future score from this record is labelled **independent OMT-0**. The
shipped ORCA2 card, sea ice, all six selectors and `unmeasured_features` are
unchanged. `git diff cd9c3d234..HEAD -- packages` is empty.

## Loud correction to round 199

Round 199 claimed that `nn_stock` would create terminal restart sentinels at
steps 2, 12 and 96 while `ln_rst_list=.true.`. **That claim is retracted.**
NEMO initialises `nitrst` from the first list item and does not use `nn_stock`
in list mode (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:94-119`). It opens only
the current `nitrst` target (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:120-146`),
then closes it and advances to the next list item
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:188-202`). A terminal step not in
the list is not written.

The old gate's expectations for step-2, step-12 and step-96 files have been
removed from the executable gate, not merely corrected in prose. The two new
twins end at step 10 and list exactly steps 1..10, so all 40 rank-complete
shards close before normal `STOP 0`. The 96-step target retains the ten-entry
Decision-103 schedule `10,20,...,90,95`; it can validly close step 10 before
the measured safety boundary.

## The oracle's own boundary

The operator's round-199 run confirmed the two-step smoke, then the first
twelve-step target reached kt=11 and NEMO stopped itself. NEMO computes the
active SSH, U, V and salinity extrema at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stpctl.f90:176-184`, applies the compiled
20 m / 10 m/s / salinity / non-finite predicates at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stpctl.f90:243-250`, and reports the locations
before `ctl_stop` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stpctl.f90:293-316`.

The prior output is finite but over the velocity bound:

| kt=11 check | value | global location |
|---|---:|---|
| `abs(ssh)` maximum | 3.853 m | `(i,j)=(9,90)` |
| `abs(U)` maximum | 3.041 m/s | `(i,j,k)=(21,84,27)` |
| `abs(V)` maximum | **10.24 m/s** | `(i,j,k)=(22,84,27)` |
| salinity minimum | 21.58 PSU | `(i,j,k)=(37,133,1)` |
| salinity maximum | 37.25 PSU | `(i,j,k)=(35,19,5)` |

The recovery gate requires those printed values and locations exactly, requires
the V value to exceed 10 m/s, requires the `output.abort` file and process exit
through `stp_ctl`, and refuses a clean 96-step continuation or any different
stop. It admits only the closed step-10 month checkpoint. Steps 20..95 remain
**UNMEASURED WITH SPEC: NEMO OMT-0 stops at kt=11**. No threshold, stabiliser,
source or binary is changed.

## Corrected acquisition contract

The launcher reuses the admitted smoke and stages three new target names:

1. twin A, `nn_itend=10`, restart list `1..10`;
2. twin B, the byte-identical protocol;
3. month-boundary target, `nn_itend=96`, restart list
   `10,20,...,90,95`, expected to stop through the exact kt=11 boundary.

The twins must provide 40 files and 100 array-equal field comparisons across
fp64 finite `sshn/un/vn/tn/sn`. The month step-10 checkpoint must be
array-equal to both twins in 20 more field comparisons. Every file is parsed
from its own NetCDF `kt`, names, dtype, dimensions, shape and payload. Later
month files and fictional terminal sentinels are neither expected nor filled.

Preflight results: `PASS_OMT0_DECK`,
`PASS_R200_OMT0_RECOVERY_PREFLIGHT`, `PASS_R200_OMT0_SMOKE_REUSE`, and
`ORCA2_ROUND200_OMT0_RECOVERY_PREFLIGHT_READY`. The canonical deck SHA-256 is
unchanged at `9279c638a9fe50b551fcca8dc62dc1c19c0fc2c55bdcae7f3501a8afeb5ca8ac`.
The extra-deck, live-module and over-capacity-list plants fire. The unit-level
wrong-oracle-stop plant fires. Record-dependent plants remain unmeasured until
the operator run.

## Prediction ledger

| prediction | verdict |
|---|---|
| R200-P1 prior smoke re-admits | **CONFIRMED**: clean completion, exact five OFF selectors, zero physical deck delta and exact zero-flux provenance. |
| R200-P2 two ten-step twins complete and compare bitwise | **UNMEASURED**: new targets do not exist until operator execution. |
| R200-P3 the month target reproduces the exact kt=11 oracle stop after closing step 10 | **UNMEASURED**: prior evidence motivated and froze the prediction; the new target must reproduce it. |
| R200-P4 no month state exists after step 10 | **UNMEASURED WITH SPEC** pending P3; the gate explicitly lists steps 20..95 unavailable only after exact reproduction. |
| R200-P5 bookkeeping only | **CONFIRMED**: no package or card diff. |
| R200-P6 controls bind | **PARTIAL**: four pre-record controls fire; ten record-dependent plants await output. |

## Validation and review

Focused round-200 plus citation-map tests pass 26/26. Shell syntax, Python
compilation and `git diff --check` pass. The required single
`tests/ocean/fidelity -n 12` battery collected 2,961 tests, reached 97%, and
was interrupted once after 15 minutes without output. It has no terminal JUnit
and no PASS claim. Before the quiet tail it reproduced four registered
pre-existing reds: SI3 scalar-math provenance, the GYRE spread-floor record,
the 13-driver scoped-stamp ratchet and the 13-emitter worktree-stamp ratchet.
All four were rerun together and reproduced; none names a round-200 file.

The default citation audit passes 274 citations with zero failures, unmapped
citations or failing map entries. This receipt passes all six distinct
compiled citations with the same zero counts.
Shifting the `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stpctl.f90:243-250` span by two
lines fires as required with `SYMBOL-NOT-AT-LINE`, exit 1. The three reports
are stored as `citations_default.json`, `citations_round200.json` and
`citations_round200_plant.json` under the evidence root.

Independent review unavailable in-sandbox: the required separate
`codex exec --sandbox read-only` invocation failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system
(os error 30)` (evidence SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`).
No independent verdict is claimed.

No configuration choice, source modification, stabiliser, tolerance,
carried-state change or sea-ice change was introduced. ASKED choices: Decision
103's OMT-0 physics and record schedule. UNASKED choices: empty.

## OPEN

The operator runs
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round200_omt0_recovery_acquisition/run.sh --run`.
The next round admits the two twins and exact kt=11 stop, then builds the
explicit OMT-0 legoESM card using the already-landed OFF arms, verifies mesh
identity, scores both ten-step ladders, and names the first non-bit statement.
The independent month is recorded as stopped at kt=11 unless the new run
refutes that frozen boundary. OMT-1 does not begin first.
