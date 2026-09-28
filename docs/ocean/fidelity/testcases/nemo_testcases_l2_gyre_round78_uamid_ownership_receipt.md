# NEMO-testcases L2 GYRE round-78 U-midpoint ownership receipt

Date: 2026-09-13.

## Verdict

The preregistered prediction is **REFUTED**. The first non-bit live U row is
substep-1 `ubb_e`, not `un_e`: 6/580 wet faces differ, with maximum absolute
difference `8.470329472543003e-22`. The three coefficient rows, `un_e`, and
`ub_e` before it are bit-exact. The resulting `ua_e` differs at 2/580 faces by
`2.117582368135751e-22`.

The midpoint statement is not the owner. The extracted production helper is
bit-exact on all 50 recorded NEMO coefficient/operand sets, all record replays
are bit-exact, and every leaf of the ordinary production-jitted kt=2 trace has
the same digest before and after extraction. No numerical fidelity arm landed.
The round stops for the carried-state decision below.

## Record and source walk

The operator-admitted record was produced at
`6c0fe440340c1c1c7ad16d8cbf547d93264bee26`. Its payload SHA-256 is
`efff2a6ab74770890221790ac7b3c7d01d07bbf9ee3a520fdad1d99041be8a0b`;
the inherited admission remains 45/69 byte-identical, 24 classified changed,
and 281 admitted representatives. The round-78 reader additionally required
the record stamp, producer stamp, fp64, JIT, x64, clean worktree, and exact
ordinary seed.

The record producer's compiled branch selects the three midpoint coefficients,
forms `ua_e` from `un_e`, `ub_e`, and `ubb_e` in written left association, and
writes those exact operands and result at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:481-509`.
The measured first three substeps are:

| substep | coefficients | `un_e` | `ub_e` | `ubb_e` | `ua_e` |
|---:|---|---|---|---|---|
| 1 | bit | bit | bit | 6/580, `8.470329472543003e-22` | 2/580, `2.117582368135751e-22` |
| 2 | bit | 580/580, `3.032539284029834e-09` | bit | bit | 580/580, `5.40127088148319e-09` |
| 3 | bit | 580/580, `5.1234040916522955e-09` | 580/580, `3.032539284029834e-09` | bit | 580/580, `6.331943583964616e-09` |

Later rows are downstream once substep 1 is non-bit. V remains withheld until
the U owner is resolved.

## First statement and carried-state blocker

This is confirmation of the already-open round-50 barotropic-memory debt, not
a new attribution. NEMO conditionally initializes the six histories, seeds the
current external velocity directly from the selected baroclinic time level,
and zeros only its accumulators at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:339-378`.
After every external substep it rotates the absolute `ubb_e <- ub_e <- un_e`
arrays, with the corresponding V and SSH rotations, at the same compiled
source's `:783-795`. It reads and writes all six absolute before/twice-before
arrays directly in restart I/O at `:991-1018`.

legoesm instead defines `bt_hist` as six final-minus-history deviations at
`state.py:648-666`, reconstructs the absolute inputs by subtracting those
deviations from the new window's current value at
`barotropic_latlon_cgrid.py:2063-2085`, and forms the next deviations by
subtraction at `:2890-2901`. Those subtraction/reconstruction pairs cannot
recover bits already rounded away. The observed six-cell Ubb footprint and
maximum exactly reproduce round 50's live-memory result.

Changing those six carried fields is a carried-state representation choice,
which this campaign forbids without the user. Moreover, round 51's registered
raw-history substitution did not improve the kt=3 U/V maxima and worsened SSH
slightly, so the preserved raw-history-only arm remains **REFUTED** and held.
The next eligible implementation must pair NEMO's absolute histories with the
upstream window-boundary depth-mean/reconciliation owner and preregister the
pair; it may not simply revive the round-51 substitution.

## Controls and evidence

The original `null-live-un-e` control is **REFUTED** because its target is
already exact. After the committed addendum, that control refused with the
verbatim message `GATE FAILED: null-live-un-e plant target is already exact`
and exited 1. The corrected comparison-only `null-live-ubb-e` plant exited 1,
moved the first boundary to substep-1 `ua_e`, and left the shared helper exact.
The independent one-ULP `ua_e` plant exited 1 and made the shared-helper gate
non-bit. Thus the observed ownership and shared-statement checks can fail.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round78/`:

- `round78_pre_refactor_trace.json`, SHA-256
  `b53b546a1714b4894c2122e538676a5a4cb0346e20eb1b0a324898430bdfed5f`;
- `round78_uamid_walk_final.json`, SHA-256
  `77fabd5cf30c217efcec2eb0fff5c3669d8449389f24599517acd632420db8b9`;
- `round78_null_live_ubb_e_plant.json`, SHA-256
  `f90cb4009f346daaaad2d07c0ae101f5a67cab15951f48e54cb299e3d0cd654c`;
- `round78_shared_result_ulp_plant.json`, SHA-256
  `d4efe048613bb897a12f8f3659f601ed387fb6102604c405953a5a43b1c1b5b3`.

The required separate review command was run twice, first exactly as specified
and then with the documented ephemeral/read-only options. Both attempts failed
before reading the diff and produced no verdict. The decisive line, quoted
verbatim, was: `Error: failed to initialize in-process app-server client:
Read-only file system (os error 30)`. Therefore no `SHIP`, `HOLD`, or
`DO NOT SHIP` reviewer verdict exists to misrepresent. No numerical diff is
being shipped.

## Rule 12 and scope

| card/lane | disposition |
|---|---|
| GYRE-zco kt=1--10 | UNREACHED: no numerical arm; the complete ordinary kt=2 production trace is pre/post bit-identical |
| GYRE days 1--30 | UNREACHED: no numerical arm and no moved row; the recorded before arm was not replaced by a scratch toggle |
| LOCK_EXCHANGE-zco | EXECUTES the extracted AB3 statement; operation order is unchanged and no state/numerical arm landed |
| OVERFLOW-zps | EXECUTES the same extracted statement under its AB3-family filter; operation order is unchanged and no state/numerical arm landed |
| DINO | SHARED-STATEMENT RISK: the helper is shared and DINO retains 96--98% per-row cancellation risk; no DINO inference or moved row is claimed |
| ORCA2 | UNMEASURED-WITH-SPEC: resolve its compiled card; record every entry, coefficient, midpoint operand/result, metric transport, reciprocal metric, accumulator exit, normalization, and boundary handoff for kt=1--10; replay in source order; preserve every AT-BAR row and forbid an earlier first-over-bar boundary |

Focused tests cover the walk, strict signed-zero comparison, digest sensitivity,
left association under JIT, citation-map self-audit, and citation-gate plants.
The focused run passed 55/55 tests. The receipt citation gate passed all seven
citations with zero unmapped entries or map-audit failures; shifting the
compiled midpoint citation by two lines exited 1 with
`SYMBOL-NOT-AT-LINE`.
The GYRE ladder and year score were not run because the preregistration marks
them UNREACHED absent a numerical candidate. NEMO source/build/run, the year
harness, reconciliation gate, freshwater pair, #1484 guard, configuration,
stabilizers, and held manifests were untouched. GitHub issue #1455 could not be
read or updated because network access was unavailable in this sandbox.

## DECISION_NEEDED

May the next round replace deviation-form `bt_hist` with NEMO's six absolute
barotropic history arrays only as a preregistered pair with the upstream
window-boundary mean/reconciliation owner? **Pick: YES** -- NEMO identity
requires the absolute carried state, but the raw-history-only arm refuted in
round 51 must remain held.

## OPEN

The campaign is paused for that carried-state answer. If YES, first
preregister and measure the window-boundary pair that distinguishes NEMO's
direct history rotation from legoESM's post-solve mean shift; require the
substep-1 `ubb_e` row to become bit-exact without moving any prior exact row or
making first-over-bar earlier; then run GYRE kt=1--10 and the recorded-before
days 1--30 score plus both tanks, with DINO risk and the ORCA2 spec above. If
NO, record that the user has explicitly accepted a known six-cell
bit-exactness gap before continuing the magnitude-ranked `un_adv` owner.
