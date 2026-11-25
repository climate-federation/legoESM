# ORCA2 round 203 — OMT-0 passive-frame record recovery

Date: 2026-10-09. Incoming tip: `4874e52ee0`. Preregistration:
`430b45303`, with the frozen fallback addendum at `02cfe2621`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round203/`.

## Result

**STOPPED_FOR_RECORD.** The operator's round-202 run proved that the proposed
every-step restart protocol is invalid for this deck. The files it left are
rank-complete NetCDF headers with `kt=1`, but none contains `sshn`, `un`, `vn`,
`tn` or `sn`. They are not a state record and no payload from them is admitted.

The owner is compiled and exact: NEMO requires restart frequency `nn_stock` to
be divisible by the unchanged surface cadence `nn_fsbc` and stops otherwise
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:341-348`). Round 202 selected 1
against 2, producing the observed `sbc_init` refusal before integration. The
old every-step renderer now refuses this combination, so the retraction lives
in the tool rather than only in this receipt.

The replacement acquisition is committed and preflight-clean. It reuses the
already admitted additions-only round-90 frame binary with the exact OMT-0
deck. NEMO records `Nbb` before surface or stage work
(`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:92-108`), then the
completed stage-1 state (`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:214-217`)
and completed stage-2/3 states
(`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:221-233`). Each of two
instrumented twins must produce 80 self-describing files: two MPI ranks,
steps 1..10 and stages 0..3, with T/S/u/v/ssh in fp64. The gate compares all
400 twin field payloads with `np.array_equal`.

A third run uses the uninstrumented scalar-math binary and the supported single
restart target `(10,)`. NEMO opens a target one step before it is written
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:94-119`) and closes it at the target
before selecting the next list member
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:188-202`). Both instrumented
terminal restart shards must therefore be byte-identical to the uninstrumented
calibration. This is the additions-only predicate on the OMT-0 deck itself.

All results are labelled **independent OMT-0**. The shipped rung-10 card, sea
ice, all six selectors and `unmeasured_features` are unchanged.

## Prediction ledger

| prediction | verdict |
|---|---|
| R203-P1 cadence-refused step 1 is a complete state | **REFUTED.** Both rank files have `kt=1` but all five required fields are absent. |
| R203-P2 second cadence refusal reproduces step 1 | **SUPERSEDED before execution.** Header-only files cannot be state twins. |
| R203-P3/P4 staggered list runs form calibrated twins | **SUPERSEDED before execution.** The admitted passive frame writer gives the whole entry/stage topology directly and retains an uninstrumented terminal calibration. |
| R203-P5 controls bind | **PARTIAL.** Cadence, additions-only source, inventory and missing-frame controls fire in preflight/tests; record-backed header/payload/twin/terminal/binary plants await acquisition. |
| R203-P6 no production behavior changes | **CONFIRMED.** `git diff 4874e52ee0..HEAD -- packages` is empty. |
| Frozen frame fallback: two 80-frame twins, 400 exact field comparisons and four byte-identical terminal checks | **UNMEASURED.** New targets do not exist before operator execution. |

The exact OMT-0 deck preflight reports only Decision 103's five OFF selections.
The frame preflight reports 80 expected frames per twin, four additions-only
call sites, five fields and zero removed source lines. The prior failed target
passes the new `PASS_R203_R202_HEADER_ONLY_REFUSAL` classification. The clean
launcher ends with `ORCA2_ROUND203_OMT0_FRAMES_PREFLIGHT_READY`.

## Review and validation

Independent review is unavailable in-sandbox. The separate
`codex exec --sandbox read-only` invocation failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system
(os error 30)`; no review verdict is claimed.

Focused OMT-0 record and frame-gate tests pass **16/16**. Shell syntax, Python
compilation and `git diff --check` pass. The default citation gate passes all
274 citations with zero failures or unmapped citations; this receipt passes
all six citations, and a shifted `sbcmod.f90` plant refuses with
`SYMBOL-NOT-AT-LINE`.

The required single `tests/ocean/fidelity -n 12` battery reached 97%; after
the pytest processes had exited its PTY stopped emitting before the terminal
summary, so the stale PTY was closed. The two observed reds reproduced in one
isolated two-test run while this receipt was uncommitted: the clean-worktree
layout plant refused at its earlier dirty-tree guard (`63` rather than expected
`69`), and the pre-existing stamp-scope ratchet listed 13 VORTEX drivers. Once
the receipt was committed, the layout plant passed **1/1** in isolation. The
sole remaining red is the known stamp-scope ratchet; it does not name or
execute a round-203 file, and no unrelated ratchet is changed in this
stopped-for-record round.

No NEMO source is modified. No configuration choice, surface cadence,
stabiliser, threshold, tolerance, carried state or sea-ice field changes.
ASKED choices: Decision 103's OMT-0 physics and rank-complete ten-step record.
UNASKED choices: empty.

## OPEN

The operator runs
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round203_omt0_frames_acquisition/run.sh --run`.
The next round admits the two frame twins and uninstrumented calibration, then
builds the explicit OMT-0 card, scores both ten-step ladders and names the first
non-bit statement. OMT-1 does not begin first.
