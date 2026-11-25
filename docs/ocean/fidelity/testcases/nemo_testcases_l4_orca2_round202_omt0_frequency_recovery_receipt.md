# ORCA2 round 202 — OMT-0 every-step record recovery

Date: 2026-10-09. Incoming tip: `55fc7fe0b3`. Preregistration:
`02dbac843`. Recovery implementation: `96bb42ee5`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round202/`.

## Result

**STOPPED_FOR_RECORD.** The operator's two round-200 NEMO runs reached normal
step-10 completion, but neither contains the preregistered kt=1..10 ladder.
Each target has only two restart shards, both named for step 1, and the resolved
log opens that file only at terminal step 10. The admission gate therefore
refuses at missing resolved restart step 2 before reading or scoring a field.

The cause is the run protocol, not NEMO physics. In list mode NEMO initialises
the next target from the first list entry, while frequency mode updates the
target from `nn_stock`; a file is opened one step before that target or on every
step only when `nn_stock == 1`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:94-119`). A list beginning at the
first model step cannot be opened one step earlier. The RK3 caller only writes
when `lrst_oce` says a restart is already open
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:271-271`), so the list never
advances; at terminal fallback the still-current step-1 name is opened. When a restart is legitimately
written, list mode closes it and advances at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:188-202`.

The executable gate now refuses a list whose first entry is the initial step or
whose entries do not leave an opening step. The replacement launcher uses the
compiled every-step frequency path: `nn_stock=1`, `ln_rst_list=.false.`, fresh
round-202 twin target names, unchanged OMT-0 physics, binary and inputs. It
reuses the previously measured step-11 month boundary; no longer-duration NEMO
run is requested. CPU preflight emits
`PASS_R202_OMT0_FREQUENCY_RECOVERY_PREFLIGHT` and
`ORCA2_ROUND202_OMT0_FREQUENCY_PREFLIGHT_READY`.

All future record and trajectory scores remain labelled **independent OMT-0**.
No given-entry result is mixed with them. The shipped rung-10 ORCA2 card, sea
ice, all six selectors and `unmeasured_features` are unchanged.

## Frozen prediction ledger

| prediction | verdict |
|---|---|
| R202-P1 existing record admits | **REFUTED.** Both twins lack steps 2..10; the gate refuses before payload scoring. |
| R202-P2 all admission controls bind | **UNMEASURED.** Clean admission stops before record plants; the new preflight controls fire. |
| R202-P3 explicit OMT-0 card resolves only five OFF modules | **UNMEASURED.** Card work is correctly blocked on record admission. |
| R202-P4 both legoESM ladders execute through kt=10 | **UNMEASURED.** No candidate trajectory was run. |
| R202-P5 no earlier retained-core debt than external SSH | **UNMEASURED.** No candidate trajectory was run. |
| R202-P6 candidate reaches kt=11 no earlier than NEMO | **UNMEASURED.** No candidate trajectory was run. |
| R202-P7 no shared production behavior changes | **CONFIRMED.** `git diff 55fc7fe0b3..HEAD -- packages` is empty. |
| R202-P8 instruments are non-vacuous | **PARTIAL.** Deck delta, live-module and adjacent-list plants fire; record plants await the new twins. |

## Retractions and controls

Round 200's claim that its two ten-step twins would provide 40 shards and 100
field comparisons is **RETRACTED**. Normal `STOP 0` proved only execution to
step 10; it did not prove restart coverage. The gate no longer accepts or
generates that adjacent list protocol, so the retraction lives in the tool.

The prior record refuses with the exact line `resolved log omitted restart
step 2`. The adjacent-list plant calls the same renderer with steps 1..10 and
must refuse on the compiled opening constraint. The frequency-mode deck test
requires step 10, stock frequency 1, explicit list mode false, no list payload,
unchanged surface cadence, and all five Decision-103 OFF arms still present.
The targeted gate suite passes 9/9.

## Review and validation

Independent read-only Codex review is unavailable in-sandbox. The required
separate invocation failed before reading the diff with `failed to initialize
in-process app-server client: Read-only file system (os error 30)`; no review
verdict is claimed.

The default citation audit passes 274 citations with zero failures and zero
unmapped citations. The round-202 audit passes all three citations; its planted
bad `stprk3.f90` anchor refuses with `SYMBOL-NOT-AT-LINE`. The final focused
tests pass 26/26. The required `tests/ocean/fidelity -n 12` battery was launched
exactly once and reached 97%; its only observed failure was the registered
pre-existing
`test_nemo_testcase_l2_gyre_round129_spread_floor_gate.py::test_record_backed_gate_passes`.
It then produced no output for the bounded long-tail wait and was interrupted
once (exit 130); no terminal battery PASS is claimed.

No NEMO source, stabiliser, safety threshold, tolerance, carried state,
physical selector or sea-ice setting was changed. ASKED choices: Decision
103's OMT-0 physics and ten-step record. UNASKED choices: empty.

## OPEN

The operator runs
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round202_omt0_frequency_acquisition/run.sh --run`.
The next round admits both every-step twins and the existing exact kt=11 month
boundary, then builds and scores the explicit OMT-0 card. OMT-1 does not begin
first.
