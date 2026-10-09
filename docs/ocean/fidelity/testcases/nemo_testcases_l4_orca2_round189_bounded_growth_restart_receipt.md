# ORCA2 round 189: bounded growth-restart correction

Date: 2026-10-08

Status: **STOPPED_FOR_RECORD**

Claim labels: every rung-0 scientific number referenced here is
**independent**. No given-entry rung-7 number is mixed into the table or
verdict. Sea ice and the shipped rung-10 card are untouched.

## Outcome

Round 188's operator-run acquisition stopped before the first time step. The
failure was the acquisition deck, not ocean physics: it assigned 11 values to
a compiled ten-element restart-list array. The round-188 READY claim is
retracted in its own receipt, its launcher now refuses, and the gate rejects
the actual failed deck.

The first round-189 repair, frequency mode with `nn_stock=95`, is also
**REFUTED before a NEMO run**. Although it would put step 95 before terminal
step 96, the compiled surface-boundary consistency check rejects an odd
restart frequency under the resolved `nn_fsbc=2`. That failed preflight is
kept as evidence rather than silently replaced.

The corrected acquisition is preflight-ready under fresh round-189 targets.
It uses the bounded explicit list `95,96`, while steps 10..90 continue to come
from the admitted round-186 twins. The operator has not run the corrected
twins, so step 95 remains **UNMEASURED-with-spec** and the round stops for the
record.

## Executed-source diagnosis

| order | compiled statement | measured consequence |
|---:|---|---|
| 1 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/in_out_manager.f90:51-51` | `nn_stocklist` has exactly ten elements. |
| 2 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domain.f90:323-325` | The round-188 config read returns `iostat=5010`; its resolved print contains only the first ten supplied values, through 95. |
| 3 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:341-349` | Both experiment length and frequency restart cadence must be divisible by `nn_fsbc`; this kills `nn_stock=95` even though the 96-step experiment length is valid. |
| 4 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:94-119` | List mode starts at 95, opens that file at step 94, then the terminal fallback can open 96 only after step 95 closes. |
| 5 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:188-202` | Writing 95 advances the list cursor to 96, preventing the step-95 filename from being reopened at terminal step 96. |

The observed round-188 `ocean.output` contains no time-step line or restart
file. The actual failed deck now produces
`ROUND188_ACTUAL_DECK_REFUSED compiled nn_stocklist capacity 10 exceeded` in
the repaired gate.

## Controlled deck delta

The corrected renderer reports `physical_delta=[]`. The complete assignment
diff is:

| field | admitted rung-0 source | round-189 acquisition |
|---|---:|---:|
| `nn_itend` | 240 | 96 |
| `nn_stock` | 240 | 96 (ignored in list mode) |
| `ln_rst_list` | absent, resolves false | `.true.` |
| `nn_stocklist` | absent, resolves ten zeros | `95,96` followed by eight inherited zeros |

No physical assignment, input, binary, forcing, initial state, selector, or
sea-ice setting moves. The launcher uses new target names, pins the committed
gate and preregistration by content SHA256, and reuses the admitted round-186
step-10..90 twins as an explicit prefix.

Artifacts:

- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round189/acquisition/namelist_growth_frequency_preflight` — **REFUTED** frequency-mode preflight.
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round189/acquisition/namelist_growth_bounded_list_preflight` — corrected deck.
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round189/acquisition/deck_preflight.log` — `RUN_PROTOCOL_ONLY`, `physical_delta=[]`.
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round189/acquisition_preflight_final.log` — READY line with the fresh target.

Launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round189_growth_acquisition/run.sh`.

## Frozen prediction ledger

| prediction | outcome | evidence |
|---|---|---|
| R189-P1 | **CONFIRMED** | Round 188 stops on namelist `iostat=5010`; the resolved print ends after the tenth list value and no time step or restart exists. |
| R189-P2 | **REFUTED** | Compiled `sbcmod` rejects `nn_stock=95` because 95 is not divisible by `nn_fsbc=2`. The failed preflight is retained. |
| R189-P2b | **CONFIRMED at preflight** | The corrected rendered deck contains only the bounded list `95,96`, reports `physical_delta=[]`, and changes only run protocol. |
| R189-P3 | **UNMEASURED-with-spec** | Corrected twins do not exist. Admission requires complete finite step-95 and step-96 payloads on both ranks with exact twin equality. |
| R189-P4 | **UNMEASURED-with-spec** | The admitted prefix is pinned, but the completed 10..95 table is withheld until the new step-95 payload admits. |
| R189-P5 | **UNMEASURED-with-spec / standing prior result** | No passive scientific ladder was rerun; the completed-RHS vertical-average boundary is neither promoted nor withdrawn. |
| R189-P6 | **PARTLY CONFIRMED** | Eight focused controls pass, including the oversized-list and disabled-list plants. Real-record plants run only after the operator produces both twins. |

## Validation and review

The focused growth-record suite passes 8/8. The corrected preflight passes;
the old launcher refuses with exit 78. The actual round-188 deck fails the new
capacity check. No package/model file changed, so no ORCA2, GYRE, DINO, tank,
or month trajectory could move in this record-protocol round.

The required read-only `codex exec` review was attempted and returned
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. The operator runs the round-189 launcher. Admission combines the immutable
   round-186 steps 10..90 with new, rank-complete twin payloads at 95 and 96.
2. Admit step 95, complete the frozen table, and confirm or refute step 10 as
   the first coarse >10x boundary.
3. Resume the already-named completed-RHS vertical-average statement only
   under its standing shared GYRE gate.
4. Re-test the private halo/V-transport unit only after that upstream
   statement is resolved. No partial operand may land.
5. Rung 7 and sea ice remain untouched; sea ice stays exactly in the card's
   existing `unmeasured_features` declaration.
