# ORCA2 round 189 preregistration — bounded restart-list correction

Date: 2026-10-08. Base: `03d7c0f7e75d08d6655e960099e8b4cb12ba3b8f`.
Every scientific number remains **independent**: the rung-0 card starts from
its own corrected, bit-exact initial state. No given-entry rung-7 number is
mixed into the tables.

## Observed failure, not a prediction

The operator ran round 188's terminal-sentinel launcher. NEMO stopped during
namelist initialisation with exit 123 and no time step or restart payload:

```text
misspelled variable in namelist namrun (cfg) iostat = 5010
```

The rendered deck supplied 11 values to `nn_stocklist`, but the executed
binary declares `nn_stocklist` as a fixed ten-element integer array at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/in_out_manager.f90:51`. The namelist reader at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/domain.f90:323-325` therefore rejects the deck.
This measured failure is retained and is not presented as a preregistered
result below.

## Frozen scope

Round 189 corrects the existing growth-record renderer and checker, disables
the invalid round-188 launcher, and adds a new fail-closed launcher with fresh
twin names under the round-189 evidence directory. It reuses the admitted
rung-0 binary, physical deck, and inputs. It requests only the still-missing
step-95 payload by setting `nn_itend=96` and `nn_stock=95` with explicit-list
mode absent. Steps 10 through 90 remain the already-admitted round-186 record
and are not rerun. Step 96 is a checked terminal sentinel and is excluded from
the scientific growth table.

No model file, physical configuration, forcing, carried state, stabiliser,
public selector, sea-ice selector, or `unmeasured_features` entry changes. No
in-executable observer is authorised. The six sea-ice selectors stay exactly
as carried by the ORCA2 card.

## Compiled protocol

At `ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:94-119`, frequency mode sets
`nitrst=95` at step 1 and opens the step-95 file at step 94. After step 95 is
written and closed at `restart.f90:188-202`, step 96 recomputes and caps the
next frequency target to `nitend=96`, so the terminal fallback opens a
distinct step-96 file. This is a run-protocol correction, not a scientific
configuration choice.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R189-P1 | Round 188 failed solely because 11 values exceeded the compiled ten-element `nn_stocklist`. | Both rank logs contain namelist `iostat=5010`; the resolved print contains exactly ten values through 95; no time step or restart exists. | Any time step or restart payload exists, or another earlier error is present: **REFUTED**; reconcile before changing the gate. |
| R189-P2 | The frequency-mode replacement is run-protocol-only. | Assignment inventory differs only in `nn_itend: 240→96` and `nn_stock: 240→95`; `ln_rst_list` and `nn_stocklist` are absent; `physical_delta=[]`. | Any other assignment moves: **REFUTED**; launcher refuses. |
| R189-P3 | Frequency mode preserves step 95 and writes step 96 separately. | Both twins and ranks contain complete finite restarts stamped 95 and 96; each step is twin-bit-exact. | Missing, truncated, misstamped, or moved payload: **REFUTED**; stop on the exact record defect under a new target. |
| R189-P4 | Combining the new step-95 payload with the already-admitted steps 10..90 completes the frozen table without moving prior evidence. | Step 95 admits; the checker identifies round 186 as the immutable prefix; step 10 recalibrates bit-exactly; step 10 remains the first strict >10x coarse boundary. | Prefix identity or boundary changes: **REFUTED**; retain and report the measured result. |
| R189-P5 | The completed-RHS vertical average remains the first source-ordered non-bit statement at kt=1 stage 1. | Existing passive ladder/replay remains unchanged; no statement claim is inferred from restart protocol. | Changed passive result: **REFUTED**; reconcile before citing either result. |
| R189-P6 | Every new correction check is non-vacuous. | Plants for explicit-list mode, wrong frequency, missing rank, twin ULP, prefix calibration, hidden deck delta, missing sentinel, truncated sentinel, and wrong sentinel header all fire. | Any plant stays green: instrument invalid; quote no record or scientific verdict. |

The round ends `STOPPED_FOR_RECORD` if the operator-run twins are unavailable.
If they admit, the receipt completes the growth table and resumes the already-
named statement only under its standing shared GYRE gate.

## Committed correction before replacement measurement

R189-P2 as written above is **REFUTED before any NEMO run**. The first repaired
preflight used frequency mode with `nn_stock=95`, but the compiled consistency
check at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:341-349` also rejects a
frequency that is not divisible by `nn_fsbc=2`. The preflight artifact is kept
as a failed prediction; it is not an admissible acquisition deck.

The corrected frozen replacement, R189-P2b, uses explicit-list mode with only
`nn_stocklist=95,96`. Two values fit the compiled ten-element array, step 95 is
written and closed before step 96, and steps 10..90 continue to come only from
the admitted round-186 prefix. The reference namelist initializes unused list
slots to zero; they are unreachable because the run ends after 96. R189-P2b is
confirmed only if the rendered assignment inventory differs by `nn_itend=96`,
`nn_stock=96`, `ln_rst_list=.true.`, and `nn_stocklist=95,96`, with
`physical_delta=[]`. Any other change or any resolved list other than
`95,96,0,0,0,0,0,0,0,0` refutes it. All other predictions retain their
original meaning, with “frequency mode” read as this bounded two-entry list.
