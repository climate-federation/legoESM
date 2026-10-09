# ORCA2 round 188 preregistration — restart-list terminal sentinel

Date: 2026-10-08. Base: `b79a200ae8b7f93e1c26717b1393d4dca45e9dac`.
Every scientific number remains **independent**: the rung-0 card starts from
its own corrected, bit-exact initial state. No given-entry rung-7 number is
mixed into the tables.

## Prior evidence, not a prediction

The operator ran round 187's replacement acquisition. NEMO stopped before its
first time step because a 95-step experiment is not divisible by the resolved
`nn_fsbc=2`; no restart payload was produced. This observed refusal is retained
as evidence and is not presented below as a preregistered result.

The executed compiled source makes the correction mechanical. With restart
output enabled, `sbc_init` rejects an experiment length not divisible by
`nn_fsbc`. After a listed restart is written, `rst_write` advances `nitrst` to
the next list member. Round 188 therefore keeps the even 96-step protocol and
adds step 96 as a terminal sentinel after the scientific checkpoint at step
95. Step 96 is run-protocol evidence only and is not added to the frozen
scientific growth table.

## Frozen scope

Round 188 changes only the growth-record renderer, its checker, focused tests,
and a new fail-closed acquisition launcher with new target names under the
round-188 evidence directory. The launcher reuses the admitted rung-0 binary,
deck inputs, and exact restart reader. It requests checkpoints
`10,20,...,90,95,96`, runs through step 96, admits/scientifically scores only
`10,20,...,90,95`, and separately proves that the terminal file is a complete
step-96 restart. Both ranks and two independent twins are required.

No model file, physical configuration, forcing, carried state, stabiliser,
public selector, sea-ice selector, or `unmeasured_features` entry changes. No
in-executable observer is authorised. The six sea-ice selectors stay exactly
as carried by the ORCA2 card.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R188-P1 | The terminal-sentinel deck is a run-protocol-only delta from rung 0. | Assignment inventory differs only in `nn_itend=96`, `nn_stock=96`, `ln_rst_list=.true.`, and `nn_stocklist=10,...,95,96`; all physical assignments are identical. | Any other assignment moves: **REFUTED**; launcher refuses. |
| R188-P2 | Appending checkpoint 96 prevents the step-95 overwrite. | Both twins and ranks contain complete finite restarts with `kt=95` and `kt=96`; step 95 is twin-bit-exact and step 96 is checked but excluded from the scientific table. | Missing/truncated/misstamped payload or any twin ULP: **REFUTED**; stop on the exact record defect under a new target. |
| R188-P3 | The admitted steps 10 through 90 remain unchanged and step 10 remains the first strict >10x coarse growth boundary. | Step-10 calibration and all twin comparisons are exact; the completed table selects step 10 using the frozen `2e-10` floor. | Any earlier payload moves or the selector changes: **REFUTED**; retain and report the measured boundary. |
| R188-P4 | The completed-RHS vertical average remains the first source-ordered non-bit statement at kt=1 stage 1. | The existing passive ladder/replay still closes and names the same statement; no new scientific claim depends on the terminal checkpoint. | A changed passive result: **REFUTED**; reconcile before citing either result. |
| R188-P5 | Every new terminal-sentinel check is non-vacuous. | Plants that remove the sentinel, truncate its payload, or substitute 95 for its header all fire, in addition to the existing missing-rank, twin-ULP, calibration, and hidden-deck plants. | Any plant stays green: instrument invalid; quote no record or scientific verdict. |

The round ends `STOPPED_FOR_RECORD` if the operator-run twins are not yet
available. If they are available and admit, the receipt records the completed
growth table and resumes the already-named statement only under its standing
shared GYRE gate.
