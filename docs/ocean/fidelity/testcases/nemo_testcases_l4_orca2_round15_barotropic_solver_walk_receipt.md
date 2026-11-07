# NEMO testcase Lane 4 — ORCA2 card round 15 barotropic solver walk receipt

Date: 2026-09-25 (measurement began 2026-09-24)

Parent: `8f58d909ecdc068412e0cf92f0ac4446ebbe8408`

Measurement tip: `208a321d251d177cbf222ca7f4c58d3154303548`

Status: **HELD.**  The two unchecked external-solver inputs do not own the
24 cm kt=1 sea-surface row.  The first non-bit statement is NEMO's cold-start
sea-surface history initialization, but its causal movement is exactly zero.
After that inert mismatch is substituted, the walk reaches the recorded slow
U forcing.  No production physics statement lands in this round.

Every solver number below is labelled **given NEMO's entry**.  The ORCA2
ladder separately retains its existing **independent with Decision-52 SSH**
label.  These populations are not mixed.  The six sea-ice selectors and the
card's `unmeasured_features` tuple are unchanged:
`staged_gm_eiv`, `linear_implicit_bottom_drag`,
`internal_wave_mixing`, `spatial_lateral_viscosity`,
`freshwater_budget_carry`, `si3_jpl5_layered_prather_state`.

## 1. Record, execution and compiled source

The admitted record is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`.
The gate ran production JIT on CPU, fp64/x64/libm, with the card resolving
`explicit_substep`, 65 substeps, `nemo_ab3am4`, `nemo_literal` continuity,
substep drag, and `forward_euler`.  Rank 0's owned 90 longitude columns are
scored; no rank-1 or full-domain claim is made.

The executing input copy is
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:287-291`.
The compiled solver then zeroes cold histories at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:339-347`,
loads its forward-entry fields at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:355-364`,
forms predictors at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:460-493`,
forms face depths/transports at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:505-536`,
updates continuity at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:550-558`,
back-interpolates SSH and forms pressure gradients at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:601-616`,
adds Coriolis and drag at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:618-652`,
and performs the executing vector update at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:666-679`.
The record writer's field order is
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:755-779`.

## 2. The two requested input substitutions

All rows in this section are **given NEMO's entry**.

| operand/arm | result |
|---|---|
| baseline end-of-step rank-0 SSH disagreement | `0.2448430937728719 m`, exactly round 14's number |
| sea-surface freshwater forcing | bit-exact, `0 / 8,794` cells, max `0.0`; replacement is an identity |
| U drag coefficient before substitution | `8,568 / 8,568`, max `0.001957455073904624` |
| V drag coefficient before substitution | `8,554 / 8,554`, max `0.0018358089155710111` |
| recorded drag reaches traced boundary | bit-exact for U and V |
| own-rate no-op control | solver SSH bit-exact |
| recorded-drag end-of-step movement | `8,794 / 13,320` cells, max `0.016513317062236965 m` |
| movement / baseline | `0.06744448784638991` (6.7444%) |
| substituted end-of-step SSH disagreement | `0.24444708169024776 m` |

Freshwater was already exact.  Drag is a measurable contributor but is below
the preregistered `0.024 m` ownership threshold.  Because the freshwater arm
is an identity, the combined arm has the drag arm's `0.016513317062236965 m`
movement and also fails the ownership threshold.

The no-op substitution uses legoESM's own rates and leaves the solver SSH
bit-identical.  A one-ULP plant in each real input channel fires on exactly
one cell: freshwater `1 / 8,794`, max `5.293955920339377e-23`, ULP 1; drag U
`1 / 8,568`, max `5.421010862427522e-20`, ULP 1.  The plant gate intentionally
returns exit 1 after writing its clean-stamped JSON.

## 3. First non-bit statement

The final gate walks every source group in substep 1 before beginning substep
2.  Its first rows are the compiled cold-history assignments, not the later
entry load.

| walk | first non-bit row | unequal | maximum | end-of-step SSH movement |
|---|---|---:|---:|---:|
| recorded drag substituted | substep 1 `eta_history_b` | `8,794 / 8,794` | `0.10891439254081828 m` | — |
| entry U/V additionally substituted | substep 1 `eta_history_b` | `8,794 / 8,794` | `0.10891439254081828 m` | `0.0 m` from entry substitution |
| six cold histories additionally substituted | substep 1 `slow_u` | `8,568 / 8,568` | `5.370080135032166e-12` | **`0.0 m`** from history substitution |

NEMO zeroes `sshbb_e`, `ubb_e`, `vbb_e`, `sshb_e`, `ub_e`, and `vb_e` in
the cited `:339-347` block.  legoESM exposes non-zero cold SSH histories, so
the statement is non-bit.  However, NEMO's first two predictor rows use the
forward weights in the cited `:460-493` block; replacing all six histories
moves no scored end-of-step SSH cell.  R15-P6 is therefore REFUTED.  This is a
mismatch, not the owner, and the preregistered stop rule says to continue
without a production fix.

The next active row is `slow_u` in the cited vector-update block.  Its partner
`slow_v` differs by `2.2928581685638914e-11`; the resulting first-substep
exits differ by `8.922594685670249e-10` (U) and
`3.809672033947605e-09` (V).  Those later rows are reported only to define
the next walk; none is attributed or landed here.

## 4. Retractions and instrument defects kept in the record

1. The first draft aggregated both recorded substeps and printed `u_entry`,
   `8,568 / 17,136`, max `8.922594685670249e-10`, as the first mismatch.
   **RETRACTED.**  All 8,568 cells belonged to substep 2.  The gate now scores
   the complete substep-1 program before substep 2 and cannot print that
   aggregate ownership claim.
2. A subsequent draft placed the entry load before the cold-history zeroing.
   The compiled source executes `:339-347` before `:355-364`; the final
   `SOURCE_ORDER` and its regression test now enforce that literal order.
3. The first two-channel plant run fired, then the evidence writer refused a
   dirty tracked tree.  The first clean rerun fired, then refused to serialize
   NumPy integer coordinates.  Coordinates are now converted to JSON-native
   integers and the admitted clean run writes the complete artifact.  Neither
   failed artifact is cited for a scientific verdict.

## 5. Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R15-P1 | **CONFIRMED** | baseline `0.2448430937728719 m`, relative departure `0.0` |
| R15-P2 | **CONFIRMED** | freshwater bit-exact, max `0.0`; identity replacement |
| R15-P3 | **CONFIRMED** | drag movement `0.016513317062236965 m < 0.024 m` |
| R15-P4 | **CONFIRMED** | combined movement is the same `0.016513317062236965 m < 0.024 m` |
| R15-P5 | **CONFIRMED** | first non-bit is substep-1 `eta_history_b`, before continuity |
| R15-P6 | **REFUTED** | causal history substitution moves end-of-step SSH by exactly `0.0 m` |
| R15-P7 | **CONFIRMED** | ladder `LADDER_MEASURED`; kt=10 entry T max `3.9430791763114783` on 430,552 cells |
| R15-P8 | **CONFIRMED** | no production statement landed; private hooks are off by default; ORCA2 and GYRE guards pass |

## 6. Landing guards

### ORCA2

The kt=1..10 ladder exits 0 with `LADDER_MEASURED` at measurement tip
`208a321d2`.  Its independent first statement remains kt=1 stage-1 T,
unattributed.  The Decision-52 bridge remains bit-exact for T, S, SSH, U and V;
the kt=10 entry-temperature row remains max `3.9430791763114783` with 430,552
unequal cells.  No registered row leaves its bar and no first-over-bar moves
earlier.

### GYRE

Base `8f58d909e` and tip `208a321d2` were run separately through the ten-step
trajectory and 30-day daily member.

| check | result |
|---|---|
| offline oracle-relative compare | **PASS**, 70 certified rows, 0 violations |
| largest oracle-residual worsening | `0.0 ULP` |
| first over bar | unchanged: U/V at kt=2 |
| residual arrays | `210 / 210` `np.array_equal` |
| daily snapshots | `30 / 30` byte-identical |
| day-30 SHA-256 | `14a7e64b4512860eded79bb12a2120885b97400ecb4acb7e6a2bd3d84a245469` |

Thus the private drag/history measurement plumbing is inert when its hooks
are absent, including the shared GYRE implementation.

## 7. Review, citations and tests

The required separate reviewer was attempted twice at committed tips.  Both
attempts failed before review because `codex exec --sandbox read-only` could
not initialize its app-server client on the read-only filesystem.  Verdict:
**independent review unavailable in-sandbox**.

The focused final-tip battery is **27 passed, 26 deselected**.  The required
`tests/ocean/fidelity -n 12` battery was launched exactly once: 1,640 tests
were collected; it reached its final tail after visibly recording six
failures and seven skips, then produced no output for 19 minutes at 99% and
was interrupted to stay within the round CPU budget.  Because xdist emits
failure IDs only in the final summary, the interrupted run cannot mechanically
classify those six marks; no PASS is claimed for that battery.  The changed
gate, private hooks, source order, JSON control, and all ten compiled citations
are covered by the passing focused battery.

The receipt citation gate maps all ten compiled strings above, pins both
range endpoints and lengths, and requires a rigid two-line shift of each to
fail.  Its final PASS and planted-failure result are recorded after this
receipt's commit.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round15/codex`.
The admitted solver artifact is `solver_gate_final_admitted.json`; the ORCA2
ladder is `ladder_final.json`; GYRE evidence is in `gyre_base`, `gyre_tip`,
`gyre_compare.json`, and `gyre_exact_identity.log`.

## 8. Choices

ASKED: substitute the sea-surface freshwater forcing and barotropic drag
coefficients, then walk the compiled split-explicit solver in source order.

UNASKED: none.  Decisions 54, 57 and 58 remain pending and untouched.  No
configuration choice, carried-state change, stabilizer or sea-ice change was
made.

## OPEN

1. **Round 16: continue the first-active-statement walk.**  Retain the private
   recorded cold-history substitution, substitute the record's `slow_u` and
   `slow_v` values at the cited vector update, measure their already expected
   sub-threshold causal movement, and then name the first non-bit arithmetic
   statement in `u_exit`/`v_exit`.  Do not land the inert cold-history mismatch.
2. The northern-fold mask/wind-stress operands (668 / 35 cells) remain
   reported, not landed.
3. Decisions 54 (three-part dyn_ldf), 57 (`r1_rho0` spelling), and 58 (second
   continuity solve) remain pending and must not be acted on without the user.
4. The independent ORCA2 year still depends on the already scheduled
   independent-initial-state completion; this round's solver result is only
   **given NEMO's entry**.
5. The wide ocean-fidelity battery's hung final tail remains an operator action;
   its interrupted run is not represented as green.
