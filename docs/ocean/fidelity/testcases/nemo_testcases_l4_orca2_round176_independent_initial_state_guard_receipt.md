# ORCA2 round 176 — independent initial-state guard receipt

Date: 2026-10-08. Base: `8f3aaa0aa`. Measurement commit:
`55ddebf21ba698491c8016d5c72ee084a5215f9a`. Claim label:
**independent hierarchy rung 0**. Status: **HELD**.

## Verdict

The admitted round-175 record is sound, but the independent walk stops before
stage 1. The rung-0 private card applies ORCA_R2's regional T/S alterations
unconditionally through `build_orca2_initial_ts`'s default
(`nemo_testcase_recipe.py:1413-1453` and
`nemo_testcase_recipe.py:1650-1651`). NEMO applies that block only
inside `nn_cfg == 2 .AND. ln_tsd_dmp`; rung 0 resolves
`ln_tsd_dmp=.false.`
(`ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/dtatsd.f90:218-255`, with the
resolved switch in
`round175/orca2_rung0_stage1_ranked_10step_np2/namelist_cfg:51`). This is the first non-bit
statement and owns the next walk.

No stage operator was scored after that boundary. Crossing it by installing
NEMO's entry would change the claim to **given NEMO's entry**, forbidden by
Decision 52's label separation.

## Record and entry gate

The self-described record admission remains
`PASS_R175_STAGE1_ADMISSION`: two 67,783,100-byte rank records, all 23 fields
finite on owned cells, exactly-once 148x180 coverage, and all 20 additions-only
restart comparisons byte-identical. NEMO's recorded stage-1 advection and
surface-source accumulators are bit-identical for both T and S, confirming the
unforced rung has no stage-1 surface tracer source.

The independent entry comparison is:

| row | active unequal | active RMS | max absolute | argmax `[j,i,k]` | exact |
|---|---:|---:|---:|---|---|
| T | 1,283 / 430,552 | 0.6188248282711377 K | 20.515075852794034 K | `[91,154,3]` | no |
| S | 720 / 430,552 | 0.012747120607447834 PSU | 0.3500000000000014 PSU | `[100,139,17]` | no |
| u | 0 / 413,030 | 0 | 0 | — | yes |
| v | 0 / 413,856 | 0 | 0 | — | yes |
| ssh | 0 / 16,433 | 0 | 0 | — | yes |

Across stored cells (including inactive storage), T has 153,200 unequal cells;
S has 720. The larger T storage count is reported, not used to replace the
active-domain statistic. The earlier transient diagnostic statement that u/v
had 15 levels was a script-printing mistake: both card and record have 30
physical levels, and u/v are bit-exact.

Both preregistered cells are exact at the independent entry: the round-174
machine maximum `[147,49,0]` has T/S
`-1.6946502868686952 / 31.42209498728475`; the production failure-column cell
`[86,159,3]` has `25.888660361689908 / 36.218905541204634`. Therefore the
regional initial-state defect does not itself explain the later 3.2847-PSU
stage-1 maximum at the fold cell; it nevertheless precedes that operator walk
globally and must be corrected first.

## Required geometry read-out

The card reads `ORCA_R2_zps_domcfg.nc`. At `[147,49]` the column is on the
northern fold, not a cyclic seam, has 27 wet levels and a 148.10058158549737 m
partial bottom cell; its four surface neighbours are wet. At `[86,159]` the
column is neither fold nor seam, is land-adjacent to west and south, has 21 wet
levels and a 260.9846894461209 m partial bottom cell. Both columns' levels 0-5
are wet, with e3t
`[10.000015488051758, 10.000818315027573, 10.002382004484389,
10.00542762603709, 10.011359574398284, 10.022913005705504]` m. The complete
mask/thickness rows are in `round176/stage1_offline_replay.json`.

## Operator table

| compiled order | result | disposition |
|---|---|---|
| independent entry T | 20.515075852794034 K maximum | first non-bit statement |
| independent entry S | 0.3500000000000014 PSU maximum | second entry debt |
| independent entry u/v/ssh | bit-exact | admitted |
| external/QCO handoff through final stage association | not evaluated | UNMEASURED: upstream entry guard owns walk |

This is not an operator exoneration. R176-P3 and R176-P5 remain UNMEASURED.

## Correction to round 174

Round 174's receipt labelled the 3.2847473521544472-PSU stage-1 result
`independent`. That label is retracted. Its gate calls `r166._setup`
(`nemo_testcase_l4_orca2_round174_stage_growth_gate.py:224`), which explicitly
assembles NEMO's kt=1 frame and installs it with `bridge_entry`
(`nemo_testcase_l4_orca2_round166_external_substep_gate.py:223-230`). Those
stage-growth numbers are **given NEMO's entry**. Their values remain valid
under that label; they are not an independent-start result.

## Predictions and controls

R176-P1, P2, P4 and P6 are CONFIRMED. R176-P3 and P5 are UNMEASURED because
the walk stopped at the earlier entry boundary. Seven non-vacuous plants fire:
admission, rank placement, source order, cell identity, active-mask registry,
first-debt selection and one-ULP output. No `packages/` file, card selector,
carried state, stabiliser, threshold or sea-ice field changed.

The required separate `codex exec --sandbox read-only` review was attempted
twice. This installed CLI rejects `--full-auto`; without it, initialization
fails with `failed to initialize in-process app-server client: Read-only file
system`. Verdict: **independent review unavailable in-sandbox**. The full log
is `round176/codex_review.log`.

## OPEN

Round 177 must make the rung-0 private card obey the compiled
`ln_tsd_dmp=.false.` guard by building the T/S input without the regional
alterations, then gate its independent kt=1 entry bit-exact before rerunning
this offline stage-1 table. Because this changes the rung-0 initial condition,
rerun the independent ten-step ladder and month; loudly supersede the previous
"independent" rung-0 trajectory numbers. Do not modify the shipped rung-10
ORCA2 card: its damping-enabled path continues to execute the alterations.

No acquisition is needed. No configuration choice is open: the compiled
guard and resolved rung-0 namelist pin the branch.
