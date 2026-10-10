# Preregistration: ORCA2 round 223 — OMT-4 admission and atomic-unit score

Date: 2026-10-10. Frozen base: `272ff3437`. Scope: admit the existing
Decision-109 OMT-4 acquisition, build its gate-local card, run both labelled
ladders, and score the complete round-217 vector unit atomically. This file is
committed before reading any round-222 restart payload or running an OMT-4
legoESM trajectory.

No NEMO run, package physics, shipped rung-0/rung-10 card, carried state,
stabiliser, sea-ice selector, threshold, scoring rule, or
`unmeasured_features` tuple changes in this round. Every OMT-4 trajectory
number will be labelled either **independent** or **given NEMO's entry**.

## Compiled restart statement and inherited record

The operator reports that the smoke, uninstrumented ten-step run, both P3
twins, and the 96-step month all reached `STOP 0`; the existing admission then
refused because `ORCA2_00000095_restart_0000.nc` carries `kt=0` rather than 95.
No payload has been inspected for this preregistration.

The compiled record build opens a listed restart at `nitrst-1`, deriving its
filename from the current `nitrst`, at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/restart.f90:94-146`. It writes and
closes only when `kt == nitrst`, then advances the fixed-size list cursor with
`MIN(nrst_lst + 1, SIZE(nn_stocklist,1))` at `restart.f90:153-202`. The month
deck fills all ten compiled list slots and ends the list at 95 while
`nn_itend=96`. Therefore the source predicts that step 96's terminal-open arm
reopens the still-current step-95 filename and truncates it without a matching
write. NEMO writes the `kt` scalar only through `iom_rstput` when
`kt == nitrst` at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/daymod.f90:405-418`.

This exact terminal-overwrite class already has a committed parser and receipt
in round 187. Round 223 extends the existing OMT-4 admission with that parser;
it does not create a second NetCDF interpretation.

## Frozen predictions and falsifiers

1. **R223-P1 — terminal overwrite, not a time convention.** CONFIRM: both
   step-95 rank shards have `kt=0` and zero payload values for every scored
   restart field, matching the source-ordered terminal reopen. REFUTE: either
   shard has `kt=95`, any scored field has a nonzero payload length, the ranks
   disagree, or a different scheduled restart is malformed. A refutation keeps
   the record refused; no NEMO rerun is inferred from it.
2. **R223-P2 — existing record admits without rerunning NEMO.** CONFIRM: both
   ten-step twins contain 80 self-described frames, all 400 field comparisons
   are array-equal, their kt=10 terminal restarts are byte-identical to the
   calibration, the completed month retains valid rank-complete restarts at
   steps 10 through 90, and step 95 is explicitly classified as terminally
   overwritten. REFUTE: any frame, twin, terminal restart, month checkpoint,
   binary, deck, stop line, or provenance check fails.
3. **R223-P3 — OMT-4 baseline.** CONFIRM: the gate-local OMT-4 card differs
   from OMT-3 only by the already-admitted tracer-advection edge, and both
   labelled baseline ladders complete kt=1..10. REFUTE: either label stops,
   resolves another config delta, loses an existing exact row, or fails its
   independent/given-entry separation.
4. **R223-P4 — tracer-advection attribution.** CONFIRM: the complete
   round-217 vector unit first recreates the registered live-W-thickness
   refusal at kt=8 on OMT-4, while the baseline completes. REFUTE: the atomic
   candidate completes kt=10, refuses at another boundary, or the baseline
   itself refuses. A confirmation attributes the compensating partner to the
   tracer-advection rung but does not authorize a partial or production
   landing; the first FCT statement is then walked offline.
5. **R223-P5 — controls.** Existing deck/frame/twin/terminal/boundary plants
   and a new terminal-overwrite non-vacuity plant must all refuse. The new
   plant changes the observed overwrite signature before classification; if it
   remains green, no month restart claim is citable.

## Stop conditions

If the existing record cannot be admitted after the source-cited checker
repair, status is `STOPPED_FOR_RECORD` with the exact missing or corrupt stream;
NEMO is not rerun in the sandbox. If it admits, both labelled ladders and the
atomic unit are scored before OMT-5. If OMT-4 carries the refusal, the round
names the first tracer-advection statement or leaves that source-ordered walk
as the explicit OPEN item; it does not land a partial cancelling unit.
