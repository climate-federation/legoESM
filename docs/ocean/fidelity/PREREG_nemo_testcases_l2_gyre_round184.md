# Preregistration — round 184, DINO developed-state bridge repair

Committed before any new Round-184 numerical comparison.  Round 182 showed
that the developed DINO twin enters its first production step with the NEMO
restart exact on wet T/S/u/v/ssh, then returns non-finite values in every
reported prognostic family.  Its downstream step-2 live-geometry refusal is a
detector, not the first failing statement.  Decision 61 landed the carried
step-entry `rn2b` route with this row explicitly unmeasured; this round repairs
only the bridge/first-step defect and appends the two-card result to the
Round-183 receipt.

No configuration value, physical formula, carried-state schema, threshold,
timestep, NEMO source, or GYRE trajectory is authorized to change.

## Compiled execution order

The compiled DINO MLF program applies the surface boundary condition before
the physics (`DINO/BLD/ppsrc/nemo/stpmlf.f90:163-190`), then vertical and
lateral physics (`DINO/BLD/ppsrc/nemo/stpmlf.f90:192-221`), momentum and its
implicit solve (`DINO/BLD/ppsrc/nemo/stpmlf.f90:228-385`), tracer surface,
shortwave, advection, lateral diffusion, and vertical solve
(`DINO/BLD/ppsrc/nemo/stpmlf.f90:442-515`), and only then boundary conditions,
time filters, and the level swap (`DINO/BLD/ppsrc/nemo/stpmlf.f90:518-580`).
The bridge walk follows that order and scores active wet cells separately from
dry/storage cells; it does not infer a first failure from the final state.

The pre-implementation search found the existing developed-state constructor
and production loop in `kamm_twin_90d.py`, the single DINO topology bridge in
`nemo_state_bridge.py`, and the Round-182 production `bn2` wrapper.  Round 184
extends that wrapper/walk rather than creating another twin harness.

## Frozen predictions and falsifiers

1. The fully built bridged state immediately before surface forcing has zero
   non-finite values in every active wet T/S/u/v/ssh family.  A non-finite wet
   entry value makes the bridge load itself the first boundary and stops the
   downstream walk.  Dry/storage non-finite values, if any, are registered but
   are not called prognostic destruction unless a later active statement reads
   them.
2. The analytic surface-forcing application preserves finite wet state.  If it
   is the first failing boundary, the walk stops there and cites NEMO's `sbc`
   call rather than an ocean operator.
3. Otherwise exactly one earliest compiled-order production boundary changes
   an active prognostic family from all-finite to non-finite.  The statement is
   named from the compiled DINO source and its direct operands are printed.
   More than one unseparated operator at that boundary refuses attribution.
4. The minimal repair changes only a bridge representation or routing operand
   so legoESM supplies the value NEMO carries at that statement.  It adds no
   clipping, stabilizer, or replacement constant.  The repaired developed
   control and carried arms both complete at least the first two production
   steps with finite active T/S/u/v/ssh and execute the mixed-layer recurrence.
5. A one-cell or one-field production plant at the repaired boundary restores
   the original failure, prints `STATUS PLANT-FIRED`, and exits nonzero.  A
   source-only assertion or a plant that reaches a pass marker is insufficient.
6. The landed carried route does not worsen either DINO card relative to the
   repaired control on the committed DINO comparison rows.  Every moved row is
   registered; a worsened certified row stops the round for a decision rather
   than being hidden.
7. DINO from-rest gates remain green.  The GYRE/tank/generic route census is
   unchanged because this is a developed-state bridge repair, not a model-card
   change.  If production model code must change, the complete executing-card
   census and Decision-43/45 gates are rerun instead.
8. The receipt citation gate and its shifted-citation plant, focused tests, and
   a separate read-only Codex review run against the final commit.  `DO NOT
   SHIP`, an unfired plant, a new test failure, or dirty commit stamp refuses
   the repair.

If the first statement requires a configuration or carried-state choice, the
round stops with `DECISION_NEEDED`; no choice is made silently.
