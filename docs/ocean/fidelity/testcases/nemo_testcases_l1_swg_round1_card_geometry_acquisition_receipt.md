# NEMO SWG level-1 round 1 — card, geometry, and acquisition receipt

Date: 2026-10-11.  Base: `17f6923ff76145ad36cd46de00ec66d5b22064d7`.
Preregistration commit: `31e9c2f9b678a4954ba778fd2ce922c7b8343bac`.
Implementation commit: `b466d706b6fb32dbd528c3b8df9a06c5463387b4`.
Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round1`.

## Verdict

**HELD.**  The round-1 implementation and deterministic gates are complete,
but the required separate Codex review did not return a verdict because this
sandbox denies the reviewer's model connection.  No acquisition should be run
until that review returns `SHIP`.

The required command was attempted with `codex exec --sandbox read-only` in the
writable clone.  Its app server was given an isolated temporary Codex state
directory because the normal state directory is read-only.  It initialized,
identified the sandbox as read-only, printed the full adversarial prompt, and
then emitted the following terminal diagnosis (verbatim):

```text
failed to connect to websocket: IO error: Operation not permitted (os error 1)
ERROR: Reconnecting... waiting for network
```

It emitted neither `SHIP` nor `DO NOT SHIP`; therefore there is no review
verdict to quote and the no-review/no-land rule holds.  The complete attempt is
`codex_review_attempt.log` in the evidence directory.

## Frozen predictions and falsifiers

The preregistration was committed before any array in the operator probe was
opened.  It froze exact geometry, reader, acquisition, and planted-violation
criteria.  One source-derived coordinate prediction was **REFUTED and retained**:

- predicted written T origin: -62,500 m; measured: -12,500 m;
- predicted zero-based wet rectangle: rows/columns 3 through 82; measured: 1
  through 80;
- discriminator: the source-local computational array is 88 by 88, while the
  written NetCDF data domain is 84 by 84 after two MPP halo rows on each side
  are removed.  The source coordinate formula was unchanged; the preregistered
  source-to-record index association was wrong.

The survey now states that correction rather than silently changing the frozen
preregistration.

## Card and fail-closed boundary

`SwgResolvedNamelist` states the shipped deck's selected and inert switches.
`SWG_DEVIATIONS` contains only the toolchain removal of `key_xios`.
`build_swg_card()` is a pure configuration of the shared lat-lon C-grid lane,
with the shipped 84 by 84 data domain, 6,400 wet T cells, one executed 500 m
layer, 3,600 s step, 172,800 steps, and reduced gravity 0.02 m/s2.

The card is intentionally not executable.  `SWG_UNMEASURED` contains exactly:

1. ENS total-vorticity with the mask selection;
2. the SWE three-stage explicit-free-surface RK3 program;
3. stage-3-only viscosity, mean wind stress, and before-velocity layer drag.

The independent ladder arm exits 2 with a named `REFUSE` listing all three.
Thus the restart comparison below is explicitly reader calibration, not a
legoESM/NEMO fidelity claim.

## Geometry measurement

The corrected source transcription is bit-identical to all 34 admitted probe
rows: T/U/V/F coordinates, eight e1/e2 metrics, source and card Coriolis,
T/U/V/F masks, and the depth/thickness records.  `geometry.log` ends
`GEOMETRY IDENTICAL`; `geometry.json` stamps implementation commit
`b466d706b6fb32dbd528c3b8df9a06c5463387b4`.

All three preregistered controls fired and exited nonzero:

- signed-y Coriolis: 84 unequal T values;
- one flipped wet cell: one unequal mask value;
- one changed layer thickness: one unequal thickness value.

## Record readers and acquisition

The restart reader admitted exactly the ten numbered after-state restarts,
checked each internal step number and 3,600 s step, required finite float64
fields, and removed only the dummy bottom velocity record.  Its self-comparison
is bit-identical and all rows are at the 1e-15 bar; the one-cell plant leaves the
bar and exits nonzero.  Again, this proves the reader and scorer, not ocean
fidelity.

The ladder also accepts the second, stage-entry record kind through the same
admission module.  The additive acquisition consists of a stage-entry writer,
an additive patch on the copied SWE step source, a new
`SWG_OMIP_L1_P3` target, and an operator-only runner.  The default invocation is
preflight only.  It verifies the existing reference executable's SHA-256
`a5b312b8b8fccadd67152d446811c7d2bc9d0e1cd55dce0d3148c8b5c6817eb9`
before doing anything.  A real operator run must admit ten times three ordered
stage frames and compare every numeric field in its own grid-T/U/V/W output
bit-for-bit with the reference probe.  Every nonzero runner exit is preceded by
a named `REFUSE` line.

Preflight passed without invoking `makenemo` or `mpirun`.  The runner must stay
on hold until the external review is available.

## First non-bit statement

There is no model-fidelity row in round 1 because execution is refused.  The
first non-bit measurement in the round is the preregistered written-coordinate
association: the predicted T origin differed by exactly 50,000 m.  It was an
index-frame error, not a changed numerical statement, and correction of the
data-domain crop makes every geometry row bit-identical.  No executable ocean
operator was changed in response.

## Rule 12 disposition

| Landed statement | Shared implementation site | Who can execute it | Other-card disposition |
|---|---|---|---|
| SWG coordinate, mask, and one-layer card construction | SWG-named helpers and builder in the shared recipe module | SWG builder only | TSUNAMI, VORTEX, GYRE, LOCK_EXCHANGE, OVERFLOW, DINO, and ORCA2 do not dispatch to the SWG case or helpers |
| SWG validation dispatch | Existing shared card validator, guarded by exact case name | SWG card only | Existing case branches are unchanged |
| Geometry, restart, and stage-entry tooling | SWG-named validation scripts | Explicit SWG CLI only | No model card imports or executes these scripts |
| Stage-entry writer | Operator runner copies it into the new NEMO target | New instrumented target only | Reference target and every legoESM card are untouched |

No shared executable numerical statement was changed, so there are no moved
ladder rows to register and no other card can execute this round's additions.
The SWG focused suite passed 23 tests and the TSUNAMI card suite passed 26.
A broader certified-card battery passed 145 tests and reported three stale
digest assertions for GYRE, LOCK_EXCHANGE, and OVERFLOW; the untouched base
checkout reproduces the same three failures with the same actual digests, so
they are pre-existing at the stated base rather than Rule-12 movement.

## Compiled-source citation map

Every source claim below is bound to the exact compiled branch that produced
the operator probe:

- Domain spacing, dimensions, gravity override, and closed lateral boundaries:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_nam.f90:154-166`.
- T/U/V/F metre coordinates, uniform metrics, and the absolute-y beta plane:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90:123-168`.
- The one-layer W/T depth ladder:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:157-168`; flat-bottom and wet-mask
  construction: `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:206-235`.
- The shipped surface freshwater, salt, and heat fluxes are zero:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_sbc.f90:84-87`.
- The executed ENS branch supplies total rather than Coriolis-only vorticity:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:250-251`.
- With one executed level, the initialized vertical-advection carry and empty
  interior loop make the bottom subtraction exactly zero:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/dynzad.f90:111-146`.
- The three stage lengths are one third, one half, and one full step:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:140,196,252`; stage 1 calls advection
  and vorticity in that order:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:148-149`.
- Stage 3 adds lateral diffusion, mean wind, and drag on the step-before
  velocity: `SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:260-273`; it then advances
  free surface and recomputes thickness ratios:
  `SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:281-283`.

## OPEN

1. Re-run the exact adversarial Codex review where its model connection is
   permitted.  Do not run acquisition unless the terminal verdict is `SHIP`.
2. After `SHIP`, the operator runs the committed acquisition runner and returns
   the admitted stage-entry record plus passivity report.
3. Round 2 starts from that admitted record and implements the three declared
   unmeasured arms in the shared lane, including end-to-end propagation of the
   configured reduced gravity, then walks stage order without moving any
   existing at-bar row.

No configuration or carried-state choice is requested in this round.
