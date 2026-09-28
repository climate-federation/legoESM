# NEMO testcase lane 2 — GYRE phase-3 trajectory preregistration

Date: 2026-09-01

Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`

Scope: the first whole-step comparison for the `GYRE-zco` card, starting at
the phase-2-certified `kt=1` Nbb/before state.  This file is committed before
the phase-3 NEMO build/run, any legoESM step, or any phase-3 score.

## Pre-implementation search

The search covered the DINO and lane-1 canonical cards, the lane-1 phase-3
entry/stage/transport/RHS/barotropic dump writers and binary readers, the
lane-1 fail-closed trajectory gate and log-log growth instrument, the GYRE
phase-1 resolved namelist, and the executed NEMO `stprk3` call graph.  The
existing machinery supplies the fp64 dump formats, time-level registry,
three-stage Wicker--Skamarock identity, NEMO TEOS-10, `nemo_sco`, FCT2,
stage barotropic correction, transport reconciliation, Demange filter, and
first-over-bar/growth reporting.  They will be reused, not forked into a
second format or an independent solver.

The search also found known post-entry configuration differences.  The
oracle resolves vector/C2/ENE momentum, isoneutral tracer Laplacian mixing,
level momentum Laplacian mixing, two-band solar penetration, nonlinear bottom
drag, TKE, enhanced vertical diffusion, and nonzero background vertical
coefficients (`cfgs/GYRE_OMIP_L2/EXP00/namelist_cfg:73-78,108-111,138-144,
159-167,185-220`).  The phase-2 campaign card deliberately carries the
collapsed lane-1 flux-form UP3 identity and zero lateral coefficients, while
some GYRE closure programs are not yet selected.  These are pre-existing
**candidate arms only**.  No term owns a discrepancy until a registered
stage/term row and a one-variable scaling check support that label.

## Oracle run and immutable identity

The run root is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10`; the new NEMO
configuration is `GYRE_OMIP_L2_P3`.  It copies the completed phase-1 case's
cpp keys, resolved configuration inputs, geometry, initial condition, `dt`,
calendar, and physics.  Only `nn_itend=10`, dump cadence, and diagnostic
instrumentation may differ.  The executable, cpp file, namelists, MY_SRC
files, mesh, output, and every binary dump receive SHA256 receipts.

The lane-1 MY_SRC pattern and bytes are reused.  Step entry writes T, S, u, v,
and SSH from `Nbb` before SBC, `stp_2D`, or any RK stage.  At `kt=1`, the
instrument additionally writes:

- `stp_2D` primary `uu_b/vv_b` and tracer `un_adv/vn_adv` frames;
- the entering momentum RHS;
- each stage's advective transports; and
- T, S, u, v, and SSH after stages 1, 2, and 3.

Although the inherited binary magic remains `NEMO_L1_*_1` for byte-for-byte
reader reuse, the configuration/run provenance identifies these as lane-2
GYRE artifacts.  Every reader must call the repository-wide
`time_level_for_dump` registry and refuse an unregistered dump or a level other
than the registered one.

## Executed full-stage registry

NEMO calls the seasonal SBC at step entry (`src/OCE/stprk3.F90:127-138`),
updates EOS/BN2 and vertical/lateral coefficients (`:156-180`), executes the
single barotropic composition (`:184-186`), then calls stages 1, 2, and 3 with
the documented level swaps (`:192-213`).  Within a stage:

- the barotropic/tracer transports are reconciled at
  `src/OCE/stprk3_stg.F90:257-304`;
- stage 1 consumes the `stp_2D` momentum RHS while stages 2/3 execute
  EOS, `nemo_sco` PGF, ENE vorticity, and vector/C2 advection (`:309-336`);
- stages 1/2 update momentum from Kbb and stage 3 adds lateral/vertical
  momentum closure (`:360-430`), followed at every stage by the barotropic
  mean correction (`:433-446`);
- every stage builds tracer transports, FCT advection, and surface forcing
  (`:456-521`); stage 3 additionally executes solar penetration, isoneutral
  lateral diffusion, and vertical tracer closure (`:568-599`).

GYRE's genuinely active forcing is the source transcription already certified
in phase 2: the 360-day clock and seasonal cosines
(`usrdef_sbc.F90:84-107`), heat/restoring fields (`:109-120`), the unmasked
EMP numerator divided by the wet denominator and wet-only mean removal
(`:122-145`), and rotated-grid wind stress (`:161-184`).  Phase 3 evaluates
those fields at each NEMO step clock and supplies them through the canonical
legoESM forcing channels.  A missing forcing channel is DEBT, not a zero-field
waiver.

## Measurements, bars, and stop rule

All card/oracle state arrays must be native fp64 under
`PrecisionPolicy.fp64()` with JAX x64 enabled.  T/S use the wet, non-dummy
T mask; u/v use the matching native wet-face masks; SSH uses the wet surface
mask.  Pointwise rows use the immutable normalized max-absolute bar `1e-15`,
with scale `max(max(abs(oracle)),1)`.  Exact identity is reported separately.

1. Reconfirm `kt=1` entry, then score the registered barotropic frames,
   transport frames, entering RHS, and all three `kt=1` stage states in oracle
   call order.  This is evidence for the complete three-stage program, not a
   substitute for the committed `kt=2` state.
2. Score `kt=2` Nbb/before T, S, u, v, and SSH.  Record the first over-bar row
   in a fail-closed field; later rows cannot erase it.
3. At the first nonexact/over-bar boundary, run only one-variable arms.  Each
   arm must name its sole changed selector/operand and leave all others fixed.
   Before any owner label, report candidate movement divided by the faithful
   error and the resulting error change.  Exact closure with source-aligned
   stage evidence may be `CONFIRMED_OWNER`; directional material improvement
   without closure is only `PLAUSIBLE_CONTRIBUTOR_NOT_OWNER`; wrong scale or
   direction refutes primary ownership.  Self-agreement or a one-sided stage
   value cannot establish two-model ownership.
4. Stop ordinary advancement at the first unresolved over-bar row.  If `kt=2`
   clears, or if the registered owner arms are exhausted without closure, use
   the explicit continuation arm through `kt=10`, retaining the first-over-bar
   record and marking the later prefix nonexact.
5. Reuse lane 1's `characterize_growth`: only fields with at least four
   positive DEBT samples receive successive ratios plus tail log-log
   power-law and semilog exponential fits.  A fit describes growth; it never
   assigns ownership.  Otherwise growth remains UNMEASURED.

The ordered one-variable candidate registry follows the executed call graph:
seasonal forcing placement/channels; barotropic composition/filter; horizontal
momentum vector/C2/ENE versus the collapsed UP3 arm; momentum lateral closure;
tracer FCT/SBC/solar/lateral closure; and TKE/EVD/background vertical closure.
An arm without a dispatchable canonical implementation stays UNMEASURED; the
gate must not emulate it privately and call that a match.

## Planted controls and withheld claims

The gate ships controls that must fail for: a `+1 C` wet `kt=1` entry
perturbation; an unregistered/wrong-level stage dump; a stage-state wet-cell
perturbation; an arm manifest changing two operands; and a synthetic control
whose candidate movement is made too small or opposite-signed for its proposed
owner label.  The first-over-bar field is mandatory whenever any row is DEBT.

Until measurement proves otherwise, `kt=2..10`, all owner labels, all growth
classifications, and any seasonal-gyre trajectory/phenomenology claim are
**UNMEASURED**.  This phase is numerical whole-step certification only.
