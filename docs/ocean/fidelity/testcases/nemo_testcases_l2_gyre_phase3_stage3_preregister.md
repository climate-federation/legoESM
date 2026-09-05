# NEMO testcase lane 2 GYRE — stage-3 completion preregistration

Date: 2026-09-03  
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`  
Reconciled baseline: `57429ecf5f377ce2bf220f36bc05313cd29e0dfd`

This freezes the next ordered measurement after the bottom-drag landing and
the Nbb TKE/EVD time-level correction.  It was derived afresh from NEMO 5.0.2
source; the abandoned round-7 WIP receipt and its nonexistent preregistration
SHAs are not evidence.

## Entering boundary

The CPU/fp64 gate now clears all 800 barotropic substep rows, both stage-1
momentum rows, and the stage-1 and stage-2 T/S/SSH operands.  Stage-2 tracer
normalized maxima are `4.542928987878046e-16` (T),
`5.786483703728924e-16` (S), and `3.2526065174565133e-18` (SSH).  The first
tracer debt is therefore the stage-3 completion returned at `kt=2`: T
`2.2049230768358347e-3`, S `8.792858585325299e-11`.  The first momentum debt
is likewise stage 3: u `9.481924730527598e-7`, v
`8.988523102725608e-7`.

## Source-ordered register

For tracers, `stprk3_stg.F90:565-600` zeros `Krhs`, then executes advection,
`tra_sbc_RK3`, QSR, lateral diffusion, and finally the implicit `tra_zdf`
time integration.  The resolved GYRE branches have no ice shelf, BBL tracer,
damping, MFC, OSMOSIS, or NPC term.  For momentum, `:317-336,357-430`
executes HPG, ENE vorticity, vector-invariant advection, level Laplacian
viscosity, then implicit `dyn_zdf`; `:433-446` applies the final fixed-depth
barotropic mean.

A WRITE-only MY_SRC record will capture the accumulated stage-3 T/S `Krhs`
immediately before `tra_zdf`, plus T/S `Kbb`, `Kmm`, and final `Kaa`.  The
candidate will expose its normal pre-implicit state only after completing the
ordinary step.  The gate will reconstruct NEMO's explicit QCO seed from the
dumped operands and score it before scoring final `Kaa`.

- If the pre-implicit seed clears `1e-15` but final `Kaa` does not, the
  implicit ZDF solve is the first owner-capable boundary.
- If the seed is already DEBT, the first source-ordered accumulated RHS term
  over bar remains the boundary; no ZDF owner label is allowed.
- A one-variable oracle-seed arm is `CONFIRMED_CAUSAL_OWNER` only if it clears
  the final field.  Movement is printed relative to the faithful residual
  before any label.

The existing direct QSR row is retained: its tendency differs by
`5.257960831729332e-13 K/s`, but the oracle-QSR injection moves the kt=2 T
row by only `1.416330476565004e-7` of its residual, so QSR is
`CAUSAL_NONPRIMARY_AT_KT2`, not exonerated globally.

## Controls and provenance

The record must validate magic, header, dimensions, fp64 dtype, finite owned
values, and a central time-level-registry entry.  The same instrumented run's
ordinary stage and kt=2 state hashes must equal the already pinned hashes, or
the record is rejected.  A planted pre-ZDF tracer value must become DEBT.
No external adversarial review has occurred for this round.

## ZDF Kmm time-level continuation

Source inspection after the first stage-3 run identifies one narrower matrix
boundary.  Stage 3 establishes `Kmm = N+1/2` and its `r3f` at
`stprk3_stg.F90:218-235`; `tra_zdf` then divides both off-diagonals by
`e3w(...,Kmm)` at `trazdf.F90:207-221`.  The candidate currently threads the
whole-step entry SSH into that divisor.  Before the production rerun, freeze
this discriminator as follows:

- the faithful arm supplies the already-materialized stage-2/N+1/2 SSH to the
  existing canonical `nemo_e3w_kmm` helper;
- a private one-variable legacy arm restores only the entry-SSH operand;
- **CONFIRM** iff the faithful final T/S rows clear `1e-15`, the legacy arm
  reproduces the prior final residual within 1%, and the oracle-operand literal
  recurrence is at bar;
- **REFUTE** otherwise.  Scaling is printed before any owner label.

The exploratory source-order arithmetic used to select this arm is post-hoc
and is not citable campaign evidence; only the committed gate reruns below may
support the receipt.

## Direct content-RHS continuation

The committed Kmm-level rerun improves final T from `4.5273e-9` to
`3.1935e-11` absolute and S from `3.4163e-10` to `8.1002e-13`, but neither
clears the bar.  The next source-order operand is therefore the actual
thickness-form content RHS passed to `tra_zdf`, not its concentration view.
The earlier private arm changed only the view while the production path kept a
separate content array, so it had no discriminating power and cannot support
an owner label.

A WRITE-only hook will expose that exact content array after the ordinary step;
the gate will compare it with `e3t(Kbb)*T(Kbb) + rDt*e3t(Kmm)*Krhs` from the
oracle record.  A separate one-variable arm will replace only this array while
retaining the candidate matrix.

- **CONFIRM content-RHS ownership** iff the direct row is DEBT at the measured
  final-residual scale and the oracle-content arm clears or materially reduces
  final T/S with movement at that scale.
- **REFUTE** if the direct row is AT-BAR or the arm is near-null.
- If the direct row is DEBT but the arm leaves a residual, ownership is shared
  or later; report the next operand honestly.  The immutable bar remains
  `1e-15`, and a planted wet-cell content perturbation must fire.

## Stage-3 advection-content continuation

The direct content row is DEBT (`3.0946e-10` T and `2.7285e-11` S absolute),
but replacing it moves only 5.39% of the faithful normalized T residual and
does not improve the final maximum.  The maxima occur at interior levels where
SBC and QSR are zero, and the oracle LDF increment is exactly zero.  The next
ordered discriminator is therefore the advection-only content before the
stage-3 physical source is added.

The shared RK3 helper will expose, WRITE-only, its already-computed
`e3t(Kbb)*T(Kbb) - rDt*div(F_T)` value.  The gate compares it against the
oracle's `e3t(Kbb)*T(Kbb) + rDt*e3t(Kmm)*Krhs_after_advection`.
**CONFIRM** the advection boundary iff this row is DEBT at the direct-content
scale; **REFUTE** iff it is AT-BAR.  A later one-variable correction must move
the final residual at its scale before any owner label.  No alternate FCT
implementation may be added: any correction belongs inside the canonical
S-37 NEMO FCT identity and its isomorphism registration.

## Stage-3 FCT transport operand

The committed advection-content gate confirms the entire direct-content debt
is already present before stage-3 physical sources.  Stage-2 T/S are already
AT-BAR, so the next ordered FCT input is the stage-3 `zFu/zFv/zFw` triplet
built at `stprk3_stg.F90:257-304` and consumed by `tra_adv` at `:463`.
The existing WRITE-only transport record and private exposure hook will be
scored on their native owned masks.  **CONFIRM transport ownership** only if a
row is DEBT and a one-variable oracle-transport arm moves the advection-content
or final residual at the same scale; **REFUTE** if all informative rows are
AT-BAR.  The stage-1 structural-zero `zFw` exception does not apply to this
stage-3 record.  A planted owned-cell perturbation must fail closed.

### Source-boundary correction (preregistered before the replacement run)

The preceding paragraph misidentified the existing record as the transport
consumed by `tra_adv`. Source inspection shows that record is emitted before
`tra_adv_trp`, while the live vector-invariant branch recomputes `zFw` inside
`tra_adv_trp` (`TRA/traadv.F90:138,220-227`) before `stprk3_stg.F90` passes
the triplet to `tra_adv`. The pre-call `zFw` comparison and its injection arm
are therefore **UNINFORMATIVE / INVALID FOR OWNERSHIP**; their numerical
movement will not be cited as campaign evidence.

A distinct WRITE-only MY_SRC record will now capture `zFu/zFv/zFw` after the
common `tra_adv_trp` boundary and immediately before passive or active tracer
advection. The ordinary stage, kt=2 state, and restart hashes must remain
bit-identical. Only this post-call triplet may drive the one-variable
transport arm. **CONFIRM transport ownership** iff an informative post-call
row is DEBT and replacement moves the advection-content or final residual at
the same scale; otherwise report the first surviving operand without an owner
label. Header, fp64 dtype, registry, finite-mask, and planted-violation checks
remain mandatory.
