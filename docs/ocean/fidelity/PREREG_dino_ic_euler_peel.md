# DINO independent-start IC/Euler peel preregistration

Date: 2026-08-30. Session:
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

## Question and frozen order

The user-supplied transfer ladder puts a 359-day free run started from NEMO's
step-2 restart in the twin class, while the fully independent start is far
away. This peel asks where the independent path first differs: analytic
initial-condition inputs, profile evaluation, or the one Euler bootstrap.
The walk stops causal attribution at the first failing rung. Later rows are
reported as conditional diagnostics and cannot supersede an earlier failure.

No GPU or new NEMO run is authorized. The committed probe may read only:

- `/data/abyssal/dbalwada/lego_bridged_kt2.npz`;
- `/data/abyssal/dbalwada/lego_bridged_d10.npz`;
- `/data/abyssal/dbalwada/RUN_KT2/`;
- `/data/abyssal/dbalwada/RUN_TRAJ_Y1/`; and
- the existing DINO `RUN_TRAJ/mesh_mask.nc` used by the recorded harness.

Every input file used by a numerical row is SHA-256 stamped. Missing, dirty,
non-finite, shape-incompatible, or differently configured inputs hard-fail.

## Source ownership

Executed DINO source is frozen as follows:

- `src/OCE/DOM/istate.F90:107-137`: a non-restart run sets
  `l_1st_euler=.true.`, initializes velocity to zero, calls
  `usr_def_istate` with `gdept(:,:,:,Kbb)`, then copies before to now;
- `cfgs/DINO/MY_SRC/usrdef_istate.F90:129-175`: `nn_initcase=4` evaluates
  the T/S profiles on that three-dimensional T-depth, takes
  `MAXVAL(gphit)` and wet-field `MINVAL` anchors, then applies the latitude
  blend;
- `cfgs/DINO/MY_SRC/stpmlf.F90:134-137`: the first step uses `rn_Dt`;
- `cfgs/DINO/MY_SRC/stpmlf.F90:578-624`: the barotropic corrector and lateral
  boundary finalize precede the filter and time-level swap;
- `cfgs/DINO/MY_SRC/stpmlf.F90:634`: restart writing follows that swap; and
- `cfgs/DINO/MY_SRC/stpmlf.F90:685-688`: only after the first step is the
  Euler flag cleared and `rDt` restored to `2*rn_Dt`.

The step-2 restart's `tb/sb/ub/vb/sshb` are therefore the step-1 carry after
the second step's Asselin filter, not the raw Euler endpoint. The raw endpoint
comparators are the already-recorded first-step `baro_dump_*_after.bin`,
`stp_dump_21_trazdf_{tem,sal}.bin`, and `spg_dump_pssh_final.bin` files. The
restart before-fields remain a distinct carry-level rung.

## Rung 0 — transfer-ladder receipt

Bind, but do not recompute or strengthen, the user-reported common-target
numbers:

| start | free integration | target/readout | SST RMS | SSH RMS |
|---|---:|---|---:|---:|
| day-180 bit-exact bridge | 180 days | NEMO day 360, full 3-D scheduled | 0.005 degC | recorded twin battery |
| NEMO step-2 bit-exact bridge | 359 days | NEMO day 360 daily-series endpoint | 0.0104 degC | 0.29 mm |
| independent analytic start | 360 days | NEMO day 360 | 0.39 degC | report if present |

The step-2 comparison is endpoint-offset by 0.94 day and lacks a scheduled
final 3-D snapshot. It supports `TWIN_CLASS_DAILY_ENDPOINT_WITH_OFFSET`, not a
bit-exact or horizon-exact final-state claim. The causal conclusion is recorded
as user-directed evidence and then tested by the peel; the ladder alone does
not distinguish IC evaluation from the Euler step.

The receipt must record that both bridge artifacts were produced from commit
`7c0f121baed6bc57243e849127040865dda2d597` with one dirty tracked file and
`LEGOESM_ALLOW_DIRTY=1`. It must also record the first ten-day attempt's
mid-run-edit refusal if a citable log is available; absent log means
`USER_REPORTED_NO_HASH`, never an invented artifact.

The restart-admission deviation is frozen as the user-added
`DINO_TWIN_MIN_SPEED` environment override. Its permanent replacement must
keep two independent conditions: every bridged field equals the selected
restart on wet cells within the existing tolerance, and the selected restart's
wet velocity vector is nonzero. Lowering the nonzero threshold must never
weaken or replace the equality condition. The effective threshold and its
source are artifact stamps.

## Rung 1 — analytic initialization, first divergence

All comparisons use fp64 and the standalone physical core convention:
NEMO's recorded `(199,52,36)` frame maps to `[2:-2,2:-2,:-1]`, yielding
legoESM's `(195,48,35)` T frame. A shape or mask mismatch is a hard failure.

Run these cumulative rows in order:

1. **Inputs.** Compare wet mask, latitude, and the exact positive-down T-depth
   consumed by NEMO with the standalone depth. Exact equality is `PASS`;
   otherwise the first divergence is `IC_INPUT_GEOMETRY`.
2. **Common-input evaluation.** Evaluate the source-ordered CASE(4) formula
   and legoESM's public T/S profile functions on identical NEMO depth and
   identical NEMO `phi_max/t_bot/s_bot` anchors. Report maximum, RMS, mismatch
   count, and ulp distance. Exact equality is `PASS`; a residual here is
   `IC_PROFILE_EVALUATION_LIBM_OR_ORDER`.
3. **Resolved standalone initialization.** Compare the actual member-0
   standalone T/S with the source-ordered NEMO CASE(4) reconstruction. Report
   the same reductions and the first differing index/value. Exact equality is
   `PASS`; otherwise report which preceding controlled substitution first
   collapses the residual: NEMO depths, NEMO anchors, then source-order
   evaluation. No later row may be called the owner if an earlier row fails.

The pointwise frozen bar is exact equality for discrete inputs and `<=1e-15`
absolute for evaluated fp64 fields. Values above that bar are debt; no
climate-scale threshold is substituted.

## Rung 2 — Euler bootstrap on a common initial state

Only after Rung 1 is printed, replace legoESM member 0's T/S with the
source-ordered NEMO CASE(4) fields while leaving every other initialized field
unchanged. Prove that the replacement changes only T/S. Advance exactly one
2700-second standalone step with the shipped faithful card and the same
seasonal forcing time used by the T1 runner.

Compare physical-interior wet values against the existing NEMO first-step
stage dumps:

- T/S after `tra_zdf`;
- U/V after `mlf_baro_corr`; and
- SSH after the split-explicit solve.

Report max absolute, RMS, mismatch count, and first differing index/value.
The first field over the `1e-15` pointwise bar is
`EULER_BOOTSTRAP_FIRST_DIVERGENCE`; exact rows before it are retained. The
comparison is a final-stage Euler composition check, not proof that every
internal term is identical.

Separately run two legoESM steps from the common initial state and compare its
carried before-level fields with the step-2 restart's before fields. This row
is labelled `FILTERED_STEP1_CARRY`; it cannot be used as the raw Euler result.

## Controls

- A planted one-cell T mismatch must fire the field-difference gate.
- A planted wrong core slice must fire the shape/mask gate.
- A planted all-zero restart velocity must fire the restart nonzero gate even
  when the bridged state equals that restart.
- The actual step-2 bridge must pass both independent invariants: bridged
  state equals the restart on wet cells, and the restart velocity is nonzero.
- Artifact provenance must expose dirty entry/exit state; an allowed dirty run
  is a receipt with reduced claim scope, not a clean run.

## Disposition

If the first divergence is in the initializer, repair only an already-defined
NEMO-faithful selector/default path and rerun the CPU peel. If the initializer
passes and Euler fails, stop at its first recorded stage and preregister the
next term-level peel before changing physics. If both pass, the unmeasured
owner lies in the transition from raw Euler endpoint to the filtered step-1
carry; do not infer it from the year-long ladder.
