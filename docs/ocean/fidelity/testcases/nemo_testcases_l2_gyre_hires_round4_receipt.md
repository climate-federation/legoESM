# NEMO testcase L2 GYRE high-resolution round 4 receipt

**Date:** 2026-10-13
**Incoming tip:** `be90847c4`
**Disposition:** **HELD for one operator acquisition; no physics changed**

## Verdict

The operator's claim is confirmed at the observable boundary: the
`r4_smag` increment is not a last-bit continuation of the `r4` resolution
debt.  Both decks are identical through the kt=2 before state; at kt=3 the
Smagorinsky deck has approximately `2.99e-5 m/s` U/V error versus
`1.3e-11 m/s` on `r4`, and by kt=10 its T error is `1.219e-2 K`.  NEMO calls
`dyn_ldf` only in RK stage 3, at
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90:387-398`, so the first
possible port-specific movement is the step-2 stage-3 momentum-LDF increment.

The existing operator records contain neither NEMO `ahmt`/`ahmf` nor the
momentum RHS immediately before and after that call.  They therefore cannot
run the required coefficient-versus-operator discriminator.  Per the
fail-closed instruction, this round makes **no Smagorinsky physics edit** and
does not guess from the final state.  It supplies a dedicated WRITE-only NEMO
config and exact acquisition commands below.

## Smagorinsky discriminator acquisition

The committed patch creates a new config named `GYRE_OMIP_L2_P3_AHM` from
`GYRE_OMIP_L2_P3`; it does not modify either the engine tree or the certified
P3 config.  Its only source changes are in `stprk3.F90` and `stprk3_stg.F90`:

- after `ldf_dyn(kstp,Nbb)`, it writes `ahmt` and `ahmf` for kt=2 and kt=3;
- immediately around stage-3 `dyn_ldf`, it writes U/V RHS before and after for
  kt=2 and kt=3; and
- every record has a versioned 16-byte magic, dimensions, time-level indices,
  and 64-bit-width assertion.

This is the discriminating measurement.  Substituting the recorded
`ahmt`/`ahmf` into legoESM will classify the owner as **COEFFICIENT** if the
step-2 momentum-LDF residue falls to the inherited `r4` level; otherwise it is
**OPERATOR**.  Only after that result may the relevant coefficient factors or
the two-pass bilaplacian be changed.  NEMO's measured coefficient program uses
the `Nbb` velocity, the tension and shear stencils, lower and upper limits,
bilaplacian `16/9` factor, `rn_Dt` upper-limit clock, square-root conversion,
and coefficient halo update in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90:462-528`.

The operator commands, from this committed checkout, are:

```text
NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2 scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires/build_ahm_writer.sh --run
EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round4/oracle_gyre_r4_smag_ahm RANKS=8 NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2 scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires/run_ahm.sh --run
```

The builder's read-only preflight applies the patch to pinned P3 sources and
verifies the resulting source digests.  The acquisition launcher verifies the
compiled markers, exact `r4_smag` namelist patch, clean legoESM commit, binary
hash, and all four discriminator records.  The acquisition itself remains an
operator action and was not run in this sandbox.

## Resolution-rung barotropic walk

The Round-3 localization is sharpened.  At rest, the first acquired
split-explicit mismatch is substep 1 `slow_u`, followed by `slow_v` and the U/V
exit velocities.  `eta_entry`, U/V entry, mid values, continuity, pressure
gradient, Coriolis, and drag are bit-equal before that boundary.  The first
active `slow_u` mismatch is 2 ULP; the maximum absolute slow-forcing residue
is `6.617444900424222e-24 m/s2`.

This record names the first differing **composite term**, not yet a raw
multiplicand.  At a rest start the depth-averaged dynamic RHS and bottom-drag
contribution are zero, leaving the wind term.  NEMO constructs the vertically
averaged RHS and then adds `(r1_rho0 * utauU) * r1_hu_0` (and its V analogue)
in `cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stp2d.f90:174-199`; `dynspg_ts` then
removes the frozen two-dimensional Coriolis term before entering the substep
loop in `cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynspg_ts.f90:284-292`.

The one-variable causal arm replaces only legoESM's frozen slow U/V forcing
with NEMO's recorded `slow_u`/`slow_v`.  At substep 1, both slow-forcing rows,
both exit-velocity rows, and both transport-metric rows become bit-exact.  The
owner is therefore upstream of the external recurrence, in the frozen wind
forcing composite.  The existing file does not record NEMO's `utau`,
`r1_rho0`, or `r1_hu_0` separately, so assigning the two-ULP composite residue
to one of those raw operands would be a guess and remains OPEN.

The arithmetic audit records 50 substeps on both sides, a quarter-degree
substep of exactly 72 s (`3600/50`) versus 288 s at the shipped resolution,
the corresponding reciprocals, the source grid expression `106000/4`, the
boxcar weights and normalization label, and the pressure-gradient and zero
rest-drag dispositions.  This refutes the split-explicit **count**, pressure
gradient, bottom drag, and averaging as the first owner; it does not relax the
exact ladder bar.

## Registered climate runs and GPU selection

The climate and movie instruments no longer hard-code one commit or accept an
arbitrary legoESM directory.  The committed
`gyre_hires_registered_runs.json` maps an explicit label to the absolute run
directory, full commit, deck, precision policy, backend, certification flag,
and physical GPU where known.  It registers the operator run as
`pre_fix_7b772afb8`, visibly labels its disposition as pre-fix, and permits a
future post-fix label without confusing the two.  Unknown labels, registry
schema changes, directory indirection, commit mismatch, policy mismatch, and
recorded-GPU mismatch refuse.

The long-run driver and wrapper require an explicit non-negative physical GPU
index.  `CUDA_VISIBLE_DEVICES` must equal that argument, the index is stored in
the run manifest and daily records, and CPU-certified mode refuses a GPU
argument.  The generic wrapper has no device default and is configured for the
five-year 1825-day run.  No replacement five-year run is requested until the
port owner is measured and fixed.

## Tests, evidence, and controls

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round4/`.

- `r4_ladder_resolution_walk.json` contains the unchanged ten-step exact-bar
  rows, one-ULP controls, the stage walk, the vertical-coefficient refutation,
  and the new barotropic first-difference/one-variable arm.
- The Round-3 `r4_ladder.json` and `r4_smag_ladder.json` remain the controlling
  side-by-side ladder table because no physics changed in this blocked round.
- Direct tests cover AHM patch applicability, its read-only pinned-source
  preflight, exact deck selection, registry refusal, manifest-policy refusal,
  explicit GPU selection/no default, climate/movie logic, and the driver.
- The receipt citation gate is run explicitly on this receipt, with its planted
  shifted-citation control.

## Inherited pins / OPEN

Inherited unchanged are the shipped one-degree card and digest, its certified
ten-step CPU/libm trajectory, the parameterised `r4` and `r4_smag` namelists,
the exact zero-unequal-float ladder bar and one-ULP plants, 31 levels, RK3,
split-explicit free surface, TKE/EVD, Redi tracers, seasonal analytic forcing,
closed free-slip walls, the native-transcendental non-certified GPU policy,
the NEMO same-file `sosstsst/vovvle3t` convention, the 122x82 eight-tile
stitching rule, and the preregistered year-5 climate bars.  No physics or deck
configuration changed in this round.

OPEN:

1. the operator must build `GYRE_OMIP_L2_P3_AHM` and acquire the four kt=2/3
   discriminator records using the commands above;
2. the next round must inject those NEMO coefficient fields, classify
   coefficient versus operator, fix the measured owner, add a regression that
   fails on revert, and rerun the side-by-side ladders;
3. only after the corrected `r4_smag` kt=3 increment falls to the inherited
   `r4` level should the operator launch a new labelled five-year run;
4. the pre-fix run may be scored only as `pre_fix_7b772afb8`; it must never be
   substituted for that post-fix run; and
5. the raw owner inside the quarter-degree wind-forcing composite remains
   unmeasured, as do the later 1/12-degree acquisition and climate comparison.
