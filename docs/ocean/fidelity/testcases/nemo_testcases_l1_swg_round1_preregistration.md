# SWG round 1 preregistration — card, geometry, records, and ladder

Frozen 2026-10-11 before opening any array in the operator's
`round0_probe/kt10` evidence.  Base commit: `17f6923ff76145ad36cd46de00ec66d5b22064d7`.
This round makes no executable-physics claim: the ENS operator, SWE RK3 step
program, and stage-3 wind/drag remain named `SWG_UNMEASURED` debt.

## Source-verified predictions

The compiled authority is
`tests/SWG_OMIP_L1/BLD/ppsrc/nemo/` in NEMO 5.0.2.

1. The mesh is 84 by 84.
   `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_nam.f90:154-160` divides 100 km by
   `nn_SWG=4`, adds two 25 km ghost cells on each side of the 2000 km basin,
   and takes the ceiling.
2. Every horizontal scale factor is exactly 25,000 m
   (`SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90:95,150-156`).  T-point
   coordinates start at -62,500 m; U/V/F coordinates use their source
   staggerings (`SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90:123-146`).
3. The T- and F-point Coriolis arrays are exactly
   `rn_f0 + rn_beta * abs(y)` with `rn_f0=0.7e-4` and `rn_beta=2e-11`
   (`SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90:164-171`).  Removing `abs` is
   the Coriolis plant.
4. There are exactly 6,400 wet T cells, a contiguous 80 by 80 block at
   zero-based rows and columns 3 through 82.  The prediction follows the
   strict U/V-coordinate bounds in
   `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:210-228`; the lonely-corner rule
   at `:229-235` removes none of this rectangle.
5. There is one wet 500 m layer.  The two-record ladder is W=(0,500) m and
   T=(250,750) m (`SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:157-168`), while
   the bottom is `jpkm1=1`
   (`SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:206-208`).

## Geometry decision rule

CONFIRM only if every admitted geometry row has zero unequal values and zero
maximum absolute error: T/U/V/F coordinates, all eight e1/e2 metric arrays,
T/F Coriolis, T mask, reference depths, and wet-layer thickness.  REFUTE at
the first unequal row.  The gate must exit nonzero for each planted violation:
signed-y Coriolis, one flipped wet cell, and one changed layer thickness.

## Restart-record decision rule

The restart reader must admit exactly ten files numbered 1 through 10, require
each file's `kt` to equal its filename, require finite float64 `sshn`, `un`,
`vn`, `uu_n`, `vv_n`, `utau_b`, `vtau_b`, `emp_b`, `qns_b`, and `sfx_b`, and
crop the single dummy bottom record from three-dimensional velocity fields.
The round-1 self-check compares every admitted after-state field with its own
decoded copy and therefore predicts bit identity; a one-cell score plant must
leave the 1e-15 bar and exit nonzero.  This is reader calibration, not a
legoESM-vs-NEMO fidelity result.

When execution debt is cleared, each real ladder row is frozen as legoESM's
independently chained after-state of step `kt` against restart `NN=kt`, scored
field-by-field at relative bar `1e-15`, with bit identity reported separately.

## Stage-entry acquisition decision rule

The new record is admitted only if steps 1 through 10 each contain exactly
three ordered stage-entry frames.  Each frame must identify its stage and time
indices and contain finite float64 stage `ssh`, `u`, `v`, `r3t`, `r3u`, `r3v`,
`r3f`, the fixed step-before velocity, and both before/current wind stresses.
The instrumented run is passive only if its own NEMO grid-T/U/V output fields
are bit-identical to the existing reference run's fields.  Missing acquisition
records are `ACQUISITION_NEEDED`, never a failed physics prediction.

## Frozen first expected boundary

Round 1 is expected to stop before a fidelity statement: `SWG_UNMEASURED`
must make the execution gate refuse the card.  The first future non-bit row is
not predicted from physics or outputs; it will be named only after the
stage-entry acquisition and the shared SWE program exist.
