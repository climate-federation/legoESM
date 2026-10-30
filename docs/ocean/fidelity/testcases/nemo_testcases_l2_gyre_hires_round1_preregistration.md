# GYRE high-resolution round 1 preregistration

**Date:** 2026-10-12  
**Claim commit:** `f04f9f1d8`  
**Lane:** NEMO 5.0.2 `GYRE_OMIP_L2`, `nn_GYRE = 4`

This file was committed before running the new-resolution ladder or either
three-year integration.  It fixes the claims and controls for round 1; it does
not claim that the required NEMO acquisition already exists.

## Resolved deck

The quarter-degree deck is exactly the operator deck:

- `nn_GYRE = 4`, hence `122 x 82 x 31` NEMO points;
- `rn_Dt = 3600 s`;
- `rn_Lv = rn_Ld = 25000 m`;
- `nn_it000 = 1`, `nn_itend = 26280`, `nn_write = 24`, and
  `nn_stock = 8760`;
- all other scheme and physics selections are inherited unchanged from the
  shipped `GYRE_OMIP_L2` namelist.

These are not inferred scalings.  The compiled oracle reads `nn_GYRE` in
`cfgs/GYRE_OMIP_L2/WORK/usrdef_nam.F90:64-68`, sets the horizontal extent to
`30 * nn_GYRE + 2` by `20 * nn_GYRE + 2` at lines 79-80, and retains
`jpkglo` at line 98.  The compiled horizontal-grid routine divides 106 km by
`nn_GYRE` at `WORK/usrdef_hgr.F90:87`, uses that spacing to locate every T/U/V/F
point at lines 91-141, assigns every scale factor at lines 147-150, and derives
the beta-plane Coriolis arrays from those positions at lines 161-168.

The analytic surface boundary condition depends on the resulting T-point
latitudes throughout: target temperature and shortwave at
`WORK/usrdef_sbc.F90:115-120`, freshwater flux and its domain-mean correction
at lines 122-145, and wind stress at lines 167-176.  Its seasonal clock uses
`kt * rn_Dt` at lines 84-107 and 153-171.  These dependencies are therefore
part of the card parameterisation and cannot be represented by changing array
sizes alone.

## Claim A: ten-step CPU ladder

The only trajectory claim at the new resolution is the stage-entry ladder at
`kt = 1..10`, run on CPU under legoESM's certified fp64/libm policy and compared
with the P3-writer NEMO acquisition from the resolved deck above.

The preregistered bar is bit identity for every compared active value of
`T`, `S`, `u`, `v`, and `ssh` at every step: unequal-value count must be zero
and maximum ULP distance must be zero.  Any nonzero unequal count is **DEBT**,
not a climate verdict.  Each field's control plants one one-ULP perturbation in
an active value and must make the exactness gate fail.  A plant that does not
fail invalidates the instrument.

The existing shipped one-degree result is a regression control, not a new
measurement.  After parameterisation its card digest and ten-step trajectory
artifact must be byte-identical to the pre-change controls recorded under
`phase3/gyrehr_rounds/round1/`.

## Claim B: three-year integrations

No pointwise or trajectory match is claimed for the GPU integration.  The
preregistered expectation is that the trajectories **will diverge** after
spin-up: this GYRE step evaluates transcendental forcing, and the quarter-degree
regime is eddying.  The first transcendental reached by the legoESM surface
forcing is the seasonal cosine in `nemo_gyre_seasonal_cosines` (called first by
the shortwave path); its NEMO counterpart is the `COS` at
`WORK/usrdef_sbc.F90:106`.  The certified CPU mode uses the libm precision
policy, whereas the GPU mode must use fp64 native transcendentals and is
therefore labelled **non-certified**.

The later comparison is limited to pattern and climate statistics computed
over identical NEMO and legoESM windows: time-mean fields, EKE, jet position,
and spatial spectra.  The exact spin-up exclusion/window and statistic-specific
acceptance bars remain **OPEN** and must be preregistered before scoring; round 1
does not silently choose them.

## Instrument controls and blind spots

- The ladder sees state values, not all internal tendencies or geometry; card
  construction tests separately cover the resolved geometry and coefficients.
- The one-ULP plants prove sensitivity at the exact bit bar; they do not
  validate a future statistical climate instrument.
- Scan/per-step parity is a driver-control measurement only.  It cannot certify
  NEMO fidelity.
- Daily snapshots and daily means cannot resolve sub-daily phase differences.

## Acquisition dependency

Round 1 must stop the new-resolution ladder at `ACQUISITION_NEEDED` until the
operator runs the committed `--deck r4` acquisition command.  Neither the
author nor the sandbox may run `makenemo`, `mpirun`, or modify NEMO sources.
