# ORCA2 round 330 preregistration — walk inside `zdf_tke`

Frozen 2026-10-24 before reading any round-328 numerical payload in this
round. Source inspection is complete; the predictions below are not
measurements.

## Source-read discriminator

NEMO allocates `en`, `zdiag`, `zd_up`, and `zd_lw` through `jpk`. It writes
the bottom boundary to `en(mbkt+1)` at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:287-310`, but builds
and solves only levels `2:jpkm1`; the terminal solve row is assigned without
an `en(jpk)` term at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:411-478`.

The certified V1 path carries only the solved `2:jpkm1` rows. Its selected
bottom-boundary arm clamps `bottom_level` into that carried range and replaces
the selected row by an identity equation at
`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:1802-1830`.
Therefore a full-depth NEMO column (`mbkt+1=jpk`) leaves the boundary outside
the solve, while the card currently clamps it onto the last solved row. A
shallower column whose `mbkt+1<=jpkm1` retains an in-solve identity row.

## Frozen predictions and falsifiers

- R330-P1 **CONFIRM:** the admitted record has 4,757 full-depth columns with
  `mbkt+1=jpk`; all correspond to the zero-based `k=28` returned-energy
  family. **REFUTE:** that count or mapping differs.
- R330-P2 **CONFIRM:** an offline replay of the current comparison card that
  omits the bottom identity scatter only where `bottom_level` lies beyond the
  carried solve has zero unequal returned-energy bits at all 4,757 active
  `k=28` interfaces, or strictly fewer inequalities there without worsening
  any earlier level. **REFUTE:** the `k=28` score is unchanged or worse.
- R330-P3 **CONFIRM:** the same replay keeps the in-range scatter for shallower
  columns and introduces zero new inequalities at every level `k<28`.
  **REFUTE:** any previously exact earlier interface becomes unequal.
- R330-P4 **CONFIRM:** reverting only the range test restores the 4,757-row
  full-depth failure, proving the control is active. **REFUTE:** the planted
  old clamp remains green.

If P1--P4 hold, the candidate is one cited V1 statement and is measured under
Decision 122/139 on all seven pairs and both labels, with every other card's
program unchanged and the rung-10 N=1 reverse/forward gradients finite. It
lands only if those gates pass. If the replay cannot isolate the source row,
or any unrecorded `zdf_sh2`, `zdf_drg`, or `zdf_mxl` input remains necessary
to decide ownership, this round instead extends the reviewed round-328
acquisition and stops for the operator-run record.
