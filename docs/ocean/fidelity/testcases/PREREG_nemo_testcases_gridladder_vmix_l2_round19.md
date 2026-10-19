# PREREG — lane B rung L2 final arithmetic walk (round 19, vmix_r19)

Frozen before changing production code or running a round-19 score.  Round
18's admitted operator record remains the oracle; no NEMO acquisition is
expected.  The target is the one-ULP `ah_wslp2` seed that leaves two T and two
S cells unequal at kt 1 and grows into later SSH/velocity debt.

All oracle citations below are to the compiled
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/` source.

## Compiled arithmetic boundary

`SUBROUTINE traldf_iso_a33` is the executing standard-isoneutral A33 routine
(`traldf_iso.f90:746-775`).  It adds the four U diffusivities and the four V
diffusivities as two bracketed pairs, multiplies each result by its wet-face
normalizer, then forms `ah_wslp2` in written left-to-right multiply/add order
(`traldf_iso.f90:782-794`).  ORCA2 selects the standard isoneutral Laplacian,
MSC, and mode-20 coefficient branches (`EXP00/namelist_cfg:301-307`), so these
statements run.

Round 18 replayed NEMO's own operands: the pair association gives 0 unequal
cells, the shared helper's left-to-right sum gives 18,465, and the same helper
under JIT gives 60,720.  The extra JIT mismatches were independently identified
as multiply-add contraction: disabling FMA returns the eager 18,465.  This
round will select NEMO's bracketed and source-rounded statement only for L2;
the shared default stays unchanged.

The earlier compiled boundary is the stored U/V face ratio.  Stage 2 stores
`0.5*(r3u(Kbb)+r3ua)` and its V twin at N+1/2
(`stprk3_stg.f90:205-213`); stage 3 names that level Kmm
(`stprk3_stg.f90:228-232`).  `dom_qco_r3c_RK3` constructs each endpoint face
ratio in source order (`domqco.f90:207-217`), and the live face thickness is
the reference thickness times one plus the stored ratio.  Round 18 measured
36 U and 20 V off-fold faces one ULP from NEMO.  Reusing the stored ratios made
the off-fold faces exact but must not replace the existing north-fold V row:
that discarded arm moved all 1,319 wet pivot-row V faces.

## Frozen predictions and falsifiers

- **P19-1, A33 sum:** an L2-only bracketed pair sum plus
  `nemo_source_round` at the written statement boundaries reproduces a NumPy
  transcription of NEMO's order exactly under JIT on synthetic traced fp64
  operands.  The planted left-to-right association must differ.  Any JIT
  mismatch to the NumPy transcription refutes the implementation.
- **P19-2, direct operator record:** after the guarded sum, production
  `ah_wslp2` has 0 unequal cells against the admitted NEMO dump.  Any unequal
  cell refutes closure.  Removing the guard must restore a nonzero mismatch.
- **P19-3, face lead:** L2 consumes its already-materialized stage-2 stored
  face ratios off the fold while retaining the current fold row.  Direct
  NEMO face-thickness mismatches fall from 36 U / 20 V to 0 / 0, and all
  1,319 wet pivot-row V faces remain exact.  Any direct mismatch or any moved
  pivot face refutes the fix.
- **P19-4, certification:** both independent and given-NEMO-entry L2 labels,
  kt 1..10, have every scored row exact or AT-BAR under D127's `1e-15` rule.
  Any DEBT row refutes L2 certification and remains named debt.

## Fixed controls

No field set, threshold, cadence, forcing, state, or acquisition changes.
The new A33 and face arithmetic is selected by an explicit L2-only card field;
all defaults remain false.  G5, V1-V5, L1, full ORCA2, DINO, OMT-0..4, rung
0, VORTEX, and SMT-1..4 must retain their pre-round jaxpr hashes or exact
values as appropriate.  GYRE's ten-step JSON rows and residual archive must
remain byte-identical to round 18.  Every new exactness check carries a planted
violation that must fail.

If the admitted record proves insufficient, this round will write and preflight
an acquisition script under the standing checklist and stop.  It will not
build or run NEMO.
