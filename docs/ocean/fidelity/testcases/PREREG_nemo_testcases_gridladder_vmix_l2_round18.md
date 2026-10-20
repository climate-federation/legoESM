# PREREG — lane B rung L2 residual walk (round 18, vmix_r18)

Frozen before running the round-18 walk or changing production code. Round
17's admitted record and observer remain the only measurements. The target is
the remaining kt-1 stage-3 lateral-diffusion trend debt: 7,878 cells exceed
`1e-6` of scale, none exceeds `1e-3`, and the maximum normalized difference is
`3.55187264925e-4`; the resulting T and S states each differ in 16,428 cells.

All oracle citations below are to the compiled
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/` source.

## Compiled-order boundary

The admitted slope observer already makes the four final `uslp`, `vslp`,
`wslpi`, and `wslpj` arrays exact. This round nevertheless checks the two
requested slope/surface leads first: the mixed-layer ramp and its stored
anchors in `ldfslp.f90:232-256` and `ldfslp.f90:284-317`, the zero surface
tracer gradient at `traldf_iso.f90:178-193`, and the zero surface vertical
flux plus first below-surface flux at `traldf_iso.f90:259-291`.

Before those flux statements, stage 2 stores the face stretch ratios as
`0.5 * (r3u(Kbb) + r3ua)` and `0.5 * (r3v(Kbb) + r3va)` at
`stprk3_stg.f90:205-214`. Stage 3 names that stored N+1/2 level `Kmm` at
`stprk3_stg.f90:228-232`. The isoneutral operator's first use is the `zA11`
and `zA22` assignment at `traldf_iso.f90:227-231`; the later horizontal flux,
vertical flux, and divergence statements are `traldf_iso.f90:233-291`.

The selected card already constructs the source-ordered stage-2 U/V ratios
for its momentum/tracer transport. Its Redi face-thickness path instead
reconstructs U/V ratios from the averaged sea level. Those are algebraically
equal but do not preserve the compiled stage-2 assignment boundary.

## Frozen prediction and falsifier

- **P18-1, predicted first non-bit statement:** the first mismatch is the
  stage-2 stored U/V ratio assignment at `stprk3_stg.f90:210-212`, first
  observed by `traldf_iso` in `zA11`/`zA22` at
  `traldf_iso.f90:230-231`. The current sea-level reconstruction will differ
  from NEMO's stored Kmm face values in at least one wet U or V face.
- **Refuting number:** **0 unequal wet face values in both U and V** between
  the current production face arrays and NEMO's values refutes P18-1. Any
  nonzero count confirms the boundary only if feeding the stored ratios makes
  those same face arrays exact.
- **P18-2, surface lead:** if P18-1 is refuted, the next predicted boundary is
  the first below-surface vertical-flux statement at
  `traldf_iso.f90:266-285`, not the already-exact slope producer. It is
  refuted if the NEMO and card `zfw_kp1` surface-interface rows are exact.
- **P18-3, landing:** a confirmed correction is L2-selected only, removes the
  first operand mismatch, and makes the kt-1 lateral trend exact or moves the
  walk to the next compiled statement. It lands only if its NEMO-value test
  fails when the wiring is removed under JIT, traced fp64 inputs, and every
  required control remains unchanged.
- **P18-4, ladder:** after the first-statement repair, both L2 labels at kt
  1..10 are predicted exact or AT-BAR. Any DEBT row refutes certification and
  remains named debt rather than being waived.

## Fixed controls

No card selection, threshold, cadence, state, or forcing changes. Jaxprs must
remain byte-identical for G5, V1-V5, OMT-0..4, rung 0, VORTEX, and SMT-1/2.
L1, full ORCA2, DINO, and SMT-3/4 values must remain bit-identical. GYRE's
recorded ten-step payload and residual archive must remain bit-identical.
The face gate must compare production values directly with NEMO values from an
admitted dump; rebuilding both sides from the same helper is not evidence.

If the admitted round-17 files do not contain the needed NEMO values, this
round will add a committed acquisition with read-only preflight,
operator-only `--run`, and `--admit-existing`, then stop with
`ACQUISITION_NEEDED`.
