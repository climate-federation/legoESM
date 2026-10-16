# Preregistration — grid ladder G5, round 3 (gridladder_r33), ice-ocean top drag

**Frozen:** 2026-10-15, before changing the card or measuring a candidate.
Record: `gridladder_rounds/round31/oracle_g5_r31`, admitted in round 2.
Bar: the ladder's existing normalized 1e-15; no threshold change.

## Fixed NEMO path

The compiled G5 deck resolves `ln_drgimp=.true.` and
`ln_drgice_imp=.true.` (`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/EXP00/namelist_ref:
817-818`), while `ln_bt_fw=.true.` (`:1091`) and wetting/drying is off.  The
top-friction namelist states `rn_Cd0=1e-3` and `rn_Uc0=0.4` (`:821-824`), but
there is no ice-shelf cavity on this deck, so the executing `zdf_drg('TOP')`
arm copies SI3's coefficient directly: `rCdU_top=rCdU_ice`
(`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/zdfdrg.f90:150,178-190`).
The admitted record already contains `rCdU_ice`; no acquisition is proposed.

The candidate will transcribe the four consumers in compiled order, behind one
G5-selected option whose default is false and which refuses a missing recorded
coefficient:

1. face coefficient `pCdU_u/v`, the bottom-pair plus top-pair followed by the
   factor one half (`dynspg_ts.f90:1404,1428-1433`);
2. forward-level top baroclinic residual correction (`:1466-1489`);
3. the frozen coefficient's active non-wetting/drying substep use
   (`dynspg_ts.f90:290-291,633-640`);
4. `dyn_zdf`'s top-cell barotropic RHS terms and implicit matrix diagonals
   (`dynzdf.f90:158-180,303-314,470-482`).

The same statements, opened at the same enclosing subroutines, are text- and
line-identical in the Decision-122 `ORCA2_OMIP_L4/BLD/ppsrc/nemo` build.

## Predictions and exact verdicts

1. The routed card coefficient equals NEMO's recorded `rCdU_ice` bit for bit
   for kt 1..10.  At kt 1 the card's U/V face coefficients equal the recorded
   `zCdU_u/v` on every scored wet face, including the 68 north-fold V faces.
2. In a kt-1 staged walk, entry remains exact and every stage-1 and stage-3
   field (ssh, uu_b, vv_b, all-level u, v, T and S) is AT-BAR.  Each consumer
   boundary is also exercised by a focused unit test and a planted coefficient
   perturbation makes at least one comparison fail.
3. GIVEN-NEMO-ENTRY and INDEPENDENT are both AT-BAR for every scored surface,
   region and depth-class row at kt 1..10.  If true, the sole remaining G5
   unmeasured label is removed and G5 is CERTIFIED.
4. If the first candidate leaves any kt-1 field above the bar, it is REFUTED;
   retain the unmeasured label, identify the first remaining compiled-order
   consumer/statement by the staged walk, and continue only with a separately
   reported causal arm.  Any kt 2..10 leave after an exact kt 1 is likewise
   DEBT, not a certification.

## Controls

- Default false must leave every pre-G5 card's configuration and arithmetic
  path unchanged.
- Selecting the option without `rCdU_ice` must raise, never substitute zero.
- Deselecting it on G5 must restore round 2's kt-1 first leave.
- A sign plant on one nonzero ice coefficient must fail coefficient parity and
  move the kt-1 result; a zero/non-reading consumer is not an acceptable
  control.
