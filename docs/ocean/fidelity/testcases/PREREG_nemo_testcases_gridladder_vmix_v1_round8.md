# PREREG — vertical-mixing lane V1 round 8 (vmix_r8)

Frozen before the landing and the kt 1..10 ladders. Record: `gridladder_vmix_rounds/round1/oracle_v1_r1`
(admitted round 3). Decisions 129 = 1 and 130 = 1 in force.

Already measured (offline, evidence `gridladder_vmix_rounds/round8/`, done BEFORE this file):
on the pivot row (card row 147 = NEMO jj=148, mirror-symmetric in NEMO's T), a numpy replay of
`zdf_sh2` (`zdfsh2.f90:98-113`) with the card's wall north face reproduces the card's kt-2
shear in 1,292 of 1,292 wet cells; the replay with NEMO's fold images is exactly mirror-symmetric
and differs from the card in 1,251 cells. NEMO's images: the pivot row's north V face is wet
(raw `vmask`), its north viscosity is `avm_k(ji,jj+1)` = the fold image of row 146 (W, +1), and
its `r3v` reads `e1e2t*ssh` at the fold image of row 146 (T, +1) — `lbc_nfd_dp`, T-pivot,
`lbcnfd.f90:584-720`, applied by `stp_RK3_stg`'s `lbc_lnk` (`stprk3_stg.f90:785-793`) and the
ssh exchange. The card keeps that face a wall (mask 0, coast factor 2), edge-repeats the
viscosity, and wraps the ratio's north neighbour to row 0.

Landing (V1-selected only): those three operands from the fold images on the top V face.

- **P1.** kt-2 `en` over the bar on the pivot row: 75 cells -> 0. Falsifier: any.
- **P2.** kt 1 stays bit-identical on every field under both labels. Falsifier: any unequal cell.
- **P3.** kt-2 `en` full-depth max miss falls below 5.57e-7. Falsifier: it does not.
- **P4.** V1 certifies (every field at the bar kt 1..10, both labels). PLAUSIBLE only; falsifier:
  any DEBT row.
