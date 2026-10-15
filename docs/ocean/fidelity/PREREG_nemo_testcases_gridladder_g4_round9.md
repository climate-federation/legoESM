# Preregistration — grid ladder G4 round 9 (gridladder_r30), stage-3 thickness ladder

Frozen before any round-30 G4 measurement. Round 29 (G1b) changed shared code
only behind G1b-selected options; G4's arithmetic is unchanged at `a6bce3f6d`.

## Fixed program and labels

- Rung G4, Decisions 120/121/123 unchanged. Record
  `gridladder_rounds/round21/oracle_g4_r21` (build `ORCA2G1_OMIP_L1_P5`).
  fp64, CPU. Bar: the ladder's existing normalized 1e-15; no threshold change.
- Before arm: the card at `a6bce3f6d` (round 8 landed): kt 1 stage-3 T
  200 unequal, max 7.105427357601002e-15; S, ssh, uu_b, vv_b, v exact; u 56
  signed-zero bits.

## NEMO's stage-3 tracer solve thicknesses (read off the compiled source)

At stage 3, Kbb = N, Kmm = N+1/2, Kaa = N+1 (`stprk3_stg.f90:223`), and every
thickness is the stored ratio `e3t_3d*(1+r3t(K)*tmask)`:

- `r3t(Kmm)` = stage 2's `r1_2*(r3t(Kbb)+r3ta)` (`stprk3_stg.f90:215`);
  `r3t(Kaa) = r3ta` (`:236`); `r3ta = ssha*r1_ht_0` (`domqco.f90:257`).
- non-solar heat divisor `e3t(Kmm)` top cell (`trasbc.f90:306-307`);
  runoff divisor `h_rnf = e3t(Kmm)` top cell, set by the stage-3 `div_hor`
  (`sbcrnf.f90:281`), read at `trasbc.f90:322-329`;
- solve RHS `e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs` (`trazdf.f90:284-289`);
- diagonal `e3t(Kaa) - (zwi+zws)` (`trazdf.f90:233`), off-diagonals
  `-p2dt*zwt/(e3w_1d*(1+r3t(Kmm)))` (`trazdf.f90:231-232`).

The card: Kbb and Kaa thicknesses already use the stored product
(`compute_ocean_jacobian`); Kmm is the thickness MEAN `0.5*(h_b+h_a)` for the
heat/runoff divisors and the RHS, and the off-diagonals use `1+eta/H` of the
mean sea level (quotient, not the stored stage-2 ratio).

## Candidate

One G4-selected option: stage 3's Kmm thickness is `e3t_0*(1+r3t_stage2)`
with NEMO's stored stage-2 ratio, in the heat divisor, the runoff divisor,
the Kmm solar divisor, the solve RHS, and the off-diagonal `e3w(Kmm)` — the
whole solve together (round 8's finding 4).

## Predictions

1. The card's Kbb, stage-3 Kmm and Kaa ratios equal the record's `e_r3t_bb`,
   `2_r3t_aa`, `3_r3t_aa` bit for bit on every wet column.
2. With the option: kt 1 stage-3 T unequal falls below 200 (target 0); S stays
   exact; ssh, uu_b, vv_b, v unchanged.
3. Both labels kt 1..10: no AT-BAR row leaves the bar; INDEPENDENT
   first-over-bar (kt 9 ssh/u/v) unchanged or later.

## Falsifiers and controls

- T unequal >= 200, or any cell above 7.105e-15, or S unequal → REFUTED; the
  option is not landed and the receipt walks the remaining cells.
- Deselecting the option must restore 200 (non-vacuity).
- Prediction 1 failing means the instrument is wrong before the physics.
