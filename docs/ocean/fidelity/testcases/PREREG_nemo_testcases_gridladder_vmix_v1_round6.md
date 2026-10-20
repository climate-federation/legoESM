# PREREG — vertical-mixing lane V1 round 6 (vmix_r6)

Frozen before measuring. Record: `gridladder_vmix_rounds/round1/oracle_v1_r1`
(admitted round 3). Deck: Decision 129 = 1 (NEMO's own TKE floors, internal-wave
mixing off), Decision 130 = 1 (new statements V1-selected only).

Question 1 (kt-1 owner): the 2 over-bar `avm` cells (10,120) and (21,113) at
w-level 20, and the 1 over-bar `avt` cell, are missed identically by NEMO's
own-input replay, which feeds the card's rn2. Round 5 named N2 (PLAUSIBLE).

Instrument: an offline numpy replay of NEMO's literal `rab_3d_t` (EOS-80
branch, `eosbn2.f90:1256-1333`) and `bn2_t` (`eosbn2.f90:1604-1640`), one
IEEE binary64 operation per Fortran operation in source order, fed NEMO's own
kt-1 step-entry T/S (`oracle_step_entry_kt00000001.bin`). At kt 1
`rn2 = rn2b` (`stprk3.f90:176-180`), so one array serves both.

- **P1.** The literal replay's rn2 differs from the card's rn2 at the 3 cells.
  Falsifier: bit-identical there (owner is not N2).
- **P2.** Feeding the literal rn2 (and NEMO's en) to the round-4/5 NEMO-input
  replay brings the 3 cells to the bar. Falsifier: any of them stays over.
- **P3.** The first non-bit statement is named by comparing the card's bn2
  intermediates (zrw, zaw, zbw, products, division) and alpha/beta with the
  replay's. Any landed fix is V1-selected only (G1-G5, full ORCA2, library
  defaults unchanged).

Question 2: after Q1, the first kt-2 over-bar row (step, stage, field, cells)
and its first non-bit statement. No prediction is frozen for its identity.
