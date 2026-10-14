# Grid ladder rung G4 — round 2 receipt (gridladder_r18)

Status: **STOPPED_FOR_DECISION.** Record admitted; nothing landed in the
model; G4 is NOT certified. The card cannot be laddered kt = 1..10 under
either label (it has no forcing evaluator, and the record holds no
per-step forcing to substitute).
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g4_round2.md
(86c7d34e7, before the comparison; one citation off by a line: the
`ssh_frc` copy is `dynspg_ts.f90:273`, not `:272`).
Code: 1355c60b5 (admission fix), 627c0c8a9 (kt = 11 witness), review fixes and this receipt's update in the last commit.
Record: gridladder_rounds/round17/oracle_g4_r17. Evidence: gridladder_rounds/round18/.
Decisions 120 and 121 pending: as-built deck.

## Take-aways

1. **The refusal was the checker's.** Three NaN words in the solver's
   recorded drag coefficient at kt = 1, all halo cells NEMO never reads.
   Fixed the check; re-admitted without re-running NEMO: ACQUISITION_DONE.
2. **Shortwave into the ocean is NEMO's bit for bit** at kt = 11 (all
   16,433 wet cells): file record, interpolation weights, albedo.
3. **Wind stress is not:** direction exact (max angle 1.9e-16 rad); size
   off on every wet cell, two populations: 16,161 cells by < 1e-6
   relative, 272 cells by 1e-6 to 80 % (up to 1.2e-3 Pa), all where the
   sea is 23-39 K warmer than the air (ice-free polar water under cold
   air: a consequence of running without ice). Over the bar.
   (Retracted from this receipt's first commit: "one-signed 4e-10 to
   4e-9" — that was the strong-stress subset only; 117 cells are low.)
4. **First non-bit statement: NEMO's NCAR bulk stress magnitude**, the
   same shared leaf the ORCA2 lane registered as owned debt.
5. **No ladder, no certification**: the next record must hold NEMO's
   per-step surface fluxes (decision below).

## 1. Admission (the run.sh line-309 refusal)

| item | value |
|---|---|
| non-finite words | `i000_zCdU_v`, kt = 1, (jj, ji) = (1, 98), (2, 77), (129, 2); same in twins a, b, c |
| inner domain (Fortran) | ji 3..182, jj 3..150: all three outside |
| NEMO | ALLOCATE uninitialised `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:122`; set on the inner domain only `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1238-1247` (tiling off); read only at `:518` and `:575`; whole array copied at `:277` |
| card reader | cuts every plane to the inner domain |
| fix | finiteness on the consumed support (inner domain of full planes, other arrays whole); halo NaNs counted in the JSON; new plant `substep-inner-nonfinite` |
| re-admission | `run.sh --admit-existing`: PASS_G1_RECORD, 60 frame fields per stage step, twin restarts passive, frames identical kt 1..10, ACQUISITION_DONE; 8 of 8 plants fired |
| record | kt = 1..10; NEMO stops at kt = 11 on \|V\| = 10.25 at (22, 84, 27), as G3 (prediction G4-P2 HELD) |

## 2. Forcing identity (INDEPENDENT label; kt = 11, the only step with NEMO fluxes)

The RK3 restarts hold no surface flux; the frames hold only two composites
(sea-level forcing = emp − rnf, momentum forcing = depth-mean RHS + stress)
at kt 1 and 2. NEMO's kt = 11 abort file holds stress, shortwave and total
heat flux of the kt = 11 call. Inputs: CORE records from NEMO's log for
kt = 11, NEMO's surface mean from twin a's step-10 restart.

| field | unequal wet cells | normalised max | status |
|---|---|---|---|
| shortwave qsr | 0 / 16,433 | 0 | bit-identical |
| zonal stress utau | 16,433 / 16,433 | 4.1e-4 | over the bar |
| meridional stress vtau | 16,433 / 16,433 | 1.0e-3 | over the bar |
| stress size, \|rel\| < 1e-6 | 16,161 cells | wind p50 7.7 m/s | SST − air p50 +0.9 K |
| stress size, 1e-6 to 1e-3 | 223 cells | wind p50 6.6 m/s | SST − air p5..p95 +23..+34 K |
| stress size, > 1e-3 | 49 cells | wind p50 5.6 m/s | SST − air p50 +32 K |
| non-solar heat, emp − rnf | not scored | — | full heat assembly not wired; NEMO's SSS restoring absent from shared code |

Predictions G4R2-P1 (stress over) and G4R2-P2 (shortwave bit) HELD. The
one-cell plant moves the at-bar row (exit 1, kt11_plant.log).

## 3. First non-bit statement

The NCAR bulk's stress magnitude: `turb_ncar` (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/sbcblk.f90:706`),
air density from the 10 m state (`:782-784`), the bulk formula giving
taum (`:787-791`), stress = taum/wndm × wind (`:800-803`). The operand
inside it is not isolated. PLAUSIBLE, from the population split: the
strongly unstable branch of `turb_ncar` — the stability-parameter clamp
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/sbcblk_algo_ncar.f90:177-178`), the
neutral-wind floor (`:207`) or the stability flag update (`:223`).
Isolating it needs NEMO's bulk intermediates (Cd, zeta) at these cells. Not
landed: no cited single statement, and the ORCA2 lane holds it as owned
debt (phase 2h O1-B: over the bar on all seven outputs).

## 4. The ORCA2 lane's forcing statements on G4

| statement (ORCA2 round) | G4 exercises it | holds on G4 |
|---|---|---|
| fld_read mapping (O1-M) | yes | yes for the daily shortwave (bit); wind direction exact; others only through the bulk |
| RGB light deposit (10) | yes | not measured: the card cannot step forced; its input qsr is bit |
| runoff tracer source / water (12, 13) | yes | not measured: rnf witnessed only inside emp − rnf |
| slow forcing substitution (16, HELD) | yes | not measured |
| NCAR bulk (O1-B, debt) | yes | no (section 2) |

## 5. Ladders kt = 1..10

Not run (prediction G4R2-P4 HELD): the INDEPENDENT label needs a card
that computes the forcing (it only names it; the ladder steps with zero
forcing and refuses G4); GIVEN-NEMO-ENTRY needs NEMO's per-step fluxes.

## 6. Decisions pending, rows each touches

- **120** (G3 hand edits): the Alboran/Red Sea cells' initial T/S → every
  G4 row through the surface mean there (bulk heat, stress, SSS restoring).
- **121** (SSS restoring): emp, hence the sea-level forcing composite and
  level-1 salinity at every step.

## 7. Tests (synchronous)

Focused battery (G1 tools, G2, G2 ladder, G3, G4): 68 passed, 9
deselected (JAX on CPU), before and after the review fixes. Non-vacuity: the new finiteness test fails on the
old checker; the old fixture's NaN sat in the halo and was moved inside
(the expectation, not the code, was wrong). Witness plant: exit 1.

## 8. Review

Single review (codex): **"VERDICT: findings"** (codex_review_r18.log), all
three confirmed and fixed in the witness commit after this receipt's first:
- HIGH, receipt claimed a one-signed 4e-10..4e-9 bias: measured 117
  negative cells, range −0.80..+0.0019; retracted, the witness now emits
  the population split (section 2).
- MEDIUM, stress plant vacuous (rows already over): the plant now runs
  only on rows at the bar unplanted (qsr) and names the rest; exit 1.
- MEDIUM, no provenance: the witness now refuses unless the abort file is
  NEMO's kt = 11 stop, NEMO's log reads the records it uses, and every
  input file is byte-equal to the abort run's (wrong abort file: exit 73).
  Refusals exit 73, never 1 (a fired plant). The NCAR defaults it
  inherits (iterations, salt factor, no relative wind) match the deck by
  reading, not by a check.

## 9. Choices

ASKED (round brief): admission fix on the consumed support; re-admit
without re-running NEMO; as-built deck.
UNASKED, each offered for revert: (1) the new plant added to G4's run.sh
plant list; (2) scoring at kt = 11 from the abort file (past the record)
as the only fp64 flux witness; (3) the campaign bar (1e-15 normalised)
for forcing rows.

No gate exists for: dual review (single, lane rule 6), claim review before
code, non-vacuity (shown above).

UNVERIFIED:
- that NEMO's kt = 11 bulk read the 10 m air fields without temporal
  interpolation (deck `.false.`; log names one record each);
- the operand inside the bulk that carries the bias.
