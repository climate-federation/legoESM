# Grid ladder rung G4 — round 2 preregistration (gridladder_r18)

Base: 1355c60b5 (G4 round 1 + the admission fix below). Record:
gridladder_rounds/round17/oracle_g4_r17 (operator's acquisition).
Evidence: gridladder_rounds/round18/. NEMO citations are to
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/<file>.f90:lines` (same sources as the
record binaries). Decisions 120 and 121 pending: as-built deck (no
hand edits, SSS restoring ON).

## 0. Already measured before this file (admission, not a card comparison)

- The record was refused for three non-finite words in `i000_zCdU_v` at
  kt = 1, (jj, ji) = (1, 98), (2, 77), (129, 2), identical in all twins.
  All three are halo cells: NEMO ALLOCATEs `CdU_v` uninitialised
  (`stp2d.f90:122`), sets it on ntsi..ntei x ntsj..ntej only
  (`dynspg_ts.f90:1238-1247`; tiling off, so the inner domain), reads it
  only there (`:518`, `:575`) and copies the whole array into the recorded
  `zCdU_v` (`:277`). Checker fixed to test finiteness on the consumed
  support (1355c60b5), record re-admitted with `--admit-existing`:
  ACQUISITION_DONE, all eight plants fired.
- What the record holds about forcing: the RK3 restarts hold NO surface
  flux (no `sbc_hc_b`, stress or emp; round 1's section 5 expected
  `sbc_hc_b` — wrong); they hold `fraqsr_1lev`, the surface means
  `sst_m`/`sss_m`/`ssu_m`/`ssv_m` and the fwb state. The substep frames
  hold `ssh_frc` = r1_rho0·(emp − rnf) (`stp2d.f90:242-245`, copied at
  `dynspg_ts.f90:272`) and `zu_frc`/`zv_frc` (depth-mean RHS plus
  r1_rho0·utauU, `stp2d.f90:198`) at kt = 1, 2. NEMO's kt = 11 abort file
  (fp64) holds utau, vtau, qsr, qsr + qns and emp − rnf of the kt = 11 sbc
  call (`diawri.f90:638-649`).
- The card has no forcing evaluator: it names the forcing; the G2/G3
  ladder steps with zero forcing and refuses G4. The shared NCAR leaf is a
  registered debt (ORCA2 lane, phase 2h: over the bar on all seven scored
  outputs, oracle-supplied inputs). No NEMO-statement SSS restoring
  (`sbcssr.f90`) exists in shared code.

## 1. This round's measurement (frozen before it runs)

M1, kt = 11 forcing witness, offline. Inputs: the CORE files the run read,
at the records NEMO's log states for kt = 11 (`ocean.output:1860-1868`:
6-hourly record 6, daily record 2, monthly record 1), mapped by the
shared `nemo_fld_interp` + `rotate_en_to_ij` (certified O1-M); the surface
state NEMO's kt = 11 bulk reads, `sst_m` from twin a's step-10 restart.
Card arithmetic: the shared `air_sea_fluxes(algo="ncar")`. Scored against
`output.abort.nc` on wet T cells (inner domain), bit test then the bar.

- G4R2-P1. utau, vtau: NOT at the bar on most wet cells (as O1-B).
  Falsifier: at the bar. PLAUSIBLE.
- G4R2-P2. qsr = (1 − albo)·SW (`sbcblk.f90:686`): bit-identical.
  Falsifier: any unequal wet cell. PLAUSIBLE.
- G4R2-P3. qns and emp − rnf: not scored (qns needs the full blk_oce_2
  assembly; emp needs NEMO's SSS restoring, absent from shared code).

## 2. Ladders kt = 1..10

- G4R2-P4. Not runnable this round under either label: INDEPENDENT needs a
  card that computes the forcing (it does not); GIVEN-NEMO-ENTRY needs
  NEMO's per-step forcing fields (the record has none). No certification.

## 3. First non-bit statement (predicted)

The NCAR bulk's turbulent transfer: `turb_ncar` called at
`sbcblk.f90:706`, its iteration in `sbcblk_algo_ncar.f90:59-238`, the
stress `utau = zztmp·pwndi` at `sbcblk.f90:802-803`.
