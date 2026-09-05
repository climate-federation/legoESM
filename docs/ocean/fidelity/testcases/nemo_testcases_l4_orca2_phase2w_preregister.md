# NEMO testcase Lane 4 — ORCA2 Phase-2w preregistration

Date: 2026-09-06

Starting parent: `26af77048a8b28f0d4f29bc9e36699919615d0a2`

Status: **PREREGISTERED BEFORE PHASE-2W MEASUREMENT OR NEW ORACLE BUILD.**

## P2W-1 — Phase-2v replacement admission

Admit twin A only if the existing acquisition gate decodes the
`NEMO_L4_TKEW_1` record from its header to exact EOF, every unowned slot is
canonical zero, all 101 streams are raw-identical A/B, all 100 inherited
streams are raw-identical to the Phase-2s root, ordinary restart/history
payloads pass the established identity rule, and all eight plants exit
nonzero.  Twin B is the reproducibility witness.  Phase-2u remains
`REJECTED_NONREPRODUCIBLE_TKE_RECORD`; earlier roots remain retained witnesses.

## P2W-2 — ordered TKE walk

The production-JIT CPU/fp64/scalar-libm walk first scores the `taum` operand
actually handed to the TKE call against the dumped `taum`.  The post-NCAR
operand-substitution boundary is exact only at `0 / n`; a mismatch is a Lane-4
ORCA2 forcing boundary.  With that operand admitted, walk NEMO source order:

1. no-Stokes `zWlc2=zcsd*taum` and the scalar `zpelc` recurrence
   (`zdftke.F90:326-345`);
2. `imlc/zhlc/zus3`, the Langmuir source, Prandtl ratio, matrix and RHS
   (`:347-420`);
3. forward/RHS/back substitution and floor (`:451-469`), then `nn_etau=1`
   (`:490-496`);
4. resolved `nn_mxl=3`, `nn_mxlice=2`, `nn_pdl=1` mixing lengths and
   `avm/avt/dissl` (`:598-724`).

Every row is cellwise on defined rank-zero cells, reports `unequal / n` and
row-scale ULP, and has a target-bit plant that exits nonzero.  Stop at the
first non-bit statement.  Shared TKE arithmetic is
`GYRE_OWNER_SHARED_TKE`; an ORCA2 selector or forcing operand is Lane 4.

## P2W-3 — Decision 12, complete NEMO `nn_eice` dispatcher

NEMO assigns mode 1 to `TANH(10*fr_i)`, mode 2 to raw `fr_i`, and mode 3 to
`MIN(4*fr_i,1)` (`zdftke.F90:253-258,828-834`).  Keep the restored mode-1
meaning, add source-literal mode 2 to the one shared dispatcher, accept exactly
0/1/2/3, and raise otherwise.  Direct JIT/fp64 bit tests and a gradient check
must pass.  GYRE/LOCK/OVERFLOW/C1D and the ORCA1 CORE2 driver must be
bit-identical before/after because they select 0 or 3, never the newly added
mode 2.  This is User Decision 12, not a new default.

## P2W-4 — ORCA1-ice ORCA2 variant oracle

Create a config-local copy of the accepted icebergs-off ORCA2 variant and
change only `namelist_ice_cfg` to the resolved ORCA1 target supplied by User
Decision 11: `jpl=1`, `nlay_i=3`, `nlay_s=3`, `nn_icesal=2`, `ln_pnd=F`,
`ln_icedA=F`, `ln_landfast_L16=T`, `ln_rhg_EVP=T` with inherited
`ln_aEVP=T`, `ln_str_H79=T`, `rn_pstar=2e4`, `rn_crhg=20`, ridging and
rafting on, and Prather advection.  All ocean options, forcing, icebergs-off
choice, scalar-math toolchain, and `(jpni,jpnj)=(2,1)` remain fixed.

The WRITE-only instrument retains every V2 stream and adds self-describing,
zero-first, rank-zero frames at the first three executed ice calls.  With the
unchanged ORCA2 `nn_fsbc=2`, those are ocean `kt=1,3,5`; there is no SI3
operator call at ocean `kt=2` (`sbcmod.F90:477,604`).  The frames are:

- thermodynamic entry/exit state and column operands in `icethd.F90` order,
  including categories, thickness/volume, surface temperature, layered ice
  and snow enthalpy, layered salinity, and SI3 atmosphere/ice flux operands;
- dynamics entry state and Prather moments, surface/ocean stresses, strength,
  `a_i/v_i/v_s`, and the Lemieux-2016 landfast basal-stress operands in the
  executed rheology path.

Each header carries magic/version/kt/frame, all resolved allocation dimensions,
an explicit per-field extent table, a payload count derived from those extents,
and binary64 precision.  Two independent user-shell twins are admissible only
at raw identity for every stream and ordinary-output identity.  If the shipped
ORCA2 initialization cannot construct the `jpl=1` state, stop on the exact NEMO
diagnostic and report the smallest source-supported initial-state change; do
not choose it silently.

## ASKED / UNASKED

| action | classification | preregistered disposition |
|---|---|---|
| admit Phase-2v twins and pin A | ASKED | exact twin/inherited/schema/identity/plants gate only |
| execute ordered TKE walk | ASKED | stop at first non-bit statement; ownership rule above |
| add NEMO `nn_eice=2` | ASKED, User Decision 12 | raw fraction; defaults and other selectors unchanged |
| mode 1 semantic record | ASKED | old legoESM meaning raw fraction; restored NEMO meaning `tanh(10*fi)`; raw behaviour moves to mode 2 |
| build ORCA1-ice ORCA2 variant | ASKED, User Decision 11 | exact listed ice selectors, everything else fixed |
| interpret requested `kt=1..3` under ORCA2 cadence | source-resolved, no choice | first three executed SI3 calls are ocean kt 1/3/5; no synthetic kt=2 operator frame |
| execute MPI/NEMO | forbidden in sandbox | prepare two hash-guarded launchers for user shell |
| change shared TKE arithmetic after a non-bit row | Lane-4-forbidden | register reproducer and hand to GYRE |
| enter EVD/IWM after an open TKE boundary | downstream/unasked | do not enter until TKE closes |
| delete retained roots, edit shipped NEMO, commit large records, or push | forbidden | none planned |
