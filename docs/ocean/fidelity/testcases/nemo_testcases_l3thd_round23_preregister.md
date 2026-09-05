# Lane 3b round 23 preregistration — ORCA1-ice real-geometry exact inputs

Date: 2026-09-06

Tracker: `climate-federation/legoESM#1699`

Parent: `2bbea6d612416c4a3d5e7fb589e3d85c57875a36`

Status: **PREREGISTERED BEFORE GATE IMPLEMENTATION OR CLAIMED SCORE.**

## Decision 11 and vehicle

User Decision 11 fixes the target at the ORCA1-resolved identity: one category,
three ice plus three snow layers, P07/BL99, salinity option 2 with
`rn_sinew=0.75`, no ponds, no lateral melt, Lemieux-2016 landfast, aEVP, and
Prather transport.  The five-category rung is dropped.  The vehicle is the
ORCA2 grid with icebergs off and that exact ice namelist, produced by Lane 4.
No dynamics implementation or five-category fallback is authorized here.

The Lane-4 branch currently exposes the phase-2w writer and schema sources at
`b7ce08cc8afa`.  Filesystem reconnaissance found failed, record-free phase-2w
roots and a completed but not yet twin-admitted phase-2x A root.  Phase-2x B is
staged but has no outputs.  A is diagnostic evidence only until B reproduces
it.

## Rule 0 — executed source order and time levels

NEMO 5.0.2 `icestp.F90:126-136,151-206` executes an ice step only when
`MOD(kt-1,nn_fsbc)==0`, computes `t_bo`, stores state, runs dynamics, converts
global-to-equivalent fields, computes atmosphere/ice fluxes, and then calls
`ice_thd`.  The resolved `nn_fsbc=2` and `rDt_ice=21600 s` therefore execute
SI3 at ocean steps 1, 3, and 5 for the requested three frames
(`icestp.F90:126`, `:377-380`; `sbcmod.F90:474-477,604`).

Inside `ice_thd`, the live order is:

1. global ENTRY before `ice_thd_frazil` (`icethd.F90:112-115`);
2. frazil collection, active-point selection, and 2-D-to-1-D conversion
   (`icethd.F90:115-140`);
3. BL99/P07 `ice_thd_zdf` (`icethd.F90:143-148`, concrete dispatcher
   `icethd_zdf.F90:47-71` and solve `icethd_zdf_bl99.F90:34-871`);
4. enabled `ice_thd_dh` (`icethd.F90:150`, `icethd_dh.F90:34-533`);
5. temperature inversion, option-2 salinity, then temperature inversion
   (`icethd.F90:152-156`, `icethd_sal.F90:60-632`);
6. no virtual-ITD or lateral-melt call because both selectors are false
   (`icethd.F90:158-163`);
7. no category remap because `jpl=1`, followed by enabled open-water growth,
   correction, aging, and LBC (`icethd.F90:176-190`,
   `icethd_do.F90:56-343`).

The Lane-4 self-describing f0 frame is current global state immediately before
item 1; f1 is current global state after item 7.  The atmosphere/ice flux
members retain their `ice_sbc_flx` input time level in both files; they are not
post-thermodynamics state.  The gate will encode this registry and reject any
unregistered frame, selector, extent, step, or time level.

## Pre-implementation search and reuse

The search found the Round-20 admission gate
`nemo_orca2_si3_exact_input_gate.py`, the existing dimension-derived legacy
stream gate, the shared `si3_column_step_arrays`, and the shared SI3 exchange
operators in `ice/sea_ice.py` and `coupler/ocean_forcing.py`.  It also found
Lane 4's self-describing phase-2w parser and synthetic formats.  This round
will extend the Round-20 gate; it will not create another thermodynamics model,
exchange module, or ORCA2 writer.

## Header reconnaissance and fixed predictions

Header-only reconnaissance of provisional phase-2x A, performed before gate
implementation, observed:

- six `NEMO_L4_O1ITHD1` files at kt 1/3/5, frames 0/1;
- `jpl=1`, `nlay_i=nlay_s=3`, binary64, 16 fields per frame;
- a full state allocation of 94 by 152 and flux allocation of 90 by 148;
- the inherited legacy thermodynamics stream has all eight call-order frames
  for five executed ice calls; and
- kt1 has 1,700 active points at global ENTRY but 1,777 at POST_ZDF/ZDF input.

The final gate is predicted to:

1. reject both failed phase-2w roots with a named missing-record `GateError`;
2. admit a valid synthetic six-file schema and make every schema/row plant exit
   nonzero;
3. admit phase-2x A's selectors and six self-described files;
4. score the eight state fields common to the new and legacy ENTRY/EXIT writers
   at **0 / n non-bit**; and
5. stop physics scoring at `POST_FRAZIL_PRE_ZDF_1D` with
   `UNMEASURED_MISSING_OPERAND`, not manufacture the 77 new columns from the
   pre-frazil frame.

A non-bit common-writer row refutes item 4 and becomes the earlier stop.  A
complete future operand frame activates the existing shared-column scorer; its
target is 0 / n and the first non-bit operand/boundary is the result.

## Required Lane-4 WRITE-only handoff if prediction 5 holds

At `icethd.F90:140`, after `CALL ice_thd_1d2d(jl,1)` and before the initialized
tendencies / `CALL ice_thd_zdf` at `:143-148`, write one self-describing
`POST_FRAZIL_PRE_ZDF_1D` frame per executed call containing:

- `nptidx` with its global rank-zero T-cell identity;
- `a_i_1d`, `h_i_1d`, `h_s_1d`, `t_su_1d`, `e_i_1d`, `e_s_1d`, `s_i_1d`,
  `sz_i_1d`, `oa_i_1d`, and `t_s_1d`; and
- the already existing ZDF-input members at that same 1-D time level:
  `qns_ice_1d`, `qsr_ice_1d`, `dqns_ice_1d`, `qtr_ice_top_1d`, `t_bo_1d`,
  `sss_1d`, `evap_ice_1d`, `sprecip_1d`, `qprec_ice_1d`, `qcn_ice_1d`,
  `qsb_ice_bot_1d`, `fhld_1d`, and `qlead_1d`
  (`icethd.F90:343-435` in the 2-D-to-1-D conversion).

For a wholly self-describing call-order score, mirror the seven registered
post-call boundaries after ZDF, DH, TEMP1, SAL, TEMP2, DO, and correction/EXIT,
with extents and payload counts derived from the write list.  The retained
legacy stream contains these values, but its field extents are implicit rather
than individually declared.

The atmosphere/ocean exchange score additionally needs full rank-zero,
self-describing operand/output frames rather than the existing one-point bulk
probe: `blk_ice_1/blk_ice_2` around `sbcblk.F90:1218-1346`;
`ice_flx_other` inputs and outputs at `icesbc.F90:307-437`; and the heat/water/
salt accumulator inputs plus outputs bracketing `ice_update_flx` at
`icestp.F90:201-213` / `iceupdate.F90:105-194`.  Include `ssmask` at the same
time level so the gate can enumerate every rank-zero wet cell rather than infer
wetness from a physically valid zero.  This is a handoff specification only;
Lane 3b will not edit or build the ORCA2 instrument.

## Coverage gain over C1D

| behaviour absent or trivial in the one-column rung | real-geometry disposition |
|---|---|
| spatially varying atmosphere/ice and ocean/ice forcing | will be certified cellwise once full exchange operands exist |
| spatially varying ice/snow state and branch census | will be certified for every recorded active rank-zero column once PRE_ZDF state exists |
| frazil activation of previously open cells | exposed by the vehicle; remains ORACLE_SUPPLIED/UNMEASURED unless separately implemented and framed |
| partial ocean cells affecting `e3t_m`, `t_bo`, `qsb_ice_bot`, `fhld`, `qlead` | will be certified through `ice_flx_other` only with its requested operand frame |
| north-fold/halo exchange and canonicalization | schema/canonical bridge is checked now; physical LBC/fold operator is not certified by a rank-zero scalar column |
| `nn_fsbc=2` cadence | frame sequence 1/3/5 is certified now; temporal operator output awaits complete operands |
| landfast L16, aEVP, Prather | selector identity is admitted; dynamics remain Lane 3a scope, not a thermodynamic claim |

## Binding controls and classifications

The synthetic dry-run will poison magic, version/extent/count, EOF,
step/frame order, selector, cadence, and each of the eight common ENTRY/EXIT
state rows.  Every plant must produce a named nonzero failure.  Real-data row
plants are private gate arguments, not physical selectors.

All results use `CONFIRMED`, `PLAUSIBLE`, or `UNMEASURED`.  AT_BAR and bit
identity remain separate.  No owner is assigned downstream of a missing input.
The C1D kt2 slab row remains registered to the GYRE lane and is untouched.

## ASKED / UNASKED

| choice/action | status | disposition before implementation |
|---|---|---|
| adopt ORCA1 single-category identity and drop five-category rung | ASKED, Decision 11 | exact selector admission only |
| build self-described exact-input parser/gate and row plants | ASKED | extend Round-20 gate |
| dry-run failed roots and synthetic schema | ASKED | fail closed / bind controls |
| use provisional phase-2x A before twin admission | UNASKED as oracle pin | diagnostic schema/bridge only, explicitly provisional |
| request missing WRITE-only operands | ASKED conditional | precise Lane-4 handoff; no Lane-3 ORCA2 build |
| implement landfast/aEVP/Prather or five-category SI3 | UNASKED / other lane | not done |
| alter the slab seed or implement prognostic `uu_b/vv_b` | GYRE-owned | not done |
| modify shipped NEMO, delete retained evidence, use GPU/`mpirun`, or push | forbidden | not done |
