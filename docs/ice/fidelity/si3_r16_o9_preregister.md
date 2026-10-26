# SI3 Ladder Rung O9 Preregistration — Round 16 (`si3_r16`): ZDF, DH and TEMP, jpl = 5, 10 + 5 layers

- Date: 2026-10-25. Lane `si3`, round 16, rung O9 (`docs/ice/fidelity/si3_ladder_design.md:229`).
- Model parent: `abd7262c11b4` (D5, O1, O2, O3, O5, O7, O8 landed).
- Status: committed BEFORE any O9 measurement; sections 2 (P), 3 (P5), 4 and 5 revised 2026-10-25 after the codex HOLD, still before any measurement. No NEMO array has been read for O9. No acquisition tool is built (DECISION 173: O8-O16 come from ONE 100-step production `ORCA2_SAS_ICE` master record; this file registers only O9's projection of it).
- Reference build cited: `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/`. `icethd_zdf_bl99.f90`, `icethd_dh.f90`, `icethd.f90`, `icethd_zdf.f90` are byte-identical (`md5sum`) across `ORCA2_SAS_ICE_{O1,O2,O5,O7}_{REF,REC}`. Flags (`ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm:27`): `-fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize`.

## 1. What executes (production deck, resolved from `namelist_ice_ref` + empty `namelist_ice_cfg` groups)

`jpl = 5`, `nlay_i = 10`, `nlay_s = 5`, `ln_icedH = T`, `ln_zdf_BL99 = T`, `ln_cndi_P07 = T`,
`rn_cnd_s = 0.31`, `rn_kappa_i = 1`, `rn_kappa_s = 10`, `nn_qtrice = 0`, `nn_snwfra = 2`,
`ln_cndflx = F` (so `ice_thd_zdf_BL99( np_cnd_OFF )`), `ln_virtual_itd = F`, `ln_zdf_chkcvg = F`,
`rn_snwblow = 0.66`, `rn_sinew = 0.75`, `nn_icesal = 4`, `ln_pnd = T` (ponds run after the category loop,
`icethd.f90:222-223`, outside O9).

Per category `jl`, with `a_i > epsi10` gathered into `nptidx` (`icethd.f90:176-185`):
`ice_thd_1d2d( jl, 1 )` (`:189`), the `dh_*` zeroing (`:192-195`), `ice_thd_zdf` (`:197`), `ice_thd_dh`
(`:199`), `ice_thd_temp` (`:201`). O9 stops there: `ice_thd_sal` (`:203`) is O10.

## 2. Arrays and call points (label: GIVEN-NEMO-ENTRY only)

Binary64, 1-D gathers `(1:npti)` and `(1:npti, 1:nlay_i|nlay_s)` as declared in `ice1d.f90`, bounds and
`nptidx(1:npti)` in each frame header. Observer copies are statement-level in `ice_thd`, inside `IF( npti > 0 )`
but outside every inner loop (checklist item 1).

- F0 run scalars: `kt`, `nit000`, `jpni`, `jpl`, `nlay_i`, `nlay_s`, `rDt_ice`, `rcnd_s`, and the module values of
  `rn_kappa_i rn_kappa_s rn_snwblow rn_sinew nn_icesal`, compared to the resolved `output.namelist.ice`.
- F1, after `icethd.f90:195`, before `:197` (stage ENTRY): `a_i at_i h_i h_s t_su t_i t_s e_i e_s sz_i s_i t_bo
  sss sst qns_ice qsr_ice dqns_ice qtr_ice_top qsb_ice_bot fhld frq_m sprecip qprec_ice evap_ice hfx_err_dif
  hfx_dif` and the 25 flux accumulators of `SI3DHFluxes` (all `_1d`).
- F2, after `:197` (ZDF exit): `t_su t_i t_s e_i e_s qns_ice qcn_ice_top qcn_ice_bot qtr_ice_bot hfx_err_dif
  hfx_dif cnd_ice t1_ice t_si`.
- F3, after `:199` (DH exit): `h_i h_s t_su a_i e_i e_s t_s sz_i s_i qml_ice`, the eight `dh_*`, the 25 fluxes.
- F4, after `:201` (TEMP exit): `t_i`.
- P (Picard), REVISED 2026-10-25 (codex HOLD item 3; the first design put an observer inside NEMO's `ji`
  loop, against checklist item 1): one statement-level observer in `ice_thd_zdf_BL99` after the `ji` loop's
  `END DO` of each `DO WHILE` pass (`icethd_zdf_bl99.f90:603`), inside the `np_cnd_OFF` branch, before the pass's closing `END DO`. It copies
  whole arrays only: `iconv` and `l_T_converged(1:npti)`, `t_su_1d(1:npti)`, `t_i_1d(1:npti,1:nlay_i)`,
  `t_s_1d(1:npti,1:nlay_s)`, `qns_ice_1d(1:npti)`. No copy sits inside any `ji`/`jk`/`jm` loop.
  What it CAN discriminate: the state after every pass, so a first difference is localised to (pass, cell);
  each cell's pass count, read as the first pass whose `l_T_converged` is true (model `n_picard`); and a
  GIVEN-NEMO-ENTRY replay of one pass from the recorded state of the pass before.
  What it CANNOT discriminate: anything inside one pass. NEMO's matrix rows `ztrid zindterm zindtbis zdiagbis`
  are per-cell scratch overwritten for every `ji`, so after the loop they hold only the last cell; a first
  difference inside a pass cannot be split between matrix assembly, the Thomas sweeps and the clamps. If that
  is needed it is a follow-up record design, brought back as DECISION_NEEDED, not an in-loop observer.

Steps: every `kt` of the master record for F0-F4; P only at the steps of section 6.

## 3. Frozen predictions (CPU fp64, production `jax.jit`, `transcendentals = "libm"`)

- P1: F2 equals `si3_ice_thd_zdf_bl99(F1, deck)` (`packages/ice/legoesm/ice/bitz_lipscomb.py`), 0 differing bits
  on every gathered cell, every category, every step.
- P2: F3 equals `si3_ice_thd_dh(F1 + F2, deck)`, 0 differing bits.
- P3: F4 equals `si3_ice_thd_temp(F3)`, 0 differing bits.
- P4: the composed `si3_ice_thd_category_stages(F1, deck)` reproduces F2-F4 (composition, not replay).
- P5: per-cell Picard count (`n_picard`) equals the first pass at which P's `l_T_converged` turns true
  (NEMO's `iconv` is one counter for all cells; it bounds `n_picard` and equals its maximum).
- Falsifier for each: any differing bit outside a preregistered platform floor. The only stated floor is
  DECISION 158's binary64 subnormal class: a cell whose F1-F4 values, or any compare operand, is subnormal is
  reported separately and never called matched. NaN-only operand-order classes (section 5) are counted and
  reported, not scored.
- The first non-bit statement, if any, is named with its `ppsrc` line before any change lands.

## 4. Controls the gate must carry

Reader: swapped `t_i`/`t_s`, one-bit payload, truncated record, wrong category index, wrong time level, wrong
`nptidx` order. Scorer plants that must turn the score red (each reported vacuous if the record lacks the
operand class): `rn_cnd_s` +1 ULP; `rn_sinew` +1 ULP; native transcendentals (fires iff an `EXP`/`**` argument
is non-trivial); reversed `minsd`/`maxsd` operand roles (fires iff a signed-zero tie is recorded); `sst := t_bo -
rt0` (fires iff a flooding cell is recorded); materialized-zero removal (fires iff a `-0` accumulates into a
zeroed `dh_*`); constant-divisor reciprocal (fires iff `h_i < 0.1` or `h_s < 1e-3`). Placement: the P observer is
checked against the real ppsrc to sit after the `ji` loop of `icethd_zdf_bl99.f90:603`, with a plant moving it
inside the loop. Admission per the ACQUISITION TOOL CHECKLIST.

## 5. Object-code facts the predictions rest on (REVISED 2026-10-25, codex HOLD item 4)

The first version rested on an approximate scratch classifier. It is replaced by a committed, exhaustive audit
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o9_minmax_audit.py`, report `nemo_si3_o9_minmax_audit.json`,
pinned site by site by `tests/ocean/fidelity/test_nemo_si3_o9_minmax_audit.py`). It recompiles each ppsrc with the
record flags plus `-g`, refuses unless the instruction stream equals the record object's, lists every
`minsd`/`maxsd`/`sqrtsd` in the O9 routines (`ice_thd_zdf`, `ice_thd_zdf_BL99`, `ice_thd_dh`, `snw_ent`,
`ice_thd_temp`, `ice_var_enthalpy`, `ice_var_vremap`, `ice_var_snwblow_1d`) through the line table to its
statement, and resolves each operand's kind (zero, constant, variable) by reaching definitions over the
function's control-flow graph. Both census directions are checked.

What it proves (CONFIRMED): 64 statements hold all such instructions (25 + 35 + 1 + 3 by object; none in
`ice_thd_zdf`); on every one of the 33 deck-path statements with a zero operand the zero is the SOURCE, so it is
what a signed-zero tie or a NaN returns; the zero-free sites compare against a nonzero constant or are
variable/variable; no `SIGN` appears in these routines; the only `sqrtsd` is `icethd.f90:285`. Four source
`MIN`/`MAX` have no instruction at their own line: `icethd_dh.f90:222` and `:300` are emitted as negated
`minsd` that the line table assigns to `:228` and `:295`; `icethd_zdf_bl99.f90:578` is folded to `ABS(.)`
(zero first operand); `icethd_dh.f90:526` (not executed, `nn_icesal == 2`) has none, PLAUSIBLY a reused
subexpression.

What it does NOT prove (narrowed): for variable/variable sites (e.g. `icethd_dh.f90:194,200,490`,
`icethd_zdf_bl99.f90:577-590`, `icevar.f90:955`) it does not name which Fortran operand each register holds.
The earlier statements that those sites "keep `t_i`" or "keep the thickness" are hand readings, PLAUSIBLE, and
they matter only for an opposite-sign zero tie or a NaN. `EXP` (radiation) and `**` (`ice_var_snwblow`, `pow`)
remain glibc calls.

## 6. Pending decisions (section 7 of the receipt)

- Picard/tridiagonal frames (P): pick every pass at `kt = 1, 2, 100`, every gathered cell, all five categories.
- Steps scored for F0-F4: pick all 100 steps of the master record.

## 7. CHOICES (ASKED/UNASKED)

- O9 projection taken from the shared master record: ASKED (DECISION 173).
- Frames F1-F4 at the four statement boundaries above, plus F0 scalars: UNASKED, design `:229` "ZDF/DH/TEMP
  entry/exits"; offered for revert.
- Picard frame steps and the 100-step scoring window: UNASKED -> DECISION_NEEDED.
- `ice_thd_1d2d( jl, 1 )` (gather and J/m2 -> J/m3, `icethd.f90:373-588`) is upstream of F1 and owned by no rung
  in the design table: UNASKED observation, reported, not built.
