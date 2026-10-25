# SI3 Ladder Rung O11 Preregistration — Round 17 (`si3_r17`): lateral melt projection of the master record

- Date: 2026-10-25. Lane `si3`, round 17, rung O11 (`docs/ice/fidelity/si3_ladder_design.md:231`).
- Model parent: `abd7262c11b4`. Model under test: `packages/ice/legoesm/ice/si3_thd_da.py` (`si3_ice_thd_da`).
- Status: committed BEFORE any O11 measurement. No NEMO array has been read for O11. No acquisition tool is built this round.
- Record: the ONE shared 100-step production `ORCA2_SAS_ICE` master record for O8-O16 (DECISION 173; `docs/ice/fidelity/si3_ladder_design.md:205-213`). O11 reads only the projection below; it does not ask for its own run.
- Reference build cited: `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/`. `icethd_da.f90` is byte-identical (`md5sum` `4a70434478af079f4c987cb8f3bd8fb4`) across `ORCA2_SAS_ICE_{O1,O2,O5,O7}_{REF,REC}`; `icethd_da.o` instruction streams are identical in `O7_REF` and `O7_REC`. Flags `ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm:27`.

## 1. What executes on the production deck

Resolved from the shipped deck (`tests/ice/unit/data/orca2_sas_ice_expref/`; `namelist_ice_cfg` sets none of these):
`ln_icedA = .true.`, `ln_icedH = .true.`, `ln_virtual_itd = .false.` (so `ice_thd_mono` does not run),
`nn_icesal = 4`, `jpl = 5`, `nlay_i = 10`, `nlay_s = 5`, `rn_beta = 1.0`, `rn_dmin = 8.`
(`namelist_ice_ref:24-27,157-158,179,183,199`); `rn_Dt = 10800.`, `nn_fsbc = 1` (`namelist_cfg:28,61`).

Call point, inside the category loop and its `IF( npti > 0 )` (`icethd.f90:176-214`):
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:210`: `IF( ln_icedA )    CALL ice_thd_da`.
Every statement of `icethd_da.f90:174-216` executes (`nn_icesal = 4` arm of `:178`). `a_i_1d` at the call is either
`+0` (set by `icethd_dh.f90:537-541` where `h_i_1d == 0`) or `> epsi10` (gather, `icethd.f90:181`); nothing between
the gather and the call writes it on this deck.

## 2. Arrays and call points (label: GIVEN-NEMO-ENTRY only)

Binary64, NEMO's own 1-D extents `(1:npti)` and `(1:npti, 1:nlay)` as declared in `ice1D`, per `kt` and per `jl`:

- D1, immediately before `icethd.f90:210`: `npti`, `nptidx(1:npti)`, `at_i_1d`, `a_i_1d`, `h_i_1d`, `h_s_1d`,
  `e_i_1d(:,1:nlay_i)` and `e_s_1d(:,1:nlay_s)` (J/m3 at this point, after `ice_thd_1d2d( jl, 1 )`),
  `s_i_1d`, `sz_i_1d(:,1:nlay_i)`, `sst_1d`, `t_bo_1d`, `sfx_lam_1d`, `hfx_thd_1d`, `wfx_lam_1d`.
- D2, immediately after `icethd.f90:210` (before `:212`): the same arrays; the routine writes only `a_i_1d`, `h_i_1d`,
  `h_s_1d`, `sfx_lam_1d`, `hfx_thd_1d`, `wfx_lam_1d`, and D2 proves the rest unchanged.
- D0 run scalars: `kt`, `jl`, `nit000`, `jpni`, `jpl`, `nlay_i`, `nlay_s`, `nn_icesal`, `rDt_ice`, `r1_Dt_ice`,
  `r1_nlay_i`, `r1_nlay_s`, and module `rn_beta`, `rn_dmin`, each compared to the run's resolved
  `output.namelist.ice` (never the deck text); refuse unless `ln_icedA = T`, `ln_virtual_itd = F`.

Steps `kt = 1..100`, every `jl` with `npti > 0`. The observer must be passive under the record build's real flags
(ACQUISITION TOOL CHECKLIST item 1): it sits in NEMO's category loop, so its census must show `ice_thd` object code
unchanged apart from the observer calls. Size (design row: 2-5 GiB) is a retained subset of the shared record.

## 3. Frozen predictions (CPU fp64, production `jax.jit`, `transcendentals="libm"`)

- P1: D2 `a_i_1d, h_i_1d, h_s_1d, sfx_lam_1d, hfx_thd_1d, wfx_lam_1d` equal `si3_ice_thd_da(D1, rn_beta, rn_dmin,
  rDt_ice, nn_icesal=4)` with 0 differing bits on every gathered cell, every `kt`, every `jl`.
- P2: every other D1 array is bit-identical at D2.
- P3: resolved `rn_beta = 0x3FF0000000000000`, `rn_dmin = 0x4020000000000000`, `rDt_ice = 10800`,
  `r1_nlay_i = 0x3FB999999999999A`, `r1_nlay_s = 0x3FC999999999999A`, `r1_Dt_ice = 1/10800` (`0x3F1845C8A0CE5129`).
- Reported per step and `jl`, claims made only for classes the record contains: cells with `a_i_1d = +0` (untouched);
  `sst - (t_bo - rt0) <= 0` (no melt); `zda = a_i` (category emptied, `h_i = h_s = +0`); NaN or subnormal operands.
- Falsifier for each: any differing bit. The first non-bit statement, if any, is named with its `ppsrc` line before
  any change lands.

## 4. Controls the gate must carry

Reader: swapped `sfx_lam_1d`/`wfx_lam_1d`, one-bit payload, truncated record, wrong `jl`, wrong time level.
Scorer plants that must turn the score red on the record: `rn_dmin` +1 ULP; `MAX` as a NaN-propagating maximum
(fires iff the record has a NaN `sst_1d`, else reported vacuous); `SUM` in reverse layer order; source rounding
removed; `a_i == 0` never true (fires iff the record has an emptied category). Admission per the ACQUISITION TOOL
CHECKLIST (reference vs record restart bytes, `STOP 0`, passivity census with a plant).

## 5. CHOICES (ASKED/UNASKED)

- Record source and window: the shared 100-step master record (DECISION 173, ASKED); every `kt` and `jl` retained (DECISION 159 (5), ASKED).
- Recording the unchanged arrays at D2 to prove P2. UNASKED; offered for revert.
- Reference build `ORCA2_SAS_ICE_O7_REF` (byte-identical alternatives). UNASKED, mechanical.
- `sst_1d`, `t_bo_1d`, `e_i_1d`, `e_s_1d` enter as recorded arrays (produced by earlier stages). UNASKED; follows the exact-input rule.
