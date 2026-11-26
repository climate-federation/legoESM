# SI3 Ladder Rung O5 Preregistration — Round 10 (`si3_r10`): five-category H79 strength + aEVP on the ORCA2 tripole

- Date: 2026-10-25. Lane `si3`, round 10, rung O5 (`docs/ice/fidelity/si3_ladder_design.md:225`).
- Model commit preregistered against: `e17ba3082c` (parent `1bb17df9ddcc35037c72f81e9dba1de853a46798`).
- Status: committed BEFORE any O5 measurement. No NEMO array has been read for O5. The acquisition tool is NOT built this round.
- Reference build cited: `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/` (the unpatched O2 reference build). Its `icedyn_rhg_evp.f90`, `icedyn_rdgrft.f90`, `icedyn.f90` and `icevar.f90` are byte-identical (`cmp`) to `ORCA2_SAS_ICE_O1_REF`, `_O1_REC`, `_O2_REC` and to lane 3's `tests/ICE_ADV2D_RHG_OMIP_L3/BLD/ppsrc/nemo/` (`diff`, empty).

## 1. Deck

The design row's diagnostic overlay on `ORCA2_SAS_ICE`, nothing else
(`docs/ice/fidelity/si3_ladder_design.md:225`): `ln_dynALL=T`,
`ln_dynRHGADV=F`, `ln_ridging=F`, `ln_rafting=F`, `ln_icethd=F`. Every other
key is the O2 reference deck (`nn_fsbc = 1`, `rn_Dt = 10800`,
`nn_hls = 2`, `ln_str_H79 = T`, `rn_pstar = 2.0e+04`, `rn_crhg = 20.0`,
`ln_str_smooth = T`, `ln_rhg_EVP = T`, `ln_aEVP = T`, `rn_creepl = 2.0e-9`,
`rn_ecc = 2.0`, `nn_nevp = 100`, `nn_rhg_chkcvg = 0`, `rn_ishlat = 2.`,
`ln_landfast_L16 = F`, `ln_ice_embd = F`), each read from the resolved
`ocean.output`/`output.namelist.ice`, never from the deck text. One MPI rank
as O1/O2 (`mpirun -np 1`), so `jpni = 1`.

Step window: PENDING USER DECISION (see section 6). Proposed: `nn_it000 = 1`,
`nn_itend = 2`, as O1/O2.

## 2. Arrays and call points

Labels: GIVEN-NEMO-ENTRY only. Every array is the full `(jpi, jpj[, k])`
binary64 allocation including both halo widths.

(a) Strength, inside `ice_dyn_rhg_evp` (`icedyn_rhg_evp.f90:268`, `CALL ice_strength`):
- A1 entry, immediately before `:268`: `a_i(:,:,1:5)`, `v_i(:,:,1:5)`, `at_i`, `tmask(:,:,1)`.
- A2 after the H79 loop (`icedyn_rdgrft.f90:1085`): `at_i` (recomputed at `icedyn_rdgrft.f90:973`), `strength` (raw).
- A3 after `icedyn_rdgrft.f90:1114` (smoothing + `lbc_lnk('T', 1)`): `strength`.

(b) aEVP, same routine:
- B1 after `icedyn_rhg_evp.f90:281` (`zsshdyn = ice_var_sshdyn( ssh_m, snwice_mass, snwice_mass_b)`):
  `u_ice`, `v_ice`, `u_ice_b`, `v_ice_b`, `zs1`, `zs2`, `zs12` (`:263-265`), `strength`, `at_i`,
  `vt_i`, `vt_s`, `vt_ip`, `vt_il`, `utau_ice`, `vtau_ice`, `drag_io`, `u_oce`, `v_oce`,
  `ssh_m`, `snwice_mass`, `snwice_mass_b`, `zsshdyn`, `ff_t`, `tmask/umask/vmask(:,:,1)`,
  `fast_umask`, `fast_vmask`, `fast_tmask`, `icb_umask/vmask/tmask`,
  `e1t e2t e1u e2u e1v e2v e1f e2f e1e2t e1e2u e1e2v e1e2f`, `r1_e1u r1_e2v r1_e1e2t r1_e1e2u r1_e1e2v r1_e1e2f`.
- B2 `fimask` after `:238` (kt = nit000 only).
- B3 inner frames `jter in {1, 2, 3, 50, 99, 100}`: `zdelta`, `zp_delt` after `:440`; `zs1`, `zs2`, `zs12` after `:504`; `u_ice`, `v_ice` after `:647`/`:754`.
- B4 exit after `:850`: `pstress1_i`, `pstress2_i`, `pstress12_i`, `u_ice`, `v_ice`; plus `divu_i`, `shear_i`, `delta_i` (`:798-843`, recorded for O6, not scored here).
- B5 run scalars: `ncom_stp`, `nit000`, `jpni`, `jpnj`, `ln_nnogather`, `nn_npfchk`, `rDt_ice`.
- C1 no-redistribution proof: `a_i`, `v_i` at `ice_dyn_rdgrft` entry and exit (`icedyn.f90:193`).

## 3. Frozen predictions (model `e17ba3082c`, `transcendentals="libm"`, CPU fp64)

- P1: A2 and A3 replay from A1 exactly (0 ULP, every cell incl. halos) through `si3_orca2_ice_strength`, fold program `nfd_full_when_absent = True`.
- P2: A1 `at_i` equals A2 `at_i` bit-for-bit. Falsifier: any difference; then zmsk (`:203`) needs the entry `at_i` as a separate input (model change).
- P3: `zsshdyn` equals `ssh_m` bit-for-bit (levitating arm, `icevar.f90:1090`).
- P4: B4 replays from B1 + A3 exactly (0 ULP) through `si3_orca2_aevp_solver` with `n_subcycles = 100`; B3 frames replay exactly at their `jter`.
- P5: `fimask` (B2) replays exactly.
- P6: C1 exit equals entry (no O6 redistribution).
- P7: B1 `e1e2t` equals `e1t * e2t` and `r1_*` equal `1/metric` bit-for-bit, so the model's in-arm reciprocals read the same numbers. Falsifier: a `domain_cfg`-read area or a precomputed reciprocal differs; then the area fields enter as recorded arrays (already how the metrics tuple works) and the reciprocal statements need a citation walk.
- First non-bit statement, if any, is named with its `ppsrc` line before any change lands.

## 4. Controls the gate must carry

Swapped-field and one-bit payload plants on the reader; wrong fold sign per
label (`fimask zdelta zp_delt u_ice v_ice zs1 zs2 zs12 strength`); strength
exchanged before smoothing; unmasked smoothing; parity swap. Each must turn
the score red.

## 5. Not exercised by this deck (stated, not claimed)

The multi-rank partial pivot-line fold (`nfd_full_when_absent = False`): one
rank always self-folds (`lbclnk.f90:2112`). Unit-tested on synthetic fold-asymmetric
input only; PLAUSIBLE for NEMO until a multi-rank record exists.

## 6. Pending decision

Step window: `nn_itend = 2` (pick; kt 2 exercises the carried stress and a
non-first-step fold) or `nn_itend = 3`.

Reviewer notes for this decision (added after review; the pick is unchanged and still with the user):
- Argon: `nn_itend = 3` additionally feeds strength and the aEVP a state advected
  by both alternating Prather sweep orders. The order switches by step at
  `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:267` (odd step: `adv_x`
  then `adv_y`) and `:317` (even step: `adv_y` then `adv_x`); the note cited
  `:324`, a line inside the even-step branch.
- codex: 2 steps buy one stress-carry transition; 3 steps add a second recurrence check.

Exact-input addendum (after review): B1's `u_ice_b`, `v_ice_b` enter the model as
their own required inputs (`icedyn_rhg_evp.f90:574,625`), never substituted by
the entry `u_ice`, `v_ice`.
