# PREREGISTRATION — TSUNAMI lane, round 2 (the RK3 build)

Date 2026-10-08. Lane tip at start `5b9bde8a196a`. Frozen before any NEMO
record of the RK3 build exists and before the B4 seam probe is run.
Scored when the operator runs
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tsunami/run_rk3.sh --run`.

## 0. The deviation (DECISION 100)

TSUNAMI is built and scored with `key_RK3` ADDED to the shipped cpp keys
(`cpp_TSUNAMI.fcm`: `key_qco key_xios key_vco_1d`). Recorded in the card as
`TSUNAMI_DEVIATIONS = (("key_RK3", False, True, "DECISION 100"),)`, the way
the TEOS-10 deviations are recorded elsewhere. `key_xios` is dropped by the
toolchain (no XIOS), as in round 1. Nothing else in the case changes; the
shipped `MY_SRC` is used as-is.

## 1. The RK3 step program the card reproduces

`nemogcm.F90:166` calls `stp_RK3`. The case's own `MY_SRC/stprk3.F90`
(diff against `src/OCE/stprk3.F90`: the `key_RK3` guard, every physics,
diagnostic, restart and AGRIF call cut) runs, per step:

| order | statement | file:line |
|---|---|---|
| 1 | surface forcing, all zero (usrdef_sbc) | `tests/TSUNAMI/MY_SRC/stprk3.F90:98` |
| 2 | external mode, once per step | `tests/TSUNAMI/MY_SRC/stprk3.F90:107` |
| 3 | stage 1, Kmm = Nbb, Kaa = Naa | `tests/TSUNAMI/MY_SRC/stprk3.F90:113` |
| 4 | stage 2 after the Nnn/Naa swap | `tests/TSUNAMI/MY_SRC/stprk3.F90:118` |
| 5 | stage 3 | `tests/TSUNAMI/MY_SRC/stprk3.F90:123` |
| 6 | swap Nbb/Naa; ssh(Naa) = 2 ssh(Nbb) - ssh(Naa) | `tests/TSUNAMI/MY_SRC/stprk3.F90:125` and `:129` |
| 7 | output of ssh, uu_b, vv_b at Nbb | `tests/TSUNAMI/MY_SRC/stprk3.F90:136` |

NOT called by the case: `zdf_phy`, `eos_rab`, `bn2`, `ldf_slp`, `ldf_*`
coefficient updates, `rst_write`, `stp_ctl`. Vertical mixing therefore stays
at the constant `rn_avm0`/`rn_avt0` set at initialisation (`ln_zdfcst`).

The external mode (`src/OCE/stp2d.F90`, not overridden): `eos` + `dyn_hpg`
at Kbb (`stp2d.F90:127-128`), `dyn_ldf` (OFF), EEN `dyn_vor` (`stp2d.F90:146`),
the after-ssh ratio guess `r3t(Kaa) = ssh(Kaa) * r1_ht_0` (`stp2d.F90:151`),
`wzv` (`stp2d.F90:155`), NO advection (`ln_dynadv_OFF`, `dynadv.F90:129`
sets `np_LIN_dyn`, which `stp2d.F90:159` does not select), the depth mean of
the 3-D RHS (`stp2d.F90:178-182`), zero wind and drag, then `dyn_spg_ts`
(`stp2d.F90:280`).

Each stage (`src/OCE/stprk3_stg.F90`, not overridden; `n_baro_upd = np_HYB`
at `stprk3_stg.F90:44`; `ln_dynadv_vec = .false.`):

- stage 1: ssh(Kaa) = 2/3 ssh(Kbb) + 1/3 ssh(N+1) and uu_b/vv_b(Kaa) =
  their N+1 values (`stprk3_stg.F90:143-145`); the N+1 thickness ratios
  (`stprk3_stg.F90:156`) and their periodic halo exchange
  (`stprk3_stg.F90:158`); r3t/r3u/r3v(Kaa) interpolated
  (`stprk3_stg.F90:166-168`).
- stage 2: ssh(Kaa) = 1/2 (ssh(Kbb) + ssh(N+1)) (`stprk3_stg.F90:206`),
  r3f at N+1/3 (`stprk3_stg.F90:213`).
- stage 3: ssh, uu_b, vv_b(Kaa) = N+1 (`stprk3_stg.F90:224-226`).
- every stage: transports for `wzv` with the barotropic correction
  (`stprk3_stg.F90:267-274`); `wzv` on transports (`stprk3_stg.F90:297`,
  the flux-form branch).
- stage 1: `dyn_adv` with `np_LIN_dyn`, i.e. nothing (`stprk3_stg.F90:315`).
- stages 2 and 3: `eos` + `dyn_hpg` (hpg_sco, overwrites Krhs under key_RK3)
  and EEN `dyn_vor` (`stprk3_stg.F90:322-327`); `dyn_adv` adds nothing
  (`stprk3_stg.F90:331-334`).
- stages 1 and 2: flux-form momentum step weighted by (1 + r3u/r3v)
  (`stprk3_stg.F90:373-378`).
- stage 3: `dyn_ldf` (OFF, `stprk3_stg.F90:400`), then `dyn_zdf` integrates
  in time (`stprk3_stg.F90:430`).
- every stage: the barotropic correction to uu_b/vv_b(Kaa)
  (`stprk3_stg.F90:440-445`).
- every stage: TRACERS ARE STEPPED. `ts(Krhs) = 0` (`stprk3_stg.F90:512`),
  `tra_adv` adds nothing (`np_NO_adv`, `traadv.F90:461`), `tra_sbc_RK3`
  adds zero forcing (`stprk3_stg.F90:521`); stages 1-2 step
  ts(Kaa) = (1 + r3t(Kbb)) ts(Kbb) / (1 + r3t(Kaa)) (`stprk3_stg.F90:552-554`);
  stage 3 `tra_ldf` (OFF) and `tra_zdf` (`stprk3_stg.F90:586`, `:598`).
- every stage: halo exchange of uu, vv, T, S at Kaa (`stprk3_stg.F90:636`).

Premise correction to the brief's "no tracers, no advection, no ldf": no
advection and no lateral diffusion is right; tracers are NOT absent. T and
S are stepped every stage by the thickness ratio alone, so they stop being
uniform wherever ssh moves, and they feed the pressure gradient at stages 2
and 3. legoESM's carrier runs flux-form tracer advection instead (gap B7).

The case's `MY_SRC/diawri.F90` (diff against `src/OCE/DIA/diawri.F90`) cuts
every 3-D and forcing field and, in the non-XIOS branch this build compiles,
writes only `sossheig`, `souubaro`, `sovvbaro` at the passed level
(`tests/TSUNAMI/MY_SRC/diawri.F90:653` to `:655`).

## 2. Frozen predictions and falsifiers

Geometry (unchanged by key_RK3: in the domain code it only selects which
r3 slots are built, `domqco.F90:124`). Carried from round 1 unchanged:
G1 coordinates bitwise, G2 every horizontal scale factor exactly 10000.0,
G3 ff_t = ff_f = f0 = 9.078896742484248e-05 bitwise, G4 e3*_0 = 100.0,
gdept_1d = (50, 150), gdepw_1d = (0, 100), tmask 1 then 0, G5 the kt = 1
entry ssh equals the card's initial ssh bitwise (325 non-zero, maximum 0.1
at 0-based (79, 39)), G6 `nn_e = 6`. Falsifier for each: one unequal cell.

kt = 1 ladder, all inside NEMO's own record (read off the code, PLAUSIBLE):

- **R1** step entry: e_ssh_bb = e_ssh_nn = the initial bump; e_uu_b_bb =
  e_vv_b_bb = 0; e_tn_k1_bb = 20 and e_sn_k1_bb = 30 everywhere.
- **R2** stage 1 HYB: 1_ssh_aa = 2/3 e_ssh_bb + 1/3 b_ssh_aa and
  1_uu_b_aa = b_uu_b_aa bitwise on owned cells; stage 2:
  2_ssh_aa = 1/2 (e_ssh_bb + b_ssh_aa); stage 3: 3_ssh_aa = b_ssh_aa,
  3_uu_b_aa = b_uu_b_aa. Falsifier: one unequal owned cell.
- **R3** (the B7 claim) 1_tn_k1_aa = 20 (1 + e_r3t_bb) / (1 + 1_r3t_aa)
  bitwise and differs from 20 in at least one cell. Falsifier: T stays
  exactly 20 everywhere after stage 1.
- **R4** closing extrapolation: f_ssh_aa = 2 f_ssh_bb - e_ssh_bb bitwise.
- **R5** NEMO's own output at kt = 5 (`sossheig`) equals f_ssh_bb of the
  kt = 5 record bitwise.
- **R6** the kt = 1 substep record: icycle and weights as round-1 L3
  (`ts_wgt`'s inputs, `dynspg_ts.F90:245`, are unchanged by key_RK3 because
  ln_bt_fw is already true and nn_bt_flt = 1).

B4 seam probe (legoESM only, no NEMO number): one card step is translation
equivariant across both periodic seams when run inside the card's y-wrap
scope: step(roll(state)) equals roll(step(state)) for an ssh bump placed on
the i-seam and on the j-seam. Prediction: bitwise on eta, uu_b, vv_b.
Falsifier: any unequal cell; the receipt names the first unequal field.

## 3. ORCA2 pointer

ORCA2 rung 0 runs this same RK3 program (`stp_RK3` with the full
`stprk3.F90`, the same `stprk3_stg.F90` and `dyn_spg_ts`) with nn_bt_flt = 3.
Every R row except R6 is a statement ORCA2 executes.
