# Round-40 instrument A — the stage-3 per-operator momentum RHS

`oracle_rkstage3_terms_kt00000001.bin`, magic **`NEMO_L2_RKTS3_1`**.

## Why it exists

GYRE's kt=2 first-over-bar is only `u` and `v`.  Round 33 localised the owner
to the stage-3 momentum RHS; round 30 bracketed that RHS as a **total** (the
pre-`dyn_ldf` frame `NEMO_L2_RKPLD_1`, 2.06e-16 on 17383 of 17400 U faces).
Stage 3 accumulates `dyn_hpg`, `dyn_vor` and `dyn_adv` into one `Krhs` slot
(`stprk3_stg.F90:324`, `:327`, `:331`), and the only per-operator record this
campaign has — `oracle_rkstage2_terms`, `NEMO_L2_RKTRM_1` — is header-locked to
`kstg == 2`.  So the stage-3 split has never been scored and no statement
inside it can be named.

## Placement

A `MY_SRC/stprk3_stg.F90` delta on the `GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2` card,
armed on `lwp .AND. kstp == nit000 .AND. kstg == 3`.  Four snapshots inside the
`CASE ( 2 , 3 )` block, then the derived stage geometry after `dyn_adv`.  The
delta removes **0** of the source card's lines and **0** of NEMO's.

`before_u` / `before_v` are a diagnostic of the **incoming `Krhs` slot only**.
Under `key_RK3` `dyn_hpg` overwrites `Krhs` (`dynhpg.F90:359-363`, `:383-387`),
so they are not an operand of anything at this stage.

`e3f_vor_Kmm` and `e3f_0vor` are carried because `key_qco` **is** defined on
this build (`cpp_GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2.fcm`): `vor_ene` divides its
potential vorticity by `e3f_0vor*(1+r3f*fe3mask)` at `dynvor.F90:490`, and the
`nn_e3f_typ` SELECT at `:493-516` is dead code.

## Frame spec

Stream, little-endian, `real(8)` payloads.

```
char*16  'NEMO_L2_RKTS3_1'
int*4 x16  version, kstp, kstg, Kbb, Kmm, Krhs, Kaa,
           jpi, jpj, jpk, jpkm1, ntsi, ntei, ntsj, ntej, STORAGE_SIZE(1._wp)
repeat until EOF:
  char*16  name (blank-padded)
  int*4 x4 rank, n1, n2, n3      ! rank 0 -> 1 value; 2 -> n1*n2; 3 -> n1*n2*n3
  real*8   payload
```

Header for this acquisition: `(1, 1, 3, 1, 2, 3, 3, 36, 26, 31, 30, …, 64)`.

## Arrays, in write order

| # | name | rank | extents |
|---|---|---|---|
| 1 | `before_u` | 3 | jpi, jpj, jpk |
| 2 | `before_v` | 3 | jpi, jpj, jpk |
| 3 | `after_hpg_u` | 3 | jpi, jpj, jpk |
| 4 | `after_hpg_v` | 3 | jpi, jpj, jpk |
| 5 | `after_vor_u` | 3 | jpi, jpj, jpk |
| 6 | `after_vor_v` | 3 | jpi, jpj, jpk |
| 7 | `after_adv_u` | 3 | jpi, jpj, jpk |
| 8 | `after_adv_v` | 3 | jpi, jpj, jpk |
| 9 | `rDt` | 0 | 1, 1, 1 |
| 10 | `r1_Dt` | 0 | 1, 1, 1 |
| 11 | `n_baro_upd` | 0 | 1, 1, 1 |
| 12 | `ln_dynadv_vec` | 0 | 1, 1, 1 |
| 13 | `ln_zad_Aimp` | 0 | 1, 1, 1 |
| 14 | `uu_Kmm` | 3 | jpi, jpj, jpk |
| 15 | `vv_Kmm` | 3 | jpi, jpj, jpk |
| 16 | `rhd` | 3 | jpi, jpj, jpk |
| 17 | `ww` | 3 | jpi, jpj, jpk |
| 18 | `r3f` | 2 | jpi, jpj, 1 |
| 19 | `r3t_Kmm` | 2 | jpi, jpj, 1 |
| 20 | `r3u_Kmm` | 2 | jpi, jpj, 1 |
| 21 | `r3v_Kmm` | 2 | jpi, jpj, 1 |
| 22 | `e3u_Kmm` | 3 | jpi, jpj, jpk |
| 23 | `e3v_Kmm` | 3 | jpi, jpj, jpk |
| 24 | `e3t_Kmm` | 3 | jpi, jpj, jpk |
| 25 | `e3w_Kmm` | 3 | jpi, jpj, jpk |
| 26 | `e3f_vor_Kmm` | 3 | jpi, jpj, jpk |
| 27 | `e3f_0vor` | 3 | jpi, jpj, jpk |
| 28 | `e3u_0` | 3 | jpi, jpj, jpk |
| 29 | `e3v_0` | 3 | jpi, jpj, jpk |
| 30 | `e3t_0` | 3 | jpi, jpj, jpk |
| 31 | `fe3mask` | 3 | jpi, jpj, jpk |
| 32 | `fmask` | 3 | jpi, jpj, jpk |
| 33 | `umask` | 3 | jpi, jpj, jpk |
| 34 | `vmask` | 3 | jpi, jpj, jpk |
| 35 | `tmask` | 3 | jpi, jpj, jpk |
| 36 | `wmask` | 3 | jpi, jpj, jpk |
| 37 | `ff_f` | 2 | jpi, jpj, 1 |
| 38 | `e1u` | 2 | jpi, jpj, 1 |
| 39 | `e2u` | 2 | jpi, jpj, 1 |
| 40 | `e1v` | 2 | jpi, jpj, 1 |
| 41 | `e2v` | 2 | jpi, jpj, 1 |
| 42 | `r1_e1u` | 2 | jpi, jpj, 1 |
| 43 | `r1_e2v` | 2 | jpi, jpj, 1 |
| 44 | `e1e2t` | 2 | jpi, jpj, 1 |
| 45 | `r1_e1e2f` | 2 | jpi, jpj, 1 |
| 46 | `r1_hf_0` | 2 | jpi, jpj, 1 |
## The operator's contribution

`dyn_hpg` = `after_hpg` (it overwrites, it does not accumulate).
`dyn_vor` = `after_vor − after_hpg`.  `dyn_adv` = `after_adv − after_vor`.
`after_adv` must equal the existing `oracle_rkstage3_preldf_kt00000001.bin`
(`NEMO_L2_RKPLD_1`) bit for bit — that is this record's own closure control.

## One command

```
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round40_stage3_terms/run.sh
```
