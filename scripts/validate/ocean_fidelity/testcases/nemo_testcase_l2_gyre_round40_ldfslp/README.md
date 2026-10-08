# Round-40 instrument B — the `ldf_slp` statement record

`oracle_ldfslp_kt00000001.bin` and `oracle_ldfslp_kt00000002.bin`, magic
**`NEMO_L2_LDFSL_1`**.

## Why it exists

Given NEMO's own before state at kt=2, legoESM's isoneutral fold reaches
`1.1728064666279615e-10` where NEMO's `ah_wslp2` reaches
`3.3898494597440722e-08` — 17400 of 17400 wet faces, 289x apart, argmax at the
first interior face.  The round-38 kt=2 trazdf record carries only `ah_wslp2`,
the **product** of the slopes and the diffusivity, and none of the operands
`ldf_slp` reads: not `prd` (=`rhd` at `Nbb`), not `pn2` (=`rn2b`), not `nmln`,
not `hmlp`, and none of `uslp`/`vslp`/`wslpi`/`wslpj`.  Every hypothesis about
which statement differs is unfalsifiable against it.

## Placement

A `MY_SRC/ldfslp.F90` copy of the shipped `src/OCE/LDF/ldfslp.F90` on the
`GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2` card, with one WRITE block immediately after
`ldf_slp`'s own `CALL lbc_lnk` (`ldfslp.F90:319`), armed on
`lwp .AND. kt <= nit000 + 1`.  Per-`kt` filename.

`kt = nit000` alone is useless: GYRE's analytic initial T and S are
horizontally uniform on wet cells, so every slope there is exactly zero and the
record can only say whether a candidate is also exactly zero.  `nit000+1` is the
first step whose before state carries structure.

The per-`jk` intermediates are scalars inside NEMO's own `DO_2D` bodies and the
**descending** `jk` loop destroys them, so each is copied into a zeroed 3-D
buffer inside the loop.  No NEMO arithmetic is touched, the loop is not
restructured, and the delta removes **0** of NEMO's own lines.  A buffer slot
the routine never assigns stays a deterministic `0._wp` — never uninitialised
memory, which is what round 39's raw-identity refusal was really about.

## Frame spec

Stream, little-endian, `real(8)` payloads.

```
char*16  'NEMO_L2_LDFSL_1'
int*4 x14  version, kt, Kbb, Kmm, jpi, jpj, jpk, jpkm1, nlb10,
           ntsi, ntei, ntsj, ntej, STORAGE_SIZE(1._wp)
repeat until EOF:
  char*16  name (blank-padded)
  int*4 x4 rank, n1, n2, n3   ! 0 -> 1 value; 1 -> n1; 2 -> n1*n2; 3 -> n1*n2*n3
  real*8   payload
```

Integer NEMO fields (`nmln`, `mikt`, `miku`, `mikv`, `mbkt`) and the logicals
(`ln_isfcav`, `ln_traldf_iso`, `l_ldfslp`) are written as `real(8)`; 1 is true.

Locals declared on an `A2D` window (`nmln`, `hmlp`, `zhmlpt`, `r1_hmlu`,
`r1_hmlv`, `r1_hmlw`) are staged through a zeroed `(jpi,jpj,jpk)` scratch over
**their own** defined loop bounds, so no undefined slot is ever written.

## Arrays, in write order

| # | name | rank | extents |
|---|---|---|---|
| 1 | `rn_slpmax` | 0 | 1, 1, 1 |
| 2 | `grav` | 0 | 1, 1, 1 |
| 3 | `zeps` | 0 | 1, 1, 1 |
| 4 | `z1_slpmax` | 0 | 1, 1, 1 |
| 5 | `ln_isfcav` | 0 | 1, 1, 1 |
| 6 | `ln_traldf_iso` | 0 | 1, 1, 1 |
| 7 | `l_ldfslp` | 0 | 1, 1, 1 |
| 8 | `gdept_1d` | 1 | jpk, 1, 1 |
| 9 | `gdepw_1d` | 1 | jpk, 1, 1 |
| 10 | `e3t_1d` | 1 | jpk, 1, 1 |
| 11 | `e3w_1d` | 1 | jpk, 1, 1 |
| 12 | `nmln` | 2 | jpi, jpj, 1 |
| 13 | `hmlp` | 2 | jpi, jpj, 1 |
| 14 | `zhmlpt` | 2 | jpi, jpj, 1 |
| 15 | `r1_hmlu` | 2 | jpi, jpj, 1 |
| 16 | `r1_hmlv` | 2 | jpi, jpj, 1 |
| 17 | `r1_hmlw` | 2 | jpi, jpj, 1 |
| 18 | `ssmask` | 2 | jpi, jpj, 1 |
| 19 | `mikt` | 2 | jpi, jpj, 1 |
| 20 | `miku` | 2 | jpi, jpj, 1 |
| 21 | `mikv` | 2 | jpi, jpj, 1 |
| 22 | `mbkt` | 2 | jpi, jpj, 1 |
| 23 | `r3t_Kmm` | 2 | jpi, jpj, 1 |
| 24 | `r3u_Kmm` | 2 | jpi, jpj, 1 |
| 25 | `r3v_Kmm` | 2 | jpi, jpj, 1 |
| 26 | `e1t` | 2 | jpi, jpj, 1 |
| 27 | `e2t` | 2 | jpi, jpj, 1 |
| 28 | `e1u` | 2 | jpi, jpj, 1 |
| 29 | `e2v` | 2 | jpi, jpj, 1 |
| 30 | `r1_e1u` | 2 | jpi, jpj, 1 |
| 31 | `r1_e2v` | 2 | jpi, jpj, 1 |
| 32 | `prd` | 3 | jpi, jpj, jpk |
| 33 | `pn2` | 3 | jpi, jpj, jpk |
| 34 | `tmask` | 3 | jpi, jpj, jpk |
| 35 | `umask` | 3 | jpi, jpj, jpk |
| 36 | `vmask` | 3 | jpi, jpj, jpk |
| 37 | `wmask` | 3 | jpi, jpj, jpk |
| 38 | `e3u_0` | 3 | jpi, jpj, jpk |
| 39 | `e3v_0` | 3 | jpi, jpj, jpk |
| 40 | `e3u_Kmm` | 3 | jpi, jpj, jpk |
| 41 | `e3v_Kmm` | 3 | jpi, jpj, jpk |
| 42 | `e3w_Kmm` | 3 | jpi, jpj, jpk |
| 43 | `gdept_Kmm` | 3 | jpi, jpj, jpk |
| 44 | `gdepw_Kmm` | 3 | jpi, jpj, jpk |
| 45 | `zdzr` | 3 | jpi, jpj, jpk |
| 46 | `zgru_iik` | 3 | jpi, jpj, jpk |
| 47 | `zgru_iikm1` | 3 | jpi, jpj, jpk |
| 48 | `zgrv_iik` | 3 | jpi, jpj, jpk |
| 49 | `zgrv_iikm1` | 3 | jpi, jpj, jpk |
| 50 | `zau` | 3 | jpi, jpj, jpk |
| 51 | `zav` | 3 | jpi, jpj, jpk |
| 52 | `zbu` | 3 | jpi, jpj, jpk |
| 53 | `zbv` | 3 | jpi, jpj, jpk |
| 54 | `zfi` | 3 | jpi, jpj, jpk |
| 55 | `zfj` | 3 | jpi, jpj, jpk |
| 56 | `zmli` | 3 | jpi, jpj, jpk |
| 57 | `zmlj` | 3 | jpi, jpj, jpk |
| 58 | `zdepu` | 3 | jpi, jpj, jpk |
| 59 | `zdepv` | 3 | jpi, jpj, jpk |
| 60 | `zwz_uv` | 3 | jpi, jpj, jpk |
| 61 | `zww_uv` | 3 | jpi, jpj, jpk |
| 62 | `zuslp_hml` | 3 | jpi, jpj, jpk |
| 63 | `zvslp_hml` | 3 | jpi, jpj, jpk |
| 64 | `zbw` | 3 | jpi, jpj, jpk |
| 65 | `zci` | 3 | jpi, jpj, jpk |
| 66 | `zcj` | 3 | jpi, jpj, jpk |
| 67 | `zai` | 3 | jpi, jpj, jpk |
| 68 | `zaj` | 3 | jpi, jpj, jpk |
| 69 | `zbi` | 3 | jpi, jpj, jpk |
| 70 | `zbj` | 3 | jpi, jpj, jpk |
| 71 | `zfk` | 3 | jpi, jpj, jpk |
| 72 | `zmlk` | 3 | jpi, jpj, jpk |
| 73 | `zck` | 3 | jpi, jpj, jpk |
| 74 | `zwz_w` | 3 | jpi, jpj, jpk |
| 75 | `zww_w` | 3 | jpi, jpj, jpk |
| 76 | `zwslpi_hml` | 3 | jpi, jpj, jpk |
| 77 | `zwslpj_hml` | 3 | jpi, jpj, jpk |
| 78 | `uslp` | 3 | jpi, jpj, jpk |
| 79 | `vslp` | 3 | jpi, jpj, jpk |
| 80 | `wslpi` | 3 | jpi, jpj, jpk |
| 81 | `wslpj` | 3 | jpi, jpj, jpk |
## Not carried, and why

`ahtu`/`ahtv` live in `ldftra`, which `ldfslp` does not `USE` and cannot
without a circular dependency (`ldftra` uses `ldfslp`).  They are constants on
this card (`nn_aht_ijk_t = 0`, `rn_Ud = 2.0e-2`, `ocean.output:663-664`), and
`ah_wslp2` — their product with the slopes — is already carried at both kt by
the round-38 trazdf record, which is where the closure check belongs.

Pre-`lbc_lnk` slopes are not carried: `lbc_lnk` on this closed box fills the
halo only, and every row scored against this record is interior.

## One command

```
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round40_ldfslp/run.sh
```
