# Preregistration — NEMO-testcases grid ladder G8, round 1 (gridladder_r45)

**Frozen:** 2026-10-17, before any G8 card change and before any G8 record
exists.  Nothing is measured against NEMO in this round; every statement
below is read off the compiled source (READ) or is a prediction (PREDICTED).

Builds cited: `G5` = `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo` (the
record binary G8 reuses; its writer-free twin is
`ORCA2G5_OMIP_L4_R31REF_FIX1`); Decision-122 pointer `ORCA2` =
`ORCA2_OMIP_L4/BLD/ppsrc/nemo`.  `ldfdyn.f90`, `dynldf.f90`,
`dynldf_lev.f90`, `iom.f90` and `lbcnfd.f90` are byte-identical in all three
builds (`cmp`), and the record build's `stp2d.f90` / `stprk3_stg.f90` call
lines equal ORCA2's, so every line number below holds in both.

## 1. Survey (search before build)

| item | value | where |
|---|---|---|
| ORCA2_OMIP_L4 `&namdyn_ldf` | three lines: `ln_dynldf_lap = .true.`, `ln_dynldf_lev = .true.`, `nn_ahm_ijk_t = -30` | `ORCA2_OMIP_L4/EXP00/namelist_cfg:371-373` |
| resolved from `namelist_ref` | `ln_dynldf_OFF=F`, `nn_dynldf_typ=0` (div-rot), `ln_dynldf_blp=F`, `ln_dynldf_hor=F`, `ln_dynldf_iso=F`; `rn_Uv=0.1`, `rn_Lv=10.e+3`, `rn_ahm_b=0.0` read but inert under -30 | `ORCA2_OMIP_L4/EXP00/namelist_ref:1107-1134` |
| namelist read, operator choice | `NAMELIST/namdyn_ldf/` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfdyn.f90:177-179` (SUBROUTINE `ldf_dyn_init` :149) |
| lap + lev -> `nldf_dyn = np_lap` (no rotation) | | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfdyn.f90:248-250` |
| `nn_ahm_ijk_t = -30`: `ahmt_3d`/`ahmf_3d` READ from `eddy_viscosity_3D.nc` with `'T'`/`'F'` lateral exchange; `zah0` not referenced | | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfdyn.f90:348-353` |
| laplacian: levels 1..jpkm1 multiplied by `tmask`/`fmask` (fmask carries `rn_shlat=2` and `strait_shlat`) | | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfdyn.f90:387-390` |
| no per-step coefficient (`l_ldfdyn_time = F`) | `ldf_dyn` not called | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:195` |
| dispatcher `np_lap` -> `dynldf_lev_lap` | | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynldf.f90:83-85` (SUBROUTINE `dyn_ldf` :60) |
| the operator: `zwf` (ahmf*e3f*curl), `zwt` (ahmt*div/e3t), U/V update | | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynldf_lev.f90:121-140` (SUBROUTINE `dynldf_lev_lap` :64) |
| call 1: 2-D RHS, `dyn_ldf(kt, Kbb, Kbb, ...)` after `dyn_hpg`, before `dyn_vor` | | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:144` (SUBROUTINE `stp_2D` :63) |
| call 2: stage 3, `dyn_ldf(kstp, Kbb, Kmm, ...)` | | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:493` (SUBROUTINE `stp_RK3_stg` :78) |
| G5 deck and its record | `ln_dynldf_OFF = .true.` (TSUNAMI) — **no lateral momentum diffusion** | `nemo_testcase_g5_orca2ice/namelist_cfg:293` |
| G5 card | `lateral_viscosity.A_h = 0.0` (OFF) on top of ORCA2's identity, which already resolves `nemo_div_curl` + `nemo_e3` + `nemo_ahm_3d_file`; the G builder does not carry the coefficient | `build_orca2g1_flat_card` |
| existing transcription | the ORCA2 card reads the file (`build_orca2_ldf_dyn_coefficients`, round-9 gate pins the file SHA-256 `fc142ce0…99cb`); the shared WS-RK3 program runs `dyn_ldf` at stage 1 and stage 3 and skips it at stage 2; ORCA2 lane OMT-3 (round 221) = OMT-2 + `A_h` 0 -> 1e5 with the same four selectors | `nemo_testcase_recipe.py`, `ocean_pe_latlon_cgrid.py`, `nemo_testcase_l4_orca2_round221_omt3_ladder_gate.py` |

So G5 does NOT run ORCA2's lateral momentum diffusion; G8 is real work, and
all of it is selection: no new physics code is expected.

**Premise correction.** The round brief says G8 keeps "G5's flux-form
momentum advection".  G5 runs momentum advection OFF (`ln_dynadv_OFF=.true.`),
as round 41 already recorded.  G8 keeps G5's setting (OFF), which is what
"G5 + lateral momentum diffusion, nothing else" requires.  Drag stays OFF.

**ORCA2 card (report only, the ORCA2 lane imports).**  Its lateral-momentum
selection matches the namelist: harmonic div-curl level operator
(`nemo_div_curl`, `nemo_e3`), coefficient read from the file
(`nemo_ahm_3d_file`, `ldfdyn.f90:348-353`), free-slip side condition with the
boundary condition carried inside `ahmf*fmask`, `A_h=1e5` as an inert
selector.  It still lists `spatial_lateral_viscosity` as UNMEASURED; no
lateral-momentum setting is GYRE-inherited.

## 2. Claim and exact target

G8 = the certified G5 deck plus ORCA2's `&namdyn_ldf` selection and nothing
else.  Deck
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g8_orca2ldf/namelist_cfg`;
`namelist_ice_cfg` is G5's, byte-identical.

| key | G5 | G8 (resolved = ORCA2's) |
|---|---|---|
| `namdyn_ldf.ln_dynldf_OFF` | `.true.` | `.false.` (ORCA2: unstated -> ref `.false.`) |
| `namdyn_ldf.ln_dynldf_lap` | `.false.` | `.true.` (ORCA2 `:371`) |
| `namdyn_ldf.ln_dynldf_lev` | unstated -> `.false.` | `.true.` (ORCA2 `:372`) |
| `namdyn_ldf.nn_ahm_ijk_t` | unstated -> `0` (inert, OFF) | `-30` (ORCA2 `:373`) |

`ln_dynldf_blp = .false.` stays stated as in G5 (= ref).  Nothing else moves
(a committed test pins the assignment diff).  The run directory also gets
ORCA2's shipped `eddy_viscosity_3D.nc` (SHA-256 `fc142ce0…99cb`, the file the
ORCA2 lane's gate pins), which NEMO opens by that fixed name.

## 3. Card change (to be built after this commit)

- New case `ORCA2G8-ldf` = G5's rung tables, with `lateral_viscosity` left at
  the ORCA2 identity's value (`A_h=1e5`, an inert positive selector under the
  read source, exactly OMT-3's edge) instead of G5's zero, and the
  coefficient `ahmt`/`ahmf` read by the ORCA2 card's own
  `build_orca2_ldf_dyn_coefficients` on the G builder's `tmask`/`fmask` and
  carried on `z_coord`.  Nothing else changes.
- The deck reader picks the `&namdyn_ldf` switch set by `ln_dynldf_lap`, and
  refuses any stated `nn_dynldf_typ`, `ln_dynldf_hor`, `ln_dynldf_iso`,
  `rn_Uv`, `rn_Lv`, `rn_ahm_b` or Smagorinsky line (ORCA2 states none; the
  card transcribes `namelist_ref`'s).
- Every other card is unchanged: only the G8 case selects the coefficient and
  the non-zero `A_h`.  Card is fail-closed: `ORCA2G8_UNMEASURED =
  (nemo_post_si3_surface_records, g8_record_not_admitted,
  lateral_momentum_diffusion_multilevel_step)`.

## 4. Predictions (frozen; scored when the record exists)

- **P1 (READ+PREDICTED).** NEMO's G8 log prints `ln_dynldf_lap = T`,
  `ln_dynldf_lev = T`, `nn_ahm_ijk_t = -30`, `iso-level laplacian operator`
  and `eddy viscosity = F(i,j,k) read in eddy_viscosity_3D.nc file`
  (`ldfdyn.f90:196`, `:200`, `:205`, `:284`, `:349`); falsified if any is absent.
- **P2 (PREDICTED).** The record is admitted with passivity PASS (writer set
  unchanged from G5).
- **P3 (PREDICTED).** kt 1 stage 1 is bit-identical to G5's behaviour for
  the LDF term: from rest `uu(Kbb) = vv(Kbb) = 0`, so the stage-1 call
  (`stp2d.f90:144`) adds an exact zero.  The first LDF-carrying statement is
  stage 3 of kt 1 (`stprk3_stg.f90:493`, on the stage's `Kbb`=rest velocity
  — so ALSO zero at kt 1) — hence the first non-zero LDF term is kt 2,
  stage 1 (`Kbb` = kt-1's after velocity).  Falsified if any field differs
  from the card at kt 1 when G5's kt 1 is bit-identical.
- **P4 (PREDICTED).** G8 is NOT bit-identical at kt 2.  The first non-bit
  statement is one of, in this order of likelihood:
  (a) the north-fold V row: NEMO's V update on the last owned row reads
  `zwt(ji,jj+1)` on the fold halo row (`dynldf_lev.f90:140`), built from
  folded velocities and the folded coefficient; the shared operator pads the
  divergence term with zero at the north edge and zeroes the polar V row;
  (b) the stage-1 `Kmm` slot: stage 1 passes `Kbb` as `Kmm`, so the U/V
  divisors read `r3u/r3v(Kbb)` (`stp2d.f90:144`, `dynldf_lev.f90:135`,
  `:139`), not the stage's mid level;
  (c) the e3 association of `zwf`/`zwt` (`e3f*(1+r3f*fe3mask)`,
  `e3t*(1+r3t(Kbb)*tmask)`) on partial cells.
  Falsified if kt 1..N are bit-identical on every field under both labels.
- **P5 (READ).** Stability: max over wet surface T cells of
  `rn_Dt * ahmt * (1/e1t^2 + 1/e2t^2)` = 0.390 (cell (j,i)=(1,49) 0-based,
  file coefficient 2000..40000 m2/s); below the explicit 0.5 bound, so the
  term itself is not expected to move NEMO's stop boundary (G5: stop while
  beginning kt 11).

## 5. Acquisition (operator runs it)

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g8_orca2ldf/run.sh`,
the G7 pattern: no build (operator choice is a namelist switch; `ldfdyn`,
`dynldf`, `dynldf_lev` are compiled in every build), G5's two round-31
binaries pinned by SHA-256, G4's inputs plus ORCA2's `eddy_viscosity_3D.nc`
(pinned), smoke 2 steps -> writer-free 100-step boundary -> three record
twins -> passivity + frame admission with `g5_records.py`.  Evidence root
`gridladder_rounds/round45/oracle_g8_r45`.

## 6. Choices

ASKED (the brief): G8's definition; ORCA2's three `&namdyn_ldf` lines; G5's
binaries and record set.
UNASKED, offered for revert: keep `ln_dynldf_blp = .false.` stated (G5's
line, equal to ref); the reader refusing unstated-in-ORCA2 `&namdyn_ldf`
lines; `A_h = 1e5` as the inert selector value (the ORCA2 identity's and
OMT-3's).
