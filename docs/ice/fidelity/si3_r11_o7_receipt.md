# SI3 round 11 receipt — O7 model preparation: the ice-atmosphere surface step

## Own errors, first

1. My first transcription let XLA turn `zta/rtt0` into a multiply by the
   reciprocal: 93 one-ULP rows against the independent oracle. The divisor
   is now materialized (`packages/core/legoesm/thermo.py:385`).
2. My first Fortran `SUM` seed `0 + x` was folded away by XLA, so an
   open-ocean `emp_ice` kept a `-0`. The seed is now materialized
   (`packages/ice/legoesm/ice/si3_surface.py:256`).
3. Round 11 first shipped a second implementation of statements lane 3
   already had, dropped `blk_ice_1`'s `zsipt`, built "production" by editing
   the O2 overlay's `ln_icedyn`, and checked 14 outputs (codex HOLD). All
   four are fixed below.
4. Commit `7a02a286da` holds only two file renames (its `git add` aborted);
   `576d922d72` is the content its message describes.

## Outcome

One NEMO-literal implementation per statement family, called by both lane 3's
C1D gate and O7. On the C1D year it moves 16,494 lane-3 values from
non-bit-identical to bit-identical with NEMO's record and changes none that
already matched. The D5 year hash is unchanged. No NEMO array was read for
O7; the acquisition stays preregistered and unbuilt.

## Build cited

`ORCA2_SAS_ICE_O2_REF`, flags (`ORCA2_SAS_ICE_O2_REF/BLD/arch_nemo.fcm:27`):
`%FCFLAGS             -fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize`.
Its `sbcblk.f90`, `sbc_phy.f90`, `icealb.f90`, `icesbc.f90`, `icevar.f90`
are byte-identical (`cmp`) to `_O1_REF`, `_O1_REC` and `_O2_REC`.

## Codex item 1 — `zsipt`

`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1086`:
`zsipt(:,:) = theta_exner( ptsui(:,:), pslp(:,:) )`. Every reader of this
local in `blk_ice_1` is a non-constant arm:
`:1102`: `CALL turb_ice_an05( rn_zqt, rn_zu, zsipt, ptair, ztmp, pqair, wndm_ice,       &`,
`:1107`: `CALL turb_ice_lu12( rn_zqt, rn_zu, zsipt, ptair, ztmp, pqair, wndm_ice, fr_i, &`,
`:1112`: `CALL turb_ice_lg15( rn_zqt, rn_zu, zsipt, ptair, ztmp, pqair, wndm_ice, fr_i, &`.
(`blk_ice_2`'s `zsipt` is its own scalar local, `:1225`.) Transcribed
(`packages/core/legoesm/core/bulk_flux.py:1646`). The test moves
`tm_su` alone and requires `zsipt` to change and every other output to stay
bit-identical (`tests/ice/unit/test_si3_surface.py:436`); a planted consumer
(`theta_zu_i = zsipt`) is caught (`tests/ice/unit/test_si3_surface.py:441`).
Not transcribed, stated: `zmsk00`, read only by `iom_put` diagnostics.

## Codex item 2 — production configuration from the production deck

Built from the shipped files, merged as NEMO reads them (each `_cfg` key
replaces its `_ref` key; no namelist group repeats in any of the four):
`packages/ice/legoesm/ice/si3_surface.py:146`. Fixtures are byte copies,
compared to the NEMO tree when it is mounted, of `cfgs/SHARED/namelist_ref`,
`cfgs/SHARED/namelist_ice_ref`, `cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg`,
`cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg` (`ORCA2_SAS_ICE` has no `BLD`;
"as built" does not exist). Result: `rn_Cd_ia = rn_Ce_ia = rn_Ch_ia = 1.4e-3`,
`ln_icedyn = T`, ponds on. It differs from the O2 run's resolved config in
exactly one field, `ln_icedyn` (`tests/ice/unit/test_si3_surface.py:351`).
The reader is the round-314 audit's, moved to one home
(`packages/core/legoesm/core/nemo_namelist.py:80`); that audit's tests still pass.

## Codex item 3 — one implementation per statement family

| family | single implementation | C1D caller | O7 caller |
|---|---|---|---|
| `e_sat`/`q_sat`/`dq_sat`, `theta_exner` | `packages/core/legoesm/thermo.py:410` | yes | yes |
| `blk_ice_1` constant arm | `packages/core/legoesm/core/bulk_flux.py:1636` | yes | yes |
| `blk_ice_2` turbulent terms | `packages/core/legoesm/core/bulk_flux.py:1668` | yes | yes |
| `ice_alb`, pond and no-pond | `packages/ice/legoesm/ice/si3_surface.py:300` | yes | yes |
| `blk_ice_2` | `packages/ice/legoesm/ice/si3_surface.py:384` | yes | yes |
| `ice_flx_other`, pointwise neighbours | `packages/ice/legoesm/ice/si3_surface.py:467` | yes | yes |

Lane 3's three private copies in `sea_ice.py` are deleted. Its gate now reads
its constants from the C1D run's own resolved namelists
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_bulk_flux_gate.py:411`).
The no-pond arm is now transcribed:
`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icealb.f90:203`: `zafrac_pnd = 0._wp`;
`:214`: `IF( ph_snw(ji,jj,jl) == 0._wp .AND. pt_su(ji,jj,jl) >= rt0 ) THEN   ;   zalb_ice = rn_alb_imlt`.
The snow-zero tests read the bit pattern, as lane 3 did
(`packages/ice/legoesm/ice/si3_surface.py:262`, `packages/ice/legoesm/ice/si3_surface.py:268`).

Codex's "C1D bits byte-identical" cannot hold with one NEMO-literal body: lane
3's copies used a different operation order. Measured instead, every value:

| control (parent `d5bcd0d428` vs current) | result |
|---|---|
| lane-3 C1D bulk card, 227,760 values | 16,494 changed; all now equal NEMO's record; 0 that matched changed |
| lane-3 card non-bit rows | 16,512 -> 18 (the 18 are the native-`exp` rows awaiting the libm policy) |
| D5 C1D year, CPU `jit`, default ISA | 237,214 values identical; shipped hash `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2` |
| D5 C1D year, CPU `jit`, `--xla_cpu_max_isa=AVX` | 222,618 values identical; same shipped hash |
| production `nemo_si3_constant` dispatch under `jit`, 20,000 cells | stress <=2 ULP, latent <=3,571 ULP, sensible <=100,652 ULP (rel 2e-11); parent `jit` disagreed with its own eager on 18,257 + 8,112 values, current `jit` == eager on all |

## Codex item 4 — oracle on every output

The independent per-cell glibc transcription now covers all 45 model leaves
(`blk_ice_1` 9 incl. `zsipt`, albedo, `blk_ice_2` 19 outputs and 11 statement
intermediates — `zcptrain`, `zcptsnw`, `zcptn`, `zsq`, `zsipt`, `z_qlw`,
`z_dqlw`, `z_qsb`, `z_dqsb`, `zevap`, `zsnw` — and the 3 lead/bottom outputs).
The key sets are asserted equal, so an unscored output fails the test. It runs
on three decks: production, the O2 overlay, and C1D (no ponds)
(`tests/ice/unit/test_si3_surface.py:398`).

## Codex item 5 — deck

Preregistration addendum 2: only the deck the user chooses is acquired; both
arms existing in the model does not choose it.

## Statements (unchanged from the first draft, now in the shared code)

- `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1082`: `wndm_ice(ji,jj) = SQRT( pwndi(ji,jj) * pwndi(ji,jj) + pwndj(ji,jj) * pwndj(ji,jj) )`
- `:1137`: `zztmp         = rhoa(ji,jj) * Cd_ice(ji,jj) * wndm_ice(ji,jj)`
- `:1231`: `qsr_ice(ji,jj,jl) = zztmp * ( 1. - palb(ji,jj,jl) ) * qsr(ji,jj)`
- `:1280`: `evap_ice (ji,jj,jl) = rn_efac * qla_ice (ji,jj,jl) * z1_rLsub    ! sublimation`
- `:1290`: `emp_ice(ji,jj) = SUM( a_i_b(ji,jj,:) * evap_ice(ji,jj,:) ) - sprecip(ji,jj) * zsnw(ji,jj)`
- `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:167`: `CALL lbc_lnk( 'icesbc', utau_ice, 'T', -1.0_wp, vtau_ice, 'T', -1.0_wp )`
- `:463`: `fhld (ji,jj) = zswitch * MAX( 0._wp, ( zqld - zqfr_pos ) * r1_Dt_ice / MAX( at_i(ji,jj), epsi10 ) ) ! divided by at_i since this is (re)multiplied by a_i in icethd_dh.F90`
- `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbc_phy.f90:692`: `zle  = rAg_i*(ztmp - 1._wp) + rBg_i*LOG10(ztmp) + rCg_i*(1._wp - zta/rtt0) + rDg_i`
- `:350`: `theta_exner_sclr = pta * ( rpref / ppa ) ** rgamma_dry`
- `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1481`: `IF( ln_pnd_alb  ) THEN ; ln_pnd_alb  = .FALSE. ; CALL ctl_warn( 'ln_pnd_alb=false when no ponds' )  ; ENDIF`

The rows that changed on the C1D card are exactly these statements' order:
stress `(rhoa*Cd)*wndm` (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1137`), `evap`/`devap` multiply by `z1_rLsub`
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1280`), `emp_ice`'s `SUM` (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1290`), `fhld`'s `* r1_Dt_ice /`
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:463`).

## Tests and gates

- `tests/ice/unit/test_si3_surface.py`, namelist reader, lane-3 bulk-gate test,
  round-314 audit test: `40 passed, 2 xfailed in 82.09s`. The xfails: the
  native-`exp` plant is vacuous on the synthetic fixture (agrees with glibc;
  still required on the record), and the subnormal-snow-under-`jit` platform
  floor (eagerly the shared code gives NEMO's answer).
- Plants that fire on the full oracle: `Cd_ia` +1 ULP, `rtt0` +1 ULP,
  `LOG(10)` +1 ULP, `SUM` from the first category, `nn_snwfra` swapped,
  planted `zsipt` consumer.
- Eager == `jit` on all 45 leaves; gradients finite, central difference on one
  `t_su` cell within 1e-5.
- `tests/ice/unit`: `391 passed, 1 skipped, 2 xfailed`. SI3 fidelity-gate
  tests: `93 passed`. Sea-ice/core/coupler/land unit set: `25 failed, 619
  passed`; parent `25 failed, 618 passed`, same 25 failures (pre-existing).
- `test_nemo_si3_scalarmath_v2_gate.py::test_full_v2_gate_and_plants` fails
  "A MY_SRC is not verbatim" on parent and current alike (NEMO tree check).
- Ratchets: the same 11 pre-existing reds as before this fix; none in a
  touched file. Saturation ratchet passes.
- Citation gate, all SI3 receipts, full map; default gate; private-import
  ratchet: quoted in the final message.
- Review: `codex review: operator`.

## CONFIRMED vs PLAUSIBLE

- CONFIRMED: bit-identity to the independent glibc transcription on every
  output, three decks, under `jit`; lane-3 per-value control; D5 hash and
  reports parent == current on both ISAs; production config read from the
  shipped deck.
- PLAUSIBLE: bit-exactness against an ORCA2 NEMO record (none read; P1-P6).

## CHOICES

- O7 acquisition deck and window. ASKED (pending with the user).
- Lane-3 C1D card now moves 16,494 values onto NEMO's record instead of
  staying byte-identical. UNASKED (consequence of one implementation);
  offered as the decision below.
- Production `nemo_si3_constant` dispatch bits change under `jit` (now
  equal to its eager arithmetic). UNASKED; same decision.
- Lane-3 gate reads its constants from the C1D run's resolved namelists
  instead of library constants (values identical). UNASKED; offered for revert.
- Namelist reader moved from the ORCA2 round-314 audit into core. UNASKED,
  mechanical (RULE 4).
- Fixtures: production deck namelists and the C1D resolved namelists
  committed under `tests/ice/unit/data/`. UNASKED; offered for revert.
- Module renamed `si3_orca2_surface` -> `si3_surface`. UNASKED; offered for revert.
- No configuration default, production config or deck changed.

Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round11/fix/`.
