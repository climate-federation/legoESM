# SI3 round 17 receipt — O11 model preparation: lateral melt (ice_thd_da)

## Own errors, first

The first commit (`2066cc5961`) shifted every line of `bitz_lipscomb.py` and
`si3_surface.py` that earlier D5/O7/O8 receipts cite (the citation gate failed
seven receipts). `caf7fd5cca` returns both files to the parent except for
alias names appended after their last cited line; all 13 SI3 receipts pass again.

## Outcome

`ice_thd_da` as it executes on the production ORCA2_SAS_ICE deck is one
NEMO-literal function in the shared ice package (`si3_thd_da.py`). It is
bit-identical to an independent per-cell CPython transcription with glibc `pow`,
under `jit`, eager, and with traced namelist scalars. Nothing calls it yet: the
C1D card selects `ln_icedA = .false.` and ORCA2's `ice_thd` stage does not exist.
The D5 C1D year is byte-identical to the parent (18 of 18 arms, CPU and AVX).
No NEMO array was read; the O11 projection of the shared master record is
preregistered (`docs/ice/fidelity/si3_r17_o11_preregister.md`); no tool built.

## Build cited

`ORCA2_SAS_ICE_O7_REF`, flags (`ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm:27`):
`%FCFLAGS             -fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize`.
`icethd_da.f90` (md5 `4a70434478af079f4c987cb8f3bd8fb4`) is byte-identical across
`ORCA2_SAS_ICE_{O1,O2,O5,O7}_{REF,REC}`; the `icethd_da.o` instruction streams of
`O7_REF` and `O7_REC` are identical.

## Search before building

Searched `packages/`, `src/`, `scripts/validate/`, `tests/ice/` for `thd_da`,
`lateral melt`, `rn_dmin`, `rn_beta`, `zastar`, `dfloe`, `zwlat`, `wfx_lam`,
`sfx_lam`, `Josberger`, `Lupkes`. Found: no transcription anywhere. The C1D
path does not run it (`scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_ice_cfg:91`:
`ln_icedA         = .false.    ! ORCA1 ice cfg:87`), and the selector field
`SI3ThermoConfig.lateral_melt` (`packages/ice/legoesm/ice/config.py:308`:
`lateral_melt: bool = False`) is read by nothing. Compiled-Fortran helpers
already existed: `MIN`/`MAX`/`SUM`/bit-zero in the surface module
(`packages/ice/legoesm/ice/si3_surface.py:243`, `packages/ice/legoesm/ice/si3_surface.py:253`,
`packages/ice/legoesm/ice/si3_surface.py:262`) and the integer-only `< 0` in the
thermodynamics module (`packages/ice/legoesm/ice/bitz_lipscomb.py:1787`). They are
reused through public aliases (`packages/ice/legoesm/ice/si3_surface.py:590-593`,
`packages/ice/legoesm/ice/bitz_lipscomb.py:1822`), not copied.

## Statements transcribed

Call point: `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:210`:
`IF( ln_icedA )    CALL ice_thd_da                       ! --- Lateral melting --- !`.
Deck values (`tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:179`: `rn_beta          =   1.0`;
`tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:183`: `rn_dmin          =   8.`;
`tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:199`: `nn_icesal        =   4`).
Every executing statement:

- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:167-170` PARAMETERs `zdmax = 300._wp`, `zcs = 0.66_wp`, `zm1 = 3.e-6_wp`, `zm2 = 1.36_wp` (`packages/ice/legoesm/ice/si3_thd_da.py:25-28`).
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:174` `zastar = 1._wp / ( 1._wp - (rn_dmin / zdmax)**(1._wp/rn_beta) )`
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:178` `IF( nn_icesal == 4 ) THEN   ;   zs_i(:) = sz_i_1d(ji,:)` (else arm `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:179` also transcribed)
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:183` `zdfloe = rn_dmin * ( zastar / ( zastar - at_i_1d(ji) ) )**rn_beta`
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:185` `zperi  = at_i_1d(ji) * rpi / ( zcs * zdfloe )`
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:187` `zwlat  = zm1 * ( MAX( 0._wp, sst_1d(ji) - ( t_bo_1d(ji) - rt0 ) ) )**zm2`
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:189` `zda_tot = MIN( zwlat * zperi * rDt_ice, at_i_1d(ji) )`
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:192` `IF( a_i_1d(ji) > 0._wp ) THEN`
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:195` `zda = MIN( a_i_1d(ji), zda_tot * a_i_1d(ji) / at_i_1d(ji) )`
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:198` `sfx_lam_1d(ji) = sfx_lam_1d(ji) + rhoi * zda * r1_Dt_ice * h_i_1d(ji) * r1_nlay_i * SUM( zs_i(:) )`
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:201-202` `hfx_thd_1d(ji) = hfx_thd_1d(ji) - zda * r1_Dt_ice * ( h_i_1d(ji) * r1_nlay_i * SUM( e_i_1d(ji,1:nlay_i) )  &` ... `+ h_s_1d(ji) * r1_nlay_s * SUM( e_s_1d(ji,1:nlay_s) ) )`
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:205` `wfx_lam_1d(ji) =  wfx_lam_1d(ji) + zda * r1_Dt_ice * ( rhoi * h_i_1d(ji) + rhos * h_s_1d(ji) )`
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:208` `a_i_1d(ji) = a_i_1d(ji) - zda`; `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:211-213` `IF( a_i_1d(ji) == 0._wp ) THEN` `h_i_1d(ji) = 0._wp` `h_s_1d(ji) = 0._wp`

Reciprocals as NEMO forms them (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icestp.f90:419`:
`r1_Dt_ice = 1._wp / rDt_ice`; `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icestp.f90:423`:
`r1_nlay_i = 1._wp / REAL( nlay_i, wp )   !--- inverse of nlay_i and nlay_s`).
`rpi`, `rt0`, `rhos`, `rhoi`: `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/phycst.f90:36`
(`3.141592653589793_wp`, equal to `math.pi` in binary64), `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/phycst.f90:50`,
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/phycst.f90:68-69`, the latter three via
`NEMO_SI3_CONSTANTS_CONFIG`, passed in (no default).

Model: `packages/ice/legoesm/ice/si3_thd_da.py:65` (`si3_ice_thd_da`), shape-agnostic over
NEMO's gathered `ice1D` cells of one category (jpl and nlay come from the array shapes).

## Object code (the MIN/MAX/SIGN lesson)

`__icethd_da_MOD_ice_thd_da` (evidence `ice_thd_da_O7_{REF,REC}.objdump.txt`): no FMA
instruction; three `pow` calls (glibc); no `SIGN`. Each compare site, pinned:

| site | instruction | meaning | model |
|---|---|---|---|
| line 192 `a_i > 0` | `comisd %xmm3(+0),%xmm15` / `ja` | ordered; `+0`, `-0`, NaN false | integer bit test (`packages/ice/legoesm/ice/si3_thd_da.py:143`) |
| line 187 `MAX(0,y)` | `maxsd %xmm7(+0),%xmm0(y)` | `y > 0 ? y : +0`; `-0`, NaN give `+0` | `fortran_max(0, y)` (`packages/ice/legoesm/ice/si3_thd_da.py:149`) |
| line 189 `MIN(x,at_i)` | `minsd %xmm1(at_i),%xmm8(x)` | `x < at_i ? x : at_i`; NaN `x` gives `at_i` | `fortran_min(at_i, x)` (`packages/ice/legoesm/ice/si3_thd_da.py:152`) |
| line 195 `MIN(a_i,q)` | `minsd %xmm13(a_i),%xmm8(q)` | `q < a_i ? q : a_i` | `fortran_min(a_i, q)` (`packages/ice/legoesm/ice/si3_thd_da.py:155`) |
| line 211 `a_i == 0` | `ucomisd` / `jp`,`jne`; `movq $0x0` | `+0` stores | bit-zero test (`packages/ice/legoesm/ice/si3_thd_da.py:170`) |

The floe block (source lines 183-189) is computed inside the `a_i > 0` branch in the object code; the
model computes it everywhere and selects, which gives the same stored bits. `SUM` is a
sequential `addsd` chain from `pxor` (+0) in layer order (`fortran_sum0`).

## FMA, constant division, transcendentals

Every operation is source-rounded (DECISION 158). Measured on this function: removing the
rounding changes `sfx/hfx/wfx` bits in 158 of 257 cells, so XLA does contract here.
XLA also turns a division by the constant 300 into a multiply by `1/300`
(85,048 of 200,000 random quotients differ); `zdmax` is therefore a rounded operand
(`packages/ice/legoesm/ice/si3_thd_da.py:50`). On the deck's `rn_dmin = 8` that rewrite happens to give the same
`zastar`; at `rn_dmin = 6.2` it moves 143 of 257 cells. `pow` follows the policy
(DECISION 162); on 20,000 bases per exponent (1.36, 1.0, 1.2, 0.3) XLA's native `pow`
equals glibc, so the glibc arm is selected but not observable here (PLAUSIBLE that it
never matters on this routine).

## Tests

`tests/ice/unit/test_si3_thd_da.py`: `14 passed in 12.71s`.

- Bitwise against an independent CPython + ctypes-glibc transcription, `nn_icesal` 4 and 2,
  257 cells with eight object-code edge cells (`a_i` `+0`/`-0`/NaN, `sst` NaN, below freezing,
  `-0` temperature excess, total melt, NaN `MIN` operands); `jit` == eager every bit; traced
  `rn_beta`/`rn_dmin` bitwise.
- Inexact pins: `zastar = 0x1.070381c0e0703p+0`; `zcs`, `zm1`, `zm2`, `rpi`, `rt0` hex patterns.
- Edge pins: NaN `sst` melts nothing (a NaN-propagating `MAX` would poison the cell);
  `at_i = +0`, `sst = +inf` empties the category through both `MIN` operand orders.
- Bit predicate: `5e-324 > 0` is true under `jit` (a float compare says false).
- Plants on the production function, all fire: NaN-propagating `MAX` (1 cell),
  NaN-propagating `MIN` (1), reversed `SUM` (151), no source rounding (158),
  `a_i == 0` never (2), `zdmax` as `1/300` multiply at `rn_dmin = 6.2` (143).
- Refusals: any missing array or scalar, missing constants, `nn_icesal` not an integer 1..4.
- Gradients finite (also at edge cells); `d/d rn_dmin` and `d/d rn_beta` match central
  differences to 1e-5 relative and are nonzero.

## Controls (parent `abd7262c11` vs current `caf7fd5cca`)

Parent from a clean detached worktree; fp64, CPU, `jit`.

| caller | parent | current |
|---|---|---|
| D5 C1D year, 18 arms, `XLA_FLAGS=""` | shipped arm `b7a7329b...`, platform floor | all 18 identical |
| D5 C1D year, 18 arms, AVX | shipped arm `b7a7329b...` | all 18 identical |
| `tests/ice/unit` (O7 surface and O8 frazil callers) | 408 passed, 1 skipped, 1 xfailed | 422 passed, 1 skipped, 1 xfailed (+14 new) |
| SI3 fidelity tests (11 files) | 2 failed, 142 passed | 1 failed, 143 passed |
| CI ratchets (10 files) | 13 failed | same 13 failed |
| citation gate, 13 SI3 receipts | all PASS | all PASS |

The fidelity failure common to both is the scalarmath v2 full gate; the parent's second
(O7 preflight) did not recur on current, as in round 15. The 13 ratchet reds are identical
(the `bitz_lipscomb.py` inline-coefficient red is still only `jk_minus_half`); the
private-import ratchet is green. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round17/` (`head_caf7fd/` for current).

## Gates

This receipt's citation gate: `PASS`, 41 citations, 0 unmapped; a planted citation makes it
`FAIL`. All 14 SI3 receipts pass. Default citation-gate test: `32 passed`. Private-import
ratchet: `6 passed`.

## Findings outside this rung (not acted on)

- The surface module's `<= 0` bit helper is fused by XLA into a flushing float compare under
  `jit` (`+5e-324` reads as `<= 0`); its module docstring already declares that subnormal floor.
- `SI3ThermoConfig.lateral_melt` is a selector nothing reads; it matches the C1D card today.

## CONFIRMED vs PLAUSIBLE

- CONFIRMED: bit-identity to the independent transcription (every input class above, `jit`,
  eager, traced scalars); object-code semantics of all five compare sites; plants fire;
  D5 hashes and listed callers unchanged; all SI3 receipts' citations resolve.
- PLAUSIBLE: bit-exactness against an ORCA2 NEMO record (none read; P1-P3 preregistered);
  that the glibc `pow` arm matters for this routine.

## CHOICES

- Lateral melt in a new module rather than appended to the thermodynamics module (keeps every cited line in place). UNASKED, mechanical; offered for revert.
- Public alias names added to the surface and thermodynamics modules instead of moving the helpers (moving them breaks cited lines). UNASKED; offered for revert.
- `a_i > 0` read on the bit pattern (positive subnormal true), as O8's `< 0`; unreachable on this deck (`a_i_1d` is `+0` or `> 1e-10`). UNASKED; offered for revert.
- Floe block computed for every cell and selected, instead of only in the branch. UNASKED; forward bits identical.
- Untaken cells divide by 1 instead of `at_i` (gradient only). UNASKED; offered for revert.
- No caller wired: C1D has `ln_icedA = .false.` and ORCA2's `ice_thd` stage is not built. UNASKED -> named, not done.
- No configuration default, production config or deck changed. GPU spot-check (DECISION 158): not run; the glibc arm is CPU-only.

Review: `codex review: operator`.
