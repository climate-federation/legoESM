# SI3 round 17 receipt — O11 model preparation: lateral melt (ice_thd_da)

## Own errors, first

The first commit (`2066cc5961`) shifted every line of `bitz_lipscomb.py` and
`si3_surface.py` that earlier D5/O7/O8 receipts cite (the citation gate failed
seven receipts). `caf7fd5cca` returns both files to the parent except for
alias names appended after their last cited line; all 13 SI3 receipts pass again.
This receipt first said nothing calls the routine and that the glibc `pow` arm is
not observable; both were wrong or incomplete, and are corrected under Resume 1.

## Outcome

`ice_thd_da` as it executes on the production ORCA2_SAS_ICE deck is one
NEMO-literal function in the shared ice package (`si3_thd_da.py`). It is
bit-identical to an independent per-cell CPython transcription with glibc `pow`,
under `jit`, eager, and with traced namelist scalars. Resume 1 makes it selectable on
the production SI3 column path behind `SeaIceConfig.si3.lateral_melt`; the C1D card keeps
it off (`ln_icedA = .false.`) and ORCA2's `ice_thd` stage does not exist yet.
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
thermodynamics module (`packages/ice/legoesm/ice/bitz_lipscomb.py:2385`). They are
reused through public aliases (`packages/ice/legoesm/ice/si3_surface.py:590-593`,
`packages/ice/legoesm/ice/bitz_lipscomb.py:2420`), not copied.

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
`zastar`; at a traced `rn_dmin = 6.2` it moves 143 of 257 cells (a compile-time
`rn_dmin` folds the quotient exactly; Resume 1). `pow` follows the policy (DECISION 162);
see Resume 1 for where the glibc arm is and is not observable.

## Tests (first landing)

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

## Controls, first landing (parent `abd7262c11` vs `caf7fd5cca`)

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

## Gates, first landing

This receipt's citation gate: `PASS`, 41 citations, 0 unmapped; a planted citation makes it
`FAIL`. All 14 SI3 receipts pass. Default citation-gate test: `32 passed`. Private-import
ratchet: `6 passed`.

## Findings outside this rung (not acted on)

- The surface module's `<= 0` bit helper is fused by XLA into a flushing float compare under
  `jit` (`+5e-324` reads as `<= 0`); its module docstring already declares that subnormal floor.

## CONFIRMED vs PLAUSIBLE

- CONFIRMED: bit-identity to the independent transcription (every input class above, `jit`,
  eager, traced scalars); object-code semantics of all five compare sites; plants fire;
  D5 hashes and listed callers unchanged; all SI3 receipts' citations resolve; Resume 1's
  production-dispatch bit-identity, plants, ties, subnormal pin and glibc firing control.
- PLAUSIBLE: bit-exactness against an ORCA2 NEMO record (none read; P1-P3 preregistered);
  that `at_i_1d` equals the entry `a_i` on the one-category C1D column (read off the code).

## CHOICES

- Lateral melt in a new module rather than appended to the thermodynamics module (keeps every cited line in place). UNASKED, mechanical; offered for revert.
- Public alias names added to the surface and thermodynamics modules instead of moving the helpers (moving them breaks cited lines). UNASKED; offered for revert.
- `a_i > 0` read on the bit pattern (positive subnormal true), as O8's `< 0`; unreachable on this deck (`a_i_1d` is `+0` or `> 1e-10`). UNASKED; offered for revert.
- Floe block computed for every cell and selected, instead of only in the branch. UNASKED; forward bits identical.
- Untaken cells divide by 1 instead of `at_i` (gradient only). UNASKED; offered for revert.
- Caller wired on the C1D column path (codex HOLD item 1, ASKED); ORCA2's `ice_thd` stage is not built.
- `&namthd_da` as a new no-default `SI3LateralMeltConfig` held by `SI3ThermoConfig.namthd_da` (None when off). UNASKED; offered for revert.
- `sst_m` as an optional ice-entry forcing field (None; refused when lateral melt is on). UNASKED; offered for revert.
- `si3_column_step_arrays(..., si3=None)`: None is `ln_icedA = .false.`, so direct D5 callers stay unchanged. UNASKED; offered for revert.
- Wiring written into blank lines and in-place line edits so no cited line moves (no re-anchoring). UNASKED, mechanical.
- Column drops the lateral-melt `sfx/hfx/wfx` (it returns no thermodynamic fluxes for any stage). UNASKED -> named, not changed.
- `at_i` = entry `a_i`; only cells gathered at entry (`a_i > epsi10`) change; `nn_icesal` from `salinity_scheme`. UNASKED; follows NEMO's gather.
- No configuration default, production config or deck changed (`lateral_melt` stays False). GPU spot-check run (Resume 1).

Review: `codex review: operator`.

## Resume 1 — codex HOLD items (commits `f37f838e42`, `4818655e89`)

1. **Selectable.** `SeaIceConfig.si3.lateral_melt` now has a consumer. NEMO calls
   `ice_thd_da` after the second `ice_thd_temp` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:205`)
   and before `ice_thd_do` (`:227`), inside the cells gathered at `:181`
   (`IF ( a_i(ji,jj,jl) > epsi10 ) THEN`); the column step does the same at
   `packages/ice/legoesm/ice/bitz_lipscomb.py:2333` through `:2446`, which applies the
   gather at `:2478`. `step_sea_ice` passes the config (`packages/ice/legoesm/ice/sea_ice.py:517`;
   keyword at `packages/ice/legoesm/ice/bitz_lipscomb.py:2275`). On needs the namelist values
   (`packages/ice/legoesm/ice/config.py:309`, class `:597-601`, no defaults) and `sst_m`
   (`packages/ice/legoesm/ice/bitz_lipscomb.py:42`); the identity check keeps every other field
   fixed (`packages/ice/legoesm/ice/config.py:316`). The C1D deck still disables it
   (`scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_ice_cfg:91`).
   D5 C1D year, wired `f37f838e42` vs parent `f87536a06d` (clean detached worktree), same
   environment: 18 of 18 arms identical on CPU (`XLA_FLAGS=""`) and on AVX, every JSON leaf
   equal except the worktree stamp; shipped arm `b7a7329b...`; `D5_PLATFORM_FLOOR`.
   Production-path test fails when the call is removed: the real line replaced by
   `post_da = post_temp2` turns the deck bitwise test red.
2. **Plants through the production dispatch** (`step_sea_ice`, `jit`, 255 cells, bitwise
   against the oracle applied to the column's own post-temperature state, then the
   non-cooling open-water step): exact at the deck, at a constant `rn_beta = 3`, and at a
   traced `rn_dmin = 6.2`; plants fire: call removed, NaN-propagating `MAX`, `a_i == 0`
   never, `1/300` on a traced `rn_dmin`, native `pow`. Not reachable through this column,
   so leaf-only: source rounding and `SUM` order (they move only the fluxes the column does
   not return), a NaN `MIN` operand (needs `at_i <= epsi10`, never gathered). A constant
   `rn_dmin` folds `rn_dmin / 300` exactly, so the `1/300` hazard is a traced-parameter
   (training) hazard only.
3. **Positive-subnormal `a_i`.** Two cells with `a_i = 5e-324` and `-0` accumulators: the
   taken branch turns `sfx_lam`/`wfx_lam` into `+0` as binary64 does; with line 143 written
   as `a_i > 0` they stay `-0` and the test fails (real-source edit, reverted). Inside the
   branch XLA CPU flushes the subnormal, so `a_i`, `h_i`, `h_s` become `+0` where binary64
   keeps `5e-324`: pinned as `D5_PLATFORM_FLOOR` (the GPU arm matches binary64 here).
4. **MIN/MAX ties.** `+0`/`-0` ties keep the first model operand (`minsd`/`maxsd`); each
   site's operands are recorded and pinned (`MAX(+0, y)` gives `+0` at `y = -0`; `at_i` and
   `a_i` first in the two `MIN`). Reversing the real line at :149 or :152 fails the test.
   No stored bit can see these ties: `pow(+-0, 1.36) = +0`, `x` and `at_i` share a sign,
   and the branch needs `a_i > 0`; so the pin is on the operands, not the outputs.
5. **glibc arm control.** 4.8 M random bases with traced exponents: native and glibc agree
   except subnormal results (flushed downstream). With a compile-time exponent XLA rewrites
   `pow` (0.5, 2, 3, -1) and differs from glibc (3: 184,417 of 900,000 bases). A constant
   `rn_beta = 3`: native moves 28 of 257 leaf cells (2 in `a_i`); the plant fails the test
   at the leaf and through the dispatch. On the deck (`rn_beta = 1`) the arms agree.
6. **GPU spot-check (DECISION 158), by counts.** Both V100S idle before (0 MiB, 0 %, no
   compute apps); GPU 1 used; idle after (0 MiB). Native `pow` (glibc arm is CPU-only).
   Leaf, deck, 257 cells: 2 `a_i`, 24 `sfx_lam`, 30 `hfx_thd`, 26 `wfx_lam` cells differ from
   glibc binary64; 2 subnormal cells: 0 differ. Whole production step GPU vs CPU, 255 cells:
   1 `concentration`, 4 `h_ice`, 6 `h_snow`, 0 `T_surface` cells differ. Not certified.

Controls (parent `f87536a06d`, model-identical to `caf7fd5cca`):

| caller | first landing | Resume 1 |
|---|---|---|
| `tests/ice/unit` | 422 passed, 1 skipped, 1 xfailed | 440 passed, 1 skipped, 1 xfailed (+18) |
| O11 tests | 14 passed | 32 passed |
| SI3 fidelity tests (11 files) | 1 failed, 143 passed | 2 failed, 142 passed (O7 preflight passes alone: 1 passed in 92.77s) |
| CI ratchets (10 files) | 13 failed | same 13 failed |

The two fidelity reds are the parent's: scalarmath v2 "A MY_SRC is not verbatim", and
the O7 preflight subprocess timeout at 120 s (also on the parent last round). One D5
dispatcher plant had a fixed signature; it now forwards the new keyword (lines unmoved).
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round17/resume1/`.

Gates (Resume 1): this receipt's citation gate `PASS`, 53 citations, 0 unmapped, full-map
audit 0 failing; a planted citation makes it `FAIL`. All 14 SI3 receipts pass. Default
citation-gate test: `32 passed in 8.16s`. Private-import ratchet: `6 passed in 6.77s`.
Review: `codex review: operator`; GLM: not run (NO GATE).
