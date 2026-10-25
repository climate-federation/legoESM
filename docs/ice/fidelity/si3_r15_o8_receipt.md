# SI3 round 15 receipt — O8 model preparation: frazil collection bookkeeping

## Own errors, first

The first landing (`5a8d17ed3d`) overstated integration: the new function
was called only by its tests, while the production C1D step kept its own
inline cooling predicate and new-ice thickness (codex HOLD). Resume 1
(`29d94bd78d`) wires the production step to it.

## Outcome

`ice_thd_frazil` as it executes on the production deck (collection off) is
one NEMO-literal function in the shared ice package, and since resume 1 the
production C1D step calls it: the open-water growth takes `ht_i_new` from it
and gathers cooling cells with the same `qlead < 0` predicate; the inline
copy is gone. The function is bit-identical to an independent per-cell
transcription under `jit` and eager, including `-0`, NaN and
negative-subnormal lead heat. The D5 C1D year is byte-identical to the
parent on all 18 arms, CPU and AVX. No NEMO array was read for O8; the
acquisition is preregistered (`docs/ice/fidelity/si3_r15_o8_preregister.md`)
and not built.

## Build cited

`ORCA2_SAS_ICE_O7_REF` (the unpatched reference of the production-deck O7
acquisition), flags (`ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm:27`):
`%FCFLAGS             -fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize`.
Its `icethd_do.f90` (SHA-256 `bdaabab8...c7c8e`) and `icethd.f90` are
byte-identical across `ORCA2_SAS_ICE_{O1,O2,O5,O7}_{REF,REC}`.

## Search before building

Searched `packages/`, `src/`, `scripts/validate/`, `tests/ice/` for
`frazil`, `hinew`, `ht_i_new`, `fraz_frac`, `qlead`, `thd_do`. Found: no
transcription of `ice_thd_frazil` anywhere; before resume 1 the C1D column
step inlined `qlead < 0.0` and the module constant `_NEW_ICE_THICKNESS = 0.05`
(`packages/ice/legoesm/ice/bitz_lipscomb.py:113`, the C1D card's `rn_hinew`)
in its open-water growth. The ocean
`physics/frazil.py` is the ocean-side supercooling fix, a different
statement. The bit-pattern zero test in `packages/ice/legoesm/ice/si3_surface.py:262`
is the `== 0` sibling; it is private to that module, so the `< 0`
predicate lives beside the new function.

## Statements transcribed

`ice_thd` calls the routine unconditionally:
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:174`:
`CALL ice_thd_frazil             !--- frazil ice: collection thickness (ht_i_new) & fraction of frazil (fraz_frac)`.
Executed on this deck (`ln_frazil = .false.`, `rn_hinew = 0.1`,
`tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:190-191`):

- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:419`: `fraz_frac(:,:) = 0._wp`
- `:422`: `WHERE( qlead(:,:) < 0._wp ) ! cooling`
- `:423`: `ht_i_new(:,:) = rn_hinew`
- `:425`: `ht_i_new(:,:) = 0._wp`
- `:428`: `IF( ln_frazil ) THEN` — false; the collection arm is refused, not transcribed.

Arrays are interior-only:
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/ice.f90:445`:
`ALLOCATE( ht_i_new (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , fraz_frac (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , STAT=ierr(ii) )`.
The routine has no category axis on this arm; `hi_max(jpl)` is read only in
the collection arm (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:474`),
so jpl=5 needs no generalisation.

Model: `packages/ice/legoesm/ice/bitz_lipscomb.py:1799` (`si3_ice_thd_frazil`),
predicate `packages/ice/legoesm/ice/bitz_lipscomb.py:1787`.

## Production wiring (resume 1)

ice_thd_do gathers its cells with the same predicate
(`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:182`:
`IF ( qlead(ji,jj)  <  0._wp ) THEN`) and reads `ht_i_new` as its new-ice
thickness (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:200`,
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:283`). The C1D step now
calls the routine (`packages/ice/legoesm/ice/bitz_lipscomb.py:1750`) with
the C1D card's values
(`scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_ice_cfg:113-114`:
`rn_hinew = 0.05`, `ln_frazil = .false.`), takes the cooling mask from the
shared predicate (`packages/ice/legoesm/ice/bitz_lipscomb.py:1751`) and
divides by `ht_i_new` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1759`).
Non-cooling cells divide `+0` by 1 rather than by `ht_i_new = +0`, which
keeps 0/0 out of the forward and the gradient; their values stay `+0`.
ORCA2's ice_thd_do itself is rung O14; nothing of it is transcribed here.

Production-path tests (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2808`,
`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2822`), on a real
C1D entry (kt = 1) widened to seven columns with
`qlead = -300, -1e-3, -5e-324, -0, +0, +5e-324, +300`, CPU `jit`:
a spy shows the step calls the routine once with the card's values; three
plants each change exactly the predicted columns — ORCA2's `rn_hinew = 0.1`
wired in (columns 0, 1), always-cooling predicate (column 6), never-cooling
predicate (columns 0, 1). With the parent model file restored, all four
fail (`4 failed`). A negative-subnormal plant is not observable in this
step's outputs (its growth is `±0`), so it stays a unit-level plant.

The frazil unit tests place their inputs on the CPU device
(`jax.device_put(..., jax.devices("cpu")[0])`) because the float-compare
plant relies on XLA CPU flushing; `jax.jit(backend="cpu")` exists in JAX
0.10.0 but warns that it is deprecated, so it was not used.

## Object code (the MIN/MAX/SIGN lesson)

`__icethd_do_MOD_ice_thd_frazil` in `ORCA2_SAS_ICE_O7_REF/BLD/obj/icethd_do.o`
(REF and REC instruction streams identical): `fraz_frac` is zeroed by a
`memset` (+0 bits); each `WHERE` cell is `comisd (qlead),%xmm0` with
`%xmm0 = +0`, then `ja` to `rn_hinew`, else `pxor` (+0). So cooling means
ordered `0 > qlead`: `-0` and NaN are not cooling, a negative subnormal is.
No `MIN`/`MAX`/`SIGN` and no `exp`/`log`/`pow` execute on this arm (the ten
`min`/`max` instructions belong to the collection arm), so no FMA barrier
(DECISION 158) and no glibc arm (DECISION 162) is involved.
Evidence: `ice_thd_frazil_O7_{REF,REC}.objdump.txt`.

XLA CPU `jit` flushes a negative subnormal to `-0` before a float compare
(measured: `jnp.where(q < 0, ...)` gives 0 for `-5e-324` under `jit`, 0.1
eager). The predicate therefore reads the bit pattern; it gives NEMO's
answer under both.

## Tests

`tests/ice/unit/test_si3_thd_frazil.py`: `13 passed in 2.39s`.

- Bitwise against the independent CPython transcription on shapes
  `(7,5)`, `(182,149)`, `(3,)`, scalar, with `-5e-324`, `-2.2e-308`, `±0`,
  `+5e-324`, `±NaN`, `±inf`.
- Inexact pin: `rn_hinew` from the shipped deck is `0x3FB999999999999A`,
  not exactly 1/10; cooling cells carry that pattern, `-0`/NaN give `+0`.
- Eager (`jax.disable_jit`) == `jit`, every bit.
- Production-path plants, all fire (cells hit): float `< 0` -> 2
  (`-5e-324`, `-2.2e-308`); `<= 0` -> 3 (`±0`, `+5e-324`); NaN-as-cooling
  -> 4; `rn_hinew` +1 ULP -> 19 (every cooling cell).
- Refusals: missing `rn_hinew` or `ln_frazil` (no defaults), `None`,
  `ln_frazil = True`, non-logical `ln_frazil`.
- Gradients finite; `d/d rn_hinew` central difference within 1e-6 and equal
  to `2 rn_hinew x` (cooling-cell count).

## Controls (parent `e8ee89ece8` vs current `5a8d17ed3d`)

| caller | parent | current |
|---|---|---|
| D5 C1D year, CPU `jit` (18 arm hashes) | shipped `b7a7329b...` | all 18 identical; report differs only in the worktree stamp |
| `tests/ice/unit` | 395 passed, 1 skipped, 1 xfailed | 408 passed, 1 skipped, 1 xfailed (+13 new) |
| SI3 fidelity tests (11 files) | 2 failed, 138 passed | same 2 failed, 138 passed |
| ratchets + default citation gate | 11 failed | same 11 failed |

The two fidelity failures (scalarmath v2 "MY_SRC not verbatim"; O7 preflight)
and the 11 ratchet reds are pre-existing and identical on both trees; the
`bitz_lipscomb.py` inline-coefficient red lists only the old
`jk_minus_half` site.

## CONFIRMED vs PLAUSIBLE

- CONFIRMED: bit-identity to the independent transcription, every class of
  input above, `jit` and eager; object-code semantics of the `WHERE`; D5
  hash and every listed caller unchanged.
- PLAUSIBLE: bit-exactness against an ORCA2 NEMO record (none read; P1-P3).

## CHOICES

- O8 acquisition deck/window/ranks, own run instead of the master record: UNASKED -> DECISION_NEEDED.
- Bit-pattern `< 0` predicate instead of a float compare, so a negative subnormal matches NEMO under `jit`. UNASKED; offered for revert.
- Collection arm refused rather than transcribed (it does not execute). UNASKED; offered for revert.
- Function placed in the existing SI3 thermodynamics module. UNASKED, mechanical.
- C1D's `_NEW_ICE_THICKNESS = 0.05` not rewired to the new function (its consumer is `ice_thd_do`, rung O14). UNASKED; finding only, nothing changed.
- No configuration default, production config or deck changed.

Review: `codex review: operator`.
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round15/`.
