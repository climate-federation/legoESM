# SI3 Ladder Rung O8 Preregistration — Round 15 (`si3_r15`): frazil collection bookkeeping

- Date: 2026-10-25. Lane `si3`, round 15, rung O8 (`docs/ice/fidelity/si3_ladder_design.md:228`).
- Model parent: `e8ee89ece8cb` (D5, O1, O2, O3, O5, O7 landed).
- Status: committed BEFORE any O8 measurement. No NEMO array has been read for O8. The acquisition tool is NOT built this round.
- Reference build cited: `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/` (unpatched reference of the production-deck O7 acquisition). Its `icethd_do.f90` and `icethd.f90` are byte-identical (`md5sum`) across `ORCA2_SAS_ICE_{O1,O2,O5,O7}_{REF,REC}`. Flags `ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm`: `-fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize`.

## 1. What executes on the production deck

The production deck resolves `ln_frazil = .false.`, `rn_hinew = 0.1`
(`&namthd_do` in `namelist_ice_ref:190-191`; `namelist_ice_cfg`'s
`&namthd_do` sets neither). `ice_thd` calls `ice_thd_frazil`
unconditionally (`icethd.f90:174`) whenever `ln_icethd` (`icestp.f90:250`).
Executed statements (`icethd_do.f90`):

- `:419` `fraz_frac(:,:) = 0._wp`
- `:422-426` `WHERE( qlead(:,:) < 0._wp ) ht_i_new(:,:) = rn_hinew ELSEWHERE ht_i_new(:,:) = 0._wp`
- `:428` `IF( ln_frazil ) THEN` — false; `:429-481` do not execute.

Object code (`ORCA2_SAS_ICE_O7_REF/BLD/obj/icethd_do.o`, `__icethd_do_MOD_ice_thd_frazil`;
REF and REC instruction streams identical): `fraz_frac` is a `memset` 0 (+0 bits);
the `WHERE` is `comisd (qlead),%xmm0[=+0]` / `ja` -> `rn_hinew`, else `pxor` (+0).
So `-0` and NaN are not cooling, and a negative subnormal IS cooling (no DAZ in
gfortran without fast-math). No `MIN`/`MAX`/`SIGN`, no `exp`/`log`/`pow` executes
on this arm (all ten `min`/`max` instructions belong to the collection arm).
Arrays are `A2D(0)`: `qlead`, `ht_i_new`, `fraz_frac` all `ALLOCATE`d
`(Nis0:Nie0, Njs0:Nje0)` (`ice.f90:414,445`).

## 2. Deck, window, ranks — PENDING USER DECISION (section 7)

Pick: an O8-only instrumented run of the **unmodified production**
`ORCA2_SAS_ICE` deck, one MPI rank, `nn_it000 = 1`, `nn_itend = 2` — the
O7 pattern (DECISION 165). The design text (`si3_ladder_design.md`, O8-O16
paragraph) says O8-O16 come from ONE instrumented 100-step master record; the
observers for O9-O16 do not exist yet, so that record cannot be built now.
This deviation needs the user's answer before any tool is written.

Every scalar is read from the run's resolved `output.namelist.ice`, never the
deck text: `ln_frazil`, `rn_hinew`, `ln_icethd`, `ln_icedO`, `jpl`; refuse
unless `ln_frazil = F` and `ln_icethd = T`.

## 3. Arrays and call points (label: GIVEN-NEMO-ENTRY only)

Binary64, interior extents `(Nis0:Nie0, Njs0:Nje0)` as declared, bounds in the
record header. Observer copies are statement-level in `ice_thd`, outside every
loop.

- F1, immediately before `icethd.f90:174`: `qlead`, and the pre-call `ht_i_new`, `fraz_frac` (to show the routine overwrites every cell).
- F2, immediately after `icethd.f90:174` (before the `DO jl` loop at `:176`): `ht_i_new`, `fraz_frac`.
- F0 run scalars: `kt`, `nit000`, `jpni`, `jpl`, `rn_hinew` (module value, compared to the resolved namelist).

Steps `kt = 1, 2`. Size: three interior arrays x 2 frames x 2 steps, well
under 10 MiB (the design row's 1-2 GiB is the master-record projection).

## 4. Frozen predictions (CPU fp64, production `jax.jit`)

- P1: F2 `ht_i_new` equals `si3_ice_thd_frazil(qlead=F1, rn_hinew=<resolved>, ln_frazil=False).ht_i_new`, 0 differing bits on every interior cell, both steps.
- P2: F2 `fraz_frac` is +0 bits on every interior cell, both steps.
- P3: resolved `rn_hinew` bits `0x3FB999999999999A`, `ln_frazil = F`, module `rn_hinew` equal.
- Falsifier for each: any differing bit. The counts of F1 `qlead` cells that are `-0`, `+0`, NaN, negative subnormal and negative normal are reported per step; a predicate claim is made only for classes the record contains.
- The first non-bit statement, if any, is named with its `ppsrc` line before any change lands.

## 5. Controls the gate must carry

Reader: swapped `ht_i_new`/`fraz_frac`, one-bit payload, truncated record,
wrong time level. Scorer plants that must turn the score red: `rn_hinew`
+1 ULP (fires iff the record has a cooling cell); `qlead <= 0` predicate
(fires iff it has a `+-0` cell); float `qlead < 0` predicate under `jit`
(fires iff it has a negative-subnormal cell — reported vacuous otherwise);
`fraz_frac` from the pre-call value. Admission: reference vs record restart
bytes identical, `STOP 0`, passivity census per the ACQUISITION TOOL
CHECKLIST.

## 6. CHOICES (ASKED/UNASKED)

- Deck, rank count, step window, own run instead of the master record: section 2. UNASKED -> DECISION_NEEDED.
- Reference build `ORCA2_SAS_ICE_O7_REF` (byte-identical alternatives). UNASKED, mechanical.
- Recording pre-call `ht_i_new`/`fraz_frac` (F1) in addition to the design's inputs/outputs. UNASKED; offered for revert.
- `qlead` enters as a recorded array (produced by `ice_flx_other`, admitted at O7). UNASKED; follows the exact-input rule.

## 7. Pending decision

O8 acquisition: (1) O8-only production-deck run, 1 rank, `kt = 1..2`
(pick: cheapest, covers `nit000` and a later step, same deck as O7);
(2) wait and take O8 from the single 100-step master record once O9-O16
observers exist; (3) O8-only production-deck run, 1 rank, 100 steps.
