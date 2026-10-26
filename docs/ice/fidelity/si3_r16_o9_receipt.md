# SI3 round 16 receipt — O9 model preparation: ZDF, DH and TEMP at jpl = 5, 10 + 5 layers

## Own errors, first

My first bit comparison failed on two leaves: XLA folded a zeroed accumulator's `0 + (-0)` to `-0`
(NEMO gives `+0`), and it rewrote a constant division `x / 0.1` as a reciprocal multiply. Both are fixed
below and both now have a gate; neither touches the D5 card's bits. The caller controls then caught a
D5 test still calling the old `_radiation` signature; fixed (test-only).

## Outcome

The C1D BL99/P07 heat diffusion and growth/melt code now runs on any layer count as one NEMO-literal path
in the shared ice package: the production salinity-4 arm, every `ice_thd_dh` flux and `dh_*` diagnostic,
the post-loop ZDF diagnostics, `ice_thd_temp`, and exact-input temperatures. On five categories of
10 + 5 layers it is bit-identical, under `jit` and eager, to an independent CPython transcription.
The D5 C1D card's growth/melt now runs through the same code, and its year is byte-identical to the
parent on all 18 arms, CPU and AVX (shipped arm `b7a7329b...`, subnormal platform floor unchanged).
No NEMO array was read; the projection of the shared master record is preregistered
(`docs/ice/fidelity/si3_r16_o9_preregister.md`) and no tool is built.

## Build cited

`ORCA2_SAS_ICE_O7_REF`, flags (`ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm:27`):
`%FCFLAGS             -fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize`.
Its `icethd_zdf_bl99.f90`, `icethd_dh.f90`, `icethd.f90`, `icethd_zdf.f90` are byte-identical across
`ORCA2_SAS_ICE_{O1,O2,O5,O7}_{REF,REC}` and to the C1D build's (which differs only by its observer and by
lacking `-fno-tree-vectorize`), so the D5 code was already a transcription of the same statements.

## Search before building

Searched `packages/`, `src/`, `scripts/validate/`, `tests/` for `zdf_bl99`, `_dh_step`, `ice_thd_temp`,
`vremap`, `snw_ent`, `snwblow`, `nlay`, `qcn_ice`, `hfx_`, `maxsd`. Found: the D5 C1D column
(`packages/ice/legoesm/ice/bitz_lipscomb.py`: 3 + 3 layers hard-coded, `nn_icesal = 2`, no DH fluxes, no
`ice_thd_temp`); the shared Thomas solve (`thomas_solve(..., "nemo_unnormalised")`), `nemo_source_round`
and the glibc `transcendentals` arm, reused unchanged; the O7 surface module's private `_fmin`/`_fmax`
(left alone, private-import ratchet). Generalised in place; no new module.

## Statements transcribed

Stage sequence, per category (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:197`):
`CALL ice_thd_zdf                      ! --- Ice-Snow temperature --- !`,
`:199` `IF( ln_icedH )    CALL ice_thd_dh                       ! --- Growing/Melting --- !`,
`:201` `CALL ice_thd_temp                     ! --- Temperature update --- !`; the view stops before
`:203` `CALL ice_thd_sal                      ! --- Ice salinity --- !` (O10).

- ZDF (`icethd_zdf_bl99.f90` 148-879, `np_cnd_OFF`): model `packages/ice/legoesm/ice/bitz_lipscomb.py:532`
  (`def _si3_zdf_bl99_core(`, Picard loop and n-row system) and `:921` (`def si3_ice_thd_zdf_bl99(`,
  exact-input entry with `zq_ini`, interface fluxes, `ice_var_enthalpy`, `hfx_err_dif`, `hfx_dif`,
  `cnd_ice`, `t1_ice`, `t_si`). `rcnd_s` is the deck's `rn_cnd_s = 0.31`, passed explicitly.
- DH (`icethd_dh.f90` 102-541, `nn_icesal = 4`): `packages/ice/legoesm/ice/bitz_lipscomb.py:1249`
  (`def _si3_ice_thd_dh_core(`), `:1522` (`def si3_ice_thd_dh(`), remaps `:1214`
  (`def _cumulative_remap(`, `snw_ent` and `ice_var_vremap`, including the salt remap
  `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90:522`:
  `IF( nn_icesal == 4 )   CALL ice_var_vremap( zh_i_old, zs_i_old, sz_i_1d(ji,:) )`).
- TEMP: `packages/ice/legoesm/ice/bitz_lipscomb.py:1506` (`def si3_ice_thd_temp(`); composition
  `:1556` (`def si3_ice_thd_category_stages(`).
- Five categories: the category body has no cross-category coupling in these three routines (`at_i` enters
  as data), so the stage function runs per gathered category; the test drives `jpl = 5` with
  `npti = 7, 3, 11, 1, 5`.

Refused rather than transcribed (not executed on this deck): `nlay_i = 1` arms, `np_cnd_ON`/`EMU`,
`ln_virtual_itd`, `nn_qtrice = 1`, `nn_snwfra /= 2`, U64 conductivity, `nn_icesal = 1, 3`.

## Object code (MIN/MAX/SIGN sites)

REVISED after the codex HOLD: the claim below that roles were read "at every executing site" rested on an
approximate scratch classifier. The exhaustive version is now committed and pinned; it is summarised in the
preregistration, section 5 (`docs/ice/fidelity/si3_r16_o9_preregister.md`). It confirms the zero-operand roles
at all 33 deck-path statements and narrows the rest: at variable/variable sites the operand each register
holds is a hand reading (PLAUSIBLE), relevant only to opposite-sign zero ties and NaN. The text of this section
is kept as first written.

The record objects, a plain recompile and a `-g` recompile of the same ppsrc with the record flags have
identical instruction streams (`icethd_zdf_bl99.o icethd_dh.o icethd.o icevar.o`). GCC 15 emits
`MAX`/`MIN` as `maxsd`/`minsd` with per-site operand roles; which operand is returned on a signed-zero tie
or NaN was read from the `-g` disassembly at every executing site and transcribed with two helpers
(`packages/ice/legoesm/ice/bitz_lipscomb.py:439`, `def _maxsd(dst, src):`; `:444`, `def _minsd(dst, src):`).
Examples:

- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90:128`: `zq_top = MAX( 0._wp, qml_ice_1d(ji) * rDt_ice )` —
  zero is the source; `+0` on a tie. Same at `:180`, `ze_s(0) = MAX( 0._wp, - qprec_ice_1d(ji) )`, where
  `qprec = +0` really ties (pinned by test).
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:589`:
  `t_i_1d(ji,jk) =  MAX( MIN( t_i_1d(ji,jk), ztmelts ), rt0 - 100._wp )` — the `MIN` keeps `t_i`
  (the first operand), unlike the clamp at `:202`, which keeps its constant.
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90:221`: `zdum = - zq_top / MAX( ze_s(jk), epsi20 )   ! thickness change`
  keeps `ze_s`; `:194` (`zdeltah    = MAX( - evap_ice_1d(ji) * r1_rhos * rDt_ice, - h_s_1d(ji) )   ! amount of snw that sublimates (<0) or deposition (>0)`)
  and `:300` (`zdum               = MAX( - zh_i(jk) , - zevap_rema * r1_rhoi )`) are emitted as a negated
  `MIN` that keeps the thickness (model `packages/ice/legoesm/ice/bitz_lipscomb.py:1332`).
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90:490`: `zdum        = MIN( zdeltah, zh_s(jk) )         ! amount of snw that floods, > 0`
  keeps `zdeltah` (model `packages/ice/legoesm/ice/bitz_lipscomb.py:1471`).
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:955`:
  `t_i_1d(ji,jk) = MIN( t_i_1d(ji,jk), ztmelts + rt0 ) ! Force t_i_1d to be lower than melting point => likely conservation issue`
  keeps `t_i`.

Two sites change roles between unrolled copies and differ only for NaN operands: the convergence
`MAX(zdti_max, ABS(.))` (both operands `>= +0`) and the basal `MIN(0, MAX(zdum, -zh_i))`; both are
registered, not scored. No `SIGN` executes in these routines (grep). `EXP` (13 call sites, radiation) and
`**` (`ice_var_snwblow`, `pow`) run on the glibc arm (DECISION 162); `ice_thd_temp`'s root is `sqrtsd`:
`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:285`:
`zccc          = SQRT( MAX( zbbb * zbbb - 4._wp * rcpi * rLfus * ztmelts, 0._wp ) )`.
Evidence: superseded scratch `round16/objaudit/`; committed audit `scripts/validate/ocean_fidelity/testcases/nemo_si3_o9_minmax_audit.py`.

## What XLA rewrites (the S1 lesson, measured)

- Constant division: under `jit`, `(2x)/0.1` differs from IEEE division in 35,410 of 100,000 draws; a
  materialized divisor gives 0. Executing sites: `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:191`
  (`isnow_comb(ji) = h_s_1d(ji) / zh_min`) and `:857`
  (`cnd_ice_1d(ji) = 2._wp * ztcond_i(ji,0) / zhi_ssl ! cnd_ice is capped by: cond_i/zhi_ssl`); model
  `packages/ice/legoesm/ice/bitz_lipscomb.py:449` (`def _div_const(x, c):`), used at `:590` and `:993`.
- Zero folding: XLA folds `0 + x` to `x`, so a `-0` added to a zeroed `dh_*` stays `-0`; the seed is
  materialized at `packages/ice/legoesm/ice/bitz_lipscomb.py:1291`. Before the fix `dh_i_sub` differed in
  5 cells and `cnd_ice` in 1 (`round16/o9_prefix_mismatches.log`).
- Every multiply-add, including inside each flux accumulation, goes through `nemo_source_round`.

## Tests

`tests/ice/unit/test_si3_o9_zdf_dh_temp.py`: `14 passed` (the final run is in the controls below).

- Bitwise against the independent transcription, five categories x (10 + 5) layers, every ZDF/DH/TEMP leaf
  (`tests/ice/unit/test_si3_o9_zdf_dh_temp.py:652`), with these classes all present: melting surface,
  snow and snow-free, internal ice melt, basal growth and melt, flooding, ice melted away, deposition,
  warm snow layers. The oracle's constants are the build's literals, not the model's.
- Eager (`jax.disable_jit`) == `jit`, every bit (`:674`).
- Inexact pins: `rn_cnd_s` `0x1.3d70a3d70a3d7p-2`, `r1_nlay_i` `0x1.999999999999ap-4`, `r1_nlay_s`,
  `rn_snwblow`, plus a digest of every output leaf (`:682`).
- Production-path plants, all fire (`:697`): `rn_cnd_s` +1 ULP, `rn_sinew` +1 ULP, `rn_kappa_i` +1 ULP,
  native transcendentals (on 64 sunlit snow-free columns; it is inert on the main batch, because a 1-ULP
  `exp` change in absorbed shortwave is below one ULP of ~260 K), reversed MIN/MAX tie roles,
  `sst := t_bo - rt0`.
- Signed-zero site `MAX(0, -qprec)` pinned through `hfx_spr` (`:727`).
- The C1D card's DH calls the shared core (`:742`), and on the D5 year a 1-ULP plant in the core's `h_i`
  changes exactly the shipped arm (`b7a7329b...` -> `0843195e...`), reverted after.
- Gradients finite; central difference on `qns_ice` within 1e-5 (`:759`).
- No defaults: every missing scalar or array and every non-production selector is refused (`:778`).

## Controls (parent `abd7262c11` vs current)

Parent from a clean detached worktree; both trees fp64, CPU, `jit`.

| caller | parent | current |
|---|---|---|
| D5 C1D year, 18 arm hashes, `XLA_FLAGS=""` | shipped arm `b7a7329b...`, platform floor | all 18 identical (clean tree `86cf00613e`) |
| D5 C1D year, 18 arm hashes, `--xla_cpu_max_isa=AVX` | shipped arm `b7a7329b...` (CPU == AVX) | all 18 identical |
| `tests/ice/unit` | 408 passed, 1 skipped, 1 xfailed | 422 passed, 1 skipped, 1 xfailed (+14 O9) |
| SI3 fidelity tests (11 files) | 1 failed, 143 passed | 1 failed, 143 passed after the fix below |
| CI ratchets (6 files, incl. private imports) | 11 failed, 12047 passed | same 11 failed, 12049 passed |
| default citation gate + all 13 SI3 receipts | pass | pass (receipt gate 43 citations, planted shift fires) |

The fidelity failure on both trees is the pre-existing scalarmath v2 "A MY_SRC is not verbatim". The
first current run had a second failure: a D5 test calls `_radiation` directly and the generalised
signature requires the extinction coefficients; the test now passes the C1D card's values
(`d5probe_current_fixed.log`: `22 passed`). The `bitz_lipscomb.py` inline-coefficient red is the
pre-existing one on both trees.

## Findings (one line each, not acted on)

- `NEMO_SI3_CONSTANTS_CONFIG.k_snow = 0.5` is the C1D card's `rn_cnd_s`, not a NEMO constant; the ORCA2
  value is 0.31. O9 passes `rn_cnd_s` explicitly; the shared config was not changed.
- C1D's flooding water enthalpy uses `t_bo - rt0` where NEMO uses `sst_1d`
  (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90:463`, `zEw            = rcp * sst_1d(ji)`); kept,
  so the D5 year is unchanged (`packages/ice/legoesm/ice/bitz_lipscomb.py:1610`). PLAUSIBLE that C1D's
  `sst_m` equals `t_bo - rt0`; not measured.
- `ice_thd_1d2d( jl, 1 )` (gather and J/m2 -> J/m3) sits upstream of O9's entry and no rung owns it.
- The preregistered Picard frame needed an observer inside NEMO's `ji` loop, against checklist item 1;
  redesigned (whole arrays after each pass), see the review fixes below.

## CONFIRMED vs PLAUSIBLE

- CONFIRMED: bit-identity to the independent transcription on every listed branch class, `jit` and eager;
  the MIN/MAX roles at every executing site, read from the record build's instruction stream; the two XLA
  rewrites and their fixes; D5 year byte-identical (18 arms, CPU and AVX) with the core proven on its path.
- PLAUSIBLE: bit-exactness against an ORCA2 NEMO record (none read; P1-P5 of the preregistration).

## CHOICES

- O9 projection from the one master record: ASKED (DECISION 173).
- Picard frame steps and the 100-step scoring window: UNASKED -> DECISION_NEEDED.
- MIN/MAX/SIGN audited at object level and transcribed from the operand roles: ASKED (the round brief asked for
  the object-level audit); the NaN-only behaviour that comes with it is not scored.
- Materialized zero seeds and true constant divides: UNASKED (forced by bit-identity); offered for revert.
- C1D DH routed through the shared core with `sst := t_bo - rt0`, `rn_snwblow = 0.66`, `rn_sinew = 0.75`,
  `nn_icesal = 2` (its existing values): UNASKED, mechanical; offered for revert.
- 58 historical receipt citations into `bitz_lipscomb.py` re-anchored to the same statements: UNASKED, mechanical.
- No configuration default, production config or deck changed.

## Review fixes (codex HOLD; Argon LAND)

1. Thomas forward sweep: NEMO's two statements
   `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:537-538`:
   `zdiagbis(jm) = ztrid   (jm,2) - ztrid(jm,1) * ztrid   (jm-1,3) / zdiagbis(jm-1)` and
   `zindtbis(jm) = zindterm(jm  ) - ztrid(jm,1) * zindtbis(jm-1  ) / zdiagbis(jm-1)` are now materialised
   with `nemo_source_round` around the product and the quotient, in the NEMO-only branch of the shared solver.
   No other caller selects that branch. Control, parent `9044f7a7f3` vs current, fp64 CPU: for all seven other
   entries of the solver (default arm, shared-factor, batched, legacy batched, cyclic reduction, periodic,
   unrounded NEMO arm) the compiled program with debug metadata removed, the output bytes and the gradient
   bytes are identical; only the NEMO arm's program changed, and its outputs and gradients are identical on the
   257-column sample. Every test file importing the solver: same counts on both trees (8 files all pass; the
   land-audit file has the same 3 failures on both, run one class per process because one long process exceeds
   the machine's memory-map limit). D5 year: all 18 arms byte-identical to the parent, CPU and AVX.
2. Picard coverage: the ZDF exit now carries each cell's pass count (`n_picard`, NEMO's per-cell
   `tice_cvgstp_1d`), compared bitwise with the independent transcription's count, and the coverage test counts
   cells with more than one pass. Every one of the 27 test cells takes 3-9 passes. Planting a constant count of 1
   turns the bitwise test red (7 cells), reverted. O9 tests: `14 passed`.
3. Picard frame redesigned: whole arrays after each pass's `ji` loop, nothing inside NEMO's inner loops;
   what it can and cannot localise is stated in the preregistration, section 2.
4. Object audit committed and pinned (`tests/ocean/fidelity/test_nemo_si3_o9_minmax_audit.py`: `10 passed`,
   including regeneration from the record build and four plants that turn it red); the claim is narrowed as above.
5. CHOICES corrected: the object-level audit was asked.

Controls after the fixes (current tree; parent `9044f7a7f3` where compared): `tests/ice/unit` `422 passed, 1 skipped,
1 xfailed`; SI3 fidelity files plus the new audit test `152 passed`, 2 failed: the pre-existing scalarmath v2 one and
the O7 preflight timing out under 16 workers (alone: `1 passed`); six CI ratchets: the same 8 failures on both trees;
default citation gate passes; all 14 SI3 receipts pass the per-receipt gate, and a planted shift of the new NEMO
citation turns it red. 62 citation-map entries and their receipt citations were re-anchored to the same statements
after the line shifts (mechanical, each re-checked by the gate). Review of these fixes: UNREVIEWED until the operator
runs codex and the second reviewer.

Choices in the fixes: barriers in the NEMO-only branch, per-cell pass count on the public ZDF outputs, the redesigned
whole-array Picard frame, the committed audit and the narrowed claim: ASKED (items 1-5 of the review). The audit's
location next to the other SI3 validators and the re-anchoring: UNASKED, mechanical; offered for revert.

Review: `codex review: operator`.
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round16/`.
