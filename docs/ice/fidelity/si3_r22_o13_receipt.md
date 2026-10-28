# SI3 Ladder Round 22 (`si3_r22`) — O13 HFN ITD-remap receipt

- **Date:** 2026-10-28
- **Parent:** `87a1c8843803`
- **Preregistration:** `docs/ice/fidelity/si3_r22_o13_score_preregister.md:1-65`, committed as `f8e3f05d8a` before any Kind 11/12 record was read.
- **Verdict:** **`PASS_O13_0_ULP`**. The production `si3_ice_itd_rem` matches every field of all 100 admitted Kind 12 EXIT records: **`0 / 1,145,653,900` unequal**, maximum absolute difference `0.0`, maximum ULP `0`, first non-bit `None`.
- **Score artifact:** `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round22/o13_master_score.json`, 22,118 bytes, SHA-256 `4e95174e554347a3b5dad598af506b42bfe2295f03b1bf0f5425940697cf6cbe`.
- **independent review: operator (Codex)**

## 1. Source and deck resolution

The compiled record build is `ORCA2_SAS_ICE_O8O16_REF`. The executing source, not a nearby uncompiled arm, says:

> `at_i(...) = SUM( a_i(...,:), dim=3 )`
>
> `CALL itd_glinear(...)`
>
> `CALL itd_shiftice ( jdonor(...), zdaice(...), zdvice(...) )`

Those statements and the boundary/remap eligibility logic are in `ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/iceitd.f90:118-365`; the linear reconstruction is `:368-429`; the sequential extensive transfer, roundoff call, `rn_amax_2d` cap, and final thickness/temperature update are `:432-597`. The roundoff callee clamps negative ice/snow/age/energy and pond values, conditionally clamps lid volume, and selects layered `szv_i` for `nn_icesal == 4` in `ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icevar.f90:856-894`.

Resolved production values are `jpl=5`, `nlay_i=10`, `nlay_s=5`, `nn_icesal=4`, `ln_pnd_LEV=True`, `ln_pnd_lids=True`, HFN `rn_himean=2.0`, `rn_himax=99.0`, and the recorded spatial `rn_amax_2d`. There is no fallback deck: the production wrapper rejects a missing `config.si3_orca2`, missing `rn_amax_2d`, mismatched category count, or mismatched spatial shape (`packages/ice/legoesm/ice/sea_ice.py:2847-2868`). The remapper also has no defaults for `epsi10`, `epsi20`, or `rt0`; production passes all three from the run's explicit `SI3ORCA2StateConfig.ice_constants` (`packages/ice/legoesm/ice/sea_ice.py:3254-3286`).

## 2. One production implementation

The canonical implementation is `si3_ice_itd_rem` in `packages/ice/legoesm/ice/itd.py:823-1079`. It transcribes, in source order:

1. sequential category summation and remap-cell eligibility;
2. displaced HFN boundaries and first-category melt-area loss, including the object-code fused line-280/282 `MINSD` order;
3. `itd_glinear` reconstruction (`packages/ice/legoesm/ice/itd.py:790-819`);
4. boundary integrations for area and volume;
5. sequential donor transfers for area, ice/snow volume, age, surface-temperature content, pond area/volume/lid, all snow and ice energy layers, and layered or bulk salt;
6. NEMO roundoff, `rn_amax_2d`, final `h_i`/`t_su`, and the pre-transfer area conservation target.

Every NEMO source operation is materialized with `nemo_source_round`; category sums use the pre-existing sequential `fortran_sum_categories`. The production sea-ice step selects this core for the existing `n_categories > 1`, `itd_remap="lipscomb2001"`, `ponds.scheme="si3_lev"` configuration and maps the generic carrier into/out of the same extensive interface (`packages/ice/legoesm/ice/sea_ice.py:3235-3307`). Other callers retain their existing generic remap branch. The AST duplicate-arithmetic guard now treats the canonical remapper as the sole O13 kernel, catches a planted copied O13 window, and there is exactly one `si3_ice_itd_rem` definition under `packages/`.

The production-path test drives basal melt on five populated HFN categories. The exact remapper moves category area, and replacing the call by a no-op changes the returned production concentration bits; the sentinel removal plant also fires (`tests/ice/unit/test_si3_o13_itd_remap.py:294-409`).

## 3. Object-code MIN/MAX/SIGN roles

The committed audit recompiles the record ppsrc with the recorded flags plus debug lines, requires an identical instruction stream, and records object SHA-256 `2642ba5ec85fc283fc288d25ade003a837a3fe6f8cccc0f32b7f3bb3d89caa71` and ppsrc SHA-256 `22baac67260037d4cde180d48ab1938e730b57f492c7d01eb6ade93c9403e7ba` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o13_minmax_audit.py:1-89`).

It finds eight emitted sites at source lines 222, 282, 289, 298, 326, 327, 331, and 334, plus source line 280 fused into the two `minsd` instructions attributed to line 282; no `SIGN` or `SQRT` site executes. The important survivor roles are:

- line 222 `MAX(hi_max, expression)`: emitted source is `hi_max` (first Fortran operand);
- fused lines 280/282: `minsd(dst=hi_max(1), src=hR(ii,1))`, then `minsd(dst=<that>, src=-zdh0)`; both source operands survive bit-changing signed-zero ties in the executing helper (`packages/ice/legoesm/ice/itd.py:1088-1092`);
- line 298 `MIN(zdh0, hi_max)`: emitted source is `zdh0` (first operand);
- line 331 `MIN(hi_max, hR)`: emitted source is `hi_max` (first operand);
- lines 289, 326, 327, and 334 use the second-operand survivor represented by `_nemo_minsd`/`_nemo_maxsd`.

The report regeneration, site signatures, census, operand roles, and five planted report corruptions are tested in `tests/ocean/fidelity/test_nemo_si3_o13_minmax_audit.py:65-109`.

## 4. Tests and conservation

The independent scalar NumPy oracle imports no legoESM model code or constants. Its mixed up/down boundary crossings and first-bin melt match true eager execution (the whole call under `jax.disable_jit`) and `jax.jit` (`tests/ice/unit/test_si3_o13_itd_remap.py:242-257`). Runtime JIT arguments, finite reverse gradients, a central-difference directional derivative, vanished-ice state, and a zero category are checked in `tests/ice/unit/test_si3_o13_itd_remap.py:260-291`.

The preregistered P4 gate checks `a_i`, `v_i`, `v_s`, `sv_i`, `oa_i`, `a_ip`, `v_ip`, `v_il`, `e_s`, `e_i`, and `szv_i`. Area is checked against the independently carried post-thin-melt/cap target; every other field is checked against its ENTRY category sum. It passes the synthetic crossings and all 100 admitted Kind-11 frames. Dropping one nonzero category receipt turns every one of the eleven field checks red, and the scorer-level dropped-`v_i` plant refuses.

Focused results:

| Gate | Result |
|---|---|
| O13 scalar oracle / explicit constants / fused ties / eager / JIT / AD / edges / live production plant | `5 passed` |
| O13 object audit and five plants | `8 passed` |
| P4 conservation and eleven dropped-transfer controls | `1 passed` |
| O8–O13 production-call/AST duplicate guard | `1 passed` |
| modified O12 configuration guards | `2 passed` |

## 5. Master score and controls

The r21 scorer was extended in place. `_score_o13_step` reads verified Kind 11/12 pairs, calls the production function with arrays as JIT arguments, checks P4 conservation, and compares every one of the 96 EXIT fields (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:951-981`). `score_o13` is the all-step mode of that same block, not a reimplementation (`:984-1036`), and the integrated O8–O13 loop calls the same helper (`:1653-1657`).

Admission pins verified during the final run:

| Artifact | SHA-256 / result |
|---|---|
| run manifest | `bfb0b71d1500b7bc91c68a9e39c8af7a654b139258c0635ad5b1f25284b31fd0` |
| resource usage | `b30a5febe9248e5008999cb419953dad40c78658542a3c7221d301825270967c` |
| records inventory | `fbd3cb4d3450f428d8444dcb145bec3c1a832554e3e7e05d7f5b6a159daadd85` |
| O13 projection (200 Kind 11/12 records) | `a94d126ddaee3194a053ef437d762707dff93045287392d4e0586aaee995a4aa` |
| O14 projection, unchanged | `8130e5ac516cf8c1549badbab2caf90cf5cce03f3167c317a10532b836572cd6` |
| all 100 steps, all 96 EXIT fields | `PASS_0_ULP`; `0 / 1,145,653,900` unequal |
| P4 conservation | `PASS_CONSERVATION`; 100 frames, 11 inventories |

The one-ULP EXIT plant fails with exactly `1 / 11,456,539` unequal at `a_i[0,0,0]`, max ULP `1`. The model mutation (one ULP in `hi_max[1]`) fails through the shared scorer. The dropped-transfer plant refuses at the P4 `v_i` conservation gate before a score can be accepted.

Current D5 `C1D_OMIP_L3` was re-run for all 8,760 JIT steps through `run_year_rescore` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:806-1883`) and returns the pinned parent hash `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`. The integrated admitted-frame test also keeps O8–O12 at zero ULP, and O14's projection digest is unchanged.

## 6. Choices and epistemic status

### ASKED choices

- Implement and score only O13; no configuration value, bar, or master record was changed.
- Extend the r21 scorer and use the admitted Kind 11 -> Kind 12 projection.
- Preserve the existing approximate generic Lipscomb behavior for non-SI3 callers while routing the exact SI3-LEV selector to the single O13 core.

### UNASKED choices

- None affecting physics or configuration. `rn_amax_2d` is an explicit runtime boundary input because it is spatial and is already present in the admitted ENTRY record; it is not a new default or tuning parameter.

### CONFIRMED

- All 100 admitted steps and all EXIT fields are bit-exact at 0 ULP.
- All 100 admitted steps pass the eleven-inventory P4 conservation gate.
- Boundary crossings, thin-ice loss, area/volume transfer, snow and ice energy, layered salt, pond area/volume/lid, roundoff, and the area cap execute in the scored production function.
- Eager equals JIT; gradients and central differences are finite; vanished ice and a zero category stay finite.
- Parent and current D5 year hashes are identical.

### PLAUSIBLE

- None used for the landing verdict. The generic `DynamicSeaIceState` wrapper represents bulk salinity as one extensive layer because that carrier does not expose ten SI3 layers; the scored O13 production boundary receives and verifies all ten layers directly.

## 7. Landing statement

O13 is implemented, production-wired, and scored at the bar. No acquisition or configuration decision is needed. Receipt citations, missing-default behavior, canonical-kernel scope, field inventory, and every cited-file digest are machine-gated.
