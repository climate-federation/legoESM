# SI3 Ladder Round 23 (`si3_r23`) — O14 model-preparation and scoring receipt

Date: 2026-10-28. Parent: `87a1c8843803`. Scope: O14 only. Preregistration:
`docs/ice/fidelity/si3_r23_o14_score_preregister.md:1-52`, committed as
`ce859f121f` before any master-record field was scored.

**Verdict: `PASS_O14_0_ULP`; MODEL PREPARATION + SCORING COMPLETE.** The single
production `si3_ice_thd_do` implementation matches all 96 kind-13 EXIT fields
for all 100 admitted steps: `0 / 1,145,653,900` unequal binary64 elements,
maximum absolute difference `0.0`, maximum ULP difference `0`, and first
non-bit `None`. The result includes signed zero, all five categories, all ten
ice layers, new-category selection, heat/freshwater/salt fluxes, and every
pass-through field.

independent review: operator (Claude)

## 1. Exact source and executing identity

The retained record build is
`round18/refused/builds_20261026T210237Z_2/ORCA2_SAS_ICE_O8O16_REF`.
Its `nemo.exe` SHA-256 is
`d3a805bcbc358c0eff1c91599b9547017de85e2a89b2cb5ee4a5380496e1289f`;
its preprocessed `icethd_do.f90` SHA-256 is
`bdaabab890d5cd4d886d315901c793c70a4ef353ff2bfb7bce6e0cdf014c7c8e`;
and `BLD/obj/icethd_do.o` SHA-256 is
`bee1eae9780e89a47ea68ac53bef028c497b105f895414da6500200e6004e5d9`.

The following manifest is parsed by the receipt-integrity test. Every value
was computed from the named file with `sha256sum`:

<!-- BEGIN REHASHABLE_SHA256 -->
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/refused/builds_20261026T210237Z_2/ORCA2_SAS_ICE_O8O16_REF/BLD/bin/nemo.exe` SHA-256 `d3a805bcbc358c0eff1c91599b9547017de85e2a89b2cb5ee4a5380496e1289f`
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/refused/builds_20261026T210237Z_2/ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90` SHA-256 `bdaabab890d5cd4d886d315901c793c70a4ef353ff2bfb7bce6e0cdf014c7c8e`
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/refused/builds_20261026T210237Z_2/ORCA2_SAS_ICE_O8O16_REF/BLD/obj/icethd_do.o` SHA-256 `bee1eae9780e89a47ea68ac53bef028c497b105f895414da6500200e6004e5d9`
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master/manifest/run_manifest.json` SHA-256 `bfb0b71d1500b7bc91c68a9e39c8af7a654b139258c0635ad5b1f25284b31fd0`
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round23/o14_master_score.json` SHA-256 `a1ca351c714f791415f757d916544879814b9831dfa9546f4cee4da10f199b0b`
<!-- END REHASHABLE_SHA256 -->

The complete executing statement blocks were transcribed from the compiled
record build, not from a different checkout:

- `ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90:177-219`:
  `at_i(...) = SUM( a_i(...,:), dim=3 )`, followed by the `qlead < 0` gather
  and the extensive-to-intensive conversions
  `e_i_2d = e_i_2d / v_i_2d * REAL(nlay_i)` and the same operation for
  `szv_i_2d`.
- `ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90:225-273`:
  `CASE ( 2 , 4 ) ; zs_newice = rn_sinew * sss_1d`, the new-ice heat content,
  `zfmdt = -qlead_1d / zdE`, `zv_newice = -zfmdt * r1_rhoi`, and all four
  heat/freshwater/salt flux updates.
- `ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90:275-320`:
  frazil-bottom allocation, concentration capping and residual retention,
  interval selection `zh_newice > hi_max(jl-1) .AND. ... <= hi_max(jl)`, and
  new-ice heat/salt mixing using the saved pre-growth area `za_b(jl)`.
- `ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90:322-375`:
  bottom/residual distribution, `sv_i` update, `ice_var_vremap` for heat and
  salt, unit restoration, and scattering `a_i`, `v_i`, `sv_i`, `e_i`, and
  `szv_i`. The absence of an `at_i` scatter is executing behavior: its EXIT is
  the initial `SUM(a_i)`, not the updated gathered `at_i_1d`.

The admitted resolved identity is unchanged: `ln_icedO=T`, `ln_leadhfx=T`,
`rn_hinew=0.1`, `ln_frazil=F`, `nn_icesal=4`, `rn_sinew=0.75`, `jpl=5`,
`nlay_i=10`, and `rDt_ice=21600 s`. No default supplies a scoring selector.

### Object-code MIN/MAX/SIGN audit

Disassembly of the exact object above established the operand/tie roles used
at each executing call site:

- Source line 246 temperature `MIN`: `dst=-epsi10`,
  `src=t_bo-rt0`; `src` survives equality because `dst` is retained only when
  strictly smaller.
- Source line 246 latent `MAX`: `dst=1-ztmelts/capped_t_bo`, `src=+0`; source
  `+0` survives a signed-zero tie.
- Source line 291 concentration-cap `MAX` and lines 294-295 residual `MAX`:
  candidate is `dst`, `+0` is `src`; source `+0` survives ties at each site.
- Source line 315 mixing-divisor `MAX`: `dst=epsi20`, `src=v_i`; source volume
  survives equality. Source line 337's identical denominator is optimized out
  inside the already-proven `at_i >= epsi20` arm.
- No executing `SIGN` instruction or call exists in this routine for the
  admitted deck.

## 2. One production implementation and production wiring

`packages/ice/legoesm/ice/bitz_lipscomb.py:1811-2055` is the only O14
implementation. It applies NEMO statement barriers, compiled operand roles,
category interval selection, residual/bottom remapping, and all ten changed
outputs. The former inline open-water arithmetic was deleted.

`packages/ice/legoesm/ice/bitz_lipscomb.py:2204-2249` wires the same function
into `si3_column_step_arrays`, with `rn_hinew`, `ln_frazil`, and `rn_sinew`
read from the resolved namelist dictionary. The one-category C1D adapter
constructs extensive inputs, calls the production kernel once, and extracts
only the C1D prognostic fields already present in its state schema.

The production test spies on that call and then replaces it with a bypass;
removing O14 changes the returned production ice thickness, so the M4 control
can fail (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:261-433`).
The AST guard continues to permit only the canonical implementation and its
updated planted duplicate fires.

## 3. Independent model tests

The scalar oracle in `tests/ice/unit/test_si3_thd_do.py:82-143` is standalone
NumPy: it imports no production helper or production constant and independently
implements category choice, fluxes, capping, bulk salt, and cumulative heat/salt
remapping.

`tests/ice/unit/test_si3_thd_do.py:157-295` checks every returned state/flux
field against that oracle, passes arrays as runtime JIT arguments, requires
real eager execution under `jax.disable_jit()` to equal JIT bit-for-bit, checks
finite vanished-ice and zero-category states, compares `jax.grad` for `qlead`,
`a_i`, `v_i`, `sv_i`, `e_i`, and `szv_i` with central differences away from
switches, refuses missing/invalid inputs, and pins the audited MIN/MAX
signed-zero roles with bit-distinct tied operands. The vanished-ice adapter
test also differentiates concentration and thickness; restoring the unsafe
denominator makes this plant fail with a NaN gradient. Result: `10 passed`.

## 4. Admitted-master scoring

The existing scorer was extended rather than reimplemented. Its cached JIT
kernel calls the production function with resolved selectors
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:875-882`).
`score_o14` verifies admission, reads kind 12 and kind 13 through the existing
verified reader, passes ENTRY arrays as JIT arguments, compares all 96 EXIT
fields bitwise, and fails with the first `(kt, category, cell, field)`
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1016-1119`).
Admission now also compares the preregistered `ln_icedO=T` and `ln_leadhfx=T`
selectors with the resolved record namelist and compares the kind-12/13
`rn_hinew` scalar header with that namelist; all three mismatch plants fire.

Full report:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round23/o14_master_score.json`,
17,725 bytes, SHA-256
`a1ca351c714f791415f757d916544879814b9831dfa9546f4cee4da10f199b0b`.

| EXIT group | Elements | Unequal | max ULP |
|---|---:|---:|---:|
| `a_i`, `v_i`, `sv_i` | 39,960,000 | 0 | 0 |
| `e_i`, `szv_i` | 266,400,000 | 0 | 0 |
| `at_i`, `sfx_opw`, `wfx_opw`, `hfx_thd`, `hfx_opw` | 13,320,000 | 0 | 0 |
| 86 pass-through fields | 825,973,900 | 0 | 0 |
| **All 96 fields** | **1,145,653,900** | **0** | **0** |

The direct scorer test covers representative `kt=(1,2,100)`, a one-ULP
`v_i` EXIT mutation, and a production-model bypass mutation
(`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:142-160`). Both
plants exit nonzero and name O14 as non-bit-exact.

## 5. Caller controls and gates

The committed D5 rescore probe (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:806-1883`)
ran all 8,760 C1D steps under CPU JIT. The shipped
`r6_step1_plus_step2_snow_update` arm retained the parent/pinned full-year
SHA-256 exactly:
`b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`.
All 621,960 stage-field rows were evaluated. Thus the production O14 wiring
changes no D5 output bit; other O8-O12 production paths and the production-call
test also pass.

Test evidence on clean review-fix commit `2187e6a234`:

- O14 unit, O8 edge, and O8-O14 scorer/production suite: `40 passed in
  874.85s`; the focused gradient, selector, and digest-integrity subset:
  `14 passed in 26.29s`.
- Full 100-step O14 score: `PASS_O14_0_ULP`.
- Safe-denominator forward proof: the full score remained 0 ULP, and the
  8,760-step D5 run retained the registered hash above across 621,960 rows.
- Receipt digest integrity: all five manifest files re-hash exactly, and the
  planted digest mismatch fails.
- This receipt's citation gate: `PASS`, 14 citations, zero unmapped/failing
  citations, and every self-test fired.
- Full default receipt-citation gate: `39 passed in 9.61s`.
- Private-import ratchet with every namespace root present: `6 passed in
  5.01s`.

## 6. CHOICES and epistemic status

**CHOICES — ASKED:** use the admitted O14 kind 12 -> 13 projection; transcribe
the compiled record build; preserve source order and object-code tie roles;
wire one production implementation; require 0 ULP; retain the D5 hash; add
independent eager/JIT/gradient/edge tests and firing plants. All were applied
unchanged.

**CHOICES — UNASKED:** none. No configuration, threshold, category boundary,
constant, cadence, state schema, tolerance, or scoring statistic changed.

**CONFIRMED:** exact retained source/object provenance; every executing O14
statement for this deck; every MIN/MAX operand role and absence of SIGN; single
production implementation and live call site; full 100-step all-field 0-ULP
score; one-ULP/model-bypass/production-removal/duplicate plants; eager/JIT and
finite-gradient behavior; unchanged D5 shipped hash.

**PLAUSIBLE:** none used for the verdict. Behavior outside `nn_icesal in
{2,4}`, or with `ln_frazil=T`, is outside this rung and is not claimed.
