# GYRE high-resolution round 16: online term attribution

Date: 2026-10-19

Incoming tip: `06cb10a3d`.  Attribution instrument and scalar-twin
preregistration: `ac511929db74fae72de2953b411da94941ae508a`.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round16/`.

## Verdict

**CONFIRMED:** the matched vector-NEMO/legoESM budgets are admissible and their
cross-provider component difference closes to `1.5558301175166989e-16 m/s`,
well below the registered `1e-12 m/s` algebraic bar.

**HELD:** no physics owner is named.  The preregistered raw winner is
`vertical_mixing`, but that row and `external_mode_correction` use different
intermediate boundaries in the two models and their signed model differences
cancel by `0.9997436040502027`.  Once the only source-aligned pair is formed,
`total_vorticity_coriolis` is the largest candidate.  It cannot yet be judged
against the requested NEMO-vs-NEMO per-term envelope because the scalar-math
NEMO twin has daily states but no online term budget.

No shared physics, card, namelist, JAX kernel, or existing evidence artifact
changed in this round.

## Finding 1 — both acquired providers pass

Disposition: admit the exact days 30--180, native surface-V,
western-boundary comparison.

The admission gate reopens both digest-bound contracts, arrays, masks,
cadences, passivity artifacts, and source manifests.  It accepts 151 records,
24 one-hour increments per record, identical 82-by-122 masks with 9,480 wet
V points, and the exact `r4_smag` deck.  Provider-local maximum closure is
`1.5151725360484924e-16 m/s` for legoESM and
`1.5265566588595902e-16 m/s` for NEMO.  The cross-provider difference of the
first seven rows closes the independently recorded total at
`1.5558301175166989e-16 m/s` maximum and
`5.2358106010775965e-18 m/s` RMS.

Evidence:

- `budget_admission.json`, SHA-256
  `aecfa74d7868179498ea70fb6f217809d56a743b4cd1ffd2209fe740fa2599c0`.
- `budget_attribution.json`, SHA-256
  `0f3bb96cafd1637eb2d8cab3af6203f8a5e35243f7e28d020ee434263ee6309e`.
- legoESM term archive SHA-256
  `9cfb68af05b735e7582ace1331bd99d470b32e40559a5cc8c147ed746c80a20b`.
- vector-NEMO term archive SHA-256
  `04062f98f34887add7698a5eff1b6a870d42533d79803eaf6303fa088283508e`.

The admission gate initially refused only NEMO's list-directed spellings
`3.5000000000000000` and `1.0000000000000000` because it matched literal
short strings.  The repaired gate parses numeric assignments and still
refuses a self-consistently rehashed `rn_csmc=3.6` plant.  No admitted value
changed.

## Finding 2 — preregistered raw ranking

The registered score is
`abs(mean(legoESM - NEMO_vector)) / RMS(NEMO_vector term)` over all 151 days
and the 1,896 active cells in the western-boundary wet-span bin.  The total is
shown as the independent closure target and is not ranked as a contributor.

| rank | raw row | signed legoESM−NEMO mean (m/s) | NEMO RMS (m/s) | registered score |
|---:|---|---:|---:|---:|
| 1 | `vertical_mixing` | `+4.0294903382489715e-3` | `1.8339635243509341e-2` | `2.1971485717934688e-1` |
| 2 | `external_mode_correction` | `-4.0305237482135756e-3` | `2.7658230605205614e-2` | `1.4572601572911112e-1` |
| 3 | `total_vorticity_coriolis` | `+1.6614925847170393e-6` | `1.7932044299130690e-2` | `9.2654945359330014e-5` |
| 4 | `smagorinsky_lateral_diffusion` | `-5.6551384240899292e-8` | `7.6735215898452412e-4` | `7.3696781300174616e-5` |
| 5 | `surface_stress` | `-4.1026376370591605e-7` | `5.7924648282904560e-3` | `7.0827148004798187e-5` |
| 6 | `pressure_kinetic_energy_gradient` | `-1.4362280207293100e-7` | `2.1060458944633723e-3` | `6.8195475915555290e-5` |
| 7 | `advection` | `-4.1178294929254298e-10` | `7.3667614682629338e-5` | `5.5897418569416569e-6` |
| — | `total_increment` | `+1.7232887144617270e-8` | `2.1369903489130334e-4` | `8.0640921721441888e-5` |

Disposition: retain this table as the exact preregistered numerical result,
but do not interpret rows 1 or 2 as physical ownership.

## Finding 3 — the raw winner is a bookkeeping-boundary mismatch

NEMO captures its pre-ZDF row before removing the prognostic barotropic
velocity.  Its implicit vertical result still lacks that velocity, and the
final external correction restores the target barotropic mean.  legoESM
subtracts the same mean before its solve but re-adds it before returning the
recorded no-stress/full solutions.  Consequently NEMO assigns the temporary
remove/restore pair to the two recorded rows while legoESM does not.

Measured discriminator: the signed vertical difference is
`+0.0040294903382489715 m/s`; external is
`-0.0040305237482135756 m/s`; their source-aligned remainder is only
`-1.0334099646033592e-6 m/s`, a `99.97436040502027%` cancellation.  This is
an instrument partition mismatch, not evidence of a vertical-mixing defect.

Disposition: combine exactly `vertical_mixing + external_mode_correction` for
the diagnostic ranking.  Surface stress remains separate.  No remainder is
invented and the original eight-row closure remains unchanged.

## Finding 4 — source-aligned candidates, not owners

| rank | source-aligned row | signed legoESM−NEMO mean (m/s) | NEMO RMS (m/s) | registered score |
|---:|---|---:|---:|---:|
| 1 | `total_vorticity_coriolis` | `+1.6614925847170393e-6` | `1.7932044299130690e-2` | `9.2654945359330014e-5` |
| 2 | `smagorinsky_lateral_diffusion` | `-5.6551384240899292e-8` | `7.6735215898452412e-4` | `7.3696781300174616e-5` |
| 3 | `surface_stress` | `-4.1026376370591605e-7` | `5.7924648282904560e-3` | `7.0827148004798187e-5` |
| 4 | `pressure_kinetic_energy_gradient` | `-1.4362280207293100e-7` | `2.1060458944633723e-3` | `6.8195475915555290e-5` |
| 5 | `implicit_zdf_plus_external_mode` | `-1.0334099646033592e-6` | `2.0883944158039562e-2` | `4.9483467145047587e-5` |
| 6 | `advection` | `-4.1178294929254298e-10` | `7.3667614682629338e-5` | `5.5897418569416569e-6` |

The inherited state-space twin measurement still establishes why this field
and region were preregistered: at day 30, surface-V candidate divergence was
`5,233.793418002661` times the scalar/vector NEMO divergence; over days
30--180 the western-boundary bin held `0.2753272077371752` of the signed
surface-V excess and growth classified as algebraic.  That state-space result
does not supply a term-by-term arithmetic envelope.

Disposition: `total_vorticity_coriolis` is the first candidate to test, not an
owner to fix.  No shared-path change is warranted from two already-diverged
trajectories without the NEMO scalar/vector term floor.

## Smallest proving test and missing acquisition

The smallest owner test after the arithmetic floor is known is one common
developed entry state.  Capture stage-3 `total_vorticity_coriolis` in legoESM
and scalar/vector NEMO, substitute NEMO's already-formed vorticity operand into
legoESM, and require the row to fall to the scalar/vector envelope without
moving any untouched row.  Only that collapse would identify a transcription
owner suitable for a shared-path fix.

The immediate missing measurement is cheaper and already preregistered:
repeat the passive 180-day NEMO online budget with scalar math.  The operator
tool uses the same writer, `r4_smag` namelist, single rank, days, cadence,
same-binary control, and pointwise closure bar; only
`arch-conda-scalarmath.fcm` replaces `arch-conda.fcm`.

From this clean committed clone, the preflight is:

```bash
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-154073996 && scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget/run.sh --preflight-scalar
```

It passes with `PREFLIGHT_OK`.  The exact operator acquisition is:

```bash
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-154073996 && EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round16/oracle_gyre_r4_smag_budget_180d_scalar RANKS=1 scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget/run.sh --run-scalar
```

After it prints `ACQUISITION_DONE`, apply the registered four-times-twin test:

```bash
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-154073996 && PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget_attribution.py --legoesm-budget-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round14/legoesm_gyre_r4_smag_budget_180d_round14 --nemo-budget-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round15/oracle_gyre_r4_smag_budget_180d_corrected/instrumented --nemo-twin-budget-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round16/oracle_gyre_r4_smag_budget_180d_scalar/instrumented --admission /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round16/budget_admission.json --state-envelope /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round11/divergence_regional_v3/gyre_hires_divergence_report.json --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round16/budget_attribution_with_scalar_twin.json
```

## Controls, tests, and review

Focused acquisition/admission/attribution tests pass 18/18.  Controls refuse a
non-closing active word, mismatched native mask, active non-finite, unsupported
precision flavour, scalar contract presented as vector, altered scalar source
registry, and rehashed wrong namelist value.  The cancelling-boundary plant
proves the raw winner is retained while the source-aligned ranking does not
call it a physics owner.  The scalar envelope plant proves that 10-times twin
clears the preregistered four-times threshold.

The clean scalar acquisition preflight ends in `PREFLIGHT_OK`.  The broader
CPU/x64 lane battery reports `245 passed, 5 deselected` in 486.39 seconds; the
five exclusions are inherited slow controls.  The line-sensitive citation
gate passes all five compiled-source citations with no unmapped or failing
entry.  Shifting the first citation by two lines produces the expected nonzero
exit and `SYMBOL-NOT-AT-LINE` refusal.

Independent-review status is `UNREVIEWED`: the required Claude reviewer was
invoked read-only but its API host failed DNS resolution, and the required
GLM-5.2 reviewer is not available in this environment.  Neither failure is
treated as a review verdict or waiver.  No physics was changed, and the
scientific conclusion remains held pending the scalar term acquisition.

## Inherited pins / OPEN

Inherited unchanged: 31 levels, WS-RK3, split-explicit free surface, TKE/EVD,
Redi tracer mixing, the `r4_smag` momentum deck, 3,600-s step, literal 360-day
seasonal clock, native-fp64 legoESM GPU policy, vector-math certified NEMO
climate reference, scalar-libm CPU ladder policy, days 30--180 budget window,
western-boundary wet-span region, eight-row registry, and absolute
`1e-12 m/s` pointwise algebraic closure bar.

OPEN, in order:

1. Operator runs the exact scalar-math budget acquisition above.
2. Score every source-aligned candidate against the preregistered four-times
   scalar/vector term envelope.
3. If `total_vorticity_coriolis` clears that floor, run the one-state,
   one-variable operand substitution before naming an owner or editing shared
   physics.
4. Post the round-16 result to issue #1455 when network access is available.

Choice audit: the user asked for the days-30--180 attribution, ranking, first
owner to fix, smallest proving test, and no physics edit without unambiguous
attribution.  No physics owner passed that bar, so none was fixed.  The only
added acquisition path is the exact scalar-NEMO term envelope required by the
requested comparison.

## Compiled-source citations

The corrected NEMO writer captures the pre-ZDF state before `dyn_zdf` removes
the prognostic barotropic velocity at
`GYRE_OMIP_L2_P3_R15BUD/BLD/ppsrc/nemo/dynzdf.f90:159-168`.  After the
implicit solve, the stage code computes and applies the final barotropic
correction, then passes the final surface V to the recorder at
`GYRE_OMIP_L2_P3_R15BUD/BLD/ppsrc/nemo/stprk3_stg.f90:504,532,534`.  The writer
therefore assigns the removal to its vertical row and the restoration to its
external row, as its explicit row equations show at
`GYRE_OMIP_L2_P3_R15BUD/BLD/ppsrc/nemo/gyre_r14_budget.f90:161-184`.

In legoESM, the vertical solve explicitly re-adds the subtracted barotropic
mean to both the production and no-stress results before returning at
`ocean_model_latlon_cgrid.py:13050-13058`.  Its budget trace then forms
vertical, stress, external, and independent total rows from those returned
boundaries at `ocean_model_latlon_cgrid.py:10425-10442`.  This executed-source
comparison is why only the vertical-plus-external pair is semantically
matched; it does not change the preregistered raw table.
