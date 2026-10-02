# Preregistration: corrected-T wall epoch, selector-lattice bisect

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen after two configuration
stops and before any replacement-lattice arm runs.

## Retractions and retained receipt

The claims that the original `entry` and repaired `entry_dep` groups were
independently revertible are **retracted**. Neither stopped configuration is a
model arm and neither produced an NPZ. The first violated the carried-slope-N2
dependency; the second also violated the literal-matrix dependency:

`tke_matrix_evaluation='nemo_literal' requires
tke_preclosure_coeff_source='carried_previous_step'`.

The clean `slope_n2_only` arm is retained at
`/tmp/dino-zdf-wall-epoch-bisect-fresh-01a04e34/arms/slope_n2_only.npz`.
Its SHA-256 is
`1395b84ec8e9f482c17bde34ebb59d273f794b7e5901ac6c3ea02c0e67d43083`,
producer `9ac2550d4f559b1c73c2b65174fc2e436025176a`; it is float64, stable,
160 samples, and changes only slope N2 to `recompute`.

## Complete executable dependency graph

The graph below comes from every `requires` guard reached by the 16 selector
families in the DINO constructor and runtime validation paths. An arrow means
the faithful selector on the left requires the faithful selector on the
right:

```text
literal solver --------> literal matrix -----> carried preclosure coefficients
                               |  |
literal Langmuir --------------+  +----------> step-entry N2 bundle
                                                     ^
carried slope N2 ------------------------------------+
```

The selector-to-selector guards are:

| Faithful selector | Required faithful selector | Executable guard |
|---|---|---|
| TKE literal matrix | carried preclosure coefficients | `tke.py:2721-2722` |
| TKE literal matrix | step-entry N2 evaluation | `tke.py:2725-2726` |
| TKE literal solver | literal matrix | `tke.py:1197-1199` |
| TKE literal Langmuir | literal matrix and step-entry N2 | `tke.py:2964-2981` |
| carried slope N2 | step-entry N2 bundle | `ocean_model_latlon_cgrid.py:4906-4911` |

The same grep found fixed-input guards, not additional selector edges:
carried coefficients require prognostic TKE, one iteration, and carried
`avm/avt` (`tke.py:2702-2712`); literal matrix requires live `e3t`, `dissl`,
W mask/surface `avm`, NEMO surface row, and the NEMO dissipation split
(`tke.py:1200-1217,1443-1450,2729-2734`); literal solver requires the NEMO
surface row, W mask, and floor positivity (`tke.py:1253-1262`); literal
Langmuir requires its bottom/W operands and one iteration
(`tke.py:2000-2004,2967-2981`); step-entry N2 requires `nemo_bn2` and raw NEMO
mesh ladders (`ocean_model_latlon_cgrid.py:6048-6055,5934-5941`); step-entry
shear and its live face metric require carried velocity/`avm`, QCO geometry,
and raw face metrics (`ocean_model_latlon_cgrid.py:6148-6239`); literal htau
requires native T-point latitude (`ocean_model_latlon_cgrid.py:7082-7067`);
carried slope N2 and literal PRD require the fixed `nemo_bn2`/S-EOS settings
(`dino.py:3206-3226`). The etau exponential, raw MXL association, final ZDF
recurrence, and remaining three slope arithmetic selectors add no
selector-to-selector `requires` edge. Literal final ZDF also refuses distinct
DDM heat/salt matrices (`ocean_model_latlon_cgrid.py:7622-7630`), which is off
in every frozen arm.

Let `R(x)` mean selector `x` is reverted. The legal reversion ideals are
exactly closed under:

- `R(preclosure) -> R(matrix, solver, Langmuir)`;
- `R(N2 stage) -> R(matrix, solver, Langmuir, slope N2)`;
- `R(matrix) -> R(solver, Langmuir)`.

Solver, Langmuir, slope N2, and the other ten selectors may be reverted
without forcing another selector. Thus the six coupled selectors have 14,
not 64, legal states; with the ten independent selectors the full lattice has
14,336 legal subsets. The committed scorer enumerates that lattice and refuses
an illegal registered arm during `--self-test`.

## Replacement arms on the legal lattice

The current control remains the hash-pinned corrected-T artifact (ratio
`1.6511846170859847`, wall share `0.17020235431211447`); the historical target
remains ratio `1.1607251697830108`, wall share `0.11941867158491266`, with the
previously frozen bands. All production, state, runtime, stress, comparator,
and control gates are unchanged.

One retained plus five new arms cover all 16 selectors:

| Arm | Legacy selectors |
|---|---|
| `slope_n2_only` (retained) | carried slope N2 only |
| `tke_core` | matrix, solver, Langmuir |
| `core_slope_n2` | matrix, solver, Langmuir, slope N2 |
| `entry_closed` | preclosure, shear stage, shear metric, N2 stage, plus the forced matrix/solver/Langmuir/slope-N2 closure |
| `mxl_zdf` | etau exponential, htau, raw MXL, final ZDF recurrence |
| `slope_rest` | PRD, reciprocal metric, live face thickness, live depth; slope N2 stays faithful |

`current`, `slope_n2_only`, `tke_core`, and `core_slope_n2` form the reachable
core x slope-N2 four-corner. Entry is measured only by
`core_slope_n2 - entry_closed`, with forced dependencies held legacy.

## Frozen bars and unreachable contrasts

The primary observables and ordered bars are unchanged. For each observable
`q`, closure is `(q_current-q_arm)/(q_current-q_historical)`:

1. `EPOCH_RESTORED`: both scores enter the historical bands and both closures
   are at least 0.50.
2. `MAJORITY_OWNER`: both closures are at least 0.50.
3. `BOUNDED_SMALL`: both absolute closures are at most 0.10.
4. Otherwise `OPEN_MIXED_OR_PARTIAL`.

The entry conditional and core-given-slope-N2 contrasts use the same 0.50 and
0.10 bars. The reachable core x slope-N2 difference-in-differences is reported
but is not promoted through a post-hoc ownership bar.

The following contrasts are structurally unreachable and will not be inferred:

- legacy preclosure with faithful matrix/solver/Langmuir;
- legacy N2 stage with faithful matrix/Langmuir or faithful carried slope N2;
- an independent entry main effect with all downstream selectors faithful;
- the entry x forced-dependency interaction needed to convert its conditional
  effect into an independent main effect.

Multiple owner contrasts mean composition. No owner triggers the already
registered all-16-legacy universe gate and complement tests. Artifact admission
still pins producer, clean tree, current initial state/mask, session, float64
160-sample SSH, NEMO ladder, start mode, reconstructed T-stress content, and all
non-arm configuration fields; the classifier plants and flicker controls remain
mandatory.
