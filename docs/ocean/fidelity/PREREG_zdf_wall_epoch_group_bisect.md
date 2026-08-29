# Preregistration: corrected-T wall epoch drift, dependency-coherent ZDF groups

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen after the metric x continuity
factorial returned `INVALID_CONTROL`, and before any selector-group arm runs.

## 2026-08-29 dependency correction before scoring

The first `entry` arm stopped during model configuration and produced no NPZ:
`gm_redi_slope_n2_evaluation=carried_step_entry` requires the pre-`zdf_phy`
TKE N2 bundle, which the legacy entry selector removes. The original statement
that all four groups were independently revertible is **retracted**. The
parallel `tke_core` arm completed stably for 160 samples at clean producer
`9ac2550d4f559b1c73c2b65174fc2e436025176a`, artifact SHA-256
`1be9010230834eca71f349aa672c20c5704c2f6b0887ed19d0b4eff01e38ee3a`;
no flicker statistic was opened. That valid independent arm is retained.

Operational amendment: the coordinator deleted that unscored run root during
cleanup. The retained-artifact optimization is therefore withdrawn without
changing any arm, contrast, or bar. The handoff reruns all five reachable arms
from one fresh root and one pinned producer; `tke_core` is admitted by the same
producer/config/state/runtime gates as the other four, not by the deleted NPZ's
archive hash.

The repaired design pins every model arm to producer `9ac2550d...`. It replaces
the unreachable entry arm with `entry_dep`, which also reverts the required
slope-N2 selector, and adds `slope_n2_only` to measure that dependency. The
entry effect is therefore registered only as the reachable conditional
contrast `entry_dep - slope_n2_only`. The fourth corner—legacy entry with
faithful carried slope N2—cannot execute, so the entry x slope-N2 interaction
is explicitly **unidentifiable**, not assumed zero.

## Question and fixed endpoints

Which family among the 16 legacy-selectable ZDF-sweep corrections moved the
five-day corrected-T wall-flicker epoch from the basin-lane value to the
current value?

The current control is the completed M0A0 artifact
`wall/legacy_generic.npz`, SHA-256
`c8c7135a12a75332cb1662052dc7c346b6bae7523dabe75e0f2223c515ac616a`,
produced cleanly at `e013e95ca54957a4454878ed7118e623da0a19ba`. Its frozen
score is ratio `1.6511846170859847`, wall share `0.17020235431211447`.

The historical endpoint is not admitted as a same-commit control because it
predates producer and initial-state hashes. It is retained only as the
epoch-target receipt: `wind_place_tcarry_explicit_flicker.json`, SHA-256
`5fd033345548dd5b80390293b0ee786d3c23f2a68055ae86afe2b616ad4e5dd0`,
ratio `1.1607251697830108`, wall share `0.11941867158491266`. Its already-frozen
corrected-T control bands are `[1.0470043852687352,1.2805669019825299]` and
`[0.0934682579050795,0.14512176885262343]`, respectively.

Every new arm retains the current control's corrected T-point carry,
`dino_wind_profile_evaluation=nemo_literal`, legacy V-face metric, generic
continuity association, cold TKE start, bridged BEFORE state, fp64, NEMO ladder
`both`, five days, and all-step SSH. The already-null metric/continuity pair and
the separately-owned wind arithmetic are not candidates.

## Reachable group reversions

Four replacement arms plus the retained `tke_core` arm cover the candidates
through executable configurations:

| Arm | Legacy selectors |
|---|---|
| `entry_dep` | preclosure coefficient lifetime; shear evaluation lifetime; shear face metric; N2 evaluation lifetime; **required** slope N2=`recompute` |
| `slope_n2_only` | slope N2=`recompute`, with faithful entry bundle |
| `tke_core` | TKE matrix association; TKE recurrence; Langmuir update association |
| `mxl_zdf` | etau exponential; htau construction; raw mixing length; final momentum/tracer ZDF recurrence |
| `slopes` | carried slope N2; density/PRD association; metric reciprocal; live face thickness; live depth association |

The exact field/value map lives in `zdf_wall_epoch_group_score.py`; artifact
admission requires the full run configuration to equal the frozen current
control after only the declared substitutions. `entry_dep` and
`slope_n2_only` intentionally overlap at slope N2 so their difference isolates
the reachable conditional entry effect. All other groups remain disjoint.

## Frozen reduction and bars

The primary observables remain the registered first-eight-step all-domain
legoESM/NEMO RMS ratio and legoESM aggregate-wall share from
`eta_flicker_decay.py`. For each observable `q`, direct-arm closure is

`C_q = (q_current - q_group_legacy) / (q_current - q_historical)`.

Classification is ordered:

1. `EPOCH_RESTORED`: both scores lie inside the historical corrected-T bands
   and both closure fractions are at least 0.50.
2. `MAJORITY_OWNER`: both closure fractions are at least 0.50.
3. `BOUNDED_SMALL`: both absolute closure fractions are at most 0.10.
4. Otherwise `OPEN_MIXED_OR_PARTIAL`.

Two conditional closures are also registered:

- entry given legacy slope N2:
  `(q_slope_n2_only - q_entry_dep) / (q_current - q_historical)`;
- the other four slope selectors given legacy slope N2:
  `(q_slope_n2_only - q_slopes) / (q_current - q_historical)`.

Conditional closure is `CONDITIONAL_MAJORITY_OWNER` only when both observables
are at least 0.50, `BOUNDED_SMALL` only when both absolute closures are at most
0.10, and otherwise `OPEN_MIXED_OR_PARTIAL`. A conditional entry owner remains
conditioned on legacy slope N2; it cannot be promoted to an independent main
effect or used to infer the missing interaction. Multiple owner contrasts mean
composition. No owner contrast triggers the all-16-legacy universe gate and
then complement pairs. No post-hoc threshold may promote a mixed response.

## Admission and controls

The current artifact and NEMO comparator are hash-pinned. New artifacts and
the retained `tke_core` artifact must share the pinned clean producer, the
current initial-state hash, masks, session,
float64 160-sample SSH, ladder, start mode, stress content, and all non-group
configuration fields. The scorer refuses any production model or twin-harness
change since the current artifact's producer. It retains the flicker probe's
land-poison and amplitude-recovery controls and plants values for every
classification branch.

This is a dependency-conditioned family localization. It does not predeclare a
unique selector owner, cannot erase a detected interaction, and records the
structurally missing corner in every score artifact.
