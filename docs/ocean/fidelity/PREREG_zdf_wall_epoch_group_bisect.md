# Preregistration: corrected-T wall epoch drift, four ZDF selector groups

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen after the metric x continuity
factorial returned `INVALID_CONTROL`, and before any selector-group arm runs.

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

## Four group reversions

Each arm reverts exactly one coherent group and leaves the other 12--13
selectors faithful:

| Arm | Legacy selectors |
|---|---|
| `entry` | preclosure coefficient lifetime; shear evaluation lifetime; shear face metric; N2 evaluation lifetime |
| `tke_core` | TKE matrix association; TKE recurrence; Langmuir update association |
| `mxl_zdf` | etau exponential; htau construction; raw mixing length; final momentum/tracer ZDF recurrence |
| `slopes` | carried slope N2; density/PRD association; metric reciprocal; live face thickness; live depth association |

These groups cover all 16 and do not overlap. The exact field/value map lives
in `zdf_wall_epoch_group_score.py`; its artifact admission requires the full
run configuration to equal the frozen current control after only that group's
declared substitutions.

## Frozen reduction and bars

The primary observables remain the registered first-eight-step all-domain
legoESM/NEMO RMS ratio and legoESM aggregate-wall share from
`eta_flicker_decay.py`. For each observable `q`, group closure is

`C_q = (q_current - q_group_legacy) / (q_current - q_historical)`.

Classification is ordered:

1. `EPOCH_RESTORED`: both scores lie inside the historical corrected-T bands
   and both closure fractions are at least 0.50.
2. `MAJORITY_OWNER`: both closure fractions are at least 0.50.
3. `BOUNDED_SMALL`: both absolute closure fractions are at most 0.10.
4. Otherwise `OPEN_MIXED_OR_PARTIAL`.

Exactly one majority/restoring group localizes the epoch drift to that group,
not yet to an individual selector. Multiple owner groups mean composition and
require complement-paired interaction arms. No owner group means the four
one-group reversions cannot distinguish distributed effects from interaction;
the next mandatory arm is the all-16-legacy same-commit universe gate, followed
by complement pairs. No post-hoc threshold may promote a mixed response.

## Admission and controls

The current artifact and NEMO comparator are hash-pinned. New artifacts must
share one clean producer, the current initial-state hash, masks, session,
float64 160-sample SSH, ladder, start mode, stress content, and all non-group
configuration fields. The scorer refuses any production model or twin-harness
change since the current artifact's producer. It retains the flicker probe's
land-poison and amplitude-recovery controls and plants values for every
classification branch.

This is a four-arm family localization. It does not predeclare a unique
selector owner and cannot erase a detected interaction.
