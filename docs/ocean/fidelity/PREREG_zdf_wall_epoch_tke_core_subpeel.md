# Preregistration amendment: TKE-core wall-epoch sub-peel

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`. Frozen after the legal-lattice result
`LOCALIZED_TO_TKE_CORE_AT_FAITHFUL_SLOPE_N2` and before any sub-peel arm runs.

## Question and fixed endpoints

Which component of the three-selector TKE core supplied the compensating error
that restored the historical wall-flicker epoch: the solver recurrence, the
Langmuir association, their interaction, or the matrix conditional on both
consumers being legacy?

The faithful control is retained at SHA-256
`c8c7135a12a75332cb1662052dc7c346b6bae7523dabe75e0f2223c515ac616a`,
with ratio `1.6511846170859847` and wall share `0.17020235431211447`.
The full-core legacy endpoint is retained at SHA-256
`1be9010230834eca71f349aa672c20c5704c2f6b0887ed19d0b4eff01e38ee3a`,
with ratio `1.163579045567568` and wall share `0.11693494406953533`.
The historical target remains ratio `1.1607251697830108`, wall share
`0.11941867158491266`; its frozen bands remain
`[1.0470043852687352,1.2805669019825299]` and
`[0.0934682579050795,0.14512176885262343]`.

Both retained artifacts are admitted by content hash plus a clean-producer and
model/harness-diff-zero receipt against selected producer
`9ac2550d4f559b1c73c2b65174fc2e436025176a`. Exact producer identity is required
for every new arm. All artifacts must carry the common initial-state hash,
land mask, corrected T-point stress, session, fp64 policy, NEMO ladder `both`,
160 all-step samples, and the full haloed eta shape `(160,199,52)`.

## Legal arms

All unlisted selectors stay at the faithful control, including faithful slope
N2 and the literal matrix in every new arm:

| Arm | Matrix | Solver | Langmuir | Status |
|---|---|---|---|---|
| `current` | literal | literal | literal | retained by hash |
| `solver_only` | literal | shared legacy | literal | new |
| `langmuir_only` | literal | literal | vectorized legacy | new |
| `solver_langmuir` | literal | shared legacy | vectorized legacy | new |
| `full_core` | factored legacy | shared legacy | vectorized legacy | retained by hash |

These are legal lattice points. The faithful solver requires the literal
matrix, and literal Langmuir requires the literal matrix; therefore a factored-
matrix arm with either faithful consumer is structurally unreachable. The
matrix contrast is consequently conditional on both consumers being legacy,
not an independent main effect at the faithful consumer settings.

## Frozen decomposition and bars

For each observable `q`, let `D_q = q_current - q_historical`. Define:

- solver main at faithful Langmuir:
  `(q_current - q_solver_only) / D_q`;
- Langmuir main at faithful solver:
  `(q_current - q_langmuir_only) / D_q`;
- solver x Langmuir interaction at faithful matrix:
  `[(q_current-q_solver_langmuir) - (q_current-q_solver_only)
  - (q_current-q_langmuir_only)] / D_q`;
- matrix given legacy solver and Langmuir:
  `(q_solver_langmuir - q_full_core) / D_q`.

The four components must sum to the already-measured full-core closure for
both observables within `5e-13`; failure stops without a verdict. The scorer
also reports solver given legacy Langmuir and Langmuir given legacy solver.

Each component is classified identically and independently:

1. `MAJORITY_OWNER` when both observable fractions are at least `0.50`.
2. `BOUNDED_SMALL` when both absolute fractions are at most `0.10`.
3. Otherwise `OPEN_MIXED_OR_PARTIAL`.

Disposition is ordered:

1. Exactly one `MAJORITY_OWNER` and all other components `BOUNDED_SMALL`:
   `LOCALIZED_TO_<component>`.
2. Exactly one majority with any open component:
   `MAJORITY_<component>_WITH_OPEN_COMPONENTS`; sole ownership is withheld.
3. Multiple majority components: `COMPOSITION_MULTIPLE_TKE_CORE_COMPONENTS`.
4. No majority: `OPEN_DISTRIBUTED_OR_SUBBAR_COMPOSITION`.

Direct-arm historical-band classifications are reported but do not override
this component decomposition. No post-hoc sign, magnitude, or one-observable
response may promote ownership.

## Controls and next action

The committed scorer plants majority, bounded-small, and mixed component
responses; plants localized, composition, and open dispositions; verifies an
exact synthetic decomposition; reruns the flicker land-poison/amplitude
controls; and requires the retained full-core score to reproduce exactly.

If one component localizes, the next measurement is the matched-state
wall/interior ladder appropriate to that component, continuing through the
Langmuir-updated RHS, literal/factored diagonals, forward/backward recurrence,
solved TKE, `avm/avt`, vertical momentum increment, and first-eight-step SSH.
This sub-peel can identify which legacy TKE-core component supplied the
compensation; it still cannot name the other model error being compensated
without that operand ladder.
