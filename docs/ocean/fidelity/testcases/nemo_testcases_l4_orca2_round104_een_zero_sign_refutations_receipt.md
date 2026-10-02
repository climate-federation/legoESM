# ORCA2 round 104 — rung-0 EEN zero-sign refutations

Date: 2026-10-02. Base `d99367ba27b1f1e4da6311e0ea2c2bf778618c3a`.
Scope is ocean only. Every numerical result below is **independent**: hierarchy
rung 0 starts from NEMO's own from-rest state. No table mixes these results
with the shipped card's Decision-52 given-entry claim.

## Verdict

**HELD.** Three separately preregistered, one-variable interpretations of
NEMO's EEN coefficient loop are refuted. The second arm is informative: it
makes every non-fold magnitude bit-exact and reduces the coefficient debt to
228--419 signed-zero/fold bits per field, but it does not meet the bit bar.
Every production edit was reverted. No `packages/` diff, card field,
configuration choice, stabilizer, threshold, carried state, sea-ice selector,
or ORCA2 `unmeasured_features` entry lands.

The next source-order discriminator needs an oracle operand not present in the
admitted records: each pre-scale accumulator paired with its exact final scale.
This round commits a fail-closed operator acquisition for that record. It is
syntax-proved and its layout/content plants fire, but it is not run here.

## Executing statement and card scope

The compiled ORCA2 arm initializes all eight arrays to positive zero, loops U
through `mbku` and V through `mbkv`, then scales each accumulated coefficient
in `ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1213-1265`.
The loop bounds are never zero: the compiled domain setup forces `mbku` and
`mbkv` to at least one at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/domzgr.f90:621-631`, while its mask setup
clears an entire face column when that bottom index is one at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dommsk.f90:218-229`.

ORCA2 rung 0 and the shipped ORCA2 card execute the `np_EEN` statement. The
VORTEX vector card also executes EEN. GYRE executes the sibling `np_ENE` arm
through the same production coefficient builder, and DINO shares its
metric-weighted barotropic coefficient path. The flux-form tanks do not
execute this vector-invariant coefficient statement. Thus any production
landing remains subject to all shared-card gates even though this held round
leaves the shared tree unchanged.

## Frozen arms — independent

Each arm used the admitted two-rank round-98 final-coefficient record and
round-96/90 split-explicit operands through the committed round-98 residual
probe. Both the coefficient and consumer one-bit plants fired before the
results were accepted.

### Arm 1 — remove the early local face masks

Commit `2fa033ee1` removed only the builder's early U/V face masks. It was
**REFUTED** and reverted by `973ec077c`. Instead of fixing zero signs, it
introduced 2,115--2,693 non-fold magnitude differences per coefficient.

| coefficient | non-fold bit unequal | non-fold magnitude unequal | fold magnitude unequal | max abs |
|---|---:|---:|---:|---:|
| `ffu_ne` | 6,166 | 2,652 | 19 | `8.388907644539505e-4` |
| `ffu_nw` | 6,208 | 2,693 | 19 | `1.7706856596698278e-3` |
| `ffu_se` | 6,027 | 2,451 | 19 | `2.5812540150707704e-3` |
| `ffu_sw` | 6,121 | 2,543 | 19 | `7.079382622149699e-4` |
| `ffv_ne` | 6,108 | 2,604 | 67 | `1.8179619797217355e-3` |
| `ffv_nw` | 6,072 | 2,568 | 66 | `1.4701409993307707e-3` |
| `ffv_se` | 5,700 | 2,115 | 10 | `2.4695867884935627e-3` |
| `ffv_sw` | 5,823 | 2,239 | 10 | `1.3049603813940708e-3` |

At the consumer, substep-1 U/V retain 398/397 active signed-zero differences.
Substep-2 expands to 6,904 U cells, maximum `1.7111407059559952e-6`, and
6,499 V cells, maximum `3.2951394806195284e-6`. R104-P1 through P3 are
**REFUTED**. Artifact `round104/een_mask_arm.json`, SHA-256
`9f46c226168e1cf4c4ea55dd3905756e368dbf2b531a3f0e681cc471a99ae51b`.

### Arm 2 — gate each recurrence by the local wet-level mask

Commit `f93a57947` removed the early masks and conditionally updated the
accumulator only at locally wet levels. It was **REFUTED** and reverted by
`d678822e3`. It is the narrowing result: every non-fold magnitude becomes
bit-exact, but signed zeros remain.

| coefficient | non-fold signed-zero unequal | fold magnitude unequal |
|---|---:|---:|
| `ffu_ne` | 228 | 0 |
| `ffu_nw` | 260 | 0 |
| `ffu_se` | 354 | 0 |
| `ffu_sw` | 336 | 0 |
| `ffv_ne` | 334 | 67 |
| `ffv_nw` | 353 | 66 |
| `ffv_se` | 260 | 0 |
| `ffv_sw` | 228 | 0 |

The source-associated consumer rows are substep-1 U 398 active / 399 full,
substep-1 V 402 / 402, substep-2 U 68 / 68 with maximum
`2.9617669311254642e-8`, and substep-2 V 0 active / 68 full (signed zero only
outside the active score). R104-P6 and P7 are **REFUTED**. Artifact
`round104/een_loop_arm.json`, SHA-256
`bdcd1b1b63498fa6dd92b279363d7acfbad2b7be8ba43b77b84ef1506d945921`.

### Arm 3 — force one recurrence level on a fully dry face

Commit `db4e964c4` made the arm-2 predicate reproduce the compiled
minimum-one bottom index. It was **REFUTED** and reverted by `0b4131050`.
All non-fold magnitudes remain exact, but the dry-domain sign support grows:

| coefficient | non-fold signed-zero unequal | fold magnitude unequal |
|---|---:|---:|
| `ffu_ne` | 3,577 | 0 |
| `ffu_nw` | 3,577 | 0 |
| `ffu_se` | 3,579 | 0 |
| `ffu_sw` | 3,579 | 0 |
| `ffv_ne` | 3,510 | 67 |
| `ffv_nw` | 3,511 | 66 |
| `ffv_se` | 3,647 | 0 |
| `ffv_sw` | 3,647 | 0 |

Substep-1 U/V retain 398/402 active signed-zero differences but expand to
3,871/3,972 full-domain differences. Substep-2 U preserves the independent
68-cell magnitude row while its full-domain sign count expands to 3,344;
substep-2 V remains active-bit-exact with 3,417 full-domain sign differences.
R104-P9 and P10 are **REFUTED**. Artifact
`round104/een_minone_arm.json`, SHA-256
`03e1d8a9296fcac2c3ef1b9f84996f5d03455dfe19fde3aecfbc761e0934835a`.

## Acquisition request

The new writer records 16 owned-slab arrays on both ranks: the eight
accumulators immediately before NEMO's final scale and the eight scale factors.
The self-describing checker reconstructs every admitted round-98 final
coefficient as `scale * accumulator` bitwise, requires exact-once global rank
coverage, and requires all 20 terminal restart shards byte-identical to the
round-98 source run. Header, name, dimensions, truncation, missing-field,
signed-zero, swapped-rank, restart-byte, source-layout, and producer-content
plants are fail-closed.

`run.sh --preflight-only` reports `SYNTAX_PROOF_PASS` and
`ORCA2_ROUND104_EEN_ACCUM_PREFLIGHT_READY`. Both launcher plants exit 69 with
`STATUS PLANT-FIRED`. The operator should run the committed launcher at the
path in OPEN; no result is claimed before admission.

## Validation and review

- Final tree has no `packages/` diff from the round base; GYRE, DINO, tanks,
  rung-0 and rung-7 trajectories are unchanged by construction.
- Acquisition writer and patched compiled source pass the gfortran syntax
  proof; a synthetic rank-complete record passes, and all eight record plants
  fire.
- Citation gate, focused tests, prescribed ocean-fidelity battery, and
  separate read-only review are recorded in the final round commit.

## OPEN

1. Operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round104_een_accum_acquisition/run.sh`.
2. Admit the record, compare each first unequal pre-scale accumulator/scale
   bit, and land only the first NEMO statement that satisfies the full bars.
3. Then isolate the independent northern-fold 66/67 magnitude debt and the
   later 68-cell substep-2 U residual.
4. The package-exposed rung-0 card and independent 240-step month remain
   unmeasured hierarchy deliverables.

## UNVERIFIED

- The exact accumulation or final-scale operand producing the zero signs.
- The new NEMO operand record, until the operator runs and admits it.
- No physics or configuration statement lands in this round.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
