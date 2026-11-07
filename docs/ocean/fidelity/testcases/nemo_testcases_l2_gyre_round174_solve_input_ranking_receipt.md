# Round 174 receipt — developed solve-input magnitude ranking

**Status: HELD.**  No production physics, configuration, carried state,
restart schema, card default, or immutable before arm changed.  The admitted
paired NEMO record **REFUTES** both frozen scientific predictions: replacing
NEMO's wet-cell temperature content with legoESM's free-running content gives
day-240 T3D RMS `1.694285453522351e-02` K, while replacing live `e3t(Kaa)`
gives `1.126600178477721e-03` K.  Content is `15.0389x` the e3t response and
`13.6497x` the production complete-K/e3w remainder.  The next walk is the
temperature-content producer; the forced-input result is magnitude ranking,
not source-exact landing proof.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round174/`.  The frozen
preregistration is commit `7e9289ef7`; the fail-closed committed scorer is
`fe07a4efa`.

The immutable production headline remains Round 163: first-over-bar kt3,
day-30 T RMS `6.572574374770603e-05` K, day-240
`1.644836070117868e-02` K, and day-360 `1.122566001855131e-02` K.  No accepted
candidate ran, so the ladder, month, year, DINO, generic GYRE, tanks, and ORCA2
rows have zero registered movement.

## Compiled statements and record meaning

The producing build reads one paired input frame per step at
`GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:136-150`.
The e3t arm selects legoESM's wet-cell diagonal factor at `:490-504`; the
content arm selects legoESM's wet-cell temperature RHS at `:583-609`; the
ordered backward solve consumes those inputs at `:619-624`.

Thus the experiment changes one input family in each independently compiled
NEMO trajectory while retaining NEMO's own dry-cell operands.  Because every
step consumes the corresponding value exported by the free legoESM trajectory,
the resulting endpoint response measures family leverage.  It does not prove
that legoESM's local content-assembly arithmetic is wrong, nor that substituting
NEMO content into a freely evolving legoESM run would produce the inverse
endpoint change.

## Admission and registered ranking

The committed Round-173 admission was rerun from this clone.  Both arms have
exact `STOP 0`, byte-identical executables and corrected inputs, byte-identical
step-1080 restarts, and step-1440 restarts different from baseline.  The reused
baseline is byte-identical to Round 125 at steps 1080 and 1440.  The scorer
then independently verifies the four-output manifest and all these hashes
before opening a restart.

| arm | wet unequal | max absolute T | day-240 T3D RMS | fraction of `1.241262968697578e-03` K remainder |
|---|---:|---:|---:|---:|
| baseline versus Round 125 | `0 / 18,000` | `0` K | `0` K | `0` |
| `e3t_wet` | `18,000 / 18,000` | `2.9849144358202295e-02` K | `1.126600178477721e-03` K | `0.907624` |
| `content_wet` | `18,000 / 18,000` | `1.8846929375568173` K | `1.694285453522351e-02` K | `13.649690` |

The two directed rows are not additive pieces of the endpoint gap.  Each is a
separate nonlinear 60-day trajectory forced every step, so fractions above
one and overlap are permitted.  The table ranks leverage only.

## Frozen predictions

| preregistered item | verdict | result |
|---|---|---|
| record admission passes every binary/restart/input check | **CONFIRMED** | both arms admitted; baseline remains Round 125 |
| baseline is BIT against Round 125 | **CONFIRMED** | `0 / 18,000`, RMS `0` |
| both arms are discriminating | **CONFIRMED** | both move all 18,000 wet temperature cells |
| e3t response is larger than content | **REFUTED** | content is `15.0389x` e3t |
| neither response reaches the `6.20631484348789e-04` K half-remainder threshold | **REFUTED** | e3t is `1.81525x`; content is `27.2994x` the threshold |
| each scale plant changes every unequal wet cell and exits nonzero | **CONFIRMED** | both move 18,000 wet, zero dry, print `STATUS PLANT-FIRED`, exit `1` |

The earlier hypothesis that live `e3t(Kaa)` is the larger remaining solve
input is retracted.  The measurement that kills it is the paired NEMO response
above, with the same binary, interval, mask, and baseline.

## Landing verdict and shared-card scope

Nothing lands.  No first non-bit production statement is claimed: the first
magnitude boundary is the compiled temperature-content assembly selected at
`:583-609`, but the paired record does not distinguish a wrong local statement
from inherited differences in its thickness, before-temperature, or accumulated
tendency operands.  Rule 12 and Decisions 43/45/55/59 do not run because the
model implementation is unchanged.  Every certified card therefore has zero
movement, and no configuration or carried-state choice was made.

The pre-implementation search found and extended the existing year-owner
instrument, its NEMO restart reader, RMS definition, and mesh-mask reader.  No
second scientific harness or weighting convention was created.

## Review, citations, and tests

The required separate command was run with `codex exec --sandbox read-only`.
It emitted no SHIP or DO NOT SHIP verdict.  Its terminal disposition verbatim
was:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore the independent review was unavailable in-sandbox; the round
continued under operator note C.

The receipt citation gate reports PASS with four citations, zero failures,
zero unmapped citations, and zero map-audit failures; all nine self-test
plants fire.  Shifting the content-assembly citation reports FAIL and exits
`1`.

The focused year-owner and citation-gate files report `55 passed in 34.91s`,
including a synthetic dirty-worktree refusal before any record is read.
The mandated single `tests/ocean/fidelity tests/ocean/unit -n 12` battery was
run once.  Four JAX/XLA workers aborted during unrelated compilations and were
replaced; a fifth crash item then produced an xdist internal error at 62%.
Its exact terminal summary is `70 failed, 5331 passed, 124 skipped, 2 xfailed,
59 warnings in 686.73s`.  The internal error suppressed the failed-ID summary,
so those 70 partial-run failures cannot be classified against the historical
87-red list.  This is incomplete full-tree coverage, not a pass; the directly
changed focused files have the clean 55-test summary above.

## Evidence hashes

| artifact | SHA-256 |
|---|---|
| `acquisition_admission.log` | `1da2b20f353cc2eb9b4c01fdb3212a5757880795a01e5d866961260c4052aa10` |
| `solve_input_pair_ranking.json` | `98cc67d1a20a098e67b17d439e3a14b6eb3a8f323bb08526847f200fdf38bae5` |
| `solve-input-e3t-scale.log` | `ee106ee48e23b9a03b3dc032ff9e2c4dcbd5444cf062c357df1a7aee6a5f577f` |
| `solve-input-content-scale.log` | `3daebfc1aed0871f156caefb76590664cc0541b466dc1be9906e4ef006da17d4` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |
| `citation_gate.json` | `1988601642a12848cf9e9bd7cdecf154651ff67b8a026647532dfdedb09ba570` |
| `citation_shifted_plant.json` | `db22d7ec486684ab7dc6fa4fb282df0856d8303ee85b63207392192a4b7c1def` |
| `focused_tests_final.log` | `e0a6d102beea33efa28d9d91fb39d0cbf27fb157b30dcd842c4f9c96f64255bd` |
| `full_fidelity_unit_tests.log` | `dd6bbdf414737ec14c5881d3dc29dcbf51bee8dbe5825a94fe77788ade6f85d4` |

## OPEN — round 175

1. Stay on the developed vertical-diffusion chain and walk the temperature
   content producer selected above.  At step 1081, given NEMO's admitted
   entry, score in compiled order the before-content term and the accumulated
   `Krhs` term that NEMO forms at `:583-609`, under production JIT and eager.
2. Use the admitted Round-125 record's `T_Kbb_in`, `T_Krhs_in`,
   `e3t_Kbb`, `e3t_Kmm`, `e3t_Kaa`, and `rhs_T` rows; reproduce the literal
   recorded RHS before interpreting a residual.  No acquisition is presently
   needed.
3. Name the first non-bit content operand or association.  If the statement is
   locally exact given NEMO operands, move upstream to the first non-bit input;
   do not infer a fix from Round 174's nonlinear forced response.
4. Keep Round 163 as the immutable production before arm.  Apply the full
   Decision 43/45 gate only if a one-variable source-exact candidate exists.
