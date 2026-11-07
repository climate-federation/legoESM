# NEMO testcase Lane 4 — ORCA2 card round 23 second-solve receipt

Date: 2026-09-26

Parent: `6da9dd3fd65aea24684394c79c0c9cb08864ba62`

Status: **LANDED — DECISION 58.**  The ORCA2 card now executes NEMO's
separate velocity-form continuity solve for momentum at RK3 stages 2 and 3.
The ten-step ladder moves first at kt=1 stage-2 momentum, the first non-bit
NEMO statement stays kt=1 stage-1 temperature, and GYRE is byte-identical.

Every ORCA2 number below is **independent with Decision-52 SSH**.  No
given-NEMO-entry solver number is mixed into a table.  The six sea-ice
selectors and the card's `unmeasured_features` tuple remain unchanged:
`staged_gm_eiv`, `linear_implicit_bottom_drag`, `internal_wave_mixing`,
`spatial_lateral_viscosity`, `freshwater_budget_carry`, and
`si3_jpl5_layered_prather_state`.

## 1. Executed compiled statements

The admitted ORCA2 build takes the vector-invariant branch.  At stages 2 and
3 it calls `wzv` on raw Kmm velocity for momentum at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3_stg.f90:323-329`.
Tracer transport separately calls `wzv` on its corrected transport at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/traadv.f90:296-300`.
The velocity-form divergence statement those momentum operands select is
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/divhor.f90:123-130`.
The executing deck proves the branch: `ln_dynadv_vec = T` in its recorded
`ocean.output`, and its three RK3 stage calls are present there.

The shared two-solve implementation already existed and GYRE already executed
it.  Round 23 changes no numerical operator: the ORCA2 card's explicit field
is the one controlled variable.

| resolved field | before, `5522bd417` | after, `46ab68361` |
|---|---|---|
| `nemo_stage_momentum_wzv_split` | `False` | `True` |

The result gate checks the full `packages/` diff from before to tip.  The only
non-comment model statement changed is that `False` to `True`; the other two
model files only bring their existing comments up to date.  The resolved
predicate changes from not executing to executing.  No default or other card
leaf moves.

## 2. ORCA2 ten-step ladder

Both arms use the same card, admitted record, surface forcing, CPU backend,
fp64 policy, ten steps, and Decision-52 SSH entry.  Both artifacts carry a
clean-worktree stamp.  Of 200 scored rows, **185 move**.  Every moved row,
including its complete before/after score and direction, is registered in
`round23/result.json`: 103 move toward NEMO by maximum absolute error and 82
move away; none has the same maximum.  This mixed response is expected
Rule-12 exposure, not grounds to remove NEMO's cited statement.

| independent checkpoint | before max error | after max error | direction |
|---|---:|---:|---|
| kt=1 stage-2 u | `0.06463349988292608` m/s | `0.06470386947382581` m/s | away |
| kt=1 stage-2 v | `0.03401471577804818` m/s | `0.034012848056840184` m/s | toward |
| kt=10 entry T | `3.947126188631776` degC | `2.8242210027469206` degC | toward |
| kt=10 stage-3 T | `0.7734225584952537` degC | `1.157338474726192` degC | away |
| kt=10 stage-3 u | `16.16081318414712` m/s | `15.365503106245665` m/s | toward |
| kt=10 stage-3 v | `41.579560180754974` m/s | `42.669598831454074` m/s | away |

The first moved row is kt=1 stage-2 u.  No kt=1 entry or stage-1 row moves,
and no formerly bit-identical entry/stage-1 row becomes non-bit at a later
step.  The first NEMO mismatch remains exactly kt=1 stage-1 T, 233,341 of
399,600 cells, maximum `0.0014770192519697467` degC.  Thus Decision 58 does
not move the owner boundary; it changes the downstream trajectory.

## 3. Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R23-P1 | **CONFIRMED** | One resolved ORCA2 leaf changes False to True and the execution predicate flips. |
| R23-P2 | **CONFIRMED** | The first movement is kt=1 stage-2 u; kt=1 entry and stage 1 are unchanged. |
| R23-P3 | **CONFIRMED** | The first NEMO mismatch remains kt=1 stage-1 T with the same score. |
| R23-P4 | **CONFIRMED** | 185 rows move; the planted kt=1 entry ULP creates a 186th row and makes the gate HELD. |
| R23-P5 | **CONFIRMED** | GYRE has 0 moved certified rows, 210/210 equal residual arrays, and 30/30 byte-identical daily snapshots. |

No prediction was rewritten after measurement.

## 4. GYRE unchanged gate

| comparison | result |
|---|---|
| certified trajectory | PASS, 70 rows, 0 ULP maximum worsening, first-over-bar unchanged at kt=3 |
| `ladder.residuals.npz` | 210/210 arrays `np.array_equal`; SHA-256 `43f37831256832c3...` both arms |
| daily snapshots | day001 through day030, 30/30 byte-identical |
| day-30 snapshot | digest `a66143733bcc9e4e`, unchanged |

This is the required shared-implementation proof.  GYRE's resolved config
already selected the statement before this round; the ORCA2-only card value
does not move its ten-step or day-30 trajectory.

## 5. Gate, review, and tests

The round-23 result gate exits 0 with `LANDED`.  Its synthetic one-ULP entry
plant exits 2 with `HELD`, changes the first moved row to kt=1 entry T, and
refutes R23-P2.  The result artifact enumerates all 185 real moved rows.

The required separate `codex exec --sandbox read-only` review was attempted
at the committed implementation tip.  It failed before reading the diff:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

The receipt citation gate passes: three citations, zero failures, zero
unmapped citations, and zero map entries failing audit.  Its real two-line
plant on the stage statement exits 1 with `SYMBOL-NOT-AT-LINE`; all nine
self-tests fire.  The final focused battery passes **84 / 84**.

The one required `tests/ocean/fidelity -n 12` battery collected 1,865 tests
and reached the inherited final-tail hang after **1,840 passed, 7 skipped,
5 failed, and 13 unfinished**.  It was interrupted after a five-minute
bounded wait and is not represented as green.  Explicit isolated reruns
identify the same five inherited failures recorded by the preceding round:

1. the round-129 record refuses because the certified stepping-gate stamp
   moved after its members ran;
2. the round-51 live-trace suffix assertion is stale;
3. the SI3 scalar-math gate says its retained `MY_SRC` is not verbatim;
4. the worktree-stamp ratchet names the same three legacy unstamped emitters;
5. the case board lacks the `hires_lane_surface` comparison-driver row.

None reads a round-23 model, gate, citation, or receipt path.  No round-23
focused test fails.

Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round23/`.

## Choices

ASKED: Decision 58 changes ORCA2's explicit second-continuity-solve choice
from False to True.

UNASKED: none.  No default, carried state, stabilizer, scoring rule, sea-ice
selector, or other model configuration changes.

## OPEN

1. Decision 54 is next: land the whole three-part lateral-diffusion
   attribution under the required GYRE, DINO, lock-exchange, and overflow
   gates.
2. The kt=10 mixed movement is fully registered (103 rows toward, 82 away);
   Rule 12 makes it the downstream walk after the cited statement stays.
3. Round 20's ranked slow-forcing producer walk remains open.
4. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
5. The independent ORCA2 year still depends on its scheduled independent
   initial-state completion.
6. The seven inherited duplicate citation-map literal keys remain open.
