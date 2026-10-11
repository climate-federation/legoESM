# ORCA2 round 246 — atomic fold unit exposes the stage V mask

Date: 2026-10-11. Frozen base: `c1b476175`. Preregistration commit:
`5fc3e7988`. Final measurement/classifier commit: `923d6a571`. Status:
**HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round246/`.

This round changes no effective package/model file, deck, selector, carried
state, stabiliser, sea-ice setting, or `unmeasured_features` entry. It adds a
committed measurement gate and tests only. Every number below is reported
separately as **independent** and **given NEMO's entry**; the two labels give
the same results. No NEMO acquisition was required.

## Compiled statement and current execution path

The executing rank-complete record build forms the stage V correction and
transport in one source-ordered unit at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279` and
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`:

```
zvb = vn_adv * r1_hv(Kmm) - vv_b(Kmm)
zFv = e1v * e3v(Kmm) * (vv(Kmm) + zvb * vmask)
```

The exact compiled statement multiplies both the live-thickness stretch and
the barotropic correction by NEMO's three-dimensional `vmask`. The current
legoESM stage path resolves D114's raw face thickness correctly, but passes
the compact mask made by `compute_face_masks_3d` into
`_nemo_stage_corrected_velocity`, then materialises `zFv`. The next walk must
route this stage mask through the same production resolved-mesh operand path;
it must not create a second mesh builder.

The committed gate is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round246_atomic_pair_gate.py`.
It first reruns round 245's direct statement gate, then runs the complete
round-217 unit on OMT-4 through kt=8 under each label, replays the current
five-operand stage transport against the admitted rank-complete record, and
classifies the first remaining boundary. All execution uses CPU fp64/x64.

## Frozen predictions

| prediction | result |
|---|---|
| R246-P1: the raw-V-mask plus T-pivot pair remains 0/0/0 at substeps 3--6 under both labels | **CONFIRMED** |
| R246-P2: the atomic complete unit removes the old live-W refusal and completes OMT-4 kt=8--10 | **REFUTED**: both labels complete kt=7 and refuse at kt=8 |
| R246-P3: the five stage-transport operands are exact after the pair, so `zFv` is exact | **REFUTED**: the three-dimensional V mask and `zFv` each differ on 668 slots |
| R246-P4: full Decision-96 landing gates run only after P1--P3 | **NOT ACTIVATED** because P2 and P3 failed |
| R246-P5: every classifier plant fires | **CONFIRMED** |

## Direct pair and atomic boundary

The direct production-shaped pair reproduces round 245 exactly. For every
substep 3, 4, 5, and 6, midpoint V, midpoint live V depth, and `zhV` each
have 0 unequal values and maximum absolute error 0 under both labels.

The complete atomic candidate then completes OMT-4 kt=1--7 under both labels.
At kt=8 both runs refuse at the same guard:

```
raw-mesh e3w_int must contain only finite values > 0
```

There is no completed kt=8 state to score. The preregistered claim that the
pair was the complete compensating partner is therefore refuted; the failure
boundary did not move later.

## First remaining statement

The five-operand replay uses the current tree's production D114 resolved
geometry route. It reproduces the executable candidate transport exactly.
The table is identical for **independent** and **given NEMO's entry**:

| operand, in NEMO statement order | unequal / support | maximum absolute error | argmax |
|---|---:|---:|---|
| `e1v` | 0 / 8,589 | 0 | `[1,49]` |
| live `e3v(Kmm)` | 0 / 226,637 | 0 | `[1,49,0]` |
| `vv(Kmm)` | 0 / 226,637 | 0 | `[1,49,0]` |
| `zvb` | 0 / 8,589 | 0 | `[1,49]` |
| three-dimensional `vmask` | **668 / 399,600** | **1** | **`[147,29,0]`** |
| completed `zFv` | **668 unequal** | candidate/oracle maxima both 3,690,843.421062493 | sign mismatch on 647 slots |

The correction inputs `vn_adv`, `r1_hv`, and `vv_b` are each 0 unequal.
Replacing the recorded V mask closes the completed `zFv` replay exactly.
Thus the first remaining source-ordered non-bit operand is the stage
transport's three-dimensional V mask, not the now-exact surface-mask/depth
pair. This also reconciles the earlier resolved-boundary diagnostic: that
diagnostic supplied a raw mask to the replay and therefore could not identify
the executable stage mask; the final five-operand probe measures the actual
argument and supersedes it for attribution.

`held_classification.json` reports
`HELD_R246_STAGE_VMASK_AND_KT8_LIVE_W`. Its label-coverage, direct-closure,
boundary, V-mask, and `zFv` plants all exit red.

## Instrument retractions

Three instrument-only failures are retained and do not support physics
claims:

1. A monolithic multi-variant run exhausted the JAX/LLVM materialisation
   budget. The measurement was split into one process per label and section.
2. The first operand replay selected round 232's retired reconstructed mesh
   route and could not reproduce the passive candidate. It was replaced by
   the current D114 production resolved-mesh route.
3. A stale private-hook name and an object-identity check rejected equivalent
   rebuilt OMT-4 cards. The gate now binds to current hooks and compares the
   resolved geometry contents. Neither refusal produced a cited number.

## Landing disposition

Nothing lands in production. The atomic candidate fails two frozen landing
prerequisites: it still refuses at kt=8, and its completed stage V transport
still carries 668 wrong mask slots. Per preregistration, the OMT-1..4, rung-0,
rung-10, GYRE-year, DINO, tank, and moved-row landing gates were not activated;
running them cannot turn a failed prerequisite into a landing.

## Evidence, tests, citations, and review

Focused round-241/245/246 and literal-continuity tests pass 34/34. The
once-only `tests/ocean/fidelity -n 12` battery reached 99% before one known
long prediction-plant worker stopped making progress: 3,197 passes, seven
skips, four registered pre-existing failures, one running node without a
verdict, and seven not scheduled when interrupted. The four failure IDs are
the SI3 scalar-math provenance gate, GYRE round-129 certified-year record,
allow-dirty stamp scope, and worktree-stamp ratchet. Serial isolation results
reproduce the four registered messages respectively: `A MY_SRC is not
verbatim`; `the certified year harness moved after the registered members
ran`; the same 13 allow-dirty offenders; and the same 13 unstamped emitters.
The prediction-plant node again produced no verdict when run alone; after its
Python process disappeared while the harness remained blocked, that isolated
attempt was interrupted and retained. No second full battery was run.

The round receipt citation gate passes 2/2 citations and the campaign default
passes 274/274, both with zero unmapped citations. Shifting the compiled
stage-transport citation by two lines exits red. `codex exec --sandbox
read-only` failed before reading the diff with `failed to initialize
in-process app-server client: Read-only file system (os error 30)`.
**Independent review unavailable in-sandbox.** Exact logs and exit markers
are retained in the evidence root.

## OPEN

Route the stage transport's three-dimensional V mask through the production
D114 resolved mesh operands, then rerun the same five-operand replay and the
same atomic complete unit. The immediate falsifier is unchanged: if `vmask`
and `zFv` do not fall from 668 to 0 unequal, or if OMT-4 still refuses at
kt=8, the mask is not the complete partner and no production landing follows.
No acquisition is needed. Decision 116 remains pending with the user and is
not re-emitted by this round.
