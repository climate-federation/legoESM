# ORCA2 round 247 — resolved stage mask closes `zFv`, atomic unit blows up at kt=5 stage 1

Date: 2026-10-11. Frozen base: `89c1bb679`. Preregistration commit:
`9dc2d937e`. Measurement candidate: `6c6d5e9c2`. Candidate retraction:
`476fe9624`. Status: **HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round247/`.

This round changes no effective package/model file, deck, selector, carried
state, stabiliser, sea-ice setting, or `unmeasured_features` entry. The exact
stage-mask candidate was measured and then removed because the complete atomic
unit failed its frozen completion predicate. The retained changes are the
preregistration, measurement gate, controls, tests, and this receipt. Every
number below is reported separately as **independent** and **given NEMO's
entry**; the two labels give identical results. No NEMO acquisition was
required.

## Compiled statement and frozen candidate

The admitted rank-complete record first constructs the barotropic V correction
from `vn_adv`, `r1_hv_0`, and `vv_b` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279`, then forms
the stage transports from the live face thickness, corrected velocity, and
three-dimensional face mask at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`.

The candidate routed the stage transport's U/V masks through the existing
D114 `nemo_qco_resolved_mesh_operands` result. It did not create a second
builder. Round 245's raw surface-V-mask plus T-pivot north-neighbour arm and
round 217's complete fold/transport unit remained atomic around that single
change.

## Frozen predictions

| prediction | result |
|---|---|
| R247-P1: stage `vmask` and completed `zFv` fall from 668 to 0 unequal under both labels | **CONFIRMED** |
| R247-P2: midpoint V, live V depth, and `zhV` remain 0/0/0 at substeps 3--6 | **CONFIRMED** |
| R247-P3: the complete unit crosses the old kt=8 guard and completes kt=10 under both labels | **REFUTED**: both labels complete kt=4 and become non-finite at kt=5 stage 1 |
| R247-P4: the full Decision-96 landing gates run only if P1--P3 hold | **NOT ACTIVATED** because P3 failed |
| R247-P5: every held-classifier plant fires | **CONFIRMED** |

## Exact transport closure

The five-operand replay uses the production D114 operand route and reproduces
the candidate executable exactly. This table is identical under both labels:

| operand, in NEMO statement order | unequal |
|---|---:|
| `e1v` | 0 |
| live `e3v(Kmm)` | 0 |
| `vv(Kmm)` | 0 |
| `zvb` | 0 |
| resolved three-dimensional `vmask` | **0** (was 668) |
| completed `zFv` | **0** (was 668) |

The correction inputs `vn_adv`, `r1_hv`, and `vv_b` are also 0 unequal. The
separate source-ordered depth pair remains exact at substeps 3, 4, 5, and 6:
midpoint V, midpoint live V depth, and `zhV` each have 0 unequal values and
maximum absolute error 0. The exact stage transport is therefore exonerated as
the remaining direct boundary, but it exposes a downstream instability in the
complete unit.

## First downstream boundary

The ordinary atomic ladder completes kt=1--4 and refuses while entering kt=5
under both labels with `raw-mesh e3w_int must contain only finite values > 0`.
The boundary probe adds no executable observer: it advances the same ordinary
state to kt=4, reads the already-supported completed stage-1/2 returns, and
reconstructs the guard arithmetic offline.

Every kt=4 candidate boundary (entry, stages 1--3) is finite, with 0 invalid
live-W thicknesses. The kt=5 entry is also finite. The first returned invalid
boundary is **kt=5 stage 1**, identically under both labels:

| quantity | result |
|---|---:|
| invalid live `e3w` slots | 476,557 |
| first invalid `[j,i,k]` | `[1,49,0]` |
| `ht0` there | 729 m |
| raw `e3w0` there | 10.00035061113249 m |
| stage-1 SSH / `r3t` / stretch / live `e3w` there | non-finite |
| non-finite SSH cells | 26,640 |
| non-finite T / S cells | 430,552 / 430,552 |
| non-finite U / V slots | 803,640 / 799,200 |

NEMO's admitted kt=4 and kt=5 entry/stage-1/stage-2 states have zero invalid
live-W slots and zero non-finite T/S/u/v/SSH values. This is not a marginal
non-positive-thickness event: the complete candidate produces a global
stage-1 blow-up from a finite kt=5 entry. Damage at the guard is downstream;
the first writing statement inside kt=5 stage 1 remains OPEN.

## Landing disposition and retraction

The package candidate was removed after measurement. A partial exact-mask
landing would split the same cancelling unit whose atomic completion was the
frozen predicate. Consequently OMT-1..4, rung-0, rung-10, GYRE-year, DINO,
tank, moved-row, and month landing gates were not activated: those gates cannot
turn a failed atomic prerequisite into a landing. The final tree has no model
diff from the frozen base, so GYRE and every other card retain their base-tree
trajectory by construction.

Two instrument corrections are retained explicitly. The first boundary run
completed the candidate calculation but its checker treated an oracle frame
dictionary as a model-state object; no science number was emitted. The fixed
reader re-used the same admitted record. The next run proved all kt=4 states
valid, so the committed probe was extended to the already-supported kt=5
exposed stages; it then named stage 1 without inserting an executable callback.

## Controls, tests, citations, and review

The held classifier passes and all seven named plants exit red:
label coverage, direct closure, terminal coverage, NEMO guard, boundary stage,
V mask, and `zFv`. Focused round-232/245/246/247 tests pass 32/32.

The once-only `tests/ocean/fidelity -n 12` battery reached 99% with 3,218
passes, seven skips, four registered pre-existing failures, and the known
`test_prediction_plant_is_fail_closed` worker stalled without a verdict. A
ten-minute serial isolation of that exact node also collected and then timed
out without a verdict. Serial isolation reproduced the other four known reds:
the certified GYRE-year harness stamp moved, SI3 `A MY_SRC is not verbatim`,
the same 13 unscoped allow-dirty drivers, and the same 13 unstamped report
emitters. No second full battery was run.

The campaign default citation gate passes 274/274 and the round receipt gate
passes 2/2, both with zero unmapped citations. Shifting the compiled
stage-transport citation by two lines exits red. `codex exec --sandbox
read-only` failed before reading the diff with `failed to initialize
in-process app-server client: Read-only file system (os error 30)`.
**Independent review unavailable in-sandbox.** Exact logs are retained in the
evidence root.

## OPEN

Walk OMT-4 kt=5 stage 1 from the finite entry state, offline from passive
completed states, in compiled stage-program order. The next probe must name
the first writer that turns the finite kt=5 entry into the global non-finite
stage-1 state, starting at `[j=1,i=49]` but checking the full support. Keep the
resolved stage-mask candidate and complete round-217 unit private and atomic;
re-score them only after that downstream owner is transcribed. Decision 116
remains pending with the user and is not re-emitted.
