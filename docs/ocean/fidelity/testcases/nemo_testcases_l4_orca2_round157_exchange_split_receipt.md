# ORCA2 round 157 — external-mode cyclic / north-fold split receipt

Date: 2026-10-05
Lane: ORCA2 hierarchy, ocean only
Base: `8a7ac4e5f`
Preregistration: `PREREG_nemo_testcases_l4_orca2_round157.md`
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round157`

## Verdict

**HELD.** Production physics is unchanged. The private round-156 complete
seven-field association was split in the compiled order into U cyclic, U
north-fold, V cyclic, and V north-fold measurement arms. The first
source-ordered non-bit operation is the U cyclic association at external
substep 1: 24 cells differ, with maximum absolute difference
`5.421010862427522e-20`. The large finite growth is owned by the V north-fold
operation: 180 association cells differ by at most
`0.00040579965574751963`, its next V-transport consumer differs on 68 cells
by at most `155776.5627856178`, and stage-1 T differs on 23,505 cells by at
most `0.1774972822409795`.

Every number in this receipt is labelled **independent**: the experiment starts
from legoESM's rung-0 initial state and does not mix in NEMO's recorded entry
state. Sea ice and the ORCA2 card's `unmeasured_features` tuple are untouched.

## Compiled source order

The compiled rung-0 call associates seven live fields together in
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`. The
resolved `jpni=2`, `jpnj=1`, `nn_hls=2`, `ln_nnogather=.TRUE.` path
first performs the east/west MPI send and fill in
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:1961-1979`, then calls
the north-fold exchange in
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:2105-2113`.

The T-pivot no-gather branch documents the duplicated U/V polar lines in
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1561-1577`, packs the
per-field extra lines and sends them in
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1629-1669`, and finally
applies the field sign while overwriting the selected pivot cells in
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1747-1766`.

The legoESM helper remains private and complete by default at
`barotropic_latlon_cgrid.py:689-739`. No card selects any round-157 component.
The hook rejects an unknown component and rejects overlap with either the
complete or one-field association arms.

## Frozen predictions

All six preregistered predictions were confirmed.

| prediction | result | independent evidence |
|---|---|---|
| R157-P1 observer is passive | **CONFIRMED** | T, S, u, v, ssh, uu_b, and vv_b are array-exact |
| R157-P2 U cyclic is non-vacuous | **CONFIRMED** | substep-1 association: 24 cells, max `5.421010862427522e-20`; next U transport: 22 cells, max `4.3655745685100555e-11` |
| R157-P3 U fold is non-vacuous | **CONFIRMED** | substep-1 association: 56 signed-zero cells; first non-bit U transport is consumer substep 3 after association substep 2: 34 cells, max `15393.357397114101` |
| R157-P4 compact V cyclic is a no-op | **CONFIRMED** | all 65 association rows, all downstream consumer rows, and every stage-1 field are array-exact |
| R157-P5 V fold reproduces round 156 | **CONFIRMED** | stage-1 T max is exactly `0.1774972822409795`; association and first consumer counts are 180 and 68 |
| R157-P6 components compose to the complete image | **CONFIRMED** | U cyclic→fold and V cyclic→fold each differ from the complete field image on 0 cells over all 65 substeps |

The U-fold signed-zero row was not misclassified: its 56 cells have different
bits and maximum arithmetic difference 0.0. The comparison-bit and
signed-zero controls therefore both prove that the classification is
bit-level, not magnitude-only.

## Component walk

All rows below are **independent**.

| component | first association difference | first downstream difference | stage-1 T |
|---|---|---|---|
| U cyclic | substep 1: 24 cells, max `5.421010862427522e-20`, argmax `[28,0]` | U transport at substep 2: 22 cells, max `4.3655745685100555e-11`, argmax `[61,0]` | 360 cells, max `3.552713678800501e-15` |
| U north fold | substep 1: 56 signed-zero cells | U transport at substep 3 after association substep 2: 34 cells, max `15393.357397114101`, argmax `[147,133]` | 10,596 cells, max `0.0016328912725630529` |
| V cyclic | exact through all 65 substeps | exact | exact |
| V north fold | substep 1: 180 cells, max `0.00040579965574751963`, argmax `[148,135]` | V transport at substep 2: 68 cells, max `155776.5627856178`, argmax `[148,135]` | 23,505 cells, max `0.1774972822409795` |

This resolves round 156's branch question without promoting a partial boundary
call to production physics. U cyclic is the first source-ordered non-bit
statement in this split. V north fold is the magnitude owner. The complete
association still has the round-156 terminal kt=8 non-finite result, so no
single component can land.

The aggregate artifact is
`exchange_split.json`
(SHA-256 `5446209923c391103adb103cf7e40a7d95d17b4cd7689cbbc68e0d4c44085acc`).
The round-92 and pinned round-90 input locations contain the same 80 inherited
round-84 frame shards byte for byte; no cross-record numerical mixture was
used.

## Production-path gates

The rung-0 ten-step production ladder remains unchanged: 200/200 rows are
unchanged, 0 AT-BAR rows leave the bar, and its first non-bit row remains kt=1
stage-1 T. The comparison artifact is
`rung0_compare.json`
(SHA-256 `e30f82f3f434c01769bed2b5950a504790424ce7734ec7180d7a7156e0d113eb`).

The shared GYRE path is also exact. Its ten-step gate compares 70 certified
rows with 0 moved rows, maximum worsening 0 ULP, and unchanged first debt at
kt=3. Both residual archives have identical keys and every array is
`np.array_equal`. All 30 daily snapshots in the required 30-day member are
byte-identical to round 156; day-30 SHA-256 is
`b017623dea468af4f4ba31761148aa52e478d60196828443b56ceeee36534180`.

No DINO, VORTEX, tank, generic-card, rung-7 candidate, or production landing
gate was run: the change is a private observer/hook split and the
preregistered terminal rule requires **HELD**. Running downstream landing
gates cannot turn the terminal complete-association refusal into a landing.

## Controls, citations, tests, and review

The comparison-bit, signed-zero, unknown-selector, overlap, and composition
plants all fire. The composition plant perturbs one V element and the gate
requires exactly one differing cell. Focused round-156/157 tests pass 16/16.

The canonical citation gate passes with 274 citations, zero failures, zero
unmapped citations, and zero failing map entries. Its planted two-line shift
of the single-line ocean-model citation fails with
`SYMBOL-NOT-AT-LINE`. Both edited model files and the cumulative receipt were
re-anchored mechanically with `difflib.SequenceMatcher`; the extent change
of the deliberately enlarged association helper was audited explicitly.

The one required `tests/ocean/fidelity -n 12` invocation is **INCOMPLETE**,
not PASS. It collected 2,633 tests and printed four failure markers at the same
suite positions as round 156's registered SI3 scalar-math provenance, GYRE
round-129 spread-record stamp, round-35 escape-scope, and worktree-stamp reds.
After the last printed 95% progress line and prolonged quiescence, no pytest
process remained but the PTY had not closed; it was interrupted once to
recover the terminal. There is no terminal summary and the battery was not
rerun. No round-157 focused test failed.

The required separate review is **independent review unavailable in-sandbox**.
The exact review error is: `failed to initialize in-process app-server client:
Read-only file system (os error 30)`. No independent PASS is claimed.

ASKED choices: none. UNASKED choices: empty. ACQUISITION_NEEDED: none.

## OPEN

1. Keep production unchanged. U cyclic is now the first source-ordered
   non-bit operation; walk its 24 compact closure cells against the exact
   two-rank east/west exchange ordering before touching the larger fold term.
2. Then split the V north-fold operation in the compiled no-gather order:
   extra-line population, neighbour selection, pivot overwrite, and sign
   application. Require the 180-cell association row and 68-cell transport
   carrier to move one registered sub-operation at a time.
3. A partial exchange or fold remains a measurement arm only. Re-run the
   complete association and rung-0 ladder before considering any landing.
