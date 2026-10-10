# ORCA2 round 235 — OMT-4 external-mode alpha owner

Date: 2026-10-10  
Base: `b4d6c1ea0`  
Status: **STOPPED_FOR_DECISION** (measurement complete; no production physics landed)

## Scope and frozen claims

This round executes the round-234 OPEN: discriminate a fold-local external-mode
error from a global operand by a source-ordered per-substep table on OMT-4.  The
original preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round235.md`.  After that
table refuted the expected slow-forcing location, the alpha replay was separately
frozen in
`docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round235_alpha_replay.md`
before the replay ran.  The shipped ORCA2 card and every sea-ice selector are
untouched. Decision 114 remains pending and was not acted on.

Every result below is reported under both required labels. "Independent" uses
legoESM's corrected rung-0/OMT initial state; "given NEMO's entry" uses the
recorded NEMO entry. The two labels are never pooled.

## Executing source and deck

The compiled OMT-4 `dynspg_ts` copies the completed three-dimensional stage
forcing into the external mode at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:289-294`, removes the
two-dimensional Coriolis contribution at `:323-328`, forms the midpoint depths,
transports, and SSH update at `:533-560`, then updates the vector-form velocity
at `:669-682` and performs the post-loop association at `:738-744` and
`:747-756`. The first global non-bit boundary measured below is instead the
half-step SSH interpolation at `:604-612`.

That statement calls `ts_bck_interp`; for substep 3 and later, its executing
branch derives all four weights from `rn_bt_alpha` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:1522-1557`. The
resolved OMT-4 deck selects filter 3 and `rn_bt_alpha=0.09` in
`orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2/namelist_cfg:383-387`.
legoESM instead defines the shared default as the GYRE value 0.07 in
`barotropic_latlon_cgrid.py:112-125`, and the active filter-3 call does not pass
a card value at `barotropic_latlon_cgrid.py:3389-3390`.

## Instrument and controls

The committed round-235 table reads the admitted OMT-4 frame/substep streams,
runs the production CPU/fp64/JIT path with a private atomic-unit arm, and compares
rank 0's owned 148 x 90 slab. It certifies every completed stage state against
the untraced path before reading an intermediate. No NEMO acquisition was
needed.

Three instrument refusals are retained as evidence rather than hidden:

- `instrument_refusal_early_trace.log`: the first hook was upstream of the
  completed stage and therefore had no passive state to certify.
- `instrument_refusal_rank_slab.log`: the candidate T field still included a
  compact halo while the oracle was the rank-owned slab.
- `instrument_refusal_all_fields_rank_slab.log`: U/V needed the same explicit
  slab conversion.

After those mechanical fixes, all completed T/S/u/v/SSH states were
array-identical with and without the side output. The label-coverage and
owner-class controls fire. The alpha gate's `deck-alpha` plant refuses a deck
changed to 0.07, and its `alpha09-bit` plant refuses one introduced ULP.

## Per-substep result

The first complete-domain difference is still fold-local: substep 1 `slow_v`,
35/13,320 cells unequal, maximum 1.6557659420864476e-06, with 0/13,050
interior cells unequal. Substep 2 remains fold-local. The first non-rounding
interior difference is substep 3 `eta_pgf`:

| label | owner class | first global operand | unequal complete | unequal interior | complete/interior max (m) | fold max (m) | argmax |
|---|---|---:|---:|---:|---:|---:|---:|
| independent | GLOBAL | substep 3 `eta_pgf` | 8,794/13,320 | 8,692/13,050 | 0.0031271104752883805 | 0.00022684827371018297 | [83,21] |
| given NEMO's entry | GLOBAL | substep 3 `eta_pgf` | 8,794/13,320 | 8,692/13,050 | 0.0031271104752883805 | 0.00022684827371018297 | [83,21] |

Thus R235-P2's class prediction is confirmed but its stronger predicted
location is **REFUTED**: the first global error is not substep-1 slow forcing.
R235-P1, P3, P4, and P5 are confirmed.

## Frozen alpha replay

The replay uses the four admitted NEMO SSH history operands and the exact
source-ordered multiply/add sequence of the compiled statement:

| replay | unequal / 13,320 | maximum absolute error (m) | result |
|---|---:|---:|---|
| shared alpha 0.07 | 8,794 | 0.0031271104752883805 | reproduces the table maximum at 0 ULP |
| deck alpha 0.09 | 0 | 0.0 | bit-exact |

The 0.09 weights are
`[0.7125337174, 0.2525882038999999, 0.03722244, -0.0023443612999999985]`.
The 0.07 weights are
`[0.6881859402, 0.2639740297, 0.04749412, 0.0003459100999999992]`.
This names the first global non-bit statement and its owner exactly: ORCA2 is
executing NEMO's `rn_bt_alpha=0.09` interpolation while legoESM evaluates the
same statement with GYRE's 0.07.

Evidence JSON SHA256:

- independent table: `166365a7f10fb99ce8b9d6c1d0d12c3a6c2ebe2d8e464431d2ae12ec93a587ef`
- given-entry table: `d7e041df74f88ee94b004e29ff0ca888933a05aee299379287fc69a9389a0290`
- classification: `a4f0c520351b250b75b76b062ffe2049ea0267919ff23b81cf6c979d5e7bcbf9`
- alpha replay: `4fa4b5770d3d5860662d4ca3b45aef5ccaddef4957db1205ec3287869daa3761`

All live evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round235/`.

## Landing disposition

No package change lands. The private measurement arm was committed so the
measurement is reproducible and then removed; the final tree has zero
`packages/` diff from `b4d6c1ea0`. Consequently no certified trajectory moves,
no row can leave the bar, and no GYRE/DINO/tank trajectory rerun is applicable
to this measurement-only disposition.

The source-exact correction needs a new explicit configuration operand (the
barotropic filter alpha) so ORCA2 can state 0.09 while GYRE retains its deck's
0.07. Creating that field is a configuration/API choice and is prohibited
without the user's decision. The statement is therefore named but not landed.

## Verification

- focused round-235 unit tests: pending final battery
- citation gate, default receipt and this receipt: pending final battery
- citation rigid-shift plant: pending final battery
- independent read-only diff review: pending
- final production-package diff from base: empty

## OPEN

1. User decision: add an explicit required NEMO barotropic-filter-alpha config
   field, state ORCA2/OMT-4 as 0.09, and preserve GYRE's 0.07 (recommended).
2. If approved, land that single source statement under the OMT-4, rung-0,
   rung-10, GYRE, DINO, tank, and moved-row gates; then resume the substep table
   after `eta_pgf`.
3. Decision 114 remains separately pending; do not couple its exact geometry
   unit to this alpha statement.
