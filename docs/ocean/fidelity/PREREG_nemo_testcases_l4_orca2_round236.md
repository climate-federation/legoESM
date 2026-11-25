# ORCA2 round 236 preregistration — OMT-4 fold-local slow-V split

Date: 2026-10-10. Frozen base: `956afff2723d40cc909d55e6cc6944212124d45b`.
Scope: measurement only. This round changes no model file, card, deck, carried
state, selector, stabiliser, sea-ice field, or `unmeasured_features` tuple.
Decisions 114 and 115 remain pending and are not acted on.

## Frozen question and source order

Round 235 measured the first complete-domain external-mode difference at
substep 1: `slow_v` differs on 35 northern-fold halo cells by at most
`1.6557659420864476e-06 m s-2`, with zero interior differences. Split that
forcing in NEMO's executing order and name the first unequal operand.

The compiled OMT-4 `stp2d` builds the vector-form depth average from
`e3v_3d`, the completed 3-D V RHS, `vmask`, and `r1_hv_0` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stp2d.f90:194-200`, applies
baroclinic drag at `:218-221`, and adds wind at `:223-230`. The compiled
external solver copies `Ve_rhs` to `zv_frc` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:289-294`, then
subtracts `zv_trd * ssvmask` at `:323-328`. This is the frozen source order:

1. reference V-face thickness `e3v_3d`;
2. completed V RHS at `Krhs`;
3. three-dimensional `vmask`;
4. stored reciprocal `r1_hv_0`;
5. depth-averaged `Ve_rhs`;
6. post-drag `Ve_rhs`;
7. post-wind/incoming `Ve_rhs`;
8. barotropic Coriolis trend `zv_trd`;
9. raw two-dimensional `ssvmask`;
10. final `zv_frc` consumed as substep `slow_v`.

## Existing record and instrument

No NEMO acquisition is authorised or expected. The admitted OMT-4 twin under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_{a,b}_np2/`
already contains `oracle_slow_forcing_kt00000001.bin`,
`oracle_bt_substeps_kt00000001.bin`, the rank-complete stage frames, and the
raw mesh/mask operands. The checker parses the self-described record and
requires exact defined twin payloads before measuring it.

Candidate operands come only from the existing passive completed-stage trace.
An ordinary run and traced run must have array-identical completed T/S/u/v/SSH
states before an intermediate is read. Both required labels are measured
separately: **independent OMT-4** and **given NEMO's entry OMT-4**. Execution is
CPU, production JIT, fp64/libm. Comparisons are exact; the `2e-10` floor is
reported separately. Only the 35 cells selected by the admitted final
`slow_v` difference are the owner support; full fold-band counts are retained
as a control.

## Frozen predictions and falsifiers

- **R236-P1 — admission and passivity.** CONFIRM if both inherited slow-forcing
  and substep streams parse, their defined twin payloads are exact, and every
  completed traced state equals its ordinary counterpart under both labels.
  REFUTE on a missing/malformed stream, twin difference, or one moved state
  bit. Plants that alter one record bit and one passivity result must refuse.
- **R236-P2 — first operand.** Prediction: the depth-average, drag, and wind
  boundaries are bit-exact on the 35-cell owner support; the first unequal
  operand is the post-copy Coriolis-removal unit (`zv_trd`, raw `ssvmask`, or
  their source-ordered product) at `dynspg_ts.f90:323-328`. CONFIRM if all
  earlier rows are exact and the first unequal row is in that unit. REFUTE by
  retaining the earlier first unequal row and its exact count/max/argmax.
- **R236-P3 — round-217 unit.** Prediction: the complete private atomic unit
  closes the final `zv_frc` only as the already registered cancelling pair;
  neither an incoming-forcing-only nor raw-mask-only replay closes the final
  row. CONFIRM if cumulative replay has that signature. REFUTE if one operand
  alone closes it or the complete replay remains unequal.
- **R236-P4 — labels.** Prediction: independent and given-entry labels name
  the same first operand with identical counts and maximum. REFUTE if they
  disagree; retain both and claim no shared owner.
- **R236-P5 — disposition.** This is measurement only. The final package diff
  must be empty. A cited statement may be proposed later only after a frozen
  arm moves the registered boundary under Decision 96; Decisions 114/115 are
  untouched.

## Outcomes

- `HELD`: the table names a first operand under both labels; no package change.
- `STOPPED_FOR_RECORD`: an exact required operand is absent; name the missing
  stream and write a fail-closed acquisition.
- `FAILED`: admission, passivity, label coverage, source order, or a plant
  fails.

ASKED choices: Decisions 103, 109, 113 and standing Decision 96. Pending and
untouched: Decisions 114 and 115. UNASKED choices: empty.
