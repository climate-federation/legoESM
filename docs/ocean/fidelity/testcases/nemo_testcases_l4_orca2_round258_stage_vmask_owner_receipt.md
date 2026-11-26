# ORCA2 round 258 — pivot V-transport mask-owner receipt

Date: 2026-10-12. Base: `d401dd65e`. Preregistration commit:
`98ebf18b1`. Measurement commit: `b54be9688`. Disposition: **HELD**.

This round changes no model, card, deck, selector, carried-state contract,
stabiliser, sea-ice declaration, or `unmeasured_features` entry. It is an
offline CPU fp64/libm production-JIT measurement on OMT-4 with the current
complete private fold/transport unit. Every model number below is labelled
separately as **independent** or **given NEMO's entry**. No NEMO acquisition
was needed.

The first invocation pointed the gate at the hierarchy's deck-only directory;
preflight refused because its required input files were absent, before model
construction or any numerical result. The admitted round-230 run directory
was then supplied and all measurements below used only its self-described
record.

## Compiled statement and existing instrument

NEMO constructs the external correction `zvb` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279`, then writes
the V transport as the ordered product `e1v`, live `e3v(Kmm)`, `vv(Kmm)`,
`zvb`, `vmask` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`.
The domain mask construction makes the 3-D V mask from adjacent T masks at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:206-213`, derives its
surface mask and applies the boundary association at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:228-232`, and its compiled
V-fold rule is
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:684-721`.

Before adding code, the round searched the committed ORCA2 instruments. It
reused round 232's five-operand parser/replay and round 257's passive
completed-stage trace. The only new seam is a private wrapper around the
production stage-transport helper which replaces the compact `v_mask_3d`
operand by the already-landed D114 resolved mask. It neither observes nor
changes an in-executable intermediate. Traced and untraced completed states
are array-identical for every field under both labels.

The admitted operand record is
`oracle_rkstage1_transport_operands_kt00000001.bin`, SHA-256
`5cd9ef3f875cf54f79ec1e3896cd3d738505046d4175c2a4bb51fd25116afb79`.
Both candidate-side and oracle-side five-operand products replay their stored
`zFv` bit-for-bit over all 399,600 slots.

## Source-ordered operand result

The table scores the exact 668 finite-valued stage-1 pivot-row slots on which
the current private unit's `zFv` differs from NEMO. Both labels produce the
same table. `unequal` includes bit differences; no row contains a non-finite
value or a signed-zero-only mismatch.

| operand in written order | unequal / 668 | max abs | argmax `[j,i,k]` |
|---|---:|---:|---:|
| `e1v` | 0 | 0 | — |
| live `e3v(Kmm)` | 0 | 0 m | — |
| `vv(Kmm)` | 0 | 0 m/s | — |
| `zvb` | 0 | 0 m/s | — |
| `vmask` | 668 | 1 | `[147,29,0]` |

Thus the first valued non-bit operand is the stage transport's compact 3-D V
mask. This is not the raw geometry path: the D114 route already makes live
`e3v` exact here, while the stage transport still receives a different compact
mask object. NEMO consumes the folded domain V mask in the cited product; the
private source-exact arm routes this consumer to that already-resolved source.

## Causal arm and the two labels

The present-tree baseline reproduces round 257's physical-half boundary to
every printed digit. The resolved-mask arm makes both the 2,700-slot pivot-row
mask and `zFv` bit-exact. The completed-stage result is at the campaign floor,
but is deliberately not called bit-exact:

| label | baseline T max / unequal | baseline S max / unequal | arm T max / unequal | arm S max / unequal |
|---|---:|---:|---:|---:|
| independent | 0.17735225873138316 K / 668 | 3.283688696376231 PSU / 668 | 2.220446049250313e-16 K / 10 | 3.552713678800501e-15 PSU / 1 |
| given NEMO's entry | 0.17735225873138316 K / 2,469 | 3.283688696376231 PSU / 668 | 2.220446049250313e-16 K / 1,811 | 3.552713678800501e-15 PSU / 1 |

The extra T bit counts under **given NEMO's entry**, and the residual arm bit
counts under both labels, have only rounding/signed-zero-scale magnitude.
They are below the `2e-10` K campaign floor. R258-P1 through P4 are confirmed;
the causal conclusion is that this mask operand owns the round-257 magnitude.

## Round-247 / round-257 reconciliation

There is no contradiction between the historical rows:

| round | arm | support | record | `vmask` unequal | `zFv` unequal |
|---|---|---|---|---:|---:|
| 247 | resolved-stage-mask private arm | rank-0 native pivot row, NEMO-live stage-1 slots | `5cd9ef3f...afb79` | 0 | 0 |
| 257 | current complete unit, without resolved-stage-mask arm | same | same | not replaced | 668 |
| 258 | current complete unit, with resolved-stage-mask arm | same | same | 0 | 0 |

Round 258 reconstructs round 247's exact mask and transport rows using the
same record while also reproducing round 257 before enabling the arm. R258-P5
is confirmed. Round 247 compared the private resolved-mask arm; round 257
reported the production-shaped held unit without that arm.

## Controls, gates, tests, and review

All ten non-vacuity plants refuse: dirty/wrong worktree, missing label,
reordered compiled operands, wrong pivot support, failed round-257
reproduction, broken candidate replay, broken round-247 reconciliation,
false owner selection, failed mask closure, and a one-bit operand
perturbation. The classifier status is
`HELD_R258_STAGE_VMASK_OWNER_NAMED`. Evidence SHA-256 values are:

- independent: `3c62f42ee9a6ce6185e73fdaafaa43ee27c5f234739e2c766dcfee58a0ac3381`
- given NEMO's entry: `274d87af2011c0b3c3820184db731ff3aa4f3f25d9cbd67d5f9209ffdd8f5999`
- classification: `088e0500f3f2b0b654cbb65c741e330cec5aa1f7945d8ba4ebb33bd74a82b8ac`
- ten plants: `277cd2b3f0ea37291950836e19453b82e8fcd4721f1b2940529f248350808fe8`

The focused round-232/247/257/258 set passed `39/39`; its log SHA-256 is
`ec11fa96e8782a8df1ac78036c07ebcf5dcad3aee4ea954583b2becf10f7576b`.
This round has no `packages/` diff, so no OMT, rung-0, rung-10, GYRE, DINO,
or tank trajectory can move and their landing gates are not activated.

The round receipt citation gate passed all five compiled citations with zero
failures or unmapped citations; the cumulative default receipt passed all 274
citations with zero failures or unmapped citations. The rigid two-line shift
of the transport-product citation refused with exit 1. Citation-gate tests
passed `17/17`. The receipt/default/plant/test evidence SHA-256 values are
`3d9d0b166b106532eeae4d1ba9ab9f4a716502414a84612cf62f770467790158`,
`8599ea52b8e9acc614fae1b0f67250edf9ba13e05fe61a39f306dbdbd49ba8c6`,
`c4347bf60e8f65fe7bb1c3c434b6503683fabbfd7904efb964f113e1e1bc5906`,
and `36b8964e8ea37da9f51d04ffbfd749613a622549e5ebe35b5df9a7fe813d80c7`.

The once-only `tests/ocean/fidelity -n 12` battery collected 3,348 tests and
reached 99% before an xdist worker stopped producing output; it was terminated
once and not repeated. Its preserved log SHA-256 is
`5edeab712c94ffe142208f1ff368a40660455fe17c521a1488b17aecec4423aa`.
The four visible reds reproduced serially in isolation (`4 failed in 7.57 s`;
log SHA-256
`97c51c9ceb2e8a0fcbf4cfebcfd515a29929468923dd201c1dfcd91748b01ce0`):

- `test_nemo_si3_scalarmath_v2_gate.py::test_full_v2_gate_and_plants` — known
  SI3 provenance red, `A MY_SRC is not verbatim`;
- `test_nemo_testcase_l2_gyre_round129_spread_floor_gate.py::test_record_backed_gate_passes`
  — known registered-record/harness-stamp red;
- `test_nemo_testcase_round35_stamp_scope.py::test_every_driver_that_arms_the_escape_scopes_it`
  — known allow-dirty scope ratchet; and
- `test_nemo_testcase_worktree_stamp.py::test_every_report_emitter_stamps_the_worktree`
  — known worktree-stamp ratchet.

The requested separate `codex exec --sandbox read-only` invocation failed
before review with `failed to initialize in-process app-server client:
Read-only file system`; **independent review unavailable in-sandbox**. Its log
SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN

R258-P6 is enforced: the resolved stage-mask arm and both held external-mode
associations remain private components of the atomic unit while Decision 116
is pending. The next authorised round must integrate this already-proved mask
route into that atomic unit, re-score kt=1 stage 1, the growth table, and
kt=8..10, and land only if the complete unit passes Decision 96 and the full
card gates. No acquisition is needed. No partial mask or association lands
from this measurement.

ASKED choices: none. Unasked configuration choices made: none. Round
disposition: **HELD**.
