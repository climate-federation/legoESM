# ORCA2 round 232 — stage-1 V-transport operand split

Date: 2026-10-10. Frozen base: `0838ebc78`. Preregistration commit:
`e666dc352`. Measurement commit: `14adf5c9a`. Status: **HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round232/`.

This round changes no package model file, card, deck, carried state,
stabiliser, sea-ice selector, or `unmeasured_features` tuple. Every number is
reported separately as **independent** and **given NEMO's entry**. The two
labels happen to give identical operand scores; both were executed and kept.
No NEMO acquisition was run.

## Compiled statement and admitted instrument

The executing ORCA2 record build first constructs the barotropic V correction
`zvb` in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279`, then writes
the stage-1 V transport as the ordered product `e1v`, live `e3v(Kmm)`,
`vv(Kmm)`, `zvb`, `vmask` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`.

The existing self-described rank-zero record has SHA-256
`5cd9ef3f875cf54f79ec1e3896cd3d738505046d4175c2a4bb51fd25116afb79`.
The candidate values come from the round-228/229 passive live-stage trace.
Under both labels, every completed state slot is array-identical between the
ordinary and traced runs. Offline replay exactly reproduces candidate `e3v`,
corrected V and `zFv` (zero unequal bits), so the instrument passes R232-P1.

NEMO's canonical writer stores zero outside its executable mask. The binding
scores therefore compare `e1v`, `e3v`, `vv`, and `zvb` only on NEMO-live
support, while `vmask` is scored everywhere. Raw full-array signatures include
canonical dry zeros and are diagnostic only; they are not owner votes.

## Five-operand result — prediction refuted

The preregistered R232-P2 prediction that `zvb` would be first is **REFUTED**.
The first unequal operand in NEMO statement order is live `e3v(Kmm)`:

| label | operand | unequal / support | max absolute | argmax |
|---|---|---:|---:|---|
| independent | `e1v` | 0 / 8,589 | 0 m | — |
| independent | `e3v(Kmm)` | 13 / 226,637 | 437.2169154512344 m | `[147,51,24]` |
| independent | `vv(Kmm)` | 0 / 226,637 | 0 m/s | — |
| independent | `zvb` | 8,589 / 8,589 | 0.00919996399945781 m/s | `[147,38]` |
| independent | `vmask` | 668 / 399,600 | 1 | `[147,29,0]` |
| given NEMO's entry | `e1v` | 0 / 8,589 | 0 m | — |
| given NEMO's entry | `e3v(Kmm)` | 13 / 226,637 | 437.2169154512344 m | `[147,51,24]` |
| given NEMO's entry | `vv(Kmm)` | 0 / 226,637 | 0 m/s | — |
| given NEMO's entry | `zvb` | 8,589 / 8,589 | 0.00919996399945781 m/s | `[147,38]` |
| given NEMO's entry | `vmask` | 668 / 399,600 | 1 | `[147,29,0]` |

All coordinates are zero-based rank-zero slab coordinates. The `e3v` argmax
is on the northern-fold row.

## Source split — the named statement

NEMO reads the V-point reference thickness directly from the domain input at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domzgr.f90:184-188`.
`E3v_0/e3v_0` names that stored three-dimensional field under the executing
partial-cell build at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/inc/domzgr_substitute.h90:114-121`. NEMO then
constructs the live V ratio at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:261-270` and consumes the
live thickness in the transport product above.

The candidate stage path instead calls the reconstructed-card builder at
`ocean_model_latlon_cgrid.py:1821-1825`; that builder derives `e3v_0` from
neighbouring T thicknesses at `vertical.py:647-654`. The already-shared
resolved builder reads the card's raw NEMO face thickness at
`vertical.py:673-748`.

On both labels:

- reconstructed reference `e3v_0` versus recorded raw reference: 13 / 226,637
  live slots unequal, maximum 437.2169154512344 m at `[147,51,24]`;
- raw recorded reference passed through the candidate live-thickness formula
  versus NEMO live `e3v(Kmm)`: **0 / 226,637 unequal**.

Thus the first non-bit statement is the source of `e3v_0`: this ORCA2 path
reconstructs a field NEMO reads explicitly. This is a source-exact attribution,
not an inference from endpoint magnitude.

## Later operands and why nothing lands

R232-P3 is not reached as preregistered because P2 is refuted, but the retained
post-hoc split shows the later debts rather than hiding them:

| correction input | unequal / support | max absolute | argmax |
|---|---:|---:|---|
| `vn_adv` | 8,589 / 8,589 | 24.341577728515905 | `[147,49]` |
| `r1_hv` | 35 / 8,589 | 0.03332976059679253 | `[147,29]` |
| carried `vv_b` | 0 / 8,589 | 0 | — |

Replacing NEMO's `vn_adv` reduces `zvb` to the remaining 35 reciprocal cells;
replacing all three correction inputs makes `zvb` exact. Even then the raw
mask debt remains. Replacing any one of `e3v`, `zvb`, or `vmask` cannot close
the full candidate `zFv`; replacing all five recorded operands replays NEMO's
`zFv` bit-for-bit (0 / 399,600 unequal), proving the record sufficient and the
product statement itself exonerated.

R232-P4 required one source-cited operand to close this boundary. It is
**REFUTED**: raw `e3v_0` is first, but later correction and mask debts are
independent. Therefore P5 is not activated, no private production candidate is
built, and no trajectory or Decision-96 landing gate is claimed. A partial
`e3v_0` landing here would knowingly split a cancelling unit.

## Controls, validation, and review

Both binding gates pass at `14adf5c9a`. Independent/given-entry JSON SHA-256
values are
`3c48ede34e2ec54d0c73f094fcb6341a7f3a4c58a334dd5a29ec5340db9d8347`
and `7e65534e3643ea91b912afbc53a5939ba5f7f76d4820a5bedce9a5345e9cb02b`.
The source-order, correction-replay, `e1v`-bit, and `e3v`-bit integrated plants
fire; the isolated binding `e3v` plant log SHA-256 is
`2049b2ee5d5f933fa1f775ecd35b7fa0c109c3bf399a1ef85b9b2cc780200d77`.
The focused unit test perturbs and detects each of the five operand scorers.
The multi-run integrated plant process was stopped after compiler-memory
exhaustion; it is not called a complete plant battery.

Independent review was attempted with a separate `codex exec --sandbox
read-only` process and exited 1 before reading the diff: `failed to initialize
in-process app-server client: Read-only file system (os error 30)`.
Independent review unavailable in-sandbox. Log SHA-256:
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

Citation and pytest results are recorded in the final validation commit.

## Preregistered predictions

| ID | disposition |
|---|---|
| R232-P1 | **CONFIRMED**: both labels are passive; all three offline candidate replays are bit-exact. |
| R232-P2 | **REFUTED**: `e3v` is first, not `zvb`; exact counts are retained above. |
| R232-P3 | **NOT REACHED as conditional**; post-hoc split retained and explicitly labelled. |
| R232-P4 | **REFUTED**: no single source-cited operand closes `zFv`. |
| R232-P5 | **NOT ACTIVATED**: no model candidate exists. |

## OPEN

Next round, use the already-shared resolved raw-mesh operand source in a
private stage-transport arm and combine it in NEMO source order with the exact
V correction (`vn_adv` plus the 35 reciprocal faces) and raw NEMO V mask. Gate
each operand against this admitted record, then score the complete atomic unit
on OMT-4, independent rung 0 and the standing external gates. Do not land raw
`e3v_0` alone. No new NEMO acquisition is presently required.

ASKED choices: Decisions 103, 109, 113, and standing Decision 96. UNASKED
choices: empty.
