# NEMO testcase Lane 4 — ORCA2 Phase-2w handoff receipt

Date: 2026-09-06

Parent: `26af77048a8b28f0d4f29bc9e36699919615d0a2`

Status: **IN PROGRESS — Phase-2v replacement admitted; new ice-variant MPI
twins still to be handed to the user shell.**

## 1. Phase-2v twin admission and V2 pin

**CONFIRMED:** both unchanged launchers completed at `time.step=10` with
`MPIRUN_RC=0` and `RUN DONE`.  The retained acquisition gate decoded the
57,266,632-byte `NEMO_L4_TKEW_1` stream from its 15-integer base header and
24-field extent table to exact EOF.  The header resolves `kt=2`, `Kbb=3`,
`Kmm=3`, binary64, `17` 3-D fields, `7` 2-D fields, and a header-derived
7,629,792-value payload.  Every unowned slot is canonical zero and the frame
is non-vacuous.

Twin A and B are raw-identical for **101 / 101** `oracle_*.bin` streams.  All
**100 / 100** inherited streams are raw-identical to the Phase-2s root.  The
ordinary-output identity control passes with dynamic counts: four restart
shards are exact bytes; eight history payloads are exact after excluding only
the global timestamp attribute; `ocean.output` is exact after the registered
WRITE-only dump notices.  Timing and launcher provenance are excluded.

Twin A is therefore pinned as the single `VARIANT_ORACLE_V2` root:

`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2v_tke_a_10step_np2`

Twin B is its independent reproducibility witness.  Phase-2s and earlier roots
remain retained witnesses.  The Phase-2u twins remain
`REJECTED_NONREPRODUCIBLE_TKE_RECORD`; nothing was deleted.

The complete 105-file output/hash inventories are external evidence:

- A manifest SHA-256 `9a98cd5e97caffe54707d33e06e34a42c1c39e8cd9a875ffc187988685079047`;
- B manifest SHA-256 `3e78e3829d87ad08eda5313d44c50b42894ca1d81094ec45c1b22ce2d208500b`;
- admission JSON SHA-256 `63d77e382f951df3bf9a5ebc1243bd3126322249f85a69057226493690f61abd`;
- admitted TKE record SHA-256
  `31675493f022f71a609142f53bbe220c111b09e9a9a352926a1aff7358770a52`.

All eight binding plants exited nonzero: magic, extent, count, truncation,
trailing byte, canonical unowned slot, undefined `jpk` workspace, and twin
identity.  This is **CONFIRMED**, not inferred from the user's byte census.

## ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| execute Phase-2v MPI twins | ASKED, user shell | CONFIRMED complete; launchers unchanged |
| admit and pin twin A | ASKED | CONFIRMED 101/101 twin and 100/100 inherited raw identity |
| retain twin B and prior roots | ASKED | CONFIRMED retained and labelled |
| pin Phase-2u TKE records | forbidden | rejected records remain flagged, never scored |
| change shared arithmetic during admission | UNASKED | none |
| delete artifacts, edit shipped NEMO, commit large records, or push | forbidden | none |
