# Round-65 post-code self-review

Date: 2026-09-12

- Scope: no production physics change survives.  The temporary native FCT
  diagnostic and the gate that could print its refuted candidate are removed.
- Retraction: the preregistered `traadv_fct.f90:503-506` and
  `zdftke.f90:430` candidates are explicitly REFUTED in the receipt.
- Evidence: record calibration, commit stamp, schema/EOF validation, and
  truncation/ULP/stamp plants are fail-closed.  All reported exactness uses
  uint64 comparison, not tolerance.
- Causality: Kmm T/S and thickness are already non-bit before the first FCT
  writer.  The receipt calls line 607 only the first observed unequal writer,
  not an owning transcription.
- Completeness: SBC/QSR/LDF are explicitly UNMEASURED because legoESM currently
  aggregates those stage-3 sources and the admitted record does not provide a
  model-side split seam.  No invented reconstruction is presented as NEMO.
- Rule 12: no candidate landed, so no card can be DISCHARGED or regressed by
  this round.  Existing trajectory values are quoted unchanged, not rerun as
  before/after evidence.
- Review status: SELF-REVIEWED; independent review is deferred to the Codex
  review session requested after this round.  No independently reviewed
  physics claim is made.

