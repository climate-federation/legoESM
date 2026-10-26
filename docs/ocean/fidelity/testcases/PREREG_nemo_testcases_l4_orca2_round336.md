# ORCA2 round 336 preregistration — `tra_zdf` operand and recurrence walk

Frozen 2026-10-26 before running the round-336 offline audit or extending the
round-333 acquisition. Base: `ba9f39512595`. This round changes no
configuration value, and Codex will not run NEMO.

## Fixed scope and source order

The registered target is the round-333 rung-10 kt-2 stage-3 set: 18 wet
temperature words and 13 wet salinity words at which the first value departure
appears on return from the implicit tracer solve. Both the given-NEMO-entry and
independent labels are scored. The operand walk is ordered as NEMO executes it:
the effective diffusivity (`avt + akz` for temperature and `avs + akz` for
salinity), `e3w(Kmm)`, `e3t(Kaa)`, the thickness-form RHS, matrix diagonals,
eliminated diagonal, forward content, and back-substituted solution.

The record-build selectors are audited from retained `ocean.output`, not
inferred: `ln_zad_Aimp=F`, `ln_zdfddm=T`, `ln_zdfevd=T`, `ln_zdfiwm=T`,
`ln_zdfmfc=F`, and the MSC isoneutral branch is active. Any changed selector,
RK level tuple, or source placement refuses the measurement.

## Measurements and predictions

- **R336-P1 — input completeness.** The admitted round-333 tracer and inner-LDF
  records are predicted to determine NEMO's `e3t(Kaa)`, `e3w(Kmm)`, and
  thickness-form temperature RHS at every registered temperature word. They
  are predicted to be bit-identical to the arrays consumed by legoESM's live
  solve. Any unequal target bit refutes that operand. The final kt-2 `avt` and
  `avs` after TKE, river-mouth, EVD, DDM, and IWM composition are predicted to
  be absent from the admitted records; an earlier-step `avt_k` is not an
  admissible substitute. Finding an admitted exact kt-2 final coefficient
  refutes the absence prediction.
- **R336-P2 — discrimination.** If all recorded inputs are exact but the final
  `avt/avs` or any inner matrix/sweep row is absent, the admitted evidence is
  predicted not to discriminate the first differing operand or statement.
  The result must remain **PLAUSIBLE** and trigger an additions-only extension
  of the reviewed round-333 acquisition. A complete admitted row walk that
  names one first unequal row refutes this prediction.
- **R336-P3 — acquisition payload.** If required, the extension records both
  tracer passes at kt 2 stage 3, with the effective matrix coefficient,
  `e3w(Kmm)`, `e3t(Kaa)`, thickness-form RHS, lower/diagonal/upper rows,
  eliminated diagonal, forward content, and final back-substitution value.
  Calls that copy payloads are outside NEMO's arithmetic loops. Admission
  requires exact declared extents, source-order field names, checksums,
  finite payloads, signed-zero preservation, wet coverage of all registered
  targets, selector/time-level/source-placement plants, a bounds fixture, and
  a production-flag observer-free versus instrumented floating-point census.
  Any missing row or passive-control failure refuses the record.
- **R336-P4 — stop/landing rule.** A model edit is eligible only if one cited
  statement is confirmed and its one-statement replacement is bit-identical
  or strictly better under Decision 122 for all seven pairs and both labels,
  with a revert-failing production-call-site test, every shared caller,
  faithful DINO probe/month and grid-ladder coverage, and finite rung-10 N=1
  reverse gradients equal to forward mode. Otherwise the round stops at the
  measured proposal or `ACQUISITION_NEEDED`.

## Controls

The offline audit must re-admit every consumed record, use production JIT on
CPU under fp64/libm, compare an independently compiled ordinary state with the
trace state, and ship plants for target ownership, record completeness, row
order, selector state, and passivity. The acquisition wrapper must import the
round-333 helpers and preserve every inherited checklist item; it may not
execute NEMO in preflight mode.
