# SI3 round 18 receipt — O8–O16 master-record acquisition tool

## Outcome

The operator-only acquisition tool is built and its read-only preflight passes.
NEMO was not run. Per Decision 173, this is one shared 100-step record, not nine
runs; per Decision 179 the production ice namelist is unmodified and Picard
counts are not scored. This preserves the exact-input separation specified for
the O rungs (`docs/ice/fidelity/si3_ladder_design.md:195-197`).

The tool records 17 boundaries from frazil through the first ocean-format
`ice_update_flx` output. O8–O16 each have a fail-closed projection reader and a
missing/foreign record plant. O9 retains final Picard state only at steps 1, 2
and 100; it does not enable the convergence-diagnostic overlay. The frazil call
is the always-executed production call (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:174`),
and O9's first operator exit follows the production ZDF call
(`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:197`).

## Storage and retention

The schema-derived worst case is 114,202,493,300 bytes (106.36 GiB). The run
requires 150,000,000 KiB free disk before building/running. Projection sizes are
O8 0.10 GiB, O9 33.46 GiB, O10 23.21 GiB, O11 19.62 GiB, O12–O15 16.21 GiB
each and O16 9.07 GiB. Shared boundary files are not copied: per-rung manifests
contain relative references, while the raw master, logs, source/binary hashes,
production decks and builds remain immutable until all nine rungs admit.

Refused evidence and builds are retained. The explicit move-aside mode is the
only supported rerun path. Admission pins the producer commit and verifies
source/executable digests, completion logs, all projections and reference versus
record restart bytes.

## Preflight evidence

The default invocation completed all nine phases without invoking NEMO:

- 17-kind reader schema, signed-zero retention, checksum/truncation/header and
  swapped-field plants;
- all nine projection readers with missing/foreign/wrong-category plants;
- byte-identical production ice namelist and namelist group membership plants;
- placement against real preprocessed sources and a misplacement plant;
- production-flag compilation and nonzero per-function FP census for five
  patched routines, with all observer counts unchanged and all injected-op
  plants firing;
- a bounds-checked execution of patched `ice_thd_da` with 10+5 layers, distinct
  cells and retained `-0.0`;
- source-derived SAS time level and explicit full-halo/interior-only branches;
- memory, disk, retention and explicit-run gates.

The direct acquisition tests and required repository gates are recorded in the
commit that carries this receipt. No physics result or fidelity claim is made.

## Choices and review

- Shared record, 100 steps, one rank, unmodified production ice namelist: ASKED.
- Production dynamics remains on and is excluded from every O8–O16 projection:
  mechanical consequence of the unmodified deck; explicitly stamped.
- Conservative worst-case gather sizing and 150 GB disk gate: UNASKED safety
  margin, offered for revision before the operator run.
- Raw master retained once; per-rung manifests reference it without copies:
  UNASKED storage mechanism, offered for revision.
- No nested Codex review was run.

independent review: operator (Claude)
