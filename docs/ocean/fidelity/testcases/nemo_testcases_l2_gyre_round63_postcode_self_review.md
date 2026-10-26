# Round 63 post-code self-review

Date: 2026-09-12

## Findings and corrections

- The first writer compile exposed an incorrect module owner for r3t; the
  redundant USE was removed and the compiled dom_oce owner retained.
- The first hand-written stage-driver patch had a malformed hunk count; it was
  corrected before any source was accepted.
- Static runner inspection found template-produced plus tokens in continued
  shell commands; all were removed, bash -n now passes, and the source-run
  resolved patterns were exercised read-only.
- Every production-source patch now adds lines only relative to its exact R46
  or shipped base. No physical expression, carried field, or configuration
  branch is changed.
- The TKE capture brackets the RHS statement inside each jj row, before that
  row's tridiagonal sweep can overwrite en.
- The content gate uses the recorded effective e3t and separately reconstructs
  that effective thickness from e3t_3d, r3t, and tmask before accepting the
  content reconstruction.
- The reader is an extension of the round-54 tracer tool; no second parser was
  introduced.

## Controls

- Preprocessing plus gfortran syntax checking passes for the writer and all
  four dry-patched compilation units with empty output.
- The focused new and inherited tracer tests pass 11/11.
- Five round-63 CLI plants (stamp, two one-ULP arms, two truncations) return
  nonzero.
- The citation-map audit passes after pinning every new endpoint to a unique
  occurrence.
- Clean-stamp validation used /tmp/gyre-r63-stamp.yOlzN8/repo at
  a7688af736af: 27/27 focused tests, 15/15 receipt citations, shifted-citation
  plant exit 1, and empty-output syntax compilation.
- Runner is operator-only, refuses dirty/uncommitted trees and pre-existing
  targets, checks resolved ocean.output, stamps each record, runs consumed-
  field admission, and includes a nonzero admission plant.

## Review disposition

The user explicitly scheduled a later Codex review session. No claim from an
acquisition is made here, and no physics is eligible to ship before that
review and the operator run.

## ASKED / UNASKED

| Kind | Item | Disposition |
|---|---|---|
| ASKED | Compile-checked writer and handoff | Complete |
| ASKED | Stop before makenemo/mpirun | Honored |
| UNASKED | Resolve measured owner without the new record | Not inferred |
| UNASKED | Change production physics | Not performed |
