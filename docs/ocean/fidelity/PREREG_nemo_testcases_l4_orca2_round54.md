# ORCA2 round 54 preregistration — repair ENS acquisition free-space preflight

Date frozen: 2026-09-27  
Base: `da29c80009876fc10af1d0b323f2e481501cee96`  
ORCA2 claim label: **given NEMO's entry** (Decision 52; no ORCA2 trajectory
measurement is planned)  
OVERFLOW claim label: **given NEMO's recorded operands**

## Frozen question and scope

The operator ran round 53's committed acquisition.  It exited before `makenemo`
or `mpirun` because the free-space preflight passed the not-yet-created parent
of `TARGET_RUN` to `df`, which returned "No such file or directory".  The target
configuration and run directory therefore remain absent.

This round may only repair that acquisition preflight, prove the repair binds,
and hand the same acquisition back to the operator.  It may not change the
writer, parser, NEMO patch, target name, run configuration, model package,
selector, carried state, threshold, score domain, stabiliser, or sea-ice field.
The held QCO package hunk stays absent.

Repository search found the established repair pattern in the round-156 GYRE
acquisition: reject an existing target, create only the target's parent, then
query that existing parent with `df`.  The round-53 script is extended in place;
no second launcher or helper is introduced.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R54-P1 | Creating only the absent target parent after the new-target refusal makes the free-space preflight executable without weakening target freshness. | A shell test begins with an absent nested parent, executes the exact ordered guard/mkdir/df pattern successfully, and proves the target itself remains absent. | Any target creation, overwrite path, or `df` failure: reject the repair. |
| R54-P2 | The repair is non-vacuous and minimal. | Removing the parent creation makes the shell test fail; the committed diff changes only the round-53 launcher, its direct test, preregistration, and receipt; `packages/` remains unchanged. | A plant stays green or another scientific/configuration file moves: reject the repair. |
| R54-P3 | Round 53's acquisition remains otherwise unchanged and ready for the operator. | Shell syntax, the existing additions-only/source-branch preflight, self-describing parser tests, source-target absence checks, and citations pass; the operator path still names the same fresh target and record. | Any writer/parser/patch digest-affecting edit or preflight regression: keep `STOPPED_FOR_RECORD` and repair before handoff. |
| R54-P4 | No scientific statement is measurable this round. | No ENS record exists after local verification; disposition remains `STOPPED_FOR_RECORD`, with R53-P2 still `UNMEASURED_WITH_SPEC`. | A record unexpectedly exists: stop and admit it under the frozen round-53 gate before making a claim. |

Failed predictions remain in the receipt as **REFUTED**.  No field list, score
mask, exact bar, or configuration choice changes after this freeze.

## Required verification

1. Add a direct test for the exact guard/create/check ordering and show it fails
   when the repair is removed.
2. Run shell syntax, round-50/52/53 focused controls, Ruff/compile checks, and
   the round-53 additions-only preflight.
3. Run a separate read-only `codex exec` review and record its verdict or the
   mandated unavailable wording.
4. Run the default and round receipt citation gates with a real planted shift,
   then the shared-card battery and `tests/ocean/fidelity -n 12` once, one
   pytest battery at a time.

## Frozen OPEN

The operator runs the repaired committed round-53 launcher.  The following
round admits the record and walks `zwz`, `zuav`, their product, and the final
addition in compiled order.  The QCO arm stays held until that walk identifies
a complete source-exact pair or the first non-bit internal statement.  The
independent Decision-52 initial-state/year and round-20 slow forcing remain
open; sea ice remains out of scope at `STOP_SELECTOR_GAP`.
