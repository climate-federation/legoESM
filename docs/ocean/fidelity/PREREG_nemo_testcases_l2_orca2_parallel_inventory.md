# Preregistration: NEMO-testcases L2 ORCA2 parallel inventory

Date: 2026-09-16. Frozen at incoming side-branch tip
`29af6665f77c` before creating or running an ORCA2 inventory probe and before
attempting any legoESM ORCA2 integration. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/parallel/orca2/`.

## Question and governing spec

The newest campaign statement classifies ORCA2 as
**UNMEASURED-WITH-SPEC**: first resolve its integrator, then record native
production-step TKE boundaries and inputs with the same provenance stamp,
exact-EOF checks, and one-ULP controls used by the GYRE statement-boundary
lane. This inventory asks the earlier admission question: can the current
tree construct and execute a native legoESM ORCA2 card from rest for kt=1--3,
and is a new NEMO acquisition actually required?

This is a zero-cost, read-only inventory lane. It may inspect the current
tree, repository history, the read-only oracle build, and existing records.
It must not run `makenemo`, `mpirun`, or a NEMO executable, change production
physics, touch the GYRE card, or modify `packages/ocean/legoesm`.

## Frozen inventory and admission predicates

The inventory will record, without substituting a surrogate:

1. whether the current recipe dispatch constructs a native ORCA2 card;
2. whether the current tree contains the exact ORCA2 forcing reader required
   by that card and a tripolar grid implementation;
3. whether a read-only ORCA2 oracle configuration and executable exist;
4. whether phase3 contains a raw ORCA2 record or only derived ORCA2 evidence;
5. whether any other campaign record contains a completed, provenance-stamped
   native ORCA2 run with kt=1--3 stage/entry frames and the requested TKE
   boundary record; and
6. whether the current card passes its own execution guard.

A legoESM kt=1--3 run is **admitted** only if the current checked-out source
constructs the ORCA2 card, provides its exact grid/recipe/forcing inputs, and
passes the card's execution validator without unresolved selected mechanisms.
If any predicate fails, the run must stop before kt=1 and the receipt must
name the first failed predicate. Historical or derived artifacts may establish
what exists, but cannot make an absent current-tree card runnable.

A new NEMO acquisition is **needed** only if no existing native ORCA2 record
has a successful completion stamp, matching executable/input provenance,
kt=1--3 entry/stage records, the native TKE record required by the newest
spec, and the corresponding exact-EOF/plant contract. A complete existing
record refutes acquisition need even if current-tree legoESM integration is
still blocked.

## Frozen predictions and falsifiers

The working prediction is:

- the read-only oracle checkout has a built ORCA2 target using RK3, QCO, ZPS,
  and TKE;
- an already completed native ORCA2 record supplies kt=1--10 entry/stage
  frames and a kt=2 TKE walk, so a new oracle run is unnecessary;
- the current GYRE-lane tip has the generic tripolar grid but no current ORCA2
  recipe dispatch or exact ORCA2 forcing reader; and
- the last historical ORCA2 card remains fail-closed on explicitly unresolved
  selected mechanisms, so a legoESM kt=1--3 run is not admitted today.

The prediction is **confirmed** only if every positive and negative predicate
is observed directly and the inventory's planted missing-record control exits
nonzero with a named `REFUSE:` line. It is **refuted** if the current tree can
construct and admit ORCA2, if the claimed record lacks any required native
frame/provenance field, or if the planted control does not fire.

If current execution is admitted, the lane will run kt=1--3 on CPU under
`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, the repository `PYTHONPATH`, and the
explicit fp64 precision policy, then report the first-over-bar table against
the existing native record. If it is not admitted, no numerical trajectory
will be manufactured: the first-over-bar table will be marked not measured at
the named pre-kt1 predicate.

## Frozen outputs and disposition

The committed probe will write a deterministic JSON inventory in the evidence
directory and expose a planted failure mode. The receipt will cite every claim
to a file and line, distinguish current source from historical source and raw
records from derived evidence, state whether acquisition is needed, and carry
an ordered OPEN list. No numerical candidate or landing claim is eligible in
this task.
