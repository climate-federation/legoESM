# ORCA2 round 279 preregistration

Date: 2026-10-15

Scope: verify the runoff-depth configuration without changing it, then diagnose
the held completed-stage top-V fold exchange at matched RK3 program points.
All measurements are ocean-only and CPU-only.  No package change is authorised
by this preregistration.

## Frozen source reading

The compiled ORCA2 `sbcrnf` branch tests `ln_rnf_depth .OR.
ln_rnf_depth_ini`; its false arm adds runoff only to level 1.  The shipped deck
does not override either `namelist_ref` default (`.false.`).  The legoESM recipe
default is `runoff_depth_spread_m=150.0`, while the rung-0 builder explicitly
sets `0.0` and no map.

The compiled ORCA2 RK3 program calls `lbc_lnk` on completed `u`, `v`, `T`, and
`S` after every stage.  The held selector fills the completed top V row, but the
current shared implementation has no corresponding completed-stage T/S fold
association at that boundary.

## Frozen predictions and falsifiers

**R279-P1 — oracle runoff semantics.**  The resolved ORCA2 deck has
`ln_rnf_depth=.false.` and `ln_rnf_depth_ini=.false.`, so NEMO deposits runoff
at the surface only.  CONFIRM if the compiled false arm is live and both flags
remain false; REFUTE if either flag is true or another live branch distributes
the runoff vertically.

**R279-P2 — card mismatch.**  Instantiating the shipped rung-10 ORCA2 card will
print `runoff_depth_spread_m=150.0`, `runoff_depth_spread_map=None`, and resolve
an active 150 m spread.  Rung 0 and OMT-1 through OMT-4 will print `0.0`, no
map, and resolve surface-only.  CONFIRM exactly those values; REFUTE if rung 10
is surface-only, if it has an explicit map, or if any dynamics-only card enables
vertical spreading.

**R279-P3 — first effect of the held statement.**  With the completed-stage
top-V selector toggled on against the same current-tree card, kt=1 stage-1 entry
and every non-V carried field remain array-identical; the first changed carried
field is completed stage-1 V on the northern fold row.  REFUTE if an entry field
moves, if the first V movement is off the northern fold support, or if another
field changes at the same earlier boundary.

**R279-P4 — first downstream growth boundary.**  Because the associated V is
first consumed by the next stage while the completed T/S pair was not folded at
NEMO's post-stage call site, the first downstream non-V movement will appear in
kt=1 stage 2 in T and/or S on the fold band.  CONFIRM if stage 1 differs only in
V and stage 2 is the first boundary with a T/S difference whose argmax is in
the fold band; REFUTE if T/S remain identical through stage 3 or if an interior
SSH/U field is the first downstream movement.

**R279-P5 — missing-partner control.**  Replaying the stage-2 tracer consumer
from the same stage-1 state after applying NEMO's T-point fold association to
the completed T/S pair will reduce the first T/S residual to the campaign floor
without undoing the top-V copy.  CONFIRM if the controlled association closes
that first residual; REFUTE if it does not move it or if the residual precedes
NEMO's post-stage association.

## Controls

The committed measurement must contain planted failures for (a) a changed
runoff scalar and (b) a deliberately corrupted fold-row value.  Each plant must
make its corresponding check refuse.  Measurements are reported separately as
"given NEMO's entry" or "independent" and never pooled.

## Landing predicate

Round 279 is diagnostic.  A runoff mismatch creates a configuration decision
and is not changed here.  The held top-V statement remains off unless a complete
cited unit passes the standing Decision-96/122 gates; a partial association is
not a landing candidate.
