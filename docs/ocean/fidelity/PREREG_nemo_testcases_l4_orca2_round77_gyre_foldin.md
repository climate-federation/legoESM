# Preregistration — ORCA2 round 77, the GYRE fold-in and the re-measurement

Written and committed while the ten-step ladders were still running and before
any ladder number of this round existed.  The fingerprints and the resolved
configuration diffs in section 0 were already measured when this was written
and are labelled as such; everything in sections 1 and 2 is frozen.

## 0. Already measured when this was written

The four conflicts are resolved, the citation map is re-anchored by symbol and
audits clean, the ORCA2 card states its after-SSH form, and the card
fingerprints and resolved-configuration diffs are taken on three trees (the
pre-merge ORCA2 tip, the GYRE lane tip, and the merged tree).

## 1. Frozen predictions

* **P1 — the entry bridge is untouched.**  In the "given NEMO's entry"
  ladder, every kt=1 entry row stays bit-identical.  REFUTED by any kt=1
  entry row that is not bit-identical.
* **P2 — the independent initial state is untouched.**  The independent
  ladder's kt=1 sea-surface row stays at 16,433 of 26,640 cells with maximum
  0.015479333813968585 m.  REFUTED by any change in that count or maximum.
* **P3 — the later independent rows MOVE, and the mixing length owns it.**
  The merged tree changes ORCA2's turbulence mixing-length floor from 1e-8 m
  to NEMO's forced 1e-3 m and adds three more mixing-length statements, so the
  kt=10 rows must move.  REFUTED if every kt=10 row is unchanged, which would
  mean the floor is not reached on this card and the fold-in is inert there.
* **P4 — GYRE and both tanks do not move.**  The only field this lane adds to
  the shared configuration resolves, on every card but ORCA2, to the behaviour
  those cards already ran, so GYRE's ten-step ladder and both tank ladders
  must be byte-identical to the GYRE lane's.  REFUTED by one differing row.
* **P5 — DINO's from-rest month is inert.**  The gate PASSes at its pinned
  value within the harness's own floor.  REFUTED by a worse day 30.

## 2. What would make this round HOLD

Any ORCA2 headline row that moves AWAY from NEMO and is owned by a field
stated in this round rather than by the merge, or any GYRE/tank/DINO row that
moves at all.  In that case the specific field is held and named, and the
fold-in still lands, because a merge is not a choice.

## 3. Not in scope

The six sea-ice selectors, the carried state, every threshold, and the
first-non-bit statement walk (round 77's external transport pair), which this
round does not advance.
