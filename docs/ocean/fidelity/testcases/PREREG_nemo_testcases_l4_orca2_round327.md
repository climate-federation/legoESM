# ORCA2 round 327 preregistration — rung-10 residual localisation

Frozen 2026-10-23 before the round-327 probe is run.  Base:
`6a465f349e367478bdee4bcdab2a5de570be931c`.

This round measures only.  It changes no model statement, card, selector, or
configuration.  The measured comparison is rung 10's current comparison card:
implicit ice-ocean top drag, L3 eddy-induced velocity on the L9 route, and
`nn_mxlice = 2` supplied with the admitted NEMO mean-ice-thickness record.

## Source-first hypothesis

The independent card and the recorded one-category run already disagree in
the ocean state before `kt=1` executes.  NEMO forms snow/ice/pond mass and
subtracts one domain-global scalar from both SSH time levels at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:440-465`.  The card independently
constructs its five-category initial ice-mass adjustment at
`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1752-1796`.
No certified ocean rung owns this initial sea-ice identity: the G5 ice-ocean
drag rung consumes a completed recorded ice-ocean state and does not create
it.

## Frozen predictions and falsifiers

1. **R327-P1 — first departure.**  Reading the current round-326 ladder JSON
   in execution order predicts the independent label first exceeds the exact
   bar at `kt=1`, `entry`, `ssh`: exactly `16,433 / 26,640` cells, maximum
   `0.015479333813968585 m`, RMS `0.012157480547162937 m`; T/S/u/v remain
   exact.  The given-NEMO-entry label predicts five exact fields at that same
   row.  Any changed row, count, maximum, RMS, or non-SSH departure refutes.
2. **R327-P2 — localisation and replay.**  Every unequal SSH value predicts
   the same signed offset to binary64 equality.  Replacing only the card SSH
   by the passive completed NEMO entry SSH predicts all five fields exact.
   The probe will report the first and maximum cell's zero-based `(j,i)`,
   latitude, longitude, surface/depth classification, recorded `fr_i > 0`
   ice classification, coast/interior classification, and fold/seam flags;
   it makes no prospective prediction about those categorical counts.  A
   nonuniform offset or any residual after the one-field replay refutes.
3. **R327-P3 — inheritance.**  Because the given-entry label is exact at the
   same `kt=1` entry row, the independent excess predicts `INHERITED_ENTRY`,
   not creation by an ocean-step statement.  Any given-entry numerical
   residual at the row refutes.
4. **R327-P4 — remaining magnitude order.**  At full-domain `kt=10` entry,
   rank the five field families by the larger maximum across the two labels.
   The frozen order and `(independent, given-entry)` maxima are: SSH
   (`0.018533089817820425`, `3.734978452696325e-05`) m; T
   (`0.011429686050284715`, `7.703006966558457e-05`) K; u
   (`0.005295100146905263`, `0.003046847350500703`) m/s; v
   (`0.0031473714762187688`, `0.003149083070336161`) m/s; S
   (`0.0018793894167785652`, `1.2237551899829668e-05`).  Any changed value
   or order refutes.  These are measured residual families, not statement
   ownership; only the initial SSH row may be assigned by this probe.
5. **R327-P5 — controls.**  Plants for row order, owner replay, localisation
   partition, and endpoint ranking must each make classification refuse.

## Disposition rule

If P1--P5 pass, the receipt reports the initial SSH owner as **CONFIRMED**,
labels every unwalked endpoint-family owner **PLAUSIBLE/UNMEASURED**, and
offers numbered choices for the initial-state/configuration question.  It
must not turn endpoint magnitude into statement attribution.  A failed
prediction is retained as **REFUTED** and reconciled before any conclusion.

ASKED: current-card localisation, passive owner-output replay, cross-label
ranking, and next-round options.

UNASKED: NEMO execution; model, card, selector, threshold, or configuration
changes; and attribution of a compounded `kt=10` residual to an unmeasured
statement.
