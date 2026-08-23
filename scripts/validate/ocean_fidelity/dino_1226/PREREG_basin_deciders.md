# PRE-REGISTRATION — the two offline deciders for candidates 1 and 2

Written BEFORE either decider produced a number, and before the 360-day
accumulation artifact they read had finished being written.
Lane: `fidelity/dino-basin-budget-1yr`. Parent result:
`docs/ocean/fidelity/dino_basin_budget_result.md` (commit `660942b08`).

## What is already established, and therefore not re-litigated here

The southern-basin transport deficit is a **depth-uniform eastward velocity
error** of about −4.6e-4 m/s in a ~300 km boundary layer against the southern
wall, with identical water masses and identical wind input, worth a year-mean
shortfall of **0.112 m³/s² per row** in the rate at which legoESM builds
circulation in those rows (23% slower than the oracle).

The ranked candidates were: (1) the non-linear bottom drag as a barotropic
momentum sink; (2) the split-explicit barotropic solve; (3) the lateral
momentum flux at the free-slip wall; (4) the enhanced vertical viscosity
(demoted). This registers the two offline deciders for (1) and (2).

Both run entirely on artifacts already on disk. No new integration.

---

## D2 — does legoESM's realised rate collapse to its own barotropic solve?

The oracle satisfies three exact identities that collapse its whole
depth-integrated row rate to one number: the barotropic solve. legoESM's step
composes differently and there is no reason it must. This asks whether it does
anyway.

**Measure.** From the 360-day accumulation artifact, per southern row and per
10-day window: the realised rate (the stage sum) against the `BARO solve`
stage row, and their ratio, using the same band mean the campaign records.

* **CONFIRM "collapses"** — median over the 13 wet southern rows of
  `|realised/BARO − 1|` **< 0.05**. Consequence: the entire 0.112 shortfall is
  inside the barotropic solve and every other stage is exonerated in one
  number. Candidate 2 becomes the single work item.
* **REFUTE** — median `|realised/BARO − 1|` **> 0.5**, or `|realised − BARO|`
  band-mean **> 0.112**, i.e. the non-barotropic stages carry more than the
  whole deficit. Consequence: the deficit must be attributed stage by stage
  and candidate 2 loses its privileged position.
* Between the two: **PARTIAL**, and the report says how much of the rate the
  barotropic solve carries rather than claiming either.

**What this cannot do, registered in advance.** legoESM's stage rows and the
oracle's are *not* row-comparable — the oracle discards the vertical solve's
column mean before the next step and legoESM keeps part of it. D2 is a
statement about legoESM's own internal composition only. It narrows *where in
legoESM* the rate lives; it does not by itself attribute the difference.

## D1 — is the bottom-drag sink stronger in legoESM?

**Measure.** The per-row bottom-drag torque `phi_drag(u, v)` — the campaign's
recorded transcription of the oracle's non-linear drag law, applied to each
model's own saved velocities, the SAME functional on both sides — at every one
of the 19 days both models saved a state, band-meaned over the southern rows
and also resolved per row.

Sign convention: `phi_drag` is a sink, so it is negative where the flow is
eastward. "legoESM's sink is stronger" means `Δ = phi_drag(lego) −
phi_drag(NEMO)` is **negative**.

* **CONFIRM "drag owns it"** — Δ band-mean over the year is negative and
  `|Δ| ≥ 0.056` m³/s² per row, i.e. at least half the 0.112 shortfall, with the
  excess concentrated in the four wall rows.
* **REFUTE** — `|Δ| ≤ 0.022` (under 20% of the shortfall), or Δ is **positive**
  (legoESM's sink weaker, which is the wrong sign to explain a weak flow).
* Between: **PARTIAL**.

**What this cannot do, registered in advance — and it is the important
caveat.** This is the drag torque each model's own state *implies* under the
shared transcription. It is **not** what either model actually applied: both
put the drag on an implicit vertical diagonal and both run a separate
barotropic in-substep drag, so the applied stress differs from this diagnostic
by the implicit factor and by the substep placement. Worse, the measurement is
**circular in one direction**: a weaker flow produces a weaker drag under a
non-linear law, so Δ negative is *consistent* with drag being the cause and
also with drag being a consequence.

That circularity is why the two deciders are registered together. Drag is a
**barotropic sink**, so if it is the owner it must show up inside the
barotropic solve. The joint reading, fixed now:

| D2 | D1 | reading |
|---|---|---|
| collapses | drag excess ≥ half | candidate 1 leads, and D2 says where it acts |
| collapses | drag excess small/positive | candidate 2 leads; the barotropic solve itself, not its drag term |
| does not collapse | drag excess ≥ half | drag acts outside the barotropic mode too — candidates 1 and 3 both live |
| does not collapse | drag excess small/positive | both demoted; candidate 3 (the wall) moves to the top |

## Self-checks that must pass before either number is read

Each can fail.

1. `phi_drag` must reproduce the recorded per-row drag torque on the SAME state
   the recorded probe used, to fp64 roundoff — otherwise the shared
   transcription has drifted.
2. The lego `v`-row offset must be decided by the day-0 identity, by the
   recorded control (one offset matching to <1e-5 m/s and the other 100× worse).
   A mis-offset `v` silently changes the drag speed.
3. The artifact D2 reads must stamp the shipped card, NEMO's vertical ladder,
   NEMO's day-of-year, 360 days and no plant — the same gate the cross-model
   comparison uses.
4. The stage sum in the artifact must reproduce the realised rate the states
   themselves show, band-meaned, to 1e-3.
5. Δ must be computed on identical days on both sides, and the day list printed.

## What neither decider may conclude

Neither is a perturbation test. A candidate that survives both is **plausible**;
confirming it needs one arm that changes that one thing and moves the number in
the predicted direction. No fix is built in this lane.

---

## POST-HOC DISCLOSURE — D2's axis was void, and this pre-registration was wrong

Added after both deciders ran, and kept here rather than quietly amended.

**D2 cannot fail.** Its gate 3 admits only artifacts stamped with the shipped
card's post-solve correction and NEMO's vertical ladder. Under exactly that
pair, `realised == BARO` is forced by three exact facts (two stage rows are
depth deviations under a depth-integral reducer; the post-solve correction
overwrites the after-level column mean with the barotropic solve's own
average). Measured at full precision the supporting cancellations are 3e-17 and
5e-14 against a scale of 1.7 — twelve orders inside the registered threshold.

**Consequence for the joint reading.** The 2×2 table above has a D2 axis with
one reachable value, so it selects nothing. The table is **void** and is not
evaluated. Reporting a cell from it would have been reporting an assumption.

**What survives, and is reported instead.** D2 keeps two findings that could
have come out otherwise: the shipped card's post-solve correction is verified
active (with it off, the row prints zero), and the time filter is exonerated at
0.02% against a 23% deficit. Both are stated as what they are.

**D1's sign clause was also wrong**, in two layers: it assumed an eastward
bottom flow (it is westward), and its replacement — "opposes, therefore
consequence" — is invalid for any sink-type cause. What decides D1 is the
registered *concentration* clause, which is a shape test and does discriminate.

**D3 was added after the fact** and is labelled as post-hoc: a per-term
accumulated cross-model table, in response to D2 turning out to be an identity.
It carries a closure gate.

**RETRACTION on D3, logged rather than edited away.** The first version of that
gate FAILED at 49 per row and the failure was written up as "the two models'
term sets are not in bijection at this level", with "build a matched term
correspondence" recommended as the next work item. That was a bug in the probe:
the oracle's barotropic operand (order 1e4) was credited into the surface group
while its post-solve row was omitted, double-counting. Repaired, the partition
closes to **2e-9 per row** on both sides. Both reviewers found this
independently and both verified the repricing before it was applied.

The closed table still attributes nothing — in the wall rows the groups cancel
310:1 against the quantity they would have to explain — and the probe says so.
Note also that legoESM's last group is a remainder, so its side of the gate is
an identity and only the oracle's side can ever fire. It did.
