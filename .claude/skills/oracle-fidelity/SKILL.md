---
name: oracle-fidelity
description: Discipline for matching a reference model (NEMO, MITgcm, Veros, Oceananigans) term by term. Use when porting/validating against an oracle, chasing a fidelity gap or climate-drift mismatch, building a twin/bridge, or whenever a comparison "should match but doesn't". Encodes the rules that were each earned by a failure.
---

You are matching a REFERENCE MODEL (the "oracle") term by term. You have its
source code. Almost every wasted day on this work came from reasoning about the
oracle instead of reading it.

## Rule 0 — READ THE ORACLE'S SOURCE FIRST. Always.

Before forming any hypothesis about what the oracle does: open its source and
read the routine. Quote `file:line`. This is not a last resort, it is the first
move.

- "NEMO probably uses X" → **read `traadv.F90`**.
- "the coefficient is presumably Y" → **read the namelist and the routine**.
- "this convention is surely Z" → **read it**.

If you catch yourself deriving what the oracle *must* do from physics or from
its outputs, stop and go read it. Symptoms you are doing this: long chains of
algebra about the oracle's discretisation; explaining a mismatch by a plausible
story; the phrase "should be equivalent to".

Every claim of "matched" or "different" must cite the oracle's line, not a norm.

## Rule 1 — Coverage, not a checklist

The worst bugs are not wrong comparisons, they are **missing** ones. If your
reader never loads an array, nothing can ever compare it, and no amount of
additional checking will find it — every check is driven by *your list*, so
anything off the list is invisible by construction.

So drive verification from **what the oracle provides**:

1. Enumerate every variable in the oracle's files (mesh, restart, namelist).
2. Each must be either **VERIFIED** against your equivalent to roundoff, or
   **WAIVED with a written reason**.
3. Anything else is a **hard failure**.

Waiving becomes a deliberate act that leaves a reason in the diff. See
`scripts/validate/ocean_fidelity/dino_1226/nemo_geometry_gate.py` for a working
example (45 arrays, 0 unaccounted).

Apply this to the mesh, the restart, AND the namelist.

## Rule 2 — Know what each gate CANNOT see

Write down every gate's blind spot; the next bug lives there.

| gate | sees | blind to |
|---|---|---|
| bit-exact state gate | T/S/u/v/η values | the GEOMETRY holding them |
| cell census | wet-cell counts, masks | cell SIZES |
| tendency comparison | instantaneous operators | slow systematic bias |
| norms / RMSE | magnitude | spatial structure, edge artifacts |

A real case: a 12.9% layer-thickness error passed a bit-exact state gate AND a
census for days — the state was right, the boxes holding it were the wrong size.

## Rule 3 — Calibrate an instrument BEFORE trusting a residual

Never report a difference without knowing the metric's noise floor. Measure it
by **total ablation**: delete a first-order process entirely and see how much
the metric moves. That is your floor.

Real case: a 1-year ACC difference of +0.6 Sv looked like agreement. Deleting
the eddy parameterisation *entirely* also moved it 0.6 Sv — the metric could not
distinguish the model from one with no eddies at all. The 3-year window gave
12.9 Sv of signal against that same floor.

If signal ≈ floor, the measurement is meaningless no matter how clean it looks.

## Rule 4 — Ablate BOTH models (ownership, not sensitivity)

To test whether subsystem X owns a residual, remove X from **both** models and
ask whether the GAP survives.

- gap survives → X is **exonerated**
- gap collapses → X **owns** it

Ablating only your model measures your sensitivity to X, which is a much weaker
claim and a classic route to a false attribution.

Verify the two ablations are the *same* ablation by reading both configs — e.g.
`use_gm_redi=False` may kill only the bolus while leaving isoneutral diffusion
at full strength, whereas the oracle's flag kills something different.

**Ablation is structurally blind to a MISSING term.** If your model lacks a term,
deleting it changes nothing and it looks exonerated. Flat insensitivity across
every subsystem is the signature of something absent — switch to Rule 1.

## Rule 5 — The oracle's diagnostic outputs are integrator bookkeeping

Trend/tendency diagnostics (NEMO `trdtra`/`trddyn`, and equivalents) are NOT
clean physics buckets. Before quoting any of them, **prove budget closure**
against the oracle's own total.

Known traps of this class: a diagnostic that is a non-additive re-diagnostic of
something already inside another bucket; a bucket that silently folds in a
different operator; a term that is ~99% integrator bookkeeping rather than
physics. Every unclosed trend comparison in one campaign produced a false
finding — five in a row.

**A residual bucket can never be an attribution.** If a term is computed as
`TOTAL − (other terms)`, a signal in it is "that term plus every other term's
error".

## Rule 6 — Prefer weighting-free invariants

Integral diagnostics can be wrong through their *weights* (area, thickness,
mask, dtype) rather than through the model. Prefer checks that need no weights:

- **Constancy preservation** — a uniform tracer must stay exactly uniform under
  pure transport. No area, no thickness, no mask enters `max|T − T₀|`.
- **Rest state** — an initially motionless, level-isopycnal state must stay at
  rest.
- **Analytic/manufactured solutions.**

When you must integrate: accumulate in float64 (a float32 reduction over ~10⁵
cells is only good to ~1e-7 — the same size as the drift you are hunting), and
state the roundoff threshold **for the actual dtype**.

## Rule 7 — Controlled comparison, one variable

Hold the eval protocol byte-identical to the baseline: same forcing and
sampling, same grid, same metric definition, same window, same masks. A metric
that moved because the protocol changed is a CONFOUND, not a result. If a
resource limit forces a lighter protocol, **re-run the baseline at that same
protocol** before comparing.

Before writing "improved"/"degraded"/"comparable", confirm the two configs
differ ONLY in the variable under test. If you cannot, claim no direction.

## Rule 8 — Faithful-but-worse is a SIGNAL, not a reason to revert

Removing a compensating error can worsen a metric while being correct. When a
faithful fix makes things worse:

- do **not** revert the fix
- do **not** flip a default to hide it
- find what the wrong version was compensating for — that is the real bug

Real case: adopting the oracle's true layer thicknesses made the model unstable.
The correct response was to find the second defect (an eddy-parameterisation
term that the wrong grid had been masking), not to keep the wrong grid.

## Rule 9 — Never add a stabilizer the oracle lacks

If the oracle runs this configuration stably and you do not, **you have another
mismatch** — that is the finding. Do not add damping, clipping, filtering or
limiters the oracle does not have. Check the oracle's switch first: it may
already be off in both, in which case enabling yours is a deviation.

## Rule 10 — Instantiate and print; never trust a declaration

Comments and docstrings go stale, defaults are overridden, and a subagent
reading a formula may report an illustrative value as the real one. Always
instantiate the config and print the number you are about to rely on.

Real cases: a comment claimed a scheme was off when it was on with a coefficient
of 200; a reported thickness of "10.0" was actually 10.138751.

## Rule 11 — Record retractions as first-class output

When a finding dies, write down **what it was and what killed it**, next to the
finding it replaces. Retracted attributions get re-discovered by the next
session otherwise. A campaign that produced five retractions and one confirmed
bug is not a failure — but only if the five are written down.

## Working order

1. Read the oracle's source for the subsystem. Build an ordered
   `N1..Nk` alignment table: oracle `file:line` vs yours, MATCH/DIFF per row.
   A CLEAN table is a real result — it eliminates a suspect.
2. Run the coverage gate (Rule 1) on mesh, restart, namelist.
3. Establish weighting-free invariants (Rule 6).
4. Calibrate your headline metric (Rule 3).
5. Only then attribute — by both-sided ablation (Rule 4) or by table.
6. Fix as a **selectable option** on the oracle-matching card, defaults
   unchanged for other users; add a direct test with a synthetic-violation
   check so it cannot pass vacuously; then re-measure.

## Anti-patterns

- Building a fourth bookkeeping probe when the previous three each produced a
  retraction. Prefer an experiment that answers the question directly.
- Sizing a candidate's impact by a global-mean scalar when the target is a
  gradient. A structured error can have leverage far above its mean.
- Asserting `(x/h)·h == x` and calling it a conservation test. Every test needs
  a synthetic-violation check proving it CAN fail.
- Reporting a number without the config that produced it.
