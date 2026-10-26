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

Apply this to the mesh, the restart, the namelist, **AND the oracle's STEP CALL
GRAPH**. Enumerate every `CALL` in the oracle's step routine (descending one
level through dispatch wrappers to the concrete routine the config runs) and
force a disposition per routine: VERIFIED (number + where), WAIVED (reason), or
UNVERIFIED (listed loudly, ranked by climate leverage). A term list built from
"terms we had reason to suspect" is a checklist, not coverage — the #1226 sweep
verified 11 suspected terms to ~1.0 while the barotropic solver and momentum
advection pieces sat unenumerated, and "verified in an earlier phase" served as
an implicit, unwritten waiver. Prior-phase verification does NOT carry over a
raised bar: re-certify or waive explicitly, in writing.

## Rule 1b — The bar is EXACT. Encode it in a gate, not in your judgment.

If the user sets the bar at corr 1.0 / ratio 1.0, then 0.99x is **DEBT**, not
"matched", "faithful", "closed", or "good enough". Those words are forbidden
for any term whose measured numbers are not at the bar.

Judgment drifts: measure 0.9923, write "FAITHFUL", move on — repeatedly, across
a whole campaign, while being told the bar each time. So do not hold the bar in
your head. Put every term's measured (corr, ratio) in a **gate script** that
classifies AT BAR / DEBT / UNMEASURED and exits non-zero unless all are AT BAR
(example: `scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py`).
Report its output, not your impression. Never relax the bar constants; only add
measurements.

Corollary: a term is not clearable by an EXPLANATION of its residual
("threshold chatter", "irreducible amplification", "branch flips"). Those are
hypotheses. Clearing requires comparing the routine's INTERNALS stage by stage
and either reaching the bar or proving the deviation lives in the oracle's own
arithmetic.

## Rule 1c — Match the oracle's PRECISION, and verify it by printing the dtype

NEMO, MITgcm and Veros are **fp64** models. An oracle comparison run in
float32 is measuring your own rounding, not your physics. f32 eps = 1.19e-7,
so anything you "find" in the 1e-8..1e-5 band may be the dtype.

**Every oracle-fidelity evaluation runs fp64.** Set it explicitly:

```python
from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())
```

**`JAX_ENABLE_X64=1` IS NOT ENOUGH.** It permits f64 arrays; it does NOT change
legoESM's precision policy. Constructors cast to `get_policy().control`, which
**defaults to float32** — so a harness can run with x64 enabled and still build
its grid in single precision.

DOUBLE-CHECK IT — do not assume (this is Rule 10 applied to dtype). Print the
dtype of the arrays you are comparing *and* of the GEOMETRY behind them:

```python
for f in ("t_depth_ref", "dz_ref", "z_full_ref", "z_half_ref", "h_partial"):
    print(f, np.asarray(getattr(z_coord, f)).dtype)   # want float64
```

A ladder that is f32 while T/S are f64 is the easy case to miss: the state
looks right and the geometry silently is not (Rule 2's blind spot again).

Real case (#1226): `create_z_star_from_thicknesses` cast NEMO's f64 `gdept_1d`
to the policy control dtype = f32, losing ~7 digits — median |rel| 2.555e-8 =
**0.21 x f32 eps**, the fingerprint of single precision. Under fp64:

```
live gdept vs NEMO   2.163e-8  ->  1.199e-16
eos_rab alpha        1.322e-9  ->  0.000e+00
bn2 (err_norm)       4.322e-9  ->  1.413e-17
zdf_mxl nmln          10/9920  ->  0/9920
```

Four terms, one dtype. Weeks had gone into hunting a physical cause for an
"unexplained 3e-6..5e-5 residual band" that was substantially float32.

**Diagnostic tell:** if a residual's median sits near a fixed fraction of
machine eps (~0.2-0.5 x eps) and is roughly FLAT across unrelated terms, suspect
the dtype before the physics. A real discretisation error has structure; a
rounding floor does not.

Corollary for the gate: record the precision every measurement was taken at,
next to the number. A figure measured at a different precision than the current
default is STALE, exactly like a figure measured at a different commit.

**No-Frankenstein precision rule.** A selectable platform/precision policy is
allowed only with explicit oracle and toolchain provenance (compiler flags,
linked math library, and binary identity). It must be one shared implementation,
never a per-card guard, and it must preserve both JIT execution and autodiff.
The selector describes the oracle's arithmetic environment; it is not a physics
arm and cannot be used to conceal a card-specific residual.

## Rule 1d — Pin the oracle's TIME LEVEL per dump, in a registry that raises

A leapfrog oracle carries three time levels, and a routine is routinely called
with T/S at one and geometry at another:

```fortran
CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nnn )   ! stpmlf.F90:184  T/S BEFORE, geom NOW
CALL eos_rab( ts(:,:,:,:,Nnn), rab_n, Nnn )   ! stpmlf.F90:185
```

Compare a dump against the wrong level and you silently substitute
`|T_now - T_before|` for "error". **This does not look like noise.** That
difference is largest in the thermocline, so it renders as a beautifully
depth-structured signal — the most convincing possible disguise.

Real case (#1226), the same mistake made TWICE in one day: feeding NOW T/S
against BEFORE-level dumps produced a 7074-cell "tail", three 1.8% alpha
"outliers", and a "levels 6-8 structure" that got a written mechanistic
explanation. At the correct level: tail EMPTY, zero outliers, no structure.
The wrong-level numbers had already been committed to the gate.

**Never infer the level from the field name or from what you loaded last.**
Read the oracle's call site, then record it ONCE in a registry that
FAILS CLOSED on anything unregistered
(`ocean/fidelity/time_levels.py::time_level_for_dump`, which raises rather than
defaulting to "now"), and let `select_ts(dump, now=, before=)` pick — so the
correct level is the DEFAULT ACTION, not a thing to remember. Registering a
dump requires citing the `file:line` that proves it; an unsourced entry is a
guess, and a guess here is the whole failure mode.

**Tell:** a residual that is near-zero in a well-mixed layer, peaks at the
thermocline, and decays with depth has the shape of a T-tendency, not of a
discretisation error. Suspect the time level before you write a mechanism.

**How it got caught** (worth copying): invert the oracle's own polynomial for
the input it implies. NEMO's dumped alpha at one "outlier" implied T = 7.103 C
where we had fed 6.858 C — not roundoff, a *different temperature*. Numbers can
be argued about; an implied input that is 0.245 C off cannot.

## Rule 1e — Reconcile a disagreeing measurement BEFORE you record it

When a new probe disagrees with a recorded number, the temptation is to explain
why the OLD one was untrustworthy (wrong run directory, stale commit, worse
precision) and record the new one. That reasoning is backwards: a disagreement
is evidence that **one of them has a bug**, and you do not know which yet.

Real case (#1226, twice in one session): a rebuilt `ldf_slp` probe — with
better provenance in every respect, fp64 and both preconditions wired —
reported `wslpi` 0.999876/1.0028 against a recorded 0.999963/1.000524. I
attributed the gap to the old probe's known flaws and committed the new
numbers as "superseding on provenance". The new probe was the broken one: it
omitted `eos_depth="geometric"` from the config and silently took a different
density convention. Corrected, it reproduced the historical value exactly.

**A better harness does not make a measurement right.** Provenance is a reason
to *trust*, never a reason to *skip reconciling*.

Procedure when two measurements of the same quantity disagree:
1. Do NOT record either yet.
2. Find a quantity both probes should agree on exactly and check it — an input,
   a mask count, a cell census. Disagreement upstream localises the bug.
3. Diff the two CONFIGS field by field, not just the code. The #1226 bug was
   one missing kwarg that changed a physical convention, invisible in the
   numerics and silent at runtime because it had a default.
4. Only record once you can say WHY they differed.

Two independent probes that CONVERGE are much stronger evidence than either
alone — that convergence is the thing worth chasing, and it is what finally
established these four rows.

Corollary: a config field with a silent default is a trap for oracle work.
Prefer an explicit value at every probe call site, and diff the assembled
config against the production one rather than trusting that defaults match.

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

## Rule 12 — A faithful fix may expose a compensating error; keep walking

A change is eligible to land only if the changed operator is shown bit-exact
given NEMO's own inputs on every card it touches.

**"DISCHARGED" IS RESERVED FOR BIT-EXACT — 0 cells unequal, not "inside the
tolerance".** A row that is AT-BAR by a `1e-15` comparison while its own
`exact` field is false is AT-BAR-NOT-EXACT, and saying DISCHARGED of it
smuggles a tolerance into a bit-exactness claim. Two such rows survived three
rounds in one campaign's receipt with their own tables contradicting the word
beside them. If such a change makes a card
worse against NEMO, that is a second error exposed: the fix stays, the worsened
row enters that card's register as debt naming the boundary, and the next round
walks it. Never reverted, never waived silently, never a per-card switch.

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
