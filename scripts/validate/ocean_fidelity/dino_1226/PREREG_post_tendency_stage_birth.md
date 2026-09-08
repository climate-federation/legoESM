# PRE-REGISTRATION — where the wall-row velocity difference is BORN inside one step

Written **before** the probe produced any number.  #1455.  Branch
`fidelity/dino-basin-budget-1yr`.

## Why

The southern-basin deficit's owner has been sought entirely inside the explicit
tendency calculation, and every term there has been compared and cleared.
Roughly a third of what reaches the wall-row velocity happens **after** the
tendency, and none of it has been compared against the oracle:

1. the split-explicit **barotropic solve**;
2. the **leap-frog recombination** of the split velocity;
3. the **implicit vertical solve** (gate row blank);
4. the **after-level reconciliation**, which *sets* the depth-uniform velocity
   while the deficit *is* depth-uniform;
5. the **Asselin time filter**.

This probe bisects one committed step across those boundaries.

## Design

* **Lane**: day 180, `RUN_SEQDUMP_D180_1R`, kt 5760 → 5761.  Both models are
  identical at entry (the bridged state is gated to `max|dT| = max|d_eta| = 0`),
  so a one-step difference is a pure OPERATOR difference with no trajectory
  feedback in it.
* **Card**: `nemo_dino_kamm_mlf`, which runs `outer_integrator="leapfrog"` —
  so the instrumented path is `_leapfrog_step`, NOT `_nemo_mlf_step`.  Verified
  from the resolved recipe, not from the wiring prose.
* **Capture**: wrap the three stage methods on the model class and record their
  inputs and outputs.  Run through the production `model.step` entry point under
  `jax.disable_jit()` so the captured arrays are concrete and the path exercised
  is the production one.
* **NEMO side**, every dump's time level taken from
  `legoesm.ocean.fidelity.time_levels`, never assumed:
  | boundary | legoESM | NEMO |
  |---|---|---|
  | after tendency + barotropic solve, recombined | input to `_apply_implicit_vertical_mixing` | `naa_A = (uu(Kbb) + rDt·uu(Krhs))·umask`, from the restart `ub` and `stp_dump_07_dynspg` |
  | after the implicit vertical solve | its output | `stp_dump_08_dynzdf_{u,v}` |
  | after the after-level reconciliation | output of `_apply_after_level_reconcile` | `baro_dump_{u,v}_after` (its own bracket, `baro_dump_*_before`, is checked too) |
  | committed after-state | the returned state | `seq_dump_postlbc_{u,v}_aaa_kt00005761` |

## Statistics, and their bars — registered now

All three are computed on the **min-rule thickness weighting**, never a layer
average.  The layer average was the instrument defect that forced the campaign's
last rescore (it over-weights the 10 m surface layer ~54× against the 545 m
bottom); every number this probe prints is thickness-weighted, and the probe
prints the layer-averaged value alongside it **only** as a labelled contrast.

* **S1 — depth-uniform wall velocity difference** `[m/s]`.
  The thickness-weighted column mean of `u_lego − u_nemo`, averaged over wall
  rows 1–4 over wet u-columns.
  **Bar: 1.6e-7 m/s per step.**  Derivation: the day-90 deficit is 4.6e-4 m/s;
  2880 steps in 90 days; 4.6e-4 / 2880 = 1.6e-7.  A stage that is born below
  this bar cannot reach the deficit even at PERFECT linear retention.
* **S2 — wall-band transport difference** `[Sv]`.
  `Σ_rows Σ_i e2u · H_u · Δū` over rows 1–4.
  **Bar: 1.0e-4 Sv per step** (0.288 Sv / 2880, same construction).
* **S3 — wall enrichment** (dimensionless).
  |S1| over wall rows 1–4 divided by |S1| over basin rows 6–13.
  **Bar: 2.0.**  The deficit is wall-concentrated; a stage that is born
  uniformly across the basin has the wrong shape to own it.

**These bars are one-directional and I register that limitation now.**  Clearing
all three is NECESSARY, not sufficient — this campaign has already measured a
badly non-linear map from operator error to transport (closing 96–98 % of a real
operator error moved the transport by 0.07 noise floors), so a per-step
difference cannot be multiplied by 2880 and called a prediction.  A stage
**below** its bar is refuted as the owner under linear retention; a stage
**above** all three is PROMOTED to the next test, which is the offline shape
comparison against the 34 % lobe fingerprint, and only a shape match earns an
option plus an A/B.

## Sign convention, fixed now

Every difference is `lego − nemo`.  The deficit is **negative** in that
convention (legoESM's flow is too weak in the eastward sense by 4.6e-4 m/s).  A
stage that is born POSITIVE is pushing the wrong way and cannot be the owner
however large it is; that is recorded as a separate REFUTE arm, not folded into
a magnitude.

## The implicit solve's 15 % figure

The one end-to-end number attached to the implicit vertical solve is a "15 %
discrepancy" that exists only in prose.  It is decomposed here into the three
quantities that can be measured separately, using the existing bracket
(`s17_dynzdf_bracket.py`, which already feeds the solve NEMO's own input):

* **input difference** — legoESM's own pre-solve velocity against NEMO's;
* **operation difference** — NEMO's input in, legoESM's solve run, compared
  against NEMO's post-stage dump (the SEEDED arm);
* **end-to-end difference** — legoESM's own input, legoESM's solve, against
  NEMO's post-stage dump (the FREE arm).

If the prose figure has no probe behind it, that is recorded as prose without
provenance and the figure is retired rather than repeated.

## Controls the probe must pass before any number is read

1. **Day-0 identity** — the bridged state matches the restart to `0.0`.
2. **Planted control** — rolling the NEMO reference by one row must blow every
   statistic up; a metric that survives the roll is translation-blind and its
   number is not read.
3. **Closure** — the per-stage births must sum to the end-to-end committed
   difference, to roundoff.  A budget that does not close means a stage boundary
   is missing and no attribution is made.
4. **NaN is fatal**; provenance (HEAD, dirty count, lane, dump paths, dtypes,
   e3t mode) is stamped on every run; the probe prints numbers and never prints
   a verdict.


---

# AMENDMENT, recorded AFTER the run and labelled as such

The two magnitude bars above are **not mutually consistent**, and I did not
notice before running.  `BAR_S1` was derived from the deficit's depth-uniform
VELOCITY (4.6e-4 m/s, a whole-southern-basin figure) and `BAR_S2` from the wall
band's TRANSPORT gap (0.288 Sv, a four-row figure).  Those describe different
regions, and on this band's cross-section (~2e10 m2) they disagree by about a
factor of thirty.  A result can therefore sit under one and over the other, and
the committed one-step difference does exactly that: +2.2e-8 m/s is 7x under
`BAR_S1` while +4.9e-4 Sv is 4.9x over `BAR_S2`.

**Nothing above is edited and no bar is moved.**  What changes is which legs the
conclusion is allowed to rest on: the SIGN leg and the SHAPE leg, both fixed in
advance and both unambiguous, carry it.  The magnitude leg is recorded as
INCONCLUSIVE for this probe until the two published figures are reconciled on a
common region — which is itself a finding, and belongs to whoever next quotes
either of them.
