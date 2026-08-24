# The DINO wall 2-step surface flicker: a launch transient with no owner

Branch `fidelity/dino-wall-flicker`, off `fix/twin-euler-start-default`.
Campaign #1455 / #1226, the "one live physics thread" as of 2026-08-24.
Both adversarial reviews returned NO-SHIP on the first draft of this document;
everything below is the post-review version and the retractions are in it.

## Verdict

The two-step (2 x 2700 s) surface mode legoESM was reported to carry at **2.9x
NEMO's amplitude with 85% of it on land-adjacent rows** is a **launch
transient**, and neither of the brief's two ranked hypotheses survives:

* **Hypothesis 2, the time filter damps the mode less at walls — REFUTED, on a
  controlled common-excitation experiment.** Given the SAME imposed
  displacement, the two models ring down at the same rate: wall-band decay
  constant 5.19 steps (legoESM) against 5.20 (NEMO), ratio **0.999**, injected
  amplitude ratio **1.000**, r² = 0.97 on both fits.
* **Hypothesis 1, a wall stencil sustains a two-step source — REFUTED AT THE
  PUBLISHED SIZE.** The locus is what decides this, not the amplitude: the
  wall's share of the two-step variance falls from **0.852 to 0.181** over five
  days while NEMO's RISES from 0.044 to 0.204 — the oracle ends up MORE
  wall-enriched than the twin. A sustained wall source would hold its locus.
* **Hypothesis 3 was gated on 1 or 2 explaining a sustained excess.** Neither
  does, so it was not reached.

**The chain consequence, which is why this thread existed.** The flicker is
**refuted as the differential noise source** feeding the convective amplifier,
by direct measurement rather than by argument: two legoESM members from the
same day-180 state differing by the ensembles' own 1e-14 perturbation have a
member-minus-member two-step field whose wall **enrichment is 1.39** against a
single member's **22.4**. The transient is common-mode and it does not survive
into the difference field an ensemble spread is made of. With the convective
switch already shown to be a shared passive amplifier, **the amplification
asymmetry now has no named owner.**

## Retractions

Loud, because each was in the first draft of this document and each was caught
by review, not by me.

1. **"legoESM's DINO card ships `barotropic_forcing_centred = False` while
   NEMO runs `ln_bt_fw = .FALSE.`" is FALSE and is WITHDRAWN.** The card under
   test sets it to **True**
   (`packages/ocean/legoesm/ocean/experiments/dino.py:1582`, inside
   `DINO_RECIPES["nemo_dino_kamm_mlf"]`, verified in the worktree the run
   actually used). I read the dataclass default at line 301 and the
   forward-Euler base card's comment block, neither of which is the
   configuration the run resolves — the exact "prove the path executes before
   blaming a line" failure. There was never a mismatch to suspect.
2. **"The transient is legoESM's own operator response, not an initial-
   condition transfer gap" was WRONG ON ITS FACTS.** A transfer gap does
   exist: the turbulence state is NOT carried over from NEMO's restart
   (`bridge_tke` defaults off, and no TKE line appears in the day-180 run log),
   so legoESM starts with cold-started vertical mixing. The claim's
   *conclusion* survives, but only because it was measured afterwards — see
   below — not because the premise held.
3. **"legoESM's late-time wall amplitude is if anything LOWER than NEMO's"
   is WITHDRAWN.** It came from subtracting a leakage floor measured over the
   first eight samples from an amplitude measured over the last half — a
   mismatched-window comparison, and inside the transient the floor is
   inflated by the very signal under study. Window-matched, the wall floor
   ratio is 1.2x, not 4x, and the corrected amplitudes are 8.82e-7 (legoESM)
   against 8.75e-7 (NEMO): a 1% difference in the OTHER direction. Both
   reviewers found this independently. The subtraction is now simply dropped;
   it was a 6% correction resting on an unverified orthogonality assumption.
4. **"Equals NEMO to 3%" is WITHDRAWN as spurious precision.** See the
   estimator table below: the same data gives 1.03, 1.21 or 1.33 depending on
   which mean is taken, with a 90% block-bootstrap interval of [0.73, 1.28].
5. **I over-claimed novelty.** That the mode decays to 1.1x over the second
   half was ALREADY published — `dino_eta_wave_field_result.md:649` says so and
   then names the flicker the live thread anyway. What is new here is the
   per-region decomposition, the locus-versus-time measurement, the
   common-excitation damping test, the second start state, and the two
   decisive runs below.
6. **The day-190 "CONFIRMS" is downgraded to "not inconsistent with".** The
   pre-registered thresholds (1.5 / 2.5) sit INSIDE the estimator's own noise:
   its bootstrap interval is [0.59, 1.86]. The rule was under-specified and
   saying so is cheaper than defending it.

## Why no new physics run was needed to answer the brief's first question

The brief asked whether 45-minute sampling can see a 90-minute mode. It can,
exactly: DINO's timestep IS 2700 s and the artifacts were dumped every step,
so the mode sits precisely at the series' Nyquist frequency. The whole
decay-versus-generation split came out of data already on disk. The four runs
that were made were forced by review, not by sampling.

## The measurements

Instrument: `scripts/validate/ocean_fidelity/dino_1226/eta_flicker_decay.py`,
built on the campaign's `eta_wave_twin.py` (its Nyquist operator, the mesh's
own wet mask, the wall/equator/interior split with the re-entrant channel's
periodic seam correctly excluded, and its leakage floor — which this branch
extends to be region- and window-aware). 18 unit tests.

Two instrument defects were found and fixed before any number here was
believed, both of them controls that could not fail:

* the shared land-poison plants a CONSTANT, and the Nyquist operator
  annihilates constants exactly — on the real data (dry-cell eta is exactly
  0.0 at every step in both models) it moved the unmasked statistic by a ratio
  of 1.000000. The control now plants an ALTERNATING poison.
* even then it could not fail, because the production area weights are already
  zero on land, so land carried no weight before the mask was ever consulted.
  The control now runs against UNMASKED weights, leaving the region mask as
  the only defence — and a test proves a mask that admits land turns it red.

### 1. Damping, on a common excitation (the clean test)

Each model minus its OWN free run, so both ring down from an identical imposed
bump. This is the lane in which decay rates are comparable at all; in the free
lane NEMO has no launch transient, so there is no NEMO rate to be slower than.

| region | injected amplitude (both) | decay constant, NEMO | legoESM | ratio | r² |
|---|---|---|---|---|---|
| wall | 3.021e-4 m | 5.20 steps | 5.19 | **0.999** | 0.969 |
| interior | 2.305e-4 m | 4.18 steps | 4.18 | 1.000 | 0.986 |
| equator | 9.702e-5 m | 14.78 steps | 14.74 | 0.997 | 0.762 |
| whole basin | 2.284e-4 m | 4.41 steps | 4.40 | 0.998 | 0.988 |

The statistic is genuinely wall-sensitive rather than interior bleed: the wall
band's decay constant differs from the interior's (5.20 against 4.18) in BOTH
models, and the two models agree on the WALL value to 0.2%. Hypothesis 2 is
dead. This also removes the "a source is masked late by saturation" objection:
the ringdown is a clean exponential over two decades of amplitude, so the
damping is linear across the range that matters.

The honest limit of this lane: the excitation is basin-scale (its own wall
enrichment is 1.08), so it measures how each model's wall band damps a wave
ARRIVING at it, not how it would damp something injected locally by a wall
stencil. Those coincide for a linear operator, which the exponential ringdown
says this one is over the amplitude range involved — but the statement is
about damping, and it does not by itself exclude a local source.

### 2. Locus versus time, day 180 — the statistic that decides hypothesis 1

Share of the two-step variance carried by the 424-cell wall band (3.8% of the
wet area):

| | first sample | first 8 | last half |
|---|---|---|---|
| legoESM | **0.852** (enr 22.4) | 0.485 | **0.181** (enr 4.7) |
| NEMO | 0.044 (enr 1.15) | 0.069 | **0.204** (enr 5.4) |

The wall localisation is a first-SAMPLE property. It is gone within a day, and
what it decays TO is a state in which the oracle is the more wall-enriched of
the two.

### 3. Amplitude, day 180 — reported with its uncertainty, not without

Wall band, window-matched floors (both sides clear 2x their own floor, so the
comparison is resolved):

| | first 8 | last half | floor (last half) |
|---|---|---|---|
| NEMO | 5.075e-7 m | 9.128e-7 m | 2.62e-7 m |
| legoESM | 3.346e-6 m | 9.381e-7 m | 3.16e-7 m |
| ratio | **6.59** | see below | |

Late-window ratio under three defensible estimators — mean of per-sample rms
**1.03**, rms over time **1.21**, median cell **1.33** — with a 90% moving-block
bootstrap interval of **[0.73, 1.28]** on the first of them. The honest
statement is that **a sustained wall excess of the published size (6.6x at the
wall, 2.9x global) is excluded, and a residual up to roughly 2x is not.**

The two late fields are also structurally different in a way an rms hides, and
this is the strongest surviving caveat: **82% of NEMO's late wall variance
lives in ten cells** (four carry half of it) against 46% for legoESM (twelve
carry half). Same magnitude, different field — NEMO's northern boundary column
against legoESM's distributed wall band. They are, further, spatially
uncorrelated late on: the area-weighted spatial correlation of the two models'
wall two-step fields over the last half is **-0.038 with a standard error of
0.055**, i.e. indistinguishable from zero (it is +0.12 over the first eight,
so the early transient does partly share structure and the late noise does
not). The two models share a LEVEL at the wall, not a mode.

### 4. Launch-independence — the transient does not set the late level

Two legoESM starts whose launch transients differ 4.9x converge to the same
late level, and both collapse their locus:

| day-180 arm | wall first 8 | wall last half | wall share, first sample -> last half |
|---|---|---|---|
| legacy Euler start | 1.633e-5 m (32.2x) | 1.194e-6 m (1.31x) | 0.358 -> 0.182 |
| bridged start | 3.346e-6 m (6.59x) | 9.381e-7 m (1.03x) | 0.852 -> 0.181 |

### 5. A second start state — day 190, pre-registered

One variable, the start state: same NEMO binary (md5 identical), same namelist
but the three step counters, same probe, same mesh. The legoESM arm ran the
day-180 arm's exact model code (the eta-waves worktree at `b0606080`, which
differs from the day-180 run's `31b7df02` in docs, probes and tests only —
`git diff --stat 31b7df02 b0606080 -- packages/ src/` is empty) with a one-line
change to the restart filename, which IS the variable under test.

Wall first-8 ratio **3.41**; wall last-half **1.22** (bootstrap [0.59, 1.86]);
wall share **0.840 -> 0.136** against NEMO's 0.174 -> 0.086. Same shape as day
180 — a launch transient whose locus collapses. Not a passed gate, because the
thresholds were inside the noise; a consistent second observation.

### 6. Is the cold turbulence start the transient's owner? — pre-registered, NO

The one initial-condition gap that survives review's scrutiny. Registered rule:
below 1.5e-6 m means the cold TKE start is a material contributor, above
2.7e-6 m means it is not.

| day-180 bridged arm | wall first 8 | wall share, first sample |
|---|---|---|
| TKE cold-started (production default) | 3.3456e-6 m | 0.852 |
| TKE bridged from NEMO's restart | 3.3185e-6 m | 0.853 |

**0.99x — the transient does not move at all.** The cold turbulence start is
REFUTED as its owner. (The gap is real and worth a separate look for its own
sake; it just does not make this transient.)

### 7. Does it cancel in an ensemble difference? — pre-registered, YES

The claim review would not accept by argument, since a common-mode
perturbation can still move threshold-gated physics (convective adjustment,
the flux limiter) onto a different branch. Measured instead: a second legoESM
member from the same day-180 state with the harness's own 1e-14 relative
perturbation (realised max relative dT 5.0e-14).

| two-step field | wall share | wall enrichment |
|---|---|---|
| a single member, first sample | 0.852 | **22.4** |
| member minus member, first sample | 0.053 | **1.39** |
| member minus member, last half | 0.077 | 2.03 |

Registered threshold was enrichment <= 3. **1.39.** The launch transient
cancels; it is not in the difference field, and it therefore cannot be the
differential noise the amplification asymmetry needs.

## The transient's footprint, for whoever picks this up

Splitting land-adjacent cells by which direction the land lies in, day 180,
first sample, area-weighted share of the two-step variance:

| | meridional wall (land E/W) | zonal wall (land N/S) | corner | interior |
|---|---|---|---|---|
| legoESM | **85.1%**, enrichment **23.9** | 0.1%, 0.4 | 0.0% | 14.8%, 0.2 |
| NEMO | 4.3%, 1.2 | 0.0%, 0.2 | 0.0% | 95.6%, 1.0 |

Entirely on the north-south running basin sidewalls, where the zonal
barotropic flux must vanish, in 3.56% of the area. Cells whose land neighbour
is to the north or south are DEPLETED. The domain's outermost rows hold no wet
cells at all in DINO, which rules out the free-slip domain edge outright.

That geometry — the zonal direction, at land-adjacent faces, with no
meridional counterpart — is what any future explanation has to match. No
suspect is named here: the one I proposed was refuted (retraction 1), the one
review proposed was measured and refuted (section 6), and the phenomenon is
worth about five hours of a five-day run.

## The brief's own first candidate, closed

The brief named "the known non-halo-aware boundary stencil in one diagnostic"
as the leading wall-stencil suspect, to be ruled in or out by checking whether
it is diagnostic-only. **The premise is wrong and the candidate is closed for a
different reason.** The stencil is the Hollingsworth-Kallberg kinetic-energy
gradient (`ke_gradient_scheme = "hollingsworth"`, NEMO `nn_dynkeg=1`), and it
is resolved ACTIVE on the card under test — it is a momentum-tendency term, not
a diagnostic. What rules it out here is the other half:

* its boundary fill (edge replication rather than zero-Dirichlet) differs by
  roughly a factor of four from the oracle's convention only where a WET
  boundary row carries non-zero velocity. Verified on this run's own output:
  the maximum absolute u and v on dry cells is exactly 0.0 at every step, so
  the two conventions coincide cell for cell here. (They would NOT coincide on
  a restart from a model that does not zero land velocities, or in a
  wetting-drying case — that debt is real and is recorded in
  `dino_wall_ldf_alignment.md`, it is simply not this run's problem.)
* the missing halo exchange is inert at single rank, which is how the twin
  runs.

## What was NOT done, and why

No fix, no A/B, and no ensemble-spread-ratio prediction. The brief gated all
three on naming a source with matching geometry AND the right
generation/damping signature. The damping signature came back at ratio 0.999,
the sustained excess is not there to fix, and the chain-closing prediction's
premise — that fixing the flicker collapses the amplification asymmetry — is
refuted by section 7. Registering a prediction whose premise is already
falsified would be theatre.

## Two prose defects found in `dino.py` while chasing retraction 1

Reported, not fixed, since they are outside this branch's scope:

* the base card's header comment says `nn_e=30` while the resolved value sixty
  lines below is 23, with a note that 30 is only the auto-off fallback. A grep
  finds the stale number first.
* the forward-Euler block's "faithfully guarded OUT of this frame (do NOT
  re-wire them here)" describes a property of that frame but sits in the base
  dictionary that the leapfrog card overrides — read literally it forbids
  exactly what line 1573 does. This is what sent me to the wrong line.

One note on the shared instrument, since this branch edits a file whose
numbers are already published: the extracted reduction keeps the original
per-sample loop rather than vectorising it. A vectorised sum agrees to twelve
digits but not bit-for-bit, and a refactor of a diagnostic must not move a
number that has already been quoted. Verified bit-identical against the
pre-refactor module on both DINO artifacts for `two_dt_mode` (every sample),
its first-8 and last-half means, its locus shares, and `two_dt_leakage_floor`
at its old call site; a unit test goes red if anyone vectorises it again.

Also worth a stamp: neither the day-180 artifact nor its log records the recipe
name or whether the turbulence state was bridged, so the configuration behind
those numbers had to be inferred. Every arm on this branch was re-run with the
configuration recorded in its log.
