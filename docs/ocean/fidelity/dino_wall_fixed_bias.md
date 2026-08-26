# The barotropic loop's fixed bias: not a wall object, and not the wind

**Outcome.** The state-constant part of the barotropic end-wall residual is
characterised directly, without any projector; the one unstarted candidate on
the card — the wind-stress placement difference — is **excluded**; and a new
candidate with a measured coefficient is named and ranked. Two loop ingredients
are refuted by measurement and two more excluded for free.

**The headline is a retraction.** This residual is **not a wall object at all**.
In the zonal component it is the largest of three alternating, zonally coherent,
basin-scale bands, and at the southern wall it is 1.004× the interior median. In
the wall-normal component the relative deficit peaks in the *interior*, at row
134, exceeding both walls. Two claims made in the first draft of this document —
"a northern wall ramp" and "ten times the interior, boundary-localised" — are
withdrawn below, along with a third about the wind's meridional channel.

**The leading candidate** is a horizontal grid metric: legoESM's zonal width of
the meridional cell face is 3.3e-05 too large at both walls, the only metric off
by more than roundoff. It predicts the wall-normal deficit at the walls to
within 5-28 % with no fitted parameter — and fails to reproduce its basin-wide
latitude structure. One cheap pre-registered arm settles it (§6).

Artifact: `results/dino_1455/baro_fixed_bias_wall_map.json`.
Probe: `scripts/validate/ocean_fidelity/dino_1226/baro_fixed_bias_wall_map.py`.
Tests: `tests/ocean/fidelity/test_baro_fixed_bias_wall_map.py` (42).

Everything below is measured in the arm where **NEMO's own frozen slow forcing
is substituted into legoESM's own loop** from a bit-identical entry state
(`max|du| = max|dv| = max|d_eta| = 0` against NEMO's restart). In that arm both
models integrate the same forcing from the same state over the same 68 substeps
with the same weights, so what survives belongs to the loop — and every
difference in how legoESM *assembles* its slow forcing has been substituted
away. That last clause is load-bearing for the wind section.

## 1. It really is a fixed field

| | per-state RMS | state-mean RMS | worst departure | min cross-state corr |
|---|---|---|---|---|
| zonal (tangential) | 1.2232–1.2239e-07 | 1.2235e-07 | **1.23 %** | +0.99986 |
| meridional (wall-normal) | 1.8471–1.8482e-07 | 1.8475e-07 | **1.53 %** | +0.99979 |

**And restricted to the rows the claim is about** — the reduction should match
the claim, so the wall rows are also measured on their own, where the field is
*more* nearly fixed, not less:

| | worst departure at the wall rows | min cross-state corr |
|---|---|---|
| zonal | **0.056 %** | +1.00000 |
| meridional | **0.102 %** | +1.00000 |

The correlation is kept **signed** on purpose: an anti-correlated pair is the
two-step mode itself and must never be allowed to read as state-constant. The
statistic is proved on a constant field and on an alternating field with known
answers before it is used.

**Caveat on the span.** These five states are five *consecutive* steps, so they
sample a nearly unchanged ocean. "Fixed" is therefore established over ~5
steps, not over a season. See §6 for the measurement that would extend it and
the artifact that blocks it.

## 2. RETRACTED as a wall object: it is a three-band basin-scale field

The first version of this section called rows 184-196 a "northern wall ramp".
**That over-reads it, and the wall framing is withdrawn.** Equally coherent,
one-signed, zonally uniform bands sit in the pure interior — the northern lobe
is only the largest of three, and they alternate in sign:

| rows | signed zonal mean | coherence | sign |
|---|---|---|---|
| 57-73 | +7.1e-08 … +1.02e-07 | 0.86-0.93 | positive |
| 121-153 | +5.0e-08 … +8.3e-08 | 0.79-0.96 | positive |
| 185-197 | −1.5e-07 … −5.5e-07 | 0.93-0.98 | **negative** |

And the southern end is not a feature at all in this component: 6.485e-08
against an interior median of 6.459e-08 is a ratio of **1.004**. So what is
measured is a meridionally banded basin-scale residual whose northern lobe
happens to reach the wall — not a boundary object.

What *is* true, and is the useful half: the field is strongly one-signed within
each band (coherence up to 0.98) and is **not** a grid-scale checkerboard (the
two-gridpoint alternating share of its RMS is 0.003-0.025 in the zonal
component and 0.056-0.094 in the meridional). A basin-scale banded structure
with almost no grid-scale content is a different suspect list from a wall
artifact.

### The wall-normal component, and a second retraction

At a *zonal* wall the wall-normal component is the *meridional* one, and it is
the convergence of the wall-normal transport that sets the sea surface against
a closed wall. Its coherence is 0.01-0.03 — a zero-mean pattern in longitude,
not an offset — and it is anti-correlated per cell with NEMO's own meridional
barotropic velocity (−0.815 at the southern wall row, −0.634 at the northern,
about −0.9 a few rows in). legoESM's wall-normal barotropic transport is
systematically *weaker* than NEMO's.

**But "about ten times the interior, and boundary-localised" is retracted.**
That factor was a denominator choice. Measured as bias RMS over velocity RMS
per row, two interior rows exceed *both* walls:

| row | 1 (S wall) | 65 | 100 | 129 | 137 | 196 (N wall) |
|---|---|---|---|---|---|---|
| relative deficit | 4.26e-05 | 2.91e-05 | 3.35e-06 | **4.56e-05** | **5.34e-05** | 3.51e-05 |

The "10×" came from taking rows near 100 as the interior baseline, and those
rows are a local *minimum* of the residual field. The relative deficit in fact
peaks at row 134. Restate it as: a broad relative weakening of the wall-normal
transport with a basin-scale latitude structure, of order 3-5e-05, in which the
walls are unexceptional.

## 3. The geometry is nearly mirror-symmetric; the bias is not

| | southern wall | northern wall | ratio N/S |
|---|---|---|---|
| latitude | −69.504 | +69.504 | \|Δ\| = 0.0000° |
| meridional grid spacing | 38934.6 m | 38934.6 m | **1.0000** |
| mean column depth | 2231.3 m | 2558.3 m | 1.147 |
| **zonal bias RMS** | 6.485e-08 | 7.605e-07 | **11.73** |

The two walls sit at the same latitude to four decimals and carry identical
meridional grid spacing, but the column depth is **14.7 % deeper** at the north
— so "symmetric" is the wrong word and the probe no longer uses it: it raises a
`GEOMETRY IS NOT SYMMETRIC` flag above 5 % and prints the largest departure. The
point survives the correction, quantitatively: a driver scaling linearly with
column depth could produce about **1.15×** of north/south contrast, and the
measured zonal contrast is **11.7×**. The depth asymmetry does not account for
it.

Whatever sets the zonal ramp is asymmetric, and the obvious asymmetric field is
the flow: NEMO's own barotropic zonal velocity is 11.67× larger at the northern
wall row than at the southern. (These are **tracer**-point geometry numbers; the
u and v wall rows are at different indices because of the staggering, and the
probe labels the grid rather than letting the reader assume.)

### The amplitude agreement: the conclusion holds, the reasoning did not

The bias amplitude ratio (11.73) matches the velocity amplitude ratio (11.67)
to 0.5 %, which looks like "the bias is a fixed fraction of the local
velocity" — one number, 8.3e-06, at both walls. The first version called this
an amplitude coincidence and refused it on the per-**cell** correlation, which
is −0.504 (south) and +0.051 (north).

**The conclusion survives; the evidence for it was weak and is replaced.** At
rows 193-197 the velocity is itself nearly zonally uniform (coherence
0.92-0.99) and so is the bias, so the per-cell correlation is computed on the
small zonally-varying leftovers of two near-constant fields and is close to
blind there. What actually refutes pointwise proportionality is that the ratio
of zonal means *saturates*: it falls from +2.8e-05 at row 193 to +6.2e-06 at
row 197 while the zonal-mean velocity **rises** 6.7-fold. No constant of
proportionality does that.

**And the amplitude agreement itself is not noise — it is a clue, and treating
it as a coincidence cost a candidate.** A mechanism with a north/south
*symmetric coefficient error* acting on an *asymmetric flow* predicts exactly
this: matching amplitude ratios, symmetric geometry, no pointwise
proportionality. That is the candidate in §6.

The `AMPLITUDE-ONLY` flag stays in the probe — it correctly says "these two
rows do not establish proportionality" — but it is a veto on one reading, not a
verdict that the agreement is meaningless.

## 4. Ingredient attribution: two refuted, two free, three open

Freeze-and-vary on loop **inputs** only — no in-loop surgery.

| ingredient | how tested | result |
|---|---|---|
| slow forcing | NEMO's own `zu_frc`/`zv_frc`/`ssh_frc` substituted | removes only **3.1×/4.2×** of the constant part (it removes ~300× of the small varying part) |
| bottom drag | NEMO's own dumped `rCdU_bot`, face-averaged by NEMO's own `dyn_drg_init` rule | **REFUTED**, collapse **0.1 %** against a pre-registered bar (owner >50 %, refuted <10 %) |
| barotropic lateral diffusion | card sets the coefficient to 0.0 | **not active** — excluded at no cost |
| barotropic divergence damping | config default 0.0, card does not override | **not active** — excluded at no cost |
| face depths | — | **open** |
| Coriolis / EEN | — | **open** |
| continuity / SSH update, BEBT blend, time filter | — | **open** |

The drag arm's staggering control separates the correct face mapping from the
best wrong one by 215×, so its null is a null about drag and not about the
alignment. Input substitution has therefore **not** localised the constant
bias: neither substitutable input owns it.

## 5. The wind-stress placement difference does not match

**Both models' wiring, read from source.** NEMO applies the wind stress
**twice, independently**: explicitly into the barotropic slow forcing
(`dynspg_ts.F90:443-444`, as `r1_rho0 · ½(utau_b + utauU) · r1_hu(Kmm)`), and
again as the top boundary condition *inside* the implicit vertical solve
(`dynzdf.F90:353-363`). Because `dyn_spg` runs before `dyn_zdf` in the step
(`stpmlf.F90:332` vs `:396`), the implicit application cannot feed the same
step's barotropic loop — which is precisely why the barotropic term is
re-derived explicitly. legoESM sets the implicit-stress option to false on every
shipped DINO card, so its wind is an explicit top-cell source that reaches the
slow forcing only through the depth-mean of the momentum tendency.

**A correction to the card's reason string.** The standing note says "NEMO
applies the stress inside the implicit vertical solve … legoESM's card applies
it explicitly", framed as the placement difference. That is only half the
wiring: for the *barotropic* forcing — the thing under test here — NEMO applies
it **explicitly too**. Separately, the placement term did **not** need
reconstructing: NEMO dumps its own barotropic wind increment
(`wnd_dump_z{u,v}_frc_inc.bin`), so the candidate's footprint is read, not
modelled. (The "dead dump slot" finding in the previous commit concerns the
per-term *trend* diagnostics, a different instrument; both statements stand.)

**THE EXCLUSION RESTS ON ONE GROUND, AND ONLY ONE.** An earlier version of
this document listed four and called them "decreasing strength". Adversarial
review took the second apart; the fourth never worked at the southern wall. The
honest structure is one decider and two partial supports.

**The decider — structural.** The fixed bias is measured in the arm where
NEMO's own `zu_frc`/`zv_frc` are substituted into legoESM's loop. Every
difference in how legoESM *assembles* its slow forcing, wind term included, is
already removed there by construction. Any wind-placement difference lives in
that assembly, so in this arm it is identically zero. **This alone settles it.**

**Support (partial) — shape, zonal channel only.** The zonal wind increment
peaks at row 48, in the westerly band, and at the wall rows is **571×** (south)
and **1308×** below that peak; its whole-domain row profile correlates with the
bias's at −0.187. The candidate vanishes where the bias is largest.

**Support (partial, and it fails at one wall) — size.** Deliberately generous:
100 % of the wind term, not the small measured placement offset.

| | measured response (0.148) | response 1.0 |
|---|---|---|
| zonal, south wall | 0.79× — **does not exclude** | 0.12× — **does not exclude** |
| zonal, north wall | 21.3× — excludes | 3.2× — excludes |
| rotated meridional, south | 2.2× — **does not exclude** | 0.32× — **does not exclude** |
| rotated meridional, north | 26.2× — excludes | 3.9× — excludes |

At the southern wall the wind term is comparable to or larger than the bias in
both channels. Size excludes at the north and **not** at the south, and the
probe now prints that verdict per row rather than leaving it to prose.

### RETRACTED: "the candidate has no wall-normal component at all"

The previous version's second ground said: NEMO's meridional wind increment is
exactly zero over all 10 348 cells (true — DINO's wind is zonal-only on both
sides), *therefore* the candidate cannot produce a wall-normal bias. **That is
a fact about the forcing being read as a fact about the response, and it is
wrong.** A zonal forcing integrated inside a *rotating* barotropic loop deposits
meridional velocity through Coriolis. Working it through with the loop's own
weights — `δv = f · F_u · dt² · Σ w_k k² / 2` — the rotated deposit at the
southern wall is **2.19e-07 m/s** against a measured meridional bias of
7.03e-08, i.e. the candidate is **3.1× larger** than the thing it was supposed
to be excluded from. The bound is generous (no bottom drag, no lateral
friction, none of the opposing sea-surface-gradient response), but a generous
bound that fails to exclude is still a failure to exclude.

The rotated bound is now computed by the probe from NEMO's own Coriolis
parameter, printed per wall with an explicit `DOES NOT EXCLUDE`, and pinned by
a test that it is linear in *f* and exactly zero at *f* = 0. Confusing a null
forcing with a null response is a mistake this campaign has paid for before, so
the refutation is kept in the source rather than only here.

**Verdict: NOT A MATCH**, on the structural ground. The chain that a match would
have unlocked — the bias going to zero at the walls, the member-difference
enrichment collapsing, the basin torque gap moving — is **not** registered,
because its premise failed.

## 6. The leading candidate, named with a measured coefficient

Raised by mechanism review and **verified here from first principles, not
relayed**: legoESM's *zonal width of the meridional cell face* is too large by
about **3.3e-05** at both walls, and it is the only horizontal metric off by
more than roundoff.

NEMO builds DINO's mesh isotropically — its two v-face scale factors are equal
**bit-for-bit** — with both evaluated at NEMO's own v-point latitude. legoESM's
matching convention fixes the *meridional* width but leaves the *zonal* one on
the cosine of the midpoint between the two adjacent tracer rows. On DINO's
stretched meridional grid that midpoint is not NEMO's v-point latitude; they
differ by up to 0.0011°. The resulting gap:

| row | 1 (S wall) | 50 | 99 (equator) | 145 | 196 (N wall) |
|---|---|---|---|---|---|
| legoESM wider by | +3.33e-05 | +1.81e-05 | +2.9e-09 | +1.71e-05 | +3.33e-05 |

**Why it is the leading candidate.** It sits *inside* the loop, so substituting
NEMO's slow forcing does not remove it. It feeds the metric-complete rotation
coefficient, which the card under test has active, and a rotation coefficient
too large by a fraction makes the balanced wall-**normal** velocity too small by
the same fraction — the right sign and the right order for the deficit in §2. At
the two wall rows the prediction lands at **1.05-1.28× the measured deficit with
no fitted parameter**.

**Why it is not yet the owner, by this document's own standard.** Two rows
agreeing is exactly the reading §3 had to retract. Over all 196 wet rows the
row-wise correlation between the predicted gap and the measured relative
deficit is **−0.045** — essentially zero — and the per-row ratio swings from
0.15 to 128. The measured deficit peaks at row 134; the predicted gap peaks at
the walls. *(The mechanism reviewer reported +0.52 for this correlation, over a
selected subset of rows. Over all rows, unselected, it is −0.045. The
disagreement is entirely the row selection, and the unselected number is the
one reported here.)*

So: right order, right sign, right place at the walls, **wrong basin-wide
shape**. That is a lead worth one cheap arm, not a verdict.

### The ranked remainder, and the pre-registered next test

1. **Override the v-face zonal metric with NEMO's own array and rerun the
   in-loop arm** from the same bit-identical entry state — the same
   freeze-and-vary that refuted bottom drag. **Pre-register: OWNER if the
   state-constant wall-normal residual collapses by more than 50 %; REFUTED
   below 10 %.** Include the staggering control (feed the un-shifted array),
   which must make things clearly worse. This settles §6 either way and is the
   only test on this list that can.
2. **Face depths**, measurable offline the same way the metrics were — compare
   the loop's face depths against NEMO's dumped ones before spending a run.
3. **Discriminate "the bias tracks the flow" from "the bias tracks the
   geometry."** Cheapest version, suggested by review and better than the
   two-day comparison: rerun the in-loop arm from a **zero entry state** with
   the same frozen forcing. Because the loop is near-linear that splits the
   residual into a forcing-driven part and a state-proportional part in two
   runs, exactly. *(The two-day route is blocked: the day-180/day-230 pair on
   disk predates two fixes to this quantity — at the same state its residual
   correlates with the current one at **−0.18** — and regenerating day 230
   needs a tiled oracle restart that exists only for day 180.)*
4. **The blend and the time filter.** A weight or phase offset gives a
   state-constant residual, but with no reason to carry a latitude shape. Low.
5. **The north fold: excluded at no cost.** DINO's grid carries no fold, and
   the fold machinery is the identity in a serial run.

### One route the substituted arm may not close

If the loop re-scales the substituted forcing by legoESM's *own* face depth
(transport form), a depth difference leaks in even though the forcing is
NEMO's. That is not wind placement, but it should be confirmed once before the
substituted arm is described as "loop-only". **Unverified.**

## 7. What is *not* claimed

* No term is named. The constant bias remains unowned.
* The overlay against the basin torque-gap row profile was **not performed**:
  no such row-profile data exists on disk, only prose. What can be said is a
  geometric contrast, and it is only that — the year budget's torque gap is a
  *southern*-wall object, while this fixed bias is northern-dominated in the
  zonal component. The two are also different quantities (a year-mean torque
  versus a per-step velocity bias), so no correlation between them is quoted.
* The 1.23 % / 1.53 % constancy is over five consecutive steps, not over a
  season (§1).
* The wind's exclusion rests on the **structural** ground alone. The size bound
  does **not** exclude it at the southern wall in either channel (§5), and the
  claim that a zero meridional forcing implies a zero meridional response is
  retracted.
* "The bias is a fixed fraction of the local velocity" is retracted (§3): the
  amplitudes agree at the wall pair and the cells do not.
* The geometry is not exactly symmetric — the northern column is 14.7 % deeper
  (§3). That is far too small to account for the 11.7× bias contrast, but the
  word "symmetric" is not used unqualified.
* The "northern wall ramp" framing is retracted (§2): it is the largest of
  three alternating basin-scale bands, and the southern wall is 1.004× the
  interior in the zonal component.
* "Ten times the interior, boundary-localised" is retracted (§2): two interior
  rows exceed both walls, and the relative deficit peaks at row 134.
* The v-face metric is a **candidate**, not the owner (§6). Its basin-wide
  shape correlation with the measured deficit is −0.045.

## 8. A scope caution this document owes the campaign

By this card's own prior measurement, the two-step projection annihilates a
state-constant field **exactly**. So the residual characterised here — which is
99.9 % state-constant — **cannot be the wall flicker** that opened this thread.
It is the largest remaining term in the barotropic per-step deposit, and that is
the only reason to keep spending on it. Anyone picking this up should state that
justification again before running arm 1, rather than inheriting it.
