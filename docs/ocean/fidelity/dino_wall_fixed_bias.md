# The barotropic loop's fixed bias: not a wall object, and not the wind

**Outcome.** The state-constant part of the barotropic end-wall residual is
characterised directly, without any projector; the wind-stress placement
difference — the one unstarted candidate on the card — is **excluded on a
structural argument, but on that argument alone**; and a new candidate with a
measured coefficient is named and ranked. Two loop ingredients are refuted by
measurement and two more excluded for free.

**Read §5 before quoting the wind verdict.** The independent size check
excludes the wind at **neither wall** once its forcing is co-located with its
response, and the structural argument that carries the exclusion has four
premises, one of which is still unverified.

**The headline is a retraction.** This residual is **not a wall object at all**.
In the zonal component it is the largest of three alternating, zonally coherent,
basin-scale bands, and at the southern wall it is 1.004× the interior median. In
the wall-normal component the relative deficit peaks in the *interior*, at row
134, exceeding both walls. Two claims made in the first draft of this document —
"a northern wall ramp" and "ten times the interior, boundary-localised" — are
withdrawn below, along with a third about the wind's meridional channel.

**The leading candidate** is a horizontal grid metric: legoESM's zonal width of
the meridional cell face is 3.3e-05 too large at both walls, the only metric off
by more than roundoff. **The pre-registered arm has now been run and the verdict
is PARTIAL** (§6a): substituting NEMO's own array removes 16 % of the
state-constant wall-normal residual basin-wide and 25 % at the wall rows, against
bars of 50 % for OWNER and 10 % for REFUTED. The candidate is a real and
correctly-shaped term — the arm's own response tracks the predicted latitude gap
at **+0.75** — but it is not the residual's owner, and a **larger sibling error in
the same coefficient** is now named.

Artifact: `results/dino_1455/baro_fixed_bias_wall_map.json`.
Probe: `scripts/validate/ocean_fidelity/dino_1226/baro_fixed_bias_wall_map.py`.
Tests: `tests/ocean/fidelity/test_baro_fixed_bias_wall_map.py` (70).

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

**The decider — structural, and it has four premises, not one.** The fixed bias
is measured in the arm where NEMO's own `zu_frc`/`zv_frc` are substituted into
legoESM's loop, so every difference in how legoESM *assembles* its slow forcing,
wind term included, is removed there by construction.

| # | premise | status |
|---|---|---|
| 1 | the barotropic loop carries no wind-stress term of its own, so there is no second route in | **verified by source** (review round 2) |
| 2 | the substitution replaces the forcing and nothing else | **verified by source** |
| 3 | these maps are the wind-**on** arm | **now asserted from the maps' own stamps** (§5a) |
| 4 | the loop does not re-scale the substituted forcing by legoESM's own face depth | **UNVERIFIED** |

So this is a **strong argument, not a closed one**, and the earlier sentence
"this alone settles it" is withdrawn. Premise 4 is the one to close before the
arm is ever again described as loop-only.

### 5a. The provenance gate premise 3 rests on, and what it still does not cover

Round-2 review found this probe **discarding** the provenance the maps already
carry. A map made with the wind off is a well-formed array on an identical wet
mask, so no numerical guard could tell it from a wind-on one, and §5 would have
compared an *unforced* ocean against a *forced* increment. Round 3 then found
the first fix inadequate in two specific ways, both now closed:

* **Agreement is not an on-state.** The gate only checked that the five maps
  agreed with each other — and five identical *wind-off* maps agree perfectly.
  Each knob is now checked against its expected value, **per map**, not against
  its neighbours. (The confirmation pass then found the on-state check for the
  vertical ladder sitting *outside* the per-map loop, so a series whose first
  map was right and whose other four were wrong would have passed. Every check
  is now inside the loop; the cross-map agreement clause was **removed** once
  it was, because with both present neither could be tested — deleting either
  one left the suite green while the other silently covered for it, which is
  the "guard that cannot fail" this work exists to refuse. Each of the three
  per-map checks now has a test that fails when it is moved back outside the
  loop — including the step-stamp check, which the deleted clause had always
  excluded, so the loop was its only guard and nothing had ever tested that.)
* **It was checking the wrong knobs.** The three switches it examined are
  continuity controls belonging to *other* probes; they reach this stamp only
  because one shared environment list is recorded for the whole directory, and
  the producer of these maps never reads them. Meanwhile the one stamped knob
  it *does* read that changes the wind — **the seasonal clock**, on a card whose
  wind has an annual cycle — was omitted entirely. The gate now checks the
  clock (it must be derived, not overridden, and derived from the step the map
  actually is), and the vertical ladder; the three inert switches are recorded
  for audit and explicitly **not** treated as a wind-on witness.

Two things measured rather than assumed while fixing it: the five states
legitimately carry **five different oracle dump directories**, which is why the
command-line directory is checked against them; and the wind increment is
**step-invariant to 4.3e-08** across all five, loaded from each and compared.

**The gap that remains, named.** The *oracle* side's wind-on state **is**
witnessed — its wind increment is nonzero on every one of the wet faces, which
a wind-off oracle run could not produce. legoESM's **own** wind-on state is not
directly stamped in these maps at all. The clock checks are the strongest proxy
the saved artifacts allow, and they are a proxy. Closing it properly means
stamping the resolved stress amplitude in the producer.

**Support (partial) — shape, zonal channel only.****Support (partial) — shape, zonal channel only.** The zonal wind increment
peaks at row 48, in the westerly band, and at the wall rows is **571×** (south)
and **1308×** below that peak; its whole-domain row profile correlates with the
bias's at −0.187. The candidate vanishes where the bias is largest.

**Support (partial) — size, and it now excludes at NEITHER wall.** Deliberately
generous: 100 % of the wind term, not the small measured placement offset.

| | measured response (0.148) | response 1.0 |
|---|---|---|
| zonal, south wall | 0.79× — **does not exclude** | 0.12× — **does not exclude** |
| zonal, north wall | 21.3× — excludes | 3.2× — excludes |
| rotated meridional, south | 0.85× — **does not exclude** | 0.13× — **does not exclude** |
| rotated meridional, north | 10.2× — excludes | **1.51× — does not exclude** |

At an assumption-free response the size bound **fails to exclude the wind at
either wall in the across-wall channel**, and at the southern wall it fails in
both channels at both responses. The probe prints the verdict and the headroom
per row.

### RETRACTED: the northern size exclusion, which the co-location error created

Round-3 review found the rotated bound taking its **forcing** from the
along-wall grid's own wall row while evaluating the **response** on the
across-wall grid's. In the south those are the same row and nothing changed. In
the **north they are different rows** — and the wind climbs steeply away from
the wall there (2.6× one row in, 25× by ten). Reading the forcing one row too
far out understated it by **2.57×** and turned a 1.51× *non*-exclusion into a
3.9× "exclusion".

Co-located properly — a meridional face is flanked by two zonal faces, and the
larger of the two is used because this is an upper bound — the northern rotated
ratio is **1.51×**, below the 3× margin. **The wind-placement candidate is NOT
excluded at the north wall by size.** This consumed exactly the 1.30× headroom
that the previous round added, which is the argument for having added it. The
probe now prints which row each side of the comparison came from.

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
tests on its *magnitude* as well as its proportionality — an independent
reviewer verified the closed form at two window lengths and confirmed it is a
genuine upper bound. Two caveats it now carries in its own output rather than
in prose:

* the small-angle step is used **outside its stated regime** — the rotation
  angle reaches **1.09 rad** by the end of the averaging window, not a small
  angle. The error is in the conservative direction (the true response is
  **6.1 % smaller**), so it remains an upper bound, but the regime is stated.
  That figure is now **derived** from the window's own weights rather than
  pasted: writing it out showed the pasted 6.5 % was slightly off, and writing
  it out *stably* mattered too — the literal `1 − cos θ` form cancels to zero
  at small angles and would have reported a 100 % overstatement in exactly the
  regime where the approximation is perfect;
* the kernel on the **real** saved weights is **1142**, not the 787.75 of the
  synthetic uniform-window case the tests pin. Both are printed, because the
  real averaging window is uniform over only 45 of its 68 substeps. Confusing a null
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
matching convention (`metric_convention="nemo_isotropic"` in
`packages/core/legoesm/grids/latlon.py`, selected by the DINO card in
`packages/ocean/legoesm/ocean/experiments/dino.py`) fixes the *meridional* width
but leaves the *zonal* one on the cosine of the midpoint between the two
adjacent tracer rows. That is where a fix would go — **not** in the barotropic
solver, which merely consumes the metric. On DINO's
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

**Why it is not yet the owner, by this document's own standard — and the probe
now says so with a flag, not with prose.** Two rows agreeing is exactly the
reading §3 had to retract, so the shape test carries a boolean verdict
(`SHAPE TEST: NOT SUPPORTED — right order, wrong latitude structure`) exactly
as §3 and §4 do. Over all 196 wet rows the row-wise correlation between the
predicted gap and the measured relative deficit is **−0.045** — essentially
zero — and the per-row ratio spans **0.03 to 1649**. The measured deficit peaks at row 134; the predicted gap peaks at
the walls. *(The mechanism reviewer reported +0.52 for this correlation, over a
selected subset of rows. Over all rows, unselected, it is −0.045. The
disagreement is entirely the row selection, and the unselected number is the
one reported here.)*

So: right order, right sign, right place at the walls, **wrong basin-wide
shape**. That is a lead worth one cheap arm, not a verdict.

## 6a. THE ARM WAS RUN. The verdict is PARTIAL

Three arms, all five states, all produced at one clean tree and scored by the
**unmodified** probe: the plain frozen-forcing baseline, NEMO's own `e1v`
substituted, and the registered staggering control. The substitution is one
array reaching the three in-loop consumers that read the same width — the
continuity divergence, the metric-complete EEN rotation coefficient, and the
ssh-average face depth — and all three are verified reachable on this card at
run time rather than read off a log afterwards.

### The registered number

| reduction of the state-constant WALL-NORMAL residual | baseline | arm | collapse |
|---|---|---|---|
| basin-wide state-mean RMS | 1.8475e-07 | 1.5484e-07 | **16.2 %** |
| restricted to the two wall rows | 2.6725e-07 | 2.0065e-07 | **24.9 %** |
| southern wall row 1 | 7.034e-08 | 6.394e-08 | **9.1 %** |
| northern wall row 196 | 3.713e-07 | 2.765e-07 | **25.6 %** |

**PARTIAL** on every reduction but one: the southern wall row alone lands at
9.1 %, 0.9 points inside the REFUTED band. The verdict does not depend on which
reduction §2 and §6 meant — the oracle velocity is bit-identical between the
arms, so the relative-deficit collapse equals the raw-RMS collapse to the digit
(9.1 % and 25.6 % either way). The arm's residual is still a fixed field (worst
state-to-state departure 1.83 %, cross-state correlation +0.9997), so the
comparison is like for like.

The **tangential** channel, which the registration did not score, moved nearly
three times as much: **46.0 %** basin-wide. §6's mechanism argued the oversized
rotation coefficient makes the *wall-normal* velocity too small; it does not
predict that.

### The staggering control: it fires, and it calibrates the response

Feeding the un-shifted array — same 10 296 cells, only the row alignment
differs — makes the wall-normal residual **292-307× WORSE** at every reduction,
against a bar registered in advance at 50 % above baseline. It is not a
degenerate path: the wrong array's metric error is 656× the candidate's and its
residual change is 510× larger, i.e. the loop's response to this metric is
near-linear across a 656× span, and no NaN or infinity is involved.

### The band decomposition: it acts on the northern lobe, not on the bands

| band (wall-normal) | baseline | arm | collapse |
|---|---|---|---|
| rows 57-73 (southern interior) | 2.787e-07 | 2.892e-07 | **−3.7 %** |
| rows 121-153 (northern interior) | 2.091e-07 | 2.149e-07 | **−2.8 %** |
| rows 185-197 (northern lobe) | 3.814e-07 | 1.460e-07 | **+61.7 %** |

So the answer to "does it act on the bands or only the walls" is **neither**.
The two interior bands are untouched — marginally worse, not better — while
rows ≥185 carry **74 %** of the arm's whole variance reduction. And inside that
lobe the collapse *falls as the wall is approached*: 83.5 % at row 185, 71 % at
192, 53 % at 194, 25.6 % at the wall itself. The wall row is the part of the
northern lobe this candidate explains **worst**, which is the opposite of the
wall-localised reading §6 was built on.

### The shape test, re-printed post-substitution — and a sharper version of it

The probe's own test is unchanged (its predicted gap comes from NEMO's mesh and
no arm touches it). Post-substitution it still returns **NOT SUPPORTED**, and
the correlation moves *away* from support, −0.045 → **−0.200**.

But the test compares a predicted metric gap against the measured *residual*,
and the arm now supplies the missing third field — the residual's response to
that metric. Against **the arm's own response** the correlation is **+0.750**
over the same 196 rows. Read together these say something sharper than "right
order, wrong shape": **the mechanism is real and its latitude footprint is
exactly where §6 predicted; the residual it was supposed to explain simply has
a different and larger owner.**

A caveat on the test's own power, which the control exposes: on the staggering
arm the shape test returns **SUPPORTED** with the measured peak moving to the
wall row. That is a residual which is ~100 % metric-driven, so it demonstrates
the test is not vacuous but says nothing about its power at the candidate's own
30 %. A scaled-candidate ladder would measure that threshold; it has not been
run.

### Why a 16 % collapse is not a small response — amplitude versus alignment

The collapse statistic conflates two things and both are measurable from the
same maps:

| channel | residual RMS | the arm's own change | corr | best-fit rescale α | variance explained |
|---|---|---|---|---|---|
| wall-normal | 1.8475e-07 | 1.0732e-07 (**58 %** of it) | +0.548 | **0.941** | **30 %** |
| tangential | 1.2235e-07 | 1.0113e-07 (**83 %** of it) | +0.852 | **1.018** | **73 %** |

α is the least-squares rescaling that would best fit the residual, and it comes
out at 0.94 and 1.02 — **the arm's amplitude is right with no fitted
parameter**. Nothing is left on the table by amplitude; the entire shortfall is
alignment. Note also what the registered bar demanded: a 50 % RMS collapse
requires the candidate to explain 75 % of the variance. That bar stands as
written, but it is a stiffer test than "is this term the dominant one".

Corollary, and it **weakens §6's own headline**: the balance argument predicts a
transfer gain of exactly 1 — a rotation coefficient too large by ε makes the
balanced wall-normal velocity too small by ε. Measured on the arm's own
response the gain is **0.77 (south) and 0.66 (north)** at the walls and **0.33**
basin-wide. §6's "the prediction lands at 1.05-1.28× the measured deficit with
no fitted parameter" was therefore two compensating errors — the prediction
over-states the true response by ~1.3-1.5× at the walls, and the deficit it was
compared against is only partly metric-driven. The two-row agreement was partly
luck.

### The sibling error this arm uncovered, and did not remove

Mechanism review asked whether a second error of the same class could be
masking the first. There is one, it is **larger**, it sits in the **same** EEN
rotation coefficient, and it has the **opposite sign**. legoESM builds the
Coriolis parameter at the vertex as the average of the two adjacent tracer-row
values; NEMO evaluates it at its own f-point latitude. Measured against NEMO's
own dumped `ff_f`:

| | median | RMS | at the walls | at the equator |
|---|---|---|---|---|
| v-face zonal metric gap (this arm's candidate) | **+1.86e-05** | 2.08e-05 | +3.35e-05 | +2.9e-09 |
| Coriolis at the vertex | **−5.48e-05** | 6.15e-05 | −2.50e-05 | −9.19e-05 |

The two are exactly complementary in LATITUDE: the metric error peaks at the
walls and vanishes at the equator, the Coriolis error does the reverse, and
their signed latitude profiles correlate at +1.000 (their *magnitudes* at
−1.000 — the same fact stated two ways). That part stands.

> **RETRACTED 2026-08-26 — the sentence that used to follow is wrong.** It read:
> "In the coefficient, which goes as `e1v · f`, a positive relative error in one
> and a negative one in the other **partially cancel**." They do — in the
> **zonal** tendency. They do **not** in the meridional one, which is the
> wall-normal component this document scores. Measured on
> `een_barotropic_coriolis` (metric-complete, +1 % on one array at a time):
> `e1v` moves the zonal tendency by 1.431e-06 and the meridional tendency by
> **exactly 0.000e+00**, because the operator carries `e1v` into the zonal
> output and `e2u` into the meridional one. In this document's own registered
> channel the Coriolis error is **unpaired**, and there is nothing for the
> metric half to cancel against.
>
> The registered three-arm test was run and agrees independently: the joint arm
> is the SUM of its halves, not more, and the two halves act on DISJOINT bands —
> the metric owns the northern lobe (61.7 % against the Coriolis half's −1.5 %),
> the Coriolis owns both interior bands (32.1 % / 25.8 % against the metric
> half's −3.7 % / −2.8 %). See `dino_coriolis_pair_result.md`.

This arm removed the smaller of the two *in the zonal tendency*, and one of two
independent errors in the meridional one.

**And one third of that Coriolis gap is not a discretisation convention at
all.** It decomposes into a uniform **−1.578e-05** and a latitude-varying
**−3.90e-05** median. The uniform part is simply a different Earth: legoESM's
`constants.Omega` is **7.292e-05**, NEMO's is **7.2921150830e-05** (2π over the
sidereal day, confirmed from NEMO's own `phycst.F90` and independently from the
dumped `ff_f`). An oracle-matching card is running on a rotation rate the oracle
does not use. Verified premises: legoESM's tracer latitudes equal NEMO's
`gphit` to 1.4e-14 degrees, so this is a constant, not a grid difference.

> **UPDATED 2026-08-26 — right, and the mechanism is one step further back than
> "the card".** The card's pinned rate DID reach every config-side consumer;
> what it never reached was the GEOMETRY, whose Coriolis arrays the NEMO bridge
> built from its own default argument (legoESM's rounded constant). The split
> was config-versus-geometry. And `7.2921150830e-05` is NEMO's `#else` branch:
> the preset carried the `key_cice` literal `7.292116e-05`, a further 1.257e-07
> away — 125x smaller than the gap that mattered, and not to be quoted as its
> peer. Both fixed; one rotation rate now reaches every site, bit-identical to
> the rate recovered from NEMO's own `ff_f`. See
> `dino_coriolis_pair_result.md` §2.

### Stamps

fp64 control dtype; vertical ladder `LEGOESM_NEMO_E3T=both`; seasonal clock
derived per state (day 180.03 → 180.16), never overridden; wind on
(`wind_through_step=True`, τ_x ∈ [−0.1999, 0.1000] Pa) at every state; entry
state bit-identical to NEMO's restart at every state
(`max|dT| = max|d_eta| = max|du| = max|dv| = 0`); card `face_depth='nemo_ssh_avg'`,
`time_filter='nemo_boxcar_ab3'`, `coriolis='een_metric'`,
`reconcile='velocity_avg'`. All fifteen maps carry one producer SHA, and every
untouched array — legoESM's own-forcing deposit, the oracle side, the weights
and the masks — is **bit-identical across the three arms at all five states**
(65 array comparisons), so the only thing that moved between them is the
substituted width. The earlier baseline, written by a `+dirty` tree, was re-run
at the clean tree and reproduces `1.8475e-07` exactly.

**Also verified at the point of use rather than relayed**, because the whole arm
rests on it: `dx_u` vs `e1u`, `dy_u` vs `e2u` and `dy_v` vs `e2v` agree with
NEMO's mesh to 1.3e-16, and the cell area to 2.3e-16. The v-face zonal width
really is the only horizontal metric off by more than roundoff — **scoped, as
adversarial review required (2026-08-27): that sweep covers the metrics the
barotropic loop reads, and does NOT cover the VERTEX area, which legoESM builds
as the exact spherical cap `R²·Δλ·|Δsin φ|` while NEMO divides the circulation
by `e1f·e2f`. Those disagree by ~1.9e-05 median — the same order as the defect
this arm removed. It sits outside the loop the arm exercised, so the arm's
premise survives; as a statement about the twin's horizontal metrics generally,
the sentence is too broad.** See the debt register.

## 6b. THE FAITHFUL CONSTRUCTION. Named, fixed, and bit-matched to the arm

The arm above substituted NEMO's `e1v` **array**. That proved the mechanism
real; it did not say why legoESM's own array was wrong. This section answers
that, ships the fix, and shows that the two are the same experiment.

### The DIFF, read from both models' source

It is **not** a different formula, a different radius, or a rounding. Both
models compute the same thing, `R·Δλ·cos φ`. They evaluate it at a **different
latitude**.

NEMO, `cfgs/DINO/MY_SRC/usrdef_hgr.F90`:

```fortran
zvj = REAL( mjg(jj,0) - nn_jeq_s, wp ) + 0.5                     ! :98
pphiv(ji,jj) = 1./rad * ASIN( TANH( rn_e1_deg *rad* zvj ) )      ! :108
pe1v (ji,jj) = ra * rad * COS( rad * pphiv(ji,jj) ) * rn_e1_deg  ! :113
pe2v (ji,jj) = ra * rad * COS( rad * pphiv(ji,jj) ) * rn_e1_deg  ! :118
```

The Mercator transform is taken at the **half-integer row index** — the
V-point's own latitude. (`ra = 6371229 m`, `rad = π/180` from `phycst.F90:26,37`;
`rn_e1_deg = 1` from `usrdef_nam.F90:30`, not overridden in DINO's
`namelist_cfg`. `ln_read_cfg = .false.`, so this file — not a `domain_cfg.nc` —
is what builds the mesh, and nothing downstream alters `e1v` but a halo fill.)

legoESM, `create_latlon_geometry` in `packages/core/legoesm/grids/latlon.py`,
built the same width from `cos(½·(lat_T[j-1] + lat_T[j]))` — the arithmetic
mean of the two adjacent **tracer latitudes**.

`asin(tanh(·))` is nonlinear, so
`asin(tanh(x+½)) ≠ ½·[asin(tanh(x)) + asin(tanh(x+1))]`: **the midpoint was
being taken in latitude space instead of in Mercator index space.** On DINO the
two latitudes differ by up to **0.0011°**, which is exactly the 3.3e-05.

The sharpest form of the diagnosis: legoESM **already had the correct face
latitudes** and was already using them for the *meridional* v-face scale factor
(NEMO's `e2v`) under the same convention. Only the *zonal* one was left on the
tracer midpoint. The two are one quantity in NEMO — `:113` and `:118` are the
same expression — and now they are one quantity here too, bit-for-bit on the
interior.

### The fix, and why it is a fix rather than a new option

It lands inside the existing `metric_convention="nemo_isotropic"` selector,
whose entire contract is *"reproduce NEMO's `usr_def_hgr.F90`"*. Under that
contract the old width was simply wrong, so no new switch is warranted: recipe
doctrine puts the faithful value on the oracle card, and `nemo_isotropic` is
selected **only** by the DINO oracle cards and by the NEMO bridge's
auto-detect. Every other grid runs `"exact"` and is byte-identical to before.

One builder change reaches every consumer, because there is only one stored
width: the model converts whatever it is handed into a `LatLonCGridGeometry` at
construction and *"all downstream operators see the enriched geometry"*, and
the shared-metric helper reads that stored array on any rich geometry. The two
END v-faces stay hard-zeroed — that is the transport-metric contract (no
meridional flux through the closed wall), and NEMO enforces the same no-flux
through its `vmask` rather than through its metric. Only the interior latitude
moved.

Direct test: `tests/ocean/unit/test_dino_vface_zonal_width_nemo.py`. It pins the
corrected width against NEMO's own array at both walls with hand-quoted values
(south wall `39899.4776639535 m` at −68.9727620197°, north wall the same at
+68.9727620197°, the near-equatorial face `111194.6894417521 m`), establishes
the row correspondence **by latitude** rather than by an assumed halo offset,
and checks every interior v-face rather than the five quoted ones — against
NEMO's transform rebuilt **end to end and independently of legoESM's own
latitudes** (`gphiv = asin(tanh(rn_e1_deg·rad·(j − nn_jeq_s + ½)))`, `:98`/`:108`),
so it verifies that the two models' latitudes agree and not merely that
legoESM is self-consistent. legoESM's v-face latitudes reproduce NEMO's own
transform to better than 1e-9 degrees across the whole interior. Every fidelity
assertion is paired with the same assertion under `"exact"`, which must FAIL —
so the test provably fires if the fix is removed, and reverting the fix was
confirmed to redden exactly those assertions and no others.

The five hand-quoted literals were themselves re-verified against NEMO's
`domain_cfg_out.nc`, matched **by latitude** (agreement 2e-12 to 5e-11 degrees,
widths to 5e-11 m), with the row map `lego j → NEMO j+1` recovered
independently — so a transcription slip cannot make the test self-consistent
and wrong.

The convention is **inert on uniform-latitude grids**, where the two
constructions are mathematically identical and there is nothing to correct;
that inertness is itself pinned by a test.

| the width, vs NEMO's `e1v` | at the walls | over the whole interior |
|---|---|---|
| `"exact"` (the old construction) | +3.3175e-05 | 3.3175e-05 max |
| `"nemo_isotropic"` (fixed) | −3.6471e-16 | 1.4895e-15 max |

Measured on the standalone NEMO-faithful R1 grid, fp64 storage, with legoESM
computing its own Mercator latitudes — its `lat_v` agrees with NEMO's `gphiv` to
**2.8e-14 degrees**, so the fix does not depend on being handed NEMO's mesh.
(At the default fp32 storage the agreement is bounded at ~1.7e-07 by the stored
dtype, still 195× better than the defect; the oracle lane runs fp64.)

### The bit-match: the arm's table IS this fix's evidence

Registered before it was run: if the corrected construction produces the same
array the arm substituted, the arm's five states already measure the fix and no
re-run is owed; if it does not, the fix reaches consumers the arm did not and
the arm's numbers may not be quoted for it.

`scripts/validate/ocean_fidelity/dino_1226/vface_width_arm_bitmatch.py`,
offline, no model run:

```
  cells compared        : 10400
  cells differing       : 0
  max |difference| [m]  : 0.000000e+00
  VERDICT: BIT-IDENTICAL.
  CONTROL (the pre-fix T-midpoint construction vs the same arm array):
    cells differing 10296, max relative 3.3485e-05  -- the control fires
```

**Zero of 10,400 cells differ — on CPU.** Three honest qualifications on that
headline. The first was caught by re-running the probe on the other backend
after the fix was already committed, and it is the sharpest: **this result is
backend-dependent.** On GPU the same probe reports **1,352 cells differing at a
maximum RELATIVE difference of 2.5e-16 — 1.1 ulp of float64** — because the
device transcendental library rounds `cos()` differently. That is **1.3e+11×
smaller than the 3.3485e-05 defect** and physically meaningless, but
"bit-identical" is a claim about bits and it is false on GPU. The probe now
stamps its backend and returns a three-valued verdict, so a CPU run can no
longer be quoted as though it were universal. The construction is the same one
on both backends; the bits are not.

The remaining two qualifications come from adversarial review. 104 of those
cells are the two end-wall rows, which the arm leaves at legoESM's own value and
which are therefore equal by construction; the probe now *asserts* they are the pole-zeroed wall (and that
NEMO's own `vmask` shows them carrying no wet faces) instead of silently counting
them as agreements, so the real statement is **0 of 10,296 substituted cells
differ on CPU, and the 104 uncompared cells are checked to be zero.**
Second, the control's 10,296 does match the touched-cell count §6a records for
the arm — but `10,296 = 198 × 52` is what the substitution overwrites on *any*
array of this shape, so it corroborates the shape, not the identity. The
discriminating check is the row alignment, which the probe re-establishes
against the arm's own two-hypothesis test (correct map 0.000e+00, best wrong map
1.218e-02) rather than assuming it.

### What the bit-match does and does NOT license — a correction

**Retracted, in place:** an earlier draft of this section said the equality means
"no re-run is owed" full stop. That overclaims, and the reason is the one this
campaign keeps relearning — **an identical array is not an identical
experiment.**

The arm substituted into the *barotropic substep loop's* kwargs and its EEN
pre-block and ran that loop. The builder fix changes the width for the whole
model. So the equality licenses exactly this:

* **Licensed.** The arm's five states already measure this width's effect
  **inside the barotropic substep loop**, and no re-run is owed to re-measure
  that. §6a's collapse figures — 16.2 % basin, 24.9 % wall rows, 9.1 % south,
  25.6 % north, staggering control 292–307× — stand as that evidence, and **the
  PARTIAL verdict is unchanged.** (The arm's own FATAL guard, that the EEN
  pre-block's copy of the width must BE the grid's `dx_v`, is what makes the two
  configurations equivalent inside the loop rather than merely similar: the
  pre-block is built from `dx_v`, so fixing `dx_v` fixes it too.)
* **NOT licensed.** The fix also moves baroclinic-side consumers the arm never
  perturbed, all of which read the stored width by value: the F-point lateral
  viscosity coefficient `ahmf = ½·rn_Uv·max(e1f,e2f)`, GM/Redi's isoneutral
  tensor component `e1v/e2v`, the bolus streamfunction and its transport→velocity
  division, MLE, and the baroclinic EEN weighting. **Their magnitude on the
  climate is unmeasured**, and no number here may be quoted for them.

Every one of those moves **toward** NEMO, and one of them is now exact. Measured
at the bridged geometry, F-point rows aligned to NEMO's:

| the lateral viscosity coefficient `ahmf`, vs NEMO's | max relative gap |
|---|---|
| before the fix | 3.3485e-05 |
| after the fix | **0.0000e+00** |

That is a direct cross-link to the lateral-friction lane: the alignment table's
row 3 recorded legoESM's F-point coefficient as matching NEMO only
approximately, attributing the residual to the discrete `max(e1,e2)` convention.
The residual was this width. With `e1f = e2f` restored, `max` is exact and so is
the coefficient. It does not reopen anything — friction's refutation rests on a
~1e-4 *relative* effect being too small to carry a 34 % transport error, and
closing a 3.3e-05 coefficient gap only makes that leg safer.

So: the fix removes a real infidelity from the twin, in more places than the arm
tested. It does not promote the candidate, and the banded residual remains
unowned.

### The conservation question, checked rather than assumed

Unlike the Coriolis placement, a metric width has no solid-body-curl trade to
lose — but "no trade" is an argument, not a measurement, and the width does sit
inside three discrete identities. All three were re-run on the corrected
geometry (`TestInvariantsSurviveTheCorrectedWidth`), and the reason they hold is
structural: each needs every operator to share **one** width, and the fix moved
that one width rather than one operator's copy of it.

* **Strain ↔ stress adjointness.** `<strain(u,v), T> = <(u,v), stress_div(T)>`
  holds to better than 1e-11 relative on the corrected `nemo_isotropic`
  geometry.
* **Continuity ↔ flux-form advection mass consistency.** The advection's v-face
  transport length and the divergence's agree to <1e-9 m, so no mass leaks
  between the two — and, separately pinned, the width they now share is the
  **corrected** one (a shared but stale metric would pass the consistency check
  while leaving the defect in place).
* **The EEN `e3f` normalization does not enter.** Read at the point of use:
  the vertex thickness is built from layer thicknesses alone, with no
  horizontal metric anywhere in it. Nothing to check.
* **The EEN metric weighting's energy identity, which does enter**, because the
  DINO card ships it and its entire justification is that it conserves the
  *physical* kinetic-energy norm rather than the per-area one. Measured, not
  reasoned: the vorticity flux still does zero net work on the corrected width,
  to better than 1e-14 of the domain kinetic energy. The width appears once in
  the transport weight and once in the energy norm and telescopes — but that is
  the class of claim this campaign keeps retracting, so it is a test.

The two pole/wall faces stay exactly zero under both conventions, so the
closed-wall no-flux property is untouched.

**One test was narrowed, deliberately.** A grids test pinned `dx_v`
bit-identical between the two conventions. Its own comment says the invariant
being protected *"holds because every operator SHARES that metric — not because
of its value"*, i.e. the pin was strictly stronger than the invariant, and the
sibling field `dy_v` had already been removed from that same list for exactly
this reason. `dx_v` is now pinned the other way: it must move, by 1e-5–1e-4
relative and no more, keep its zeroed end faces, and equal `dy_v` on the
interior.

### What the fix EXPOSED: a cancelling pair, and the campaign's own Rule 8

Raised by adversarial review and measured here rather than relayed. legoESM
builds the **vertex** area as the exact spherical cap `R²·Δλ·|Δsin φ|` and
divides the circulation by it to form relative vorticity; NEMO divides by
`e1f·e2f`. Those are different quantities, and this fix moves one of them:

| `|area_q − e1f·e2f| / (e1f·e2f)`, interior rows | median | max |
|---|---|---|
| before the fix | 7.79e-06 | 2.54e-05 |
| **after the fix** | **2.22e-05** | **4.16e-05** |

Read this correctly. The vertex-area gap is **pre-existing and untouched** —
`area_q` is built exactly as it always was, and NEMO always divided by
`e1f·e2f`. What changed is that the compensating error in `e1f` is gone, so the
gap is no longer partly hidden. **This is the Rule-8 cancelling-pair pattern
this campaign has now recorded five times:** raising faithfulness on one half of
a pair makes a derived metric look worse.

It does not argue for reverting — the campaign's own standing answer to Rule 8
is *"a joint arm, never a revert"*, and one metric now matching the oracle
exactly is not a defect. No identity breaks: `q` enters the AL81 triad
symmetrically, so the kinetic-energy cancellation measured above is
`q`-independent and EEN enstrophy conservation is a property of the triad
structure rather than of `q`'s divisor.

### §6c — the vertex area: the diff named, the pair REFUTED as a pair, the fix shipped

**Same day, 2026-08-27.** Probe: `vertex_area_pair_analysis.py` (self-test
PASS). Tests: `tests/ocean/unit/test_dino_vertex_area_nemo.py`.

**THE CONSTRUCTION DIFF.** It is not a different radius, dlon or rounding, and
it is not a different latitude either — it is a different **QUADRATURE**. NEMO
(`usrdef_hgr.F90`) builds the F-cell area as a PRODUCT of two scale factors
taken at the F-point's own Mercator latitude:

    zfj   = REAL(mjg(jj,0) - nn_jeq_s) + 0.5                     :97
    pphif = 1./rad * ASIN(TANH(rn_e1_deg*rad*zfj))               :109
    pe1f  = ra * rad * COS(rad*pphif) * rn_e1_deg                :114
    pe2f  = ra * rad * COS(rad*pphif) * rn_e1_deg                :118

legoESM built the exact spherical cap between the two adjacent TRACER
latitudes. On a Mercator coordinate `sin φ = tanh(Δλ·j)`, so
`d(sin φ)/dj = Δλ·cos²φ` and the cap is `R²Δλ²` times the **exact interval
integral** of `cos²φ` while NEMO's product is `R²Δλ²` times its **midpoint
value**. Exact quadrature versus the midpoint rule on `sech²`:

    relative gap  =  (Δλ² / 12) · (3 sin²φ_f − 1)

| | signed median | \|·\| median | \|·\| max | sign change |
|---|---|---|---|---|
| measured, vs NEMO's dumped mesh | +1.1747e-05 | 2.2230e-05 | 4.1585e-05 | −35.08° |
| the closed form above | +1.1748e-05 | — | 4.1585e-05 | ±35.26° |
| **measured / predicted** | **0.999984** (min 0.9993, max 1.0022) | | | |

The **sign change** is the discriminating feature: no wrong radius, dlon or
rounding can produce one, and it is where `3sin²φ = 1`.

`zfj` (:97) and `zvj` (:96) carry the SAME `+0.5` row offset, so `pphif ==
pphiv` and **`e1f·e2f == e1v·e2v` exactly** on NEMO's own mesh (0.0 relative).
The F-cell area is therefore the SQUARE of the v-face width §6b just corrected,
and the fix reuses that width rather than deriving a second latitude.

| the vertex area vs NEMO's `e1f·e2f`, interior rows | signed median | \|·\| max |
|---|---|---|
| exact (old) | +1.0791e-05 | 4.0965e-05 |
| **nemo_isotropic (fixed)** | **+4.9e-16** | **3.1e-15** |

Measured on the model's own NEMO-faithful R1 grid against NEMO's transform
rebuilt end to end, so the fix does not depend on being handed NEMO's mesh. The
two **wall rows** are deliberately left on the old cap: they span pole-to-first-
tracer-row, are fully masked, and sit under the curl's positivity guard.

**THE PAIR ANALYSIS — REFUTED AS A PAIR.** Rule 8's standing answer is a joint
arm, so this was run BEFORE shipping. The vertex area and the vertex Coriolis
are the same class of defect at the same point (`−(Δλ²/4)cos²φ_f` for the
Coriolis half, signed median −3.998e-05) and they meet in ONE quantity: the
F-point absolute vorticity `ζ + f` that EEN's `q` is built from.

*Pre-registered before any number was read:* the two are a cancelling pair
requiring a joint fix **iff the area half is within a factor of 10 of the
Coriolis half at the median in that channel.**

Measured on NEMO's own day-5760 restart, 333,318 wet F-points:

| | median | p99 | max |
|---|---|---|---|
| \|ζ\| [1/s] | 2.0513e-08 | 2.4048e-06 | — |
| \|f\| [1/s] | 1.0184e-04 | 1.3613e-04 | — |
| area half of the `ζ+f` error [1/s] | 4.0182e-13 | 7.5738e-11 | — |
| Coriolis half [1/s] | 3.0018e-09 | 4.2748e-09 | — |
| **ratio area / Coriolis** | **1.5050e-04** | 6.867e-02 | 4.249e-01 |

**1.51e-04 against a bar of 1e-01 — BELOW by a factor of 660.** The two
REINFORCE at 59.9% of wet points; at the 0.51% of points where the area half
does exceed the bar, 79.6% REINFORCE, so removing it helps there too.

> **RETRACTION, loud, and it reversed a sign.** The first version of this
> section reported *"removing the area half moves the total error by
> +0.055%"* — a **worsening**. That number was a **ratio of two medians of two
> different distributions**, which is not the change in the error, and it
> contradicted the reinforce/oppose split printed directly above it (if a
> majority reinforce, removing one half must help at the median). The correct
> **paired, per-point** statistic is:
>
> | | |
> |---|---|
> | median per-point change | **−0.001232%** (an IMPROVEMENT) |
> | mean | −0.103% |
> | improves at | **59.9%** of wet F-points, worsens at 40.1% |
> | energy-norm ratio (L2 after / before) | 0.999542 |
> | max \|error\| | 4.2753e-09 → 4.2748e-09 |
>
> Wrong sign and ~45× too large. The fix *reduces* the absolute-vorticity
> error at the median, at the maximum, and in the L2 norm. Caught by
> adversarial review, not by me; the probe now computes the paired statistic
> and the ratio-of-medians is gone.

**REGIONAL, because a global median hides a structural enrichment.** The
Coriolis half vanishes at the equator while the area half tracks relative
vorticity, so the ratio is worst there:

| band | ratio median | fraction above the 0.1 bar |
|---|---|---|
| \|lat\| < 5° | 5.74e-03 | 3.11% |
| \|lat\| 5–40° | 5.86e-05 | 0.01% |
| \|lat\| > 40° | 1.93e-04 | 0.63% |

Still 17× under the bar even in the equatorial band, and the maximum anywhere
is 0.42, so the area half never dominates — but **"unpaired-safe" carries the
qualifier "except within a few degrees of the equator"**, where the margin is
one order rather than three. In the
other active channel — the NEMO-faithful lateral viscosity, which divides by
`e1f·e2f` — `f` does not appear at all, so the area is structurally unpaired
there and the fix strictly improves it. A consumer sweep found the one
genuinely self-cancelling use of the vertex area (the Smagorinsky/om4p25
raw-stress path, where it multiplies back out) is **INACTIVE on the DINO card**
(`C_smag = 0`).

**VERDICT: unpaired-safe. Shipped alone.** This is the first of the five
recorded Rule-8 instances to be REFUTED as a pair by measurement rather than
handled with a joint arm.

**THE COST, named rather than buried.** The (cap, cell-average) pair is what
makes the discrete curl of solid-body rotation equal the discrete `f`
*exactly*, so the fix gives that up:

| `\|curl(solid body) − f\| / f`, interior rows | median | \|·\| max |
|---|---|---|
| legoESM before (cap, cell_average) | −1.1e-16 | 1.6e-14 |
| **after this fix (e1f·e2f, cell_average)** | **+1.1748e-05** | 4.1585e-05 |
| Coriolis half alone (cap, face_latitude) | −3.9022e-05 | 7.6149e-05 |
| **NEMO itself (e1f·e2f, face_latitude)** | **−2.7274e-05** | **1.0153e-04** |

**THE BAR, APPLIED HERE TOO — and here it FAILS.** The first version of this
section scored the pre-registered bar only in the absolute-vorticity channel,
where it passes by 660×, and never applied it in this one. A bar applied only
where it passes is not a bar. In this channel the ratio of the two halves is
`|3sin²φ − 1| / (3cos²φ)`: **median 0.333, range 0.005–4.53 — AT OR ABOVE the
0.1 bar.** By the registered criterion, *in this channel the two ARE a
cancelling pair.* And the column an oracle lane actually cares about, distance
to NEMO, was not computed at all; it is **3.116e-05 → 3.902e-05 at the median
(worse)**, 1.0153e-04 → 7.6149e-05 at the max (better), improving at 58% of
rows.

**SO WHY SHIP ALONE.** The two channels disagree, and the argument for
preferring one has to be stated rather than assumed — this document chose its
channel before measuring (the pre-registration names the absolute vorticity),
but it did not anticipate a second channel that fails. The discriminator:

* the **absolute vorticity is a TERM in the equations** — every tendency the
  model integrates passes through `q`, and there the area half is 1.5e-04 of
  the Coriolis half because real ocean relative vorticity sits four orders
  below planetary;
* the **solid-body identity is a DIAGNOSTIC PROPERTY of the operator pair** —
  it is a statement about one specific idealised flow, not a quantity the
  model advances.

State-dependent channel decides; the diagnostic channel is the recorded cost.
NEMO's own discretisation is *less* solid-body-consistent than what this fix
lands on, so the twin's internal consistency now sits between legoESM's and the
oracle's. This is the same trade the `coriolis_placement` selector already
documents; it is recorded, not hidden. **If the Coriolis half is ever fixed
too, this channel is where the joint arm must be scored.**

**CONSERVATION, measured rather than assumed.** The vertex area is a pure
DIVISOR in every active DINO consumer, so no identity constrains its VALUE —
what is constrained is that every consumer shares ONE value, and one did not:
`vertex_area_cgrid` recomputed the cap while the curl read the stored array.
They agreed to dtype roundoff while both were caps and would now differ by
2.2e-05, so that helper was moved onto the stored array (its own contract is
that `A·ζ` IS the circulation the curl formed).

**Test non-vacuity, corrected after review:** with the source change stashed,
**7 assertions go RED** — 6 in the vertex-area module plus the narrowed
convention pin in the grids suite — not the 5 first reported. The rest stay
green by design and now say so: two are no-regression checks (the wall rows,
and the q-independent energy identity), and the pair-analysis class is
closed-form receipts for the numbers above rather than a test of this diff.

| identity | result |
|---|---|
| curl / vertex-area helper read ONE array | equal; control fires at 4.1e-05 |
| circulation `A·ζ` invariant under an area-only swap | < 1e-13 relative |
| curl's response equals the midpoint-rule gap | < 1e-9 over 8,460 cells |
| EEN metric-weighting energy identity | net work < 1e-14 of domain KE |
| wall rows stay positive, unchanged, and hard-zeroed in the helper | holds |

**ONE-STATE RESPONSE: INERT.** Pre-registered as material only above 1%.
Measured at the single day-220 replay state (`substep_traj_compare`,
`LEGOESM_NEMO_E3T=both`, fp64):

| statistic | before | after | change |
|---|---|---|---|
| wall max diff, jn=68 | 7.2393e-02 | 7.2385e-02 | **−0.011%** |
| meridional deposit max\|dV_avg\| | 5.035e-05 | 5.033e-05 | **−0.040%** |
| substituted-forcing max\|dV_sub\| | 2.629e-06 | 2.629e-06 | 0.000% |
| `zu_frc` ACC residual | +9.0521e-05 Sv | +9.0521e-05 Sv | 0.000% |

Both moving statistics move TOWARD NEMO. **Stated against my own prediction,
which was wrong by 100×:** I predicted ≲1e-4 %, measured 1.1e-2 %. The verdict
and the direction survive; the magnitude estimate does not, and I have not
established which path carries the extra factor — that is PLAUSIBLE-at-best and
is not claimed.

**THE SCALING TEST, added after review, because a one-state null whose own
prediction was falsified by 100× is not yet a result.** The campaign's standing
rule is that a proposed mechanism must survive a scaling test: perturb by a
known factor and check the response follows. The area perturbation was scaled
×10 through the same one-state harness (temporary builder knob, reverted; the
×1 arm reproduced the shipped number exactly, so the knob is inert at unity).

| arm | wall max diff, jn=68 | response vs baseline |
|---|---|---|
| baseline (no fix) | 7.2393e-02 | — |
| **×1 (shipped)** | 7.2385e-02 | −8.0e-06 |
| **×10** | 7.2310e-02 | −8.3e-05 |

**Response ratio 10.4 for a 10× perturbation** (predicted 10.0 if linear).

*Provenance limitation, stated rather than glossed:* the replay harness stamps
only the environment variables it knows about, so the scale factor does **not**
appear in the arms' own artifacts — it is recorded here and in the commit
instead. The control that the knob was actually in the path is the ×1 arm
reproducing the shipped number **exactly** (7.2385e-02) while ×10 produced a
different one; an unwired knob would have returned the same value for both.
The diagnostic prints 5 significant figures, so the ×1 response is 8 quanta and
the ratio carries ~±12% — 10.4 ± 1.3 against a predicted 10.0. **The response
is LINEAR in the perturbation to the resolution available**, which is what
licenses extrapolating the one-state margin instead of merely asserting it: the
null is a real ~1e-4 sensitivity, not a cancellation that happened at this
state. No multi-state arm was run, now on the strength of this rather than on
the pre-registration alone.

*Not claimed:* the meridional deposit moved 5.035e-05 → 5.033e-05 → 5.023e-05,
a ratio of 6.0, but it is printed to 4 significant figures so the ×1 response is
2 quanta and the ratio is bounded at ±50%. That statistic cannot discriminate
linear from non-linear here and is recorded, not used.

### The ranked remainder, after the arm

1. **Substitute NEMO's own `ff_f` for the vertex Coriolis** — three arms (f
   alone, `e1v` alone which is done, and both together), pre-registering the
   verdict on the **joint** arm. It is the same freeze-and-vary at the same
   cost, it is the only test that can settle whether the two coefficient errors
   were cancelling, and its target is 3× the RMS of the one just removed.
   Separately and cheaper: legoESM's `Omega` is not NEMO's, which is a constant,
   not a scheme.
2. ~~**Override the v-face zonal metric with NEMO's own array**~~ — **DONE, see
   §6a. PARTIAL: 16.2 % basin-wide, 24.9 % at the wall rows, against 50 %/10 %.
   The staggering control fired at 292-307× worse.**
3. **Face depths**, measurable offline the same way the metrics were — compare
   the loop's face depths against NEMO's dumped ones before spending a run.
4. **Discriminate "the bias tracks the flow" from "the bias tracks the
   geometry."** Cheapest version, suggested by review and better than the
   two-day comparison: rerun the in-loop arm from a **zero entry state** with
   the same frozen forcing. Because the loop is near-linear that splits the
   residual into a forcing-driven part and a state-proportional part in two
   runs, exactly. *(The two-day route is blocked: the day-180/day-230 pair on
   disk predates two fixes to this quantity — at the same state its residual
   correlates with the current one at **−0.18** — and regenerating day 230
   needs a tiled oracle restart that exists only for day 180.)*
5. **The blend and the time filter.** A weight or phase offset gives a
   state-constant residual, but with no reason to carry a latitude shape. Low.
6. **The north fold: excluded at no cost.** DINO's grid carries no fold, and
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
* The v-face metric is a **candidate**, not the owner (§6) — and the arm has now
  measured that: it removes 16.2 % of the state-constant wall-normal residual
  basin-wide and 24.9 % at the wall rows, which is **PARTIAL**, not ownership
  (§6a). Its basin-wide shape correlation with the measured deficit is −0.045
  post-substitution −0.200, and the probe's own shape test returns NOT
  SUPPORTED in both.
* **RETRACTED: "the prediction lands at 1.05-1.28× the measured deficit with no
  fitted parameter" is not the endorsement it reads as** (§6a). The balance
  argument predicts a transfer gain of 1; the arm's measured gain is 0.77/0.66
  at the walls and 0.33 basin-wide. The two-row agreement was two compensating
  errors — an over-stated prediction against a deficit that is only partly
  metric-driven.
* **The bias is NOT owned by this metric at the walls specifically.** The arm's
  effect is a northern-lobe object (rows ≥185 carry 74 % of its whole variance
  reduction) whose collapse *falls* from 83.5 % at row 185 to 25.6 % at the wall
  row, and the two interior bands are untouched (§6a).
* The arm removed **one of two opposing coefficient errors**. legoESM's Coriolis
  at the vertex differs from NEMO's `ff_f` by −5.48e-05 median, 3× the metric
  gap's RMS, with the opposite sign and a complementary latitude shape; a third
  of it is simply a different rotation rate (§6a). No claim is made about what
  the joint arm would give — that test is unrun.
* The wall-normal collapse's **amplitude** is not the shortfall: the arm's own
  change is 58 % of the residual's RMS at a best-fit rescaling of 0.94, i.e. the
  amplitude is right unfitted and the whole shortfall is alignment (§6a).
* The shape test's non-vacuity demonstration is **untested at the candidate's
  own amplitude** (§6a): it returns SUPPORTED on the staggering control, whose
  residual is ~100 % metric-driven, and that says nothing about its power at
  30 %.
* The wind's structural exclusion has **four** premises, two verified by source,
  one now gated on the maps' provenance, and one still **unverified** (§5).
  "This alone settles it" is withdrawn.
* **The wind is not excluded at the north wall by size** (§5). The earlier
  northern exclusion was an artifact of reading the forcing one row off the
  response's row, and is retracted. At an assumption-free response the size
  bound excludes at neither wall in the across-wall channel.
* The rotated bound uses the small-angle step outside its regime (1.09 rad by
  window end). The error is conservative (~6.5 %), so it is still an upper
  bound (§5).
* legoESM's own wind-on state is **not** stamped in the maps; the gate's clock
  checks are a proxy (§5a). The oracle side is directly witnessed.

## 8. A scope caution this document owes the campaign

By this card's own prior measurement, the two-step projection annihilates a
state-constant field **exactly**. So the residual characterised here — which is
99.9 % state-constant — **cannot be the wall flicker** that opened this thread.
It is the largest remaining term in the barotropic per-step deposit, and that is
the only reason to keep spending on it. Anyone picking this up should state that
justification again before running arm 1, rather than inheriting it.
