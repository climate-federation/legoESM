# The Coriolis pair: the premise is RETRACTED, and the two halves are disjoint

**Outcome.** The campaign's ranked-first next action (`dino_campaign_synthesis.md`
§3 item 1) was registered on a premise that is **false in the channel it
registered**. legoESM's vertex-Coriolis error and its v-face zonal-metric error
are **not** a cancelling pair in the wall-normal tendency: the v-face zonal
width enters the **zonal** tendency only, and its effect on the meridional
tendency **through the Coriolis operator is exactly zero**. In the registered
channel the Coriolis error is unpaired *in that operator*, and the three arms
bear it out — the two perturbations superpose **exactly**, with no interaction,
and the two halves own **disjoint bands**.

**Scope that sentence carefully, because I first wrote it too strongly.**
"Exactly zero" is a statement about the **Coriolis term**, not about the loop:
`e1v` still reaches the meridional velocity through the continuity divergence
and the ssh-average face depth, and the metric arm's meridional response is
**58 % of the baseline residual's own amplitude** (pre-fix tree; 60 % at HEAD).
At system level its effect
is emphatically not zero — it simply does not travel by the Coriolis route the
"cancelling pair" claim invoked.

The registered verdict is **PARTIAL**, the staggering control **fires**, and the
pre-registered condition therefore **withholds** the 90-day gate run.

Two code defects were found on the way. One had the entire oracle-matching lane
building its Coriolis on a different planet from the oracle; it is fixed here.

Pre-registration: `scripts/validate/ocean_fidelity/dino_1226/PREREG_coriolis_pair.md`
(committed before any arm was scored). Probes: `coriolis_omega_routing_audit.py`,
`coriolis_pair_arm_compare.py`, and the `DINO_1455_SUB_CORIOLIS` arm inside
`substep_traj_compare.py`. Artifacts: `results/dino_1455/arms_head/` (the HEAD re-run, which every number
in §3 comes from) and `results/dino_1455/arms/` (the superseded two-Earth pass,
kept for the cross-tree comparison).

---

## 1. RETRACTION: the pair does not cancel in the registered channel

`dino_campaign_synthesis.md` §3 item 1, `dino_wall_fixed_bias.md` ("The sibling
error this arm uncovered") and `PREREG_coriolis_pair.md` all state that the two
errors "enter the same EEN rotation coefficient `e1v · f` with opposite signs
and partially cancel". **Measured on the operator itself** (`een_barotropic_
coriolis`, metric-complete, +1 % on one array at a time, fp64):

| array perturbed +1 % | max Δ zonal tendency | max Δ meridional tendency |
|---|---:|---:|
| `e1v` (v-face zonal width) | 1.431e-06 | **0.000e+00** |
| `e2u` | 0.000e+00 | 1.285e-06 |
| `f_vtx` (vertex Coriolis) | 1.431e-06 | 1.285e-06 |

The metric-complete operator scales the meridional velocity by `e1v` before the
triad and normalises the **zonal** output by `e1u`; the meridional output
carries `e2u` and is normalised by `e2v`. `e1v` therefore cannot reach the
meridional tendency **through this operator** at all. The Coriolis error
reaches both.

**It does reach it by another route**, and the arms measure that: `e1v` also
feeds the continuity divergence and the ssh-average face depth, both of which
act on sea level and therefore on the meridional velocity. The metric arm's
meridional response has basin RMS 1.073e-07 against a 1.847e-07 baseline
residual. So the correct statement is *"the pair does not cancel in the
Coriolis coefficient of the meridional tendency"* — not *"the metric does
nothing there"*.

**So the pair is real in the ZONAL tendency and absent from the Coriolis
coefficient of the meridional one — and the pre-registration scores the
meridional (wall-normal) component.** The
verdict below is reported as registered; the registration itself is retracted as
mis-aimed. Credit: the mechanism review caught this; it was then re-measured
independently before being written down.

The arms confirm it without reference to the operator algebra, and by the right
statistic: the two perturbation **fields superpose exactly** (‖(Δ_E1V + Δ_F) −
Δ_JOINT‖ / ‖Δ_JOINT‖ = 3.5e-06). They are linearly independent contributions,
with no interaction of either sign. **Do not use the `joint − sum` percentages
for this** — a per-cent difference between a joint arm and the sum of two halves
is a second-order norm-geometry effect fully determined by the exact
superposition, and carries no information about cancellation either way. That
framing appeared in an earlier draft and in the harness, and is withdrawn.

---

## 2. STEP 0 — the routing audit, and the wrong planet

Measured by instantiation, not inference: the audit builds the same objects
every fidelity probe and the 90-day twin build, then recovers the rotation rate
each Coriolis site actually received (`omega = f / (2 sin φ)`, equatorial rows
dropped as ill-conditioned, bound printed).

**Before the fix** — the split is not "card versus global constant", it is
**config versus geometry**:

| site class | rate received |
|---|---|
| every config-side site (card pin, config property, `constants.Omega`, physics block) | 7.292116e-05 |
| every geometry-side site (`geometry.omega`, `f_T`, `f_u`, `f_v`, `vertex_coriolis`, `coriolis_at_faces`) | **7.292000e-05** |
| NEMO's own dumped `ff_f`, inverted against its own `gphiv` | **7.292115083046e-05** (row spread 5.6e-16) |

The card's pin never reached the Coriolis arrays: the bridge that builds the
twin's geometry took ω from its own default argument, and no DINO caller
overrode it. The production lat-lon path was unaffected (it builds the grid from
the card, and `ensure_geometry` carries the grid's own rate). **The
oracle-matching path was the one on the wrong Earth**, which is the exact
inversion of what an oracle harness is for.

**And the card's pin cited the wrong branch.** `phycst.F90:88-92` reads

```fortran
rsiday = rday / ( 1._wp + rday / rsiyea )
#if defined key_cice
  omega  = 7.292116e-05
#else
  omega  = 2._wp * rpi / rsiday
#endif
```

DINO is not `key_cice`, so it integrates the **sidereal-day expression**; the
preset carried the `key_cice` literal, 1.257e-07 relative away. Transcribing the
expression reproduces the rate recovered from NEMO's own `ff_f` to **8.5e-15**.

Two gaps, of very different size, and only one is material: the geometry/config
gap is **1.578e-05**, the pin/branch gap **1.257e-07** — 125× smaller. Do not
quote them as if they were peers.

**After the fix**, re-run as the receipt: **one** rotation rate reaches every
legoESM site, and it is bit-identical (`+0.000e+00`) to the rate recovered from
the oracle's own array. The vertex gap that remains is **pure placement** —
constant part exactly 0, median −3.622e-05 — and the row-map median falls from
5.480e-05 to 3.902e-05.

### The boundary of what the split can claim

On a grid with **constant** latitude spacing the row average is exactly
`cos(Δφ/2)` times the face value — a uniform factor, indistinguishable from a
rotation-rate change. DINO's Mercator spacing is stretched, which is the only
reason the constant and the convention are separable at all. Both cases are
unit-tested with hand-computed answers.

---

## 3. STEP 2 — the registered arms

Five directories, three registered arms, all produced sequentially on one device
at one tree, from the same bit-identical entry state with the same substituted
(NEMO's own frozen) slow forcing, over the same 68 substeps, at the same five
consecutive states `kt = 5760…5764`.

### The instrument was validated against a known answer three times

*(On the FIRST pass, against the two-Earth tree the §6a numbers were recorded
on — which is what makes it a valid known-answer check. The HEAD numbers are
further down and are not expected to reproduce these, since the baseline moved.)*

The first-pass **E1V** arm reproduces the recorded `dino_wall_fixed_bias.md` §6a
numbers **to the digit** on all six reductions (basin 16.2 %, wall rows 24.9 %,
south 9.1 %, north 25.6 %, tangential 46.0 %, northern lobe 61.7 %) and on the
alignment/amplitude triple (+0.548 / 0.941 / 0.300 wall-normal, +0.852 / 1.018 /
0.726 tangential). The routing probe independently reproduces the campaign's
recorded −5.48e-05 row-map median.

### THE ARMS WERE RE-RUN AT HEAD, ON THE ONE-EARTH MODEL

The first pass ran on a tree carrying two rotation rates, which made arm F a
two-variable arm and put the baseline on a geometry that no longer exists. All
five directories were re-run at `99e170f6e` into `results/dino_1455/arms_head/`;
the pre-fix maps are kept at `results/dino_1455/arms/` for the comparison below.
**Everything in this section is the HEAD re-run.**

#### The new baseline, and whether the pre-registered bars still transfer

| reduction | pre-fix (2 Earths) | **HEAD (1 Earth)** | change |
|---|---:|---:|---:|
| wall-normal basin | 1.8475e-07 | **1.7806e-07** | −3.62 % |
| both wall rows | 2.6725e-07 | 2.6719e-07 | −0.02 % |
| south wall row | 7.0338e-08 | 7.1429e-08 | +1.55 % |
| north wall row | 3.7135e-07 | 3.7106e-07 | −0.08 % |
| band 57–73 | 2.7875e-07 | 2.5483e-07 | **−8.58 %** |
| band 121–153 | 2.0909e-07 | 1.9450e-07 | **−6.97 %** |
| lobe 185–197 | 3.8139e-07 | 3.8438e-07 | +0.78 % |
| tangential basin | 1.2235e-07 | 1.1834e-07 | −3.28 % |

**The bars transfer, and the reason is stated rather than assumed.** The
pre-registered thresholds (OWNER > 50 % on both the basin and the lobe,
REFUTED < 10 %) are *fractional collapses of a state-constant residual*. Two
things have to hold for them to keep their meaning, and both are measured:
the object is still state-constant at HEAD (worst state-to-state departure
**1.59 %**, min signed cross-state correlation **+0.99977**, against 1.53 % /
+0.99979 before), and the denominator moved **3.6 %** — an order of magnitude
inside the nearest threshold, so no arm changes verdict band on account of it.
Concretely: turning the joint arm's 29.8 % into an OWNER call would need the
basin baseline to be **40 % larger** than it is, eleven times the move.

**Quote both numbers, because the scalar is far steadier than the object under
it.** The baseline's *magnitude* moved 3.6 %, but the baseline *field* moved
**11.5 %** (L2 of the five-state residual stack, pre-fix against HEAD). The bar
transferring cleanly is a statement about a ratio, not a claim that this is the
same residual.

**The Ω fix alone is NOT the headline** — it moved the basin residual 3.6 %.
**Its latitude signature is.** The rotation-rate error was uniform in latitude,
and removing it moved *only the two interior bands* (−8.6 %, −7.0 %) while the
walls and the northern lobe did not move at all (−0.02 %, +0.78 %). Those are
exactly the bands the Coriolis half owns and exactly the ones the metric half
does not. The wrong Earth was living in the Coriolis half's territory, which is
why the first pass's arm F was a two-variable arm — and the re-run makes it a
clean one.

#### The rate-and-convention split, now separable

Arm F substitutes NEMO's own `ff_f`, which on the old tree removed the rotation
rate *and* the placement convention together. With the rate fixed, the two
separate, on the wall-normal basin reduction:

| component | collapse |
|---|---:|
| rate **and** convention together (pre-fix arm F) | +9.93 % |
| **rate alone** (the baseline's own move) | **+3.62 %** |
| **convention alone** (HEAD arm F) | **+6.55 %** |

They compose multiplicatively: `(1 − 0.0362)(1 − 0.0655) = 0.900692` against
`(1 − 0.0993) = 0.900692`. **That identity is NOT an independent check and is
not offered as one** — it telescopes. Because arm F's absolute residual is the
same on both trees, `(BASE_head/BASE_pre)·(F_head/BASE_head) = F_head/BASE_pre`
identically, so the six-digit agreement encodes exactly one fact: that arm F is
invariant across the fix. That fact is measured directly, and far more tightly,
in the table below (1.6e-11, not 1e-6). Cite that, not this.

**And the split is SEQUENTIAL, not a share.** "Convention alone = 6.55 %" means
*convention, given the rate is already fixed*. The other ordering — the
convention's effect on the old Earth — would need an arm that was never run.
The placement convention is roughly two thirds of what substituting NEMO's
array used to buy, in this ordering.

#### The check that the two trees differ in exactly one thing

An arm that **substitutes** NEMO's own `ff_f` replaces legoESM's `f` outright,
so the rotation-rate fix cannot reach it; an arm that **keeps** legoESM's own
`f` must move. Measured, as `max|pre-fix − HEAD| / max|field|`:

| arm | what it does with f | change across the fix |
|---|---|---:|
| BASE | keeps legoESM's | 5.0e-02 |
| E1V | keeps legoESM's | 8.4e-02 |
| F | substitutes NEMO's | **1.6e-11** |
| JOINT | substitutes NEMO's | **2.6e-11** |
| CTRLF | substitutes NEMO's | **2.1e-13** |

Exactly the two arms that should have moved did, and the three that should not
did not. (They are not bit-identical — 1e-11 relative, not 1e-16 — so something
still differs in the last bits of an inert path; it is eleven orders below the
smallest effect in the table and is not chased here, but it is recorded rather
than rounded to "identical".)

### The collapse table — HEAD

State-constant residual, five-state mean, % collapse against the HEAD baseline:

| reduction | BASE | E1V | F | E1V+F | JOINT |
|---|---:|---:|---:|---:|---:|
| wall-normal basin | 1.7806e-07 | +18.9 | +6.5 | +25.4 | **+29.8** |
| wall-normal, both wall rows | 2.6719e-07 | +25.2 | +0.4 | +25.5 | +25.9 |
| south wall row | 7.1429e-08 | +8.5 | −1.1 | +7.5 | +7.3 |
| north wall row | 3.7106e-07 | +25.9 | +0.4 | +26.3 | +26.6 |
| band, southern interior 57–73 | 2.5483e-07 | −3.6 | **+25.7** | +22.1 | +24.8 |
| band, northern interior 121–153 | 1.9450e-07 | −2.6 | **+20.2** | +17.6 | +19.0 |
| band, northern lobe 185–197 | 3.8438e-07 | **+61.8** | −0.7 | +61.0 | **+62.0** |
| tangential basin | 1.1834e-07 | +47.7 | +4.5 | +52.2 | +53.2 |

**The disjointness survives the deconfounding, which is the point of the
re-run.** With the rotation rate no longer riding along, the Coriolis half is a
pure convention arm and it *still* owns both interior bands (+25.7 %, +20.2 %)
and *still* does nothing to the northern lobe (−0.7 %); the metric half still
owns the lobe (+61.8 %) and still moves the interior bands the wrong way
(−3.6 %, −2.6 %). Every number is smaller than on the old tree — because the Ω
fix already removed part of what arm F used to remove — and the structure is
unchanged.

**The halves still superpose exactly:**
`‖(Δ_E1V + Δ_F) − Δ_JOINT‖ / ‖Δ_JOINT‖` = **3.47e-06** (wall-normal),
**4.12e-06** (tangential). There is no interaction, on either tree. A cancelling
pair would show one.

### The registered verdict at HEAD

- JOINT basin-wide **+29.8 %** — below the OWNER bar (was +32.3 % pre-fix; it
  *fell*, because the Ω fix took interior-band error out of the baseline that
  arm F used to be credited with removing).
- JOINT northern lobe **+62.0 %** — above it (essentially unchanged from +61.8 %,
  as it must be: nothing the Ω fix touched lives there).

**VERDICT: PARTIAL** — the same band as the first pass, on a cleaner
measurement. The **staggering control FIRES** at **+12 808 %** above baseline
against a +50 % bar. Every arm's residual is still a fixed field.

**Two caveats on the control, adopted from review rather than argued with.**
(a) It is `(metric off, Coriolis staggered)`, so it is *not* one variable off
the joint arm it gates; what it establishes is that the loop reads this array at
all, and the joint arm inherits that. The stricter `(metric on, Coriolis
staggered)` control was not run. (b) It fires at **328×** the candidate's own
amplitude on the corrected model (1.281e-02 against the convention-only
3.902e-05). It was 234× against the two-Earth tree's combined 5.480e-05, and
quoting *that* number after the fix understates the mismatch — the control is a
worse match to the signal than the old line said. So it licenses "the loop reads
this array", **not** "the loop resolves the candidate". For the latter the only
evidence is arm F's own measured response.

**The 90-day gate is NOT bought**, by the pre-registered condition (joint arm
must clear 50 % per-step; it reads 29.8 %).

### Post-hoc, and labelled as such: alignment versus amplitude

Not the registered rule. The sharper diagnostic the mechanism review proposed —
correlation of each arm's own response field against the baseline residual, plus
the best-fit rescale:

| arm | channel | corr | best-fit rescale α | variance explained |
|---|---|---:|---:|---:|
| E1V | wall-normal | +0.586 | 0.971 | 0.343 |
| F | wall-normal | +0.364 | 1.185 | 0.132 |
| **JOINT** | **wall-normal (registered)** | +0.716 | 1.108 | 0.513 |
| E1V | tangential | +0.859 | 0.997 | 0.737 |
| F | tangential | +0.387 | 1.964 | 0.150 |
| **JOINT** | **tangential (the pair's real channel)** | **+0.884** | **0.992** | **0.781** |

(HEAD re-run. The pre-fix tree gave +0.740 / 1.109 / 0.548 and +0.891 / 0.995 /
0.795 — the deconfounding barely moves this diagnostic.)

**Amplitude is right with no fitted parameter** (α within 11 % of 1 on the
registered channel, within 1 % on the tangential one); the shortfall is
alignment. On the channel where the pair actually exists the joint arm explains
**78 %** of the variance. Read against the review's proposed bar (corr > 0.85
*and* α ∈ [0.8, 1.25]) the joint arm **passes on the tangential channel
(+0.884, 0.992) and fails on the wall-normal one (+0.716)** — which is the same conclusion the operator
measurement reached, arrived at independently.

---

## 4. STEP 1 — what was fixed, and what it costs

**(a) The rotation-rate routing — a straight bug fix.** `NEMO_CONSTANTS_CONFIG`
now transcribes NEMO's sidereal-day expression instead of the `key_cice`
literal, and `bridge_nemo_to_legoesm_topo` defaults to NEMO's Earth instead of
legoESM's rounded constant. The Coriolis guard that let this pass was 63× too
loose; it is tightened from 1e-3 to 1e-9 and made precision-aware (in fp32 the
stored array only carries ~1.2e-07, so a fixed 1e-9 would fail on rounding
rather than on physics). The metric guards keep their own 1e-3 bound — tightening
one guard must not silently re-scope two others.

**(b) `coriolis_placement`, a selectable option, default off and
bit-identical.** `"cell_average"` is the shipped construction; `"face_latitude"`
evaluates `f = 2 Ω sin φ_face` at the v-face latitude, NEMO's `ff_f` convention.
It lives in ONE array — the geometry's `f_v` — which every C-grid Coriolis path
reads through one of exactly two helpers (`vertex_coriolis` for the barotropic
EEN pre-block and the 3-D EEN/ENE vorticity flux; `coriolis_at_faces` for the
semi-implicit and `explicit_ab2` face-f Coriolis). `f_T`/`f_u` are deliberately
untouched: on a lat-lon grid the u-point shares the tracer row's latitude, so
`f_u` is already `f` at its own point.

### THE CONSERVATION TRADE, stated loudly because it is real

The row average is **not** arbitrary. It is exactly the discrete curl of
solid-body rotation on this grid — circulation over the dual-cell area gives
`Ω(sin φⱼ + sin φⱼ₋₁)`, which *is* the row average, using the grid's own vertex
area. Measured with legoESM's own curl operator on the DINO latitudes:

| construction | median \|curl(solid body) − f_v\| / 2Ω |
|---|---:|
| row average (`cell_average`) | **4.1e-15 — exact, to machine precision** |
| f at the face latitude | **2.0e-05** |

(Re-measured with legoESM's own curl operator and now a committed test,
`tests/ocean/unit/test_coriolis_placement.py`, rather than a throwaway probe.
The first row is *exact*, not merely 10× better, so the trade is larger than an
earlier draft of this document stated.)

So `"face_latitude"` trades a **10× degradation in discrete planetary-vorticity
(Kelvin/Stokes) consistency** for oracle fidelity: after the change the
planetary part is no longer the discrete curl of anything, and a fluid in exact
solid-body co-rotation acquires a spurious potential vorticity of order 1e-08
per second, fixed in space.

**It does NOT trade energy or enstrophy conservation ON THE EEN PATH.** The
Sadourny/AL81 identities are generic in the vertex field — a *random* vertex
Coriolis array gives a work residual of ≤ 4.3e-19 — so the conservation
property depends on every paired `(u, v)` contribution sharing ONE vertex
value, not on how that value was computed. **Scope it there and no further:**
the option also feeds `coriolis_at_faces`, the semi-implicit and
`explicit_ab2` face-f Coriolis, whose work cancellation is *not* the AL81
identity — it depends on the `(f_u, f_v)` pair, and this flag moves `f_v` while
leaving `f_u`. That path's work residual is **UNMEASURED**. And NEMO itself uses the face-latitude value with the same
scheme, so the oracle is on the less curl-consistent choice. (Enstrophy is
argued from the coefficient algebra and the machine-zero energy result, not
measured on a time-stepped run: PLAUSIBLE, not CONFIRMED.)

This is why the option is **default off** and will stay off outside oracle
cards.

---

## 5. A first-order defect found on the way, and why it did not bias the arm

legoESM's `f_v` carries the **tracer-row** value at both end rows rather than a
face value. That is a **first-order half-cell error**, not the second-order
convention the arm is about. Measured on the DINO mesh:

| row | relative error vs NEMO |
|---|---:|
| array median | −5.48e-05 |
| northern lobe (185–197) | −2.73e-05 |
| **north wall row (jpj = 199)** | **−1.108e-03** |
| its own neighbour (row 198) | −2.497e-05 |

20× the median and **44× its own neighbour**. `"face_latitude"` fixes it (it
uses the true wall latitude, where `sin = ±1`), and that is unit-tested.

**It does not bias the arm**, and that is measured rather than argued: the arm's
reachability control corrupts both end vertex rows — along with the `f_u`/`f_v`
loop kwargs and `grid.f_u`/`f_v`/`f_T`, seven arrays, each verified nonzero
(max 1.3692e-04) before being scaled by 1e3 — and the loop's carry comes back
**bit-identical, 0.000e+00**. The last wet v-face is row 196 and the AL81 triad
cannot reach vertex row 199. Re-running the arms to substitute an unreachable
row would have bought nothing.

---

## 6. What this does and does not claim

- **Not** that the vertex Coriolis owns the southern-basin −0.95 Sv deficit. The
  object here is the barotropic loop's state-constant wall-normal residual, a
  per-step object.
- **Not** a climate statement. No 90-day run was bought, by registered
  condition.
- The in-loop **zonal deposit** collapsing 69.5 % under the Coriolis half is a
  **signed, area-weighted** reduction — the stand-in for circumpolar transport —
  and it is NOT the RMS the verdict is registered on. The same half moves the
  tangential RMS only 7.6 %. Two different reductions of the same field; they
  must never be quoted as one number.

  **But the gap between them is itself the finding, and it is geometric, not an
  artifact.** There is no sign-structured cancellation: the Coriolis arm's
  *response field* correlates with the baseline residual at **+0.364** at HEAD
  (positive means it removes something aligned with the residual, which is what
  a real fix does). An earlier draft quoted **−0.456** here — that was the
  pre-fix tree AND a different pairing, and the sign made it read as
  anti-alignment; both are corrected. Measured directly, the
  transport-deposit direction captures only **14.3 %** of the baseline
  residual's norm, while the Coriolis arm's response is **52.7 %** aligned with
  it — and the metric arm's is **0.2 %**, i.e. essentially orthogonal (HEAD
  re-run; the pre-fix tree gave 16.9 % / 48.6 % / 0.2 %, so the deconfounding
  sharpened the concentration rather than explaining it away). The
  Coriolis error's damage is ~3× concentrated in the net-transport mode
  relative to the residual as a whole, and the metric error is not in that mode
  at all. That is the strongest argument in this document for re-registering on
  the transport channel.

## 7. Named next steps, not started

0. ~~Re-run the five arms on HEAD.~~ **DONE** — §3 is the HEAD re-run, arm F is
   now a clean convention arm, and the verdict is unchanged at PARTIAL.
1. **Re-register on the zonal channel.** That is where the pair exists, where
   the joint arm already explains 80 % of the variance at α = 0.995, and where
   the transport-relevant deposit lives.
2. ~~A rotation-rate-only arm to separate the constant from the convention.~~
   **DONE, and without a new arm**: the baseline's own move under the Ω fix
   (+3.62 %) IS the rate-only collapse, and it composes multiplicatively with
   the HEAD arm F (+6.55 %) to reproduce the old two-variable +9.93 % to six
   digits. See §3.
3. **Swap the RMS-collapse bar for correlation-plus-rescale.** An RMS threshold
   conflates alignment with amplitude and is sign-blind.
4. **The wall-row first-order defect** is real, is 20× the convention error, and
   is unreachable only in *this* loop. Its reach in the 3-D scheme is unmeasured.
