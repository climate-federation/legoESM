# The Coriolis pair: the premise is RETRACTED, and the two halves are disjoint

**Outcome.** The campaign's ranked-first next action (`dino_campaign_synthesis.md`
§3 item 1) was registered on a premise that is **false in the channel it
registered**. legoESM's vertex-Coriolis error and its v-face zonal-metric error
are **not** a cancelling pair in the wall-normal tendency: the v-face zonal
width enters the **zonal** tendency only, and its effect on the meridional
tendency is **exactly zero**. In the registered channel the Coriolis error is
unpaired, nothing cancels it, and the three arms bear this out — the joint arm
is the **sum** of its halves, never more. The two halves own **disjoint bands**.

The registered verdict is **PARTIAL**, the staggering control **fires**, and the
pre-registered condition therefore **withholds** the 90-day gate run.

Two code defects were found on the way. One had the entire oracle-matching lane
building its Coriolis on a different planet from the oracle; it is fixed here.

Pre-registration: `scripts/validate/ocean_fidelity/dino_1226/PREREG_coriolis_pair.md`
(committed before any arm was scored). Probes: `coriolis_omega_routing_audit.py`,
`coriolis_pair_arm_compare.py`, and the `DINO_1455_SUB_CORIOLIS` arm inside
`substep_traj_compare.py`. Artifacts: `results/dino_1455/arms/`.

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
meridional tendency at all. The Coriolis error reaches both.

**So the pair is real in the ZONAL tendency and ABSENT in the meridional one —
and the pre-registration scores the meridional (wall-normal) component.** The
verdict below is reported as registered; the registration itself is retracted as
mis-aimed. Credit: the mechanism review caught this; it was then re-measured
independently before being written down.

The arms confirm it without reference to the operator algebra: a cancelling pair
predicts the joint arm **exceeds** the sum of its halves. It does not.

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

The re-run **E1V** arm reproduces the recorded `dino_wall_fixed_bias.md` §6a
numbers **to the digit** on all six reductions (basin 16.2 %, wall rows 24.9 %,
south 9.1 %, north 25.6 %, tangential 46.0 %, northern lobe 61.7 %) and on the
alignment/amplitude triple (+0.548 / 0.941 / 0.300 wall-normal, +0.852 / 1.018 /
0.726 tangential). The routing probe independently reproduces the campaign's
recorded −5.48e-05 row-map median.

### The collapse table

State-constant residual, five-state mean, % collapse against BASE:

| reduction | BASE | E1V | F | E1V+F | JOINT | joint − sum |
|---|---:|---:|---:|---:|---:|---:|
| wall-normal basin | 1.8475e-07 | +16.2 | +9.9 | +26.1 | **+32.3** | +6.2 |
| wall-normal, both wall rows | 2.6725e-07 | +24.9 | +0.4 | +25.3 | +25.9 | +0.6 |
| south wall row | 7.0338e-08 | +9.1 | −2.6 | +6.5 | +5.8 | −0.6 |
| north wall row | 3.7135e-07 | +25.6 | +0.5 | +26.0 | +26.7 | +0.7 |
| band, southern interior 57–73 | 2.7875e-07 | −3.7 | **+32.1** | +28.4 | +31.3 | +2.9 |
| band, northern interior 121–153 | 2.0909e-07 | −2.8 | **+25.8** | +23.0 | +24.7 | +1.6 |
| band, northern lobe 185–197 | 3.8139e-07 | **+61.7** | −1.5 | +60.2 | **+61.8** | +1.6 |
| tangential basin | 1.2235e-07 | +46.0 | +7.6 | +53.6 | +54.7 | +1.1 |
| in-loop zonal deposit | 1.2610e-03 Sv/step | −2.4 | **+69.5** | +67.1 | +67.1 | −0.0 |

**The two halves are disjoint.** The metric owns the northern lobe (61.7 %) and
does nothing to the interior bands (−3.7 %, −2.8 %). The Coriolis owns both
interior bands (32.1 %, 25.8 %) and the zonal deposit (69.5 %), and does nothing
to the lobe (−1.5 %). This matches the two error profiles — the metric gap peaks
at the walls, the Coriolis gap at the equator — and it explains the campaign's
three-band decomposition with one term per band.

**Additivity.** `joint − (E1V + F)` is +0.6 to +6.2 points, ≤ 2.9 on every
reduction but the basin. A cancelling pair predicts the joint arm to *exceed*
the sum; it equals it. The pair framing is refuted by the arms as well as by
the operator.

### The registered verdict

Registered on the **joint arm alone**: OWNER above 50 % on **both** the
basin-wide and the northern-lobe reduction, REFUTED below 10 %.

- JOINT basin-wide **+32.3 %** — below the OWNER bar.
- JOINT northern lobe **+61.8 %** — above it.

**VERDICT: PARTIAL.**

**The staggering control FIRES**: the un-shifted `ff_f` puts the wall-normal
basin residual **+12 340 %** above baseline, against a registered bar of +50 %.
Every arm's residual is still a fixed field (worst state-to-state departure
1.53–2.26 %, min signed cross-state correlation ≥ +0.9995), so the comparison is
like for like.

**Two caveats on the control, adopted from review rather than argued with.**
(a) It is `(metric off, Coriolis staggered)`, so it is *not* one variable off
the joint arm it gates; what it establishes is that the loop reads this array at
all, and JOINT inherits that. (b) It fires at **234×** the candidate's own
amplitude (1.281e-02 against 5.480e-05), so it licenses "the loop reads this
array", **not** "the loop resolves 5.5e-05". For the latter the evidence is the
Coriolis arm's own measured response.

### The 90-day gate: NOT RUN, per the registered condition

The pre-registration made the gate conditional on the joint arm clearing 50 %
per-step. It reads 32.3 %. The gate is **not** run and the chain prediction is
**not** issued. This is the compute-discipline rule working as intended.

### Post-hoc, and labelled as such: alignment versus amplitude

Not the registered rule. The sharper diagnostic the mechanism review proposed —
correlation of each arm's own response field against the baseline residual, plus
the best-fit rescale:

| arm | channel | corr | best-fit rescale α | variance explained |
|---|---|---:|---:|---:|
| E1V | wall-normal | +0.548 | 0.941 | 0.300 |
| F | wall-normal | +0.436 | 1.077 | 0.190 |
| **JOINT** | **wall-normal (registered)** | +0.740 | 1.109 | 0.548 |
| E1V | tangential | +0.852 | 1.018 | 0.726 |
| F | tangential | +0.456 | 1.738 | 0.208 |
| **JOINT** | **tangential (the pair's real channel)** | **+0.891** | **0.995** | **0.795** |

**Amplitude is right with no fitted parameter** (α within 11 % of 1 on the
registered channel, within 0.5 % on the tangential one); the shortfall is
alignment. On the channel where the pair actually exists the joint arm explains
**80 %** of the variance. Read against the review's proposed bar (corr > 0.85
*and* α ∈ [0.8, 1.25]) the joint arm **passes on the tangential channel and
fails on the wall-normal one** — which is the same conclusion the operator
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
| row average (`cell_average`) | 2.1e-06 |
| f at the face latitude | **2.1e-05** (10× worse) |

So `"face_latitude"` trades a **10× degradation in discrete planetary-vorticity
(Kelvin/Stokes) consistency** for oracle fidelity: after the change the
planetary part is no longer the discrete curl of anything, and a fluid in exact
solid-body co-rotation acquires a spurious potential vorticity of order 1e-08
per second, fixed in space.

**It does NOT trade energy or enstrophy conservation.** The Sadourny/AL81
identities are generic in the vertex field — a *random* vertex Coriolis array
gives a work residual of ≤ 4.3e-19 — so the conservation property depends on
every paired `(u, v)` contribution sharing ONE vertex value, not on how that
value was computed. And NEMO itself uses the face-latitude value with the same
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

## 7. Named next steps, not started

1. **Re-register on the zonal channel.** That is where the pair exists, where
   the joint arm already explains 80 % of the variance at α = 0.995, and where
   the transport-relevant deposit lives.
2. **A rotation-rate-only arm** would separate the constant from the convention
   inside arm F. It is now moot for future arms — the constant is fixed — but
   the F numbers in §3 were recorded *before* that fix and therefore carry both.
3. **Swap the RMS-collapse bar for correlation-plus-rescale.** An RMS threshold
   conflates alignment with amplitude and is sign-blind.
4. **The wall-row first-order defect** is real, is 20× the convention error, and
   is unreachable only in *this* loop. Its reach in the 3-D scheme is unmeasured.
