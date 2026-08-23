# PRE-REGISTRATION — the one-variable lateral-viscosity ablation at the southern wall

Written 2026-08-23, BEFORE any number from either arm existed.
Registered by: #1455 wall-friction lane. Issue: #1455.

## The question

`docs/ocean/fidelity/dino_basin_budget_result.md` Part 4 measured that legoESM
piles ~4.4 mm too much sea surface against DINO's southern free-slip wall,
decaying with an e-folding scale of 2.6 rows ~ 112 km, and that the westward
circulation lobe in the four wall rows (69.5S..68.4S) is 34% too strong while
the rest of the basin is right to 3%. The lobe is 91% geostrophic, so geostrophy
names a target, not an owner. The only discriminating number so far is the
LENGTH SCALE: 112 km is not the grid scale (1 row = 43 km) and not the
barotropic deformation radius (~1500 km), but it is within 30% of the Munk
frictional boundary-layer width computed with the oracle's own viscosity at
these rows (5387 m2/s -> 87 km = 2.0 rows). That is one coincidence and one
number, so the friction hypothesis is PLAUSIBLE and untested.

This arm tests it by perturbation, which is the only thing that can promote it.

## The one variable

`DINOConfig.U_M` (NEMO `rn_Uv`), 0.27 -> 0.54 m/s, through the new
`kamm_twin_90d.py --u-m` flag. `U_M` is read in exactly one place on this lane,
`dino.py:3020` `A_h_base = 0.5*U_M*R*dlon`, so doubling it doubles the lateral
viscosity coefficient everywhere and changes nothing else: the card sets
`B_h = 0` (the biharmonic raises if nonzero under this operator),
`barotropic_diffusion_alpha = 0`, and `A_h_eq_boost`/`A_h_cap_boost`/`A_h_floor`
are all at their no-op defaults. The runner's `smoke-check` mode asserts the
doubling reaches the built `lateral_viscosity.A_h`, so the knob cannot be
vacuous.

Everything else is byte-identical between the two arms: same commit, same
restart (`RUN_STEPDUMP/DINO_00005760_restart.nc`, NEMO day 180), same vertical
ladder (`LEGOESM_NEMO_E3T=both`), same 90 days, same snapshot days, same
`--bridge-before --surface-tendency-placement leapfrog_rhs`, same fp64 policy.
Both arms are re-run at THIS commit; no arm from an earlier commit is reused.

## (a) The number each arm produces

1. `G4` = the day-90 southern-basin zonal-transport gap (legoESM - NEMO) summed
   over the four wall rows 69.5S..68.4S (grid rows j=1..4; row j=0 is entirely
   dry). Baseline expectation from the parent document: the whole-basin day-90
   gap is -0.43 Sv and the wall rows carry ~95% of it.
2. `A_wall` = the day-90 sea-surface excess against the wall [mm] and `L_e`, its
   meridional e-folding scale [rows], from the same exponential fit
   `southern_wall_balance.py` used to report 4.4 mm / 2.6 rows.
3. The 5-metric 90-day acceptance gate (`acceptance_gate_90d.py`), including the
   circumpolar channel transport, which is the campaign's crown jewel and is
   reported for BOTH arms whatever the verdict.

## (b) What confirms and what refutes

Munk scaling: the frictional boundary-layer width goes as the cube root of the
viscosity, so doubling widens it by 2^(1/3) = 1.26x. A boundary current whose
transport is set upstream and whose width is set by friction then has its peak
amplitude scaled by 1/1.26, i.e. -21%; the parent document registered "-26% for
a doubling" from the same cube root. Both readings live in the same band, so the
registered bands below are deliberately wider than the gap between them.

* **CONFIRM (friction owns the wall scale)**: `|G4|` moves by **>= 15%**
  between the arms, OR `L_e` moves by **>= 15%**. Direction is recorded but is
  not part of the pass condition: a shrink toward zero is the Munk-predicted
  sign, a growth means the lateral operator still owns the lobe with the
  opposite sign, and either way friction is the live lever.
* **REFUTE (friction exonerated)**: BOTH `|G4|` and `L_e` move by **< 5%**.
  Then the wall lobe is not set by the lateral-friction layer, Task 2 (a wall
  treatment fix) does NOT run, and the barotropic free-surface solve returns to
  the top of the candidate list. This is to be reported loudly.
* **NO VERDICT**: any outcome between 5% and 15% on both quantities. Reported as
  no verdict; no candidate is promoted or retired on it.

Noise floor for the comparison: the #1492 item-2.1 micro-ensemble floor at day
90 is 0.091 Sv on the circumpolar transport; the wall-row gap has no published
floor of its own, so the arm reports `|G4_2x - G4_1x|` next to 0.091 Sv as an
order-of-magnitude sanity bound and does not claim a move smaller than that.

## (c) Why no cheaper offline test answers it

The quantity under test is an EQUILIBRATED boundary-layer width, not a
single-step tendency. An offline tendency comparison on a saved state (which
this lane also runs, as the alignment table's numeric row) measures the
operator; it cannot measure the layer the operator sets up over days, and the
parent document's own stage and per-term tables are closed by a 310:1
cancellation, so no bookkeeping decomposition of the existing states can
substitute. The cost is two 90-day twins at ~215 s of wall clock each,
i.e. ~0.12 GPU-hours total, which is below the 1 GPU-hour threshold that would
require this justification at all; it is written because the rule is that every
run is pre-registered, not because the cost is large.

## Recorded before the fact

Nothing about either arm's outcome is known at the time of writing. If the
result lands in the NO VERDICT band it will be reported as such rather than
re-cut against a new threshold.

---

# ADDENDUM, written 2026-08-23 after dual adversarial review, before the 1.5x arm ran

Two independent reviewers read the two arms above. Both found the same defect in
this pre-registration and each found one the other did not. What follows is
registered BEFORE the third arm was launched; the two existing arms' numbers are
known and are restated here so the reader can see exactly what was and was not
in hand when the new bands were written.

## What is withdrawn from the registration above

**W1. The signed four-row sum `G4` is withdrawn as the primary metric, and the
CONFIRM it produced is withdrawn with it.** `G4` is a signed sum over four rows,
and under a doubled viscosity the per-row error becomes DIPOLAR inside that
window (+0.218, +0.123, -0.059, -0.237 Sv on rows 1-4). The signed sum therefore
falls 85% while the sum of the individual rows' magnitudes RISES from 0.288 to
0.636 Sv, and across the whole southern band from 0.446 to 1.800 Sv. Every row
got worse; they cancel. A metric that a sign flip can turn from a degradation
into an 85% improvement is not a metric. **The registered metric set gains
`sum |gap|` over rows 1-4 and over rows 0-13, and no verdict may be issued on a
signed sum alone.**

**W2. The Munk `A^(1/3)` prediction is withdrawn as the wrong scaling law, not
merely as a mis-estimate.** `(A/beta)^(1/3)` balances `beta*psi_x` against
`A*del4(psi)` in a boundary layer whose cross-shore direction is zonal, i.e. on
a MERIDIONAL wall. DINO's southern wall is ZONAL and the current along it is
zonal, and the basin is closed in longitude, so the depth-integrated meridional
transport vanishes at every latitude at steady state and `beta*v` never enters
the balance. The 87 km the parent document quotes has no derivation on this
wall; its agreement with the measured 112 km is arithmetic. A frictional
sidewall layer here scales as `sqrt(A/r_bottom)`, which with NEMO's own drag
constants is of order 500 km, and the 90-day viscous spreading scale
`sqrt(A*t)` is about 205 km. Neither is 112 km either.

**W3. The e-folding scale `L_e` is withdrawn as a mechanism discriminator.** On
this grid the first baroclinic deformation radius is roughly 9 km against a 40
km cell, so any wall-trapped baroclinic structure is resolution-limited to two
or three cells whatever produces it. The measured 2.69 rows IS that limit. It
therefore carries no information about which term made the anomaly, and the
parent document's argument that "112 km is neither the grid scale nor the
deformation radius" does not hold: 2.69 rows is the grid scale for a smooth
decaying structure. `L_e` stays in the reported table as a shape descriptor and
is no longer a CONFIRM route.

**W4. "One variable" was true of the CONFIG and false of the REGION.** Doubling
the lateral viscosity changes legoESM's own velocity by about 12% RMS in every
latitude band of the model, including 70N, and moves the sea surface more
outside the southern basin (5-6.6 mm) than inside it (1.8-2.2 mm). It is a
globally-acting parameter read through a four-row window, so an improvement seen
in that window has to survive a compensation check before it counts.

## What the two existing arms are now taken to have established

Restated with the metric set of W1, and labelled:

* **CONFIRMED**: the four wall rows are strongly sensitive to the lateral
  viscosity. Every norm moves by far more than the registered 15%: the signed
  sum by 85%, the unsigned by 120%, and the change is 3.7 times the day-90
  ensemble floor.
* **CONFIRMED**: doubling the viscosity DEGRADES the southern basin. Unsigned
  error 2.2x worse over the four wall rows and 4.0x worse over rows 0-13.
* **CONFIRMED**: the band total is nearly invariant (-0.4258 to -0.4180 Sv,
  1.8%). With free slip at the wall the meridionally-integrated lateral friction
  telescopes to a single stress at the band's northern edge, so this is close to
  a property of the operator rather than a discovery -- which makes it an
  argument AGAINST lateral friction owning a band-integrated deficit.
* **NOT ESTABLISHED**: that friction owns the wall lobe. The verdict registered
  above is withdrawn.

## The third arm, registered now

**One variable, `rn_Uv` = 0.405 (1.5x), everything else byte-identical to the
other two arms, same commit, all three re-run together so that no arm predates
the artifact stamp the reader now requires.**

It separates the only two readings of the existing pair that survive review:

* **Reading A -- an effective-coefficient deficit.** legoESM's realized lateral
  dissipation at these rows is genuinely about 1.8x too weak (the zero crossing
  of the signed `G4` between the two existing arms sits at 1.82-1.87x). Then the
  UNSIGNED basin error `sum |gap|` over rows 0-13 falls from 1x through 1.5x and
  is at or near a minimum somewhere below 2x.
* **Reading B -- the viscosity is a lever on someone else's error.** Then
  `sum |gap|` rises MONOTONICALLY from 1x through 1.5x to 2x, and the signed
  `G4` crosses zero purely by cancellation.

**Registered discriminator**: `sum |gap|` over rows 0-13 at 1.5x, against 0.4459
(1x) and 1.8002 (2x).
  * below 0.4459 -> **Reading A**, and the effective-coefficient deficit becomes
    the named next work item.
  * above 1.8002, or monotone between the two endpoints within 10% of the
    straight line joining them -> **Reading B**, and the lateral viscosity is
    recorded as a lever and not the owner.
  * anything else -> reported as neither, with the numbers.

**Second registered reading, on the same arm**: the upper density contrast error
against its 5.5e-4 kg/m3 gate, alongside the circumpolar transport error. A real
improvement moves both monotonically in the same direction. A compensating error
shows the transport metric with an interior optimum while the density metric
degrades monotonically. The two existing arms already show transport 5.6x better
while its own controlling density contrast is 6x worse, which is the
compensation signature; the third point says whether that is monotone.

**Cost**: one 90-day twin, about 215 s. All three arms re-run: about 11 minutes.

Nothing about the third arm's outcome is known at the time of writing.
