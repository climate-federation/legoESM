# PRE-REGISTRATION — the wall balance test, and two controls

Written BEFORE any number existed. Lane: `fidelity/dino-basin-budget-1yr`.
Parent: `docs/ocean/fidelity/dino_basin_budget_result.md`, commit `c73198004`.

## Why a balance test rather than another decomposition

Every bookkeeping route into this defect has now been closed by cancellation:
the stage table is an identity, and the per-term table closes to 2e-9 but its
groups cancel 310:1 against the quantity they would have to explain. A balance
test asks a different question — *is the flow in the balance it should be in?* —
and its numerator is the quantity of interest, so it cannot be swamped that way.

The measured defect: legoESM's **westward** circulation lobe in the four rows
against the southern wall is **34%** too strong, the rest of the basin is right
to 3%, and a rigid displacement of the profile explains under 1%.

## B1 — is the too-strong lobe in geostrophic balance?

A depth-mean flow in near-geostrophic balance is set by the sea-surface slope
plus the depth-integrated density gradient. **The two models' densities agree to
4e-5 kg/m³**, so between the two models the density term cancels and any
difference in the depth-mean flow must be matched by a difference in the
sea-surface slope if the lobe is geostrophic.

**Measure**, per wall row, on both models' saved states at all 19 matched days:

* `R` — the depth-integrated row circulation, the campaign's recorded reducer;
* `R_ssh` — the same reducer applied to the geostrophic velocity the model's own
  sea-surface field implies, `u_g = −(g/f)·∂η/∂y`, on the identical geometry;
* the ratio `legoESM/NEMO` of each, and the fraction of `ΔR` that `ΔR_ssh`
  explains.

**Tolerance** is the ensemble spread of the same functional: all four members
of each side at day 360, combined as the campaign's two-sided floor. A
difference inside twice that floor is not a difference.

* **CONFIRM geostrophic** — `ΔR_ssh` explains **≥ 70%** of `ΔR` in the wall
  rows, and the sea-surface-slope ratio is within ±10 percentage points of the
  circulation ratio (1.34). **Consequence: the owner is whatever sets the
  sea-surface set-up against the wall — the barotropic solve's wall
  treatment — and the next lane works on that.**
* **REFUTE geostrophic** — `ΔR_ssh` explains **≤ 30%**. **Consequence: the lobe
  is ageostrophic and the owner is friction or advection at the wall.**
* Between: **PARTIAL**, and the report says what fraction, with no verdict.

**What this cannot do.** It is a diagnosis of the STATE, not of the operator: a
sea-surface slope that is too large is equally consistent with the barotropic
solve producing it and with something else forcing the solve to produce it. It
picks the branch, not the line of code. And near a wall the flow is not exactly
geostrophic even in the oracle, so the *fraction explained* is the statistic,
never the residual on its own.

## C1 — are the per-term group differences physics or diagnostic staging?

The per-term table's groups differ by 4–8% while the totals agree to 0.02%,
which is the signature of a time-level or staging difference in the
diagnostics rather than of physics.

**Measure**: the same group differences in the FIRST 10-day window, when the
two trajectories still agree to the precision at which the states are stored,
against the year mean.

* **CONFIRM staging** — the window-1 group differences are within a factor of
  two of the year-mean ones, i.e. present before the trajectories have
  meaningfully separated. Consequence: the term table's group differences are
  not physics and must never be attributed.
* **REFUTE staging** — window-1 differences are under a fifth of the year-mean
  ones, i.e. they grow with the trajectory separation. Consequence: the group
  differences are physical and a matched term correspondence would be worth
  building after all.

## C2 — does legoESM's fused pressure-gradient diagnostic exclude the surface term?

The per-term table pairs legoESM's fused kinetic-energy-plus-pressure
diagnostic against the oracle's `keg + zad + hpg`. The oracle's `hpg` is the
hydrostatic gradient only; its surface gradient is a separate term. If
legoESM's fused diagnostic carried the surface gradient too, that pairing would
be wrong and the group differences would be an artifact of the pairing.

**Measure**: read the assembly in the production code and state which pressure
enters it, with the line. This is a code fact, not a run.

## Review

One reviewer suffices for a read-only balance probe; two if anything surprises.
