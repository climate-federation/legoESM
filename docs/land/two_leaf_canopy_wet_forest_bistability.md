# Two-leaf canopy: GPP/LE collapse at dense wet broadleaf forests

## Symptom

The two-leaf canopy surface scheme severely under-predicts gross primary
productivity and evapotranspiration at dense, high-insolation evergreen
broadleaf forests. Diagnosed at two FLUXNET sites (offline `run_ec_site`,
2015-2018):

| site | LAI | GPP mod / obs | LE mod / obs | H mod / obs (midday) |
|------|-----|---------------|--------------|----------------------|
| AU-Tum (Tumbarumba, wet eucalypt) | 5.8 | 9 / 24 | 92 / 246 | 328 / 184 |
| GF-Guy (Guyaflux, tropical rainforest) | 6.0 | ~3 / 10 | ~46 / 93 | high |

Both fail identically (~4-5x low GPP/LE, excess sensible heat). The temperate
sites the scheme was validated at (US-MMS, DE-Obe, US-Ton) are unaffected.

## Root cause: a bistable canopy-air-humidity <-> stomatal-conductance coupling

The two-leaf canopy air space (`q_c`) is coupled to the leaves through the
Ball-Berry stomatal model (`gs` proportional to leaf-surface relative humidity).
At a dense canopy under intense radiation the time-integrated closure settles
into a **self-reinforcing closed state**:

```
dry canopy air -> high leaf-to-air VPD -> Ball-Berry gs collapses
   -> transpiration fails -> canopy air stays dry            (positive feedback)
   -> leaves overheat (+6..+8 C over air, growing with SW)
   -> surplus energy dumped as sensible heat (H > 400 W/m2)
   -> assimilation becomes CO2-supply-limited (Vcmax-insensitive)
```

Measured at AU-Tum midday: leaf-air gap +6.3 C, `gs` stuck at ~0.001 m/s
*independent of light level* (the tell-tale of a supply-limited fixed point).

The **open branch** (moist canopy air, open stomata, GPP ~20 umol) is an equally
valid solution: a fresh cold-start of the identical canopy (same forcing and
state) reliably finds it. The system is genuinely **bistable**, and the full
integration falls into the wrong basin at these sites while the working
temperate sites sit in the open basin.

## Systematically ruled out

Each was tested, not assumed (probes in `scripts/tmp/_probe_canopy_*.py`,
`_probe_fullrun_internals.py`):

| suspect | test | result |
|---------|------|--------|
| Soil-moisture stress | cold-start at `w_frac_rz` 0.85 -> 0.70 | stays healthy |
| Vcmax | driver 32 -> 70 | no change (supply-limited) |
| Temperature acclimation (`TgC`) | cold-start at TgC=9.1 C | stays healthy (+ analytic: Topt~28 C either way) |
| Energy-balance LE cap | `le_cap_mode="off"` on full run | identical |
| Soil-surface evap resistance (#671/this work) | series off on full run | slightly *worse* -> cleared |
| Driver radiation quality | field comparison vs gap-filled sites | identical (PPFD_IN, PPFD_DIF, diffuse ratios present) |
| Aerodynamic decoupling to atmosphere | H is high (400+) | canopy is well-coupled |
| Radiative transfer + Farquhar core | standalone at Ci=280 ppm, LAI 5.8 | GPP ~30, healthy |

## Why it is a closure (not physics) problem

The Newton initial guess is a cold `[T_air, T_air, 0.7*Ca, 0.7*Ca, T_air, q_c_init]`
each timestep with `q_c_init = 0.5*(q_sat(Ts) + q_atm)`, and the outer Picard
loop re-solves from that same guess each pass while only `Ts_bc` evolves
(`surface_scheme/two_leaf_canopy.py`). With a cool soil boundary the cold guess
lands in the open basin; with the prognostic (hot) soil boundary at these sites
it lands in the closed basin. So the branch selection is governed by the
initial canopy-air humidity relative to the soil boundary temperature -- a
conditioning/initialisation issue in the coupled closure, not the leaf
biophysics.

## The two branches are both stable fixed points

The closed state is a genuine **converged** fixed point of the coupled
canopy-soil system, not an iteration artifact:

- Picard passes 6 / 20 / 40 -> identical GPP 5.2 (fully converged).
- Initial canopy-air humidity `q_c` weight 0.5 / 0.9 / 1.0 -> identical
  (the Newton returns the closed branch regardless of where it starts).

A fresh cold-start of the *isolated* canopy (`compute_two_leaf_canopy_fluxes`
with a static soil-temperature callback) instead returns the **open** branch
(GPP ~20, gs open) for the same forcing -- and stays open even with the soil
skin forced +15 C, `w_frac_rz` down to 0.70, or `TgC`=9.1. So both branches
exist and are stable; the difference is that the full multilayer step, with the
real prognostic soil-thermal feedback, co-evolves the soil and the
non-transpiring canopy into the self-consistent *closed* state, whereas the
temperate sites and the static-soil probe land open.

The branch selection could **not** be reduced to any single scalar input passed
to the canopy (all of `w_frac_rz`, `TgC`, soil-skin temperature magnitude,
`q_c` init, `w_frac_soil_evap`, soil-surface resistance, LE cap, wind, and the
Picard count were tested individually and none flip the isolated solve). It is
an emergent property of the coupled canopy<->soil-thermal iteration under the
dense-canopy / intense-radiation regime.

## Fix direction (status: NOT yet solved)

Because both branches are stable, a robust fix requires **branch selection**,
not a parameter or an initial-guess tweak:

- **Continuation / homotopy**: solve the coupled canopy-soil closure at a
  sequence of increasing radiative load (or decreasing aerodynamic resistance)
  starting from a regime where only the open branch exists, and continue to the
  full forcing -- so the solver tracks the open branch.
- **Open-branch bias with validation**: seed the coupled iteration from a
  transpiring state and confirm (against US-MMS / DE-Obe / US-Ton) that the
  working sites are unchanged, plus a synthetic dense-canopy/high-radiation
  bistability unit test.

Attempts made this session that did **not** work:

- Moistening the `q_c` initial guess (weight 0.5 -> 1.0): no effect -- the
  converged branch is init-independent.
- Increasing the Picard pass count (6 -> 20 -> 40): no effect -- already
  converged (confirms a stable fixed point, not slow convergence).
- Warm-starting / continuing the canopy Newton across Picard passes (each pass
  from the previous solution instead of a cold restart): no effect -- the first
  pass already lands on the closed branch, so continuation stays closed.

The fix is real numerical-methods work on the coupled closure (a genuine forcing
homotopy, or an instrumented match of the exact soil-thermal trajectory the
multilayer step feeds the canopy at a collapsed step) and is left as scoped
follow-up. The photosynthesis, radiative-transfer, and leaf-biophysics code is
NOT implicated.

## Reproduce

```
# full-run internals (leaf T, gs, canopy-air T, TgC), binned by shortwave:
JAX_ENABLE_X64=1 python scripts/tmp/_probe_fullrun_internals.py AU-Tum
# cold-start single-column bisection (state independence):
JAX_ENABLE_X64=1 python scripts/tmp/_probe_canopy_coupled.py AU-Tum
```
