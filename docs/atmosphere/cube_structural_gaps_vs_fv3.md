# Structural gaps: legoESM cube path vs the FV3 duo-grid oracle

External survey (2026-08-02) of what the cube path **does not have**, or has in
materially different form, walking FV3's `fv_dynamics` → `dyn_core` → `c_sw` /
`d_sw` / `nh_core` sequence. Motivated by a pattern the audit exposed: the cube
lanes pass their gates while two defects persist (a tuned-down but never
eliminated W2 imprint, and a dead Held-Suarez jet at ~13 m/s vs ~30 expected).
Gates green with physics wrong is the signature of **missing pieces**, not of
individual broken lines.

Legend: **M** missing integrated mechanism · **A** material approximation or
replacement · **C** different convention or staggering.

| FV3 phase | legoESM live path | Gap |
|---|---|---|
| Outer split dynamics (`fv_dynamics.F90:450,502`) | generic tendency + `dispatch_integrator(..., "ssp_rk3")` (`primitive_eq_cdgrid.py:1583,1644`) | **M/A** — FV3's staged forward–backward acoustic update is absent from the PE route |
| Prognostic wind placement (`fv_dynamics.F90:103`, `sw_core.F90:150`) | HS enters cell-centred; each public step is cc → corner → cc (`primitive_eq_cdgrid.py:315,2028`) | **C/M** — *the largest active representation mismatch*; lift and exit are four-point averages |
| `c_sw` C-grid half-step (`sw_core.F90:79,482,3361`) | local D→C interpolation + four-corner average (`primitive_eq_cdgrid.py:328,332`) | **M/A** — `dgrid_to_cgrid` is a two-point projection, not `d2a2c_vect` |
| `p_grad_c` / `geopk` (`dyn_core.F90:581,678,1591`) | cc geopotential + Arakawa–Lamb corner gradient (`primitive_eq_cdgrid.py:344,478`) | **A/M** — `use_fv3_lin_pgf` is explicitly inert under RK3 |
| `d_sw1…6` transport/momentum (`dyn_core.F90:891,1026,1172,1321`) | flux-form continuity with face-averaged `dp`, advective T/tracer, generic centre diffusion | **M/A** — the ports exist, but no PE caller composes the FV3 mass/KE/vorticity stages |
| Lagrangian vertical coordinate + remap (`fv_dynamics.F90:568`, `fv_mapz.F90:62`) | fixed sigma/hybrid layers, Eulerian upwind vertical advection | **M** — a PPM remap prototype exists but is explicitly not integrated (`_future/vertical_remap.py`) |
| Grid/metrics/duo exchange (`fv_arrays.F90:324`, `fv_duogrid.F90:85`) | HS defaults to equiangular/no-duogrid; ED+duo is an opt-in factory | **C/M** — FV3-native seam angles intentionally off for the legacy solver |
| Dissipation + KE→heat closure (`dyn_core.F90:817,1837`, `sw_core.F90:1948`) | generic `A_h`, hyperdiffusion, continuous div damping; FV3-like knobs default off | **A/M** — not "no damping"; placement, selectivity, time discretisation and energy pathway differ |
| Tracer PPM + flux capacitor | advective tendencies; flux-form moisture is first-order split | **M/A**, irrelevant to dry W2 and dry HS |
| `nh_core` Riemann / `update_dz` | cell-centred split-explicit compressible solver | **M/A**, irrelevant to both reported symptoms |

`d2a2c` and `c_sw` are **not absent from the repository** — both have bit-exact
oracle leaf tests (`test_fv3_native_d2a2c_duo.py:111`,
`test_fv3_native_c_sw_oracle.py:100`). The structural gap is that they are not
the **live multilevel PE update**.

## Impact ranking

| Gap | W2 imprint | dead jet |
|---|---:|---:|
| Native ED+duogrid geometry/halo/stagger bundle | **1** | 5 |
| cc↔corner projection; missing live `d2a2c_vect` | **2** | **1** |
| Full `c_sw`/`d_sw` forward–backward sequence | 3 | **2** |
| Lagrangian vertical coordinate/remap | irrelevant | **3** |
| Dissipation / KE→heat closure | 4 | 4 |
| Exact FV3 pressure/external-mode staging | 5 | 5 |
| Tracer PPM/remap · NH Riemann · FMS MPI | irrelevant | irrelevant |

**The first four gaps are not tunable.** No coefficient turns four-point
projections into a covariant D→A→C reconstruction, RK3 into the staged FB
sequence, fixed-coordinate advection into a Lagrangian remap, or
equiangular/no-duo into ED+duogrid. Relevant here: the recorded `A_h` factorial
**saturates near 13.8 m/s** rather than recovering a healthy jet
(`run_atmosphere_test_matrix.py:1108`) — dissipation tuning has already been
tried and cannot close it.

## Frozen-threshold checks (declared before running)

1. **HS outer-projection A/B — highest-value jet test.** C48 HS, 20 days, twice,
   identical forcing/seed/dt/coefficients. Arm A converts once with
   `hydrostatic_to_fv3` and *retains* `FV3HydrostaticState` between steps; arm B
   keeps the current public cc route. Metric: area- and pressure-weighted
   30–60° / 200–800 hPa EKE at day 20.
   **A/B ≥ 1.50 ⇒ projection materially suppresses eddies; ≤ 1.10 ⇒ deprioritise.**
2. **Frozen one-step native-core residual — highest-value W2 test.** Same native
   ED+duo C48 W2 state through one native six-face stage chain and one production
   `FV3Edge` step; compare increments excluding a two-cell panel-edge ring,
   `r = ‖ΔX_prod − ΔX_native‖ / ‖ΔX_native‖` for `delp`, `u`, `v`.
   **max(r) ≥ 0.25 ⇒ material interior disagreement; ≤ 0.05 ⇒ evidence against the
   horizontal core.** Use the six-face assembly, not the single-tile duo `d_sw`
   port (that one fails closed by design — the Fortran pipeline needs inter-panel
   flux averaging).
3. **Split by symptom.**
   - *W2*: one-day C48 legacy (equiangular/no-duo) vs ED+duo production A/B.
     **native/legacy ≤ 0.50 ⇒ material.** Not an ED-only result: the switch
     changes geometry *and* duo halos (`matrix:2760`).
   - *Jet*: column-only remap discriminator at HS days 2/5/10 — 100 columns of
     largest vertical mass flux, current Eulerian one-step vertical update vs
     `fv_mapz::remap_2d` on the same constructed interfaces.
     **RMS ≥ 0.20 of the current vertical-transport increment on two checkpoints
     ⇒ material.** Far cheaper than a full multilevel port.

## Consistency with earlier rounds

The cc↔corner projection ranking **#1 for the jet** is the same object as the
"persistent-D" arm proposed independently in the earlier ladder — two separate
analyses converging on the repeated D→C exit / C→D re-entry as the prime suspect.
That is corroboration, not confirmation: neither has been measured yet.

## MEASURED: the outer projection materially suppresses eddies (2026-08-02)

Frozen check #1 above has been run. **CONFIRMED**: removing the extra outer
cell-centre↔corner projection roughly doubles both eddy kinetic energy and the
eddy momentum flux at day 20.

C48 / L30 / dt=300 / 20 days, configuration mirrored to the matrix HS lane
(`hyperdiff = 1.0e16`, `div_damp = 1.5e7`, `A_h` from the matrix formula), both
arms identical except where the prognostic state lives between steps:

| arm | banded EKE (J/kg) | signed u′v′ (m²/s²) | max &#124;U&#124; (m/s) |
|---|---|---|---|
| A — persistent D | 5.195e-03 | **+4.173e-04** | 6.620 |
| B — cc round-trip (current) | 1.976e-03 | +2.208e-04 | 4.737 |

**EKE ratio A/B = 2.63**, against the pre-declared **≥ 1.50 = material**
(≤ 1.10 would have deprioritised it). A C12 smoke gave **3.24** — same direction,
independently. `u′v′` is poleward — the correct sign for an eddy-driven jet — and
**1.9× larger** in arm A, so this is not an EKE-only artefact: the flux that
actually drives the jet responds too. The instantaneous jet is 40% stronger.

Controls passed *before* the arms ran: the physical `U = 30cos φ` vector-lift
preflight including seams (2/2), and an orthonormality + rotation control
(`max|cos²+sin²−1| = 7.7e-08`, grid dtype float32, tol 1e-6). Earlier attempts of
this probe voided themselves three times on mis-set control tolerances and once
on the wrong configuration entirely — each void was the instrument working.

### What this does and does not establish

**CONFIRMED** — under this configuration and seed, at day 20, the extra outer
projection costs a factor ~2.6 in banded EKE and ~1.9 in `u′v′`.

**NOT established.** This is a **lower bound** on the projection's total cost:
`physics_fn` converts to cell-centre on every RK stage in *both* arms, so the
inner conversions are common-mode and unmeasured. Twenty days is not
Held-Suarez equilibrium, and arm A's 6.6 m/s is still far from the ~30 m/s
benchmark — the projection is **a** material suppressor, not demonstrably the
whole cause. One seed, one configuration.

### Consequence

This promotes the cc↔corner projection from "ranked #1 by inspection" to
**measured material**, and is the first quantitative attribution for the dead
jet. Keeping the prognostic state in D staggering between steps is therefore not
a fidelity nicety — it is worth a factor of ~2 in eddy activity here.
