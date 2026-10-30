# Consistent plant-hydraulic stress across the three arms (design)

Status: agreed design, not yet implemented (2026-10-30).  Written for whoever
implements it; read `docs/land/optimality_switches.md` first for the current
arm switches.

## Goal

Give all three arms (baseline, least_cost, phydro) the same short-term drought
physics, a root-xylem-leaf hydraulic model, with an equal number of tunable
parameters per arm, while keeping the classical empirical beta(theta) stress
available as a switch value.

## Decisions taken

1. **beta(theta) stays.** `transpiration_stress` gains a third value,
   `hydraulic`, next to `beta_theta` and `phydro`.  Under `beta_theta` nothing
   changes from today's code path (fStress on Vcmax and the slope, root_frac *
   beta_k uptake weights, 6-unknown closure).  The `optimality` wrapper sets the
   arm's default; `-o physics.transpiration_stress=beta_theta` runs the
   classical version of any arm with the same code.  The two stress modes never
   act in one run.
2. **Non-stomatal limitation inside FvCB is required (currently a bug).** No
   arm limits Vcmax/Jmax by leaf water potential: `psi50_jmax_mpa` and
   `ck_jmax` were removed from `PHydroConfig` in 5564c394d and were unused
   before that.  Restore them as `psi50_vcmax_mpa` / `ck_vcmax` and apply a
   Weibull factor W_V(psi_L) = 2^(-(psi_L/psi50_vcmax)^ck_vcmax) to Vcmax25
   (Jmax follows through rjv25) inside the per-leaf-class photosynthesis call
   of the canopy closure, in ALL arms under `hydraulic` and `phydro`.  psi_L is
   a Newton unknown, so the factor is part of the residual, not a
   pre-multiplier.
3. **Acclimation:** the least-cost P-model acclimation sees no soil state
   (atmosphere only); P-hydro's acclimated optimum uses running means of soil
   psi.  This is accepted as a difference between the hypotheses; no empirical
   soil factor is added to the P-model beta.
4. **One parameter set per arm for both leaf classes.** Sunlit and shaded
   leaves have their own psi_L and their own draw-down unknown, but share g1,
   psi50_g1 and gamma (shade parameters are poorly identifiable from fluxes).
5. **Canopy hold (done, 88e6c890d):** a column is held only on a non-finite
   state/response or an implausible answer (skin T outside 180-350 K, |H| or
   |LE| > 1500 W/m2), no longer on the closure's adjoint-grade convergence
   flag (which held ~17 % of land column-hours).  A held hour's water is still
   not applied (by choice); the `water_budget` tape shows it.
6. **gplant is a fixed shared constant** (4e-3 mol m-2 leaf s-1 MPa-1), not
   tuned: it trades off with psi50_xylem.

## Short-term physics under `transpiration_stress=hydraulic` (baseline, least_cost)

Per leaf class c in {sun, shade}, every hour, solved jointly with the existing
canopy energy balance (Tf_sun, Tf_sh, Ci_sun, Ci_sh, Tc, q_c):

* supply: E_c = integral_{psi_L,c}^{psi_s} gplant * P(psi) dpsi, with the xylem
  Weibull P (psi50_xylem, ck_xylem), exactly `supply_transpiration` in
  `land/phydro.py`; psi_L,c = psi_s - dpsi_c.
* demand: Medlyn gs_c = g0 + 1.6 (1 + g1 * W_g1(psi_L,c) / sqrt(D)) A_c / Ca,
  A_c from FvCB with Vcmax25 * W_V(psi_L,c).
* closure row for dpsi_c: LAI_c * E_c(dpsi_c) / D = gs_w,c (supply equals
  demand).  This replaces phydro's stationarity row; same two unknowns.
* converged (gs, An, psi_L) per class come out of the one Newton solve.

`phydro` keeps its profit stationarity row (A - gamma*dpsi^2) and gains W_V on
Vcmax as in decision 2.

## Leaf-class area and the root path (as implemented today)

* fSun = (1 - exp(-kb*CI*LAI)) / (kb*LAI), kb = 0.5/cos(SZA), CI clumping;
  LAI_sun = fSun*LAI, LAI_sh = (1-fSun)*LAI (`canopy/radiative_transfer.py`).
* Soil-to-root conductance per ground area from `soil_root_supply` (Bonan 2014
  A23-A25, 500 g/m2 fine roots by root fraction, layers in parallel) gives a
  conductance-weighted psi_s; the soil-to-root resistance is currently masked to
  zero (c55207318), so the two classes are independent given psi_s.
* Add a switch `leaf_coupling: independent | shared_root`.  `shared_root`:
  psi_root = psi_s - R_soil*(LAI_sun E_sun + LAI_sh E_sh), both classes hang
  off psi_root.  Default `independent` (today's behaviour).

## Tunable parameters (five per arm)

| slot | baseline | least_cost | phydro |
|---|---|---|---|
| 1 | Vcmax25,top | beta (unit-cost ratio, now 146) | beta (unit-cost ratio) |
| 2 | g1 | kphio (intrinsic quantum yield) | kphio |
| 3 | psi50 xylem | psi50 xylem | psi50 xylem |
| 4 | psi50 Vcmax | psi50 Vcmax | psi50 Vcmax |
| 5 | psi50 g1 | psi50 g1 | gamma (acclimation and both leaves) |

Shared and NOT tuned (one value for all arms): gplant, ck_xylem, ck_vcmax,
ck_g1, root_biomass, alpha_cost.

## Code layout

* **New `land/plant_hydraulics.py`** (split out of `phydro.py`, which keeps the
  profit optimum): `soil_root_supply` (moved), `supply_transpiration`,
  `weibull_vulnerability`, `LeafHydraulics`, `leaf_supply_gs_w`; new
  `leaf_psi(dpsi, psi_s, ...)`, `weibull_stress(psi_L, psi50, ck)`.  Config
  `PlantHydraulicsConfig`: gplant, psi50_xylem, ck_xylem, root_biomass,
  psi50_vcmax, ck_vcmax, psi50_g1, ck_g1, leaf_coupling.  `PHydroConfig` keeps
  alpha_cost and gamma_cost.
* **`canopy/solver.py`:** 8-unknown state whenever leaf hydraulics is on
  (`hydraulic` or `phydro`), not only for phydro.  New
  `_hydraulic_stress(x, bundle)` returning W_V and W_g1 per class from psi_L,
  applied in the photosynthesis call and the Medlyn slope.  Draw-down rows by
  stomatal model: `phydro` -> `instantaneous_foc`; `medlyn` -> supply-demand.
  The bundle carries the Weibull parameters as per-column arrays so the
  implicit adjoint differentiates with respect to them (gradient-based tuning).
* **`surface_scheme/two_leaf_canopy.py`:** `transpiration_stress=hydraulic` ->
  fStress_vcmax = 1 and slope stress = 1; build the hydraulic bundle for any
  stomatal model; seed dpsi from the previous step; tape psi_L,sun, psi_L,sh,
  W_V, W_g1.
* **`multilayer_land.py`:** compute and thread `PhydroSupply` whenever
  hydraulics is on; root-uptake weights from the supply's layer conductances
  under `hydraulic` and `phydro` (root_frac*beta_k only under `beta_theta`).
* **`p_model.py`:** beta and kphio as per-arm tunables (already config fields);
  skip the beta(theta)-stressed rjv25 path under `hydraulic`.
* **Switches (`config.py`, `lmip_config.py`):** two axes,
  `optimality in {none, least_cost, phydro}` and
  `transpiration_stress in {beta_theta, hydraulic, phydro}`; the wrapper
  defaults none/least_cost -> hydraulic, phydro -> phydro; either overridable.
  `submit_lmip_ginsburg.sh` arms unchanged otherwise.
* **Tuning interface:** an `ArmTunables` NamedTuple per arm with the five slots
  and a mapping to config fields, so a fitting script sees one shape for all
  arms.

## Tests to pin it

* psi50 -> -inf: the hydraulic baseline reproduces today's unstressed Medlyn
  column exactly.
* The supply-demand row is zero at convergence (each class).
* phydro with the W_V factor disabled (psi50_vcmax -> -inf) is bit-identical to
  today.
* W_V lowers GPP monotonically with drying in all three arms.
* psi_L,sun <= psi_L,sh at midday (sunlit leaves transpire more).
* `beta_theta` mode unchanged (existing tests).

## Suggested order

1. Hydraulics module + Medlyn supply-demand row (baseline arm).
2. W_V inside FvCB for all hydraulic modes, phydro included (decision 2).
3. least_cost under `hydraulic` (mostly config).
4. Switch axes and the `ArmTunables` interface.
5. Tuning: EC sites first, then a stratified column subset of the global grid
   from a common restart (the model is column-independent, so a subset is an
   exact sample), full-globe single years for finalists, 30-year chains last.
