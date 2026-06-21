# Super-Droplet Method (SDM) microphysics — oracle digest + port plan

Faithful JAX port of the **Super-Droplet Method** (Shima et al. 2009, QJRMS 135:1307-1320),
using ERF `Source/Microphysics/SuperDropletsMoist` + `Source/Particles/ERF_SuperDropletPC*`
+ `Source/MaterialProperties/*` (branch `development`) as the oracle. Switchable
microphysics scheme `scheme="sdm"`. Differentiability: not required (Monte-Carlo
coalescence is non-diff); condensation growth IS differentiable and kept so.

ERF is BSD-3 (open source); used as an algorithmic reference, independent JAX reimpl.

## Oracle file map (fetched to /tmp/erf_sdm during research; not committed)
- `ERF_SuperDropletsMoist{.H,Advance,PhaseChange,Init,Utils}.cpp` — Eulerian moisture-model wrapper (grid<->particle glue, phaseChange driver, q_t closure, latent heating).
- `ERF_SuperDropletPCDefinitions.H` — superdroplet SoA attributes; `SD_effective_radius`, `SD_total_mass`, `SD_dry_radius`; enum kernel/term-vel/integrator types.
- `ERF_SuperDropletPCMassChange.{H,cpp}` — diffusional growth ODE (`dRsqdt`) + RK3BS/RK4/BE/CN/DIRK2 integrators + Newton solver; coefficient setup.
- `ERF_SuperDropletPCCoalescence.{H,cpp}` — Shima MC collision-coalescence + kernels (Golovin/sedimentation/Long/Hall/Brownian).
- `ERF_SuperDropletPCAdvection.cpp` — interpolate flow vel, terminal velocity, sediment.
- `ERFPCParticleToMesh.H` — CIC particle->mesh deposition (mass density per cell volume).
- `ERF_MaterialProperties.{H,cpp}` — saturation funcs, latent heat, Rv, density, curvature `coeffCurv`, solute `coeffVPSolute`, diffusion `coeffMolecularDiffusion`.
- `ERF_TerminalVelocity.H` — RogersYau / AtlasUlbrich / CloudRainShima.
- `ERF_Constants.H`, `ERF_EOS.H` — constants + compressible EOS.

## Superdroplet representation (SoA pytree)
Per super-droplet i: `multiplicity` xi_i [-] (real, # real droplets represented),
`radius` R_i [m] (effective wet radius), water mass m_w_i, solute mass m_s_i (+ ion/MW),
insoluble mass m_p_i, cell/bin index, `active` flag. Velocity for hydrodynamic kernel.
- effective radius: m_t = m_w + m_s + (rho_w/rho_p) m_p ; R_eff = cbrt(m_t/(4/3 pi rho_w)).
- water solute moles: N_s = sum_j m_solute_j * ion_j / MW_j (van't Hoff).

## (1) Condensation / evaporation — diffusional growth (per droplet)
Integrate u = R^2 with (ERF `dRsqdt::rhs_func`, exact):
```
lambda_v = 2 D / sqrt(8 T R_v / pi)          # vapor mean free path
Kn   = lambda_v / R                          # Knudsen number
dcf  = (1 + Kn) / (1 + 2 Kn (1 + Kn))        # Fukuta-Walter transition correction
F_k  = (L/(R_v T) - 1) * (L rho_l)/(K T)     # heat-conduction term
F_d  = (rho_l R_v T)/(dcf D e_s)             # vapor-diffusion term (Kn-corrected D)
d(R^2)/dt = 2(S-1)/(F_k+F_d)                 # Maxwell (alpha)
          - 2 (a/T)/(F_k+F_d) * (1/R)        # Kelvin curvature (beta), a = 2 sigma/(R_v rho_l)
          + 2 b N_s/(F_k+F_d) * (1/R^3)      # Raoult solute  (gamma), b = coeffVPSolute
```
Jacobian (for implicit): d/d(R^2) of beta/R + gamma/R^3 terms = `dRsqdt::rhs_jac`.
Coefficients (water): L=L_v=2.5e6, K=therco=2.40e-2 W/m/K, R_v=461.505, rho_l=rho_w=1000,
D=diffelq=2.21e-5 m^2/s (const for water), sigma=0.076148325 N/m =>
a=2 sigma/(R_v rho_w)=3.300e-7 [m K] (curvature term uses a/T), b=4.3e-6 [m^3/mol] (solute).
S = saturation ratio = e/e_sat (vapor-pressure based; e = p·r/(eps+r) from the vapor
mixing ratio r — see thermo.relative_humidity, NOT the mixing-ratio ratio q_v/q_sat).
For pure water droplets w/o aerosol the
curvature+solute terms are negligible at R>~1um => classic Maxwell: d(R^2)/dt=2(S-1)/(F_k+F_d).
Integrators: adaptive-substep with dt=cfl/|tau|, tau=rhs_jac; default explicit (RK4/RK3BS),
implicit BE/CN/DIRK2 via Newton for stiff (small R). Port: sub-stepped RK4 + analytic R^2 form.
After: d_mass = 4/3 pi rho_l (R_new^3 - R_old^3); m_w += d_mass, clamp >=0.

## Grid coupling (phase-change back-reaction; total water conserved)
- Deposit particle water mass -> grid liquid density; R<r_rain(=40um)->q_c, R>=r_rain->q_r; /rho_air.
- q_t fixed during phase change: q_v_new = q_t - q_c - q_r (vapor = residual).
- Latent heating: theta += (theta/T) (L/Cp) (q_v_old - q_v_new). [Cp=Cp_d dry]
- Recompute T,p from theta (EOS), recompute S; iterate `num_substeps_phase_change`.

## (2) Collision-coalescence — Shima 2009 Monte-Carlo (per cell/bin)
For each cell with n super-droplets:
1. random shuffle of the n SD indices.
2. linear sampling: floor(n/2) candidate non-overlapping pairs (inds[p], inds[n-1-p]).
3. per pair order by multiplicity: i = larger xi, j = smaller xi (xi_i >= xi_j).
4. kernel K(R_i,R_j) [m^3/s] (choice): Golovin b(X_i+X_j),b=1.5e3,X=4/3pi R^3;
   hydrodynamic E*pi(R_i+R_j)^2 |v_i-v_j| with E from Long/Hall/sedimentation; +Brownian opt.
5. scaled probability (Shima):
   ```
   prob_ij     = K / V_bin                       # V_bin = cell volume / n_bins
   prob_sd_ij  = max(xi_i, xi_j) * prob_ij
   scaling     = 0.5 n (n-1) / floor(n/2)         # candidate->all-pairs correction
   P           = prob_sd_ij * scaling * dt
   ```
6. stochastic integer count gamma = floor(P) + (1 if U(0,1) < P-floor(P) else 0);
   gamma = min(gamma, floor(xi_i/xi_j)); remainder = xi_i - gamma*xi_j.
7. multiplicity update (xi_i>=xi_j; "donor"=larger xi loses count, "grower"=smaller xi grows):
   - remainder>0: xi_i -= gamma*xi_j ; R_j=cbrt(gamma R_i^3 + R_j^3); m_j += gamma*m_i (per species).
   - remainder==0 (xi_i==gamma*xi_j): split equal — dm=floor(xi_j/2); xi_i=dm; xi_j-=dm;
     R_i=R_j=cbrt(gamma R_i^3 + R_j^3); masses equalized.
Invariant: total real-droplet water mass sum_i xi_i*m_i conserved by coalescence.

## (3) Terminal velocity
- RogersYau:    v = k1 R^2, k1=1.233e8 (Stokes, cloud).
- AtlasUlbrich: v = 3.778 D^0.67 (D in mm) (rain).
- CloudRainShima: SCALE-SDM piecewise (Stokes+slip / Beard Re(Nda) / Bond-number large rain), cgs.

## (4) Sedimentation/advection
z += dt (w_flow - v_terminal). Box/parcel model: closed (no sedimentation) for Golovin/condensation tests.

## Time-step order (ERF Advance)
inject -> phaseChange(substeps) -> advect+sediment -> coalescence -> recycle.

## legoESM port design
New sub-package `packages/atmosphere/legoesm/atmosphere/physics/microphysics/sdm/`:
- `particles.py`  — `SuperDropletState` SoA pytree + constructors + derived (R_eff, total mass).
- `condensation.py` — growth law + sub-stepped integrator (vmap over particles). Differentiable.
- `kernels.py`   — Golovin + hydrodynamic kernels + terminal velocities.
- `coalescence.py` — Shima MC (jax.random key), multiplicity update. Non-diff (documented).
- `coupling.py`  — particle<->grid deposition, latent heating, q_t closure.
- `config.py`    — `SDMConfig` NamedTuple (kernel, term-vel, r_rain, n_substeps, integrator,
                   solver tols, surface tension/curvature/solute coeffs, init dist params). All
                   tunables here (audit: no magic numbers in hot loops). Constants via
                   `legoesm.constants`; saturation via `legoesm.thermo` (audit-mandated, NOT ERF Tetens).
- `box_model.py` — persistent-particle drivers (lax.scan): `run_box` composes
  condensation + Shima coalescence (ERF order) in a well-mixed box; `run_parcel`
  is the adiabatic activation parcel. Natural Lagrangian SDM home + test cases.
- `__init__.py`  — exports `sdm_microphysics` (column operator) + box/parcel drivers.
Each physics leaf (`condensation`, `kernels`, `coalescence`, `coupling`, `column`)
carries `__physics_contract__`; `particles`/`config`/`box_model` are plumbing/
drivers (EXCLUDED from the contract gate).

### Switchable wiring
- `MicrophysicsConfig`: add `scheme` literal `"sdm"` + `sdm: SDMConfig = SDMConfig()`.
- `integration.py` factory `_get_microphysics_fn`: add `"sdm" -> sdm_microphysics, config.sdm`.
- plane `_PLANE_MIN_TRACER_SLOTS["sdm"]` + tracer-slot map.
- `ExperimentConfig.validate_strict` membership set + `test_validate_strict_coverage`/`test_dispatch_hardening` baselines.

### Column-interface limitation (be explicit — "no laziness")
legoESM `micro_fn(T,q_v,hydrometeors,p_full,p_half,rho,dz,dt,cfg)->MicrophysicsOutput` is
STATELESS per call; SDM is Lagrangian w/ persistent particles. Column operator runs a
single-step reconstruct(grid q_c/q_r/N_c -> ephemeral SD population)->evolve 1 dt->deposit
tendency. True multi-step Lagrangian persistence lives in `box_model.py` (and a future
particle-state-threading interface). Documented, not hidden.

## Validation (simple cases)
- `test_sdm_condensation.py`: fixed-S Maxwell growth — R^2 grows linearly d(R^2)/dt=2(S-1)/(F_k+F_d);
  Kelvin/Raoult equilibrium (Kohler) radius at S slightly <1. Analytic.
- `test_sdm_golovin.py`: collision-coalescence box vs analytic Golovin solution (Shima 2009 Fig 4);
  mass-density spectrum moments converge; total water mass sum(xi*m) conserved.
- per-leaf unit tests (tendencies directly).
- conservation: water mass (vapor+liquid) under condensation; sum(xi*m) under coalescence.

## Status (2026-06-11)
Iterations 1-12 complete on branch `feat/sdm-microphysics`, each codex-reviewed
+ hardened. Modules: `particles`, `condensation`, `kernels`, `coalescence`,
`coupling`, `box_model`, `column`, `sedimentation`.
- **All 5 oracle collision kernels**: Golovin, sedimentation, Long, Hall (full
  21×15 table, bilinear, cap-only-on-large-branch), Brownian (Seinfeld-Pandis
  Fuchs form, ERF-faithful ADDITIVE `include_brownian` wiring).
- **Sedimentation + surface rain accumulation** (`sedimentation.py`): terminal-
  velocity fall, crossing deposits ξm → precip [kg/m²], exact conservation,
  `column_rainout` driver.
- **Condensation integrator family** (`condensation_integrator`): fixed-substep
  `rk4`/`euler` (reverse-mode differentiable, default), ERF adaptive
  stiffness-based `rk4_adaptive` (dt=cfl/|τ| via the oracle's approximate
  `drsq_dt_jac`, accepted-step-only cap, partial-on-cap = ERF semantics), and
  implicit `be` (ERF NewtonSolver + TI::be, unconditionally stable for stiff
  Köhler haze). Adaptive family is jit-only (not reverse-diff).
Follow-up branch `feat/sdm-followups` (iters 13-15, codex-clean): ERF
constant-multiplicity initialization spectra (`init.py`: exponential-mass +
truncated-lognormal-radius, exact erfinv; dedups the validation samplers),
particle recycling (`recycling.py`: inactive -> fresh dry aerosol with the
exact 1e-15 m water seed), and the CN/DIRK2 implicit integrators — the ERF
`SDMassChangeTIMethod` enum is now COMPLETE (rk4/euler fixed-differentiable +
rk4_adaptive/be/cn/dirk2 on one shared adaptive outer loop).
Remaining oracle gaps (future work): resolved-flow particle advection (needs
the Eulerian wind-field coupling architecture); the ERF 'sampled'
importance-multiplicity initialization mode. Validations passing:
- **Golovin box (collision)**: ensemble number decay matches analytic
  `N(t)=N0 exp(-(b/ρ_w)Lt)` to 0.36% and 2nd mass moment to 1.9%; `Σξm` conserved.
- **Adiabatic parcel (condensation/activation)**: supersaturation peaks (~1.018)
  then relaxes; droplets 1->9.5 um; LWC 0->0.16 g/kg; total water conserved ~1e-12;
  moist adiabat.
- **Column operator**: condensation/evaporation signs, total-water + latent-heat
  consistency, donor-clamp positivity, S=1 round-trip, factory dispatch.
`scheme="sdm"` is selectable, strict-validated, dispatched, and driver-buildable.

### Cross-validation vs PySDM (`scripts/validate/validate_sdm_vs_pysdm.py`)
PySDM (https://open-atmos.github.io/PySDM/, open-atmos — the reference
open-source SDM, Shima lineage) run as an INDEPENDENT second oracle on the
canonical Shima-2009 Golovin box (N0=2^23 m^-3, X̄=1.19e5 µm³, b=1500/s,
2^15 super-droplets, 3600 s): **number density agrees to 0.0-1.6%** across
three decades of decay, M2 to 0.6-7.4% (MC-tail noise), mass spectra to
0.5-12% L1, LWC conserved by both (0.2% apart from sampling). Two-process
harness (PySDM runs in its own venv, `pysdm_golovin_reference.py` dumps .npz;
the repo-venv validator compares) so PySDM's numba stack never touches the
jax env. Strongest consistency evidence: two independent Monte-Carlo
implementations of the same algorithm agreeing within ensemble noise.

### Activation-parcel cross-validation vs PySDM (`validate_sdm_parcel_vs_pysdm.py`)
Second independent-oracle case: Arabas & Shima (2017)-style monodisperse
ammonium-sulfate parcel, PySDM (κ-Köhler κ=0.72, implicit condensation) vs
legoESM (ideal van't Hoff i=3, explicit `rk4_adaptive`): **S_max−1 within 2.0%,
peak timing 1.0%, final radius 1.2%, LWC 3.7%, T within 0.05 K; identical
0.115 µm haze equilibrium** — the activation chain agrees across two
independent Köhler forms, saturation curves, and integrators.

### Smoke / oracle-consistency validation (`scripts/validate/validate_sdm_smoke.py`)
End-to-end ALL-PASS (stable, physically realistic, oracle-consistent):
- **Golovin collision box vs analytic Scott (1968)** — the exact benchmark ERF /
  Shima (2009 Fig. 4) validate against: the super-droplet **number decay matches
  `N(t)=N0 exp(-b/ρ_w·L·t)` to 1.2-4.5%** over τ=b·L·t = 0.5..3, mass conserved to
  1e-16. (2nd mass moment matches at moderate τ; its giant-drop tail is
  under-sampled by a finite super-droplet count past τ~1 — a known SDM
  convergence property, diagnostic-only there.) Running the ERF C++/AMReX oracle
  is unnecessary: its published acceptance test IS this analytic comparison, and
  the per-formula numerics were matched term-by-term in the unit tests.
- **Warm-rain box (`run_box`, condensation + collision)** — rain forms
  (q_rain 0 -> 1.27 g/kg), N 1e8 -> 1e4 /m^3, droplets 18 -> 825 um, water
  conserved; autoconversion emerges from the resolved Long-kernel collisions.
- **Adiabatic parcel** — realistic activation: peak supersaturation +0.44%,
  droplets 8 -> 18.6 um, LWC 1.21 g/kg, total water conserved to ~1e-15, moist
  adiabat. (A physical caveat learned here: pure-water droplets started
  *subsaturated* evaporate to the radius floor where the curvature term traps
  them — start at cloud base, or carry aerosol/solute, which real SDM does.)

### Column-adapter scope + exposure (codex iter-5)
The stateless column path (`sdm/column.py`) does **diffusional condensation/
evaporation only** on a mean droplet reconstructed from `q_c` + prescribed `cdnc`
(donor-clamped, water-conserving). Thin-cloud closure (fresh-review fix): when
the fixed-cdnc inversion would give a sub-`r_min_reconstruct` (1 um) droplet,
the closure holds the droplet at `r_min_reconstruct` and reduces the effective
number (`N_eff = q_c·ρ/m(r_min)`), so no nm-scale Kelvin-barrier artifacts and
the tendency is continuous in `q_c` (`dq_c ∝ q_c` for thin cloud). Collision-coalescence (`coalescence.py`) and
aerosol activation need a persistent droplet population: the composed persistent
path is `box_model.run_box` (condensation + Shima coalescence); sedimentation/
precip of the particles is not yet implemented. None of these run in the column
op. Consequently `scheme="sdm"` is registered + strict-valid but is
**intentionally not exposed in the AMIP CLI** (`scripts/run/run_amip.py` choices)
— same treatment as `p3`/`ml_emulator` — because a condensation-only scheme is
not a complete precipitating microphysics for a full climate run.

## Faithfulness caveats (intentional deviations from the ERF oracle)
- **Constants follow legoESM canonical values, not ERF's** (repo audit forbids
  hardcoded/duplicated physical constants): `R_v=461.51` (ERF 461.5),
  `L_v=2.501e6` (ERF 2.5e6), saturation via `thermo.saturation_vapor_pressure`
  Tetens curve (ERF uses SAM `erf_esatw`). Differences <0.04%; the *method* is
  faithful. This is a deliberate WONTFIX (codex iter-1 finding #3).
- **Condensation integrator is fixed-substep RK4/Euler**; ERF's adaptive
  stiffness-based dt with step-halving + Newton-implicit (BE/CN/DIRK2) options
  are the planned refinement. Fixed substeps stiffen as R->dry-radius; the
  `_R_SQ_FLOOR` keeps it finite (codex iter-1 finding #5).
- **`cfg` is a static argument** (carries str/bool fields) — jit via closure or
  `static_argnames`, as every legoESM scheme does (codex iter-1 finding #1).

## Iteration plan
1. [this iter] oracle digest + `particles.py` + `condensation.py` + `config.py` skeleton + Maxwell-growth test + contracts. codex review. commit.
2. kernels + terminal velocity + tests. codex. commit.
3. Shima MC coalescence + Golovin box test (analytic). codex. commit.
4. coupling (deposition/latent heat/q_t) + box_model driver + parcel test. codex. commit.
5. switchable wiring (config/factory/validate_strict/slot map) + column operator + integration test + matrix entry. codex. commit.
6. conservation + differentiability (condensation grad) checks; docs. codex. commit.
```
