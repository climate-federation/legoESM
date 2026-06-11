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
S = saturation ratio = q_v / q_sat (= RH). For pure water droplets w/o aerosol the
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
- `box_model.py` — persistent-particle box/parcel driver (lax.scan), natural SDM home + test case.
- `__init__.py`  — exports `sdm_microphysics` (column operator) + box driver.
Each physics leaf carries `__physics_contract__`.

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
