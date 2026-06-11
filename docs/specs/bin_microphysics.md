# Fast Spectral-Bin Microphysics (FSBM-2) — faithful port spec

**Oracle**: WRF `phys/module_mp_fast_sbm.F` (FSBM-2, Hebrew University Cloud
Model; Khain et al. 2004 JAS 61:2963; Shpund et al. 2019 JGR 124:9800). Local
copy: `/tmp/wrf_sbm/module_mp_fast_sbm.F` (9053 lines, not committed).

**Goal**: switchable `MicrophysicsConfig.scheme="fast_sbm"` Eulerian bin
scheme, differentiable end-to-end (`jax.grad`), reusing repo helpers
(`legoesm.thermo`, `constants`, `microphysics.output.sedimentation_tendency`,
SDM kernels where physics overlaps).

## Oracle structure (what we are porting)

Four size distributions on mass-doubling bins (`m_{k+1}=2m_k`):

| Distribution | Bins | Notes |
|---|---|---|
| Liquid drops (cloud+rain) | NKR=33 | single liquid spectrum; KRDROP=15 (~50 um) cloud/rain split for diagnostics |
| Snow/ice crystals | 33 | with rime fraction `RF3R` |
| Graupel **or** hail | 33 | `hail_opt=1` default (hail fall speeds) |
| Aerosol (CCN) | NKR_aerosol=43 | 3-lognormal init (nucleation/accumulation/coarse) |

Key oracle constants: `COL=0.23105=ln2/3` (log-radius increment), quadrature
width `dm=3·COL·x=ln2·x`, moments `QC=(1/ρ)Σ3·COL·f·x²`, `QNC=(1/ρ)Σ3·COL·f·x`
(FAST_SBM lines 4993–4996). CGS internally; our port is SI with CGS only in
oracle-diff tests.

Process inventory (oracle subroutine → port module → status):

| Process | Oracle | Port | Status |
|---|---|---|---|
| Bin grid + moments | parameter block, QC/QNC diags | `fast_sbm/grid.py` | **done (iter 1)** |
| Collision kernels | `Kernals_KS` (l. 6238) — NOTE: only pressure-interpolates file-read `YW*` tables | computed kernels (reuse `sdm/kernels.py` Golovin/Hall/Long) + optional table load | Golovin **done (iter 2)**; Hall/efficiency todo |
| Collision-coalescence (Bott flux) | `coll_xxx_lwf` + `courant_bott_KS` | `fast_sbm/collision.py` | **liquid self-collection done (iter 2)**; LWF variant (snow), xyx/xyz cross-species todo |
| Diffusional growth (cond/evap dep/sub) | `JERRATE_KS`→`JERTIMESC_KS`→`JERSUPSAT_KS`→`JERDFUN_KS`/`JERNEWF_KS`→`ONECOND1` | `diffusional_growth.py` + `supersaturation.py` + `remap.py` + `condensation_driver.py` | **warm chain + ONECOND1 driver done (iters 3-6)** |
| Drop nucleation (CCN activation) | `JERNUCL01_KS`, `WATER_NUCLEATION`, `LogNormal_modes_Aerosol` | `fast_sbm/nucleation.py` | todo |
| Freezing/melting | `FREEZ`, melting block in FAST_SBM | `fast_sbm/ice_phase.py` | todo |
| Breakup (collisional + spontaneous) | `coll_breakup_KS`, `Spont_Rain_BreakUp` | `fast_sbm/breakup.py` | todo |
| Sedimentation per bin | fall-speed tables `VR1..VR5` + advection in FAST_SBM | `fast_sbm/sedimentation.py` (reuse `output.sedimentation_tendency`) | todo |
| Column driver + scheme wiring | `FAST_SBM` subroutine | `fast_sbm/column.py` + `integration.py` dispatch | **switchable `scheme="fast_sbm"` done (iter 7)** — stateless adapter (reconstruct→evolve→project); per-bin prognostic tracers later |

**Lookup-table strategy**: WRF reads tables (`capacity33.asc`, masses,
terminal velocities, kernels `YW*`, breakup `PKIJ/QKJ`) from data files NOT in
the WRF git repo. Faithfulness plan: (a) grid masses derived analytically
(doubling from 2 um — reproduces documented landmarks), (b) collision kernels
via in-code `Kernals_KS` path, (c) terminal velocities from published
formulations already in `sdm/kernels.py` where identical, else ported, (d)
capacities analytic (sphere/oblate per category). Any residual table needed →
small committed `.npy` with provenance, like RRTMGP.

**Differentiability strategy**: oracle's remap (`JERNEWF_KS`) and Bott scheme
use integer bin shifts + positivity clamps — port keeps the *algorithm* but
implements branches as `jnp.where` with smooth-safe formulations; hard
`wrf_error_fatal` guards become finite clamps documented per-site. No
`lax.cond` on traced data for bin loops — vectorized over bins.

**Physics contracts**: every tendency-producing module ships
`__physics_contract__` + acceptance test before the body (CLAUDE.md
guardrails). Helpers (`grid.py`) live in `test_physics_contracts.py`
EXCLUDED.

## Iteration log

### Iter 1 (2026-06-11)
- Fetched oracle (9053 l). Mapped subsystem: dispatch in
  `microphysics/integration.py` (hardened ValueError), SDM = subpackage
  precedent, `MicrophysicsOutput` union container.
- Extracted oracle conventions: NKR=33/43, COL=ln2/3, dm=3·COL·x, f in
  cm⁻³g⁻¹ (→ SI m⁻³kg⁻¹), QC/QNC quadrature, KRDROP=15.
- **`fast_sbm/grid.py`**: `mass_doubling_grid` (bitwise-exact doubling),
  `radius_from_mass`, `bin_mass_widths` (=ln2·m exact), `number_density`,
  `mass_density`, `discretize_lognormal` (exact CDF-difference number
  projection; promoted `sdm.init.lognormal_cdf` to public for reuse).
- Tests `tests/atmosphere/microphysics/unit/test_fast_sbm_grid.py`: oracle
  landmarks (2 um, bin15≈50 um, top≈3.25 mm), quadrature == oracle formula,
  lognormal number exact + mass vs analytic 3rd moment (midpoint bias
  +ln²2/24 = +2.0% bracketed explicitly), truncation bounds, `jax.grad`
  correctness (linear-moment gradient == m·dm), jit + float32, ValueError
  guard.
- Tracer-state converters `f_from_bin_mixing_ratios`/`bin_mixing_ratios_from_f`
  (oracle carries per-bin mixing ratios ``chem_new``; ``f = q ρ_air/(m dm)``
  = its ``ρ/(3 COL x²)``) + round-trip/QC-sum test. (Surfaced by the codex
  pass before it hung; review re-run scheduled with iter 2.)
- Codex adversarial review attempt 1 hung after ~40 min (no log progress);
  cancelled, findings up to hang folded in; full review re-runs at iter 2.

### Iter 2 (2026-06-11)
- **`fast_sbm/collision.py`** — Bott flux collision-coalescence port:
  `precompute_collision_tables` (oracle `courant_bott_KS`; generalized
  Courant `ln(x0/m_{k-1})/ln(m_k/m_{k-1})` so refinement studies reuse the
  solver; ints/floats static NumPy), `bott_coalescence` (oracle
  `coll_xxx_lwf` at `fl≡1` — mass-only liquid self-collection; exact
  statement-order replication incl. `j==i`/`k==j` aliasing, salvage branch,
  gmin floors; `lax.scan` over 561 pairs in oracle order), `g↔f`
  converters, `collision_ck_matrix` (= `K·dt·dlnr`, oracle `Kernals_KS`
  contract). Ships `__physics_contract__`.
- `discretize_exponential` added to grid (Golovin init).
- Kernel reuse: `sdm.kernels.golovin_kernel` (no re-derivation).
- Validation (`test_fast_sbm_collision.py`, 8 tests): Courant-table
  structure on the doubling grid (self-pairs → k=j+1 @ c=0; mixed → k=j,
  c=ln(1+m_i/m_j)/ln2 exact), empty-spectrum fixed point, single-bin
  Gauss-Seidel cascade conservation, mass conservation 1e-13 over 20 steps
  + monotone number decay, Golovin box vs analytic moment laws
  (N: −7% dt-converged spatial bias at NKR=33, M2: +22% broadening
  overshoot — both bracketed + signed), **grid-refinement convergence**
  (halving dlnm cuts N error ≥25%), `jax.grad` through the scan (dN/db<0
  finite), f/g moment consistency.
- Notable: WRF kernels are file-read tables (`YWLL_*` etc.), NOT computed
  in-code — `Kernals_KS` only pressure-interpolates them. Port strategy
  updated: computed kernels (reuse SDM Hall/Long/Golovin) as default;
  optional oracle-table loader later for bit-level fidelity.
- Codex adversarial review (iters 1-2): verdict FAIL → all fixed in
  `eb2f4c51`: CRITICAL f32 0/0 NaN at Bott flux `x1→0` (analytic limit
  `flux→gsk·c` + f32 value/grad regression test), refined-grid top-pair
  omission documented (oracle never assigns), scalar `rho_air` accepted,
  contract text overstatement fixed. Transcription itself verified faithful
  (statement order, index translation, clamps, salvage, aliasing).

### Iter 3 (2026-06-11)
- **`fast_sbm/diffusional_growth.py`** — oracle `JERRATE_KS`/`JERTIMESC_KS`:
  `vapor_diffusivity` (D_ref·(p₀/p)(T/T₀)^1.94), `ventilation_factor`
  (PK: Re=2rV/ν via oracle's (m/ρ)^⅓ form, X=√Re·Sc^⅓, branch at Re=2.5
  kept faithfully discontinuous, cap 5.0), `drop_growth_coefficient`
  (B = 4πC·f_vent/(F_D+F_K), capacitance=r for drops),
  `supersat_relaxation_integral` (SFN = Σf·B·dm/ρ_air). Ships contract.
- **`fast_sbm/config.py`** — `FastSBMConfig` (oracle reference coefficients
  D=0.211 cm²/s, ν=0.13 cm²/s, exponent 1.94, VENTPL_MAX=5 — kept as scheme
  params since repo constants use different reference states).
- Saturation from `legoesm.thermo` (NOT a POLYSVP port — repo single-curve
  rule; <0.5% difference, model consistency wins).
- Validation (6 tests): **cross-implementation pin** — ventilation-off B
  reproduces `sdm.condensation.drsq_dt` Maxwell core to 2e-3 (Knudsen
  residual) with matched D; diffusivity reference-state collapse;
  ventilation limits (f(V=0)=1 exactly, cap, monotone); B>0 monotone in
  size; SFN linearity + explicit formula + physical window; T/p gradients
  finite.

### Iter 4 (2026-06-11)
- **`fast_sbm/supersaturation.py`** — oracle `JERSUPSAT_KS` warm branch:
  exact linear-ODE step `dS/dt=−R·S+F` → (`S_new`, `S_int=∫S dt`, the
  driver of bin growth `Δm=B·S_int`). Single `expm1` formulation replaces
  the oracle's |R·dt|≶1e-6 branch pair (their Taylor EXPM1 = workaround for
  Fortran exp precision; `jnp.expm1` is that limit exactly) — analytically
  identical, cancellation-free, smooth gradients incl. R=0.
  `supersat_relaxation_rate` = oracle `RW=(OPER2+B5L·AL1)·DOPL·SFN` with
  L_v/R_v and L_v/c_pd derived from constants (oracle hardcodes 5.42e3/2500
  roundings — ≤0.5% documented deviation). Ships contract.
- Validation (7 tests): verbatim oracle large-x branch 1e-13; verbatim
  small-x Taylor branch (tolerance 1e-8 = the ORACLE's own catastrophic
  cancellation noise, port is cancellation-free); ballistic R=0 branch
  exact; equilibrium F/R + pure decay S_int→S0/R; independent RK2 ODE
  solve 1e-7; RW vs oracle rounded constants (7e-3) and vs derived
  constants (1e-14); grads finite across R=0/1e-13/1e-6/2 incl. batched.
- Codex re-review (fixes + iter 3): CONDITIONAL PASS → conditions fixed in
  `72dfd61a` (saturation-curve deviation corrected to verified ±3.4% at
  243/310 K extremes — was understated; L_v/p_atm_std deltas + Re=2.5
  ventilation jump documented). Flux-limit fix verified sound (no
  where-trap; 1e-6 gate appropriate for f32).

### Iter 5 (2026-06-11)
- **`fast_sbm/remap.py`** — oracle `JERDFUN_KS`/`JERNEWF_KS`:
  `condensation_new_masses` (exact m^{2/3} growth update, oracle floor),
  `remap_spectrum` = Kovetz–Olund 2-point packet split (ψ=f·m; conserves
  Σψ AND Σψ·m exactly; below-grid evaporation loss; 1024·m_top sentinel),
  3-point anti-diffusive correction (smoothing criteria with Fortran
  operator precedence, positivity guards, EXIT-kills-pass semantics via
  scan carry flag), drop-tail merge (window bins 6–12 1-based,
  COEFF_REMAPING=1/150, KMAX edge search, right-to-left cascade,
  unrolled static window). Evaporation disables 3-point + merge (oracle
  IEvap/IDROP). Negative-ψ: oracle hard-stops; port returns `min_psi`
  diagnostic (no silent clamp). Ships contract.
- Validation (8 tests): identity exact; doubling → exact one-bin shift;
  KO conservation (number 1e-12; post-remap mass == post-GROWTH mass,
  both ±3-point); evaporation monotone-number/no-negatives; growth-law
  limits (s_int=0 → 2-ulp identity inside remap shortcut; overshoot
  floor); tail-merge folds spurious tail + window mass conserved;
  d(mass)/d(S_int) finite positive THROUGH the remap; jit+vmap columns
  with s_int sign mix.

### Iter 6 (2026-06-11)
- **`fast_sbm/condensation_driver.py`** — oracle `ONECOND1` assembled:
  S from `thermo.relative_humidity`, B/SFN/R chain, JERSUPSAT step with
  in-step forcing 0 (oracle passes `DYN1=0`), growth+remap, exact closure
  `q−=Δq_c`, `T+=(L_v/c_pd)Δq_c`. Key oracle finding: this version's
  substep limiter is vestigial (`DTNEWL=min(DT,TIMEREV)` → ONE pass) and
  the final `JERDFUN_NEW(SUPINTW)` remap of the original spectrum then
  coincides with the in-loop remap — port implements the single-pass form
  directly (documented; wrap in `lax.scan` if a future oracle re-enables
  `DT_WATER_COND`). Ships contract (conserves moisture+energy).
- **Simple case passes**: supersaturated parcel (S=2%, 100 cm⁻³ @ 10 µm)
  condenses — total water invariant 1e-13, `c_pd ΔT = −L_v Δq_v` 1e-12,
  S decays toward equilibrium without overshoot; RH 90% parcel evaporates
  (number non-increasing); saturated+empty spectrum = exact fixed point;
  40-step scan: |S| monotone ↓, ends <10% of initial; d(dq_c)/d(T,q)
  finite, d(dq_c)/dq > 0. 47 fast_sbm tests green total.
- Codex review (iters 4-5) must-fixes, all applied: (1) KO + 3-point
  exact-match is an OVERWRITE in the oracle (not add) — KO now a
  sequential scan replicating assignment-then-add ordering; (2) boundary
  hits `m_new == grid node` now take the LOWER interval
  (`searchsorted side='left'` = oracle first-match search; the doubling
  grid hits this exactly); (3) small-x guard now uses series for
  `em/R`,`tail` so d/dR at R→0 is the analytic derivative (was zeroed);
  (4) `COEFF_REMAPING` = oracle literal 0.0066667 (not exact 1/150);
  (5) sentinel-overflow doc fixed (fraction LOST, not kept); (6) contracts
  note piecewise gradients (remap) / Re=2.5 jump; (7) RW constants
  decision recorded: derived `L_v/R_v`, `L_v/c_pd`, `epsilon` from
  `legoesm.constants` (≤0.5% from oracle roundings) — physics-faithful
  over rounding-faithful, consistently with the saturation-curve policy.
  New edge tests: mixed exact/non-exact overwrite, boundary equality,
  negative new mass, sentinel loss fraction, empty merge window,
  evaporation-disables-passes equivalence, dS/dR correctness at R=0
  (central-difference pinned). 54 fast_sbm tests green.

### Iter 7 (2026-06-11) — SWITCHABLE SCHEME
- **`fast_sbm/column.py`** — `scheme="fast_sbm"` live: stateless column
  adapter (SDM pattern, but resolving the FULL 33-bin spectrum): bulk
  (q_c, q_r, N_r) → mass-exact lognormal cloud + exponential rain modes →
  ONECOND1 condensation + Bott coalescence (computed Hall/Long kernels,
  `terminal_velocity_cloud_rain_shima` reused) → bulk tendencies split at
  KRDROP. **Autoconversion/accretion emerge from the resolved collection
  equation — no tuned rate anywhere.** Limitations documented: no
  activation (clear cell stays clear), precipitation=0 until
  sedimentation, warm-only. Contract ships.
- Wiring: `integration.py` dispatch + import, `MicrophysicsConfig.
  fast_sbm` field + docstring, `_PLANE_MIN_TRACER_SLOTS["fast_sbm"]=9`,
  `driver/config.py` `_valid_microphysics` += fast_sbm, package
  `__init__` exports (FastSBMConfig, fast_sbm_microphysics). Config grew
  adapter fields (cdnc, cloud_geom_std, n_rain_floor, collision_kernel,
  golovin_b); `_kernel_matrix` raises ValueError on unknown kernel.
- `mass_doubling_grid` got a NumPy twin (`mass_doubling_grid_np`) so
  collision-table precompute stays host-side under jit/grad traces.
- Validation (6 tests): dispatch via PUBLIC MicrophysicsConfig (+unknown
  scheme raises), unknown kernel raises, (ncol,nlev) condensation closure
  (dq_v=−dq_l, dT=(L/c_p)dq_l, ice identically 0), **emergent
  autoconversion** (dense cloud ≫ thin cloud rain production at S=0,
  liquid conserved by coalescence 5e-9), clear-cell fixed point, jit+grad
  through the full operator. Ratchets green (2078), microphysics tree 180,
  hydrostatic integration 68.
