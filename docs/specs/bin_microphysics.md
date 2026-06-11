# Fast Spectral-Bin Microphysics (FSBM-2) — faithful port spec

**Oracle**: WRF `phys/module_mp_fast_sbm.F` (FSBM-2, Hebrew University Cloud
Model; Khain et al. 2004 JAS 61:2963; Shpund et al. 2019 JGR 124:9800). Local
copy: `/tmp/wrf_sbm/module_mp_fast_sbm.F` (9053 lines, not committed).

**Goal**: switchable `MicrophysicsConfig.scheme="fast_sbm"` Eulerian bin
scheme, differentiable end-to-end (`jax.grad`), reusing repo helpers
(`legoesm.thermo`, `constants`, `microphysics.output.sedimentation_tendency`,
SDM kernels/velocities where physics overlaps).

## Oracle structure

Four size distributions on mass-doubling bins (`m_{k+1}=2m_k`): liquid drops
(NKR=33, KRDROP=15 ≈50 µm cloud/rain split), snow (33, rime `RF3R`),
graupel/hail (33), aerosol/CCN (43, 3-lognormal). Key constants:
`COL=0.23105=ln2/3`, quadrature `dm=3·COL·x=ln2·x`, moments
`QC=(1/ρ)Σ3·COL·f·x²`, `QNC=(1/ρ)Σ3·COL·f·x` (FAST_SBM l.4993–4996). Oracle is
CGS; port is SI (CGS only inside oracle-diff tests).

## Process inventory (oracle → port module → status)

| Process | Oracle | Port | Status |
|---|---|---|---|
| Bin grid + moments | param block, QC/QNC | `grid.py` | **done** |
| Collision kernels | `Kernals_KS` (only p-interpolates file-read `YW*` tables) | computed (reuse `sdm/kernels.py` Golovin/Hall/Long) | Golovin/Hall/Long **done**; oracle-table load todo |
| Collision-coalescence (Bott) | `coll_xxx_lwf`+`courant_bott_KS` | `collision.py` | **liquid self-collection + cross-species riming (`coll_xyx`) done**; LWF/`dm_rime` tracking + ice-ice aggregation (`coll_xyz`) todo |
| Diffusional growth | `JERRATE`→`JERTIMESC`→`JERSUPSAT`→`JERDFUN`/`JERNEWF`→`ONECOND1` | `diffusional_growth.py`+`supersaturation.py`+`remap.py`+`condensation_driver.py` | **warm chain + ONECOND1 done** |
| Drop nucleation (CCN) | `JERNUCL01_KS`, `WATER_NUCLEATION` | `nucleation.py` | **done** — Köhler r_crit + lognormal-tail, deficit-activation in column |
| Sedimentation per bin | `FALFLUXHUCM_Z` + `VR1..VR5` | `sedimentation.py` (REUSES `output.sedimentation_tendency`) | **done** — static substeps, per-level fall speeds, precip live |
| Column driver + wiring | `FAST_SBM` | `column.py` + `integration.py` + `kernel_registry.py` | **switchable `scheme="fast_sbm"` done** — stateless adapter |
| Freezing (immersion) | `FREEZ` (Bigg) | `freezing.py` | **done (iter 11)** — wired in column as liquid→ice + fusion heat; ice-spectrum carry-through deferred |
| Melting | `J_W_MELT` (Jiwen Fan) | `melting.py` | **done (iter 12)** — wired in column with q_i carry-through (freeze↔melt loop closed) |
| Breakup | `coll_breakup_KS`, `Spont_Rain_BreakUp` | `breakup.py` | todo |

**Lookup-table strategy**: WRF data-file tables (`capacity33.asc`, masses,
velocities, kernels `YW*`, breakup `PKIJ/QKJ`) are NOT in its git repo, so the
port derives them: grid masses analytic (doubling from 2 µm — reproduces
documented landmarks), kernels computed (Hall/Long/Golovin), velocities from
`sdm.kernels.terminal_velocity_cloud_rain_shima`, capacities analytic
(sphere). Any residual table → small committed `.npy` with provenance.

**Differentiability**: oracle integer bin-shifts + positivity clamps + hard
`wrf_error_fatal` guards → `jnp.where` smooth-safe forms + finite clamps,
`lax.scan` over bins (no `lax.cond` on traced bin loops). Adaptive runtime
trip counts (remap substeps, fall NSUB) → static config constants (reverse-AD
constraint, documented).

**Contracts**: every tendency module ships `__physics_contract__` + acceptance
test before the body. Helpers (`grid.py`, `config.py`) in
`test_physics_contracts.py` EXCLUDED.

## Faithfulness decisions (the non-obvious ones)

- **Constants from `legoesm.constants`** (L_v, R_v, c_pd, ε, σ_w), not the
  oracle's CGS roundings (`AL1=2500`, `BB1_MY=5.42e3`, `AKOE=3.3e-5`,
  `COEFF_VISCOUS`): ≤0.5% off, physics-faithful over rounding-faithful.
  Reference coefficients with no constant (D=0.211, ν=0.13 cm²/s, exponent
  1.94, VENTPL_MAX=5) live in `FastSBMConfig`.
- **Saturation from `legoesm.thermo`** (not oracle `POLYSVP`/`A·exp(−B/T)`):
  ~+0.1% at 273 K, up to ≈3.4% at 243/310 K — model-consistency rule
  (re-derived curves caused false CI supersat before).
- **`COEFF_REMAPING` = oracle literal 0.0066667** (not exact 1/150) — flips
  merges in the threshold band.
- **Single-substep ONECOND1** is faithful: this oracle's substep limiter is
  vestigial (`DTNEWL=min(DT,TIMEREV)` → one pass) and its final
  `JERDFUN_NEW(SUPINTW)` then equals the in-loop `JERDFUN(D1N)`.
- **KRDROP cloud/rain split** 0-based = `< KRDROP-1` (oracle `IF(KRR<15)`,
  1-based bins 1..14 cloud → 50 µm bin is RAIN).
- **CCN activation is deficit-diagnostic**: cloud number relaxes toward the
  Köhler target `N_CCN·frac_above(r_crit)` (self-limiting), no persistent
  aerosol depletion (the stateless-adapter gap vs oracle `FCCNR`).

## Validation highlights

- **Golovin box** (collision): mass conserved 1e-13, N(t)/M2(t) analytic
  moment laws bracketed at NKR=33 (spatial bias signed + documented),
  refinement-convergent.
- **SDM↔SBM cross-check**: ventilation-off growth coefficient B reproduces
  `sdm.condensation.drsq_dt` Maxwell core to 2e-3.
- **Supersaturation ODE**: both oracle branches verbatim (1e-13 large-x;
  small-x to the oracle's own cancellation noise — port is cancellation-free);
  dS/dR analytic at R=0.
- **Remap**: identity exact; doubling → exact one-bin shift; KO conserves
  number + grown mass 1e-12; oracle exact-match overwrite + boundary-interval
  + EXIT semantics replicated; edge tests (negative mass, sentinel loss,
  empty merge window, evaporation-disables-passes).
- **ONECOND1 parcel**: supersaturated condenses with q+q_c invariant 1e-13,
  `c_pd ΔT=−L_v Δq_v` 1e-12, S monotone-relaxes.
- **Köhler activation**: A vs oracle AKOE 4.4%, B≈0.72 (ammonium sulfate κ),
  r_crit∝s^{−2/3} + oracle RCRITI 1e-12, monotone + reservoir-bounded,
  deficit self-limits over steps.
- **Sedimentation**: column+precip invariant 1e-12, positivity at CFL 10.8,
  per-level velocity field, d(precip)/dV>0.
- **Column scheme**: dispatch via public config + resolves through production
  `MICROPHYSICS_REGISTRY`; emergent autoconversion (dense≫thin, no tuned
  rate); clear supersaturated cell forms cloud with total-water closure 1e-9;
  jit+grad through the full operator.

## Codex adversarial reviews (all findings resolved)

- **iters 1-2** FAIL→fixed: float32 `0/0` NaN at Bott flux `x1→0` (analytic
  limit `flux→gsk·c` + f32 grad regression); re-review CONDITIONAL→fixed
  (saturation-deviation docs).
- **iters 4-5** must-fixes: KO/3-point exact-match is OVERWRITE not add
  (sequential scan); boundary hits take lower interval; small-x guard series
  so dR-gradient analytic; oracle-literal `COEFF_REMAPING`.
- **iters 6-7** 2 HIGH: KRDROP off-by-one (`cloud_bins=KRDROP-1`); `fast_sbm`
  accepted by validate_strict but missing from production
  `MICROPHYSICS_REGISTRY` → KeyError (added entry + buildability literal).
  MEDIUM: prognostic N_c used when present; AMIP CLI choices += fast_sbm.
- **iters 8-9** 1 CRITICAL + 3 HIGH: stateless aerosol depletion / double
  activation → **deficit-activation** (relax toward target, self-limiting);
  fixed-reference fall speeds → **per-level** `VR1(K,KR)`. Remaining documented
  gaps: no persistent `FCCNR` budget, single-bin seed placement (oracle
  spreads bins 1–8), fixed-reference collision kernel (weak p-dependence).

### Iter 13 (2026-06-11) — end-to-end integration + float32 AD hardening
- **`scheme="fast_sbm"` validated through the REAL production pipeline**:
  `make_microphysics_physics(config, "hydrostatic")` → `physics_fn(state,
  grid, sigma)` on a Held-Suarez cubed-sphere state (shapes, finite
  tendencies, `jax.grad`). Added to `tests/atmosphere/hydrostatic/unit/
  test_microphysics.py::TestIntegrationHydrostatic`.
- That test surfaced **three float32 NaN-gradient traps** over a dry
  atmosphere (forward finite, grad NaN), all fixed:
  1. `nucleation.critical_dry_radius`: a constant `jnp.inf` in a
     `where(s>0,…,inf)` poisoned the VJP (`inf·0`); and `(4/(B s²))^{1/3}`
     made the reciprocal VJP carry `1/u²≈1e48` → float32 overflow.
     Reformulated to the algebraically identical `s^{-2/3}` (VJP `s^{-5/3}`,
     in-range) on a floored `s`, gate left to the caller's `S>1`.
  2. `supersaturation.supersat_relaxation_rate`: OPER2 `ε/((…)q)` is a 1/0
     at `q_v=0`; with `sfn=0` (no droplets) the product is `∞·0=NaN`.
     Floored `q_v` in that term → empty cell gives `R=0`.
  3. `column._reconstruct_spectrum`/`_reconstruct_ice`: the exact-mass
     rescale `q·ρ/mass_shape` overflowed (`1/mass²`) when a near-zero mean
     mass drove `r_med` below the grid (`mass_shape→0`). Floored the mean
     mass at the smallest bin so the mode stays on-grid (rescale≈1, bounded
     grad) — thin-cloud closure, never binds for real cloud (≳5 µm).
- Regression test `test_float32_grad_dry_atmosphere` (dry column, `rho(T)`,
  float32) pins all three. 94 fast_sbm + 284 microphysics/integration green.

### Iter 14 (2026-06-11) — cross-species riming
- **`collision.py` `bott_riming`** (oracle `coll_xyx_lwf`): ice collector
  bin `j` + liquid bin `i` → ice bin `k=ima(i,j)`, full `(i,j)` grid via
  new `precompute_riming_tables` (Courant matrix is symmetric in
  `m_i+m_j`, so reuses the same per-pair geometry). Same Bott flux split /
  aliasing / gmin semantics as self-collection, two spectra, product to
  ice. Total ice+liquid mass conserved; LWF/`dm_rime` tracking deferred
  (rimed liquid treated as fully frozen → caller releases fusion heat).
- Wired in column: supercooled cells (`T<0 °C`) rime cloud liquid onto
  ice → `dq_i` up, `dq_c/dq_r` down, `+(L_f/c_pd)·rimed` heat; gated by
  `jnp.where(T<T_freeze)`, vapor-neutral so the total-water closure is
  unchanged. 5 riming kernel tests (full-grid table, mass conservation +
  ice growth, empty-species no-op, d/dkernel>0) + a column test
  (seed-ice cell converts more cloud→ice than freezing-only, closure
  holds). 100 fast_sbm tests green.

### Iter 15 (2026-06-11) — ice aggregation + multi-step conservation
- **Ice-ice aggregation** (snow formation): it is ice *self*-collection,
  so it REUSES `bott_coalescence` + the self-collection Courant geometry on
  the ice spectrum (crystal-crystal sticking IS mathematically self-
  collection). The collision KERNEL is an explicit APPROXIMATION — the
  scaled liquid kernel `ck·ice_aggregation_efficiency` (~0.1), NOT an
  ice-specific kernel; a true ice kernel (fall speeds, branched-crystal
  cross-sections, T-dependent sticking) lands with the multi-ice-habit
  iteration. Gated on `T<0 °C` (codex iter-15: warm ice is melting away,
  not aggregating). Mass-conserving, no phase change → no latent heat,
  closure unchanged.
- **Multi-step trajectory conservation test**: full scheme (warm + ice +
  riming + aggregation) run 30 steps feeding tendencies back; total water
  (vapor+cloud+rain+ice) minus accumulated surface precip conserved to
  2e-3 — validates the scheme as a stable, conservative INTEGRATOR (not
  just per-step). 108 fast_sbm + integration tests green.

## Multi-ice-habit subsystem (in progress, branch `feat/fast-sbm-multi-ice`)

Oracle carries 5 distributions: drops (`FF1`), ice crystals (`FF2`, 3 habits
columns/plates/dendrites), snow (`FF3`), graupel (`FF4`), hail (`FF5`). The
single-ice-spectrum port (above) collapses these; the multi-ice subsystem
separates them, each with its own fall speed, capacitance, and cross-species
collection (`coll_xyz` between every pair). Incremental build:

- **Iter 1 (2026-06-11)** — habit-routed freezing. `freeze_step_routed`
  (oracle `FREEZ` `KRFREEZE` split): frozen drops in bins `< krfreeze`
  (=21, 1-based oracle `KR≤KRFREEZ`) → pristine ice crystals, larger
  (frozen rain) → hail/graupel. Same Bigg rate + fusion heat as
  single-category `freeze_step`; the two categories sum EXACTLY to the
  single-category ice (5 tests: sum-equals-single 1e-14, split-at-krfreeze,
  mass conservation, no-op above freezing, differentiable). Foundation for
  carrying distinct ice categories through the column. Codex: PASS (habit
  collapse + grad test notes applied).
- **Iter 2 (2026-06-11)** — per-bin ice terminal velocities by category
  (`ice_fall_speed.py`): computed replacement for the oracle's file-read
  `VR2..VR5` tables — `V=a·D^b·(ρ₀/ρ_air)^½`, `D=(6m/πρ_cat)^{1/3}`,
  Locatelli-Hobbs (1974) coefficients + bulk density per habit (snow
  a=11.72/b=0.41/ρ=100; graupel a=124/b=0.66/ρ=400, all in `FastSBMConfig`).
  Physical crossover: at equal mass fluffy low-density snow is larger so
  falls faster at small sizes; dense graupel wins in the precip regime
  (>~170 µm) and reaches a far higher max — the reason to separate
  categories. 7 tests (monotone, graupel>snow precip regime, mm-size
  magnitudes, density correction exact, array broadcast, unknown-category
  raises, differentiable). Codex: PASS-WITH-NOTES → ρ_ref=1.2 + tight
  coefficient test applied.
- **Iter 3 (2026-06-11)** — column carries TWO ice categories: crystal/snow
  (`q_i`) and graupel/hail (`q_g`). `freeze_step_routed` sends small frozen
  drops → snow, frozen rain → graupel; both melt above 0 °C; riming +
  aggregation act on the snow category; **each category now SEDIMENTS at its
  own fall speed** (rain via Shima, snow + graupel via `ice_fall_speed`) —
  ice precipitates for the first time (previously trapped in-column). Output
  `dq_i_dt` (snow) + `dq_g_dt` (graupel); total-water closure extended to
  `−dq_v = dq_c+dq_r+dq_i+dq_g+precip`. Tests: graupel precipitates >1.4×
  faster than equal-mass snow (the multi-category payoff), cold-cell ice
  falls without melting (loss == precip), aggregation conserves vs precip,
  multistep trajectory conserves all 5 species, supercooled freeze closure
  incl. both categories. 116 fast_sbm + 242 microphysics/integration green.
  Codex iter-3 review: PASS-WITH-NOTES (all 6 invariants PASS); LOW notes
  (docstring, graupel-melt-ladder, q_g>0 asserts) applied.
- **Iter 4 (2026-06-11)** — **graupel riming** (closes the comprehensive
  review's lone HIGH/FAIL): the oracle rimes BOTH ice categories
  (`coll_xyx_lwf(g4/g5,g1,…)` l.8495/8534), not snow only. Now snow rimes
  cloud first, then graupel collects the REMAINING cloud; both supercooled-
  gated, both release fusion heat (`dT_rime` over the total rimed liquid).
  Comprehensive review confirmed mass/energy/differentiability all CLEAN;
  this fills the one missing growth/heating path. Added the two flagged
  test gaps: warm-cell graupel melt (dq_g<0 + 5-species closure) and
  supercooled graupel riming (seed graupel converts more cloud→ice than
  freeze-only). 5-species closure stays exact (5.8e-12) with graupel
  riming on; float32 dry-atmosphere grad still finite. 119 fast_sbm green.

## Remaining work (warm + full ice phase done → bit-exact FSBM-2)

Multi-ice-category habits (separate snow/graupel/hail spectra + their
cross-species `coll_xyz`), LWF + `dm_rime` riming-fraction tracking,
collisional + spontaneous breakup (needs oracle `PKIJ/QKJ` tables),
oracle kernel-table loader for bit-fidelity, per-bin prognostic tracers
(replace per-step reconstruction), persistent aerosol reservoir, ice
sedimentation (ice currently does not fall).

**Codex iter-14 review** (HIGH/FAIL → fixed): `precompute_riming_tables`
emitted the full `(i,j)` grid, but oracle `coll_xyx_lwf` (`jmin=i; do
j=jmin+indc`, `indc=1`) skips `j≤i` — the collector ice bin is strictly
larger than the collected droplet (riming sweeps up *smaller* droplets).
Restricted to `j>i` (`(n-1)(n-2)/2` pairs); removed pairs were spurious
small-ice-collects-large-liquid interactions. Test updated to the oracle
traversal; added a warm-cell gradient test through the where-discarded
riming branch. g-space mass conservation confirmed machine-precision
(3.3e-14). Same-grid (single ice spectrum) = documented port
simplification vs the oracle's per-habit `x,y` grids.

## Iteration log (compressed at iter 10)

Iters 1–9 built the warm-rain chain bottom-up, each commit codex-reviewed +
fixed to clean before the next (see commit history on `feat/bin-microphysics`
and the "Faithfulness decisions" / "Codex reviews" sections above for the
substance). Per-iter prose was folded into those sections at the iter-10
shrink.

- **Iter 10 (2026-06-11)** — codex iters-8-9 fixes: deficit CCN activation
  (self-limiting, no aerosol-reservoir state), per-level sedimentation fall
  speeds (`sediment_bins` accepts `(ncol,nlev,nkr)` velocity), remaining
  stateless gaps documented. Spec text shrunk. 78 fast_sbm tests green.
  Codex re-review: CONDITIONAL PASS (only a loose test bound, tightened).
- **Iter 11 (2026-06-11)** — **`freezing.py`**: Bigg (1953) immersion
  freezing (oracle `FREEZ`), per-bin rate `P=m·A·exp(−B(m)·ΔT)`, frozen
  fraction `1−exp(−P·dt)`, liquid→ice + `(L_f/c_pd)Δq_ice` fusion heat;
  CGS Bigg coeffs in `FastSBMConfig` (mass kg→g internally). Wired in
  column: supercooled cells now produce `dq_i_dt` + fusion warming
  (ice-spectrum carry-through/collision/melting/ice-sedimentation deferred
  — ice is a diagnostic sink for now). 6 freezing tests (oracle PF formula
  1e-12, mass conservation, deeper-supercooling/larger-drops monotone,
  d/dT); cold-cell column test (ice + total-water closure incl. dq_i,
  supersaturated to avoid the S≈1 cancellation). Codex ADV-10-4 self-limit
  test tightened to assert ~0 activation when target already met. 85
  fast_sbm tests green. Codex review iter 11: PASS (only caveat = the
  cross-step ice persistence, addressed next).
- **Iter 12 (2026-06-11)** — **`melting.py`**: Jiwen-Fan constant-timescale
  melting (oracle `J_W_MELT`), size-dependent rate ladder (small bins full,
  mid `0.5/50`, large `0.683/120` s⁻¹ — thresholds in `FastSBMConfig`),
  ice→liquid + `−(L_f/c_pd)Δq` cooling. **Closes the freeze↔melt loop**:
  column now reconstructs an ice spectrum from carried `q_i`, melts it
  above 0 °C before warm physics, and the net `dq_i` = (melt-consumed +
  freeze-produced) — `q_i` is a real carried tracer (fixes iter-11 codex
  caveat). Total-water closure proven incl. melt/freeze internal transfers.
  6 melting tests (oracle rate ladder, mass conservation, small-bin full
  melt, **freeze→melt round-trip conserves**, d/dT) + 2 column tests
  (warm cell melts carried ice with closure; cold cell preserves ice). 91
  fast_sbm tests green.
