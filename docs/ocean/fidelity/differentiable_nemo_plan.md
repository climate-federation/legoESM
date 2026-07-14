# Differentiable NEMO in legoESM — implementation plan

Status: DRAFT 2026-07-14. Synthesized from an 8-way NEMO 5.0.2 ↔ legoESM
side-by-side audit (full per-subsystem reports in
`~/oracle-builds/nemo5/gap_audit/{0..8}_*.md`; NEMO oracle built + running at
`~/oracle-builds/nemo5/nemo_5.0.2`, conda env `nemo-build`).

## Progress log

**2026-07-14 (Phase A foundation):**
- ✅ **EOS-80 Roquet-55 port** (`eos="nemo_eos80"`) built + reviewed (2 agents) +
  **certified against NEMO's compiled run to 4.5e-13 kg/m³** over 18k GYRE cells
  (`tests/ocean/unit/test_nemo_roquet_eos.py`). P0 #1 done. TEOS-10 set still TODO.
- ✅ **Certificate plumbing proven — and simpler than §2 assumed.** `iom_rstput`
  (the restart writer) emits native NetCDF **without XIOS**; only `iom_put` is the
  no-op. So the trend dump is `compute trend → iom_rstput`, no raw-nf90 writer
  needed. Demonstrated by `MY_SRC/restart.F90` dumping `rhd` (the EOS ground truth).
- ✅ **GYRE un-rotated** (`MY_SRC/usrdef_hgr.F90`, `zsin_alpha=0`): stock GYRE is a
  45°-rotated beta-plane (2-D lat, rotated-axis u/v) that legoESM's `LatLonGrid`
  can't hold; density was grid-independent but every *tendency* couples to velocity
  direction, so the regular-grid config is required. Now a regular lat-lon box,
  stable, legoESM-representable.
- ▢ **NEXT: state bridge** (`nemo_io.py` + `nemo_state_bridge.py`, mirror
  `mitgcm_state_bridge.py`) → `probe_latlon_cgrid` tendency diff. Then the momentum
  trend dump (`MY_SRC/trddyn` via `iom_rstput`).
- ▢ **Trajectory/statistics tier** (complements the tendency tier): once the bridge
  runs legoESM-GYRE, do a longish two-model run and compare climate/statistics
  (NOT trajectory — chaotic). A 2-yr NEMO-GYRE reference run establishes the oracle
  half.

## 0. Goal & verdict

**Goal.** A *differentiable* re-expression of NEMO's discrete ocean operator
inside legoESM, certified by a **single-timestep tendency match** (read a NEMO
restart → one step → compare every per-term dU/dt, dT/dt, dS/dt, dη/dt to
legoESM to ~roundoff, before chaos). The certificate gives an *existence
guarantee* — that legoESM's solution manifold contains NEMO's attractor — so
that "can we find NEMO's climate" becomes a pure optimization question, cleanly
separated from model capacity.

**Verdict from the audit: legoESM is far closer than expected.** Almost every
NEMO OCE subsystem already exists in legoESM as one of:
- **EXACT** — byte/algebra-identical port (S-EOS, geothermal, advective BBL,
  Richardson/double-diffusion/internal-wave mixing, MLE, the NCAR/COARE bulk
  kernel, FCT-2 block, EEN/AL81 vorticity core, Hollingsworth KE-gradient, z\*
  vertical coordinate + partial cells).
- **VARIANT** — structurally correct, discretely different (RK3 family,
  barotropic predictor, TKE internals, neutral slopes, GM form, tracer
  iso-Laplacian, EOS-80/TEOS-10 polynomial).
- **MISSING** — genuinely absent (Roquet-55 EOS polynomial, ENE lat-lon
  vorticity, OSMOSIS BL, EKE-GEOMETRIC structure fn, interior `tradmp`,
  `nn_fwb`).

So this is **"reconcile + add faithful options + wire a recipe,"** not "build a
model." The single hard *enabler* is a NEMO-side trend writer (below).

## 1. Subsystem scorecard

| # | subsystem | overall | GYRE-relevant gaps | evidence |
|---|---|---|---|---|
| 1 | time-stepping / barotropic | **VARIANT** | RK3 family wrong (SSP vs Wicker-Skamarock); barotropic Matsuno vs AB3-AM4 + non-NEMO extras; time-filter cosine≠boxcar | `1_timestepping.md` |
| 2 | momentum (vor/keg/hpg/ldf) | **EXACT core / VARIANT geom** | EEN core matched; **GYRE ENE not on lat-lon**; C2 keg = mean-of-squares≠legoESM square-of-mean; PGF wiring | `2_momentum.md` |
| 3 | tracer transport + iso-Lap | **near-EXACT / PARTIAL** | FCT-2 exact but recipe picks ppm_fct; iso-Lap over-counts msc=F implicit diagonal; slope taper differs | `3_tracer_transport.md` |
| 4 | vertical mixing / conv / drag | **VARIANT** | TKE surface-BC/self-diff/Prandtl/mxl deltas; EVD K_conv 100× low; drag TKE-coupling absent | `4_vertical_mixing.md` |
| 5 | EOS + vertical coordinate | **EOS VARIANT / vcoord EXACT** | **Roquet-55 poly missing** (GYRE uses EOS-80); z\* exact | `5_eos_vcoord.md` |
| 6 | surface forcing + tracer BC | **bulk EXACT / BC PARTIAL** | GYRE = 3 config fixes; trasbc timestep placement; constants c_sw/ρ0 | `6_surface_forcing.md` |
| 7 | mesoscale eddy (GM/slope/MLE/EKE) | **MLE EXACT / rest VARIANT** | slopes N²-vs-ρ + cap/ML-ramp/Shapiro (feeds GYRE Redi); GM skew vs bolus; EKE partial | `7_mesoscale_eddy.md` |
| 8 | infra: trends / halo / metrics | **enabler + bridge** | **trend `iom_put` is a no-op w/o XIOS → need MY_SRC writer**; tracer diag struct missing; north-fold in harness | `8_infra_trends_halo.md` |

## 2. The certificate mechanism (harness — build first, it gates everything)

**NEMO side (`MY_SRC`, we control the build):**
1. `MY_SRC/trddyn.F90` + `MY_SRC/trdtra.F90` — replace the `iom_put` calls in
   `trd_dyn_iom`/`trd_tra_iom` with **direct writes via the native `iom_nf90`
   layer** (iom_put is a compiled no-op without XIOS — the trends are computed
   and silently dropped otherwise). Handle: (a) the `MOD(kt,2)` even-step gate
   on split tracer trends — bypass for a step-1 cert; (b) the RK3-unvalidated
   warning (`trdini.F90:79`, #487) — restrict the first cert to robust terms
   (hpg/ldf/zdf/`jptra_totad`) or use the MLF stepper.
2. Momentum trends: 13 terms (`trd_oce.F90:62-76`: hpg/spg/keg/rvo/pvo/zad/ldf/
   zdf/bfr/atf/tau…). Tracer: 21 terms (`:34-54`); use `jptra_totad`
   (`traadv.F90:394`) for advection (split xad/yad/zad are commented out).
3. Driver: short spin → write restart → 1-step run with the writer → NetCDF of
   every term + `mesh_mask.nc`.

**legoESM side:**
4. Bridge (in the fidelity harness, NOT the model — per
   `oracle_recipe_strategy.md`): read NEMO restart + `mesh_mask` → legoESM
   `LatLonCGridOceanState`; halo/north-fold/staggering conventions handled here,
   verified by equivariance tests.
5. Reuse `MomentumTendencyDiagnostics` (`state.py:593`, already modeled on NEMO
   `trd_*` with a machine-precision closure test). **Build the missing
   per-term `TracerTendencyDiagnostics`** (only lumped `dT_diss` today).
6. Diff on **regrouped sums** — legoESM lumps KE+PGF and applies planetary
   Coriolis outside the struct, vs NEMO's separate keg/hpg/pvo/rvo.

**First run on GYRE (regular grid)** to sidestep curvilinear metrics + north-fold.

## 3. Free wins — recipe re-wiring (config only, no new numerics)

`nemo_recipe.py` is DINO/ORCA-oriented and mis-wired in several places that the
audit caught. These cost ~1 line each and immediately improve fidelity:

- **FCT order**: `tracer_advection="ppm_fct"` → `"fct2"` (GYRE/NEMO nn_fct=2).
- **Time filter**: `barotropic_time_filter="cosine"` → boxcar (helper
  `compute_nemo_boxcar_centred_weights` already exists).
- **EOS (DINO)**: `eos="veros_gsw"` → `nemo_seos` (S-EOS form already exact).
- **PGF (DINO)**: `smc03` → `adcroft` + `pgf_quadrature="nemo_trapezoid"`
  (DINO uses `ln_hpg_sco`, not `ln_hpg_djc`).
- **Bottom drag**: `"legacy"` → `nemo_loglayer` (faithful law already present).
- **GM coeff (ORCA)**: constant κ=600 → Treguier `nn_aht_ijk_t=21` adaptive
  (legoESM ships it, unwired).
- **Solar (GYRE)**: RGB → 2-band Jerlov type-I; disable Hallberg resolution fn
  (no NEMO equivalent — MOM6-style).

## 4. New numerics to build, prioritized

### P0 — required for the GYRE certificate
- **Roquet-55 EOS polynomial** (EOS-80 + TEOS-10 coeff sets) + α,β for `bn2`.
  *Dominant tendency lever* — feeds baroclinic PGF at every cell.
- **NEMO-faithful neutral slope** as a selectable `slope_scheme`: build ∂zρ from
  N² (not ρ-difference), hard-MIN denominator cap at `rn_slpmax`, Δz grid cap,
  **mixed-layer linear flattening**, 16-pt/Shapiro coastal smoother. (Feeds
  GYRE's iso-Laplacian Redi.)
- **iso-Laplacian msc=F handling**: put the `S²∂z` diagonal fully implicit in
  `trazdf` (explicit `traldf` flux = 0); legoESM currently over-counts it
  explicitly.
- **TKE deltas**: Dirichlet surface BC `en(1)=max(emin0,(ebb/ρ)·taum)`
  ebb=67.83; self-diffusion α=1 (vs 30); Prandtl `nn_pdl=1`; mixing-length
  `nn_mxl=3` (min/√ up-down) + floor.
- **EVD convection**: fix `K_conv` (100× too small → match `rn_evd=100`),
  `min(rn2,rn2b)≤−1e-12` trigger.
- **`trasbc` timestep-placement** reconciliation (leapfrog 0.5-avg vs RK3).
- **`TracerTendencyDiagnostics`** per-term struct (§2.5).
- **Constants**: NEMO set via `ConstantsConfig` (c_sw=3991.87, ρ0=1026 vs Veros
  3994/1035) — cross-cutting (EOS ref density + every buoyancy/heat term).
- **ENE lat-lon vorticity** OR (pragmatic) configure the NEMO GYRE oracle with
  `ln_dynvor_een` to match legoESM's existing AL81/EEN. **C2 mean-of-squares
  KE-gradient** OR configure GYRE `nn_dynkeg=1` (Hollingsworth, already exact).

### P1 — exact time-stepping (harder, needed for multi-step trajectory)
- **RK3 Wicker-Skamarock** 3-stage `(Δt/3,Δt/2,Δt)` re-evaluating the coupled
  RHS each stage → new integrator `rk3_ws`.
- **Barotropic AB3-AM4 forward-backward** + half-step-back SSH → new
  `barotropic_solver="forward_backward_ab3am4"`; **gate off the non-NEMO extras**
  (BEBT blend, divergence damping, MAXVEL clip, Laplacian η-diffusion) for the
  fidelity card.

### P2 — DINO / ORCA
- EEN geometry deltas (F-point `e3f` masked-average `nn_e3f_typ`, coastal
  `fmask` vs Neumann-fill) — matter on partial cells / z\*.
- GM **advective-bolus** form option (vs the current Griffies skew flux).
- EKE-**GEOMETRIC** structure function S=N²/N²_ref + linear dissipation.
- OSMOSIS BL (only for `ln_zdfosm` configs — legoESM has Large-94 KPP, not
  OSMOSIS).
- Curvilinear `mesh_mask` ingestion + tripole north-fold bridge.
- Interior `tradmp`, `nn_fwb`, diffusive BBL (`nn_bbl_ldf`), non-penetrative
  convection `tranpc` iterative form.

## 5. Recipe structure (recipe = pure config, options in canonical modules)

Ship **per-config NEMO-faithful recipe cards**, each just *selecting* the
faithful blocks (never a bespoke solver):
- `nemo_gyre_v1` — GYRE's exact scheme set (or the legoESM-matched subset for
  rung 0). NEW.
- `nemo_dino_v1` — reuse/fix the existing DINO card (§3 rewiring).
- `nemo_orca_v1` — adds partial cells, tripole, TEOS-10, bulk, GM adaptive.

All new options land in the canonical modules (`eos.py`, slope/advection
dispatch, `vertical_mixing/`, integrator + barotropic dispatch) with their own
unit tests; the cards only wire them. Mimicry glue (halo strip, axis transpose,
time-level handling) lives in the fidelity harness.

## 6. Differentiability watch (consolidated)

- **FCT/Zalesak limiter** (min/max/sign): piecewise-linear, valid a.e.
  subgradients, and the cert is *primal* — **NOT a blocker**; the float32
  `Q/inc` NaN hazard is already fixed (`grad_safe_ratio`).
- **NEMO's hard-MIN slope clamp** is nondiff; legoESM's tanh taper is smooth — a
  genuine tension between "faithful" and "differentiable." For the differentiable
  card, prefer smooth variants where they don't move the tendency past
  tolerance; otherwise accept a.e. subgradients.
- **Gate off for the fidelity card**: MAXVEL `jnp.clip` (zero subgradient),
  adaptive-implicit `wAimp` Courant kinks, eta-floor `maximum`. Keep the
  barotropic loop on the `scan` path (`differentiable_barotropic`), not
  `fori_loop`. Iteration counts stay static.
- MPI mass-redistribute allreduce — confirm AD-safe under sharding.

## 7. Sequenced roadmap

- **Phase A — harness + pipeline proof.** MY_SRC trend writer, restart→1-step
  dump, legoESM bridge, TracerTendencyDiagnostics, regrouped-sum diff.
  Configure the NEMO GYRE oracle to legoESM's already-EXACT blocks (S-EOS or
  EOS-80-once-built, EEN, Hollingsworth, FCT2, geothermal off). **Milestone:
  machine-precision match on the robust term subset (hpg/ldf/zdf/totad).**
- **Phase B — GYRE full certificate.** Build the P0 numerics; match all 13
  momentum + 21 tracer terms to ~roundoff on GYRE step-1. **Milestone: GYRE
  tendency certificate.**
- **Phase C — exact time-stepping.** P1 (`rk3_ws`, AB3-AM4 barotropic); check
  multi-step trajectory divergence stays at roundoff for a few steps.
- **Phase D — DINO 1°.** P2 subset (z\* already exact, EEN geom deltas, S-EOS,
  GM). **Milestone: DINO certificate** (deterministic, published reference).
- **Phase E — ORCA/tripole.** Curvilinear mesh + north-fold + TEOS-10 + bulk +
  sea ice → full differentiable NEMO; then the optimization question (find
  NEMO's climate via differentiable calibration).

## 8. Bottom line

The differentiable-NEMO target is **credible and mostly assembled**. The
critical path is: (1) the MY_SRC trend writer (small, unblocks everything);
(2) the Roquet EOS polynomial + NEMO slope + TKE deltas (the P0 numerics);
(3) a GYRE-faithful recipe card. Exact time-stepping (RK3-WS + AB3-AM4) is the
one genuinely new dynamical-core chunk. Everything else is reconcile-and-wire on
blocks legoESM already has.
