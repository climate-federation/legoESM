# EKE mixing-length (`eke_len`) variant — progress log

Spec: `docs/ocean/fidelity/eke_len_build_spec.md`. Branch: `matching_Veros_oracle`.
One dated entry per gate: what changed, the exact gate command + result, commit
hash, every micro-decision. Newest at the bottom. Adversarial review per gate
(physics-validator agent + `/code-review`; the codex CLI it normally drives is
absent on this machine).

## Gate status
- [x] L1 pure mixing-length functions (rhines + deformation radius + composite) + config + unit tests
- [x] L2 wire `int_N_dz` + β + scheme dispatch into the coupling (default bit-identical)
- [x] L3 differentiability (grad through the rhines length finite + nonzero)
- [x] L4 oracle confirmation (reproduce Veros `eke_len`/`L_rossby`/`L_rhines`/`K_gm` to machine precision)
- [x] L5 recipe adoption (flip EKE on in `build_acc_model_config`)
- [x] L6 regression lock + measure-first free-run re-check

---

### 2026-05-29 · L1 done — pure Rhines-limited mixing-length functions + config
Added to `lateral_mixing/eke.py` (all pure, differentiable):
- `eke_rhines_length(E, beta, cfg)` = √(√(max(E,0)+1e-30)/max(β,1e-16))  [Veros L_rhines].
- `eke_deformation_radius(int_N_dz, f, beta, cfg)`: c1=max(int_N_dz,0)/π; min(c1/max(|f|,1e-16),
  √(c1/(2·max(β,1e-16))+1e-30))  [Veros L_rossby, equatorially limited]. `int_N_dz`=∫N dz=C_rossby·π.
- `eke_len_composite(L_def, L_rhines, cfg)` = max(l_min, min(eke_cross·L_def, eke_crhin·L_rhines)).
- EKEConfig fields: `mixing_length_scheme="rossby"` (default = legoESM pre-eke_len `max(L_rossby,l_min)`),
  `eke_cross=1.0`, `eke_crhin=1.0` (Veros settings.py defaults). `validate_eke_config` raises on unknown
  scheme + non-positive cross/crhin (dispatch discipline). β-floor `_DENOM_FLOOR=1e-16` mirrors Veros
  `max(·,1e-16)` (numerical safety floor; exempt). No duplicate N²/N numerics — `int_N_dz` exposed from
  the shared `_eady_growth_and_length` at L2; L1 functions take it as input.
8 new direct unit tests: config defaults, validation raises, Rhines formula+monotone-in-E+β-floor,
deformation-radius both branches (midlat c1/|f| at -45°, equatorial √(c1/2β) cap as |f|→0)+zero-at-c1=0,
composite picks-min+floors, full-ACC-regime eke_len ~8 km (≪ saturated ~200 km Visbeck L).
GATE: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 pytest tests/ocean/unit/{test_eke,test_eke_regression,
test_baroclinic_decomposition,test_veros_acc_recipe}.py` → 56 passed (48 prior bit-identical — default
"rossby" path unchanged, E8 golden holds — + 8 new). Commit: <this commit>.
ADVERSARIAL REVIEW (physics-validator agent; codex CLI absent on this machine so it used its own
analysis): independently reimplemented Veros eke_len in numpy → BIT-EXACT match; jax.grad edge-point
probe (E=0,c1=0,β=0,|f|=0) → all forward+grad finite (no inf·0→NaN); units/sign/dispatch/constants all
clean. Verdict PASS, no code changes; one doc-precision nit on a test comment fixed.
NEXT: L2 — expose `int_N_dz` from `_eady_growth_and_length` (= Σ N·dz_half, already computed as
`_col[...,2]`); add `beta` arg + scheme dispatch to `compute_eke_kappa_gm`; analytic β=2Ω cosφ/R (from
config.constants + grid.cos_lat) in `compute_eke_step_kappa`. Default "rossby" must stay bit-identical.

### 2026-05-29 · L2 done — β + ∫N dz + scheme dispatch wired (default bit-identical)
- `_eady_growth_and_length` → returns `int_N_dz` (4-tuple); column reduction UNIFIED out of the
  if/else (`_stack=[ones,sigma,N]` always) so `sigma_bar`/`L`/`wet_col` stay bit-identical while
  `int_N_dz = ∫N dz` is exposed (reuses the shared N — no duplicate numerics). Both callers updated
  (`compute_visbeck_kappa_gm` discards it; `compute_eke_kappa_gm` uses it).
- `compute_eke_kappa_gm` → `beta` kwarg + dispatch on `eke_cfg.mixing_length_scheme` (static Python
  str ⇒ trace-time): "rossby"=`max(L_rossby,l_min)` (existing), "rhines"=`eke_len_composite(
  eke_deformation_radius(int_N_dz,|f|,β), eke_rhines_length(E,β))`; ValueError on unknown / on
  rhines-without-beta.
- `compute_eke_step_kappa` → analytic β=2Ω cosφ/R from `grid.cos_lat` + `omega`/`r_earth` kwargs;
  step call site passes `config.constants.{Omega,R_earth}` (Veros-pinned).
GATE: `pytest tests/ocean/unit/{test_eke,test_eke_regression,test_baroclinic_decomposition,
test_gm_redi_latlon_cgrid,test_latlon_cgrid_ocean,test_veros_acc_recipe}.py` → 122 passed
(121 prior bit-identical — Visbeck/GM-Redi + E8 EKE golden + decomposition golden all hold — + the
L2 dry-column robustness test).
ADVERSARIAL REVIEW (physics-validator): 6 dimensions CLEAN (bit-identical refactor proven
ALGEBRAICALLY incl. dry column; β = df/dy exact, β>0 both hemispheres, cos_lat clamp irrelevant in
ACC band; Ω/R from ConstantsConfig; dispatch-on-static; caller-completeness; AD-safe, no double-abs;
eke=None regression-safe). 1 MINOR latent bug FOUND+FIXED: `int_N_dz` was returned UNMASKED → NaN
L+grad on a fully-dry column (jacobian=0 ⇒ N²=0/0) in the rhines path (ACC never hits it — land uses
H_bathy=H_max ⇒ jacobian≈1 — so goldens passed, but a true-land global config would). FIX: wet-mask
`int_N_dz` like sigma_bar/N_bar (bit-identical on wet; L=l_min on land). Locked by
`test_L2_rhines_dry_column_finite_no_nan`. (Reviewer also flagged a PRE-EXISTING decomposition-golden
flake under broad acc/veros co-execution — XLA-CPU reduction order, NOT L2; tracked as separate debt.)
NEXT: L3 — committed grad test through the rhines length at typical values (finite + nonzero); the
edge cases (E=0, dry column) are already locked by L1's probe + the L2 dry-column test.

### 2026-05-29 · L3 done — differentiability at typical values (finite + nonzero)
Two committed grad tests in `test_eke.py`: (a) `jax.grad` through the pure rhines chain
(`eke_deformation_radius` + `eke_rhines_length` + `eke_len_composite`) is finite AND > 0 in BOTH
limiting regimes — d(eke_len)/dE>0 when Rhines limits (small E), d(eke_len)/d(∫N dz)>0 when the
deformation radius limits (large E); (b) grad of the prognostic `kappa_GM` (rhines coupling) w.r.t.
BOTH E and rho is finite + nonzero (E=0.05 makes the deformation radius the limiter so kappa carries
gradient through √E AND ∫N dz(rho)) — mirrors E5 for the rossby path. Edge cases (E=0, β=0, |f|=0,
dry column) already locked by L1's grad probe + `test_L2_rhines_dry_column_finite_no_nan`.
GATE: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 pytest tests/ocean/unit/test_eke.py -q` → 35 passed.

### 2026-05-29 · L4 done — oracle confirmation: eke_len reproduced to MACHINE PRECISION
`scripts/validate/ocean_fidelity/compare_eke_len_veros.py`: ran Veros ACC (enable_eke, 864000 s), captured
`eke,K_gm,eke_len,L_rossby,L_rhines,Nsqr,dzw,maskW,coriolis_t,beta`, and fed Veros's OWN
`(int_N_dz=Σ√N²·dzw·maskW, |coriolis_t|, beta, eke)` into the L1 functions. At the diagnostic time
level (tau=2):
  L_rossby max_rel_err = 5.80e-16 (corr 1.0)   [eke_deformation_radius]
  L_rhines max_rel_err = 0.00e+00 (corr 1.0)   [eke_rhines_length]
  eke_len  max_rel_err = 0.00e+00 (corr 1.0)   [eke_len_composite]
  K_gm     max_rel_err = 0.00e+00 (corr 1.0)   [eke_kappa_gm — re-confirms E9]
Veros eke_len mean 8.3 km / max 47.3 km; Veros L_rossby mean 56.9 km / max 221 km — i.e. the Rhines
scale (L_rhines mean 8.29 km) IS the limiter, exactly the ~25x reduction E9 flagged. This is the
deeper analog of E9 (which matched K_gm given Veros's eke_len); now the eke_len machinery ITSELF is
reproduced bit-exactly. The Rhines-limited mixing-length FORM matches the oracle ⇒ EKE can be adopted
apples-to-apples in the ACC recipe (L5). Informational gate (needs Veros); repro:
`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/compare_eke_len_veros.py`.
NEXT: L5 — flip EKE on in `build_acc_model_config` (`eke=EKEConfig(mixing_length_scheme="rhines",
eke_cross=2.0, eke_crhin=1.0)`; other params already default to ACC); update the stale
`build_acc_physics_config` docstring + strategy §8 ledger; lock with a recipe test.

### 2026-05-29 · L5 done — EKE flipped on in the ACC recipe (rhines)
- `ACC_GM_REDI_CONFIG.eke = EKEConfig(mixing_length_scheme="rhines", eke_cross=2.0, eke_crhin=1.0)`
  (other params already match ACC: c_k=0.4, c_eps=0.5, l_min=100, k_max=1e4, k_iso=1000, superbee).
  `kappa_GM=1000` retained as the EKE-off fallback.
- Stale `build_acc_physics_config` docstring (said "EKE NOT yet flipped on") → "ADOPTED"; strategy §8
  ledger eke_len row → DONE+adopted, EKE-closure row → adopted.
- **Integration bug surfaced + fixed:** the EKE-on free-run uses `jax.lax.scan`, whose carry must
  keep a CONSTANT pytree. `initial_state.eke` was `None` but `model.step` returns it as a `Field`
  (None→Field structure change) ⇒ scan `TypeError`. E6 only tested the step via a Python loop, so this
  was latent. FIX: `build_acc_state` seeds `eke=Field(e_min·land_mask)` from step 0 when EKE is on
  (Veros likewise starts eke small). VERIFIED the tier-2 frozen probe does NOT read `eke` (it calls
  `gm_redi_tracer_tendency_latlon` WITHOUT `kappa_gm_override` ⇒ constant-kappa path), so the committed
  tier-2 report is unchanged.
GATE: `pytest tests/ocean/unit/test_veros_acc_recipe.py tests/ocean/fidelity/test_run_acc_freerun.py
-m "slow or not slow"` → 27 passed (25 recipe incl. new EKE-adoption asserts + rest-IC + the @slow
2-day EKE-on free-run: stable, finite, develops an eastward ACC — first end-to-end exercise of the
rhines path through model.step).
NEXT: L6 — eke_len-active regression golden (rtol=1e-12, like E8) + measure-first free-run re-check
(re-run run_acc_freerun.py EKE-on; report whether the +232% KE / +70% transport gap moves; update the
driver's stale "EKE OFF" known-difference note).

### 2026-05-29 · L6 done — rhines regression golden + measure-first free-run re-check
L6a (regression lock): added a parallel rhines golden `fixtures/eke_step_regression_rhines_golden.npz`
+ `test_eke_rhines_step_regression_bit_identical` (rtol=1e-12), locking the rhines `eke_len` step
through `model.step`. Refactored `test_eke_regression.py` (`_build_case(eke_cfg)`/`_produce(eke_cfg)` +
shared `_check_golden`) so the original "rossby" golden stays BIT-IDENTICAL (DRY, no duplicate test
logic). Rhines golden non-degenerate: max eke 0.040, kappa_gm 12–7614 m²/s, eke_len 19–95 km, ≥0+finite.
GATE: `pytest tests/ocean/unit/test_eke_regression.py` → 3 passed.

L6b (MEASURE-FIRST free-run re-check, EKE-on 30-day vs Veros — the headline finding):
| metric | legoESM | Veros | % diff (EKE-OFF baseline) |
|---|---|---|---|
| ACC_transport_Sv | 59.19 | 34.75 | **+70.4% (was +70%)** |
| total_KE_J | 1.031e16 | 3.101e15 | **+232.6% (was +232%)** |
| vol_mean_T_C | 6.282 | 6.364 | −1.3% (was −1.3%) |
| T_max_C | 14.61 | 14.94 | −2.2% (was −2.2%) |
**Turning EKE on did NOT move the transport/KE gap** — it is essentially unchanged from the EKE-OFF
run (commit 7dad7512). WHY: over a 30-day spin-up `eke` starts at e_min and the EKE dissipation
timescale `L/(c_eps·√E)` is multi-year, so the prognostic kappa_GM is still ~0.3 m²/s (vs the constant
1000 it replaced) — and the gap is unchanged. This FALSIFIES the "EKE-off is what matters" hypothesis
from 7dad7512: the GM *skew* coefficient (0.3 vs 1000) is not the 30-day lever.
**CAVEAT (L5+L6 review, Issue 1):** the prognostic kappa replaces 1000 ONLY in the GM skew term. Veros
ACC also sets `enable_eke_isopycnal_diffusion=True` ⇒ `K_iso = K_gm` (the Redi *tracer* diffusivity is
the prognostic kappa, ~0.3 cold), but legoESM still holds `kappa_Redi=1000` constant — a ~3000× K_iso
mismatch over the whole run. So the unchanged gap is consistent with BOTH the **integrator** gap
(legoESM SSP-RK3 vs Veros leapfrog+AB2; audit gap #7) AND the **unmatched K_iso=K_gm coupling** — it
is NOT cleanly attributable to the integrator alone. The GM coefficient is apples-to-apples (machine-
exact form, L4); the Redi diffusivity is NOT yet — reproducing `K_iso=K_gm` (prognostic Redi) is the
documented next must-build, and the free-run should be re-measured AFTER that lands. Updated the
driver's `_KNOWN_DIFFERENCES` accordingly. Repro: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python
scripts/validate/ocean_fidelity/run_acc_freerun.py --years 0.0821918`.

# Follow-on build: prognostic Redi `K_iso = K_gm` (R-gates)

The L5+L6 review (Issue 1) surfaced that only the GM *skew* coefficient was EKE-driven, while Veros
ACC also sets `enable_eke_isopycnal_diffusion=True` ⇒ the Redi *tracer* diffusivity `K_iso = K_gm`.
This makes the full EKE/GM-Redi closure apples-to-apples.

### 2026-05-29 · R1 done — array-capable `kappa_Redi` + `EKEConfig.isopycnal_diffusion`
Made `kappa_Redi` array-capable through BOTH GM/Redi flux builders, mirroring the existing `kappa_GM`
broadcast: triad → `kappa_Redi_{c,u,v,w}` (face-interpolated via `interp_cell_to_{u,v}face`), centered →
`kappa_Redi_b` (cell-centred). A scalar passes through unchanged ⇒ the constant-coefficient path is
BIT-IDENTICAL. Added `EKEConfig.isopycnal_diffusion: bool = False` (Veros default; ACC = True). When
`kappa_Redi == kappa_GM` (the K_iso=K_gm coupling) the `(kappa_Redi-kappa_GM)` horizontal off-diagonal
cancels exactly, as in Veros. GATE: `pytest test_baroclinic_decomposition + test_gm_redi_latlon_cgrid +
test_eke{,_regression} + test_latlon_cgrid_ocean + test_veros_acc_recipe` → **125 passed** (goldens +
30 GM/Redi all bit-identical) + 4 new `TestKappaRediArray` tests (const-array==scalar to rtol 1e-13;
K_iso=K_gm override changes the tendency ~2× for a non-f(ρ) tracer).

### 2026-05-29 · R2 done — step `kappa_redi_override` + ACC adoption + 3rd golden
Public `gm_redi_tracer_tendency_latlon` gains `kappa_redi_override`; `kappa_Redi_eff = override if set
else cfg.kappa_Redi` flows to the triad/centered builders. Step (`_step_impl`) sets `kappa_redi_override
= kappa_gm_override` when `eke.isopycnal_diffusion`. ACC recipe: `isopycnal_diffusion=True`. Added a 3rd
regression golden `eke_step_regression_rhines_kiso_golden.npz` (rhines + K_iso=K_gm) — verified active
(differs from the constant-Redi rhines golden by 1.1e-6 in T; small because the golden's tracer is
≈f(ρ) so the Redi cancellation mostly holds). GATE: `pytest test_eke_regression + test_veros_acc_recipe
+ test_gm_redi_latlon_cgrid + test_baroclinic_decomposition + test_latlon_cgrid_ocean +
test_run_acc_freerun -m "slow or not slow"` → **97 passed** (incl. the @slow EKE+Redi free-run smoke).

### 2026-05-29 · R3 done — oracle (K_iso=K_gm machine-exact) + measure-first free-run
Oracle (`compare_eke_len_veros.py` extended): Veros's OWN `K_iso == K_gm` to MACHINE PRECISION
(max_rel_err 0.0, mean 33 m²/s, 19560 wet cells); legoESM reproduces it transitively
(`K_iso = kappa_redi_override = kappa_gm_override` = the prognostic kappa that L4 matched to Veros
K_gm machine-exact). The R3 oracle script ALSO captures `K_iso` and verifies Veros's own
`K_iso == K_gm`. (Honesty note, R4 review item 6: the tier-2 frozen-state probe `tendency_probe.py`
does NOT exercise the override — it uses the constant-kappa GM path — so K_iso=K_gm is validated
through `model.step` + the `rhines_kiso` golden + the Veros-side oracle, NOT the per-process probe.) **MEASURE-FIRST (EKE+Redi-on 30-day free-run): the gap is UNCHANGED**
(ACC transport +70.3%, total KE +233.2% — vs +70.4%/+232.6% GM-only at L6 and +70%/+232% EKE-off).
With the WHOLE EKE/GM-Redi closure now apples-to-apples, the 30-day gap does not move ⇒ **conclusively
the integrator** (SSP-RK3 vs Veros leapfrog+AB2) + the spin-up transient, NOT the eddy closure. GM/Redi
affects the slow (multi-year) baroclinic adjustment, not the fast (days) barotropic jet spin-up.

## FINAL STATUS
The eke_len Rhines mixing length (L1–L6) AND prognostic Redi `K_iso=K_gm` (R1–R3) are complete +
honestly green: the FULL EKE/GM-Redi closure is now apples-to-apples with Veros ACC (form, params,
`eke_len`, `K_gm` AND `K_iso` all oracle machine-exact). The measure-first free-run, with every eddy
knob matched, leaves the ACC-transport/KE gap unchanged at +70%/+233% — so the dominant remaining ACC
free-run gap is **the time integrator** (legoESM SSP-RK3 vs Veros leapfrog+AB2+Robert-Asselin; audit
gap #7). NEXT: wire `outer_integrator="leapfrog_ab2"` (a standalone `timestepping/leapfrog_ab2.py` existed but was deleted 2026-10 as unwired; recover it from git e86e0b86d;
it needs the τ-1 carry threaded through the ocean step/scan) — the I-gates. Also open: a
developed-flow eke-bridge for a tier-2 prognostic-GM-tendency comparison.
