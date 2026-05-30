# EKE mixing-length (`eke_len`) variant — progress log

Spec: `docs/ocean_fidelity/eke_len_build_spec.md`. Branch: `matching_Veros_oracle`.
One dated entry per gate: what changed, the exact gate command + result, commit
hash, every micro-decision. Newest at the bottom. Adversarial review per gate
(physics-validator agent + `/code-review`; the codex CLI it normally drives is
absent on this machine).

## Gate status
- [x] L1 pure mixing-length functions (rhines + deformation radius + composite) + config + unit tests
- [x] L2 wire `int_N_dz` + β + scheme dispatch into the coupling (default bit-identical)
- [ ] L3 differentiability (grad through the rhines length finite + nonzero)
- [ ] L4 oracle confirmation (reproduce Veros `eke_len`/`L_rossby`/`L_rhines`/`K_gm` to machine precision)
- [ ] L5 recipe adoption (flip EKE on in `build_acc_model_config`)
- [ ] L6 regression lock + measure-first free-run re-check

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
