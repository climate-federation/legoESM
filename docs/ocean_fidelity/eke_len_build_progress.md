# EKE mixing-length (`eke_len`) variant — progress log

Spec: `docs/ocean_fidelity/eke_len_build_spec.md`. Branch: `matching_Veros_oracle`.
One dated entry per gate: what changed, the exact gate command + result, commit
hash, every micro-decision. Newest at the bottom. Adversarial review per gate
(physics-validator agent + `/code-review`; the codex CLI it normally drives is
absent on this machine).

## Gate status
- [x] L1 pure mixing-length functions (rhines + deformation radius + composite) + config + unit tests
- [ ] L2 wire `int_N_dz` + β + scheme dispatch into the coupling (default bit-identical)
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
