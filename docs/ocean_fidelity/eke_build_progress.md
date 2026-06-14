# Prognostic EKE build — progress log

Spec: `docs/ocean_fidelity/eke_build_spec.md`. Branch: `matching_Veros_oracle`.
One dated entry per iteration: what changed, the exact gate command + result, commit hash,
every micro-decision. Newest at the bottom.

## Gate status
- [x] E1 EKE closure module (pure) + EKEConfig + direct unit tests (8 green)
- [x] E2 dispatch + GM/Redi coupling (compute_eke_kappa_gm; GMRediConfig.eke; validate_eke_config)
- [x] E3 positivity (semi-implicit dissipation; E >= 0 by construction, no clip)
- [x] E4 budget closure (advection + lateral diffusion conserve integral-E, machine-eps)
- [x] E5 differentiability (grad through closure + coupling finite + nonzero)
- [x] E6 state threading (field + step integration + restart) + zero-behaviour-when-OFF
- [x] E7 idealized channel (E spins up bounded; kappa_GM responds)
- [x] E8 regression lock (EKE-active golden case)
- [x] E9 oracle confirmation (informational): closure verified vs Veros K_gm to machine precision;
      eke_len gap documented; recipe adoption deferred behind the eke_len variant

---

### 2026-05-29 · iter 1 · E1 done — pure EKE closure module
Created src/legoesm/ocean/physics/lateral_mixing/eke.py: EKEConfig (c_k=0.4, c_eps=0.5,
l_min=100.0, k_iso=1000, advection="superbee", e_min, kappa_gm_max) + pure functions
eke_mixing_length (L=max(L_rossby,l_min)), eke_kappa_gm (c_k·L·√E, clamped [0,kappa_gm_max]),
eke_local_tendency (P-eps; P=kappa_GM·sigma², eps=c_eps·E^{3/2}/L). Design: E is 2-D
(depth-integrated) to match the 2-D Visbeck kappa_GM the GM/Redi already accepts; the closure is
PURE on (E, sigma, L) — sigma (=<N|S|>_z Eady rate) and L (Rossby radius) come from the SHARED
compute_visbeck_kappa_gm machinery (no duplicate N²/slope/L numerics), wired at E2. Energetically
consistent: P=kappa_GM·sigma²=kappa_GM·M⁴/N² (GM mean-APE→EKE conversion). 8 direct unit tests
green (kappa_GM nonneg+monotone+capped, mixing-length floor, zero-at-E=0, production form,
dissipation ≤0 with E^{3/2} scaling, finite on a field).
NEXT: E2 — wire a prognostic-kappa_GM mode into GM/Redi (config dispatch, ValueError on unknown);
this makes eke.py live. Reuse the Visbeck path (which already accepts a 2-D kappa_GM array) +
factor a shared (sigma_bar, L) helper from compute_visbeck_kappa_gm so EKE + Visbeck share it.

### 2026-05-29 · iter 2 · E2 done — GM/Redi prognostic-kappa_GM coupling + config + validation
Factored `_eady_growth_and_length(rho, S_x, S_y, z_coord, jacobian, f, visbeck_cfg)` out of
`compute_visbeck_kappa_gm` (BIT-IDENTICAL — 49 Visbeck/GM-Redi/decomposition tests green) so the
N²/slope/Rossby-length numerics live in ONE place. Added `compute_eke_kappa_gm(E, rho, S_x, S_y,
z_coord, jacobian, f, visbeck_cfg, eke_cfg)` in _gm_redi_common: reuses the shared helper for
(sigma_bar, L) + the EKE closure for kappa_GM = c_k·L·√E (2-D, wet-masked); returns (kappa_GM,
sigma_bar, L) — sigma_bar/L feed the EKE local source/sink. Added `GMRediConfig.eke: EKEConfig |
None = None` (presence-based selection; uses visbeck length params) + `validate_eke_config`
(fail-fast on non-physical params). 11 EKE tests green (config presence, validation raises,
prognostic kappa monotone-in-E + >=0 + finite + L floored + kappa=0 at E=0). No duplicate numerics.
NEXT: E3 (positivity of E under the closure), E5 (differentiability) — both testable on the pure
closure now. E4 (budget closure) needs the E-transport operator (2-D advection by the depth-mean
flow + iso-diffusion, reusing divergence_cgrid) — define it conservatively, then E6 wires it +
the `eke` state field into the step (the cross-cutting part; SegmentCarry discipline).

### 2026-05-29 · iter 3 · E3 (positivity) + E5 (differentiability) done
Added `eke_apply_local_source(E, sigma, L, cfg, dt)`: forward-explicit production +
SEMI-IMPLICIT dissipation -> E_{n+1} = (E_n + dt·P)/(1 + dt·c_eps·√E_n/L), num>=0 denom>=1, so
E>=0 BY CONSTRUCTION (no clip/mask — doctrine-clean; the standard stable treatment of the
quadratic EKE dissipation). E3 tests: E stays >=0 + finite over 50 steps for dt up to 10 days
(incl. huge dt where explicit Euler would go negative); E grows from ~1e-6 toward a bounded
steady state under forcing. E5: jax.grad through compute_eke_kappa_gm + eke_apply_local_source is
finite + nonzero. 14 EKE tests green.
NEXT: E4 — conservative 2-D E-transport operator (flux-form advection by the depth-mean flow via
divergence_cgrid + lateral iso-diffusion via laplacian_cgrid; both conserve integral-E by
telescoping); test conservation to machine-eps. Then E6 (state field + step threading; the
cross-cutting SegmentCarry part).

### 2026-05-29 · iter 4 · E4 done — conservative 2-D E-transport
Added eke_horizontal_transport(E, U_bar, V_bar, grid, eke_cfg, mask, u_mask, v_mask) in
gm_redi_latlon_cgrid.py: flux-form advection of E by the depth-mean flow (upwind to faces) via
divergence_cgrid + lateral diffusion via k_iso·laplacian_cgrid — both conserve the area-integral
of E by telescoping (periodic lon; no-flux N/S walls). Reuses the shared operators (no duplicate
numerics). E4 test: advection-only, diffusion-only, and combined each conserve integral(E·area) to
<1e-12 (float64 grid). NOTE: as with the flux-form F5, the test U_bar must respect the C-grid
periodic-wrap invariant (U_bar[:,n_lon]==U_bar[:,0]) the real model maintains, else the advective
flux doesn't telescope. 15 EKE tests green (E1-E5).
NEXT: E6 — the cross-cutting state threading. Add `eke` to LatLonCGridOceanState + rest_state init
+ the step loop (transport via eke_horizontal_transport + local source/sink via
eke_apply_local_source, using compute_eke_kappa_gm for kappa_GM into GM/Redi) + SegmentCarry +
restart I/O + channel-packing + the compiled-segment ref loops. DEFAULT OFF (gm_redi.eke=None) ⇒
bit-identical existing (decomposition gate + suite green). This is the largest/riskiest gate.

### 2026-05-29 · iter 5 · E6a — eke state field added (inert), bit-identical existing
KEY SCOPE REDUCTION: LatLonCGridOceanState is NOT carried in SegmentCarry (grep: only restart.py
references it) — so the SegmentCarry / compiled-segment discipline does NOT apply to ocean EKE.
E6 reduces to: state field + init + step integration + restart.
E6a: added `eke: object = None` as the last field of LatLonCGridOceanState (inert; matches the
existing T_som/S_flux_div_prev optional-None pattern). Default None -> not integrated, not read ->
ZERO behaviour change. Verified: 47 tests green (decomposition gate + general lat-lon C-grid + EKE)
— bit-identical existing.
NEXT (E6b — step integration, the intricate coupling): in LatLonCGridOceanModel._step_impl, when
config.gm_redi.eke is not None: compute (kappa_GM, sigma_bar, L) = compute_eke_kappa_gm(state.eke,
rho, slopes, grid.f, gm_redi.visbeck, gm_redi.eke) [reusing the slope helpers], pass kappa_GM into
gm_redi_tracer_tendency_latlon (add a `kappa_gm_override` param, default None = existing path), and
integrate state.eke: eke_horizontal_transport(E, U_bar, V_bar, ...) + eke_apply_local_source(E,
sigma_bar, L, dt). Gate on eke OFF -> step unchanged (existing tests), eke ON -> E integrates +
stays positive/finite (new step test). Then E6c (restart I/O for eke), E7 (channel spinup), E8/E9.
Design note: a first version may recompute rho+slopes for eke (redundant with gm_redi's internal
computation) — correct, flag compute-once as an optimization.

### 2026-05-29 · iter 6 · E6 done — step integration (E6b) + restart round-trip (E6c)
E6b: wired the prognostic-EKE branch into LatLonCGridOceanModel._step_impl's GM/Redi block,
gated on `gm_cfg.eke is not None` (default None ⇒ existing constant/Visbeck path bit-identical).
When active: E = state.eke.data (or e_min if None) → compute_eke_step_kappa(T_mid,S_mid,eta,H,E,...)
returns (kappa_GM, sigma_bar, L); kappa_GM is passed into gm_redi_tracer_tendency_latlon via the new
`kappa_gm_override` param (precedence override > visbeck > constant); E is integrated one step:
E_t = E + dt·eke_horizontal_transport(E, U_bar, V_bar, ...) [U_bar/V_bar = level-mean flow, masked,
preserves periodic wrap so ∫E telescopes] then E_new = eke_apply_local_source(E_t, sigma_bar, L, dt)
[semi-implicit, E≥0 by construction]; state_new._replace(eke=Field(E_new·land_mask)).
E6c: ocean restart I/O is FIELD-GENERIC (ocean/restart.py _iter_state_fields → only Fields are
serialised), so eke=None (OFF) is skipped ⇒ existing test_restart_round_trip_bit_identical stays
green (bit-identical), and eke=Field (ON) round-trips automatically — locked by a new test.
SCOPE: SegmentCarry / compiled-segment ref loops + ml/channel_packing are atmosphere-PE only;
LatLonCGridOceanState is not carried in SegmentCarry, so those E6 sub-items are N/A for ocean EKE
(verified iter 5: only restart.py references the ocean state).
Gates: EKE-on step test (test_E6_step_integrates_eke_field: 15 steps, E≥0 + finite + grows under
baroclinic forcing) + restart round-trip (test_E6_restart_round_trip_eke_field) + bit-identical-OFF
(79 green: 17 EKE + 2 decomposition + 30 GM/Redi + 30 lat-lon C-grid).
Command: JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 pytest tests/ocean/unit/{test_eke,test_baroclinic_
decomposition,test_gm_redi_latlon_cgrid,test_latlon_cgrid_ocean}.py → 79 passed.
NEXT: E7 — idealized baroclinic channel with EKE on: E spins up to a BOUNDED level (no blow-up, no
negative), kappa_GM responds to E (varies in space, ≥0), runs vs EKE-off both stable. Then E8
(regression lock: EKE-active golden) + E9 (oracle confirmation + recipe adoption, informational).

### 2026-05-29 · iter 7 · E7 (idealized channel) + E8 (regression lock) done
E7: tests/ocean/unit/test_eke.py — a coarse re-entrant baroclinic channel
(create_latlon_grid(16,32), 6 levels, polar walls, a meridional tanh T front ->
baroclinic slopes -> Eady-rate forcing). EKE on (test_E7_channel_eke_spins_up_
bounded_and_kappa_responds): over 30 steps E stays >=0 + finite, spins up above the
floor (>5x e_min; observed ~30x), stays BOUNDED (<<1e3, no blow-up); kappa_GM read
off the final state via compute_eke_step_kappa is >=0, finite, spatially varying
(std>0), and DIFFERS from the baseline-at-uniform-E (it responded to the evolved E).
NOTE: the absolute E level is small because EKE equilibrates on a multi-year
dissipation timescale L/(c_eps·√E) while the gate runs ~hours — the gate verifies
the SIGN of evolution + boundedness + kappa response, NOT equilibrium (documented in
the test). EKE off (test_E7_channel_stable_with_eke_off): the same channel with a
constant GM kappa runs finite/stable — EKE adds the prognostic closure without
destabilising. 19 EKE tests green.
E8: tests/ocean/unit/test_eke_regression.py + fixtures/eke_step_regression_golden.npz
— a committed golden of the EKE-active step (5 steps, structured Gaussian-band initial
E so transport + semi-implicit source/sink + kappa coupling are all non-degenerate),
locked at rtol=1e-12 (mirrors test_baroclinic_decomposition). FORCED to the fp64
precision policy (set_policy/restore) so the whole step is float64 — the model
otherwise casts to its fp32 storage policy and a float32 golden would not hold 1e-12
cross-platform. Locks eke + kappa_gm + sigma + L + T + S; key-set lock + golden-exists
guard + a non-degeneracy sanity (max E > 1e-2). 23 tests green (19 EKE + 2 regression +
2 decomposition).
Command: JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 pytest tests/ocean/unit/{test_eke,
test_eke_regression,test_baroclinic_decomposition}.py → 23 passed.
NEXT: E9 — oracle confirmation (informational). The ACC recipe build_acc_physics_config
docstring still says "EKE is a Veros-only closure (no legoESM equivalent yet)" — now
STALE. Document the legoESM EKE block in the §8 ledger, update that docstring, and
(doctrine rule H, since Veros ACC is enable_eke=True) ADOPT eke=EKEConfig() in
ACC_GM_REDI_CONFIG / build_acc_model_config so the recipe is apples-to-apples. E9 is
informational (not pass/fail) — record correlation-with-Veros status (Veros eke field +
kappa_GM) if a developed-flow Veros snapshot is available; otherwise document the gap.

### 2026-05-29 · iter 8 · E9 done — oracle confirmation (closure verified to machine precision)
Ran a developed-flow Veros ACC integration (enable_eke; runlen 864000 s = 10 d) and captured
Veros's 3-D eke / K_gm / eke_len. KEY FINDING (verified from veros/core/eke.py + acc.py):
  Veros K_gm = min(eke_k_max, eke_c_k·eke_len·√eke)  — IDENTICAL FORM to legoESM eke_kappa_gm
  Veros dissipation c_int = eke_c_eps·√eke/eke_len   — IDENTICAL to legoESM eps = c_eps·E^{3/2}/L
The ACC setup sets eke_c_k=0.4, eke_c_eps=0.5, eke_k_max=1e4, eke_lmin=100, superbee advection,
isopycnal diffusion — ALL match EKEConfig() defaults exactly. NUMERICAL CONFIRMATION: feeding
Veros's OWN eke + eke_len into legoESM eke_kappa_gm reproduces Veros's K_gm to MACHINE PRECISION
at the matching time level (tau=2: max_rel_err=0.0, corr=1.000000, 19560 wet cells). Reproducible:
scripts/validate/ocean_fidelity/compare_eke_kappa_veros.py.
GAP SURFACED (the next must-build, NOT part of E1-E9): the MIXING LENGTH differs.
  Veros eke_len = max(eke_lmin, min(eke_cross·L_rossby, eke_crhin·L_rhines)), eke_cross=2, with the
    eddy-energy Rhines scale L_rhines=√(√E/β) -> developed ACC: ~8 km mean, 47 km max.
  legoESM EKE L = the shared Visbeck first-baroclinic Rossby radius -> ~200 km (saturated), ~25x larger.
  So the closure FORM is oracle-exact, but a STANDALONE legoESM EKE run would give kappa_GM ~25x
  Veros's until L is reconciled. Logged as the "extend L (eke_len variant)" follow-up in the
  strategy doc §8 EKE ledger (Rhines limiting + eke_cross/eke_crhin scaling as a selectable length).
ADOPTION DECISION (doctrine rule H, "if validated"): EKE closure validated (form+params machine-
exact) but NOT flipped on in build_acc_model_config yet — GMRediConfig.eke stays None — because the
eke_len gap would make the recipe's prognostic kappa_GM ~25x too large. Adoption is deferred behind
the eke_len variant. ALSO E9 is 2-D-vs-3-D aware: legoESM E is 2-D (depth-integrated, by design to
match the 2-D Visbeck kappa_GM); Veros eke is 3-D — the per-cell closure match holds because the
formula is pointwise; a 2-D field comparison would use depth-reduced Veros eke.
Stale-doc corrections (EKE now exists): build_acc_physics_config docstring, strategy doc §8 ledger
(EKE row -> DONE + new eke_len row) + §"apples-to-apples" must-build line, phase_g_recipe_fidelity_
plan.md (2 notes).
Commands: JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/compare_eke_kappa_veros.py
=> "reproduces Veros K_gm to MACHINE PRECISION".

## FINAL STATUS
E1-E8 honestly green + committed (E1-E5: earlier iters; E6: f07301d1; E7+E8: d4b5fd6c).
E9 documented + oracle-verified (this iter). The EKE closure is a canonical, differentiable,
positivity-preserving, conservative, oracle-form-exact block. Remaining for full ACC apples-to-
apples (NEW follow-up, beyond this build spec): the eke_len mixing-length variant + then recipe
adoption + a developed-flow eke-bridge for the tier-2 GM-tendency comparison.
