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
- [ ] E7 idealized channel (E spins up bounded; kappa_GM responds)
- [ ] E8 regression lock (EKE-active golden case)
- [ ] E9 oracle confirmation (informational) + recipe adoption

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
