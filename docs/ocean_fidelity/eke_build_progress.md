# Prognostic EKE build — progress log

Spec: `docs/ocean_fidelity/eke_build_spec.md`. Branch: `matching_Veros_oracle`.
One dated entry per iteration: what changed, the exact gate command + result, commit hash,
every micro-decision. Newest at the bottom.

## Gate status
- [x] E1 EKE closure module (pure) + EKEConfig + direct unit tests (8 green)
- [x] E2 dispatch + GM/Redi coupling (compute_eke_kappa_gm; GMRediConfig.eke; validate_eke_config)
- [ ] E3 positivity (E >= 0 / E_min)
- [ ] E4 budget closure (advection + iso-diffusion conserve integral-E)
- [ ] E5 differentiability
- [ ] E6 state threading + zero-behaviour-when-OFF (existing bit-identical)
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
