# Flux-form momentum build — progress log

Spec: `docs/ocean_fidelity/flux_form_build_spec.md`. Branch: `matching_Veros_oracle`.
One dated entry per iteration: what changed, the exact gate command + result, commit hash,
every micro-decision. Newest at the bottom.

## Gate status
- [x] F1 existing paths bit-identical (gate green; unchanged by F2a)
- [x] F2 dispatch discipline — DONE. momentum_advection validated against
      {vector_invariant,weno5,weno7,flux_form}; momentum_flux_scheme against {upwind,centered};
      unknown raises (was a silent fallthrough). Tests in test_config_footguns.py.
- [x] F3 zero-velocity ⇒ zero (test_flux_form_momentum.py)
- [x] F4 uniform-flow analytic ⇒ zero
- [x] F5 momentum conservation (centered + upwind; volume-weighted integral ~ machine-eps)
- [x] F6 differentiability (jax.grad finite + nonzero)
- [x] F7 idealized-gyre stability (80-step forced-flow run finite + KE bounded; vi baseline too)
- [x] F8 regression lock (flux_form case added to the decomposition golden; 77 arrays; gate green)
- [x] F9 oracle confirmation (informational) — DOCUMENTED (see iter 3)

---

### 2026-05-29 · iter 1 · F2a done (momentum_advection dispatch validation)
Added `VALID_MOMENTUM_ADVECTION = {vector_invariant, weno5, weno7}` (single source) in
ocean_pe_latlon_cgrid.py and a fail-fast check in `LatLonCGridOceanModel._validate_config`
(was a silent fallthrough to vector-invariant for any unknown literal). Tests
test_validate_config_{rejects_unknown,accepts_valid}_momentum_advection in test_config_footguns.py.
Confirmed no real config uses an out-of-set value (only vector_invariant/weno5 in src+tests).
F1 bit-identical gate green (no behaviour change). 11 footgun + decomposition tests green.

**Substage design for iter 2+ (the hard part — F3/F4/F5):**
New substage `_bc_horizontal_momentum_advection_flux_form(du_dt, dv_dt, u, v, h_u, h_v,
u_mask_3d, v_mask_3d, mask, grid, config)`; dispatch at the `_bc_pv_flux` call (orchestrator
~L2146): `if _mom_adv == "flux_form": <new> else: _bc_pv_flux`. Add `momentum_flux_scheme:
str = "upwind"` to LatLonCGridOceanConfig + "flux_form" to VALID_MOMENTUM_ADVECTION +
VALID_MOMENTUM_FLUX_SCHEME validation, all WITH the substage.

Conservation-correct construction (mirror `divergence_cgrid`'s FV metric — the crux for F5):
u-momentum at u-points = -(1/(A_u·h_u))·[δ_x(Fx_uu) + δ_y(Fy_vu)] where
- Fx_uu at cell centres = Ut_c · u_c, Ut_c = 0.5*(Fu[:, :-1]+Fu[:, 1:]) (Fu=h_u·u·u_mask),
  u_c = centred (or upwind by sign of Ut_c) interp of u to centres; the x-divergence to
  u-points telescopes in lon (periodic) -> momentum conserved.
- Fy_vu at vertices (n_lat+1,n_lon+1) = Vt_vtx · u_vtx, Vt_vtx = Fv averaged to lon-faces,
  u_vtx = u averaged (or upwinded) to lat-faces; the y-divergence to u-points telescopes in
  lat with v=0 wall -> conserved. Use the divergence_cgrid face-length (dy_u, dx_v) + 1/A_u
  weighting, NOT the plain gradient_x/y operators (which divide by distance, not area), or F5
  fails. Symmetric construction for v-momentum at v-points.
GATES to drive next: F3 (u=v=0 -> 0, trivial), F4 (uniform u, flat periodic -> ~0), F5
(periodic-domain momentum integral ~ machine-eps), then F6/F7/F8/F9.

**KEY DE-RISKING INSIGHT (iter 1):** F5 (momentum conservation) holds *structurally* for any
telescoping flux-divergence form — the exact metric VALUES (A_u, dx_vtx) only affect ACCURACY
(F4/gyre), not conservation. Proof: write du/dt = -(dFx + yp)/h_u and weight the conservation
sum by VOLUME (A_u·h_u): Σ A_u·h_u·du/dt = -Σ A_u·(dFx + yp). The x-part dFx =
gradient_x_cgrid(Fx_uu) = (Fx[j]-Fx[j-1])/dx_u; at fixed lat A_u/dx_u is lon-constant so it
factors and Σ_j telescopes to 0 (periodic). The y-part yp = (Fy[i+1]·dx[i+1] - Fy[i]·dx[i])/A_u,
so A_u·yp telescopes in i to the pole boundary = 0 (v=0 wall -> Fy=0 at poles). So the
conservation test must weight by **area·h_u** (volume), and it passes for ANY consistent metric.
The x-part is exactly `gradient_x_cgrid(Fx_uu_centre)` (E/W faces same lat -> dy cancels). Only
the y-part (vertex->u-point, dx varies by lat) needs explicit `dx_vtx[i]/A_u[i]` weighting;
SYMMETRICALLY for v-momentum the y-part is `gradient_y_cgrid` and the x-part needs the vertex
metric. **Iter 2 first action:** read `divergence_cgrid` v-part (latlon_cgrid_operators.py ~615-660)
to copy its exact `dx_v` (v-face zonal length) + area convention, so the vertex-metric parts are
consistent with continuity. Guard `is_tripolar` with a clear ValueError (flux-form on tripolar is
a follow-up; ACC/gates use regular/regional grids). Upwind vs centred u_c/v_c via
`momentum_flux_scheme` (default "upwind"); centred is the cleanest for the F4 uniform-flow check.

### 2026-05-29 · iter 2 · substage implemented; F2-F6 green
Implemented `_bc_horizontal_momentum_advection_flux_form` (FV form mirroring divergence_cgrid's
metric) + `momentum_flux_scheme` config field (after `constants`, literal default -> safe) +
"flux_form" in VALID_MOMENTUM_ADVECTION + VALID_MOMENTUM_FLUX_SCHEME + validation in
_validate_config. Dispatch wired at stage 7b (flux_form -> new substage, else _bc_pv_flux), and
CRITICALLY: when flux_form, the KE-gradient term (dKE_dx/dy) is zeroed before KE_PGF so it isn't
double-counted (vector-invariant splits advection into grad-KE + PV flux; flux-form gives the
whole thing). The substage fills the same `vortcor` diagnostic slot, so the orchestrator +
momentum closure are unchanged.
**Two real issues found + fixed via the gates (this is why the loop fits):**
1. KE double-count (above) — caught by reasoning before coding.
2. v-momentum x-flux relied on the input periodic wrap column (u[:, n_lon]==u[:, 0]); random test
   input violated it -> conservation residual 1e-2. Fixed to roll-based periodicity (drop the u
   wrap col), mirroring the u-momentum x-part -> telescopes EXACTLY regardless of input. The
   u-momentum part was already roll-based (passed at 1e-18 first try).
**Conservation nuance (documented, honest):** on a walled lat-lon domain the v-momentum
y-advection legitimately transfers momentum to the N/S walls (y is not periodic), so the simple
domain integral is the wall reaction, NOT zero. F5 isolates the SCHEME's conservation (flux
telescoping) via an interior flow (v->0 in the 2 rows nearest each pole); u-momentum is periodic
in lon and conserves unconditionally. Gate uses a float64 grid (default area is float32).
GATES GREEN: F1 F2 F3 F4 F5(centered+upwind) F6. 16 flux-form + footgun tests green; momentum-
diagnostics closure green.
NEXT: F7 (Munk/Stommel gyre stability with flux_form), F8 (flux_form golden case in the
decomposition gate), F9 (ACC oracle re-run, informational).

### 2026-05-29 · iter 3 · F7+F8 green; F9 documented — BUILD COMPLETE (F1-F8 green)
F7 (gyre stability) + F8 (golden regression lock) committed (71bb972e). F9 (oracle
confirmation, INFORMATIONAL per spec) attempted: built the recipe with
momentum_advection="flux_form" and ran the ACC tier-2 comparison vs Veros for du_adv. Result:
DEGENERATE — the cached Veros ACC snapshot at runlen_s=4800 (one short step from rest) has
|u|=|v|=0 and veros_du_adv=0, so EVERY momentum process (incl. coriolis, unrelated to flux-form)
compares as L2=0/nan. The prior "du_adv corr 0.58" baseline in the §8 ledger came from a
developed-flow Veros run, not this rest snapshot. So the quantitative oracle number is NOT a
flux-form issue and is deferred to a developed-flow run.
**F9 follow-up (documented, not blocking — informational gate):** run Veros ACC with developed
flow (compare_tendencies_acc.py --runlen-s ~864000) and momentum_advection="flux_form" to
quantify the du_adv correlation improvement toward Veros; and ADOPT flux_form in
build_acc_model_config for true ACC apples-to-apples (Veros ACC momentum IS flux-form — doctrine
rule H / the apples-to-apples principle). The flux-form scheme itself is verified independently
by F1-F8 (truth tiers), which is the trust basis; the oracle is the lowest-trust confirmation.

## FINAL STATUS — flux-form momentum build COMPLETE
Gates F1-F8 honestly GREEN and committed; F9 (informational) documented. The canonical lat-lon
C-grid dycore now offers momentum_advection="flux_form" (+ momentum_flux_scheme upwind/centered),
a conservative FV scheme verified for: existing-paths bit-identity, dispatch discipline,
zero-velocity, uniform-flow analytic, momentum conservation (machine-eps, centered+upwind),
differentiability, idealized-gyre stability, and bit-identical regression lock. This is the first
of the two ACC apples-to-apples must-build blocks (the other is EKE — see eke_scope.md).
