# Flux-form momentum build — progress log

Spec: `docs/ocean_fidelity/flux_form_build_spec.md`. Branch: `matching_Veros_oracle`.
One dated entry per iteration: what changed, the exact gate command + result, commit hash,
every micro-decision. Newest at the bottom.

## Gate status
- [x] F1 existing paths bit-identical (gate green; unchanged by F2a)
- [~] F2 dispatch discipline — F2a DONE (momentum_advection validated against
      {vector_invariant,weno5,weno7}; unknown raises; was a silent fallthrough). flux_form +
      momentum_flux_scheme validation land WITH the substage.
- [ ] F3 zero-velocity ⇒ zero
- [ ] F4 uniform-flow analytic ⇒ zero
- [ ] F5 momentum conservation (periodic domain)
- [ ] F6 differentiability
- [ ] F7 idealized-gyre stability
- [ ] F8 regression lock (flux_form golden case)
- [ ] F9 oracle confirmation (informational)

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
