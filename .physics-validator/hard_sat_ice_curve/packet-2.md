You are an independent ADVERSARIAL physics reviewer for LegoESM. ROUND 2.
In Round 1 you raised 6 findings on the mixed-phase (ice-curve) hard-saturation
drain (TTL dehydration fix). I have fixed the confirmed ones and refuted two with
argument. VERIFY each disposition; find any NEW bug the fixes introduced. Cite
line numbers. Be CONCISE; end with
`OVERALL VERDICT: <no substantive findings | findings remain: ...>`.

Full current diffs in this repo:
`.physics-validator/hard_sat_ice_curve/warm_rain.diff`
`.physics-validator/hard_sat_ice_curve/model_driver.diff`
`.physics-validator/hard_sat_ice_curve/config.diff`

# Round-1 findings and dispositions

## F1 silent no-op off MPAS — FIXED
`validate_strict` now refuses `hard_sat_ice_curve=True` unless
`grid.grid_type=='mpas'` (config.py). The drain is only wired into the MPAS
post-step hook. Test `test_validate_refuses_non_mpas_ice_curve`.

## F2 non-ice-scheme fallback (energy creation / orphan ice) — FIXED (two layers)
1. `validate_strict` now requires `microphysics=='morrison'` (the only scheme
   carrying prognostic q_i + N_i + a nucleation mass to deposit into). Kessler /
   Seifert-Beheng (placeholder zero-ice) are refused.
2. Defense in depth: the post-step now DEGRADES to the energy-EXACT liquid drain
   when `q_i is None` — it passes `ice_curve=(ice_curve and q_i is not None)` to
   `hard_saturation_drain` and heats with `L_v` into q_c, so `c_pd*T + L_v*q_v`
   is exact. The previous branch heated with the blended (L_s-weighted) latent
   heat while binning liquid q_c, injecting `(1-w)*L_f*dq` of energy. Test
   `test_no_qi_reservoir_degrades_to_liquid_drain`.
```python
    _use_ice = bool(ice_curve) and (q_i is not None)
    rate = hard_saturation_drain(T, q_v, p_full, dt, hard_threshold,
                                 hard_max_heating_K, ice_curve=_use_ice)
    dq = rate * dt
    if _use_ice:
        _l_cp = mixed_phase_l_over_cp(T).astype(T.dtype)
        T_new = T + _l_cp * dq
        w_liq = mixed_phase_liquid_fraction(T).astype(q_v.dtype)
        q_c_new = q_c + w_liq * dq
        q_i_new = q_i + (1.0 - w_liq) * dq
    else:
        T_new = T + (constants.L_v / constants.c_pd) * dq
        q_c_new = q_c + dq
        q_i_new = q_i
```
Rationale for depositing the cold fraction to q_i (NOT q_c): at TTL the LIQUID
saturation sits ~60% above ice saturation, so liquid q_c formed at the ice curve
is deeply liquid-SUBsaturated and Morrison would immediately re-evaporate it,
undoing the dehydration. Ice is the only reservoir the drain can stick to.

## F3 ice mass with zero ice number — FIXED (seed N_i)
The driver now seeds `dN_i = dq_i / mi0`, `mi0 = 4/3 pi rho_ci r_nuc^3` from the
Morrison sub-config — Morrison's OWN Cooper-nucleation mass<->number closure
(`dq_i_nuc = dN_i_nuc * mi0`). Orphan ice (N_i=0) would be M2005-deposition-inert
(`EPSI ~ N_i^(2/3)`) and non-sedimenting (`lambda_i=(rho_ci pi N_i/q_i)^(1/3)=0`).
Number is additive, no latent heat -> water/energy budgets unchanged. Helper
`_seed_nucleated_ice_number` + driver call site (model_driver.diff). Test
`test_seed_nucleated_ice_number`.
```python
def _seed_nucleated_ice_number(N_i, dq_i, ice_nuc_mass):
    return N_i + jnp.maximum(dq_i, 0.0) / jnp.maximum(ice_nuc_mass, 1.0e-30)
# driver: _dq_i_dep = _qi_hs - _qi_fld.data ;  N_i += _dq_i_dep / mi0
```

## F4 "not the model's phase enthalpy / 700 J/kg" — REFUTED
Morrison heats deposition with **L_s** (morrison.py ~1190:
`L_s*(dq_i_dep+dq_i_nuc+prds+prdg)/c_pd`) and its invariant is
`h=c_pd*T+L_v*q_v-L_f*q_ice`. Our `L_eff=w*L_v+(1-w)*L_s` matches Morrison's
deposition convention, so with the ice fraction routed to q_i the post-step
conserves Morrison's h to the SAME residual `(1-w)*(L_s-L_v-L_f)*dq` that
Morrison's OWN deposition carries. With `L_s=2.834e6, L_v=2.501e6, L_f=3.337e5`,
`L_s-L_v-L_f = -700 J/kg` (0.21% of L_f) — a **pre-existing constants.py
`L_s != L_v+L_f` inconsistency** shared by Morrison, NOT introduced here, and not
fixable in the drain without desyncing from Morrison's L_s deposition. Flagged
for a separate constants PR. The frozen-L statement `c_pd*T+L_eff(T0)*q_v` is
exact (test `test_enthalpy_conserved_with_blended_latent_heat`; probe max|dH|=2.8e-14).
Do you accept this, or do you want L_eff rebuilt as `L_v+(1-w)*L_f` (and Morrison's
deposition left on L_s, i.e. the drain deliberately DIFFERENT from Morrison)?

## F5 frozen w(T0) routing / above-freezing q_i — REFUTED (documented edge)
`w(T0)` routing is REQUIRED for exact enthalpy with the frozen-L solve: splitting
by `w(t_new)` breaks `c_pd*T+L_eff(T0)*q_v` conservation by `(w(t_new)-w(T0))*L_f*dq`.
The only artefact is a cap-sized adjustment from `T0 in [268.15,273.15)` landing
just above freezing with a little q_i; Morrison's `melt_ice` (now with valid
seeded N_i) melts it next step (releases L_f) — self-correcting, narrow band,
irrelevant at TTL (T<<233, w=0). Accept, or do you want a melt/clamp in the hook?

## F6 test evidence gap — FIXED
- Genuine fp32 (x64-OFF) branch: `test_fp32_ice_curve_branch_subprocess`
  (child process, x64 off; finite/positive/on-curve float32).
- FD agreement for d/dT and d/dp (not just finite d/dq_v):
  `test_gradient_matches_central_fd` (rel<=1e-4, TTL + mixed).
- MPAS/Morrison/no-reservoir validate + degradation tests.

# TEST + NUMERICAL STATUS (JAX_ENABLE_X64=1, node l40048)
- `test_hard_sat_ice_curve.py` (22) + `test_hard_saturation_adjustment.py` (24)
  = **46 passed**. Broader: +212 passed in CLI/warm-rain/qv-smoothing (the only
  2 fails are the pre-existing `mcfarlane` vs `mcfarlane+hines` gwd default,
  independent of this change — confirmed at clean HEAD).
- Physics probe (bracket monotonicity, byte-identical OFF, fp32 parity,
  grad-vs-FD, water+enthalpy conservation, heating cap = 5.000 K exact, on-curve
  landing, RHi gate): FAILURES: NONE.

# SCRUTINIZE
The N_i-seeding layering (driver reaches into Morrison sub-config for rho_ci /
r_nuc); the `_use_ice` degradation dtype/consistency; the validate guards'
field access (`self.grid.grid_type`, `self.microphysics`); whether restricting
to Morrison is too narrow or correctly scoped; any NEW conservation/gradient
regression from the poststep restructure; F4/F5 acceptance.
