# Round-1 codex findings — classification & disposition

## Finding 1 — silent no-op off MPAS — CONFIRMED, FIXED
`hard_sat_ice_curve` only acts in the MPAS post-step hook (`grid_type=='mpas'`);
off-MPAS it is silently inert. Fix: `validate_strict` now refuses
`hard_sat_ice_curve=True` unless `grid.grid_type == 'mpas'`.
Test: `test_validate_refuses_non_mpas_ice_curve`.

## Finding 2 — q_i existence ≠ ice-scheme capability; energy-creating fallback — CONFIRMED, FIXED
Two fixes:
1. `validate_strict` now requires `microphysics == 'morrison'` (the only scheme
   with prognostic q_i + N_i + a nucleation mass this drain deposits to).
   Kessler / Seifert-Beheng placeholder-ice configs are refused.
   Test: `test_validate_refuses_non_morrison_ice_curve`.
2. Defense-in-depth: the post-step DEGRADES to the energy-EXACT liquid drain
   when `q_i is None` (rate AND heating liquid, `c_pd*T + L_v*q_v` exact) instead
   of heating with the blended L_s-weighted latent heat while binning liquid q_c
   (which injected `(1-w)*L_f*dq` of spurious energy). Test:
   `test_no_qi_reservoir_degrades_to_liquid_drain`.

## Finding 3 — ice mass created with zero ice number — CONFIRMED, FIXED
The drain deposited q_i without N_i. Morrison M2005 deposition self-gates as
`EPSI ~ N_i^(2/3)` (orphan ice can't grow/sublimate) and the fall-speed PSD slope
`lambda_i=(rho_ci pi N_i/q_i)^(1/3)` is degenerate at N_i=0 (ice can't sediment).
Fix: seed `dN_i = dq_i / mi0` with `mi0 = 4/3 pi rho_ci r_nuc^3` — Morrison's OWN
Cooper-nucleation mass<->number closure (`dq_i_nuc = dN_i_nuc * mi0`), read from
the Morrison sub-config. Number is additive and carries no latent heat, so
water/energy budgets are unchanged. Helper `_seed_nucleated_ice_number`; test
`test_seed_nucleated_ice_number`.
NB codex's "landing removes the supersaturation nucleation needs": the drain
fires only above RH_ice>1.1 and runs AFTER microphysics each step, so Morrison
has already nucleated on the step's supersaturation; the drain corrects the
dycore transport spike on the final state.

## Finding 4 — "the claimed enthalpy is not the model's phase enthalpy / 700 J/kg" — REFUTED (with correction context)
Morrison's invariant IS `h = c_pd*T + L_v*q_v - L_f*q_ice`, and Morrison heats
deposition with **L_s** (`morrison.py:~1190`: `L_s*(dq_i_dep+...)/c_pd`). Our
`L_eff = w*L_v + (1-w)*L_s` matches Morrison's deposition convention exactly, so
routing the ice fraction to q_i conserves Morrison's phase enthalpy to the SAME
residual Morrison itself carries. That residual is `(1-w)*(L_s - L_v - L_f)*dq`
with the constants `L_s=2.834e6`, `L_v=2.501e6`, `L_f=3.337e5` giving
`L_s-L_v-L_f = -700 J/kg` (0.21% of L_f). This is a **pre-existing model-wide
`L_s != L_v+L_f` inconsistency** in `constants.py` that Morrison's own deposition
shares identically — NOT introduced by this change, and NOT fixable here without
DE-SYNCING the post-step from Morrison's deposition heating. Building L_eff from
`L_v + (1-w)*L_f` would make the post-step exactly conserve `h` but INCONSISTENT
with Morrison's L_s deposition. Out of scope; flagged for a separate
constants-consistency PR. The post-step's own frozen-L statement
`c_pd*T + L_eff(T0)*q_v` is exact (probe F: max|dH|=2.8e-14).

## Finding 5 — frozen w(T0) routing vs t_new curve; above-freezing q_i — REFUTED (documented edge)
The `w(T0)` routing is REQUIRED, not a bug: to conserve enthalpy exactly with a
frozen-L solve, the phase split must use the SAME w(T0) as the frozen L_eff(T0).
Splitting by `w(t_new)` would break `c_pd*T + L_eff(T0)*q_v` conservation
(residual `(w(t_new)-w(T0))*L_f*dq`). The only artefact is a cap-sized adjustment
from `T0 in [268.15, 273.15)` landing slightly above freezing while holding a
little q_i; that ice is melted by Morrison's `melt_ice` (now with a valid seeded
N_i) on the next microphysics call, releasing L_f — self-correcting, narrow band,
and irrelevant at the TTL temperatures this fix targets (T << 233 K, w=0).

## Finding 6 — test evidence gap — CONFIRMED, FIXED
- Genuine fp32 (x64-OFF) branch: `test_fp32_ice_curve_branch_subprocess` (child
  process with x64 off — finite, positive, on-curve float32 result).
- FD agreement for d/dT and d/dp (not just finite d/dq_v):
  `test_gradient_matches_central_fd` (rel <= 1e-4 at TTL and mixed cells).
- Non-MPAS, non-Morrison, no-q_i-reservoir cases: the validate + degradation
  tests above.
