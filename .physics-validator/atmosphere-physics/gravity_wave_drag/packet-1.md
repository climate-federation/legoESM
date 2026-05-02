# Adversarial review: legoESM gravity wave drag

You are an independent adversarial physics reviewer. Read the GWD source files below.

Convention: column shape `(ncol, nlev)`, surface at `[:, -1]`, top at `[:, 0]`.

Hypotheses to verify or rebut:

1. **`hines.py:90` — `sigma_grown = sigma_gw * rho_ratio[:, k]` may be wrong**.
   - `rho_ratio[:, k] = sqrt(rho_sfc / rho[:, k])` — cumulative surface-to-k ratio.
   - The carry holds `sigma_new` from previous level's iteration.
   - Per WKB, amplitude grows as `sqrt(rho_(k+1)/rho_k)` between adjacent levels.
   - The code multiplies by `sqrt(rho_sfc/rho_k)` at every step, which compounds across levels.
   - In absence of dissipation: at k=nlev-2, sigma = sigma_init * sqrt(rho_sfc/rho[nlev-2]). At k=nlev-3, sigma = previous * sqrt(rho_sfc/rho[nlev-3]) = sigma_init * sqrt(rho_sfc/rho[nlev-2]) * sqrt(rho_sfc/rho[nlev-3]). **The expected value should be sigma_init * sqrt(rho_sfc/rho[nlev-3]).**
   - Effect: amplitude growth wildly overshoots WKB; saturation cap at sigma_sat fires aggressively to mask this.

2. **`hines.py:81` — `sigma_sat = N / (m_star * rho_ratio)`**.
   - `rho_ratio` decreases with altitude (rho_sfc/rho > 1 aloft).
   - This makes `sigma_sat` smaller aloft — opposite of expected (gravity wave saturation amplitude should be larger aloft).
   - Suspect copy-paste typo; should likely be `* rho_ratio` (multiplied) or equivalent formulation.

3. **All GWD schemes — no gradient tests in `tests/unit/test_physics_gwd.py`** (verified by grep).

4. **`mcfarlane.py:75, 100` — hardcoded sharpness 20 and softmin 50** in body, not config.

5. **`mcfarlane.py:86` — `tau_0 = jnp.clip(tau_0, 0.0, 10.0)`** with hardcoded upper bound 10 (units Pa or kg/m²/s).

6. **`lindzen.py:91-100`, `mcfarlane.py:96-106`, `hines.py:85-102` — same scan-then-T-then-reverse ordering pattern**. Verify correctness in each.

7. **Frictional heating formula `dT_dt = -(u du_dt + v dv_dt) / c_pd`** in all schemes. Verify sign.

Files:
- `src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py` (90 LOC).
- `src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py` (123 LOC).
- `src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py` (126 LOC).
- `src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py` (125 LOC).
- `src/legoesm/atmosphere/physics/gravity_wave_drag/config.py`.

For each finding: file path : line range, summary, technical explanation (≤3 sentences), severity (critical|major|minor).

If no findings beyond those flagged, end with `NO ADDITIONAL SUBSTANTIVE FINDINGS`.
