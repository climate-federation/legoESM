# Adversarial review packet 1 — prognostic sea-ice skin temperature, MPAS AMIP lane

You are an independent adversarial physics + software reviewer. Find every bug,
sign error, unit inconsistency, broken-gradient pattern, conservation violation,
tuple-contract mismatch, restart-continuity break, and test-case discrepancy in
this implementation. Cite file:line. If you believe a candidate concern is NOT a
bug, say so and explain why. Be concrete and skeptical. Numerical evidence beats
verbal reasoning.

## Motivation
First ClimateEval scorecard: polar `tas` warm bias +8.6 K. The prescribed-SST
AMIP anchor pins ice-covered cells at the CONSTANT `cfg.T_ice = 271.35 K`
year-round via `blend_surface_temperature(sst, sic, T_ice) = sic*T_ice +
(1-sic)*sst`. Real central-Arctic winter skin is ~245-250 K. This change makes
the ice component of that blend a prognostic Semtner (1976) zero-layer skin.

All changes are UNCOMMITTED working tree on `mpas-stability-campaign` @ 649d8480e.

## The physics (Semtner 1976 zero-layer + slab inertia)
Conductive flux `k_i/h * (T_f - T_s)` through a climatological ice slab toward
the seawater freezing point `T_f`, plus the slab's HALF thermal inertia
`C = rho_ice * c_pi * h / 2`, turning the diagnostic zero-layer balance into a
stably integrable prognostic skin:

    C dT_s/dt = F_net_down(atm) + (k_i/h)(T_f - T_s)

Constants (all pre-existing in `legoesm.constants`): `k_ice_default=2.04`
(Untersteiner), `rho_ice=917`, `c_pi=2106`, `T_freeze_ocean=271.35`.
At h=2 m: C_areal=1.931e6 J/m^2/K, g_cond=k/h=1.02 W/m^2/K, slab timescale
C/g = 21.9 days (~3 weeks). Zero-layer winter equilibrium
`T_s = T_f + F_net_down*h/k`; at F=-25 W/m^2 -> 24.5 K below freezing (~246.8 K).

## Sign convention trace (I already verified these — challenge them)
`_sfc_diag` tuple contract (built in
`atmosphere/dynamics/gcm/primitive_eq_mpas.py:868-883`, consumed by the CMOR
feed at `model_driver.py:5534-5547`):

    slot 0 = sw_net_sfc  [W/m^2, + INTO surface (down)]
    slot 1 = lw_net_sfc  [W/m^2, + INTO surface (down); NET, so <0 in polar night]
    slot 2 = precip
    slot 3 = lw_up_toa, slot 4 = sw_up_toa, slot 5 = sw_down_toa
    slot 6 = shflx_sfc   [W/m^2, + UPWARD out of surface]
    slot 7 = lhflx_sfc   [W/m^2, + UPWARD out of surface]

Turbulence-scheme sign convention (uniform across all schemes, e.g.
`turbulence/surface_layer.py:47,109`: `shflx = rho c_pd Ch |V| (T_sfc - T)`,
"Positive upward"; `holtslag_boville.py:92`, `louis.py:84`, `tke.py:101`,
`ysu.py:138`, `edmf.py:115`: "shflx > 0 upward, lhflx > 0 upward").

Ice-skin assembly (`model_driver.py:6688-6698`):
`F_net_down = sw_net + lw_net - shflx - lhflx` = (down radiative) - (up
turbulent) = net downward energy gain of the skin. Winter polar night:
lw_net<0 (net LW loss), shflx/lhflx typically small or downward under a cold
skin -> F_net_down<0 -> skin cools. STABILIZING feedback: skin cools ->
d(sigma T^4)/dT>0 so upward LW falls -> lw_net rises (less negative); shflx =
rho c_pd Ch |V| (T_sfc - T_air) falls -> both oppose further cooling.

## Independent numerical probes (this reviewer ran these; CPU, x64)
`scripts/tmp/_probe_ice_skin.py`:
1. FD Jacobian of the helper w.r.t. F_net vs `jax.jacrev`:
   max|autodiff - FD| = 8.2e-11; diag dT_new/dF = 0.04279 = analytic r/(1+rg)
   exactly; all rows non-zero. Differentiable, no dead gradient.
2. Lagged coupled daily iteration with linear surface-flux feedback
   F_net(T)=F0 - lam*(T-T_ref), amplification |1 - r*lam|/(1 + r*g):
   - lam=5  -> amp=0.742, converges monotone to 255.3 K, no tail oscillation
   - lam=10 -> amp=0.528, converges monotone to 257.4 K, no oscillation
   - lam=20 -> amp=0.101, converges monotone to 258.6 K, no oscillation
   - lam=47 -> amp=1.055, oscillates but the melt cap PINS it at 271.35 (fails
     safe to old behavior, not divergence)
   - lam=60 -> amp=1.611, likewise capped at 271.35
   Realistic surface feedback lam ~ 10 W/m^2/K (LW ~3.5 + SH ~6.5) -> amp 0.53,
   monotone convergence. Conductive term is backward-Euler (unconditionally
   stable); only the LAGGED atmospheric feedback is explicit, stable for
   realistic lam at daily dt.
3. jit parity: max|diff|=0; vmap (ensemble axis) parity: max|diff|=0.

## Baseline tests (all green, via srun on the interactive alloc)
253 passed: `test_mpas_ice_skin.py` (18), `test_run_amip_cli.py`,
`test_mpas_land_boundary.py`, `test_mpas_qv_smoothing.py`,
`test_mpas_cmor_flux_feed.py`.

---

## SOURCE — the pure helper (`packages/tools/legoesm/forcing/surface_utils.py`)
```python
_ICE_SKIN_FLOOR_K = 185.0

def prognostic_ice_skin_temperature(
    T_skin, F_net_down_W_m2, sic, dt_s, h_ice_m,
    T_freeze_K=constants.T_freeze_ocean,
    k_ice_W_m_K=constants.k_ice_default,
):
    C_areal = 0.5 * constants.rho_ice * constants.c_pi * h_ice_m  # [J/m^2/K]
    g_cond = k_ice_W_m_K / h_ice_m                                # [W/m^2/K]
    r = dt_s / C_areal
    T_new = (T_skin + r * (F_net_down_W_m2 + g_cond * T_freeze_K)) \
        / (1.0 + r * g_cond)
    T_new = jnp.clip(T_new, _ICE_SKIN_FLOOR_K, T_freeze_K)
    return jnp.where(sic > 0.0, T_new, T_freeze_K)
```
`blend_surface_temperature(sst, sic, T_ice) = sic*T_ice + (1-sic)*sst`
(unchanged) — with the feature ON, `T_ice` is the (nCells,) skin array.

## SOURCE — config (`packages/coupler/legoesm/driver/config.py`)
New `ExperimentConfig` fields:
```python
mpas_ice_skin_prognostic: bool = False    # byte-identical default
mpas_ice_thickness_m: float = 2.0
```
`validate_strict` (MPAS-lane only): refuses on non-MPAS discretizations
("has its own surface/ice tiles"); refuses `radiation == "none"` (no fluxes to
integrate); refuses a non-default thickness without the boolean gate (inert);
bounds thickness to [0.1, 10] finite.

## SOURCE — driver integration (`packages/coupler/legoesm/driver/model_driver.py`)
`_run_mpas` setup (~6149):
```python
_ice_skin_on = bool(getattr(cfg, "mpas_ice_skin_prognostic", False))
if _ice_skin_on and not _sst_forcing:
    raise ValueError("... needs the prescribed SST/SIC surface forcing ...")
if _ice_skin_on and self._voronoi_layout is not None:
    raise ValueError("... not wired for the distributed Voronoi (MPI) lane ...")
```
Seed / checkpoint-resume (~6169):
```python
if _ice_skin_on:
    from legoesm.forcing.surface_utils import prognostic_ice_skin_temperature
    _h_ice = float(cfg.mpas_ice_thickness_m)
    _staged_skin = self._carry_aux.get("ice_T_skin") if isinstance(self._carry_aux, dict) else None
    if _staged_skin is not None:
        _skin = jnp.asarray(_staged_skin).reshape(-1)
        if _skin.shape != (_ncell,):
            raise ValueError(f"checkpoint ice_T_skin shape {_skin.shape} != (nCells={_ncell},) — mesh mismatch.")
        self._ice_T_skin = _skin
    else:
        self._ice_T_skin = jnp.full(_ncell, constants.T_freeze_ocean)
```
Blend closure `_compute_T_sfc(day)` (~6215):
```python
_sst, _sic = self.get_sst_sic(day)
_ice_component = self._ice_T_skin if _ice_skin_on else _T_ice
_ts = blend_surface_temperature(jnp.asarray(_sst), jnp.asarray(_sic), _ice_component).reshape(-1)
# ... optional land-lapse correction applied AFTER the blend ...
```
Daily-boundary advance — fires ONCE per canonical forcing day (inside
`if _fd_int != _last_force_day:`, `_last_force_day` set at :6727), dt_s=86400 s
(~6674):
```python
if _ice_skin_on:
    _sd = getattr(self.model, "_sfc_diag", None)
    def _slot(i):
        return (_sd[i].data if (_sd is not None and len(_sd) > i and _sd[i] is not None) else None)
    _swn, _lwn = _slot(0), _slot(1)
    _shf, _lhf = _slot(6), _slot(7)
    if _swn is not None and _lwn is not None:            # skip day 0 (no fluxes yet)
        _f_net = jnp.asarray(_swn).reshape(-1) + jnp.asarray(_lwn).reshape(-1)
        if _shf is not None: _f_net = _f_net - jnp.asarray(_shf).reshape(-1)
        if _lhf is not None: _f_net = _f_net - jnp.asarray(_lhf).reshape(-1)
        _, _sic_now = self.get_sst_sic(_force_day_canonical)
        self._ice_T_skin = prognostic_ice_skin_temperature(
            self._ice_T_skin, _f_net, jnp.asarray(_sic_now).reshape(-1),
            dt_s=86400.0, h_ice_m=_h_ice)
_forcing_daily["T_sfc"] = _compute_T_sfc(_force_day_canonical)
```
Checkpoint SAVE (~4257) adds `_save["ice_T_skin"] = np.asarray(self._ice_T_skin)`
only when `getattr(self, "_ice_T_skin", None) is not None` (absent on runs
without the feature -> byte-identical restart). LOAD (~4835) drops any prior
staging then stages the checkpoint's `ice_T_skin` into `_carry_aux` for the
`_run_mpas` seed overlay (mirrors the `physstate_*` stale-drop precedent).

## SOURCE — CLI (`scripts/run/run_amip.py`)
`--mpas-ice-skin-prognostic` (BooleanOptionalAction, default False),
`--mpas-ice-thickness-m` (float, default None -> config default 2.0), wired
into `build_config_from_args`. Round-trip tested.

---

## Attack surface (explicit — please probe each)
1. **Energy-balance sign end-to-end.** Is `F_net_down = sw_net + lw_net - shflx
   - lhflx` correct given slots 0/1 are +into-surface and 6/7 are +up? A flipped
   turbulent sign WARMS the winter skin (masking the fix). Verify against the
   turbulence-scheme convention cited above.
2. **Lagged-flux feedback loop.** Skin cools -> physics recomputes fluxes at the
   new anchor next day. Converge or oscillate? See probe #2. Is my linear
   stability argument (amp = |1-r*lam|/(1+r*g)) sound, and is the implicit
   conductive damping enough at daily dt for realistic lam?
3. **Spin-up transient.** Fresh run seeds at T_f (271.35) and relaxes over ~3
   weeks; months 1-2 biased warm toward old behavior. Scorecard uses months
   3-12. Acceptable? Any faster-seed argument?
4. **Melt-season / sic retreat.** Melt cap at T_f sheds energy with NO melt
   bookkeeping (fine for prescribed ice). Does cap+open-water-snap interact
   badly as sic seasonally retreats/advances? Flickering marginal cells
   (`sic > 0.0` strict)?
5. **Blend broadcasting.** Per-cell (nCells,) skin array where a scalar T_ice was
   — shape/dtype correctness through `blend_surface_temperature` and the
   subsequent land-lapse correction (order: blend THEN lapse).
6. **Restart continuity.** (a) Runs WITHOUT the feature: `ice_T_skin` absent from
   their checkpoints, byte-identical? (b) Loading an OLD checkpoint with the
   feature ON seeds fresh at T_f (staged skin absent) — documented/correct? (c)
   The save-before-run laundering guard: could a stale `_ice_T_skin` from a prior
   run on a reused driver leak into a checkpoint? (d) Does the once-per-canonical-
   day gate plus `_sfc_diag=None` at a fresh link's first boundary double-count or
   drop a daily advance across a 12h chain link?
7. **`_compute_T_sfc` closure reads `self._ice_T_skin` at call time** (eager daily
   call). Any JIT staleness (is the daily blend eager, outside the compiled MPAS
   step)?
8. **Global scalar h_ice** vs hemispheric asymmetry (Arctic ~2 m, Antarctic ~1 m)
   — documented limitation or latent bug?
9. **Purity / differentiability** of the helper (clip + where): dead-gradient
   regions? The daily advance mutates `self._ice_T_skin` (host-side eager state)
   — does that break the compiled-segment autodiff path, or is it safely outside
   the traced step?

Please give: confirmed bugs (file:line + why + fix), refuted concerns (why not a
bug), and any NEW issues I did not list. If you find nothing substantive, say so
and justify each candidate.
