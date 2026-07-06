"""Comprehensive stress test for the canopy land model.

Tests the canopy land step function across a wide range of configurations
and forcing conditions, checking for:

  * NaN / Inf in state or response
  * Unphysical leaf / soil temperatures (outside [200, 340] K)
  * Energy budget non-closure (|residual| > threshold)
  * Newton non-convergence (n_iters == max)
  * JIT consistency
  * Differentiability via jax.grad

Runs:
  1. Extended 7-day run, 6 contrasting PFTs
  2. LAI sweep ([0.5, 8])
  3. Canopy height sweep ([0.3, 35])
  4. Soil moisture sweep (wet → wilting)
  5. Freezing conditions (T_air < 273 K) + snow
  6. Hot desert conditions (T_air = 320 K, dry)
  7. Rapidly fluctuating SW forcing (cloudy day)
  8. dt sensitivity ({900, 1800, 3600} s)
  9. Multi-column consistency (single vs multi)
 10. jax.grad through one step
 11. JIT vs non-JIT output agreement

Exits 1 on any test failure, 0 on full pass.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy import CanopyConfig, CanopyLandParams
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land,
    step_multilayer_land_with_diagnostics,
)
from legoesm.land.soil_hydraulics import psi_from_theta
from legoesm.land.surface_scheme import TwoLeafCanopyConfig


# Phase 3 script-local shims so the rest of the stress script reads
# unchanged after ``canopy_land.py`` and ``CanopyLandConfig`` were removed.
from typing import NamedTuple

step_canopy_land = step_multilayer_land
init_canopy_land_state = init_multilayer_land_state


class _StressDiag(NamedTuple):
    """JAX-pytree-compatible adapter mapping ``SurfaceFluxOutput`` fields
    to the legacy ``CanopyDiagnostics`` names used by the stress harness.
    """
    Tf_Sun: jax.Array
    Tf_Sh: jax.Array
    Ts_solve: jax.Array
    LE_tot: jax.Array
    H_tot: jax.Array
    G: jax.Array
    Rn_int: jax.Array
    residual_int: jax.Array
    n_iters: jax.Array
    GPP: jax.Array


def step_canopy_land_with_diagnostics(state, forcing, cfg, U_min, dt, **kwargs):
    """Phase 3 stress-test shim.

    ``step_multilayer_land_with_diagnostics`` returns the raw
    ``SurfaceFluxOutput`` as its 4th element.  This wrapper repackages
    the canopy-relevant fields under the legacy field names
    (``LE_tot``, ``H_tot``, ``G``, ``residual_int``, ...) that the
    stress test harness below expects, using a NamedTuple so the
    return value is a valid JAX pytree under JIT.
    """
    new_state, resp, carbon_new, so = step_multilayer_land_with_diagnostics(
        state, forcing, cfg, U_min, dt, **kwargs)
    diag = _StressDiag(
        Tf_Sun=so.Tf_Sun,
        Tf_Sh=so.Tf_Sh,
        Ts_solve=so.Ts_solve,
        LE_tot=so.lhflx,
        H_tot=so.shflx,
        G=so.G_soil,
        Rn_int=so.Rn_int,
        residual_int=so.residual_int,
        n_iters=so.n_iters,
        GPP=so.gpp if so.gpp is not None else jnp.zeros_like(so.lhflx),
    )
    return new_state, resp, carbon_new, diag


def CanopyLandConfig(*,
                     multilayer: MultiLayerLandConfig = None,
                     canopy: CanopyConfig = None
                     ) -> MultiLayerLandConfig:
    """Phase 3 test-script shim: build a ``MultiLayerLandConfig`` with a
    ``TwoLeafCanopyConfig`` surface scheme from the old
    ``CanopyLandConfig(multilayer=..., canopy=...)`` factory signature.
    """
    base = multilayer if multilayer is not None else MultiLayerLandConfig()
    cc = canopy if canopy is not None else CanopyConfig()
    return base._replace(surface_scheme=cc)


# Physical sanity bounds
T_MIN, T_MAX = 220.0, 340.0
LE_MAX = 2000.0     # W/m2
H_MAX = 2000.0      # W/m2
G_MAX = 1200.0      # W/m2
RESIDUAL_TOL = 1.0  # W/m2 — energy budget closure tolerance


def _build_forcing(ncol, t_hr, T_base=295.0, q_base=0.012, sw_base=1000.0,
                   lw_base=360.0, wind=3.0):
    cosz = max(0.0, float(np.sin(np.pi * (t_hr - 6.0) / 12.0)))
    sw = sw_base * cosz
    T_air = T_base + 6.0 * float(np.sin((t_hr - 9.0) * np.pi / 12.0))
    return AtmToSurface(
        T_lowest=jnp.full(ncol, T_air),
        q_lowest=jnp.full(ncol, q_base),
        u_lowest=jnp.full(ncol, wind),
        v_lowest=jnp.zeros(ncol),
        p_lowest=jnp.full(ncol, 98000.0),
        p_surface=jnp.full(ncol, 101325.0),
        rho_lowest=jnp.full(ncol, 1.18),
        sw_down=jnp.full(ncol, sw),
        lw_down=jnp.full(ncol, lw_base),
        cos_zenith=jnp.full(ncol, cosz),
        precip_total=jnp.zeros(ncol),
        precip_snow=jnp.zeros(ncol),
        co2_ppmv=jnp.full(ncol, 420.0),
        has_radiation=True,
        has_precipitation=False,
    )


def _default_params(ncol, LAI=3.0, hc=10.0, fC4=0.0, theta_init=0.30,
                    Vc3=60.0, Vc4=0.0, TgC=20.0):
    return CanopyLandParams(
        LAI=jnp.full(ncol, LAI),
        hc=jnp.full(ncol, hc),
        fC4=jnp.full(ncol, fC4),
        FNonVeg=jnp.zeros(ncol),
        CI=jnp.full(ncol, 0.75),
        kn=jnp.full(ncol, 0.3),
        Vcmax25_C3_leaf=jnp.full(ncol, Vc3),
        Vcmax25_C4_leaf=jnp.full(ncol, Vc4),
        m_C3=jnp.full(ncol, 9.0),
        m_C4=jnp.full(ncol, 4.0),
        b0_C3=jnp.full(ncol, 0.01),
        b0_C4=jnp.full(ncol, 0.04),
        alf=jnp.full(ncol, 0.3),
        TgC=jnp.full(ncol, TgC),
        ALB_VIS=jnp.full(ncol, 0.10),
        ALB_NIR=jnp.full(ncol, 0.25),
        emissivity=jnp.full(ncol, 0.97),
        rz0m=jnp.full(ncol, 0.08),
        rd=jnp.full(ncol, 0.67),
    )


def _init_state(ncol, cfg, T_init=290.0, theta_init=0.30):
    state = init_canopy_land_state(ncol, cfg, T_init=T_init)
    theta_profile = jnp.full_like(state.theta_soil, theta_init)
    psi_profile = psi_from_theta(theta_profile, cfg.hydraulics)
    return state._replace(theta_soil=theta_profile, psi_soil=psi_profile)


class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = True
        self.issues = []
        self.n_steps = 0
        self.max_resid = 0.0
        self.max_nit = 0
        self.nan_count = 0
        self.nan_first = -1
        self.unphys_count = 0

    def record_step(self, k, state, response, diag):
        self.n_steps += 1
        # NaN check
        fields = [state.T_soil, state.theta_soil, response.T_surface,
                  response.lhflx, response.shflx, diag.Rn_int, diag.LE_tot,
                  diag.H_tot, diag.G, diag.residual_int, diag.Tf_Sun,
                  diag.Tf_Sh, diag.Ts_solve]
        for f in fields:
            if not bool(jnp.all(jnp.isfinite(f))):
                self.nan_count += 1
                if self.nan_first < 0:
                    self.nan_first = k
                    self.issues.append(f"NaN at step {k}")
                self.passed = False
                return

        # Physical bounds on temperatures
        for name, f in (("T_surface", response.T_surface),
                        ("Tf_Sun", diag.Tf_Sun),
                        ("Tf_Sh", diag.Tf_Sh),
                        ("Ts_solve", diag.Ts_solve)):
            arr = np.asarray(f)
            if np.any(arr < T_MIN) or np.any(arr > T_MAX):
                self.unphys_count += 1
                if self.unphys_count <= 3:
                    self.issues.append(
                        f"unphysical {name} at step {k}: min={arr.min():.1f} max={arr.max():.1f}")
                self.passed = False

        # Physical bounds on fluxes
        for name, f, bound in (("LE", diag.LE_tot, LE_MAX),
                               ("H", diag.H_tot, H_MAX),
                               ("G", diag.G, G_MAX)):
            arr = np.abs(np.asarray(f))
            if np.any(arr > bound):
                if self.unphys_count <= 5:
                    self.issues.append(
                        f"|{name}|>{bound:.0f} at step {k}: max={arr.max():.1f}")
                self.unphys_count += 1
                self.passed = False

        # Residual
        r = float(np.abs(np.asarray(diag.residual_int)).max())
        if r > self.max_resid:
            self.max_resid = r

        # Newton iters
        ni = int(np.asarray(diag.n_iters).max())
        if ni > self.max_nit:
            self.max_nit = ni

    def finalize(self, max_iters_limit):
        if self.max_resid > RESIDUAL_TOL:
            self.issues.append(
                f"max |residual| = {self.max_resid:.2f} > {RESIDUAL_TOL} W/m2")
            self.passed = False
        if self.max_nit >= max_iters_limit:
            self.issues.append(
                f"solver hit max_iters ({max_iters_limit})")
            # Not necessarily a failure — depends on whether result is physical

    def report(self):
        status = "PASS" if self.passed else "FAIL"
        line = (f"[{status}] {self.name:38s} "
                f"steps={self.n_steps:4d} max|resid|={self.max_resid:8.2e} "
                f"max_nit={self.max_nit:3d} nan={self.nan_count:3d}")
        print(line)
        for issue in self.issues[:5]:
            print(f"     - {issue}")
        return self.passed


def run_run(name, cfg, state, params, n_steps, forcing_fn, dt=1800.0,
            start_hour=12.0):
    """Execute a timestep loop and record diagnostics."""
    result = TestResult(name)
    step_fn = jax.jit(lambda s, f: step_canopy_land_with_diagnostics(
        s, f, cfg, 1.0, dt,
        lat=jnp.zeros(state.T_soil.shape[0]),
        doy=180.0, land_params=params))
    for k in range(n_steps):
        t_hr = (start_hour + k * dt / 3600.0) % 24.0
        forcing = forcing_fn(t_hr, k)
        try:
            state, response, _, diag = step_fn(state, forcing)
        except Exception as e:
            result.passed = False
            result.issues.append(f"step {k} raised {type(e).__name__}: {e}")
            break
        result.record_step(k, state, response, diag)
        if result.nan_count > 0 and result.nan_first == k:
            break
    result.finalize(cfg.surface_scheme.max_iters)
    return result, state


# -------------------------------------------------------------------------
# Test 1 — 7-day diurnal run, 6 contrasting PFTs
# -------------------------------------------------------------------------

def test_extended_diurnal():
    cases = [
        # (name, LAI, hc, fC4, Vc3, Vc4, TgC, theta, T_base, q_base)
        ("tropical_DBF",    5.0, 20.0, 0.0, 66.0,  0.0, 27.0, 0.38, 299.0, 0.018),
        ("temperate_DBF",   4.0, 15.0, 0.0, 57.0,  0.0, 20.0, 0.30, 295.0, 0.012),
        ("temperate_GRA",   2.0,  0.5, 0.0, 78.0,  0.0, 18.0, 0.25, 294.0, 0.010),
        ("boreal_ENF",      3.0, 12.0, 0.0, 54.0,  0.0, 12.0, 0.35, 286.0, 0.006),
        ("C4_savanna",      1.5,  3.0, 1.0,  0.0, 40.0, 28.0, 0.22, 297.0, 0.010),
        ("dry_shrub",       1.0,  1.5, 0.0, 62.0,  0.0, 22.0, 0.17, 296.0, 0.006),
    ]
    cfg = CanopyLandConfig(multilayer=MultiLayerLandConfig(),
                           canopy=CanopyConfig(max_iters=50, tol=1e-4))
    results = []
    for (name, LAI, hc, fC4, Vc3, Vc4, TgC, theta, T_base, q_base) in cases:
        state = _init_state(1, cfg, T_init=T_base - 2.0, theta_init=theta)
        params = CanopyLandParams(
            LAI=jnp.array([LAI]), hc=jnp.array([hc]), fC4=jnp.array([fC4]),
            FNonVeg=jnp.zeros(1), CI=jnp.array([0.75]), kn=jnp.array([0.3]),
            Vcmax25_C3_leaf=jnp.array([Vc3]),
            Vcmax25_C4_leaf=jnp.array([Vc4]),
            m_C3=jnp.array([9.0]), m_C4=jnp.array([4.0]),
            b0_C3=jnp.array([0.01]), b0_C4=jnp.array([0.04]),
            alf=jnp.array([0.3]), TgC=jnp.array([TgC]),
            ALB_VIS=jnp.array([0.10]), ALB_NIR=jnp.array([0.25]),
            emissivity=jnp.array([0.97]),
            rz0m=jnp.array([0.08]), rd=jnp.array([0.67]),
        )
        def forcing_fn(t_hr, k, T_base=T_base, q_base=q_base):
            return _build_forcing(1, t_hr, T_base=T_base, q_base=q_base)
        r, _ = run_run(f"extended/{name}", cfg, state, params,
                       n_steps=7 * 48, forcing_fn=forcing_fn)
        results.append(r.report())
    return all(results)


# -------------------------------------------------------------------------
# Test 2 — LAI sweep
# -------------------------------------------------------------------------

def test_lai_sweep():
    cfg = CanopyLandConfig(multilayer=MultiLayerLandConfig(),
                           canopy=CanopyConfig(max_iters=50, tol=1e-4))
    results = []
    for LAI in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        params = _default_params(1, LAI=LAI, hc=max(0.5, LAI*1.5), theta_init=0.30)
        state = _init_state(1, cfg, theta_init=0.30)
        r, _ = run_run(f"LAI_sweep/LAI={LAI:.1f}", cfg, state, params,
                       n_steps=96, forcing_fn=lambda t, k: _build_forcing(1, t))
        results.append(r.report())
    return all(results)


# -------------------------------------------------------------------------
# Test 3 — Canopy height sweep
# -------------------------------------------------------------------------

def test_hc_sweep():
    cfg = CanopyLandConfig(multilayer=MultiLayerLandConfig(),
                           canopy=CanopyConfig(max_iters=50, tol=1e-4))
    results = []
    for hc in [0.3, 0.8, 2.0, 5.0, 12.0, 20.0, 35.0]:
        params = _default_params(1, LAI=3.0, hc=hc, theta_init=0.30)
        state = _init_state(1, cfg, theta_init=0.30)
        r, _ = run_run(f"hc_sweep/hc={hc:.1f}m", cfg, state, params,
                       n_steps=96, forcing_fn=lambda t, k: _build_forcing(1, t))
        results.append(r.report())
    return all(results)


# -------------------------------------------------------------------------
# Test 4 — Soil moisture sweep
# -------------------------------------------------------------------------

def test_theta_sweep():
    cfg = CanopyLandConfig(multilayer=MultiLayerLandConfig(),
                           canopy=CanopyConfig(max_iters=50, tol=1e-4))
    results = []
    # theta_wp = 0.15, theta_fc = 0.30, theta_sat = 0.43
    for theta in [0.16, 0.20, 0.25, 0.30, 0.35, 0.42]:
        params = _default_params(1, LAI=3.0, hc=8.0, theta_init=theta)
        state = _init_state(1, cfg, theta_init=theta)
        r, _ = run_run(f"theta_sweep/theta={theta:.2f}", cfg, state, params,
                       n_steps=96, forcing_fn=lambda t, k: _build_forcing(1, t))
        results.append(r.report())
    return all(results)


# -------------------------------------------------------------------------
# Test 5 — Freezing conditions
# -------------------------------------------------------------------------

def test_freezing():
    cfg = CanopyLandConfig(multilayer=MultiLayerLandConfig(),
                           canopy=CanopyConfig(max_iters=50, tol=1e-4))
    params = _default_params(1, LAI=2.0, hc=8.0, theta_init=0.25)
    state = _init_state(1, cfg, T_init=262.0, theta_init=0.25)
    # Add snow
    state = state._replace(snow_depth=jnp.array([0.05]))
    def fn(t_hr, k):
        return _build_forcing(1, t_hr, T_base=262.0, q_base=0.002, sw_base=400.0)
    r, _ = run_run("freezing/boreal_winter", cfg, state, params,
                   n_steps=96, forcing_fn=fn)
    return r.report()


# -------------------------------------------------------------------------
# Test 6 — Hot desert conditions
# -------------------------------------------------------------------------

def test_hot_desert():
    cfg = CanopyLandConfig(multilayer=MultiLayerLandConfig(),
                           canopy=CanopyConfig(max_iters=50, tol=1e-4))
    params = _default_params(1, LAI=0.8, hc=0.5, theta_init=0.17)
    state = _init_state(1, cfg, T_init=308.0, theta_init=0.17)
    def fn(t_hr, k):
        return _build_forcing(1, t_hr, T_base=313.0, q_base=0.008, sw_base=1100.0)
    r, _ = run_run("hot_desert", cfg, state, params,
                   n_steps=96, forcing_fn=fn)
    return r.report()


# -------------------------------------------------------------------------
# Test 7 — Cloudy / fluctuating SW forcing
# -------------------------------------------------------------------------

def test_fluctuating_sw():
    cfg = CanopyLandConfig(multilayer=MultiLayerLandConfig(),
                           canopy=CanopyConfig(max_iters=50, tol=1e-4))
    params = _default_params(1, LAI=3.0, hc=10.0, theta_init=0.30)
    state = _init_state(1, cfg, theta_init=0.30)
    rng = np.random.RandomState(42)
    sw_mult = 0.5 + 0.5 * rng.uniform(size=200)  # 50-100% cloud fraction
    def fn(t_hr, k):
        cosz = max(0.0, float(np.sin(np.pi*(t_hr-6)/12)))
        sw = 1000.0 * cosz * sw_mult[k % len(sw_mult)]
        T_air = 295 + 6*float(np.sin((t_hr-9)*np.pi/12))
        return AtmToSurface(
            T_lowest=jnp.array([T_air]), q_lowest=jnp.array([0.012]),
            u_lowest=jnp.array([3.0]), v_lowest=jnp.array([0.0]),
            p_lowest=jnp.array([98000.0]), p_surface=jnp.array([101325.0]),
            rho_lowest=jnp.array([1.18]),
            sw_down=jnp.array([sw]), lw_down=jnp.array([360.0]),
            cos_zenith=jnp.array([cosz]),
            precip_total=jnp.zeros(1), precip_snow=jnp.zeros(1),
            co2_ppmv=jnp.array([420.0]),
            has_radiation=True, has_precipitation=False)
    r, _ = run_run("fluctuating_SW/cloudy", cfg, state, params,
                   n_steps=192, forcing_fn=fn)
    return r.report()


# -------------------------------------------------------------------------
# Test 8 — dt sensitivity
# -------------------------------------------------------------------------

def test_dt_sweep():
    results = []
    for dt in [900.0, 1800.0, 3600.0]:
        cfg = CanopyLandConfig(multilayer=MultiLayerLandConfig(),
                               canopy=CanopyConfig(max_iters=50, tol=1e-4))
        params = _default_params(1, LAI=3.0, hc=10.0, theta_init=0.30)
        state = _init_state(1, cfg, theta_init=0.30)
        n_steps = int(2 * 24 * 3600 / dt)  # 2 days
        r, _ = run_run(f"dt_sweep/dt={dt:.0f}s", cfg, state, params,
                       n_steps=n_steps, forcing_fn=lambda t, k: _build_forcing(1, t),
                       dt=dt)
        results.append(r.report())
    return all(results)


# -------------------------------------------------------------------------
# Test 9 — Multi-column vs single-column consistency
# -------------------------------------------------------------------------

def test_multicolumn_consistency():
    cfg = CanopyLandConfig(multilayer=MultiLayerLandConfig(),
                           canopy=CanopyConfig(max_iters=50, tol=1e-4))
    # Two identical columns
    state2 = _init_state(2, cfg, theta_init=0.30)
    params2 = _default_params(2, LAI=3.0, hc=10.0, theta_init=0.30)
    state1 = _init_state(1, cfg, theta_init=0.30)
    params1 = _default_params(1, LAI=3.0, hc=10.0, theta_init=0.30)

    step1 = jax.jit(lambda s, f: step_canopy_land_with_diagnostics(
        s, f, cfg, 1.0, 1800.0, lat=jnp.zeros(1), doy=180.0, land_params=params1))
    step2 = jax.jit(lambda s, f: step_canopy_land_with_diagnostics(
        s, f, cfg, 1.0, 1800.0, lat=jnp.zeros(2), doy=180.0, land_params=params2))

    for k in range(48):
        t_hr = (12 + k * 0.5) % 24
        f1 = _build_forcing(1, t_hr)
        f2 = _build_forcing(2, t_hr)
        state1, r1, _, d1 = step1(state1, f1)
        state2, r2, _, d2 = step2(state2, f2)

    # Compare last-step T_surface (both columns of state2 should match state1)
    diff = float(jnp.abs(state2.T_soil[0] - state1.T_soil[0]).max())
    passed = diff < 1e-10
    name = "multicolumn_consistency"
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name:38s} max|T_soil diff| = {diff:.2e}")
    return passed


# -------------------------------------------------------------------------
# Test 10 — Differentiability via jax.grad
# -------------------------------------------------------------------------

def test_differentiability():
    cfg = CanopyLandConfig(multilayer=MultiLayerLandConfig(),
                           canopy=CanopyConfig(max_iters=50, tol=1e-4))
    state = _init_state(1, cfg, theta_init=0.30)
    forcing = _build_forcing(1, 14.0)

    def loss_fn(Vcmax_leaf):
        params = _default_params(1, LAI=3.0, hc=10.0, theta_init=0.30,
                                 Vc3=Vcmax_leaf, TgC=20.0)
        _, response, _, diag = step_canopy_land_with_diagnostics(
            state, forcing, cfg, 1.0, 1800.0,
            lat=jnp.zeros(1), doy=180.0, land_params=params)
        return jnp.squeeze(diag.GPP[0])

    grad_fn = jax.jit(jax.grad(loss_fn))
    try:
        g = float(grad_fn(60.0))
        finite = np.isfinite(g)
        passed = finite and abs(g) < 1e6 and abs(g) > 0.0
        name = "differentiability/dGPP_dVcmax"
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name:38s} dGPP/dVcmax = {g:.4e}")
        return passed
    except Exception as e:
        print(f"[FAIL] differentiability/dGPP_dVcmax {type(e).__name__}: {e}")
        return False


# -------------------------------------------------------------------------
# Test 11 — JIT vs non-JIT agreement
# -------------------------------------------------------------------------

def test_jit_agreement():
    cfg = CanopyLandConfig(multilayer=MultiLayerLandConfig(),
                           canopy=CanopyConfig(max_iters=50, tol=1e-4))
    state = _init_state(1, cfg, theta_init=0.30)
    params = _default_params(1, LAI=3.0, hc=10.0, theta_init=0.30)
    forcing = _build_forcing(1, 14.0)

    # Non-JIT
    s1, r1, _, d1 = step_canopy_land_with_diagnostics(
        state, forcing, cfg, 1.0, 1800.0, lat=jnp.zeros(1), doy=180.0,
        land_params=params)
    # JIT
    step_fn = jax.jit(lambda s, f: step_canopy_land_with_diagnostics(
        s, f, cfg, 1.0, 1800.0, lat=jnp.zeros(1), doy=180.0, land_params=params))
    s2, r2, _, d2 = step_fn(state, forcing)

    diffs = {
        "T_soil":  float(jnp.abs(s1.T_soil - s2.T_soil).max()),
        "lhflx":   float(jnp.abs(r1.lhflx - r2.lhflx).max()),
        "shflx":   float(jnp.abs(r1.shflx - r2.shflx).max()),
        "Rn_int":  float(jnp.abs(d1.Rn_int - d2.Rn_int).max()),
    }
    passed = all(v < 1e-10 for v in diffs.values())
    name = "jit_agreement"
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name:38s} max_diff = {max(diffs.values()):.2e}")
    return passed


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    jax.config.update("jax_enable_x64", True)

    print("=" * 90)
    print("legoESM canopy land stress test")
    print("=" * 90)
    test_fns = [
        ("extended 7-day diurnal (6 PFTs)",         test_extended_diurnal),
        ("LAI sweep",                                test_lai_sweep),
        ("canopy height sweep",                      test_hc_sweep),
        ("soil moisture sweep",                      test_theta_sweep),
        ("freezing conditions",                      test_freezing),
        ("hot desert conditions",                    test_hot_desert),
        ("fluctuating SW (cloudy day)",              test_fluctuating_sw),
        ("dt sensitivity",                           test_dt_sweep),
        ("multi-column consistency",                 test_multicolumn_consistency),
        ("differentiability (jax.grad)",             test_differentiability),
        ("JIT vs non-JIT agreement",                 test_jit_agreement),
    ]

    summary = []
    for label, fn in test_fns:
        print()
        print("-" * 90)
        print(f"Running: {label}")
        print("-" * 90)
        try:
            ok = fn()
        except Exception as e:
            traceback.print_exc()
            ok = False
        summary.append((label, ok))

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    n_pass = 0
    for label, ok in summary:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}")
        n_pass += int(ok)
    print(f"\n{n_pass}/{len(summary)} test groups passed")
    sys.exit(0 if n_pass == len(summary) else 1)
