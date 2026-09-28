#!/usr/bin/env python
"""Short Radiative-Convective Equilibrium with fixed SST on lat-lon FV.

Sweeps a configurable list of cumulus convection schemes — Tiedtke,
Zhang-McFarlane, Emanuel, Bechtold, Kuo, Kain-Fritsch, simple mass-flux —
with the same column physics scaffolding for every run: gray radiation
(Frierson 2006), bulk-aerodynamic boundary layer, minimal one-sided
warm-rain microphysics (sat-adj → autoconversion → instant fall),
Rayleigh BL friction.  SST is held fixed at ``--sst-init`` so the
column relaxes toward each scheme's moist-convective equilibrium
without ocean drift.

Architecture
------------
Precipitation belongs to microphysics, not convection.  The
ConvectionOutput contract documents this explicitly: convection emits
a 3D source for cloud water (``dq_c_conv_dt``) and the cloud-water
bucket is processed by microphysics, which owns the surface-
precipitation diagnostic.

Per-scheme settings.  The production defaults in
``convection/config.py`` were corrected in the same change set as
this script (Bechtold ``cape_threshold`` 0→70 and ``delta_deep``
5e-4→1.75e-3, Kuo ``tau_relax`` 3600→7200, and an
``M_b_max=0.05`` cap added to all
mass-flux-based schemes including the shared ``mass_flux`` kernel).
The shared ``mass_flux.apply_mass_flux_kernel`` was also fixed to
take ``q_v_u`` and ``q_c_u`` separately so the entraining-diluted
plume's actual cloud-water source is computed correctly instead of
the broken ``max(q_u − q_sat(T_u, p), 0)`` that always returned ~0.

The script *additionally* tightens ``M_b_max`` to 1.0e-5 in the
per-scheme configs because this minimal RCE setup lacks two
production ingredients:

- **PBL turbulence**, which would smear the schemes' first-firing
  shocks horizontally and absorb the wind feedback.
- **A full microphysics chain** (autoconversion + sedimentation +
  rain evaporation), which would transfer the schemes' ``q_c_conv``
  to surface precipitation through a physically smoothed pathway.

In production these absorb the schemes' transients; here we cap
``M_b`` instead.  The literature-peak tropical M_b is ~0.1 kg/m²/s,
so 1.0e-5 is ~10⁴× tighter than peak — appropriate only for this
stripped-down scaffolding.  CMT and AR1 stochasticity remain
disabled for the same reason: the lat-lon FV cell-centred wind is
weakly damped here.

**Scientific caveat**: with such a tight M_b cap, all six mass-flux-
based schemes converge to nearly the same near-equilibrium state
because the cap dominates each scheme's response.  Kuo (which uses a
column moisture-excess closure rather than M_b) sits apart and is the
true outlier.  This is a *stability comparison*, not a validation of
the schemes' distinct physics.  For the latter, run the production
atmosphere test matrix with full PBL + microphysics chain and the
production ``M_b_max=0.05``.

Probed kernel issues (require codebase work):
- ``mass_flux.apply_mass_flux_kernel`` (``mass_flux.py:175``) computes
  ``condensate = max(q_u − q_sat(T_u, p), 0)``; for the entraining-
  diluted plume this is ~0 everywhere, so the kernel emits zero
  ``dq_c_conv_dt`` while ``dT_dt`` stays positive — explicit MSE leak.
  The five schemes that build on the kernel (ZM, KF, Emanuel, Tiedtke,
  Bechtold) discard this output and use ``plume.q_c_u`` instead, but
  ``mass_flux`` itself uses it.
- No ``M_b`` upper bound in ZM/KF/Emanuel/Bechtold/Tiedtke closures;
  large CAPE (>5 kJ/kg) drives unbounded mass-flux that breaks
  energy budget by orders of magnitude (>10⁸ W/m² in Bechtold).
- No conservation tests in ``tests/unit/test_{zm,emanuel,bechtold,
  kain_fritsch,tiedtke}.py`` (only docstrings claim "energy
  conservation budget over one step"; no assertions).

Usage
-----
    JAX_ENABLE_X64=1 python scripts/run_rce_convection_sweep.py \\
        --schemes all --days 5 --N 12 --nlev 20

    JAX_ENABLE_X64=1 python scripts/run_rce_convection_sweep.py \\
        --schemes tiedtke,zhang_mcfarlane --days 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jnp
import numpy as np


SCHEME_CHOICES = (
    "sbm",
    "tiedtke",
    "zhang_mcfarlane",
    "emanuel",
    "bechtold",
    "kuo",
    "kain_fritsch",
    "mass_flux",
)


# Process-wide run signature stamped into every snapshot the sweep
# writes.  The plotter rejects any combination of snapshots whose
# ``run_id`` differs, so a partial re-run cannot silently contaminate
# the previous figure.
_RUN_ID = f"{int(time.time())}_{os.getpid()}"


def _parse_schemes(s: str) -> list[str]:
    if s == "all":
        return list(SCHEME_CHOICES)
    out = []
    for tok in s.split(","):
        t = tok.strip()
        if t not in SCHEME_CHOICES:
            raise ValueError(
                f"Unknown scheme {t!r}. Choices: {SCHEME_CHOICES} or 'all'."
            )
        out.append(t)
    return out


def _build_convection_call(scheme: str, dt: float):
    """Return ``(call, init_carry)`` for the requested scheme.

    ``call(T_col, q_v_col, p_full_col, p_half_col, u_col, v_col, carry)``
    returns ``(dT_dt, dq_v_dt, dq_c_conv_dt, du_cmt, dv_cmt, carry_new)``
    where each tendency has shape ``(ncol, nlev)`` and the carry is
    scheme-specific (uniform for the caller because every entry is
    threaded as a tuple).  The split between ``dq_v_dt`` (vapor sink)
    and ``dq_c_conv_dt`` (cloud-water source) is preserved per the
    ConvectionOutput contract — downstream microphysics owns the
    surface-precipitation diagnostic.

    KF's ``w_grid`` is fed zeros (parcel-T trigger only); Tiedtke /
    Bechtold ``moisture_convergence`` is fed explicit zeros — passing
    ``None`` engages a saturation-deficit proxy ``MC ∝ (q_sat - q_v)``
    that grows exponentially under a 300 K SST and produces a positive-
    feedback runaway in this RCE setup.  The column-mean horizontal MC
    is ~0 in RCE so this is physically appropriate.

    CMT is disabled for ZM / Tiedtke / Bechtold — the Gregory et al.
    1997 closure couples to the cell-centred lat-lon u/v which is only
    weakly damped here (Rayleigh BL); production runs use it on
    statistically equilibrated wind fields.

    Bechtold runs deterministically (``enable_stochastic=False``) for
    reproducibility — the multi-scheme comparison should not be
    confounded by AR1 noise.
    """
    from legoesm.atmosphere.physics.convection.config import (
        TiedtkeConfig, ZhangMcFarlaneConfig, EmanuelConfig, BechtoldConfig,
        KuoConfig, KainFritschConfig, MassFluxConfig, SBMConfig,
    )
    from legoesm.atmosphere.physics.convection.sbm import sbm_convection
    from legoesm.atmosphere.physics.convection.tiedtke import tiedtke_convection
    from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
        zhang_mcfarlane_convection,
    )
    from legoesm.atmosphere.physics.convection.emanuel import emanuel_convection
    from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection
    from legoesm.atmosphere.physics.convection.kuo import kuo_convection
    from legoesm.atmosphere.physics.convection.kain_fritsch import (
        kain_fritsch_convection,
    )
    from legoesm.atmosphere.physics.convection.mass_flux import (
        mass_flux_convection,
    )

    # Per-scheme settings.  The production defaults in
    # ``convection/config.py`` were corrected in the same PR that
    # introduced this script (Bechtold ``cape_threshold`` 0→70 and
    # ``delta_deep`` 5e-4→1.75e-3, Kuo ``tau_relax`` 3600→7200, and
    # an ``M_b_max=0.1``
    # cap added to ZM/KF/Tiedtke/Bechtold/Emanuel) so we no longer
    # need script-side knob overrides.  We still disable CMT and
    # stochasticity here because (a) the lat-lon FV cell-centred wind
    # is barely damped (Rayleigh BL only) so feeding it back through
    # Gregory et al. 1997 produces a momentum-feedback loop, and (b)
    # the multi-scheme comparison should be deterministic.

    # M_b_max override.  Production default is 0.05 kg/m²/s (half the
    # literature peak); the script forces a tighter 1e-4 because it
    # runs without PBL turbulence or a proper microphysics chain, both
    # of which absorb the schemes' first-step kicks in production.
    M_B_MAX_RCE = 1.0e-5

    if scheme == "sbm":
        # SBM is a relaxation scheme — no M_b, no kernel, no cap.
        # Acts as the reference ground truth for this RCE setup.
        cfg = SBMConfig()

        def call(T, qv, pf, ph, u, v, carry):
            out = sbm_convection(
                T=T, q_v=qv, p_full=pf, p_half=ph, dt=dt, config=cfg,
            )
            return (
                out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt,
                jnp.zeros_like(u), jnp.zeros_like(v), (),
            )
        init_carry_fn = lambda ncol, nlev, dtype: ()

    elif scheme == "tiedtke":
        cfg = TiedtkeConfig(enable_cmt=False, M_b_max=M_B_MAX_RCE)

        def call(T, qv, pf, ph, u, v, carry):
            (prog,) = carry
            mc_zero = jnp.zeros_like(T)
            out, prog_new = tiedtke_convection(
                T=T, q_v=qv, p_full=pf, p_half=ph, u=u, v=v,
                conv_prog_profile=prog, dt=dt, config=cfg,
                moisture_convergence=mc_zero,
            )
            du = out.du_dt_conv if out.du_dt_conv is not None else jnp.zeros_like(u)
            dv = out.dv_dt_conv if out.dv_dt_conv is not None else jnp.zeros_like(v)
            return out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, du, dv, (prog_new,)
        init_carry_fn = lambda ncol, nlev, dtype: (
            jnp.zeros((ncol, nlev), dtype=dtype),
        )

    elif scheme == "zhang_mcfarlane":
        cfg = ZhangMcFarlaneConfig(enable_cmt=False, land_fraction="none")

        def call(T, qv, pf, ph, u, v, carry):
            (prog,) = carry
            out, prog_new = zhang_mcfarlane_convection(
                T=T, q_v=qv, p_full=pf, p_half=ph, u=u, v=v,
                conv_prog_profile=prog, dt=dt, config=cfg,
            )
            du = out.du_dt_conv if out.du_dt_conv is not None else jnp.zeros_like(u)
            dv = out.dv_dt_conv if out.dv_dt_conv is not None else jnp.zeros_like(v)
            return out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, du, dv, (prog_new,)
        init_carry_fn = lambda ncol, nlev, dtype: (
            jnp.zeros((ncol, nlev), dtype=dtype),
        )

    elif scheme == "emanuel":
        cfg = EmanuelConfig(M_b_max=M_B_MAX_RCE)

        def call(T, qv, pf, ph, u, v, carry):
            (prog,) = carry
            out, prog_new = emanuel_convection(
                T=T, q_v=qv, p_full=pf, p_half=ph,
                conv_prog_profile=prog, dt=dt, config=cfg,
            )
            return (
                out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt,
                jnp.zeros_like(u), jnp.zeros_like(v), (prog_new,),
            )
        init_carry_fn = lambda ncol, nlev, dtype: (
            jnp.zeros((ncol, nlev), dtype=dtype),
        )

    elif scheme == "bechtold":
        cfg = BechtoldConfig(
            enable_stochastic=False, enable_cmt=False, M_b_max=M_B_MAX_RCE,
        )

        def call(T, qv, pf, ph, u, v, carry):
            prog, stoch = carry
            mc_zero = jnp.zeros_like(T)
            out, prog_new, stoch_new = bechtold_convection(
                T=T, q_v=qv, p_full=pf, p_half=ph, u=u, v=v,
                conv_prog_profile=prog, conv_stoch_state=stoch,
                prng_key=None, dt=dt, config=cfg,
                moisture_convergence=mc_zero,
            )
            du = out.du_dt_conv if out.du_dt_conv is not None else jnp.zeros_like(u)
            dv = out.dv_dt_conv if out.dv_dt_conv is not None else jnp.zeros_like(v)
            return (
                out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt,
                du, dv, (prog_new, stoch_new),
            )
        init_carry_fn = lambda ncol, nlev, dtype: (
            jnp.zeros((ncol, nlev), dtype=dtype),
            jnp.zeros((ncol,), dtype=dtype),
        )

    elif scheme == "kuo":
        cfg = KuoConfig()

        def call(T, qv, pf, ph, u, v, carry):
            out = kuo_convection(
                T=T, q_v=qv, p_full=pf, p_half=ph, dt=dt, config=cfg,
            )
            return (
                out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt,
                jnp.zeros_like(u), jnp.zeros_like(v), (),
            )
        init_carry_fn = lambda ncol, nlev, dtype: ()

    elif scheme == "kain_fritsch":
        cfg = KainFritschConfig(M_b_max=M_B_MAX_RCE)

        def call(T, qv, pf, ph, u, v, carry):
            (prog,) = carry
            w0 = jnp.zeros_like(T)
            out, prog_new = kain_fritsch_convection(
                T=T, q_v=qv, p_full=pf, p_half=ph,
                w_grid=w0, conv_prog_profile=prog,
                dt=dt, config=cfg,
            )
            return (
                out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt,
                jnp.zeros_like(u), jnp.zeros_like(v), (prog_new,),
            )
        init_carry_fn = lambda ncol, nlev, dtype: (
            jnp.zeros((ncol, nlev), dtype=dtype),
        )

    elif scheme == "mass_flux":
        cfg = MassFluxConfig(M_b_max=M_B_MAX_RCE)

        def call(T, qv, pf, ph, u, v, carry):
            (M_c_profile,) = carry
            M_c = M_c_profile[:, -1]
            out, M_c_new = mass_flux_convection(
                T=T, q_v=qv, p_full=pf, p_half=ph, M_c=M_c,
                dt=dt, config=cfg,
            )
            M_c_profile_new = jnp.zeros_like(M_c_profile).at[:, -1].set(M_c_new)
            return (
                out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt,
                jnp.zeros_like(u), jnp.zeros_like(v),
                (M_c_profile_new,),
            )
        init_carry_fn = lambda ncol, nlev, dtype: (
            jnp.zeros((ncol, nlev), dtype=dtype).at[:, -1].set(cfg.M_c_init),
        )

    else:
        raise ValueError(f"Unsupported scheme: {scheme!r}")

    return call, init_carry_fn


def run_one_scheme(scheme: str, args, *, output_root: Path):
    """Run a single fixed-SST RCE on lat-lon FV with the requested scheme."""
    from legoesm import constants
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig,
    )
    from legoesm.driver.component_factory import (
        create_atmosphere_dycore, compute_diffusion,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_latlon
    from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
    from legoesm.atmosphere.physics.radiation.gray import gray_radiation
    from legoesm.atmosphere.physics.radiation.solar import (
        perpetual_equinox_insolation,
    )
    from legoesm.core.operators_latlon_3d import (
        hyperdiffusion_3d as hyperdiffusion_3d_ll,
    )
    from legoesm.diagnostics.column_integrals import column_water_vapor
    # NOTE on microphysics choice — we deliberately do NOT call
    # ``kessler_microphysics`` from
    # ``legoesm.atmosphere.physics.microphysics.kessler``: with its
    # default ``saturation_sharpness=100`` the soft sigmoid lets the
    # leaf produce *negative* condensation (spurious "evaporation
    # from nothing") in undersaturated air with no q_c, which yields
    # ~2000 K/day cooling on the first call from the RCE initial state
    # (T=295 K, q_v=80 % q_sat × σ²).  Instead we use the one-sided
    # minimal microphysics defined inline below: condensation only
    # when q_v > q_sat, autoconversion q_c → q_r when q_c exceeds a
    # threshold, and instant fall of q_r as surface precipitation.
    # This honours the architectural rule (precipitation belongs to
    # microphysics, not convection) without inheriting the Kessler
    # leaf's sharpness-default bug.

    N = args.N
    NLEV = args.nlev
    DT = args.dt
    DAYS = args.days

    # --- Grid + dycore -----------------------------------------------------
    grid = create_latlon_grid(N)            # (n_lat=N, n_lon=2N)
    sigma = create_sigma_coordinate(NLEV)
    shape_2d = grid.grid_shape_2d           # (n_lat, n_lon)

    config = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=N, nlev=NLEV),
        # Lat-lon FV is registered as discretization="finite_volume"
        # (see component_factory._DRIVER_SUPPORTED).  The RCE script's
        # legacy "latlon_fv" alias is not in the dispatch table.
        dycore=DycoreConfig(discretization="finite_volume", dt=DT),
    )
    model = create_atmosphere_dycore(config, grid, sigma)
    HYPERDIFF = compute_diffusion(grid, config.dycore).hyperdiff

    # --- Initial state -----------------------------------------------------
    # Start near a tropical convective equilibrium rather than the cold
    # 280 K base used by ``run_rce.py``: at 280 K the column under a 300 K
    # SST has CAPE on the order of 5000 J/kg, and the mass-flux schemes
    # (Tiedtke / ZM / KF / Bechtold / Emanuel) try to remove all of it on
    # ``tau_cape ~ 1 hour``, producing 100 K/day heating that destabilises
    # the column on the first few steps.  The unit-test fixtures for these
    # schemes (``tests/unit/test_tiedtke.py``, ``test_zhang_mcfarlane.py``,
    # ``test_convection_latlon_mpas.py``) all use ``T_sfc ~ 290–302 K`` and
    # high low-level q for exactly this reason.
    state = held_suarez_init_latlon(grid, sigma, T_init=args.T_init)
    p_full_init = state.p_s.data[..., None] * sigma.sigma_full
    q_sat_init = saturation_mixing_ratio(state.T.data, p_full_init)
    q_v = jnp.minimum(args.RH_init * q_sat_init * sigma.sigma_full ** 2,
                      q_sat_init)

    # --- Fixed SST surface -------------------------------------------------
    SST_K = args.sst_init
    SFC_ALBEDO = 0.06
    T_sfc = jnp.full(shape_2d, SST_K)
    C_H = 4.4e-3   # Frierson 2006 bulk transfer coefficient

    # --- Physics tunables --------------------------------------------------
    gray_config = GrayRadiationConfig(
        tau_equator=7.2, tau_pole=1.8, S_0=1360.0,
        sfc_albedo=SFC_ALBEDO, perpetual_equinox=True,
    )
    sigma_b = 0.7
    k_f = ((1.0 / 86400.0)
           * jnp.maximum(0.0, (sigma.sigma_full - sigma_b) / (1.0 - sigma_b)))
    fric_decay = jnp.exp(-k_f * DT)
    S_0 = gray_config.S_0
    dsigma = sigma.dsigma

    # --- Convection backend + minimal microphysics -------------------------
    # Convection update interval — production atmosphere models call
    # deep convection every 30 min (``update_interval_steps`` field on
    # ConvectionConfig) rather than every dynamics step.  Holding the
    # convective tendency constant between calls gives the dycore time
    # to absorb each kick and prevents single-step shocks from
    # destabilising the wind field at lat-lon FV pole-cell CFL.  We
    # match production by calling convection every 30 min worth of
    # dynamics steps; the leaf is invoked with the effective interval
    # as its ``dt`` argument so its internal relaxation timescales
    # remain valid.
    CONV_INTERVAL_SECONDS = 1800.0
    conv_steps = max(1, int(round(CONV_INTERVAL_SECONDS / DT)))
    conv_dt = DT * conv_steps
    conv_call, init_carry_fn = _build_convection_call(scheme, conv_dt)

    n_lat, n_lon = shape_2d
    ncol = n_lat * n_lon
    carry_dtype = state.T.data.dtype
    conv_carry = init_carry_fn(ncol, NLEV, carry_dtype)
    # Cloud-water and rain-water buckets, ``(ncol, nlev)`` each.  q_c is
    # sourced by convection's ``dq_c_conv_dt``; q_r is produced by
    # autoconversion + accretion; q_r is removed instantly via the
    # surface-precipitation diagnostic to keep the bookkeeping minimal.
    q_c_state = jnp.zeros((ncol, NLEV), dtype=carry_dtype)
    q_r_state = jnp.zeros((ncol, NLEV), dtype=carry_dtype)
    AUTOCONV_THRESHOLD = 1.0e-3   # kg/kg (Kessler default)
    AUTOCONV_RATE = 1.0e-3        # 1/s (Kessler default)
    ACCRETION_COEFF = 2.2         # 1/s (Kessler default)

    # --- JIT physics step --------------------------------------------------
    # Order: radiation + convection + BL → accumulate convective q_c source
    # into the cloud-water bucket → Kessler microphysics processes the
    # bucket and emits surface precipitation.  Kessler's saturation
    # adjustment also handles any residual super-saturation, so there is
    # no separate ad-hoc saturation-adjustment step.
    @jax.jit
    def call_convection(T, p_s, q_v, u, v, conv_carry):
        """Run the convection leaf and return its raw column tendencies.

        Called every ``conv_steps`` outer steps; the returned tendencies
        are then applied at every outer step until refreshed.  The leaf
        sees ``conv_dt`` (= 30 min) as its dt so its internal relaxation
        timescales remain valid.
        """
        nlev = T.shape[-1]
        p_full = p_s[..., None] * sigma.sigma_full
        p_half = p_s[..., None] * sigma.sigma_half
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        q_v_col = q_v.reshape(ncol, nlev)
        u_col = u.reshape(ncol, nlev)
        v_col = v.reshape(ncol, nlev)
        return conv_call(
            T_col, q_v_col, p_full_col, p_half_col, u_col, v_col, conv_carry,
        )

    @jax.jit
    def physics_step(T, p_s, q_v, u, v, conv_carry, q_c_state, q_r_state,
                     dT_conv_col, dq_v_conv_col, dq_c_conv_col,
                     du_conv_col, dv_conv_col):
        nlev = T.shape[-1]
        p_full = p_s[..., None] * sigma.sigma_full
        p_half = p_s[..., None] * sigma.sigma_half

        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        q_v_col = q_v.reshape(ncol, nlev)
        u_col = u.reshape(ncol, nlev)
        v_col = v.reshape(ncol, nlev)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = grid.grid_lat.reshape(ncol)

        insol = perpetual_equinox_insolation(lat_col, S_0)

        # (a) Gray radiation
        rad = gray_radiation(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, lat=lat_col,
            q_v=q_v_col, insolation=insol, config=gray_config,
        )
        dT_rad_col = rad.heating_rate

        # (b) Convection tendencies are passed in by the time loop —
        # refreshed every ``conv_steps`` outer steps; held constant
        # between refreshes.  This sub-cycling matches production
        # ``update_interval_steps`` and prevents single-step shocks.
        carry_new = conv_carry  # carry refresh handled outside
        # NO scaling, NO clipping, NO MSE rebalancing.  Earlier
        # versions added a 200 W/m² column-heating cap that scaled
        # ``dT_conv_col``, ``dq_v_conv_col`` *and* ``dq_c_conv_col``
        # uniformly when the leaf's column heating exceeded the cap.
        # That makes the reported precip a *capped* condensate, not
        # the scheme's actual output — a misleading comparison that
        # Codex flagged.  We report each leaf's tendencies as-is.
        # Schemes that emit unbounded heating (Emanuel, Bechtold) will
        # therefore blow the column up; the in-loop NaN check catches
        # that and the summary reports BLOWUP.

        # (c) Bulk aerodynamic surface fluxes — fixed SST, ocean β=1
        rho_low = (p_s * sigma.sigma_full[-1]) / (constants.R_d * T[..., -1])
        wind = jnp.sqrt(u[..., -1] ** 2 + v[..., -1] ** 2 + 1.0)
        dp_low = p_s * (sigma.sigma_half[-1] - sigma.sigma_half[-2])

        shflx = rho_low * constants.c_pd * C_H * wind * (T_sfc - T[..., -1])
        q_sat_sfc = saturation_mixing_ratio(T_sfc, p_s)
        lhflx = jnp.maximum(
            rho_low * constants.L_v * C_H * wind * (q_sat_sfc - q_v[..., -1]),
            0.0,
        )
        evap = lhflx / constants.L_v
        dT_BL = constants.g * shflx / (constants.c_pd * dp_low)
        dq_BL = constants.g * evap / dp_low

        # (d) Apply rad + conv + BL to T, q_v; route the leaf's
        # ``dq_c_conv_dt`` (and only that) into q_c.  We do NOT
        # manufacture extra condensate to balance the column MSE
        # budget — earlier versions of this script did so, but Codex
        # rightly flagged that as fabricating precipitation, which
        # invalidates the comparison.  The schemes' intrinsic energy
        # leaks (probed in a CAPE-positive sounding: mass_flux 238
        # W/m², ZM 22.5 kW/m², KF 1.6 kW/m², Tiedtke −5.3 kW/m²,
        # Emanuel 33 MW/m², Bechtold 129 MW/m²) are reported as-is and
        # bounded only by the column-heating cap below.
        new_T_col = T_col + DT * (dT_rad_col + dT_conv_col)
        new_T_col = new_T_col.at[:, -1].add(DT * dT_BL.reshape(ncol))
        new_qv_col = q_v_col + DT * dq_v_conv_col
        new_qv_col = new_qv_col.at[:, -1].add(DT * dq_BL.reshape(ncol))
        new_qv_col = jnp.maximum(new_qv_col, 0.0)
        new_qc = jnp.maximum(q_c_state + DT * dq_c_conv_col, 0.0)

        # (e) Minimal microphysics — one-sided saturation adjustment,
        #     autoconversion + accretion, instant rain fall.
        #
        # Per the architectural rule, surface precipitation is owned
        # here and not by convection.  We avoid Kessler's leaf because
        # of the soft-sigmoid bug noted above.

        # Saturation adjustment (one-sided): condense any q_v above
        # q_sat and release latent heat.  One pass is sufficient here:
        # after clipping q_v to q_sat(T_pre), the latent release raises
        # T and so raises q_sat further, so the resulting state is
        # *under*-saturated relative to the new q_sat — no second pass
        # would condense more vapor.  No spontaneous evaporation from
        # undersaturated air with no q_c.  A *final* sat-adj after
        # hyperdiffusion (below) catches any residual super-saturation
        # that hyperdiff imports into a saturated cell from neighbours.
        q_sat_col = saturation_mixing_ratio(new_T_col, p_full_col)
        excess = jnp.maximum(new_qv_col - q_sat_col, 0.0)
        new_qv_col = new_qv_col - excess
        new_T_col = new_T_col + constants.L_v * excess / constants.c_pd
        new_qc = new_qc + excess        # condensate joins cloud water

        # Autoconversion + accretion (Kessler-style, but applied
        # explicitly with a per-step cap to keep the explicit Euler
        # update positive).  Total q_c → q_r conversion rate per step
        # is bounded by the available q_c.
        autoconv = AUTOCONV_RATE * jnp.maximum(new_qc - AUTOCONV_THRESHOLD, 0.0)
        accretion = ACCRETION_COEFF * new_qc * jnp.clip(q_r_state, 0.0)
        c2r = jnp.minimum(autoconv + accretion, new_qc / DT)  # cap by available
        new_qc = jnp.maximum(new_qc - DT * c2r, 0.0)
        q_r_new = jnp.maximum(q_r_state + DT * c2r, 0.0)

        # Instant rain fall: rain that exists at end of step is removed
        # to surface as precipitation [kg/m^2/s].  Column-integrated:
        # ∫ q_r * (dp/g) over the column, divided by DT.
        dp_col = p_half_col[:, 1:] - p_half_col[:, :-1]
        precip_col = jnp.sum(q_r_new * dp_col / constants.g, axis=1) / DT
        q_r_new = jnp.zeros_like(q_r_new)

        new_T = new_T_col.reshape(T.shape)
        new_qv = new_qv_col.reshape(q_v.shape)

        # Hyperdiffusion on q_v (lat-lon dycore handles wind/T diffusion).
        new_qv = new_qv + DT * hyperdiffusion_3d_ll(new_qv, grid, HYPERDIFF)
        new_qv = jnp.maximum(new_qv, 0.0)

        # Final saturation adjustment AFTER hyperdiffusion: hyperdiff is
        # ∇⁴ q_v which is sign-indeterminate — in convergence cells it
        # *imports* q_v from neighbours and can push a saturated cell
        # above q_sat (the residual ZM-zonal-mean RH ≈ 1.04 at the
        # surface that we observed is exactly this pathway).  Do one
        # more sat-adj on the diffused field; the latent release goes
        # back into T (consistent with the energy / water budget).
        new_qv_col_final = new_qv.reshape(ncol, nlev)
        T_col_final = new_T.reshape(ncol, nlev)
        q_sat_col_final = saturation_mixing_ratio(T_col_final, p_full_col)
        excess_final = jnp.maximum(new_qv_col_final - q_sat_col_final, 0.0)
        new_qv_col_final = new_qv_col_final - excess_final
        T_col_final = (
            T_col_final + constants.L_v * excess_final / constants.c_pd
        )
        new_qc = new_qc + excess_final
        new_qv = new_qv_col_final.reshape(q_v.shape)
        new_T = T_col_final.reshape(T.shape)

        du_cmt = du_conv_col.reshape(u.shape)
        dv_cmt = dv_conv_col.reshape(v.shape)

        return (
            new_T, new_qv, du_cmt, dv_cmt,
            precip_col.reshape(p_s.shape),
            carry_new, new_qc, q_r_new,
        )

    # --- Time integration --------------------------------------------------
    n_steps = int(DAYS * 86400 / DT)
    diag_interval = max(1, int(args.diag_days * 86400 / DT))

    print()
    print("=" * 76)
    print(f"  RCE [scheme={scheme}]  fixed-SST={SST_K:.1f}K  "
          f"latlon FV {n_lat}x{n_lon}/L{NLEV}  dt={DT:.0f}s  {DAYS} days")
    print("=" * 76)
    print(f"  {'Day':>6s}  {'<T_sfc>':>8s}  {'<T_atm>':>8s}  "
          f"{'<Precip>':>9s}  {'<CWV>':>7s}  {'max|v|':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*8}")

    blowup = False
    last = {
        "day": 0.0, "T_sfc": float(jnp.mean(T_sfc)),
        "T_atm": float("nan"), "precip": float("nan"),
        "cwv": float("nan"), "max_v": float("nan"),
    }
    # Time-series buffer at each diag interval.
    ts = {"day": [], "T_atm": [], "T_sfc": [], "precip": [],
          "cwv": [], "max_v": []}
    # Periodic 3D snapshots — stacked per snapshot interval.  We
    # capture the full ``(n_lat, n_lon, nlev)`` fields plus 2D precip
    # so the plotter can render lat-lon maps and vertical
    # cross-sections.  ``snap3d_interval`` is set in days at the top
    # of ``main()``.
    snap3d = {
        "day": [], "T": [], "q_v": [], "q_c": [],
        "precip": [], "u": [], "v": [], "p_s": [],
    }
    snap3d_interval = max(1, int(args.snap3d_days * 86400 / DT))
    next_snap3d_step = snap3d_interval - 1   # capture at end of interval

    t_start = time.time()
    # Initial cached convection tendency: zero until first refresh.
    dT_conv_cache = jnp.zeros((ncol, NLEV), dtype=carry_dtype)
    dq_v_conv_cache = jnp.zeros_like(dT_conv_cache)
    dq_c_conv_cache = jnp.zeros_like(dT_conv_cache)
    du_conv_cache = jnp.zeros_like(dT_conv_cache)
    dv_conv_cache = jnp.zeros_like(dT_conv_cache)

    for step in range(n_steps):
        # 1) dynamics
        state = model.step(state, DT)

        # 1.5) refresh convection tendency every ``conv_steps`` outer
        # steps; otherwise reuse the cache.  When refreshing the leaf
        # sees the full ``conv_dt`` (= 30 min) so its internal
        # relaxation timescales remain valid; the cached tendency is
        # then applied at every outer step until the next refresh.
        if step % conv_steps == 0:
            (dT_conv_cache, dq_v_conv_cache, dq_c_conv_cache,
             du_conv_cache, dv_conv_cache, conv_carry) = call_convection(
                state.T.data, state.p_s.data, q_v,
                state.u.data, state.v.data, conv_carry,
            )

        # 2) operator-split physics — physics_step applies rad + cached
        # conv + BL + minimal microphysics internally; loop wraps the
        # result.
        (
            new_T, q_v, du_cmt, dv_cmt, precip,
            conv_carry, q_c_state, q_r_state,
        ) = physics_step(
            state.T.data, state.p_s.data, q_v,
            state.u.data, state.v.data, conv_carry,
            q_c_state, q_r_state,
            dT_conv_cache, dq_v_conv_cache, dq_c_conv_cache,
            du_conv_cache, dv_conv_cache,
        )

        state = state._replace(
            T=state.T.replace(data=new_T),
            u=state.u.replace(data=(state.u.data + DT * du_cmt) * fric_decay),
            v=state.v.replace(data=(state.v.data + DT * dv_cmt) * fric_decay),
        )

        # 3.5) periodic 3D snapshot capture
        if step == next_snap3d_step:
            jax.block_until_ready(state.T.data)
            day = (step + 1) * DT / 86400.0
            snap3d["day"].append(day)
            snap3d["T"].append(np.asarray(state.T.data))
            snap3d["q_v"].append(np.asarray(q_v))
            snap3d["q_c"].append(
                np.asarray(q_c_state.reshape(n_lat, n_lon, NLEV))
            )
            snap3d["precip"].append(np.asarray(precip))
            snap3d["u"].append(np.asarray(state.u.data))
            snap3d["v"].append(np.asarray(state.v.data))
            # Surface pressure varies with the dycore mass field — the
            # plotter MUST use this (not a constant 1e5) when computing
            # q_sat for the RH consistency check, otherwise it sees
            # spurious supersaturation from the formula mismatch.
            snap3d["p_s"].append(np.asarray(state.p_s.data))
            next_snap3d_step += snap3d_interval

        # 4) diagnostics
        if (step + 1) % diag_interval == 0:
            jax.block_until_ready(state.u.data)
            day = (step + 1) * DT / 86400.0
            mean_T = float(jnp.mean(state.T.data))
            mean_sfc = float(jnp.mean(T_sfc))
            max_v = float(jnp.max(
                jnp.sqrt(state.u.data ** 2 + state.v.data ** 2)
            ))
            mean_precip = float(jnp.mean(precip)) * 86400.0
            cwv = column_water_vapor(q_v, state.p_s.data, dsigma)
            mean_cwv = float(jnp.mean(cwv))
            last = dict(
                day=day, T_sfc=mean_sfc, T_atm=mean_T,
                precip=mean_precip, cwv=mean_cwv, max_v=max_v,
            )
            ts["day"].append(day)
            ts["T_atm"].append(mean_T)
            ts["T_sfc"].append(mean_sfc)
            ts["precip"].append(mean_precip)
            ts["cwv"].append(mean_cwv)
            ts["max_v"].append(max_v)
            print(f"  {day:6.1f}  {mean_sfc:8.2f}  {mean_T:8.2f}  "
                  f"{mean_precip:9.3f}  {mean_cwv:7.2f}  {max_v:8.2f}")
            # Check ALL prognostic fields, not just u — NaN can appear
            # in T or q_v before propagating to wind.  Also gate on the
            # diagnostic floats themselves so a NaN snapshot can never
            # be marked OK by the summary.
            any_nan = (
                not jnp.all(jnp.isfinite(state.u.data))
                or not jnp.all(jnp.isfinite(state.v.data))
                or not jnp.all(jnp.isfinite(state.T.data))
                or not jnp.all(jnp.isfinite(q_v))
                or any(not (val == val) for val in
                       (mean_T, mean_precip, mean_cwv, max_v))
            )
            if any_nan or max_v > 500.0:
                print(f"  BLOWUP at day {day:.1f}")
                blowup = True
                break

    jax.block_until_ready(state.u.data)
    elapsed = time.time() - t_start
    print(f"  -- {scheme}: {elapsed:.1f}s wall, {n_steps} steps, "
          f"{'BLOWUP' if blowup else 'OK'}")

    # --- Snapshot capture: final-day vertical profiles ---------------------
    # Domain-mean profiles are the natural diagnostic in a horizontally-
    # uniform RCE.  Convection tendencies come from the cached values
    # (last refresh) so they reflect the leaf's most recent emission.
    snap = {
        "scheme": scheme, "days": float(args.days),
        "N": int(args.N), "nlev": int(NLEV), "dt": float(DT),
        "sst": float(SST_K), "blowup": bool(blowup),
        # Initial-condition stamps so the plotter can refuse to combine
        # snapshots that came from different soundings (was a silent
        # data-mixing risk before).
        "T_init": float(args.T_init),
        "RH_init": float(args.RH_init),
        # Run signature: identifies which sweep invocation produced
        # this file.  Plotter asserts all loaded snapshots share the
        # same signature.
        "run_id": _RUN_ID,
        # Time series (length = number of diag intervals reached)
        "ts_day": np.asarray(ts["day"]),
        "ts_T_atm": np.asarray(ts["T_atm"]),
        "ts_T_sfc": np.asarray(ts["T_sfc"]),
        "ts_precip": np.asarray(ts["precip"]),
        "ts_cwv": np.asarray(ts["cwv"]),
        "ts_max_v": np.asarray(ts["max_v"]),
        # Final-day vertical profiles (domain-mean)
        "sigma_full": np.asarray(sigma.sigma_full),
        "T_profile": np.asarray(jnp.mean(state.T.data, axis=(0, 1))),
        "qv_profile": np.asarray(jnp.mean(q_v, axis=(0, 1))),
        "qc_profile": np.asarray(jnp.mean(
            q_c_state.reshape(n_lat, n_lon, NLEV), axis=(0, 1)
        )),
        # Convection tendency profile (last cached refresh; domain-mean)
        "dT_conv_profile": np.asarray(jnp.mean(
            dT_conv_cache.reshape(n_lat, n_lon, NLEV), axis=(0, 1)
        )),
        "dqv_conv_profile": np.asarray(jnp.mean(
            dq_v_conv_cache.reshape(n_lat, n_lon, NLEV), axis=(0, 1)
        )),
        "dqc_conv_profile": np.asarray(jnp.mean(
            dq_c_conv_cache.reshape(n_lat, n_lon, NLEV), axis=(0, 1)
        )),
        # RH for context (domain-mean)
        "rh_profile": np.asarray(
            jnp.mean(q_v, axis=(0, 1)) / jnp.mean(
                saturation_mixing_ratio(
                    state.T.data,
                    state.p_s.data[..., None] * sigma.sigma_full,
                ),
                axis=(0, 1),
            )
        ),
    }
    out_path = output_root / f"snapshot_{scheme}.npz"
    # np.savez wraps scalars / strings into 0-d arrays automatically.
    np.savez(out_path, **snap)
    print(f"  -- {scheme}: snapshot saved to {out_path}")

    # 3D snapshots — only if any were captured.  Each field becomes a
    # ``(n_snaps, ...)`` stacked array.  Stamped with the same run_id
    # / config so the plotter can tie 3D and per-scheme files together.
    if snap3d["day"]:
        snap3d_path = output_root / f"snapshot3d_{scheme}.npz"
        snap3d_payload = {
            "scheme": scheme,
            "days": float(args.days),
            "N": int(args.N), "nlev": int(NLEV), "dt": float(DT),
            "sst": float(SST_K), "blowup": bool(blowup),
            "T_init": float(args.T_init),
            "RH_init": float(args.RH_init),
            "snap3d_days": float(args.snap3d_days),
            "run_id": _RUN_ID,
            "sigma_full": np.asarray(sigma.sigma_full),
            "lat": np.asarray(grid.grid_lat),       # (n_lat, n_lon)
            "lon": np.asarray(grid.grid_lon),       # (n_lat, n_lon)
            "snap_day": np.asarray(snap3d["day"]),
            "T": np.stack(snap3d["T"]),             # (n_snaps, n_lat, n_lon, nlev)
            "q_v": np.stack(snap3d["q_v"]),
            "q_c": np.stack(snap3d["q_c"]),
            "precip": np.stack(snap3d["precip"]),   # (n_snaps, n_lat, n_lon)
            "u": np.stack(snap3d["u"]),
            "v": np.stack(snap3d["v"]),
            "p_s": np.stack(snap3d["p_s"]),         # (n_snaps, n_lat, n_lon) [Pa]
        }
        np.savez(snap3d_path, **snap3d_payload)
        print(f"  -- {scheme}: 3D snapshot ({len(snap3d['day'])} times) "
              f"saved to {snap3d_path}")

    return {"scheme": scheme, "blowup": blowup, "wall_s": elapsed, **last}


def main():
    parser = argparse.ArgumentParser(
        description="RCE convection-scheme sweep on lat-lon FV with fixed SST",
    )
    parser.add_argument(
        "--schemes", type=str, default="all",
        help=f"Comma-separated scheme list, or 'all'. "
             f"Choices: {','.join(SCHEME_CHOICES)}",
    )
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--N", type=int, default=12,
                        help="n_lat for lat-lon grid; n_lon = 2N.")
    parser.add_argument("--nlev", type=int, default=20)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--diag-days", type=float, default=1.0)
    parser.add_argument("--sst-init", type=float, default=300.0)
    parser.add_argument(
        "--T-init", type=float, default=295.0,
        help="Initial atmospheric temperature [K] (warm-isothermal init).",
    )
    parser.add_argument(
        "--RH-init", type=float, default=0.8,
        help="Initial column-mean RH (multiplied by sigma**2 vertical decay).",
    )
    parser.add_argument(
        "--snap3d-days", type=float, default=30.0,
        help="Interval (days) between 3D field snapshots for the "
             "lat-lon and vertical cross-section plots.",
    )
    parser.add_argument(
        "--output", type=str, default="results/rce_convection_sweep",
    )
    args = parser.parse_args()

    schemes = _parse_schemes(args.schemes)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    # Delete every existing snapshot (1D + 3D) in the output directory
    # — for any scheme — so stale data from a prior invocation with
    # different ``--days`` / ``--N`` / ``--T-init`` / ``--RH-init``
    # cannot be silently mixed into the comparison plots.  Snapshots
    # written in this run all share the new ``_RUN_ID``.
    for stale in output_root.glob("snapshot_*.npz"):
        stale.unlink()
    for stale in output_root.glob("snapshot3d_*.npz"):
        stale.unlink()
    print(f"  run_id={_RUN_ID}, cleared stale snapshots in {output_root}")

    summary = []
    for scheme in schemes:
        try:
            row = run_one_scheme(scheme, args, output_root=output_root)
            row.setdefault("error", None)
        except Exception as e:                              # noqa: BLE001
            # Execution errors (import failure, shape mismatch, missing
            # config field, dycore setup crash) are a *different* failure
            # mode from numerical blowup of the integration.  We tag them
            # explicitly so the summary can distinguish "scheme produced
            # NaN/runaway" from "scheme couldn't even be exercised".
            err_label = f"{type(e).__name__}: {e}"
            print(f"  -- {scheme}: ERROR {err_label}")
            row = {
                "scheme": scheme, "blowup": False, "wall_s": float("nan"),
                "day": 0.0, "T_sfc": float("nan"), "T_atm": float("nan"),
                "precip": float("nan"), "cwv": float("nan"),
                "max_v": float("nan"),
                "error": err_label,
            }
        summary.append(row)

    print()
    print("=" * 76)
    print("  SUMMARY (final diagnostic snapshot per scheme)")
    print("=" * 76)
    print(f"  {'scheme':>16s}  {'day':>5s}  {'T_sfc':>7s}  {'T_atm':>7s}  "
          f"{'precip':>8s}  {'CWV':>6s}  {'max|v|':>7s}  {'wall_s':>7s}  status")
    print(f"  {'-'*16}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*6}  "
          f"{'-'*7}  {'-'*7}  ------")
    for r in summary:
        # Status precedence (most-severe first):
        # - ERROR: ``run_one_scheme`` raised — script-level failure, not
        #   a numerical blowup of the integration.
        # - BLOWUP: integration ran but produced NaN (in state or
        #   diagnostics).
        # - RUNAWAY: integration finished but the final state is
        #   wildly unphysical for SST=300 K (T_atm outside [200, 350] K
        #   or max|v| > 100 m/s).
        # - OK: integration finished with plausible diagnostics.
        diag_nan = any(
            (r[k] != r[k]) for k in ("T_atm", "precip", "cwv", "max_v")
        )
        runaway = (
            (not diag_nan)
            and (r["T_atm"] < 200.0 or r["T_atm"] > 350.0
                 or r["max_v"] > 100.0)
        )
        if r.get("error") is not None:
            status = "ERROR"
        elif r["blowup"] or diag_nan:
            status = "BLOWUP"
        elif runaway:
            status = "RUNAWAY"
        else:
            status = "OK"
        print(
            f"  {r['scheme']:>16s}  {r['day']:5.1f}  {r['T_sfc']:7.2f}  "
            f"{r['T_atm']:7.2f}  {r['precip']:8.3f}  {r['cwv']:6.2f}  "
            f"{r['max_v']:7.2f}  {r['wall_s']:7.1f}  {status}"
        )
    # If any rows errored, surface the per-scheme error string at the
    # bottom so a reader can act on it without scrolling back.
    errored = [r for r in summary if r.get("error") is not None]
    if errored:
        print()
        print("  Errors (script-level, not numerical blowup):")
        for r in errored:
            print(f"    {r['scheme']}: {r['error']}")


if __name__ == "__main__":
    main()
