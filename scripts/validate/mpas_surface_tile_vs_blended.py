"""RETRACTED AS A MEASUREMENT OF THE PRODUCTION MPAS LANE (#1320).

Codex review, 2026-09-04, four HIGH findings.  The numbers this script produced
were quoted on #1320 and have been withdrawn there.  Do not cite them.

WHAT IS WRONG, so it is fixed rather than re-run:

1. The "blended" arm is a strawman.  When the interactive land supplies its
   fluxes, the MPAS turbulence factory does NOT run one bulk scheme on a
   blended skin temperature over land -- it area-weights the FLUXES,
   ``(1 - f_land) * SH_ocean + f_land * SH_land`` and likewise for latent heat
   (``turbulence/integration.py`` ~L770-812), fed by ``forcing["shflx_land"]``
   / ``["lhflx_land"]``, wired at ``model_driver.py:10563``.  It also takes the
   land's SOLVED boundary humidity ``forcing["q_sfc_land"]`` verbatim over
   land, superseding the beta throttle reproduced below.  So for the land tile
   the lane ALREADY does the flux-level aggregation the port would buy.
2. Gustiness resolves to the native 600 m default here; production sets
   ``surface_gustiness_zi: 300.0``.  The configuration must be RESOLVED, not
   reconstructed as it is below.
3. The ice tile is invented (MOST at z0=1e-3); the real tiled path uses
   ``bulk_scheme="constant"`` (``physics_pipeline.py:1032``).
4. The tests exercise only this file's own two arms, so the production path
   could be fixed and every test would still pass.

WHAT A CORRECT VERSION MUST DO: resolve the production ``ExperimentConfig`` and
read the tile configs off it; drive ``_make_mpas_turbulence`` with the same
forcing channels a run threads (including ``shflx_land``/``lhflx_land`` and
``q_sfc_land``); and scope the comparison to what is genuinely untiled on that
lane -- the SEA-ICE tile, momentum/stress, and the ocean tile's stability being
evaluated against a partly land-influenced blended state.

Kept in the tree rather than deleted because the correct version is a
modification of this one and the four defects above are the specification for
it.

--------------------------------------------------------------------------
THE ORIGINAL HEADER, kept for the specification it carries:

How much does the MPAS blended surface cost, in W/m^2? (#1320)

The MPAS lane runs an UNTILED surface: one blended ``T_sfc``/``q_sfc`` per cell
is fed to a single bulk scheme, with the land fraction handled by a skin-T
blend and a beta-limited humidity.  The FV pipeline instead computes fluxes
SEPARATELY on the ocean / sea-ice / land tiles -- each with its own surface
temperature, its own saturation humidity and its own bulk scheme -- and
area-weights the resulting fluxes.

Blending is exact only for a FIXED shared transfer coefficient.  A
stability-dependent bulk scheme recomputes the exchange coefficient from the
blended surface temperature, and that coefficient is a nonlinear function of
the surface-air contrast, so the blend is mis-averaged wherever two tiles have
genuinely different stability -- a cold-air outbreak over a coastal cell, or
the ice edge.

This script QUANTIFIES that error offline, on single columns, before anyone
buys the port.  It answers one question: how many W/m^2 does the approximation
cost at its worst, and at what land/ice fraction.

It prints NUMBERS ONLY and no verdict -- a probe that decides what its own
output means gets that opinion quoted back as evidence.

PROVENANCE, so these numbers are not quoted as "the model's error": the columns
below are ARCHETYPES with contrasts imposed by hand (a 25 K land-ocean skin
difference, a 15 K ice-air deficit), not samples drawn from a run.  They bound
what the approximation costs where the tiles genuinely disagree; what it costs
IN a run depends on the joint distribution of skin contrast, wind and cover
fraction over the actual mesh, which only model output can supply.  Note also
that a large skin contrast AND a strong wind is itself an unusual pairing --
wind destroys strong surface stability -- so the high-wind rows are an upper
bound, not a typical case.

Reuse (no re-derivation): the tiled path is
``legoesm.atmosphere.physics.turbulence.surface_layer.compute_tiled_surface_fluxes``
and the blended path is ``compute_surface_fluxes`` from the same module -- the
two functions the model itself calls.  The land tile's config is built the way
``ExperimentConfig._land_tile_surface_cfg`` builds it (``bulk_scheme="most"``,
``z0=surface_z0_land``, no gustiness), and the blended humidity uses the same
``beta_limited_surface_humidity`` helper the MPAS turbulence factory uses.

Usage::

    python scripts/validate/mpas_surface_tile_vs_blended.py
    python scripts/validate/mpas_surface_tile_vs_blended.py --wind 12 --beta 0.3
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import (  # noqa: E402
    SurfaceLayerConfig,
)
from legoesm.atmosphere.physics.turbulence.surface_layer import (  # noqa: E402
    SurfaceTileSpec,
    beta_limited_surface_humidity,
    compute_surface_fluxes,
    compute_tiled_surface_fluxes,
)
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402


# Production values, not invented ones: config/amip/amip_production.yaml sets
# surface_bulk_scheme: coare3 (the OCEAN tile) and surface_z0_land: 0.1, and
# ExperimentConfig._land_tile_surface_cfg derives the land tile from the ocean
# one by swapping in "most" + that roughness and dropping the gustiness.
_OCEAN_CFG = SurfaceLayerConfig(bulk_scheme="coare3", z0=1.0e-4, z_ref=10.0)
_LAND_CFG = _OCEAN_CFG._replace(
    bulk_scheme="most", z0=0.1, gustiness_w_zi=0.0)
_ICE_CFG = _OCEAN_CFG._replace(bulk_scheme="most", z0=1.0e-3,
                               gustiness_w_zi=0.0)


def _column(*, T_air, q_air, wind, p_sfc, rho):
    n = 1
    z = jnp.zeros((n,))
    return dict(
        u=jnp.full((n,), wind), v=z,
        T=jnp.full((n,), T_air), q_v=jnp.full((n,), q_air),
        rho=jnp.full((n,), rho),
    )


def _pair(*, frac_land, frac_ice, T_ocean, T_ice, T_land, beta, col, p_sfc):
    """(tiled, blended) sensible/latent/stress for one mosaic column."""
    one = jnp.ones((1,))
    f_land = one * frac_land
    f_ice = one * frac_ice
    f_ocean = one * (1.0 - frac_land - frac_ice)

    q_ocean = saturation_mixing_ratio(one * T_ocean, one * p_sfc)
    q_ice = saturation_mixing_ratio(one * T_ice, one * p_sfc)
    # The land tile's own boundary humidity is beta-throttled saturation at the
    # LAND skin temperature -- the tiled counterpart of what the blended path
    # does to the blended value.
    q_land = beta * saturation_mixing_ratio(one * T_land, one * p_sfc)

    tiles = SurfaceTileSpec(
        frac_ocean=f_ocean, frac_ice=f_ice, frac_land=f_land,
        T_ocean=one * T_ocean, T_ice=one * T_ice, T_land=one * T_land,
        q_sfc_ocean=q_ocean, q_sfc_ice=q_ice, q_sfc_land=q_land,
    )
    tiled = compute_tiled_surface_fluxes(
        col["u"], col["v"], col["T"], col["q_v"], col["rho"],
        tiles, _OCEAN_CFG, _ICE_CFG, _LAND_CFG)

    # The MPAS blended surface: area-weighted skin temperature, saturation at
    # that blend, then the land fraction's humidity gradient throttled by beta
    # (turbulence/integration.py) -- and ONE bulk scheme for the whole cell.
    T_blend = (f_ocean * T_ocean + f_ice * T_ice + f_land * T_land)
    q_blend = saturation_mixing_ratio(T_blend, one * p_sfc)
    q_blend = beta_limited_surface_humidity(
        q_blend, col["q_v"], f_land, beta)
    blended = compute_surface_fluxes(
        col["u"], col["v"], col["T"], col["q_v"],
        T_blend, q_blend, col["rho"], _OCEAN_CFG)
    return tiled, blended


def _row(label, frac_land, frac_ice, tiled, blended):
    sh_t, lh_t = float(tiled[2][0]), float(tiled[3][0])
    sh_b, lh_b = float(blended[2][0]), float(blended[3][0])
    tau_t = float(np.hypot(float(tiled[0][0]), float(tiled[1][0])))
    tau_b = float(np.hypot(float(blended[0][0]), float(blended[1][0])))
    return dict(
        case=label, frac_land=frac_land, frac_ice=frac_ice,
        sh_tiled=sh_t, sh_blended=sh_b, d_sh=sh_b - sh_t,
        lh_tiled=lh_t, lh_blended=lh_b, d_lh=lh_b - lh_t,
        tau_tiled=tau_t, tau_blended=tau_b, d_tau=tau_b - tau_t,
        d_turb=(sh_b + lh_b) - (sh_t + lh_t),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wind", type=float, default=8.0, help="10 m wind [m/s]")
    p.add_argument("--beta", type=float, default=0.4,
                   help="soil moisture availability on the land tile")
    p.add_argument("--p-sfc", type=float, default=1.0e5, help="surface p [Pa]")
    args = p.parse_args(argv)

    rows = []

    # --- coastal cell: cold winter land against a much warmer ocean ---------
    # Land skin 20 K below the air (strongly STABLE over land) while the ocean
    # is 5 K above it (UNSTABLE over water) -- the contrast that makes one
    # shared stability argument wrong for both tiles.
    T_air = constants.T_freeze
    col = _column(T_air=T_air, q_air=2.0e-3, wind=args.wind,
                  p_sfc=args.p_sfc, rho=1.25)
    for f in (0.1, 0.25, 0.5, 0.75, 0.9):
        t, b = _pair(frac_land=f, frac_ice=0.0,
                     T_ocean=T_air + 5.0, T_ice=constants.T_freeze,
                     T_land=T_air - 20.0, beta=args.beta, col=col,
                     p_sfc=args.p_sfc)
        rows.append(_row("coastal_cold_air_outbreak", f, 0.0, t, b))

    # --- ice edge: cold ice against open water at the freezing point -------
    T_air_ice = 253.15
    col_ice = _column(T_air=T_air_ice, q_air=0.5e-3, wind=args.wind,
                      p_sfc=args.p_sfc, rho=1.35)
    for f in (0.1, 0.25, 0.5, 0.75, 0.9):
        t, b = _pair(frac_land=0.0, frac_ice=f,
                     T_ocean=constants.T_freeze_ocean,
                     T_ice=T_air_ice - 15.0,
                     T_land=T_air_ice, beta=1.0, col=col_ice,
                     p_sfc=args.p_sfc)
        rows.append(_row("ice_edge", 0.0, f, t, b))

    hdr = (f"{'case':28s} {'f_land':>7s} {'f_ice':>6s} "
           f"{'SH_tiled':>9s} {'SH_blend':>9s} {'dSH':>8s} "
           f"{'LH_tiled':>9s} {'LH_blend':>9s} {'dLH':>8s} "
           f"{'dSH+dLH':>8s} {'d|tau|':>8s}")
    print(f"wind={args.wind} m/s  beta_land={args.beta}  "
          f"p_sfc={args.p_sfc/100:.0f} hPa   (fluxes W/m^2, positive UP; "
          f"d = blended - tiled)")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['case']:28s} {r['frac_land']:7.2f} {r['frac_ice']:6.2f} "
              f"{r['sh_tiled']:9.2f} {r['sh_blended']:9.2f} {r['d_sh']:8.2f} "
              f"{r['lh_tiled']:9.2f} {r['lh_blended']:9.2f} {r['d_lh']:8.2f} "
              f"{r['d_turb']:8.2f} {r['d_tau']:8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
