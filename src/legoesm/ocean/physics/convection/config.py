"""Configuration for ocean convection schemes."""

from __future__ import annotations

from typing import NamedTuple


class EnhancedDiffusionConfig(NamedTuple):
    """Enhanced diffusion where N^2 < 0.

    The convective diffusivity ``K_conv`` is normally large (~1 m²/s) to
    rapidly homogenize an unstable column.  When the diffusion operator
    is applied explicitly (``apply_diffusion=True`` in
    ``enhanced_diffusion_convection``), explicit-Euler stability requires
    ``K · dt / dz² ≤ 0.5``.  With ``dz ≈ 10 m`` and ``dt ≈ 3600 s`` this
    forces ``K ≤ 0.014`` — three orders of magnitude below the desired
    value.  ``cfl_dt_estimate`` and ``cfl_safety`` parameterize the
    safety cap applied internally to ``K`` along the explicit path; the
    implicit path (``apply_diffusion=False``) bypasses the cap because
    backward-Euler is unconditionally stable.
    """
    K_conv: float = 1.0        # Convective diffusivity [m^2/s]
    K_bg: float = 1e-5         # Background diffusivity [m^2/s]
    smooth_transition: bool = True
    sigmoid_sharpness: float = 1e6
    cfl_dt_estimate: float = 3600.0  # Reference dt for explicit-CFL cap [s]
    cfl_safety: float = 0.45         # Stability margin (≤ 0.5)


class PlumeConfig(NamedTuple):
    """Entraining mass-flux convective plume."""
    epsilon: float = 1e-3       # Entrainment rate [1/m]
    alpha_plume: float = 0.1    # Detrainment tendency scaling
    # Plume vertical velocity [m/s].  Used directly as ``w_p`` in
    # dT/dt = w_p · α · ε · (T_p − T_env).  Despite the legacy
    # ``_min`` suffix, this is the actual plume speed for the
    # unresolved-plume detrainment closure (see plume.py:75-88).
    w_plume_min: float = 0.01
    T_excess: float = 0.05      # Initial plume temperature excess [K]
    # Sharpness of the smooth active-mask transition on ``delta_rho``.
    # Larger values approach a hard switch; ``1e4`` corresponds to a
    # transition width of ~1e-4 kg/m^3 in density anomaly.  Configurable
    # so coarser/finer EOS regimes can retune without source edits.
    active_sigmoid_sharpness: float = 1e4


class OceanConvectionConfig(NamedTuple):
    """Top-level ocean convection configuration."""
    scheme: str = "none"  # "enhanced_diffusion", "plume", "none"
    enhanced_diffusion: EnhancedDiffusionConfig = EnhancedDiffusionConfig()
    plume: PlumeConfig = PlumeConfig()
