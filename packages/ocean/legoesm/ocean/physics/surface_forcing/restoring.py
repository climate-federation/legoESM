"""SST/SSS restoring to target profiles."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.surface_forcing.config import RestoringConfig
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput
from legoesm.ocean.physics.surface_forcing._shared import linear_relaxation

__physics_contract__ = {
    "summary": (
        "Newtonian (Haney) SST/SSS restoring: relax surface temperature and "
        "salinity toward prescribed target profiles, dT/dt = -(T - T*)/tau_T "
        "and dS/dt = -(S - S*)/tau_S, applied in the top ocean layer."
    ),
    "inputs": {
        "T": "degC", "S": "psu", "grid.grid_lat": "rad",
        "cfg.T_star_eq": "degC", "cfg.S_star": "psu",
        "cfg.tau_T": "s", "cfg.tau_S": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "degC/s", "dS_dt": "psu/s",
        "Q_net": "W/m^2", "tau_x": "N/m^2", "tau_y": "N/m^2",
    },
    "sign_convention": (
        "Relaxation OPPOSES the anomaly — the tendency has the opposite sign to "
        "(T - T*) and (S - S*), so it always drives T, S toward the target; "
        "relaxation timescales tau_T, tau_S > 0; a non-conservative surface "
        "source/sink (adds/removes heat & salt to reach the target); z positive "
        "up. An unknown T_profile raises ValueError."
    ),
    # Restoring is a surface source/sink toward a target; not conservative.
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Haney, R. L. (1971), JPO 1, 241-248 (Newtonian surface relaxation)",
    "idealized_test": (
        "tests/ocean/unit/test_restoring_extension.py — T above target cools, "
        "below target warms, magnitude ~ |T - T*|/tau; T = T* gives zero "
        "tendency; an unknown T_profile raises."
    ),
}


def _make_target(cfg: RestoringConfig, lat, dtype, kind: str):
    """Resolve the restoring target. ``kind`` ∈ {"T", "S"}.

    Priority: user-provided 2D array > built-in cosine/constant formula.
    """
    if kind == "T":
        if cfg.T_star_array is not None:
            return jnp.asarray(cfg.T_star_array).astype(dtype)
        if cfg.T_profile == "cosine":
            return (cfg.T_star_eq
                    - (cfg.T_star_eq - cfg.T_star_pole) * jnp.sin(lat) ** 2
                    ).astype(dtype)
        if cfg.T_profile == "constant":
            return jnp.full_like(lat, cfg.T_star_eq, dtype=dtype)
        # Fail fast rather than silently treating a typo as "constant"
        # (slopbuster Pass 4).  ``T_profile`` is a static config string.
        raise ValueError(
            f"Unknown T_profile {cfg.T_profile!r}; expected 'cosine' or "
            "'constant'"
        )
    # kind == "S"
    if cfg.S_star_array is not None:
        return jnp.asarray(cfg.S_star_array).astype(dtype)
    return jnp.full_like(lat, cfg.S_star, dtype=dtype)


def restoring_surface_forcing(
    T: jnp.ndarray,
    S: jnp.ndarray,
    grid,
    cfg: RestoringConfig,
    *,
    sw_down=None,
    dt: float | None = None,
    rho_0: float | None = None,
    c_p: float | None = None,
    dz_0: float | None = None,
) -> SurfaceForcingOutput:
    """Apply SST/SSS restoring to target profiles in the surface layer.

    Parameters
    ----------
    T, S : array
        3D tracer fields (any grid layout). Trailing axis is the
        vertical level; index 0 is the surface.
    grid : any grid with ``grid_lat`` attribute
    cfg : RestoringConfig
    sw_down : array, optional
        Surface shortwave [W/m²]; only used when ``cfg.subtract_qsr=True``.
        Subtracted from the surface T tendency after the W/m² → K/s
        conversion (paper eq 8 non-solar split). Shape must match
        ``grid.grid_lat``.
    dt : float, optional
        Baroclinic timestep [s]. Required when ``cfg.implicit=True``.
    rho_0, c_p, dz_0 : float, optional
        Required for the Q_sr-subtraction term (W/m² → K/s conversion).

    Returns
    -------
    SurfaceForcingOutput
        ``dT_dt`` and ``dS_dt`` have the shape of ``T``/``S`` with
        non-zero values only in the surface layer (``[..., 0]``).

    Notes on integration mode
    -------------------------
    With ``cfg.implicit=False`` (default), returns a *forward-Euler*
    tendency ``-(T − T*) / τ``. The dycore's tracer update
    ``T_new = T_old + dt · dT_dt`` then gives the standard explicit
    relaxation. Stable iff ``dt < 2·τ`` AND there's no destabilising
    feedback through other operators.

    With ``cfg.implicit=True``, returns an *effective* tendency that,
    when applied via the dycore's same forward-Euler tracer update,
    yields the *analytical implicit-Euler* result
    ``T_new = (T_old + dt·T*/τ) / (1 + dt/τ)``. By algebra:

        dT_dt_eff = (T_new − T_old) / dt = (T* − T_old) / (τ + dt)

    so the implicit step is just a denominator change ``τ → τ + dt``.
    Stable for any dt; the new value is bounded between ``T_old`` and
    ``T*``. Q_sr (when subtract_qsr=True) is treated explicitly because
    it's a known external forcing, not a function of T.
    """
    shape_3d = T.shape
    dtype = T.dtype
    lat = grid.grid_lat

    T_star = _make_target(cfg, lat, dtype, "T")
    S_star = _make_target(cfg, lat, dtype, "S")

    if cfg.implicit:
        if dt is None:
            raise ValueError(
                "RestoringConfig.implicit=True requires `dt` to be passed "
                "to restoring_surface_forcing(...)."
            )
        eff_tau_T = cfg.tau_T + dt
        eff_tau_S = cfg.tau_S + dt
    else:
        eff_tau_T = cfg.tau_T
        eff_tau_S = cfg.tau_S

    # Newtonian relaxation toward the zonal target (shared kernel, #518 item 9).
    # ``neg_field_minus_target`` reproduces the prior -(f - target)/tau form
    # bit-for-bit.
    surf_dT = linear_relaxation(
        T[..., 0], T_star, eff_tau_T, numerator="neg_field_minus_target")
    surf_dS = linear_relaxation(
        S[..., 0], S_star, eff_tau_S, numerator="neg_field_minus_target")

    if cfg.subtract_qsr:
        if sw_down is None:
            raise ValueError(
                "RestoringConfig.subtract_qsr=True requires `sw_down`."
            )
        if rho_0 is None or c_p is None or dz_0 is None:
            raise ValueError(
                "RestoringConfig.subtract_qsr=True requires rho_0, c_p, dz_0 "
                "for the W/m² → K/s conversion."
            )
        surf_dT = surf_dT - sw_down / (rho_0 * c_p * dz_0)

    nlev = shape_3d[-1]
    pad_axes_r = ((0, 0),) * (len(shape_3d) - 1)
    dT_dt = jnp.pad(surf_dT[..., None], (*pad_axes_r, (0, nlev - 1)))
    dS_dt = jnp.pad(surf_dS[..., None], (*pad_axes_r, (0, nlev - 1)))

    du_dt = jnp.zeros(shape_3d, dtype=dtype)
    dv_dt = jnp.zeros(shape_3d, dtype=dtype)
    Q_net = jnp.zeros_like(lat)
    tau_x = jnp.zeros_like(lat)
    tau_y = jnp.zeros_like(lat)

    return SurfaceForcingOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=Q_net, tau_x=tau_x, tau_y=tau_y,
    )
