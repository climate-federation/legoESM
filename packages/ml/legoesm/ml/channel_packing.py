"""Channel packing: convert legoESM states to/from flat tensors.

Provides functions to pack spectral model states into dense arrays
of shape (n_lat, n_lon, n_channels) suitable for the SFNO, and to
unpack SFNO outputs back into state objects.

Three packing schemes:
- **Shallow Water (SW)**: 4 channels — vor, div, phi, phis
- **Primitive Equations (PE 3D)**: 4*nlev + 2 surface channels on the
  MODEL SIGMA LEVELS (no sigma→pressure interpolation is performed)
- **Ocean**: 4*nlev + 2 surface — u, v, T, S at each level + eta, H_bathy

``WB2_PRESSURE_LEVELS`` below is a DATA-side constant (default ERA5
level selection for ``training/era5_to_state.py`` and
``ml/data/era5_loader.py``); it plays no role in the packing itself.
"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from legoesm.atmosphere.dynamics.gcm.spectral_sw import SpectralSWState
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralHydrostaticState

from legoesm.grids.gaussian import (
    GaussianGrid,
    sh_synthesis,
    sh_synthesis_3d,
    sh_analysis,
    sh_analysis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
    uv_from_vordiv_3d,
)
from legoesm.atmosphere.dynamics.gcm.spectral_sw import SpectralSWState
from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralHydrostaticState
from legoesm.atmosphere.physics._shared import zero_like_tracers

# WeatherBench2 standard pressure levels [hPa]
WB2_PRESSURE_LEVELS = (
    1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50
)


class SWChannelSpec(NamedTuple):
    """Channel layout for shallow water packing.

    4 channels: vor, div, phi, phis
    """
    n_channels: int = 4
    vor_idx: int = 0
    div_idx: int = 1
    phi_idx: int = 2
    phis_idx: int = 3


class PE3DChannelSpec(NamedTuple):
    """Channel layout for 3D primitive equation packing.

    Channels: [u(nlev), v(nlev), T(nlev), q(nlev), lnps, phis, ...]
    """
    nlev: int = 13
    n_base_vars: int = 4  # u, v, T, q
    n_surface: int = 2    # lnps, phis

    @property
    def n_channels(self) -> int:
        return self.n_base_vars * self.nlev + self.n_surface

    @property
    def u_slice(self) -> slice:
        return slice(0, self.nlev)

    @property
    def v_slice(self) -> slice:
        s = self.nlev
        return slice(s, s + self.nlev)

    @property
    def T_slice(self) -> slice:
        s = 2 * self.nlev
        return slice(s, s + self.nlev)

    @property
    def q_slice(self) -> slice:
        s = 3 * self.nlev
        return slice(s, s + self.nlev)

    @property
    def lnps_idx(self) -> int:
        return self.n_base_vars * self.nlev


# ============================================================================
# Shallow Water packing
# ============================================================================

def pack_sw_state(state, grid: GaussianGrid) -> jnp.ndarray:
    """Pack a SpectralSWState into a dense tensor for the SFNO.

    Parameters
    ----------
    state : SpectralSWState
        Spectral shallow water state (vor_hat, div_hat, phi_hat, phis_hat).
    grid : GaussianGrid
        Grid for SH synthesis.

    Returns
    -------
    array, shape (n_lat, n_lon, 4)
        Packed grid-space fields: [vor, div, phi, phis].
    """
    vor = sh_synthesis(grid, state.vor_hat.data)
    div = sh_synthesis(grid, state.div_hat.data)
    phi = sh_synthesis(grid, state.phi_hat.data)
    phis = sh_synthesis(grid, state.phis_hat.data)
    return jnp.stack([vor, div, phi, phis], axis=-1)


def unpack_sw_output(
    output: jnp.ndarray,
    state: "SpectralSWState",
    grid: GaussianGrid,
    mode: str = "state_update",
) -> "SpectralSWState":
    """Unpack SFNO output back into a SpectralSWState.

    Parameters
    ----------
    output : array, shape (n_lat, n_lon, 4)
        SFNO output: [vor, div, phi, phis] in grid space.
    state : SpectralSWState
        Original state (used for Field metadata and phis in tendency mode).
    grid : GaussianGrid
        Grid for SH analysis.
    mode : str
        "state_update": output is the new state directly.
        "tendencies": output is tendencies (dvor/dt, ddiv/dt, dphi/dt, 0).

    Returns
    -------
    SpectralSWState
        New state or tendencies in spectral space.
    """
    vor_grid = output[..., 0]
    div_grid = output[..., 1]
    phi_grid = output[..., 2]

    vor_hat = sh_analysis(grid, vor_grid.astype(jnp.float64))
    div_hat = sh_analysis(grid, div_grid.astype(jnp.float64))
    phi_hat = sh_analysis(grid, phi_grid.astype(jnp.float64))

    if mode == "tendencies":
        phis_hat = jnp.zeros_like(state.phis_hat.data)
    else:
        phis_hat = state.phis_hat.data

    return SpectralSWState(
        vor_hat=state.vor_hat.replace(data=vor_hat),
        div_hat=state.div_hat.replace(data=div_hat),
        phi_hat=state.phi_hat.replace(data=phi_hat),
        phis_hat=state.phis_hat.replace(data=phis_hat),
    )


# ============================================================================
# 3D Primitive Equation packing
# ============================================================================

def pack_pe_state(
    state,
    grid: GaussianGrid,
    sigma_coord=None,
) -> jnp.ndarray:
    """Pack a SpectralHydrostaticState into a dense tensor.

    The q channel is populated from ``state.tracers["q_v"]`` when
    present (the canonical spectral PE moisture tracer); falls back to
    zeros for the legacy dry pipeline.  Without this, the SFNO never
    sees ERA5 humidity even when the IC carries it.

    Parameters
    ----------
    state : SpectralHydrostaticState
        Spectral PE state.
    grid : GaussianGrid
        Grid for SH transforms.
    sigma_coord : SigmaCoordinate, optional
        Unused; accepted for call-site compatibility (the PE dycore
        bridges pass it positionally).  Channels are packed on the
        model sigma levels directly.

    Returns
    -------
    array, shape (n_lat, n_lon, n_channels)
        Packed grid-space fields.
    """
    # 3D fields: (n_lat, n_lon, nlev)
    T = sh_synthesis_3d(grid, state.T_hat.data)

    # u, v from vorticity-divergence
    u_cos, v_cos = uv_from_vordiv_3d(
        grid, state.vor_hat.data, state.div_hat.data
    )
    cos_lat_3d = grid.cos_lat[:, None, None]
    u = u_cos / jnp.maximum(cos_lat_3d, 1e-6)
    v = v_cos / jnp.maximum(cos_lat_3d, 1e-6)

    # Surface fields: (n_lat, n_lon)
    lnps = sh_synthesis(grid, state.lnps_hat.data)
    phis = sh_synthesis(grid, state.phis_hat.data)

    # Pull q_v from the tracer dict when present.  Container is duck-
    # typed (Field vs raw) for symmetry with the dycore RHS / physics
    # bridges.  Cast to T's dtype to keep the channel tensor uniform.
    if state.tracers is not None and "q_v" in state.tracers:
        _qv_raw = state.tracers["q_v"]
        _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
        q = _qv_data.astype(T.dtype)
    else:
        q = jnp.zeros_like(T)

    packed = jnp.concatenate(
        [u, v, T, q,
         lnps[..., None], phis[..., None]],
        axis=-1,
    )
    return packed


def unpack_pe_output(
    output: jnp.ndarray,
    state,
    grid: GaussianGrid,
    mode: str = "state_update",
) -> "SpectralHydrostaticState":
    """Unpack SFNO output back into a SpectralHydrostaticState.

    The q output channel is read and emitted as ``state.tracers["q_v"]``
    (grid-space, matching the spectral PE tracer convention) when the
    input state carries a q_v tracer.  In ``mode="tendencies"`` this
    is the SFNO's predicted ``dq_v/dt`` (consumed by the dycore RHS
    via the physics_tendency.tracers path); in ``mode="state_update"``
    it is the new q_v field directly.

    Mirrors :func:`pack_pe_state` — without this round-trip, the SFNO
    output's q channel would be silently discarded.

    Parameters
    ----------
    output : array, shape (n_lat, n_lon, n_channels)
        SFNO output in grid space.
    state : SpectralHydrostaticState
        Original state for metadata.
    grid : GaussianGrid
        Grid for SH analysis.
    mode : str
        "state_update" or "tendencies".

    Returns
    -------
    SpectralHydrostaticState
    """
    if mode not in ("state_update", "tendencies"):
        raise ValueError(
            f"Unknown unpack mode: {mode!r}. "
            f"Choose 'state_update' or 'tendencies'."
        )
    nlev = state.T_hat.data.shape[-1]
    spec = PE3DChannelSpec(nlev=nlev)

    u = output[..., spec.u_slice].astype(jnp.float64)
    v = output[..., spec.v_slice].astype(jnp.float64)
    T = output[..., spec.T_slice].astype(jnp.float64)
    q = output[..., spec.q_slice].astype(jnp.float64)
    lnps = output[..., spec.lnps_idx].astype(jnp.float64)

    # T → spectral
    T_hat = sh_analysis_3d(grid, T)

    # lnps → spectral
    lnps_hat = sh_analysis(grid, lnps)

    # u, v → vor, div in spectral space
    a = grid.radius
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a
    cos_lat_3d = grid.cos_lat[:, None, None]

    u_cos = u * cos_lat_3d
    v_cos = v * cos_lat_3d

    # vor = curl(u,v), div = div(u,v)
    vor_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, u_cos)
    )
    div_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, v_cos)
    )

    if mode == "tendencies":
        phis_hat = jnp.zeros_like(state.phis_hat.data)
    else:
        phis_hat = state.phis_hat.data

    # Build the tracer dict for the output state.  We mirror the input
    # state's tracer pytree:
    #   * if input has q_v → emit q_v (the SFNO's predicted dq_v/dt
    #     in tendency mode, or new q_v in state-update mode)
    #   * additional tracer keys (q_c, q_i, q_r, ...) get no network
    #     prediction.  In "tendencies" mode they are mirrored as ZEROS
    #     (zero tendency = unchanged — correct, and keeps the pytree
    #     structure aligned for jax.tree.map).  In "state_update" mode
    #     the output IS the next state, so zeros would ERASE q_c/q_i;
    #     carry the INPUT state's values through unchanged instead.
    #   * if input.tracers is None → output also None (legacy dry path)
    tracers_out = None
    if state.tracers is not None:
        if mode == "tendencies":
            tracers_out = zero_like_tracers(state.tracers) or {}
        else:  # state_update (mode validated at entry)
            tracers_out = dict(state.tracers)
        if "q_v" in state.tracers:
            template = state.tracers["q_v"]
            if hasattr(template, "data") and hasattr(template, "replace"):
                tracers_out["q_v"] = template.replace(
                    data=q.astype(template.data.dtype),
                )
            else:
                tracers_out["q_v"] = q.astype(template.dtype)

    return SpectralHydrostaticState(
        vor_hat=state.vor_hat.replace(data=vor_hat),
        div_hat=state.div_hat.replace(data=div_hat),
        T_hat=state.T_hat.replace(data=T_hat),
        lnps_hat=state.lnps_hat.replace(data=lnps_hat),
        phis_hat=state.phis_hat.replace(data=phis_hat),
        tracers=tracers_out,
    )
