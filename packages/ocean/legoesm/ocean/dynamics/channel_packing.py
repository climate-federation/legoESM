"""Spectral-ocean channel packing for the SFNO ocean dynamical core.

Moved out of ``ml/channel_packing.py`` so the shared ML packing hub no longer
imports the ocean component (and ``ocean.sfno_ocean`` no longer imports the ML
hub that pulls in the *atmosphere* component) — that cross-layer coupling made
atmosphere and ocean mutually dependent through ``ml``, which blocked component
independence (see the import-linter contracts in ``pyproject.toml``).

The packing is intra-ocean: it round-trips a ``SpectralOceanState`` through the
dense channel tensor the SFNO consumes.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.grids.gaussian import (
    GaussianGrid,
    sh_analysis,
    sh_analysis_3d,
    sh_analysis_dmu_3d,
    sh_analysis_oc2_3d,
    sh_synthesis,
    sh_synthesis_3d,
    uv_from_vordiv_3d,
)
from legoesm.ocean.state import SpectralOceanState


class OceanChannelSpec(NamedTuple):
    """Channel layout for spectral ocean packing.

    Channels: [u(nlev), v(nlev), T(nlev), S(nlev), eta, H_bathy]
    """
    nlev: int = 10
    n_base_vars: int = 4  # u, v, T, S
    n_surface: int = 2    # eta, H_bathy

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
    def S_slice(self) -> slice:
        s = 3 * self.nlev
        return slice(s, s + self.nlev)

    @property
    def eta_idx(self) -> int:
        return self.n_base_vars * self.nlev

    @property
    def H_bathy_idx(self) -> int:
        return self.n_base_vars * self.nlev + 1


def pack_ocean_state(
    state,
    grid: GaussianGrid,
) -> jnp.ndarray:
    """Pack a SpectralOceanState into a dense tensor for the SFNO.

    Parameters
    ----------
    state : SpectralOceanState
        Spectral ocean state.
    grid : GaussianGrid
        Grid for SH transforms.

    Returns
    -------
    array, shape (n_lat, n_lon, n_channels)
        Packed grid-space fields: [u(nlev), v(nlev), T(nlev), S(nlev), eta, H_bathy].
    """
    # 3D fields: (n_lat, n_lon, nlev)
    T = sh_synthesis_3d(grid, state.T_hat.data)
    S = sh_synthesis_3d(grid, state.S_hat.data)

    # u, v from vorticity-divergence
    u_cos, v_cos = uv_from_vordiv_3d(
        grid, state.vor_hat.data, state.div_hat.data
    )
    cos_lat_3d = grid.cos_lat[:, None, None]
    u = u_cos / jnp.maximum(cos_lat_3d, 1e-6)
    v = v_cos / jnp.maximum(cos_lat_3d, 1e-6)

    # Apply land mask
    mask_3d = state.land_mask_grid.data[..., None]
    u = u * mask_3d
    v = v * mask_3d

    # Surface fields: (n_lat, n_lon)
    eta = sh_synthesis(grid, state.eta_hat.data)
    H_bathy = sh_synthesis(grid, state.H_bathy_hat.data)

    packed = jnp.concatenate(
        [u, v, T, S,
         eta[..., None], H_bathy[..., None]],
        axis=-1,
    )
    return packed


def unpack_ocean_output(
    output: jnp.ndarray,
    state,
    grid: GaussianGrid,
    mode: str = "state_update",
) -> SpectralOceanState:
    """Unpack SFNO output back into a SpectralOceanState.

    Parameters
    ----------
    output : array, shape (n_lat, n_lon, n_channels)
        SFNO output in grid space.
    state : SpectralOceanState
        Original state for metadata and static fields.
    grid : GaussianGrid
        Grid for SH analysis.
    mode : str
        "state_update" or "tendencies".

    Returns
    -------
    SpectralOceanState
    """
    nlev = state.T_hat.data.shape[-1]
    spec = OceanChannelSpec(nlev=nlev)

    u = output[..., spec.u_slice].astype(jnp.float64)
    v = output[..., spec.v_slice].astype(jnp.float64)
    T = output[..., spec.T_slice].astype(jnp.float64)
    S = output[..., spec.S_slice].astype(jnp.float64)
    eta = output[..., spec.eta_idx].astype(jnp.float64)

    # Apply land mask
    mask = state.land_mask_grid.data
    mask_3d = mask[..., None]
    u = u * mask_3d
    v = v * mask_3d
    eta = eta * mask

    # T, S → spectral
    T_hat = sh_analysis_3d(grid, T)
    S_hat = sh_analysis_3d(grid, S)

    # eta → spectral
    eta_hat = sh_analysis(grid, eta)

    # u, v → vor, div in spectral space
    a = grid.radius
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a
    cos_lat_3d_f64 = grid.cos_lat[:, None, None]

    u_cos = u * cos_lat_3d_f64
    v_cos = v * cos_lat_3d_f64

    vor_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, u_cos)
    )
    div_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, v_cos)
    )

    if mode == "tendencies":
        # Static fields have zero tendencies
        H_bathy_hat = jnp.zeros_like(state.H_bathy_hat.data)
        land_mask_grid = jnp.zeros_like(state.land_mask_grid.data)
    else:
        H_bathy_hat = state.H_bathy_hat.data
        land_mask_grid = state.land_mask_grid.data

    return SpectralOceanState(
        vor_hat=state.vor_hat.replace(data=vor_hat),
        div_hat=state.div_hat.replace(data=div_hat),
        T_hat=state.T_hat.replace(data=T_hat),
        S_hat=state.S_hat.replace(data=S_hat),
        eta_hat=state.eta_hat.replace(data=eta_hat),
        H_bathy_hat=state.H_bathy_hat.replace(data=H_bathy_hat),
        land_mask_grid=state.land_mask_grid.replace(data=land_mask_grid),
    )
