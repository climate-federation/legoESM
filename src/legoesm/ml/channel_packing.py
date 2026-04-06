"""Channel packing: convert legoESM states to/from flat tensors.

Provides functions to pack spectral model states into dense arrays
of shape (n_lat, n_lon, n_channels) suitable for the SFNO, and to
unpack SFNO outputs back into state objects.

Three packing schemes:
- **Shallow Water (SW)**: 4 channels — vor, div, phi, phis
- **Primitive Equations (PE 3D)**: 4*nlev + surface + forcings on
  WeatherBench2 pressure levels
- **Ocean**: 4*nlev + 2 surface — u, v, T, S at each level + eta, H_bathy
"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from legoesm.atmosphere.dynamics.spectral_sw import SpectralSWState
    from legoesm.atmosphere.dynamics.spectral_pe import SpectralHydrostaticState
    from legoesm.ocean.state import SpectralOceanState

from legoesm.core.field import Field
from legoesm.grids.gaussian import (
    GaussianGrid,
    sh_synthesis,
    sh_synthesis_3d,
    sh_analysis,
    sh_analysis_3d,
)

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

    @property
    def phis_idx(self) -> int:
        return self.n_base_vars * self.nlev + 1


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
    from legoesm.atmosphere.dynamics.spectral_sw import SpectralSWState

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

    Parameters
    ----------
    state : SpectralHydrostaticState
        Spectral PE state.
    grid : GaussianGrid
        Grid for SH transforms.
    sigma_coord : SigmaCoordinate, optional
        For computing u, v from vor, div.

    Returns
    -------
    array, shape (n_lat, n_lon, n_channels)
        Packed grid-space fields.
    """
    from legoesm.grids.gaussian import uv_from_vordiv_3d

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

    # Pack: [u(nlev), v(nlev), T(nlev), zeros_q(nlev), lnps, phis]
    nlev = T.shape[-1]
    q_placeholder = jnp.zeros_like(T)  # humidity placeholder

    packed = jnp.concatenate(
        [u, v, T, q_placeholder,
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
    from legoesm.atmosphere.dynamics.spectral_pe import SpectralHydrostaticState
    from legoesm.grids.gaussian import (
        sh_analysis_oc2_3d,
        sh_analysis_dmu_3d,
    )

    nlev = state.T_hat.data.shape[-1]
    spec = PE3DChannelSpec(nlev=nlev)

    u = output[..., spec.u_slice].astype(jnp.float64)
    v = output[..., spec.v_slice].astype(jnp.float64)
    T = output[..., spec.T_slice].astype(jnp.float64)
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

    return SpectralHydrostaticState(
        vor_hat=state.vor_hat.replace(data=vor_hat),
        div_hat=state.div_hat.replace(data=div_hat),
        T_hat=state.T_hat.replace(data=T_hat),
        lnps_hat=state.lnps_hat.replace(data=lnps_hat),
        phis_hat=state.phis_hat.replace(data=phis_hat),
    )


# ============================================================================
# Ocean packing
# ============================================================================

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
    from legoesm.grids.gaussian import uv_from_vordiv_3d

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
) -> "SpectralOceanState":
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
    from legoesm.ocean.state import SpectralOceanState
    from legoesm.grids.gaussian import (
        sh_analysis_oc2_3d,
        sh_analysis_dmu_3d,
    )

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
