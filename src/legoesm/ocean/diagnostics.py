"""Ocean diagnostics: deformation radius and energy/enstrophy spectra.

Deformation radius follows Chelton et al. (1998) / Gill (1982):
    L_d = (1 / (pi |f|)) * integral_{-H}^{0} N(z) dz

Energy and enstrophy spectra use 2D FFT with isotropic wavenumber binning.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import compute_buoyancy_frequency, rho_0 as _RHO_0
from legoesm.ocean.vertical import OceanZStarCoordinate


# ============================================================================
# Phase 1c: First baroclinic deformation radius
# ============================================================================


def first_baroclinic_deformation_radius(
    rho: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    f_coriolis: jnp.ndarray,
    rho_ref: float = _RHO_0,
    f_min: float = 1e-10,
) -> jnp.ndarray:
    """First baroclinic deformation radius (Rossby radius).

    L_d = (1 / (pi |f|)) * integral_{-H}^{0} N(z) dz

    where N = sqrt(max(N^2, 0)) is the Brunt-Vaisala frequency.

    Parameters
    ----------
    rho : (..., nlev)
        In-situ density [kg/m^3].
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    jacobian : (...)
        Dynamic z-star Jacobian.
    f_coriolis : (...)
        Coriolis parameter [1/s], same horizontal shape as ``jacobian``.
    rho_ref : float
        Reference density for buoyancy frequency.
    f_min : float
        Floor on |f| to prevent blow-up near the equator.

    Returns
    -------
    L_d : (...)
        Deformation radius [m].  One value per water column.
    """
    # N^2 at interior interfaces, shape (..., nlev-1)
    N2 = compute_buoyancy_frequency(
        rho, z_coord.dz_ref, jacobian, rho_ref=rho_ref, g=constants.g,
    )
    # Tiny positive floor avoids NaN gradients in unstable layers,
    # where ``sqrt(0)`` × ``maximum``-mask backward yields ``inf*0=NaN``.
    # See _gm_redi_common.py:154 for the analogous fix.
    N = jnp.sqrt(jnp.maximum(N2, 1e-30))

    # Interface thicknesses for depth-weighted averaging.
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]  # (..., nlev)
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])  # (..., nlev-1)

    # Depth-weighted average of N, then scale to full water column depth.
    # dz_half only spans interior interfaces, missing half-layers at the
    # surface and bottom.  Using N_bar * H_col extrapolates N to the
    # boundaries — matches the Visbeck (1997) Rossby-radius convention
    # in compute_visbeck_kappa_gm().
    w_total = jnp.sum(dz_half, axis=-1)
    N_bar = jnp.sum(N * dz_half, axis=-1) / jnp.maximum(w_total, 1e-30)
    H_col = jnp.sum(dz_actual, axis=-1)

    f_safe = jnp.maximum(jnp.abs(f_coriolis), f_min)
    L_d = N_bar * H_col / (jnp.pi * f_safe)

    return L_d


# ============================================================================
# Phase 1d: Isotropic energy and enstrophy spectra
# ============================================================================


def _isotropic_spectrum_2d(
    field: jnp.ndarray,
    dx: float,
    detrend: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Isotropic power spectrum from a 2D field via FFT and radial binning.

    Parameters
    ----------
    field : (ny, nx)
        Real-valued 2D field on a regular grid.
    dx : float
        Grid spacing [m] (assumed uniform in both directions).
    detrend : bool
        If True, remove the spatial mean before computing the FFT.

    Returns
    -------
    k_bins : (n_bins,)
        Wavenumber bin centres [1/m].
    spectrum : (n_bins,)
        Power spectral density in each annular bin.
    """
    ny, nx = field.shape

    if detrend:
        field = field - jnp.mean(field)

    # 2D FFT and power spectrum (normalised by number of grid points)
    field_hat = jnp.fft.fft2(field) / (nx * ny)
    power_2d = jnp.abs(field_hat) ** 2

    # Wavenumber grids [1/m]
    kx = jnp.fft.fftfreq(nx, d=dx)
    ky = jnp.fft.fftfreq(ny, d=dx)
    kx_2d, ky_2d = jnp.meshgrid(kx, ky)
    k_mag = jnp.sqrt(kx_2d ** 2 + ky_2d ** 2)

    # Fundamental wavenumber
    dk = 1.0 / (max(nx, ny) * dx)

    # Bin into isotropic shells
    n_bins = max(nx, ny) // 2
    bin_indices = jnp.floor(k_mag / dk + 0.5).astype(jnp.int32)
    bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)

    # Sum power in each shell and normalise by dk to get spectral density
    spectrum = jnp.zeros(n_bins, dtype=field.dtype)
    spectrum = spectrum.at[bin_indices].add(power_2d)
    spectrum = spectrum / dk

    k_bins = jnp.arange(n_bins) * dk

    return k_bins, spectrum


def isotropic_energy_spectrum(
    u_h: jnp.ndarray,
    v_h: jnp.ndarray,
    dx: float,
    detrend: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Isotropic kinetic energy spectrum from cell-centre velocities.

    KE(k) = 0.5 * (|u_hat(k)|^2 + |v_hat(k)|^2), binned isotropically.

    Parameters
    ----------
    u_h : (n_lat, n_lon)
        Zonal velocity at cell centres [m/s].
    v_h : (n_lat, n_lon)
        Meridional velocity at cell centres [m/s].
    dx : float
        Grid spacing [m] (assumed approximately uniform).
    detrend : bool
        Remove spatial mean before FFT.

    Returns
    -------
    wavenumbers : (n_bins,)
        Wavenumber bin centres [1/m].
    ke_spectrum : (n_bins,)
        Kinetic energy spectral density [m^3/s^2].
    """
    k, spec_u = _isotropic_spectrum_2d(u_h, dx, detrend=detrend)
    _, spec_v = _isotropic_spectrum_2d(v_h, dx, detrend=detrend)

    return k, 0.5 * (spec_u + spec_v)


def isotropic_enstrophy_spectrum(
    zeta: jnp.ndarray,
    dx: float,
    detrend: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Isotropic enstrophy spectrum from cell-centre relative vorticity.

    Z(k) = 0.5 * |zeta_hat(k)|^2, binned isotropically.

    Parameters
    ----------
    zeta : (n_lat, n_lon)
        Relative vorticity at cell centres [1/s].
    dx : float
        Grid spacing [m].
    detrend : bool
        Remove spatial mean before FFT.

    Returns
    -------
    wavenumbers : (n_bins,)
        Wavenumber bin centres [1/m].
    enstrophy_spectrum : (n_bins,)
        Enstrophy spectral density [m/s^2].
    """
    k, spec = _isotropic_spectrum_2d(zeta, dx, detrend=detrend)

    return k, 0.5 * spec
