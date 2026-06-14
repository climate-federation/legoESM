"""Ocean diagnostics: deformation radius and energy/enstrophy spectra.

Deformation radius follows Chelton et al. (1998) / Gill (1982):
    L_d = (1 / (pi |f|)) * integral_{-H}^{0} N(z) dz

Energy and enstrophy spectra use 2D FFT with isotropic wavenumber binning.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import compute_buoyancy_frequency, rho_0 as _RHO_0, wright_eos
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
    # ``w_total`` and the ``N_bar`` numerator share the ``dz_half``
    # weight on the same axis — fuse into one stacked reduction.
    _stack = jnp.stack([jnp.ones_like(N), N], axis=-1) * dz_half[..., None]
    _col = jnp.sum(_stack, axis=-2)
    w_total = _col[..., 0]
    N_bar = _col[..., 1] / jnp.maximum(w_total, 1e-30)
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


# ============================================================================
# Mixed-layer depth (de Boyer Montegut 2004/2022; Treguier et al. 2023 OMIP)
# ============================================================================


def mixed_layer_depth(
    T: jnp.ndarray,
    S: jnp.ndarray,
    z_centers: jnp.ndarray,
    *,
    delta_sigma: float = 0.03,
    ref_depth_m: float = 10.0,
    wet_mask: jnp.ndarray | None = None,
    bottom_depth: jnp.ndarray | None = None,
    eos_fn=wright_eos,
    p_ref_pa: float = 0.0,
) -> jnp.ndarray:
    """Density-threshold mixed-layer depth [m] (de Boyer Montegut method).

    MLD = shallowest depth below ``ref_depth_m`` where the potential density
    exceeds the value at the reference depth by ``delta_sigma``::

        MLD = min{ z > z_ref : sigma_theta(z) - sigma_theta(z_ref) >= delta_sigma }

    with linear interpolation in z to the exact crossing, anchored on a virtual
    reference point ``(z_ref, dsigma=0)`` so the search starts at 10 m rather than
    the surface.  This is the OMIP / de Boyer Montegut (2022) diagnostic used by
    Treguier et al. (2023, GMD 16:3849).

    THRESHOLD: ``delta_sigma`` default 0.03 kg/m^3 is the de Boyer Montegut /
    Treguier literature value.  NEMO ORCA1 RUN_REF writes ``mldr10_1`` with
    ``dsigma = 0.01`` wrt 10 m, so a comparison against that field MUST pass
    ``delta_sigma=0.01`` (a larger threshold gives a DEEPER MLD, so 0.03-vs-0.01
    would bias the model deep).

    Parameters
    ----------
    T, S : array, shape (..., nlev)
        Potential temperature [degC] and salinity [PSU] at level centres.
    z_centers : array, shape (nlev,)
        Level-centre depths [m, positive down], strictly increasing.
    delta_sigma : float
        Potential-density threshold [kg/m^3].
    ref_depth_m : float
        Reference depth [m] for the density anomaly (10 m, OMIP standard).
    wet_mask : array, shape (..., nlev) or None
        Per-level ocean mask in {0,1}.  Dry levels are excluded from the search
        and the reference interpolation.  ``None`` -> all wet.
    bottom_depth : array, shape (...) or None
        Per-column sea-floor depth [m] (e.g. ``H_bathy``).  Used as the MLD when
        the column is fully mixed (no crossing) -- NEMO/dBM convention.  ``None``
        -> deepest wet level centre (underestimates by up to half a bottom cell;
        a logged approximation, prefer passing H_bathy).
    eos_fn : callable
        ``fn(T, S, p) -> rho`` [kg/m^3], p in Pa.  Default ``wright_eos`` (the
        nonlinear seawater EOS) -- use it for NEMO-method parity regardless of
        what EOS the run's dynamics used.
    p_ref_pa : float
        Reference pressure for the potential density [Pa].  0.0 = sigma-theta
        referenced to the surface (dBM/NEMO convention); do NOT reference to the
        10 m pressure (the criterion is wrt the 10 m density VALUE, not pressure).

    Returns
    -------
    mld : array, shape (...)
        Mixed-layer depth [m].  NaN over fully-dry (land) columns.
    """
    nlev = z_centers.shape[0]
    lead = T.shape[:-1]
    z = jnp.asarray(z_centers, dtype=T.dtype)                       # (nlev,)
    if wet_mask is None:
        wet = jnp.ones(T.shape, dtype=T.dtype)
    else:
        wet = jnp.asarray(wet_mask, dtype=T.dtype)

    # Potential density anomaly sigma_theta = rho(T,S,p_ref) - 1000.
    sigma = eos_fn(T, S, jnp.asarray(p_ref_pa, dtype=T.dtype)) - 1000.0   # (..., nlev)

    # First wet level per column (shallowest ocean cell) -> its sigma is the
    # fallback reference when the 10 m bracket levels are not both wet.
    first_wet = jnp.argmax(wet > 0.5, axis=-1)                      # (...)
    sigma_top_wet = jnp.take_along_axis(sigma, first_wet[..., None], axis=-1)[..., 0]

    # Reference-depth density: linear interpolation of sigma onto ref_depth from
    # the two bracketing level centres (shared 1-D z, so the weights are static --
    # z_centers must be a CONCRETE array, not a jit-traced value).  If ref_depth
    # is above the first centre (coarse grid) OR either bracket is DRY, fall back
    # to the shallowest wet level's sigma (avoids interpolating through rock /
    # inventing surface structure by extrapolation).
    if float(z[0]) >= ref_depth_m:
        sigma_ref = sigma_top_wet
    else:
        i_hi = int(jnp.searchsorted(z, jnp.asarray(ref_depth_m, dtype=z.dtype), side="right"))
        i_hi = min(max(i_hi, 1), nlev - 1)
        i_lo = i_hi - 1
        w = (ref_depth_m - float(z[i_lo])) / (float(z[i_hi]) - float(z[i_lo]))
        sigma_interp = (1.0 - w) * sigma[..., i_lo] + w * sigma[..., i_hi]
        both_wet = (wet[..., i_lo] > 0.5) & (wet[..., i_hi] > 0.5)
        sigma_ref = jnp.where(both_wet, sigma_interp, sigma_top_wet)

    dsig = sigma - sigma_ref[..., jnp.newaxis]                      # (..., nlev)

    # Search only wet levels strictly below the reference depth.
    below_ref = (z > ref_depth_m)[(None,) * len(lead) + (slice(None),)]
    searchable = (wet > 0.5) & jnp.broadcast_to(below_ref, dsig.shape)
    exceed = (dsig >= delta_sigma) & searchable                    # (..., nlev)
    has_crossing = jnp.any(exceed, axis=-1)                        # (...)

    # First exceeding level (argmax of the boolean; 0 when none -> guarded below).
    idx = jnp.argmax(exceed.astype(jnp.int32), axis=-1)            # (...)
    idx_lo = jnp.maximum(idx - 1, 0)

    z_k = z[idx]                                                   # (...)
    z_km1 = z[idx_lo]
    dsig_k = jnp.take_along_axis(dsig, idx[..., None], axis=-1)[..., 0]
    dsig_km1 = jnp.take_along_axis(dsig, idx_lo[..., None], axis=-1)[..., 0]
    wet_km1 = jnp.take_along_axis(wet, idx_lo[..., None], axis=-1)[..., 0]

    # Lower bracket = the previous level ONLY if it is below the reference depth
    # AND wet (a dry gap between 10 m and the crossing must not corrupt the
    # interpolation); otherwise anchor on the virtual reference point (z_ref,
    # dsig=0).
    use_prev = (z_km1 > ref_depth_m) & (wet_km1 > 0.5)
    z_lo = jnp.where(use_prev, z_km1, jnp.asarray(ref_depth_m, dtype=T.dtype))
    dsig_lo = jnp.where(use_prev, dsig_km1, jnp.zeros_like(dsig_km1))

    denom = dsig_k - dsig_lo
    frac = jnp.where(jnp.abs(denom) > 1e-12,
                     (delta_sigma - dsig_lo) / jnp.where(denom == 0.0, 1.0, denom),
                     0.0)
    frac = jnp.clip(frac, 0.0, 1.0)
    mld_cross = z_lo + (z_k - z_lo) * frac

    # Fully mixed (no crossing) -> bottom depth.
    if bottom_depth is None:
        # Deepest wet level centre per column (logged approximation).
        wet_depth = jnp.where(wet > 0.5, z[(None,) * len(lead) + (slice(None),)], 0.0)
        bottom = jnp.max(wet_depth, axis=-1)
    else:
        bottom = jnp.asarray(bottom_depth, dtype=T.dtype)

    mld = jnp.where(has_crossing, mld_cross, bottom)

    # A crossing can never be deeper than the sea floor (defensive against an
    # inconsistent wet_mask/bottom_depth pair).
    mld = jnp.minimum(mld, bottom)

    # Shallow column (sea floor above the reference depth): the whole column is
    # the mixed layer -> MLD = bottom depth.
    mld = jnp.where(bottom < ref_depth_m, bottom, mld)

    # Land / fully-dry columns -> NaN.
    column_wet = jnp.any(wet > 0.5, axis=-1)
    mld = jnp.where(column_wet, mld, jnp.nan)
    return mld
