"""Gaussian grid and spherical harmonic transforms for spectral methods.

Implements a regular Gaussian grid (equispaced longitude, Gaussian quadrature
latitude) with forward/inverse spherical harmonic (SH) transforms. The SH
transforms use real FFT in longitude and matrix-multiply with precomputed
associated Legendre polynomials in latitude.

Triangular truncation: 0 <= m <= n <= n_max, with n_sh = (n_max+1)*(n_max+2)/2
spectral coefficients (only m >= 0 stored; conjugate symmetry for real fields).

References
----------
- Durran, D. R. (2010). Numerical Methods for Fluid Dynamics.
- Hack & Jakob (1992). Description of a Global Shallow Water Model Based on
  the Spectral Transform Method. NCAR Technical Note.
- Swarztrauber, P. N. (1996). Spectral Transform Methods for Solving the
  Shallow-Water Equations on the Sphere. Monthly Weather Review.
"""

from __future__ import annotations

import math
from typing import NamedTuple
import warnings

import numpy as np
import jax
import jax.numpy as jnp

from legoesm import constants


# =============================================================================
# Grid definition
# =============================================================================

class GaussianGrid(NamedTuple):
    """Gaussian grid with precomputed spectral transform matrices.

    All arrays are JAX arrays (float64 or complex128).
    """
    n_lat: int              # Number of latitude points
    n_lon: int              # Number of longitude points
    n_max: int              # Spectral truncation (triangular)
    radius: float           # Sphere radius [m]
    lat: jax.Array          # Gaussian latitudes [rad], shape (n_lat,), S->N
    lon: jax.Array          # Equispaced longitudes [rad], shape (n_lon,)
    lat2d: jax.Array        # 2D latitude, shape (n_lat, n_lon)
    lon2d: jax.Array        # 2D longitude, shape (n_lat, n_lon)
    cos_lat: jax.Array      # cos(lat), shape (n_lat,)
    sin_lat: jax.Array      # sin(lat), shape (n_lat,)
    f: jax.Array            # Coriolis = 2*Omega*sin(lat), shape (n_lat, n_lon)
    weights: jax.Array      # Gaussian quadrature weights, shape (n_lat,)
    Pnm: jax.Array          # Assoc. Legendre P_n^m, (n_lat, n_sh)
    Hnm: jax.Array          # -(1-mu^2)*dP/dmu = cos(lat)*dP/dtheta, (n_lat, n_sh)
    Pnm_oc2: jax.Array      # P_n^m / cos^2(lat), (n_lat, n_sh)
    Dnm: jax.Array           # dP_n^m/dmu = -Hnm/cos^2(lat), (n_lat, n_sh)
    n_sh: int               # Number of spectral coefficients
    ls: jax.Array           # Total wavenumber n for each SH index, (n_sh,)
    ms: jax.Array           # Zonal wavenumber m for each SH index, (n_sh,)
    lap: jax.Array          # Spectral Laplacian = -n(n+1)/a^2, (n_sh,)
    ilap: jax.Array         # Inverse Laplacian (0 for n=0), (n_sh,)

    # ------------------------------------------------------------------
    # GridProtocol properties
    # ------------------------------------------------------------------

    @property
    def grid_lat(self) -> jax.Array:
        return self.lat2d

    @property
    def grid_lon(self) -> jax.Array:
        return self.lon2d

    @property
    def grid_area(self):
        dlon = 2.0 * jnp.pi / self.n_lon
        return self.radius ** 2 * dlon * jnp.broadcast_to(
            self.weights[:, None], (self.n_lat, self.n_lon)
        )

    @property
    def grid_total_area(self):
        return jnp.sum(self.grid_area)

    @property
    def grid_coriolis(self) -> jax.Array:
        return self.f

    @property
    def grid_radius(self) -> float:
        return self.radius

    @property
    def grid_n_columns(self) -> int:
        return self.n_lat * self.n_lon

    def to_columns(self, field):
        extra = field.shape[2:]
        return field.reshape(self.n_lat * self.n_lon, *extra)

    def from_columns(self, cols):
        extra = cols.shape[1:]
        return cols.reshape(self.n_lat, self.n_lon, *extra)


def create_gaussian_grid(
    n_max: int,
    radius: float = constants.R_earth,
    *,
    dealiasing: str = "quadratic",
    allow_unsupported_backend: bool = False,
    legoesm_config=None,
) -> GaussianGrid:
    """Create a Gaussian grid with precomputed SH transform matrices.

    Parameters
    ----------
    n_max : int
        Spectral truncation.
    radius : float
        Sphere radius [m].
    dealiasing : str
        Dealiasing rule for the transform grid size:
        - ``"quadratic"`` (default): n_lat = 3*(n_max+1)//2.
          Exact for products of 2 spectral fields (shallow water).
        - ``"cubic"``: n_lat = 2*(n_max+1).
          Exact for products of 3 spectral fields (primitive equations).
        - ``"linear"``: n_lat = n_max + 1. No dealiasing.
    allow_unsupported_backend : bool
        If True, bypass the backend compatibility check (expert only).
        Default False.
    legoesm_config : Config, optional
        legoESM global configuration. If provided, the
        ``atmosphere.spectral.allow_unsupported`` value is used
        (overrides *allow_unsupported_backend*).
    """
    # Extract allow_unsupported from global config if provided
    if legoesm_config is not None:
        allow_unsupported_backend = bool(
            legoesm_config.get("atmosphere.spectral.allow_unsupported", False)
        )

    from legoesm.core.hardware import check_spectral_backend, get_backend
    backend = get_backend()

    # On Metal we can still run spectral dynamics by hosting grid/transforms on
    # CPU and routing the spectral model there. Keep strict x64 requirement.
    if backend == "METAL":
        if not jax.config.jax_enable_x64:
            raise ValueError(
                "Gaussian spectral grid on Metal requires JAX_ENABLE_X64=True "
                "for CPU spectral fallback."
            )
        if not allow_unsupported_backend:
            warnings.warn(
                "Metal backend detected. Creating Gaussian spectral grid on CPU "
                "for spectral fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
        allow_unsupported_backend = True

    # Guard: spectral code requires float64/complex128
    check_spectral_backend(allow_unsupported=allow_unsupported_backend)

    # Grid dimensions based on dealiasing rule
    if dealiasing == "cubic":
        n_lat = 2 * (n_max + 1)
    elif dealiasing == "quadratic":
        n_lat = 3 * (n_max + 1) // 2
    elif dealiasing == "linear":
        n_lat = n_max + 1
    else:
        raise ValueError(
            f"Unknown dealiasing={dealiasing!r}. "
            "Use 'quadratic', 'cubic', or 'linear'."
        )
    # Make n_lat even for FFT efficiency
    if n_lat % 2 != 0:
        n_lat += 1
    n_lon = 2 * n_lat

    # Gaussian quadrature points and weights on [-1, 1]
    # x = cos(theta) = sin(lat), so lat = arcsin(x)
    x_gauss, w_gauss = np.polynomial.legendre.leggauss(n_lat)
    # Sort south to north (x from -1 to +1 => lat from -pi/2 to +pi/2)
    idx = np.argsort(x_gauss)
    x_gauss = x_gauss[idx]
    w_gauss = w_gauss[idx]

    lat_np = np.arcsin(x_gauss)  # Gaussian latitudes [rad]
    lon_np = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)  # [0, 2pi)

    cos_lat_np = np.cos(lat_np)
    sin_lat_np = np.sin(lat_np)  # = x_gauss

    # Spectral index arrays
    n_sh = (n_max + 1) * (n_max + 2) // 2
    ls_np = np.zeros(n_sh, dtype=np.int32)
    ms_np = np.zeros(n_sh, dtype=np.int32)
    k = 0
    for n in range(n_max + 1):
        for m in range(n + 1):
            ls_np[k] = n
            ms_np[k] = m
            k += 1

    # Laplacian eigenvalues: -n(n+1)/a^2
    nn = ls_np.astype(np.float64)
    lap_np = -nn * (nn + 1.0) / (radius * radius)
    ilap_np = np.zeros_like(lap_np)
    np.divide(1.0, lap_np, where=nn > 0, out=ilap_np)

    # Compute associated Legendre polynomials and their derivatives
    Pnm_np, Hnm_np = _compute_legendre(n_max, x_gauss, cos_lat_np)

    # Precompute matrices for spectral div/curl operators:
    # Pnm_oc2 = Pnm / cos²(lat) — used in sh_analysis for zonal derivative terms
    # Dnm = dPnm/dmu = -Hnm / cos²(lat) — used in meridional derivative terms
    cos2_lat_np = cos_lat_np * cos_lat_np  # cos²(lat), always > 0 at Gauss pts
    Pnm_oc2_np = Pnm_np / cos2_lat_np[:, None]
    Dnm_np = -Hnm_np / cos2_lat_np[:, None]

    # Build 2D grids
    lon2d_np, lat2d_np = np.meshgrid(lon_np, lat_np)
    f_np = 2.0 * constants.Omega * sin_lat_np[:, None] * np.ones((1, n_lon))

    target_device = jax.devices("cpu")[0] if backend == "METAL" else None

    def _to_jax(array, dtype):
        np_arr = np.asarray(array, dtype=dtype)
        if target_device is None:
            return jnp.array(np_arr, dtype=dtype)
        return jax.device_put(np_arr, target_device)

    return GaussianGrid(
        n_lat=n_lat,
        n_lon=n_lon,
        n_max=n_max,
        radius=float(radius),
        lat=_to_jax(lat_np, np.float64),
        lon=_to_jax(lon_np, np.float64),
        lat2d=_to_jax(lat2d_np, np.float64),
        lon2d=_to_jax(lon2d_np, np.float64),
        cos_lat=_to_jax(cos_lat_np, np.float64),
        sin_lat=_to_jax(sin_lat_np, np.float64),
        f=_to_jax(f_np, np.float64),
        weights=_to_jax(w_gauss, np.float64),
        Pnm=_to_jax(Pnm_np, np.float64),
        Hnm=_to_jax(Hnm_np, np.float64),
        Pnm_oc2=_to_jax(Pnm_oc2_np, np.float64),
        Dnm=_to_jax(Dnm_np, np.float64),
        n_sh=n_sh,
        ls=_to_jax(ls_np, np.int32),
        ms=_to_jax(ms_np, np.int32),
        lap=_to_jax(lap_np, np.float64),
        ilap=_to_jax(ilap_np, np.float64),
    )


# =============================================================================
# Associated Legendre polynomials (computed with numpy at init time)
# =============================================================================

def _compute_legendre(
    n_max: int,
    x: np.ndarray,
    cos_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fully normalized associated Legendre polynomials and derivatives.

    Uses the standard three-term recursion with full (4pi) normalization:
        integral over sphere of |Y_n^m|^2 dA = 1

    Parameters
    ----------
    n_max : int
        Maximum total wavenumber.
    x : ndarray, shape (n_lat,)
        sin(lat) = cos(colatitude) at Gaussian latitudes.
    cos_lat : ndarray, shape (n_lat,)
        cos(lat) = sin(colatitude).

    Returns
    -------
    Pnm : ndarray, shape (n_lat, n_sh)
        P_n^m(x) for each latitude and spectral index.
    Hnm : ndarray, shape (n_lat, n_sh)
        dP_n^m/d(theta) for each latitude, where theta = colatitude.
    """
    n_lat = len(x)
    n_sh = (n_max + 1) * (n_max + 2) // 2
    Pnm = np.zeros((n_lat, n_sh), dtype=np.float64)
    Hnm = np.zeros((n_lat, n_sh), dtype=np.float64)

    sin_theta = cos_lat  # sin(colatitude) = cos(latitude)

    def _idx(n, m):
        return n * (n + 1) // 2 + m

    # Sectoral seed: P_m^m
    # P_0^0 = 1/sqrt(4*pi)
    Pnm[:, _idx(0, 0)] = 1.0 / np.sqrt(4.0 * np.pi)

    # Build P_m^m for m = 1, ..., n_max via:
    # P_m^m = -sqrt((2m+1)/(2m)) * sin_theta * P_{m-1}^{m-1}
    for m in range(1, n_max + 1):
        factor = np.sqrt((2.0 * m + 1.0) / (2.0 * m))
        Pnm[:, _idx(m, m)] = -factor * sin_theta * Pnm[:, _idx(m - 1, m - 1)]

    # Tesseral recursion: P_{m+1}^m from P_m^m
    for m in range(n_max):
        n = m + 1
        a_coeff = np.sqrt(
            (2.0 * n - 1.0) * (2.0 * n + 1.0) / ((n - m) * (n + m))
        )
        Pnm[:, _idx(n, m)] = a_coeff * x * Pnm[:, _idx(m, m)]

    # General recursion: P_n^m from P_{n-1}^m and P_{n-2}^m
    for m in range(n_max + 1):
        for n in range(m + 2, n_max + 1):
            a_coeff = np.sqrt(
                (4.0 * n * n - 1.0) / (n * n - m * m)
            )
            b_coeff = np.sqrt(
                ((n - 1.0) ** 2 - m * m) / (4.0 * (n - 1.0) ** 2 - 1.0)
            )
            Pnm[:, _idx(n, m)] = (
                a_coeff * (x * Pnm[:, _idx(n - 1, m)]
                           - b_coeff * Pnm[:, _idx(n - 2, m)])
            )

    # Compute derivatives: dP_n^m/d(theta)
    # Using the recurrence:
    #   dP_n^m/dtheta = n*x/sin^2(theta) * P_n^m
    #                 - sqrt((2n+1)/(2n-1) * (n^2-m^2)) / sin(theta) * P_{n-1}^m
    # But this is unstable near poles. Instead use:
    #   dP_n^m/dtheta = (1/sin_theta) * [n*x*P_n^m - sqrt((2n+1)/(2n-1)*(n^2-m^2))*P_{n-1}^m]
    # For n=m (sectoral), use:
    #   dP_m^m/dtheta = m * (x / sin_theta) * P_m^m
    # which is m * cot(theta) * P_m^m at non-polar points.

    # Use a safer recursion for derivatives. The standard approach:
    # H_n^m = -(1-mu^2)*dP_n^m/dmu  (= cos(lat)*dP/d(colatitude))
    # Using the identity: (1-x^2)*dP/dx = (n+1)*eps_{n,m}*P_{n-1}^m - n*eps_{n+1,m}*P_{n+1}^m
    # where eps_{n,m} = sqrt((n^2 - m^2) / (4*n^2 - 1))
    # So H = -(1-x^2)*dP/dx = n*eps_{n+1,m}*P_{n+1}^m - (n+1)*eps_{n,m}*P_{n-1}^m
    for m in range(n_max + 1):
        for n in range(m, n_max + 1):
            k = _idx(n, m)
            # Term 1: -(n+1) * eps_{n,m} * P_{n-1}^m
            if n > m:
                eps_nm = np.sqrt((n * n - m * m) / (4.0 * n * n - 1.0))
                Hnm[:, k] -= (n + 1.0) * eps_nm * Pnm[:, _idx(n - 1, m)]
            # Term 2: +n * eps_{n+1,m} * P_{n+1}^m
            if n < n_max:
                n1 = n + 1
                eps_n1m = np.sqrt(
                    (n1 * n1 - m * m) / (4.0 * n1 * n1 - 1.0)
                )
                Hnm[:, k] += n * eps_n1m * Pnm[:, _idx(n1, m)]

    return Pnm, Hnm


# =============================================================================
# Spectral index helpers
# =============================================================================

def _sh_idx(n: int, m: int) -> int:
    """Flat spectral index for (n, m) in triangular truncation."""
    return n * (n + 1) // 2 + m


# =============================================================================
# Spherical harmonic transforms
# =============================================================================

def sh_analysis(grid: GaussianGrid, field_grid: jax.Array) -> jax.Array:
    """Forward spherical harmonic transform: grid -> spectral.

    Parameters
    ----------
    grid : GaussianGrid
    field_grid : array, shape (n_lat, n_lon)
        Real-valued field on the Gaussian grid.

    Returns
    -------
    coeffs : complex array, shape (n_sh,)
        Spectral coefficients for m >= 0.
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    n_max = grid.n_max
    n_sh = grid.n_sh

    # 1. FFT in longitude -> Fourier coefficients for each latitude
    # rfft gives m = 0, 1, ..., n_lon/2   (shape: n_lat x (n_lon//2 + 1))
    f_hat_lon = jnp.fft.rfft(field_grid, axis=1)  # (n_lat, n_lon//2+1)

    # Normalize: FFT gives sum, we need mean * 2pi for the SH convention
    # Actually the standard: f_m(lat) = (1/n_lon) * sum_j f(lat, lon_j) * e^{-im*lon_j}
    # rfft already gives the un-normalized sum, so divide by n_lon
    f_hat_lon = f_hat_lon / n_lon  # (n_lat, n_lon//2+1)

    # 2. Legendre transform: for each m, contract over latitude
    # coeffs[k] = 2*pi * sum_j w_j * P_n^m(x_j) * f_hat_lon(j, m)
    # where k = _sh_idx(n, m)

    # Extract the needed Fourier modes (m = 0, 1, ..., n_max)
    # f_hat_lon has modes 0..n_lon//2, we need 0..n_max
    f_m = f_hat_lon[:, :n_max + 1]  # (n_lat, n_max+1)

    # Build coefficients: for each (n,m), sum over latitudes
    # coeffs[k] = 2*pi * sum_lat [ w[lat] * Pnm[lat, k] * f_m[lat, m_of_k] ]
    # We can do this as a matrix multiply if we construct the weighted Legendre matrix

    # Weighted Legendre: (n_lat, n_sh) * weights[:, None]
    wPnm = grid.Pnm * grid.weights[:, None]  # (n_lat, n_sh)

    # For each spectral index k with zonal wavenumber m_k, pick f_m[:, m_k]
    # and contract: coeffs[k] = 2*pi * sum_lat wPnm[lat, k] * f_m[lat, m_k]

    # Gather the right Fourier mode for each spectral index
    ms = grid.ms  # (n_sh,) int
    f_m_gathered = f_m[:, ms]  # (n_lat, n_sh)

    # Contract over latitude
    coeffs = 2.0 * jnp.pi * jnp.sum(wPnm * f_m_gathered, axis=0)  # (n_sh,)

    return coeffs


def sh_synthesis(grid: GaussianGrid, coeffs: jax.Array) -> jax.Array:
    """Inverse spherical harmonic transform: spectral -> grid.

    Parameters
    ----------
    grid : GaussianGrid
    coeffs : complex array, shape (n_sh,)
        Spectral coefficients (m >= 0).

    Returns
    -------
    field_grid : real array, shape (n_lat, n_lon)
        Field on the Gaussian grid.
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    n_max = grid.n_max

    # 1. Legendre synthesis: for each latitude and m, sum over n
    # f_m(lat, m) = sum_{n=m}^{n_max} coeffs[k(n,m)] * Pnm[lat, k(n,m)]
    # This is a matrix multiply: f_m = Pnm @ coeffs -> but we need per-m grouping

    # Scatter-add: for each spectral index k with wavenumber m_k,
    # f_m[lat, m_k] += Pnm[lat, k] * coeffs[k]
    ms = grid.ms  # (n_sh,)

    # Pnm * coeffs -> (n_lat, n_sh), then scatter-add into (n_lat, n_max+1)
    contributions = grid.Pnm * coeffs[None, :]  # (n_lat, n_sh)

    # Use segment_sum or scatter to accumulate per m
    # f_m shape: (n_lat, n_max+1)
    f_m = jnp.zeros((n_lat, n_max + 1), dtype=jnp.complex128)
    # For each m, sum contributions from all n >= m
    # This can be done as a scatter:
    f_m = f_m.at[:, ms].add(contributions)

    # 2. Inverse FFT in longitude
    # Pad f_m to the right shape for irfft: (n_lat, n_lon//2 + 1)
    f_hat_full = jnp.zeros((n_lat, n_lon // 2 + 1), dtype=jnp.complex128)
    f_hat_full = f_hat_full.at[:, :n_max + 1].set(f_m)

    # irfft expects the un-normalized FFT convention: multiply by n_lon
    field_grid = jnp.fft.irfft(f_hat_full * n_lon, n=n_lon, axis=1)

    return field_grid.real


def sh_analysis_oc2(grid: GaussianGrid, field_grid: jax.Array) -> jax.Array:
    """Forward SH transform with 1/cos²(lat) weighting (Pnm_oc2 matrix).

    Computes: 2π * ∫ f_m(μ) * [Pnm(μ)/cos²φ] dμ
    Used for the zonal (∂/∂λ) terms in spectral div/curl operators.
    Avoids dividing f by cos²φ in physical space (pole-safe).
    """
    n_max = grid.n_max

    f_hat_lon = jnp.fft.rfft(field_grid, axis=1) / grid.n_lon
    f_m = f_hat_lon[:, :n_max + 1]

    wPnm_oc2 = grid.Pnm_oc2 * grid.weights[:, None]
    f_m_gathered = f_m[:, grid.ms]

    coeffs = 2.0 * jnp.pi * jnp.sum(wPnm_oc2 * f_m_gathered, axis=0)
    return coeffs


def sh_analysis_dmu(grid: GaussianGrid, field_grid: jax.Array) -> jax.Array:
    """Forward SH transform with dPnm/dμ weighting (Dnm matrix).

    Computes: 2π * ∫ f_m(μ) * [dPnm/dμ] dμ
    Used for the meridional (∂/∂μ) terms in spectral div/curl operators.
    The spectral divergence ∂V/∂μ term = -sh_analysis_dmu(V).
    """
    n_max = grid.n_max

    f_hat_lon = jnp.fft.rfft(field_grid, axis=1) / grid.n_lon
    f_m = f_hat_lon[:, :n_max + 1]

    wDnm = grid.Dnm * grid.weights[:, None]
    f_m_gathered = f_m[:, grid.ms]

    coeffs = 2.0 * jnp.pi * jnp.sum(wDnm * f_m_gathered, axis=0)
    return coeffs


def sh_analysis_H(grid: GaussianGrid, field_grid: jax.Array) -> jax.Array:
    """Forward SH transform using derivative Legendre polynomials Hnm.

    Same as sh_analysis but uses grid.Hnm instead of grid.Pnm. This is
    used for computing spectral divergence and curl of (u*cos_lat, v*cos_lat).

    Parameters
    ----------
    grid : GaussianGrid
    field_grid : array, shape (n_lat, n_lon)

    Returns
    -------
    coeffs : complex array, shape (n_sh,)
    """
    n_max = grid.n_max
    n_sh = grid.n_sh

    f_hat_lon = jnp.fft.rfft(field_grid, axis=1) / grid.n_lon
    f_m = f_hat_lon[:, :n_max + 1]

    wHnm = grid.Hnm * grid.weights[:, None]
    f_m_gathered = f_m[:, grid.ms]

    coeffs = 2.0 * jnp.pi * jnp.sum(wHnm * f_m_gathered, axis=0)

    return coeffs


# =============================================================================
# Velocity from vorticity and divergence
# =============================================================================

def uv_from_vordiv(
    grid: GaussianGrid,
    vor_hat: jax.Array,
    div_hat: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute (u*cos_lat, v*cos_lat) on the grid from spectral vor & div.

    Uses streamfunction psi and velocity potential chi:
        psi_hat = -inv_lap * vor_hat
        chi_hat = -inv_lap * div_hat
        u*cos_lat = -d(psi)/d(lat) + (1/cos_lat) * d(chi)/d(lon)
                  = d(psi)/d(theta) + (1/cos_lat) * d(chi)/d(lon)
        v*cos_lat = (1/cos_lat) * d(psi)/d(lon) + d(chi)/d(lat)
                  = (1/cos_lat) * d(psi)/d(lon) - d(chi)/d(theta)

    where theta is colatitude (d/d(lat) = -d/d(theta)).

    In spectral space:
        d/d(theta) -> multiply by Hnm (derivative Legendre)
        d/d(lon) -> multiply by im (applied during synthesis)

    Parameters
    ----------
    grid : GaussianGrid
    vor_hat : complex array, shape (n_sh,)
    div_hat : complex array, shape (n_sh,)

    Returns
    -------
    u_cos : real array, shape (n_lat, n_lon)
        u * cos(lat) on the grid.
    v_cos : real array, shape (n_lat, n_lon)
        v * cos(lat) on the grid.
    """
    a = grid.radius

    # Streamfunction and velocity potential in spectral space
    psi_hat = -grid.ilap * vor_hat / (a * a) * (a * a)  # = -ilap * vor_hat
    # Actually: ilap = -a^2 / (n(n+1)), so psi_hat = ilap * vor_hat already
    # gives psi in units that need scaling. Let's be careful:
    #
    # Laplacian eigenvalue: lap = -n(n+1)/a^2
    # inv_lap = -a^2/(n(n+1))  (for n>0)
    # psi_hat = inv_lap * vor_hat  (on the sphere, nabla^2 psi = vor)
    # chi_hat = inv_lap * div_hat  (nabla^2 chi = div)
    psi_hat = grid.ilap * vor_hat  # ilap = 1/lap = -a^2/(n(n+1))
    chi_hat = grid.ilap * div_hat

    # d(psi)/d(theta) on grid: synthesize using Hnm
    # This gives sum_nm psi_hat[nm] * Hnm[lat, nm] * exp(im*lon) / a
    # The /a comes from the sphere: gradient on sphere has 1/a factor

    # d/d(theta) component: uses Hnm in synthesis
    dpsi_dtheta = _sh_synthesis_H(grid, psi_hat) / a
    dchi_dtheta = _sh_synthesis_H(grid, chi_hat) / a

    # d/d(lon) component: multiply coeffs by im, then standard synthesis
    dpsi_dlon = sh_synthesis(grid, 1j * grid.ms * psi_hat) / a
    dchi_dlon = sh_synthesis(grid, 1j * grid.ms * chi_hat) / a

    # u*cos_lat = d(psi)/d(theta) + d(chi)/d(lon)
    u_cos = dpsi_dtheta + dchi_dlon

    # v*cos_lat = d(psi)/d(lon) - d(chi)/d(theta)
    v_cos = dpsi_dlon - dchi_dtheta

    return u_cos, v_cos


def _sh_synthesis_H(grid: GaussianGrid, coeffs: jax.Array) -> jax.Array:
    """Inverse SH transform using derivative Legendre Hnm (instead of Pnm).

    Produces the theta-derivative of the field on the grid.
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    n_max = grid.n_max
    ms = grid.ms

    contributions = grid.Hnm * coeffs[None, :]
    f_m = jnp.zeros((n_lat, n_max + 1), dtype=jnp.complex128)
    f_m = f_m.at[:, ms].add(contributions)

    f_hat_full = jnp.zeros((n_lat, n_lon // 2 + 1), dtype=jnp.complex128)
    f_hat_full = f_hat_full.at[:, :n_max + 1].set(f_m)

    field_grid = jnp.fft.irfft(f_hat_full * n_lon, n=n_lon, axis=1)
    return field_grid.real


# =============================================================================
# Spectral operators
# =============================================================================

def spectral_laplacian(grid: GaussianGrid, coeffs: jax.Array) -> jax.Array:
    """Apply spectral Laplacian: multiply by -n(n+1)/a^2."""
    return grid.lap * coeffs


def spectral_inverse_laplacian(
    grid: GaussianGrid, coeffs: jax.Array,
) -> jax.Array:
    """Apply inverse Laplacian (n=0 mode stays zero)."""
    return grid.ilap * coeffs


def spectral_hyperdiffusion(
    grid: GaussianGrid,
    coeffs: jax.Array,
    nu: float,
    order: int = 2,
) -> jax.Array:
    """Apply spectral hyperdiffusion: ``-nu * [n(n+1)/a^2]^p * coeffs``.

    This damps small scales.  The sign is always negative (dissipative)
    for any positive order.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order!r}")
    if nu < 0.0:
        raise ValueError(f"nu must be >= 0, got {nu!r}")
    if not math.isfinite(float(nu)):
        raise ValueError(f"nu must be finite, got {nu!r}")
    if nu == 0.0:
        return jnp.zeros_like(coeffs)

    a2 = grid.radius * grid.radius
    nn = grid.ls.astype(jnp.float64)
    eig = nn * (nn + 1.0) / a2  # n(n+1)/a^2
    damping = -nu * eig ** order
    damping = jnp.where(jnp.isfinite(damping), damping, 0.0)
    return damping * coeffs


# =============================================================================
# 3D (level-wise) transform wrappers via vmap
# =============================================================================

def sh_analysis_3d(grid: GaussianGrid, field_3d: jax.Array) -> jax.Array:
    """Forward SH transform per vertical level.

    Parameters
    ----------
    field_3d : (n_lat, n_lon, nlev) real array.

    Returns
    -------
    (n_sh, nlev) complex array.
    """
    f_t = jnp.moveaxis(field_3d, -1, 0)  # (nlev, n_lat, n_lon)
    result = jax.vmap(lambda f: sh_analysis(grid, f))(f_t)  # (nlev, n_sh)
    return jnp.moveaxis(result, 0, -1)  # (n_sh, nlev)


def sh_synthesis_3d(grid: GaussianGrid, coeffs_3d: jax.Array) -> jax.Array:
    """Inverse SH transform per vertical level.

    Parameters
    ----------
    coeffs_3d : (n_sh, nlev) complex array.

    Returns
    -------
    (n_lat, n_lon, nlev) real array.
    """
    c_t = jnp.moveaxis(coeffs_3d, -1, 0)  # (nlev, n_sh)
    result = jax.vmap(lambda c: sh_synthesis(grid, c))(c_t)  # (nlev, n_lat, n_lon)
    return jnp.moveaxis(result, 0, -1)  # (n_lat, n_lon, nlev)


def sh_analysis_oc2_3d(
    grid: GaussianGrid, field_3d: jax.Array,
) -> jax.Array:
    """Forward SH transform with 1/cos^2 weighting, per level.

    Parameters
    ----------
    field_3d : (n_lat, n_lon, nlev).

    Returns
    -------
    (n_sh, nlev) complex.
    """
    f_t = jnp.moveaxis(field_3d, -1, 0)
    result = jax.vmap(lambda f: sh_analysis_oc2(grid, f))(f_t)
    return jnp.moveaxis(result, 0, -1)


def sh_analysis_dmu_3d(
    grid: GaussianGrid, field_3d: jax.Array,
) -> jax.Array:
    """Forward SH transform with dPnm/dmu weighting, per level.

    Parameters
    ----------
    field_3d : (n_lat, n_lon, nlev).

    Returns
    -------
    (n_sh, nlev) complex.
    """
    f_t = jnp.moveaxis(field_3d, -1, 0)
    result = jax.vmap(lambda f: sh_analysis_dmu(grid, f))(f_t)
    return jnp.moveaxis(result, 0, -1)


def uv_from_vordiv_3d(
    grid: GaussianGrid,
    vor_hat_3d: jax.Array,
    div_hat_3d: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reconstruct (u*cos_lat, v*cos_lat) at all levels from spectral vor/div.

    Parameters
    ----------
    vor_hat_3d, div_hat_3d : (n_sh, nlev) complex arrays.

    Returns
    -------
    u_cos, v_cos : (n_lat, n_lon, nlev) real arrays.
    """
    vor_t = jnp.moveaxis(vor_hat_3d, -1, 0)  # (nlev, n_sh)
    div_t = jnp.moveaxis(div_hat_3d, -1, 0)

    def single_level(v, d):
        return uv_from_vordiv(grid, v, d)

    u_cos_t, v_cos_t = jax.vmap(single_level)(vor_t, div_t)
    return jnp.moveaxis(u_cos_t, 0, -1), jnp.moveaxis(v_cos_t, 0, -1)


def spectral_hyperdiffusion_3d(
    grid: GaussianGrid,
    coeffs_3d: jax.Array,
    nu: float,
    order: int = 2,
) -> jax.Array:
    """Apply spectral hyperdiffusion to 3D spectral field, per level.

    Pointwise in spectral space -- no vmap needed.

    Parameters
    ----------
    coeffs_3d : (n_sh, nlev) complex.

    Returns
    -------
    (n_sh, nlev) complex.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order!r}")
    if nu < 0.0:
        raise ValueError(f"nu must be >= 0, got {nu!r}")
    if not math.isfinite(float(nu)):
        raise ValueError(f"nu must be finite, got {nu!r}")
    if nu == 0.0:
        return jnp.zeros_like(coeffs_3d)

    a2 = grid.radius * grid.radius
    nn = grid.ls.astype(jnp.float64)
    eig = nn * (nn + 1.0) / a2
    damping = -nu * eig ** order  # (n_sh,)
    damping = jnp.where(jnp.isfinite(damping), damping, 0.0)
    return damping[:, None] * coeffs_3d
