"""CMIP6 ocean-diagnostic expansion: variance, dianeutral mixing, OSNAP.

Adds three diagnostics required by the CMIP6 Omon / Oyr archive +
the OSNAP intercomparison protocol:

1. **Tracer variance production** ``ε_X = 2·K·|∇X|²``: turbulent
   conversion of mean-tracer variance into sub-grid energy.  Per-cell
   per-level field for X ∈ {T, S}; used by the AMOC stability / OHC
   variance budgets.

2. **Dianeutral diffusivity inferred** ``K_dia``: sum of the
   diagnosed dianeutral components (vertical mixing + GM/Redi
   slope-dependent + tidal) at each cell.  Produced as an Omon
   diagnostic for the CMIP6 mixing-budget tables.

3. **OSNAP-class meridional section transports**: volume (Sv),
   heat (PW), salt (kg/s) transports across a meridional section
   bounded by ``[lon_min, lon_max]`` at a target latitude.  Provides
   OSNAP East (~58°N, 31°W → 7°W) and OSNAP West (~53°N, 53°W →
   33°W) presets matching the observational protocol (Lozier 2017).

References
----------
* Lozier, M. S., et al. (2017). Overturning in the Subpolar North
  Atlantic Program: A new international ocean observing system.
  *Bull. Amer. Meteor. Soc.*, 98(4), 737–752.
* Griffies, S. M., et al. (2016). OMIP contribution to CMIP6.
  *Geosci. Model Dev.*, 9(9), 3231–3296.

All helpers are pure NumPy (callers convert from JAX arrays via
``np.asarray``) so they run cheaply at the per-year diagnostic
cadence without forcing a re-compile.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from legoesm import constants


SV = 1.0e6           # 1 Sverdrup [m³/s]
PW = 1.0e15          # 1 Petawatt [W]


# ==============================================================================
# Tracer variance production
# ==============================================================================

def tracer_variance_production(
    X: np.ndarray,
    K: np.ndarray,
    dz: np.ndarray,
    *,
    axis_vertical: int = -1,
) -> np.ndarray:
    """Per-cell turbulent variance production ``ε_X = 2 · K · |∇_v X|²``.

    Approximates the VERTICAL component of the variance-production
    budget term ``2 · K · |∇X|²`` for a tracer ``X`` (T, S, ...).
    The horizontal-gradient contribution lives in the lateral
    mixing scheme and is reported separately by callers via
    ``A_h · |∇_h X|²``.

    Centered finite differences in the vertical use cell-centre
    spacing.  For uniform layer thickness ``h`` the centre-to-centre
    distance from cell ``k-1`` to ``k+1`` is

        z_{k+1} − z_{k-1} = h_k + 0.5·(h_{k-1} + h_{k+1})

    (half-cell of each end-cell plus the full thickness of the
    middle cell).  The gradient at ``k`` is then approximated by

        (∂X/∂z)_k ≈ (X_{k+1} − X_{k-1}) / Δz_k

    with ``z`` positive downward.  Surface/bottom rows fall back to
    one-sided differences using ``0.5·(h_0 + h_1)`` for the surface
    and ``0.5·(h_{n-2} + h_{n-1})`` for the bottom.

    Parameters
    ----------
    X : ndarray ``(..., nlev)``
        Tracer field (e.g. T in °C, S in PSU).  Trailing axis is
        vertical level (index 0 = surface, ``-1`` = bottom).
    K : ndarray ``(..., nlev)``
        Vertical diffusivity κ [m²/s] co-located with X.
    dz : ndarray ``(..., nlev)``
        Layer thickness [m] co-located with X.
    axis_vertical : int
        Which axis is the vertical.  Default ``-1``.

    Returns
    -------
    eps : ndarray (same shape as X)
        Variance-production rate [unit²/s] (e.g. K²/s for T).
    """
    if axis_vertical != -1:
        X = np.moveaxis(X, axis_vertical, -1)
        K = np.moveaxis(K, axis_vertical, -1)
        dz = np.moveaxis(dz, axis_vertical, -1)
    nlev = X.shape[-1]
    if nlev < 2:
        return np.zeros_like(X)

    grad = np.zeros_like(X)
    # Interior centred-difference: gradient at cell ``k`` uses cell-
    # centre separation ``h_k + 0.5·(h_{k-1} + h_{k+1})`` so the
    # finite-volume gradient matches the analytical derivative for
    # uniform-h grids (with positive-down convention).
    interior = slice(1, -1)
    denom_int = (
        dz[..., 1:-1]
        + 0.5 * (dz[..., :-2] + dz[..., 2:])
    )
    denom_int = np.where(np.abs(denom_int) > 1e-12, denom_int, 1.0)
    grad[..., interior] = (X[..., 2:] - X[..., :-2]) / denom_int

    # Surface one-sided diff: cell-centre separation between k=0 and
    # k=1 is ``0.5·(h_0 + h_1)``.
    denom_s = 0.5 * (dz[..., 0] + dz[..., 1])
    denom_s = np.where(np.abs(denom_s) > 1e-12, denom_s, 1.0)
    grad[..., 0] = (X[..., 1] - X[..., 0]) / denom_s

    # Bottom one-sided diff: same recipe at the seafloor end.
    denom_b = 0.5 * (dz[..., -2] + dz[..., -1])
    denom_b = np.where(np.abs(denom_b) > 1e-12, denom_b, 1.0)
    grad[..., -1] = (X[..., -1] - X[..., -2]) / denom_b

    eps = 2.0 * K * grad * grad
    if axis_vertical != -1:
        eps = np.moveaxis(eps, -1, axis_vertical)
    return eps


def global_tracer_variance(
    X: np.ndarray,
    h: np.ndarray,
    area: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Volume-weighted global ``⟨X²⟩ − ⟨X⟩²`` (variance) of a tracer.

    Parameters
    ----------
    X : ndarray ``(..., n_lat, n_lon, nlev)``
        Tracer field.
    h : ndarray ``(..., n_lat, n_lon, nlev)``
        Layer thickness [m].
    area : ndarray ``(..., n_lat, n_lon)``
        Cell area [m²].
    mask : ndarray ``(..., n_lat, n_lon)`` or None
        Ocean mask (1 = wet).
    """
    vol = h * area[..., None]
    if mask is not None:
        vol = vol * mask[..., None]
    total_vol = float(np.sum(vol))
    if total_vol <= 0.0:
        return 0.0
    mean = float(np.sum(X * vol)) / total_vol
    var = float(np.sum((X - mean) ** 2 * vol)) / total_vol
    return var


# ==============================================================================
# Dianeutral diffusivity inferred
# ==============================================================================

def dianeutral_diffusivity_inferred(
    K_vertical: np.ndarray,
    K_tidal: Optional[np.ndarray] = None,
    K_gm_dia: Optional[np.ndarray] = None,
    K_redi_dia: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sum the dianeutral-mixing components into a single field.

    CMIP6 Omon / Oyr expects a single ``K_dia(x, y, z)`` field
    representing the total dianeutral diffusivity from all
    parameterisations.  Components left as ``None`` contribute zero.

    Returns
    -------
    K_dia : ndarray (same shape as ``K_vertical``) [m²/s]
    """
    K = np.asarray(K_vertical, dtype=np.float64).copy()
    for extra in (K_tidal, K_gm_dia, K_redi_dia):
        if extra is not None:
            K = K + np.asarray(extra, dtype=np.float64)
    return np.maximum(K, 0.0)


# ==============================================================================
# Meridional section transports
# ==============================================================================

class SectionTransport(NamedTuple):
    """Transport across a meridional section.

    Sign convention: NORTHWARD transports are positive.

    Fields
    ------
    volume_Sv : total volume transport [Sv].
    heat_PW : potential-temperature × c_p × ρ_0 × volume transport [PW].
    salt_kg_per_s : salinity × volume transport [kg(salt)/s].
    overturning_Sv : strength of overturning cell from the depth-
        integrated streamfunction ψ = -cumsum(V_zonal) within the
        section [Sv].  Positive = northward upper / southward deep.
    """
    volume_Sv: float
    heat_PW: float
    salt_kg_per_s: float
    overturning_Sv: float


def meridional_section_transport(
    v_face: np.ndarray,
    h_partial: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    grid,
    *,
    target_lat_deg: float,
    lon_min_deg: float,
    lon_max_deg: float,
    lat_tol_deg: float = 2.0,
    rho_0: Optional[float] = None,
    c_p: Optional[float] = None,
) -> SectionTransport:
    """Transports across a meridional section on a lat-lon C-grid.

    Picks the v-face row closest to ``target_lat_deg`` and integrates
    over the longitude band ``[lon_min_deg, lon_max_deg]``.  Wrap-
    around bands (``lon_min > lon_max``) span the dateline.

    Sign convention: positive volume / heat / salt transports are
    NORTHWARD.  Overturning strength is the RAPID-style max
    ``-min(ψ)`` over depth.

    Parameters
    ----------
    v_face : ndarray ``(n_lat+1, n_lon, nlev)``
        Meridional velocity at v-faces [m/s].
    h_partial : ndarray ``(n_lat, n_lon, nlev)``
        Layer thickness [m].
    T : ndarray ``(n_lat, n_lon, nlev)``
        Potential temperature [°C].
    S : ndarray ``(n_lat, n_lon, nlev)``
        Practical salinity [PSU].
    grid : LatLonGrid-like
        Must expose ``radius`` and ``lat``/``lon`` in radians (or
        ``lat_v`` + ``dlon``).
    target_lat_deg : float
        Latitude of the section [°N].
    lon_min_deg, lon_max_deg : float
        Longitude band of the section [°E, ``[-180, 180]``-wrapped].
    lat_tol_deg : float
        If no v-face row falls within ``lat_tol_deg`` of the target,
        return all-NaN.
    rho_0, c_p : float, optional
        Reference seawater density [kg/m³] + heat capacity
        [J/(kg·K)].  Default to ``constants.rho_ocean`` /
        ``constants.c_sw``.

    Returns
    -------
    :class:`SectionTransport`
    """
    if rho_0 is None:
        rho_0 = float(constants.rho_ocean)
    if c_p is None:
        c_p = float(constants.c_sw)

    v_np = np.asarray(v_face, dtype=np.float64)
    h_np = np.asarray(h_partial, dtype=np.float64)
    T_np = np.asarray(T, dtype=np.float64)
    S_np = np.asarray(S, dtype=np.float64)

    n_lat_v, n_lon, nlev = v_np.shape
    R = getattr(grid, "radius", constants.R_earth)
    dlon = getattr(grid, "dlon", 2.0 * np.pi / n_lon)

    # v-face latitudes [degrees].
    lat_v = getattr(grid, "lat_v", None)
    if lat_v is None:
        lat_c = np.asarray(getattr(grid, "lat", None))
        dlat = getattr(grid, "dlat", None)
        if lat_c is not None and dlat is not None:
            lat_v_rad = np.concatenate([
                [lat_c[0] - 0.5 * float(dlat)],
                lat_c + 0.5 * float(dlat),
            ])
        else:
            lat_v_rad = np.linspace(-np.pi / 2, np.pi / 2, n_lat_v)
    else:
        lat_v_rad = np.asarray(lat_v)
    lat_v_deg = np.degrees(lat_v_rad)

    # Closest v-face row.
    j = int(np.argmin(np.abs(lat_v_deg - target_lat_deg)))
    if abs(float(lat_v_deg[j]) - target_lat_deg) > lat_tol_deg:
        return SectionTransport(float("nan"), float("nan"), float("nan"), float("nan"))

    # Longitude mask for the section band.
    lon_c = np.asarray(getattr(grid, "lon", np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)))
    lon_deg = np.degrees(lon_c)
    lon_wrapped = ((lon_deg + 180.0) % 360.0) - 180.0
    if lon_min_deg <= lon_max_deg:
        lon_mask = (lon_wrapped >= lon_min_deg) & (lon_wrapped <= lon_max_deg)
    else:
        lon_mask = (lon_wrapped >= lon_min_deg) | (lon_wrapped <= lon_max_deg)

    # Edge length dx at this latitude row.
    cos_lat = float(np.cos(lat_v_rad[j]))
    dx_v = R * dlon * cos_lat                                   # scalar

    # Centred edge thickness from the two adjacent cells (north +
    # south of v-face).  Pole rows (j=0 or j=n_lat_v-1) get one-sided.
    if 0 < j < n_lat_v - 1:
        h_e = 0.5 * (h_np[j - 1] + h_np[j])                     # (n_lon, nlev)
        T_e = 0.5 * (T_np[j - 1] + T_np[j])
        S_e = 0.5 * (S_np[j - 1] + S_np[j])
    elif j == 0:
        h_e = h_np[0]
        T_e = T_np[0]
        S_e = S_np[0]
    else:
        h_e = h_np[-1]
        T_e = T_np[-1]
        S_e = S_np[-1]

    # Per-column volume flux per level [m³/s].
    F_vol = v_np[j] * h_e * dx_v                                # (n_lon, nlev)
    F_vol_band = F_vol * lon_mask[:, None]

    # Section totals.
    volume_m3s = float(np.sum(F_vol_band))
    heat_W = float(rho_0 * c_p * np.sum(T_e * F_vol_band))
    salt_kg_s = float(rho_0 * 1.0e-3 * np.sum(S_e * F_vol_band))
    # Salt unit: ``rho_0 * S [PSU = g/kg = 1e-3 kg/kg] * F_vol [m³/s]``
    # = kg/m³ * 1e-3 * m³/s = kg(salt)/s.

    # Overturning streamfunction within the band: cumulative sum of
    # depth-integrated zonal-mean transport.  Matches the lat-lon
    # ``moc_streamfunction`` convention ``ψ = -cumsum``.
    F_zonal = F_vol_band.sum(axis=0)                            # (nlev,)
    psi_m3s = -np.cumsum(F_zonal)
    overturning_Sv = float(-np.nanmin(psi_m3s) / SV)

    return SectionTransport(
        volume_Sv=volume_m3s / SV,
        heat_PW=heat_W / PW,
        salt_kg_per_s=salt_kg_s,
        overturning_Sv=overturning_Sv,
    )


# ==============================================================================
# OSNAP presets
# ==============================================================================

# OSNAP East: nominal 58°N, 31°W → 7°W (Iceland Basin to Scotland
# shelf).  Lozier 2017 used a slightly curved transect; we
# approximate with a zonal section since the model section width is
# much larger than the curvature offset.
OSNAP_EAST_LAT_DEG = 58.0
OSNAP_EAST_LON_MIN_DEG = -31.0
OSNAP_EAST_LON_MAX_DEG = -7.0

# OSNAP West: nominal 53°N, 53°W → 44°W (Labrador Sea).
OSNAP_WEST_LAT_DEG = 53.0
OSNAP_WEST_LON_MIN_DEG = -53.0
OSNAP_WEST_LON_MAX_DEG = -44.0


def osnap_east_transport(
    v_face: np.ndarray,
    h_partial: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    grid,
    *,
    rho_0: Optional[float] = None,
    c_p: Optional[float] = None,
) -> SectionTransport:
    """OSNAP East transport preset (58°N, 31°W → 7°W)."""
    return meridional_section_transport(
        v_face, h_partial, T, S, grid,
        target_lat_deg=OSNAP_EAST_LAT_DEG,
        lon_min_deg=OSNAP_EAST_LON_MIN_DEG,
        lon_max_deg=OSNAP_EAST_LON_MAX_DEG,
        rho_0=rho_0, c_p=c_p,
    )


def osnap_west_transport(
    v_face: np.ndarray,
    h_partial: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    grid,
    *,
    rho_0: Optional[float] = None,
    c_p: Optional[float] = None,
) -> SectionTransport:
    """OSNAP West transport preset (53°N, 53°W → 44°W)."""
    return meridional_section_transport(
        v_face, h_partial, T, S, grid,
        target_lat_deg=OSNAP_WEST_LAT_DEG,
        lon_min_deg=OSNAP_WEST_LON_MIN_DEG,
        lon_max_deg=OSNAP_WEST_LON_MAX_DEG,
        rho_0=rho_0, c_p=c_p,
    )
