"""Fourier polar filter for lat-lon grid CFL stability.

The polar CFL problem: dx = R*dlon*cos(lat) -> 0 at poles, making
explicit time-stepping unstable unless high-wavenumber modes are damped.

The filter truncates Fourier modes in longitude at each latitude,
keeping only wavenumbers that satisfy the CFL condition for the fastest
wave (external gravity wave) at the given time step.

The maximum stable wavenumber at each latitude is:
    k_max = CFL_limit * R * cos(lat) / (c_max * dt)

where c_max is the maximum wave speed (external gravity wave) and
CFL_limit is the RK3 stability limit (sqrt(3) ~ 1.73).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid

# ---------------------------------------------------------------------------
# Distributed longitude context (2-D lat-lon MPI, proc_lon>1)
# ---------------------------------------------------------------------------
# The polar filter is a per-latitude-row longitude rFFT, so it needs the WHOLE
# longitude circle on one rank.  Under a longitude split each rank owns only a
# lon SLICE, so the filter must first gather the full lon axis, rFFT/mask/irFFT
# on it, then scatter its block back.  The 2-D MPI step
# (``make_latlon_2d_mpi_step``) injects an AD-safe gather/scatter pair here
# (bound to its ``LatLon2DLayout`` + longitude row sub-communicator) BEFORE it
# builds the jitted step; the filter reads this process-global at trace time,
# exactly like the halo backend (``grids.halo.get_halo_backend``).  This keeps
# ``polar_filter`` free of a ``parallel.latlon_mpi`` import (no cycle) and the
# dycore call sites (``fourier_filter(field, grid, mask)``) unchanged.  ``None``
# (serial / lat-band proc_lon==1 / SPMD) → the local rFFT path below, unchanged.
_LON_GATHER_CTX = None   # (gather_full: fn(field)->full, scatter_block: fn(full)->block)


def set_polar_filter_lon_gather(gather_full, scatter_block) -> None:
    """Install the longitude gather/scatter pair used by :func:`fourier_filter`
    under a 2-D (proc_lon>1) lat-lon MPI decomposition.  Both must be AD-safe
    (the filter runs inside the differentiated tendency).  Pass ``None, None``
    to clear (restore the serial local-rFFT path)."""
    global _LON_GATHER_CTX
    _LON_GATHER_CTX = (
        None if gather_full is None else (gather_full, scatter_block))


def get_polar_filter_lon_gather():
    """Return the active ``(gather_full, scatter_block)`` pair or ``None``."""
    return _LON_GATHER_CTX


def compute_polar_filter_mask(
    grid: LatLonGrid,
    dt: float = 600.0,
    max_wave_speed: float = 300.0,
    cutoff_lat_deg: float = 60.0,
    safety_factor: float = 0.85,
    *,
    is_v_face: bool = False,
    n_lon_override: int | None = None,
) -> jnp.ndarray:
    """Precompute the Fourier polar filter mask.

    Returns a 2D mask (n_lat, n_freq) for cell-centered fields or
    (n_lat+1, n_freq) for v-face fields, where n_freq = n_lon//2 + 1,
    suitable for multiplying the output of jnp.fft.rfft.

    Parameters
    ----------
    grid : LatLonGrid
        The lat-lon grid.
    dt : float
        Time step [seconds].
    max_wave_speed : float
        Maximum wave speed [m/s] (external gravity wave speed).
    cutoff_lat_deg : float
        Latitude (degrees) beyond which filtering is applied.
    safety_factor : float
        Fraction of the theoretical CFL limit to use (< 1 for margin).
    is_v_face : bool, optional
        When True, build a mask for v-face fields (lat-interface
        values, shape ``(n_lat+1, ...)``) using ``grid.cos_lat_v`` /
        the half-cell-offset lat-interface coordinates.  Codex review
        Stage 3-E round 2 caught that applying the cell-centered mask
        to v-face indices introduces a half-cell lat offset that
        admits ``k`` modes the actual v-face CFL forbids.
    n_lon_override : int, optional
        Use this longitude count for the wavenumber axis (``n_freq`` and
        the ``k`` clip) instead of ``grid.n_lon``.  Under a 2-D lat-lon
        MPI split the rank-local grid carries the LOCAL ``n_lon``, but the
        polar filter rFFTs the GATHERED full longitude circle, so the mask
        must span the GLOBAL ``n_lon``.  The latitude terms stay rank-local
        (``grid.lat`` / ``grid.cos_lat`` are the owned rows).  ``None`` →
        ``grid.n_lon`` (serial / full-lon-per-rank).

    Returns
    -------
    jax.Array : Float mask, shape (n_lat, n_freq) if ``is_v_face``
                is False, else (n_lat+1, n_freq).
    """
    n_lon = int(n_lon_override) if n_lon_override is not None else grid.n_lon
    # RK3 stability limit for centered differences: sqrt(3)
    cfl_limit = jnp.sqrt(3.0) * safety_factor

    if is_v_face:
        # v-face: read the precomputed ``grid.lat_v`` and
        # ``grid.cos_lat_v`` arrays directly.  These were placed at
        # ``LatLonGrid``-construction time using the cell-axis ``lat``
        # available at the call site — so under MPI band decomposition
        # the rank-local grid carries the rank-local v-face arrays
        # (length ``n_lat_local+1``).  A previous implementation
        # reconstructed the v-face axis with ``±π/2`` padding which
        # silently broke interior ranks; Codex review Stage 3-E
        # round 3 caught that regression.
        cos_lat_face = grid.cos_lat_v
        lat_face = grid.lat_v
    else:
        cos_lat_face = grid.cos_lat
        lat_face = grid.lat

    # CFL-based max wavenumber at each latitude
    max_k_cfl = cfl_limit * grid.radius * cos_lat_face / (max_wave_speed * dt)
    max_k_cfl = jnp.clip(max_k_cfl, 1.0, float(n_lon // 2))

    # Only apply filtering poleward of cutoff
    cutoff_rad = jnp.deg2rad(cutoff_lat_deg)
    is_poleward = jnp.abs(lat_face) > cutoff_rad
    max_k = jnp.where(is_poleward, max_k_cfl, float(n_lon // 2))

    # Build mask
    n_freq = n_lon // 2 + 1
    wavenumbers = jnp.arange(n_freq)
    mask = (wavenumbers[None, :] <= max_k[:, None]).astype(jnp.float32)

    return mask


def fourier_filter(
    field: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Apply Fourier polar filter to a 2D field using a precomputed mask.

    Parameters
    ----------
    field : jax.Array
        2D field, shape (n_lat, n_lon).
    grid : LatLonGrid
        The lat-lon grid.
    mask : jax.Array
        Precomputed filter mask, shape (n_lat, n_freq).

    Returns
    -------
    jax.Array : Filtered field, shape (n_lat, n_lon).
    """
    ctx = _LON_GATHER_CTX
    if ctx is not None:
        # 2-D lat-lon MPI (proc_lon>1): gather the full longitude circle for
        # this rank's rows, rFFT/mask/irFFT on it (``mask`` is already
        # global-``n_freq``), then scatter this rank's lon block back.  Both
        # collectives are AD-safe (custom-VJP allgather / allreduce).
        gather_full, scatter_block = ctx
        full = gather_full(field)                   # (n_lat_local, n_lon_global)
        n_lon_g = full.shape[-1]
        hat = jnp.fft.rfft(full, axis=-1)
        filt = jnp.fft.irfft(hat * mask, n=n_lon_g, axis=-1)
        return scatter_block(filt)                  # (n_lat_local, n_lon_local)
    field_hat = jnp.fft.rfft(field, axis=-1)
    return jnp.fft.irfft(field_hat * mask, n=grid.n_lon, axis=-1)


def fourier_filter_3d(
    field_3d: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Apply Fourier polar filter to a 3D field using a precomputed mask.

    Parameters
    ----------
    field_3d : jax.Array
        3D field, shape (n_lat, n_lon, nlev).
    grid : LatLonGrid
        The lat-lon grid.
    mask : jax.Array
        Precomputed filter mask, shape (n_lat, n_freq).

    Returns
    -------
    jax.Array : Filtered field, shape (n_lat, n_lon, nlev).
    """
    ctx = _LON_GATHER_CTX
    if ctx is not None:
        # 2-D lat-lon MPI (proc_lon>1): gather full lon (axis 1), filter,
        # scatter this rank's block back.  lon_gather_full handles the
        # trailing ``nlev`` axis; ``mask`` is already global-``n_freq``.
        gather_full, scatter_block = ctx
        full = gather_full(field_3d)                # (n_lat_local, n_lon_g, nlev)
        n_lon_g = full.shape[1]
        hat = jnp.fft.rfft(full, axis=1)
        filt = jnp.fft.irfft(hat * mask[:, :, None], n=n_lon_g, axis=1)
        return scatter_block(filt)
    field_hat = jnp.fft.rfft(field_3d, axis=1)
    return jnp.fft.irfft(field_hat * mask[:, :, None], n=grid.n_lon, axis=1)
