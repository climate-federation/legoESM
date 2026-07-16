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

import hashlib
import math
import os
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from legoesm import constants


def _sh_chunk_size() -> int:
    """Return the trailing-axis chunk size for the 3D SH transforms.

    The 3D SH analysis/synthesis kernels in this module materialise a
    ``(n_lat, n_sh, n_batch)`` intermediate on the way to the FFT.  At
    T127 with ~30 levels and complex128 this is ~1.5 GB; at T255 with
    60 levels it is ~12 GB — enough to OOM on 16-40 GB GPUs.

    When ``LEGOESM_SH_CHUNK_SIZE`` is set to a positive integer N, the
    3D paths split the trailing batch axis into chunks of size N and
    process them with a static Python loop, capping the peak working
    set at ``(n_lat, n_sh, N) * dtype_bytes``.  N=0 (the default)
    keeps the original single-pass behaviour.

    The branch is purely Python-static at trace time, so the chunked
    path remains AD-compatible.
    """
    raw = os.environ.get("LEGOESM_SH_CHUNK_SIZE", "0")
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _maybe_chunk_trailing(
    fn, arr: jax.Array, chunk: int, axis: int = -1,
) -> jax.Array:
    """Apply ``fn`` to ``arr`` in chunks along ``axis``.

    ``fn`` must accept and return arrays with the same trailing axis
    layout.  When ``chunk == 0`` or ``arr.shape[axis] <= chunk``,
    forwards the whole array in a single call (no Python loop).
    Otherwise splits along ``axis`` and concatenates outputs.
    """
    n = arr.shape[axis]
    if chunk <= 0 or n <= chunk:
        return fn(arr)
    parts = []
    start = 0
    while start < n:
        end = min(start + chunk, n)
        sub = jax.lax.dynamic_slice_in_dim(arr, start, end - start, axis=axis)
        parts.append(fn(sub))
        start = end
    return jnp.concatenate(parts, axis=axis)


# =============================================================================
# Batched-GEMM Legendre step (scaling review 2026-06-13 lever #2; OPT-IN)
# =============================================================================
# The default 3-D analysis/synthesis Legendre step is a broadcast-multiply +
# ``jnp.sum`` / ``segment_sum`` over the gathered ``(n_lat, n_sh, nlev)``
# intermediate — memory-bound, never a tensor-core GEMM.  Reformulated here as
# a batched ``dot_general`` over the zonal wavenumber ``m`` (contract latitude,
# free (n, nlev)).  Mathematically identical (same products + reduction, only
# the summation grouping changes) → parity to fp round-off; AD-safe.  OPT-IN
# via ``LEGOESM_SH_GEMM`` because the win needs fp64 tensor cores (Ampere+);
# on Turing/CPU the default reduce is competitive (design:
# docs/performance/scaling/sh_gemm_design_2026-06-13.md).


def _sh_gemm_enabled() -> bool:
    """True when the opt-in batched-GEMM Legendre path is requested.

    Read at trace time (static Python bool) so the branch never doubles the
    trace and stays AD-compatible — same discipline as
    :func:`_sh_chunk_size`.  Case-insensitive so ``False``/``FALSE``/``No``
    do NOT accidentally enable the opt-in path (codex 2026-06-13)."""
    return os.environ.get("LEGOESM_SH_GEMM", "0").strip().lower() not in (
        "0", "", "false", "no", "off")


def _sh_gemm_bf16_enabled() -> bool:
    """True when the SH-GEMM Legendre contraction should run in bfloat16.

    Sub-mode of ``LEGOESM_SH_GEMM`` (no effect unless the GEMM path is also
    enabled): casts the Legendre matmul operands to bf16 with fp32 accumulate
    — the MXU/tensor-core reduced-precision throughput path on Ampere+/TPU.
    Read at trace time (static Python bool), same discipline as
    :func:`_sh_gemm_enabled`.

    FORWARD / INFERENCE-ONLY: bf16 yields ~3-decimal-digit gradients, so this
    must NOT be enabled on the differentiable-training path. The FFT stays
    complex128 and the semi-implicit solve is untouched — only the Legendre
    contraction is reduced-precision."""
    return os.environ.get("LEGOESM_SH_GEMM_BF16", "0").strip().lower() not in (
        "0", "", "false", "no", "off")


def _legendre_contract(
    spec: str, mat_real: jax.Array, field: jax.Array, compute_bf16: bool,
) -> jax.Array:
    """``jnp.einsum(spec, mat_real, field)`` with an optional bf16 compute path.

    ``mat_real`` is a REAL Legendre matrix; ``field`` may be real or complex.
    When ``compute_bf16`` is False this is byte-for-byte the default einsum (so
    the non-bf16 GEMM path stays bit-identical to the legacy reduction). When
    True, operands are cast to bfloat16 and accumulated in float32 (the
    ``preferred_element_type``); a complex ``field`` is contracted as two real
    matmuls (W·(a+ib) = W·a + i·W·b, the Legendre matrix being real) so bf16
    never has to represent a complex value. The result is upcast back to the
    field's dtype."""
    if not compute_bf16:
        return jnp.einsum(spec, mat_real, field)
    m16 = mat_real.astype(jnp.bfloat16)
    if jnp.iscomplexobj(field):
        re = jnp.einsum(spec, m16, field.real.astype(jnp.bfloat16),
                        preferred_element_type=jnp.float32)
        im = jnp.einsum(spec, m16, field.imag.astype(jnp.bfloat16),
                        preferred_element_type=jnp.float32)
        return (re + 1j * im).astype(field.dtype)
    return jnp.einsum(spec, m16, field.astype(jnp.bfloat16),
                      preferred_element_type=jnp.float32).astype(field.dtype)


def _flat_to_bym(
    mat: jax.Array, ms: jax.Array, ls: jax.Array, n_max: int,
) -> jax.Array:
    """Scatter a flat ``(n_lat, n_sh)`` SH matrix to the dense ``(m, n, lat)``
    block layout ``(n_max+1, n_max+1, n_lat)`` (zero where ``n < m``).

    ``out[ms[k], ls[k], :] = mat[:, k]`` so for triangular truncation
    ``out[m, n, lat] == mat[lat, _sh_idx(n, m)]``.  Built on demand from the
    grid's existing Legendre matrices (no new pytree fields)."""
    n_lat = mat.shape[0]
    out = jnp.zeros((n_max + 1, n_max + 1, n_lat), dtype=mat.dtype)
    return out.at[ms, ls, :].set(mat.T)


def _analysis_legendre_gemm(
    W: jax.Array, f_m: jax.Array, ms: jax.Array, ls: jax.Array, n_max: int,
) -> jax.Array:
    """Batched-GEMM Legendre analysis contraction.

    ``W`` is a weighted Legendre matrix ``(n_lat, n_sh)`` (wPnm / wPnm_oc2 /
    wDnm); ``f_m`` is the per-latitude Fourier field ``(n_lat, n_max+1, nlev)``
    (m-indexed, NO gather).  Returns ``coeffs`` ``(n_sh, nlev)`` =
    ``Σ_lat W[lat,k]·f_m[lat, m(k)]`` — identical to
    ``Σ_lat W[:,k,None]·f_m[:,m(k)]`` but via one batched ``dot_general`` over
    m.  Caller applies the ``2π`` prefactor (matching the legacy kernels)."""
    W_bym = _flat_to_bym(W, ms, ls, n_max)            # (m, n, lat)
    coeffs_bym = _legendre_contract(
        "mnl,lmv->mnv", W_bym, f_m, _sh_gemm_bf16_enabled())  # (m, n, nlev)
    return coeffs_bym[ms, ls, :]                       # (n_sh, nlev)


def _synthesis_legendre_gemm(
    P: jax.Array, coeffs: jax.Array, ms: jax.Array, ls: jax.Array,
    n_max: int,
) -> jax.Array:
    """Batched-GEMM Legendre synthesis contraction (inverse of
    :func:`_analysis_legendre_gemm`).

    ``P`` is a Legendre matrix ``(n_lat, n_sh)`` (Pnm / Hnm); ``coeffs`` is
    ``(n_sh, nlev)``.  Returns ``f_m`` ``(n_lat, n_max+1, nlev)`` =
    ``Σ_{n>=m} P[lat, k(n,m)]·coeffs[k(n,m)]`` grouped by m — identical to the
    legacy ``segment_sum`` by ``ms`` but via one batched ``dot_general``."""
    nlev = coeffs.shape[-1]
    P_bym = _flat_to_bym(P, ms, ls, n_max)            # (m, n, lat)
    coeffs_bym = jnp.zeros(
        (n_max + 1, n_max + 1, nlev), dtype=coeffs.dtype,
    ).at[ms, ls, :].set(coeffs)                        # (m, n, nlev)
    return _legendre_contract(
        "mnl,mnv->lmv", P_bym, coeffs_bym, _sh_gemm_bf16_enabled())  # (n_lat,n_max+1,nlev)


# =============================================================================
# Grid definition
# =============================================================================

class GaussianGrid(NamedTuple):
    """Gaussian grid with precomputed spectral transform matrices.

    All arrays are JAX arrays (float64 or complex128).

    Memory Scaling
    ~~~~~~~~~~~~~~
    The grid stores full Legendre polynomial matrices in memory. For a spectral
    truncation with n_sh spectral coefficients and n_lat latitude points:

    - Each Legendre matrix (Pnm, Hnm, Pnm_oc2, Dnm) takes ~8 * n_lat * n_sh bytes
      (float64). Plus their weighted versions (wPnm, wPnm_oc2, wDnm), which
      are also ~8 * n_lat * n_sh bytes each.

    - Total: ~8 * n_lat * n_sh * 8 = ~64 * n_lat * n_sh bytes per grid.

    For standard truncations:

    - **T170** (n_max=170): n_sh ≈ 14,706, typical n_lat ≈ 512 (quadratic dealiasing)
      → ~480 MB per grid instance.

    - **T340** (n_max=340): n_sh ≈ 58,366, typical n_lat ≈ 1024
      → ~4.8 GB per grid instance.

    On multi-process runs (e.g., 8-GPU system with MPI), each process loads its own
    grid copy, so total memory usage scales roughly as (# processes) × (# grids) ×
    memory-per-grid. Use grid caching / singleton patterns in large distributed runs.
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
    wPnm: jax.Array         # weights * Pnm (precomputed for SH analysis), (n_lat, n_sh)
    wPnm_oc2: jax.Array     # weights * Pnm_oc2 (precomputed), (n_lat, n_sh)
    wDnm: jax.Array         # weights * Dnm (precomputed), (n_lat, n_sh)
    n_sh: int               # Number of spectral coefficients
    ls: jax.Array           # Total wavenumber n for each SH index, (n_sh,)
    ms: jax.Array           # Zonal wavenumber m for each SH index, (n_sh,)
    lap: jax.Array          # Spectral Laplacian = -n(n+1)/a^2, (n_sh,)
    ilap: jax.Array         # Inverse Laplacian (0 for n=0), (n_sh,)
    subgrid_topo_stddev: object = None  # jax.Array (n_lat, n_lon) [m] | None — oro-GWD launch h_topo

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

    @property
    def grid_shape_2d(self) -> tuple[int, ...]:
        return (self.n_lat, self.n_lon)

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
    omega: float = constants.Omega,
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
    omega : float
        Planetary rotation rate [rad/s] used to populate ``f =
        2·omega·sin(lat)``.  Default ``constants.Omega`` (Earth).  Set
        to ``0.0`` for the canonical DCMIP 2008 §3-1 non-rotating
        gravity-wave test, or to ``constants.Omega · X`` for the
        Wedi-Smolarkiewicz 2009 small-planet framework.
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

    Warnings
    --------
    Memory usage scales quadratically with truncation level. At high truncations
    (n_max ≥ 340), expect ~4+ GB of memory per grid instance (see GaussianGrid
    "Memory Scaling" section for details). In distributed MPI runs with many
    processes, consider caching grid instances or implementing lazy loading
    strategies if memory becomes a bottleneck.
    """
    # Hard guard: spectral transforms require float64/complex128
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "Spectral/Gaussian grids require JAX_ENABLE_X64=True. "
            "Set the environment variable JAX_ENABLE_X64=1 or call "
            "jax.config.update('jax_enable_x64', True) before importing."
        )
    # Precision policy guard: spectral paths are float64-only.
    try:
        from legoesm.core.precision import get_policy
        policy = get_policy()
        if policy.compute == jnp.float32 and policy.storage == jnp.float32:
            import warnings
            warnings.warn(
                "Spectral/Gaussian grid created under fp32 precision policy. "
                "Spectral transforms require float64; grid arrays will be "
                "float64 regardless of the global policy.",
                stacklevel=2,
            )
    except Exception:
        pass

    # Extract allow_unsupported from global config if provided
    if legoesm_config is not None:
        allow_unsupported_backend = bool(
            legoesm_config.get("atmosphere.spectral.allow_unsupported", False)
        )

    from legoesm.runtime.backend import check_spectral_backend, get_backend
    backend = get_backend()

    # On the Apple GPU (mps) backend we can still run spectral dynamics by
    # hosting grid/transforms on CPU and routing the spectral model there.
    # Keep strict x64 requirement.
    if backend == "mps":
        if not jax.config.jax_enable_x64:
            raise ValueError(
                "Gaussian spectral grid on Apple GPU (mps) requires "
                "JAX_ENABLE_X64=True for CPU spectral fallback."
            )
        if not allow_unsupported_backend:
            warnings.warn(
                "Apple GPU (mps) backend detected. Creating Gaussian spectral "
                "grid on CPU for spectral fallback.",
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
    f_np = 2.0 * float(omega) * sin_lat_np[:, None] * np.ones((1, n_lon))

    target_device = jax.devices("cpu")[0] if backend == "mps" else None

    def _to_jax(array, dtype):
        np_arr = np.asarray(array, dtype=dtype)
        if target_device is None:
            return jnp.array(np_arr, dtype=dtype)
        return jax.device_put(np_arr, target_device)

    # Precompute weighted Legendre matrices (avoid recomputing every SH analysis)
    w_col = w_gauss[:, None]  # (n_lat, 1)
    wPnm_np = Pnm_np * w_col
    wPnm_oc2_np = Pnm_oc2_np * w_col
    wDnm_np = Dnm_np * w_col

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
        wPnm=_to_jax(wPnm_np, np.float64),
        wPnm_oc2=_to_jax(wPnm_oc2_np, np.float64),
        wDnm=_to_jax(wDnm_np, np.float64),
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
    n_lon = grid.n_lon
    n_max = grid.n_max

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

    # Gather the right Fourier mode for each spectral index
    ms = grid.ms  # (n_sh,) int
    f_m_gathered = f_m[:, ms]  # (n_lat, n_sh)

    # Contract over latitude using precomputed weighted Legendre matrix
    coeffs = 2.0 * jnp.pi * jnp.sum(grid.wPnm * f_m_gathered, axis=0)  # (n_sh,)

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

    Notes
    -----
    Uses ``jax.ops.segment_sum`` to group spectral contributions by
    zonal wavenumber *m* without materializing the full ``(n_lat, n_sh)``
    intermediate.  Peak memory is ``O(n_lat × n_sh)`` for the
    element-wise product but the segment_sum reduces it immediately,
    and XLA's buffer reuse typically avoids the peak allocation.
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    n_max = grid.n_max
    ms = grid.ms  # (n_sh,)

    # Element-wise Legendre * coefficients, then group-sum by m.
    # contributions shape: (n_lat, n_sh) — XLA may fuse with segment_sum.
    contributions = grid.Pnm * coeffs[None, :]  # (n_lat, n_sh)

    # segment_sum groups along the spectral axis by wavenumber m.
    # Result shape: (n_lat, n_max + 1).
    f_m = jax.ops.segment_sum(
        contributions.T,  # (n_sh, n_lat)
        ms,
        num_segments=n_max + 1,
    ).T  # (n_lat, n_max + 1)

    # Inverse FFT in longitude.
    f_hat_full = jnp.zeros((n_lat, n_lon // 2 + 1), dtype=jnp.complex128)
    f_hat_full = f_hat_full.at[:, :n_max + 1].set(f_m)

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

    f_m_gathered = f_m[:, grid.ms]

    coeffs = 2.0 * jnp.pi * jnp.sum(grid.wPnm_oc2 * f_m_gathered, axis=0)
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

    f_m_gathered = f_m[:, grid.ms]

    coeffs = 2.0 * jnp.pi * jnp.sum(grid.wDnm * f_m_gathered, axis=0)
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

    # d/d(theta) and d/d(lon) components — batch (psi, chi) along a
    # trailing axis so the SH synthesis runs once on (n_sh, 2) for
    # each of the two synthesis variants.  Uses the 3D-native
    # synthesis (``sh_synthesis_3d`` / ``sh_synthesis_H_3d``), which
    # treats any trailing axis (level *or* tracer/component) as a
    # passive batch — for the SW 2D path the ``2`` plays the role of
    # ``nlev=2``.  4 SH syntheses → 2.
    pc_hat = jnp.stack([psi_hat, chi_hat], axis=-1)  # (n_sh, 2)
    pc_dtheta = sh_synthesis_H_3d(grid, pc_hat) / a
    dpsi_dtheta = pc_dtheta[..., 0]
    dchi_dtheta = pc_dtheta[..., 1]

    pc_dlon = sh_synthesis_3d(grid, (1j * grid.ms)[:, None] * pc_hat) / a
    dpsi_dlon = pc_dlon[..., 0]
    dchi_dlon = pc_dlon[..., 1]

    # u*cos_lat = d(psi)/d(theta) + d(chi)/d(lon)
    u_cos = dpsi_dtheta + dchi_dlon

    # v*cos_lat = d(psi)/d(lon) - d(chi)/d(theta)
    v_cos = dpsi_dlon - dchi_dtheta

    return u_cos, v_cos


def sh_synthesis_H(grid: GaussianGrid, coeffs: jax.Array) -> jax.Array:
    """Inverse SH transform using derivative Legendre Hnm (instead of Pnm).

    Produces the theta-derivative of the field on the grid.

    Mirrors the ``sh_synthesis`` rewrite that replaced a scatter-add
    (``zeros + at[:, ms].add``) with ``jax.ops.segment_sum``.  On GPU
    the scatter-add falls back to atomic ops and is 5–20× slower than
    a segment sum; this path is hit every spectral PE step via
    ``uv_from_vordiv``.
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    n_max = grid.n_max
    ms = grid.ms

    contributions = grid.Hnm * coeffs[None, :]
    f_m = jax.ops.segment_sum(
        contributions.T,
        ms,
        num_segments=n_max + 1,
    ).T  # (n_lat, n_max + 1)

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


def dealiasing_mask(grid: GaussianGrid, fraction: float = 0.667) -> jax.Array:
    """Orszag 2/3-rule de-aliasing mask for spectral nonlinear products.

    Returns a real ``(n_sh,)`` multiplier that is ``1.0`` for every
    spherical-harmonic coefficient with total wavenumber
    ``n <= floor(fraction * n_max)`` and ``0.0`` above it.  Multiplying a
    spectral *tendency* (or a transformed nonlinear product) by this mask
    discards the upper ``1 - fraction`` band of wavenumbers, which is
    where the quadratic/cubic products fold spurious aliased power back
    into the resolved spectrum on a triangular-truncation Gaussian grid.

    The default ``fraction = 2/3`` is the Orszag (1971) rule: a quadratic
    nonlinearity ``a*b`` of two fields truncated at ``n_max`` produces
    content up to ``2*n_max``; retaining only ``n <= (2/3)*n_max`` of each
    factor guarantees the aliased part (``n > n_max`` folded down) lands
    above the retained band and is removed.

    This is the single canonical de-aliasing mechanism shared by every
    spectral dycore (``spectral_pe``, ``spectral_nh``, ``spectral_sw``);
    do not re-derive the ``ls <= n_cut`` truncation inline.

    Parameters
    ----------
    grid : GaussianGrid
        Carries ``ls`` (total wavenumber ``n`` per SH index) and the
        triangular truncation ``n_max``.
    fraction : float
        Retained fraction of the spectrum.  ``2/3`` (default) is the
        standard Orszag rule for quadratic nonlinearities.  Values
        ``<= 0`` return an all-ones mask (de-aliasing disabled), so a
        caller can gate the feature on a single config float without a
        Python ``if`` around the multiply.

    Returns
    -------
    (n_sh,) float64 array of 1.0 / 0.0.

    References
    ----------
    - Orszag, S. A. (1971): On the elimination of aliasing in
      finite-difference schemes by filtering high-wavenumber components.
      J. Atmos. Sci., 28, 1074.
    """
    if fraction <= 0.0:
        return jnp.ones((grid.n_sh,), dtype=jnp.float64)
    n_cut = int(fraction * grid.n_max)
    return jnp.where(grid.ls <= n_cut, 1.0, 0.0).astype(jnp.float64)


# =============================================================================
# 3D (level-wise) transform wrappers via vmap
# =============================================================================

def sh_analysis_3d(grid: GaussianGrid, field_3d: jax.Array) -> jax.Array:
    """Forward SH transform per vertical level (3D-native).

    Same numeric algorithm as :func:`sh_analysis` but with the
    longitude FFT, the Legendre weight, and the latitude sum all
    batched along the trailing level axis — no per-level moveaxis +
    vmap.

    Memory: when ``LEGOESM_SH_CHUNK_SIZE > 0``, the trailing level
    axis is processed in chunks to cap the ``(n_lat, n_sh, n_batch)``
    intermediate at ``n_batch = chunk_size`` per inner call (iter 4).

    Parameters
    ----------
    field_3d : (n_lat, n_lon, nlev) real array.

    Returns
    -------
    (n_sh, nlev) complex array.
    """
    # Value (unweighted Pnm) analysis is exactly the ``weight = wPnm`` case
    # of the shared weighted kernel, which carries both the legacy latitude
    # sum and the opt-in ``LEGOESM_SH_GEMM`` ``dot_general`` path.  One
    # analysis core for the value / oc2 / dmu variants (no duplicated
    # Legendre numerics); trailing-axis chunking handled inside.
    return _sh_analysis_weighted_3d(grid, field_3d, grid.wPnm)


def _synthesis_3d_with_matrix(
    grid: GaussianGrid, coeffs_3d: jax.Array, P_matrix: jax.Array,
) -> jax.Array:
    """Shared batched inverse SH synthesis for a given Legendre matrix.

    ``P_matrix`` selects the transform: ``grid.Pnm`` → value synthesis
    (:func:`sh_synthesis_3d`); ``grid.Hnm`` → θ-derivative synthesis
    (:func:`sh_synthesis_H_3d`).  Both the legacy ``segment_sum`` path and
    the opt-in ``LEGOESM_SH_GEMM`` ``dot_general`` path
    (:func:`_synthesis_legendre_gemm`) live here ONCE, so the two public
    variants share a single body — no duplicated Legendre numerics.
    Trailing-axis chunking via ``LEGOESM_SH_CHUNK_SIZE`` (iter 4).
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    n_max = grid.n_max
    ms = grid.ms  # (n_sh,)

    def _kernel(coeffs):
        if _sh_gemm_enabled():
            f_m = _synthesis_legendre_gemm(
                P_matrix, coeffs, grid.ms, grid.ls, n_max)  # (n_lat,n_max+1,nlev)
        else:
            # contributions: (n_lat, n_sh, nlev_chunk) — P_matrix broadcasts.
            contributions = P_matrix[..., None] * coeffs[None, :, :]
            # segment_sum operates on the leading axis; permute
            # (n_sh, n_lat, n_batch), group, then permute back to
            # (n_lat, n_max+1, n_batch).
            f_m = jnp.swapaxes(
                jax.ops.segment_sum(
                    jnp.swapaxes(contributions, 0, 1),
                    ms,
                    num_segments=n_max + 1,
                ),
                0, 1,
            )
        nlev_chunk = coeffs.shape[-1]
        f_hat_full = jnp.zeros(
            (n_lat, n_lon // 2 + 1, nlev_chunk), dtype=jnp.complex128,
        )
        f_hat_full = f_hat_full.at[:, :n_max + 1, :].set(f_m)
        return jnp.fft.irfft(f_hat_full * n_lon, n=n_lon, axis=1).real

    return _maybe_chunk_trailing(_kernel, coeffs_3d, _sh_chunk_size(), axis=-1)


def sh_synthesis_3d(grid: GaussianGrid, coeffs_3d: jax.Array) -> jax.Array:
    """Inverse SH transform per vertical level (3D-native).

    Same numeric algorithm as :func:`sh_synthesis` but with the
    ``n_sh`` -> ``m`` segment-sum done in a single batched call over
    ``(n_sh, n_lat, nlev)`` instead of vmap'ing the 2D path per level.
    Eliminates the per-level moveaxis + vmap dance and lets XLA fuse
    the FFT across the level axis.

    Memory: when ``LEGOESM_SH_CHUNK_SIZE > 0`` the trailing level axis
    is processed in chunks (see :func:`_maybe_chunk_trailing`, iter 4).

    Parameters
    ----------
    coeffs_3d : (n_sh, nlev) complex array.

    Returns
    -------
    (n_lat, n_lon, nlev) real array.
    """
    return _synthesis_3d_with_matrix(grid, coeffs_3d, grid.Pnm)


def sh_synthesis_H_3d(grid: GaussianGrid, coeffs_3d: jax.Array) -> jax.Array:
    """3D-native counterpart of :func:`sh_synthesis_H`.

    Returns the θ-derivative of the inverse SH transform at every
    vertical level in a single batched ``segment_sum`` + IRFFT, instead
    of vmap'ing the 2D path per level.

    Memory: trailing-axis chunking via ``LEGOESM_SH_CHUNK_SIZE`` —
    see :func:`_maybe_chunk_trailing` (iter 4).

    Routes through the shared :func:`_synthesis_3d_with_matrix` core with
    ``grid.Hnm`` (the θ-derivative Legendre matrix), so it inherits the
    opt-in ``LEGOESM_SH_GEMM`` ``dot_general`` path identically to
    :func:`sh_synthesis_3d`.
    """
    return _synthesis_3d_with_matrix(grid, coeffs_3d, grid.Hnm)


def _sh_analysis_weighted_3d(
    grid: GaussianGrid, field_3d: jax.Array, weight_matrix: jax.Array,
) -> jax.Array:
    """Batched forward SH analysis weighted by ``weight_matrix`` (n_lat, n_sh).

    rfft over longitude → slice m≤n_max → ``2π·Σ_lat (weight·f_m)``,
    trailing-axis chunked.  THE shared analysis core for every scalar
    forward variant — value (``wPnm``, via :func:`sh_analysis_3d`), ``oc2``
    (``wPnm_oc2``, 1/cos²) and ``dmu`` (``wDnm``, dPnm/dμ) — carrying BOTH
    the legacy latitude sum and the opt-in ``LEGOESM_SH_GEMM``
    ``dot_general`` path (:func:`_analysis_legendre_gemm`).  The only
    difference between the variants is ``weight_matrix``; the Legendre
    numerics live in one place.
    """
    n_max = grid.n_max

    def _kernel(field):
        f_hat_lon = jnp.fft.rfft(field, axis=1) / grid.n_lon
        f_m = f_hat_lon[:, :n_max + 1, :]               # (n_lat, n_max+1, n_batch)
        if _sh_gemm_enabled():
            # m-indexed GEMM contraction — no gather (matches the legacy
            # output to ~1e-12 in f64; see tests/unit/test_sh_gemm.py).
            return 2.0 * jnp.pi * _analysis_legendre_gemm(
                weight_matrix, f_m, grid.ms, grid.ls, n_max)
        f_m_gathered = f_m[:, grid.ms, :]
        return 2.0 * jnp.pi * jnp.sum(
            weight_matrix[:, :, None] * f_m_gathered, axis=0,
        )

    return _maybe_chunk_trailing(_kernel, field_3d, _sh_chunk_size(), axis=-1)


def sh_analysis_oc2_3d(
    grid: GaussianGrid, field_3d: jax.Array,
) -> jax.Array:
    """Forward SH transform with 1/cos² weighting, 3D-native.

    Same numeric algorithm as :func:`sh_analysis_oc2` but with the
    longitude FFT, Legendre weighting, and latitude sum batched along
    the trailing level axis.  Trailing-axis chunking via
    ``LEGOESM_SH_CHUNK_SIZE`` (iter 4).
    """
    return _sh_analysis_weighted_3d(grid, field_3d, grid.wPnm_oc2)


def sh_analysis_dmu_3d(
    grid: GaussianGrid, field_3d: jax.Array,
) -> jax.Array:
    """Forward SH transform with dPnm/dμ weighting, 3D-native.

    Same numeric algorithm as :func:`sh_analysis_dmu` but batched along
    the trailing level axis.  Trailing-axis chunking via
    ``LEGOESM_SH_CHUNK_SIZE`` (iter 4).
    """
    return _sh_analysis_weighted_3d(grid, field_3d, grid.wDnm)


def sh_analysis_oc2_dmu_3d(
    grid: GaussianGrid, field_3d: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Combined oc2 + dmu forward SH transforms, sharing the FFT + gather.

    The standalone ``sh_analysis_oc2_3d`` and ``sh_analysis_dmu_3d``
    each compute ``rfft(field) → slice → gather`` before applying their
    Legendre weight matrix (``wPnm_oc2`` vs ``wDnm``).  When both are
    called on the SAME input (3 sites in spectral_pe_tendencies), the
    FFT + gather is duplicated.

    This combined entry point runs the FFT + gather once and applies
    both Legendre weight matrices.  Returns ``(oc2_result, dmu_result)``
    matching the standalone outputs.

    Parameters
    ----------
    grid : GaussianGrid
    field_3d : jax.Array, shape (n_lat, n_lon, ...)

    Returns
    -------
    oc2 : jax.Array — same shape and value as ``sh_analysis_oc2_3d(grid, field_3d)``
    dmu : jax.Array — same shape and value as ``sh_analysis_dmu_3d(grid, field_3d)``
    """
    n_max = grid.n_max
    f_hat_lon = jnp.fft.rfft(field_3d, axis=1) / grid.n_lon
    f_m = f_hat_lon[:, :n_max + 1, :]
    twoπ = 2.0 * jnp.pi
    if _sh_gemm_enabled():
        # One shared FFT, two GEMM contractions (no gather) — matches the
        # two standalone GEMM analyses and the legacy fused output.
        oc2 = twoπ * _analysis_legendre_gemm(
            grid.wPnm_oc2, f_m, grid.ms, grid.ls, n_max)
        dmu = twoπ * _analysis_legendre_gemm(
            grid.wDnm, f_m, grid.ms, grid.ls, n_max)
    else:
        f_m_gathered = f_m[:, grid.ms, :]
        oc2 = twoπ * jnp.sum(grid.wPnm_oc2[:, :, None] * f_m_gathered, axis=0)
        dmu = twoπ * jnp.sum(grid.wDnm[:, :, None] * f_m_gathered, axis=0)
    return oc2, dmu


# Pole-safe floor for the 1/cos φ in the geographic spectral gradient (Gaussian
# grids carry no pole row, so this only bounds the highest |lat| Gaussian latitude).
_GRADIENT_COS_LAT_MIN = 1.0e-6


def spectral_gradient_3d(
    grid: GaussianGrid, coeffs_3d: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Geographic east/north horizontal gradient of a 3D scalar from its SH spectrum.

    Returns ``(dfdx, dfdy)`` on the Gaussian grid where ``dfdx = (1/(a cos φ)) ∂f/∂λ``
    (eastward) and ``dfdy = (1/a) ∂f/∂φ`` (northward), using the zonal derivative
    ``∂/∂λ → i·m`` (applied in :func:`sh_synthesis_3d`) and the meridional
    Legendre-derivative synthesis :func:`sh_synthesis_H_3d` (``∂/∂φ = -∂/∂θ`` for
    colatitude θ).  The shared spectral gradient operator (sibling of
    :func:`vordiv_from_uv_3d` / :func:`uv_from_vordiv_3d`); the spectral NH dycore
    and the column-forcing geostrophic diagnostic both use it (no parallel copy).

    Parameters
    ----------
    coeffs_3d : ``(n_sh, nlev)`` complex SH spectrum.

    Returns
    -------
    dfdx, dfdy : ``(n_lat, n_lon, nlev)`` real arrays.
    """
    a = grid.radius
    ims = grid.ms.astype(jnp.float64)
    cos_lat = jnp.clip(grid.cos_lat[:, None], _GRADIENT_COS_LAT_MIN, None)[..., None]
    dfdx = sh_synthesis_3d(grid, (1j * ims)[:, None] * coeffs_3d) / (a * cos_lat)
    dfdy = -sh_synthesis_H_3d(grid, coeffs_3d) / (a * cos_lat)
    return dfdx, dfdy


def uv_from_vordiv_3d(
    grid: GaussianGrid,
    vor_hat_3d: jax.Array,
    div_hat_3d: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reconstruct (u*cos_lat, v*cos_lat) at all levels from spectral
    vor/div — 3D-native.

    Same algorithm as :func:`uv_from_vordiv` but each synthesis is
    batched across both the (psi, chi) potentials *and* all vertical
    levels, using the trailing-axis-passive-batch property of
    ``sh_synthesis_3d`` / ``sh_synthesis_H_3d``.  4 per-call SH
    syntheses collapse to 2 (one ``segment_sum`` + IRFFT for both
    ``d/dtheta`` and ``d/dlon`` — psi/chi share the kernel).

    Parameters
    ----------
    vor_hat_3d, div_hat_3d : (n_sh, nlev) complex arrays.

    Returns
    -------
    u_cos, v_cos : (n_lat, n_lon, nlev) real arrays.
    """
    a = grid.radius

    # Streamfunction and velocity potential in spectral space
    # (ilap is shape (n_sh,) — broadcasts over the trailing level axis).
    psi_hat = grid.ilap[:, None] * vor_hat_3d
    chi_hat = grid.ilap[:, None] * div_hat_3d

    n_sh_pc, nlev_pc = psi_hat.shape
    # Stack (psi, chi) along a trailing axis and fold into the level
    # dim so each SH synthesis runs once on a thicker (n_sh, nlev*2)
    # tensor instead of being called twice on (n_sh, nlev).
    pc_stack = jnp.stack([psi_hat, chi_hat], axis=-1)  # (n_sh, nlev, 2)
    pc_flat = pc_stack.reshape(n_sh_pc, nlev_pc * 2)

    pc_dtheta_flat = sh_synthesis_H_3d(grid, pc_flat) / a
    pc_dtheta = pc_dtheta_flat.reshape(
        pc_dtheta_flat.shape[0], pc_dtheta_flat.shape[1], nlev_pc, 2,
    )
    dpsi_dtheta = pc_dtheta[..., 0]
    dchi_dtheta = pc_dtheta[..., 1]

    im_pc = (1j * grid.ms)[:, None] * pc_flat
    pc_dlon_flat = sh_synthesis_3d(grid, im_pc) / a
    pc_dlon = pc_dlon_flat.reshape(
        pc_dlon_flat.shape[0], pc_dlon_flat.shape[1], nlev_pc, 2,
    )
    dpsi_dlon = pc_dlon[..., 0]
    dchi_dlon = pc_dlon[..., 1]

    u_cos = dpsi_dtheta + dchi_dlon
    v_cos = dpsi_dlon - dchi_dtheta
    return u_cos, v_cos


def vordiv_from_uv_3d(
    grid: GaussianGrid,
    u_grid: jax.Array,
    v_grid: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Forward transform a grid-space vector field to spectral (vor, div).

    Inverse of :func:`uv_from_vordiv_3d` (up to the n=0 mode, which is
    annihilated by both directions because vor and div have no constant
    mode on the sphere).

    Implements the Hack & Jakob (1992) / Bourke (1972) spectral
    divergence and curl operators in pole-safe form.  For a vector field
    ``F = (F_x, F_y)`` with ``A = F_x · cos φ`` and ``B = F_y · cos φ``::

        div_hat  = (im/a) · sh_oc2(A) - (1/a) · sh_dmu(B)
        vor_hat  = (im/a) · sh_oc2(B) + (1/a) · sh_dmu(A)

    where ``sh_oc2`` carries an embedded ``1/cos²φ`` weighting (pole-safe)
    and ``sh_dmu`` carries the dPnm/dμ kernel.

    Parameters
    ----------
    grid : GaussianGrid
    u_grid, v_grid : (n_lat, n_lon, nlev) real arrays
        Grid-space vector components in **physical** units (NOT pre-multiplied
        by ``cos φ``).

    Returns
    -------
    vor_hat, div_hat : (n_sh, nlev) complex arrays.
    """
    a = grid.radius
    cos_lat_3d = grid.cos_lat[:, None, None]
    A = u_grid * cos_lat_3d   # F_x · cos φ
    B = v_grid * cos_lat_3d   # F_y · cos φ

    # Stack (A, B) along a trailing axis and fold into the level dim so
    # each SH-analysis variant runs once on a thicker
    # (n_lat, n_lon, nlev*2) tensor — matches the
    # ``uv_from_vordiv_3d`` / spectral PE batching pattern.  4 SH
    # forwards collapse to 2.
    n_lat_t, n_lon_t, nlev_t = A.shape
    AB_stack = jnp.stack([A, B], axis=-1)
    AB_flat = AB_stack.reshape(n_lat_t, n_lon_t, nlev_t * 2)
    AB_oc2_flat = sh_analysis_oc2_3d(grid, AB_flat)
    AB_dmu_flat = sh_analysis_dmu_3d(grid, AB_flat)
    AB_oc2 = AB_oc2_flat.reshape(AB_oc2_flat.shape[0], nlev_t, 2)
    AB_dmu = AB_dmu_flat.reshape(AB_dmu_flat.shape[0], nlev_t, 2)
    A_oc2, B_oc2 = AB_oc2[..., 0], AB_oc2[..., 1]
    A_dmu, B_dmu = AB_dmu[..., 0], AB_dmu[..., 1]

    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a

    div_hat = im_over_a[:, None] * A_oc2 - one_over_a * B_dmu
    vor_hat = im_over_a[:, None] * B_oc2 + one_over_a * A_dmu
    return vor_hat, div_hat


# ---------------------------------------------------------------------------
# Exact left-inverse grid-winds -> spectral vor/div analysis (#976)
# ---------------------------------------------------------------------------
#
# ``vordiv_from_uv_3d`` above implements the Bourke (1972) / Hack-Jakob (1992)
# ``oc2``/``dmu`` divergence-curl operators.  Those are the CORRECT spectral
# divergence/vorticity operators for the dycore's own de-aliased nonlinear
# FLUX products, and are cheap.  They are NOT, however, an exact left-inverse
# of the streamfunction/velocity-potential wind synthesis
# (:func:`uv_from_vordiv_3d`) at the TRUNCATION BOUNDARY: the meridional
# derivative maps a mode of total wavenumber ``n`` to ``n±1``, so the
# reconstructed winds carry power at ``n = n_max + 1`` that the truncated
# analysis basis cannot represent.  Continuously the telescoping
# ``∫(1-µ²)P'_nP'_{n'} = n(n+1)δ_{nn'} - m²∫P_nP_{n'}/(1-µ²)`` closes and
# ``A∘S = Id``; discretely the missing ``n_max+1`` partner leaves the
# ``n = n_max`` row unbalanced.  The result is a pole-concentrated
# amplifier — every input mode leaks spurious power into ``(n_max, m)`` and
# the ``(n_max, m)`` mode self-amplifies (~×21 per pass at T85), which
# blows up the ``carry -> state -> carry -> state`` round trip used by the
# WB2 eval driver (#951).  Adding Gaussian latitudes does NOT help
# (verified: identical gain for linear/quadratic/cubic grids) because it is
# an operator-truncation defect, not a quadrature-resolution defect.
#
# The exact left-inverse of the synthesis on the band-limited subspace,
# pole rows included, is obtained by solving the (per zonal wavenumber ``m``)
# least-squares system ``[U_m; V_m] = B_m · [psi; chi]`` where ``B_m`` is the
# real synthesis matrix built from the SAME ``Pnm``/``Hnm`` operators
# ``uv_from_vordiv_3d`` uses.  ``pinv(B_m)`` recovers ``(psi, chi)`` exactly
# for any winds in ``range(B_m)`` and gives the stable least-squares
# projection otherwise (no pole amplification).  ``B_m`` and its pseudo-
# inverse depend only on the grid, so they are precomputed once (host,
# numpy) and cached; the runtime path is a differentiable FFT + batched
# matmul + scatter (JIT/``jax.grad``/pytree safe, complex128).
_VORDIV_PINV_CACHE: dict = {}


def _vordiv_pinv_operators(grid: GaussianGrid):
    """Build & cache the per-``m`` pseudo-inverse of the wind synthesis.

    Returns ``(BP, lap_pad, sh_index, n_sh, n_pad)`` where ``BP`` has shape
    ``(n_max+1, 2*n_pad, 2*n_lat)`` (complex128), ``lap_pad`` is
    ``(n_max+1, n_pad)`` and ``sh_index`` is an int32 scatter map
    ``(n_max+1, n_pad)`` (``-1`` for padding slots).

    The operators depend only on ``grid`` (``Pnm``/``Hnm``/``lap``), so this
    reads *concrete* grid arrays; ``grid`` must be a static/closure object
    (the established contract for every spectral transform here), never a
    JIT-traced argument.
    """
    n_max = int(grid.n_max)
    n_lat = int(grid.n_lat)
    a = float(grid.radius)
    Pnm = np.asarray(grid.Pnm, dtype=np.float64)
    Hnm = np.asarray(grid.Hnm, dtype=np.float64)
    lap = np.asarray(grid.lap, dtype=np.float64)
    ms = np.asarray(grid.ms)
    ls = np.asarray(grid.ls)
    n_sh = int(Pnm.shape[1])
    # Max modes for any single m: (m..n_max) => at most n_max+1 (the m=0 column).
    n_pad = n_max + 1

    # Cache key: the scalar shape signature PLUS a byte-exact content hash of
    # every operator the inverse depends on, so a custom/modified/reordered
    # grid (same scalars, different Pnm/Hnm/lap/ms/ls) does NOT alias another
    # grid's inverse.
    _h = hashlib.blake2b(digest_size=16)
    for _arr in (Pnm, Hnm, lap, ms, ls):
        _c = np.ascontiguousarray(_arr)
        _h.update(str((_c.shape, _c.dtype.str)).encode())
        _h.update(_c.tobytes())
    key = (n_max, n_lat, a, _h.hexdigest())
    ops = _VORDIV_PINV_CACHE.get(key)
    if ops is not None:
        return ops

    # Guard against silent OOM at very high truncation: the padded operator is
    # ``64·(n_max+1)²·n_lat`` bytes (complex128).  ~117 MiB at T106, ~460 MiB
    # at T170, ~3.5 GiB at T340.  ``carry_to_spectral_state`` is an IC/eval
    # conversion (not the dycore hot loop); fail LOUDLY rather than thrash.
    bp_bytes = 16 * (n_max + 1) * (2 * n_pad) * (2 * n_lat)
    _bp_budget = int(os.environ.get("LEGOESM_VORDIV_PINV_MAX_BYTES",
                                    str(2 * 1024**3)))  # 2 GiB default
    if bp_bytes > _bp_budget:
        raise MemoryError(
            f"vordiv_from_uv_exact_3d operator for T{n_max} needs "
            f"{bp_bytes / 1024**3:.2f} GiB (> "
            f"{_bp_budget / 1024**3:.2f} GiB budget). Raise "
            "LEGOESM_VORDIV_PINV_MAX_BYTES to allow it, or keep the exact "
            "wind round trip to eval/IC at supported truncations."
        )

    BP = np.zeros((n_max + 1, 2 * n_pad, 2 * n_lat), dtype=np.complex128)
    lap_pad = np.zeros((n_max + 1, n_pad), dtype=np.float64)
    sh_index = np.full((n_max + 1, n_pad), -1, dtype=np.int32)

    for m in range(n_max + 1):
        idx = np.where((ms == m) & (ls > 0))[0]  # exclude n=0 (no vor/div mean)
        k = int(idx.size)
        if k == 0:
            continue
        P = Pnm[:, idx]
        H = Hnm[:, idx]
        imP = (1j * m) * P
        # Synthesis (see uv_from_vordiv_3d):
        #   U_m = (H · psi + im·P · chi) / a
        #   V_m = (im·P · psi - H · chi) / a
        top = np.concatenate([H, imP], axis=1)          # (n_lat, 2k)  -> U_m
        bot = np.concatenate([imP, -H], axis=1)         # (n_lat, 2k)  -> V_m
        B = np.concatenate([top, bot], axis=0) / a      # (2 n_lat, 2k)
        Bp = np.linalg.pinv(B)                           # (2k, 2 n_lat)
        BP[m, :k, :] = Bp[:k, :]                         # psi rows
        BP[m, n_pad:n_pad + k, :] = Bp[k:, :]            # chi rows
        lap_pad[m, :k] = lap[idx]
        sh_index[m, :k] = idx

    ops = (
        jnp.asarray(BP),
        jnp.asarray(lap_pad),
        jnp.asarray(sh_index),
        n_sh,
        n_pad,
    )
    _VORDIV_PINV_CACHE[key] = ops
    return ops


def vordiv_from_uv_exact_3d(
    grid: GaussianGrid,
    u_grid: jax.Array,
    v_grid: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Exact left-inverse of :func:`uv_from_vordiv_3d` (#976).

    Grid-space winds ``(u, v)`` (physical units, NOT pre-multiplied by
    ``cos φ``) -> spectral ``(vor_hat, div_hat)``.  Unlike
    :func:`vordiv_from_uv_3d` this is the exact LEFT-INVERSE of the
    streamfunction/velocity-potential synthesis on the band-limited
    subspace (``range(B_m)``): for winds that are a synthesis of some
    band-limited ``(vor, div)`` it recovers them to machine precision, pole
    rows included, so ``carry -> state -> carry`` round trips are idempotent
    at every truncation (no polar amplification).  For winds OUTSIDE that
    subspace (raw ERA5) it returns the minimum-Euclidean-residual fit at the
    Gaussian nodes — a stable projection, but NOT the area-weighted (metric)
    adjoint; the fixed point of the round trip is still exact.

    Solves, per zonal wavenumber ``m``, the least-squares system
    ``[U_m; V_m] = B_m · [psi; chi]`` (``B_m`` built from the SAME
    ``Pnm``/``Hnm`` the synthesis uses) with the precomputed ``pinv(B_m)``,
    then ``vor = lap · psi`` / ``div = lap · chi``.  Differentiable
    (FFT + matmul + scatter), complex128; ``n=0`` vor/div are exactly zero.

    Parameters
    ----------
    u_grid, v_grid : (n_lat, n_lon, nlev) real arrays.

    Returns
    -------
    vor_hat, div_hat : (n_sh, nlev) complex arrays.
    """
    BP, lap_pad, sh_index, n_sh, n_pad = _vordiv_pinv_operators(grid)

    n_max = grid.n_max
    nlev = u_grid.shape[2]
    cos_lat_3d = grid.cos_lat[:, None, None]
    u_cos = u_grid * cos_lat_3d
    v_cos = v_grid * cos_lat_3d

    # Fourier coefficients (m = 0..n_max) — same normalization as sh_analysis.
    U = jnp.fft.rfft(u_cos, axis=1)[:, : n_max + 1, :] / grid.n_lon
    V = jnp.fft.rfft(v_cos, axis=1)[:, : n_max + 1, :] / grid.n_lon
    # (n_lat, n_max+1, nlev) -> (n_max+1, n_lat, nlev)
    U = jnp.moveaxis(U, 1, 0)
    V = jnp.moveaxis(V, 1, 0)
    rhs = jnp.concatenate([U, V], axis=1)  # (n_max+1, 2 n_lat, nlev)

    # Per-m least-squares solve:
    # (n_max+1, 2 n_pad, 2 n_lat) @ (n_max+1, 2 n_lat, nlev)
    sol = jnp.einsum("mij,mjl->mil", BP, rhs)  # (n_max+1, 2 n_pad, nlev)
    psi = sol[:, :n_pad, :]
    chi = sol[:, n_pad:, :]
    vor_pad = lap_pad[:, :, None] * psi  # (n_max+1, n_pad, nlev)
    div_pad = lap_pad[:, :, None] * chi

    # Scatter (m, local-n) -> flat SH index.  Each valid index appears once;
    # padding slots map to index 0 with a zeroed contribution.
    flat_idx = sh_index.reshape(-1)             # ((n_max+1)*n_pad,)
    valid = (flat_idx >= 0)[:, None]            # (·, 1)
    safe_idx = jnp.where(flat_idx >= 0, flat_idx, 0)
    zero_c = jnp.zeros((), dtype=jnp.complex128)
    vp = jnp.where(valid, vor_pad.reshape(-1, nlev), zero_c)
    dp = jnp.where(valid, div_pad.reshape(-1, nlev), zero_c)
    zeros = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    vor_hat = zeros.at[safe_idx].add(vp)
    div_hat = zeros.at[safe_idx].add(dp)
    return vor_hat, div_hat


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
