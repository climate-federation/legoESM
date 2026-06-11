"""Bridge: Veros snapshot ↔ legoESM ``LatLonCGridOceanState``.

Veros and legoESM both use the **Arakawa C-grid** (u at east faces,
v at north faces, scalars at cell centres) — verified against Veros
source ``veros/variables.py`` which defines ``U_GRID = ("xu", "yt",
"zt")`` and ``V_GRID = ("xt", "yu", "zt")``.

Differences this module bridges:

1. **Halos**: Veros arrays carry 2-cell halos on each side
   (``vs.u`` shape ``(nx+4, ny+4, nz, n_tau)``). legoESM has none
   internally — operators handle boundaries via ``jnp.roll``.

2. **Time-level dim**: Veros leapfrog stores 3 time levels indexed
   by ``vs.tau`` (current), ``vs.taup1`` (next), ``vs.taum1``
   (previous). legoESM stores only the current state.

3. **Vertical index direction**: Veros has ``k=0`` deepest, ``k=nz-1``
   surface. legoESM has ``k=0`` surface, ``k=nz-1`` deepest. The
   bridge reverses the vertical axis.

4. **u-face count**: Veros u has ``nx`` zonal samples (after halo
   strip); legoESM lat-lon C-grid u has ``n_lon + 1`` (the extra
   sample is the periodic wrap-around face). The bridge sets
   ``legoesm_u[:, n_lon, :] = legoesm_u[:, 0, :]``.

5. **Veros (xt, yt) vs legoESM (lat, lon)**: Veros stores arrays
   as ``(x, y, z, tau)``; legoESM as ``(lat, lon, level)``. The
   bridge transposes the first two axes.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm.ocean.fidelity.veros_runner import VerosResult
from legoesm.ocean.state import LatLonCGridOceanState


# Veros's halo width (2 cells on each side; standard for the leapfrog +
# Adams-Bashforth stencils).
_VEROS_HALO = 2


def _strip_halo_xy(arr: np.ndarray) -> np.ndarray:
    """Strip 2-cell halos from the first two axes."""
    return arr[_VEROS_HALO:-_VEROS_HALO, _VEROS_HALO:-_VEROS_HALO]


def _veros_xy_to_legoesm_latlon(arr: np.ndarray) -> np.ndarray:
    """Transpose Veros's (x, y, ...) to legoESM's (lat=y, lon=x, ...)."""
    # arr is (nx, ny, ...) → swap axes 0 and 1.
    return np.swapaxes(arr, 0, 1)


def _reverse_z(arr: np.ndarray) -> np.ndarray:
    """Reverse the last (vertical) axis: Veros k=0 deep → legoESM k=0 surface."""
    return arr[..., ::-1]


def veros_u_centered_z_centres(dz_top_down: np.ndarray) -> np.ndarray:
    """Veros cell-centre depths ``zt`` via pyOM's ``u_centered_grid`` recursion.

    Veros does NOT place its T points at cell midpoints: ``u_centered_grid``
    (core/numerics.py:9-21, applied to z at :74-76) builds the REFLECTED
    recursion ``zt[k] = 2·zw[k-1] − zt[k-1]`` with ``zw_raw[k] = Σ_{i≥1}
    dzt[i]`` and ``zt[0] = zw[0] − dzt[0]/2``, then shifts so the surface
    interface is 0.  On a STRETCHED grid this differs from midpoints by up to
    half a cell (global_4deg: 25 m), and the derived centre spacing ``dzw =
    zt[k+1]−zt[k]`` (the denominator of every Veros vertical gradient and the
    implicit-solve metric) alternates around the midpoint value — global_4deg
    interior dzw = [30, 110, 90, 190, 190, …] m (top-down) vs midpoint
    [60, 85, 120, 165, 215, …].  A faithful recipe must build its
    ``z_full_ref``/``dz_half_ref`` from THIS construction or every isoneutral
    slope (∝ 1/∂_zρ), K_33 (∝ S²) and vertical gradient inherits an
    alternating per-level bias (tier-2 isolation: F_z rms lego/Veros by level
    = [1.71, 0.84, 1.26, 0.88, 1.10, …] — exactly the dz_half/dzw ratios).
    On a UNIFORM grid the recursion reduces to midpoints (the flat-bottom ACC
    channel divides the same stretched ddz by 2.5, so it is NOT uniform and
    carries the same bias).

    Parameters
    ----------
    dz_top_down : (nlev,) layer thicknesses, k=0 = surface (legoESM order).

    Returns
    -------
    z_full_ref : (nlev,) cell-centre depths, top-down, negative (legoESM
        convention) — bit-identical to Veros's ``zt`` (reversed).
    """
    dzt = np.asarray(dz_top_down, dtype=np.float64)[::-1]  # Veros bottom-up
    nz = dzt.shape[0]
    zw = np.zeros(nz)
    zw[1:] = np.cumsum(dzt[1:])
    zt = np.zeros(nz)
    zt[0] = zw[0] - dzt[0] * 0.5
    for k in range(1, nz):
        zt[k] = 2.0 * zw[k - 1] - zt[k - 1]
    zt = zt - zw[-1]
    return zt[::-1]


def _extract_veros_var(
    result: VerosResult, name: str, tau: int = 1,
) -> np.ndarray:
    """Pull a (nx, ny, nz) snapshot of a Veros variable.

    Strips halos, selects time level ``tau`` (1 = current; Veros's
    ``vs.tau`` for a freshly-stepped state), transposes to
    ``(lat, lon, level)``, reverses the vertical axis.

    Parameters
    ----------
    result : VerosResult
        Output from ``run_veros``.
    name : str
        Variable name (e.g. ``"u"``, ``"temp"``, ``"du_cor"``).
    tau : int
        Time level. 1 selects the "current" level for arrays of
        shape ``(..., 3)``; ignored for arrays without a time dim.

    Returns
    -------
    array
        Shape ``(n_lat, n_lon, n_lev)`` or ``(n_lat, n_lon)`` for 2-D
        fields. Float64 numpy array.
    """
    arr = result.variables[name]
    arr = np.asarray(arr, dtype=np.float64)
    # Strip halos on (x, y).
    arr = _strip_halo_xy(arr)
    # If the last dim is a time-level axis (size 3), select tau.
    if arr.ndim >= 2 and arr.shape[-1] == 3:
        arr = arr[..., tau]
    # 3-D (x, y, z) or 2-D (x, y) — transpose first two axes.
    arr = _veros_xy_to_legoesm_latlon(arr)
    # Reverse vertical axis if it exists.
    if arr.ndim == 3:
        arr = _reverse_z(arr)
    return arr


# ---------------------------------------------------------------------------
# State translation: Veros snapshot → LatLonCGridOceanState
# ---------------------------------------------------------------------------


def _wrap_u_periodic(u_cell: np.ndarray) -> np.ndarray:
    """Convert Veros u of shape ``(n_lat, n_lon, n_lev)`` to legoESM u
    of shape ``(n_lat, n_lon + 1, n_lev)`` by appending the
    wrap-around face. Veros's u at xu_index = i is the eastern face
    of T-cell i; legoESM's u_face[:, j] is the western face of
    T-cell j (= eastern face of T-cell j-1). With both periodic and
    aligned at the wrap, the legoESM eastern face at column n_lon
    equals legoESM at column 0."""
    extra = u_cell[:, :1, :]
    return np.concatenate([u_cell, extra], axis=1)


def _pad_v_walls(v_cell: np.ndarray) -> np.ndarray:
    """Convert Veros v of shape ``(n_lat, n_lon, n_lev)`` to legoESM v
    of shape ``(n_lat + 1, n_lon, n_lev)``. Veros's v at yu_index = j
    is the northern face of T-cell j; legoESM's v_face[i, :] is the
    southern face of T-cell i (= northern face of T-cell i-1).
    Append a zero row at the north wall (ACC closed N boundary)."""
    zero = np.zeros_like(v_cell[:1, :, :])
    return np.concatenate([v_cell, zero], axis=0)


class StateBridgeOutput(NamedTuple):
    """Result of :func:`veros_snapshot_to_legoesm_state`."""
    state: LatLonCGridOceanState
    info: dict   # provenance / shapes / indexing diagnostics


def veros_snapshot_to_legoesm_state(
    result: VerosResult,
    base_state: LatLonCGridOceanState,
    tau: int = 1,
) -> StateBridgeOutput:
    """Initialise a legoESM lat-lon C-grid state from a Veros snapshot.

    Parameters
    ----------
    result : VerosResult
        Output of :func:`legoesm.ocean.fidelity.veros_runner.run_veros`.
    base_state : LatLonCGridOceanState
        Empty legoESM state at the matching grid (built from
        :func:`build_acc_state` or similar). Provides the grid /
        land-mask / face-mask templates; T, S, u, v, eta values are
        overwritten from the Veros snapshot.
    tau : int
        Veros time-level index to extract (default 1 = "current").

    Returns
    -------
    StateBridgeOutput
        ``state`` is the populated ``LatLonCGridOceanState``; ``info``
        contains shape and tau metadata for debugging.
    """
    temp = _extract_veros_var(result, "temp", tau=tau)   # (n_lat, n_lon, n_lev)
    salt = _extract_veros_var(result, "salt", tau=tau)
    u_cell = _extract_veros_var(result, "u", tau=tau)
    v_cell = _extract_veros_var(result, "v", tau=tau)

    # Veros u sits at the EAST face of each T-cell; legoESM u_face at
    # column j is the WEST face of T-cell j. Index mapping:
    #     legoesm_u[:, 0, :]    = legoesm_u[:, n_lon, :] (periodic wrap)
    # so the "wrap" face is just a duplicate of column 0 / n_lon.
    u_face = _wrap_u_periodic(u_cell)        # (n_lat, n_lon + 1, n_lev)
    v_face = _pad_v_walls(v_cell)             # (n_lat + 1, n_lon, n_lev)

    # eta: Veros has it in result.variables when captured; fall back to
    # zero (rest-state) if absent.
    if "psi" in result.variables:
        # Veros uses streamfunction psi for the barotropic — not eta.
        # For tier-2 (per-process tendency) the absolute eta doesn't
        # matter much; the PGF tendency depends on grad(eta). Start with
        # zero eta and let the integrator spin it up from the surface
        # forcing.
        eta = np.zeros_like(base_state.eta.data)
    else:
        eta = np.zeros_like(base_state.eta.data)

    # Shape sanity-check vs. the base state.
    expected_T = base_state.T.data.shape
    expected_u = base_state.u.data.shape
    expected_v = base_state.v.data.shape
    if temp.shape != expected_T:
        # When Veros has ny=42 cells and legoESM has 44 (with N/S walls),
        # we pad the Veros 42-row data into the interior rows of the
        # 44-row legoESM array — leaving the N/S wall rows untouched.
        if (
            temp.shape[1:] == expected_T[1:]
            and temp.shape[0] + 2 == expected_T[0]
        ):
            temp = _pad_y_walls(temp)
            salt = _pad_y_walls(salt)
            u_face = _pad_y_walls(u_face)
            v_face = _pad_y_walls(v_face)
        else:
            raise ValueError(
                f"Veros snapshot T shape {temp.shape} does not match "
                f"base_state T shape {expected_T}, and the y-wall "
                f"padding heuristic does not apply."
            )

    state = base_state._replace(
        T=base_state.T.replace(data=jnp.asarray(temp)),
        S=base_state.S.replace(data=jnp.asarray(salt)),
        u=base_state.u.replace(data=jnp.asarray(u_face)),
        v=base_state.v.replace(data=jnp.asarray(v_face)),
        eta=base_state.eta.replace(data=jnp.asarray(eta)),
    )

    info = {
        "tau": tau,
        "veros_T_shape_after_strip": list(temp.shape),
        "legoesm_T_shape": list(expected_T),
        "legoesm_u_shape": list(expected_u),
        "legoesm_v_shape": list(expected_v),
        "veros_version": result.provenance.get("veros_version"),
    }
    return StateBridgeOutput(state=state, info=info)


# ---------------------------------------------------------------------------
# Tendency translation: Veros per-process diagnostics → ProbeResult-shaped dict
# ---------------------------------------------------------------------------


# Map of Veros tendency variable name → (legoESM probe field name).
# Some legoESM probe fields aggregate multiple Veros tendencies, so the
# mapping is many-to-one in some places.
#
# Specifically, Veros's ``du_adv`` covers what legoESM's
# ``MomentumTendencyDiagnostics`` splits across ``vortcor_u +
# vertadv_u`` (and the Coriolis term is in ``du_cor`` separately).
# Veros's ``du_mix`` corresponds to legoESM's ``av_vert_u +
# botdrag_u`` (vertical viscosity + bottom drag). Mapping these
# 1:1 isn't strictly possible without further splitting — for the
# first tier-2 acceptance run we compare them as aggregates.

VEROS_TO_LEGOESM_MOMENTUM = {
    "du_cor": "coriolis_u",
    "dv_cor": "coriolis_v",
    "du_adv": "veros_du_adv",   # aggregate; compare to vortcor+vertadv on legoESM side
    "dv_adv": "veros_dv_adv",
    "du_mix": "veros_du_mix",
    "dv_mix": "veros_dv_mix",
}

VEROS_TO_LEGOESM_TRACER = {
    "dtemp_hmix": "veros_dT_hmix",
    "dtemp_vmix": "veros_dT_vmix",
    "dtemp_iso":  "veros_dT_iso",
    "dsalt_hmix": "veros_dS_hmix",
    "dsalt_vmix": "veros_dS_vmix",
    "dsalt_iso":  "veros_dS_iso",
}

VEROS_TENDENCY_CAPTURE_VARS: tuple[str, ...] = (
    "u", "v", "temp", "salt", "rho",
    "surface_taux", "surface_tauy",
    *VEROS_TO_LEGOESM_MOMENTUM.keys(),
    *VEROS_TO_LEGOESM_TRACER.keys(),
)


def _pad_y_walls(arr: np.ndarray) -> np.ndarray:
    """Pad a (n_lat, ...) array with zero rows at the N + S walls so it
    matches the legoESM lat-lon C-grid convention of ``n_lat + 2``
    rows on a ``create_regional_latlon_grid`` output."""
    pad = np.zeros_like(arr[:1])
    return np.concatenate([pad, arr, pad], axis=0)


def extract_veros_tendencies(
    result: VerosResult,
    tau: int = 1,
    pad_y_walls: bool = True,
) -> dict:
    """Pull all per-process tendency arrays from a Veros result, in
    legoESM-shaped form (lat, lon, level).

    Parameters
    ----------
    result : VerosResult
    tau : int
        Veros time-level index.
    pad_y_walls : bool
        When ``True`` (default), pads each array with one zero row at
        both the north and south ends so the shape matches legoESM's
        ``create_regional_latlon_grid`` convention (``n_lat + 2``
        rows). When ``False``, returns Veros interior shape
        ``(NY, NX, NZ)``.
    """
    pad = _pad_y_walls if pad_y_walls else (lambda a: a)
    out: dict[str, np.ndarray] = {}
    for veros_name, lego_name in VEROS_TO_LEGOESM_MOMENTUM.items():
        if veros_name in result.variables:
            out[lego_name] = pad(_extract_veros_var(result, veros_name, tau=tau))
    for veros_name, lego_name in VEROS_TO_LEGOESM_TRACER.items():
        if veros_name in result.variables:
            out[lego_name] = pad(_extract_veros_var(result, veros_name, tau=tau))
    if "rho" in result.variables:
        # Veros stores ``vs.rho`` as the density ANOMALY (``rho - rho_0``)
        # — see ``veros/core/density/nonlinear_eq{1,2,3}.py`` which all
        # return ``rho_anom`` and ``veros/core/numerics.py:265`` which
        # assigns ``vs.rho = get_rho(...)``. legoESM's ``probe.rho`` is
        # the in-situ density (anomaly + ``rho_0``), so we add Veros's
        # canonical rho_0 = 1024 here for an apples-to-apples comparison.
        out["rho"] = pad(_extract_veros_var(result, "rho", tau=tau)) + 1024.0
    return out


__all__ = (
    "StateBridgeOutput",
    "VEROS_TENDENCY_CAPTURE_VARS",
    "VEROS_TO_LEGOESM_MOMENTUM",
    "VEROS_TO_LEGOESM_TRACER",
    "extract_veros_tendencies",
    "veros_snapshot_to_legoesm_state",
)
