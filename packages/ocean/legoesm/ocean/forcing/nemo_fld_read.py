"""NEMO ``fld_read`` weighted forcing operators for native ORCA cards.

The host loads NetCDF records and the NEMO-generated ``srcNN``/``wgtNN``
tables.  The numerical map itself is pure JAX and follows
``fldread.F90:1120-1241,1349-1579``: four weighted source points, followed by
the three four-term gradient groups for bicubic weights.  It is shared by the
ORCA2 fidelity card and the future ORCA1 card; neither card owns a second
regridding formula.

Arrays use NetCDF/Python ``(j, i)`` ordering.  NEMO's source indices are
one-based Fortran-flat indices (``i`` fastest); :func:`decode_source_indices`
converts them once on the host.  Time record selection/interpolation remains a
separate operation because it is deck metadata, not spatial interpolation.
"""

from __future__ import annotations

import ctypes
import math

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.source_rounding import nemo_source_round as _sr

_LIBM = ctypes.CDLL("libm.so.6")
_LIBM_SINCOS = _LIBM.sincos
_LIBM_SINCOS.argtypes = (
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
)
_LIBM_SINCOS.restype = None
_LIBM_TAN = _LIBM.tan
_LIBM_TAN.argtypes = (ctypes.c_double,)
_LIBM_TAN.restype = ctypes.c_double


def _scalar_sincos(value: float) -> tuple[float, float]:
    """The glibc scalar ``sincos`` call emitted by the oracle compiler."""
    sine = ctypes.c_double()
    cosine = ctypes.c_double()
    _LIBM_SINCOS(value, ctypes.byref(sine), ctypes.byref(cosine))
    return sine.value, cosine.value


def decode_source_indices(src_tables: np.ndarray) -> np.ndarray:
    """Convert NEMO weight-file ``srcNN`` values to zero-based flat indices."""
    source = np.asarray(src_tables, dtype=np.float64)
    if source.ndim != 3 or source.shape[0] < 4:
        raise ValueError(f"expected (>=4,j,i) source tables, got {source.shape}")
    rounded = np.rint(source)
    if not np.array_equal(source, rounded) or np.any(rounded < 1):
        raise ValueError("NEMO srcNN tables must contain positive integer values")
    return rounded.astype(np.int64) - 1


def _weighted_add(total, weight, value):
    return _sr(_sr(total) + _sr(_sr(weight) * _sr(value)))


def nemo_fld_interp(source, source_indices, weights, *, bicubic: bool):
    """Apply NEMO's bilinear/bicubic ``fld_interp`` source program.

    ``source`` is one forcing record ``(src_j,src_i)``.  ``source_indices`` is
    ``(4,dst_j,dst_i)`` after :func:`decode_source_indices`; ``weights`` has
    4 planes for bilinear or 16 for bicubic.  The static ``bicubic`` flag makes
    the function production-JIT safe.
    """
    source = jnp.asarray(source, dtype=jnp.float64)
    indices = jnp.asarray(source_indices, dtype=jnp.int64)
    weight = jnp.asarray(weights, dtype=jnp.float64)
    if indices.ndim != 3 or indices.shape[0] != 4:
        raise ValueError(f"source_indices must be (4,j,i), got {indices.shape}")
    expected_weights = 16 if bicubic else 4
    if weight.shape != (expected_weights,) + indices.shape[1:]:
        raise ValueError(
            f"weights must be {(expected_weights,) + indices.shape[1:]}, "
            f"got {weight.shape}"
        )
    src_j, src_i = source.shape
    ii = indices % src_i
    jj = indices // src_i
    if not isinstance(source_indices, jax.core.Tracer):
        # Host-provided NEMO tables are static data.  Reject corruption rather
        # than clipping it into a different interpolation program.
        host_indices = np.asarray(source_indices)
        if np.any(host_indices >= src_i * src_j):
            raise ValueError("source index outside forcing record")

    total = jnp.zeros(indices.shape[1:], dtype=jnp.float64)
    for corner in range(4):
        total = _weighted_add(total, weight[corner], source[jj[corner], ii[corner]])

    if bicubic:
        differences = []
        for corner in range(4):
            ip = (ii[corner] + 1) % src_i
            im = (ii[corner] - 1) % src_i
            jp = jnp.minimum(jj[corner] + 1, src_j - 1)
            jm = jnp.maximum(jj[corner] - 1, 0)
            di = _sr(source[jj[corner], ip] - source[jj[corner], im])
            dj = _sr(source[jp, ii[corner]] - source[jm, ii[corner]])
            dij = _sr(
                _sr(source[jp, ip] - source[jp, im])
                - _sr(source[jm, ip] - source[jm, im])
            )
            differences.append((di, dj, dij))
        # fldread.F90 uses three complete jn=1:4 loops: every i-gradient,
        # then every j-gradient, then every ij-gradient.  Preserve both that
        # inter-corner order and the written ``wgt * scale * difference``
        # association.
        for offset, scale, component in ((4, 0.5, 0), (8, 0.5, 1), (12, 0.25, 2)):
            for corner in range(4):
                plane = corner + offset
                difference = differences[corner][component]
                term = _sr(_sr(_sr(weight[plane]) * _sr(scale)) * difference)
                total = _sr(_sr(total) + term)
    return total


def nemo_t_rotation_from_domain(
    glamt_deg: np.ndarray,
    gphit_deg: np.ndarray,
    glamv_deg: np.ndarray,
    gphiv_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar-libm T-point ``gcost,gsint`` from ``geo2ocean.F90:167-297``.

    Rotation is static grid construction, so it is intentionally evaluated on
    the host one scalar at a time, matching the campaign's static-geometry
    scalar-libm precedent.  The returned arrays feed the production-JIT
    :func:`rotate_en_to_ij` operator.  The south edge has no preceding interior
    V row and is the geographic arm for ORCA_R2; it is initialized to (1,0),
    as NEMO's post-angle boundary handling resolves it.
    """
    glamt = np.asarray(glamt_deg, dtype=np.float64)
    gphit = np.asarray(gphit_deg, dtype=np.float64)
    glamv = np.asarray(glamv_deg, dtype=np.float64)
    gphiv = np.asarray(gphiv_deg, dtype=np.float64)
    if not (glamt.shape == gphit.shape == glamv.shape == gphiv.shape):
        raise ValueError("T/V longitude and latitude arrays must share a shape")
    ny, nx = glamt.shape
    cosine = np.ones((ny, nx), dtype=np.float64)
    sine = np.zeros((ny, nx), dtype=np.float64)
    rpi = np.float64(3.141592653589793)
    rad = rpi / np.float64(180.0)
    for jj in range(1, ny):
        for ji in range(nx):
            lam, phi = float(glamt[jj, ji]), float(gphit[jj, ji])
            lam_v, phi_v = float(glamv[jj, ji]), float(gphiv[jj, ji])
            lam_b, phi_b = float(glamv[jj - 1, ji]), float(gphiv[jj - 1, ji])
            sin_lam, cos_lam = _scalar_sincos(float(rad) * lam)
            sin_v, cos_v = _scalar_sincos(float(rad) * lam_v)
            sin_b, cos_b = _scalar_sincos(float(rad) * lam_b)
            tan_lam = _LIBM_TAN(float(rpi) / 4.0 - float(rad) * phi / 2.0)
            tan_v = _LIBM_TAN(float(rpi) / 4.0 - float(rad) * phi_v / 2.0)
            tan_b = _LIBM_TAN(float(rpi) / 4.0 - float(rad) * phi_b / 2.0)
            xn = 0.0 - (cos_lam + cos_lam) * tan_lam
            yn = 0.0 - (sin_lam + sin_lam) * tan_lam
            xv = (cos_v + cos_v) * tan_v - (cos_b + cos_b) * tan_b
            yv = (sin_v + sin_v) * tan_v - (sin_b + sin_b) * tan_b
            norm = max(
                math.sqrt((xn * xn + yn * yn) * (xv * xv + yv * yv)),
                1.0e-14,
            )
            sine[jj, ji] = (xn * yv - yn * xv) / norm
            cosine[jj, ji] = (xn * xv + yn * yv) / norm
            if abs(lam_v - lam_b) % 360.0 < 1.0e-8:
                sine[jj, ji], cosine[jj, ji] = 0.0, 1.0
    return cosine, sine


def rotate_en_to_ij(u_east, v_north, cosine, sine):
    """NEMO ``rot_rep(...,'T','en->i/j')`` in one production-JIT call."""
    u = jnp.asarray(u_east, dtype=jnp.float64)
    v = jnp.asarray(v_north, dtype=jnp.float64)
    c = jnp.asarray(cosine, dtype=jnp.float64)
    s = jnp.asarray(sine, dtype=jnp.float64)
    if not (u.shape == v.shape == c.shape == s.shape):
        raise ValueError("wind and rotation arrays must share a shape")
    ui = _sr(_sr(_sr(u) * _sr(c)) + _sr(_sr(v) * _sr(s)))
    vj = _sr(_sr(_sr(v) * _sr(c)) - _sr(_sr(u) * _sr(s)))
    return ui, vj


# --- fld_read time records: monthly climatology, 'yearly' file -------------
_NSECD = 86400                                                  # daymod nsecd
_NOLEAP_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)  # daymod.f90:182


def _fortran_half(value: int) -> int:
    """Fortran ``INTEGER / 2``: truncation toward zero."""
    return -((-value) // 2) if value < 0 else value // 2


def nemo_clim_monthly_record_centres(n_years: int, *, nleapy: int = 0):
    """Record centres of a 12-record climatological monthly file.

    ``sn_* = '<file>', -1., '<var>', .true., .true., 'yearly'`` with no
    ``+``/``-`` record shift.  Times are integer seconds since Jan 1st 00h of
    the nit000 year, from the PREVIOUS year's records (``fld_init`` reads them
    when the first step precedes the January centre, fldread.f90:281) through
    year ``n_years - 1``.  Month boundaries are ``nmonth_beg``
    (daymod.f90:203-209), the yearly-file slice is fldread.f90:916, and each
    centre is NEMO's integer rounded average, fldread.f90:934-936.  Returns
    ``(centres_s, record_index)``: int64 arrays, ``record_index`` zero-based
    into the file's 12 records (a climatology reuses them every year).
    """
    if nleapy != 0:
        raise ValueError(
            f"nleapy={nleapy}: only the 365-day calendar (nn_leapy=0) is "
            "transcribed")
    if n_years < 1:
        raise ValueError("n_years must be >= 1")
    bounds = [-365 * _NSECD]
    for _ in range(n_years + 1):
        for days in _NOLEAP_MONTH_DAYS:
            bounds.append(bounds[-1] + days * _NSECD)
    centres = [
        _fortran_half(b0) + _fortran_half(b1)
        + max(int(math.fmod(b0, 2)), int(math.fmod(b1, 2)))
        for b0, b1 in zip(bounds[:-1], bounds[1:])
    ]
    index = [k % 12 for k in range(len(centres))]
    return np.asarray(centres, dtype=np.int64), np.asarray(index, dtype=np.int64)


def nemo_fld_time_interpolate(records, record_index, centres_s, isecsbc):
    """``fld_read``'s two-record time interpolation at model time ``isecsbc``.

    fldread.f90:244-246::

        ztinta = REAL(isecsbc - nrec(2,ibb),wp) / REAL(nrec(2,iaa) - nrec(2,ibb),wp)
        ztintb = 1. - ztinta
        fnow   = ztintb * fdta(:,:,:,ibb) + ztinta * fdta(:,:,:,iaa)

    The after record is the first centre >= ``isecsbc`` (fld_update,
    fldread.f90:309-361: no update while ``isecsbc <= nrec(2,iaa)``; an update
    selects the first centre > ``isecsbc``) and the before record the one
    preceding it.  ``isecsbc`` is integer seconds held in float64.  A time
    outside ``centres_s`` returns NaN, never a clamped record.
    """
    centres = jnp.asarray(centres_s, dtype=jnp.float64)
    t = jnp.asarray(isecsbc, dtype=jnp.float64)
    ia = jnp.searchsorted(centres, t, side="left")
    inside = (ia > 0) & (ia < centres.shape[0])
    ia = jnp.clip(ia, 1, centres.shape[0] - 1)
    ib = ia - 1
    ztinta = (t - centres[ib]) / (centres[ia] - centres[ib])
    ztinta = jnp.where(inside, ztinta, jnp.nan)
    ztintb = 1.0 - ztinta
    index = jnp.asarray(record_index)
    rec = jnp.asarray(records, dtype=jnp.float64)
    return _sr(_sr(_sr(ztintb) * rec[index[ib]])
               + _sr(_sr(ztinta) * rec[index[ia]]))


__all__ = (
    "decode_source_indices",
    "nemo_clim_monthly_record_centres",
    "nemo_fld_time_interpolate",
    "nemo_fld_interp",
    "nemo_t_rotation_from_domain",
    "rotate_en_to_ij",
)
