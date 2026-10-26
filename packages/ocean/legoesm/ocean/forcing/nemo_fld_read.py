"""NEMO ``fld_read`` weighted forcing operators for native ORCA cards.

The host loads NetCDF records and the NEMO-generated ``srcNN``/``wgtNN``
tables.  The numerical map itself is pure JAX and follows
``fldread.F90:1120-1241,1349-1579``: four weighted source points, followed by
the three four-term gradient groups for bicubic weights.  It is shared by the
ORCA2 fidelity card and the future ORCA1 card; neither card owns a second
regridding formula.

Arrays use NetCDF/Python ``(j, i)`` ordering.  NEMO's source indices are
one-based Fortran-flat indices (``i`` fastest); :func:`decode_source_indices`
converts them once on the host.  Time record selection remains a separate
operation because it is deck metadata; :func:`nemo_fld_time_interp` is the
``ln_tint`` blend applied once the two record times are known.
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


def nemo_fld_time_interp(fdta_before, fdta_after, isecsbc, nrec_sec_before,
                         nrec_sec_after):
    """Literal ``fld_read`` temporal interpolation (``ln_tint = T``).

    fldread.F90:244-246: ``ztinta = REAL(isecsbc - nrec(2,ibb)) /
    REAL(nrec(2,iaa) - nrec(2,ibb))``, ``ztintb = 1. - ztinta``,
    ``fnow = ztintb*fdta(ibb) + ztinta*fdta(iaa)``.  The record times are
    integer seconds; their differences are formed in integer arithmetic and
    converted exactly.  A yearly climatology (``freqh = -12``) holds the same
    field in both slots, and the blend can still move it by one ULP.
    """
    before = jnp.asarray(nrec_sec_before, dtype=jnp.int64)
    num = (jnp.asarray(isecsbc, dtype=jnp.int64) - before).astype(jnp.float64)
    den = (jnp.asarray(nrec_sec_after, dtype=jnp.int64) - before).astype(
        jnp.float64)
    ztinta = _sr(_sr(num) / _sr(den))
    ztintb = _sr(1.0 - ztinta)
    fb = jnp.asarray(fdta_before, dtype=jnp.float64)
    fa = jnp.asarray(fdta_after, dtype=jnp.float64)
    return _sr(_sr(_sr(ztintb) * _sr(fb)) + _sr(_sr(ztinta) * _sr(fa)))


def _nemo_nint(value):
    """Fortran ``NINT`` (ties away from zero), returned as int64."""
    value = _sr(jnp.asarray(value, dtype=jnp.float64))
    rounded = jnp.where(
        value >= 0.0, jnp.floor(value + 0.5), jnp.ceil(value - 0.5)
    )
    return rounded.astype(jnp.int64)


def nemo_fld_model_time(
    *, nsec_year, nsec_1jan000, forcing_stride, timestep_seconds, time_offset
):
    """NEMO ``fld_read`` model time at the middle of a forcing interval.

    This is ``fldread.f90:195-206``'s no-``kit`` arm.  Every operand is
    explicit because calendar state and the component offset are physical
    inputs, not safe defaults.
    """
    stride = _sr(jnp.asarray(forcing_stride, dtype=jnp.float64))
    half_span = _sr(_sr(0.5) * _sr(stride - _sr(1.0)))
    offset_span = _sr(half_span + _sr(time_offset))
    seconds = _sr(offset_span * _sr(timestep_seconds))
    rounded = _nemo_nint(seconds)
    return (
        jnp.asarray(nsec_year, dtype=jnp.int64)
        + jnp.asarray(nsec_1jan000, dtype=jnp.int64)
        + rounded
    )


def _trunc_divide_by_two(value):
    value = jnp.asarray(value, dtype=jnp.int64)
    return jnp.where(value >= 0, value // 2, -((-value) // 2))


def nemo_fld_record_centres(record_edges, rec_shft):
    """Construct interpolated-record centres from integer edge times.

    Literal ``fldread.f90:932-936`` arithmetic, including truncating integer
    division, Fortran ``MOD`` for negative edges, and ``NINT(rec_shft*width)``.
    ``record_edges`` includes both ends and therefore returns one fewer centre.
    """
    edge = jnp.asarray(record_edges, dtype=jnp.int64)
    if edge.ndim != 1 or edge.shape[0] < 2:
        raise ValueError("record_edges must be a one-dimensional edge array")
    left = edge[:-1]
    right = edge[1:]
    shift = _nemo_nint(
        _sr(_sr(rec_shft) * _sr((right[0] - left[0]).astype(jnp.float64)))
    )
    left_half = _trunc_divide_by_two(left)
    right_half = _trunc_divide_by_two(right)
    left_mod = left - _trunc_divide_by_two(left) * 2
    right_mod = right - _trunc_divide_by_two(right) * 2
    odd_carry = jnp.maximum(left_mod, right_mod)
    return left_half + right_half + odd_carry + shift


def nemo_fld_select_bracket(model_seconds, record_centres):
    """Select NEMO's retained-before and current-after record indices.

    The caller supplies previous/current/next-file centres after NEMO's file
    policy has assembled them.  It must include a record on each side; file
    cycling/extrapolation remains explicit loader policy, not a numeric default.
    ``fldread.f90:309`` advances the pair only for ``model_seconds > after``;
    equality therefore retains the record as the after slot.
    """
    centres = np.asarray(record_centres, dtype=np.int64)
    if centres.ndim != 1 or centres.shape[0] < 2:
        raise ValueError("record_centres must contain at least two records")
    if np.any(np.diff(centres) <= 0):
        raise ValueError("record_centres must be strictly increasing")
    after = int(np.searchsorted(centres, int(model_seconds), side="left"))
    if after == 0 or after == centres.size:
        raise ValueError("record_centres do not bracket model_seconds")
    before = after - 1
    return before, after


def nemo_fld_time_interpolate(
    before_record, after_record, *, model_seconds, before_seconds, after_seconds
):
    """Literal S0-F temporal blend of already spatially mapped records.

    Returns ``(fnow, after_weight, before_weight)`` in NEMO assignment order.
    Source-rounding barriers materialize every intermediate for the CPU-fp64
    oracle arm.
    """
    before = jnp.asarray(before_record, dtype=jnp.float64)
    after = jnp.asarray(after_record, dtype=jnp.float64)
    if before.shape != after.shape:
        raise ValueError("before_record and after_record must share a shape")
    elapsed_i = jnp.asarray(model_seconds, dtype=jnp.int64) - jnp.asarray(
        before_seconds, dtype=jnp.int64
    )
    width_i = jnp.asarray(after_seconds, dtype=jnp.int64) - jnp.asarray(
        before_seconds, dtype=jnp.int64
    )
    elapsed = _sr(elapsed_i.astype(jnp.float64))
    width = _sr(width_i.astype(jnp.float64))
    after_weight = _sr(elapsed / width)
    before_weight = _sr(_sr(1.0) - after_weight)
    before_term = _sr(before_weight * _sr(before))
    after_term = _sr(after_weight * _sr(after))
    fnow = _sr(before_term + after_term)
    return fnow, after_weight, before_weight


__all__ = (
    "decode_source_indices",
    "nemo_fld_time_interp",
    "nemo_fld_interp",
    "nemo_fld_model_time",
    "nemo_fld_record_centres",
    "nemo_fld_select_bracket",
    "nemo_fld_time_interpolate",
    "nemo_t_rotation_from_domain",
    "rotate_en_to_ij",
)
