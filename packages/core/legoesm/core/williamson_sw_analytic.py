"""Analytic Williamson (1992) shallow-water initial states, in
GEOGRAPHIC (east/north) components, shared by every grid.

One home for these formulas.  Grid-specific work -- projecting onto a
C-grid, a D-grid covariant edge basis, an MPAS dual mesh, or a spectral
basis -- stays with each caller; only the analytic field of (lon, lat)
lives here.

Covers Williamson case 2 (solid body), case 5 (flow over an isolated
mountain) and case 6 (Rossby-Haurwitz wave 4).  Before this module those
fields were written out separately in
``tests/atmosphere/shallow_water/test_cases/williamson.py``,
``.../williamson_latlon.py``, ``.../williamson_mpas.py``,
``tests/test_cases/williamson_extended.py`` and
``tests/test_cases/dcmip2008/rossby_haurwitz_6_0.py``, and the
FV3-native duo lane was about to add another copy.

★ TWO OF THE COPIES DISAGREED WITH THE OTHERS, so consolidating them
changed numbers.  Both defects and their evidence are documented on the
function that now owns them: the case-6 height's ``B`` coefficient
(``cos^(R-1)`` in all three copies, ``cos^R`` in both authorities) and
the case-5 mountain radius (great-circle in the lat-lon and MPAS copies,
clipped-planar in the cubed-sphere copy and in both authorities).

★ DEFECT FIXED IN THE MOVE, and it changes numbers.  All three previous
copies wrote the height's wave coefficient as ``B ~ cos^(R-1)(lat)``.
Both authorities say ``cos^R``:

  - Williamson et al. (1992) Eq. 145;
  - the pinned FV3 duo oracle, ``tools/test_cases.F90:1226``:
        B = (2.*(omega+omg)*rk / ((r+1)*(r+2))) * (COS(...)**r) * ...

and the structure of the case's own streamfunction agrees with them:
``psi`` carries the wave as ``a^2 K cos^R(lat) sin(lat) cos(R lon)``, so
the balanced height's wave part must scale as ``cos^R`` too.  At R = 4
the old form used ``cos^3`` where ``cos^4`` belongs -- a real change to
the initial condition of every W6 run, not a refactor.

Constants are ARGUMENTS, never module globals: the FV3-native lane must
run on the oracle's GFS ``radius``/``omega``/``Grav``
(``fv3_native_gridstruct.FV3_RADIUS_M`` etc.), while the lat-lon, MPAS
and spectral lanes use ``legoesm.constants``.  Mixing the two flavours
is a ~3e-5 relative shift in radius alone, which is far above any
parity tolerance this repo works at.

``xp`` selects the array module so the same code serves NumPy callers
(the FV3-native duo lattice) and ``jax.numpy`` callers (the traced
shallow-water test cases) without a second implementation.
"""

from __future__ import annotations

import numpy as _np

# --- Solid-body rotation, Williamson et al. (1992) cases 2 and 5 ---
# (test_cases.F90 case(2) and case(5):1173-1209)
W2_U0_ROTATION_DAYS = 12.0        # coeff-ok: Williamson 1992 case-2 period
W2_GH0 = 2.94e4                   # coeff-ok: Williamson 1992 case-2 gh0
W5_UBAR_MS = 20.0                 # coeff-ok: test_cases.F90:1175
W5_H0_M = 5960.0                  # coeff-ok: test_cases.F90:1176 gh0/Grav
W5_MOUNTAIN_RADIUS_RAD = _np.pi / 9.0    # coeff-ok: test_cases.F90:1178
W5_MOUNTAIN_HEIGHT_M = 2000.0            # coeff-ok: test_cases.F90:1187

# ★ THE MOUNTAIN CENTRE IS NOT AGREED BETWEEN THE TWO AUTHORITIES, so it
# is an explicit argument with two named defaults rather than a literal:
#   Williamson et al. (1992) case 5 puts it at lambda_c = 3*pi/2 (270E);
#   the pinned FV3 oracle puts it at pi/2 (90E), test_cases.F90:1179.
# Both use theta_c = pi/6 (30N).  Which one is correct depends on which
# reference a run is being scored against -- an FV3-native duo run must
# use the FV3 centre, a generic Williamson convergence study the
# Williamson one.  Picking silently is how a cross-grid comparison
# becomes a confound.
W5_CENTER_WILLIAMSON = (3.0 * _np.pi / 2.0, _np.pi / 6.0)
W5_CENTER_FV3 = (_np.pi / 2.0, _np.pi / 6.0)

# --- Rossby-Haurwitz wave 4, Williamson et al. (1992) test case 6 ---
# (test_cases.F90:1216-1218 -- the oracle keeps omg and rk as two
#  symbols with the same value; they are kept separate here so a future
#  deck that splits them does not need this file edited.)
RH4_WAVENUMBER = 4.0
RH4_OMEGA_WAVE_HZ = 7.848e-6      # coeff-ok: Williamson 1992 case-6 omega
RH4_K_HZ = 7.848e-6               # coeff-ok: Williamson 1992 case-6 K
RH4_MEAN_DEPTH_M = 8.0e3          # coeff-ok: Williamson 1992 case-6 h0


def solid_body_rotation_speed(radius: float,
                              rotation_days: float = W2_U0_ROTATION_DAYS
                              ) -> float:
    """``Ubar = 2 pi a / (12 days)`` (``test_cases.F90`` case 2)."""
    return 2.0 * _np.pi * radius / (rotation_days * 86400.0)


def solid_body_winds(lon, lat, *, u0: float, alpha: float = 0.0, xp=_np):
    """Solid-body rotation ``(u_east, v_north)`` at rotation angle
    ``alpha``, shared by Williamson cases 2 and 5.

    ``test_cases.F90``'s ``init_winds`` case-2 projection; ``alpha = 0``
    collapses to ``u = u0 cos(lat)``, ``v = 0``.
    """
    return (u0 * (xp.cos(lat) * xp.cos(alpha)
                  + xp.sin(lat) * xp.cos(lon) * xp.sin(alpha)),
            -u0 * xp.sin(lon) * xp.sin(alpha))


def solid_body_geopotential(lon, lat, *, radius: float, omega: float,
                            u0: float, gh0: float, alpha: float = 0.0,
                            xp=_np):
    """Geostrophically balanced ``g*h`` of the solid-body state.

    ``gh = gh0 - (a Om u0 + u0^2/2) S^2`` with
    ``S = -cos(lon) cos(lat) sin(alpha) + sin(lat) cos(alpha)``
    (``test_cases.F90:1203-1205``, and the case-2 sibling).  Surface
    geopotential is NOT subtracted here -- case 5's ``- phis`` is the
    caller's business, since ``h`` and ``h_free`` differ by exactly that.
    """
    s = (-xp.cos(lon) * xp.cos(lat) * xp.sin(alpha)
         + xp.sin(lat) * xp.cos(alpha))
    # The oracle's operation tree, both terms (:1203-1205):
    # (Ubar*Ubar)/2. not 0.5*u0*u0, and S**2 not s*s (codex r1 #1: the
    # s*s form is one fp32 ULP off the ** tree at phi=0.8517...).
    return gh0 - (radius * omega * u0 + (u0 * u0) / 2.0) * s ** 2


def williamson_5_mountain_height(lon, lat, *,
                                 center=W5_CENTER_WILLIAMSON,
                                 r0: float = W5_MOUNTAIN_RADIUS_RAD,
                                 h_s0: float = W5_MOUNTAIN_HEIGHT_M,
                                 xp=_np):
    """Isolated-mountain surface height [m], ``h_s = h_s0 (1 - r/r0)``.

    ``r`` is the PLANAR (lon, lat)-plane distance, CLIPPED at ``r0``::

        r = sqrt(min(r0^2, dlon^2 + dlat^2))

    which is what both authorities specify -- Williamson et al. (1992)
    case 5 (``r^2 = min(R^2, (lam-lam_c)^2 + (th-th_c)^2)``) and
    ``test_cases.F90:1185-1187``.  The clip is self-zeroing: at
    ``r = r0`` the height is exactly 0 and stays 0 outside, so no
    ``where(r < r0, ..., 0)`` mask is needed.

    ★ NOT a great-circle distance.  ``williamson_mpas.py`` previously
    used ``arccos(sin th_c sin th + cos th_c cos th cos(lam-lam_c))``,
    which is a DIFFERENT mountain (it is isotropic on the sphere, while
    the planar form is stretched in longitude by ``1/cos(lat)``, i.e.
    ~15% at 30N).  Any W5 comparison across those two grids was
    measuring the mountain difference as well as the scheme difference.

    ``dlon`` is wrapped to the nearest periodic image first, so callers
    storing longitude in ``[-pi, pi)`` and callers storing it in
    ``[0, 2pi)`` get the same mountain.
    """
    lon_c, lat_c = center
    dlon = xp.mod(lon - lon_c + _np.pi, 2.0 * _np.pi) - _np.pi
    dlat = lat - lat_c
    r = xp.sqrt(xp.minimum(r0 * r0, dlon * dlon + dlat * dlat))
    return h_s0 * (1.0 - r / r0)


def rossby_haurwitz_4_geopotential(lon, lat, *, radius: float,
                                   omega: float, gh0: float,
                                   wavenumber: float = RH4_WAVENUMBER,
                                   omega_wave: float = RH4_OMEGA_WAVE_HZ,
                                   k_wave: float = RH4_K_HZ,
                                   xp=_np):
    """Geopotential ``g*h`` of the Rossby-Haurwitz wave-4 state.

    Williamson et al. (1992) Eq. 145; ``tools/test_cases.F90:1222-1230``.
    Returns ``g*h`` (m^2/s^2), which is what FV3 stores in ``delp`` on
    the shallow-water lane; divide by ``g`` for a depth in metres.

    ``gh0`` is the resting geopotential, i.e. ``g * h0`` with
    ``h0 = 8000 m`` -- passed in rather than computed here so the caller
    owns which ``g`` flavour is in play (:1215 uses the oracle's ``Grav``).

    The ``A`` term is written in the oracle's LITERAL factored form,
    ``cos^{2R} * cos^{-2}``, not the algebraically equal ``cos^{2R-2}``,
    so the rounding matches the Fortran.

    RETRACTED (measured 2026-08-08): an earlier version of this docstring
    claimed the factored form yields ``0 * inf = nan`` at an exact pole,
    where the epsilon-regularised copies it replaced returned a plausible
    number.  That is FALSE in float64 -- ``cos(pi/2)`` evaluates to
    6.1e-17, never to 0.0, so ``cos^{-2}`` is 2.7e32 and the product is a
    perfectly finite 5.3e-99.  Both factorings return the same finite
    value at the representable pole.  The removal of the ``+ 1e-30``
    guards is therefore a fidelity change only, NOT a new failure signal,
    and callers on a pole-containing grid get no warning from this
    function.
    """
    r = wavenumber
    w = omega_wave
    k = k_wave
    c = xp.cos(lat)
    a_t = (0.5 * w * (2.0 * omega + w) * c ** 2
           + 0.25 * k * k * c ** (r + r)
           * ((r + 1.0) * c ** 2 + (2.0 * r * r - r - 2.0)
              - 2.0 * (r * r) * c ** (-2.0)))
    # cos^R -- NOT cos^(R-1); see the module docstring.
    b_t = ((2.0 * (omega + w) * k / ((r + 1.0) * (r + 2.0)))
           * c ** r
           * ((r * r + 2.0 * r + 2.0) - ((r + 1.0) * c) ** 2))
    c_t = (0.25 * k * k * c ** (2.0 * r)
           * ((r + 1.0) * c ** 2 - (r + 2.0)))
    return gh0 + radius * radius * (a_t
                                    + b_t * xp.cos(r * lon)
                                    + c_t * xp.cos(2.0 * r * lon))


def rossby_haurwitz_4_winds(lon, lat, *, radius: float,
                            wavenumber: float = RH4_WAVENUMBER,
                            omega_wave: float = RH4_OMEGA_WAVE_HZ,
                            k_wave: float = RH4_K_HZ,
                            xp=_np):
    """``(u_east, v_north)`` in m/s.

    Williamson et al. (1992) Eq. 146-147;
    ``tools/test_cases.F90:1241-1243`` (and the identical pair at
    :1254-1256, evaluated on the other edge midpoint).

    This is exactly ``curl`` of the case's streamfunction

        psi = -a^2 w sin(lat) + a^2 K cos^R(lat) sin(lat) cos(R lon)

    i.e. ``u = -(1/a) dpsi/dlat``, ``v = (1/(a cos lat)) dpsi/dlon`` --
    the relation the unit test certifies numerically.
    """
    r = wavenumber
    w = omega_wave
    k = k_wave
    c = xp.cos(lat)
    s = xp.sin(lat)
    u_east = (radius * w * c
              + radius * k * c ** (r - 1.0)
              * (r * s * s - c * c) * xp.cos(r * lon))
    v_north = -radius * k * r * s * xp.sin(r * lon) * c ** (r - 1.0)
    return u_east, v_north
