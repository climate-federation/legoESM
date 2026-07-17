"""Astronomical (equilibrium) tidal forcing — a barotropic body force on momentum.

This is the barotropic **tide GENERATOR** (the astronomical tide-generating
potential expressed as an equivalent equilibrium sea-surface elevation, which
drives the momentum equation).  It is complementary to — and distinct from — the
existing tidal *mixing* schemes: the Jayne & St-Laurent internal-tide diapycnal
diffusivity (``physics/vertical_mixing/tidal.py``) and the internal-wave mixing
(``physics/vertical_mixing/internal_wave_mixing.py``) parameterise where tidal
ENERGY dissipates; this module supplies the FORCE that raises the barotropic
tide in the first place.  Closes an audit gap vs Oceananigans / ClimaOcean,
which ship an equilibrium-tide forcing.

Physics
-------
The equilibrium tide is the sea-surface elevation the ocean would take in
instantaneous hydrostatic balance with the tide-generating potential Phi_tide:

    g * eta_eq = -Phi_tide   ->   eta_eq = -Phi_tide / g,

so eta_eq is HIGH toward the sub-lunar / sub-solar point (the tidal bulge).  The
horizontal tractive acceleration on the water column is then

    a_tide = -grad_h(Phi_tide) = +g * grad_h(eta_eq_eff)      [m/s^2]

i.e. the force points UP the equilibrium-elevation gradient, TOWARD the bulge
(see the SIGN CONVENTION note in ``__physics_contract__`` and ``tidal_acceleration``).
Equivalently, in the momentum equation the pressure-gradient term becomes
``-g grad(eta - eta_eq_eff)`` (the standard NEMO / MOM6 / ROMS form), so the
tidal contribution is ``+g grad(eta_eq_eff)``.

The equilibrium elevation is a sum over tidal constituents c:

    eta_eq(phi, lambda, t) = sum_c  A_c * G_{n_c}(phi) * cos(omega_c t + chi_c + n_c lambda)

with the degree-2 spherical-harmonic latitude structure per species n
(x eastward = +lambda east, y northward = +phi):

    n = 0  long-period  (Mf, Mm):      G_0(phi) = (3 sin^2(phi) - 1) / 2 ,  no lon dependence
    n = 1  diurnal      (K1,O1,P1,Q1): G_1(phi) = sin(2 phi) ,              phase + 1*lambda
    n = 2  semidiurnal  (M2,S2,N2,K2): G_2(phi) = cos^2(phi) ,              phase + 2*lambda

(the un-normalised associated Legendre functions P_2^n of degree 2).  With
constituent phases chi_c = 0 the bulge sits at lambda = 0 at t = 0, and the
argument ``omega t + n lambda`` gives a WESTWARD-propagating bulge (constant
phase => d(lambda)/dt = -omega/n < 0), tracking the apparent westward motion of
the Moon / Sun.

Effective elevation (elastic Earth + self-attraction & loading)
---------------------------------------------------------------
The astronomical forcing is reduced by two standard scalar factors:

* ``love_factor`` = (1 + k_2 - h_2) ~ 0.69 — the body-tide (elastic-Earth) Love
  combination (k_2 ~ 0.302, h_2 ~ 0.612): the solid Earth deforms under the
  tidal potential, adding its own potential (k_2) and moving the reference
  surface (h_2).
* ``beta_sal`` ~ 0.94 — the Accad & Pekeris (1978) SCALAR self-attraction &
  loading approximation.  A rigorous SAL term is a spherical-harmonic
  convolution of the model's own eta; the scalar form replaces it with a single
  multiplicative factor on the forcing.  This is the standard first-order
  approximation used by many barotropic tide models.

    eta_eq_eff = beta_sal * love_factor * amplitude_scale * eta_eq

Both factors (and a spin-up / sensitivity ``amplitude_scale``) are config-tunable.

DOCUMENTED SIMPLIFICATIONS (see ``__physics_contract__``)
--------------------------------------------------------
1. SIMPLIFIED ASTRONOMICAL ARGUMENT.  The per-constituent phase ``chi_c`` is a
   constant (default 0), NOT the true Doodson/astronomical argument V_0 + u at a
   reference epoch.  The angular FREQUENCIES, EQUILIBRIUM AMPLITUDES, spatial
   GEOMETRY and RELATIVE constituent phasing are physical; only the ABSOLUTE
   phase relative to a real calendar date is a simplification.  Upgrade path:
   populate ``phase_rad`` from the Doodson numbers at a chosen epoch.
2. NODAL MODULATION (18.6-yr f, u factors) is omitted (f = 1, u = 0).
3. SCALAR SAL (item above) — no eta-convolution.
4. AMPLITUDE CONVENTION — the tabulated ``amplitude_m`` are the standard
   equilibrium (Cartwright-Tayler-Edden 1971/1973; Doodson 1921) constituent
   amplitudes in the max-elevation normalisation matching the G_n above
   (G_2 max = 1 at the equator, G_1 max = 1 at +/-45 deg, G_0 max = 1 at the poles).
5. PHASE PRECISION — the fast time-phase ``omega*t`` is reduced modulo 2*pi in
   ``t``'s native precision before casting to the field dtype, so a float32 field
   keeps full phase for typical runs. Over LONG (multi-year, t ~ 1e8 s)
   integrations a float32 ``t`` still loses ~1e-3 rad of phase (~0.06 deg) BEFORE
   the reduction; enable ``JAX_ENABLE_X64`` (the ocean sci-run default), under
   which the model wire feeds a float64 ``t`` and the phase is exact.

References
----------
Cartwright, D. E., & Tayler, R. J. (1971). New computations of the tide-generating
potential. *Geophys. J. R. astr. Soc.*, 23, 45-73.
Cartwright, D. E., & Edden, A. C. (1973). Corrected tables of tidal harmonics.
*Geophys. J. R. astr. Soc.*, 33, 253-264.
Accad, Y., & Pekeris, C. L. (1978). Solution of the tidal equations for the M2
and S2 tides in the world oceans from a knowledge of the tidal potential alone.
*Phil. Trans. R. Soc. Lond. A*, 290, 235-266.
Pugh, D., & Woodworth, P. (2014). *Sea-Level Science*. Cambridge Univ. Press.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.operators_latlon_cgrid import (
    gradient_x_cgrid,
    gradient_y_cgrid,
)

__physics_contract__ = {
    "summary": (
        "Astronomical equilibrium tidal forcing: the barotropic tide-generating "
        "potential expressed as an equivalent equilibrium sea-surface elevation "
        "eta_eq_eff, whose gradient is an OPT-IN body force a = +g grad(eta_eq_eff) "
        "on the horizontal (barotropic) momentum equation. Sums the major "
        "semidiurnal (M2,S2,N2,K2), diurnal (K1,O1,P1,Q1) and long-period (Mf,Mm) "
        "constituents with elastic-Earth (Love) and scalar SAL corrections."
    ),
    "inputs": {
        "lat": "rad", "lon": "rad", "t": "s",
        "love_factor": "1", "beta_sal": "1", "amplitude_scale": "1",
    },
    "outputs": {"eta_eq_eff": "m", "a_x": "m/s^2", "a_y": "m/s^2"},
    "sign_convention": (
        "x eastward (+lon east), y northward (+lat north). eta_eq_eff is the "
        "equilibrium ELEVATION, HIGH toward the sub-lunar/sub-solar bulge "
        "(positive amplitudes with G_2=cos^2, at chi=0,t=0,lambda=0 the bulge is "
        "at the equator/lambda=0). The tractive acceleration points TOWARD the "
        "bulge (up the eta_eq gradient): a = +g*grad(eta_eq_eff) [a_x=+g*deta/dx "
        "at u-faces, a_y=+g*deta/dy at v-faces], NOT -g*grad. In the momentum eqn "
        "this is the +g*grad(eta_eq_eff) piece of the standard pressure-gradient "
        "form -g*grad(eta - eta_eq_eff) (NEMO/MOM6/ROMS). Disabled -> identity."
    ),
    # An EXTERNAL astronomical momentum forcing (like wind stress): it injects
    # tidal energy into the ocean and conserves nothing INTERNALLY. It is, however,
    # globally NEUTRAL: the degree-2 potential has no degree-0 part so the global
    # area-weighted mean of eta_eq is ~0 (no net mass/barotropic source), and
    # integral(grad(eta_eq)) over the closed sphere = 0 (no net momentum injection).
    # Those are properties of the forcing FIELD, not interior conservation laws of
    # the scheme -> "none".
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Cartwright-Tayler-Edden 1971/1973; Accad & Pekeris 1978; "
                 "Doodson 1921; Pugh & Woodworth 2014",
    "idealized_test": (
        "tests/ocean/unit/test_tidal_forcing.py — each constituent omega matches "
        "its known period (M2=12.4206h,S2=12h,K1=23.9345h,O1=25.819h); "
        "semidiurnal geometry peaks at the equator (cos^2), diurnal at +/-45 deg "
        "(sin2phi), long-period ~ (3sin^2-1)/2; global area-weighted mean ~0; "
        "a=+g grad(eta) points toward the equilibrium high; grad wrt t and a "
        "config amplitude is finite; disabled=byte-identical."
    ),
}

__param_spec__ = {
    "TidalForcingConfig": {
        "scheme_key": "ocean.tidal_forcing",
        "excluded": {},
        "params": {
            # (1 + k2 - h2) elastic-Earth body-tide reduction of the astronomical
            # forcing. Physical value ~0.69; loosely bounded around it.
            "love_factor": {
                "units": "1", "bounds": (0.6, 0.75), "tunable_tier": 2,
                "transform": "sigmoid", "category": "tidal_forcing",
                "reference": "Love numbers k2~0.302, h2~0.612 (1+k2-h2~0.69)",
                "shape": None,
            },
            # Accad-Pekeris scalar self-attraction & loading factor on the forcing.
            "beta_sal": {
                "units": "1", "bounds": (0.85, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "tidal_forcing",
                "reference": "Accad & Pekeris 1978 scalar SAL approximation",
                "shape": None,
            },
            # Global forcing multiplier (spin-up ramp / sensitivity). Default 1.
            "amplitude_scale": {
                "units": "1", "bounds": (0.0, 2.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "tidal_forcing",
                "reference": "spin-up ramp / sensitivity knob (not a physical constant)",
                "shape": None,
            },
        },
    },
}

# --- Species codes (degree-2 spherical-harmonic order n = number of cycles per
#     revolution in longitude; == the longitude multiplier nu) ---
_SPECIES_LONG_PERIOD = 0    # G_0(phi) = (3 sin^2 phi - 1)/2, no lon dependence
_SPECIES_DIURNAL = 1        # G_1(phi) = sin(2 phi),          phase + 1*lambda
_SPECIES_SEMIDIURNAL = 2    # G_2(phi) = cos^2(phi),          phase + 2*lambda


class _TidalConstituent(NamedTuple):
    """One equilibrium-tide constituent (module-level table datum, NOT a config).

    ``omega_rad_s`` and ``period_hours`` are stored as INDEPENDENT literals and
    cross-checked in the test (2*pi/omega == period*3600) so a transcription
    typo in either is caught.
    """
    name: str
    omega_rad_s: float    # angular frequency [rad/s]
    period_hours: float   # period [h] (provenance / cross-check)
    amplitude_m: float    # equilibrium amplitude A_c [m] (CTE 1971/1973)
    species: int          # 0 long-period, 1 diurnal, 2 semidiurnal
    phase_rad: float      # reference (Greenwich) phase chi_c [rad] (simplified: 0)


# ==============================================================================
# Equilibrium tidal constituents (Cartwright-Tayler-Edden 1971/1973; Doodson 1921).
#
# omega computed from the canonical constituent periods (Schureman 1958 / IHO
# standard) and verified to 2*pi/omega == period in the unit test. Amplitudes are
# the standard equilibrium (max-elevation) values in the G_n normalisation above;
# their absolute magnitudes carry a convention uncertainty (flagged for review),
# but the frequencies, species geometry and zero global-mean are exact.
# ==============================================================================
_CONSTITUENTS: tuple[_TidalConstituent, ...] = (
    # --- Semidiurnal (species 2, G_2 = cos^2 phi) ---
    _TidalConstituent("M2", 1.405189027e-4, 12.4206012, 0.242334, _SPECIES_SEMIDIURNAL, 0.0),
    _TidalConstituent("S2", 1.454441043e-4, 12.0000000, 0.112841, _SPECIES_SEMIDIURNAL, 0.0),
    _TidalConstituent("N2", 1.378797076e-4, 12.6583475, 0.046398, _SPECIES_SEMIDIURNAL, 0.0),
    _TidalConstituent("K2", 1.458423012e-4, 11.9672361, 0.030704, _SPECIES_SEMIDIURNAL, 0.0),
    # --- Diurnal (species 1, G_1 = sin 2 phi) ---
    _TidalConstituent("K1", 7.292115822e-5, 23.9344697, 0.141565, _SPECIES_DIURNAL, 0.0),
    _TidalConstituent("O1", 6.759774406e-5, 25.8193417, 0.100514, _SPECIES_DIURNAL, 0.0),
    _TidalConstituent("P1", 7.252294586e-5, 24.0658902, 0.046843, _SPECIES_DIURNAL, 0.0),
    _TidalConstituent("Q1", 6.495854106e-5, 26.8683567, 0.019256, _SPECIES_DIURNAL, 0.0),
    # --- Long-period (species 0, G_0 = (3 sin^2 phi - 1)/2) ---
    _TidalConstituent("Mf", 5.323398946e-6, 327.8599387, 0.041742, _SPECIES_LONG_PERIOD, 0.0),
    _TidalConstituent("Mm", 2.639195197e-6, 661.3111655, 0.022026, _SPECIES_LONG_PERIOD, 0.0),
)


class TidalForcingConfig(NamedTuple):
    """Astronomical (equilibrium) tidal-forcing configuration.

    Disabled by default (``enabled=False``) so existing ocean runs are
    bit-exact.  The three ``include_*`` species selectors and the scalar
    Love / SAL / amplitude factors are the tunable surface.
    """
    # --- feature gate (static Python bool; a disabled run is byte-identical) ---
    enabled: bool = False
    # --- scalar corrections (literal defaults; see __param_spec__ for bounds) ---
    love_factor: float = 0.69        # (1 + k2 - h2), body-tide elastic-Earth [-]
    beta_sal: float = 0.94           # Accad & Pekeris 1978 scalar SAL [-]
    amplitude_scale: float = 1.0     # spin-up / sensitivity multiplier [-]
    # --- constituent-species subset selector (static Python bools) ---
    include_semidiurnal: bool = True    # M2, S2, N2, K2
    include_diurnal: bool = True        # K1, O1, P1, Q1
    include_long_period: bool = True    # Mf, Mm


def _species_included(species: int, config: TidalForcingConfig) -> bool:
    """Static (Python-bool) species gate — compile-time constant, no traced branch."""
    if species == _SPECIES_SEMIDIURNAL:
        return config.include_semidiurnal
    if species == _SPECIES_DIURNAL:
        return config.include_diurnal
    if species == _SPECIES_LONG_PERIOD:
        return config.include_long_period
    # Table integrity guard: species codes are fixed in _CONSTITUENTS, so an
    # unknown code means a corrupted table, not a user typo -> fail loudly.
    raise ValueError(f"unknown tidal species code {species!r}")


def _species_geometry(species: int, sin_lat: jnp.ndarray, lat: jnp.ndarray) -> jnp.ndarray:
    """Degree-2 latitude structure G_n(phi) for the given species (static ``species``).

    G_0 = (3 sin^2 - 1)/2 (long-period), G_1 = sin(2 phi) (diurnal),
    G_2 = cos^2 = 1 - sin^2 (semidiurnal).
    """
    if species == _SPECIES_SEMIDIURNAL:
        return 1.0 - sin_lat * sin_lat            # cos^2(phi)
    if species == _SPECIES_DIURNAL:
        return jnp.sin(2.0 * lat)                 # sin(2 phi)
    if species == _SPECIES_LONG_PERIOD:
        return (3.0 * sin_lat * sin_lat - 1.0) / 2.0
    raise ValueError(f"unknown tidal species code {species!r}")


def equilibrium_tide_elevation(lat, lon, t_seconds, config: TidalForcingConfig | None = None):
    """Effective equilibrium tidal elevation eta_eq_eff [m] (pure, vectorised).

    Sums the enabled constituents and applies the scalar Love / SAL / amplitude
    corrections.  Pure function of the coordinate arrays, time and (static)
    config — differentiable in ``t_seconds`` and in the scalar config factors.

    Parameters
    ----------
    lat, lon : array-like [rad]
        Latitude and longitude (RADIANS) at the evaluation points.  Any broadcast
        shape; for the C-grid pressure gradient these are the TRACER-point
        coordinates ``grid.lat2d`` / ``grid.lon2d``.
    t_seconds : scalar [s]
        Elapsed model time.  Passed as an explicit (traceable) argument — never
        captured in a closure — so a per-step changing ``t`` does not freeze at
        compile time (SegmentForcing doctrine).
    config : TidalForcingConfig, optional
        Species selectors + scalar factors.  ``None`` -> defaults (all species,
        standard Love/SAL).  ``enabled`` is NOT consulted here (this pure kernel
        always evaluates); the gate lives in :func:`apply_tidal_forcing`.

    Returns
    -------
    eta_eq_eff : array [m]
        Effective equilibrium elevation, broadcast over ``lat``/``lon``.
    """
    cfg = config if config is not None else TidalForcingConfig()
    lat = jnp.asarray(lat)
    lon = jnp.asarray(lon)
    # Respect the ambient precision (x64 when enabled, else float32 finite-volume).
    dtype = jnp.result_type(lat.dtype, lon.dtype, jnp.float32)
    lat = lat.astype(dtype)
    lon = lon.astype(dtype)
    # Keep t at its NATIVE precision (do NOT downcast to the field dtype): the
    # time-phase omega*t is reduced modulo 2*pi BELOW before being cast, so a
    # float32 field does not lose tidal phase on long integrations (omega*t can
    # reach ~1e5 rad, where float32 has only ~1e-2 rad resolution; the reduced
    # phase in [0, 2*pi) keeps full precision).
    t = jnp.asarray(t_seconds)
    sin_lat = jnp.sin(lat)

    eta = jnp.zeros(jnp.broadcast_shapes(lat.shape, lon.shape), dtype=dtype)
    for c in _CONSTITUENTS:
        if not _species_included(c.species, cfg):
            continue  # static Python skip -> constituent contributes nothing
        geom = _species_geometry(c.species, sin_lat, lat)
        # Argument omega*t + chi + n*lambda. n=species is the longitude multiplier
        # nu (0 for long-period -> no lon dependence). Westward-propagating bulge.
        # Reduce the fast time-phase mod 2*pi in t's own precision, then cast the
        # small [0,2*pi) result to the field dtype and add the longitude phase.
        time_phase = jnp.mod(c.omega_rad_s * t + c.phase_rad, constants.TWO_PI)
        arg = time_phase.astype(dtype) + c.species * lon
        eta = eta + c.amplitude_m * geom * jnp.cos(arg)

    # Effective elevation: elastic Earth (Love) x scalar SAL x amplitude scale.
    scale = cfg.beta_sal * cfg.love_factor * cfg.amplitude_scale
    return scale * eta


def tidal_acceleration(grid, t_seconds, config: TidalForcingConfig, *, g: float = constants.g):
    """Barotropic tidal acceleration (a_x, a_y) [m/s^2] on the lat-lon C-grid.

    Evaluates ``eta_eq_eff`` at TRACER (cell-centre) points and takes its
    horizontal gradient with the CANONICAL C-grid operators
    (:func:`gradient_x_cgrid` / :func:`gradient_y_cgrid`) — the SAME operators the
    model uses for the surface pressure gradient ``-g grad(eta)`` — so the
    discretisation, staggering and masking are consistent with the barotropic
    solver by construction.

    SIGN CONVENTION (x eastward, y northward): the tractive force points TOWARD
    the equilibrium high (sub-lunar bulge), i.e. UP the ``eta_eq_eff`` gradient:

        a_x = +g * d(eta_eq_eff)/dx   at u-faces,  shape (n_lat, n_lon+1)
        a_y = +g * d(eta_eq_eff)/dy   at v-faces,  shape (n_lat+1, n_lon)

    This is the +g*grad(eta_eq_eff) piece of the standard momentum pressure
    gradient -g*grad(eta - eta_eq_eff).  (It is NOT -g*grad(eta_eq): with eta_eq
    high toward the Moon, -g*grad would point AWAY from the bulge — unphysical.)

    Parameters
    ----------
    grid : LatLonGrid
        Provides ``lat2d`` / ``lon2d`` [rad] (tracer points) and the metric the
        gradient operators need.
    t_seconds : scalar [s]
        Elapsed model time (traced per-step argument, not a closure capture).
    config : TidalForcingConfig
        Species selectors + scalar factors.
    g : float, optional
        Gravitational acceleration [m/s^2]; defaults to ``constants.g``.  Pass the
        ocean config's ``g`` at the call site for exact PGF consistency.

    Returns
    -------
    (a_x, a_y) : tuple of arrays [m/s^2]
        Eastward acceleration at u-faces and northward acceleration at v-faces.
    """
    eta_eq = equilibrium_tide_elevation(grid.lat2d, grid.lon2d, t_seconds, config)
    a_x = g * gradient_x_cgrid(eta_eq, grid)   # u-faces (n_lat, n_lon+1)
    a_y = g * gradient_y_cgrid(eta_eq, grid)   # v-faces (n_lat+1, n_lon)
    return a_x, a_y


def apply_tidal_forcing(du_dt, dv_dt, grid, t_seconds, config: TidalForcingConfig,
                        *, g: float = constants.g, u_mask=None, v_mask=None):
    """Add the equilibrium-tide body force to the BAROTROPIC (u, v) tendencies.

    Application hook mirroring ``ocean.coupler.geothermal_apply``: the physics
    core (:func:`tidal_acceleration`) is grid-coupled but state-agnostic; this
    thin wrapper adds it to the barotropic momentum tendencies behind the static
    ``config.enabled`` gate.

    ``du_dt`` / ``dv_dt`` are the BAROTROPIC (2-D) u-/v-tendencies (or the slow
    barotropic forcing ``F_slow_u`` / ``F_slow_v``): shapes ``(n_lat, n_lon+1)``
    and ``(n_lat+1, n_lon)``, matching ``tidal_acceleration``.

    ``config`` is a STATIC (compile-time) argument: ``enabled`` and the species
    selectors gate Python control flow, so pass it closed-over or via
    ``jax.jit(..., static_argnames="config")`` — never as a dynamic tracer.

    Masking: closed/land faces must receive NO tidal acceleration. Pass
    ``u_mask``/``v_mask`` (broadcasting against ``a_x``/``a_y``) to zero them
    here; if omitted the CALLER must mask immediately downstream (the production
    wire adds the tide to ``F_slow_u/v``, which the barotropic substep multiplies
    by ``u_mask``/``v_mask``).

    ``config is None`` / ``config.enabled=False`` / ``t_seconds is None`` ->
    returns the inputs UNCHANGED (same objects) => byte-identical. The
    ``t_seconds`` gate matters: a caller that has no model time cannot evaluate
    an equilibrium tide, and silently substituting t=0 would apply a WRONG,
    frozen tide rather than none.

    The result is cast back to the ``du_dt``/``dv_dt`` dtype. This is
    LOAD-BEARING, not cosmetic: ``tidal_acceleration`` can return float64 under
    x64 (grid geometry built via ``jnp.linspace`` defaults), and promoting the
    barotropic slow forcing would break the ``fori_loop`` carry-type invariant
    in the substep. The wrapper previously omitted the cast while the production
    call sites did it inline -- so anyone wiring this per its own docstring
    would have broken the carry invariant. Owning the cast here is what makes
    the function safe to actually use.
    """
    if config is None or not config.enabled or t_seconds is None:
        # Feature gate on a static Python bool (NOT jnp.where): tide-off traces
        # no tidal ops and returns the caller's exact arrays -> bit-identical.
        return du_dt, dv_dt
    a_x, a_y = tidal_acceleration(grid, t_seconds, config, g=g)
    if u_mask is not None:
        a_x = a_x * u_mask
    if v_mask is not None:
        a_y = a_y * v_mask
    return du_dt + a_x.astype(du_dt.dtype), dv_dt + a_y.astype(dv_dt.dtype)
