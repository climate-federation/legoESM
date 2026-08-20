"""
JAX/Python translation of the shared orbital mechanics module.

Provides solar zenith angle, declination, and Earth orbital parameter
calculations following Berger (1978).

References
----------
Berger, A. 1978. Long-Term Variations of Daily Insolation and
Quaternary Climatic Changes. J. Atmos. Sci. 35:2362-2367.

Berger, A. 1978. A Simple Algorithm to Compute Long-Term Variations
of Daily Insolation. Contribution 18, Institut d'Astronomie et de
Géophysique, Université Catholique de Louvain.

Berger, A., Loutre, M., and Tricot, C. 1993. Insolation and Earth
Orbital Periods. J. Geophys. Res. 98:10341-10362.

Original Fortran module: shr_orb_mod
"""

from __future__ import annotations

import math

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import pi  # pi = rpi

# ---------------------------------------------------------------------------
# Series coefficients for shr_orb_params — Fortran parameter arrays
# ---------------------------------------------------------------------------

# --- Obliquity (47 terms): amplitude (arc s), rate (arc s/yr), phase (deg) ---
_poblen = 47

_obamp = (
    -2462.2214466,
    -857.3232075,
    -629.3231835,
    -414.2804924,
    -311.7632587,
    308.9408604,
    -162.5533601,
    -116.1077911,
    101.1189923,
    -67.6856209,
    24.9079067,
    22.5811241,
    -21.1648355,
    -15.6549876,
    15.3936813,
    14.6660938,
    -11.7273029,
    10.2742696,
    6.4914588,
    5.8539148,
    -5.4872205,
    -5.4290191,
    5.1609570,
    5.0786314,
    -4.0735782,
    3.7227167,
    3.3971932,
    -2.8347004,
    -2.6550721,
    -2.5717867,
    -2.4712188,
    2.4625410,
    2.2464112,
    -2.0755511,
    -1.9713669,
    -1.8813061,
    -1.8468785,
    1.8186742,
    1.7601888,
    -1.5428851,
    1.4738838,
    -1.4593669,
    1.4192259,
    -1.1818980,
    1.1756474,
    -1.1316126,
    1.0896928,
)

_obrate = (
    31.609974,
    32.620504,
    24.172203,
    31.983787,
    44.828336,
    30.973257,
    43.668246,
    32.246691,
    30.599444,
    42.681324,
    43.836462,
    47.439436,
    63.219948,
    64.230478,
    1.010530,
    7.437771,
    55.782177,
    0.373813,
    13.218362,
    62.583231,
    63.593761,
    76.438310,
    45.815258,
    8.448301,
    56.792707,
    49.747842,
    12.058272,
    75.278220,
    65.241008,
    64.604291,
    1.647247,
    7.811584,
    12.207832,
    63.856665,
    56.155990,
    77.448840,
    6.801054,
    62.209418,
    20.656133,
    48.344406,
    55.145460,
    69.000539,
    11.071350,
    74.291298,
    11.047742,
    0.636717,
    12.844549,
)

_obphas = (
    251.9025,
    280.8325,
    128.3057,
    292.7252,
    15.3747,
    263.7951,
    308.4258,
    240.0099,
    222.9725,
    268.7809,
    316.7998,
    319.6024,
    143.8050,
    172.7351,
    28.9300,
    123.5968,
    20.2082,
    40.8226,
    123.4722,
    155.6977,
    184.6277,
    267.2772,
    55.0196,
    152.5268,
    49.1382,
    204.6609,
    56.5233,
    200.3284,
    201.6651,
    213.5577,
    17.0374,
    164.4194,
    94.5422,
    131.9124,
    61.0309,
    296.2073,
    135.4894,
    114.8750,
    247.0691,
    256.6114,
    32.1008,
    143.6804,
    16.8784,
    160.6835,
    27.5932,
    348.1074,
    82.6496,
)

# --- Eccentricity / fixed vernal equinox longitude (19 terms) ---
_pecclen = 19

_ecamp = (
    0.01860798,
    0.01627522,
    -0.01300660,
    0.00988829,
    -0.00336700,
    0.00333077,
    -0.00235400,
    0.00140015,
    0.00100700,
    0.00085700,
    0.00064990,
    0.00059900,
    0.00037800,
    -0.00033700,
    0.00027600,
    0.00018200,
    -0.00017400,
    -0.00012400,
    0.00001250,
)

_ecrate = (
    4.2072050,
    7.3460910,
    17.8572630,
    17.2205460,
    16.8467330,
    5.1990790,
    18.2310760,
    26.2167580,
    6.3591690,
    16.2100160,
    3.0651810,
    16.5838290,
    18.4939800,
    6.1909530,
    18.8677930,
    17.4255670,
    6.1860010,
    18.4174410,
    0.6678630,
)

_ecphas = (
    28.620089,
    193.788772,
    308.307024,
    320.199637,
    279.376984,
    87.195000,
    349.129677,
    128.443387,
    154.143880,
    291.269597,
    114.860583,
    332.092251,
    296.414411,
    145.769910,
    337.237063,
    152.092288,
    126.839891,
    210.667199,
    72.108838,
)

# --- Moving vernal equinox longitude of perihelion (78 terms) ---
_pmvelen = 78

_mvamp = (
    7391.0225890,
    2555.1526947,
    2022.7629188,
    -1973.6517951,
    1240.2321818,
    953.8679112,
    -931.7537108,
    872.3795383,
    606.3544732,
    -496.0274038,
    456.9608039,
    346.9462320,
    -305.8412902,
    249.6173246,
    -199.1027200,
    191.0560889,
    -175.2936572,
    165.9068833,
    161.1285917,
    139.7878093,
    -133.5228399,
    117.0673811,
    104.6907281,
    95.3227476,
    86.7824524,
    86.0857729,
    70.5893698,
    -69.9719343,
    -62.5817473,
    61.5450059,
    -57.9364011,
    57.1899832,
    -57.0236109,
    -54.2119253,
    53.2834147,
    52.1223575,
    -49.0059908,
    -48.3118757,
    -45.4191685,
    -42.2357920,
    -34.7971099,
    34.4623613,
    -33.8356643,
    33.6689362,
    -31.2521586,
    -30.8798701,
    28.4640769,
    -27.1960802,
    27.0860736,
    -26.3437456,
    24.7253740,
    24.6732126,
    24.4272733,
    24.0127327,
    21.7150294,
    -21.5375347,
    18.1148363,
    -16.9603104,
    -16.1765215,
    15.5567653,
    15.4846529,
    15.2150632,
    14.5047426,
    -14.3873316,
    13.1351419,
    12.8776311,
    11.9867234,
    11.9385578,
    11.7030822,
    11.6018181,
    -11.2617293,
    -10.4664199,
    10.4333970,
    -10.2377466,
    10.1934446,
    -10.1280191,
    10.0289441,
    -10.0034259,
)

_mvrate = (
    31.609974,
    32.620504,
    24.172203,
    0.636717,
    31.983787,
    3.138886,
    30.973257,
    44.828336,
    0.991874,
    0.373813,
    43.668246,
    32.246691,
    30.599444,
    2.147012,
    10.511172,
    42.681324,
    13.650058,
    0.986922,
    9.874455,
    13.013341,
    0.262904,
    0.004952,
    1.142024,
    63.219948,
    0.205021,
    2.151964,
    64.230478,
    43.836462,
    47.439436,
    1.384343,
    7.437771,
    18.829299,
    9.500642,
    0.431696,
    1.160090,
    55.782177,
    12.639528,
    1.155138,
    0.168216,
    1.647247,
    10.884985,
    5.610937,
    12.658184,
    1.010530,
    1.983748,
    14.023871,
    0.560178,
    1.273434,
    12.021467,
    62.583231,
    63.593761,
    76.438310,
    4.280910,
    13.218362,
    17.818769,
    8.359495,
    56.792707,
    8.448301,
    1.978796,
    8.863925,
    0.186365,
    8.996212,
    6.771027,
    45.815258,
    12.002811,
    75.278220,
    65.241008,
    18.870667,
    22.009553,
    64.604291,
    11.498094,
    0.578834,
    9.237738,
    49.747842,
    2.147012,
    1.196895,
    2.133898,
    0.173168,
)

_mvphas = (
    251.9025,
    280.8325,
    128.3057,
    348.1074,
    292.7252,
    165.1686,
    263.7951,
    15.3747,
    58.5749,
    40.8226,
    308.4258,
    240.0099,
    222.9725,
    106.5937,
    114.5182,
    268.7809,
    279.6869,
    39.6448,
    126.4108,
    291.5795,
    307.2848,
    18.9300,
    273.7596,
    143.8050,
    191.8927,
    125.5237,
    172.7351,
    316.7998,
    319.6024,
    69.7526,
    123.5968,
    217.6432,
    85.5882,
    156.2147,
    66.9489,
    20.2082,
    250.7568,
    48.0188,
    8.3739,
    17.0374,
    155.3409,
    94.1709,
    221.1120,
    28.9300,
    117.1498,
    320.5095,
    262.3602,
    336.2148,
    233.0046,
    155.6977,
    184.6277,
    267.2772,
    78.9281,
    123.4722,
    188.7132,
    180.1364,
    49.1382,
    152.5268,
    98.2198,
    97.4808,
    221.5376,
    168.2438,
    161.1199,
    55.0196,
    262.6495,
    200.3284,
    201.6651,
    294.6547,
    99.8233,
    213.5577,
    154.1631,
    232.7153,
    138.3034,
    204.6609,
    106.5938,
    250.4676,
    332.3345,
    27.3039,
)


# ---------------------------------------------------------------------------
# shr_orb_cosz
# ---------------------------------------------------------------------------


def shr_orb_cosz(jday: float, lat: float, lon: float, declin: float) -> float:
    """
    Return the cosine of the solar zenith angle.

    Mirrors Fortran function ``shr_orb_cosz`` (lines 22-36).

    Assumes 365.0 days per year.  The formula (Fortran line 34) is::

        cos(zen) = sin(lat)*sin(declin)
                 - cos(lat)*cos(declin)*cos(jday*2π + lon)

    Args:
        jday:   Julian calendar day including fraction (1.xx – 365.xx).
        lat:    Geodetic latitude (radians).
        lon:    Longitude (radians).
        declin: Solar declination (radians).

    Returns:
        Cosine of the solar zenith angle (dimensionless).
    """
    return math.sin(lat) * math.sin(declin) - math.cos(lat) * math.cos(declin) * math.cos(
        jday * 2.0 * pi + lon
    )


# ---------------------------------------------------------------------------
# shr_orb_decl
# ---------------------------------------------------------------------------


def shr_orb_decl(
    calday: float,
    eccen: float,
    mvelpp: float,
    lambm0: float,
    obliqr: float,
) -> tuple:
    """
    Compute the eccentricity factor and solar declination.

    Mirrors Fortran subroutine ``shr_orb_decl`` (lines 38-88).

    Uses formulas from Berger (1978) J. Atmos. Sci. 35:2362-2367.

    **Algorithm** (Fortran lines 65-84):

    1. Compute mean longitude at present day::

           lambm = lambm0 + (calday - ve) * 2π / dayspy
           lmm   = lambm - mvelpp

    2. Compute Earth's true longitude (Berger 1978 eq.)::

           sinl = sin(lmm)
           lamb = lambm + eccen*(2*sinl + eccen*(1.25*sin(2*lmm)
                  + eccen*((13/12)*sin(3*lmm) - 0.25*sinl)))

    3. Compute inverse normalised sun–Earth distance::

           invrho = (1 + eccen*cos(lamb - mvelpp)) / (1 - eccen²)

    4. Set outputs::

           delta = asin(sin(obliqr) * sin(lamb))
           eccf  = invrho²

    Local constants: ``dayspy = 365.0``, ``ve = 80.5`` (vernal equinox
    calendar day; Fortran lines 49-50).

    Args:
        calday: Calendar day including fraction.
        eccen:  Orbital eccentricity.
        mvelpp: Moving vernal equinox longitude of perihelion plus π (rad).
        lambm0: Mean longitude of perihelion at vernal equinox (rad).
        obliqr: Earth's obliquity in radians.

    Returns:
        Tuple ``(delta, eccf)`` where

        - ``delta`` — solar declination angle (radians).
        - ``eccf``  — Earth-sun distance factor ``(1/r)²``.
    """
    dayspy = 365.0  # Days per year — Fortran line 49
    ve = 80.5  # Vernal equinox calendar day — Fortran line 50

    # Mean longitude of perihelion at present day — Fortran lines 65-66
    lambm = lambm0 + (calday - ve) * 2.0 * pi / dayspy
    lmm = lambm - mvelpp

    # Earth's true longitude — Fortran lines 68-70
    sinl = math.sin(lmm)
    lamb = lambm + eccen * (
        2.0 * sinl
        + eccen
        * (1.25 * math.sin(2.0 * lmm) + eccen * ((13.0 / 12.0) * math.sin(3.0 * lmm) - 0.25 * sinl))
    )

    # Inverse normalised sun/earth distance — Fortran line 76
    invrho = (1.0 + eccen * math.cos(lamb - mvelpp)) / (1.0 - eccen * eccen)

    # Solar declination and eccentricity factor — Fortran lines 79-80
    delta = math.asin(math.sin(obliqr) * math.sin(lamb))
    eccf = invrho * invrho

    return delta, eccf


# ---------------------------------------------------------------------------
# shr_orb_params
# ---------------------------------------------------------------------------


def shr_orb_params(iyear_AD: int) -> tuple:
    """
    Calculate Earth's orbital parameters for a given year.

    Mirrors Fortran subroutine ``shr_orb_params`` (lines 90-220).

    Implements Dave Thresher's formulation of Berger (1978).  Valid to
    approximately ±1,000,000 years from present.

    **Algorithm** (Fortran lines 148-214):

    1. ``yb4_1950AD = 1950 - iyear_AD``; ``years = -yb4_1950AD``.

    2. **Obliquity** (``obliq``, degrees): cosine series of 47 terms::

           obliq = 23.320556 + Σ obamp[i]*psecdeg
                               * cos((obrate[i]*psecdeg*years + obphas[i])*degrad)

    3. **Eccentricity** (``eccen``): cos/sin series of 19 terms::

           cossum = Σ ecamp[i]*cos((ecrate[i]*psecdeg*years + ecphas[i])*degrad)
           sinsum = Σ ecamp[i]*sin((ecrate[i]*psecdeg*years + ecphas[i])*degrad)
           eccen  = sqrt(cossum² + sinsum²)

    4. **Fixed vernal equinox longitude** (``fvelp``, radians): derived
       from ``atan2(sinsum, cossum)`` with quadrant correction matching
       the Fortran ``if/else`` chain (Fortran lines 170-183).

    5. **Moving vernal equinox longitude** (``mvelp``, degrees): sine
       series of 78 terms::

           mvelp = fvelp/degrad + 50.439273*psecdeg*years
                   + 3.392506 + Σ mvamp[i]*psecdeg
                                 * sin((mvrate[i]*psecdeg*years + mvphas[i])*degrad)

       Normalised to [0, 360).

    6. **Derived quantities** (Fortran lines 200-214)::

           obliqr = obliq * degrad
           mvelpp = (mvelp + 180) * degrad
           beta   = sqrt(1 - eccen²)
           lambm0 = 2*((0.5*e + 0.125*e³)*(1+β)*sin(mvelpp)
                       - 0.25*e²*(0.5+β)*sin(2*mvelpp)
                       + 0.125*e³*(1/3+β)*sin(3*mvelpp))

    Args:
        iyear_AD: Year AD for which to compute orbital parameters
            (e.g. 1995).

    Returns:
        Tuple ``(eccen, obliq, mvelp, obliqr, lambm0, mvelpp)`` where

        - ``eccen``  — orbital eccentricity (dimensionless).
        - ``obliq``  — obliquity in degrees.
        - ``mvelp``  — moving vernal equinox longitude of perihelion
          (degrees, in [0, 360)).
        - ``obliqr`` — obliquity in radians.
        - ``lambm0`` — mean longitude of perihelion at vernal equinox
          (radians).
        - ``mvelpp`` — moving vernal equinox longitude of perihelion
          plus π (radians).
    """
    psecdeg = 1.0 / 3600.0  # arc seconds → degrees — Fortran line 111
    degrad = pi / 180.0  # degrees → radians — Fortran line 148

    yb4_1950AD = 1950.0 - float(iyear_AD)
    years = -yb4_1950AD  # Fortran lines 150-151

    # ------------------------------------------------------------------
    # 2. Obliquity — Fortran lines 159-164
    # ------------------------------------------------------------------
    obsum = 0.0
    for i in range(_poblen):
        obsum += (
            _obamp[i] * psecdeg * math.cos((_obrate[i] * psecdeg * years + _obphas[i]) * degrad)
        )
    obliq = 23.320556 + obsum

    # ------------------------------------------------------------------
    # 3. Eccentricity — Fortran lines 167-178
    # ------------------------------------------------------------------
    cossum = 0.0
    for i in range(_pecclen):
        cossum += _ecamp[i] * math.cos((_ecrate[i] * psecdeg * years + _ecphas[i]) * degrad)

    sinsum = 0.0
    for i in range(_pecclen):
        sinsum += _ecamp[i] * math.sin((_ecrate[i] * psecdeg * years + _ecphas[i]) * degrad)

    eccen2 = cossum * cossum + sinsum * sinsum
    eccen = math.sqrt(eccen2)
    eccen3 = eccen2 * eccen

    # ------------------------------------------------------------------
    # 4. Fixed vernal equinox longitude — Fortran lines 170-183
    # Exact quadrant logic from the Fortran if/else chain.
    # ------------------------------------------------------------------
    if abs(cossum) <= 1.0e-8:
        if sinsum == 0.0:
            fvelp = 0.0
        elif sinsum < 0.0:
            fvelp = 1.5 * pi
        else:  # sinsum > 0
            fvelp = 0.5 * pi
    elif cossum < 0.0:
        fvelp = math.atan(sinsum / cossum) + pi
    else:  # cossum > 0
        if sinsum < 0.0:
            fvelp = math.atan(sinsum / cossum) + 2.0 * pi
        else:
            fvelp = math.atan(sinsum / cossum)

    # ------------------------------------------------------------------
    # 5. Moving vernal equinox longitude — Fortran lines 186-198
    # ------------------------------------------------------------------
    mvsum = 0.0
    for i in range(_pmvelen):
        mvsum += (
            _mvamp[i] * psecdeg * math.sin((_mvrate[i] * psecdeg * years + _mvphas[i]) * degrad)
        )

    mvelp = fvelp / degrad + 50.439273 * psecdeg * years + 3.392506 + mvsum

    # Normalise to [0, 360) — Fortran lines 197-201
    while mvelp < 0.0:
        mvelp += 360.0
    while mvelp >= 360.0:
        mvelp -= 360.0

    # ------------------------------------------------------------------
    # 6. Derived quantities — Fortran lines 204-214
    # ------------------------------------------------------------------
    obliqr = obliq * degrad
    mvelpp = (mvelp + 180.0) * degrad

    beta = math.sqrt(1.0 - eccen2)

    lambm0 = 2.0 * (
        (0.5 * eccen + 0.125 * eccen3) * (1.0 + beta) * math.sin(mvelpp)
        - 0.25 * eccen2 * (0.5 + beta) * math.sin(2.0 * mvelpp)
        + 0.125 * eccen3 * (1.0 / 3.0 + beta) * math.sin(3.0 * mvelpp)
    )

    return eccen, obliq, mvelp, obliqr, lambm0, mvelpp
