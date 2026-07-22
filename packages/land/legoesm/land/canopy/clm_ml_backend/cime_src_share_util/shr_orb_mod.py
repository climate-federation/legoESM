"""
JAX translation of shr_orb_mod module.

Calculates Earth's orbital parameters, solar declination, and zenith angle
using Berger (1978) algorithms for long-term variations in Earth's orbit.

References:
    Berger, Andre. 1978. Long-Term Variations of Daily Insolation and
    Quaternary Climatic Changes. J. of the Atmo. Sci. 35:2362-2367.

    Berger, Andre. 1978. A Simple Algorithm to Compute Long-Term Variations
    of Daily Insolation. Contribution 18, Institute of Astronomy and
    Geophysics, Universite Catholique de Louvain, Belgium.

Translated from: shr_orb_mod.F90, lines 1-467
"""

from typing import NamedTuple, Tuple

import jax.numpy as jnp
from jax import Array, jit, lax

# =============================================================================
# Constants
# =============================================================================

PI = jnp.pi

# =============================================================================
# Type Definitions
# =============================================================================


class OrbitalParams(NamedTuple):
    """Container for Earth's orbital parameters.

    Attributes:
        eccen: Orbital eccentricity (dimensionless)
        obliq: Obliquity in degrees
        mvelp: Moving vernal equinox longitude of perihelion (degrees)
        obliqr: Earth's obliquity in radians
        lambm0: Mean longitude of perihelion at vernal equinox (radians)
        mvelpp: Moving vernal equinox longitude of perihelion plus pi (radians)
    """

    eccen: jnp.ndarray
    obliq: jnp.ndarray
    mvelp: jnp.ndarray
    obliqr: jnp.ndarray
    lambm0: jnp.ndarray
    mvelpp: jnp.ndarray


# =============================================================================
# Orbital Parameter Series Data (Berger 1978 Tables)
# =============================================================================

# Obliquity series parameters (47 terms)
# Amplitudes in arc seconds
OBAMP = jnp.array(
    [
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
        -1.7056854,
        1.6560716,
        1.4437463,
        0.8803925,
        -0.8468189,
        -0.7674114,
        -0.6827364,
        -0.6089556,
        0.6060708,
        -0.5905702,
        0.5817298,
    ],
    dtype=jnp.float64,
)

# Rates in arc seconds per year
OBRATE = jnp.array(
    [
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
    ],
    dtype=jnp.float64,
)

# Phases in degrees
OBPHAS = jnp.array(
    [
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
    ],
    dtype=jnp.float64,
)

# Eccentricity series parameters (19 terms)
# Amplitudes (dimensionless)
ECAMP = jnp.array(
    [
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
    ],
    dtype=jnp.float64,
)

# Rates in arc seconds per year
ECRATE = jnp.array(
    [
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
    ],
    dtype=jnp.float64,
)

# Phases in degrees
ECPHAS = jnp.array(
    [
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
    ],
    dtype=jnp.float64,
)

# Moving vernal equinox series parameters (78 terms)
# Amplitudes in arc seconds
MVAMP = jnp.array(
    [
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
        117.0673862,
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
    ],
    dtype=jnp.float64,
)

# Rates in arc seconds per year
MVRATE = jnp.array(
    [
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
        1.160091,
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
    ],
    dtype=jnp.float64,
)

# Phases in degrees
MVPHAS = jnp.array(
    [
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
    ],
    dtype=jnp.float64,
)

# Conversion factors
PSECDEG = 1.0 / 3600.0  # Arc seconds to degrees


# =============================================================================
# Public Functions
# =============================================================================


@jit
def shr_orb_cosz(jday: Array, lat: Array, lon: Array, declin: Array) -> Array:
    """
    Return the cosine of the solar zenith angle.

    Assumes 365.0 days per year.

    Args:
        jday: Julian calendar day (1.xx to 365.xx) [radians for phase calculation]
        lat: Centered latitude (radians)
        lon: Centered longitude (radians)
        declin: Solar declination (radians)

    Returns:
        Cosine of the solar zenith angle (dimensionless)

    Notes:
        - Pure function, JIT-compatible
        - Vectorized for array inputs
        - Formula: cos(zenith) = sin(lat)*sin(declin) - cos(lat)*cos(declin)*cos(jday*2*pi + lon)
        - The jday term represents the hour angle in the original formulation

    Reference: Fortran lines 23-41
    """
    cosz = jnp.sin(lat) * jnp.sin(declin) - jnp.cos(lat) * jnp.cos(declin) * jnp.cos(
        jday * 2.0 * PI + lon
    )

    return cosz


def shr_orb_decl(
    calday: jnp.ndarray,
    eccen: float,
    mvelpp: float,
    lambm0: float,
    obliqr: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute eccentricity factor and solar declination.

    Uses day value where a round day (such as 213.0) refers to 0z at
    Greenwich longitude.

    Based on formulas from Berger, Andre 1978: Long-Term Variations of Daily
    Insolation and Quaternary Climatic Changes. J. of the Atmo. Sci.
    35:2362-2367.

    Args:
        calday: Calendar day, including fraction. Shape: (...)
        eccen: Eccentricity (dimensionless)
        mvelpp: Moving vernal equinox longitude of perihelion plus pi (radians)
        lambm0: Mean longitude of perihelion at the vernal equinox (radians)
        obliqr: Earth's obliquity in radians

    Returns:
        delta: Solar declination angle (radians). Shape matches calday.
        eccf: Earth-sun distance factor (i.e. (1/r)**2). Shape matches calday.

    Physics:
        1. Calculate mean longitude (lambm) from vernal equinox reference
        2. Compute Earth's true longitude (lamb) using eccentric anomaly
        3. Calculate inverse normalized sun/earth distance (invrho)
        4. Compute solar declination from obliquity and true longitude
        5. Compute eccentricity factor as (1/r)^2

    Reference: Fortran lines 42-106
    """
    # Constants
    dayspy = 365.0  # Days per year
    ve = 80.5  # calday of vernal equinox (assumes Jan 1 = calday 1)

    # Calculate mean longitude at present day
    lambm = lambm0 + (calday - ve) * 2.0 * jnp.pi / dayspy
    lmm = lambm - mvelpp

    # Calculate Earth's true longitude using Berger 1978 formula
    sinl = jnp.sin(lmm)
    lamb = lambm + eccen * (
        2.0 * sinl
        + eccen
        * (1.25 * jnp.sin(2.0 * lmm) + eccen * ((13.0 / 12.0) * jnp.sin(3.0 * lmm) - 0.25 * sinl))
    )

    # Calculate inverse normalized sun/earth distance
    invrho = (1.0 + eccen * jnp.cos(lamb - mvelpp)) / (1.0 - eccen * eccen)

    # Set solar declination and eccentricity factor
    delta = jnp.arcsin(jnp.sin(obliqr) * jnp.sin(lamb))
    eccf = invrho * invrho

    return delta, eccf


@jit
def shr_orb_params(iyear_AD: int) -> OrbitalParams:
    """Calculate Earth's orbital parameters for a given year.

    Uses Dave Thresher's formula based on Berger (1978) to compute long-term
    variations in Earth's orbital parameters.

    Args:
        iyear_AD: Year to calculate orbit (Anno Domini)

    Returns:
        OrbitalParams: Named tuple containing:
            - eccen: Orbital eccentricity
            - obliq: Obliquity in degrees
            - mvelp: Moving vernal equinox longitude (degrees)
            - obliqr: Earth's obliquity in radians
            - lambm0: Mean longitude of perihelion at vernal equinox (radians)
            - mvelpp: Moving vernal equinox longitude of perihelion plus pi (radians)

    Reference:
        Fortran lines 109-465
        Berger, A. (1978). A Simple Algorithm to Compute Long-Term Variations
        of Daily Insolation. Contribution 18, Institute of Astronomy and
        Geophysics, Universite Catholique de Louvain, Belgium.

    Note:
        Algorithm valid only to 1,000,000 years past or future.
    """
    # Convert input to array for consistency
    iyear_AD = jnp.array(iyear_AD, dtype=jnp.float64)

    # Degree to radian conversion
    degrad = PI / 180.0

    # Years before 1950 AD
    yb4_1950AD = 1950.0 - iyear_AD
    years = -yb4_1950AD

    # =========================================================================
    # Part 1: Obliquity calculation
    # =========================================================================

    # Summation of cosine series for obliquity
    def obliq_body(i: int, acc: jnp.ndarray) -> jnp.ndarray:
        """Accumulate obliquity series term."""
        arg = (OBRATE[i] * PSECDEG * years + OBPHAS[i]) * degrad
        term = OBAMP[i] * PSECDEG * jnp.cos(arg)
        return acc + term

    obsum = lax.fori_loop(0, OBAMP.shape[0], obliq_body, jnp.array(0.0))

    # Obliquity = epsilon star + series summation
    obliq = 23.320556 + obsum

    # =========================================================================
    # Part 2: Eccentricity calculation
    # =========================================================================

    # Cosine summation for eccentricity
    def eccen_cossum_body(i: int, acc: jnp.ndarray) -> jnp.ndarray:
        """Accumulate eccentricity cosine series term."""
        arg = (ECRATE[i] * PSECDEG * years + ECPHAS[i]) * degrad
        term = ECAMP[i] * jnp.cos(arg)
        return acc + term

    cossum = lax.fori_loop(0, ECAMP.shape[0], eccen_cossum_body, jnp.array(0.0))

    # Sine summation for eccentricity
    def eccen_sinsum_body(i: int, acc: jnp.ndarray) -> jnp.ndarray:
        """Accumulate eccentricity sine series term."""
        arg = (ECRATE[i] * PSECDEG * years + ECPHAS[i]) * degrad
        term = ECAMP[i] * jnp.sin(arg)
        return acc + term

    sinsum = lax.fori_loop(0, ECAMP.shape[0], eccen_sinsum_body, jnp.array(0.0))

    # Compute eccentricity
    eccen2 = cossum * cossum + sinsum * sinsum
    eccen = jnp.sqrt(eccen2)
    eccen3 = eccen2 * eccen

    # =========================================================================
    # Part 3: Fixed vernal equinox longitude of perihelion (fvelp)
    # =========================================================================

    # Case 1: abs(cossum) <= 1.0E-8
    fvelp_case1a = 0.0
    fvelp_case1b = 1.5 * PI
    fvelp_case1c = 0.5 * PI

    fvelp_case1 = jnp.where(
        sinsum == 0.0, fvelp_case1a, jnp.where(sinsum < 0.0, fvelp_case1b, fvelp_case1c)
    )

    # Case 2: cossum < 0.0
    fvelp_case2 = jnp.arctan2(sinsum, cossum) + PI

    # Case 3: cossum > 0.0
    fvelp_case3a = jnp.arctan2(sinsum, cossum) + 2.0 * PI
    fvelp_case3b = jnp.arctan2(sinsum, cossum)
    fvelp_case3 = jnp.where(sinsum < 0.0, fvelp_case3a, fvelp_case3b)

    # Select among main cases
    abs_cossum_small = jnp.abs(cossum) <= 1.0e-8
    cossum_negative = cossum < 0.0

    fvelp = jnp.where(
        abs_cossum_small, fvelp_case1, jnp.where(cossum_negative, fvelp_case2, fvelp_case3)
    )

    # =========================================================================
    # Part 4: Moving vernal equinox longitude of perihelion (mvelp)
    # =========================================================================

    # Summation for moving vernal equinox
    def mvsum_body(i: int, acc: jnp.ndarray) -> jnp.ndarray:
        """Accumulate moving vernal equinox series term."""
        arg = (MVRATE[i] * PSECDEG * years + MVPHAS[i]) * degrad
        term = MVAMP[i] * PSECDEG * jnp.sin(arg)
        return acc + term

    mvsum = lax.fori_loop(0, MVAMP.shape[0], mvsum_body, jnp.array(0.0))

    # Compute mvelp
    mvelp = fvelp / degrad + 50.439273 * PSECDEG * years + 3.392506 + mvsum

    # Normalize mvelp to [0, 360)
    mvelp = jnp.mod(mvelp, 360.0)

    # =========================================================================
    # Part 5: Final conversions
    # =========================================================================

    # Convert obliquity to radians
    obliqr = obliq * degrad

    # Add 180 degrees to mvelp and convert to radians
    mvelpp = (mvelp + 180.0) * degrad

    # Calculate mean longitude at vernal equinox (lambm0)
    beta = jnp.sqrt(1.0 - eccen2)

    term1 = (0.5 * eccen + 0.125 * eccen3) * (1.0 + beta) * jnp.sin(mvelpp)
    term2 = 0.25 * eccen2 * (0.5 + beta) * jnp.sin(2.0 * mvelpp)
    term3 = 0.125 * eccen3 * (1.0 / 3.0 + beta) * jnp.sin(3.0 * mvelpp)

    lambm0 = 2.0 * (term1 - term2 + term3)

    return OrbitalParams(
        eccen=eccen, obliq=obliq, mvelp=mvelp, obliqr=obliqr, lambm0=lambm0, mvelpp=mvelpp
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "shr_orb_cosz",
    "shr_orb_decl",
    "shr_orb_params",
    "OrbitalParams",
]
