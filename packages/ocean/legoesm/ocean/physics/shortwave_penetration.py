"""Subsurface shortwave penetration heating.

Two selectable schemes (``ShortwavePenetrationConfig.scheme``):

``"jerlov_2band"`` (default)
    Spatially-uniform two-band exponential absorption (Paulson & Simpson
    1977, Jerlov water types).  Cheap, no chlorophyll input.

``"rgb_chl"``
    Faithful port of NEMO 5.0.1 ``tra_qsr`` RGB scheme (``ln_qsr_rgb``):
    an infrared band plus three visible (red/green/blue) bands whose
    extinction lengths are chlorophyll-dependent via the Morel & Maritorena
    (2001) 61-class look-up table (``trc_oce.F90::trc_oce_tab``).  With
    ``rgb_chl_profile="morel_berthon"`` the column chlorophyll follows the
    Morel & Berthon (1989) analytical vertical profile (NEMO
    ``nn_chlprfl=1``); with ``"surface"`` the surface value is extended
    downward (``nn_chlprfl=0``).  Clear (low-Chl) subtropical water lets
    blue light penetrate ~60 m, cooling the surface; productive
    (high-Chl) subpolar water traps light near the surface.  This spatial
    contrast is the physical lever for the subtropical warm / subpolar cold
    SST biases against NEMO.

Without any penetration, all SW heating lands in the surface layer,
producing unrealistically warm SST and shallow mixed layers.

References
----------
Paulson, C. A. & Simpson, J. J. (1977): Irradiance measurements in the
    upper ocean. J. Phys. Oceanogr., 7(6), 952-956.
Jerlov, N. G. (1976): Marine Optics. Elsevier, 231 pp.
Morel, A. & Berthon, J.-F. (1989): Surface pigments, algal biomass
    profiles ... Limnol. Oceanogr., 34(8), 1545-1562.
Lengaigne, M. et al. (2007): Influence of the oceanic biology on the
    tropical Pacific climate ... Clim. Dyn., 28, 503-516.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.ocean.eos import rho_0 as _RHO_0_DEFAULT, c_sw as _C_SW_DEFAULT


__physics_contract__ = {
    "summary": (
        "Subsurface shortwave penetration heating. Two schemes: a uniform "
        "two-band Jerlov exponential (Paulson & Simpson 1977), and a faithful "
        "port of NEMO tra_qsr RGB (ln_qsr_rgb): an infrared band plus three "
        "visible bands (red/green/blue) whose extinction lengths follow "
        "chlorophyll via the Morel & Maritorena (2001) 61-class table, with "
        "an optional Morel & Berthon (1989) analytical vertical Chl profile "
        "(NEMO nn_chlprfl=1). Light reaching the seabed is deposited in the "
        "deepest wet level (no-flux bottom). Clear low-Chl water penetrates "
        "deeper (cooler surface); turbid high-Chl water traps light near top."
    ),
    "inputs": {
        "sw_down": "W/m^2", "chl_surface": "mg/m^3", "dz_live": "m",
        "wet_cell": "1", "dz_ref": "m", "z_half_ref": "m", "jacobian": "1",
        "rho_0": "kg/m^3", "c_sw": "J/(kg K)",
    },
    "outputs": {"dT_dt": "degC/s"},
    "sign_convention": (
        "sw_down >= 0 is net shortwave INTO the ocean (post-albedo); the "
        "returned dT/dt >= 0 everywhere it heats, summing over the column to "
        "deposit 100% of sw_down in the wet cells (no light lost below the "
        "seabed)."
    ),
    # Column-integral heat closure: sum_k (rho_0*c_sw*dz*dT/dt) == sw_down to
    # floating-point tolerance, because the interface-flux telescopes and the
    # bottom no-flux mask routes the residual into the last wet level.
    "conserves": ["energy"],
    "differentiable": True,
    "reference": (
        "Paulson & Simpson (1977) JPO 7 952-956; Morel & Berthon (1989) "
        "Limnol. Oceanogr. 34 1545-1562; Morel & Maritorena (2001) JGR 106 "
        "7163-7180; Lengaigne+ (2007) Clim. Dyn. 28 503-516; NEMO 5.0.1 "
        "TRA/traqsr.F90 + trc_oce.F90 (ORCA1 RUN_REF: ln_qsr_rgb, nn_chldta=1, "
        "nn_chlprfl=1, rn_abs=0.58, rn_si0=0.35)."
    ),
    "idealized_test": (
        "tests/ocean/unit/test_rgb_chl_penetration.py: NEMO class-index "
        "regression (Chl=0.03->11, 1->41, 10->61), single-column band-sum vs "
        "an independent hand recursion, exact column heat closure, dry-column "
        "zero, and Morel-Berthon profile regression against the NEMO polynomial."
    ),
}

__param_spec__ = {
    "ShortwavePenetrationConfig": {
        "scheme_key": "ocean.sw_penetration",
        "excluded": {},
        "params": {
            "rgb_ir_fraction": {
                "units": "1", "bounds": (0.4, 0.7), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "NEMO rn_abs (ORCA1 default 0.58); Paulson & Simpson (1977)",
                "shape": None,
            },
            "rgb_ir_extinction_m": {
                "units": "m", "bounds": (0.1, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "NEMO rn_si0 (ORCA1 default 0.35 m); Paulson & Simpson (1977)",
                "shape": None,
            },
        },
    },
}


# ==============================================================================
# Jerlov water type parameters
# ==============================================================================
# Two-band model: I(z) = Q_sw * [R * exp(z/zeta1) + (1-R) * exp(z/zeta2)]
# where z is negative (depth below surface), zeta1/zeta2 are e-folding depths.

class JerlovParams(NamedTuple):
    """Two-band parameters for a Jerlov water type."""
    R: float        # Fraction in short-wavelength band (red/IR)
    zeta1: float    # e-folding depth of band 1 [m] (short, ~red/IR)
    zeta2: float    # e-folding depth of band 2 [m] (long, ~blue/green)


# Standard Jerlov water types (Paulson & Simpson 1977, Table 1).
JERLOV_TYPES: dict[str, JerlovParams] = {
    "I":   JerlovParams(R=0.58, zeta1=0.35, zeta2=23.0),
    "IA":  JerlovParams(R=0.62, zeta1=0.60, zeta2=20.0),
    "IB":  JerlovParams(R=0.67, zeta1=1.00, zeta2=17.0),
    "II":  JerlovParams(R=0.77, zeta1=1.50, zeta2=14.0),
    "III": JerlovParams(R=0.78, zeta1=1.40, zeta2=7.9),
}


class ShortwavePenetrationConfig(NamedTuple):
    """Configuration for subsurface SW penetration.

    Parameters
    ----------
    scheme : str
        ``"jerlov_2band"`` (default) or ``"rgb_chl"``.  See module docstring.
    water_type : str
        Jerlov water type ("I", "IA", "IB", "II", "III") for the two-band
        scheme.  Type I = clearest open ocean, Type III = coastal/turbid.
        Default "II" is a reasonable global average.
    rgb_ir_fraction : float
        ``rgb_chl`` only.  Fraction of net SW in the infrared band absorbed
        in the very-near surface (NEMO ``rn_abs``; ORCA1 default 0.58).  The
        remaining ``1 - rgb_ir_fraction`` is split equally among R, G, B.
    rgb_ir_extinction_m : float
        ``rgb_chl`` only.  Infrared e-folding depth [m] (NEMO ``rn_si0``;
        ORCA1 default 0.35 m).
    rgb_chl_profile : str
        ``rgb_chl`` only.  ``"morel_berthon"`` (NEMO ``nn_chlprfl=1``,
        analytical vertical Chl profile with deep-Chl maximum) or
        ``"surface"`` (``nn_chlprfl=0``, surface Chl extended downward).
    """
    scheme: str = "jerlov_2band"
    water_type: str = "II"
    rgb_ir_fraction: float = 0.58       # NEMO rn_abs
    rgb_ir_extinction_m: float = 0.35   # NEMO rn_si0 [m]
    rgb_chl_profile: str = "morel_berthon"


# ==============================================================================
# NEMO RGB chlorophyll scheme (faithful port of tra_qsr / trc_oce_tab)
# ==============================================================================
# 61-class Red-Green-Blue attenuation look-up table, transcribed verbatim from
# NEMO 5.0.1 ``src/OCE/trc_oce.F90`` (trc_oce_tab), columns ztab(2:4,:) =
# (blue, green, red) inverse-extinction-length [1/m].  Row j (0-based) is
# chlorophyll class j+1; class index from Chl is the NEMO formula
# ``NINT(41 + 20*log10(Chl))`` with Chl clamped to [0.03, 10] mg/m^3.
# Reference: Morel & Maritorena (2001) JGR 106(C4):7163-7180; Lengaigne+ 2007.
_RGB_ATTENUATION_BGR: tuple = (
    (0.01618, 0.07464, 0.37807),  # class  1
    (0.01654, 0.07480, 0.37823),  # class  2
    (0.01693, 0.07499, 0.37840),  # class  3
    (0.01736, 0.07518, 0.37859),  # class  4
    (0.01782, 0.07539, 0.37879),  # class  5
    (0.01831, 0.07562, 0.37900),  # class  6
    (0.01885, 0.07586, 0.37923),  # class  7
    (0.01943, 0.07613, 0.37948),  # class  8
    (0.02005, 0.07641, 0.37976),  # class  9
    (0.02073, 0.07672, 0.38005),  # class 10
    (0.02146, 0.07705, 0.38036),  # class 11
    (0.02224, 0.07741, 0.38070),  # class 12
    (0.02310, 0.07780, 0.38107),  # class 13
    (0.02402, 0.07821, 0.38146),  # class 14
    (0.02501, 0.07866, 0.38189),  # class 15
    (0.02608, 0.07914, 0.38235),  # class 16
    (0.02724, 0.07967, 0.38285),  # class 17
    (0.02849, 0.08023, 0.38338),  # class 18
    (0.02984, 0.08083, 0.38396),  # class 19
    (0.03131, 0.08149, 0.38458),  # class 20
    (0.03288, 0.08219, 0.38526),  # class 21
    (0.03459, 0.08295, 0.38598),  # class 22
    (0.03643, 0.08377, 0.38676),  # class 23
    (0.03842, 0.08466, 0.38761),  # class 24
    (0.04057, 0.08561, 0.38852),  # class 25
    (0.04289, 0.08664, 0.38950),  # class 26
    (0.04540, 0.08775, 0.39056),  # class 27
    (0.04811, 0.08894, 0.39171),  # class 28
    (0.05103, 0.09023, 0.39294),  # class 29
    (0.05420, 0.09162, 0.39428),  # class 30
    (0.05761, 0.09312, 0.39572),  # class 31
    (0.06130, 0.09474, 0.39727),  # class 32
    (0.06529, 0.09649, 0.39894),  # class 33
    (0.06959, 0.09837, 0.40075),  # class 34
    (0.07424, 0.10040, 0.40270),  # class 35
    (0.07927, 0.10259, 0.40480),  # class 36
    (0.08470, 0.10495, 0.40707),  # class 37
    (0.09056, 0.10749, 0.40952),  # class 38
    (0.09690, 0.11024, 0.41216),  # class 39
    (0.10374, 0.11320, 0.41502),  # class 40
    (0.11114, 0.11639, 0.41809),  # class 41
    (0.11912, 0.11984, 0.42142),  # class 42
    (0.12775, 0.12356, 0.42500),  # class 43
    (0.13707, 0.12757, 0.42887),  # class 44
    (0.14715, 0.13189, 0.43304),  # class 45
    (0.15803, 0.13655, 0.43754),  # class 46
    (0.16978, 0.14158, 0.44240),  # class 47
    (0.18248, 0.14701, 0.44765),  # class 48
    (0.19620, 0.15286, 0.45331),  # class 49
    (0.21102, 0.15918, 0.45942),  # class 50
    (0.22703, 0.16599, 0.46601),  # class 51
    (0.24433, 0.17334, 0.47313),  # class 52
    (0.26301, 0.18126, 0.48080),  # class 53
    (0.28320, 0.18981, 0.48909),  # class 54
    (0.30502, 0.19903, 0.49803),  # class 55
    (0.32858, 0.20898, 0.50768),  # class 56
    (0.35404, 0.21971, 0.51810),  # class 57
    (0.38154, 0.23129, 0.52934),  # class 58
    (0.41125, 0.24378, 0.54147),  # class 59
    (0.44336, 0.25725, 0.55457),  # class 60
    (0.47804, 0.27178, 0.56870),  # class 61
)

# Chlorophyll clamp bounds (NEMO: 0.03 <= Chl <= 10 mg/m^3 before class index).
_CHL_MIN: float = 0.03
_CHL_MAX: float = 10.0

# RGB class-index formula (NEMO trc_oce.F90): itab = NINT(offset + slope*log10(Chl)).
# Fixed published constants — not tunable.
_RGB_CLASS_INDEX_OFFSET: float = 41.0
_RGB_CLASS_INDEX_SLOPE: float = 20.0

# --- Morel & Berthon (1989) analytical vertical Chl profile coefficients ----
# Verbatim from NEMO 5.0.1 src/OCE/TRA/traqsr.F90 qsr_RGBc CASE(1) (nn_chlprfl=1).
# Horner-form polynomial fits in zlogc = ln(Chl): log(zCze), log(zCtot), log(zze)
# with a high-Chl branch, 1/delpsi, zCb, zCmax, zpsimax. Published fixed fit
# constants (Morel & Berthon 1989, Limnol. Oceanogr. 34 1545-1562) — not tunable.
_MB89_ZCZE_C0: float = 0.113328685307
_MB89_ZCZE_C1: float = 0.803
_MB89_ZCTOT_C0: float = 3.703768066608
_MB89_ZCTOT_C1: float = 0.459
_MB89_ZZE_C0: float = 6.34247346942
_MB89_ZZE_C1: float = 0.746
_MB89_ZZE_BRANCH: float = 4.62497281328
_MB89_ZZE_ALT_C0: float = 5.298317366548
_MB89_ZZE_ALT_C1: float = 0.293
_MB89_DELPSI_C0: float = 0.710
_MB89_DELPSI_C1: float = 0.159
_MB89_DELPSI_C2: float = 0.021
_MB89_ZCB_C0: float = 0.768
_MB89_ZCB_C1: float = 0.087
_MB89_ZCB_C2: float = 0.179
_MB89_ZCB_C3: float = 0.025
_MB89_ZCMAX_C0: float = 0.299
_MB89_ZCMAX_C1: float = 0.289
_MB89_ZCMAX_C2: float = 0.579
_MB89_ZPSIMAX_C0: float = 0.6
_MB89_ZPSIMAX_C1: float = 0.640
_MB89_ZPSIMAX_C2: float = 0.021
_MB89_ZPSIMAX_C3: float = 0.115


def _rgb_class_row(chl: jnp.ndarray) -> jnp.ndarray:
    """0-based row into the RGB table for chlorophyll ``chl`` [mg/m^3].

    Faithful to NEMO ``itab = NINT(41 + 20*log10(Chl))`` (1-based) with Chl
    clamped to [0.03, 10]; the 0-based row is ``itab - 1`` clipped to [0, 60].
    ``NINT`` (round-half-up for positive args) is reproduced as
    ``floor(x + 0.5)`` rather than ``jnp.round`` (banker's rounding).
    """
    chl_c = jnp.clip(chl, _CHL_MIN, _CHL_MAX)
    x = _RGB_CLASS_INDEX_OFFSET + _RGB_CLASS_INDEX_SLOPE * jnp.log10(chl_c)
    itab = jnp.floor(x + 0.5)            # NEMO NINT for positive x
    row = jnp.clip(itab - 1.0, 0.0, 60.0)
    return row.astype(jnp.int32)


def _morel_berthon_chl_column(
    chl_surface: jnp.ndarray, gdepw_bottom: jnp.ndarray
) -> jnp.ndarray:
    """Morel & Berthon (1989) analytical vertical Chl profile (NEMO nn_chlprfl=1).

    Parameters
    ----------
    chl_surface : array, shape (...)
        Surface chlorophyll [mg/m^3].
    gdepw_bottom : array, shape (..., nlev)
        Depth of the bottom interface of each cell [m, positive down]
        (NEMO ``gdepw(jk+1)``).

    Returns
    -------
    array, shape (..., nlev)
        Chlorophyll at each model level [mg/m^3], clamped to [0.03, 10].

    Notes
    -----
    Verbatim polynomial coefficients from NEMO ``traqsr.F90`` qsr_RGBc
    ``CASE(1)``.  All operations are smooth (log / exp / poly) and therefore
    differentiable; ``chl_surface`` is treated as static forcing.
    """
    chl_c = jnp.clip(chl_surface, _CHL_MIN, _CHL_MAX)
    zlogc = jnp.log(chl_c)                                   # natural log
    zc1 = _MB89_ZCZE_C0 + _MB89_ZCZE_C1 * zlogc             # log(zCze)
    zc2 = _MB89_ZCTOT_C0 + _MB89_ZCTOT_C1 * zlogc          # log(zCtot)
    zc3 = _MB89_ZZE_C0 - _MB89_ZZE_C1 * zc2                 # log(zze)
    zc3 = jnp.where(zc3 > _MB89_ZZE_BRANCH, _MB89_ZZE_ALT_C0 - _MB89_ZZE_ALT_C1 * zc2, zc3)
    zCze = jnp.exp(zc1)
    inv_delpsi = 1.0 / (_MB89_DELPSI_C0 + zlogc * (_MB89_DELPSI_C1 + zlogc * _MB89_DELPSI_C2))
    inv_zze = jnp.exp(-zc3)
    zCb = _MB89_ZCB_C0 + zlogc * (_MB89_ZCB_C1 - zlogc * (_MB89_ZCB_C2 + zlogc * _MB89_ZCB_C3))
    zCmax = _MB89_ZCMAX_C0 - zlogc * (_MB89_ZCMAX_C1 - zlogc * _MB89_ZCMAX_C2)
    zpsimax = _MB89_ZPSIMAX_C0 - zlogc * (
        _MB89_ZPSIMAX_C1 - zlogc * (_MB89_ZPSIMAX_C2 + zlogc * _MB89_ZPSIMAX_C3)
    )
    # Dimensionless depth psi = gdepw / zze, broadcast over levels.
    zpsi = inv_zze[..., jnp.newaxis] * gdepw_bottom
    chl_z = zCze[..., jnp.newaxis] * (
        zCb[..., jnp.newaxis]
        + zCmax[..., jnp.newaxis]
        * jnp.exp(-(((zpsi - zpsimax[..., jnp.newaxis]) * inv_delpsi[..., jnp.newaxis]) ** 2))
    )
    return jnp.clip(chl_z, _CHL_MIN, _CHL_MAX)


def shortwave_penetration_rgb_tendency(
    sw_down: jnp.ndarray,
    chl_surface: jnp.ndarray,
    dz_live: jnp.ndarray,
    wet_cell: jnp.ndarray,
    config: ShortwavePenetrationConfig = ShortwavePenetrationConfig(),
    rho_0: float = _RHO_0_DEFAULT,
    c_sw: float = _C_SW_DEFAULT,
) -> jnp.ndarray:
    """NEMO RGB chlorophyll SW penetration temperature tendency.

    Faithful port of NEMO 5.0.1 ``tra_qsr`` ``qsr_RGBc`` (IR + R + G + B
    bands).  Uses the **live** (z*/partial-cell) layer thickness ``dz_live``
    for the optical-depth integral so absorption is placed at the correct
    physical depths over arbitrary bathymetry (the equal-recursion identity
    of NEMO's per-level ``exp(-e3t*k)`` written as a cumulative single
    exponential).  100% of incident SW is deposited in the wet column: the
    no-flux bottom boundary (NEMO ``wmask`` on the deepest wet w-level)
    routes all light that reaches the seabed into the last wet level.

    Parameters
    ----------
    sw_down : array, shape (...)
        Net shortwave into the ocean surface [W/m^2] (post-albedo, the full
        ``qsr`` — NEMO partitions 100% of it across the four bands).
    chl_surface : array, shape (...)
        Surface chlorophyll [mg/m^3].
    dz_live : array, shape (..., nlev)
        Live layer thickness [m] (z*-scaled, partial-cell aware).  Dry cells
        carry ``dz_live = 0``.
    wet_cell : array, shape (..., nlev)
        Wet-cell mask in {0, 1} (1 = ocean).
    config : ShortwavePenetrationConfig
    rho_0, c_sw : float
        Reference seawater density [kg/m^3] and specific heat [J/(kg K)].

    Returns
    -------
    array, shape (..., nlev)
        Temperature tendency dT/dt [K/s] from RGB SW absorption.
    """
    if config.scheme != "rgb_chl":
        raise ValueError(
            "shortwave_penetration_rgb_tendency is the RGB kernel but got "
            f"scheme={config.scheme!r}; use apply_shortwave_penetration(...) or "
            "pass ShortwavePenetrationConfig(scheme='rgb_chl')."
        )
    table = jnp.asarray(_RGB_ATTENUATION_BGR, dtype=sw_down.dtype)  # (61, 3) = (B,G,R)
    wet = wet_cell.astype(dz_live.dtype)

    # Per-level chlorophyll -> per-level RGB extinction coefficients.
    if config.rgb_chl_profile == "morel_berthon":
        gdepw_bottom = jnp.cumsum(dz_live, axis=-1)              # bottom-interface depth
        chl_z = _morel_berthon_chl_column(chl_surface, gdepw_bottom)
    elif config.rgb_chl_profile == "surface":
        chl_z = jnp.broadcast_to(
            chl_surface[..., jnp.newaxis], dz_live.shape
        )
    else:
        raise ValueError(
            f"unknown rgb_chl_profile {config.rgb_chl_profile!r} "
            "(expected 'morel_berthon' or 'surface')"
        )
    row = _rgb_class_row(chl_z)                                  # (..., nlev) int
    coeffs = table[row]                                         # (..., nlev, 3) = (B,G,R)
    k_blue = coeffs[..., 0]
    k_green = coeffs[..., 1]
    k_red = coeffs[..., 2]
    k_ir = 1.0 / config.rgb_ir_extinction_m

    # Surface partition of qsr: IR + equal R/G/B (NEMO rn_abs split).
    frac_ir = config.rgb_ir_fraction
    frac_rgb = (1.0 - config.rgb_ir_fraction) / 3.0

    # Cumulative optical depth at each interface (nlev+1), tau[...,0] = 0.
    def _interface_fraction(frac_band, k_band_per_level):
        incr = dz_live * k_band_per_level                       # (..., nlev)
        tau = jnp.cumsum(incr, axis=-1)                         # (..., nlev)
        zeros = jnp.zeros(incr.shape[:-1] + (1,), dtype=incr.dtype)
        tau = jnp.concatenate([zeros, tau], axis=-1)            # (..., nlev+1)
        return frac_band * jnp.exp(-tau)

    I_ir = _interface_fraction(frac_ir, jnp.broadcast_to(k_ir, dz_live.shape))
    I_red = _interface_fraction(frac_rgb, k_red)
    I_green = _interface_fraction(frac_rgb, k_green)
    I_blue = _interface_fraction(frac_rgb, k_blue)
    I_total = I_ir + I_red + I_green + I_blue                   # (..., nlev+1) fraction of qsr

    # Wet mask on interfaces (NEMO wmask): surface face = top wet cell; an
    # interior face is wet iff both adjacent cells are wet; the face below the
    # deepest wet cell is dry -> all remaining light deposited in that cell.
    face0 = wet[..., :1]
    face_interior = wet[..., :-1] * wet[..., 1:]               # (..., nlev-1)
    face_bottom = jnp.zeros_like(wet[..., :1])
    face_wet = jnp.concatenate([face0, face_interior, face_bottom], axis=-1)  # (..., nlev+1)
    I_face = I_total * face_wet

    frac_absorbed = I_face[..., :-1] - I_face[..., 1:]         # (..., nlev)
    dz_safe = jnp.where(dz_live > 0.0, dz_live, 1.0)
    dT_dt = sw_down[..., jnp.newaxis] * frac_absorbed / (rho_0 * c_sw * dz_safe)
    return jnp.where(dz_live > 0.0, dT_dt, 0.0)


def apply_shortwave_penetration(
    config: ShortwavePenetrationConfig,
    sw_down: jnp.ndarray,
    *,
    dz_ref: jnp.ndarray | None = None,
    z_half_ref: jnp.ndarray | None = None,
    jacobian: jnp.ndarray | None = None,
    dz_live: jnp.ndarray | None = None,
    wet_cell: jnp.ndarray | None = None,
    chl: jnp.ndarray | None = None,
    rho_0: float = _RHO_0_DEFAULT,
    c_sw: float = _C_SW_DEFAULT,
) -> jnp.ndarray:
    """Dispatch SW penetration on ``config.scheme`` (hardened: raises on unknown).

    The two-band path needs ``dz_ref``/``z_half_ref``/``jacobian``; the
    ``rgb_chl`` path needs ``chl``/``dz_live``/``wet_cell``.  Missing inputs
    for the selected scheme raise ``ValueError`` (fail-early, never a silent
    fallback to the other scheme).
    """
    if config.scheme == "jerlov_2band":
        if dz_ref is None or z_half_ref is None or jacobian is None:
            raise ValueError(
                "jerlov_2band SW penetration requires dz_ref, z_half_ref, jacobian"
            )
        return shortwave_penetration_tendency(
            sw_down, dz_ref, z_half_ref, jacobian, config, rho_0, c_sw
        )
    if config.scheme == "rgb_chl":
        if chl is None:
            raise ValueError("rgb_chl SW penetration requires a chlorophyll field (chl=)")
        if dz_live is None or wet_cell is None:
            raise ValueError("rgb_chl SW penetration requires dz_live and wet_cell")
        return shortwave_penetration_rgb_tendency(
            sw_down, chl, dz_live, wet_cell, config, rho_0, c_sw
        )
    raise ValueError(
        f"unknown shortwave penetration scheme {config.scheme!r} "
        "(expected 'jerlov_2band' or 'rgb_chl')"
    )


def shortwave_penetration_tendency(
    sw_down: jnp.ndarray,
    z_coord_dz_ref: jnp.ndarray,
    z_coord_z_half_ref: jnp.ndarray,
    jacobian: jnp.ndarray,
    config: ShortwavePenetrationConfig = ShortwavePenetrationConfig(),
    rho_0: float = _RHO_0_DEFAULT,
    c_sw: float = _C_SW_DEFAULT,
    z_half_stretch: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute 3D temperature tendency from subsurface SW absorption.

    Parameters
    ----------
    sw_down : array, shape (...,)
        Downwelling shortwave at the sea surface [W/m²].
    z_coord_dz_ref : array, shape (nlev,)
        Reference layer thicknesses [m].
    z_coord_z_half_ref : array, shape (nlev+1,)
        Reference interface depths [m] (negative, z_half_ref[0]=0).
    jacobian : array, shape (...,)
        Dynamic z-star Jacobian (eta + H) / H.
    config : ShortwavePenetrationConfig
    rho_0 : float
        Reference seawater density [kg/m³].
    c_sw : float
        Specific heat of seawater [J/(kg·K)].
    z_half_stretch : array, shape (..., 1) or (...,) or None
        NEMO ``key_qco`` z* stretch factor ``(1 + r3t)`` (``eos.nemo_r3t_stretch``)
        applied to BOTH the interface depths and the layer thickness before
        the absorption profile is evaluated — i.e. NEMO's LIVE
        ``gdepw(Kmm) = gdepw_0*(1+r3t)`` (``domzgr_substitute.h90:139``,
        ``domqco.F90:160``), consumed by ``qsr_2BD`` at
        ``traqsr.F90:665-712`` (``zatt(k+1) = [rn_abs*exp(-gdepw(k+1,Kmm)*
        r1_si0) + (1-rn_abs)*exp(-gdepw(k+1,Kmm)*r1_si1)]*r1_rho0_rcp``).
        ``None`` (default) keeps the STATIC reference ladder — BIT-IDENTICAL
        to the prior behaviour, so non-bridged recipes (no live free-surface
        stretch available) are unaffected. Measured on the DINO RUN_GDB
        kt=57601 twin (#1226 ``tra_sbc_tem_piece_decompose.py`` Part 2b): the
        live ladder shrinks the all-levels pointwise-|rel| err_norm median
        2.022e-05 -> 4.979e-07 (~40x, below the c_p-truncation floor).

    Returns
    -------
    array, shape (..., nlev)
        Temperature tendency dT/dt [K/s] from SW absorption.
    """
    if config.scheme != "jerlov_2band":
        raise ValueError(
            "shortwave_penetration_tendency is the two-band Jerlov kernel but got "
            f"scheme={config.scheme!r}; call apply_shortwave_penetration(...) to "
            "dispatch the rgb_chl scheme (needs chl/dz_live/wet_cell)."
        )
    params = JERLOV_TYPES[config.water_type]
    R = params.R
    zeta1 = params.zeta1
    zeta2 = params.zeta2

    # Interface depths (negative), shape (nlev+1,) for the STATIC reference
    # ladder, or (..., nlev+1) once a per-column live stretch is applied.
    # Uses reference z (not dynamic z*J) for the absorption profile UNLESS
    # z_half_stretch is given (see the z_half_stretch docstring above).
    # Error is O(eta/H) ~ O(1e-4), negligible vs Jerlov parameter
    # uncertainty.  Standard practice in MOM6, NEMO, and POP.
    if z_half_stretch is None:
        z_half = z_coord_z_half_ref
    else:
        z_half = z_coord_z_half_ref * z_half_stretch[..., jnp.newaxis]

    # SW flux at each interface: I(z) = Q_sw * [R*exp(z/zeta1) + (1-R)*exp(z/zeta2)]
    # z_half[..., 0] = 0 (surface), z_half[..., -1] = -H_max (bottom)
    I_half = R * jnp.exp(z_half / zeta1) + (1.0 - R) * jnp.exp(z_half / zeta2)
    # Shape: (nlev+1,) static, or (..., nlev+1) once per-column stretched.

    # Fraction absorbed in each layer = I_half[k] - I_half[k+1] (last axis:
    # the level axis in both the static (nlev+1,) and stretched (..., nlev+1)
    # cases, so index the LAST axis explicitly rather than the bare [:-1]/
    # [1:] this used before the stretched (batched) shape was introduced).
    # Without correction, frac_absorbed sums to 1 - I_half[-1] (the
    # remainder reaches the bathymetric bottom and is "lost" from the
    # column heat budget).  For deep open ocean (H >> zeta2 = 23 m)
    # the leakage is negligible, but for shelf seas / lakes (H ≈ 50 m,
    # I_half[-1] ≈ 0.06) the loss is non-trivial.  Add the leaked
    # fraction to the bottom layer so the column always absorbs the
    # full surface SW (boundary condition: total absorption at the
    # bottom; backscatter from the seafloor is neglected).
    frac_absorbed = I_half[..., :-1] - I_half[..., 1:]  # (..., nlev)
    frac_absorbed = frac_absorbed.at[..., -1].add(I_half[..., -1])

    # Actual layer thickness. When z_half_stretch is given, the SAME stretch
    # multiplies the reference thickness (NEMO's single live e3t(:,:,:,Kmm) =
    # e3t_0*(1+r3t), domzgr_substitute.h90:139 — the identical factor used
    # for z_half above, not a second independent quantity) instead of the
    # dynamic jacobian; jacobian keeps its existing (non-live-ladder) role
    # for callers that pass z_half_stretch=None.
    if z_half_stretch is None:
        dz_actual = z_coord_dz_ref * jacobian[..., jnp.newaxis]  # (..., nlev)
    else:
        dz_actual = z_coord_dz_ref * z_half_stretch[..., jnp.newaxis]  # (..., nlev)

    # Temperature tendency: dT/dt = Q_sw * frac / (rho_0 * c_sw * dz).
    # Dry / land cells have ``jacobian = 0`` → ``dz_actual = 0`` so the
    # division would produce Inf/NaN that downstream summation cannot
    # mask out (``NaN * 0 = NaN`` in IEEE).  Use a safe denominator and
    # gate the output by ``dz_actual > 0`` so dry columns contribute
    # exactly zero heating and gradients stay clean.
    dz_safe = jnp.where(dz_actual > 0.0, dz_actual, 1.0)
    dT_dt_raw = (
        sw_down[..., jnp.newaxis] * frac_absorbed / (rho_0 * c_sw * dz_safe)
    )
    return jnp.where(dz_actual > 0.0, dT_dt_raw, 0.0)
