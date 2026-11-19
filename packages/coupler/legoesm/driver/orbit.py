"""The run's one Earth orbit, shared by every solar-geometry consumer.

``experiment_orbit(cfg)`` is the single resolver: radiation (RRTMGP and gray),
the daily-mean zenith/insolation forcing and the MPAS land zenith all read the
orbit it returns, so no component carries its own orbital constants.

``None`` means the circular-orbit (Cooper) approximation of
``radiation.solar`` (``orbital_insolation=False``).  Otherwise the orbit is
present day (``radiation.solar.earth_orbit``), the Berger (1978) orbit of
``orbit_year`` (CESM ``shr_orb_params`` port), or user-set elements.  Calendar
convention (PMIP4): the vernal equinox stays at calendar day 80.5 (21 March)
for every orbit; ``radiation.solar`` anchors the solar longitude there.
"""

from __future__ import annotations

from legoesm.atmosphere.physics.radiation.solar import (
    OrbitalParameters, earth_orbit,
)

# shr_orb_params validity: about +-1 Myr around its 1950 AD reference epoch.
_BERGER_REF_YEAR = 1950
_BERGER_SPAN_YEARS = 1_000_000
# radiation.solar's solar longitude is Berger's cubic series in e (no Kepler
# solve): max true-anomaly error 0.008 deg at e=0.1, 0.13 deg at 0.2, 0.68 deg
# at 0.3 (vs exact Kepler).  Earth stays below ~0.07; larger e needs a solver.
_MAX_ECCENTRICITY = 0.1
_ORBIT_ELEMENT_FIELDS = (
    "orbit_eccentricity", "orbit_obliquity_deg", "orbit_long_perihelion_deg")


def orbit_from_year(year: int) -> OrbitalParameters:
    """Berger (1978) orbit for calendar ``year`` AD (negative = BC).

    CESM's ``mvelp`` is the longitude of perihelion minus 180 deg (its
    ``shr_orb_params`` adds the 180 back), so the radiation convention is
    ``mvelp + 180``.
    """
    _check_year(year)
    from legoesm.land.canopy.clm_ml_backend.clm_share.shr_orb_mod import (
        shr_orb_params,
    )
    eccen, obliq_deg, mvelp_deg, *_ = shr_orb_params(year)
    return OrbitalParameters(
        eccentricity=float(eccen),
        obliquity_deg=float(obliq_deg),
        long_perihelion_deg=(float(mvelp_deg) + 180.0) % 360.0,
    )


def _check_year(year) -> None:
    if isinstance(year, bool) or not isinstance(year, int):
        raise ValueError(f"orbit_year={year!r} must be an integer year AD")
    if abs(year - _BERGER_REF_YEAR) > _BERGER_SPAN_YEARS:
        raise ValueError(
            f"orbit_year={year} outside the Berger (1978) validity range "
            f"{_BERGER_REF_YEAR}+-{_BERGER_SPAN_YEARS}; set the orbital "
            "elements directly instead")


def validate_orbit_fields(cfg) -> None:
    """Refuse contradictory or out-of-range orbit settings."""
    year = getattr(cfg, "orbit_year", None)
    elements = {f: getattr(cfg, f, None) for f in _ORBIT_ELEMENT_FIELDS}
    set_elements = [f for f, v in elements.items() if v is not None]
    if (year is not None or set_elements) and not getattr(
            cfg, "orbital_insolation", False):
        raise ValueError(
            "orbit_year / orbit_* elements need orbital_insolation=True "
            "(--orbital-insolation); without it the circular approximation "
            "would silently ignore them")
    if year is not None and set_elements:
        raise ValueError(
            f"orbit_year and {set_elements} are both set; choose one")
    if year is not None:
        _check_year(year)
    ecc = elements["orbit_eccentricity"]
    if ecc is not None and not 0.0 <= ecc <= _MAX_ECCENTRICITY:
        raise ValueError(
            f"orbit_eccentricity={ecc} must be in [0, {_MAX_ECCENTRICITY}] "
            "(the solar-longitude series is not accurate beyond)")
    obl = elements["orbit_obliquity_deg"]
    if obl is not None and not 0.0 <= obl <= 180.0:
        raise ValueError(f"orbit_obliquity_deg={obl} must be in [0, 180]")
    per = elements["orbit_long_perihelion_deg"]
    if per is not None and not 0.0 <= per < 360.0:
        raise ValueError(
            f"orbit_long_perihelion_deg={per} must be in [0, 360)")


def experiment_orbit(cfg) -> OrbitalParameters | None:
    """The one orbit of this run (``None`` = circular approximation)."""
    validate_orbit_fields(cfg)
    if not getattr(cfg, "orbital_insolation", False):
        return None
    year = getattr(cfg, "orbit_year", None)
    if year is not None:
        return orbit_from_year(year)
    base = earth_orbit()
    ecc = getattr(cfg, "orbit_eccentricity", None)
    obl = getattr(cfg, "orbit_obliquity_deg", None)
    per = getattr(cfg, "orbit_long_perihelion_deg", None)
    return OrbitalParameters(
        eccentricity=base.eccentricity if ecc is None else float(ecc),
        obliquity_deg=base.obliquity_deg if obl is None else float(obl),
        long_perihelion_deg=(base.long_perihelion_deg if per is None
                             else float(per)),
    )
