"""Observation-error model for calibration: the structural-error floor ``P_e``.

Calibration (Gjini, Morzfeld & Watson-Parris) uses an observation-error
covariance ``R_tau = P_tau + P_e``:

* ``P_tau`` is INTERNAL VARIABILITY — how much a statistic moves when only the
  weather changes.  It is measured empirically from a control run chopped into
  windows and scaled ``1/tau`` to the window length.  It is NOT defined here.
* ``P_e`` is the STRUCTURAL ERROR floor — the limit on how accurately this model
  can ever reproduce an observed statistic.  That is what this module defines.

Without a correctly sized ``P_e`` an ensemble-Kalman optimiser will keep pushing
parameters toward their bounds trying to close biases that no parameter can
close.  Sizing it wrongly in the other direction makes a statistic unfittable.

Why this is a TABLE and not a formula
-------------------------------------
The reference paper sets ``P_e = (0.05 * |reference|)^2`` for every statistic.
That rule is only valid when the statistic has a PHYSICALLY MEANINGFUL ZERO and
its reference sits far from that zero.  Measured on this model's own output,
it fails in two different directions:

* ``tas`` — 5% of 287 K is **14 K**.  Kelvin's zero is arbitrary, so a fraction
  of an absolute temperature is not a temperature error scale; the rule would
  accept essentially any bias.  Too LOOSE.
* ``net_toa`` — 5% of +0.9 W/m^2 is **0.045 W/m^2**, against a measured
  -15.6 W/m^2 bias.  The reference is a small residual of two ~340 W/m^2 terms,
  so a fraction of it is meaningless.  Too TIGHT.

So every statistic carries an EXPLICIT floor spec with a written reason, and a
guard (:func:`validate_floor_spec`) refuses the two configurations that produced
those failures.  See :data:`OBSERVATION_FLOORS`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, NamedTuple

# --- The reference paper's fractional rule, kept for the statistics it suits --
FRACTIONAL_FLOOR = 0.05

# A fractional floor is only meaningful when the reference is a substantial
# fraction of the statistic's characteristic scale.  ``net_toa`` fails this at
# 0.9 / 340 = 0.003.  Every simple (non-residual) statistic passes at 1.0.
MIN_REFERENCE_TO_SCALE_RATIO = 0.1

# --- Structural floor for tas [K]: DERIVED, not asserted --------------------
# The paper's 5% convention is KEPT; what changes is the temperature scale it
# multiplies.  5% of an ABSOLUTE temperature is meaningless (arbitrary zero), so
# it is applied instead to the observed SPATIAL standard deviation of the tas
# climatology — a temperature DIFFERENCE, which has a genuine physical zero.
#
#   ERA5   spatial std of tas climatology = 14.101 K  ->  5% = 0.705 K
#   MERRA2 spatial std of tas climatology = 14.461 K  ->  5% = 0.723 K
#
# Two independent reanalyses agree to 2.5%, so the scale is not an artifact of
# one product.  Rounded to 0.7 K: the 5% is a convention, not a 3-digit
# measurement, and implying more precision would be false.
#
# Sanity ordering (checked, and it holds): this floor must EXCEED the internal
# variability that P_tau models separately, or it would be inert.  Observed
# interannual std of global-mean tas is 0.34 K (ERA5) / 0.21 K (MERRA2), and
# the ERA5-MERRA2 observational spread is 0.077 K.  So
#   observational uncertainty 0.08 K  <  internal variability 0.34 K
#                                     <  structural floor 0.70 K
# which is the sensible ordering.
TAS_STRUCTURAL_SIGMA_K = 0.7

FloorKind = Literal["fractional", "absolute", "excluded"]


class FloorSpec(NamedTuple):
    """How one statistic's structural-error floor is set, and why.

    ``kind``:

    * ``"fractional"`` — ``sigma = value * |reference|`` (the paper's rule).
      Requires ``physical_zero`` and a ``characteristic_scale``.
    * ``"absolute"`` — ``sigma = value``, in ``units``.  Use when the
      statistic's zero is arbitrary (temperature in K) or when it is a small
      residual of large terms.
    * ``"excluded"`` — the statistic is NOT part of the vector the optimiser
      fits.  It may still be reported as a diagnostic.

    ``physical_zero`` — does zero mean "none of this quantity"?  True for
    fluxes, precipitation, water paths.  False for absolute temperature.
    A fractional floor on a statistic whose zero is arbitrary is REJECTED.

    ``characteristic_scale`` — the magnitude of the LARGEST term entering the
    statistic, in ``units``.  For a directly measured statistic this is just
    its own magnitude (ratio 1.0).  For a residual like
    ``net_toa = rsdt - rsut - rlut`` it is the ~340 W/m^2 insolation, which is
    what exposes the reference as near-zero on its own scale.

    ``reason`` — REQUIRED.  One line saying why this treatment, so the next
    person can see at a glance and does not "fix" a deliberate choice.
    """

    kind: FloorKind
    value: float | None
    units: str
    physical_zero: bool
    characteristic_scale: float | None
    reason: str


OBSERVATION_FLOORS: dict[str, FloorSpec] = {
    # --- the paper's 5% rule applies cleanly: physical zero, reference far
    #     from it, not a residual of larger terms ---
    "rsut": FloorSpec(
        "fractional", FRACTIONAL_FLOOR, "W/m2", True, 98.856,
        "reflected SW: zero means no reflection (physical); reference "
        "98.9 W/m2 is its own scale, so the 5% rule is well posed"),
    "rlut": FloorSpec(
        "fractional", FRACTIONAL_FLOOR, "W/m2", True, 240.513,
        "outgoing LW: zero means no emission (physical); reference "
        "240.5 W/m2 is its own scale, so the 5% rule is well posed"),
    "pr": FloorSpec(
        "fractional", FRACTIONAL_FLOOR, "mm/day", True, 3.086,
        "precipitation: zero means no rain (physical); reference "
        "3.09 mm/day is its own scale, so the 5% rule is well posed"),
    "prw": FloorSpec(
        "fractional", FRACTIONAL_FLOOR, "kg/m2", True, 24.273,
        "column water vapour: zero means a dry column (physical); reference "
        "24.3 kg/m2 is its own scale, so the 5% rule is well posed"),

    # --- FIX 2: absolute floor, because Kelvin's zero is arbitrary ---
    "tas": FloorSpec(
        "absolute", TAS_STRUCTURAL_SIGMA_K, "K", False, None,
        "near-surface air temperature in K has an ARBITRARY zero, so a "
        "fraction of 287 K (= 14 K) is not a temperature-error scale and "
        "would accept any bias; floor = 5% of the OBSERVED SPATIAL STD of the "
        "tas climatology (14.1 K ERA5 / 14.5 K MERRA2), a temperature "
        "difference with a physical zero - keeps the paper's 5% convention, "
        "fixes the scale it multiplies"),

    # --- FIX 1: excluded from the fitted vector, kept as a diagnostic ---
    "net_toa": FloorSpec(
        "excluded", None, "W/m2", True, 340.382,
        "net_toa = rsdt - rsut - rlut is an EXACT linear combination of rsut "
        "and rlut (both already fitted) plus rsdt. Fitting it double-counts "
        "the SW and LW errors and makes R near-singular for EKI. The ONLY new "
        "information it carries is the rsdt bias (measured -3.08 W/m2 vs "
        "CERES) - and rsdt is PRESCRIBED insolation that no tuned parameter "
        "can move, so fitting net_toa would push cloud parameters to "
        "compensate a solar-forcing error. REPORTED AS A DIAGNOSTIC, never "
        "fitted. Do NOT 'restore' this - the exclusion is deliberate."),
}

# Statistics reported to a reader but never fitted.  Kept explicit so a reader
# of the scorecard still sees the headline energy-balance number.
DIAGNOSTIC_ONLY: tuple[str, ...] = tuple(
    k for k, v in OBSERVATION_FLOORS.items() if v.kind == "excluded")


def validate_floor_spec(name: str, spec: FloorSpec) -> FloorSpec:
    """Reject the floor configurations that have already burned us once.

    Raises on:

    * a fractional floor for a statistic whose zero is not physical
      (the ``tas`` trap — too loose);
    * a fractional floor whose reference is a small fraction of the
      statistic's characteristic scale (the ``net_toa`` trap — too tight);
    * a missing reason, a non-positive floor, or a value on an excluded entry.
    """
    if not spec.reason or not spec.reason.strip():
        raise ValueError(f"floor spec for {name!r} has no reason; every entry "
                         "must say why it is treated this way")
    if spec.kind == "excluded":
        if spec.value is not None:
            raise ValueError(
                f"{name!r} is excluded from the observation vector but carries "
                f"a floor value {spec.value!r}; excluded entries take None")
        return spec
    if spec.value is None or spec.value <= 0.0:
        raise ValueError(
            f"floor value for {name!r} must be positive, got {spec.value!r}")
    if spec.kind == "absolute":
        return spec
    if spec.kind != "fractional":
        raise ValueError(
            f"unknown floor kind {spec.kind!r} for {name!r}; expected "
            "'fractional', 'absolute' or 'excluded'")
    # --- fractional-only guards ---
    if not spec.physical_zero:
        raise ValueError(
            f"{name!r} uses a FRACTIONAL floor but its zero is not physical. "
            "A fraction of a value measured from an arbitrary zero (e.g. "
            "temperature in K) is not an error scale - it is meaninglessly "
            "loose. Use kind='absolute' with a floor in the statistic's own "
            "units.")
    if spec.characteristic_scale is None or spec.characteristic_scale <= 0.0:
        raise ValueError(
            f"{name!r} uses a FRACTIONAL floor and must declare a positive "
            f"characteristic_scale, got {spec.characteristic_scale!r}")
    return spec


def check_reference_against_scale(name: str, spec: FloorSpec,
                                  reference: float) -> None:
    """Fail loudly when a fractional floor meets a near-zero reference.

    This is the ``net_toa`` trap: a statistic that is a small residual of much
    larger terms has a reference close to zero on its own scale, so a fraction
    of it is absurdly tight.  Checked against the DECLARED
    ``characteristic_scale`` rather than against the reference itself, because
    the reference is exactly the quantity that has gone small.
    """
    if spec.kind != "fractional":
        return
    scale = spec.characteristic_scale
    ratio = abs(reference) / scale
    if ratio < MIN_REFERENCE_TO_SCALE_RATIO:
        raise ValueError(
            f"{name!r} uses a FRACTIONAL floor but its reference "
            f"({reference:g} {spec.units}) is only {ratio:.4f} of its "
            f"characteristic scale ({scale:g} {spec.units}), below the "
            f"{MIN_REFERENCE_TO_SCALE_RATIO} limit. A fraction of a near-zero "
            "residual is a meaninglessly TIGHT floor - the optimiser would "
            "chase it to the parameter bounds. Use kind='absolute', or exclude "
            "the statistic if it is a combination of ones already fitted.")


def validate_all_floors(
    floors: Mapping[str, FloorSpec] | None = None,
    references: Mapping[str, float] | None = None,
) -> None:
    """Validate the whole table; with ``references``, also run the scale check."""
    floors = OBSERVATION_FLOORS if floors is None else floors
    for name, spec in floors.items():
        validate_floor_spec(name, spec)
        if references is not None and name in references:
            check_reference_against_scale(name, spec, references[name])


def observation_statistics(
    floors: Mapping[str, FloorSpec] | None = None) -> tuple[str, ...]:
    """The statistics the optimiser actually fits (excluded ones removed)."""
    floors = OBSERVATION_FLOORS if floors is None else floors
    return tuple(k for k, v in floors.items() if v.kind != "excluded")


def is_fitted(name: str,
              floors: Mapping[str, FloorSpec] | None = None) -> bool:
    """Whether ``name`` enters the fitted observation vector."""
    floors = OBSERVATION_FLOORS if floors is None else floors
    if name not in floors:
        raise KeyError(f"no floor spec for statistic {name!r}; add one to "
                       "OBSERVATION_FLOORS before using it")
    return floors[name].kind != "excluded"


def structural_sigma(name: str, reference: float,
                     floors: Mapping[str, FloorSpec] | None = None) -> float:
    """Structural-error standard deviation for ``name`` [statistic's units].

    Raises for an excluded statistic — asking for its floor means it is about
    to be fitted, which is precisely what the exclusion forbids.
    """
    floors = OBSERVATION_FLOORS if floors is None else floors
    if name not in floors:
        raise KeyError(f"no floor spec for statistic {name!r}; add one to "
                       "OBSERVATION_FLOORS before using it")
    spec = validate_floor_spec(name, floors[name])
    if spec.kind == "excluded":
        raise ValueError(
            f"{name!r} is excluded from the observation vector and has no "
            f"structural floor. Reason: {spec.reason}")
    if spec.kind == "absolute":
        return float(spec.value)
    check_reference_against_scale(name, spec, reference)
    return float(spec.value) * abs(float(reference))


def structural_variance(name: str, reference: float,
                        floors: Mapping[str, FloorSpec] | None = None) -> float:
    """``P_e`` diagonal entry for ``name`` (sigma squared)."""
    s = structural_sigma(name, reference, floors)
    return s * s


def build_structural_covariance_diagonal(
    references: Mapping[str, float],
    floors: Mapping[str, FloorSpec] | None = None,
) -> dict[str, float]:
    """``P_e`` as a diagonal, keyed by statistic — excluded ones omitted.

    ``references`` maps statistic -> observed value.  A reference missing for a
    FITTED statistic raises rather than silently dropping that row from ``R``.
    """
    floors = OBSERVATION_FLOORS if floors is None else floors
    out: dict[str, float] = {}
    for name in observation_statistics(floors):
        if name not in references:
            raise KeyError(
                f"no reference value for fitted statistic {name!r}; every "
                "fitted statistic needs one or R would be missing a row")
        out[name] = structural_variance(name, references[name], floors)
    return out


# Fail at import time if the shipped table is self-inconsistent.
validate_all_floors()
