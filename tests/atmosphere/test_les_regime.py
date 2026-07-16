"""Unit tests for :mod:`legoesm.atmosphere.dynamics.les.les_regime`.

Stage-5 LES regime selection: CAPE-based shallow/deep dispatch (raise on
unknown), per-regime resolution, and the stretched-grid validity guard.
"""

from __future__ import annotations

import math

import pytest
from legoesm.atmosphere.dynamics.les.les_regime import (
    LESRegimeConfig,
    LESResolutionConfig,
    les_resolution_for_column,
    les_resolution_for_regime,
    select_les_regime,
    validate_les_resolution,
    validate_regime_config,
)


def test_select_regime_by_cape():
    assert select_les_regime(0.0) == "shallow"
    assert select_les_regime(500.0) == "shallow"
    assert select_les_regime(2500.0) == "deep"
    # Exactly at threshold -> deep (>=).
    assert select_les_regime(1000.0) == "deep"


def test_select_regime_custom_threshold():
    cfg = LESRegimeConfig(deep_cape_threshold_J_kg=2000.0)
    assert select_les_regime(1500.0, cfg) == "shallow"
    assert select_les_regime(2500.0, cfg) == "deep"


def test_select_regime_rejects_non_finite_cape():
    with pytest.raises(ValueError, match="non-finite CAPE"):
        select_les_regime(float("nan"))
    with pytest.raises(ValueError, match="non-finite CAPE"):
        select_les_regime(math.inf)


def test_resolution_for_regime():
    shallow = les_resolution_for_regime("shallow")
    deep = les_resolution_for_regime("deep")
    # Deep is coarser dx, bigger domain/top than shallow.
    assert deep.dx_m > shallow.dx_m
    assert deep.domain_top_m > shallow.domain_top_m


def test_default_domains_contain_regime_eddies():
    """Horizontal domain (nx·dx) must be wide enough for the regime."""
    shallow = les_resolution_for_regime("shallow")
    deep = les_resolution_for_regime("deep")
    assert shallow.nx * shallow.dx_m >= 6_000.0   # shallow cumulus ~6 km+
    assert deep.nx * deep.dx_m >= 50_000.0         # deep convection ~50 km+


def test_resolution_unknown_regime_raises():
    with pytest.raises(ValueError, match="Unknown LES regime"):
        les_resolution_for_regime("mesoscale")


def test_resolution_for_column_dispatch():
    regime, res = les_resolution_for_column(2500.0)
    assert regime == "deep"
    assert res.domain_top_m == pytest.approx(20000.0)
    regime2, res2 = les_resolution_for_column(100.0)
    assert regime2 == "shallow"
    assert res2.domain_top_m == pytest.approx(4000.0)


def test_validate_resolution_accepts_defaults():
    validate_les_resolution(les_resolution_for_regime("shallow"))
    validate_les_resolution(les_resolution_for_regime("deep"))


def test_production_les_domain_spans_the_boundary_layer_for_flux_convergence():
    """The PRODUCTION LES domain (``nx·dx``) must be at least as wide as it is tall
    (``domain_top_m``) so the horizontal-mean resolved flux — diagnosed from a single
    final SNAPSHOT (``run_forced_les`` returns the final state) — converges over
    several boundary-layer eddies rather than a single under-resolved one.  The config
    comment claims the domain 'contains several of the regime's largest eddies'; this
    PINS that requirement on the shipped defaults (shallow 6.4 km vs 4 km top = 1.6×;
    deep 51.2 km vs 20 km top = 2.56×) so a future edit shrinking ``nx`` below the
    eddy scale — which would make the single-snapshot diagnosis spatially noisy and
    the LES-informed correction unreliable — fails CI.  (Runtime ``validate_les_
    resolution`` does NOT enforce this: the deliberately-tiny unit-test domains must
    still run; the requirement is on PRODUCTION fidelity, so it is pinned here.)

    Complements ``test_default_domains_contain_regime_eddies`` (which pins the ABSOLUTE
    width 6/50 km): this pins the SCALE-RELATIVE width ≥ depth ratio (catching a deeper
    ``domain_top`` not matched by a wider domain, which the absolute floor would miss)
    plus horizontal isotropy."""
    for regime in ("shallow", "deep"):
        res = les_resolution_for_regime(regime)
        domain_width_m = res.nx * res.dx_m
        assert domain_width_m >= res.domain_top_m, (
            f"{regime} LES domain {domain_width_m} m is narrower than its depth "
            f"{res.domain_top_m} m — cannot contain a boundary-layer eddy")
        assert res.nx == res.ny            # an isotropic horizontal domain


def test_validate_resolution_rejects_nonpositive():
    base = les_resolution_for_regime("shallow")
    with pytest.raises(ValueError, match="dx_m must be"):
        validate_les_resolution(base._replace(dx_m=0.0))
    with pytest.raises(ValueError, match="nlev must be"):
        validate_les_resolution(base._replace(nlev=-1))


def test_validate_resolution_rejects_no_stretch_room():
    # dz_sfc * nlev >= domain_top -> no room to stretch.
    bad = LESResolutionConfig(
        dx_m=50.0, nx=32, ny=32, nlev=100, domain_top_m=1000.0, dz_sfc_m=20.0
    )  # 20*100 = 2000 >= 1000
    with pytest.raises(ValueError, match="room to stretch"):
        validate_les_resolution(bad)


def test_validate_resolution_rejects_equality_boundary():
    # dz_sfc * nlev == domain_top is also invalid (strict < required).
    bad = LESResolutionConfig(
        dx_m=50.0, nx=32, ny=32, nlev=100, domain_top_m=2000.0, dz_sfc_m=20.0
    )  # 20*100 == 2000
    with pytest.raises(ValueError, match="room to stretch"):
        validate_les_resolution(bad)


def test_validate_regime_config_rejects_negative_threshold():
    with pytest.raises(ValueError, match="threshold"):
        validate_regime_config(LESRegimeConfig(deep_cape_threshold_J_kg=-1.0))


def test_resolution_for_column_validates():
    # A config whose selected regime box is invalid must fail loudly.
    bad_deep = LESResolutionConfig(
        dx_m=200.0, nx=128, ny=128, nlev=80, domain_top_m=20000.0, dz_sfc_m=300.0
    )  # 300*80 = 24000 >= 20000
    cfg = LESRegimeConfig(deep=bad_deep)
    with pytest.raises(ValueError, match="room to stretch"):
        les_resolution_for_column(2500.0, cfg)
    # A negative CAPE threshold is also rejected by the composing entry point.
    with pytest.raises(ValueError, match="threshold"):
        les_resolution_for_column(100.0, LESRegimeConfig(deep_cape_threshold_J_kg=-5.0))
