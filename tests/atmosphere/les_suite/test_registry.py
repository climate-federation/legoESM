"""Tests for the LES-suite case registry + default catalog."""
from __future__ import annotations

from pathlib import Path

import pytest
from legoesm.atmosphere.les_suite import (
    ANCHOR_CASES,
    REGIMES,
    SGS_CHOICES,
    LESCase,
    LESGrid,
    RegistryError,
    clear_registry,
    dry_shear_buoyancy_grid,
    get_case,
    get_cases_for_regime,
    list_cases,
    list_regimes,
    register_case,
    register_default_catalog,
)

_REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is process-global; isolate every test."""
    clear_registry()
    yield
    clear_registry()


def _mk_grid() -> LESGrid:
    return LESGrid(nx=8, ny=8, nz=16, Lx_m=400.0, Ly_m=400.0, Lz_m=800.0, dt_s=0.5)


def _mk_case(name: str = "c", **kw) -> LESCase:
    base = dict(
        name=name, regime="dry_convective", core="spectral", grid=_mk_grid(),
        duration_hours=1.0, surface_theta_flux_K_m_s=0.06, geostrophic_wind_m_s=None,
        sgs_variants=("lasd",), driver="run_spectral_cbl.py", reference="ref",
        ci_marker="fast", description="",
    )
    base.update(kw)
    return LESCase(**base)


# --- validation -------------------------------------------------------------
def test_grid_rejects_nonpositive():
    with pytest.raises(RegistryError):
        LESGrid(nx=0, ny=8, nz=8, Lx_m=1.0, Ly_m=1.0, Lz_m=1.0, dt_s=1.0).validate()
    with pytest.raises(RegistryError):
        LESGrid(nx=8, ny=8, nz=8, Lx_m=1.0, Ly_m=1.0, Lz_m=1.0, dt_s=-1.0).validate()


def test_grid_label():
    assert _mk_grid().label == "8x8x16"


@pytest.mark.parametrize("bad", [
    dict(regime="tropical_cyclone"),
    dict(core="compressible"),
    dict(ci_marker="whenever"),
    dict(sgs_variants=()),
    dict(sgs_variants=("lasd", "lasd")),
    dict(sgs_variants=("nonexistent_sgs",)),
    dict(duration_hours=0.0),
    dict(geostrophic_wind_m_s=-3.0),
])
def test_case_validate_rejects(bad):
    with pytest.raises(RegistryError):
        _mk_case(**bad).validate()


def test_case_validate_accepts_none_axes():
    # both axis coords may be None (free-convection + interactive surface)
    _mk_case(surface_theta_flux_K_m_s=None, geostrophic_wind_m_s=None).validate()


# --- registry mechanics -----------------------------------------------------
def test_register_get_and_duplicate():
    register_case(_mk_case("a"))
    assert get_case("a").name == "a"
    with pytest.raises(RegistryError):
        register_case(_mk_case("a"))          # duplicate name
    with pytest.raises(RegistryError):
        register_case("not a case")           # type: ignore[arg-type]


def test_get_unknown_raises():
    with pytest.raises(RegistryError):
        get_case("nope")


def test_register_validates_before_storing():
    with pytest.raises(RegistryError):
        register_case(_mk_case("bad", regime="nope"))
    assert not list_cases()                   # nothing stored on validation failure


def test_selectors():
    register_case(_mk_case("conv1", regime="dry_convective"))
    register_case(_mk_case("stab1", regime="dry_stable",
                           surface_theta_flux_K_m_s=-0.01, geostrophic_wind_m_s=8.0,
                           driver="run_spectral_sbl.py"))
    assert [c.name for c in get_cases_for_regime("dry_convective")] == ["conv1"]
    assert [c.name for c in get_cases_for_regime("dry_stable")] == ["stab1"]
    assert get_cases_for_regime("stratocumulus") == []
    assert list_regimes() == ["dry_convective", "dry_stable"]  # canonical order
    assert [c.name for c in list_cases()] == ["conv1", "stab1"]  # sorted
    with pytest.raises(RegistryError):
        get_cases_for_regime("nope")


# --- default catalog --------------------------------------------------------
def test_default_catalog_registers_anchors_and_dry_grid():
    cases = register_default_catalog()
    # 4 anchors + 4 fluxes x 3 winds = 16
    assert len(cases) == 16
    assert len(list_cases()) == 16
    names = {c.name for c in cases}
    assert {"cbl_nieuwstadt", "sbl_gabls1", "bomex_cu", "dycoms_rf01_sc"} <= names


def test_all_default_cases_validate_and_have_unique_names():
    cases = register_default_catalog()
    names = [c.name for c in cases]
    assert len(names) == len(set(names)), "duplicate case names"
    for c in cases:
        c.validate()   # must not raise


def test_anchors_are_real_grid_is_provisional():
    register_default_catalog()
    for anchor in ("cbl_nieuwstadt", "sbl_gabls1", "bomex_cu", "dycoms_rf01_sc"):
        assert get_case(anchor).provisional_axis is False
    for c in list_cases():
        if c.name.startswith("dry_grid_"):
            assert c.provisional_axis is True


def test_anchor_drivers_exist_on_disk():
    for c in ANCHOR_CASES:
        p = _REPO / "scripts" / "run" / c.driver
        assert p.is_file(), f"{c.name} references missing driver {c.driver}"


def test_anchor_sgs_and_ci_within_choices():
    for c in ANCHOR_CASES:
        assert set(c.sgs_variants) <= set(SGS_CHOICES)
        assert c.regime in REGIMES


# --- dry grid generator -----------------------------------------------------
def test_dry_grid_shape_and_regime_classification():
    grid = dry_shear_buoyancy_grid(fluxes_K_m_s=(-0.01, 0.0, 0.05),
                                   winds_m_s=(0.0, 6.0))
    assert len(grid) == 6
    by_name = {c.name: c for c in grid}
    # flux<0 -> stable ; ~0 -> neutral ; >0 -> convective
    assert by_name["dry_grid_b0s0"].regime == "dry_stable"
    assert by_name["dry_grid_b1s0"].regime == "dry_neutral"
    assert by_name["dry_grid_b2s0"].regime == "dry_convective"
    # wind=0 -> geostrophic None; wind>0 -> set
    assert by_name["dry_grid_b2s0"].geostrophic_wind_m_s is None
    assert by_name["dry_grid_b2s1"].geostrophic_wind_m_s == 6.0
    # stable -> SBL driver, else CBL driver
    assert by_name["dry_grid_b0s0"].driver == "run_spectral_sbl.py"
    assert by_name["dry_grid_b2s0"].driver == "run_spectral_cbl.py"


def test_dry_grid_flags_sheared_convective_driver_gap():
    grid = dry_shear_buoyancy_grid(fluxes_K_m_s=(0.05,), winds_m_s=(8.0,))
    assert "needs a unified dry driver" in grid[0].description


def test_dry_grid_rejects_empty_axes():
    with pytest.raises(ValueError):
        dry_shear_buoyancy_grid(fluxes_K_m_s=(), winds_m_s=(1.0,))
