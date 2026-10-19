"""Tests for the LES-suite case registry + default catalog."""
from __future__ import annotations

from pathlib import Path

import pytest
from legoesm.atmosphere.les_suite import (
    ANCHOR_CASES,
    MOIST_CASES,
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
    # 4 anchors + 4 further cloudy/moist cases (incl. GABLS3) + 4 fluxes x 3 winds
    # + 26 published full-setup cases = 46
    assert len(cases) == 46
    assert len(list_cases()) == 46
    clear_registry()
    assert len(register_default_catalog(include_published=False)) == 20
    names = {c.name for c in cases}
    assert {"cbl_nieuwstadt", "sbl_gabls1", "bomex_cu", "dycoms_rf01_sc",
            "rico_cu", "dycoms_rf02_sc", "astex_sc"} <= names


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
    for c in ANCHOR_CASES + MOIST_CASES:
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


def test_cooling_rate_validation_and_gabls1_is_cooling_forced():
    with pytest.raises(RegistryError):   # both forcings set
        _mk_case(surface_cooling_rate_K_per_h=0.25).validate()
    for bad in (0.0, -0.25, True):
        with pytest.raises(RegistryError):
            _mk_case(surface_theta_flux_K_m_s=None,
                     surface_cooling_rate_K_per_h=bad).validate()
    _mk_case(surface_theta_flux_K_m_s=None, surface_cooling_rate_K_per_h=0.25).validate()
    by_name = {c.name: c for c in register_default_catalog()}
    gabls = by_name["sbl_gabls1"]
    assert gabls.surface_cooling_rate_K_per_h == 0.25   # Beare et al. (2006)
    assert gabls.surface_theta_flux_K_m_s is None


# The drivers' sponge bases before they read them from the catalog: (fraction of the
# domain height, the driver's default domain height [m]).
_OLD_DRIVER_SPONGE = {
    "sbl_gabls1": (0.75, 400.0),       # run_spectral_sbl / run_les_suite: 0.75 * Lz
    "bomex_cu": (0.75, 3000.0),        # run_bomex_les: 0.75 * args.Lz (main and SDM)
    "dycoms_rf01_sc": (0.85, 1500.0),  # run_dycoms_les: 0.85 * args.Lz
    "dycoms_rf02_sc": (0.85, 1500.0),
    "astex_sc": (0.85, 2500.0),
    "rico_cu": (0.75, 4000.0),         # run_rico_les: 0.75 * args.Lz
    "gabls3": (None, 600.0),           # run_gabls3_les: 600 m
}


def test_catalog_sponge_base_is_what_each_driver_applied_before():
    from legoesm.atmosphere.les_suite.catalog import (
        PUBLISHED_CASES,
        catalog_case,
        dry_shear_buoyancy_grid,
    )
    for name, (frac, lz) in _OLD_DRIVER_SPONGE.items():
        case = catalog_case(name)
        if frac is None:            # an absolute base on a fixed domain (GABLS3)
            assert case.grid.Lz_m == 800.0 and case.sponge_base_m() == lz, name
            continue
        assert case.grid.Lz_m == lz, name
        # bit-identical at the default domain and at any --Lz a run passes
        for h in (lz, 200.0, 1800.0, 3000.0):
            assert case.sponge_base_m(h) == frac * h, (name, h)
    assert catalog_case("cbl_nieuwstadt").sponge_base_m() is None
    for c in dry_shear_buoyancy_grid():   # run_les_suite: 0.75 Lz on the stable points
        want = 0.75 * c.grid.Lz_m if c.regime == "dry_stable" else None
        assert c.sponge_base_m() == want, c.name
    for c in PUBLISHED_CASES:
        assert c.sponge_base_m() == c.setup.sponge_bottom_m, c.name


def test_sponge_base_frac_is_validated():
    base = dict(name="x", regime="dry_stable", core="spectral",
                grid=LESGrid(nx=8, ny=8, nz=8, Lx_m=100.0, Ly_m=100.0, Lz_m=100.0,
                             dt_s=0.1),
                duration_hours=1.0, surface_theta_flux_K_m_s=-0.01,
                geostrophic_wind_m_s=5.0, sgs_variants=("lasd",),
                driver="run_spectral_sbl.py", reference="ref")
    for bad in (0.0, 1.0, -0.2, True):
        with pytest.raises(RegistryError, match="sponge_base_frac"):
            LESCase(**base, sponge_base_frac=bad).validate()
    LESCase(**base, sponge_base_frac=0.75).validate()


def test_every_sponge_driver_case_resolves_a_sponge_and_setups_reject_a_fraction():
    import dataclasses

    from legoesm.atmosphere.les_suite.catalog import (
        ANCHOR_CASES,
        MOIST_CASES,
        PUBLISHED_CASES,
        dry_shear_buoyancy_grid,
    )
    sponge_drivers = {"run_bomex_les.py", "run_rico_les.py", "run_dycoms_les.py",
                      "run_spectral_sbl.py", "run_gabls3_les.py"}
    for c in ANCHOR_CASES + MOIST_CASES + tuple(dry_shear_buoyancy_grid()):
        if c.driver in sponge_drivers:
            assert c.sponge_base_m() is not None, c.name
    with pytest.raises(RegistryError, match="sponge_base_frac"):
        dataclasses.replace(PUBLISHED_CASES[0], sponge_base_frac=0.75).validate()


def test_dycoms_deck_names_map_to_catalog_cases():
    import importlib.util
    import sys
    run = _REPO / "scripts" / "run"
    sys.path.insert(0, str(run))
    spec = importlib.util.spec_from_file_location("run_dycoms_les",
                                                  run / "run_dycoms_les.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from legoesm.atmosphere.les_suite.catalog import catalog_case
    assert mod._CATALOG_NAME == {"dycoms": "dycoms_rf01_sc", "rf02": "dycoms_rf02_sc",
                                 "astex": "astex_sc"}
    assert set(mod._CATALOG_NAME) == set(mod._STRATOCUMULUS_CASES)
    for deck, name in mod._CATALOG_NAME.items():
        case = catalog_case(name)
        assert case.driver == "run_dycoms_les.py"
        # the deck's own domain is the catalog case's
        assert case.grid.Lz_m == mod._STRATOCUMULUS_CASES[deck]["lz_m"]
