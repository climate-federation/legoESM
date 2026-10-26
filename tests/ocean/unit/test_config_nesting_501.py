"""#501 config-grouping invariants — DynBottomDragConfig (the first nested group).

Locks the byte-identical flat↔nested contract: the storage is nested
(``config.bottom_drag.bottom_drag_r``) but the flat construction / YAML / kwarg
interface is unchanged via ``from_flat`` + ``flat_fields``.  When the next group
nests, extend the field lists here.
"""
import pytest

from legoesm.ocean.state import (
    BarotropicConfig,
    DynBottomDragConfig,
    LatLonCGridOceanConfig,
    RuntimeChecksConfig,
    PolarFilterConfig,
)

_BD = ("bottom_drag_r", "bottom_drag_bbl_thickness", "bottom_drag_bg_velocity")
_BT = ("n_barotropic_substeps", "bebt", "barotropic_solver",
       "barotropic_implicit_preconditioner", "rigid_lid_cg_tol",
       "barotropic_slow_forcing_ab2")  # representative barotropic fields
_RC = ("enable_runtime_checks", "max_abs_eta_m", "temperature_min_c",
       "temperature_max_c", "salinity_min_psu", "salinity_max_psu")


def test_bottom_drag_is_nested_not_flat():
    """The grouped fields moved OUT of the top-level config INTO the sub-config."""
    top = set(LatLonCGridOceanConfig._fields)
    assert "bottom_drag" in top
    for f in _BD:
        assert f not in top, f"{f} must be nested, not a top-level field"
        assert f in DynBottomDragConfig._fields
    assert isinstance(LatLonCGridOceanConfig().bottom_drag, DynBottomDragConfig)


def test_from_flat_distributes_grouped_fields():
    c = LatLonCGridOceanConfig.from_flat(bottom_drag_r=2.5e-3,
                                         bottom_drag_bbl_thickness=10.0)
    assert c.bottom_drag.bottom_drag_r == 2.5e-3
    assert c.bottom_drag.bottom_drag_bbl_thickness == 10.0
    assert c.bottom_drag.bottom_drag_bg_velocity == 0.0  # default


def test_from_flat_passes_ungrouped_and_empty():
    # ungrouped flat fields still construct flat; empty -> all defaults.
    c = LatLonCGridOceanConfig.from_flat(A_h=3.0e4, eos="wright")
    assert c.lateral_viscosity.A_h == 3.0e4 and c.eos == "wright"
    assert LatLonCGridOceanConfig.from_flat() == LatLonCGridOceanConfig()


def test_from_flat_nested_passthrough():
    bd = DynBottomDragConfig(bottom_drag_r=9.0)
    assert LatLonCGridOceanConfig.from_flat(bottom_drag=bd).bottom_drag is bd


def test_from_flat_unknown_field_is_loud():
    with pytest.raises(TypeError):
        LatLonCGridOceanConfig.from_flat(bottom_drag_typo=1.0)


def test_flat_fields_is_one_to_one_with_pre_grouping_set():
    ff = LatLonCGridOceanConfig.flat_fields()
    # nested field names are NOT flat keys; their members ARE
    assert "bottom_drag" not in ff and "barotropic" not in ff
    assert "runtime_checks" not in ff
    assert "lateral_viscosity" not in ff
    assert "polar_filter" not in ff
    for f in (*_BD, *_BT):
        assert f in ff
    # every other (non-grouped) top-level field is unchanged
    for f in set(LatLonCGridOceanConfig._fields) - {"bottom_drag", "barotropic",
                                                     "runtime_checks", "lateral_viscosity",
                                                     "polar_filter"}:
        assert f in ff


def test_yaml_flat_key_still_routes_to_nested():
    """The public flat ``ocean.bottom_drag_r`` YAML key survives the grouping."""
    from legoesm.ocean.config import OceanExperimentConfig  # noqa: PLC0415
    cfg = OceanExperimentConfig({"grid": {"type": "latlon_cgrid"},
                           "ocean": {"bottom_drag_r": 2.5e-3}}).to_ocean_config()
    assert cfg.bottom_drag.bottom_drag_r == 2.5e-3


def test_yaml_unknown_flat_key_still_rejected():
    from legoesm.ocean.config import OceanExperimentConfig  # noqa: PLC0415
    with pytest.raises(ValueError, match="unknown ocean config field"):
        OceanExperimentConfig({"grid": {"type": "latlon_cgrid"},
                         "ocean": {"bottom_drag_typo": 1.0}}).to_ocean_config()


def test_checkpoint_dict_roundtrip_preserves_nested_drag():
    from legoesm.ocean.config import (  # noqa: PLC0415
        ocean_config_from_dict,
        ocean_config_to_dict,
    )
    c = LatLonCGridOceanConfig.from_flat(bottom_drag_r=1.7e-3,
                                         bottom_drag_bg_velocity=0.1)
    back = ocean_config_from_dict(ocean_config_to_dict(c))
    assert back.bottom_drag.bottom_drag_r == 1.7e-3
    assert back.bottom_drag.bottom_drag_bg_velocity == 0.1


def test_nested_replace_swaps_subconfig():
    c = LatLonCGridOceanConfig.from_flat(bottom_drag_r=1.0)
    c2 = c._replace(bottom_drag=c.bottom_drag._replace(bottom_drag_r=0.0))
    assert c2.bottom_drag.bottom_drag_r == 0.0
    assert c.bottom_drag.bottom_drag_r == 1.0  # original untouched


def test_old_flat_checkpoint_decodes_into_nested():
    """A PRE-grouping checkpoint (flat ``bottom_drag_r`` at the top level) decodes
    INTO the nested sub-config via from_flat, not silently dropped to the default."""
    from legoesm.ocean.config import ocean_config_from_dict  # noqa: PLC0415
    tag = f"{LatLonCGridOceanConfig.__module__}:LatLonCGridOceanConfig"
    old = {"__type__": tag, "bottom_drag_r": 3.3e-3}  # legacy flat checkpoint shape
    cfg = ocean_config_from_dict(old)
    assert cfg.bottom_drag.bottom_drag_r == 3.3e-3


def test_from_flat_both_flat_and_nested_is_loud():
    """Passing BOTH flat ``bottom_drag_r`` and nested ``bottom_drag=`` is a
    conflict -> loud TypeError (no silent precedence)."""
    with pytest.raises(TypeError):
        LatLonCGridOceanConfig.from_flat(
            bottom_drag_r=1.0, bottom_drag=DynBottomDragConfig())


# ----------------------------------------------------------- BarotropicConfig (PR2)

def test_barotropic_is_nested_not_flat():
    top = set(LatLonCGridOceanConfig._fields)
    assert "barotropic" in top
    # 37 = 22 (PR2 baseline) + 15 fields added since (pin had drifted; found
    # red at 28 before barotropic_een_seed, #1226 item 4, made it 29; then a
    # pre-existing +1 on this branch put it at 30; +barotropic_reconcile_target
    # (NEMO dyn_spg_ts N6) makes it 31; +barotropic_after_reconcile (NEMO
    # mlf_baro_corr, stpmlf.F90:754-765) makes it 32; the seed, PGF, and
    # continuity arithmetic selectors make 35; the literal EEN coefficient
    # selector (dynspg_ts.F90:1517-1565) makes 36; literal time-mean transport
    # accumulation (dynspg_ts.F90:734-737,999-1000) makes 37).
    assert len(BarotropicConfig._fields) == 37
    for f in _BT:
        assert f not in top, f"{f} must be nested, not a top-level field"
        assert f in BarotropicConfig._fields
    assert isinstance(LatLonCGridOceanConfig().barotropic, BarotropicConfig)


def test_barotropic_from_flat_distributes():
    c = LatLonCGridOceanConfig.from_flat(barotropic_solver="implicit_cn",
                                         n_barotropic_substeps=40, bebt=0.3)
    assert c.barotropic.barotropic_solver == "implicit_cn"
    assert c.barotropic.n_barotropic_substeps == 40
    assert c.barotropic.bebt == 0.3


def test_barotropic_flat_fields_and_both_groups_coexist():
    ff = LatLonCGridOceanConfig.flat_fields()
    assert "barotropic" not in ff and "bottom_drag" not in ff
    for f in (*_BT, *_BD):
        assert f in ff
    # the two nested groups distribute together in one from_flat call
    c = LatLonCGridOceanConfig.from_flat(barotropic_solver="rigid_lid",
                                         bottom_drag_r=2.5e-3)
    assert c.barotropic.barotropic_solver == "rigid_lid"
    assert c.bottom_drag.bottom_drag_r == 2.5e-3


def test_barotropic_yaml_flat_key_routes_to_nested():
    from legoesm.ocean.config import OceanExperimentConfig  # noqa: PLC0415
    cfg = OceanExperimentConfig({"grid": {"type": "latlon_cgrid"},
                                 "ocean": {"barotropic_solver": "implicit_cn"}}).to_ocean_config()
    assert cfg.barotropic.barotropic_solver == "implicit_cn"


def test_barotropic_old_flat_checkpoint_decodes_into_nested():
    from legoesm.ocean.config import ocean_config_from_dict  # noqa: PLC0415
    tag = f"{LatLonCGridOceanConfig.__module__}:LatLonCGridOceanConfig"
    cfg = ocean_config_from_dict({"__type__": tag, "barotropic_solver": "rigid_lid",
                                  "n_barotropic_substeps": 12})
    assert cfg.barotropic.barotropic_solver == "rigid_lid"
    assert cfg.barotropic.n_barotropic_substeps == 12


# ------------------------------------------------------- RuntimeChecksConfig (PR3)

def test_runtime_checks_is_nested_not_flat():
    top = set(LatLonCGridOceanConfig._fields)
    assert "runtime_checks" in top
    assert len(RuntimeChecksConfig._fields) == 6
    for f in _RC:
        assert f not in top, f"{f} must be nested, not a top-level field"
        assert f in RuntimeChecksConfig._fields
    assert isinstance(LatLonCGridOceanConfig().runtime_checks, RuntimeChecksConfig)


def test_min_water_column_m_deliberately_stays_flat():
    """``min_water_column_m`` is a physical wet-cell floor read by grid-agnostic
    shared code (``ocean_conservation_fixer``) that also runs on the cube
    ``OceanConfig``; nesting it LatLon-only would break that uniform read, so it
    is intentionally NOT part of RuntimeChecksConfig.  Lock that decision."""
    assert "min_water_column_m" in set(LatLonCGridOceanConfig._fields)
    assert "min_water_column_m" not in RuntimeChecksConfig._fields
    assert LatLonCGridOceanConfig().min_water_column_m == 0.5


def test_runtime_checks_from_flat_distributes():
    c = LatLonCGridOceanConfig.from_flat(enable_runtime_checks=True,
                                         temperature_max_c=40.0,
                                         salinity_min_psu=2.0)
    assert c.runtime_checks.enable_runtime_checks is True
    assert c.runtime_checks.temperature_max_c == 40.0
    assert c.runtime_checks.salinity_min_psu == 2.0
    assert c.runtime_checks.max_abs_eta_m == 1.0e4  # default


def test_runtime_checks_flat_fields_all_three_groups_coexist():
    ff = LatLonCGridOceanConfig.flat_fields()
    assert "runtime_checks" not in ff
    assert "lateral_viscosity" not in ff
    assert "polar_filter" not in ff
    for f in (*_RC, *_BT, *_BD):
        assert f in ff
    assert "min_water_column_m" in ff  # ungrouped, stays a flat key
    # all three nested groups distribute together in one from_flat call
    c = LatLonCGridOceanConfig.from_flat(enable_runtime_checks=True,
                                         barotropic_solver="rigid_lid",
                                         bottom_drag_r=2.5e-3)
    assert c.runtime_checks.enable_runtime_checks is True
    assert c.barotropic.barotropic_solver == "rigid_lid"
    assert c.bottom_drag.bottom_drag_r == 2.5e-3


def test_runtime_checks_yaml_flat_key_routes_to_nested():
    from legoesm.ocean.config import OceanExperimentConfig  # noqa: PLC0415
    cfg = OceanExperimentConfig({"grid": {"type": "latlon_cgrid"},
                                 "ocean": {"temperature_max_c": 42.0}}).to_ocean_config()
    assert cfg.runtime_checks.temperature_max_c == 42.0


def test_runtime_checks_old_flat_checkpoint_decodes_into_nested():
    from legoesm.ocean.config import ocean_config_from_dict  # noqa: PLC0415
    tag = f"{LatLonCGridOceanConfig.__module__}:LatLonCGridOceanConfig"
    cfg = ocean_config_from_dict({"__type__": tag, "enable_runtime_checks": True,
                                  "salinity_max_psu": 48.0})
    assert cfg.runtime_checks.enable_runtime_checks is True
    assert cfg.runtime_checks.salinity_max_psu == 48.0


def test_flat_get_resolves_grouped_and_ungrouped_by_flat_name():
    """flat_get is the read-side inverse of from_flat for ALL groups + the
    ungrouped flat fields — reflective code that iterates flat names keeps
    working regardless of which group (if any) a field landed in."""
    c = LatLonCGridOceanConfig.from_flat(max_abs_eta_m=5.0e3, temperature_min_c=-3.0,
                                         barotropic_solver="rigid_lid",
                                         bottom_drag_r=1.0e-3, min_water_column_m=0.7)
    for name in _RC:
        assert c.flat_get(name) == getattr(c.runtime_checks, name)
    assert c.flat_get("max_abs_eta_m") == 5.0e3
    assert c.flat_get("temperature_min_c") == -3.0
    assert c.flat_get("barotropic_solver") == "rigid_lid"          # other group
    assert c.flat_get("bottom_drag_r") == 1.0e-3                   # other group
    assert c.flat_get("min_water_column_m") == 0.7                 # ungrouped, flat


def test_replace_flat_distributes_grouped_overrides():
    """replace_flat is the _replace analog of from_flat (#501) — flat override
    names (recipe scheme presets / CLI splat) distribute into their sub-configs."""
    c = LatLonCGridOceanConfig.from_flat()
    c2 = c.replace_flat(A_h=2.0e4, C_smag=0.1, barotropic_solver="rigid_lid",
                        bottom_drag_r=1.0e-3, eos="wright")
    assert c2.lateral_viscosity.A_h == 2.0e4 and c2.lateral_viscosity.C_smag == 0.1
    assert c2.barotropic.barotropic_solver == "rigid_lid"
    assert c2.bottom_drag.bottom_drag_r == 1.0e-3
    assert c2.eos == "wright"  # ungrouped stays flat
    assert c.lateral_viscosity.A_h == 1.0e4  # original untouched


# ------------------------------------------------------------- PolarFilterConfig

_PF = ("use_polar_filter", "polar_filter_cutoff_lat_deg",
       "polar_filter_max_wave_speed", "polar_filter_safety_factor")


def test_polar_filter_is_nested_not_flat():
    top = set(LatLonCGridOceanConfig._fields)
    assert "polar_filter" in top
    assert len(PolarFilterConfig._fields) == 4
    for f in _PF:
        assert f not in top and f in PolarFilterConfig._fields
    assert isinstance(LatLonCGridOceanConfig().polar_filter, PolarFilterConfig)


def test_polar_filter_from_flat_distributes_and_flat_get():
    c = LatLonCGridOceanConfig.from_flat(use_polar_filter=True,
                                         polar_filter_cutoff_lat_deg=55.0)
    assert c.polar_filter.use_polar_filter is True
    assert c.polar_filter.polar_filter_cutoff_lat_deg == 55.0
    assert c.polar_filter.polar_filter_safety_factor == 0.85  # default
    for f in _PF:
        assert c.flat_get(f) == getattr(c.polar_filter, f)


def test_polar_filter_yaml_and_old_checkpoint():
    from legoesm.ocean.config import (OceanExperimentConfig,  # noqa: PLC0415
                                      ocean_config_from_dict)
    cfg = OceanExperimentConfig({"grid": {"type": "latlon_cgrid"},
                                 "ocean": {"use_polar_filter": True}}).to_ocean_config()
    assert cfg.polar_filter.use_polar_filter is True
    tag = f"{LatLonCGridOceanConfig.__module__}:LatLonCGridOceanConfig"
    back = ocean_config_from_dict({"__type__": tag, "polar_filter_cutoff_lat_deg": 50.0})
    assert back.polar_filter.polar_filter_cutoff_lat_deg == 50.0
