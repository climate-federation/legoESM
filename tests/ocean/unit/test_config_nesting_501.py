"""#501 config-grouping invariants — DynBottomDragConfig (the first nested group).

Locks the byte-identical flat↔nested contract: the storage is nested
(``config.bottom_drag.bottom_drag_r``) but the flat construction / YAML / kwarg
interface is unchanged via ``from_flat`` + ``flat_fields``.  When the next group
nests, extend the field lists here.
"""
import pytest

from legoesm.ocean.state import DynBottomDragConfig, LatLonCGridOceanConfig

_BD = ("bottom_drag_r", "bottom_drag_bbl_thickness", "bottom_drag_bg_velocity")


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
    assert c.A_h == 3.0e4 and c.eos == "wright"
    assert LatLonCGridOceanConfig.from_flat() == LatLonCGridOceanConfig()


def test_from_flat_nested_passthrough():
    bd = DynBottomDragConfig(bottom_drag_r=9.0)
    assert LatLonCGridOceanConfig.from_flat(bottom_drag=bd).bottom_drag is bd


def test_from_flat_unknown_field_is_loud():
    with pytest.raises(TypeError):
        LatLonCGridOceanConfig.from_flat(bottom_drag_typo=1.0)


def test_flat_fields_is_one_to_one_with_pre_grouping_set():
    ff = LatLonCGridOceanConfig.flat_fields()
    assert "bottom_drag" not in ff          # the nested field name is NOT a flat key
    for f in _BD:
        assert f in ff                       # its members ARE flat keys
    # every other top-level field is unchanged
    for f in set(LatLonCGridOceanConfig._fields) - {"bottom_drag"}:
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
