"""LES-tuned turbulence defaults: SCM applies by default, AMIP opts in.

Gate for `configs/tuned/turbulence_les.yaml` + `les_tuned.py`: the tuned values
must actually reach the active scheme's sub-config by default, an opt-out must
restore the library defaults, every value must sit inside its `__param_spec__`
bounds, and the same YAML must be a valid AMIP `--params` file.
"""
import pathlib

import pytest
import yaml

from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.turbulence.les_tuned import (
    _read_tuned_yaml_text,
    apply_les_tuned_turbulence,
    load_les_tuned_overrides,
    scm_turbulence_config,
)

# Schemes present in the committed YAML (CLUBB is intentionally absent).
_TUNED_SCHEMES = ["smagorinsky", "louis", "tke", "mynn25", "clubb_lite",
                  "holtslag_boville", "ysu", "edmf"]


def test_yaml_exists_and_parses():
    # packaged resource resolves (raises if missing); loader validates fully
    _read_tuned_yaml_text(None)
    grouped = load_les_tuned_overrides()
    # 8 scheme config classes, CLUBB absent
    assert len(grouped) == 8
    assert "CLUBBConfig" not in grouped


@pytest.mark.parametrize("scheme", _TUNED_SCHEMES)
def test_scm_default_applies_tuned_values(scheme):
    """The SCM builder changes the active scheme's sub-config by default."""
    tuned = scm_turbulence_config(scheme)                 # les_tuned=True default
    plain = scm_turbulence_config(scheme, les_tuned=False)
    assert tuned.scheme == scheme
    sub_tuned = getattr(tuned, scheme)
    sub_plain = getattr(plain, scheme)
    # at least one field must have moved off the library default
    moved = [f for f in sub_tuned._fields
             if isinstance(getattr(sub_tuned, f), float)
             and getattr(sub_tuned, f) != getattr(sub_plain, f)]
    assert moved, f"{scheme}: tuned config identical to library default"


def test_opt_out_is_library_default():
    plain = scm_turbulence_config("ysu", les_tuned=False)
    assert plain == TurbulenceConfig(scheme="ysu")


def test_inactive_schemes_untouched():
    """Only the active scheme's sub-config is spliced; others stay default."""
    tuned = scm_turbulence_config("ysu")
    base = TurbulenceConfig(scheme="ysu")
    assert tuned.louis == base.louis        # a non-active scheme is unchanged
    assert tuned.smagorinsky == base.smagorinsky


def test_clubb_returned_unchanged():
    """CLUBB has no tuned entry yet -> no error, config unchanged."""
    turb = TurbulenceConfig(scheme="clubb")
    assert apply_les_tuned_turbulence(turb) == turb


def test_every_value_in_spec_bounds():
    """Each tuned value must sit inside its __param_spec__ (lo, hi)."""
    from legoesm.atmosphere.physics.turbulence import config as turb_config
    grouped = load_les_tuned_overrides()
    for cls_name, fields in grouped.items():
        spec = turb_config.__param_spec__[cls_name]["params"]
        for field, value in fields.items():
            lo, hi = spec[field]["bounds"]
            assert lo <= value <= hi, (
                f"{cls_name}.{field} = {value} outside [{lo}, {hi}]")


def test_yaml_is_valid_amip_params_format():
    """Same file must be a flat `qualified_name: value` map (the --params shape)."""
    raw = yaml.safe_load(_read_tuned_yaml_text(None)[0])
    assert isinstance(raw, dict) and raw
    for k, v in raw.items():
        assert k.startswith("atm.turb.") and isinstance(v, (int, float)), (k, v)


def test_typod_class_key_raises(tmp_path):
    """A class-name typo must fail loudly, not silently leave the scheme untuned."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("atm.turb.YSUConfg.Pr_t: 0.6\n")     # 'YSUConfg' typo
    with pytest.raises(ValueError, match="unknown turbulence config class"):
        load_les_tuned_overrides(str(bad))


def test_unknown_field_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("atm.turb.YSUConfig.not_a_field: 0.6\n")
    with pytest.raises(ValueError, match="no spec'd param"):
        load_les_tuned_overrides(str(bad))


def test_out_of_bounds_value_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("atm.turb.YSUConfig.Pr_t: 99.0\n")    # bounds (0.33, 3.0)
    with pytest.raises(ValueError, match="outside spec bounds"):
        load_les_tuned_overrides(str(bad))


def test_non_numeric_value_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("atm.turb.YSUConfig.Pr_t: true\n")    # bool coerces to 1.0 -> reject
    with pytest.raises(ValueError, match="is not a number"):
        load_les_tuned_overrides(str(bad))


def test_scheme_none_is_noop():
    """scheme='none' (valid, no sub-config) must return unchanged, not raise."""
    turb = scm_turbulence_config("none")          # les_tuned default True
    assert turb == TurbulenceConfig(scheme="none")


def test_write_active_scheme_params_is_single_scheme_slice(tmp_path):
    # P1 #1: AMIP opt-in needs only the active scheme's params (the full file
    # aborts on unselected schemes). The slice must be a valid --params doc with
    # exactly one turbulence class.
    from legoesm.atmosphere.physics.turbulence.les_tuned import (
        write_active_scheme_params)
    out = tmp_path / "louis_slice.yaml"
    n = write_active_scheme_params("louis", str(out))
    assert n > 0
    doc = yaml.safe_load(out.read_text())
    classes = {k.rsplit(".", 1)[0] for k in doc}
    assert len(classes) == 1 and next(iter(classes)) == "atm.turb.LouisConfig"
    for k, v in doc.items():
        assert k.startswith("atm.turb.") and isinstance(v, (int, float))


def test_write_active_scheme_params_raises_for_untuned_scheme(tmp_path):
    from legoesm.atmosphere.physics.turbulence.les_tuned import (
        write_active_scheme_params)
    with pytest.raises(ValueError, match="no tuned entry"):
        write_active_scheme_params("clubb", str(tmp_path / "x.yaml"))
