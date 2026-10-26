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

# The 8 FLAT closures (params land directly on the sub-config). CLUBB is tuned
# too but nested (CLUBBParams) — covered by dedicated tests below, not here.
_TUNED_SCHEMES = ["smagorinsky", "louis", "tke", "mynn25", "clubb_lite",
                  "holtslag_boville", "ysu", "edmf"]


def test_yaml_exists_and_parses():
    # packaged resource resolves (raises if missing); loader validates fully
    _read_tuned_yaml_text(None)
    grouped = load_les_tuned_overrides()
    # 8 flat scheme classes + CLUBB's nested CLUBBParams = 9 override groups
    assert len(grouped) == 9
    assert "CLUBBParams" in grouped        # CLUBB now tuned (nested)
    assert len(grouped["CLUBBParams"]) >= 30


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


def test_clubb_applies_nested_tuned_prognostic():
    """CLUBB is tuned: its coefficients live in the nested CLUBBParams, and
    the tuned config must be prognostic (that is how they were fit)."""
    tuned = scm_turbulence_config("clubb")
    plain = scm_turbulence_config("clubb", les_tuned=False)
    assert plain.clubb is None                 # library default: opt-in / off
    assert tuned.clubb is not None and tuned.clubb.prognostic is True
    dflt = type(tuned.clubb.params)()
    moved = [f for f in tuned.clubb.params._fields
             if getattr(tuned.clubb.params, f) != getattr(dflt, f)]
    assert len(moved) >= 30, moved


def test_every_value_in_spec_bounds():
    """Each tuned value must sit inside its __param_spec__ (lo, hi)."""
    from legoesm.atmosphere.physics.turbulence import config as turb_config
    from legoesm.atmosphere.physics.turbulence import clubb as clubb_mod
    # CLUBBParams is spec'd in clubb.py, the flat classes in config.py -- the
    # loader merges both, so the bounds check must too.
    merged = {**turb_config.__param_spec__, **clubb_mod.__param_spec__}
    grouped = load_les_tuned_overrides()
    for cls_name, fields in grouped.items():
        spec = merged[cls_name]["params"]
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
    """A scheme absent from the (custom) YAML must raise, not emit nothing."""
    from legoesm.atmosphere.physics.turbulence.les_tuned import (
        write_active_scheme_params)
    only_ysu = tmp_path / "only_ysu.yaml"
    only_ysu.write_text("atm.turb.YSUConfig.Pr_t: 0.6\n")
    with pytest.raises(ValueError, match="no tuned entry"):
        write_active_scheme_params("louis", str(tmp_path / "x.yaml"),
                                   path=str(only_ysu))


def test_clubb_amip_slice_refused(tmp_path):
    """CLUBB opt-in via a --params slice alone is UNSAFE (cannot set prognostic)
    and must raise, not emit a slice that runs diagnostic physics."""
    from legoesm.atmosphere.physics.turbulence.les_tuned import (
        write_active_scheme_params)
    with pytest.raises(ValueError, match="CLUBB cannot be opted into"):
        write_active_scheme_params("clubb", str(tmp_path / "clubb.yaml"))


def test_clubb_diagnostic_config_refused():
    """Prognostic-fit CLUBB coefficients must not be applied to a diagnostic config."""
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    turb = TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(prognostic=False))
    with pytest.raises(ValueError, match="prognostic"):
        apply_les_tuned_turbulence(turb)


def test_clubb_missing_clubbparams_raises(tmp_path):
    """CLUBB selected but the YAML has no CLUBBParams -> raise, not silent default."""
    only_ysu = tmp_path / "only_ysu.yaml"
    only_ysu.write_text("atm.turb.YSUConfig.Pr_t: 0.6\n")
    turb = TurbulenceConfig(scheme="clubb")
    with pytest.raises(ValueError, match="no CLUBBParams entry"):
        apply_les_tuned_turbulence(turb, path=str(only_ysu))
