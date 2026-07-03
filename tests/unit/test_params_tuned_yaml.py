"""config/cmip/params_tuned.yaml — the tuned-parameterization calibration file
(File 2 of the experiment+parameters split).

Guards that every key is a real parameter-registry qualified name
(``scheme_key.field``) and every value is within that parameter's
``__param_spec__`` bounds — so a hand edit or a dropped-in trained set cannot
drift to a typo'd key or an out-of-bounds value.  Imports the param registry
(run under the param-test sbatch, not bare login)."""

from __future__ import annotations

from pathlib import Path

import yaml

from legoesm.training.param_collector import build_registry

REPO = Path(__file__).resolve().parents[2]
PARAMS = REPO / "config" / "cmip" / "params_tuned.yaml"


def test_params_tuned_keys_are_registry_params_in_bounds():
    doc = yaml.safe_load(PARAMS.read_text())
    assert isinstance(doc, dict) and doc, "params_tuned.yaml is empty/not a mapping"
    reg = {m.qualified_name: m for m in build_registry()}
    for key, value in doc.items():
        assert key in reg, (
            f"{key!r} is not a registry parameter (expected 'scheme_key.field'; "
            f"e.g. 'atm.clouds.CloudConfig.q_c_diagnostic')")
        lo, hi = reg[key].bounds
        assert lo <= value <= hi, f"{key}={value} out of __param_spec__ bounds [{lo},{hi}]"


def test_tuned_cloud_condensate_present():
    doc = yaml.safe_load(PARAMS.read_text())
    assert doc["atm.clouds.CloudConfig.q_c_diagnostic"] == 3.0e-4
