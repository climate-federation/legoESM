"""Direct controls for the committed WW3 input/metric adapter."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from scripts.validate.validate_wave_ww3 import configure
from scripts.validate.wave_fidelity.case_io import moments, read_config
from scripts.validate.wave_fidelity.compare import errors, circular_errors


def test_metrics_detect_a_planted_error_and_zero_reference():
    reference = np.asarray([0., 1., 2.])
    prediction = reference.copy()
    prediction[1] += 1.
    score = errors(prediction, reference, np.ones(3, dtype=bool))
    assert score["max_absolute"] == 1.
    assert score["relative_l2"] > 0.
    assert score["max_absolute_index"] == [1]
    assert errors(prediction, np.zeros(3), np.ones(3, dtype=bool))["relative_l2"] is None
    assert circular_errors(np.asarray([1.]), np.asarray([359.]), True)["max_absolute"] == 2.


def test_common_moments_include_lowest_frequency_bin():
    spectrum = np.zeros((2, 3), dtype=float)
    spectrum[0, 0] = 1.
    result = moments(spectrum, np.asarray([0., 180.]), np.asarray([.05, .055, .0605]))
    assert result["swh"] > 0
    assert np.isfinite(result["mwp"])


@pytest.mark.parametrize("name", ["tp1_1", "tp1_2", "tp1_4", "tp2_1"])
def test_oracle_case_configs_select_explicit_no_source_dynamics(name):
    path = Path(__file__).parents[3]/"scripts/validate/wave_fidelity/configs"/(name+".json")
    config = read_config(path)
    case = dict(x=np.arange(3.), y=np.arange(3.), theta=np.asarray([0., 180.]),
                freq=np.asarray([.05, .055, .0605]), depth=np.ones((3, 3))*50,
                initial=np.ones((3, 3, 2, 3)), wet=np.ones((3, 3), bool),
                boundary_mask=np.zeros((3, 3), bool))
    cfg, state, boundary, fixed = configure(case, config, jnp.float32)
    cfg.validate()
    assert cfg.network.parameterization == "none"
    assert cfg.network.use_transport_networks is False
    assert cfg.grid.boundary_mode == "physical"
    assert cfg.grid.coordinate_type == config["coordinate_type"]
    assert boundary.shape == state.spectra.shape
