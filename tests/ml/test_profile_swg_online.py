"""Direct test for the SWG online profiling entry point."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "bench" / "profile_swg_online.py"
SPEC = importlib.util.spec_from_file_location("profile_swg_online", SCRIPT)
profile = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(profile)


def test_profile_metadata_is_real_network_and_no_training():
    metadata = profile.profile_metadata()
    network = metadata["network"]
    assert network["parameter_count"] == 7696
    assert network["shifted_slice_layer_count"] == 5
    assert "convolution_count" not in network
    assert network["receptive_field_cells"] == 11
    assert network["output"] == "boundary-zero native F-point potential"
    assert metadata["optimizer_factory"] == "legoesm.ml.training.create_optimizer"
    assert metadata["training_performed"] is False


def test_profile_device_selection_is_explicit_and_fail_closed(monkeypatch):
    monkeypatch.setattr(profile.jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="explicit physical_gpu"):
        profile.run_profile(1, "forward", "template")
    with pytest.raises(ValueError, match="differs"):
        profile.run_profile(1, "forward", "template", physical_gpu=0)
    monkeypatch.setattr(profile.jax, "default_backend", lambda: "cpu")
    with pytest.raises(ValueError, match="only valid"):
        profile.run_profile(1, "forward", "template", physical_gpu=0)
