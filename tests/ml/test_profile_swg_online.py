"""Direct test for the SWG online profiling entry point."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "bench" / "profile_swg_online.py"
SPEC = importlib.util.spec_from_file_location("profile_swg_online", SCRIPT)
profile = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(profile)


def test_profile_metadata_is_real_network_and_no_training():
    metadata = profile.profile_metadata()
    network = metadata["network"]
    assert network["parameter_count"] == 7696
    assert network["receptive_field_cells"] == 11
    assert network["output"] == "boundary-zero native F-point potential"
    assert metadata["optimizer_factory"] == "legoesm.ml.training.create_optimizer"
    assert metadata["training_performed"] is False
