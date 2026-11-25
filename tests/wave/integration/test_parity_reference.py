"""Parity against a stored NeuralSWM reference trajectory.

The reference arrays are large (a full 37x45x24x30 spectrum) and are not
committed.  Generate them once with the committed probe::

    PYTHONPATH=~/NeuralSWM python scripts/validate/validate_wave_parity.py \
        --emit --neuralswm-root ~/NeuralSWM \
        --checkpoint-dir <orbax-dir> --out /tmp/wave_parity.npz

then run::

    LEGOESM_WAVE_PARITY_REFERENCE=/tmp/wave_parity.npz \
    LEGOESM_WAVE_WEIGHTS=<converted.npz> \
    python -m pytest tests/wave/integration/test_parity_reference.py

The test is skipped when either file is absent: it needs the trained
weights, which are an external artefact.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import pytest

_REFERENCE = os.environ.get("LEGOESM_WAVE_PARITY_REFERENCE")
_WEIGHTS = os.environ.get("LEGOESM_WAVE_WEIGHTS")

pytestmark = pytest.mark.skipif(
    not (_REFERENCE and _WEIGHTS),
    reason="set LEGOESM_WAVE_PARITY_REFERENCE and LEGOESM_WAVE_WEIGHTS to run")


def test_matches_neuralswm_reference():
    script = (Path(__file__).resolve().parents[3]
              / "scripts/validate/validate_wave_parity.py")
    spec = importlib.util.spec_from_file_location("validate_wave_parity", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.compare(argparse.Namespace(
        weights=_WEIGHTS, reference=_REFERENCE))
