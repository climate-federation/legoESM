#!/usr/bin/env python
"""Convert a NeuralSWM Orbax checkpoint to the flat NPZ the wave package loads.

NeuralSWM stores the trained MLP model as an Orbax ``PyTreeCheckpointer``
directory.  legoESM does not depend on Orbax, so this one-shot converter
rewrites the weights into a flat ``.npz`` keyed by the Equinox attribute
path (``parameterization.mlp.linear0.weight`` etc.).  Transposes are
applied here because Flax stores ``(in, out)`` kernels and Equinox stores
``(out, in)`` weights.

Run it in an environment that has both ``orbax-checkpoint`` and JAX (the
NeuralSWM environment works)::

    python scripts/data/convert_neuralswm_checkpoint.py \
        --checkpoint-dir ~/NeuralSWM/examples/well_trained_checkpoints/case_c9_dist_260515 \
        --out ~/NeuralSWM/examples/well_trained_checkpoints/case_c9_dist_260515.npz

The source commit that produced the checkpoint should be recorded in the
PR body; the conversion is deterministic for a given directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def flatten_model_state(model_state: dict) -> dict[str, np.ndarray]:
    """Map the Orbax ``model_state`` tree onto Equinox attribute paths.

    Structurally a pure function so it can be unit-tested with a synthetic
    dictionary (see ``tests/wave/unit/test_checkpoint_conversion.py``).
    """
    out: dict[str, np.ndarray] = {}

    parameterization = model_state["parameterization"]["mlp"]
    for index in range(4):
        layer = parameterization[f"linear{index}"]
        out[f"parameterization.mlp.linear{index}.weight"] = np.transpose(
            np.asarray(layer["kernel"]))
        out[f"parameterization.mlp.linear{index}.bias"] = np.asarray(layer["bias"])
    for index in range(3):
        norm = parameterization[f"norm{index}"]
        out[f"parameterization.mlp.norm{index}.weight"] = np.asarray(norm["scale"])
        out[f"parameterization.mlp.norm{index}.bias"] = np.asarray(norm["bias"])

    for name in ("theta_transport_network", "k_transport_network"):
        layers = model_state[name]["layers"]
        for index in range(len(layers)):
            layer = layers[str(index)]
            if "kernel" in layer:
                weight = np.transpose(np.asarray(layer["kernel"]))
            else:
                weight = np.asarray(layer["scale"])
            out[f"{name}.layers.{index}.weight"] = weight
            out[f"{name}.layers.{index}.bias"] = np.asarray(layer["bias"])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import orbax.checkpoint as ocp

    restored = ocp.PyTreeCheckpointer().restore(args.checkpoint_dir)
    model_state = restored.get("model_state", restored)
    flat = flatten_model_state(model_state)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **flat)
    total = sum(leaf.size for leaf in flat.values())
    print(f"wrote {args.out} ({len(flat)} arrays, {total} values)")


if __name__ == "__main__":
    main()
