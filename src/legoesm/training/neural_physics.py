"""Backward-compatible re-export — moved to legoesm.atmosphere.physics.neural_physics."""

from legoesm.atmosphere.physics.neural_physics import (  # noqa: F401
    NeuralPhysics,
    _pack_column_features,
    _unpack_column_output,
    make_neural_step_unified,
    make_hybrid_step_unified,
)
