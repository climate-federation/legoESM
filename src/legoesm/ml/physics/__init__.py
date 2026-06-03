"""Joint ML physics parameterization package."""

from legoesm.ml.physics.data import (
    DEFAULT_SAMPLE_DAYS,
    PhysicsColumnBatch,
    PhysicsTeacherDataset,
    concatenate_physics_teacher_datasets,
    capture_physics_teacher_snapshot,
    generate_amip_physics_teacher_dataset,
)
from legoesm.ml.physics.evaluate import (
    PhysicsEvaluationMetrics,
    evaluate_physics_parameterization,
    predict_physics_parameterization_targets,
)
from legoesm.ml.physics.io import (
    PhysicsNormalizationBundle,
    load_physics_checkpoint,
    load_physics_stats,
    save_physics_checkpoint,
    save_physics_stats,
)
from legoesm.ml.physics.model import (
    PhysicsParameterizationModel,
    pack_physics_parameterization_features,
    pack_physics_parameterization_targets,
    unpack_physics_parameterization_targets,
)
from legoesm.ml.physics.train import (
    PhysicsModelConfig,
    PhysicsTrainingConfig,
    PhysicsTrainingResult,
    train_physics_parameterization,
)

__all__ = [
    "PhysicsColumnBatch",
    "PhysicsEvaluationMetrics",
    "PhysicsModelConfig",
    "PhysicsNormalizationBundle",
    "PhysicsParameterizationModel",
    "PhysicsTeacherDataset",
    "PhysicsTrainingConfig",
    "PhysicsTrainingResult",
    "DEFAULT_SAMPLE_DAYS",
    "concatenate_physics_teacher_datasets",
    "capture_physics_teacher_snapshot",
    "evaluate_physics_parameterization",
    "generate_amip_physics_teacher_dataset",
    "load_physics_checkpoint",
    "load_physics_stats",
    "pack_physics_parameterization_features",
    "pack_physics_parameterization_targets",
    "predict_physics_parameterization_targets",
    "save_physics_checkpoint",
    "save_physics_stats",
    "train_physics_parameterization",
    "unpack_physics_parameterization_targets",
]
