"""Training infrastructure for differentiable dycore + WeatherBench.

Provides three training modes:
1. Physics parameter tuning via gradient descent through the dycore
2. Neural GCM: neural network coupled to the dycore for physics
3. SFNO coupled to dycore: spherical Fourier neural operator + dynamics
"""

from legoesm.training.training_driver import (  # noqa: F401
    train_physics_params,
    train_neural_gcm,
    train_sfno_coupled,
)
from legoesm.training.vertical_interp import (  # noqa: F401
    interp_pressure_to_sigma,
)
from legoesm.training.losses import LossConfig, combined_loss  # noqa: F401
from legoesm.training.trainable_params import TrainablePhysicsParams  # noqa: F401
from legoesm.core.param_overrides import apply_param_overrides  # noqa: F401
from legoesm.training.param_collector import (  # noqa: F401
    build_trainable_params,
)
from legoesm.training.trainable_ocean_params import (  # noqa: F401
    GEOMETRIC_TRAINABLE,
    TrainableOceanParams,
)
from legoesm.training.etki import etki_update, run_etki  # noqa: F401
from legoesm.training.aimip_params import (  # noqa: F401
    AIMIP_CLASSICAL_CONSTRAINTS,
    AIMIPClassicalParams,
    make_aimip_classical_spectral_physics,
)
from legoesm.training.dycore_rollout import (  # noqa: F401
    RolloutConfig,
    differentiable_rollout,
    single_day_rollout,
)
from legoesm.training.era5_to_state import (  # noqa: F401
    TrainingERA5Config,
    ERA5Slice,
    era5_to_spectral_carry,
    era5_to_cubedsphere_carry,
)
