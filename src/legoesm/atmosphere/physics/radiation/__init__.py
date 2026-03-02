"""Atmospheric radiation parameterization for legoESM.

Two backends are available:
1. **Gray**: Two-stream gray radiation (Frierson et al. 2006) — self-contained
2. **RRTMGP**: Full correlated-k radiation via jax-rrtmgp — optional dependency

Both produce the same `RadiationOutput` interface.

Use `make_radiation_physics()` to create a physics function matching
your dynamical core's `step_with_physics` signature.

Example
-------
>>> from legoesm.atmosphere.physics.radiation import (
...     RadiationConfig, make_radiation_physics,
... )
>>> config = RadiationConfig(scheme="gray")
>>> physics_fn = make_radiation_physics(config, model_type="hydrostatic")
>>> new_state = model.step_with_physics(state, dt, physics_fn=physics_fn)
"""

from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig,
    RadiationConfig,
    RRTMGPConfig,
)
from legoesm.atmosphere.physics.radiation.output import RadiationOutput
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.solar import (
    cos_zenith_angle,
    daily_mean_insolation,
    perpetual_equinox_insolation,
    solar_declination,
)
from legoesm.atmosphere.physics.radiation.integration import (
    make_radiation_physics,
)
