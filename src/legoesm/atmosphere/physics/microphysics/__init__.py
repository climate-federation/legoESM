"""Atmospheric microphysics parameterization for legoESM.

Seven backends are available, spanning warm-rain through mixed-phase to ML:
1. **Kessler**: Warm-rain one-moment (Kessler 1969)
2. **Sundqvist**: Large-scale diagnostic condensation (Sundqvist 1989)
3. **Seifert-Beheng**: Two-moment warm rain (Seifert & Beheng 2001)
4. **Morrison**: Double-moment ice+liquid (Morrison et al. 2005)
5. **Thompson**: Hybrid moment with graupel (Thompson et al. 2008)
6. **P3**: Predicted Particle Properties single-category ice (Morrison & Milbrandt 2015)
7. **ML Emulator**: Equinox MLP surrogate

All produce the same `MicrophysicsOutput` interface.

Use `make_microphysics_physics()` to create a physics function matching
your dynamical core's `step_with_physics` signature.

Example
-------
>>> from legoesm.atmosphere.physics.microphysics import (
...     MicrophysicsConfig, make_microphysics_physics,
... )
>>> config = MicrophysicsConfig(scheme="kessler")
>>> physics_fn = make_microphysics_physics(config, model_type="nonhydrostatic", dt=1.0)
"""

from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
    KesslerConfig,
    SundqvistConfig,
    SeifertBehengConfig,
    MorrisonConfig,
    ThompsonConfig,
    P3Config,
    MLEmulatorConfig,
)
from legoesm.atmosphere.physics.microphysics.output import (
    MicrophysicsOutput,
    HydrometeorState,
    make_zero_hydrometeors,
    make_zero_output,
    sedimentation_tendency,
)
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
from legoesm.atmosphere.physics.microphysics.p3 import p3_microphysics
from legoesm.atmosphere.physics.microphysics.ml_emulator import (
    ml_microphysics,
    MicrophysicsEmulator,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
