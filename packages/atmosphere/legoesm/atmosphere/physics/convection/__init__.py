"""Atmospheric convection parameterization for legoESM.

Five backends are available:
1. **SBM**: Simplified Betts-Miller relaxation (Frierson 2007) — for aquaplanet
2. **DCA**: Deep Convective Adjustment — simplest baseline
3. **Kuo**: Moisture convergence (Kuo 1965/1974)
4. **Mass-Flux**: Prognostic mass-flux (Arakawa-Wu type)
5. **EDMF**: Simplified eddy-diffusivity mass-flux

All produce the same `ConvectionOutput` interface (stateless schemes)
or `(ConvectionOutput, prognostic_var)` tuple (prognostic schemes).

Use `make_convection_physics()` to create a physics function matching
your dynamical core's `step_with_physics` signature.

Example
-------
>>> from legoesm.atmosphere.physics.convection import (
...     ConvectionConfig, make_convection_physics,
... )
>>> config = ConvectionConfig(scheme="sbm")
>>> physics_fn = make_convection_physics(config, model_type="hydrostatic", dt=300.0)
>>> new_state = model.step_with_physics(state, dt, physics_fn=physics_fn)
"""

from legoesm.atmosphere.physics.convection.config import (
    AhmedNeelinDCAConfig,
    ConvectionConfig,
    DCAConfig,
    SBMConfig,
    KuoConfig,
    MassFluxConfig,
    ConvectiveEDMFConfig,
)
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.atmosphere.physics.convection.dca import (
    ahmed_neelin_dca,
    dca_convection,
)
from legoesm.atmosphere.physics.convection.kuo import kuo_convection
from legoesm.atmosphere.physics.convection.mass_flux import (
    edmf_convection,
    mass_flux_convection,
)
from legoesm.atmosphere.physics.convection.integration import (
    make_convection_physics,
)
