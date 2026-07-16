"""Atmospheric turbulence / boundary layer parameterization for legoESM.

Seven backends are available, with increasing complexity:
1. **Smagorinsky**: deformation/stability-dependent eddy diffusivity — simplest baseline
2. **Louis (1979)**: stability-dependent diffusion — standard GCM scheme
3. **TKE / MY2.5**: prognostic turbulent kinetic energy closure
4. **CLUBB-lite**: higher-order closure skeleton (delegates to TKE)
5. **Holtslag-Boville**: nonlocal K-profile with counter-gradient correction
6. **YSU**: nonlocal K-profile with entrainment flux at PBL top
7. **EDMF**: eddy-diffusivity mass-flux unified framework

All produce the same `TurbulenceOutput` interface.

Use `make_turbulence_physics()` to create a physics function matching
your dynamical core's `step_with_physics` signature.

Example
-------
>>> from legoesm.atmosphere.physics.turbulence import (
...     TurbulenceConfig, make_turbulence_physics,
... )
>>> config = TurbulenceConfig(scheme="smagorinsky")
>>> physics_fn = make_turbulence_physics(config, model_type="hydrostatic", dt=300.0)
>>> new_state = model.step_with_physics(state, dt, physics_fn=physics_fn)
"""

from legoesm.atmosphere.physics.turbulence.config import (
    TurbulenceConfig,
    SurfaceLayerConfig,
    SmagorinskyConfig,
    LouisConfig,
    TKEConfig,
    CLUBBLiteConfig,
    HoltslagBovilleConfig,
    YSUConfig,
    TurbulentEDMFConfig,
)
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.smagorinsky import smagorinsky_turbulence
from legoesm.atmosphere.physics.turbulence.louis import louis_turbulence
from legoesm.atmosphere.physics.turbulence.tke import tke_turbulence
from legoesm.atmosphere.physics.turbulence.clubb_lite import clubb_lite_turbulence
from legoesm.atmosphere.physics.turbulence.holtslag_boville import (
    holtslag_boville_turbulence,
    diffuse_theta_with_countergradient,
)
from legoesm.atmosphere.physics.turbulence.ysu import ysu_turbulence
from legoesm.atmosphere.physics.turbulence.edmf import edmf_turbulence
from legoesm.atmosphere.physics.turbulence.pbl_height import (
    PBLHeightConfig,
    compute_bulk_richardson,
    diagnose_pbl_height,
    diagnose_pbl_height_interp,
)
from legoesm.atmosphere.physics.turbulence.integration import (
    make_turbulence_physics,
)
