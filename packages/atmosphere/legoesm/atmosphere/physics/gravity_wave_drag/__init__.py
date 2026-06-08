"""Gravity wave drag parameterization for legoESM.

Six backends are available, with increasing complexity:
1. **Rayleigh**: simple Rayleigh friction drag
2. **Lindzen (1981)**: smoothed orographic GWD
3. **McFarlane (1987)**: orographic GWD with launch flux control
4. **Hines (1997)**: Doppler-spread non-orographic GWD
5. **Prognostic Spectral**: multi-azimuthal prognostic spectral GWD
6. **ML Emulator**: neural network GWD surrogate (Equinox MLP)

All produce the same `GWDOutput` interface.

Use `make_gwd_physics()` to create a physics function matching
your dynamical core's `step_with_physics` signature.

Example
-------
>>> from legoesm.atmosphere.physics.gravity_wave_drag import (
...     GravityWaveDragConfig, make_gwd_physics,
... )
>>> config = GravityWaveDragConfig(scheme="rayleigh")
>>> physics_fn = make_gwd_physics(config, model_type="hydrostatic", dt=300.0)
>>> new_state = model.step_with_physics(state, dt, physics_fn=physics_fn)
"""

from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    GravityWaveDragConfig,
    RayleighConfig,
    LindzenConfig,
    McFarlaneConfig,
    HinesConfig,
    PrognosticSpectralConfig,
    E3SMCAMConfig,
    E3SMOrographicConfig,
    E3SMFrontalConfig,
    E3SMBeresConfig,
    GWDMLEmulatorConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
    prognostic_spectral_gwd,
)
from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import (
    e3sm_cam_gwd,
    gw_prof,
    gw_oro_src,
    gw_cm_src,
    gw_beres_src,
    build_stand_in_mfcc,
    gw_drag_prof,
    gw_ediff,
    gw_diff_tridiag_coeffs,
    gw_diff_tend,
    gw_taucd_net,
    momentum_energy_conservation,
    newtonian_alpha_profile,
)
from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import (
    ml_gwd,
    GWDEmulator,
)
from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
    make_gwd_physics,
)
