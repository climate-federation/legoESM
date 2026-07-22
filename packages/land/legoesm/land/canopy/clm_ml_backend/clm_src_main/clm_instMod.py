"""
JAX translation of clm_instMod Fortran module.

Instances and definitions of all CLM component data types.
Provides module-level singletons for every state container used
throughout the CLM and CLMml physics, together with initialization
and restart entry points.

Original Fortran module: clm_instMod
Fortran lines 1-105
"""

from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.CanopyStateType import (
    canopystate_type,
)  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.CanopyStateType import (
    init_allocate_from_bounds as _canopystate_Init,
)
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.EnergyFluxType import (
    energyflux_type,
)  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.EnergyFluxType import (
    init as _energyflux_Init,
)
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.FrictionVelocityMod import (
    frictionvel_type,
)  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.FrictionVelocityMod import (
    init_friction_velocity as _frictionvel_Init,
)
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SoilStateInitTimeConstMod import SoilStateInitTimeConst  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SoilStateType import (
    create_soilstate as _soilstate_Init,
)
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SoilStateType import (
    soilstate_type,
)  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SolarAbsorbedType import init as _solarabs_Init
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SolarAbsorbedType import solarabs_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SurfaceAlbedoMod import SurfaceAlbedoInitTimeConst  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SurfaceAlbedoType import (
    init_allocate as _surfalb_Init,
)
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SurfaceAlbedoType import (
    surfalb_type,
)  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.TemperatureType import (
    Init as _temperature_Init,
)
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.TemperatureType import (
    temperature_type,
)  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterDiagnosticBulkType import (
    Init as _waterdiagnosticbulk_Init,
)
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterDiagnosticBulkType import (
    waterdiagnosticbulk_type,
)  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterFluxBulkType import (
    Init as _waterfluxbulk_Init,
)
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterFluxBulkType import (
    waterfluxbulk_type,
)  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterStateBulkType import (
    Init as _waterstatebulk_Init,
)
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterStateBulkType import (
    waterstatebulk_type,
)  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterType import Init as _water_Init
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterType import water_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.atm2lndType import Init as _atm2lnd_Init
from legoesm.land.canopy.clm_ml_backend.clm_src_main.atm2lndType import atm2lnd_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.initVerticalMod import initVertical  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.ncdio_pio import file_desc_t  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.wateratm2lndBulkType import (
    Init as _wateratm2lndbulk_Init,
)
from legoesm.land.canopy.clm_ml_backend.clm_src_main.wateratm2lndBulkType import (
    wateratm2lndbulk_type,
)  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import (
    create_mlcanopy as _mlcanopy_Init,
)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import (
    mlcanopy_type,
)  # noqa: F401  # CLMml

# ---------------------------------------------------------------------------
# Module-level singletons (Fortran lines 40-55)
# These are initialised to None and populated by clm_instInit.
# Mirrors Fortran public module-level instances.
# ---------------------------------------------------------------------------

atm2lnd_inst: atm2lnd_type = None  # type: ignore[assignment]
wateratm2lndbulk_inst: wateratm2lndbulk_type = None  # type: ignore[assignment]
soilstate_inst: soilstate_type = None  # type: ignore[assignment]
water_inst: water_type = None  # type: ignore[assignment]
waterstatebulk_inst: waterstatebulk_type = None  # type: ignore[assignment]
waterfluxbulk_inst: waterfluxbulk_type = None  # type: ignore[assignment]
waterdiagnosticbulk_inst: waterdiagnosticbulk_type = None  # type: ignore[assignment]
canopystate_inst: canopystate_type = None  # type: ignore[assignment]
temperature_inst: temperature_type = None  # type: ignore[assignment]
energyflux_inst: energyflux_type = None  # type: ignore[assignment]
frictionvel_inst: frictionvel_type = None  # type: ignore[assignment]
surfalb_inst: surfalb_type = None  # type: ignore[assignment]
solarabs_inst: solarabs_type = None  # type: ignore[assignment]
mlcanopy_inst: mlcanopy_type = None  # type: ignore[assignment]  # CLMml


# ---------------------------------------------------------------------------
# Public: initialize all component instances
# ---------------------------------------------------------------------------


def clm_instInit(bounds: bounds_type) -> None:
    """
    Initialize all public CLM component-type instances.

    Mirrors Fortran subroutine ``clm_instInit`` (lines 62-83).

    Initialization order matches the Fortran source exactly:

    1. ``initVertical`` — vertical grid setup.
    2. ``atm2lnd_inst.Init`` — atmosphere-to-land forcing.
    3. ``wateratm2lndbulk_inst.Init`` — bulk atm-to-land water fluxes.
    4. ``soilstate_inst.Init`` — soil state allocation.
    5. ``SoilStateInitTimeConst`` — soil hydraulic/thermal constants.
    6. ``water_inst.Init`` — snow water.
    7. ``waterstatebulk_inst.Init`` — bulk water state.
    8. ``waterfluxbulk_inst.Init`` — bulk water fluxes.
    9. ``waterdiagnosticbulk_inst.Init`` — bulk water diagnostics.
    10. ``canopystate_inst.Init`` — canopy state.
    11. ``temperature_inst.Init`` — temperature state.
    12. ``energyflux_inst.Init`` — energy fluxes.
    13. ``frictionvel_inst.Init`` — friction velocity.
    14. ``surfalb_inst.Init`` — surface albedo.
    15. ``solarabs_inst.Init`` — solar absorption.
    16. ``SurfaceAlbedoInitTimeConst`` — albedo lookup tables.
    17. ``mlcanopy_inst.Init`` — multilayer canopy (CLMml).

    Args:
        bounds: Decomposition bounds for the local MPI task, supplying
            ``begg``, ``endg``, ``begc``, ``endc``, ``begp``, and
            ``endp``.
    """
    global atm2lnd_inst, wateratm2lndbulk_inst, soilstate_inst
    global water_inst, waterstatebulk_inst, waterfluxbulk_inst
    global waterdiagnosticbulk_inst, canopystate_inst, temperature_inst
    global energyflux_inst, frictionvel_inst, surfalb_inst, solarabs_inst
    global mlcanopy_inst

    # Fortran line 65
    initVertical(bounds)

    # Fortran lines 66-82
    atm2lnd_inst = _atm2lnd_Init(bounds)
    wateratm2lndbulk_inst = _wateratm2lndbulk_Init(bounds)
    soilstate_inst = _soilstate_Init(bounds)
    soilstate_inst = SoilStateInitTimeConst(bounds, soilstate_inst)
    water_inst = _water_Init(bounds)
    waterstatebulk_inst = _waterstatebulk_Init(bounds)
    waterfluxbulk_inst = _waterfluxbulk_Init(bounds)
    waterdiagnosticbulk_inst = _waterdiagnosticbulk_Init(bounds)
    canopystate_inst = _canopystate_Init(bounds)  # type: ignore[arg-type]
    temperature_inst = _temperature_Init(bounds)  # type: ignore[arg-type]
    energyflux_inst = _energyflux_Init(bounds)  # type: ignore[arg-type]
    frictionvel_inst = _frictionvel_Init(bounds)  # type: ignore[arg-type]
    surfalb_inst = _surfalb_Init(bounds)  # type: ignore[arg-type]
    solarabs_inst = _solarabs_Init(bounds)  # type: ignore[arg-type]
    SurfaceAlbedoInitTimeConst(bounds)
    mlcanopy_inst = _mlcanopy_Init(bounds.begp, bounds.endp)  # CLMml


# ---------------------------------------------------------------------------
# Public: restart (define / write / read)
# ---------------------------------------------------------------------------


def clm_instRest(
    bounds: bounds_type,
    ncid: file_desc_t,
    flag: str,
) -> None:
    """
    Define, write, or read the CLM restart file.

    Mirrors Fortran subroutine ``clm_instRest`` (lines 85-102).

    In the Fortran source, most component restart calls are commented
    out; only ``mlcanopy_inst.restart`` is active. The commented calls
    are preserved below as Python comments at matching positions.

    Args:
        bounds: Decomposition bounds for the local MPI task.
        ncid: NetCDF file descriptor (PIO).
            Fortran: ``type(file_desc_t), intent(inout) :: ncid``.
        flag: Restart operation selector. One of ``'define'``,
            ``'write'``, or ``'read'``.
            Fortran: ``character(len=*), intent(in) :: flag``.
    """
    # Commented-out restart calls (Fortran lines 97-99):
    # atm2lnd_inst.restart(bounds, ncid, flag=flag)
    # soilstate_inst.restart(bounds, ncid, flag=flag)
    # canopystate_inst.restart(bounds, ncid, flag=flag)
    # mlcanopy_inst.restart(bounds, ncid, flag=flag)     # CLMml — Fortran line 101 (not yet implemented)
