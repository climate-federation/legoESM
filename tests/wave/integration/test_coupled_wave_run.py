"""Short coupled run through atmosphere, dynamic ocean, KPP, and wave."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.coupled_config import preset_aquaplanet
from legoesm.driver.coupled_esm_driver import CoupledESMDriver
from legoesm.wave.checkpoint import _path_name
from legoesm.wave.config import WaveGSEConfig, WaveTimeConfig
from legoesm.wave.model import WaveModel
from tests.wave.unit.test_wave_coupling import _global_wave_config


def _kpp_ocean_config():
    """The KPP stack must be explicit; the driver never invents one."""
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import (
        EnhancedDiffusionConfig, OceanConvectionConfig)
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        KPPConfig, VerticalMixingConfig)
    from legoesm.ocean.state import LatLonCGridOceanConfig

    return LatLonCGridOceanConfig.from_flat()._replace(
        physics=OceanPhysicsConfig(
            surface_forcing=SurfaceForcingConfig(scheme="none"),
            vertical_mixing=VerticalMixingConfig(
                scheme="kpp", kpp=KPPConfig(K_conv=1.0)),
            lateral_mixing=LateralMixingConfig(scheme="none"),
            bottom_drag=BottomDragConfig(scheme="none"),
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0)),
            shortwave_penetration=None,
        ))


@pytest.mark.parametrize("physics_only", [False, True])
def test_dynamic_ocean_kpp_wave_run(tmp_path, physics_only):
    wave_cfg = _global_wave_config(lon_step=45.0, lat_step=45.0)._replace(
        time=WaveTimeConfig(
            dt_global_s=600, dt_x_s=600, dt_y_s=600,
            dt_th_s=600, dt_k_s=600, dt_stot_s=600, dt_gse_s=0,
        ),
        gse=WaveGSEConfig(enabled=False),
    )
    if physics_only:
        wave_cfg = wave_cfg._replace(network=wave_cfg.network._replace(
            parameterization="none"))
    model = WaveModel(wave_cfg, key=jax.random.key(0))
    leaves, _ = jax.tree_util.tree_flatten_with_path(model)
    weights_path = tmp_path / "wave_weights.npz"
    np.savez(weights_path, **{
        _path_name(path): np.asarray(leaf) for path, leaf in leaves
    })

    atmosphere = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(
            energy_fixer=False, dt=600.0, model_type="hydrostatic",
            discretization="finite_volume",
        ),
        output=OutputConfig(diag_days=1),
        radiation="gray",
        days=1800/86400,  # three events: the checkpoint has odd spatial phase
    )
    coupled = preset_aquaplanet(
        ocean_mode="dynamic", ocean_nlev=3, ocean_dt_s=300.0,
        ocean_config=_kpp_ocean_config(),
        coupling_dt=600.0, wave_mode="active", wave_config=wave_cfg,
        wave_weights_path="" if physics_only else str(weights_path), kpp_langmuir=True,
    )
    driver = CoupledESMDriver(atmosphere, coupled, output_dir=tmp_path / "run")
    driver.setup()
    if physics_only:
        # Source-free transport needs a nonzero initial packet. No weight file
        # is supplied; restart must preserve that weight-free identity too.
        driver._wave_state = driver._wave_state._replace(
            spectra=driver._wave_state.spectra.at[:, 3, 1, 0, :].set(0.01))
        assert driver._wave_weights_hash == ""
    assert driver.run() == "COMPLETED"
    assert int(driver._wave_state.spatial_phase) == 1
    assert driver._wave_state.spectra.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(driver._wave_state.spectra)))
    assert driver._wave_stokes.shape == (8, 16)
    assert bool(jnp.all(jnp.isfinite(driver._wave_stokes)))
    assert bool(jnp.any(driver._wave_stokes > 0))
    assert bool(jnp.all(jnp.isfinite(driver.ocean_state.T.data)))
    assert driver.resolved_component_configs()["wave"] == wave_cfg

    # The next ocean segment consumes this lagged Stokes field.  A resumed run
    # must reconstruct it from the saved spectrum before that segment starts.
    np.savez(tmp_path / "coupled_day_0000.npz",
             **driver.coupled_checkpoint_arrays())
    resumed = CoupledESMDriver(
        atmosphere, coupled, output_dir=tmp_path / "resumed")
    resumed.setup()
    np.savez(tmp_path / "coupled_day_0001.npz",
             **resumed.coupled_checkpoint_arrays())
    resumed.load_coupled_checkpoint(1.0, checkpoint_dir=tmp_path, required=True)
    assert resumed._wave_stokes is None
    resumed.load_coupled_checkpoint(0.0, checkpoint_dir=tmp_path, required=True)
    # The restored spectrum is the file value (exact); the lagged Stokes field
    # is RECOMPUTED from it, and float32 reduction order can differ by one ULP
    # between the restored buffer layout and the live one.
    np.testing.assert_array_equal(
        np.asarray(resumed._wave_state.spectra),
        np.asarray(driver._wave_state.spectra),
    )
    np.testing.assert_allclose(
        np.asarray(resumed._wave_stokes),
        np.asarray(driver._wave_stokes), rtol=1e-6, atol=0,
    )
    np.testing.assert_array_equal(
        resumed._wave_state.spatial_phase, driver._wave_state.spatial_phase)

    # A wave-active checkpoint must not be resumed with the component off:
    # dropping the spectrum would make the restart incomparable.
    coupled_off = coupled._replace(
        wave_mode="off", wave_config=None, wave_weights_path="")
    off_driver = CoupledESMDriver(
        atmosphere, coupled_off, output_dir=tmp_path / "off")
    off_driver.setup()
    with pytest.raises(ValueError, match="waves off"):
        off_driver.load_coupled_checkpoint(
            0.0, checkpoint_dir=tmp_path, required=True)

    # Refusal must precede any ocean restoration: poison a candidate's ocean
    # leaves while removing the required phase, and verify the live state.
    broken = driver.coupled_checkpoint_arrays()
    broken.pop("wave_spatial_phase")
    for key, value in tuple(broken.items()):
        if key.startswith("ocean3d_") and np.issubdtype(value.dtype, np.floating):
            broken[key] = value+1
    np.savez(tmp_path/"coupled_day_0002.npz", **broken)
    before = np.asarray(resumed.ocean_state.T.data).copy()
    with pytest.raises(ValueError, match="wave state"):
        resumed.load_coupled_checkpoint(
            2., checkpoint_dir=tmp_path, required=True)
    np.testing.assert_array_equal(resumed.ocean_state.T.data, before)

    # Same-shaped metadata changes must also refuse before restoring ocean.
    for day, (key, value) in enumerate((
        ("wave_config_hash", np.asarray("0"*64)),
        ("wave_weights_hash", np.asarray("0"*64)),
        ("wave_spatial_phase", np.asarray(.5)),
        ("wave_spatial_phase", np.asarray(2, dtype=np.int32)),
        ("wave_spatial_phase", np.asarray([0, 1], dtype=np.int32)),
    ), start=3):
        broken = driver.coupled_checkpoint_arrays()
        broken[key] = value
        for name, leaf in tuple(broken.items()):
            if (name.startswith("ocean3d_")
                    and np.issubdtype(leaf.dtype, np.floating)):
                broken[name] = leaf+1
        np.savez(tmp_path/f"coupled_day_{day:04d}.npz", **broken)
        with pytest.raises(ValueError, match="config/grid|spatial phase"):
            resumed.load_coupled_checkpoint(
                float(day), checkpoint_dir=tmp_path, required=True)
        np.testing.assert_array_equal(resumed.ocean_state.T.data, before)

    # Compare the next real wave/coupler step under identical next forcing.
    # This certifies the wave feedback continuation; atmosphere restart is
    # tested separately by the existing driver suite.
    forcing = driver._build_atm_forcing(1800/86400)
    response = driver._last_sfc_response
    driver._step_wave(response, forcing)
    resumed._step_wave(response, forcing)
    # Same float32 reduction-order caveat as above: the phase is exact, the
    # spectrum and Stokes field agree to round-off.
    np.testing.assert_array_equal(
        resumed._wave_state.spatial_phase, driver._wave_state.spatial_phase)
    np.testing.assert_allclose(
        np.asarray(resumed._wave_state.spectra),
        np.asarray(driver._wave_state.spectra), rtol=1e-6, atol=0)
    np.testing.assert_allclose(
        np.asarray(resumed._wave_stokes),
        np.asarray(driver._wave_stokes), rtol=1e-6, atol=0)
