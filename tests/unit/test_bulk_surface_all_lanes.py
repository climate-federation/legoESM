"""The constant-coefficient surface exchange reaches every lane when no
turbulence scheme runs (shared law in ``turbulence/surface_layer``)."""
import json
import os
import subprocess
import sys
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import charged_latent_heat, saturation_specific_humidity


def _inputs():
    T_low = jnp.array([290.0, 300.0, 250.0])
    q_low = jnp.array([0.008, 0.015, 0.0005])
    u = jnp.array([3.0, -7.0, 0.0])
    v = jnp.array([4.0, 1.0, 0.0])
    p_low = jnp.array([99000.0, 100500.0, 98000.0])
    p_s = jnp.array([100000.0, 101300.0, 99000.0])
    T_sfc = jnp.array([295.0, 302.0, 255.0])
    return T_low, q_low, u, v, p_low, p_s, T_sfc


def test_constant_bulk_fluxes_follow_the_law():
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        constant_bulk_surface_fluxes)
    T_low, q_low, u, v, p_low, p_s, T_sfc = _inputs()
    C_H, C_E = 1.3e-3, 1.5e-3
    shflx, lhflx = constant_bulk_surface_fluxes(
        T_low, q_low, u, v, p_low, p_s, T_sfc, C_H, C_E)
    rho = p_low / (constants.R_d * T_low)
    wind = np.sqrt(np.asarray(u) ** 2 + np.asarray(v) ** 2 + 1.0)
    np.testing.assert_allclose(
        shflx, rho * constants.c_pd * C_H * wind * (T_sfc - T_low), rtol=1e-12)
    qs = saturation_specific_humidity(T_sfc, p_s)
    np.testing.assert_allclose(
        lhflx, rho * charged_latent_heat("constant", T_sfc) * C_E * wind
        * (qs - q_low), rtol=1e-12)
    assert float(jnp.min(shflx)) > 0.0 and float(jnp.min(lhflx)) > 0.0


def test_equilibrium_gives_zero_flux():
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        constant_bulk_surface_fluxes)
    T_low, _, u, v, p_low, p_s, _ = _inputs()
    q_sat = saturation_specific_humidity(T_low, p_s)
    shflx, lhflx = constant_bulk_surface_fluxes(
        T_low, q_sat, u, v, p_low, p_s, T_low, 1e-3, 1e-3)
    assert float(jnp.max(jnp.abs(shflx))) == 0.0
    assert float(jnp.max(jnp.abs(lhflx))) == 0.0


def test_kick_inverts_the_charged_latent_heat():
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        bottom_layer_surface_kick)
    T_sfc = jnp.array([300.0, 260.0])
    shflx = jnp.array([20.0, -5.0])
    lhflx = jnp.array([150.0, 10.0])
    dp = jnp.array([2000.0, 1500.0])
    dT, dq, evap = bottom_layer_surface_kick(shflx, lhflx, T_sfc, dp)
    np.testing.assert_allclose(
        evap, lhflx / charged_latent_heat("constant", T_sfc), rtol=1e-12)
    np.testing.assert_allclose(dq, constants.g * evap / dp, rtol=1e-12)
    np.testing.assert_allclose(
        dT, constants.g * (shflx + lhflx - constants.L_v * evap)  # latent-ok: moist-enthalpy reference
        / (constants.c_pd * dp), rtol=1e-12)


def test_make_physics_refuses_bulk_next_to_turbulence_and_on_other_models():
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    with pytest.raises(ValueError, match="already owns the surface flux"):
        make_physics(PhysicsConfig(turbulence=TurbulenceConfig(scheme="louis")),
                     model_type="mpas", surface_bulk_coeffs=(1e-3, 1e-3, True))
    with pytest.raises(ValueError, match="silently drop"):
        make_physics(PhysicsConfig(), model_type="nonhydrostatic",
                     surface_bulk_coeffs=(1e-3, 1e-3, True))


def test_make_physics_applies_the_default_bulk_when_turbulence_is_off():
    """Direct callers get the default exchange too: no surface temperature at
    all refuses (no silent no-flux); a prescribed skin is exchanged with,
    column by column (the 'no override' sentinel exchanges nothing); None
    opts out."""
    from legoesm.atmosphere.forcing.scm.scm import (
        make_column_state, make_scm_grid)
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    nlev = 8
    cfg = PhysicsConfig(radiation=RadiationConfig(scheme="none"),
                        convection=ConvectionConfig(scheme="none"),
                        turbulence=TurbulenceConfig(scheme="none"))
    grid = make_scm_grid(latitude_deg=0.0)
    sigma = create_sigma_coordinate(nlev, sigma_top=0.01)
    state = make_column_state(nlev, T_profile=jnp.linspace(220.0, 290.0, nlev))

    def dT(fn, **kw):
        out = fn(state, grid, sigma, **kw)
        return (out[0] if isinstance(out, tuple) else out).dT_dt.data

    fn = make_physics(cfg, model_type="hydrostatic", dt=60.0)
    with pytest.raises(ValueError, match="surface temperature"):
        fn(state, grid, sigma)
    ps = init_physics_state(ncol=1, nlev=nlev, physics_config=cfg)
    assert float(jnp.max(jnp.abs(dT(fn, phys_state=ps)))) == 0.0
    hot = dT(fn, phys_state=ps._replace(
        surface_T_sfc_override=jnp.full((1,), 300.0)))
    assert float(hot[..., -1].max()) > 0.0
    assert float(jnp.max(jnp.abs(hot[..., :-1]))) == 0.0
    off = make_physics(cfg, model_type="hydrostatic", dt=60.0,
                       surface_bulk_coeffs=None)
    assert float(jnp.max(jnp.abs(dT(off)))) == 0.0


# --- every lane: constant vs none after 2 steps, all other physics off -------

_LANE_SCRIPT = textwrap.dedent('''
    import json, sys, jax, numpy as np
    from legoesm.driver.config import (ExperimentConfig, GridConfig,
                                       DycoreConfig, OutputConfig)
    from legoesm.driver.model_driver import ModelDriver
    out, grid, dyc, extra = sys.argv[1], *map(json.loads, sys.argv[2:5])
    res = {}
    for sbs in ("constant", "none"):
        cfg = ExperimentConfig(
            grid=GridConfig(**grid),
            dycore=DycoreConfig(model_type="hydrostatic", **dyc),
            output=OutputConfig(output_dir=f"{out}/{sbs}", diag_days=0,
                                checkpoint_days=0),
            days=2 * dyc["dt"] / 86400.0, dataset="analytical",
            fix_moisture=True, distributed=False, surface_bulk_scheme=sbs,
            **{"n_devices": 1, "turbulence": "none", "radiation": "none",
               "convection": "none", "microphysics": "none",
               "gravity_wave_drag": "none", "cloud_scheme": "none", **extra})
        d = ModelDriver(cfg, output_dir=cfg.output.output_dir)
        d.setup()
        assert d.run(compiled=True) == "COMPLETED"
        s = d.state
        T = s.T if hasattr(s, "T") else s.T_hat      # spectral: T_hat
        tr = getattr(s, "tracers", None) or {}
        qv = tr.get("q_v") if tr else d.tracers.get("q_v")
        res[sbs] = (np.asarray(getattr(T, "data", T)),
                    None if qv is None
                    else np.asarray(getattr(qv, "data", qv)))
    dT = float(np.abs(res["constant"][0] - res["none"][0]).max())
    dq = (None if res["constant"][1] is None
          else float(np.abs(res["constant"][1] - res["none"][1]).max()))
    print("RESULT " + json.dumps({"dT": dT, "dq": dq}))
''')

_CUBE_TILED = dict(dt=100.0, discretization="cdgrid", energy_fixer=False,
                   hyperdiff_scale=0.0, div_damp_scale=0.0, a_h_scale=0.0)
_LATLON = (dict(grid_type="latlon", resolution=8, nlev=4),
           dict(discretization="latlon_cgrid", dt=100.0))
_MPAS = (dict(grid_type="mpas", resolution=3, nlev=8, vertical_coord="hybrid"),
         dict(discretization="mpas", dt=300.0))
_SPECTRAL = (dict(grid_type="gaussian", resolution=8, nlev=8),
             dict(discretization="spectral", dt=600.0, energy_fixer=False))

# (grid, dycore, extra, devices, moist: vapour must change too).  The
# lat-band and dynamics-only tiled lanes carry no vapour in their state:
# sensible heat only there.
LANES = {
    "latband_2dev": (*_LATLON, dict(enable_latlon_spmd=True, n_devices=2), 2,
                     False),
    "latband_2dev_segments": (*_LATLON, dict(
        enable_latlon_spmd=True, n_devices=2,
        latlon_spmd_compiled_segments=True), 2, False),
    "tiled_cube_dry": (dict(grid_type="cubed_sphere", resolution=8, nlev=4),
                       _CUBE_TILED, dict(n_devices=24), 24, False),
    "tiled_cube_kessler": (dict(grid_type="cubed_sphere", resolution=8,
                                nlev=4), _CUBE_TILED,
                           dict(n_devices=24, microphysics="kessler"), 24,
                           True),
    "mpas_dry": (*_MPAS, {}, 1, False),
    "mpas_kessler": (*_MPAS, dict(microphysics="kessler"), 1, True),
    "spectral_dry": (*_SPECTRAL, dict(precision="fp64"), 1, False),
    "spectral_kessler": (*_SPECTRAL, dict(precision="fp64",
                                          microphysics="kessler"), 1, True),
    "fv3_duo_column_dry": (dict(grid_type="cubed_sphere", resolution=8,
                                nlev=5),
                           dict(discretization="fv3_duo", column_lane=True,
                                dt=600.0), dict(precision="fp64"), 1, False),
}


def _run_lane(tmp_path, grid, dyc, extra, ndev):
    # A child launched inside a Slurm step must not inherit the step's MPI
    # launch variables (Open MPI would treat it as a direct srun launch).
    base = {k: v for k, v in os.environ.items()
            if not k.startswith(("SLURM_", "PMI", "PMIX", "OMPI_"))}
    env = {**base, "JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": f"--xla_force_host_platform_device_count={ndev}"}
    proc = subprocess.run(
        [sys.executable, "-c", _LANE_SCRIPT, str(tmp_path), json.dumps(grid),
         json.dumps(dyc), json.dumps(extra)],
        env=env, capture_output=True, text=True, timeout=3600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT "):])


@pytest.mark.slow
@pytest.mark.parametrize("lane", sorted(LANES))
def test_lane_applies_bulk_surface_exchange(tmp_path, lane):
    grid, dyc, extra, ndev, moist = LANES[lane]
    r = _run_lane(tmp_path, grid, dyc, extra, ndev)
    assert r["dT"] > 1e-6, r
    if moist:
        assert r["dq"] is not None and r["dq"] > 1e-12, r
