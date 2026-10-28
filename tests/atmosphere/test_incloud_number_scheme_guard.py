"""In-cloud (aerosol/CCN) droplet number on the non-CAM6 cloud schemes (#2072 ask 2).

cam6_clubb has an oracle (CAM6 MG2 REL, test_reff_incloud_oracle.py).
``resolved`` is the cloud-resolving convention: a cloudy cell is fully cloudy,
so the M2005 PSD on the cell q_c IS the in-cloud pairing (WRF v4.8.0
module_mp_morr_two_moment.F EFFC on cell QC3D/NC3D) and the flag changes
nothing.  sundqvist / xu_randall have no reference that pairs an in-cloud number
with their cloud fraction, so the in-cloud flag is refused there -- on the MPAS
column lane (radiation physics_fn) and on the FV / coupled pipeline lane, which
now flags its CCN number as in-cloud like the MPAS lane.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.clouds import cloud_fraction as cf_mod
from legoesm.atmosphere.physics.clouds.config import CloudConfig

NCOL, NLEV = 3, 4


def _column():
    p_full = jnp.broadcast_to(jnp.asarray([3.0e4, 5.0e4, 7.0e4, 9.0e4]), (NCOL, NLEV))
    dp = jnp.full((NCOL, NLEV), 2.0e4)
    T = jnp.broadcast_to(jnp.asarray([240.0, 260.0, 275.0, 288.0]), (NCOL, NLEV))
    q_v = jnp.full((NCOL, NLEV), 8.0e-3)
    q_c = jnp.asarray([[0.0, 1.0e-5, 2.0e-4, 5.0e-4]] * NCOL)
    n_c = jnp.full((NCOL, NLEV), 1.0e8)   # CCN number [1/m3], > 1
    return dict(T=T, p_full=p_full, q_v=q_v, dp=dp, q_cloud=q_c,
                q_ice=jnp.zeros((NCOL, NLEV)), n_cloud=n_c)


@pytest.mark.parametrize("scheme", ["sundqvist", "xu_randall"])
def test_in_cloud_number_refused_without_oracle(scheme):
    kw = _column()
    with pytest.raises(ValueError, match="in-cloud"):
        cf_mod.compute_cloud_properties(config=CloudConfig(scheme=scheme),
                                        n_cloud_in_cloud=True, **kw)
    # the grid-mean prognostic tracer (flag False) is unaffected
    cf_mod.compute_cloud_properties(config=CloudConfig(scheme=scheme),
                                    n_cloud_in_cloud=False, **kw)


def test_resolved_in_cloud_is_the_cell_pairing():
    kw = _column()
    cfg = CloudConfig(scheme="resolved")
    a = cf_mod.compute_cloud_properties(config=cfg, n_cloud_in_cloud=True, **kw)
    b = cf_mod.compute_cloud_properties(config=cfg, n_cloud_in_cloud=False, **kw)
    np.testing.assert_array_equal(np.asarray(a.r_eff_liq), np.asarray(b.r_eff_liq))


class _Stop(Exception):
    pass


def _pipeline_core(monkeypatch, radiation, aer_on):
    from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp as rrtmgp_mod
    from legoesm.driver import physics_pipeline as pp
    from legoesm.driver.config import ExperimentConfig
    from legoesm.grids.latlon import create_latlon_grid
    from tests.atmosphere.test_pipeline_ocean_albedo_oracle import (
        _EchoSolver, _sigma,
    )
    monkeypatch.setattr(rrtmgp_mod.RRTMGP, "from_legoesm_config",
                        staticmethod(lambda cfg: _EchoSolver()))
    pipe = pp.build_physics_pipeline(
        create_latlon_grid(4, 6), _sigma(NLEV),
        # kk2000 named: the unset mg law resolves to kk2000_cam6, which this
        # lane (no CLUBB relvar) refuses; kk2000 is what it ran before.
        ExperimentConfig(radiation=radiation, microphysics="morrison",
                         morrison_warm_rain_scheme="kk2000",
                         nc_from_aerosol=True, diurnal_cycle=False))
    s2 = pipe.adapter.shape_2d
    s3 = (*s2, NLEV)
    aer = jnp.full((pipe.adapter.ncol, NLEV), 0.05) if aer_on else None
    return pipe.compute_radiation_core(
        jnp.full(s3, 270.0), jnp.full(s2, 1.0e5), jnp.full(s3, 3.0e-3),
        jnp.full(s2, 295.0), jnp.zeros(s2), jnp.full(s2, 0.4),
        jnp.zeros(s2), 80.0, 43200.0, jnp.zeros(0), constants.S_0, None, aer,
        cloud_scheme="sundqvist", q_c=jnp.full(s3, 1.0e-4),
        N_c=jnp.zeros(s3), u=jnp.zeros(s3), v=jnp.zeros(s3), dt=600.0)


@pytest.mark.parametrize("radiation, aer_on, want", [
    ("rrtmgp", True, True), ("rrtmgp", False, False), ("gray", True, False)])
def test_pipeline_flags_ccn_number_in_cloud(monkeypatch, radiation, aer_on, want):
    """The FV / coupled pipeline hands its Andreae CCN number to the cloud
    optics with ``n_cloud_in_cloud=True`` when rrtmgp consumes the radius
    (same footing and gating as the MPAS lane); gray discards it."""
    seen = {}

    def _spy(**kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(cf_mod, "compute_cloud_properties", _spy)
    with pytest.raises(_Stop):
        _pipeline_core(monkeypatch, radiation, aer_on)
    assert seen["n_cloud_in_cloud"] is want


def test_pipeline_rrtmgp_refuses_ccn_number_without_oracle(monkeypatch):
    with pytest.raises(ValueError, match="in-cloud"):
        _pipeline_core(monkeypatch, "rrtmgp", True)


@pytest.mark.parametrize("cloud_scheme", ["sundqvist", "xu_randall"])
def test_mpas_radiation_refuses_ccn_number_without_oracle(cloud_scheme):
    """MPAS / FV3-duo column lane: the hydrostatic radiation physics_fn (what
    ``_run_column`` builds via make_physics(model_type='mpas')) with the CCN
    number and a cloud scheme that has no in-cloud pairing reference raises."""
    from legoesm.atmosphere.physics.radiation import integration as rad_int
    from legoesm.atmosphere.physics.radiation.config import (
        RadiationConfig, RRTMGPConfig,
    )
    from tests.atmosphere.hydrostatic.unit.test_aerosol_ccn_combined import (
        _aerosol_forcing, _moist_state, _setup,
    )
    grid, sigma = _setup()
    fn = rad_int.make_radiation_physics(
        RadiationConfig(scheme="rrtmgp", cloud_scheme=cloud_scheme,
                        rrtmgp=RRTMGPConfig(include_clouds=True)),
        model_type="hydrostatic", nc_from_aerosol=True)
    with pytest.raises(ValueError, match="in-cloud"):
        fn(_moist_state(grid, sigma), grid, sigma,
           forcing={**_aerosol_forcing(grid, sigma, 0.1),
                    "day_of_year": jnp.asarray(80.0)})


def test_pipeline_gray_ccn_run_unaffected(monkeypatch):
    """Gray discards the droplet radius, so the guard never fires there."""
    out = _pipeline_core(monkeypatch, "gray", True)
    assert np.all(np.isfinite(np.asarray(out[0])))
