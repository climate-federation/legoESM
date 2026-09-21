"""Ocean surface-layer corrections on the MPAS lane (both opt-in).

1. ``compute_surface_fluxes(..., z_ref=<array>)`` hands the MOST solver the
   real height of its inputs.  With the lowest full level at ~150 m the
   COARE latent heat flux for the SAME state is 15-50 % lower than when the
   inputs are labelled as 10 m (neutral log law: (ln(10/z0)/ln(150/z0))^2 ~
   0.66).  Fails when the keyword is not honoured (reduction 0).  z_ref =
   10 m array must equal the config default bit for bit (the off path).
2. ``ocean_surface_q_sat`` with the 0.98 factor is exactly 0.98 x the
   fresh-water value at the same T, p.
3. The MPAS turbulence bridge applies the two switches (the height switch also
   hands the solver the potential temperature at the input height, COARE's
   dry-adiabatic correction): with
   ``ocean_q_sfc_saline`` the ocean q_sfc it uses is 0.98*q_sat(SST, p_s),
   and with ``z_ref_model_level`` its latent flux drops -- checked by CALLING
   the bridge with a spy on compute_surface_fluxes.  Fails when the bridge
   ignores the switches.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.surface_layer import compute_surface_fluxes  # noqa: E402
from legoesm.core.bulk_flux import ocean_surface_q_sat  # noqa: E402
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402


def _state(n=3):
    q_sfc = jnp.full((n,), float(saturation_mixing_ratio(jnp.asarray(300.2), jnp.asarray(1.0e5))))
    return (jnp.full((n,), 7.0), jnp.zeros((n,)), jnp.full((n,), 299.0),
            jnp.full((n,), 0.0176), jnp.full((n,), 300.2), q_sfc, jnp.full((n,), 1.15))


def test_z_ref_keyword_lowers_the_flux_for_inputs_taken_at_150_m():
    cfg = SurfaceLayerConfig(bulk_scheme="coare3")
    args = _state()
    lh_default = np.asarray(compute_surface_fluxes(*args, cfg)[3])
    lh_ten = np.asarray(compute_surface_fluxes(*args, cfg, z_ref=jnp.full((3,), 10.0))[3])
    lh_150 = np.asarray(compute_surface_fluxes(*args, cfg, z_ref=jnp.full((3,), 150.0))[3])
    np.testing.assert_array_equal(lh_ten, lh_default)
    reduction = (lh_default - lh_150) / lh_default
    assert np.all(reduction > 0.15) and np.all(reduction < 0.50), reduction


def test_saline_q_sat_is_the_fresh_value_times_0_98():
    q_s = ocean_surface_q_sat(jnp.array([300.2]), 1.0e5, saline_factor=0.98)
    q_f = saturation_mixing_ratio(jnp.array([300.2]), jnp.array([1.0e5]))
    np.testing.assert_allclose(np.asarray(q_s), 0.98 * np.asarray(q_f), rtol=1e-12)


def test_mpas_bridge_applies_both_switches():
    import types

    from legoesm.atmosphere.physics.turbulence import integration as ti
    from legoesm.atmosphere.physics.turbulence import surface_layer as sl
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.core.field import Field
    from legoesm.core.state import MPASHydrostaticState
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    nlev = 16
    mesh = create_voronoi_mesh(3, lloyd_iterations=3)
    ncol = mesh.nCells
    sigma = create_sigma_coordinate(nlev)
    p_s = jnp.full((ncol,), 1.0e5)
    T = jnp.broadcast_to(jnp.linspace(300.0, 220.0, nlev)[None, :], (ncol, nlev))
    q = jnp.broadcast_to(jnp.linspace(0.017, 1e-5, nlev)[None, :], (ncol, nlev))
    state = MPASHydrostaticState(
        u=Field(jnp.full((mesh.nEdges, nlev), 4.0)), T=Field(T), p_s=Field(p_s),
        phis=Field(jnp.zeros((ncol,))), tracers={"q_v": Field(q)})
    sst = jnp.full((ncol,), 301.0)
    f_land = np.zeros(ncol); f_land[:3] = 1.0     # three land cells
    forcing = {"T_sfc": sst, "lhflx_land": jnp.zeros((ncol,)), "shflx_land": jnp.zeros((ncol,))}
    seen = {}
    real = sl.compute_surface_fluxes   # the bridge imports it at call time

    def spy(*a, **k):
        seen["q_sfc"] = np.asarray(a[5]); seen["z_ref"] = k.get("z_ref"); seen["T_in"] = np.asarray(a[2])
        out = real(*a, **k); seen["lh"] = np.asarray(out[3]); return out

    def run(zml, qsal):
        tc = ti.materialize_sub_config(TurbulenceConfig(scheme="louis"))
        surf = tc.louis.surface._replace(bulk_scheme="coare3", z_ref_model_level=zml,
                                         ocean_q_sfc_saline=qsal)
        tc = tc._replace(louis=tc.louis._replace(surface=surf))
        sl.compute_surface_fluxes = spy
        try:
            fn = ti.make_turbulence_physics(tc, model_type="mpas", dt=600.0,
                                            f_land=f_land)
            fn(state, mesh, sigma, forcing=forcing)
        finally:
            sl.compute_surface_fluxes = real
        return dict(seen)

    off = run(False, False)
    assert off["z_ref"] is None
    np.testing.assert_allclose(off["q_sfc"], np.asarray(
        saturation_mixing_ratio(sst, sigma.pressure_at_full(p_s)[:, -1])), rtol=1e-12)
    sal = run(False, True)
    ocean = f_land < 0.5
    np.testing.assert_allclose(sal["q_sfc"][ocean], 0.98 * np.asarray(
        saturation_mixing_ratio(sst, p_s))[ocean], rtol=1e-6)   # blend round-off
    # the LAND cells' humidity is untouched by the ocean switch (codex P1)
    np.testing.assert_array_equal(sal["q_sfc"][~ocean], off["q_sfc"][~ocean])
    hgt = run(True, False)
    z = np.asarray(hgt["z_ref"])
    assert z.shape == (ncol,) and np.all(z > 50.0) and np.all(z < 400.0), z
    assert np.all(hgt["lh"] < off["lh"])
    # potential temperature at the input height: T + g/c_p * z (COARE)
    from legoesm import constants
    np.testing.assert_allclose(hgt["T_in"] - off["T_in"], constants.g / constants.c_pd * z, rtol=1e-12)


def test_height_switch_accepted_and_saline_refused_on_structured_lanes():
    """codex whole-branch review P2, updated: the level height now reaches
    every lane (each kernel calls surface_fluxes_at_lowest_level), so it is
    honoured, not refused; the sea-water humidity still needs a land fraction
    the structured-grid factories do not carry, so it is refused there
    instead of silently ignored.  That the height is HONOURED (not merely
    accepted) is pinned at the kernel by test_surface_reference_height
    (correction lowers the latent flux; switch off is byte-identical) and on
    the MPAS bridge by test_mpas_bridge_applies_both_switches above."""
    import pytest
    from legoesm.atmosphere.physics.turbulence import integration as ti
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    tc = ti.materialize_sub_config(TurbulenceConfig(scheme="louis"))
    surf = tc.louis.surface._replace(bulk_scheme="coare3", z_ref_model_level=True)
    tc = tc._replace(louis=tc.louis._replace(surface=surf))
    assert callable(ti.make_turbulence_physics(tc, model_type="hydrostatic", dt=600.0))
    tc = tc._replace(louis=tc.louis._replace(surface=surf._replace(ocean_q_sfc_saline=True)))
    with pytest.raises(NotImplementedError, match="land fraction"):
        ti.make_turbulence_physics(tc, model_type="hydrostatic", dt=600.0)
