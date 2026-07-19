

def test_fcrit2_couples_launch_cap_and_saturation():
    """Oracle coupling (E3SM effkwv = kwv*fcrit2, gw_common.F90:153+493-494;
    gw_oro.F90:166): fcrit2 scales the SATURATION stress with the SAME
    marginal-instability criterion as the launch Froude cap — previously it
    was wired only to the launch (a coefficient faithful in isolation but
    attached to a different structural object).  Identity at the default
    fcrit2 = 1 (byte-identical legacy); at fcrit2 = 0.5 the saturation
    ceiling must halve, visible as weaker upper-level drag on a column whose
    stress rides the saturation cap."""
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import (
        mcfarlane_gwd)
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        McFarlaneConfig)
    from legoesm import constants
    ncol, nlev = 1, 30
    # strong flow over a big mountain, N constant: stress saturates aloft.
    p_half = jnp.linspace(100e2, 1000e2, nlev + 1)[None, :]
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    T = jnp.full((ncol, nlev), 280.0)
    u = jnp.full((ncol, nlev), 25.0)
    v = jnp.zeros((ncol, nlev))
    z_full = -7500.0 * jnp.log(p_full / 1000e2)
    z_half = -7500.0 * jnp.log(jnp.clip(p_half, 1.0, None) / 1000e2)
    rho = p_full / (constants.R_d * T)
    h_topo = jnp.full((ncol,), 800.0)
    lat = jnp.zeros((ncol,))
    kw = dict(u=u, v=v, T=T, p_full=p_full, p_half=p_half,
              z_full=z_full, z_half=z_half, rho=rho, lat=lat,
              dt=600.0)
    kw["h_topo_col"] = h_topo
    out1 = mcfarlane_gwd(config=McFarlaneConfig(fcrit2=1.0), **kw)
    outh = mcfarlane_gwd(config=McFarlaneConfig(fcrit2=0.5), **kw)
    du1, duh = out1.du_dt, outh.du_dt
    assert jnp.all(jnp.isfinite(du1)) and jnp.all(jnp.isfinite(duh))
    # DISCRIMINANT: this fixture keeps the launch Froude cap INACTIVE
    # (h^2 = 6.4e5 < fcrit2*(U/N)^2 at both values), so the OLD launch-only
    # wiring gives fcrit2 NO effect at all; with the oracle coupling the
    # halved saturation ceiling forces MORE breaking/deposition and the drag
    # must CHANGE (it increases here — lower ceiling, same launch).
    tot1 = float(jnp.sum(jnp.abs(du1)))
    toth = float(jnp.sum(jnp.abs(duh)))
    assert abs(toth - tot1) > 1e-6 * tot1, (
        "fcrit2 dead on a cap-inactive column: the saturation coupling is "
        "not wired (launch-only regression)")
    assert toth > tot1, "halved saturation ceiling must deposit more drag"
