"""Rule 0 tracer-mass budget of the spectral NH dycore (DCMIP-2025 TC3).

With no physics there are no sources or sinks, so ``in - out - d(storage)``
reduces to ``d(total water)/W = 0`` to round-off.  The former advective
tracer form (q prognostic, upwind -w dq/dz) lost 3.4e-4 of the TC3 water in
4 min (3.4e-6 in the 10 steps below); the flux form must close to ~1e-13.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants as c  # noqa: E402
from legoesm.atmosphere.dynamics.gcm.dcmip2025_ic import TC3_PARAMS  # noqa: E402
from legoesm.atmosphere.dynamics.gcm.spectral_nh import (  # noqa: E402
    SpectralCompressibleEulerModel,
    SpectralNHConfig,
    dcmip25_tc3_init_spectral,
)
from legoesm.grids.gaussian import (  # noqa: E402
    create_gaussian_grid,
    sh_analysis_3d,
    sh_synthesis_3d,
)

_N_MAX, _NLEV, _DT, _NSTEPS = 21, 20, 3.0, 10


def _tc3_model():
    x = float(TC3_PARAMS["small_earth_factor"])
    state, hc, tm = dcmip25_tc3_init_spectral(
        create_gaussian_grid(_N_MAX), n_levels=_NLEV)
    cfg = SpectralNHConfig(
        n_acoustic_substeps=25, semi_implicit_acoustic=True,
        sponge_width=8000.0, sponge_coeff=0.15,
        hyperdiff_coeff=2.338e15 * 8.0 / x ** 4, small_earth_factor=x,
        rotating=bool(TC3_PARAMS.get("rotating", True)),
        fix_mass=True, anchor_mass_to_initial=True)
    grid = create_gaussian_grid(_N_MAX, radius=c.R_earth / x)
    return SpectralCompressibleEulerModel(grid, hc, tm, cfg), state


def _run(model, state):
    for _ in range(_NSTEPS):
        state = model.step(state, _DT)
    return state


def test_tc3_total_water_closes_without_physics():
    model, state = _tc3_model()
    w0 = np.asarray(model.compute_tracer_mass(state))
    end = _run(model, state)
    w1 = np.asarray(model.compute_tracer_mass(end))
    # The run must actually move water (non-vacuous): q_v redistributes.
    w = sh_synthesis_3d(model.grid, end.w_hat.data)
    assert float(jnp.max(jnp.abs(w))) > 1e-2
    # ... and tracer transport is live (a frozen tracer would close trivially).
    d_rq = sh_synthesis_3d(
        model.grid, end.tracers_hat.data[..., 0] - state.tracers_hat.data[..., 0])
    assert float(jnp.max(jnp.abs(d_rq))) > 1e-9
    rel = (w1.sum() - w0.sum()) / w0.sum()
    assert abs(rel) < 1e-12, f"total water budget residual {rel:.3e}"
    # No physics: each tracer (q_v, q_c, q_r) closes on its own.
    per = np.abs(w1 - w0) / w0.sum()
    assert np.all(per < 1e-12), f"per-tracer residuals {per}"


def test_tc3_uniform_mixing_ratio_stays_uniform():
    """Tracer flux = q * (the mass flux that moves rho'), horizontally and
    vertically, so a uniform q stays uniform (consistency with continuity)."""
    model, state = _tc3_model()
    g, hc = model.grid, model.height_coord
    q0s = (0.01, 0.002, 0.0005)          # every tracer, distinct values
    rq = jnp.broadcast_to(hc.rho_ref, (g.n_lat, g.n_lon, _NLEV))
    tr = jnp.stack([sh_analysis_3d(g, q0 * rq) for q0 in q0s], axis=-1)
    state = state._replace(tracers_hat=state.tracers_hat.replace(data=tr))
    end = _run(model, state)
    rho = hc.rho_ref + sh_synthesis_3d(g, end.rho_prime_hat.data)
    w = sh_synthesis_3d(g, end.w_hat.data)
    assert float(jnp.max(jnp.abs(w))) > 1e-2    # vertical mass flux is live
    # ... and the tracer density itself moved (a frozen tracer cannot pass).
    d_rq = sh_synthesis_3d(g, end.tracers_hat.data[..., 0] - tr[..., 0])
    assert float(jnp.max(jnp.abs(d_rq))) > 1e-9
    for t, q0 in enumerate(q0s):
        q = sh_synthesis_3d(g, end.tracers_hat.data[..., t]) / rho
        assert float(jnp.max(jnp.abs(q / q0 - 1.0))) < 1e-10, t


def test_vertical_transport_takes_the_donor_cell():
    """Prescribed signed interface fluxes: each face carries the UPWIND
    cell's q (index 0 = top).  A reversed donor still telescopes and keeps
    a uniform q uniform, so only a cellwise check can catch it.  Two columns
    (distinct rho, q, J, and flux patterns), two tracers, distinct dz, so a
    column-, tracer- or level-broadcast mix-up changes the answer."""
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
        _tracer_vertical_transport,
    )
    rho = np.array([[1.0, 2.0, 4.0], [1.5, 2.5, 3.0]])          # (col, nlev)
    qs = np.stack([np.array([[0.1, 0.2, 0.3], [0.4, 0.15, 0.6]]),
                   np.array([[0.5, 0.7, 0.9], [0.05, 0.8, 0.35]])],
                  axis=-1)                                     # (col, nlev, tr)
    rq = rho[..., None] * qs
    # Interfaces 0..3 (top..bottom); 0 and 3 are boundaries.  Face 1 sits
    # between lev 0 (above) and lev 1 (below); face 2 between lev 1 and 2.
    # Column 0 diverges from the middle cell (both faces donor lev 1);
    # column 1 converges into it (face 1 downward -> donor lev 0, face 2
    # upward -> donor lev 2).
    flux = np.array([[0.0, 0.5, -0.25, 0.0], [0.0, -0.3, 0.6, 0.0]])
    donors = ((1, 1), (0, 2))
    dz = np.array([10.0, 20.0, 40.0])
    J = np.array([2.0, 0.8])
    out = np.asarray(_tracer_vertical_transport(
        jnp.asarray(rq), jnp.asarray(rho), jnp.asarray(flux),
        jnp.asarray(dz), jnp.asarray(J)))
    for c in range(2):
        for t in range(2):
            f1 = flux[c, 1] * qs[c, donors[c][0], t]
            f2 = flux[c, 2] * qs[c, donors[c][1], t]
            expect = rq[c, :, t] - np.array(
                [0.0 - f1, f1 - f2, f2 - 0.0]) / (dz * J[c])
            np.testing.assert_allclose(out[c, :, t], expect, rtol=0,
                                       atol=1e-15)
        # Column tracer mass sum(J * dz * rq) is unchanged, per tracer.
        np.testing.assert_allclose((out[c] * dz[:, None] * J[c]).sum(0),
                                   (rq[c] * dz[:, None] * J[c]).sum(0),
                                   rtol=0, atol=1e-13)


@pytest.mark.parametrize("semi_implicit", [False, True])
def test_acoustic_mass_flux_reconstructs_rho_increment(semi_implicit):
    """The returned substep-integrated flux is exactly what moved rho'
    (stretched grid, nonzero flow), for both acoustic kernels."""
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
        _acoustic_substeps_grid,
        _acoustic_substeps_grid_semi_implicit,
    )
    from legoesm.grids.vertical import (
        compute_terrain_metric,
        create_height_coordinate_from_z_half,
    )
    nlev = 12
    dz = 50.0 * 1.4 ** np.arange(nlev)
    z_half = jnp.asarray(np.concatenate([[0.0], np.cumsum(dz)])[::-1])
    hc = create_height_coordinate_from_z_half(z_half)
    # Spatially varying terrain so J != 1 differs per column: a kernel
    # that dropped the 1/J would fail the reconstruction below.
    tm = compute_terrain_metric(
        jnp.array([[0.0, 150.0, 300.0], [450.0, 600.0, 900.0]]), hc)
    assert float(jnp.ptp(tm.jacobian)) > 0.05
    rng = np.random.default_rng(0)
    w = jnp.asarray(rng.normal(0.0, 1.0, (2, 3, nlev + 1))).at[..., 0].set(0.0).at[..., -1].set(0.0)
    th = jnp.asarray(rng.normal(0.0, 0.5, (2, 3, nlev)))
    rho = jnp.asarray(rng.normal(0.0, 1e-3, (2, 3, nlev)))
    fn = _acoustic_substeps_grid_semi_implicit if semi_implicit else _acoustic_substeps_grid
    _, _, rho_new, mflux = fn(w, th, rho, 0.2, 5, hc, tm, SpectralNHConfig())
    assert float(jnp.max(jnp.abs(mflux))) > 1e-3
    assert float(jnp.max(jnp.abs(rho_new - rho))) > 1e-6   # rho' really moved
    np.testing.assert_allclose(mflux[..., 0], 0.0, atol=0)
    np.testing.assert_allclose(mflux[..., -1], 0.0, atol=0)
    drho = -(mflux[..., :-1] - mflux[..., 1:]) / hc.dz / tm.jacobian[..., None]
    np.testing.assert_allclose(rho_new - rho, drho, rtol=0, atol=1e-15)


def test_batched_horizontal_tracer_tendency_matches_per_tracer():
    """Tracer t's slow (horizontal) tendency from the folded multi-tracer
    batch equals the single-tracer calculation for t (no axis mix-up)."""
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
        spectral_nh_slow_tendencies,
    )
    model, state = _tc3_model()
    g, hc = model.grid, model.height_coord
    lon, lat = g.lon2d[..., None], g.lat[:, None, None]
    zs = hc.z_full[None, None, :] / 5000.0
    profiles = [jnp.cos(lon) * jnp.exp(-zs), jnp.sin(2 * lon + lat) * zs,
                jnp.cos(3 * lat) * jnp.sin(lon) + 0.3 * zs]
    tr = jnp.stack([sh_analysis_3d(g, hc.rho_ref * (1.0 + 0.5 * p))
                    for p in profiles], axis=-1)
    state = state._replace(tracers_hat=state.tracers_hat.replace(data=tr))
    batched = spectral_nh_slow_tendencies(
        state, g, hc, model.terrain_metric, model.config).tracers_hat.data
    for t in range(tr.shape[-1]):
        single = state._replace(tracers_hat=state.tracers_hat.replace(
            data=tr[..., t:t + 1]))
        one = spectral_nh_slow_tendencies(
            single, g, hc, model.terrain_metric, model.config).tracers_hat.data
        assert float(jnp.max(jnp.abs(one))) > 0.0
        np.testing.assert_allclose(np.asarray(batched[..., t]),
                                   np.asarray(one[..., 0]), rtol=0,
                                   atol=1e-12 * float(jnp.max(jnp.abs(one))))


def test_horizontal_tracer_tendency_manufactured_solid_body():
    """Independent reference: solid-body zonal flow u = U cos(phi) (div = 0)
    advecting rho*q = rho_ref * c * cos(phi) cos(lambda) has the exact
    flux-form tendency +(U/a) rho_ref c cos(phi) sin(lambda) (n = 1, so the
    truncation and the 2/3 de-aliasing mask keep it exactly)."""
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
        spectral_nh_slow_tendencies,
    )
    from legoesm.grids.gaussian import sh_analysis_dmu_3d, sh_analysis_oc2_3d
    model, state = _tc3_model()
    g, hc = model.grid, model.height_coord
    a, u0, c0 = g.radius, 10.0, 0.01
    cos_phi = jnp.cos(g.lat)[:, None, None]
    lon = g.lon2d[..., None]
    u_cos = jnp.broadcast_to(u0 * cos_phi ** 2, (g.n_lat, g.n_lon, _NLEV))
    zero = jnp.zeros_like(u_cos)
    im_a = 1j * g.ms.astype(jnp.float64) / a
    vor = im_a[:, None] * sh_analysis_oc2_3d(g, zero) + sh_analysis_dmu_3d(g, u_cos) / a
    div = im_a[:, None] * sh_analysis_oc2_3d(g, u_cos) - sh_analysis_dmu_3d(g, zero) / a
    rq = hc.rho_ref * c0 * cos_phi * jnp.cos(lon)
    state = state._replace(
        vor_hat=state.vor_hat.replace(data=vor),
        div_hat=state.div_hat.replace(data=div),
        w_hat=state.w_hat.replace(data=jnp.zeros_like(state.w_hat.data)),
        tracers_hat=state.tracers_hat.replace(
            data=jnp.stack([sh_analysis_3d(g, rq)] * 3, axis=-1)))
    tend = spectral_nh_slow_tendencies(
        state, g, hc, model.terrain_metric, model.config).tracers_hat.data
    exact = (u0 / a) * hc.rho_ref * c0 * cos_phi * jnp.sin(lon)
    scale = float(jnp.max(jnp.abs(exact)))
    for t in range(3):
        got = sh_synthesis_3d(g, tend[..., t])
        assert float(jnp.max(jnp.abs(got - exact))) < 1e-9 * scale, t


_TERRAIN_DEFECT = (
    "KNOWN DEFECT shared with rho': over terrain the grid-space 1/J before "
    "the spectral projection breaks the J-weighted integral (TC2a T21: "
    "tracer AND dry mass both +7.6e-10 in 10 steps, fixer off).  Needs a "
    "J-weighted prognostic (J rho', J rho q); design question, not fixed.")


def test_tc2a_terrain_tracer_mass_closes_without_fixer():
    """Tracks the terrain Rule 0 gap.  Setup / stepping errors and
    non-finite results FAIL normally; only the finite budget residual is
    xfailed, and the test FAILS once the residual closes (stale marker)."""
    from legoesm.atmosphere.dynamics.gcm.dcmip2025_ic import TC2_PARAMS
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
        dcmip25_tc2_init_spectral,
    )
    x = float(TC2_PARAMS["small_earth_factor"])
    state, hc, tm = dcmip25_tc2_init_spectral(
        create_gaussian_grid(_N_MAX), n_levels=_NLEV, subcase="a")
    assert float(tm.jacobian.min()) < 0.95          # terrain is present
    g = create_gaussian_grid(_N_MAX, radius=c.R_earth / x)
    rq = jnp.broadcast_to(hc.rho_ref, (g.n_lat, g.n_lon, _NLEV)) * 0.01
    state = state._replace(tracers_hat=state.tracers_hat.replace(
        data=sh_analysis_3d(g, rq)[..., None]))
    cfg = SpectralNHConfig(
        n_acoustic_substeps=20, semi_implicit_acoustic=True,
        sponge_width=15000.0, sponge_coeff=1.0 / (0.1 * 86400.0),
        hyperdiff_coeff=2.338e15 * 4.0 / x ** 4, small_earth_factor=x,
        rotating=bool(TC2_PARAMS.get("rotating", True)), fix_mass=False)
    model = SpectralCompressibleEulerModel(g, hc, tm, cfg)
    t0 = float(model.compute_tracer_mass(state)[0])
    m0 = float(model.compute_dry_mass(state))
    for _ in range(5):
        state = model.step(state, 6.0)
    rel = float(model.compute_tracer_mass(state)[0]) / t0 - 1.0
    dry = float(model.compute_dry_mass(state)) / m0 - 1.0
    assert np.isfinite(rel) and np.isfinite(dry)
    if abs(rel) >= 1e-12:
        # Pinned to the measured gap (+1.887e-10 at 5 steps, same sign for
        # tracer and dry mass) so a continuity regression cannot hide here.
        assert 1.5e-10 < rel < 2.3e-10, f"terrain residual left the known gap: {rel:.3e}"
        # The gap is the rho' one, inherited exactly (same flux, uniform q).
        assert abs(rel - dry) < 1e-12, (rel, dry)
        pytest.xfail(f"{_TERRAIN_DEFECT} residual {rel:.3e}")
    pytest.fail(f"terrain residual now closes ({rel:.3e}): remove the xfail "
                "and make this a plain budget assertion")
