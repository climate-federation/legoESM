"""``cloud_optics_inhomogeneity="two_region"``: ecRad McICA gamma scaling per
max_random subcolumn cell (clouds/subcolumns.ecrad_inhomogeneity_scaling).

Reference: ecRad 49r1 radiation_cloud_generator.F90 lines 187-194 (op_inhom =
op**(1/ratio)) and 350-363 (repeat the draw of the cell above with probability
op_inhom, else a fresh uniform; od *= sample_from_pdf(fsd, u)), with
set_overlap_param and the IFS decorrelation length 0.75 + 2.149 cos^2(lat) km,
ratio 0.5 (cloud_overlap_decorr_len, NDECOLAT=2).
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.clouds import subcolumns as sub
from legoesm.atmosphere.physics.radiation.ecrad_ifs.mcica_generator import (
    sample_from_pdf,
)

jax.config.update("jax_enable_x64", True)


def _column(ncol=16, nlev=10, top_pa=2.0e4):
    p_half = np.linspace(top_pa, 1.0e5, nlev + 1)[None, :] + np.linspace(0.0, 0.9, ncol)[:, None]
    T_half = np.linspace(220.0, 290.0, nlev + 1)[None, :].repeat(ncol, 0)
    lat = np.linspace(-1.4, 1.4, ncol)
    return jnp.asarray(p_half), jnp.asarray(T_half), jnp.asarray(lat)


def _cf(ncol, nlev):
    cf = np.zeros((ncol, nlev))
    cf[:, 2:5] = [0.3, 0.6, 0.5]
    cf[:, 7] = 0.8
    return jnp.asarray(cf)


def _reference(mask, fsd, lat, p_half, T_half, r1, r2):
    """Plain-loop ecRad recursion with the overlap parameter written out."""
    L = (0.75 + 2.149 * np.cos(lat) ** 2) * 1000.0
    rg = constants.R_d / constants.g
    p, T = np.asarray(p_half), np.asarray(T_half)
    n_sub, ncol, nlev = mask.shape
    op = np.empty((ncol, nlev - 1))
    op[:, 0] = np.exp(-(rg / L) * T[:, 1] * np.log(p[:, 2] / p[:, 1]))
    for j in range(1, nlev - 1):
        op[:, j] = np.exp(-(0.5 * rg / L) * T[:, j + 1] * np.log(p[:, j + 2] / p[:, j]))
    opi = op ** 2.0
    out = np.zeros(mask.shape)
    for s in range(n_sub):
        for c in range(ncol):
            c_prev = u_prev = 0.0
            for k in range(nlev):
                w = c_prev * (r2[s, c, k] < opi[c, k - 1]) if k > 0 else 0.0
                u = w * u_prev + (1.0 - w) * r1[s, c, k]
                out[s, c, k] = mask[s, c, k] * float(
                    sample_from_pdf(jnp.asarray(fsd), jnp.asarray(u), sharpness=math.inf))
                c_prev, u_prev = mask[s, c, k], u
    return out


def _uniforms(n_sub, nlev, shift):
    return [np.asarray(r) for r in sub.inhomogeneity_uniforms(n_sub, nlev, shift, jnp.float64)]


def test_matches_the_ecrad_recursion_on_the_lane_mask():
    ncol, nlev, n_sub, fsd = 6, 10, 8, 0.9
    p_half, T_half, lat = _column(ncol, nlev)
    shift = jnp.linspace(0.05, 0.95, ncol)
    mask = sub.generate_subcolumns(_cf(ncol, nlev), n_sub, shift=shift, width=0.0)
    got = np.asarray(sub.ecrad_inhomogeneity_scaling(mask, fsd, lat, p_half, T_half, shift=shift))
    r1, r2 = _uniforms(n_sub, nlev, shift)
    want = _reference(np.asarray(mask, np.float64), fsd, np.asarray(lat), p_half, T_half, r1, r2)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    assert np.any(np.asarray(mask)) and np.all(got[~np.asarray(mask)] == 0.0)


@pytest.mark.parametrize("width", [0.0, 0.05])
def test_homogeneous_cloud_reproduces_the_plain_subcolumn_paths(width):
    """fsd = 0: in-cloud path x scaling is exactly the plain lane's path, for
    boolean and smooth masks alike (cover is weighted once, not squared)."""
    ncol, nlev, n_sub = 4, 10, 8
    p_half, T_half, lat = _column(ncol, nlev)
    shift = jnp.full(ncol, 0.3)
    cf = _cf(ncol, nlev)
    lwp = jnp.asarray(np.random.default_rng(3).random((ncol, nlev))) * cf
    mask = sub.generate_subcolumns(cf, n_sub, shift=shift, width=width)
    x = sub.ecrad_inhomogeneity_scaling(mask, 0.0, lat, p_half, T_half, shift=shift)
    ic, _ = sub.in_cloud_paths(cf, lwp, lwp)
    plain, _ = sub.subcolumn_paths(mask, cf, lwp, lwp)
    np.testing.assert_allclose(np.reshape(np.asarray(x * ic[None]), (n_sub * ncol, nlev)),
                               np.asarray(plain), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("n_sub", [8, 1])
def test_cell_scaling_has_unit_mean_and_the_requested_spread(n_sub):
    ncol, nlev, fsd = 4096, 4, 0.75
    p_half, T_half, lat = _column(ncol, nlev)
    mask = jnp.ones((n_sub, ncol, nlev), bool)
    shift = jnp.asarray(np.random.default_rng(0).random(ncol))
    x = np.asarray(sub.ecrad_inhomogeneity_scaling(mask, fsd, lat, p_half, T_half, shift=shift))
    assert abs(x.mean() - 1.0) < 0.02
    assert abs(x.std() / x.mean() - fsd) < 0.05


def test_inhomogeneities_decorrelate_with_layer_separation():
    def corr(dp):
        ncol, n_sub = 512, 8
        p_half = jnp.asarray(np.array([5.0e4, 5.0e4 + 50.0, 5.0e4 + 50.0 + dp,
                                       5.0e4 + 100.0 + 2 * dp])[None, :].repeat(ncol, 0))
        T_half = jnp.full((ncol, 4), 260.0)
        shift = jnp.asarray(np.random.default_rng(1).random(ncol))
        x = np.asarray(sub.ecrad_inhomogeneity_scaling(
            jnp.ones((n_sub, ncol, 3), bool), 0.75, jnp.zeros(ncol), p_half, T_half, shift=shift))
        return np.corrcoef(x[..., 1].ravel(), x[..., 2].ravel())[0, 1]
    near, far = corr(100.0), corr(4.0e4)
    assert near > 0.9 and far < 0.3, (near, far)


def test_inhomogeneity_streams_are_not_the_mask_stream():
    u_mask = np.asarray(sub._shifted_uniforms(8, 10, jnp.full(3, 0.4), jnp.float64))
    r1, r2 = _uniforms(8, 10, jnp.full(3, 0.4))
    assert not np.allclose(u_mask, r1) and not np.allclose(u_mask, r2)
    assert not np.allclose(r1, r2)


def test_gradient_in_fsd_matches_finite_difference():
    ncol, nlev = 4, 10
    p_half, T_half, lat = _column(ncol, nlev)
    shift = jnp.linspace(0.1, 0.9, ncol)
    mask = sub.generate_subcolumns(_cf(ncol, nlev), 8, shift=shift, width=0.05)
    paths = jnp.asarray(np.random.default_rng(2).random((ncol, nlev)))

    def f(fsd):
        return jnp.sum(sub.ecrad_inhomogeneity_scaling(
            mask, fsd, lat, p_half, T_half, shift=shift) * paths[None])
    fsd = 0.825    # mid-way between gamma-table knots (step 0.05)
    g = float(jax.grad(f)(fsd))
    eps = 1e-6
    fd = (float(f(fsd + eps)) - float(f(fsd - eps))) / (2 * eps)
    assert np.isfinite(g) and g != 0.0
    assert g == pytest.approx(fd, rel=1e-5)


def test_jit_parity():
    """Recursion at 1e-12 (fp64); the integer-hashed uniforms bit-identical."""
    from legoesm.atmosphere.physics.radiation.ecrad_ifs.mcica_generator import (
        inhomogeneity_od_scaling)
    ncol, nlev, n_sub = 4, 10, 8
    shift = jnp.linspace(0.1, 0.9, ncol)
    mask = sub.generate_subcolumns(_cf(ncol, nlev), n_sub, shift=shift, width=0.0).astype(jnp.float64)
    r1, r2 = (jnp.asarray(r) for r in _uniforms(n_sub, nlev, shift))
    repeat = (r2 < 0.6).astype(jnp.float64)
    f = lambda m, rep, r: inhomogeneity_od_scaling(m, rep, jnp.asarray(0.75), r)  # noqa: E731
    np.testing.assert_allclose(np.asarray(jax.jit(f)(mask, repeat, r1)),
                               np.asarray(f(mask, repeat, r1)), rtol=1e-12, atol=1e-12)
    g = lambda s: sub.inhomogeneity_uniforms(n_sub, nlev, s, jnp.float64)  # noqa: E731
    for a, b in zip(jax.jit(g)(shift), g(shift)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_fp32_jit_parity_of_the_full_scaling():
    """fp32: the draw must not depend on compile mode (a float offset of a
    ~5e4 argument once wrapped to different draws, scalings differing by 5)."""
    ncol, nlev, n_sub = 128, 36, 8
    p_half, T_half, lat = (jnp.asarray(a, jnp.float32) for a in _column(ncol, nlev))
    shift = jnp.arange(ncol, dtype=jnp.float32) / ncol
    cf = jnp.asarray(np.random.default_rng(5).random((ncol, nlev)), jnp.float32)
    mask = sub.generate_subcolumns(cf, n_sub, shift=shift, width=0.0)
    f = lambda m, s: sub.ecrad_inhomogeneity_scaling(m, 0.75, lat, p_half, T_half, shift=s)  # noqa: E731
    np.testing.assert_allclose(np.asarray(jax.jit(f)(mask, shift)), np.asarray(f(mask, shift)),
                               rtol=1e-5, atol=1e-6)


def test_backend_hands_the_solver_the_scaled_paths(monkeypatch):
    """Through the REAL radiation backend: the solver receives in-cloud path x
    the ecRad cell scaling, not the plain subcolumn path."""
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.radiation import integration as rad_int
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import cam_interface_temperature

    class _Stop(Exception):
        pass

    def run(inhom):
        seen = {}

        class _Solver:
            def solve_columns(self, **kw):
                seen.update(kw)
                raise _Stop()

        cc = CloudConfig(scheme="sundqvist", cloud_vertical_overlap_optics="max_random",
                         cloud_n_subcolumns=n_sub, cloud_subcolumn_mask_width=0.0,
                         cloud_optics_inhomogeneity=inhom, cloud_fsd=1.0)
        cfg = RadiationConfig(scheme="rrtmgp", cloud_scheme="sundqvist", cloud_config=cc)
        with pytest.raises(_Stop):
            rad_int._call_radiation_backend(
                cfg, T=T, p_full=p_full, p_half=p_half, sfc_temperature=t_sfc, lat=lat,
                q_v=jnp.full((ncol, nlev), 1e-4), insolation=jnp.zeros(ncol),
                q_cloud=jnp.full((ncol, nlev), 1e-5), q_ice=jnp.zeros((ncol, nlev)),
                rrtmgp_solver=_Solver())
        return np.asarray(seen["cloud_path_liq"]).reshape(n_sub, ncol, nlev), seen

    ncol, nlev, n_sub = 8, 10, 8
    p_half, _, lat = _column(ncol, nlev)
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    T = jnp.asarray(np.linspace(220.0, 285.0, nlev)[None, :].repeat(ncol, 0))
    t_sfc = jnp.full(ncol, 290.0)
    cf = _cf(ncol, nlev)
    real = rad_int.compute_cloud_properties
    monkeypatch.setattr(rad_int, "compute_cloud_properties",
                        lambda *a, **k: real(*a, **k)._replace(cloud_fraction=cf))
    plain, _ = run("constant")
    scaled, _ = run("two_region")
    shift = sub.cam_seed_shift(p_full)
    mask = sub.generate_subcolumns(cf, n_sub, shift=shift, width=0.0)
    x = np.asarray(sub.ecrad_inhomogeneity_scaling(
        mask, 1.0, lat, p_half, cam_interface_temperature(T, p_full, p_half, t_sfc), shift=shift))
    np.testing.assert_allclose(scaled, plain * x, rtol=1e-12, atol=0.0)
    assert np.any(np.abs(x[np.asarray(mask)] - 1.0) > 0.1)


@pytest.mark.parametrize("n_sub", [8, 1])
@pytest.mark.parametrize("cf_value", [0.3, 0.7])
def test_partly_cloudy_cells_see_an_unbiased_draw(n_sub, cf_value):
    """The inhomogeneity uniforms must be independent of the mask draw: the
    mean scaling over the cells a partly cloudy layer makes cloudy is 1."""
    ncol, nlev, fsd = 4096, 3, 0.75
    p_half, T_half, lat = _column(ncol, nlev)
    shift = jnp.asarray(np.random.default_rng(4).random(ncol))
    cf = jnp.full((ncol, nlev), cf_value)
    mask = sub.generate_subcolumns(cf, n_sub, shift=shift, width=0.0)
    x = np.asarray(sub.ecrad_inhomogeneity_scaling(mask, fsd, lat, p_half, T_half, shift=shift))
    m = np.asarray(mask)[..., 0]
    assert m.sum() > 500
    assert abs(x[..., 0][m].mean() - 1.0) < 0.03


def test_single_layer_is_refused():
    p_half, T_half, lat = _column(2, 1)
    with pytest.raises(ValueError, match=">= 2 layers"):
        sub.ecrad_inhomogeneity_scaling(jnp.ones((8, 2, 1), bool), 0.75, lat, p_half, T_half)
