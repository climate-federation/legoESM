"""#1819 opt-in: implicit (Matern) GenBE horizontal correlation on MPAS meshes,
U_h = (I - sL)^{-2}, whose inverse (I - sL)^2 is exact."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.da.control_vector import build_control_spec
from legoesm.da.gen_be import GenBEParams, GenBETransform

NLEV = 2
LEN_SCALE_M = 5.0e5
# Stated tolerance.  Measured on this mesh (float64): 7.3e-12 for B^-1 B x,
# 1.3e-13 for U^-1 U v; half the Chebyshev count gives 7e-7.
INVERSE_RTOL = 1e-10


def _transform(grid, scheme="implicit_matern", len_scale=None, wind="identity"):
    kw = {} if scheme is None else {"horizontal_scheme": scheme}
    n = grid.grid_n_columns
    f3 = lambda val: grid.from_columns(jnp.full((n, NLEV), val))
    f2 = lambda val: grid.from_columns(jnp.full((n,), val))
    template = HydrostaticState(
        u=Field(data=f3(0.0), name="u", dims=(), units="m/s"),
        v=Field(data=f3(0.0), name="v", dims=(), units="m/s"),
        T=Field(data=f3(280.0), name="T", dims=(), units="K"),
        p_s=Field(data=f2(1.0e5), name="p_s", dims=(), units="Pa"),
        phis=Field(data=f2(0.0), name="phis", dims=(), units="m2/s2"),
        tracers={},
    )
    n_ch = 1 + 3 * NLEV
    reg = np.zeros((n_ch, NLEV))
    reg[1 + 2 * NLEV:, :] = 0.3 * np.eye(NLEV)
    params = GenBEParams(
        vert_eig_vec=jnp.stack([jnp.eye(NLEV)] * 3),
        vert_eig_val=jnp.stack([jnp.array([4.0, 1.0])] * 3),
        std_ps=jnp.asarray(100.0),
        reg_coeff=jnp.asarray(reg),
        len_scale=jnp.full((n_ch,), LEN_SCALE_M) if len_scale is None else len_scale,
        tracer_names=(),
        n_levels=NLEV,
        wind_transform=wind,
    )
    spec = build_control_spec(template, grid, fields=("u", "v", "T", "p_s"))
    return GenBETransform(params, spec, grid, n_diffusion_iter=400, **kw), spec


@pytest.fixture(scope="module")
def production():
    """The 40962-cell production MPAS mesh (resolution 6)."""
    from legoesm.grids.factory import create_grid

    grid = create_grid("mpas", 6, lloyd_iterations=0)
    assert grid.grid_n_columns == 40962
    return (grid,) + _transform(grid)


def _white(spec, seed):
    return jax.random.normal(jax.random.PRNGKey(seed), (spec.total_size,))


def test_inverse_consistency_on_the_production_mesh(production):
    """||B^-1 B x - x|| / ||x|| on white noise (dominated by grid-scale modes,
    the ones the explicit kernel attenuates by ~1e-30)."""
    _, B, spec = production
    U = B.sqrt_multiply
    for seed in (0, 1):
        x = _white(spec, seed)
        _, Ut = jax.vjp(U, jnp.zeros_like(x))
        Bx = U(Ut(x)[0])
        err = jnp.linalg.norm(B.inv_multiply(Bx) - x) / jnp.linalg.norm(x)
        assert float(err) < INVERSE_RTOL


def test_U_inverse_U_is_identity_on_the_production_mesh(production):
    _, B, spec = production
    v = _white(spec, 2)
    err = jnp.linalg.norm(B._inverse(B.sqrt_multiply(v)) - v) / jnp.linalg.norm(v)
    assert float(err) < INVERSE_RTOL


def test_horizontal_smooth_roundtrip(production):
    _, B, _ = production
    b = jax.random.normal(jax.random.PRNGKey(3), (B._ncol, B._n_total_ch))
    y = B._horiz_smooth(b) / B._horiz_norm[None, :]
    back = B._horiz_smooth(y * B._horiz_norm[None, :], inverse=True)
    assert float(jnp.linalg.norm(back - b) / jnp.linalg.norm(b)) < INVERSE_RTOL


def test_horizontal_kernel_is_a_nonnegative_bump(production):
    """(I - sL)^{-1} is an M-matrix inverse: the response to a point is >= 0
    everywhere and largest at the point."""
    _, B, _ = production
    c = 1234
    delta = jnp.zeros((B._ncol, B._n_total_ch)).at[c, :].set(1.0)
    k = np.asarray(B._horiz_smooth(delta))
    assert k.min() > -1e-12 * k.max()
    assert np.all(np.argmax(k, axis=0) == c)


def test_default_scheme_is_explicit_and_still_refuses_the_inverse(production):
    grid, _, spec = production
    B, spec = _transform(grid, scheme=None)
    assert B.horizontal_scheme == "explicit"
    with pytest.raises(NotImplementedError):
        B.inv_multiply(_white(spec, 4))


def test_unknown_scheme_and_non_mpas_grid_raise():
    from legoesm.grids.factory import create_grid

    mpas = create_grid("mpas", 2, lloyd_iterations=0)
    with pytest.raises(ValueError, match="unknown horizontal_scheme"):
        _transform(mpas, scheme="implicit")
    with pytest.raises(ValueError, match="needs an MPAS mesh"):
        _transform(create_grid("gaussian", 10))


def test_psi_chi_wind_still_refuses_the_inverse(production):
    """Constant psi/chi are null modes of the wind transform, whatever the
    horizontal kernel."""
    grid, _, spec = production
    B, spec = _transform(grid, wind="mpas_helmholtz")
    with pytest.raises(NotImplementedError):
        B.inv_multiply(_white(spec, 5))


def test_per_channel_length_scales(production):
    """Each channel smooths with its own length scale: a mixed-scale transform
    matches, channel by channel, transforms built with that scale everywhere."""
    grid, B0, _ = production
    n_ch = B0._n_total_ch
    scales = jnp.linspace(2.0e5, 1.0e6, n_ch)
    B, _ = _transform(grid, len_scale=scales)
    b = jax.random.normal(jax.random.PRNGKey(6), (B._ncol, n_ch))
    mixed = B._horiz_smooth(b)
    for j in range(n_ch):
        Bj, _ = _transform(grid, len_scale=jnp.full((n_ch,), scales[j]))
        ref = Bj._horiz_smooth(b)[:, j]
        assert float(jnp.linalg.norm(mixed[:, j] - ref) / jnp.linalg.norm(ref)) < INVERSE_RTOL


@pytest.mark.parametrize("bad, match", [(jnp.nan, "must be finite"),
                                        (-5.0e5, "must be finite and >= 0"),
                                        (1.0e8, "Chebyshev iterations per solve"),
                                        # convergence rate rounds to 1; bound overflows
                                        (1.0e21, "Chebyshev iterations per solve"),
                                        (1.0e200, "Chebyshev iterations per solve")])
def test_bad_length_scale_raises(production, bad, match):
    grid, B0, _ = production
    with pytest.raises(ValueError, match=match):
        _transform(grid, len_scale=jnp.full((B0._n_total_ch,), LEN_SCALE_M).at[2].set(bad))


def test_building_under_grad_wrt_length_scale_refuses(production):
    grid, B0, spec = production
    v = _white(spec, 7)

    def f(L):
        B, _ = _transform(grid, len_scale=jnp.full((B0._n_total_ch,), L))
        return jnp.sum(B.sqrt_multiply(v))

    with pytest.raises(ValueError, match="needs concrete len_scale"):
        jax.grad(f)(LEN_SCALE_M)


def test_ill_conditioned_inverse_refuses(production):
    """B^-1 B x error grows ~kappa^4 eps (measured 2.4e-7 at 2000 km, 1.3e-3 at
    6000 km); past the bound inv_multiply must refuse, not return it."""
    grid, B0, spec = production
    B, _ = _transform(grid, len_scale=jnp.full((B0._n_total_ch,), 2.0e6))
    with pytest.raises(ValueError, match="condition number"):
        B.inv_multiply(_white(spec, 8))
