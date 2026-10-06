"""#1819: GEN_BE 4D-Var in the control variable v (J_b = 1/2|v|^2).

The v-space cost needs only B^{1/2}, so it is defined on MPAS meshes where
B^{-1} is not; where B^{-1} exists it is the same cost."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.da.control_vector import build_control_spec, state_to_control
from legoesm.da.cost_function import build_cost_fn, build_vspace_cost_fn
from legoesm.da.gen_be import GenBEParams, GenBETransform
from legoesm.da.incremental import IncrementalConfig, incremental_4dvar
from legoesm.da.observation import DirectObsOperator, Observation

NLEV = 2


class _Identity:
    def step(self, state, dt):
        return state


def _template(grid):
    n = grid.grid_n_columns
    f3 = lambda val: grid.from_columns(jnp.full((n, NLEV), val))
    f2 = lambda val: grid.from_columns(jnp.full((n,), val))
    return HydrostaticState(
        u=Field(data=f3(0.0), name="u", dims=(), units="m/s"),
        v=Field(data=f3(0.0), name="v", dims=(), units="m/s"),
        T=Field(data=f3(280.0), name="T", dims=(), units="K"),
        p_s=Field(data=f2(1.0e5), name="p_s", dims=(), units="Pa"),
        phis=Field(data=f2(0.0), name="phis", dims=(), units="m2/s2"),
        tracers={},
    )


def _transform(grid, wind):
    n_ch = 1 + 3 * NLEV
    reg = np.zeros((n_ch, NLEV))
    reg[1 + 2 * NLEV:, :] = 0.3 * np.eye(NLEV)  # psi -> T balance, as GEN_BE fits
    params = GenBEParams(
        vert_eig_vec=jnp.stack([jnp.eye(NLEV)] * 3),
        vert_eig_val=jnp.stack([jnp.array([4.0, 1.0])] * 3),
        std_ps=jnp.asarray(100.0),
        reg_coeff=jnp.asarray(reg),
        len_scale=jnp.full((n_ch,), 4.0e5),
        tracer_names=(),
        n_levels=NLEV,
        wind_transform=wind,
    )
    template = _template(grid)
    spec = build_control_spec(template, grid, fields=("u", "v", "T", "p_s"))
    return GenBETransform(params, spec, grid, n_diffusion_iter=10), spec, template


def _t_obs(template, cells_value):
    shape = template.T.data.shape
    flat = [np.unravel_index(c, shape[:-1]) for c in (3, 17)]
    idx = tuple(jnp.array([f[k] for f in flat]) for k in range(len(shape) - 1)) + (
        jnp.array([0, 1]),)
    return (Observation(values=jnp.array([cells_value, cells_value]),
                        errors=jnp.array([0.5, 0.5]), time_index=0,
                        operator=DirectObsOperator("T", idx)),)


@pytest.fixture(scope="module")
def mpas():
    from legoesm.grids.factory import create_grid

    grid = create_grid("mpas", 3, lloyd_iterations=0)
    return (grid,) + _transform(grid, "mpas_helmholtz")


@pytest.fixture(scope="module")
def gauss():
    from legoesm.grids.factory import create_grid

    grid = create_grid("gaussian", 10)
    return (grid,) + _transform(grid, "identity")


def test_U_is_linear_with_no_offset_and_vjp_adjoint(mpas):
    _, B, spec, _ = mpas
    U = B.sqrt_multiply
    k = jax.random.split(jax.random.PRNGKey(0), 3)
    a, b = (jax.random.normal(kk, (spec.total_size,)) for kk in k[:2])
    w = jax.random.normal(k[2], U(a).shape)
    scale = float(jnp.linalg.norm(U(a)))
    assert float(jnp.linalg.norm(U(jnp.zeros_like(a)))) == 0.0
    assert float(jnp.linalg.norm(U(a + 2.0 * b) - U(a) - 2.0 * U(b))) < 1e-12 * scale
    _, vjp = jax.vjp(U, a)
    lhs = float(jnp.dot(U(a), w))
    rhs = float(jnp.dot(a, vjp(w)[0]))
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_constant_chi_is_a_null_direction_of_U(mpas):
    """A per-level spatially constant chi control maps to zero increment: the
    gauge the minimiser never excites (its J_o gradient is exactly zero)."""
    _, B, spec, _ = mpas
    off, size, _ = next((e.offset, e.size, e.shape) for e in spec.entries if e.field_name == "v")
    gauge = jnp.zeros(spec.total_size).at[off:off + size].set(1.0)
    v = jax.random.normal(jax.random.PRNGKey(1), (spec.total_size,))
    du = B.sqrt_multiply(v + 3.0 * gauge) - B.sqrt_multiply(v)
    assert float(jnp.max(jnp.abs(du))) < 1e-10 * float(jnp.max(jnp.abs(B.sqrt_multiply(v))))


def test_vspace_cost_equals_xspace_cost_where_the_inverse_exists(gauss):
    """On the Gaussian grid (exact spectral inverse) J_v(v) = J_x(x_b + U v) and
    their directional derivatives agree, for v in the space U^{-1} U preserves."""
    grid, B, spec, template = gauss
    obs = _t_obs(template, 281.0)
    x_b = state_to_control(template, spec)
    args = (_Identity(), x_b, obs, B, spec, template, 1.0, 1)
    J_v = build_vspace_cost_fn(*args)
    J_x = build_cost_fn(*args)
    J_xv = lambda v: J_x(x_b + B.sqrt_multiply(v))

    proj = lambda w: B._inverse(B.sqrt_multiply(w))
    k1, k2 = jax.random.split(jax.random.PRNGKey(2))
    v = proj(jax.random.normal(k1, (spec.total_size,)))
    d = proj(jax.random.normal(k2, (spec.total_size,)))
    assert float(jnp.linalg.norm(proj(v) - v) / jnp.linalg.norm(v)) < 1e-10
    assert float(J_v(v)) == pytest.approx(float(J_xv(v)), rel=1e-9)
    gv, gx = jax.grad(J_v)(v), jax.grad(J_xv)(v)
    assert float(jnp.dot(d, gv)) == pytest.approx(float(jnp.dot(d, gx)), rel=1e-8)


def _run(grid_tuple, use_preconditioning=True, method="lbfgs", n_steps=1):
    _, B, spec, template = grid_tuple
    cfg = IncrementalConfig(n_outer=2, n_inner=60, inner_gtol=1e-10,
                            inner_method=method, use_preconditioning=use_preconditioning)
    obs = _t_obs(template, 283.0)
    obs = tuple(o._replace(time_index=n_steps - 1) for o in obs)
    return incremental_4dvar(_Identity(), template, obs, B, spec,
                             dt=1.0, n_steps=n_steps, config=cfg)


def test_incremental_runs_on_mpas_and_moves_toward_the_obs(mpas):
    analysis, diag = _run(mpas)
    T_cols = np.asarray(mpas[0].to_columns(analysis.T.data))
    assert 280.0 < T_cols[3, 0] < 283.0 and 280.0 < T_cols[17, 1] < 283.0
    assert diag.cost_history[-1] < diag.cost_history[0]


def test_unpreconditioned_genbe_on_mpas_fails_at_entry(mpas):
    with pytest.raises(ValueError, match="use_preconditioning"):
        _run(mpas, use_preconditioning=False)


@pytest.mark.parametrize("name,method,n_steps",
                         [("mpas", "lbfgs", 1), ("gauss", "lbfgs", 1), ("mpas", "cg", 3)])
def test_vspace_analysis_is_the_closed_form_blue(name, method, n_steps, request):
    """Identity model + linear obs: the exact analysis is
    x_b + U J^T (J J^T + R)^{-1} d with J = H U.  The x-space path is NOT a
    reference here: on the Gaussian grid its B^{-1} is exact only on the
    subspace U^{-1}U preserves; this analysis's v lies 51% (in norm) outside
    it, and the x-space analysis misses this one by 10% of the increment."""
    from legoesm.da.control_vector import control_to_state

    grid, B, spec, template = request.getfixturevalue(name)
    obs = _t_obs(template, 283.0)[0]
    x_b = state_to_control(template, spec)
    Hx = lambda x: obs.operator(control_to_state(x, spec, template))
    J = jax.jacrev(lambda v: Hx(x_b + B.sqrt_multiply(v)))(jnp.zeros(spec.total_size))
    v_a = J.T @ jnp.linalg.solve(J @ J.T + jnp.diag(obs.errors ** 2), obs.values - Hx(x_b))
    T_blue = np.asarray(control_to_state(x_b + B.sqrt_multiply(v_a), spec, template).T.data)

    analysis, _ = _run((grid, B, spec, template), method=method, n_steps=n_steps)
    inc_blue = np.max(np.abs(T_blue - 280.0))
    assert inc_blue > 0.1
    assert np.max(np.abs(np.asarray(analysis.T.data) - T_blue)) < 1e-6 * inc_blue


def test_cycling_runs_on_mpas_and_second_cycle_moves_further(mpas):
    from legoesm.da.cycling import CyclingConfig, run_cycling

    grid, B, spec, template = mpas
    inc_cfg = IncrementalConfig(n_outer=1, n_inner=60, inner_gtol=1e-10, inner_method="lbfgs")
    cfg = CyclingConfig(window_length=1, cycle_length=1, dt=1.0, n_cycles=2, incremental=inc_cfg)
    obs = _t_obs(template, 283.0)
    final, diags = run_cycling(_Identity(), template, (obs, obs), B, spec, cfg, grid=grid)
    one, _ = _run(mpas)
    T_final, T_one = (np.asarray(grid.to_columns(s.T.data))[3, 0] for s in (final, one))
    assert len(diags) == 2 and 280.0 < T_one < T_final < 283.0


def test_unpreconditioned_genbe_with_an_inverse_warns(gauss, caplog):
    with caplog.at_level("WARNING", logger="legoesm.da.incremental"):
        _run(gauss, use_preconditioning=False)
    assert "exact only on the subspace" in caplog.text
