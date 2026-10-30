"""#1819: GEN_BE's MPAS B^{-1} is refused (not implemented), and the probe that
shows it is not implementable at production settings measures what it claims."""
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_PROBE = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "validate" / "genbe_mpas_inverse_conditioning_1819.py")


def _probe():
    spec = importlib.util.spec_from_file_location("genbe_probe_1819", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mesh():
    from legoesm.grids.factory import create_grid

    return create_grid("mpas", 3, lloyd_iterations=0)


def test_power_iteration_finds_the_dense_lambda_min(mesh):
    probe = _probe()
    lap, _, _ = probe.genbe_laplacian(mesh)
    n = mesh.grid_n_columns
    dense = np.asarray(jax.jacfwd(lap)(jnp.zeros(n)))
    exact = float(np.linalg.eigvals(dense).real.min())
    assert probe.lambda_min(lap, n) == pytest.approx(exact, rel=1e-3)


def test_attenuation_tracks_the_closed_form_and_the_production_smoother(mesh):
    """GenBETransform's own forward smoothing multiplies the probe's most
    negative eigenmode by exactly the probe's predicted factor."""
    probe = _probe()
    n_iter, len_scale = 400, 1.5e6
    lap, step, transform = probe.genbe_laplacian(mesh, len_scale_m=len_scale, n_iter=n_iter)
    n = mesh.grid_n_columns
    dense = np.asarray(jax.jacfwd(lap)(jnp.zeros(n)))
    vals, vecs = np.linalg.eig(dense)
    k = int(np.argmin(vals.real))
    mode = jnp.asarray(vecs[:, k].real)
    channels = jnp.tile(mode[:, None], (1, transform._n_total_ch))
    out = transform._diffusion_smooth(channels)[:, 1]
    measured = float(jnp.dot(out, mode) / jnp.dot(mode, mode))
    assert float(jnp.linalg.norm(out - measured * mode) / jnp.linalg.norm(out)) < 1e-6
    discrete, closed = probe.attenuation(float(vals[k].real), step, len_scale, n_iter)
    assert measured == pytest.approx(discrete, rel=1e-6)
    assert discrete < 1e-2
    assert closed == pytest.approx(discrete, rel=0.5)


def test_inv_multiply_refuses_a_real_voronoi_mesh(mesh):
    from legoesm.da.control_vector import ControlEntry, ControlVectorSpec
    from legoesm.da.gen_be import GenBEParams, GenBETransform

    n = mesh.grid_n_columns
    entries, off = [], 0
    for name, shape in (("u", (n, 1)), ("T", (n, 1)), ("p_s", (n,))):
        entries.append(ControlEntry(name, off, int(np.prod(shape)), shape, "identity"))
        off += int(np.prod(shape))
    params = GenBEParams(
        vert_eig_vec=jnp.stack([jnp.eye(1)] * 3), vert_eig_val=jnp.ones((3, 1)),
        std_ps=jnp.asarray(1.0), reg_coeff=jnp.zeros((4, 1)),
        len_scale=jnp.full((4,), 5.0e5), tracer_names=(), n_levels=1,
        wind_transform="identity",
    )
    t = GenBETransform(params, ControlVectorSpec(tuple(entries), off), mesh, n_diffusion_iter=20)
    v = jnp.ones(off)
    assert jnp.all(jnp.isfinite(t.sqrt_multiply(v)))
    with pytest.raises(NotImplementedError, match="exp\\(-L\\^2"):
        t.inv_multiply(v)
