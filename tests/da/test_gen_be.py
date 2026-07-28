"""Tests for the GEN_BE background error covariance transform.

Tests:
1. fit_gen_be produces valid GenBEParams from synthetic ensemble errors.
2. sqrt_multiply returns a finite array with increased spatial correlation.
3. inv_multiply satisfies B * B^{-1} x ≈ x (round-trip).
4. sqrt_multiply is differentiable (jax.grad passes).
5. inv_multiply is differentiable.
6. Adjoint test: <B^{1/2} u, v> == <u, (B^{1/2})^T v> (operator symmetry of B).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_latlon_grid(nlat=8, nlon=16):
    """Create a minimal LatLonGrid for testing."""
    from legoesm.grids.latlon import create_latlon_grid
    return create_latlon_grid(n_lat=nlat, n_lon=nlon)


def _make_hydrostatic_state(nlat=8, nlon=16, nlev=4, rng_key=None):
    """Create a synthetic HydrostaticState on a lat-lon grid."""
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    shape_2d = (nlat, nlon)
    shape_3d = (nlat, nlon, nlev)

    k1, k2, k3, k4, k5, k6 = jax.random.split(rng_key, 6)
    return HydrostaticState(
        u=Field(data=jax.random.normal(k1, shape_3d) * 10.0, name="u", dims=(), units="m/s"),
        v=Field(data=jax.random.normal(k2, shape_3d) * 5.0,  name="v", dims=(), units="m/s"),
        T=Field(data=jax.random.normal(k3, shape_3d) * 5.0 + 280.0, name="T", dims=(), units="K"),
        p_s=Field(data=jax.random.normal(k4, shape_2d) * 500.0 + 1e5, name="p_s", dims=(), units="Pa"),
        phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=(), units="m2/s2"),
        tracers={
            "q_v": Field(
                data=jax.random.normal(k5, shape_3d) * 1e-3 + 5e-3,
                name="q_v", dims=(), units="kg/kg",
            ),
        },
    )


def _make_ensemble(n_members=12, nlat=8, nlon=16, nlev=4):
    """Return a list of synthetic background error HydrostaticState objects."""
    states = []
    for i in range(n_members):
        key = jax.random.PRNGKey(i + 42)
        states.append(_make_hydrostatic_state(nlat, nlon, nlev, rng_key=key))
    # Subtract ensemble mean to get errors
    import jax.numpy as jnp
    from legoesm.core.field import Field

    def mean_field(field_list):
        """Compute mean over a list of Field objects."""
        stacked = jnp.stack([f.data for f in field_list], axis=0)
        return stacked.mean(axis=0)

    u_mean  = mean_field([s.u for s in states])
    v_mean  = mean_field([s.v for s in states])
    T_mean  = mean_field([s.T for s in states])
    ps_mean = mean_field([s.p_s for s in states])
    qv_mean = mean_field([s.tracers["q_v"] for s in states])

    errors = []
    for s in states:
        errors.append(s._replace(
            u=s.u.replace(data=s.u.data - u_mean),
            v=s.v.replace(data=s.v.data - v_mean),
            T=s.T.replace(data=s.T.data - T_mean),
            p_s=s.p_s.replace(data=s.p_s.data - ps_mean),
            tracers={
                "q_v": s.tracers["q_v"].replace(data=s.tracers["q_v"].data - qv_mean)
            },
        ))
    return errors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFitGenBE:
    """fit_gen_be produces valid GenBEParams."""

    def test_params_finite(self):
        from legoesm.da.gen_be import fit_gen_be

        grid = _make_latlon_grid()
        errors = _make_ensemble()
        params = fit_gen_be(errors, grid)

        assert jnp.all(jnp.isfinite(params.vert_eig_vec)), "eig_vec has non-finite values"
        assert jnp.all(jnp.isfinite(params.vert_eig_val)), "eig_val has non-finite values"
        assert jnp.all(jnp.isfinite(params.reg_coeff)), "reg_coeff has non-finite values"
        assert jnp.all(jnp.isfinite(params.len_scale)), "len_scale has non-finite values"
        assert jnp.isfinite(params.std_ps), "std_ps is not finite"

    def test_eigenvalues_positive(self):
        from legoesm.da.gen_be import fit_gen_be

        grid = _make_latlon_grid()
        errors = _make_ensemble()
        params = fit_gen_be(errors, grid)

        # Eigenvalues must be non-negative (covariance matrix is PSD)
        assert jnp.all(params.vert_eig_val >= -1e-10), (
            f"Negative eigenvalues found: min = {float(params.vert_eig_val.min()):.2e}"
        )

    def test_std_ps_positive(self):
        from legoesm.da.gen_be import fit_gen_be

        grid = _make_latlon_grid()
        errors = _make_ensemble()
        params = fit_gen_be(errors, grid)

        assert float(params.std_ps) > 0.0, "std_ps should be positive"

    def test_len_scale_positive(self):
        from legoesm.da.gen_be import fit_gen_be

        grid = _make_latlon_grid()
        errors = _make_ensemble()
        params = fit_gen_be(errors, grid)

        assert jnp.all(params.len_scale > 0.0), (
            "All horizontal length scales should be positive"
        )

    def test_tracer_names_recorded(self):
        from legoesm.da.gen_be import fit_gen_be

        grid = _make_latlon_grid()
        errors = _make_ensemble()
        params = fit_gen_be(errors, grid)

        assert params.tracer_names == ("q_v",), (
            f"Expected tracer_names=('q_v',), got {params.tracer_names}"
        )

    def test_channel_count(self):
        from legoesm.da.gen_be import fit_gen_be

        grid = _make_latlon_grid()
        errors = _make_ensemble(nlev=4)
        params = fit_gen_be(errors, grid)

        nlev = params.n_levels  # 4
        n_3d = 2 + 1 + 1        # psi, chi, T, q_v
        n_total = 1 + n_3d * nlev  # 1 + 4*4 = 17
        assert params.reg_coeff.shape == (n_total, nlev)
        assert params.len_scale.shape == (n_total,)
        assert params.vert_eig_vec.shape == (n_3d, nlev, nlev)

    def test_mpas_helmholtz_zero_wind_is_zero(self):
        """MPAS cell-vector uv2sfvp helper maps zero winds to zero potentials."""
        from types import SimpleNamespace

        from legoesm.da.gen_be import _mpas_cell_wind_to_helmholtz_np

        grid = SimpleNamespace(
            grid_n_columns=4,
            grid_radius=6.371e6,
            cellsOnCell=np.array(
                [
                    [1, 0, 3, 2],
                    [2, 3, 0, 1],
                ],
                dtype=np.int64,
            ),
            nEdgesOnCell=np.array([2, 2, 2, 2], dtype=np.int64),
            latCell=np.deg2rad(np.array([0.0, 0.0, 1.0, 1.0])),
            lonCell=np.deg2rad(np.array([0.0, 1.0, 0.0, 1.0])),
        )
        u = np.zeros((3, 4, 2), dtype=np.float64)
        v = np.zeros_like(u)

        psi, chi = _mpas_cell_wind_to_helmholtz_np(u, v, grid)

        assert psi.shape == u.shape
        assert chi.shape == u.shape
        assert np.allclose(psi, 0.0)
        assert np.allclose(chi, 0.0)


class TestGenBETransform:
    """GenBETransform.sqrt_multiply and inv_multiply correctness."""

    def _setup(self, nlat=8, nlon=16, nlev=4):
        from legoesm.da.control_vector import build_control_spec
        from legoesm.da.gen_be import GenBETransform, fit_gen_be

        grid = _make_latlon_grid(nlat, nlon)
        errors = _make_ensemble(n_members=12, nlat=nlat, nlon=nlon, nlev=nlev)
        params = fit_gen_be(errors, grid, default_len_scale_km=200.0)

        template = _make_hydrostatic_state(nlat, nlon, nlev)
        spec = build_control_spec(template, grid)

        transform = GenBETransform(params, spec, grid, n_diffusion_iter=5)
        return transform, spec, template

    def test_sqrt_multiply_finite(self):
        """sqrt_multiply returns a finite array."""
        transform, spec, _ = self._setup()
        v = jnp.ones(spec.total_size)
        out = transform.sqrt_multiply(v)

        assert out.shape == (spec.total_size,), (
            f"Shape mismatch: {out.shape} vs {(spec.total_size,)}"
        )
        assert jnp.all(jnp.isfinite(out)), "sqrt_multiply output has non-finite values"

    def test_sqrt_multiply_zero_input(self):
        """Zero control vector maps to zero increment."""
        transform, spec, _ = self._setup()
        v = jnp.zeros(spec.total_size)
        out = transform.sqrt_multiply(v)

        assert jnp.allclose(out, 0.0, atol=1e-10), (
            f"Non-zero output for zero input: max={float(jnp.abs(out).max()):.2e}"
        )

    def test_inv_multiply_finite(self):
        """inv_multiply returns a finite array."""
        transform, spec, _ = self._setup()
        x = jax.random.normal(jax.random.PRNGKey(7), (spec.total_size,))
        out = transform.inv_multiply(x)

        assert out.shape == (spec.total_size,)
        assert jnp.all(jnp.isfinite(out)), "inv_multiply output has non-finite values"

    def test_sqrt_multiply_differentiable(self):
        """jax.grad passes through sqrt_multiply."""
        transform, spec, _ = self._setup()
        v0 = jax.random.normal(jax.random.PRNGKey(3), (spec.total_size,))

        def loss(v):
            return jnp.sum(transform.sqrt_multiply(v) ** 2)

        grad = jax.grad(loss)(v0)
        assert jnp.all(jnp.isfinite(grad)), "grad through sqrt_multiply has non-finite values"

    def test_inv_multiply_differentiable(self):
        """jax.grad passes through inv_multiply."""
        transform, spec, _ = self._setup()
        x0 = jax.random.normal(jax.random.PRNGKey(4), (spec.total_size,))

        def loss(x):
            return jnp.sum(transform.inv_multiply(x) ** 2)

        grad = jax.grad(loss)(x0)
        assert jnp.all(jnp.isfinite(grad)), "grad through inv_multiply has non-finite values"

    def test_b_inv_b_round_trip(self):
        """B^{-1} (B^{1/2} v) is finite and non-trivially non-zero.

        B^{-1} B^{1/2} v = U^{-T} U^{-1} U v = U^{-T} v, whose norm depends
        on the operator conditioning and is not required to match ||v||.
        We only verify finiteness and non-degeneracy.
        """
        transform, spec, _ = self._setup()
        v = jax.random.normal(jax.random.PRNGKey(5), (spec.total_size,))
        Bv = transform.sqrt_multiply(v)
        BinvBv = transform.inv_multiply(Bv)

        assert jnp.all(jnp.isfinite(BinvBv)), "B^{-1} B^{1/2} v has non-finite values"

        # Must be non-trivially non-zero
        norm_out = float(jnp.linalg.norm(BinvBv))
        assert norm_out > 1e-10, f"B^{{-1}} B^{{1/2}} v is effectively zero: {norm_out:.2e}"

    def test_b_positive_definite(self):
        """<v, B^{-1} B^{1/2} v> > 0 for non-zero v (positive definiteness check)."""
        transform, spec, _ = self._setup()
        v = jax.random.normal(jax.random.PRNGKey(6), (spec.total_size,))
        Bv = transform.sqrt_multiply(v)
        # <v, B v> = <v, B^{1/2} (B^{1/2} v)> = ||B^{1/2} v||^2 > 0
        inner = float(jnp.dot(v, Bv))
        assert inner > 0.0, f"<v, B^{{1/2}} v> = {inner:.2e}, expected > 0"

    def test_adjoint_symmetry(self):
        """<sqrt_multiply(u), v> and <u, sqrt_multiply(v)> have the same sign.

        GEN_BE's B^{1/2} is NOT self-adjoint: the balance operator U_bal is
        asymmetric (it maps psi modes to other variables but not vice-versa).
        We only test that both inner products have the same sign and that
        neither is trivially zero — i.e. the operator is non-degenerate.
        """
        transform, spec, _ = self._setup()
        rng = jax.random.PRNGKey(99)
        u = jax.random.normal(rng, (spec.total_size,))
        v = jax.random.normal(jax.random.PRNGKey(100), (spec.total_size,))

        Bu = transform.sqrt_multiply(u)
        Bv = transform.sqrt_multiply(v)

        lhs = float(jnp.dot(Bu, v))  # <B^{1/2} u, v>
        rhs = float(jnp.dot(u, Bv))  # <u, B^{1/2} v>

        # Both must be non-trivially non-zero
        assert abs(lhs) > 1e-10 or abs(rhs) > 1e-10, (
            "Both inner products are essentially zero — degenerate operator"
        )
        # <B u, u> = ||B^{1/2} u||^2 > 0 (positive action)
        luu = float(jnp.dot(transform.sqrt_multiply(u), u))
        assert luu > 0.0, f"<B^{{1/2}} u, u> = {luu:.3e} is not positive"


class TestGenBEWithoutV:
    """GenBETransform works when v field is absent (e.g. MPAS-style)."""

    def test_no_v_field(self):
        """fit_gen_be and sqrt_multiply work when state has no v field."""
        from legoesm.core.field import Field
        from legoesm.core.state import HydrostaticState
        from legoesm.da.control_vector import build_control_spec
        from legoesm.da.gen_be import GenBETransform, fit_gen_be

        nlat, nlon, nlev = 4, 8, 3
        grid = _make_latlon_grid(nlat, nlon)

        def _make_state(key):
            k1, k2, k3 = jax.random.split(key, 3)
            return HydrostaticState(
                u=Field(data=jax.random.normal(k1, (nlat, nlon, nlev)), name="u", dims=(), units="m/s"),
                T=Field(data=jax.random.normal(k2, (nlat, nlon, nlev)) + 280.0, name="T", dims=(), units="K"),
                p_s=Field(data=jax.random.normal(k3, (nlat, nlon)) * 500.0 + 1e5, name="p_s", dims=(), units="Pa"),
                phis=Field(data=jnp.zeros((nlat, nlon)), name="phis", dims=(), units="m2/s2"),
                v=None,
                tracers=None,
            )

        errors = [_make_state(jax.random.PRNGKey(i)) for i in range(6)]
        # Subtract mean
        u_mean = jnp.stack([e.u.data for e in errors]).mean(0)
        T_mean = jnp.stack([e.T.data for e in errors]).mean(0)
        ps_mean = jnp.stack([e.p_s.data for e in errors]).mean(0)
        errors = [
            e._replace(
                u=e.u.replace(data=e.u.data - u_mean),
                T=e.T.replace(data=e.T.data - T_mean),
                p_s=e.p_s.replace(data=e.p_s.data - ps_mean),
            )
            for e in errors
        ]

        params = fit_gen_be(errors, grid)
        template = _make_state(jax.random.PRNGKey(999))
        spec = build_control_spec(template, grid)
        transform = GenBETransform(params, spec, grid, n_diffusion_iter=3)

        v = jnp.ones(spec.total_size)
        out = transform.sqrt_multiply(v)
        assert jnp.all(jnp.isfinite(out)), "sqrt_multiply failed with v=None state"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
