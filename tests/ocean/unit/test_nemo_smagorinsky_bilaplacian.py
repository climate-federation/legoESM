"""Direct tiny-grid controls for NEMO mode-32 momentum diffusion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_vertex_mask,
    nemo_smagorinsky_bilaplacian_coefficients_cgrid,
    nemo_smagorinsky_bilaplacian_tendency_e3_cgrid,
)


def _case():
    ny, nx, nz = 5, 7, 2
    grid = create_beta_plane_cgrid_geometry(
        ny, nx, dx_m=25_000.0, dy_m=25_000.0, f0=1.0e-4, beta=2.0e-11
    )
    mask = jnp.ones((ny, nx), dtype=jnp.float64)
    mask = mask.at[[0, -1], :].set(0.0).at[:, [0, -1]].set(0.0)
    u_mask = jnp.concatenate(
        [jnp.zeros((ny, 1)), mask[:, :-1] * mask[:, 1:], jnp.zeros((ny, 1))], axis=1
    )
    v_mask = jnp.concatenate([jnp.zeros((1, nx)), mask[:-1] * mask[1:], jnp.zeros((1, nx))], axis=0)
    cell3 = jnp.broadcast_to(mask[..., None], (ny, nx, nz))
    u3 = jnp.broadcast_to(u_mask[..., None], (ny, nx + 1, nz))
    v3 = jnp.broadcast_to(v_mask[..., None], (ny + 1, nx, nz))
    vertex3 = jnp.broadcast_to(
        compute_vertex_mask(mask, grid=grid)[..., None], (ny + 1, nx + 1, nz)
    )
    return grid, mask, u_mask[..., None], v_mask[..., None], cell3, vertex3, u3, v3


def _numpy_mode32_isotropic(u, v, cell_mask, vertex_mask, c_smag, dt_s, minfac, maxfac):
    """Independent source-literal mode-32 reference for the 25 km test grid."""
    length = np.float64(25_000.0)
    reciprocal = np.float64(1.0) / length
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    tmask = np.asarray(cell_mask, dtype=np.float64)
    fmask = np.asarray(vertex_mask, dtype=np.float64)

    tension = (u[:, 1:] * reciprocal - u[:, :-1] * reciprocal) * reciprocal * length - (
        v[1:] * reciprocal - v[:-1] * reciprocal
    ) * reciprocal * length
    dtensq = tension * tension * tmask

    shear = np.zeros(fmask.shape, dtype=np.float64)
    du = (u[1:] * reciprocal - u[:-1] * reciprocal) * reciprocal * length
    dv = (v * reciprocal - np.roll(v, 1, axis=1) * reciprocal) * reciprocal * length
    shear[1:-1, :-1] = du[:, :-1] + dv[1:-1]
    shear[:, -1] = shear[:, 0]
    dshesq = shear * shear * fmask

    zcmsmag = (np.float64(c_smag) / np.pi) ** 2
    esq = length * length
    zdelta = zcmsmag * esq
    zstab_lo = (
        (np.float64(16.0) / 9.0) * minfac * minfac / (np.float64(2.0) * 12.0 * 12.0 * zcmsmag)
    )
    zstab_up = maxfac / (np.float64(4.0) * zcmsmag * 2.0 * dt_s)

    dsh_t = 0.25 * ((dshesq[:-1, :-1] + dshesq[1:, :-1]) + (dshesq[:-1, 1:] + dshesq[1:, 1:]))
    deformation_t = np.sqrt(dtensq + dsh_t)
    speed_t = (u[:, 1:] * u[:, 1:] + v[1:] * v[1:]) + (u[:, :-1] * u[:, :-1] + v[:-1] * v[:-1])
    lap_t = np.minimum(
        np.maximum(zdelta * deformation_t, np.sqrt(speed_t * zdelta * zstab_lo)),
        zdelta * zstab_up,
    )

    dt_w = np.roll(dtensq, 1, axis=1)
    dt_pad = np.pad(dtensq, ((1, 1), (0, 0), (0, 0)))
    dt_w_pad = np.pad(dt_w, ((1, 1), (0, 0), (0, 0)))
    dt_f_core = 0.25 * ((dt_pad[:-1] + dt_pad[1:]) + (dt_w_pad[:-1] + dt_w_pad[1:]))
    dt_f = np.concatenate([dt_f_core, dt_f_core[:, :1]], axis=1)
    deformation_f = np.sqrt(dshesq + dt_f)
    u_pad = np.pad(u, ((1, 1), (0, 0), (0, 0)))
    v_west = np.concatenate([np.roll(v, 1, axis=1), np.roll(v, 1, axis=1)[:, :1]], axis=1)
    v_east = np.concatenate([v, v[:, :1]], axis=1)
    speed_f = (u_pad[:-1] * u_pad[:-1] + v_west * v_west) + (
        u_pad[1:] * u_pad[1:] + v_east * v_east
    )
    lap_f = np.minimum(
        np.maximum(zdelta * deformation_f, np.sqrt(speed_f * zdelta * zstab_lo)),
        zdelta * zstab_up,
    )
    return (
        # ldfdyn.f90:521-534 applies neither completed-point mask in dynamic
        # mode 32; the deformation inputs above retain their source masks.
        np.sqrt(0.125 * esq * lap_t),
        np.sqrt(0.125 * esq * lap_f),
    )


def _numpy_nemo_bilaplacian_uniform(
    u, v, ahmt, ahmf, cell_mask, u_mask, v_mask, *, length=25_000.0, thickness=100.0,
):
    """Independent NumPy transcription of dynldf_lev_blp's two source passes."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ahmt = np.asarray(ahmt, dtype=np.float64)
    ahmf = np.asarray(ahmf, dtype=np.float64)
    tmask = np.asarray(cell_mask, dtype=np.float64)
    umask = np.asarray(u_mask, dtype=np.float64)
    vmask = np.asarray(v_mask, dtype=np.float64)
    length = np.float64(length)
    thickness = np.float64(thickness)

    def pass_once(uu, vv):
        # dynldf_lev.f90:271-291 / :311-331, specialized only after retaining
        # the source's div/curl staggering.  Periodic halo values are harmless
        # because the synthetic basin's outer T ring is dry.
        uu = uu * umask
        vv = vv * vmask
        flux_u = length * thickness * uu
        flux_v = length * thickness * vv
        flux_sum = (flux_u[:, 1:] - flux_u[:, :-1]) + (
            flux_v[1:] - flux_v[:-1]
        )
        zdiv = ahmt * tmask * flux_sum / (length * length * thickness)

        v_metric = length * vv
        v_pad = np.concatenate([v_metric[:, -1:], v_metric, v_metric[:, :1]], axis=1)
        dv = v_pad[:, 1:] - v_pad[:, :-1]
        u_metric = length * uu
        u_pad = np.pad(u_metric, ((1, 1), (0, 0), (0, 0)))
        du = u_pad[:-1] - u_pad[1:]
        zcur = ahmf * thickness * (dv + du) / (length * length)

        zdiv_x = np.concatenate([zdiv[:, -1:], zdiv, zdiv[:, :1]], axis=1)
        grad_div_u = (zdiv_x[:, 1:] - zdiv_x[:, :-1]) / length
        zdiv_y = np.pad(zdiv, ((1, 1), (0, 0), (0, 0)))
        grad_div_v = (zdiv_y[1:] - zdiv_y[:-1]) / length
        curl_u = -(zcur[1:] - zcur[:-1]) / (length * thickness)
        curl_v = (zcur[:, 1:] - zcur[:, :-1]) / (length * thickness)
        return (
            (curl_u + grad_div_u) * umask,
            (curl_v + grad_div_v) * vmask,
        )

    lap_u, lap_v = pass_once(u, v)
    second_u, second_v = pass_once(lap_u, lap_v)
    return -second_u, -second_v


def test_mode32_zero_flow_is_exact_zero_with_finite_gradient():
    grid, mask, um, vm, cell3, fm, u, v = _case()
    u = jnp.zeros_like(u)
    v = jnp.zeros_like(v)

    def total(uu, vv):
        ahmt, ahmf = nemo_smagorinsky_bilaplacian_coefficients_cgrid(
            uu,
            vv,
            grid,
            3.5,
            3600.0,
            mask=mask,
            cell_mask=cell3,
            u_mask=um,
            v_mask=vm,
            vertex_mask=fm,
        )
        return jnp.sum(ahmt) + jnp.sum(ahmf)

    assert float(total(u, v)) == 0.0
    gu, gv = jax.grad(total, argnums=(0, 1))(u, v)
    assert np.all(np.isfinite(np.asarray(gu)))
    assert np.all(np.isfinite(np.asarray(gv)))

    @jax.jit
    def traced_dynamic_dt(dt_s):
        return nemo_smagorinsky_bilaplacian_coefficients_cgrid(
            u,
            v,
            grid,
            3.5,
            dt_s,
            mask=mask,
            cell_mask=cell3,
            u_mask=um,
            v_mask=vm,
            vertex_mask=fm,
        )

    traced_t, traced_f = traced_dynamic_dt(jnp.asarray(3600.0))
    assert float(jnp.sum(traced_t) + jnp.sum(traced_f)) == 0.0


@pytest.mark.parametrize(
    ("minfac", "maxfac"),
    ((0.0, 1.0e12), (1000.0, 1.0e12), (0.0, 1.0e-9)),
    ids=("deformation", "lower_limit", "upper_limit"),
)
def test_mode32_nonzero_coefficients_match_source_literal_reference(minfac, maxfac):
    grid, mask, um, vm, cell3, fm, u_masked, v_masked = _case()
    jj_u, ii_u, kk_u = jnp.indices(u_masked.shape)
    jj_v, ii_v, kk_v = jnp.indices(v_masked.shape)
    u = (0.04 * jnp.sin(0.31 * ii_u + 0.17 * jj_u + kk_u)) * u_masked
    v = (0.03 * jnp.cos(0.27 * ii_v - 0.13 * jj_v + kk_v)) * v_masked
    got_t, got_f = nemo_smagorinsky_bilaplacian_coefficients_cgrid(
        u,
        v,
        grid,
        3.5,
        3600.0,
        minfac=minfac,
        maxfac=maxfac,
        mask=mask,
        cell_mask=cell3,
        u_mask=um,
        v_mask=vm,
        vertex_mask=fm,
    )
    ref_t, ref_f = _numpy_mode32_isotropic(u, v, cell3, fm, 3.5, 3600.0, minfac, maxfac)
    np.testing.assert_allclose(np.asarray(got_t), ref_t, rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(np.asarray(got_f), ref_f, rtol=2.0e-15, atol=0.0)

    # The old defect multiplied the completed F coefficient by the four-cell
    # mask.  Pin NEMO's live free-slip boundary ring independently so reverting
    # the production fix cannot be hidden by the source-literal reference.
    perimeter = (np.asarray(fm) == 0.0) & (ref_f > 0.0)
    assert np.count_nonzero(perimeter) > 0
    assert np.all(np.asarray(got_f)[perimeter] > 0.0)

    if minfac == 1000.0:
        # Planted source violation: omitting the bilaplacian 16/9 lower-limit
        # factor must move a nonzero coefficient.
        wrong_t, wrong_f = _numpy_mode32_isotropic(u, v, cell3, fm, 3.5, 3600.0, 750.0, maxfac)
        assert np.count_nonzero(ref_t != wrong_t) + np.count_nonzero(ref_f != wrong_f) > 0


def test_mode32_rejects_anisotropic_regular_grid():
    grid, mask, um, vm, cell3, fm, u, v = _case()
    anisotropic = grid._replace(dy_T=grid.dy_T * 1.01)
    with pytest.raises(ValueError, match="exact isotropic"):
        nemo_smagorinsky_bilaplacian_coefficients_cgrid(
            u,
            v,
            anisotropic,
            3.5,
            3600.0,
            mask=mask,
            cell_mask=cell3,
            u_mask=um,
            v_mask=vm,
            vertex_mask=fm,
        )


def test_mode32_bilaplacian_matches_independent_source_literal_two_pass():
    grid, mask, um, vm, cell3, fm, u_masked, v_masked = _case()
    jj_u, ii_u, kk_u = jnp.indices(u_masked.shape)
    jj_v, ii_v, kk_v = jnp.indices(v_masked.shape)
    u = (0.04 * jnp.sin(0.31 * ii_u + 0.17 * jj_u + kk_u)) * u_masked
    v = (0.03 * jnp.cos(0.27 * ii_v - 0.13 * jj_v + kk_v)) * v_masked
    ahmt, ahmf = nemo_smagorinsky_bilaplacian_coefficients_cgrid(
        u, v, grid, 3.5, 3600.0, mask=mask, cell_mask=cell3, u_mask=um, v_mask=vm, vertex_mask=fm
    )
    # NEMO's reference e3 arrays remain geometrical on dry/free-slip points;
    # masks are separate multiplicands in dynldf_lev.  Supplying the six live
    # operands also exercises the exact source-order branch used by GYRE.
    h = jnp.full(cell3.shape, 100.0)
    e3u = jnp.full(u.shape, 100.0)
    e3v = jnp.full(v.shape, 100.0)
    e3f = jnp.full(fm.shape, 100.0)
    thickness_operands = (h, e3u, e3v, e3f, e3u, e3v)
    got_u, got_v = nemo_smagorinsky_bilaplacian_tendency_e3_cgrid(
        u, v, grid, ahmt, ahmf, h, mask=mask, u_mask=um, v_mask=vm,
        vertex_mask=fm, thickness_operands=thickness_operands,
    )
    ref_u, ref_v = _numpy_nemo_bilaplacian_uniform(
        u, v, ahmt, ahmf, cell3, um, vm,
    )
    np.testing.assert_allclose(np.asarray(got_u), ref_u, rtol=1.0e-13, atol=1.0e-30)
    np.testing.assert_allclose(np.asarray(got_v), ref_v, rtol=1.0e-13, atol=1.0e-30)
    assert np.max(np.abs(np.asarray(got_u))) > 0.0
    assert np.max(np.abs(np.asarray(got_v))) > 0.0

    # Planted old defect: erasing the live free-slip F ring must move the
    # independently transcribed NEMO result.
    wrong_u, wrong_v = _numpy_nemo_bilaplacian_uniform(
        u, v, ahmt, np.asarray(ahmf) * np.asarray(fm), cell3, um, vm,
    )
    assert np.count_nonzero(ref_u != wrong_u) + np.count_nonzero(ref_v != wrong_v) > 0


@pytest.mark.slow
def test_mode32_card_executes_one_complete_jitted_gyre_step():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_SMAG_NAMELIST,
        build_gyre_zco_card,
        gyre_surface_forcings,
    )

    old = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
        namelist = GYRE_R4_SMAG_NAMELIST._replace(
            nn_GYRE=1,
            rn_Dt_s=14_400.0,
            rn_Ld_m=100_000.0,
            nn_itend=1,
            nn_stock=1,
            nn_write=1,
        )
        card = build_gyre_zco_card(namelist)
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config
        )
        freshwater, surface = gyre_surface_forcings(card, card.recipe.initial_state, 1)
        state = model.step(
            card.recipe.initial_state, card.dt_s, freshwater=freshwater, surface_forcing=surface
        )
        jax.block_until_ready(state.eta.data)
        for name in ("T", "S", "u", "v", "eta"):
            assert np.all(np.isfinite(np.asarray(getattr(state, name).data)))
    finally:
        set_policy(old)
