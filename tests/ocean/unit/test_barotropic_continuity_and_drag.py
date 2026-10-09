"""Findings #6 (single-owner bottom drag) and #8 (continuity-consistent
barotropic transport weights) for the lat-lon C-grid explicit barotropic
substep solver ``barotropic_substeps_latlon_cgrid``.

#8 — the time-averaged transport ``Hu_avg`` returned by the solver must satisfy
the discrete continuity invariant the flux-form tracer step relies on,

    div(Hu_avg) == (eta_old - eta_avg) / dt,

for EVERY ``barotropic_time_filter`` (box / cosine / power_law).  This is what
preserves a uniform tracer.  The earlier flat ``1/n`` transport weight broke it
for box (~95% residual) and cosine (~99%); the SM2005 tail-sum weight fixes it.

#6 — the explicit barotropic substep must NOT re-apply ``implicit_bottom_drag_
factor`` on top of the depth-mean bottom drag already carried in ``F_slow``;
doing so doubled the effective barotropic-mode drag to ~2r/H.  The substep now
owns no drag, so a barotropic mode forced ONLY by ``F_slow_u = -r*U/H`` decays
at the single rate ``r/H``.

#1226 ``barotropic_face_depth="nemo_ssh_avg"`` — the NEMO ``dynspg_ts.F90``
``zhup2_e``/``zhvp2_e`` (flux depth, :568-592) and ``zsshu_a``/``hu_e`` (drag/
update depth, :658-666,771-778) face-depth rule: a fixed still-water reference
depth plus an e1e2-area-weighted 2-point average of the dynamic ssh, in place
of lego's default min-rule.  Gated by ``BarotropicConfig.barotropic_face_depth``
(default ``"min_rule"``, bit-identical legacy).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.vertical import create_ocean_z_star, compute_layer_thickness
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    barotropic_substeps_latlon_cgrid,
    _depth_average_to_faces,
    _nemo_ssh_avg_apply,
    _nemo_ssh_avg_prep,
    _nemo_literal_barotropic_pressure_gradient,
    _nemo_literal_seed_depth_mean,
    _nemo_literal_slow_depth_mean,
    nemo_literal_accumulate_transport,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    divergence_cgrid, gradient_y_cgrid, min_cell_to_uface, min_cell_to_vface,
)


def _flat_basin(n_lat=24, n_lon=48, H=4000.0, lat_cap_deg=80.0):
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z = create_ocean_z_star(n_levels=4, H_max=H, dz_surface=50.0, dz_deep=1500.0)
    latd = np.asarray(grid.lat2d) * 180.0 / np.pi
    H_bathy = jnp.full((grid.n_lat, grid.n_lon), H)
    land_mask = jnp.asarray(np.where(np.abs(latd) < lat_cap_deg, 1.0, 0.0))
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        H_max=H, H_bathy_override=H_bathy, land_mask_override=land_mask)
    return grid, z, state


def _cfg(**over):
    # #644 (config grouping) nested the barotropic + bottom-drag knobs into
    # BarotropicConfig / the bottom-drag sub-config. Route flat test kwargs to the
    # right leaf by NamedTuple field membership (robust to exact field placement).
    cfg = LatLonCGridOceanConfig(fix_eta_drift=False)
    baro = cfg.barotropic._replace(
        bebt=0.0, maxvel_barotropic=0.0, barotropic_diffusion_alpha=0.0,
        barotropic_local_subcycle_clamp=False,
        barotropic_solver="explicit_substep",
    )
    drag = cfg.bottom_drag._replace(bottom_drag_r=0.0)
    for k, v in over.items():
        if k in drag._fields:
            drag = drag._replace(**{k: v})
        elif k in baro._fields:
            baro = baro._replace(**{k: v})
        else:
            cfg = cfg._replace(**{k: v})
    return cfg._replace(barotropic=baro, bottom_drag=drag)


class TestNemoLiteralTransportAccumulation:
    """Round 59: raw za2*zhU*r1_e2u recurrence, one final division."""

    def test_source_association_and_cancelled_form_violation(self):
        """The control proves this test detects the old cancelled topology."""
        e2u = jnp.asarray([[1.0000000000000002, 1.7, 3.1]], dtype=jnp.float64)
        e1v = jnp.asarray(
            [[1.3, 2.9], [1.0000000000000004, 4.7]], dtype=jnp.float64)
        grid = SimpleNamespace(dy_u=e2u, dx_v=e1v)
        hu = jnp.asarray([[4000.125, 73.25, 8100.5]], dtype=jnp.float64)
        hv = jnp.asarray([[27.125, 9100.75], [5300.5, 61.25]], dtype=jnp.float64)
        u = jnp.asarray([[0.173, -0.219, 0.037]], dtype=jnp.float64)
        v = jnp.asarray([[0.117, -0.193], [0.071, 0.233]], dtype=jnp.float64)
        um = jnp.ones_like(u)
        vm = jnp.ones_like(v)
        raw = jnp.asarray(45.0, dtype=jnp.float64)
        got_u, got_v = nemo_literal_accumulate_transport(
            jnp.zeros_like(u), jnp.zeros_like(v), raw,
            hu, hv, u, v, um, vm, grid)
        expected_u = (raw * ((e2u * u) * hu)) * (1.0 / e2u)
        expected_v = (raw * ((e1v * v) * hv)) * (1.0 / e1v)
        np.testing.assert_array_equal(np.asarray(got_u), np.asarray(expected_u))
        np.testing.assert_array_equal(np.asarray(got_v), np.asarray(expected_v))

        # Planted old implementation: pre-normalised/cancelled H*U. It is
        # algebraically equal and therefore a credible regression, but must be
        # bit-distinct for at least one face on these operands.
        cancelled_u = raw * (hu * u)
        cancelled_v = raw * (hv * v)
        assert (not np.array_equal(np.asarray(got_u), np.asarray(cancelled_u))
                or not np.array_equal(np.asarray(got_v),
                                      np.asarray(cancelled_v))), (
            "planted cancelled-form violation did not fire")

        def objective(u_arg):
            out_u, out_v = nemo_literal_accumulate_transport(
                jnp.zeros_like(u_arg), jnp.zeros_like(v), raw,
                hu, hv, u_arg, v, um, vm, grid)
            return jnp.sum(out_u) + jnp.sum(out_v)

        jitted = jax.jit(objective)(u)
        grad = jax.grad(objective)(u)
        assert np.isfinite(float(jitted))
        assert np.all(np.isfinite(np.asarray(grad)))
        assert np.max(np.abs(np.asarray(grad))) > 0.0

    def test_literal_scan_and_fori_match(self):
        grid, z, state = _flat_basin(n_lat=8, n_lon=16)
        key_u, key_v = jax.random.split(jax.random.PRNGKey(59))
        state = state._replace(
            u=state.u.replace(data=(
                0.03 * jax.random.normal(key_u, state.u.data.shape)
                * state.u_mask.data[..., None])),
            v=state.v.replace(data=(
                0.03 * jax.random.normal(key_v, state.v.data.shape)
                * state.v_mask.data[..., None])))
        base = _cfg(
            barotropic_time_filter="nemo_boxcar_centred",
            barotropic_transport_accumulation_evaluation="nemo_literal")
        scan_cfg = base._replace(barotropic=base.barotropic._replace(
            differentiable_barotropic=True))
        fori_cfg = base._replace(barotropic=base.barotropic._replace(
            differentiable_barotropic=False))
        scan_state, scan_flux = barotropic_substeps_latlon_cgrid(
            state, 30.0, 6, grid, z, scan_cfg)
        fori_state, fori_flux = barotropic_substeps_latlon_cgrid(
            state, 30.0, 6, grid, z, fori_cfg)
        for a, b in ((scan_state.eta.data, fori_state.eta.data),
                     (scan_state.u.data, fori_state.u.data),
                     (scan_state.v.data, fori_state.v.data),
                     (scan_flux[0], fori_flux[0]),
                     (scan_flux[1], fori_flux[1])):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_generic_default_is_byte_pinned_and_bad_selectors_are_red(self):
        grid, z, state = _flat_basin(n_lat=8, n_lon=16)
        default = _cfg(barotropic_time_filter="box")
        explicit = _cfg(
            barotropic_time_filter="box",
            barotropic_transport_accumulation_evaluation="generic")
        a, flux_a = barotropic_substeps_latlon_cgrid(
            state, 30.0, 4, grid, z, default)
        b, flux_b = barotropic_substeps_latlon_cgrid(
            state, 30.0, 4, grid, z, explicit)
        for x, y in ((a.eta.data, b.eta.data), (a.u.data, b.u.data),
                     (a.v.data, b.v.data), (flux_a[0], flux_b[0]),
                     (flux_a[1], flux_b[1])):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

        bad = _cfg(barotropic_transport_accumulation_evaluation="bogus")
        with pytest.raises(ValueError, match="transport_accumulation"):
            barotropic_substeps_latlon_cgrid(state, 30.0, 4, grid, z, bad)
        wrong_filter = _cfg(
            barotropic_time_filter="box",
            barotropic_transport_accumulation_evaluation="nemo_literal")
        with pytest.raises(ValueError, match="requires barotropic_time_filter"):
            barotropic_substeps_latlon_cgrid(
                state, 30.0, 4, grid, z, wrong_filter)


class TestBarotropicContinuityInvariant:
    """Finding #8: div(Hu_avg) == (eta_old - eta_avg)/dt for every filter."""

    @pytest.mark.parametrize("time_filter", ["box", "cosine", "power_law"])
    def test_continuity_invariant(self, time_filter):
        grid, z, state = _flat_basin()
        key = jax.random.PRNGKey(1)
        u0 = 0.05 * jax.random.normal(key, state.u.data.shape) * state.u_mask.data[..., None]
        v0 = 0.05 * jax.random.normal(
            jax.random.split(key)[0], state.v.data.shape) * state.v_mask.data[..., None]
        state = state._replace(u=state.u.replace(data=u0), v=state.v.replace(data=v0))
        dt = 1800.0
        n = 30
        cfg = _cfg(barotropic_time_filter=time_filter)
        eta_old = state.eta.data
        sn, (Hu, Hv) = barotropic_substeps_latlon_cgrid(
            state, dt / n, n, grid, z, cfg, add_barotropic_coriolis=True)
        eta_avg = sn.eta.data
        div = divergence_cgrid(
            Hu, Hv, grid, u_mask=state.u_mask.data, v_mask=state.v_mask.data)
        resid = np.asarray(div - (eta_old - eta_avg) / dt)
        m = np.asarray(state.land_mask.data) > 0
        max_resid = float(np.max(np.where(m, np.abs(resid), 0.0)))
        signal = float(np.max(np.abs(np.asarray((eta_old - eta_avg) / dt))))
        rel = max_resid / max(signal, 1e-30)
        assert rel < 1e-5, (
            f"[{time_filter}] continuity invariant residual rel={rel:.3e} "
            f"(abs {max_resid:.3e}); transport weights inconsistent with eta_avg "
            "(finding #8)")

    def test_uniform_tracer_preserved_cosine(self):
        """The invariant's PURPOSE: with continuity-consistent transport, the
        column-integrated thickness change ``eta_avg - eta_old`` equals
        ``-dt*div(Hu_avg)``, so a uniform tracer advected by the resulting mass
        fluxes is preserved.  Pin the column-sum identity directly (cosine)."""
        grid, z, state = _flat_basin()
        key = jax.random.PRNGKey(5)
        u0 = 0.04 * jax.random.normal(key, state.u.data.shape) * state.u_mask.data[..., None]
        v0 = 0.04 * jax.random.normal(
            jax.random.split(key)[0], state.v.data.shape) * state.v_mask.data[..., None]
        state = state._replace(u=state.u.replace(data=u0), v=state.v.replace(data=v0))
        dt = 1800.0
        n = 24
        cfg = _cfg(barotropic_time_filter="cosine")
        eta_old = state.eta.data
        sn, (Hu, Hv) = barotropic_substeps_latlon_cgrid(
            state, dt / n, n, grid, z, cfg, add_barotropic_coriolis=True)
        eta_avg = sn.eta.data
        div = divergence_cgrid(
            Hu, Hv, grid, u_mask=state.u_mask.data, v_mask=state.v_mask.data)
        # h_new - h_old (column sum) == eta_avg - eta_old; tracer continuity needs
        # (eta_avg - eta_old) + dt*div(Hu_avg) == 0.
        m = np.asarray(state.land_mask.data) > 0
        lhs = np.asarray((eta_avg - eta_old) + dt * div)
        err = float(np.max(np.where(m, np.abs(lhs), 0.0)))
        amp = float(np.max(np.where(m, np.abs(np.asarray(eta_avg - eta_old)), 0.0)))
        assert err < 1e-6 * max(amp, 1.0), (
            f"uniform-tracer continuity violated by {err:.3e} (amp {amp:.3e})")


class TestBottomDragSingleOwner:
    """Finding #6: effective barotropic-mode drag is r/H (not 2r/H).

    Force the barotropic mode with ONLY the depth-mean bottom drag
    ``F_slow_u = -r*U/H`` (no PGF, no Coriolis, flat resting eta) and check that
    ``U_bar`` decays per outer step by the SINGLE implicit factor
    ``1/(1 + r*dt/H)`` — i.e. the substep does NOT additionally damp it.
    """

    def test_effective_barotropic_drag_is_r_over_H(self):
        grid, z, state = _flat_basin()
        H = 4000.0
        r = 1.0e-3
        # Uniform barotropic zonal flow; zero eta.
        U0 = 0.2
        u0 = jnp.full(state.u.data.shape, U0) * state.u_mask.data[..., None]
        state = state._replace(u=state.u.replace(data=u0))
        dt = 1800.0
        n = 36
        cfg = _cfg(barotropic_time_filter="box", bottom_drag_r=r)
        # F_slow_u = depth-mean bottom drag = -r*U/H (the ONLY forcing).  No
        # Coriolis (add_barotropic_coriolis=False) and zero PGF (flat eta).
        u_mask = state.u_mask.data
        F_slow_u = (-r * U0 / H) * u_mask
        F_slow_v = jnp.zeros((grid.n_lat + 1, grid.n_lon))
        sn, _ = barotropic_substeps_latlon_cgrid(
            state, dt / n, n, grid, z, cfg,
            F_slow_u=F_slow_u, F_slow_v=F_slow_v,
            add_barotropic_coriolis=False)
        # The time-averaged U_bar_avg is returned through the 3D velocity; read
        # the mean ocean u.  With single-owner drag the barotropic forcing is the
        # explicit -r*U/H applied each substep -> U decays by ~ (1 - r*dt/H).
        m = np.asarray(u_mask) > 0
        u_new = np.asarray(sn.u.data)[..., 0]
        u_mean = float(np.mean(np.where(m, u_new, np.nan)[~np.isnan(np.where(m, u_new, np.nan))]))
        # Expected single-rate decay of the averaged state.  Frozen F_slow
        # gives U(t) = U0 - (r U0/H) t, which is LINEAR in t, and the weighted
        # mean of a linear function is exactly its value at the window's
        # CENTROID.  Since 2026-08-12 the box/cosine averaging window is
        # centred on t+dt (it used to be centred on t+dt/2, which propagated
        # gravity waves at half speed -- see
        # tests/ocean/unit/test_barotropic_accuracy.py), so the centroid value
        # is U0*(1 - r*dt/H), not the old U0*(1 - r*dt/(2H)).
        #
        # EXPECTATION UPDATED, NOT THE CODE: the old numbers here encoded the
        # mis-centred window. This case is in fact a second, independent
        # confirmation that the window is centred correctly -- U is linear, so
        # the measured mean must land on the analytic value at t+dt to
        # roundoff, and it does (0.199910).
        expected_mean = U0 * (1.0 - r * dt / H)
        # Double-counting the drag would decay at 2r/H over the same window.
        single = abs(u_mean - expected_mean) / expected_mean
        double_mean = U0 * (1.0 - 2.0 * r * dt / H)
        # NON-VACUITY (codex 2026-08-12): a 2% bound is looser than the whole
        # effect -- undamped U0 = 0.2 sits within 2% of 0.199910 and is also
        # closer to the single proxy than the double one, so the original
        # bounds passed even with NO drag at all. Bound the error against the
        # single/double SEPARATION (9e-5) instead, and assert the flow
        # actually decelerated.
        assert u_mean < U0 - 0.25 * (U0 - expected_mean), (
            f"mean u={u_mean:.6f} is essentially undamped (U0={U0}); the "
            f"drag never acted, so the comparison below proves nothing")
        assert abs(u_mean - expected_mean) < 0.25 * abs(expected_mean
                                                        - double_mean), (
            f"mean u={u_mean:.6f} is not tight to the single-owner value "
            f"{expected_mean:.6f} relative to the {abs(expected_mean - double_mean):.2e} "
            f"single/double separation")
        assert single < 0.02, (
            f"barotropic drag mean u={u_mean:.6f} vs single-owner "
            f"{expected_mean:.6f} (rel {single:.3e}); drag may be double-counted")
        assert abs(u_mean - expected_mean) < abs(u_mean - double_mean), (
            "barotropic drag closer to the DOUBLE-counted rate than the "
            "single-owner rate (finding #6 regression)")

    def test_private_drag_rate_override_reaches_the_substep(self):
        """Round-15 fidelity hook substitutes rates, never configuration."""
        grid, z, state = _flat_basin(n_lat=8, n_lon=16)
        u0 = jnp.full(state.u.data.shape, 0.2) \
            * state.u_mask.data[..., None]
        state = state._replace(u=state.u.replace(data=u0))
        cfg = _cfg(
            barotropic_time_filter="box", barotropic_drag_substep=True)
        cfg = cfg._replace(constants=cfg.constants._replace(g=0.0))
        zero = (jnp.zeros_like(state.u_mask.data),
                jnp.zeros_like(state.v_mask.data))
        active = (jnp.full_like(state.u_mask.data, 1.0e-3),
                  jnp.full_like(state.v_mask.data, 1.0e-3))
        no_drag, _ = barotropic_substeps_latlon_cgrid(
            state, 30.0, 2, grid, z, cfg, add_barotropic_coriolis=False,
            _nemo_drag_rate_test_override=zero)
        with_drag, _ = barotropic_substeps_latlon_cgrid(
            state, 30.0, 2, grid, z, cfg, add_barotropic_coriolis=False,
            _nemo_drag_rate_test_override=active)
        wet = np.asarray(state.u_mask.data) > 0.0
        assert np.max(np.abs(
            np.asarray(with_drag.u.data)[..., 0][wet]
            - np.asarray(no_drag.u.data)[..., 0][wet])) > 0.0

    def test_private_drag_rate_override_refuses_bad_shapes(self):
        grid, z, state = _flat_basin(n_lat=8, n_lon=16)
        cfg = _cfg(
            barotropic_time_filter="box", barotropic_drag_substep=True)
        bad = (jnp.zeros((1, 1)), jnp.zeros_like(state.v_mask.data))
        with pytest.raises(ValueError, match="shape mismatch"):
            barotropic_substeps_latlon_cgrid(
                state, 30.0, 2, grid, z, cfg,
                _nemo_drag_rate_test_override=bad)


class TestBarotropicFaceDepthNemoSshAvg:
    """#1226 ``barotropic_face_depth="nemo_ssh_avg"`` (dynspg_ts.F90:568-592
    flux depth; :658-666,771-778 drag/update depth)."""

    def test_unknown_face_depth_raises(self):
        grid, z, state = _flat_basin()
        cfg = _cfg(barotropic_face_depth="bogus_scheme")
        with pytest.raises(ValueError, match="barotropic_face_depth"):
            barotropic_substeps_latlon_cgrid(
                state, 60.0, 4, grid, z, cfg, add_barotropic_coriolis=False)

    def test_association_selector_holds_face_depth_and_drag_fixed(self):
        """The climate A axis changes arithmetic, not face-depth physics.

        With g=0 and Coriolis off, the velocity update contains only explicit
        bottom drag. Identical U/V outputs therefore prove both arms used the
        same carry-level H_u/H_v drag denominators. Identical returned Hu/Hv
        prove the same flux-depth operands; bit-distinct eta proves the
        registered generic versus NEMO-literal divergence association fired.
        """
        grid, z, state = _flat_basin(n_lat=8, n_lon=16)
        key_u, key_v = jax.random.split(jax.random.PRNGKey(1226))
        u0 = (0.017 + 0.013 * jax.random.normal(key_u, state.u.data.shape)) \
            * state.u_mask.data[..., None]
        v0 = (-0.019 + 0.011 * jax.random.normal(key_v, state.v.data.shape)) \
            * state.v_mask.data[..., None]
        jj = jnp.arange(grid.n_lat, dtype=jnp.float64)[:, None]
        ii = jnp.arange(grid.n_lon, dtype=jnp.float64)[None, :]
        eta0 = (0.031 + 0.00017 * ii + 0.00023 * jj) * state.land_mask.data
        state = state._replace(
            u=state.u.replace(data=u0), v=state.v.replace(data=v0),
            eta=state.eta.replace(data=eta0))
        base = _cfg(
            barotropic_time_filter="box", barotropic_face_depth="nemo_ssh_avg",
            bottom_drag_r=1.7e-3)
        base = base._replace(constants=base.constants._replace(g=0.0))
        generic = base._replace(barotropic=base.barotropic._replace(
            barotropic_continuity_evaluation="generic"))
        literal = base._replace(barotropic=base.barotropic._replace(
            barotropic_continuity_evaluation="nemo_literal"))
        sn_g, (hu_g, hv_g) = barotropic_substeps_latlon_cgrid(
            state, 117.391304, 1, grid, z, generic,
            add_barotropic_coriolis=False)
        sn_l, (hu_l, hv_l) = barotropic_substeps_latlon_cgrid(
            state, 117.391304, 1, grid, z, literal,
            add_barotropic_coriolis=False)
        np.testing.assert_array_equal(np.asarray(hu_g), np.asarray(hu_l))
        np.testing.assert_array_equal(np.asarray(hv_g), np.asarray(hv_l))
        np.testing.assert_array_equal(np.asarray(sn_g.u.data),
                                      np.asarray(sn_l.u.data))
        np.testing.assert_array_equal(np.asarray(sn_g.v.data),
                                      np.asarray(sn_l.v.data))
        assert not np.array_equal(np.asarray(sn_g.eta.data),
                                  np.asarray(sn_l.eta.data)), (
            "generic and nemo_literal produced bit-identical eta; the "
            "association selector did not exercise distinct arithmetic")

    def test_default_min_rule_byte_identical_to_pre_change(self):
        """Default is "min_rule" — must reproduce the pre-#1226-field
        min-rule face depth exactly (the field is purely additive)."""
        grid, z, state = _flat_basin()
        key = jax.random.PRNGKey(3)
        u0 = 0.05 * jax.random.normal(key, state.u.data.shape) * state.u_mask.data[..., None]
        v0 = 0.05 * jax.random.normal(
            jax.random.split(key)[0], state.v.data.shape) * state.v_mask.data[..., None]
        state = state._replace(u=state.u.replace(data=u0), v=state.v.replace(data=v0))
        cfg_default = _cfg(barotropic_time_filter="cosine")
        cfg_explicit = _cfg(barotropic_time_filter="cosine",
                             barotropic_face_depth="min_rule")
        assert cfg_default.barotropic.barotropic_face_depth == "min_rule"
        sn_a, (Hu_a, Hv_a) = barotropic_substeps_latlon_cgrid(
            state, 60.0, 30, grid, z, cfg_default, add_barotropic_coriolis=True)
        sn_b, (Hu_b, Hv_b) = barotropic_substeps_latlon_cgrid(
            state, 60.0, 30, grid, z, cfg_explicit, add_barotropic_coriolis=True)
        np.testing.assert_array_equal(np.asarray(sn_a.eta.data), np.asarray(sn_b.eta.data))
        np.testing.assert_array_equal(np.asarray(sn_a.u.data), np.asarray(sn_b.u.data))
        np.testing.assert_array_equal(np.asarray(sn_a.v.data), np.asarray(sn_b.v.data))
        np.testing.assert_array_equal(np.asarray(Hu_a), np.asarray(Hu_b))
        np.testing.assert_array_equal(np.asarray(Hv_a), np.asarray(Hv_b))

    def test_nemo_ssh_avg_matches_analytic_uface_formula(self):
        """Flat bottom + a longitude ssh ramp: the u-face flux depth returned
        after ONE box-filtered substep must equal NEMO's own formula
        ``hu_0 + 0.5 * r1_e1e2u * (area[W]*eta[W] + area[E]*eta[E])``
        (dynspg_ts.F90:583-586), evaluated from the SAME grid metrics the
        solver uses (``area`` = the exact spherical T-cell area,
        ``dx_u*dy_u`` = the u-point's own metric area — these two area
        conventions differ by O(dlat^2), so the expected value is derived
        from the actual metrics, not assumed to be a bit-exact plain mean)."""
        grid, z, state = _flat_basin(n_lat=8, n_lon=16, H=1000.0, lat_cap_deg=90.0)
        i = np.arange(grid.n_lon)
        eta_ramp = 0.01 + 0.001 * i
        eta0 = jnp.asarray(np.broadcast_to(eta_ramp, (grid.n_lat, grid.n_lon)).copy())
        state = state._replace(eta=state.eta.replace(data=eta0))
        u1 = jnp.ones_like(state.u.data) * state.u_mask.data[..., None]
        state = state._replace(u=state.u.replace(data=u1))

        cfg = _cfg(barotropic_time_filter="box",
                   barotropic_face_depth="nemo_ssh_avg",
                   barotropic_continuity_evaluation="nemo_literal")
        sn, (Hu, Hv) = barotropic_substeps_latlon_cgrid(
            state, 1.0, 1, grid, z, cfg, add_barotropic_coriolis=False)
        Hu = np.asarray(Hu)

        geom = ensure_geometry(grid)
        area = np.asarray(grid.area)
        r1_e1e2u = 1.0 / np.asarray(geom.dx_u * geom.dy_u)
        area_w = np.roll(area, 1, axis=1)
        eta_w = np.roll(eta0, 1, axis=1)
        ssh_avg_u = 0.5 * r1_e1e2u[:, :-1] * (area_w * eta_w + area * eta0)
        ssh_avg_u = np.concatenate([ssh_avg_u, ssh_avg_u[:, 0:1]], axis=1)
        expected = 1000.0 + ssh_avg_u
        np.testing.assert_allclose(Hu, expected, rtol=0, atol=1e-9)

    def test_nemo_ssh_avg_equals_min_rule_at_rest(self):
        """At eta=0 (rest state) the ssh-average term is identically zero, so
        BOTH face-depth rules must reduce to the SAME still-water reference
        depth (the min-rule of H_bathy alone) — including at a bathymetric
        step, where they are provably equal (the coordinator's more general
        "min_rule <= nemo_ssh_avg away from rest" claim is NOT provable in
        general from the formula alone, since a nonzero, non-uniform eta can
        push the average either above or below the min-rule value depending
        on which side is shallower and which side's eta is larger — so this
        test asserts the one direction the formula DOES guarantee: exact
        equality at rest, not an inequality away from it)."""
        grid, z, state = _flat_basin(n_lat=8, n_lon=16, H=1000.0, lat_cap_deg=90.0)
        H_bathy = np.full((grid.n_lat, grid.n_lon), 1000.0)
        H_bathy[:, 8] = 200.0  # a shallow step at column 8
        state = state._replace(H_bathy=state.H_bathy.replace(data=jnp.asarray(H_bathy)))
        u1 = jnp.ones_like(state.u.data) * state.u_mask.data[..., None]
        state = state._replace(u=state.u.replace(data=u1))

        results = {}
        for mode in ("min_rule", "nemo_ssh_avg"):
            cfg = _cfg(barotropic_time_filter="box", barotropic_face_depth=mode)
            sn, (Hu, Hv) = barotropic_substeps_latlon_cgrid(
                state, 1.0, 1, grid, z, cfg, add_barotropic_coriolis=False)
            results[mode] = np.asarray(Hu)
        np.testing.assert_array_equal(results["min_rule"], results["nemo_ssh_avg"])
        # Sanity: the step is actually visible in the returned face depth.
        assert results["min_rule"][0, 8] == pytest.approx(200.0)


class TestBarotropicSeedFaceDepth:
    """#1226 round 2 item 1: the barotropic substep loop's ENTRY seed
    (``U_bar``/``V_bar``, lego's re-derived stand-in for NEMO's persistent
    ``un_e``/``vn_e = puu_b/pvv_b(Kbb)``) can be re-weighted by the SAME NEMO
    ssh-average face-depth rule as ``barotropic_face_depth`` — matching how
    ``puu_b``/``pvv_b`` are themselves finalized at the end of every prior
    step (``dynspg_ts.F90:963-966,978-979``, the non-RK3/nn_bt_flt=2 branch).

    Gated by ``BarotropicConfig.barotropic_seed_face_depth`` (default
    ``"min_rule"``, bit-identical legacy) — distinct from
    ``barotropic_face_depth``, which governs the IN-SUBSTEP flux/drag face
    thickness once the loop is already running (see the state.py field
    docstrings for the full NEMO citation).
    """

    def test_nemo_entry_inverse_has_finite_closed_meridional_faces(self):
        """DINO developed ssh must not turn the two storage faces into NaN."""
        grid, _, state = _flat_basin(n_lat=8, n_lon=16, H=1000.0)
        geom = ensure_geometry(grid)
        dtype = state.eta.data.dtype
        prep = _nemo_ssh_avg_prep(
            state.H_bathy.data, state.land_mask.data, grid, dtype, None)
        eta = (jnp.asarray(0.125, dtype=dtype)
               * state.land_mask.data)

        def entry_inverse(eta_arg):
            return _nemo_ssh_avg_apply(
                eta_arg, state.u_mask.data, state.v_mask.data, grid,
                geom.area, prep, return_entry_inverse=True)[2:]

        r1_u, r1_v = entry_inverse(eta)
        r1_u_jit, r1_v_jit = jax.jit(entry_inverse)(eta)
        for value in (r1_u, r1_v, r1_u_jit, r1_v_jit):
            assert np.isfinite(np.asarray(value)).all()
        np.testing.assert_array_equal(np.asarray(r1_v)[[0, -1]], 0.0)
        np.testing.assert_array_equal(np.asarray(r1_v_jit)[[0, -1]], 0.0)

    def test_unknown_seed_face_depth_raises(self):
        with pytest.raises(ValueError, match="barotropic_seed_face_depth"):
            _depth_average_to_faces(
                jnp.zeros((4, 5, 2)), jnp.zeros((5, 4, 2)), jnp.ones((4, 4, 2)),
                jnp.asarray(1.0), jnp.ones((4, 4)), jnp.ones((4, 5)),
                jnp.ones((5, 4)), seed_face_depth="bogus_scheme")

    def test_unknown_seed_face_depth_raises_via_substep_entry(self):
        """Same guard, exercised through the real substep-loop entry point
        (not just the leaf helper) — the config value the user actually
        sets."""
        grid, z, state = _flat_basin()
        cfg = _cfg(barotropic_seed_face_depth="bogus_scheme")
        with pytest.raises(ValueError, match="barotropic_seed_face_depth"):
            barotropic_substeps_latlon_cgrid(
                state, 60.0, 4, grid, z, cfg, add_barotropic_coriolis=False)

    def test_unknown_seed_evaluation_raises_at_leaf_and_substep(self):
        args = (
            jnp.zeros((4, 5, 2)), jnp.zeros((5, 4, 2)),
            jnp.ones((4, 4, 2)), jnp.asarray(1.0), jnp.ones((4, 4)),
            jnp.ones((4, 5)), jnp.ones((5, 4)),
        )
        with pytest.raises(ValueError, match="barotropic_seed_evaluation"):
            _depth_average_to_faces(*args, seed_evaluation="bogus_scheme")

        grid, z, state = _flat_basin()
        cfg = _cfg(barotropic_seed_evaluation="bogus_scheme")
        with pytest.raises(ValueError, match="barotropic_seed_evaluation"):
            barotropic_substeps_latlon_cgrid(
                state, 60.0, 4, grid, z, cfg, add_barotropic_coriolis=False)

    def test_nemo_literal_requires_nemo_ssh_avg_face_depth(self):
        with pytest.raises(ValueError, match="requires.*nemo_ssh_avg"):
            _depth_average_to_faces(
                jnp.zeros((4, 5, 2)), jnp.zeros((5, 4, 2)),
                jnp.ones((4, 4, 2)), jnp.asarray(1.0), jnp.ones((4, 4)),
                jnp.ones((4, 5)), jnp.ones((5, 4)),
                seed_face_depth="min_rule", seed_evaluation="nemo_literal")

    def test_nemo_literal_source_order_jit_grad_and_planted_reversal(self):
        """Pin ``istate.F90:149-155`` association independently of geometry.

        The cancellation-heavy terms make source order observable.  Reversing
        the level recurrence is the planted red control: it must not reproduce
        the registered surface-to-bottom result.
        """
        field_np = np.asarray([[[1.0e16, 1.0, -1.0e16, 1.0]]], dtype=np.float64)
        h_np = np.ones_like(field_np)
        mask_np = np.ones((1, 1), dtype=np.float64)
        r1_np = np.asarray([[0.25]], dtype=np.float64)

        expected = np.zeros((1, 1), dtype=np.float64)
        for jk in range(field_np.shape[-1]):
            expected = expected + h_np[..., jk] * field_np[..., jk]
        expected = (expected * r1_np) * mask_np

        field = jnp.asarray(field_np)
        h_face = jnp.asarray(h_np)
        face_mask = jnp.asarray(mask_np)
        r1_live = jnp.asarray(r1_np)
        got = _nemo_literal_seed_depth_mean(field, h_face, face_mask, r1_live)
        got_jit = jax.jit(_nemo_literal_seed_depth_mean)(
            field, h_face, face_mask, r1_live)
        np.testing.assert_array_equal(np.asarray(got), expected)
        np.testing.assert_array_equal(np.asarray(got_jit), expected)

        reversed_acc = np.zeros((1, 1), dtype=np.float64)
        for jk in reversed(range(field_np.shape[-1])):
            reversed_acc = reversed_acc + h_np[..., jk] * field_np[..., jk]
        reversed_value = (reversed_acc * r1_np) * mask_np
        assert not np.array_equal(reversed_value, expected), (
            "planted reversed vertical recurrence did not separate from the "
            "NEMO source order")

        grad = jax.grad(lambda f: jnp.sum(
            _nemo_literal_seed_depth_mean(f, h_face, face_mask, r1_live)))(field)
        assert np.all(np.isfinite(np.asarray(grad)))
        assert np.any(np.asarray(grad) != 0.0)

    def test_nemo_slow_depth_mean_materializes_compiled_products(self):
        """The fused reduction is a firing control for ``stp2d`` association."""

        field_np = np.asarray(
            [[[-0.0006998246757532552, 50.213546544575564]]],
            dtype=np.float64,
        )
        h_np = np.asarray(
            [[[752.9006358246162, 0.7354579872041004]]],
            dtype=np.float64,
        )
        field, h_face = jnp.asarray(field_np), jnp.asarray(h_np)
        level_mask = jnp.ones_like(field)
        face = jnp.ones((1, 1), dtype=jnp.float64)

        expected = np.zeros((1, 1), dtype=np.float64)
        for jk in range(field_np.shape[-1]):
            expected = expected + (
                h_np[..., jk] * field_np[..., jk])
        expected = expected * np.asarray(face) * np.asarray(face)

        got = jax.jit(_nemo_literal_slow_depth_mean)(
            field, h_face, level_mask, face, face)
        fused = jax.jit(lambda f, h: jnp.sum(f * h, axis=-1))(
            field, h_face)
        np.testing.assert_array_equal(np.asarray(got), expected)
        assert not np.array_equal(np.asarray(fused), expected), (
            "fused reduction plant did not separate from source association")

        grad = jax.grad(lambda f: jnp.sum(_nemo_literal_slow_depth_mean(
            f, h_face, level_mask, face, face)))(field)
        assert np.all(np.isfinite(np.asarray(grad)))
        assert np.any(np.asarray(grad) != 0.0)

    def test_nemo_literal_before_ssh_is_observable_against_now_control(self):
        """A nonuniform BEFORE/NOW SSH pair must not collapse to one seed.

        This is the unit-scale red control for the time-level owner measured in
        round 18: the MLF caller supplies ``eta_init`` (BEFORE), and the seed
        thickness/reciprocal must be constructed from that same value.
        """
        grid, z, state = _flat_basin(
            n_lat=8, n_lon=16, H=1000.0, lat_cap_deg=90.0)
        key_u, key_v = jax.random.split(jax.random.PRNGKey(149155))
        u3 = jax.random.normal(key_u, state.u.data.shape) * state.u_mask.data[..., None]
        v3 = jax.random.normal(key_v, state.v.data.shape) * state.v_mask.data[..., None]
        eta_before = (0.31 * jnp.sin(grid.lon2d)
                      + 0.09 * jnp.cos(2.0 * grid.lat2d)) * state.land_mask.data
        eta_now = (-0.27 * jnp.cos(2.0 * grid.lon2d)
                   + 0.07 * jnp.sin(grid.lat2d)) * state.land_mask.data
        area = grid.area.astype(jnp.float64)
        common = (jnp.asarray(0.0), state.land_mask.data,
                  state.u_mask.data, state.v_mask.data, grid)

        h_before = compute_layer_thickness(
            eta_before, state.H_bathy.data, z, min_water_column_m=0.0)
        seed_before = _depth_average_to_faces(
            u3, v3, h_before, *common,
            seed_face_depth="nemo_ssh_avg", seed_evaluation="nemo_literal",
            eta_dyn=eta_before, H_bathy=state.H_bathy.data, area=area,
            z_coord=z)
        h_now = compute_layer_thickness(
            eta_now, state.H_bathy.data, z, min_water_column_m=0.0)
        seed_now = _depth_average_to_faces(
            u3, v3, h_now, *common,
            seed_face_depth="nemo_ssh_avg", seed_evaluation="nemo_literal",
            eta_dyn=eta_now, H_bathy=state.H_bathy.data, area=area,
            z_coord=z)

        assert np.any(np.asarray(seed_before[0]) != np.asarray(seed_now[0]))
        assert np.any(np.asarray(seed_before[1]) != np.asarray(seed_now[1]))

    def test_nemo_literal_prefers_carried_reference_mesh_operands(self):
        """The bridge's exact e3/hu/hv/area operands own the literal path.

        A deliberately different model-native ``h_k`` makes the fallback a
        red control: if dispatch silently stops consuming the carried NEMO
        reference mesh, the bit-exact expected seed below changes.
        """
        n_lat, n_lon, nlev = 3, 4, 3
        rng = np.random.default_rng(149155)
        u_native = rng.normal(size=(n_lat, n_lon, nlev))
        v_native = rng.normal(size=(n_lat, n_lon, nlev))
        u3 = np.concatenate([u_native[:, -1:], u_native], axis=1)
        v3 = np.concatenate([np.zeros_like(v_native[:1]), v_native], axis=0)
        h_k = np.ones((n_lat, n_lon, nlev), dtype=np.float64)
        eta = np.asarray([
            [0.11, -0.07, 0.03, 0.19],
            [-0.13, 0.05, 0.17, -0.02],
            [0.09, 0.01, -0.15, 0.08],
        ])
        e3 = np.broadcast_to(
            np.asarray([1.25, 2.5, 5.0]), (n_lat, n_lon, nlev)).copy()
        hu0 = e3.sum(axis=-1)
        hv0 = hu0.copy()
        area_t = 2.0 + np.arange(n_lat * n_lon).reshape(n_lat, n_lon) / 13.0
        area_u = 3.0 + np.arange(n_lat * n_lon).reshape(n_lat, n_lon) / 17.0
        area_v = 4.0 + np.arange(n_lat * n_lon).reshape(n_lat, n_lon) / 19.0
        z_coord = SimpleNamespace(
            nemo_e3t_0=jnp.asarray(e3), nemo_hu_0=jnp.asarray(hu0),
            nemo_hv_0=jnp.asarray(hv0), nemo_e1e2t=jnp.asarray(area_t),
            nemo_e1e2u=jnp.asarray(area_u), nemo_e1e2v=jnp.asarray(area_v))
        u_mask = np.ones((n_lat, n_lon + 1), dtype=np.float64)
        v_mask = np.ones((n_lat + 1, n_lon), dtype=np.float64)
        v_mask[0] = 0.0
        v_mask[-1] = 0.0
        mask = np.ones((n_lat, n_lon), dtype=np.float64)
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)

        area_eta = area_t * eta
        r3u = (0.5 * (area_eta + np.roll(area_eta, -1, axis=1))
               / hu0 / area_u)
        north = np.concatenate([area_eta[1:], np.zeros_like(area_eta[:1])], axis=0)
        r3v = 0.5 * (area_eta + north) / hv0 / area_v
        wet_v = np.ones((n_lat, n_lon, nlev), dtype=np.float64)
        wet_v[-1] = 0.0

        def source_mean(field, live_h, live_r1, wet):
            acc = np.zeros(field.shape[:2], dtype=np.float64)
            for jk in range(nlev):
                acc = acc + ((live_h[..., jk] * field[..., jk])
                             * wet[..., jk])
            return acc * live_r1

        expected_u_native = source_mean(
            u_native, e3 * (1.0 + r3u[..., None]),
            (1.0 / hu0) / (1.0 + r3u), np.ones_like(e3))
        expected_v_native = source_mean(
            v_native, (e3 * (1.0 + r3v[..., None])) * wet_v,
            ((1.0 / hv0) / (1.0 + r3v)) * wet_v[..., 0], wet_v)
        expected_u = np.concatenate(
            [expected_u_native[:, -1:], expected_u_native], axis=1)
        expected_v = np.concatenate(
            [np.zeros_like(expected_v_native[:1]), expected_v_native], axis=0)

        args = (jnp.asarray(u3), jnp.asarray(v3), jnp.asarray(h_k),
                jnp.asarray(0.0), jnp.asarray(mask), jnp.asarray(u_mask),
                jnp.asarray(v_mask), grid)
        got_u, got_v = _depth_average_to_faces(
            *args, seed_face_depth="nemo_ssh_avg",
            seed_evaluation="nemo_literal", eta_dyn=jnp.asarray(eta),
            H_bathy=jnp.full_like(jnp.asarray(eta), 8.75),
            area=jnp.asarray(area_t), z_coord=z_coord)
        np.testing.assert_array_equal(np.asarray(got_u), expected_u)
        np.testing.assert_array_equal(np.asarray(got_v), expected_v)

        # Without the carried mesh the literal seed needs the card's own
        # reference ladder (e3u_0 = min-rule of e3t_0, domain.F90:145); a
        # caller with neither gets a refusal, never a silently different
        # number.
        with pytest.raises(ValueError, match="reference ladder"):
            _depth_average_to_faces(
                *args, seed_face_depth="nemo_ssh_avg",
                seed_evaluation="nemo_literal", eta_dyn=jnp.asarray(eta),
                H_bathy=jnp.full_like(jnp.asarray(eta), 8.75),
                area=jnp.asarray(area_t), z_coord=None)

    def test_default_min_rule_byte_identical_to_pre_change(self):
        """Default is "min_rule" — the new kwarg is purely additive; a run
        with the option left at default must reproduce a run from BEFORE the
        option existed (i.e. calling ``_depth_average_to_faces`` with no
        ``seed_face_depth``/``eta_dyn``/``H_bathy``/``area`` kwargs at all)
        exactly, and the full substep loop must be byte-identical whether or
        not the field is explicitly set to "min_rule"."""
        grid, z, state = _flat_basin()
        key = jax.random.PRNGKey(11)
        u0 = 0.05 * jax.random.normal(key, state.u.data.shape) * state.u_mask.data[..., None]
        v0 = 0.05 * jax.random.normal(
            jax.random.split(key)[0], state.v.data.shape) * state.v_mask.data[..., None]
        state = state._replace(u=state.u.replace(data=u0), v=state.v.replace(data=v0))
        eta0 = 0.3 * jnp.sin(2.0 * grid.lon2d) * state.land_mask.data
        state = state._replace(eta=state.eta.replace(data=eta0))

        # Leaf-level: with vs without the new kwargs at all (pre-#1226-round-2
        # call signature).
        h_k = compute_layer_thickness(state.eta.data, state.H_bathy.data, z,
                                      min_water_column_m=0.0)
        min_wc = jnp.asarray(0.0)
        U_pre, V_pre = _depth_average_to_faces(
            state.u.data, state.v.data, h_k, min_wc, state.land_mask.data,
            state.u_mask.data, state.v_mask.data, grid)
        U_post, V_post = _depth_average_to_faces(
            state.u.data, state.v.data, h_k, min_wc, state.land_mask.data,
            state.u_mask.data, state.v_mask.data, grid,
            seed_face_depth="min_rule", seed_evaluation="generic")
        np.testing.assert_array_equal(np.asarray(U_pre), np.asarray(U_post))
        np.testing.assert_array_equal(np.asarray(V_pre), np.asarray(V_post))

        # Full substep loop: default config vs explicit "min_rule".
        cfg_default = _cfg(barotropic_time_filter="cosine")
        cfg_explicit = _cfg(barotropic_time_filter="cosine",
                            barotropic_seed_face_depth="min_rule",
                            barotropic_seed_evaluation="generic",
                            barotropic_pgf_evaluation="generic")
        assert cfg_default.barotropic.barotropic_seed_face_depth == "min_rule"
        assert cfg_default.barotropic.barotropic_seed_evaluation == "generic"
        sn_a, (Hu_a, Hv_a) = barotropic_substeps_latlon_cgrid(
            state, 60.0, 30, grid, z, cfg_default, add_barotropic_coriolis=True)
        sn_b, (Hu_b, Hv_b) = barotropic_substeps_latlon_cgrid(
            state, 60.0, 30, grid, z, cfg_explicit, add_barotropic_coriolis=True)
        np.testing.assert_array_equal(np.asarray(sn_a.eta.data), np.asarray(sn_b.eta.data))
        np.testing.assert_array_equal(np.asarray(sn_a.u.data), np.asarray(sn_b.u.data))
        np.testing.assert_array_equal(np.asarray(sn_a.v.data), np.asarray(sn_b.v.data))
        np.testing.assert_array_equal(np.asarray(Hu_a), np.asarray(Hu_b))
        np.testing.assert_array_equal(np.asarray(Hv_a), np.asarray(Hv_b))

    def test_nemo_literal_pgf_uses_face_metrics_and_source_order(self):
        grid = ensure_geometry(create_latlon_grid(n_lat=5, n_lon=8))
        # Make the V metric observably non-reconstructible from the legacy
        # cell-height average, as on the NEMO Mercator bridge.
        scale = 1.0 + 2.0e-4 * jnp.arange(6, dtype=jnp.float64)[:, None]
        grid = grid._replace(dy_v=grid.dy_v * scale)
        jj = jnp.arange(5, dtype=jnp.float64)[:, None]
        ii = jnp.arange(8, dtype=jnp.float64)[None, :]
        eta = 0.17 * jnp.sin(0.3 * ii) + 0.11 * jnp.cos(0.4 * jj)
        um = jnp.ones((5, 9), dtype=jnp.float64)
        vm = jnp.ones((6, 8), dtype=jnp.float64).at[0].set(0.0).at[-1].set(0.0)
        g = jnp.asarray(9.80665)
        pu, pv = _nemo_literal_barotropic_pressure_gradient(eta, grid, g, um, vm)

        e = np.asarray(eta)
        du = np.roll(e, -1, axis=1) - e
        expected_u_native = ((-float(g) * du)
                             * (1.0 / np.asarray(grid.dx_u)[:, 1:]))
        expected_u = np.concatenate(
            [expected_u_native[:, -1:], expected_u_native], axis=1)
        dv = e[1:] - e[:-1]
        expected_v = np.concatenate([
            np.zeros_like(e[:1]),
            ((-float(g) * dv) * (1.0 / np.asarray(grid.dy_v)[1:-1])),
            np.zeros_like(e[:1]),
        ], axis=0)
        np.testing.assert_array_equal(np.asarray(pu), expected_u)
        np.testing.assert_array_equal(np.asarray(pv), expected_v)
        generic_v = -g * gradient_y_cgrid(eta, grid)
        assert np.any(np.asarray(generic_v)[1:-1] != expected_v[1:-1])

    def test_ground_truth_column_reimplementation(self):
        """Independent, from-scratch NumPy re-derivation of the
        "nemo_ssh_avg" seed on a tiny synthetic column — NOT calling any
        legoESM helper (no ``min_cell_to_uface``, no
        ``nemo_ssh_avg_face_depth``, no ``depth_average_to_faces``) so it
        cannot share a bug with the implementation under test.

        Ground truth: for z-star (a single per-column Jacobian scales every
        level identically), the barotropic mean at a u-face under either
        rule is

            U_bar = sum_k(u_k * h_face_k) / sum_k(h_face_k)

        where ``h_face_k = h_face_k^{minrule} * (H_face^{nemo} /
        H_face^{minrule})`` is level-INDEPENDENT scaling, so algebraically

            U_bar^{nemo} = U_bar^{minrule} * (H^{nemo}_u / H^{minrule}_u)
                          [... ONLY when u is the SAME at every level, i.e.
                           a purely barotropic column with no shear, where
                           the ratio scaling is trivially exact regardless
                           of shear too, since sum_k(u*h*ratio)/sum_k(h*ratio)
                           = ratio*sum_k(u*h) / (ratio*sum_k(h)) = the SAME
                           depth mean as min_rule for ANY shear, because the
                           ratio is a level-independent CONSTANT that cancels
                           top and bottom].

        This is the sharpest possible ground-truth check: under z-star, a
        level-independent face-thickness rescale can NEVER change the
        depth-mean velocity (the ratio cancels in the num/denom) — it only
        changes the face DEPTH the caller sees, never the seeded velocity.
        Verify exactly that (both with uniform and SHEARED synthetic u) by
        hand-coding the min-rule and nemo-rule 2-D total depths and the
        depth-mean formula directly in raw NumPy, independent of the module
        under test.
        """
        n_lat, n_lon, nlev = 3, 4, 3
        rng = np.random.default_rng(0)
        # Synthetic column: flat bathymetry with ONE shallow step (column 2),
        # small ssh perturbation, SHEARED velocity (differs per level).
        H_bathy = np.full((n_lat, n_lon), 100.0)
        H_bathy[:, 2] = 40.0
        mask = np.ones((n_lat, n_lon))
        eta = 0.05 * np.sin(np.arange(n_lon))[None, :] * np.ones((n_lat, 1))
        dz_ref = np.array([10.0, 30.0, 60.0])  # sums to H_max=100
        H_max = dz_ref.sum()
        J = (eta + H_bathy) / H_max                     # z-star Jacobian
        h_k = dz_ref[None, None, :] * J[..., None]       # (n_lat, n_lon, nlev)
        u3 = rng.standard_normal((n_lat, n_lon + 1, nlev))  # sheared per level
        u_mask = np.ones((n_lat, n_lon + 1))
        u_mask[:, 0] = 0.0  # west wall closed (periodic wrap not exercised here)

        # --- ground truth #1: min-rule 3-D face thickness + plain depth mean
        h_u_min = np.minimum(np.roll(h_k, 1, axis=1), h_k)
        h_u_min = np.concatenate([h_u_min, h_u_min[:, 0:1]], axis=1)
        U_bar_min = (np.sum(u3 * h_u_min, axis=-1)
                    / np.maximum(np.sum(h_u_min, axis=-1), 1e-30)) * u_mask

        # --- ground truth #2: hand-coded NEMO 2-D total face depth (min-rule
        # reference + a PLAIN 2-point mean in place of the e1e2-metric
        # weighting — equivalent on a uniform grid where e1u*e2u is constant
        # per row, which create_latlon_grid's Mercator metric is NOT in
        # general; so here we build a UNIFORM-metric synthetic column
        # specifically so the plain mean is the exact NEMO formula, keeping
        # this ground truth free of any grid-geometry helper).
        H_total = H_bathy  # rest-state total depth for the reference term
        H_u_ref = np.minimum(np.roll(H_total, 1, axis=1), H_total)
        H_u_ref = np.concatenate([H_u_ref, H_u_ref[:, 0:1]], axis=1)
        eta_w = np.roll(eta, 1, axis=1)
        ssh_avg_u = 0.5 * (eta_w + eta)  # uniform e1e2 metric -> plain mean
        ssh_avg_u = np.concatenate([ssh_avg_u, ssh_avg_u[:, 0:1]], axis=1)
        H_u_nemo = H_u_ref + ssh_avg_u

        H_total_dyn = np.sum(h_k, axis=-1)
        H_u_min_total = np.minimum(np.roll(H_total_dyn, 1, axis=1), H_total_dyn)
        H_u_min_total = np.concatenate(
            [H_u_min_total, H_u_min_total[:, 0:1]], axis=1)

        ratio_u = H_u_nemo / np.maximum(H_u_min_total, 1e-30)
        h_u_nemo_3d = h_u_min * ratio_u[..., None]
        U_bar_nemo_handcoded = (
            np.sum(u3 * h_u_nemo_3d, axis=-1)
            / np.maximum(np.sum(h_u_nemo_3d, axis=-1), 1e-30)) * u_mask

        # Claim: for z-star, rescaling every level's face thickness by the
        # SAME (level-independent) ratio leaves the depth-MEAN unchanged —
        # the ratio cancels between numerator and denominator — regardless
        # of vertical shear in u3.
        np.testing.assert_allclose(
            U_bar_nemo_handcoded, U_bar_min, rtol=0, atol=1e-12,
            err_msg=("ground-truth column re-derivation: a level-independent "
                     "z-star face-thickness rescale must leave the depth-mean "
                     "velocity unchanged"))

    def test_nemo_ssh_avg_seed_changes_face_depth_not_velocity_zstar(self):
        """Direct test of the actual implementation
        (``_depth_average_to_faces(seed_face_depth="nemo_ssh_avg")``) against
        the SAME invariance claim as the ground-truth test above: rescaling
        the min-rule face thickness by the NEMO/min-rule total-depth ratio
        must leave ``U_bar``/``V_bar`` EXACTLY equal to the min_rule result.
        The ratio is a per-COLUMN scalar for ANY vertical coordinate (not
        just z-star), so it cancels between the numerator and denominator of
        the thickness-weighted mean AWAY FROM THE WATER-COLUMN FLOOR (this
        test uses ``min_water_column_m=0.0``, i.e. the floor disabled) —
        measured 2026-07-27 on the DINO Y5 twin: bit-identical seeds,
        ~1e-16 output round-off; see the
        ``BarotropicConfig.barotropic_seed_face_depth`` docstring — NEMO's
        own qco per-column e3u stretch cancels identically, so the
        convention gap cancels in both models on DINO's deep-basin columns.
        This is NOT a general inertness proof: at the PRODUCTION
        ``min_water_column_m=0.5`` default, the floor can bind
        asymmetrically on a shelf column and break the equality — see
        ``test_shelf_column_floor_breaks_inertness_at_production_default``
        below."""
        grid, z, state = _flat_basin(n_lat=8, n_lon=16, H=1000.0, lat_cap_deg=90.0)
        H_bathy = np.full((grid.n_lat, grid.n_lon), 1000.0)
        H_bathy[:, 8] = 250.0
        state = state._replace(H_bathy=state.H_bathy.replace(data=jnp.asarray(H_bathy)))
        eta0 = 0.4 * jnp.sin(2.0 * grid.lon2d) * state.land_mask.data
        state = state._replace(eta=state.eta.replace(data=eta0))
        key = jax.random.PRNGKey(21)
        u0 = jax.random.normal(key, state.u.data.shape) * state.u_mask.data[..., None]
        v0 = jax.random.normal(
            jax.random.split(key)[0], state.v.data.shape) * state.v_mask.data[..., None]
        state = state._replace(u=state.u.replace(data=u0), v=state.v.replace(data=v0))

        h_k = compute_layer_thickness(state.eta.data, state.H_bathy.data, z,
                                      min_water_column_m=0.0)
        min_wc = jnp.asarray(0.0)
        args = (state.u.data, state.v.data, h_k, min_wc, state.land_mask.data,
               state.u_mask.data, state.v_mask.data, grid)
        U_min, V_min = _depth_average_to_faces(*args, seed_face_depth="min_rule")
        U_nemo, V_nemo = _depth_average_to_faces(
            *args, seed_face_depth="nemo_ssh_avg", eta_dyn=state.eta.data,
            H_bathy=state.H_bathy.data, area=grid.area.astype(state.eta.data.dtype))
        np.testing.assert_allclose(np.asarray(U_nemo), np.asarray(U_min),
                                   rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.asarray(V_nemo), np.asarray(V_min),
                                   rtol=0, atol=1e-9)

    def test_shelf_column_floor_breaks_inertness_at_production_default(self):
        """CRITICAL adversarial-review finding on commit 7da6d7989: the
        "VELOCITY-SEED INERT by construction" claim is FALSE in general — it
        held only in the tests/twins that disabled the water-column floor
        (``min_water_column_m=0.0``) or used a flat deep basin.  At the
        PRODUCTION default ``min_water_column_m=0.5``
        (``LatLonCGridOceanConfig.min_water_column_m``,
        ``ocean_tendency_common.column_depth``), the two face-depth rules'
        OWN total-column-depth references can straddle the floor on a thin
        shelf column: the min-rule total depth stays floor-UNBOUND while the
        NEMO ssh-average reference (a fixed ``hu_0`` + a half-weighted
        2-point ssh average, structurally different from a plain column
        sum) is floor-BOUND, or vice versa.  ``max(Sigma h, floor)`` is then
        NOT a common per-column scalar between the two rules, so it does
        NOT cancel out of the thickness-weighted mean — the "it always
        cancels" argument in the docstrings assumed a floor-free or
        floor-symmetric column.

        Reproduction (reviewer's shelf-column construction, reproduced here
        independently): a 300 m deep basin with ONE thin shelf column
        (``H_bathy=0.2 m``) at a ``+0.5 m`` ssh anomaly, so the min-rule
        total depth is ``H_bathy + eta = 0.7 m`` (floor-unbound) but the
        NEMO ssh-average reference lands at ``~0.448 m`` (floor-bound at
        the 0.5 m production default) — matching the reviewer's H=300 m /
        H_u_minrule=0.6 / H_u_nemo=0.449 counter-example order of
        magnitude.  Even a spatially UNIFORM (unsheared) seed velocity
        differs by ~10% between "min_rule" and "nemo_ssh_avg" at that face.

        Non-vacuous both ways: also assert the two modes agree exactly on a
        DEEP column (no floor interaction anywhere), so this test would
        fail loudly if the helper were changed to disagree everywhere
        (not just at the floor).
        """
        n_lat, n_lon = 8, 16
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
        z = create_ocean_z_star(n_levels=4, H_max=300.0, dz_surface=50.0, dz_deep=100.0)
        H_bathy = np.full((grid.n_lat, grid.n_lon), 300.0)
        shelf_col = 9
        H_bathy[:, shelf_col] = 0.2  # thin shelf column, bathy alone < floor
        land_mask = np.ones((grid.n_lat, grid.n_lon))
        state = rest_state_latlon_cgrid_ocean(
            grid, z, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
            H_max=300.0, H_bathy_override=jnp.asarray(H_bathy),
            land_mask_override=jnp.asarray(land_mask))
        eta0 = np.zeros((grid.n_lat, grid.n_lon))
        eta0[:, shelf_col] = 0.5  # pushes min-rule total depth to 0.7 m (floor-unbound)
        eta0 = jnp.asarray(eta0) * state.land_mask.data
        state = state._replace(eta=state.eta.replace(data=eta0))
        # Spatially uniform seed velocity (no vertical shear, no horizontal
        # structure) — isolates the floor effect from any shear/pattern.
        u_uniform = jnp.ones_like(state.u.data) * state.u_mask.data[..., None]
        v_zero = jnp.zeros_like(state.v.data)
        state = state._replace(u=state.u.replace(data=u_uniform),
                               v=state.v.replace(data=v_zero))

        min_wc = 0.5  # LatLonCGridOceanConfig.min_water_column_m production default
        h_k = compute_layer_thickness(state.eta.data, state.H_bathy.data, z,
                                      min_water_column_m=min_wc)
        args = (state.u.data, state.v.data, h_k, jnp.asarray(min_wc),
                state.land_mask.data, state.u_mask.data, state.v_mask.data, grid)
        U_min, _ = _depth_average_to_faces(*args, seed_face_depth="min_rule")
        U_nemo, _ = _depth_average_to_faces(
            *args, seed_face_depth="nemo_ssh_avg", eta_dyn=state.eta.data,
            H_bathy=state.H_bathy.data, area=grid.area.astype(state.eta.data.dtype))
        U_min = np.asarray(U_min)
        U_nemo = np.asarray(U_nemo)

        # The east face of the shelf column (index shelf_col+1, since u-face
        # j is between T-cells j-1 and j on this grid's roll convention —
        # verified directly: this is the face whose min-rule/nemo total
        # depths straddle the floor).
        shelf_face = shelf_col
        rel_diff = abs(U_nemo[1, shelf_face] - U_min[1, shelf_face]) / abs(U_min[1, shelf_face])
        assert rel_diff > 0.08, (
            f"expected the floor to break inertness by >8% at the shelf "
            f"face (production min_water_column_m=0.5); got {rel_diff:.4%} "
            f"(U_min={U_min[1, shelf_face]!r}, U_nemo={U_nemo[1, shelf_face]!r})")

        # Non-vacuous: a DEEP column (far from the shelf, no floor
        # interaction) must still agree exactly — the floor-breaks-inertness
        # claim is a LOCAL effect, not a wholesale disagreement.
        deep_face = 3
        np.testing.assert_allclose(
            U_min[:, deep_face], U_nemo[:, deep_face], rtol=0, atol=1e-9,
            err_msg="deep-column faces (no floor interaction) must still "
                   "agree between min_rule and nemo_ssh_avg seeds")

    def test_kamm_recipes_select_nemo_ssh_avg_seed(self):
        """Card selection: both DINO kamm recipes (FE and MLF) opt into
        ``barotropic_seed_face_depth="nemo_ssh_avg"``; every other recipe
        stays at the legacy default."""
        from legoesm.ocean.experiments.dino import (
            DINO_RECIPES, dino_config_for_recipe, dino_lat_lon_model_config,
        )
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.vertical import create_ocean_z_star as _mk_z

        for name in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
            c = dino_config_for_recipe(name)
            assert c.barotropic_seed_face_depth == "nemo_ssh_avg", name
            assert c.barotropic_seed_evaluation == "nemo_literal", name
            assert (c.barotropic_transport_accumulation_evaluation
                    == "nemo_literal"), name
            assert (c.barotropic_een_coefficient_evaluation
                    == "nemo_literal"), name
            assert c.barotropic_pgf_evaluation == "nemo_literal", name
            grid = create_latlon_grid(n_lat=8, n_lon=16)
            mc, _ = dino_lat_lon_model_config(grid, c)
            assert mc.barotropic.barotropic_seed_face_depth == "nemo_ssh_avg", name
            assert mc.barotropic.barotropic_seed_evaluation == "nemo_literal", name
            assert (mc.barotropic.barotropic_transport_accumulation_evaluation
                    == "nemo_literal"), name
            assert (mc.barotropic.barotropic_een_coefficient_evaluation
                    == "nemo_literal"), name
            assert mc.barotropic.barotropic_pgf_evaluation == "nemo_literal", name

        for name, spec in DINO_RECIPES.items():
            if name in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
                continue
            c = dino_config_for_recipe(name)
            # #1226: nemo_ssh_avg seed is a kamm-only override; every other
            # recipe must stay at the bit-identical legacy default.
            assert c.barotropic_seed_face_depth == "min_rule", name
            assert c.barotropic_seed_evaluation == "generic", name
            assert (c.barotropic_transport_accumulation_evaluation
                    == "generic"), name
            assert c.barotropic_een_coefficient_evaluation == "generic", name
            assert c.barotropic_pgf_evaluation == "generic", name
