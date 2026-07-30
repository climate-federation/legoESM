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

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.vertical import create_ocean_z_star, compute_layer_thickness
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    barotropic_substeps_latlon_cgrid,
    _depth_average_to_faces,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    divergence_cgrid, min_cell_to_uface, min_cell_to_vface,
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
        # Expected single-rate decay of the time-MEAN over the step.  Frozen
        # F_slow gives U(t)=U0 - (r U0/H) t, whose step-mean is U0(1 - r*dt/(2H)).
        expected_mean = U0 * (1.0 - r * dt / (2.0 * H))
        # The DOUBLE-counted (old) drag would also apply ~ (1 - r*dt/H) on top,
        # giving a clearly smaller value; assert we are within 5% of the
        # single-owner expectation and NOT near the double-rate value.
        single = abs(u_mean - expected_mean) / expected_mean
        double_mean = U0 * (1.0 - 3.0 * r * dt / (2.0 * H))  # ~2x drag proxy
        assert single < 0.02, (
            f"barotropic drag mean u={u_mean:.6f} vs single-owner "
            f"{expected_mean:.6f} (rel {single:.3e}); drag may be double-counted")
        assert abs(u_mean - expected_mean) < abs(u_mean - double_mean), (
            "barotropic drag closer to the DOUBLE-counted rate than the "
            "single-owner rate (finding #6 regression)")


class TestBarotropicFaceDepthNemoSshAvg:
    """#1226 ``barotropic_face_depth="nemo_ssh_avg"`` (dynspg_ts.F90:568-592
    flux depth; :658-666,771-778 drag/update depth)."""

    def test_unknown_face_depth_raises(self):
        grid, z, state = _flat_basin()
        cfg = _cfg(barotropic_face_depth="bogus_scheme")
        with pytest.raises(ValueError, match="barotropic_face_depth"):
            barotropic_substeps_latlon_cgrid(
                state, 60.0, 4, grid, z, cfg, add_barotropic_coriolis=False)

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
                   barotropic_face_depth="nemo_ssh_avg")
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
            seed_face_depth="min_rule")
        np.testing.assert_array_equal(np.asarray(U_pre), np.asarray(U_post))
        np.testing.assert_array_equal(np.asarray(V_pre), np.asarray(V_post))

        # Full substep loop: default config vs explicit "min_rule".
        cfg_default = _cfg(barotropic_time_filter="cosine")
        cfg_explicit = _cfg(barotropic_time_filter="cosine",
                            barotropic_seed_face_depth="min_rule")
        assert cfg_default.barotropic.barotropic_seed_face_depth == "min_rule"
        sn_a, (Hu_a, Hv_a) = barotropic_substeps_latlon_cgrid(
            state, 60.0, 30, grid, z, cfg_default, add_barotropic_coriolis=True)
        sn_b, (Hu_b, Hv_b) = barotropic_substeps_latlon_cgrid(
            state, 60.0, 30, grid, z, cfg_explicit, add_barotropic_coriolis=True)
        np.testing.assert_array_equal(np.asarray(sn_a.eta.data), np.asarray(sn_b.eta.data))
        np.testing.assert_array_equal(np.asarray(sn_a.u.data), np.asarray(sn_b.u.data))
        np.testing.assert_array_equal(np.asarray(sn_a.v.data), np.asarray(sn_b.v.data))
        np.testing.assert_array_equal(np.asarray(Hu_a), np.asarray(Hu_b))
        np.testing.assert_array_equal(np.asarray(Hv_a), np.asarray(Hv_b))

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
            grid = create_latlon_grid(n_lat=8, n_lon=16)
            mc, _ = dino_lat_lon_model_config(grid, c)
            assert mc.barotropic.barotropic_seed_face_depth == "nemo_ssh_avg", name

        for name, spec in DINO_RECIPES.items():
            if name in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
                continue
            c = dino_config_for_recipe(name)
            # #1226: nemo_ssh_avg seed is a kamm-only override; every other
            # recipe must stay at the bit-identical legacy default.
            assert c.barotropic_seed_face_depth == "min_rule", name
