"""The C-grid barotropic Coriolis 2Δx rotational null mode and its faithful cure.

Root cause of the §5 eddy-resolving turbulent blow-up (docs/issues/
barotropic_mode_noise.md §A): the in-substep barotropic Coriolis interpolates V
to u-points with the 4-point average ``0.25·(V[i]+V[i+1] + V_west[i]+V_west[i+1])``
(``barotropic_latlon_cgrid.py`` L377-379), where ``V_west = roll(V, 1, axis=lon)``.
A 2Δx ZONAL checkerboard, V[:,j] = (-1)^j, makes V_west = −V, so the average
vanishes and the discrete Coriolis operator exerts NO restoring on it — the
divergent high-zonal-wavenumber Arakawa-Lamb (1977) null mode, free to grow under
the eddy field until the run blows up.

The faithful cure (``coriolis_scheme="explicit_ab2"``) routes the planetary
Coriolis to the barotropic mode through the AB2-extrapolated slow forcing F_slow
(Oceananigans split-explicit convention: ∂_tU = −gH∇η + G^U, NO in-substep
Coriolis) → ``add_barotropic_coriolis=False`` → the null-mode operator is never
applied.  No dissipation backstop.

These tests drive the REAL ``barotropic_substeps_latlon_cgrid`` (no duplicated
stencil numerics) to pin both halves: (1) the checkerboard is a Coriolis null
mode — it produces no U response while a smooth V does; (2) ``add_barotropic_
coriolis=False`` (the cure) removes the in-substep Coriolis coupling entirely.

The full 160×128×50 80-day survival of the cured stack is validated offline by
``scripts/tmp/_silvestri_eps_validate.py`` (too expensive for CI); these unit
tests pin the *mechanism* that makes that survival hold.
"""

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import barotropic_substeps_latlon_cgrid
from legoesm.ocean.experiments.silvestri_baroclinic_jet import (
    SilvestriJetConfig,
    build_silvestri_baroclinic_jet_setup,
)

set_policy(PrecisionPolicy.fp64())


def _rest_setup():
    """Small §5 setup, then quiesce: u=v=eta=0, alpha=0 (isolate Coriolis)."""
    r = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    cfg = r.model_config._replace(barotropic=r.model_config.barotropic._replace(barotropic_diffusion_alpha=0.0))
    s = r.initial_state
    z_u = jnp.zeros_like(s.u.data)
    z_v = jnp.zeros_like(s.v.data)
    z_e = jnp.zeros_like(s.eta.data)
    s = s._replace(
        u=Field(data=z_u, name="u", dims=s.u.dims, units=s.u.units),
        v=Field(data=z_v, name="v", dims=s.v.dims, units=s.v.units),
        eta=Field(data=z_e, name="eta", dims=s.eta.dims, units=s.eta.units))
    return r, cfg, s


def _set_v(s, v3d):
    return s._replace(v=Field(data=v3d, name="v", dims=s.v.dims, units=s.v.units))


def _checkerboard_v(s):
    """2Δx zonal checkerboard: v[:,j,:] = (-1)^j (the Coriolis null mode).

    The 4-point V→u average includes ``V_west = roll(V, 1, axis=lon)``
    (``barotropic_latlon_cgrid.py`` L377): for a zonal checkerboard
    V_west = −V, so ``V_bar_c + V_west = 0`` and V_at_u vanishes — the
    divergent high-zonal-wavenumber Arakawa-Lamb mode the Coriolis
    operator cannot see (docs/issues/barotropic_mode_noise.md §A)."""
    nlat_v, nlon, nlev = s.v.data.shape
    sign = ((-1.0) ** jnp.arange(nlon))[None, :, None]
    return jnp.broadcast_to(sign, (nlat_v, nlon, nlev)).astype(s.v.data.dtype)


def _smooth_v(s):
    """A smooth (DC) v field of the same amplitude for the control."""
    return jnp.ones_like(s.v.data)


def _u_field(r, cfg, s, add_cor):
    """One barotropic step from rest with the given v; return the u field."""
    out, _ = barotropic_substeps_latlon_cgrid(
        s, dt_s=30.0, n_substeps=10, grid=r.grid, z_coord=r.z_coord,
        config=cfg, add_barotropic_coriolis=add_cor)
    return np.asarray(out.u.data)


def _coriolis_u(r, cfg, s):
    """Coriolis-ONLY U contribution = |u(cor on) − u(cor off)| (isolates the
    Coriolis term from the continuity→PGF gravity-wave response, which both
    paths share)."""
    return float(np.max(np.abs(
        _u_field(r, cfg, s, add_cor=True) - _u_field(r, cfg, s, add_cor=False))))


def test_checkerboard_is_coriolis_null_mode():
    """A 2Δx zonal checkerboard V drives ~no Coriolis U; a smooth V drives a lot."""
    r, cfg, s = _rest_setup()
    cor_checker = _coriolis_u(r, cfg, _set_v(s, _checkerboard_v(s)))
    cor_smooth = _coriolis_u(r, cfg, _set_v(s, _smooth_v(s)))
    # The smooth field feels the Coriolis operator...
    assert cor_smooth > 1e-4, f"smooth V felt no Coriolis: {cor_smooth:.2e}"
    # ...the checkerboard is annihilated by the 4-point average → null mode:
    # the discrete Coriolis exerts no restoring on it.
    assert cor_checker < 1e-3 * cor_smooth, (
        f"checkerboard is NOT a Coriolis null mode: cor_checker={cor_checker:.2e} "
        f"vs cor_smooth={cor_smooth:.2e}")


def test_cure_removes_in_substep_coriolis():
    """add_barotropic_coriolis=False (the explicit_ab2 cure) drops the term."""
    r, cfg, s = _rest_setup()
    s_smooth = _set_v(s, _smooth_v(s))
    u_on = _u_field(r, cfg, s_smooth, add_cor=True)
    u_off = _u_field(r, cfg, s_smooth, add_cor=False)
    # With Coriolis ON a smooth (divergence-free) V drives U purely via Coriolis;
    # with it OFF (the cure) there is no in-substep Coriolis and no U at all.
    assert float(np.max(np.abs(u_on))) > 1e-4, (
        f"control: Coriolis-on must drive U, got {np.max(np.abs(u_on)):.2e}")
    assert float(np.max(np.abs(u_off))) < 1e-12, (
        f"cure: add_barotropic_coriolis=False must remove the in-substep "
        f"Coriolis (no U from a divergence-free V), got {np.max(np.abs(u_off)):.2e}")


def _een_cfg(cfg):
    return cfg._replace(barotropic=cfg.barotropic._replace(
        barotropic_coriolis="een"))


def test_een_does_NOT_restore_the_checkerboard_null_mode():
    """EEN CANNOT restore the zonal 2Δx mode — it reduces to the 4-pt average.

    REWRITTEN 2026-07-29.  This test previously asserted the OPPOSITE
    ("EEN restores the null mode", ``cor_ck_een > 0.1 * cor_sm_een``) and
    passed only because of a PERIODIC-SEAM INDEX BUG in
    ``pv_flux_al81_partial_cell`` / ``pv_flux_ene`` (fixed 2026-07-29): the
    west-neighbour v-flux was built by rolling an already-wrapped ``(n_lon+1)``
    array, so u-face 0 received cell 0 instead of cell ``n-1``.  The measured
    "restoration" lived ENTIRELY in that one column —

        buggy: col 0 = 3.64e-02, col 1 = 1.13e-03, INTERIOR = 2.90e-07
        fixed: uniform 2.58e-07 at every column

    — and the metric here is ``np.max``, so a single bad column carried it.

    The correct statement is ANALYTIC, not empirical: for a UNIFORM ``q`` the
    AL81 12-point triad collapses to ``3q₀/12 = q₀/4`` on all four triads,
    i.e. EXACTLY the 4-point average ``0.25·q₀·(F_SW+F_SE+F_NW+F_NE)``
    (verified: ``|AL81 − 4pt| = 2.7e-20``).  The barotropic call uses ``ζ=0``,
    so its ``q`` IS uniform up to ``f``.  The 4-point average annihilates the
    zonal 2Δx checkerboard, therefore SO DOES EEN.  This matches
    ``.claude/ralph_barotropic_coriolis_redesign_task.md:27``: "No
    null-mode-free LINEAR C-grid f×U exists (the V→u lon-average always kills
    2Δx-lon)."  Corroboration: a controlled 30-day nemo_dino_kamm probe found
    ``avg`` and ``een`` give identical deep-equatorial KE growth.
    """
    r, cfg, s = _rest_setup()
    een = _een_cfg(cfg)
    ck = _set_v(s, _checkerboard_v(s))
    sm = _set_v(s, _smooth_v(s))
    cor_ck_avg = _coriolis_u(r, cfg, ck)
    cor_ck_een = _coriolis_u(r, een, ck)
    cor_sm_een = _coriolis_u(r, een, sm)
    # the 4-pt average annihilates the checkerboard (unchanged control)
    assert cor_ck_avg < 1e-3 * cor_sm_een, (
        f"control: avg should annihilate the checkerboard, got {cor_ck_avg:.2e}")
    # ...and EEN annihilates it too, because it REDUCES to that average.
    assert cor_ck_een < 1e-3 * cor_sm_een, (
        f"EEN should NOT restore the zonal 2Dx null mode (it reduces to the "
        f"4-pt average for uniform q): cor_ck_een={cor_ck_een:.2e} vs "
        f"cor_sm_een={cor_sm_een:.2e}.  A LARGE value here means the periodic "
        f"seam has reopened -- see test_pv_flux_periodic_seam_is_closed.")


def test_pv_flux_periodic_seam_is_closed():
    """REGRESSION for the 2026-07-29 seam bug: u-face 0 == u-face n_lon.

    On a zonally periodic grid the first and last columns of a ``(n_lon+1)``
    u-face array are the SAME physical face.  ``divergence_cgrid`` telescopes a
    row to ``(u[n_lon] − u[0])·dy``, so any disagreement is a FABRICATED VOLUME
    SOURCE: before the fix the barotropic solver gained 1.86e10 m³ per
    68-substep window in a CLOSED domain.

    This asserts the invariant DIRECTLY on both PV-flux operators, which the
    budget tests did not (``test_al81_budget.py`` explicitly skips i=0 as
    "periodic wrap, not exercised").
    """
    import jax.numpy as _jnp
    import numpy as _np

    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        pv_flux_al81_partial_cell,
        pv_flux_ene,
    )
    rng = _np.random.default_rng(0)
    n_lat, n_lon, nlev = 6, 8, 2
    f = lambda *sh: _jnp.asarray(rng.normal(size=sh))          # noqa: E731
    zeta = f(n_lat + 1, n_lon + 1, nlev)
    h_vtx = _jnp.abs(f(n_lat + 1, n_lon + 1, nlev)) + 1.0
    h_v, v = _jnp.abs(f(n_lat + 1, n_lon, nlev)) + 1.0, f(n_lat + 1, n_lon, nlev)
    h_u, u = _jnp.abs(f(n_lat, n_lon + 1, nlev)) + 1.0, f(n_lat, n_lon + 1, nlev)
    # periodic inputs: the u-face seam column IS column 0
    h_u = _jnp.concatenate([h_u[:, :-1], h_u[:, 0:1]], axis=1)
    u = _jnp.concatenate([u[:, :-1], u[:, 0:1]], axis=1)
    zeta = _jnp.concatenate([zeta[:, :-1], zeta[:, 0:1]], axis=1)
    h_vtx = _jnp.concatenate([h_vtx[:, :-1], h_vtx[:, 0:1]], axis=1)
    um = _jnp.ones((n_lat, n_lon + 1, nlev))
    vm = _jnp.ones((n_lat + 1, n_lon, nlev))
    vtx = _jnp.ones((n_lat + 1, n_lon + 1))

    for name, fn in (("al81", pv_flux_al81_partial_cell), ("ene", pv_flux_ene)):
        du, _ = fn(zeta, h_vtx, h_v, v, h_u, u, um, vm, vtx)
        seam = float(_np.max(_np.abs(_np.asarray(du)[:, 0, :]
                                     - _np.asarray(du)[:, n_lon, :])))
        scale = float(_np.max(_np.abs(_np.asarray(du))))
        assert seam <= 1e-12 * max(scale, 1e-300), (
            f"{name}: periodic seam OPEN -- u-face 0 != u-face n_lon "
            f"(|diff|={seam:.3e}, field max={scale:.3e}).  divergence_cgrid "
            f"turns this into a spurious volume source.")


def test_een_barotropic_coriolis_conserves_energy():
    """The EEN barotropic Coriolis does ~no work (Σ hu·A·U·cor_u + hv·A·V·cor_v
    ≈ 0), the defining property of the enstrophy-conserving triad.  Residual is
    limited by the coastal Neumann fill / partial cells (same character as the
    baroclinic AL81 operator), not machine precision on a walled basin."""
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _build_een_barotropic_inputs,
        een_barotropic_coriolis,
    )
    from legoesm.ocean.vertical import compute_layer_thickness
    setup = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    grid = ensure_geometry(setup.grid)
    st = setup.initial_state
    h_bathy = st.H_bathy.data.astype(jnp.float64)
    mask = st.land_mask.data.astype(jnp.float64)
    um = st.u_mask.data.astype(jnp.float64)
    vm = st.v_mask.data.astype(jnp.float64)
    h_k = compute_layer_thickness(
        jnp.zeros_like(h_bathy), h_bathy, setup.z_coord,
        min_water_column_m=setup.model_config.min_water_column_m)
    pre = _build_een_barotropic_inputs(h_k, grid, mask, um, vm, jnp.float64)
    nlat, nlon = mask.shape
    rng = np.random.default_rng(0)
    u_r = jnp.asarray(rng.standard_normal((nlat, nlon + 1))) * um
    v_r = jnp.asarray(rng.standard_normal((nlat + 1, nlon))) * vm
    cu, cv = een_barotropic_coriolis(u_r, v_r, pre)
    area = grid.area
    a_u = 0.5 * (jnp.roll(area, 1, 1) + area)
    a_u = jnp.concatenate([a_u, a_u[:, :1]], 1)
    a_v = jnp.concatenate([area[:1], 0.5 * (area[:-1] + area[1:]), area[-1:]], 0)
    work = float(jnp.sum(pre["hu"] * a_u * u_r * cu)
                 + jnp.sum(pre["hv"] * a_v * v_r * cv))
    scale = float(jnp.sum(jnp.abs(pre["hu"] * a_u * u_r * cu))
                  + jnp.sum(jnp.abs(pre["hv"] * a_v * v_r * cv)))
    assert abs(work) / scale < 1e-2, (
        f"EEN Coriolis does spurious work: rel={work / scale:.2e}")


def test_een_pre_step_matches_substep_zero_live_term():
    """``barotropic_coriolis_een_pre_step`` (the live-split F_slow subtraction) IS
    exactly the substep-0 live EEN Coriolis the loop applies — so subtracting it
    from F_slow cancels the double-count at substep 0 (the node-16 LIVE cure).

    Pins the argument wiring: the helper must compose the SAME
    ``_depth_average_to_faces`` + ``_build_een_barotropic_inputs`` +
    ``een_barotropic_coriolis`` the substep loop calls internally (a swapped u/v
    or wrong thickness would break this equality)."""
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _build_een_barotropic_inputs,
        _depth_average_to_faces,
        barotropic_coriolis_een_pre_step,
        een_barotropic_coriolis,
    )
    from legoesm.ocean.vertical import compute_layer_thickness
    setup = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    grid = ensure_geometry(setup.grid)
    st = setup.initial_state
    h_bathy = st.H_bathy.data.astype(jnp.float64)
    mask = st.land_mask.data.astype(jnp.float64)
    um = st.u_mask.data.astype(jnp.float64)
    vm = st.v_mask.data.astype(jnp.float64)
    h_k = compute_layer_thickness(
        jnp.zeros_like(h_bathy), h_bathy, setup.z_coord,
        min_water_column_m=setup.model_config.min_water_column_m)
    nlat, nlon, nlev = h_k.shape
    rng = np.random.default_rng(1)
    u3 = jnp.asarray(rng.standard_normal((nlat, nlon + 1, nlev))) * um[..., None]
    v3 = jnp.asarray(rng.standard_normal((nlat + 1, nlon, nlev))) * vm[..., None]
    mwc = jnp.asarray(setup.model_config.min_water_column_m, dtype=jnp.float64)
    cu, cv = barotropic_coriolis_een_pre_step(
        u3, v3, h_k, grid, mask, um, vm, mwc, jnp.float64)
    # independent reconstruction of the substep-0 live term
    pre = _build_een_barotropic_inputs(h_k, grid, mask, um, vm, jnp.float64)
    U, V = _depth_average_to_faces(u3, v3, h_k, mwc, mask, um, vm, grid)
    cu_ref, cv_ref = een_barotropic_coriolis(U, V, pre)
    np.testing.assert_allclose(np.asarray(cu), np.asarray(cu_ref), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(cv), np.asarray(cv_ref), rtol=0, atol=0)
    # non-vacuous: the term is actually non-trivial
    assert float(np.max(np.abs(np.asarray(cu)))) > 0.0


def test_explicit_ab2_config_gates_in_substep_coriolis():
    """The §5 faithful stack wires coriolis_scheme=explicit_ab2 (=> term off)."""
    r = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    mc = r.model_config
    assert mc.coriolis_scheme == "explicit_ab2"
    assert mc.barotropic.barotropic_solver == "explicit_substep"
    assert mc.barotropic.barotropic_slow_forcing_ab2 is True
    assert mc.barotropic.barotropic_diffusion_alpha == 0.0


# =====================================================================
# METRIC-COMPLETE EEN (barotropic_coriolis="een_metric"): NEMO ffu/ffv
# retain the e1v/r1_e1u (u) + e2u/r1_e2v (v) horizontal scale factors the
# per-unit-width "een" drops (dynspg_ts.F90:1349-1379) — the coefficients NEMO's
# dyn_cor_2D actually uses, so this is the strictly more NEMO-faithful barotropic
# Coriolis. These gates pin (a) the fold is EXACTLY an identity when the metrics
# are uniform (so "een_metric" ≡ "een" ≡ f·V̄ in the flat-uniform limit) and
# (b) it is a latitude-scaling correction (≈0 at the equator, O(1%) at high lat)
# that reduces the high-lat Coriolis energy-budget residual.
# NB (twin-verified 2026-07-19): this ~1% correction does NOT cure the DINO
# |lat|~68° 2Δx ETA runaway (the FE step-twin still NaNs at s26, near byte-
# identical to "een") — that is a free-surface mode via the barotropic
# PGF/continuity coupling, not the Coriolis V→u averaging the EEN restores.
# =====================================================================


def _een_pre_from_silvestri(metric_complete):
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _build_een_barotropic_inputs,
    )
    from legoesm.ocean.vertical import compute_layer_thickness
    setup = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    grid = ensure_geometry(setup.grid)
    st = setup.initial_state
    h_bathy = st.H_bathy.data.astype(jnp.float64)
    mask = st.land_mask.data.astype(jnp.float64)
    um = st.u_mask.data.astype(jnp.float64)
    vm = st.v_mask.data.astype(jnp.float64)
    h_k = compute_layer_thickness(
        jnp.zeros_like(h_bathy), h_bathy, setup.z_coord,
        min_water_column_m=setup.model_config.min_water_column_m)
    pre = _build_een_barotropic_inputs(
        h_k, grid, mask, um, vm, jnp.float64, metric_complete=metric_complete)
    return grid, mask, um, vm, pre


def test_een_metric_reduces_to_een_on_uniform_metrics():
    """Flat-uniform limit (gate a): when the horizontal scale factors are
    UNIFORM (e1u≡e1v, e2u≡e2v) the NEMO metric fold — scale the flux velocity
    by the neighbour width, divide the output by the local width — is EXACTLY
    an identity, so "een_metric" reproduces "een" to machine precision.  On a
    lat-lon grid they legitimately DIFFER (that difference IS the fix)."""
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        een_barotropic_coriolis,
    )
    _, _, um, vm, pre_plain = _een_pre_from_silvestri(metric_complete=False)
    # Force uniform metrics onto a metric_complete pre: any nonzero constant c
    # gives (·c)/(c) = identity → must equal the plain "een" output bit-for-bit
    # up to fp round-off.
    c1, c2 = 7.0e4, 1.1e5
    pre_uniform = dict(pre_plain)
    pre_uniform["metric_complete"] = True
    pre_uniform["e1u"] = jnp.full_like(pre_plain["hu"], c1)
    pre_uniform["e1v"] = jnp.full_like(pre_plain["hv"], c1)
    pre_uniform["e2u"] = jnp.full_like(pre_plain["hu"], c2)
    pre_uniform["e2v"] = jnp.full_like(pre_plain["hv"], c2)
    nlat, nlon = um.shape[0], vm.shape[1]
    rng = np.random.default_rng(3)
    u_r = jnp.asarray(rng.standard_normal((nlat, nlon + 1))) * um
    v_r = jnp.asarray(rng.standard_normal((nlat + 1, nlon))) * vm
    cu_p, cv_p = een_barotropic_coriolis(u_r, v_r, pre_plain)
    cu_m, cv_m = een_barotropic_coriolis(u_r, v_r, pre_uniform)
    np.testing.assert_allclose(np.asarray(cu_m), np.asarray(cu_p),
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(cv_m), np.asarray(cv_p),
                               rtol=1e-12, atol=1e-14)
    # non-vacuous: the term is genuinely non-trivial
    assert float(np.max(np.abs(np.asarray(cu_p)))) > 0.0


def _highlat_channel_pre(metric_complete, lat_s=55.0, lat_n=75.0,
                         n_lat=40, n_lon=16):
    """Coast-free (periodic-x) high-latitude channel, all interior wet, flat
    bottom — isolates the O(Δcosφ) metric error from any coastal Neumann fill."""
    from legoesm.grids.latlon import create_regional_latlon_grid, ensure_geometry
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _build_een_barotropic_inputs,
    )
    grid1d, land = create_regional_latlon_grid(
        n_lat=n_lat, n_lon=n_lon, lat_south=lat_s, lat_north=lat_n,
        periodic_x=True)
    grid = ensure_geometry(grid1d)
    mask = jnp.asarray(land, dtype=jnp.float64)   # interior all wet, N/S walls
    um, vm = compute_face_masks(mask, grid)
    um = um.astype(jnp.float64)
    vm = vm.astype(jnp.float64)
    ny, nx = mask.shape
    h_k = jnp.broadcast_to(
        (mask * 1000.0)[..., None], (ny, nx, 1)).astype(jnp.float64)
    pre = _build_een_barotropic_inputs(
        h_k, grid, mask, um, vm, jnp.float64, metric_complete=metric_complete)
    return grid, mask, um, vm, pre


def _energy_rel(grid, pre, u_r, v_r, cu, cv):
    """|Σ (e1·e2·h)·(u·cor_u + v·cor_v)| / Σ|·| — the Coriolis energy-budget
    residual under the NEMO discrete KE norm (u-/v-cell volume e1·e2·h =
    grid.dx_*·dy_*·h). The AL81/EEN operator is an energy-ENSTROPHY compromise,
    NOT a machine-precision energy conserver on a walled basin (see the sibling
    test docstring), so this is O(1e-2..1e-3); the metric fold reduces the
    LAT-DEPENDENT slice of it."""
    a_u = grid.dx_u * grid.dy_u
    a_v = grid.dx_v * grid.dy_v
    wu = pre["hu"] * a_u * u_r * cu
    wv = pre["hv"] * a_v * v_r * cv
    work = float(jnp.sum(wu) + jnp.sum(wv))
    scale = float(jnp.sum(jnp.abs(wu)) + jnp.sum(jnp.abs(wv)))
    return abs(work) / scale


def _metric_correction_and_energy(lat_s, lat_n, seed):
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        een_barotropic_coriolis,
    )
    g_p, _, um, vm, pre_p = _highlat_channel_pre(
        metric_complete=False, lat_s=lat_s, lat_n=lat_n)
    g_m, _, _, _, pre_m = _highlat_channel_pre(
        metric_complete=True, lat_s=lat_s, lat_n=lat_n)
    nlat, nlon = um.shape[0], vm.shape[1]
    rng = np.random.default_rng(seed)
    u_r = jnp.asarray(rng.standard_normal((nlat, nlon + 1))) * um
    v_r = jnp.asarray(rng.standard_normal((nlat + 1, nlon))) * vm
    cu_p, cv_p = een_barotropic_coriolis(u_r, v_r, pre_p)
    cu_m, cv_m = een_barotropic_coriolis(u_r, v_r, pre_m)
    corr = float(np.max(np.abs(np.asarray(cu_m) - np.asarray(cu_p)))) / max(
        float(np.max(np.abs(np.asarray(cu_p)))), 1e-30)
    return corr, _energy_rel(g_p, pre_p, u_r, v_r, cu_p, cv_p), \
        _energy_rel(g_m, pre_m, u_r, v_r, cu_m, cv_m)


def test_een_metric_correction_scales_with_latitude():
    """High-lat gate (gate b): the NEMO e1v/r1_e1u + e2u/r1_e2v metric fold is
    the term the EEN enstrophy budget needs where ∂cosφ ≠ 0.  It must therefore
    (i) VANISH in the near-uniform equatorial band and (ii) become a material
    O(1%) correction at 55–75°N, where it also REDUCES the Coriolis energy-budget
    residual — the exact latitude signature of NEMO's high-lat metric term.
    (A machine-precision energy gate is NOT claimed: the isolated AL81 operator
    is an energy-enstrophy compromise, not an exact energy conserver — the flat-
    uniform identity gate above is the machine-precision anchor.  And this ~1%
    correction is NOT the DINO |lat|~68° eta-runaway fix: the FE/MLF step-twin
    still blows with een_metric — see the module header.)"""
    corr_eq, _, _ = _metric_correction_and_energy(-5.0, 5.0, seed=4)
    corr_hi, rel_p, rel_m = _metric_correction_and_energy(55.0, 75.0, seed=4)
    # (i) metric ≡ identity in the near-uniform equatorial band...
    assert corr_eq < 1e-3, f"metric wrongly active at the equator: {corr_eq:.2e}"
    # (ii) ...and a material, latitude-growing correction at high lat...
    assert corr_hi > 5e-3, f"metric fold too weak at high lat: {corr_hi:.2e}"
    assert corr_hi > 10.0 * corr_eq, (
        f"metric correction does not scale with latitude: "
        f"eq={corr_eq:.2e} hi={corr_hi:.2e}")
    # (iii) BOTH energy residuals stay small.
    #
    # REWRITTEN 2026-07-29.  This previously asserted ``rel_m < rel_p`` -- that
    # the metric fold REDUCES the high-latitude energy residual.  That claim
    # rested on the periodic-seam index bug fixed the same day: with the seam
    # open the seam error DOMINATED this channel's energy budget, and the fold
    # happened to reduce it.  Closing the seam improved BOTH residuals by 1-2
    # orders of magnitude and FLIPPED their order:
    #
    #     buggy:  rel_plain 3.224e-03   rel_metric 2.810e-03   (fold "helps")
    #     fixed:  rel_plain 1.935e-05   rel_metric 2.825e-04   (fold costs)
    #
    # i.e. the fold was never the thing reducing the residual; the seam bug was
    # the thing inflating it.  Once the dominant error is gone the fold
    # contributes its own, ~15x larger than plain.  So the ordering is NOT
    # asserted either way -- what IS asserted is that both stay small, which is
    # a real gate (it would catch the seam reopening: rel_plain would jump 167x).
    assert rel_p < 1.0e-3, f"plain Coriolis energy residual too large: {rel_p:.2e}"
    assert rel_m < 5.0e-3, f"metric Coriolis energy residual too large: {rel_m:.2e}"


def test_een_metric_dispatches_and_does_NOT_restore_null_mode():
    """``een_metric`` is accepted end-to-end AND (like ``een``) annihilates 2Δx.

    REWRITTEN 2026-07-29 alongside
    :func:`test_een_does_NOT_restore_the_checkerboard_null_mode` -- the
    "restores" half of the original assertion was the periodic-seam artifact.
    The metric fold changes the Coriolis COEFFICIENT (a latitude-dependent
    e3f/metric correction, still asserted in
    :func:`test_een_metric_correction_scales_with_latitude`); it does not change
    the fact that the triad reduces to the 4-point average for uniform ``q``,
    which annihilates the zonal 2Δx mode.  The DISPATCH half is the part worth
    keeping, so it is kept and strengthened.
    """
    r, cfg, s = _rest_setup()
    metric = cfg._replace(barotropic=cfg.barotropic._replace(
        barotropic_coriolis="een_metric"))
    ck = _set_v(s, _checkerboard_v(s))
    sm = _set_v(s, _smooth_v(s))
    cor_ck_avg = _coriolis_u(r, cfg, ck)
    cor_ck_m = _coriolis_u(r, metric, ck)
    cor_sm_m = _coriolis_u(r, metric, sm)
    # it dispatches and does real work on a SMOOTH field...
    assert cor_sm_m > 1e-4, (
        f"een_metric did not dispatch / did no work on a smooth V: {cor_sm_m:.2e}")
    # ...and annihilates the checkerboard, exactly as the 4-pt average does.
    assert cor_ck_avg < 1e-3 * cor_sm_m
    assert cor_ck_m < 1e-3 * cor_sm_m, (
        f"een_metric should NOT restore the zonal 2Dx null mode: "
        f"{cor_ck_m:.2e} vs smooth {cor_sm_m:.2e}.  A large value means the "
        f"periodic seam has reopened.")



# ----------------------------------------------------------------------
# #1226 dyn_cor_2d: the card's EEN options must REACH the barotropic path
# ----------------------------------------------------------------------
#
# NEMO's barotropic Coriolis coefficients are built from ``ff_f(ji,jj) /
# e3f_vor(ji,jj,jk)`` RAW -- the SAME ``e3f_vor`` array ``vor_een`` uses,
# with no boundary fill and (DINO: ``ln_dynvor_msk=.false.``) no ``fmask``
# (``dynspg_ts.F90:1517-1531``).  So a card that selects
# ``een_q_boundary="nemo_live"`` + ``een_e3f_scheme="nemo_avg"``
# (``nn_e3f_typ=1``) for the 3-D EEN must get them in the BAROTROPIC EEN
# too.  Before 2026-08 ``_build_een_barotropic_inputs`` hard-coded the
# min-rule ``e3f`` and let ``pv_flux_al81_partial_cell`` fall through to its
# ``q_boundary="neumann_fill"`` DEFAULT, silently ignoring both card
# settings on this path.

def _stepped_channel(**cfg_kw):
    """Rest channel with a bathymetry STEP (so min-rule e3f != nemo_avg e3f)
    and a coast (so the q boundary treatment is live)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    n_lat, n_lon, n_levels, H_max = 8, 10, 5, 3000.0
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z = create_ocean_z_star(n_levels=n_levels, H_max=H_max)
    H = np.full((n_lat, n_lon), H_max)
    H[:, :3] = 900.0                     # shelf step
    H[0, :] = 0.0                        # coast row (dry)
    H_bathy = jnp.asarray(H)
    land = jnp.asarray((H > 0.0).astype(np.float64))
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        H_bathy_override=H_bathy, land_mask_override=land)
    cfg_kw.setdefault("barotropic_coriolis", "een_metric")
    cfg_kw.setdefault("A_v", 0.0)
    cfg_kw.setdefault("K_v", 0.0)
    cfg_kw.setdefault("A_h", 0.0)
    cfg_kw.setdefault("K_h", 0.0)
    config = LatLonCGridOceanConfig.from_flat(**cfg_kw)
    return grid, z, state, config


def _een_pre(grid, z, state, config, **kw):
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _build_een_barotropic_inputs,
    )
    from legoesm.ocean.vertical import compute_layer_thickness
    h_bathy = state.H_bathy.data.astype(jnp.float64)
    h_k = compute_layer_thickness(
        jnp.zeros_like(h_bathy), h_bathy, z,
        min_water_column_m=config.min_water_column_m)
    return _build_een_barotropic_inputs(
        h_k, ensure_geometry(grid), state.land_mask.data.astype(jnp.float64),
        state.u_mask.data.astype(jnp.float64),
        state.v_mask.data.astype(jnp.float64), jnp.float64, **kw)


def test_een_barotropic_inputs_thread_e3f_scheme():
    """``een_e3f_scheme`` reaches the vertex thickness: "nemo_avg" must give
    a DIFFERENT ``h_vtx`` from "min" at the bathymetry step, and must be
    bit-identical to the SHARED production helper ``een_e3f_h_vtx`` (no
    re-derived inline copy of the divisor -- the #1226 item-10 failure)."""
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import een_e3f_h_vtx
    from legoesm.ocean.vertical import compute_layer_thickness
    grid, z, state, config = _stepped_channel()
    pre_min = _een_pre(grid, z, state, config, een_e3f_scheme="min")
    pre_avg = _een_pre(grid, z, state, config, een_e3f_scheme="nemo_avg")
    h_min = np.asarray(pre_min["h_vtx"])
    h_avg = np.asarray(pre_avg["h_vtx"])
    assert h_min.dtype == np.float64 and h_avg.dtype == np.float64
    n_diff = int((np.abs(h_min - h_avg) > 1e-9).sum())
    print(f"\n  h_vtx min vs nemo_avg: {n_diff} of {h_min.size} entries differ")
    assert n_diff > 0, "een_e3f_scheme is being ignored (h_vtx identical)"

    # ... and it IS the shared helper's output, not a look-alike.
    h_bathy = state.H_bathy.data.astype(jnp.float64)
    h_k = compute_layer_thickness(
        jnp.zeros_like(h_bathy), h_bathy, z,
        min_water_column_m=config.min_water_column_m)
    for scheme, pre in (("min", pre_min), ("nemo_avg", pre_avg)):
        ref, fu, uu = een_e3f_h_vtx(h_k, None, None, ensure_geometry(grid), scheme)
        assert fu is None and uu is None
        np.testing.assert_array_equal(np.asarray(pre["h_vtx"]), np.asarray(ref))


def test_een_barotropic_inputs_thread_q_boundary_into_the_operator():
    """``een_q_boundary`` reaches ``pv_flux_al81_partial_cell``: the option
    is carried on ``pre`` and the two settings give DIFFERENT coefficients
    at the coast (non-vacuous -- a threaded-but-unused string would leave
    the two identical)."""
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        een_barotropic_coriolis,
    )
    grid, z, state, config = _stepped_channel()
    pre_fill = _een_pre(grid, z, state, config, een_q_boundary="neumann_fill")
    pre_live = _een_pre(grid, z, state, config, een_q_boundary="nemo_live")
    assert pre_fill["q_boundary"] == "neumann_fill"
    assert pre_live["q_boundary"] == "nemo_live"
    nlat, nlon = np.asarray(state.land_mask.data).shape
    rng = np.random.default_rng(3)
    um = state.u_mask.data.astype(jnp.float64)
    vm = state.v_mask.data.astype(jnp.float64)
    u_r = jnp.asarray(rng.standard_normal((nlat, nlon + 1))) * um
    v_r = jnp.asarray(rng.standard_normal((nlat + 1, nlon))) * vm
    cu_f, cv_f = een_barotropic_coriolis(u_r, v_r, pre_fill)
    cu_l, cv_l = een_barotropic_coriolis(u_r, v_r, pre_live)
    d = max(float(np.abs(np.asarray(cu_f) - np.asarray(cu_l)).max()),
            float(np.abs(np.asarray(cv_f) - np.asarray(cv_l)).max()))
    print(f"  max|cor(neumann_fill) - cor(nemo_live)| = {d:.3e}")
    assert d > 0.0, "een_q_boundary is carried but not used by the operator"


def test_een_barotropic_unknown_option_values_raise():
    """Dispatch hardening on the two newly-threaded options."""
    import pytest as _pytest
    grid, z, state, config = _stepped_channel()
    with _pytest.raises(ValueError, match="unknown een_e3f_scheme"):
        _een_pre(grid, z, state, config, een_e3f_scheme="bogus")
    with _pytest.raises(ValueError, match="unknown een_q_boundary"):
        _een_pre(grid, z, state, config, een_q_boundary="bogus")


def test_config_een_options_reach_the_operator_through_the_real_solver():
    """END-TO-END WIRING GATE (not a field-exists assertion).

    Drives the REAL ``barotropic_substeps_latlon_cgrid`` and SPIES on the
    ``q_boundary`` kwarg / ``h_vtx`` argument the barotropic path actually
    hands ``pv_flux_al81_partial_cell``.  A config carrying the NEMO-faithful
    values must produce ``q_boundary="nemo_live"`` and the ``nemo_avg``
    vertex thickness at the operator; the default config must produce
    ``"neumann_fill"`` and the min-rule thickness.

    This is the gap the fix closes: the DINO card set both fields for the
    3-D EEN and the barotropic path silently used its own defaults.
    """
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as bmod

    def _spy(config):
        grid, z, state, cfg = _stepped_channel()
        cfg = cfg._replace(**config)
        seen = []
        real = bmod.pv_flux_al81_partial_cell

        def _wrapped(*a, **kw):
            seen.append((kw.get("q_boundary", "<default>"), np.asarray(a[1])))
            return real(*a, **kw)

        bmod.pv_flux_al81_partial_cell = _wrapped
        try:
            barotropic_substeps_latlon_cgrid(
                state, 60.0, 2, grid, z, cfg, add_barotropic_coriolis=True)
        finally:
            bmod.pv_flux_al81_partial_cell = real
        assert seen, "the EEN barotropic Coriolis never ran"
        return seen, grid, z, state, cfg

    seen_live, grid, z, state, cfg_live = _spy(
        dict(een_q_boundary="nemo_live", een_e3f_scheme="nemo_avg"))
    seen_def, *_ = _spy(dict())

    assert {q for q, _ in seen_live} == {"nemo_live"}, (
        f"card value did not reach the operator: {sorted({q for q, _ in seen_live})}")
    assert {q for q, _ in seen_def} == {"neumann_fill"}, (
        f"default drifted: {sorted({q for q, _ in seen_def})}")

    h_avg = np.asarray(_een_pre(grid, z, state, cfg_live,
                                een_e3f_scheme="nemo_avg")["h_vtx"])
    h_min = np.asarray(_een_pre(grid, z, state, cfg_live,
                                een_e3f_scheme="min")["h_vtx"])
    assert not np.array_equal(h_avg, h_min)          # non-vacuous
    np.testing.assert_array_equal(seen_live[0][1], h_avg)
    np.testing.assert_array_equal(seen_def[0][1], h_min)


def test_dino_nemo_card_selects_the_nemo_live_een_options():
    """The card side of the chain: the NEMO DINO recipe asks for the
    faithful options (``ln_dynvor_msk=.false.`` + ``nn_e3f_typ=1``), which
    the test above proves now reach the barotropic operator."""
    from legoesm.ocean.experiments.dino import DINO_RECIPES
    card = DINO_RECIPES["nemo_dino_kamm_mlf"]
    assert card["een_q_boundary"] == "nemo_live"
    assert card["een_e3f_scheme"] == "nemo_avg"


# ---------------------------------------------------------------------------
# dz_ref symmetry between the 3-D EEN and the barotropic EEN (#1455 F3)
# ---------------------------------------------------------------------------
# The 3-D caller forwards ``dz_ref=z_coord.dz_ref`` to ``een_e3f_h_vtx``
# (ocean_pe_latlon_cgrid.py, _bc_pv_flux call site), which under
# ``een_e3f_scheme="nemo_avg"`` replaces the BIG_H sentinel at a FULLY-DRY
# vertex with NEMO's per-level-uniform reference ``e3f_0`` (#1226 item 10).
# The barotropic path used to pass nothing, so the two built DIFFERENT e3f at
# those vertices -- contradicting the "must be the SAME operator" rationale
# that motivated threading een_q_boundary/een_e3f_scheme in the first place.

def test_dz_ref_reaches_the_barotropic_e3f_and_is_inert_on_the_tendency():
    """Threading ``dz_ref`` (a) genuinely CHANGES ``h_vtx`` at fully-dry
    vertices under ``nemo_avg`` -- so this test is not vacuous -- and (b)
    leaves the barotropic Coriolis tendency BIT-IDENTICAL, because those
    vertices are already zeroed by the surrounding face masks.

    (b) is the measurement, not an assumption: it is asserted at exactly 0.0
    against a non-zero signal scale, so a real change would show up.

    NB the fully-dry vertex has to be built EXPLICITLY (a 2x2 block of
    zero-thickness T-cells).  ``_stepped_channel`` + ``compute_layer_thickness``
    never produces one -- ``min_water_column_m`` floors every column to a
    positive thickness, so ``wet_count > 0`` everywhere and the fallback branch
    is unreachable.  A first draft of this test used that fixture and measured
    0 of 495 entries differing; it would have passed vacuously had the
    non-vacuity assert not been there.
    """
    from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _build_een_barotropic_inputs, een_barotropic_coriolis,
    )
    from legoesm.ocean.vertical import create_ocean_z_star

    n_lat, n_lon, nlev = 6, 8, 4
    grid = ensure_geometry(create_latlon_grid(n_lat=n_lat, n_lon=n_lon))
    z = create_ocean_z_star(n_levels=nlev, H_max=3000.0)
    dz_ref = np.asarray(z.dz_ref)
    assert dz_ref.ndim == 1 and np.all(dz_ref > 0.0)

    wet = np.ones((n_lat, n_lon))
    wet[2:4, 3:5] = 0.0                      # 2x2 dry block -> a dry VERTEX
    h_k = jnp.asarray(wet[:, :, None] * dz_ref[None, None, :])
    mask = jnp.asarray(wet)
    u_mask = jnp.asarray(np.concatenate(
        [wet * np.roll(wet, 1, axis=1), (wet * np.roll(wet, 1, axis=1))[:, :1]],
        axis=1))
    v_mask = jnp.asarray(np.concatenate(
        [np.zeros((1, n_lon)), wet[:-1] * wet[1:], np.zeros((1, n_lon))], axis=0))

    def _pre(**kw):
        return _build_een_barotropic_inputs(
            h_k, grid, mask, u_mask, v_mask, jnp.float64,
            een_e3f_scheme="nemo_avg", **kw)

    pre_none = _pre()
    pre_dz = _pre(dz_ref=z.dz_ref)
    h_none_qb = {}
    h_none = np.asarray(pre_none["h_vtx"])
    h_dz = np.asarray(pre_dz["h_vtx"])
    n_diff = int((h_none != h_dz).sum())
    print(f"\n  h_vtx dz_ref=None vs dz_ref: {n_diff} of {h_none.size} differ")
    assert n_diff > 0, (
        "dz_ref never reached een_e3f_h_vtx -- this fixture has no fully-dry "
        "vertex, so the test would be vacuous")
    # The differing entries are exactly the BIG_H sentinel -> dz_ref[k] swap.
    assert np.all(h_none[h_none != h_dz] > 1e29)

    # ...and the "min" branch is dz_ref-independent (documented contract).
    _min = dict(h_k=h_k, grid=grid, mask=mask, u_mask=u_mask, v_mask=v_mask,
                dtype=jnp.float64, een_e3f_scheme="min")
    np.testing.assert_array_equal(
        np.asarray(_build_een_barotropic_inputs(**_min)["h_vtx"]),
        np.asarray(_build_een_barotropic_inputs(
            **_min, dz_ref=z.dz_ref)["h_vtx"]))

    # Tendency: bit-identical, measured against a non-zero signal scale.
    # Gated under BOTH q_boundary rules -- ``nemo_live`` is the one the DINO
    # card selects AND the one that uses a dry-vertex q raw instead of
    # filling it, so "inert" measured only under the default would not cover
    # the production path.
    rng = np.random.default_rng(4)
    U = jnp.asarray(rng.normal(size=np.asarray(pre_none["hu"]).shape) * 0.1)
    V = jnp.asarray(rng.normal(size=np.asarray(pre_none["hv"]).shape) * 0.1)
    for qb in ("neumann_fill", "nemo_live"):
        p_n = _pre(een_q_boundary=qb)
        p_d = _pre(een_q_boundary=qb, dz_ref=z.dz_ref)
        h_none_qb[qb] = np.asarray(p_n["h_vtx"])
        cu_n, cv_n = een_barotropic_coriolis(U, V, p_n)
        cu_d, cv_d = een_barotropic_coriolis(U, V, p_d)
        scale = max(float(np.abs(np.asarray(cu_n)).max()),
                    float(np.abs(np.asarray(cv_n)).max()))
        d = max(float(np.abs(np.asarray(cu_n) - np.asarray(cu_d)).max()),
                float(np.abs(np.asarray(cv_n) - np.asarray(cv_d)).max()))
        print(f"  q_boundary={qb:13s} barotropic Coriolis: max|diff| = "
              f"{d:.3e} at signal {scale:.3e}")
        assert scale > 0.0, f"{qb}: signal is zero -- check is vacuous"
        assert not np.array_equal(h_none_qb[qb],
                                  np.asarray(p_d["h_vtx"])), (
            f"{qb}: dz_ref did not change h_vtx -- check is vacuous")
        assert d == 0.0, f"{qb}: dz_ref changed the tendency by {d:.3e}"


def test_real_solver_forwards_dz_ref_to_the_barotropic_e3f():
    """The production substep loop actually passes ``z_coord.dz_ref`` (not
    ``None``) down to ``een_e3f_h_vtx`` -- the wiring, not just the helper
    signature.  Spies the shared helper at its real import site (the
    barotropic module resolves it lazily, inside the function)."""
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pemod
    grid, z, state, cfg = _stepped_channel(een_e3f_scheme="nemo_avg",
                                           een_q_boundary="nemo_live")
    seen = []
    real = pemod.een_e3f_h_vtx

    def _spy(h_k, Fu, u, g, scheme, dz_ref=None):
        seen.append(dz_ref)
        return real(h_k, Fu, u, g, scheme, dz_ref=dz_ref)

    pemod.een_e3f_h_vtx = _spy
    try:
        barotropic_substeps_latlon_cgrid(
            state, 60.0, 2, grid, z, cfg, add_barotropic_coriolis=True)
    finally:
        pemod.een_e3f_h_vtx = real
    assert seen, "een_e3f_h_vtx never ran in the barotropic path"
    assert all(d is not None for d in seen), (
        f"barotropic path still passes dz_ref=None ({seen})")
    for d in seen:
        np.testing.assert_array_equal(np.asarray(d), np.asarray(z.dz_ref))


def test_nemo_literal_een_builder_jit_gradient_and_face_mapping():
    """Literal triad/reduction/post-factor arithmetic survives JIT intact."""
    from types import SimpleNamespace

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_een_coefficients,
        een_barotropic_coriolis,
    )
    from legoesm.ocean.vertical import NemoEENBarotropicOperands

    rng = np.random.default_rng(122628)
    ny, nx, nz = 5, 8, 6
    shape2, shape3 = (ny, nx), (ny, nx, nz)
    e3u = 5.0 + rng.random(shape3)
    e3v = 6.0 + rng.random(shape3)
    e3f = 7.0 + rng.random(shape3)
    ones3 = np.ones(shape3)
    metric = lambda offset: offset + rng.random(shape2)
    raw = NemoEENBarotropicOperands(
        ff_f=jnp.asarray(1.0e-4 * rng.normal(size=shape2)),
        e3u_0=jnp.asarray(e3u), e3v_0=jnp.asarray(e3v),
        e3f_0=jnp.asarray(e3f), umask=jnp.asarray(ones3),
        vmask=jnp.asarray(ones3), fmask=jnp.asarray(ones3),
        fe3mask=jnp.asarray(ones3),
        hu_0=jnp.asarray(e3u.sum(axis=-1)),
        hv_0=jnp.asarray(e3v.sum(axis=-1)),
        hf_0=jnp.asarray(e3f.sum(axis=-1)),
        e1t=jnp.asarray(metric(10.0)), e2t=jnp.asarray(metric(20.0)),
        e1u=jnp.asarray(metric(30.0)), e2u=jnp.asarray(metric(40.0)),
        e1v=jnp.asarray(metric(50.0)), e2v=jnp.asarray(metric(60.0)),
        e1f=jnp.asarray(metric(70.0)), e2f=jnp.asarray(metric(80.0)),
    )
    z = SimpleNamespace(nemo_een_barotropic=raw)
    eta = jnp.asarray(0.1 * rng.normal(size=shape2))
    ua_native = jnp.asarray(rng.normal(size=shape2))
    va_native = jnp.asarray(rng.normal(size=shape2))
    ua = jnp.concatenate((ua_native[:, -1:], ua_native), axis=1)
    va = jnp.concatenate((jnp.zeros_like(va_native[:1]), va_native), axis=0)

    def evaluate(eta_arg, u_arg, v_arg):
        coeff = _nemo_literal_een_coefficients(eta_arg, z, jnp.float64)
        pre = {
            "coefficient_evaluation": "nemo_literal",
            "literal_coefficients": coeff,
        }
        cu, cv = een_barotropic_coriolis(u_arg, v_arg, pre)
        return tuple(coeff[name] for name in sorted(coeff)) + (cu, cv)

    eager = evaluate(eta, ua, va)
    compiled = jax.jit(evaluate)(eta, ua, va)
    for got, expected in zip(compiled, eager):
        scale = max(float(np.sqrt(np.mean(np.asarray(expected) ** 2))),
                    np.finfo(np.float64).tiny)
        relative_max = (
            float(np.max(np.abs(np.asarray(got) - np.asarray(expected))))
            / scale)
        assert relative_max <= 1.0e-15

    cu, cv = compiled[-2:]
    # Native east/north arrays are mapped only after application: periodic U
    # gets a redundant west face; V gets an exactly zero south-wall face.
    np.testing.assert_array_equal(np.asarray(cu[:, 0]), np.asarray(cu[:, -1]))
    np.testing.assert_array_equal(np.asarray(cv[0]), 0.0)

    direction = jnp.cos(jnp.arange(eta.size, dtype=jnp.float64)).reshape(shape2)

    def loss(scale):
        values = evaluate(eta + scale * direction, ua, va)
        return sum(jnp.sum(value * value) for value in values)

    assert np.isfinite(float(jax.grad(loss)(jnp.asarray(0.0))))


def test_nemo_literal_een_builder_requires_complete_raw_bridge_bundle():
    """The faithful selector fails red instead of rebuilding missing operands."""
    from types import SimpleNamespace

    import pytest

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_een_coefficients,
    )

    with pytest.raises(ValueError, match="bridge-carried raw NEMO"):
        _nemo_literal_een_coefficients(
            jnp.zeros((3, 4)),
            SimpleNamespace(nemo_een_barotropic=None), jnp.float64)


def test_nemo_literal_ene_coefficients_match_source_recurrence_and_red_scale():
    """ENE uses the 1/4 two-point F stencil, not the EEN 1/12 triads."""
    from types import SimpleNamespace

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_een_coefficients,
    )
    from legoesm.ocean.vertical import NemoEENBarotropicOperands

    rng = np.random.default_rng(169902)
    ny, nx, nz = 4, 7, 5
    shape2, shape3 = (ny, nx), (ny, nx, nz)
    e3u = 2.0 + rng.random(shape3)
    e3v = 3.0 + rng.random(shape3)
    e3f = 4.0 + rng.random(shape3)
    ff = 1.0e-4 * rng.normal(size=shape2)
    e1u = 10.0 + rng.random(shape2)
    e1v = 11.0 + rng.random(shape2)
    e2u = 12.0 + rng.random(shape2)
    e2v = 13.0 + rng.random(shape2)
    ones3 = np.ones(shape3)
    raw = NemoEENBarotropicOperands(
        ff_f=jnp.asarray(ff),
        e3u_0=jnp.asarray(e3u), e3v_0=jnp.asarray(e3v),
        e3f_0=jnp.asarray(e3f), umask=jnp.asarray(ones3),
        vmask=jnp.asarray(ones3), fmask=jnp.asarray(ones3),
        fe3mask=jnp.asarray(ones3),
        hu_0=jnp.asarray(e3u.sum(axis=-1)),
        hv_0=jnp.asarray(e3v.sum(axis=-1)),
        hf_0=jnp.asarray(e3f.sum(axis=-1)),
        e1t=jnp.ones(shape2), e2t=jnp.ones(shape2),
        e1u=jnp.asarray(e1u), e2u=jnp.asarray(e2u),
        e1v=jnp.asarray(e1v), e2v=jnp.asarray(e2v),
        e1f=jnp.ones(shape2), e2f=jnp.ones(shape2),
    )
    z = SimpleNamespace(nemo_een_barotropic=raw)
    actual = {
        name: np.asarray(value)
        for name, value in _nemo_literal_een_coefficients(
            jnp.zeros(shape2), z, jnp.float64, scheme="ene"
        ).items()
    }

    def shift(value, di=0, dj=0):
        out = np.roll(value, di, axis=1) if di else value
        return np.roll(out, dj, axis=0) if dj else out

    r1_hu = 1.0 / e3u.sum(axis=-1)
    r1_hv = 1.0 / e3v.sum(axis=-1)
    u_neighbor = {
        "nw": (e3v, e3f, ff, e1v),
        "ne": (shift(e3v, -1, 0), e3f, ff, shift(e1v, -1, 0)),
        "sw": (shift(e3v, 0, 1), shift(e3f, 0, 1),
               shift(ff, 0, 1), shift(e1v, 0, 1)),
        "se": (shift(e3v, -1, 1), shift(e3f, 0, 1),
               shift(ff, 0, 1), shift(e1v, -1, 1)),
    }
    v_neighbor = {
        "nw": (shift(e3u, 1, -1), shift(e3f, 1, 0),
               shift(ff, 1, 0), shift(e2u, 1, -1)),
        "ne": (shift(e3u, 0, -1), e3f, ff, shift(e2u, 0, -1)),
        "sw": (shift(e3u, 1, 0), shift(e3f, 1, 0),
               shift(ff, 1, 0), shift(e2u, 1, 0)),
        "se": (e3u, e3f, ff, e2u),
    }

    def literal_coefficient(face, neighbor, divisor, f_factor,
                            local_metric, r1_h, neighbor_metric):
        term = np.multiply(face, neighbor)
        term = np.multiply(term, np.ones_like(term))
        term = np.divide(term, divisor)
        acc = np.zeros_like(r1_h)
        for jk in range(nz):
            acc = np.add(acc, term[..., jk])
        scale = np.multiply(0.25, np.divide(1.0, local_metric))
        scale = np.multiply(scale, r1_h)
        scale = np.multiply(scale, neighbor_metric)
        scale = np.multiply(scale, f_factor)
        return np.multiply(scale, acc)

    expected = {}
    for corner, (neighbor, divisor, f_factor, metric) in u_neighbor.items():
        expected[f"ffu_{corner}"] = literal_coefficient(
            e3u, neighbor, divisor, f_factor, e1u, r1_hu, metric)
    for corner, (neighbor, divisor, f_factor, metric) in v_neighbor.items():
        expected[f"ffv_{corner}"] = literal_coefficient(
            e3v, neighbor, divisor, f_factor, e2v, r1_hv, metric)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])

    # The formerly folded ``ff/e3f`` form is numerically close but not the
    # same binary64 program: this makes the source-association pin fail if the
    # Round-19 owner is reintroduced.
    folded = 0.25 / e1u * r1_hu * e1v * np.sum(
        e3u * e3v * (ff[..., None] / e3f), axis=-1)
    assert not np.array_equal(actual["ffu_nw"], folded)

    # Non-vacuity: the EEN scale is a material violation on this nonzero case.
    wrong_een_scale = expected["ffu_nw"] / 3.0
    assert np.max(np.abs(actual["ffu_nw"] - wrong_een_scale)) > 1.0e-8
