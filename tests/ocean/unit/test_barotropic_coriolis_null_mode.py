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

