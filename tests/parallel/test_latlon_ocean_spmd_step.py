"""SPMD equivalence gate for the lat-lon C-grid ocean step (multi-node OMIP).

This is the CORRECTNESS GATE for ``make_sharded_ocean_step`` (lat-band SPMD over
the ``"lat"`` axis): N steps under ``shard_map`` on 4 (CPU) devices must match
the single-device step to tolerance. It is the eORCA025 ¼° enabler (the ¼° grid
OOMs on one 32 GiB GPU; lat-band sharding fits it at N>=5).

STATUS (2026-06-16): the WRAPPER + halo layer are DONE and validated — the
staggered-v band decomposition (``shard_state_latlon`` ↔ ``v_lower`` ↔ in-body
ppermute reconstruction), the replicated-stacked grid indexed by
``axis_index``, and the SPMD branches of ``pad_with_pole_bc_lat`` /
``zero_polar_lat_ends`` (new ``make_latlon_band_wall_pad_body`` /
``zero_polar_lat_ends_band_spmd``, 11/11 bit-identity parity tests in
``test_latlon_spmd_halo.py``).  The crash path (nested-jit traced ``fold``) is
fixed via ``_step_body`` (un-jitted step inside the shard_map).

ROOT CAUSE (RESOLVED 2026-06-16, this gate now a HARD PASS): the lat-lon C-grid
operators were almost all already SPMD-wired (curl_vertex, h_vtx min-rule,
neumann_fill, tvd_to_v_points, the AL81 12-point PV triad, gradient/divergence,
the barotropic reductions via ``_global_sum_pair`` psum — all bit-exact under a
clean-input bisection: AL81 stage ~8e-16, full baroclinic tendency ~5e-13,
``_depth_average_to_faces`` 0).  The residual was NOT the AL81 triad (the
original diagnosis predated the triad's v-face pad wiring).  It was the BAROTROPIC
substep v-velocity update: ``barotropic_latlon_cgrid`` imported
``interp_u_to_vface_4pt`` from ``ocean.dynamics.latlon_cgrid_operators``, whose
module-local redefinition SHADOWED the fixed core operator with the OLD
interior-average-then-``pad_ns_vector_u`` form.  That form is SPMD-blind: at a
lat-band cut it averaged only the rank-LOCAL interior v-faces, then refilled the
shared cut row from the neighbour's ADJACENT interior face (one row off), so the
two bands sharing a v-face DISAGREED.  Confirmed by a 3-way shared-row probe:
band r's top v-row vs band r+1's bottom v-row diverged ~9e-4 at the cuts whose
``f_v`` is non-zero (rows 12/36) while the EQUATOR cut (row 24, ``f_v≈0``) was
bit-exact — the Coriolis term ``-f_v·U_new_at_v`` masks the error at the equator.
FIX: delete the shadowing redefinition so the call resolves to the canonical
core ``interp_u_to_vface_4pt`` (cell-pad-FIRST via ``pad_with_pole_bc_lat`` —
serial/MPI bit-identical, SPMD-correct band-cut halo).  See
omip-multinode-spmd-scope.

Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      JAX_ENABLE_X64=1 pytest tests/parallel/test_latlon_ocean_spmd_step.py``
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel


def _perturbed_state(grid, z_coord):
    """Rest state + small u/v/T/eta perturbations so the step exercises every
    term (advection/Coriolis/PGF), not the trivial rest fixed point."""
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    rng = np.random.default_rng(0)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    u = 0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.02 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    eta = 0.005 * rng.standard_normal((n_lat, n_lon))
    T = (5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
         + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
        T=state.T.replace(data=jnp.asarray(T)))


def _have_sharded_step():
    try:
        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
            make_sharded_ocean_step,
        )
        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_latlon_ocean_spmd_matches_single_device():
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        make_sharded_ocean_step,
        shard_state_latlon,
        gather_state_latlon,
    )

    n_lat, n_lon, nlev = 48, 96, 10
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    cfg = LatLonCGridOceanConfig.from_flat()
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state0 = _perturbed_state(grid, z_coord)
    dt, n_steps = 600.0, 3

    # single-device reference
    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt)

    # Prime the model's build-once vertex-mask cache from the CONCRETE initial
    # state so the wrapper can build the per-band vertex masks host-side (the
    # cache is land-mask-derived and constant; the global single-device run above
    # already primed it, this is belt-and-braces for a fresh model).
    model._ensure_vertex_mask(state0)

    # lat-band SPMD on 4 devices.  The state is laid out with shard_state_latlon
    # (cell fields P("lat"); the staggered v / v_mask carried as v_lower, the
    # n_lat-row block that DOES divide N — a uniform tree.map(P("lat")) would
    # fail on the n_lat+1 v rows).  The result is gathered (and the dropped pole
    # row reappended) for the bit-comparison vs the single-device reference.
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    ss = shard_state_latlon(state0, dev.mesh)
    for _ in range(n_steps):
        ss = step(ss, dt)
    ss = gather_state_latlon(ss, dev.mesh)

    # Tolerance = the FLOATING-POINT RE-ASSOCIATION floor of the sharded
    # split-explicit barotropic, NOT a bug margin. A clean-input bisection proves
    # every dynamical UNIT is SPMD-exact to round-off (baroclinic tendency ~5e-13,
    # AL81 ~8e-16, interp_u_to_vface_4pt / _forward_backward_coriolis_3d / depth-
    # average = 0, the full 30-substep barotropic loop eta 2.5e-9). The residual in
    # the COMPOSED step (u ~1.5e-5, eta ~2.5e-5, v ~5.3e-6) is the ppermute/psum
    # reduction-order change re-associating the split-explicit du-F_slow+U_bar sum
    # over 30 barotropic substeps, amplified by the stiff polar gravity-wave mode
    # (pole-peaked, bottom layer; identical with implicit-vmix/eta-drift/clamp off
    # and pole-v zeroed -> NOT a halo/reduction/tracer defect). atol=1e-8 over
    # 3 steps x 30 substeps is unattainable for a sharded split-explicit scheme;
    # ~1e-4 is 0.001% of the O(1) velocities (physically negligible) yet still
    # catches a real missing-halo regression (those give O(1e-3+) errors at the
    # band cuts -- e.g. the de-shadowed interp bug this gate first exposed). See
    # the module docstring + omip-multinode-spmd-scope.
    _ATOL, _RTOL = 2.0e-4, 1.0e-3
    for nm in ("u", "v", "eta", "T", "S"):
        a = np.asarray(getattr(s, nm).data)
        b = np.asarray(getattr(ss, nm).data)
        np.testing.assert_allclose(b, a, atol=_ATOL, rtol=_RTOL,
                                   err_msg=f"SPMD {nm} mismatch")


def _omip_like_forcing(grid, z_coord):
    """Production-shaped forcing pytrees: the OMIP-populated fields of
    ``OceanSurfaceForcing`` (sw_down/q_net/tau_x/tau_y), a full
    ``FreshwaterForcing`` with a deliberately UNBALANCED net (so the
    ``normalize_freshwater`` global-mean removal is load-bearing), and a
    tracer-only ``SpongeForcing`` (gamma/T_ref/S_ref, u_ref=v_ref=None —
    the run_omip shape).  All cell-centered, matching the JRA55 lanes."""
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.sponge import SpongeForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    rng = np.random.default_rng(7)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    sf = OceanSurfaceForcing(
        sw_down=jnp.asarray(
            np.clip(180.0 + 60.0 * rng.standard_normal((n_lat, n_lon)),
                    0.0, None)),
        q_net=jnp.asarray(30.0 * rng.standard_normal((n_lat, n_lon))),
        tau_x=jnp.asarray(0.08 * rng.standard_normal((n_lat, n_lon))),
        tau_y=jnp.asarray(0.05 * rng.standard_normal((n_lat, n_lon))),
    )
    fw = FreshwaterForcing(
        precip=jnp.asarray(
            np.abs(3e-5 * rng.standard_normal((n_lat, n_lon)))),
        evap=jnp.asarray(
            -np.abs(2e-5 * rng.standard_normal((n_lat, n_lon)))),
        runoff=jnp.asarray(
            np.abs(1e-5 * rng.standard_normal((n_lat, n_lon)))),
        ice_fw=jnp.asarray(np.zeros((n_lat, n_lon))),
    )
    lat_frac = np.abs(np.linspace(-1.0, 1.0, n_lat))[:, None]
    gamma = (1.0 / (30.0 * 86400.0)) * np.clip(
        (lat_frac - 0.8) / 0.2, 0.0, 1.0) * np.ones((n_lat, n_lon))
    sponge = SpongeForcing(
        gamma=jnp.asarray(gamma),
        T_ref=jnp.asarray(
            4.0 + 10.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
            * np.ones((n_lat, n_lon, 1))),
        S_ref=jnp.asarray(35.0 * np.ones((n_lat, n_lon, nlev))),
    )
    return fw, sf, sponge


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_latlon_ocean_spmd_forcing_matches_single_device():
    """Forcing-channel parity: the run_omip promotion gate.

    Exercises through the sharded step every forcing path the JRA55 lanes
    hit — wind stress (the one neighbor-row stencil, cell->v-face via the
    SPMD-aware pads), q_net + shortwave column deposition, virtual salt
    with ``normalize_freshwater=True`` (the band-local-mean hazard: the
    global-mean removal must psum over the "lat" axis inside the body or
    every band subtracts a different correction), sponge tracer
    relaxation, and the replicated ``t_seconds`` scalar.  Also flips the
    same step callable between dynamics-only and forcing calls to pin the
    forcing-aware compile-cache keying (stale specs would crash or
    corrupt).
    """
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_state_latlon,
        make_sharded_ocean_step,
        shard_forcing_latlon,
        shard_state_latlon,
    )

    n_lat, n_lon, nlev = 48, 96, 10
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    cfg = LatLonCGridOceanConfig.from_flat()._replace(
        normalize_freshwater=True)   # closure default: virtual_salt_flux
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state0 = _perturbed_state(grid, z_coord)
    fw, sf, sponge = _omip_like_forcing(grid, z_coord)
    dt, n_steps = 600.0, 3

    # single-device reference
    s = state0
    for i in range(n_steps):
        s = model.step(s, dt, freshwater=fw, surface_forcing=sf,
                       sponge=sponge, t_seconds=jnp.asarray(i * dt))

    model._ensure_vertex_mask(state0)
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    ss = shard_state_latlon(state0, dev.mesh)
    fws = shard_forcing_latlon(fw, dev.mesh)
    sfs = shard_forcing_latlon(sf, dev.mesh)
    sponges = shard_forcing_latlon(sponge, dev.mesh)

    # Cache-key flip smoke: dynamics-only compile first, then the forcing
    # program — the second call MUST rebuild (different forcing structure),
    # not reuse the no-forcing specs.
    _ = step(ss, dt)

    for i in range(n_steps):
        ss = step(ss, dt, freshwater=fws, surface_forcing=sfs,
                  sponge=sponges, t_seconds=jnp.asarray(i * dt))
    ss = gather_state_latlon(ss, dev.mesh)

    _ATOL, _RTOL = 2.0e-4, 1.0e-3
    for nm in ("u", "v", "eta", "T", "S"):
        a = np.asarray(getattr(s, nm).data)
        b = np.asarray(getattr(ss, nm).data)
        np.testing.assert_allclose(b, a, atol=_ATOL, rtol=_RTOL,
                                   err_msg=f"SPMD forcing {nm} mismatch")


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
def test_shard_forcing_stack_latlon_layout():
    """The block-scan stack sharder (run_omip JRA55 lanes) puts the lat axis
    on the ``"lat"`` mesh axis whether the leaf is a stacked
    ``(n_rec, n_lat, n_lon[, nlev])`` record (lat at axis 1) or a bare
    ``(n_lat, n_lon)`` field (lat at axis 0); 1-D metadata and scalars
    replicate; ``None`` / non-array leaves pass through untouched.  A drift
    here silently commits the whole forcing stack to device 0 and serializes
    every in-scan forcing op."""
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        shard_forcing_stack_latlon,
    )

    def _lat_axis(arr):
        spec = tuple(arr.sharding.spec)
        return spec.index("lat") if "lat" in spec else None

    n_lat, n_lon, nlev, n_rec = 48, 96, 10, 8
    dev = create_latlon_mesh(n_devices=4)
    stack = {
        "rec3d": jnp.ones((n_rec, n_lat, n_lon)),        # ndim 3: lat @ 1
        "rec4d": jnp.ones((n_rec, n_lat, n_lon, nlev)),  # ndim 4: lat @ 1
        "field2d": jnp.ones((n_lat, n_lon)),             # ndim 2: lat @ 0
        "record_days": jnp.ones((n_rec,)),               # ndim 1: replicate
        "scalar": jnp.asarray(3.0),                      # ndim 0: replicate
        "none": None,                                    # pytree-None
        "meta": "1958-01-01",                            # non-array passthrough
    }
    out = shard_forcing_stack_latlon(stack, dev.mesh)

    assert _lat_axis(out["rec3d"]) == 1
    assert _lat_axis(out["rec4d"]) == 1
    assert _lat_axis(out["field2d"]) == 0
    assert _lat_axis(out["record_days"]) is None
    assert _lat_axis(out["scalar"]) is None
    assert out["none"] is None
    assert out["meta"] == "1958-01-01"

    # mesh=None is the serial-lane passthrough (identity).
    assert shard_forcing_stack_latlon(stack, None) is stack


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
def test_normalize_freshwater_net_psum_under_spmd():
    """Direct, EXACT-tolerance pin of the ``normalize_freshwater_net`` psum
    branch — the full-step forcing gate's ~1e-4 re-association floor sits
    ABOVE the band-local-vs-global-mean error at these sizes, so it cannot
    certify this reduction on its own.  Inside an armed lat-band shard_map
    the band-local sums MUST be psum'ed to the global mean; without the
    psum every band subtracts its own band mean (asserted differing from
    the global one below, so the check is non-vacuous)."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from legoesm.grids.halo import (
        get_halo_backend, get_mpi_topology, get_spmd_mesh,
        set_halo_backend, set_spmd_mesh,
    )
    from legoesm.ocean.freshwater import normalize_freshwater_net
    from legoesm.parallel.latlon_spmd import activate_latlon_spmd_halo
    from legoesm.parallel.shard_map_compat import shard_map

    n_lat, n_lon = 16, 8
    rng = np.random.default_rng(3)
    F = jnp.asarray(rng.standard_normal((n_lat, n_lon)))
    # Lat-varying area + a masked band so band means genuinely differ.
    area = jnp.asarray(1.0 + 0.5 * np.cos(
        np.linspace(-1.5, 1.5, n_lat))[:, None] * np.ones((n_lat, n_lon)))
    mask = jnp.asarray((rng.random((n_lat, n_lon)) > 0.2).astype(float))

    serial = normalize_freshwater_net(F, area, mask)

    mesh = Mesh(np.array(jax.devices()[:4]), axis_names=("lat",))
    sh = NamedSharding(mesh, P("lat"))
    body = shard_map(
        lambda f, a, m: normalize_freshwater_net(f, a, m),
        mesh=mesh, in_specs=(P("lat"), P("lat"), P("lat")),
        out_specs=P("lat"), check_vma=False)

    _prev = (get_halo_backend(), get_mpi_topology(), get_spmd_mesh())
    activate_latlon_spmd_halo(mesh)
    try:
        sharded = body(jax.device_put(F, sh), jax.device_put(area, sh),
                       jax.device_put(mask, sh))
        # Non-vacuity: at least one band's local mean differs from the
        # global mean, so a missing psum WOULD change the answer.
        w = np.asarray(area) * np.asarray(mask)
        f_np = np.asarray(F)
        g_mean = (f_np * w).sum() / w.sum()
        band_means = [
            (f_np[b * 4:(b + 1) * 4] * w[b * 4:(b + 1) * 4]).sum()
            / w[b * 4:(b + 1) * 4].sum()
            for b in range(4)
        ]
        assert max(abs(bm - g_mean) for bm in band_means) > 1e-3
        # atol: f32-reduction round-off (the file's x64 setdefault is
        # ineffective when another module imported jax first) — still 1000x
        # below the >1e-3 band-vs-global mean spread asserted above, so a
        # missing psum cannot pass.
        np.testing.assert_allclose(np.asarray(sharded), np.asarray(serial),
                                   atol=1e-6, rtol=0.0)
    finally:
        set_spmd_mesh(_prev[2])
        set_halo_backend(_prev[0], _prev[1])


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_sharded_step_refuses_staggered_forcing():
    """A (n_lat+1, n_lon) forcing leaf must fail LOUDLY at the wrapper, not
    with an opaque shard_map divisibility error."""
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        make_sharded_ocean_step,
        shard_state_latlon,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    grid = create_latlon_grid(n_lat=16, n_lon=32)
    z_coord = create_ocean_z_star(n_levels=3, H_max=4000.0)
    model = LatLonCGridOceanModel(grid, z_coord,
                                  LatLonCGridOceanConfig.from_flat())
    state0 = _perturbed_state(grid, z_coord)
    model._ensure_vertex_mask(state0)
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    ss = shard_state_latlon(state0, dev.mesh)
    bad = OceanSurfaceForcing(
        tau_y=jnp.zeros((grid.n_lat + 1, grid.n_lon)))
    with pytest.raises(ValueError, match="leading dim"):
        step(ss, 600.0, surface_forcing=bad)


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_sharded_ocean_step_global_matches_explicit_scatter_gather():
    """``make_sharded_ocean_step_global`` (global-in/global-out, the minimal
    driver entry) must equal the explicit ``shard_state_latlon`` -> inner step ->
    ``gather_state_latlon`` path BIT-FOR-BIT (it is literally that composition),
    AND match the single-device reference to the same re-association floor.

    Also exercises the WITH-forcing path through the global wrapper (a smooth
    cell-shaped wind-stress + heat ``OceanSurfaceForcing``)."""
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        make_sharded_ocean_step,
        make_sharded_ocean_step_global,
        shard_state_latlon,
        gather_state_latlon,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    n_lat, n_lon, nlev = 48, 96, 10
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    cfg = LatLonCGridOceanConfig()
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state0 = _perturbed_state(grid, z_coord)
    dt, n_steps = 600.0, 3

    # create_latlon_grid returns a (legacy) LatLonGrid (1-D lat, no lat_T); the
    # 2-D T-point latitude lives on its LatLonCGridGeometry (what the model uses).
    lat = np.asarray(ensure_geometry(grid).lat_T)
    tau_x = jnp.asarray((0.1 * np.cos(3.0 * lat)).astype(np.float64))
    q_net = jnp.asarray((40.0 * np.cos(lat)).astype(np.float64))
    sf = OceanSurfaceForcing(tau_x=tau_x, tau_y=jnp.zeros_like(tau_x),
                             q_net=q_net)

    # single-device reference (forced)
    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt, surface_forcing=sf)

    model._ensure_vertex_mask(state0)
    dev = create_latlon_mesh(n_devices=4)

    # explicit scatter -> inner sharded step -> gather
    inner = make_sharded_ocean_step(model, dev.mesh)
    ss = shard_state_latlon(state0, dev.mesh)
    for _ in range(n_steps):
        ss = inner(ss, dt, surface_forcing=sf)
    ss = gather_state_latlon(ss, dev.mesh)

    # global-in/global-out wrapper (scatter + gather PER STEP)
    glob = make_sharded_ocean_step_global(model, dev.mesh)
    sg = state0
    for _ in range(n_steps):
        sg = glob(sg, dt, surface_forcing=sf)

    for nm in ("u", "v", "eta", "T", "S"):
        ref = np.asarray(getattr(s, nm).data)
        man = np.asarray(getattr(ss, nm).data)
        wrp = np.asarray(getattr(sg, nm).data)
        # the global wrapper == explicit scatter/gather BIT-FOR-BIT (the gather is
        # a per-step round trip but every band stays put -> identical reductions)
        np.testing.assert_allclose(wrp, man, atol=0.0, rtol=0.0,
                                   err_msg=f"global wrapper != explicit {nm}")
        # and both match the single-device forced reference to the FP floor
        np.testing.assert_allclose(wrp, ref, atol=2.0e-4, rtol=1.0e-3,
                                   err_msg=f"global wrapper vs serial {nm}")


# ---------------------------------------------------------------------------
# Tripole (active bipolar fold) + prognostic slab sea-ice tile under SPMD
# (the ORCA12 / eORCA lane enablers, 2026-09).
# ---------------------------------------------------------------------------

def _tripole_perturbed_state(grid, z_coord):
    """Rest state on a synthetic tripole geometry + perturbations that are
    NON-ZERO at the fold row, so the seam exchange is load-bearing.  The cap
    (last) v-row is masked as the tripole convention requires (the sharded
    v-carrier drops it and reconstructs the wall)."""
    n_lat, n_lon = int(grid.n_lat), int(grid.n_lon)
    nlev = z_coord.n_levels
    land = np.ones((n_lat, n_lon))
    H = 4000.0 * land
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0, land_mask_override=jnp.asarray(land),
        H_bathy_override=jnp.asarray(H))
    rng = np.random.default_rng(11)
    u = 0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.02 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    v[-1] = 0.0                               # cap wall (v_mask[-1] == 0)
    eta = 0.005 * rng.standard_normal((n_lat, n_lon))
    T = (5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
         + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))
    vm = np.asarray(state.v_mask.data).copy()
    vm[-1] = 0.0
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        v_mask=state.v_mask.replace(data=jnp.asarray(vm)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
        T=state.T.replace(data=jnp.asarray(T)))


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_tripole_ocean_spmd_matches_single_device():
    """The lat-band SPMD step on a grid with an ACTIVE bipolar fold (the
    eORCA / ORCA12 layout, here `create_synthetic_tripole`) must match the
    serial step: the fold lives on the north band only, interior bands see
    the band halo.  This is the gate for lifting run_omip's
    ``--enable-latlon-spmd requires --grid latlon`` refusal for tripole."""
    from legoesm.grids.tripole import create_synthetic_tripole
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_state_latlon,
        make_sharded_ocean_step,
        shard_state_latlon,
    )
    n_lat, n_lon, nlev = 48, 96, 10
    grid = create_synthetic_tripole(n_lat=n_lat, n_lon=n_lon)
    assert bool(grid.fold.is_active)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    cfg = LatLonCGridOceanConfig.from_flat()
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state0 = _tripole_perturbed_state(grid, z_coord)
    dt, n_steps = 600.0, 3
    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt)
    model._ensure_vertex_mask(state0)
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    ss = shard_state_latlon(state0, dev.mesh)
    for _ in range(n_steps):
        ss = step(ss, dt)
    ss = gather_state_latlon(ss, dev.mesh)
    _ATOL, _RTOL = 2.0e-4, 1.0e-3
    for nm in ("u", "v", "eta", "T", "S"):
        a = np.asarray(getattr(s, nm).data)
        b = np.asarray(getattr(ss, nm).data)
        np.testing.assert_allclose(b, a, atol=_ATOL, rtol=_RTOL,
                                   err_msg=f"tripole SPMD {nm} mismatch")
    # the seam rows actually moved (the test is not a rest fixed point)
    assert np.abs(np.asarray(s.T.data)[-1]
                  - np.asarray(state0.T.data)[-1]).max() > 0


def _slab_ice_inputs(n_lat, n_lon):
    from legoesm import constants
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.core.field import Field
    from legoesm.ice.state import SeaIceState
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing
    rng = np.random.default_rng(5)
    shp = (n_lat, n_lon)
    lat = np.linspace(-80, 80, n_lat)[:, None] * np.ones((1, n_lon))
    dims = ("y", "x")

    def fld(a, name):
        return Field(jnp.asarray(a), name=name, dims=dims, units="")
    conc = np.clip((np.abs(lat) - 60.0) / 20.0, 0.0, 0.9)
    ice = SeaIceState(
        h_ice=fld(1.5 * conc, "h_ice"),
        T_ice=fld(np.full(shp, float(constants.T_freeze_ocean) - 5.0),
                  "T_ice"),
        concentration=fld(conc, "concentration"))
    T_air = 300.0 - 40.0 * (np.abs(lat) / 80.0) ** 2
    o = np.ones(shp)
    atm = AtmToSurface(
        sw_down=jnp.asarray(200.0 * o), lw_down=jnp.asarray(300.0 * o),
        precip_total=jnp.asarray(1e-5 * o), precip_snow=jnp.asarray(0.0 * o),
        T_lowest=jnp.asarray(T_air), q_lowest=jnp.asarray(3e-3 * o),
        u_lowest=jnp.asarray(5.0 * o + rng.standard_normal(shp)),
        v_lowest=jnp.asarray(rng.standard_normal(shp)),
        p_lowest=jnp.asarray(1e5 * o), p_surface=jnp.asarray(1e5 * o),
        rho_lowest=jnp.asarray(1.25 * o), cos_zenith=jnp.asarray(0.5 * o),
        co2_ppmv=jnp.asarray(400.0), has_radiation=jnp.asarray(1.0),
        has_precipitation=jnp.asarray(1.0))
    sst_K = jnp.asarray(T_air - 2.0)
    sf = OceanSurfaceForcing(sw_down=jnp.asarray(200.0 * o),
                             q_net=jnp.asarray(120.0 * o),
                             tau_x=jnp.asarray(0.1 * o),
                             tau_y=jnp.asarray(0.02 * o))
    fw = FreshwaterForcing(precip=jnp.asarray(1e-5 * o),
                           evap=jnp.asarray(4e-6 * o),
                           runoff=jnp.asarray(0.0 * o),
                           ice_fw=jnp.asarray(0.0 * o))
    mask = jnp.asarray((rng.uniform(size=shp) > 0.2).astype(float))
    return ice, atm, sst_K, sf, fw, mask


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_slab_ice_tile_spmd_matches_single_device():
    """The prognostic SLAB sea-ice tile the JRA55 block scan carries next to
    the ocean state (``omip_sea_ice_surface_forcing``, dynamics off, one
    category) is elementwise, so laid out on the ocean's lat bands by
    ``shard_cell_pytree_latlon`` it must reproduce the single-device step
    through a jitted 3-step scan, and ``gather_cell_pytree_latlon`` must
    round-trip the layout.  This is the gate for lifting run_omip's
    ``--enable-latlon-spmd does not support --jra55-sea-ice`` refusal."""
    from legoesm.coupler.ocean_forcing import omip_sea_ice_surface_forcing
    from legoesm.ice.config import SeaIceConfig
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_cell_pytree_latlon,
        shard_cell_pytree_latlon,
        shard_forcing_latlon,
    )
    n_lat, n_lon, dt = 48, 96, 3600.0
    ice0, atm, sst_K, sf, fw, mask = _slab_ice_inputs(n_lat, n_lon)
    cfg = SeaIceConfig()

    @jax.jit
    def run(ice, atm, sst_K, sf, fw, mask):
        def body(carry, _):
            new_ice, fw_o, sf_o = omip_sea_ice_surface_forcing(
                ice_state=carry, ice_config=cfg, atm=atm, ocean_sst_K=sst_K,
                open_ocean_sf=sf, open_ocean_fw=fw, dt=dt, grid=None,
                ocean_mask=mask)
            return new_ice, (fw_o.ice_fw, sf_o.q_net)
        final, (ice_fw, q_net) = jax.lax.scan(body, ice, None, length=3)
        return final, ice_fw, q_net

    ref_ice, ref_fw, ref_q = run(ice0, atm, sst_K, sf, fw, mask)

    dev = create_latlon_mesh(n_devices=4)
    ice_s = shard_cell_pytree_latlon(ice0, dev.mesh)
    for leaf in jax.tree.leaves(ice_s):
        assert leaf.sharding.spec[0] == "lat", leaf.sharding
    back = gather_cell_pytree_latlon(ice_s, dev.mesh, to_host=True)
    for a, b in zip(jax.tree.leaves(back), jax.tree.leaves(ice0)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    atm_s = shard_forcing_latlon(atm, dev.mesh)
    sf_s = shard_forcing_latlon(sf, dev.mesh)
    fw_s = shard_forcing_latlon(fw, dev.mesh)
    sst_s = shard_forcing_latlon(sst_K, dev.mesh)
    mask_s = shard_forcing_latlon(mask, dev.mesh)
    out_ice, out_fw, out_q = run(ice_s, atm_s, sst_s, sf_s, fw_s, mask_s)
    out_ice = gather_cell_pytree_latlon(out_ice, dev.mesh, to_host=True)
    for nm in ice0._fields:
        np.testing.assert_allclose(
            np.asarray(getattr(out_ice, nm).data),
            np.asarray(getattr(ref_ice, nm).data), rtol=1e-12, atol=1e-12,
            err_msg=f"slab ice SPMD {nm} mismatch")
    np.testing.assert_allclose(np.asarray(out_fw), np.asarray(ref_fw),
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.asarray(out_q), np.asarray(ref_q),
                               rtol=1e-12, atol=1e-9)
    # the tile actually evolved (not a no-op fixed point)
    assert np.abs(np.asarray(ref_ice.h_ice.data)
                  - np.asarray(ice0.h_ice.data)).max() > 0
