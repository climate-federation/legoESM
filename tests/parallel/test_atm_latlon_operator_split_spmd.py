"""Parity gate for the lat-band-SPMD operator-split atmosphere step.

``make_sharded_operator_split_step`` runs the SAME operator-split integration as
the serial ``_single_step`` (dynamics -> dry-mass fixer -> column-local physics
-> Euler write-back -> saturation/moisture-fix/smoothing/Rayleigh -> carry pack)
on a lat-band-sharded ``SegmentCarry``. Because the physics is PURELY
column-local it is decomposition-INVARIANT: the sharded (mesh=4dev) trajectory
must match the serial (mesh=None) one BITWISE in every physics-touched /
column-local field, and only to the FV-PPM cut bound in the dynamical fields
(u,v,p_s) — the same limited-FV-PPM boundary-order residual the dynamics-only
SPMD step already carries.

The physics is a column-local MOCK ``step_unified`` (small constant warming +
an AR1 tke evolution ``0.9*tke + 1e-3``): it exercises the full sharded
MECHANICS — band cell<->C-grid dynamics, the dry-mass fixer psum, the flattened
tke carry shard, the hyperdiffusion halo, the Euler write-back + accumulators —
without depending on a full physics harness (the real-scheme end-to-end parity
is the driver-level gate). Host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``), x64.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig)
import dataclasses

from legoesm.core.conservation import global_area_sum
from legoesm.core.operators_latlon_3d import hyperdiffusion_3d
from legoesm.driver.compiled_segments import (
    _SplitStepStatics, SegmentCarry, pack_carry, pack_forcing)
from legoesm.driver.physics_pipeline import PhysicsOutput
from legoesm.driver.sharded_operator_split_step import (
    make_sharded_operator_split_step, shard_operator_split_carry,
    shard_operator_split_forcing)

N_DEV = 4
N_LAT = 16
N_LON = 16
NLEV = 8
DT = 100.0


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    return jax.sharding.Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))


def _mock_step_unified_stateful(
    need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic, lat, lon,
    day_of_year, seconds_of_day, dt, solar_weights, s_0, o3_vmr, aerosol_od,
    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa, **kwargs,
):
    """Column-local mock physics: constant warming + slight drying + precip,
    and an AR1 tke evolution so the prognostic carry is genuinely exercised.
    Every output depends ONLY on this column -> decomposition-invariant."""
    s3, s2 = T.shape, p_s.shape
    phys_out = PhysicsOutput(
        dT_dt=jnp.full(s3, 1e-5), dq_v_dt=jnp.full(s3, -1e-9),
        dq_c_dt=jnp.zeros(s3), dq_r_dt=jnp.zeros(s3),
        precip=jnp.full(s2, 1e-6),
        sw_net_sfc=jnp.zeros(s2), lw_net_sfc=jnp.zeros(s2),
        sw_up_toa=jnp.zeros(s2), lw_up_toa=jnp.zeros(s2),
        sw_down_toa=jnp.zeros(s2),
        du_dt=jnp.zeros(s3), dv_dt=jnp.zeros(s3),
        dq_i_dt=jnp.zeros(s3), dq_s_dt=jnp.zeros(s3), dq_g_dt=jnp.zeros(s3),
        dN_c_dt=jnp.zeros(s3), dN_r_dt=jnp.zeros(s3), dN_i_dt=jnp.zeros(s3),
        conv_prog=conv_prog,
    )
    held_new = (held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                held_sw_up_toa, held_lw_up_toa, held_sw_down_toa)
    tke = kwargs.get("tke")
    if tke is not None:
        phys_out = phys_out._replace(tke=(0.9 * tke + 1e-3).astype(tke.dtype))
    return phys_out, held_new, kwargs.get("T_land")


def _build_harness(use_polar_filter=False):
    """Real lat-lon C-grid model (dynamics-only: config fix_mass=False, the
    operator-split mass fixer is separate) + a column-local mock physics carry.
    ``use_polar_filter`` exercises the band-sliced polar-mask path (else the
    sharded band would filter with the global-length mask)."""
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
                              omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=False, use_polar_filter=use_polar_filter, use_ppm_transport=True,
        time_integrator="ssp_rk3")
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)

    rng = np.random.default_rng(2027)
    eps = 1e-3
    lat2d = np.asarray(grid.lat)
    if lat2d.ndim == 1:
        lat2d = lat2d[:, None] * np.ones((1, N_LON))
    u = jnp.asarray(eps * rng.standard_normal((N_LAT, N_LON, NLEV)))
    v0 = eps * rng.standard_normal((N_LAT, N_LON, NLEV))
    v = jnp.asarray(v0)
    T = jnp.asarray(280.0 + 20.0 * np.cos(lat2d)[..., None]
                    + eps * rng.standard_normal((N_LAT, N_LON, NLEV)))
    p_s = jnp.asarray(1.0e5 + 500.0 * np.cos(lat2d)
                      + 10.0 * rng.standard_normal((N_LAT, N_LON)))
    phis = jnp.zeros((N_LAT, N_LON))
    # Spatially varying moisture so the fixers' global sums are non-trivial.
    q_v = jnp.asarray(0.01 + 0.004 * np.cos(lat2d)[..., None]
                      * np.ones((1, 1, NLEV)))
    q_c = jnp.zeros((N_LAT, N_LON, NLEV))
    q_r = jnp.zeros((N_LAT, N_LON, NLEV))

    state = HydrostaticState(
        u=Field(u, name="u", dims=("lat", "lon", "level"), units="m/s"),
        v=Field(v, name="v", dims=("lat", "lon", "level"), units="m/s"),
        T=Field(T, name="T", dims=("lat", "lon", "level"), units="K"),
        p_s=Field(p_s, name="p_s", dims=("lat", "lon"), units="Pa"),
        phis=Field(phis, name="phis", dims=("lat", "lon"), units="m2/s2"),
    )
    ncol = N_LAT * N_LON
    tke0 = jnp.full((ncol, NLEV), 1e-4)
    target_mass = global_area_sum(p_s, grid)
    z2 = jnp.zeros((N_LAT, N_LON))
    z3 = jnp.zeros((N_LAT, N_LON, NLEV))
    carry = pack_carry(
        state, q_v, q_c, q_r, conv_prog=None,
        held_dT_rad=z3, held_sw_net_sfc=z2, held_lw_net_sfc=z2,
        held_sw_up_toa=z2, held_lw_up_toa=z2, held_sw_down_toa=z2,
        step_index=jnp.int32(0),
        target_moisture=jnp.asarray(0.0), target_mass=target_mass,
        tke=tke0,
    )
    forcing = pack_forcing(
        sst=jnp.full((N_LAT, N_LON), 300.0), sic=jnp.zeros((N_LAT, N_LON)),
        day_of_year=1.0, seconds_of_day=0.0,
        solar_weights=jnp.ones(14), s_0=constants.S_0,
        o3_vmr=jnp.zeros((N_LAT, N_LON, NLEV)),
        aerosol_od=jnp.zeros((N_LAT, N_LON)),
    )
    statics = _SplitStepStatics(
        step_unified=_mock_step_unified_stateful, forcing=forcing,
        lat=grid.lat, lon=grid.lon, dt=DT,
        tau_equator=None, tau_pole=None, sbm_tau_c=None, sbm_RH_ref=None,
        C_H=None, C_E=None, albedo_ice=None, albedo_ocean=None,
        ghg_vmr_override=None, hs_newtonian_relax=None,
        energy_consistent_moisture_clip=False,
        do_sat_adjust=True, fix_moisture=False,
        sigma_full=sigma.sigma_full, dsigma=sigma.dsigma, grid=grid,
        owned_mask=None, qv_smooth_coeff=0.02,
        fric_decay=jnp.ones((NLEV,)), hyperdiffusion_3d=hyperdiffusion_3d,
    )
    return model, statics, carry, forcing


@pytest.mark.parametrize("use_polar_filter", [False, True])
def test_operator_split_spmd_matches_serial(use_polar_filter):
    mesh = _mesh()
    model, statics, carry, forcing = _build_harness(use_polar_filter)
    n_steps = 3

    step_none = make_sharded_operator_split_step(
        model, None, statics, fix_mass=True, rad_update_steps=1, start_day=0.0)
    step_dev = make_sharded_operator_split_step(
        model, mesh, statics, fix_mass=True, rad_update_steps=1, start_day=0.0)

    # Serial reference (mesh=None): full-grid operator-split step.
    c_s = carry
    for _ in range(n_steps):
        c_s = step_none(c_s, forcing)

    # Sharded (mesh=4dev): shard the carry + forcing, run, gather.
    c_b = shard_operator_split_carry(carry, mesh)
    f_b = shard_operator_split_forcing(forcing, mesh)
    # Non-vacuity: the carry is ACTUALLY lat-PARTITIONED (not merely replicated
    # across the mesh — num_devices==N_DEV holds for a replicated P() too). The
    # spec must carry "lat" so each device owns a distinct band.
    assert "lat" in c_b.T.sharding.spec, \
        f"carry T not lat-partitioned (spec={c_b.T.sharding.spec})"
    assert "lat" in c_b.tke.sharding.spec, \
        f"flattened tke not lat-partitioned (spec={c_b.tke.sharding.spec})"
    for _ in range(n_steps):
        c_b = step_dev(c_b, f_b)
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    c_b = jax.tree.map(lambda x: np.asarray(jax.device_put(x, rep)), c_b)

    # Non-vacuity: tke genuinely evolved off its 1e-4 floor (AR1 -> ~0.0037).
    tke_grew = float(np.max(np.abs(np.asarray(c_s.tke) - 1e-4)))
    assert tke_grew > 1e-3, f"tke did not evolve ({tke_grew:.2e}) — vacuous"

    # Truly column-local fields — physics-only, NOT dynamically advected (the
    # C-grid dynamics is dry: moisture + the physics carries ride physics
    # alone). Decomposition-INVARIANT -> essentially bitwise (fp64 roundoff).
    for f in ("q_v", "q_c", "q_r", "tke", "precip_accum", "held_dT_rad",
              "shflx_accum", "conv_prog"):
        a = np.asarray(getattr(c_s, f))
        b = np.asarray(getattr(c_b, f))
        assert a.shape == b.shape, f"{f}: shape {a.shape} vs {b.shape}"
        worst = float(np.max(np.abs(a - b)))
        assert worst < 1e-11, \
            f"column-local field {f}: sharded vs serial {worst:.3e} (not bitwise)"

    # Dynamically-advected fields: FV-PPM cut-truncation bound (the residual is
    # confined to cut-adjacent rows and scales with the field's meridional
    # gradient — the 20 K T gradient gives ~1e-5 abs T, ~3e-8 abs winds, all far
    # below the ~1e-3 that a REAL decomposition bug produces). T / p_s keep a
    # TIGHT rtol (1e-6) so a real error can't hide; the small-magnitude winds
    # (~1e-2, near zero at points) are bounded by atol.
    _dyn_tol = {"u": (1e-3, 1e-6), "v": (1e-3, 1e-6),
                "T": (1e-6, 1e-6), "p_s": (1e-6, 1e-6)}
    for f, (rt, at) in _dyn_tol.items():
        a = np.asarray(getattr(c_s, f))
        b = np.asarray(getattr(c_b, f))
        assert a.shape == b.shape, f"{f}: shape {a.shape} vs {b.shape}"
        np.testing.assert_allclose(
            b, a, rtol=rt, atol=at,
            err_msg=f"dynamical {f}: sharded diverged beyond the FV-PPM bound")

    # Non-vacuity: the band decomposition GENUINELY ran (T cut truncation is
    # present -> sharding did not collapse to single-device).
    t_diff = float(np.max(np.abs(np.asarray(c_s.T) - np.asarray(c_b.T))))
    assert t_diff > 1e-12, (
        f"T bit-identical to serial ({t_diff:.2e}) — the band decomposition did "
        f"not actually run; the gate would be vacuous.")


def _mock_step_unified_rad_marking(need_rad, *args, **kwargs):
    """``_mock_step_unified_stateful`` that MARKS each radiation step.

    On a radiation step every held field is bumped by 1; on a held step every
    held field is returned untouched.  ``need_rad`` is a TRACED boolean, so the
    selection is ``jnp.where`` (never a Python ``if``).  After a run each held
    field therefore equals the number of steps that took the radiation branch,
    and comparing consecutive steps says exactly WHICH steps refreshed.

    Marking ALL SIX held fields, not just the surface LW, is deliberate: a
    cadence defect confined to the heating profile or the TOA fluxes would be
    invisible in a surface-only check.
    """
    phys_out, held_new, T_land = _mock_step_unified_stateful(
        need_rad, *args, **kwargs)
    bump = jnp.where(need_rad, 1.0, 0.0)
    return (phys_out,
            tuple(h + bump.astype(h.dtype) for h in held_new),
            T_land)


@pytest.mark.parametrize("rad_update_steps", [1, 2, 3])
def test_sharded_step_radiates_on_exactly_the_cadence_steps(rad_update_steps):
    """EVENT-LEVEL cadence gate: which steps refresh radiation, not just what
    the fluxes look like at the end.

    The driver-level parity test compares final ``held_lw_net_sfc`` against the
    serial lane, which is an integrated, indirect signal -- a wrong phase, a
    wrong refresh COUNT, or a defect confined to the heating profile can all hide
    inside a small final-flux difference (codex adversarial review).  This drives
    ``make_sharded_operator_split_step`` one step at a time and asserts the
    refresh set is EXACTLY ``{i : (i+1) % rad_update_steps == 0}``, that all six
    held fields are byte-unchanged between refreshes, and that they keep their
    lat-band sharding across both a refresh and a hold step.
    """
    mesh = _mesh()
    model, statics, carry, forcing = _build_harness()
    statics = dataclasses.replace(
        statics, step_unified=_mock_step_unified_rad_marking)
    step = make_sharded_operator_split_step(
        model, mesh, statics, fix_mass=True,
        rad_update_steps=rad_update_steps, start_day=0.0)

    HELD = ("held_dT_rad", "held_sw_net_sfc", "held_lw_net_sfc",
            "held_sw_up_toa", "held_lw_up_toa", "held_sw_down_toa")
    c = shard_operator_split_carry(carry, mesh)
    assert "lat" in c.T.sharding.spec, "carry not lat-partitioned — vacuous"
    f = shard_operator_split_forcing(forcing, mesh)

    n_steps = 7
    refreshed, prev = [], {k: 0.0 for k in HELD}
    for i in range(n_steps):
        c = step(c, f)
        for name in HELD:
            leaf = getattr(c, name)
            # Sharding must survive BOTH branches (codex: no existing test
            # asserts held_* sharding after a conditional step).
            assert "lat" in leaf.sharding.spec, (
                f"{name} lost its lat-band sharding at step {i} "
                f"(spec={leaf.sharding.spec})")
        now = {k: float(np.asarray(jax.device_put(
            getattr(c, k), jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()))).max()) for k in HELD}
        # Every held field must move together, or not at all.
        moved = {k for k in HELD if now[k] > prev[k] + 0.5}
        assert moved in (set(), set(HELD)), (
            f"step {i}: held fields disagree on whether radiation ran "
            f"(moved={sorted(moved)}) — a partial refresh")
        if moved:
            refreshed.append(i)
        else:
            for k in HELD:
                assert now[k] == pytest.approx(prev[k]), (
                    f"step {i}: {k} changed on a HELD step ({prev[k]} -> "
                    f"{now[k]}); held radiation must be bit-stable between "
                    f"refreshes")
        prev = now

    expected = [i for i in range(n_steps) if (i + 1) % rad_update_steps == 0]
    assert refreshed == expected, (
        f"radiation refreshed on steps {refreshed} at "
        f"rad_update_steps={rad_update_steps}; the cadence "
        f"(i+1)%{rad_update_steps}==0 asks for {expected}. "
        f"{'Every step -> the need_rad predicate is being discarded.' if refreshed == list(range(n_steps)) else 'Phase error.'}")
