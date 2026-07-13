"""AB2 time-centering of the barotropic slow forcing F_slow.

This is the faithful cure for the §5 eddy-resolving turbulent blow-up (the C-grid
barotropic Coriolis 2Δx null mode): with ``coriolis_scheme="explicit_ab2"`` the
planetary Coriolis reaches the barotropic mode via F_slow (no in-substep f·V_at_u
→ no null mode), and ``barotropic_slow_forcing_ab2=True`` AB2-extrapolates F_slow
(F_slow_eff = 3/2·F_slow^n − 1/2·F_slow^{n-1}) to match Oceananigans' AB2-extrapolated
Gᵁ, which keeps the barotropic geostrophic balance (without it the SSH drifts and
blows up). NO dissipation backstop.

Tests:
1. Flag OFF (default): the prev fields stay None (no behaviour change / pytree growth).
2. Flag ON: the prev fields become Fields (= the current F_slow) and the flag is LIVE
   (the state evolution differs from flag-off).
3. Flag ON threads through ``jax.lax.scan`` (pytree carry is stable).
4. Flag ON conserves volume to machine precision (adiabatic).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.silvestri_baroclinic_jet import (
    SilvestriJetConfig, build_silvestri_baroclinic_jet_setup)
from legoesm.ocean.vertical import compute_layer_thickness


@pytest.fixture(autouse=True)
def _fp64_precision_policy():
    """Pin the legoESM precision policy to fp64 for the duration of each test.

    JAX ``jax_enable_x64`` (the suite enables it) is NOT sufficient on its own:
    the legoESM precision policy is a separate process-global that still defaults
    to fp32, so the ocean state/Fields come back fp32 while the AB2 ``F_slow``
    prev carry (seeded here from fp64 zeros / x64-enabled construction) stays
    fp64 — ``jax.lax.scan`` then (correctly) rejects the fp64->fp32 carry
    narrowing.  A module-level ``set_policy`` runs ONCE at import and both (a)
    leaks fp64 into sibling test modules sharing the xdist worker and (b) is
    itself clobbered by any sibling that later sets fp32, so the policy seen by
    these tests is whatever ran last.  Pin fp64 per-test and RESTORE the prior
    policy on teardown so neither direction of cross-test contamination occurs.
    """
    _prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(_prev)


def _setup(ab2_fslow):
    r = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    # §5 wires the faithful stack (explicit_ab2 + ab2_fslow + alpha=0); flip
    # ONLY barotropic_slow_forcing_ab2 to isolate it.
    mc = r.model_config._replace(barotropic=r.model_config.barotropic._replace(barotropic_slow_forcing_ab2=ab2_fslow))
    m = LatLonCGridOceanModel(r.grid, r.z_coord, mc)

    def zf(d):
        return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                     dims=d.dims, units=d.units)
    s = r.initial_state._replace(
        T_incr_prev=zf(r.initial_state.T), S_incr_prev=zf(r.initial_state.S),
        u_incr_prev=zf(r.initial_state.u), v_incr_prev=zf(r.initial_state.v))
    if not ab2_fslow:
        # ensure the prev fields are None when the flag is off
        s = s._replace(F_slow_u_prev=None, F_slow_v_prev=None)
    return r, m, s


def test_flag_off_prev_stays_none():
    """Default (flag off): F_slow prev stays None — no pytree growth, no AB2."""
    r, m, s = _setup(False)
    s1 = m.step(s, 900.0)
    assert s1.F_slow_u_prev is None and s1.F_slow_v_prev is None


def test_flag_requires_explicit_ab2_coriolis():
    """Dispatch-hardening: flag on with matsuno_split Coriolis must raise (the
    AB2 would extrapolate a Coriolis-free F_slow while the null mode persists)."""
    r = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    bad = r.model_config._replace(
        barotropic=r.model_config.barotropic._replace(barotropic_slow_forcing_ab2=True),
        coriolis_scheme="matsuno_split")
    with pytest.raises(ValueError, match="explicit_ab2"):
        LatLonCGridOceanModel(r.grid, r.z_coord, bad)


def test_flag_requires_total_ab2_scope():
    """Dispatch-hardening: flag on with ab2_scope='advective' must raise (the
    stored prev omits the du_diss depth-mean -> time-inconsistent AB2)."""
    r = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    bad = r.model_config._replace(
        barotropic=r.model_config.barotropic._replace(barotropic_slow_forcing_ab2=True),
        ab2_scope="advective")
    with pytest.raises(ValueError, match="advective"):
        LatLonCGridOceanModel(r.grid, r.z_coord, bad)


def test_flag_on_unseeded_prev_raises_clear_error():
    """Flag on but prev unseeded (None) -> clear ValueError at trace time, not a
    cryptic lax.scan carry-structure error."""
    r, m, _ = _setup(True)
    s = r.initial_state._replace(F_slow_u_prev=None, F_slow_v_prev=None)
    with pytest.raises(ValueError, match="seeded"):
        m.step(s, 900.0)


def test_flag_on_stores_current_F_slow_and_is_live():
    """Flag on: prev becomes a Field; the flag CHANGES the evolution vs off."""
    # §5 default already wires the flag ON — verify build_silvestri seeded the prev.
    r_on = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    assert r_on.model_config.barotropic.barotropic_slow_forcing_ab2 is True
    assert r_on.initial_state.F_slow_u_prev is not None

    r, m_on, s_on = _setup(True)
    _, m_off, s_off = _setup(False)
    # one step: prev stored on the on-path
    s_on1 = m_on.step(s_on, 900.0)
    assert isinstance(s_on1.F_slow_u_prev, Field)
    # second step: AB2 extrapolation is active -> diverges from the off-path
    s_on2 = m_on.step(s_on1, 900.0)
    s_off1 = m_off.step(s_off, 900.0)
    s_off2 = m_off.step(s_off1, 900.0)
    d = float(jnp.max(jnp.abs(s_on2.u.data - s_off2.u.data)))
    assert d > 1e-12, "barotropic_slow_forcing_ab2 must change the evolution"


def test_flag_on_scan_pytree_stable():
    """Flag on threads through jax.lax.scan (carry pytree stable across steps)."""
    r, m, s = _setup(True)
    blk = jax.jit(lambda st, n: jax.lax.scan(
        lambda c, _: (m.step(c, 900.0), None), st, None, length=n)[0],
        static_argnames=("n",))
    s_out = blk(s, 5)  # would raise if the carry structure changed step-to-step
    assert bool(jnp.all(jnp.isfinite(s_out.u.data)))


def test_seed_scan_carry_seeds_prev_fields():
    """``seed_scan_carry`` must zero-seed ``F_slow_{u,v}_prev`` when the flag is
    on (so ANY scan driver — e.g. run_dino with the 'oceananigans' card — gets a
    stable carry without hand-seeding like the silvestri driver), and must leave
    them ``None`` when the flag is off (no pytree growth)."""
    r, m, _ = _setup(True)
    s = r.initial_state._replace(F_slow_u_prev=None, F_slow_v_prev=None)
    seeded = m.seed_scan_carry(s, 900.0)
    assert isinstance(seeded.F_slow_u_prev, Field)
    assert isinstance(seeded.F_slow_v_prev, Field)
    # Depth-mean forcing lives on the 2-D face grids: u (n_lat, n_lon+1),
    # v (n_lat+1, n_lon) — and the cold-start prev is exactly zero.
    assert seeded.F_slow_u_prev.data.shape == s.u.data.shape[:2]
    assert seeded.F_slow_v_prev.data.shape == s.v.data.shape[:2]
    assert float(jnp.max(jnp.abs(seeded.F_slow_u_prev.data))) == 0.0
    assert float(jnp.max(jnp.abs(seeded.F_slow_v_prev.data))) == 0.0
    # Idempotent: re-seeding an already-seeded carry keeps the Fields.
    reseeded = m.seed_scan_carry(seeded, 900.0)
    assert isinstance(reseeded.F_slow_u_prev, Field)
    # And the seeded carry actually steps (the unseeded one raises — covered by
    # test_flag_on_unseeded_prev_raises_clear_error).
    s1 = m.step(seeded, 900.0)
    assert bool(jnp.all(jnp.isfinite(s1.u.data)))
    # Flag off: seeding must NOT grow the pytree.
    _, m_off, s_off = _setup(False)
    seeded_off = m_off.seed_scan_carry(s_off, 900.0)
    assert seeded_off.F_slow_u_prev is None
    assert seeded_off.F_slow_v_prev is None


def test_flag_on_conserves_volume():
    """Adiabatic (no restoring): volume conserved to machine precision."""
    r, m, s = _setup(True)
    area = np.asarray(r.grid.area)

    def vol(st):
        h = np.asarray(compute_layer_thickness(
            st.eta.data, st.H_bathy.data, r.z_coord,
            min_water_column_m=r.model_config.min_water_column_m))
        msk = np.asarray(st.land_mask.data)[:, :, None]
        return float(np.sum(h * msk * area[:, :, None]))
    v0 = vol(s)
    blk = jax.jit(lambda st, n: jax.lax.scan(
        lambda c, _: (m.step(c, 900.0), None), st, None, length=n)[0],
        static_argnames=("n",))
    s = blk(s, 20)
    v1 = vol(s)
    assert abs(v1 - v0) / v0 < 1e-12, f"volume drift {(v1 - v0) / v0:.2e}"


def test_seed_scan_carry_partial_pair_seeded():
    """A PARTIAL prev carry (one Field, one None — e.g. a hand-built restart)
    must be completed as a pair: the missing component zero-seeded, the present
    one PRESERVED (codex: a u-only guard skipped this case and the step then
    raised / flipped None->Field mid-scan)."""
    r, m, _ = _setup(True)
    s0 = r.initial_state._replace(F_slow_u_prev=None, F_slow_v_prev=None)
    full = m.seed_scan_carry(s0, 900.0)
    marked = Field(data=full.F_slow_u_prev.data + 1.2345e-3,
                   name="F_slow_u_prev", dims=("lat", "lon_u"), units="m/s^2")
    partial = s0._replace(F_slow_u_prev=marked, F_slow_v_prev=None)
    seeded = m.seed_scan_carry(partial, 900.0)
    assert isinstance(seeded.F_slow_u_prev, Field)
    assert isinstance(seeded.F_slow_v_prev, Field)
    # present component preserved bit-for-bit, missing one zero-filled
    assert float(jnp.max(jnp.abs(seeded.F_slow_u_prev.data - marked.data))) == 0.0
    assert float(jnp.max(jnp.abs(seeded.F_slow_v_prev.data))) == 0.0
    # and it steps
    s1 = m.step(seeded, 900.0)
    assert bool(jnp.all(jnp.isfinite(s1.u.data)))


def test_canonical_factory_defaults_flag_for_fe_coriolis_combo():
    """The Oceananigans canonical-config factory must default
    barotropic_slow_forcing_ab2=True for the EFFECTIVE explicit_ab2 x
    implicit_cn x ab2 combo (the FE-Coriolis hazard), NOT set it when the deck
    overrides Coriolis to matsuno, and honor an explicit caller override
    without a duplicate-kwarg TypeError."""
    from legoesm.ocean.fidelity.oceananigans_recipe import (
        oceananigans_canonical_ocean_config,
    )
    eos = {"rho_ref": 1026.0, "alpha": 2e-4, "beta": 8e-4,
           "T_ref": 10.0, "S_ref": 35.0}
    cfg = oceananigans_canonical_ocean_config(eos_linear=eos)
    assert cfg.flat_get("barotropic_slow_forcing_ab2") is True
    cfg_m = oceananigans_canonical_ocean_config(
        eos_linear=eos, coriolis_scheme="matsuno_split")
    assert not cfg_m.flat_get("barotropic_slow_forcing_ab2")
    # explicit caller override wins + no duplicate-kwarg crash
    cfg_off = oceananigans_canonical_ocean_config(
        eos_linear=eos, barotropic_slow_forcing_ab2=False)
    assert not cfg_off.flat_get("barotropic_slow_forcing_ab2")
    # an outer_integrator override via **overrides must be part of the
    # EFFECTIVE combo (codex): forward_euler outer -> no AB2 default, and no
    # duplicate-kwarg TypeError for a bundle key arriving via **overrides.
    cfg_fe = oceananigans_canonical_ocean_config(
        eos_linear=eos, outer_integrator="forward_euler")
    assert not cfg_fe.flat_get("barotropic_slow_forcing_ab2")
