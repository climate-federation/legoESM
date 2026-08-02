"""Direct unit test for the RUN restart pair ``ocean.restart.save_run_restart``
/ ``load_run_restart`` (+ ``run_restart_metadata``).

This is the checkpoint an OMIP driver resumes from, so its contract is
stricter than the older ``save_restart``: every PROGNOSTIC and STATIC slot is
persisted (see ``_SLOT_POLICY``), including the carries that are NOT ``Field``s
and that ``save_restart`` therefore drops silently — the rigid-lid
streamfunction quintet (``psi``/``dpsi``/``dpsi_prev``/``dpsin``/``dpsin_prev``,
raw arrays) and the NEMO AB3/AM4 barotropic history ``bt_hist`` (a tuple of
arrays) — plus the UNMANGLED 12-field ``DynamicSeaIceState`` and the absolute
step counter (every OMIP forcing index is a pure function of it).  DIAGNOSTIC
slots are deliberately excluded, and a slot in neither bucket raises.

The tests here are deliberately mechanical ratchets:

* ``test_every_state_slot_is_classified`` fails when a NEW slot is added to
  ``LatLonCGridOceanState`` without deciding how the restart serialises it —
  the "a carry cannot be silently missed" guarantee;
* ``test_seed_scan_carry_slots_round_trip`` enumerates the carries from
  ``LatLonCGridOceanModel.seed_scan_carry`` itself (the authoritative gate
  list) rather than from a hand-copied list, so a new gate is covered the day
  it lands;
* ``test_save_raises_on_unserialisable_slot`` proves the format fails LOUD
  instead of dropping a carry it does not understand.
"""
from __future__ import annotations

import json
import os
from typing import NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ice.state import DynamicSeaIceState, init_dynamic_ice_state
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.restart import (
    load_run_restart,
    run_restart_metadata,
    save_run_restart,
)
from legoesm.ocean.state import LatLonCGridOceanState
from legoesm.ocean.vertical import create_ocean_z_star


N_LAT, N_LON, NLEV = 6, 8, 4
H_MAX = 1000.0


def _base_state():
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    z_coord = create_ocean_z_star(n_levels=NLEV, H_max=H_MAX)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=12.0, T_deep=2.0, S_uniform=35.0,
        H_max=H_MAX,
    )
    return grid, z_coord, state


def _ice_state(n_categories=1):
    shape = (N_LAT, N_LON)
    ice = init_dynamic_ice_state(shape, n_categories=n_categories)
    # Populate EVERY slot with a distinct, non-default pattern so a dropped or
    # aliased field cannot pass by coincidence.
    rng = np.random.default_rng(1234)
    upd = {}
    for i, name in enumerate(DynamicSeaIceState._fields):
        ref = getattr(ice, name)
        upd[name] = ref.replace(
            data=jnp.asarray(rng.random(ref.data.shape) + float(i + 1),
                             dtype=ref.data.dtype))
    return ice._replace(**upd)


# ---------------------------------------------------------------------------
# Slot-kind classification: the "no carry can be silently missed" ratchet
# ---------------------------------------------------------------------------

# How each LatLonCGridOceanState slot is expected to serialise.  A slot added
# to the state without an entry here makes the test below RED, which is the
# point: the author has to decide (and cover) the new carry's persistence.
_SLOT_KIND = {
    # always-present prognostics + static geometry
    "u": "field", "v": "field", "T": "field", "S": "field", "eta": "field",
    "H_bathy": "field", "land_mask": "field", "u_mask": "field",
    "v_mask": "field", "w": "field",
    # optional Field carries
    "T_som": "field", "S_som": "field",
    "T_flux_div_prev": "field", "S_flux_div_prev": "field",
    "eke": "field", "tke": "field", "dtke": "field", "eke_diss": "field",
    "T_incr_prev": "field", "S_incr_prev": "field",
    "u_incr_prev": "field", "v_incr_prev": "field",
    "u_before": "field", "v_before": "field", "T_before": "field",
    "S_before": "field", "eta_before": "field",
    "F_slow_u_prev": "field", "F_slow_v_prev": "field",
    "tau_x_prev": "field", "tau_y_prev": "field",
    "freshwater_eta_prev": "field",
    # NOT Fields: raw arrays / tuple-of-arrays that save_restart drops silently
    "psi": "array", "dpsi": "array", "dpsi_prev": "array",
    "dpsin": "array", "dpsin_prev": "array",
    "bt_hist": "tuple",
}


def test_every_state_slot_is_classified():
    """A new LatLonCGridOceanState slot must be classified here (and hence
    covered by the round-trip below) — otherwise a future carry silently
    cold-starts on resume."""
    missing = sorted(set(LatLonCGridOceanState._fields) - set(_SLOT_KIND))
    assert not missing, (
        f"LatLonCGridOceanState slot(s) {missing} are not classified for the "
        "run restart.  Decide how each persists (Field / raw array / tuple), "
        "add it to _SLOT_KIND, and confirm the round-trip covers it.")
    stale = sorted(set(_SLOT_KIND) - set(LatLonCGridOceanState._fields))
    assert not stale, f"_SLOT_KIND lists removed slot(s) {stale}"


# --- the CLOSED prognostic / static / diagnostic partition ------------------

def test_persistence_policy_covers_every_supported_state_class():
    """``_SLOT_POLICY`` must classify every slot of every state class the
    restart can be handed.  This is the ratchet that stops a newly added field
    (the #1442 ``mass_flux_u``/``mass_flux_v`` case) from falling into neither
    bucket and being silently dropped or silently resurrected."""
    from legoesm.core.state import MPASOceanState
    from legoesm.ice.state import SeaIceState
    from legoesm.ocean.restart import _SLOT_POLICY
    from legoesm.ocean.state import OceanState

    for cls in (OceanState, LatLonCGridOceanState, MPASOceanState,
                SeaIceState, DynamicSeaIceState):
        missing = sorted(set(cls._fields) - set(_SLOT_POLICY))
        assert not missing, (
            f"{cls.__name__} slot(s) {missing} have no restart persistence "
            "policy.  Classify each in legoesm.ocean.restart._SLOT_POLICY.")


def test_unclassified_slot_raises_at_save_and_at_validate():
    """A slot outside the policy fails LOUD — at setup via
    validate_restart_policy, and again at save."""
    from legoesm.ocean.restart import (
        classify_restart_slots, validate_restart_policy,
    )

    class _NewState(NamedTuple):
        T: object = None
        brand_new_carry: object = None

    st = _NewState(T=Field(data=jnp.zeros((2, 2)), name="T"),
                   brand_new_carry=jnp.zeros((2, 2)))
    for fn in (classify_restart_slots, validate_restart_policy):
        with pytest.raises(KeyError, match="brand_new_carry"):
            fn(st)
    with pytest.raises(KeyError, match="brand_new_carry"):
        save_run_restart("/dev/null", st, step=0, time_days=0.0,
                         grid_type="latlon")


def test_diagnostic_slots_are_excluded_and_not_restored(tmp_path):
    """``w`` is recomputed from the prognostic state every step, so it is
    DELIBERATELY not persisted: it must be absent from the archive, listed in
    the archive's ``_excluded`` record, and left at the TEMPLATE value on load.

    This is the run_omip asymmetry (codex #1442) made impossible: there a
    diagnostic field was written but silently dropped on load because the
    fresh template slot was ``None``."""
    from legoesm.ocean.restart import _SLOT_DIAGNOSTIC, _SLOT_POLICY

    assert _SLOT_POLICY["w"] == _SLOT_DIAGNOSTIC
    _, _, state = _base_state()
    marked = state._replace(
        w=state.w.replace(data=jnp.full_like(state.w.data, 7.0)))
    path = tmp_path / "r.npz"
    save_run_restart(path, marked, step=3, time_days=1.0, grid_type="latlon")

    with np.load(path, allow_pickle=False) as f:
        assert "w" not in f.files, "a diagnostic slot was persisted"
        assert "w" in json.loads(str(f["_excluded"]))
    meta = run_restart_metadata(path)
    assert "w" not in meta["slots"]

    fresh = _base_state()[2]
    got, _, _ = load_run_restart(path, fresh, grid_type="latlon")
    # Template value kept (NOT the 7.0 from the writer, and NOT dropped to None
    # — the slot must stay a Field so the step's `state.w.replace(...)` works).
    assert isinstance(got.w, Field)
    np.testing.assert_array_equal(np.asarray(got.w.data),
                                  np.asarray(fresh.w.data))
    assert float(np.max(np.asarray(got.w.data))) != 7.0


def test_load_refuses_an_archive_that_persisted_a_diagnostic(tmp_path):
    """Defence against an archive written by a different policy: restoring a
    stale diagnostic would resurrect a wrong value and, on the scan path, flip
    an optional slot None->Field and break the carry treedef."""
    _, _, state = _base_state()
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")
    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files}
    slots = json.loads(str(payload["_slot_kinds"]))
    slots["w"] = {"kind": "field", "meta": ["w", ["lat", "lon", "level"],
                                            "m/s", "", "cell"]}
    payload["_slot_kinds"] = np.asarray(json.dumps(slots))
    payload["w"] = np.asarray(state.w.data)
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="diagnostic"):
        load_run_restart(path, _base_state()[2], grid_type="latlon")


def _fill_all_slots(state):
    """Populate EVERY optional slot with a distinct pattern of its expected
    kind, so the round-trip exercises all three serialisation paths."""
    rng = np.random.default_rng(7)
    upd = {}
    for i, name in enumerate(LatLonCGridOceanState._fields):
        if getattr(state, name) is not None:
            continue
        kind = _SLOT_KIND[name]
        if kind == "field":
            upd[name] = Field(
                data=jnp.asarray(rng.random((N_LAT, N_LON, NLEV)) + i),
                name=name, dims=("lat", "lon", "level"), units="1")
        elif kind == "array":
            upd[name] = jnp.asarray(rng.random((N_LAT + 1, N_LON + 1)) + i)
        elif kind == "tuple":
            upd[name] = tuple(
                jnp.asarray(rng.random((N_LAT, N_LON)) + i + k)
                for k in range(6))
        else:  # pragma: no cover - _SLOT_KIND is closed by the test above
            raise AssertionError(f"unhandled kind {kind!r}")
    return state._replace(**upd)


def _assert_slot_equal(name, got, want):
    if isinstance(want, Field):
        assert isinstance(got, Field), f"{name}: Field -> {type(got).__name__}"
        np.testing.assert_array_equal(np.asarray(got.data),
                                      np.asarray(want.data),
                                      err_msg=f"slot {name} changed")
        assert got.data.dtype == want.data.dtype, f"slot {name} dtype changed"
        # ALL Field metadata is pytree AUX DATA (Field.tree_flatten), i.e. part
        # of the treedef: a restored carry whose name/dims/units/long_name/
        # staggering drifted breaks lax.scan's equal-treedef contract and any
        # tree_map against a step output.  The leapfrog carry stores
        # ``u_before`` under the name ``'u'`` precisely for this reason.
        assert (got.name, got.dims, got.units, got.long_name,
                got.staggering) == (want.name, want.dims, want.units,
                                    want.long_name, want.staggering), (
            f"slot {name}: Field aux-data (treedef) not preserved")
    elif isinstance(want, tuple):
        assert isinstance(got, tuple) and len(got) == len(want), (
            f"{name}: tuple[{len(want)}] -> {got!r}")
        for k, (g, w) in enumerate(zip(got, want)):
            np.testing.assert_array_equal(np.asarray(g), np.asarray(w),
                                          err_msg=f"slot {name}[{k}]")
    else:
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want),
                                      err_msg=f"slot {name} changed")


def test_all_slots_round_trip_bit_exact(tmp_path):
    """Every slot — Field, raw array AND tuple — survives save -> load
    unchanged, and the step counter comes back."""
    _, _, state = _base_state()
    state = _fill_all_slots(state)
    ice = _ice_state()
    path = tmp_path / "restart.npz"

    save_run_restart(path, state, step=4321, time_days=17.5,
                     grid_type="latlon", ice_state=ice)
    got, got_ice, meta = load_run_restart(path, _base_state()[2],
                                          ice_template=_ice_state(),
                                          grid_type="latlon")

    # Strongest single check: the restored carry has the SAME pytree treedef
    # (Field aux-data included), so it can be fed straight back into a scan.
    assert (jax.tree_util.tree_structure(got)
            == jax.tree_util.tree_structure(state))
    assert (jax.tree_util.tree_structure(got_ice)
            == jax.tree_util.tree_structure(ice))
    assert meta["step"] == 4321
    assert meta["time_days"] == pytest.approx(17.5)
    assert meta["grid_type"] == "latlon"
    from legoesm.ocean.restart import _SLOT_DIAGNOSTIC, _SLOT_POLICY
    for name in LatLonCGridOceanState._fields:
        if _SLOT_POLICY[name] == _SLOT_DIAGNOSTIC:
            continue      # excluded by policy — covered by its own test below
        _assert_slot_equal(name, getattr(got, name), getattr(state, name))
    # All 12 sea-ice slots, unmangled (no land-masking, no category aggregation).
    assert len(DynamicSeaIceState._fields) == 12
    for name in DynamicSeaIceState._fields:
        _assert_slot_equal(f"ice_{name}", getattr(got_ice, name),
                           getattr(ice, name))


def test_non_field_carries_are_dropped_by_the_OLD_save_restart(tmp_path):
    """Motivation guard: the pre-existing ``save_restart`` silently loses the
    raw-array / tuple carries, which is exactly why the run restart exists.
    If ``save_restart`` ever learns to keep them this test goes red and the
    module docstring must be corrected."""
    from legoesm.ocean.restart import load_restart, save_restart

    _, _, state = _base_state()
    state = _fill_all_slots(state)
    save_restart(state, tmp_path / "old.npz")
    back = load_restart(tmp_path / "old.npz", _base_state()[2])
    assert back.psi is None and back.bt_hist is None


def test_save_raises_on_unserialisable_slot(tmp_path):
    """A carry whose container the format does not understand FAILS LOUD."""
    _, _, state = _base_state()

    class _Opaque:
        pass

    state = state._replace(eke=_Opaque())
    with pytest.raises(TypeError, match=r"unsupported type"):
        save_run_restart(tmp_path / "x.npz", state, step=1, time_days=0.0,
                         grid_type="latlon")


def test_grid_type_mismatch_is_a_hard_error(tmp_path):
    _, _, state = _base_state()
    save_run_restart(tmp_path / "r.npz", state, step=1, time_days=0.0,
                     grid_type="latlon")
    with pytest.raises(ValueError, match="does not match the run's grid_type"):
        load_run_restart(tmp_path / "r.npz", state, grid_type="tripole")


def test_ice_presence_mismatch_is_a_hard_error(tmp_path):
    """Ice in the archive with no template (and the reverse) must raise: a
    silently cold-started pack rebuilds at the seed T_ice/S_ice and dumps a
    large spurious surface flux on step 1."""
    _, _, state = _base_state()
    ice = _ice_state()
    save_run_restart(tmp_path / "with_ice.npz", state, step=1, time_days=0.0,
                     grid_type="latlon", ice_state=ice)
    with pytest.raises(ValueError, match="no ice_template was supplied"):
        load_run_restart(tmp_path / "with_ice.npz", state, grid_type="latlon")

    save_run_restart(tmp_path / "no_ice.npz", state, step=1, time_days=0.0,
                     grid_type="latlon")
    with pytest.raises(ValueError, match="holds no ice state"):
        load_run_restart(tmp_path / "no_ice.npz", state, ice_template=ice,
                         grid_type="latlon")


def test_truncated_archive_is_a_hard_error(tmp_path):
    """A manifest entry with no matching array must raise, never resume with
    that carry silently cold-started."""
    _, _, state = _base_state()
    state = _fill_all_slots(state)
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=2, time_days=1.0, grid_type="latlon")
    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files if k != "tke"}   # drop one carry
    np.savez(path, **payload)
    with pytest.raises(ValueError, match=r"missing|truncated"):
        load_run_restart(path, _base_state()[2], grid_type="latlon")


def test_static_geometry_mismatch_is_a_hard_error(tmp_path):
    """A resume whose bathymetry/mask differs from the writing leg is a
    confound, not a continuation."""
    _, _, state = _base_state()
    save_run_restart(tmp_path / "r.npz", state, step=1, time_days=0.0,
                     grid_type="latlon")
    _, _, other = _base_state()
    # FLIP the value (not "set 0"), so the test cannot pass vacuously if the
    # rest-state mask happens to already be 0 at that cell.
    other = other._replace(
        land_mask=other.land_mask.replace(
            data=other.land_mask.data.at[0, 0].set(
                1.0 - other.land_mask.data[0, 0])))
    with pytest.raises(ValueError, match="static geometry slot"):
        load_run_restart(tmp_path / "r.npz", other, grid_type="latlon")


def test_shape_mismatch_is_a_hard_error(tmp_path):
    """Resuming at a different resolution / ITD category count must raise, not
    silently _replace a wrong-shaped array into the state."""
    _, _, state = _base_state()
    ice1 = _ice_state(n_categories=1)
    save_run_restart(tmp_path / "r.npz", state, step=1, time_days=0.0,
                     grid_type="latlon", ice_state=ice1)
    ice5 = init_dynamic_ice_state((N_LAT, N_LON, 5), n_categories=5)
    with pytest.raises(ValueError, match="different resolution"):
        load_run_restart(tmp_path / "r.npz", state, ice_template=ice5,
                         grid_type="latlon")


def test_metadata_rejects_a_non_run_restart(tmp_path):
    from legoesm.ocean.restart import save_restart

    _, _, state = _base_state()
    save_restart(state, tmp_path / "legacy.npz")
    with pytest.raises(ValueError, match="not a run restart"):
        run_restart_metadata(tmp_path / "legacy.npz")


# ---------------------------------------------------------------------------
# The authoritative carry enumeration: seed_scan_carry itself
# ---------------------------------------------------------------------------

# Each entry is a MINIMAL valid config that opens a different seed_scan_carry
# gate.  ab2 outer and ab2 tracer cannot be combined (the model rejects the
# double-count), and leapfrog requires the explicit-AB2 Coriolis — so these are
# three separate cases, not one.
@pytest.mark.parametrize("gates", [
    {"tracer_time_integrator": "ab2"},        # -> T_flux_div_prev/S_flux_div_prev
    {"outer_integrator": "ab2"},              # -> {T,S,u,v}_incr_prev
    {"outer_integrator": "leapfrog",          # -> {u,v,T,S,eta}_before
     "coriolis_scheme": "explicit_ab2"},
])
def test_seed_scan_carry_slots_round_trip(tmp_path, gates):
    """Enumerate the carries from ``seed_scan_carry`` (the gate list that
    decides what the scan carries) and assert EVERY slot it seeds survives the
    restart round-trip.  Enumerating from the model — not from a hand-copied
    list — is what makes a newly-gated carry covered automatically."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanConfig, LatLonCGridOceanModel,
    )

    grid, z_coord, state = _base_state()
    cfg = LatLonCGridOceanConfig(**gates)
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    seeded = model.seed_scan_carry(state, 600.0)

    from legoesm.ocean.restart import _SLOT_DIAGNOSTIC, _SLOT_POLICY
    # Everything seed_scan_carry populated, MINUS the slots the persistence
    # policy deliberately excludes (recomputed every step).
    slots = [n for n in LatLonCGridOceanState._fields
             if getattr(seeded, n) is not None
             and _SLOT_POLICY[n] != _SLOT_DIAGNOSTIC]
    # The gated configs must actually add carries beyond the base prognostics,
    # otherwise this test would be vacuous for the ab2 / leapfrog cases.
    base = [n for n in LatLonCGridOceanState._fields
            if getattr(state, n) is not None]
    if gates:
        assert set(slots) - set(base), (
            f"config {gates} seeded no extra carry — the gate list moved and "
            "this test is no longer exercising it")

    path = tmp_path / "seeded.npz"
    save_run_restart(path, seeded, step=9, time_days=0.0, grid_type="latlon")
    meta = run_restart_metadata(path)
    assert set(meta["slots"]) == set(slots), (
        "save_run_restart did not persist every slot seed_scan_carry seeded")
    got, _, _ = load_run_restart(path, state, grid_type="latlon")
    for name in slots:
        _assert_slot_equal(name, getattr(got, name), getattr(seeded, name))
    # Treedef identity is the load-bearing property for a scan resume: the
    # leapfrog carry stores u_before/v_before/... under the SOURCE Field names
    # ('u', 'v', ...) and a name drift would fail scan reconciliation.
    assert (jax.tree_util.tree_structure(got)
            == jax.tree_util.tree_structure(seeded))
