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
    # init_dynamic_ice_state requires the trailing CATEGORY axis in `shape`
    # when n_categories > 1 (it validates shape[-1] == n_categories).
    shape = ((N_LAT, N_LON) if n_categories == 1
             else (N_LAT, N_LON, n_categories))
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
    # NOT Fields: raw arrays / tuple-of-arrays that save_restart drops silently.
    # The centred-forcing trio mirrors OceanSurfaceForcing.tau_x/tau_y and the
    # freshwater_eta_tendency output, all of which are bare arrays
    # (_seed_centred_forcing_carry stores them unwrapped).
    "tau_x_prev": "array", "tau_y_prev": "array",
    "freshwater_eta_prev": "array",
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
    w_arr = np.asarray(state.w.data)
    # A WELL-FORMED entry (format 3 records shape+dtype): the point of this
    # test is the POLICY refusal, so the forged manifest must be exactly what a
    # writer with a different policy would emit, not a malformed one that the
    # schema check would reject first for an unrelated reason.
    slots["w"] = {"kind": "field",
                  "meta": ["w", ["lat", "lon", "level"], "m/s", "", "cell"],
                  "shape": list(w_arr.shape), "dtype": str(w_arr.dtype)}
    payload["_slot_kinds"] = np.asarray(json.dumps(slots))
    payload["w"] = w_arr
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


def test_subset_manifest_is_a_hard_error(tmp_path):
    """codex r1 HIGH: dropping a slot from the manifest AND its payload used to
    load fine, leaving the fresh template's cold-start value in place.  The
    recorded inventory now makes that a hard error."""
    _, _, state = _base_state()
    state = _fill_all_slots(state)
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")
    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files if k != "tke"}
    slots = json.loads(str(payload["_slot_kinds"]))
    del slots["tke"]                     # remove the MANIFEST entry too
    payload["_slot_kinds"] = np.asarray(json.dumps(slots))
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="persisted but the manifest"):
        load_run_restart(path, _base_state()[2], grid_type="latlon")


def test_slab_ice_archive_into_a_dynamic_template_is_a_hard_error(tmp_path):
    """codex r1 HIGH: a 3-field SeaIceState archive restored into a 12-field
    DynamicSeaIceState template would leave nine fields cold."""
    from legoesm.core.field import Field as _F
    from legoesm.ice.state import SeaIceState

    _, _, state = _base_state()
    z = jnp.zeros((N_LAT, N_LON))
    slab = SeaIceState(h_ice=_F(data=z, name="h_ice"),
                       T_ice=_F(data=z, name="T_ice"),
                       concentration=_F(data=z, name="ice_concentration"))
    save_run_restart(tmp_path / "slab.npz", state, step=1, time_days=0.0,
                     grid_type="latlon", ice_state=slab)
    with pytest.raises(ValueError, match="different state layouts"):
        load_run_restart(tmp_path / "slab.npz", state,
                         ice_template=_ice_state(), grid_type="latlon")


def test_a_carry_the_parent_lacked_cannot_be_continued(tmp_path):
    """A gate turned ON between legs is a new experiment, not a continuation:
    the parent has no tke to hand over, so resuming a TKE-carrying run from it
    must raise rather than silently seed the carry.

    SCOPE: this fires only when the resuming run PRE-SEEDS the carry (as the
    template here does).  run_omip_core2 loads into an all-None rest state, so
    its gate-flip case is NOT covered by this check — see the limitation note
    in load_run_restart.
    """
    _, _, state = _base_state()
    save_run_restart(tmp_path / "r.npz", state, step=1, time_days=0.0,
                     grid_type="latlon")
    live = _base_state()[2]._replace(
        tke=Field(data=jnp.ones((N_LAT, N_LON, NLEV - 1)), name="tke"))
    with pytest.raises(ValueError, match="were UNSET in"):
        load_run_restart(tmp_path / "r.npz", live, grid_type="latlon")


@pytest.mark.parametrize("kw,msg", [
    (dict(dt_seconds=1800.0), "dt_seconds"),
    (dict(n_forcing_records=99), "n_forcing_records"),
])
def test_continuation_fingerprint_mismatch_is_a_hard_error(tmp_path, kw, msg):
    """codex r1 HIGH: `step` alone does not pin the forcing — _idx_t(step, dt,
    n_rec) also depends on dt and the record count, so a parent at dt=900
    resumed at dt=1800 reads different records while every other check passes.
    """
    _, _, state = _base_state()
    save_run_restart(tmp_path / "r.npz", state, step=4, time_days=4 * 900.0 / 86400.0,
                     grid_type="latlon", dt_seconds=900.0,
                     n_forcing_records=8)
    good = dict(dt_seconds=900.0, n_forcing_records=8)
    # Control: the matching fingerprint loads.
    load_run_restart(tmp_path / "r.npz", _base_state()[2],
                     grid_type="latlon", **good)
    with pytest.raises(ValueError, match=msg):
        load_run_restart(tmp_path / "r.npz", _base_state()[2],
                         grid_type="latlon", **{**good, **kw})


def test_inconsistent_step_and_time_is_a_hard_error(tmp_path):
    """time_days must equal step*dt; if the writer's two provenance fields
    disagree, neither can be trusted to place the resume in time."""
    _, _, state = _base_state()
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=4, time_days=99.0, grid_type="latlon",
                     dt_seconds=900.0)
    with pytest.raises(ValueError, match=r"provenance is\s+inconsistent"):
        load_run_restart(path, _base_state()[2], grid_type="latlon")


def test_five_category_ice_round_trips(tmp_path):
    """codex r1 MEDIUM: a multi-category (--ice-categories 5) pack must survive
    the round trip, not just be rejected on a shape mismatch."""
    _, _, state = _base_state()
    ice5 = _ice_state(n_categories=5)
    save_run_restart(tmp_path / "r.npz", state, step=1, time_days=0.0,
                     grid_type="latlon", ice_state=ice5)
    _, got_ice, _ = load_run_restart(
        tmp_path / "r.npz", _base_state()[2],
        ice_template=_ice_state(n_categories=5), grid_type="latlon")
    for name in DynamicSeaIceState._fields:
        _assert_slot_equal(f"ice_{name}", getattr(got_ice, name),
                           getattr(ice5, name))
    assert (jax.tree_util.tree_structure(got_ice)
            == jax.tree_util.tree_structure(ice5))


def test_stripped_v2_metadata_is_a_hard_error(tmp_path):
    """codex r2 HIGH: the v2 identity/inventory keys were OPTIONAL, so deleting
    _inventory (or retagging a v1 archive as _format=2) skipped the exact-layout
    checks entirely.  Every required key must now be present."""
    _, _, state = _base_state()
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")
    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files if k != "_inventory"}
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="missing required run-restart"):
        load_run_restart(path, _base_state()[2], grid_type="latlon")


def test_inventory_and_manifest_must_agree(tmp_path):
    """codex r2 HIGH: relabelling a persisted slot 'absent' while leaving its
    payload used to let the loader skip it.  The two records are cross-checked
    in BOTH directions now."""
    _, _, state = _base_state()
    state = _fill_all_slots(state)
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")
    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files}
    inv = json.loads(str(payload["_inventory"]))
    inv["tke"] = "absent"                       # lie about a persisted slot
    payload["_inventory"] = np.asarray(json.dumps(inv))
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="inventory does not"):
        load_run_restart(path, _base_state()[2], grid_type="latlon")


def test_config_fingerprint_mismatch_is_a_hard_error(tmp_path):
    """codex r2 HIGH: dt/n_rec/x64 do not pin the viscosity, EOS, mixing,
    barotropic or sea-ice settings — all of which change step N+1."""
    _, _, state = _base_state()
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon",
                     config_fingerprint="abc123")
    # Control: the matching fingerprint loads.
    load_run_restart(path, _base_state()[2], grid_type="latlon",
                     config_fingerprint="abc123")
    with pytest.raises(ValueError, match="config_fingerprint"):
        load_run_restart(path, _base_state()[2], grid_type="latlon",
                         config_fingerprint="different")


def test_parent_lineage_is_recorded(tmp_path):
    """codex r2 LOW: a child leg records which restart it resumed FROM."""
    _, _, state = _base_state()
    save_run_restart(tmp_path / "child.npz", state, step=2, time_days=0.0,
                     grid_type="latlon", parent="/runs/leg1/restart.npz")
    assert (run_restart_metadata(tmp_path / "child.npz")["parent"]
            == "/runs/leg1/restart.npz")
    save_run_restart(tmp_path / "fresh.npz", state, step=0, time_days=0.0,
                     grid_type="latlon")
    assert run_restart_metadata(tmp_path / "fresh.npz")["parent"] == ""


def test_a_stripped_fingerprint_cannot_bypass_the_check(tmp_path):
    """codex r3 HIGH: _require_same silently SKIPPED a missing saved value, so
    an archive written without a fingerprint bypassed the configuration check
    that the resuming run explicitly asked for."""
    _, _, state = _base_state()
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")
    # No config_fingerprint / dt_seconds were written; a caller that supplies
    # them must NOT be silently let through.
    with pytest.raises(ValueError, match="records no config_fingerprint"):
        load_run_restart(path, _base_state()[2], grid_type="latlon",
                         config_fingerprint="abc")
    with pytest.raises(ValueError, match="records no dt_seconds"):
        load_run_restart(path, _base_state()[2], grid_type="latlon",
                         dt_seconds=900.0)
    # Opting out (passing nothing) still loads — the checks are caller-driven.
    load_run_restart(path, _base_state()[2], grid_type="latlon")


def test_empty_inventory_cannot_skip_the_layout_checks(tmp_path):
    """codex r3 HIGH: `if inventory:` meant an EMPTY or null inventory skipped
    every exact-layout check instead of failing."""
    _, _, state = _base_state()
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")
    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files}
    payload["_inventory"] = np.asarray(json.dumps({}))
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="empty or malformed"):
        load_run_restart(path, _base_state()[2], grid_type="latlon")


def test_orphan_payload_array_is_a_hard_error(tmp_path):
    """codex r4 HIGH: relabel a slot 'absent', delete its manifest entry, but
    LEAVE its array in the npz — every inventory/manifest check passed and the
    carry silently cold-started.  Unreferenced payload keys now raise.

    This is the cheap tamper/corruption case, distinct from a fully coordinated
    rewrite (which would need a signature to detect)."""
    _, _, state = _base_state()
    state = _fill_all_slots(state)
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")
    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files}          # keep the tke ARRAY
    slots = json.loads(str(payload["_slot_kinds"]))
    del slots["tke"]                                   # drop the manifest entry
    payload["_slot_kinds"] = np.asarray(json.dumps(slots))
    inv = json.loads(str(payload["_inventory"]))
    inv["tke"] = "absent"                              # ...and relabel it
    payload["_inventory"] = np.asarray(json.dumps(inv))
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="no\\s+manifest entry references"):
        load_run_restart(path, _base_state()[2], grid_type="latlon")


def test_tuple_carry_elements_are_not_mistaken_for_orphans(tmp_path):
    """Non-vacuity guard for the orphan check: a tuple carry stores its
    elements under `name::i` keys, which the reference set must expand — else
    every bt_hist archive would be rejected."""
    _, _, state = _base_state()
    state = _fill_all_slots(state)          # populates bt_hist (a 6-tuple)
    assert isinstance(state.bt_hist, tuple) and len(state.bt_hist) == 6
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")
    got, _, _ = load_run_restart(path, _base_state()[2], grid_type="latlon")
    _assert_slot_equal("bt_hist", got.bt_hist, state.bt_hist)


def test_underscore_named_payload_cannot_hide_a_carry(tmp_path):
    """codex r5 HIGH: the orphan check partitioned on the '_' prefix, so
    renaming the hidden `tke` array to `_tke` made it vanish from the payload
    set and the carry cold-started anyway.  Unknown metadata keys now raise."""
    _, _, state = _base_state()
    state = _fill_all_slots(state)
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")
    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files}
    payload["_tke"] = payload.pop("tke")          # smuggle it into the
    slots = json.loads(str(payload["_slot_kinds"]))   # reserved namespace
    del slots["tke"]
    payload["_slot_kinds"] = np.asarray(json.dumps(slots))
    inv = json.loads(str(payload["_inventory"]))
    inv["tke"] = "absent"
    payload["_inventory"] = np.asarray(json.dumps(inv))
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="unrecognised metadata key"):
        load_run_restart(path, _base_state()[2], grid_type="latlon")


class _KeyAccessSpy:
    """Wrap an ``NpzFile`` and record every member that is actually READ.

    ``NpzFile.files`` comes from the zip CENTRAL DIRECTORY, so listing names
    inflates nothing; ``f[key]`` decompresses the member.  Recording
    ``__getitem__`` is therefore an exact probe of "was the payload
    materialised", not a proxy for it.
    """

    def __init__(self, inner, touched):
        self._inner = inner
        self._touched = touched
        self._entered = 0

    @property
    def files(self):
        return self._inner.files

    def __getitem__(self, key):
        self._touched.append(key)
        return self._inner[key]

    def __contains__(self, key):
        return key in self._inner

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)


def test_a_mismatched_restart_is_rejected_before_the_payload_is_read(
        tmp_path, monkeypatch):
    """codex r6 MEDIUM: the single-open fix materialised EVERY payload array
    before the format / grid / dt / fingerprint / time checks, so a
    wrong-config resume decompressed the whole state (multi-GB in production)
    only to reject it on a one-line header mismatch.  The header and the
    key-name checks now run first, inside the same single open.

    NON-VACUITY is built in: the CONTROL at the end runs a SUCCESSFUL load
    through the same spy and asserts the payload keys DO appear, so the empty
    list below cannot come from a probe that never fires.
    """
    _, _, state = _base_state()
    state = _fill_all_slots(state)
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=2, time_days=0.5, grid_type="latlon",
                     dt_seconds=21600.0, n_forcing_records=8,
                     config_fingerprint="parent-config")

    touched: list[str] = []
    opens: list[str] = []
    real_load = np.load

    def _spy_load(*a, **kw):
        # Positional OR keyword: np.load's parameter is named `file`, and a
        # keyword call would otherwise record "" and fail this test for the
        # wrong reason (codex r8 LOW).
        opens.append(str(a[0] if a else kw.get("file", "")))
        return _KeyAccessSpy(real_load(*a, **kw), touched)

    monkeypatch.setattr(np, "load", _spy_load)

    # --- REJECTION PATH: not one payload array may be inflated ------------
    with pytest.raises(ValueError, match="config_fingerprint"):
        load_run_restart(path, _base_state()[2], grid_type="latlon",
                         dt_seconds=21600.0, n_forcing_records=8,
                         config_fingerprint="this-legs-config")
    early = sorted({k for k in touched if not k.startswith("_")})
    assert early == [], (
        f"a mismatched restart decompressed payload array(s) {early} before "
        "rejecting the header")
    # It DID read the header (else the assertion above would be trivially
    # satisfied by a loader that read nothing at all).
    assert "_config_fingerprint" in touched
    # ...and the r5 SINGLE-OPEN property still holds (codex r7): two opens
    # raced with the writer's atomic cadence replacement, yielding state from
    # archive B under archive A's header.  Counting opens here means a
    # regression to two opens cannot slip past this test either.
    assert opens == [str(path)], f"expected exactly one np.load, got {opens}"

    # --- CONTROL: the same spy sees the arrays on the success path --------
    touched.clear()
    opens.clear()
    got, _, _ = load_run_restart(path, _base_state()[2], grid_type="latlon",
                                 dt_seconds=21600.0, n_forcing_records=8,
                                 config_fingerprint="parent-config")
    payload_read = {k for k in touched if not k.startswith("_")}
    assert {"T", "S", "tke"} <= payload_read, (
        f"the spy saw only {sorted(payload_read)} on a successful load, so the "
        "rejection-path assertion proves nothing")
    assert opens == [str(path)], f"expected exactly one np.load, got {opens}"
    _assert_slot_equal("tke", got.tke, state.tke)


@pytest.mark.parametrize("key,forged,msg", [
    # A 0-d FLOAT step: rank 0, so the r6 guard passed it.  With _time_days
    # edited to match, every other check passed and the driver's int() then
    # truncated it — state at one step, forcing index at another.
    ("_step", np.asarray(2.5), "counts whole steps"),
    ("_format", np.asarray("2"), "dtype"),        # string where an int is due
    ("_time_days", np.asarray(float("nan")), "not finite"),
    ("_dt_seconds", np.asarray(0.0), "non-positive timestep"),
])
def test_header_values_must_satisfy_their_schema(tmp_path, key, forged, msg):
    """codex r7 LOW: rank 0 alone is not a schema.  Each header field has a
    known dtype class and range because the writer emits it from a known Python
    type; anything else means the archive was edited or written by an
    incompatible writer."""
    _, _, state = _base_state()
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=2, time_days=0.5, grid_type="latlon",
                     dt_seconds=21600.0)
    # CONTROL: untouched, it loads.
    load_run_restart(path, _base_state()[2], grid_type="latlon")

    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files}
    payload[key] = forged
    if key == "_step":
        # Keep step*dt == time_days so the EXISTING provenance cross-check
        # cannot be what rejects it — the schema must.
        payload["_time_days"] = np.asarray(2.5 * 21600.0 / 86400.0)
    np.savez(path, **payload)
    with pytest.raises(ValueError, match=msg):
        load_run_restart(path, _base_state()[2], grid_type="latlon")


def test_a_retyped_or_resized_payload_array_is_a_hard_error(tmp_path):
    """codex r8 HIGH: the loader could only compare a restored slot against the
    TEMPLATE, and ``run_omip_core2`` loads into a rest state whose optional
    carries are all ``None`` — so their shape check was SKIPPED and a
    wrong-shaped ``tke`` loaded.  Dtype was weaker still: ``_as_jnp`` only
    asserts JAX does not demote what the file holds, so an f64 field rewritten
    as f32 passed.  The manifest now records shape+dtype per slot and the
    payload is cross-checked against it."""
    _, _, state = _base_state()
    state = _fill_all_slots(state)
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")

    # (a) SHAPE, on a carry the resuming template leaves None (the case the
    #     template comparison could never catch).
    fresh = _base_state()[2]
    assert fresh.tke is None, "the template must leave tke unseeded here"
    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files}
    orig = payload["tke"]
    payload["tke"] = np.zeros((orig.shape[0], orig.shape[1], orig.shape[2] + 1),
                              dtype=orig.dtype)
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="manifest entry records"):
        load_run_restart(path, _base_state()[2], grid_type="latlon")

    # (b) DTYPE, on a prognostic: rewrite T at the OTHER float width.  Which
    #     width the state is built at is not the point (and is not asserted
    #     here — it varies with the rest-state builder); the point is that a
    #     retyped payload no longer matches the manifest.
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")
    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files}
    saved_dtype = payload["T"].dtype
    other = np.float32 if saved_dtype == np.float64 else np.float64
    payload["T"] = payload["T"].astype(other)
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="records .*float"):
        load_run_restart(path, _base_state()[2], grid_type="latlon")


def test_the_manifest_records_shape_and_dtype_for_every_persisted_array(
        tmp_path):
    """Non-vacuity guard for the cross-check above: if the writer stopped
    recording specs, the loader's comparison would silently have nothing to
    compare against."""
    _, _, state = _base_state()
    state = _fill_all_slots(state)
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")
    meta = run_restart_metadata(path)
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: (f[k].shape, str(f[k].dtype))
                  for k in f.files if not k.startswith("_")}
    seen = 0
    for name, entry in meta["slots"].items():
        if entry["kind"] == "tuple":
            for i, spec in enumerate(entry["specs"]):
                got = arrays[f"{name}::{i}"]
                assert (tuple(spec["shape"]), spec["dtype"]) == got, name
                seen += 1
        else:
            got = arrays[name]
            assert (tuple(entry["shape"]), entry["dtype"]) == got, name
            seen += 1
    assert seen == len(arrays), (
        f"{len(arrays) - seen} payload array(s) have no shape/dtype record")


@pytest.mark.parametrize("entry,msg", [
    ({"kind": "blob"}, "cannot decode"),
    ({"kind": "tuple", "n": -1}, r"integer in \[0"),
    ({"kind": "tuple"}, r"integer in \[0"),
    ({"kind": "tuple", "n": 10 ** 9}, r"integer in \[0"),
    ({"kind": "tuple", "n": 2, "elems": [None]}, "element-metadata"),
    ({"kind": "field", "meta": [], "shape": [1], "dtype": "float64"},
     "5-element"),
    ({"kind": "field", "meta": ["tke", "notalist", "1", "", "cell"],
      "shape": [1], "dtype": "float64"}, "list of strings"),
    ({"kind": "array"}, "records shape None"),
    ({"kind": "array", "shape": [-1], "dtype": "float64"},
     "non-negative integers"),
    ({"kind": "array", "shape": [1], "dtype": ""}, "non-empty dtype"),
    ({"kind": "tuple", "n": 1, "elems": [None]}, "shape/dtype records"),
    ("not-a-dict", "expected an object"),
])
def test_a_malformed_manifest_entry_is_rejected_before_the_payload_is_read(
        tmp_path, monkeypatch, entry, msg):
    """codex r7 MEDIUM: an unknown manifest ``kind`` survived every key-name and
    layout check and was only rejected inside ``_decode_slot`` — after the whole
    payload had been inflated.  A malformed tuple entry was worse: the key
    expansion does ``int(entry['n'])`` and raised a bare KeyError."""
    _, _, state = _base_state()
    state = _fill_all_slots(state)
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")
    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files}
    slots = json.loads(str(payload["_slot_kinds"]))
    slots["tke"] = entry
    payload["_slot_kinds"] = np.asarray(json.dumps(slots))
    np.savez(path, **payload)

    touched: list[str] = []
    real_load = np.load
    monkeypatch.setattr(
        np, "load", lambda *a, **kw: _KeyAccessSpy(real_load(*a, **kw), touched))
    with pytest.raises(ValueError, match=msg):
        load_run_restart(path, _base_state()[2], grid_type="latlon")
    assert [k for k in touched if not k.startswith("_")] == [], (
        "a malformed manifest inflated the payload before being rejected")


@pytest.mark.parametrize("key,forged", [
    ("_step", np.asarray([2])),                 # numeric: .item() accepts (1,)
    ("_grid_type", np.asarray(["latlon"])),     # string: str() gives "['x']"
])
def test_a_size_one_array_cannot_masquerade_as_scalar_metadata(
        tmp_path, key, forged):
    """codex r6 LOW: ``ndarray.item()`` accepts ANY size-one array and
    ``str()`` of one yields ``"['latlon']"``, so a payload array reshaped to
    ``(1,)`` and renamed to a declared metadata key was read as that header
    field — while simultaneously escaping the orphan check, which skips the
    underscore namespace.  Every header field is written as a 0-d scalar, so
    the parser now demands rank 0."""
    _, _, state = _base_state()
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=2, time_days=0.5, grid_type="latlon",
                     dt_seconds=21600.0)
    # CONTROL: the untouched archive loads (so a green test below is the rank
    # guard firing, not an unrelated rejection).
    load_run_restart(path, _base_state()[2], grid_type="latlon")

    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files}
    assert payload[key].ndim == 0, "the writer no longer emits a 0-d scalar"
    payload[key] = forged
    np.savez(path, **payload)
    with pytest.raises(ValueError, match=r"has shape \(1,\)"):
        load_run_restart(path, _base_state()[2], grid_type="latlon")


def test_writer_metadata_keys_are_all_declared(tmp_path):
    """Drift gate: every '_' key the writer emits must be in
    _RUN_METADATA_KEYS, else the reader would call it an orphan payload.  This
    is the ratchet that stops the exact regression I shipped once (adding
    _config_fingerprint/_parent without telling the reader broke every load)."""
    from legoesm.ocean.restart import _RUN_METADATA_KEYS

    _, _, state = _base_state()
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon",
                     dt_seconds=900.0, n_forcing_records=8,
                     config_fingerprint="abc", parent="p.npz", sha="deadbeef")
    with np.load(path, allow_pickle=False) as f:
        emitted = {k for k in f.files if k.startswith("_")}
    assert emitted <= _RUN_METADATA_KEYS, (
        f"writer emits undeclared metadata {sorted(emitted - _RUN_METADATA_KEYS)}")


def test_inconsistent_sea_ice_records_are_a_hard_error(tmp_path):
    """codex r5 LOW: an empty ice manifest bypassed validation of
    _ice_inventory / _ice_class, so the three ice records could disagree."""
    _, _, state = _base_state()
    path = tmp_path / "r.npz"
    save_run_restart(path, state, step=1, time_days=0.0, grid_type="latlon")
    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files}
    payload["_ice_class"] = np.asarray("DynamicSeaIceState")   # but no slots
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="inconsistent sea-ice records"):
        load_run_restart(path, _base_state()[2], grid_type="latlon")


def test_metadata_rejects_a_non_run_restart(tmp_path):
    from legoesm.ocean.restart import save_restart

    _, _, state = _base_state()
    save_restart(state, tmp_path / "legacy.npz")
    with pytest.raises(ValueError, match="not a run restart"):
        run_restart_metadata(tmp_path / "legacy.npz")


# ---------------------------------------------------------------------------
# The authoritative carry enumeration: seed_scan_carry itself
# ---------------------------------------------------------------------------

def _add_land(state):
    """Mask the southernmost row as land (and the matching faces).

    Uses replace_land_mask so land_mask/u_mask/v_mask stay consistent — a bare
    ``_replace(land_mask=...)`` leaves stale face masks and leaks mass through
    walls (CLAUDE.md).
    """
    from legoesm.ocean.init_latlon_cgrid import replace_land_mask
    lm = state.land_mask.data.at[0, :].set(0.0)
    return replace_land_mask(state, lm)


def _config_for_gates(gates):
    """Build a LatLonCGridOceanConfig opening ONE seed_scan_carry gate.

    The underscore-prefixed keys are shorthands for gates that live in nested
    sub-configs (``physics.vertical_mixing``, ``gm_redi``, ``barotropic``)
    rather than as top-level fields; everything else is passed through
    verbatim.  Import paths verified against the production builders in
    run_omip_core2 (build_tripole_vmix_config / build_tripole).
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanConfig,
    )
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        TKEConfig, VerticalMixingConfig,
    )
    from legoesm.ocean.state import BarotropicConfig

    gates = dict(gates)
    gates.pop("_needs_forcing", None)
    gates.pop("_needs_land", None)
    kw = {}

    def _tke_cfg(**tke_kw):
        # The lat-lon C-grid REJECTS OceanPhysicsConfig's default
        # lateral_mixing scheme ("harmonic" is cubed-sphere-only), so every
        # other module must be explicitly off — same shape as the production
        # tripole builder.
        from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
        from legoesm.ocean.physics.lateral_mixing.config import (
            LateralMixingConfig,
        )
        from legoesm.ocean.physics.surface_forcing.config import (
            SurfaceForcingConfig,
        )
        return dict(
            physics=OceanPhysicsConfig(
                vertical_mixing=VerticalMixingConfig(
                    scheme="tke", tke=TKEConfig(prognostic=True, **tke_kw)),
                lateral_mixing=LateralMixingConfig(scheme="none"),
                surface_forcing=SurfaceForcingConfig(scheme="none"),
                bottom_drag=BottomDragConfig(scheme="none"),
                shortwave_penetration=None),
            implicit_vertical_mixing=True,
        )

    if gates.pop("_tke", False):
        kw.update(_tke_cfg())
    if gates.pop("_tke_adv", False):
        kw.update(_tke_cfg(advection_scheme="superbee"))
    if gates.pop("_eke", False):
        kw["gm_redi"] = GMRediConfig(eke=EKEConfig())
    if gates.pop("_eke3d", False):
        kw["gm_redi"] = GMRediConfig(eke=EKEConfig(eke_3d=True))
    if gates.pop("_slow_ab2", False):
        kw["barotropic"] = BarotropicConfig(barotropic_slow_forcing_ab2=True)
    if "barotropic" in gates:
        kw["barotropic"] = BarotropicConfig(
            barotropic_solver=gates.pop("barotropic"))
    kw.update(gates)
    return LatLonCGridOceanConfig(**kw)


# Each entry is a MINIMAL valid config that opens a different seed_scan_carry
# gate.  ab2 outer and ab2 tracer cannot be combined (the model rejects the
# double-count), and leapfrog requires the explicit-AB2 Coriolis — so these are
# three separate cases, not one.
@pytest.mark.parametrize("gates", [
    {"tracer_time_integrator": "ab2"},        # -> T_flux_div_prev/S_flux_div_prev
    {"outer_integrator": "ab2"},              # -> {T,S,u,v}_incr_prev
    {"outer_integrator": "leapfrog",          # -> {u,v,T,S,eta}_before
     "coriolis_scheme": "explicit_ab2"},
    # barotropic_forcing_centred is only valid under the leapfrog integrator
    # (the ½(before+now) average reads u_before/tau_x_prev), and the AB2 slow
    # forcing is only valid with the explicit-AB2 Coriolis — the model
    # validates both, so the companions are part of the case, not decoration.
    {"barotropic_forcing_centred": True,      # -> tau_{x,y}_prev
     "outer_integrator": "leapfrog",          #    (seeded FROM the forcing)
     "coriolis_scheme": "explicit_ab2",
     "_needs_forcing": True},
    {"_slow_ab2": True,                       # -> F_slow_{u,v}_prev
     "coriolis_scheme": "explicit_ab2",       #    (+ the incr_prev quartet:
     "outer_integrator": "ab2"},              #    explicit_ab2 Coriolis is
                                              #    unstable under forward Euler)
    {"_tke": True},                           # -> tke (+ dtke when advected)
    {"_tke_adv": True},                       # -> tke + dtke
    {"_eke": True},                           # -> eke
    {"_eke3d": True},                         # -> eke (3-D) + eke_diss
    {"barotropic": "rigid_lid",               # -> psi/dpsi/dpsi_prev/dpsin/...
     "_needs_land": True},                    #    (islands pin the streamfn)
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
    if gates.get("_needs_land"):
        # The rigid-lid solver refuses a land-free domain (the streamfunction
        # has nothing to pin to), so this gate needs a coastline.
        state = _add_land(state)
    cfg = _config_for_gates(gates)
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    # barotropic_forcing_centred seeds tau_{x,y}_prev FROM the step's forcing
    # (NEMO's nit000 "before = now" rule), so without a surface_forcing kwarg
    # it seeds nothing and the case would be vacuous.
    step_kwargs = {}
    if gates.get("_needs_forcing"):
        from legoesm.ocean.freshwater import FreshwaterForcing
        from legoesm.ocean.state import OceanSurfaceForcing
        _z2 = jnp.zeros((N_LAT, N_LON))
        step_kwargs["surface_forcing"] = OceanSurfaceForcing(
            tau_x=_z2 + 0.05, tau_y=_z2 - 0.02, q_net=_z2, sw_down=_z2)
        # freshwater too: without it the case seeds only the tau pair and
        # freshwater_eta_prev is never exercised (codex r2 MEDIUM).
        step_kwargs["freshwater"] = FreshwaterForcing(
            precip=_z2 + 1.0e-5, evap=_z2, runoff=_z2, ice_fw=_z2)
    seeded = model.seed_scan_carry(state, 600.0, **step_kwargs)

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
    if gates.get("_needs_forcing"):
        assert {"tau_x_prev", "tau_y_prev", "freshwater_eta_prev"} <= set(slots), (
            "barotropic_forcing_centred must seed the tau pair AND "
            f"freshwater_eta_prev; got {sorted(set(slots) - set(base))}")
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
