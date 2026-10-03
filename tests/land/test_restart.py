"""Restart / checkpoint I/O tests (Phase C)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.restart import load_land_restart, save_land_restart
from legoesm.land.state import MultiLayerLandState

jax.config.update("jax_enable_x64", True)

_NCOL, _NLAY = 5, 8


def _fake_state(seed=0, with_tgc=True):
    rng = np.random.default_rng(seed)
    return MultiLayerLandState(
        T_soil=jnp.asarray(280.0 + 3.0 * rng.standard_normal((_NCOL, _NLAY))),
        psi_soil=jnp.asarray(-1.0 * rng.uniform(0.5, 2.0, (_NCOL, _NLAY))),
        theta_soil=jnp.asarray(rng.uniform(0.15, 0.45, (_NCOL, _NLAY))),
        runoff_surface=jnp.asarray(rng.uniform(0, 1e-5, _NCOL)),
        runoff_subsurface=jnp.asarray(rng.uniform(0, 1e-5, _NCOL)),
        snow_depth=jnp.asarray(rng.uniform(0, 20, _NCOL)),
        snow_age=jnp.asarray(rng.uniform(0, 1e6, _NCOL)),
        TgC=jnp.asarray(rng.uniform(-20, 25, _NCOL)) if with_tgc else None,
    )


def test_round_trip_bit_identical(tmp_path):
    st = _fake_state(seed=1)
    save_land_restart(tmp_path / "r.npz", st, land_mode="multilayer",
                      t_end_s=8760.0 * 3600.0, n_steps_completed=8760,
                      metadata={"tag": "hello"})
    st2, meta = load_land_restart(tmp_path / "r.npz",
                                  expected_land_mode="multilayer",
                                  expected_ncol=_NCOL, expected_n_layers=_NLAY)
    # every field byte-for-byte identical after a save+load
    for field in ("T_soil", "psi_soil", "theta_soil",
                  "runoff_surface", "runoff_subsurface",
                  "snow_depth", "snow_age", "TgC"):
        a = np.asarray(getattr(st, field))
        b = np.asarray(getattr(st2, field))
        np.testing.assert_array_equal(a, b, err_msg=f"mismatch in {field}")
    assert meta["t_end_s"] == 8760.0 * 3600.0
    assert meta["n_steps_completed"] == 8760
    assert meta["land_mode"] == "multilayer"
    assert meta["metadata"] == {"tag": "hello"}


def test_load_without_tgc(tmp_path):
    st = _fake_state(seed=2, with_tgc=False)
    save_land_restart(tmp_path / "r.npz", st, land_mode="multilayer",
                      t_end_s=0.0, n_steps_completed=0)
    st2, _ = load_land_restart(tmp_path / "r.npz",
                               expected_land_mode="multilayer",
                               expected_ncol=_NCOL, expected_n_layers=_NLAY)
    assert st2.TgC is None


def test_land_mode_mismatch_raises(tmp_path):
    save_land_restart(tmp_path / "r.npz", _fake_state(),
                      land_mode="multilayer", t_end_s=0.0, n_steps_completed=0)
    with pytest.raises(ValueError, match="land_mode"):
        load_land_restart(tmp_path / "r.npz",
                          expected_land_mode="slab",
                          expected_ncol=_NCOL, expected_n_layers=_NLAY)


def test_ncol_mismatch_raises(tmp_path):
    save_land_restart(tmp_path / "r.npz", _fake_state(),
                      land_mode="multilayer", t_end_s=0.0, n_steps_completed=0)
    with pytest.raises(ValueError, match="ncol"):
        load_land_restart(tmp_path / "r.npz",
                          expected_land_mode="multilayer",
                          expected_ncol=_NCOL + 1, expected_n_layers=_NLAY)


def test_n_layers_mismatch_raises(tmp_path):
    save_land_restart(tmp_path / "r.npz", _fake_state(),
                      land_mode="multilayer", t_end_s=0.0, n_steps_completed=0)
    with pytest.raises(ValueError, match="n_layers"):
        load_land_restart(tmp_path / "r.npz",
                          expected_land_mode="multilayer",
                          expected_ncol=_NCOL, expected_n_layers=_NLAY + 1)


def test_slab_mode_save_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError, match="slab"):
        save_land_restart(tmp_path / "r.npz", _fake_state(),
                          land_mode="slab", t_end_s=0.0, n_steps_completed=0)


def test_merge_land_restart_into_template_fixes_structure():
    """#746: a loaded restart carries only the core prognostic fields (optional
    structural fields default to None); merging it onto a template that HAS the
    optional fields as arrays restores the full pytree structure while keeping
    the restart's prognostic values — so the resumed/coupled lax.scan carry
    input matches its output."""
    from legoesm.land.restart import merge_land_restart_into_template

    loaded = _fake_state(seed=1)                      # optional fields = None
    assert loaded.surface_water is None
    # Template with the optional fields populated as arrays (what
    # init_multilayer_land_state / a stepped state looks like).
    template = _fake_state(seed=2)._replace(
        surface_water=jnp.zeros((_NCOL,)),
        snow_bands=jnp.zeros((_NCOL, 3)),
        ice_bands=jnp.zeros((_NCOL, 3)),
    )
    merged = merge_land_restart_into_template(loaded, template)

    # Core prognostic fields come from the LOADED restart (bit-identical).
    for f in ("T_soil", "psi_soil", "theta_soil", "runoff_surface",
              "runoff_subsurface", "snow_depth", "snow_age", "TgC"):
        np.testing.assert_array_equal(
            np.asarray(getattr(merged, f)), np.asarray(getattr(loaded, f)))
    # Optional structural fields come from the TEMPLATE (arrays, not None) — the
    # pytree structure now matches a stepped state.
    assert merged.surface_water is not None
    assert merged.snow_bands is not None
    import jax
    assert (jax.tree_util.tree_structure(merged)
            == jax.tree_util.tree_structure(template))


def test_merge_land_restart_shape_skew_raises():
    """A soil-layer / resolution skew between the restart and the template must
    raise, not silently reshape."""
    from legoesm.land.restart import merge_land_restart_into_template

    loaded = _fake_state(seed=3)                      # (5, 8)
    template = MultiLayerLandState(
        T_soil=jnp.zeros((_NCOL, 10)),               # 10 layers != restart's 8
        psi_soil=jnp.zeros((_NCOL, 10)),
        theta_soil=jnp.zeros((_NCOL, 10)),
        runoff_surface=jnp.zeros(_NCOL),
        runoff_subsurface=jnp.zeros(_NCOL),
        snow_depth=jnp.zeros(_NCOL), snow_age=jnp.zeros(_NCOL),
    )
    with pytest.raises(ValueError, match="soil-layer skew"):
        merge_land_restart_into_template(loaded, template)


# ---------------------------------------------------------------------------
# The layer COUNT does not identify a soil column. Ten layers over 3 m and ten
# over 6.4 m have the same array shapes, so a warm start across them loaded
# without complaint and read the profile at the wrong depths. Found by codex
# on the calibrated-LMIP PR, which is what made the column configurable.
# ---------------------------------------------------------------------------

def _grid(n_layers=_NLAY, total_depth=3.0):
    from legoesm.land.soil_grid import SoilGridConfig
    return SoilGridConfig(n_layers=n_layers, total_depth=total_depth)


def _write(tmp_path, state, soil_grid):
    return save_land_restart(
        tmp_path / "r.npz", state, land_mode="multilayer", t_end_s=1.0,
        n_steps_completed=1, soil_grid=soil_grid)


def test_a_deeper_column_with_the_same_layer_count_is_refused(tmp_path):
    st = _fake_state(seed=7)
    _write(tmp_path, st, _grid(total_depth=3.0))
    with pytest.raises(ValueError, match="wrong depths"):
        load_land_restart(tmp_path / "r.npz", expected_land_mode="multilayer",
                          expected_ncol=_NCOL, expected_n_layers=_NLAY,
                          expected_soil_grid=_grid(total_depth=6.375))


def test_the_same_column_loads(tmp_path):
    st = _fake_state(seed=7)
    _write(tmp_path, st, _grid(total_depth=3.0))
    loaded, _ = load_land_restart(
        tmp_path / "r.npz", expected_land_mode="multilayer", expected_ncol=_NCOL,
        expected_n_layers=_NLAY, expected_soil_grid=_grid(total_depth=3.0))
    assert loaded.T_soil.shape == (_NCOL, _NLAY)


def test_a_file_without_the_geometry_warns_rather_than_pretending(tmp_path):
    """The published initial states predate the stamp, so this cannot raise --
    but it must not read as a verified match either.

    Only on the HISTORICAL DEFAULT column: that is the one the published
    states are on, so it is the only one where an unstamped file is a
    reasonable thing to be handed. Every other column refuses.
    """
    from legoesm.land.soil_grid import SoilGridConfig

    st = _fake_state(seed=7)
    save_land_restart(tmp_path / "r.npz", st, land_mode="multilayer",
                      t_end_s=1.0, n_steps_completed=1)   # no soil_grid
    with pytest.warns(RuntimeWarning, match="cannot be checked"):
        load_land_restart(tmp_path / "r.npz", expected_land_mode="multilayer",
                          expected_ncol=_NCOL, expected_n_layers=_NLAY,
                          expected_soil_grid=SoilGridConfig())


def test_an_unstamped_file_can_be_refused_outright(tmp_path):
    """Warning is not a check for the file most people load.

    The published initial states predate the interface stamp, so an unstamped
    file cannot be refused by default. But a run on a column that is NOT the
    historical default has no business accepting one: an old file is then
    almost certainly on the other column, and a warning it scrolls past is how
    the wrong soil profile gets used anyway.
    """
    st = _fake_state(seed=9)
    save_land_restart(tmp_path / "r.npz", st, land_mode="multilayer",
                      t_end_s=1.0, n_steps_completed=1)     # no soil_grid
    with pytest.raises(ValueError, match="not the historical default"):
        load_land_restart(tmp_path / "r.npz", expected_land_mode="multilayer",
                          expected_ncol=_NCOL, expected_n_layers=_NLAY,
                          expected_soil_grid=_grid(total_depth=3.0),
                          require_soil_grid=True)


def test_the_two_spellings_of_a_soil_column_are_one_column():
    """Two independent fixes for one defect met at a merge.

    One identifies a soil column by its layer INTERFACES, the other by its
    layer THICKNESSES. They are the same fact -- the interfaces are the
    running sum -- so both spellings are accepted and reduced to one form
    before anything is compared. If they ever stop agreeing, every guard built
    on them is comparing different things.
    """
    from legoesm.land.restart import _soil_dz_from
    from legoesm.land.soil_grid import make_soil_grid

    grid = _grid(total_depth=3.0)
    dz = make_soil_grid(grid).dz
    np.testing.assert_allclose(
        np.asarray(_soil_dz_from(grid, None, what="t")),
        np.asarray(_soil_dz_from(None, dz, what="t")))
    # Both together are allowed when they agree, and refused when they do not.
    _soil_dz_from(grid, dz, what="t")
    with pytest.raises(ValueError, match="DIFFERENT columns"):
        _soil_dz_from(grid, np.asarray(dz) * 2.0, what="t")


def test_a_file_stamped_one_way_is_checkable_the_other_way(tmp_path):
    """Files written by either lane are in the wild, so either loads."""
    from legoesm.land.soil_grid import make_soil_grid

    grid = _grid(total_depth=3.0)
    dz = np.asarray(make_soil_grid(grid).dz)
    other = _grid(total_depth=6.375)

    # written from the grid, checked against thicknesses
    a = tmp_path / "a.npz"
    save_land_restart(a, _fake_state(seed=11), land_mode="multilayer",
                      t_end_s=1.0, n_steps_completed=1, soil_grid=grid)
    load_land_restart(a, expected_land_mode="multilayer", expected_ncol=_NCOL,
                      expected_n_layers=_NLAY, expected_soil_dz=dz)
    with pytest.raises(ValueError, match="wrong depths"):
        load_land_restart(a, expected_land_mode="multilayer",
                          expected_ncol=_NCOL, expected_n_layers=_NLAY,
                          expected_soil_dz=make_soil_grid(other).dz)

    # written from thicknesses, checked against the grid
    b = tmp_path / "b.npz"
    save_land_restart(b, _fake_state(seed=12), land_mode="multilayer",
                      t_end_s=1.0, n_steps_completed=1, soil_dz=dz)
    load_land_restart(b, expected_land_mode="multilayer", expected_ncol=_NCOL,
                      expected_n_layers=_NLAY, expected_soil_grid=grid)
    with pytest.raises(ValueError, match="wrong depths"):
        load_land_restart(b, expected_land_mode="multilayer",
                          expected_ncol=_NCOL, expected_n_layers=_NLAY,
                          expected_soil_grid=other)


def test_the_pre_load_check_reads_the_stamp_the_writer_wrote(tmp_path):
    """A reader that keeps its own key after the writer moves returns 'no
    stamp' for every file and switches the check off in silence -- which is
    exactly what happened once here."""
    from legoesm.land.restart import load_land_restart_soil_dz
    from legoesm.land.soil_grid import make_soil_grid

    grid = _grid(total_depth=3.0)
    path = tmp_path / "r.npz"
    save_land_restart(path, _fake_state(seed=13), land_mode="multilayer",
                      t_end_s=1.0, n_steps_completed=1, soil_grid=grid)
    got = load_land_restart_soil_dz(path)
    assert got is not None, "the pre-load check cannot see a stamp it wrote"
    np.testing.assert_allclose(np.asarray(got),
                               np.asarray(make_soil_grid(grid).dz))


# ---------------------------------------------------------------------------
# Second reviewer, on the merge: the edges of carrying two stamps for one fact.
# ---------------------------------------------------------------------------

def test_two_stamps_that_disagree_are_refused(tmp_path):
    """The writer derives both from one column, but a file is not always
    written by this writer: a post-processing tool can rewrite one key and not
    the other. Picking a winner is how the wrong profile gets used in silence
    (GLM-5.2)."""
    from legoesm.land.soil_grid import make_soil_grid

    path = tmp_path / "r.npz"
    save_land_restart(path, _fake_state(seed=21), land_mode="multilayer",
                      t_end_s=1.0, n_steps_completed=1,
                      soil_grid=_grid(total_depth=3.0))
    d = dict(np.load(path, allow_pickle=False))
    # Rewrite ONE stamp, as an external tool would.
    d["soil_dz"] = np.asarray(make_soil_grid(_grid(total_depth=6.375)).dz,
                              dtype=np.float64)
    np.savez_compressed(path, **d)
    with pytest.raises(ValueError, match="TWO soil-column stamps that disagree"):
        load_land_restart(path, expected_land_mode="multilayer",
                          expected_ncol=_NCOL, expected_n_layers=_NLAY,
                          expected_soil_grid=_grid(total_depth=3.0))
    # ...and it is refused even when the run's column happens to match the
    # stamp that was NOT rewritten -- otherwise the file passes on the strength
    # of one of two records that are known to contradict each other.
    with pytest.raises(ValueError, match="TWO soil-column stamps that disagree"):
        load_land_restart(path, expected_land_mode="multilayer",
                          expected_ncol=_NCOL, expected_n_layers=_NLAY,
                          expected_soil_grid=_grid(total_depth=6.375))


def test_a_stamp_that_is_not_a_column_is_refused(tmp_path):
    """The n versus n+1 relation is enforced, not inferred: guessing at a
    malformed stamp is worse than saying it cannot be read."""
    path = tmp_path / "r.npz"
    save_land_restart(path, _fake_state(seed=22), land_mode="multilayer",
                      t_end_s=1.0, n_steps_completed=1,
                      soil_grid=_grid(total_depth=3.0))
    d = dict(np.load(path, allow_pickle=False))
    d.pop("soil_dz")
    d["soil_z_interface"] = np.array([0.0, 1.0, 0.5, 2.0])   # not increasing
    np.savez_compressed(path, **d)
    with pytest.raises(ValueError, match="increasing layer"):
        load_land_restart(path, expected_land_mode="multilayer",
                          expected_ncol=_NCOL, expected_n_layers=_NLAY,
                          expected_soil_grid=_grid(total_depth=3.0))


def test_reconstructing_a_column_from_its_interfaces_is_well_conditioned():
    """The measurement that decided against an absolute tolerance floor.

    A reviewer argued one was needed: differencing interfaces to recover
    thicknesses was said to carry an error of order eps x TOTAL depth, which
    on a 3 m column would be 1.4e-5 relative on a 2.6 cm top layer -- past the
    relative tolerance, refusing valid files. Measured, it is not: the shallow
    interfaces are themselves small, so differencing near the surface
    subtracts small numbers and the error scales with the LOCAL depth.

    This pins the real number over the thinnest column this model builds, so
    the tolerance stays justified by a measurement rather than by an argument.
    """
    from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
    from legoesm.land.restart import _SOIL_DZ_RTOL

    worst = 0.0
    for n_layers, depth, growth in ((8, 3.0, 2.0), (10, 3.0, 1.5),
                                    (8, 6.375, 2.0)):
        g = make_soil_grid(SoilGridConfig(n_layers=n_layers, total_depth=depth,
                                          growth_factor=growth))
        dz = np.asarray(g.dz, dtype=np.float64)
        zi = np.asarray(g.z_interface, dtype=np.float64)
        # the same column stored at single precision, then differenced
        recovered = np.diff(zi.astype(np.float32).astype(np.float64))
        worst = max(worst, float(np.max(np.abs(recovered - dz) / dz)))
    assert worst < 0.1 * _SOIL_DZ_RTOL, (
        f"recovering thicknesses from single-precision interfaces now costs "
        f"{worst:.2e} relative, within a factor of ten of the {_SOIL_DZ_RTOL:g} "
        f"tolerance; the comparison needs an absolute floor after all")


def test_a_translated_column_is_not_mistaken_for_the_right_one():
    """Thicknesses are the DIFFERENCES of the interfaces, so a column shifted
    bodily downwards differences to exactly the right thicknesses. Accepting
    it puts every state value a metre from where it belongs (codex)."""
    from legoesm.land.restart import _recorded_soil_dz
    from legoesm.land.soil_grid import make_soil_grid

    class _Npz(dict):
        @property
        def files(self):
            return list(self)

    grid = _grid(total_depth=3.0)
    zi = np.asarray(make_soil_grid(grid).z_interface, dtype=np.float64)
    np.testing.assert_allclose(
        _recorded_soil_dz(_Npz(soil_z_interface=zi)), np.diff(zi))
    with pytest.raises(ValueError, match="AT the surface"):
        _recorded_soil_dz(_Npz(soil_z_interface=zi + 1.0), "shifted.npz")


def test_any_non_default_column_requires_a_stamp(tmp_path):
    """The predicate is the COLUMN, not a preset's name.

    Three callers asked for the strict behaviour by passing a calibration
    flag, which left every other non-default column -- any run that sets its
    own layer count or depth -- accepting an unstamped file and reading its
    profile at the wrong depths. Both reviewers found this independently.
    """
    from legoesm.land.soil_grid import SoilGridConfig

    path = tmp_path / "legacy.npz"
    save_land_restart(path, _fake_state(seed=31), land_mode="multilayer",
                      t_end_s=1.0, n_steps_completed=1)      # unstamped

    # The historical default may still load one: the published states are on
    # that column and predate the stamp.
    with pytest.warns(RuntimeWarning, match="cannot be checked"):
        load_land_restart(path, expected_land_mode="multilayer",
                          expected_ncol=_NCOL, expected_n_layers=_NLAY,
                          expected_soil_grid=SoilGridConfig())

    # Any other column may not, with no flag passed anywhere.
    with pytest.raises(ValueError, match="not the historical default"):
        load_land_restart(path, expected_land_mode="multilayer",
                          expected_ncol=_NCOL, expected_n_layers=_NLAY,
                          expected_soil_grid=_grid(total_depth=3.0))


# --- prognostic carbon in the land restart (additive) -----------------------
def test_restart_round_trips_carbon_pools_and_phi(tmp_path):
    """Pools + permafrost phi survive a save/load cycle, and a biophysics-only
    restart reports both as None rather than fabricating them."""
    from legoesm.land.carbon.config import CarbonState

    state, ncol = _fake_state(seed=3), _NCOL
    kw = dict(land_mode="multilayer", t_end_s=1.0, n_steps_completed=1,
              soil_grid=_grid())
    carbon = CarbonState(**{f: jnp.asarray(np.full(ncol, 10.0 * (i + 1)))
                            for i, f in enumerate(CarbonState._fields)})
    phi = jnp.asarray(np.linspace(0.0, 1.0, ncol))

    p = save_land_restart(tmp_path / "c.npz", state, carbon_state=carbon,
                          soil_frozen_fraction=phi, **kw)
    _s, meta = load_land_restart(p, expected_land_mode="multilayer",
                                 expected_ncol=ncol)
    got = meta["carbon_state"]
    assert got is not None
    for i, f in enumerate(CarbonState._fields):
        np.testing.assert_allclose(np.asarray(getattr(got, f)), 10.0 * (i + 1))
    np.testing.assert_allclose(np.asarray(meta["soil_frozen_fraction"]),
                               np.linspace(0.0, 1.0, ncol))

    # Biophysics-only: absent, not invented.
    p2 = save_land_restart(tmp_path / "b.npz", state, **kw)
    _s2, meta2 = load_land_restart(p2, expected_land_mode="multilayer",
                                   expected_ncol=ncol)
    assert meta2["carbon_state"] is None
    assert meta2["soil_frozen_fraction"] is None


def test_restart_with_partial_carbon_pools_raises(tmp_path):
    """A half-written carbon state must fail loudly, not resume from whatever
    pools happen to be present."""
    from legoesm.land.carbon.config import CarbonState

    state, ncol = _fake_state(seed=4), _NCOL
    kw = dict(land_mode="multilayer", t_end_s=1.0, n_steps_completed=1,
              soil_grid=_grid())
    p = save_land_restart(tmp_path / "c.npz", state, **kw)
    d = dict(np.load(p, allow_pickle=True))
    for f in CarbonState._fields[:-1]:                 # drop ONE pool
        d[f"carbon_{f}"] = np.full(ncol, 1.0)
    np.savez_compressed(p, **d)
    with pytest.raises(ValueError, match="missing"):
        load_land_restart(p, expected_land_mode="multilayer",
                          expected_ncol=ncol)


def test_snow_node_temperature_round_trips_and_grafts(tmp_path):
    """The snow-node temperature is prognostic: it round-trips exactly, is
    grafted onto a snow-node template, and is absent when never written."""
    from legoesm.land.restart import merge_land_restart_into_template
    st = _fake_state(seed=3)._replace(
        T_snow=jnp.asarray([250.0, 255.5, 261.0, 266.25, 271.0]))
    save_land_restart(tmp_path / "r.npz", st, land_mode="multilayer",
                      t_end_s=0.0, n_steps_completed=0)
    st2, _ = load_land_restart(tmp_path / "r.npz",
                               expected_land_mode="multilayer",
                               expected_ncol=_NCOL, expected_n_layers=_NLAY)
    np.testing.assert_array_equal(np.asarray(st2.T_snow), np.asarray(st.T_snow))
    tmpl = _fake_state(seed=4)._replace(T_snow=jnp.zeros(_NCOL))
    merged = merge_land_restart_into_template(st2, tmpl)
    np.testing.assert_array_equal(np.asarray(merged.T_snow), np.asarray(st.T_snow))
    save_land_restart(tmp_path / "r0.npz", _fake_state(seed=5),
                      land_mode="multilayer", t_end_s=0.0, n_steps_completed=0)
    st3, _ = load_land_restart(tmp_path / "r0.npz",
                               expected_land_mode="multilayer",
                               expected_ncol=_NCOL, expected_n_layers=_NLAY)
    assert st3.T_snow is None


def test_snow_node_restart_into_a_run_without_it_is_refused():
    from legoesm.land.restart import merge_land_restart_into_template
    st = _fake_state(seed=6)._replace(T_snow=jnp.full(_NCOL, 260.0))
    with pytest.raises(ValueError, match="T_snow"):
        merge_land_restart_into_template(st, _fake_state(seed=7))
