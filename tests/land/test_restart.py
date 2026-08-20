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
    but it must not read as a verified match either."""
    st = _fake_state(seed=7)
    save_land_restart(tmp_path / "r.npz", st, land_mode="multilayer",
                      t_end_s=1.0, n_steps_completed=1)   # no soil_grid
    with pytest.warns(RuntimeWarning, match="cannot be checked"):
        load_land_restart(tmp_path / "r.npz", expected_land_mode="multilayer",
                          expected_ncol=_NCOL, expected_n_layers=_NLAY,
                          expected_soil_grid=_grid(total_depth=3.0))


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
