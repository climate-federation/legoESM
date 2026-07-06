"""Multilayer (Richards) land state must survive a checkpoint round-trip (#730).

Chain-enable for the C48 SOTA-multilayer year: the prognostic soil/snow/carbon
columns ride ``carry_aux`` (namespaced ``land_ml_*``) exactly like the
double-moment tracers, so a ``--restart-from`` resumes the deep-soil spin-up
instead of cold-starting.  Guards both halves — the save-side pack in
``_checkpoint_carry_aux`` and the load-side rebuild in
``_restore_land_ml_from_carry_aux`` — across a real npz write.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from legoesm.driver.model_driver import ModelDriver
from legoesm.land.state import MultiLayerLandState


def _state(scale: float) -> MultiLayerLandState:
    # 9 distinct arrays so a dropped/swapped field is caught, not masked.
    return MultiLayerLandState(*[
        scale * (i + 1) * np.ones((4, 6) if i < 3 else (4,), dtype=np.float32)
        for i in range(9)
    ])


def test_land_ml_survives_carry_aux_npz_roundtrip(tmp_path):
    saved = _state(1.0)
    src = SimpleNamespace(
        _carry_aux={}, _land_ml_state=saved,
        _double_moment_step_inputs=lambda: {},
        config=SimpleNamespace(convection="none"),
    )

    aux = ModelDriver._checkpoint_carry_aux(src)
    assert aux is not None and any(k.startswith("land_ml_") for k in aux)

    # Actually go through the disk the restart chain uses.
    np.savez(tmp_path / "c.npz", **aux)
    loaded = dict(np.load(tmp_path / "c.npz"))

    # Restart lands on the cold-start state; restore must overwrite it.
    dst = SimpleNamespace(_carry_aux=loaded, _land_ml_state=_state(99.0))
    ModelDriver._restore_land_ml_from_carry_aux(dst)

    for f in MultiLayerLandState._fields:
        np.testing.assert_allclose(
            getattr(dst._land_ml_state, f), getattr(saved, f))
    # land_ml_* popped, not left to re-save as junk.
    assert not any(k.startswith("land_ml_") for k in dst._carry_aux)


def test_none_optional_fields_survive_roundtrip(tmp_path):
    """The PRODUCTION shape: driver setup passes no TgC_init, so TgC is None
    (state.py Optional field). np.asarray(None) would pickle a 0-d object
    array into the npz and crash the load-side jnp.asarray — None fields must
    be skipped on save and left untouched on restore."""
    saved = _state(1.0)._replace(TgC=None)
    src = SimpleNamespace(
        _carry_aux={}, _land_ml_state=saved,
        _double_moment_step_inputs=lambda: {},
        config=SimpleNamespace(convection="none"),
    )
    aux = ModelDriver._checkpoint_carry_aux(src)
    assert "land_ml_TgC" not in aux          # None never serialized
    np.savez(tmp_path / "c.npz", **aux)      # must not need allow_pickle
    loaded = dict(np.load(tmp_path / "c.npz"))

    dst = SimpleNamespace(_carry_aux=loaded,
                          _land_ml_state=_state(99.0)._replace(TgC=None))
    ModelDriver._restore_land_ml_from_carry_aux(dst)
    assert dst._land_ml_state.TgC is None    # stays None, not resurrected
    np.testing.assert_allclose(dst._land_ml_state.T_soil, saved.T_soil)


def test_restore_is_noop_for_slab_land():
    # Slab run (_land_ml_state None) must ignore any stray land_ml_* keys AND
    # strip them, so they never leak forward into the next slab checkpoint.
    dst = SimpleNamespace(
        _carry_aux={"land_ml_T_soil": np.ones(3)}, _land_ml_state=None)
    ModelDriver._restore_land_ml_from_carry_aux(dst)
    assert dst._land_ml_state is None
    assert "land_ml_T_soil" not in dst._carry_aux   # popped, not left to re-save


def test_partial_checkpoint_raises(tmp_path):
    # A truncated / version-skewed checkpoint (a saved field missing) must fail
    # loud — restoring only the present fields would leave the rest silently at
    # cold-start values (a mixed restart), exactly what #730 exists to prevent.
    saved = _state(1.0)
    src = SimpleNamespace(
        _carry_aux={}, _land_ml_state=saved,
        _double_moment_step_inputs=lambda: {},
        config=SimpleNamespace(convection="none"),
    )
    aux = ModelDriver._checkpoint_carry_aux(src)
    del aux["land_ml_snow_depth"]                    # simulate a dropped column
    dst = SimpleNamespace(_carry_aux=dict(aux), _land_ml_state=_state(99.0))
    with pytest.raises(ValueError, match="does not match"):
        ModelDriver._restore_land_ml_from_carry_aux(dst)


def test_unknown_field_raises():
    dst = SimpleNamespace(
        _carry_aux={f"land_ml_{f}": np.ones((4, 6) if i < 3 else (4,),
                                            dtype=np.float32)
                    for i, f in enumerate(MultiLayerLandState._fields[:7])}
        | {"land_ml_bogus": np.ones(4, dtype=np.float32)},
        _land_ml_state=_state(99.0)._replace(TgC=None, surface_water=None))
    with pytest.raises(ValueError, match="unknown field"):
        ModelDriver._restore_land_ml_from_carry_aux(dst)


def test_shape_mismatch_raises():
    # A checkpoint at a different soil-layer count / resolution must be refused,
    # not broadcast into a wrong-shaped restart column.
    good = {f"land_ml_{f}": np.ones((4, 6) if i < 3 else (4,), dtype=np.float32)
            for i, f in enumerate(MultiLayerLandState._fields[:7])}
    good["land_ml_T_soil"] = np.ones((4, 8), dtype=np.float32)   # 8 != 6 layers
    dst = SimpleNamespace(
        _carry_aux=good,
        _land_ml_state=_state(99.0)._replace(TgC=None, surface_water=None))
    with pytest.raises(ValueError, match="shape"):
        ModelDriver._restore_land_ml_from_carry_aux(dst)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path
        test_land_ml_survives_carry_aux_npz_roundtrip(Path(d))
    test_restore_is_noop_for_slab_land()
    print("ok")
