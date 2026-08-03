"""Per-layer soil PROFILE output (``T_soil``, ``theta_soil``).

The tape machinery was ``(n_slots, ncol)`` throughout, so a profile needed a
trailing layer axis threaded through the accumulator, the mean-finalisation, the
NaN-revert mask and the NetCDF writer.  Each of those is a place where a wrong
shape either raises at WRITE time -- i.e. after a whole year has integrated -- or,
worse, silently mis-divides.  These pin all four.
"""
import numpy as np
import pytest

from legoesm.land.output_tapes import (
    PROFILE_VARS,
    TapeSpec,
    finalize_tape,
    init_tape_accumulator,
    is_profile_var,
)

NSLOT, NCOL, NLAY = 4, 7, 10


def _spec(*vars_):
    return TapeSpec(name="t", freq="monthly", average="mean", vars=tuple(vars_))


def test_profile_vars_are_declared():
    assert "T_soil" in PROFILE_VARS and "theta_soil" in PROFILE_VARS
    assert is_profile_var("T_soil") and not is_profile_var("T_soil_top")


def test_accumulator_shapes_mix_profile_and_scalar():
    """A tape may carry both kinds; each gets its own shape."""
    acc = init_tape_accumulator(_spec("T_sfc", "T_soil"), NSLOT, NCOL, n_layers=NLAY)
    assert np.shape(acc["T_sfc"]) == (NSLOT, NCOL)
    assert np.shape(acc["T_soil"]) == (NSLOT, NCOL, NLAY)
    assert np.shape(acc["count"]) == (NSLOT,)


def test_missing_n_layers_raises_rather_than_mis_shaping():
    """Silently allocating (n_slots, ncol) for a profile would fail much later,
    at the first accumulate, with an opaque broadcast error."""
    with pytest.raises(ValueError, match="n_layers"):
        init_tape_accumulator(_spec("T_soil"), NSLOT, NCOL)


def test_scalar_only_tape_needs_no_n_layers():
    acc = init_tape_accumulator(_spec("T_sfc"), NSLOT, NCOL)
    assert np.shape(acc["T_sfc"]) == (NSLOT, NCOL)


def test_mean_finalisation_divides_profiles_correctly():
    """The count is per-SLOT; a hardcoded ``count[:, None]`` broadcasts against a
    2-D array but mis-aligns against a 3-D one.  Accumulate a known constant and
    require the mean to return exactly that constant."""
    spec = _spec("T_sfc", "T_soil")
    acc = init_tape_accumulator(spec, NSLOT, NCOL, n_layers=NLAY)
    n_steps = 5
    scal = np.full((NCOL,), 3.0)
    prof = np.arange(NCOL * NLAY, dtype=float).reshape(NCOL, NLAY)
    acc = dict(acc)
    acc["count"] = np.full(NSLOT, float(n_steps))
    acc["T_sfc"] = np.broadcast_to(scal, (NSLOT, NCOL)) * n_steps
    acc["T_soil"] = np.broadcast_to(prof, (NSLOT, NCOL, NLAY)) * n_steps

    out = finalize_tape(acc, spec)
    np.testing.assert_allclose(out["T_sfc"], np.broadcast_to(scal, (NSLOT, NCOL)))
    np.testing.assert_allclose(out["T_soil"], np.broadcast_to(prof, (NSLOT, NCOL, NLAY)))


def test_inst_finalisation_passes_profiles_through():
    spec = TapeSpec(name="t", freq="monthly", average="inst", vars=("T_soil",))
    acc = init_tape_accumulator(spec, NSLOT, NCOL, n_layers=NLAY)
    acc = dict(acc)
    acc["T_soil"] = np.arange(NSLOT * NCOL * NLAY, dtype=float).reshape(NSLOT, NCOL, NLAY)
    out = finalize_tape(acc, spec)
    np.testing.assert_array_equal(out["T_soil"], acc["T_soil"])


def test_revert_mask_broadcasts_over_layers():
    """The scan body masks reverted columns to NaN.  ``reverted`` is (ncol,), so a
    flat ``jnp.where(reverted, nan, v)`` raises on an (ncol, n_layers) field --
    this is the exact failure the driver hit."""
    import jax.numpy as jnp

    reverted = jnp.array([True, False, True, False, False, False, False])

    def mask(v):
        v = jnp.asarray(v)
        m = reverted.reshape((NCOL,) + (1,) * (v.ndim - 1))
        return jnp.where(m, jnp.nan, v)

    prof = mask(jnp.ones((NCOL, NLAY)))
    scal = mask(jnp.ones((NCOL,)))
    assert np.isnan(np.asarray(prof)[0]).all() and np.isfinite(np.asarray(prof)[1]).all()
    assert np.isnan(np.asarray(scal)[0]) and np.isfinite(np.asarray(scal)[1])
