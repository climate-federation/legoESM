"""Units contract for the ``seg_precip`` coupling key.

The atmosphere carry accumulates precipitation over a segment
(``precip_accum += precip_step * dt``, compiled_segments.py:1197) in kg/m2,
but the coupler contract ``AtmToSurface.precip_total``
(packages/core/legoesm/core/coupling_fields.py:18) is a RATE in kg/m2/s,
positive-downward into the surface.  The compiled driver path published the RAW
ACCUMULATION under ``_carry_aux['seg_precip']``, which both ``CoupledESMDriver``
(:1556) and ``EarthSystemDriver`` (:205) then handed to the coupler as a rate --
too large by the segment duration in seconds (~8.64e4x for a 1-day segment).

These tests are pure: they never build a grid, a model or a segment.  They do
import jax, so per the login-node policy run them under sbatch/srun, not on the
head node -- the AST tripwire at the bottom is the only part that is pure
Python.

Tolerances: JAX defaults to float32 unless JAX_ENABLE_X64=1, and the
accumulation loops here run up to 336 sequential float32 adds, whose relative
round-off reaches ~1e-5.  rtol is set accordingly; a tighter rtol would make
these tests fail on the very fix they are guarding.
"""
import ast
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.driver.compiled_segments import segment_accum_to_rate


def _model_driver_source() -> str:
    import legoesm.driver.model_driver as md
    return pathlib.Path(md.__file__).read_text()


def _model_driver_code() -> str:
    """``model_driver.py`` with comments and docstrings stripped.

    The "no open-coded normalisation" assertion below is a source grep, so it
    must look at CODE only: the fix's own explanatory comment legitimately
    quotes the old ``seg_precip / _seg_dur`` expression to say why it moved,
    and a raw-text grep flags that quotation as a violation.  Tokenising and
    dropping COMMENT/STRING tokens keeps the assertion strict about code while
    letting the code document itself.
    """
    import io
    import tokenize
    src = _model_driver_source()
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):  # pragma: no cover
        return src  # degrade to the raw text rather than silently passing
    return " ".join(out)


# ----------------------------------------------------------------------
# 1. The normalizer inverts the accumulator, for any segment shape.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("seg_steps,dt", [
    (1, 1800.0),      # one step
    (48, 1800.0),     # 1 day @ dt=1800
    (24, 3600.0),     # 1 day @ dt=3600
    (288, 300.0),     # 1 day @ dt=300  (same day, different seg_steps)
    (7 * 48, 1800.0), # 1 week
    (13, 1800.0),     # short trailing segment (n_steps_total not a multiple)
    (96, 900.0),      # after one adaptive-dt halving
])
def test_constant_rain_recovers_the_true_rate(seg_steps, dt):
    """A constant-rain segment must yield the TRUE rain rate, independent of
    seg_steps / dt (i.e. independent of diag_days and of any adaptive-dt
    halving).  Reproduces the real accumulator expression from
    ``compiled_segments.split_physics_single_rank`` line 1197.
    """
    true_rate = jnp.asarray([1.0e-5, 3.5e-5, 0.0, 1.2e-4])  # kg/m2/s

    accum = jnp.zeros_like(true_rate)
    for _ in range(seg_steps):
        accum = accum + true_rate * dt          # <- the model's accumulator

    # Sanity: the accumulator really is an accumulation, not a rate.
    np.testing.assert_allclose(
        np.asarray(accum), np.asarray(true_rate) * seg_steps * dt,
        rtol=1e-4, atol=1e-12)

    rate = segment_accum_to_rate(accum, seg_steps, dt)
    np.testing.assert_allclose(np.asarray(rate), np.asarray(true_rate),
                               rtol=1e-4, atol=1e-12)


def test_rate_is_invariant_across_segment_lengths():
    """Same physical rain, two different segment decompositions -> same rate.
    This is the 'independent of diag_days / seg_steps' requirement."""
    true_rate = jnp.asarray([2.0e-5, 4.0e-5])

    def _accum(seg_steps, dt):
        a = jnp.zeros_like(true_rate)
        for _ in range(seg_steps):
            a = a + true_rate * dt
        return segment_accum_to_rate(a, seg_steps, dt)

    np.testing.assert_allclose(np.asarray(_accum(48, 1800.0)),
                               np.asarray(_accum(288, 300.0)),
                               rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(np.asarray(_accum(48, 1800.0)),
                               np.asarray(true_rate), rtol=1e-4, atol=1e-12)


def test_budget_closes_over_the_segment():
    """rate * duration == accum to round-off: a consumer integrating this rate
    over the SAME duration (coupled_esm_driver._segment_hook sub-cycles
    dt_segment = seg_steps * DT into n_sub steps of sub_dt) re-integrates
    exactly the water the atmosphere rained out.

    SCOPE: this closes the INTERFACE budget only when the consumer's window
    equals the segment.  It does NOT certify the end-to-end coupled water
    budget, because the coupler callback fires on the DIAG cadence while
    _carry_aux is rewritten every SEGMENT (model_driver.py:9000 vs :9073).
    """
    seg_steps, dt = 48, 1800.0
    true_rate = jnp.asarray([3.0e-5, 0.0, 9.9e-5])
    accum = true_rate * seg_steps * dt
    rate = segment_accum_to_rate(accum, seg_steps, dt)
    np.testing.assert_allclose(np.asarray(rate) * (seg_steps * dt),
                               np.asarray(accum), rtol=1e-6, atol=1e-12)


def test_zero_duration_raises():
    with pytest.raises(ValueError, match="positive"):
        segment_accum_to_rate(jnp.zeros(3), 0, 1800.0)


def test_negative_duration_raises():
    with pytest.raises(ValueError, match="positive"):
        segment_accum_to_rate(jnp.zeros(3), 4, -300.0)


def test_dtype_is_preserved():
    """The divisor is a static Python float, so no dtype promotion and no
    retrace: a float32 accumulator stays float32."""
    accum = jnp.zeros(4, dtype=jnp.float32)
    assert segment_accum_to_rate(accum, 48, 1800.0).dtype == jnp.float32


# ----------------------------------------------------------------------
# 2. Tripwire: the compiled driver must publish the RATE, not the accumulation.
#    This is the test that FAILS before the fix (the dict binds the bare Name
#    ``seg_precip``, the raw accumulation) and PASSES after (it binds a call to
#    ``segment_accum_to_rate``).  A tripwire, not a proof -- it cannot run the
#    model, so it asserts the binding rather than the number.  Pure Python +
#    ast, no jax execution: safe to run anywhere.
# ----------------------------------------------------------------------
def test_compiled_driver_publishes_seg_precip_as_a_rate():
    tree = ast.parse(_model_driver_source())

    compiled_producers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        # Discriminator for the COMPILED-path carry_aux literal: it is the only
        # seg_precip writer that also carries the mass/moisture targets.  The
        # other two writers (_sfc_diag[2] at :5802 -- a Subscript assignment,
        # not a Dict -- and phys_out.precip at :9744, whose dict lacks
        # target_moisture) already store rates and must keep binding a bare
        # name.  Verified unique against the current tree.
        if {"seg_precip", "target_moisture", "seg_shflx"} <= keys:
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "seg_precip":
                    compiled_producers.append(v)

    assert len(compiled_producers) == 1, (
        "expected exactly one compiled-path carry_aux literal writing "
        f"'seg_precip'; found {len(compiled_producers)}"
    )

    value = compiled_producers[0]
    assert isinstance(value, ast.Call), (
        "_carry_aux['seg_precip'] binds the RAW segment accumulation "
        "[kg/m2]; the coupler contract AtmToSurface.precip_total is a RATE "
        "[kg/m2/s] (core/coupling_fields.py:18). It must be normalised by "
        "the segment duration at the producer."
    )
    fname = (value.func.id if isinstance(value.func, ast.Name)
             else getattr(value.func, "attr", None))
    assert fname == "segment_accum_to_rate", (
        f"_carry_aux['seg_precip'] is normalised by {fname!r}; expected the "
        "shared helper segment_accum_to_rate(accum, seg_steps, dt) so the "
        "coupled and diagnostic precip rates cannot diverge."
    )


def test_diagnostic_and_coupled_precip_use_the_same_normalizer():
    """Both the CMOR diagnostic rate and the coupler-facing rate must go
    through the shared helper, or a future edit can silently desynchronise
    the number written to output from the number handed to the ocean."""
    src = _model_driver_source()
    assert src.count("segment_accum_to_rate(seg_precip") >= 2, (
        "expected segment_accum_to_rate to be used for BOTH the carry_aux "
        "coupling entry and the seg_precip_rate diagnostic"
    )
    # CODE only -- see _model_driver_code: the fix's comment quotes the old
    # expression on purpose, and a raw-text grep would flag that quotation.
    code = _model_driver_code()
    assert "seg_precip / _seg_dur" not in code, (
        "the open-coded seg_precip / _seg_dur normalisation should now route "
        "through segment_accum_to_rate"
    )
