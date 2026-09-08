"""What the centred averaging window damps, and where it does NOT damp more.

WHY THIS EXISTS. Re-centring the split-explicit barotropic averaging window on
t+dt fixed a half-step phase error that made external gravity waves propagate
~2x too slow. It also destabilised one case: a 200-day Eady run on an
unstructured MPAS channel, where the free surface reaches 5192 m against a
100 m blow-up threshold (isolated to that single commit, reproduced
bit-identically on origin/main + it alone).

The leading hypothesis from review was that the OLD window carried incidental
damping at small scales which the case relied on, and that the fix removed it.
MEASURED HERE, and the hypothesis is refuted AT THE PERIOD THE COUPLING LIVES
AT -- two baroclinic steps -- where the centred window damps MORE, not less
(cosine 0.849 -> 0.500, box 0.637 -> 0.017).

It is NOT refuted across the spectrum, and an earlier version of this docstring
said so wrongly (codex 2026-08-13). At the substep Nyquist and at one baroclinic
step the BOX window is LESS damping after the change (0 -> 1/59), because an
even-length box sums (-1)^j to exactly zero and an odd-length one to 1. The
production MPAS path uses the COSINE kernel, where the new window is at least as
damping everywhere probed -- which is what keeps the conclusion standing for the
case in question, but the general claim does not.

The window is a filter in TIME over substeps, so the frequency axis is cycles
per substep: 0.5 is the substep Nyquist, 1/n is one baroclinic step.

THE COMPLETED SUBSTEP SWEEP (n = 30 / 60 / 120, one variable):

    n     FB substeps per window   CFL     outcome
    30            59               0.033   max|eta| 5192 m
    60           119               0.017   NaN
   120           239               0.008   NaN

The averaging window spans 2n-1 substeps of dt_s = dt/n, so the total
excursion is ~2*dt for ALL of them -- only the STEP COUNT changes. The
failure gets monotonically worse as the steps get smaller and more
numerous, which is the opposite of what a CFL problem does.

PLAUSIBLE, not established: a defect that worsens with step COUNT at fixed
elapsed time is a per-step process rather than a per-time one -- for
example reconstruction noise injected once per substep on the distorted
Voronoi cells, or a per-step leak in a momentum/Coriolis pair that is not
exactly energy-conserving under forward-backward substepping. Separating
those needs the free surface's spatial structure just before onset.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# x64 EXPLICITLY: the weights are requested as float64 and the pinned
# values are compared at 1e-7, which fp32 cannot honour (codex 2026-08-13).
import jax
jax.config.update("jax_enable_x64", True)

jnp = pytest.importorskip("jax.numpy")

if jnp.zeros(1, dtype=jnp.float64).dtype != jnp.float64:   # pragma: no cover
    pytest.skip("x64 unavailable; the pinned tolerances are meaningless "
                "in fp32", allow_module_level=True)

_REPO = Path(__file__).resolve().parents[3]
_MODULE = "packages/ocean/legoesm/ocean/dynamics/barotropic_common.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _frozen_old_weights(n, use_cosine):
    """The PRE-2026-08-12 window, frozen as a formula.

    NOT read from origin/main. An earlier version of this file did that, and
    it is a landmine: once this branch merges, origin/main CONTAINS the new
    code, so the comparison silently becomes new-against-new and the length
    tripwire below turns into a permanent spurious failure (codex
    2026-08-13). The old construction is four lines; freezing it is what
    makes "not less damping than before" a durable statement.

        w_i = 1 + cos(2*pi*(i - n/2)/n)   (cosine)   i = 0 .. n-1
        w_i = 1                           (box)
    """
    i = np.arange(n, dtype=np.float64)
    if use_cosine:
        return 1.0 + np.cos(2.0 * np.pi * (i - n / 2.0) / n)
    return np.ones(n, dtype=np.float64)


def _response(w, freq_cycles_per_substep):
    """|W(f)| of the averaging kernel h = w/sum(w), normalised to 1 at DC.

    Magnitude only, so the time origin is irrelevant: shifting an FIR
    filter's origin multiplies the response by a phase factor that abs()
    removes. Comparing two windows of DIFFERENT length at the same f is
    legitimate because dt_s is unchanged by the fix, so a given f is the
    same physical period for both -- the differing support lengths are
    precisely why their attenuation differs (codex confirmed).
    """
    w = np.asarray(w, dtype=np.float64)
    tot = float(np.sum(w))
    assert tot > 0.0, "a zero-sum window has no DC-normalised response"
    j = np.arange(w.size)
    return float(abs(np.sum(w * np.exp(-2j * np.pi
                                       * freq_cycles_per_substep * j))) / tot)


_N = 30
#: MEASURED 2026-08-13 at n = 30. The claim this file exists to support is
#: the FIRST row: at the period the barotropic-baroclinic coupling lives
#: at, the centred window damps MORE, so the regression is not explained by
#: the fix having removed damping there.
#:
#: The other rows are why that claim is NARROW, and an earlier version of
#: this docstring overclaimed it as "every frequency that matters" (codex
#: 2026-08-13). At the substep Nyquist and at one baroclinic step the BOX
#: window is LESS damping after the change -- 0 -> 1/59 -- because an even
#: box sums (-1)^j to zero and an odd one to 1. Tiny in absolute terms, but
#: it means the lost-damping hypothesis is refuted at the 2-dt period ONLY,
#: not across the spectrum.
_EXPECTED = {
    # (use_cosine, freq)          (old, new)
    (True,  0.5 / _N): (0.84882764, 0.50000000),   # period 2*dt  <- the claim
    (False, 0.5 / _N): (0.63691075, 0.01694915),
    (True,  1.0 / _N): (0.50000000, 0.00000000),   # period dt
    (False, 1.0 / _N): (0.00000000, 0.01694915),   # new LESS damping here
    (True,  0.5):      (0.00000000, 0.00000000),   # substep Nyquist
    (False, 0.5):      (0.00000000, 0.01694915),   # new LESS damping here
}


def _new_weights(use_cosine, n=_N):
    mod = _load(_REPO / _MODULE, "_baro_new")
    return np.asarray(mod.compute_filter_weights(
        n, jnp.float64, use_cosine=use_cosine)[0], dtype=np.float64)


@pytest.mark.parametrize("key", sorted(_EXPECTED))
def test_the_measured_response_is_pinned(key):
    """The four-decimal values, not just their ordering.

    Without this the file supports only the SIGN of the comparison, and the
    numbers quoted in the commit message and the docs are unbacked.
    """
    use_cosine, freq = key
    exp_old, exp_new = _EXPECTED[key]
    got_old = _response(_frozen_old_weights(_N, use_cosine), freq)
    got_new = _response(_new_weights(use_cosine), freq)
    assert got_old == pytest.approx(exp_old, abs=1e-7), key
    assert got_new == pytest.approx(exp_new, abs=1e-7), key


@pytest.mark.parametrize("use_cosine", [False, True])
def test_more_damping_at_the_coupling_period(use_cosine):
    """THE CLAIM: at a period of two baroclinic steps the centred window
    passes less, so the regression is not a lost-damping story THERE."""
    f = 0.5 / _N
    r_old = _response(_frozen_old_weights(_N, use_cosine), f)
    r_new = _response(_new_weights(use_cosine), f)
    assert r_new < r_old, (
        f"centred window passes MORE at the 2-dt period "
        f"({r_new:.6f} vs {r_old:.6f})")


def test_the_box_window_is_LESS_damping_at_the_substep_nyquist():
    """The counter-example that bounds the claim above.

    Stated as a test so the narrowness cannot quietly disappear from the
    story: an even-length box kills the Nyquist exactly, an odd-length one
    leaves 1/N.
    """
    r_old = _response(_frozen_old_weights(_N, False), 0.5)
    r_new = _response(_new_weights(False), 0.5)
    assert r_old == pytest.approx(0.0, abs=1e-12)
    assert r_new == pytest.approx(1.0 / (2 * _N - 1), abs=1e-9)
    assert r_new > r_old


@pytest.mark.parametrize("use_cosine", [False, True])
def test_the_window_is_longer_positive_and_consistently_normalised(use_cosine):
    """2n-1 substeps, strictly positive, and w_total IS the sum."""
    mod = _load(_REPO / _MODULE, "_baro_new")
    w, w_total = (np.asarray(x, dtype=np.float64) for x in
                  mod.compute_filter_weights(_N, jnp.float64,
                                             use_cosine=use_cosine)[:2])
    assert w.size == 2 * _N - 1
    assert np.all(np.isfinite(w)) and np.all(w > 0.0), (
        "the centred cosine window must be strictly positive -- that is what "
        "removed the old n<2 degeneracy where 1+cos(-pi) == 0")
    # The accumulator is normalised by w_total downstream, so THAT is the
    # invariant, not sum(w) == 1 (codex 2026-08-13).
    assert float(w_total) == pytest.approx(float(w.sum()), rel=1e-12)
    assert float((w / w_total).sum()) == pytest.approx(1.0, rel=1e-12)


def test_the_response_probe_is_not_vacuous():
    """A pass-through filter scores 1 everywhere; a two-point mean kills the
    Nyquist. Without this the comparisons could be measuring nothing."""
    assert _response(np.array([1.0]), 0.5) == pytest.approx(1.0)
    assert _response(np.array([0.5, 0.5]), 0.5) == pytest.approx(0.0, abs=1e-12)
