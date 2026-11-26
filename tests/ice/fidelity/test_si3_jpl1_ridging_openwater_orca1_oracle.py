"""SI3 jpl=1 ridging open water vs the NEMO 5.0.1 ORCA1 tier-4 record.

NEMO keeps every listed cell in the ridging loop (icedyn_rdgrft.F90:285-320) and
updates open water for all of them (:651-654) with the opening clamp
(:594-602) and the converged reset (:304-308).  The port used to update open
water only while gross closing > 0, so cells whose entry ``ato_i + a_i`` was
below one stopped after one pass.

Fixture ``fixtures/orca1_t4/rdg_openwater.{npz,json}``: 37 cells of the ORCA1
step (provenance in the JSON).  The record ran snow/pond retention 0.5 while
this arm's validator pins 1.0; retention scales only the snow and pond
inventories, so only retention-independent fields are scored.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice.ridging import (
    SI3JPL1RidgingConfig,
    SI3JPL1RidgingState,
    apply_si3_jpl1_ridging,
)

_FIX = Path(__file__).parent / "fixtures" / "orca1_t4" / "rdg_openwater.npz"
_OPEN_WATER_REL_TOL = 1.0e-13  # vote V2 (fixture max 5.4e-14, NEMO roundoff cell)
_ICE_REL_TOL = 1.0e-14  # vote V2 (fixture max 2.2e-16)
_CLOSURE_TOL = 1.0e-10  # NEMO epsi10 convergence test, icedyn_rdgrft.F90:303
# NEMO hard form (oracle mode).  A config without the field is already hard
# (pre-smoothing code); the fallback keeps per-test non-vacuity readable.
_HARD = (SI3JPL1RidgingConfig(open_water_limit_sharpness=math.inf)
         if "open_water_limit_sharpness" in SI3JPL1RidgingConfig._fields
         else SI3JPL1RidgingConfig())
_STORED_GAP_ABS_TOL = 1.0e-12  # vote W3
_SMOOTH_GAP_MAX = 5.0e-3  # vote W3: smooth-default open-water gap vs NEMO (fixture 1.48e-3)


@pytest.fixture(autouse=True)
def _fp64_policy():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(old)


def _load():
    return dict(np.load(_FIX))


def _state(d, divergence_scale=1.0):
    st = SI3JPL1RidgingState(
        ice_area=jnp.asarray(d["a_i"]), open_water_area=jnp.asarray(d["ato_i"]),
        ice_volume=jnp.asarray(d["v_i"]), snow_volume=jnp.asarray(d["v_s"]),
        age_content=jnp.asarray(d["oa_i"]), pond_area=jnp.asarray(d["a_ip"]),
        pond_volume=jnp.asarray(d["v_ip"]), pond_lid_volume=jnp.asarray(d["v_il"]),
        snow_enthalpy=jnp.asarray(d["e_s"]), ice_enthalpy=jnp.asarray(d["e_i"]),
        ice_salt_content=jnp.asarray(d["szv_i"]))
    return st, jnp.asarray(d["divu_i"]) * divergence_scale, jnp.asarray(d["delta_i"])


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    den = np.maximum(np.abs(a), np.abs(b))
    return np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)


def test_open_water_matches_nemo_record() -> None:
    d = _load()
    out, _ = apply_si3_jpl1_ridging(*_state(d), float(d["rDt_ice"]), config=_HARD)
    r = _rel(out.open_water_area, d["ato_i_out"])
    assert r.max() <= _OPEN_WATER_REL_TOL, (r.max(), int(np.argmax(r)))
    # the cells the old gate missed are in the fixture and move by O(1e-2)
    assert _rel(d["base_ato_i_out"], d["ato_i_out"]).max() > 1.0e-2


def test_retention_independent_ice_fields_match_nemo_record() -> None:
    d = _load()
    out, _ = apply_si3_jpl1_ridging(*_state(d), float(d["rDt_ice"]), config=_HARD)
    for port, key in ((out.ice_area, "a_i_out"), (out.ice_volume, "v_i_out"),
                      (out.age_content, "oa_i_out"), (out.ice_enthalpy, "e_i_out")):
        assert _rel(port, d[key]).max() <= _ICE_REL_TOL, key


def test_listed_cells_close_the_area_like_nemo() -> None:
    d = _load()
    out, losses = apply_si3_jpl1_ridging(*_state(d), float(d["rDt_ice"]), config=_HARD)
    closure = np.abs(1.0 - (np.asarray(out.open_water_area) + np.asarray(out.ice_area)))
    nemo_closure = np.abs(1.0 - (d["ato_i_out"] + d["a_i_out"]))
    listed = np.asarray(losses.iterations) >= 1
    # NEMO-converged cells close to epsi10; deficit cells still open after the
    # 19 passes (each pass adds at most a_i of open water) keep NEMO's residual
    converged = listed & (nemo_closure < _CLOSURE_TOL)
    assert converged.sum() >= 10 and (listed & ~converged).sum() >= 3
    assert np.all(closure[converged] < _CLOSURE_TOL)
    assert np.max(np.abs(closure - nemo_closure)[listed]) < 1.0e-12
    # cells outside NEMO's ridging list keep their entry open water
    assert np.array_equal(np.asarray(out.open_water_area)[~listed], d["ato_i"][~listed])


def test_open_water_update_leaves_ice_inventories_unchanged() -> None:
    """Ice volume / heat / salt equal the NEMO record bitwise-to-roundoff and
    are invariant under the open-water continuation (zero ridge porosity)."""
    d = _load()
    st, div, dlt = _state(d)
    out, _ = apply_si3_jpl1_ridging(st, div, dlt, float(d["rDt_ice"]), config=_HARD)
    for port, entry in ((out.ice_volume, st.ice_volume), (out.ice_enthalpy, st.ice_enthalpy),
                        (out.ice_salt_content, st.ice_salt_content)):
        assert _rel(port, entry).max() <= _ICE_REL_TOL


def test_open_water_gradient_finite_and_matches_fd() -> None:
    d = _load()
    dt = float(d["rDt_ice"])

    def f(scale):
        st, div, dlt = _state(d, scale)
        return apply_si3_jpl1_ridging(st, div, dlt, dt, config=_HARD)[0].open_water_area

    _, tangent = jax.jvp(f, (jnp.float64(1.0),), (jnp.float64(1.0),))
    g = jax.grad(lambda s: jnp.sum(f(s)))(jnp.float64(1.0))
    assert np.all(np.isfinite(np.asarray(tangent))) and np.isfinite(float(g))
    # The 19-pass opening clamp makes open water piecewise smooth in the
    # divergence; kinks sit within ~1e-7 relative of the record state, so the
    # central difference uses a step below them.
    eps = 1.0e-9
    fd = (np.asarray(f(jnp.float64(1.0 + eps))) - np.asarray(f(jnp.float64(1.0 - eps)))) / (2 * eps)
    t = np.asarray(tangent)
    big = np.abs(t) > 1.0e-6
    assert big.sum() >= 8
    ratio = np.abs(fd[big] / t[big] - 1.0)
    assert np.median(ratio) < 1.0e-4 and np.max(ratio) < 5.0e-3


def test_fixture_provenance_present() -> None:
    meta = json.loads(_FIX.with_suffix(".json").read_text())
    assert "ORCA1_T4DYN" in meta["source"] and "t4_rdg_record" in meta["record"]
    assert meta["cell_classes"]["deficit_base_miss"] >= 10


def test_smooth_default_reproduces_stored_gap_vs_nemo() -> None:
    """Default (smooth) open-water limits: class B gap vs NEMO, stored per cell."""
    d = _load()
    out, _ = apply_si3_jpl1_ridging(*_state(d), float(d["rDt_ice"]))
    for port, key in ((out.open_water_area, "smooth_ato_i_out"), (out.ice_area, "smooth_a_i_out"),
                      (out.ice_volume, "smooth_v_i_out"), (out.ice_enthalpy, "smooth_e_i_out")):
        bound = _STORED_GAP_ABS_TOL * max(1.0, np.abs(d[key]).max())
        assert np.max(np.abs(np.asarray(port) - d[key])) <= bound, key
    gap = np.abs(np.asarray(out.open_water_area) - d["ato_i_out"])
    assert 1.0e-6 < gap.max() <= _SMOOTH_GAP_MAX
    # the ramp leaves ice volume / heat at NEMO's values
    assert _rel(out.ice_volume, d["v_i_out"]).max() <= _ICE_REL_TOL
    assert _rel(out.ice_enthalpy, d["e_i_out"]).max() <= _ICE_REL_TOL
    # planted violation: the hard output must NOT reproduce the stored smooth gap
    hard, _ = apply_si3_jpl1_ridging(*_state(d), float(d["rDt_ice"]), config=_HARD)
    assert np.max(np.abs(np.asarray(hard.open_water_area) - d["smooth_ato_i_out"])) > 1.0e-6


def test_smooth_default_check_grads() -> None:
    d = _load()
    dt = float(d["rDt_ice"])

    def f(scale):
        st, div, dlt = _state(d, scale)
        return apply_si3_jpl1_ridging(st, div, dlt, dt)[0].open_water_area

    check_grads(f, (jnp.float64(1.0),), order=1, modes=("fwd", "rev"), eps=1.0e-9)
    # non-vacuity: the derivative is nonzero on cells the ramp actually moved
    _, tangent = jax.jvp(f, (jnp.float64(1.0),), (jnp.float64(1.0),))
    in_band = np.abs(d["smooth_ato_i_out"] - d["ato_i_out"]) > 1.0e-9
    assert in_band.sum() >= 3
    assert np.all(np.isfinite(np.asarray(tangent)))
    assert np.max(np.abs(np.asarray(tangent)[in_band])) > 1.0e-9


def test_sharpness_validation() -> None:
    d = _load()
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="open_water_limit_sharpness"):
            apply_si3_jpl1_ridging(*_state(d), float(d["rDt_ice"]),
                                   config=SI3JPL1RidgingConfig(open_water_limit_sharpness=bad))
    assert SI3JPL1RidgingConfig().open_water_limit_sharpness == 100.0
