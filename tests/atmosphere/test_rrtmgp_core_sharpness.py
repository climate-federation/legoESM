"""Static ``sharpness`` of the RRTMGP gas-optics core (default inf).

inf is the unsmoothed path, bitwise; a finite value smooths the tropopause split
(p <= p_ref_tropo, a value jump in the tables) and the temperature clips to the
table range.  Callers other than the ecRad IFS lane never pass it.
"""
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import gas_optics as rgo
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

jax.config.update("jax_enable_x64", True)

S = 100.0


def _lib():
    return RRTMGP.from_legoesm_config(RRTMGPConfig(include_clouds=False)).optics_lib


def _cells(p, t):
    """(1, 1, n) state with the RRTMGP library's default gases."""
    p = jnp.asarray(p, jnp.float64)[None, None, :]
    t = jnp.broadcast_to(jnp.asarray(t, jnp.float64), p.shape)
    molecules = jnp.full_like(p, 2.0e23)
    return p, t, molecules


def _bound(lib, is_lw, p, t, sharpness):
    p, t, m = _cells(p, t)
    kw = {} if sharpness is None else {"sharpness": sharpness}
    b = lib.with_all_gpts(is_lw, p, t, m, None, **kw)
    names = ("tau_lw", "planck_fraction") if is_lw else ("tau_sw", "rayleigh")
    return {k: np.asarray(b._gas_all_gpts[k]) for k in names}


def _pt(lib, is_lw):
    return float((lib.gas_optics_lw if is_lw else lib.gas_optics_sw).p_ref_tropo)


# Frozen outputs of the core BEFORE the sharpness argument existed (main efa650f75e,
# fp64 CPU), on the profile built in _frozen_profile (votes K1: codex/GLM/author a).
FROZEN = Path(__file__).parent / "oracle_fixtures" / "rrtmgp_core_hard_main.npz"


def _frozen_profile():
    rng = np.random.default_rng(0)
    p = jnp.asarray(np.sort(np.exp(rng.uniform(np.log(1.0), np.log(1.1e5), (3, 1, 60))), -1))
    t = jnp.asarray(rng.uniform(140.0, 370.0, p.shape))      # crosses both table ends
    return p, t, jnp.full_like(p, 2.0e23)


def test_default_path_is_the_pre_change_core_bitwise():
    lib = _lib()
    p, t, m = _frozen_profile()
    out = {}
    for is_lw in (True, False):
        b = lib.with_all_gpts(is_lw, p, t, m, None)
        out.update({k: np.asarray(v) for k, v in b._gas_all_gpts.items()})
        if is_lw:
            for g in (0, 77, 255):
                src = b.compute_planck_sources(p, t, g, sfc_temperature=t[:, :, 0],
                                               level_temperature=t[:, :, 1:])
                out.update({f"{k}_{g}": np.asarray(v) for k, v in src.items()})
    with np.load(FROZEN) as ref:
        assert sorted(ref.files) == sorted(out)
        for k in ref.files:
            assert np.array_equal(ref[k], out[k]), k


def test_tropopause_split_continuous_when_smoothed():
    """The hard split jumps (measured up to 100 % per g-point); the blend is
    continuous: its jump across p_ref_tropo shrinks with the separation."""
    lib = _lib()

    def jumps(is_lw, eps, s):
        pt = _pt(lib, is_lw)
        out = _bound(lib, is_lw, np.array([pt * (1.0 + eps), pt * (1.0 - eps)]), 220.0, s)
        res = {}
        for k, v in out.items():
            a, b = v[..., 0], v[..., 1]
            res[k] = np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300))
        return res

    for is_lw in (True, False):
        hard = jumps(is_lw, 1e-11, math.inf)
        far, near = jumps(is_lw, 1e-9, S), jumps(is_lw, 1e-11, S)
        for k in hard:
            assert hard[k] > 1e-3, (is_lw, k, hard[k])                 # the hard form jumps
            assert near[k] < 0.05 * far[k] or far[k] < 1e-12, (is_lw, k, near[k], far[k])


def test_exact_outside_the_band():
    lib = _lib()
    for is_lw in (True, False):
        pt = _pt(lib, is_lw)
        half = 0.5 * 0.2 / S                       # ln p half-width of the blend
        p = pt * np.exp(np.array([-3.0, -2.0, 2.0, 3.0]) * half)
        p = np.concatenate([[0.5], p, [1.2e5]])   # also beyond both table ends
        a, b = _bound(lib, is_lw, p, 230.0, S), _bound(lib, is_lw, p, 230.0, math.inf)
        for k in a:
            assert np.array_equal(a[k], b[k]), (is_lw, k)


def test_temperature_clip_inside_the_table():
    t_ref = jnp.asarray(_lib().gas_optics_lw.t_ref)
    lo, hi = float(t_ref[0]), float(t_ref[-1])
    t = jnp.linspace(lo - 20.0, hi + 20.0, 4001)
    smooth = np.asarray(rgo._clip_to_table_range(t, t_ref, S))
    assert np.array_equal(np.asarray(rgo._clip_to_table_range(t, t_ref, math.inf)),
                          np.asarray(jnp.clip(t, lo, hi)))
    assert smooth.min() >= lo and smooth.max() <= hi           # never leaves the table
    assert np.all(np.diff(smooth) >= 0.0)                       # monotone
    inner = (np.asarray(t) >= lo * (1 + 1 / S)) & (np.asarray(t) <= hi * (1 - 1 / S))
    assert np.array_equal(smooth[inner], np.asarray(t)[inner])  # exact inside
    g = jax.vmap(jax.grad(lambda x: rgo._clip_to_table_range(x, t_ref, S)))(t)
    assert np.all(np.isfinite(np.asarray(g)))
    band = (np.asarray(t) > lo) & (np.asarray(t) < lo * (1 + 1 / S))
    assert np.all(np.asarray(g)[band] > 0.0)                    # smooth, not flat, in the band
    # vote K2: the band departs from the hard clip, and the slope is continuous at
    # each join (both ends of both bands)
    clip = lambda x, s: rgo._clip_to_table_range(jnp.asarray(x), t_ref, s)  # noqa: E731
    for mid in (lo * (1 + 0.5 / S), hi * (1 - 0.5 / S)):
        assert float(clip(mid, S)) != float(clip(mid, math.inf)), mid
    dclip = jax.grad(lambda x: clip(x, S))
    for j in (lo, lo * (1 + 1 / S), hi * (1 - 1 / S), hi):
        left, right = float(dclip(j - 1e-9)), float(dclip(j + 1e-9))
        assert abs(left - right) <= 1e-6, (j, left, right)
