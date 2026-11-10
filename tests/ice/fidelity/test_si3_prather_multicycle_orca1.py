"""NEMO 5.0.1 ORCA1 oracle for SI3 Prather with several advection cycles per ice step (icycle 2 and 3).

Fixture ``fixtures/orca1_t4/prather_multicycle_orca1.{npz,json}``: the tier-4 ORCA1_T4DYN build (NEMO 5.0.1, eORCA1,
nn_hls 2, F-pivot) at kt 17281 rerun with the restart's delayed CFL (``DELAY__r8_cflice``) set to 1.0 and 2.0, so
NEMO's own selector (``icedyn_adv_pra.F90:124-126``) runs icycle 2 and 3; NEMO code unchanged.  Provenance in the json.

1. Corrections inside every cycle (``DO jt = 1, icycle`` at :142 encloses Hbig_pra :407, Hsnow_pra :411,
   ice_var_zapneg :421 and the open-water update :424-429): one window per arm where applying them once after all
   cycles misses NEMO.  ``advect_si3_prather_with_cycle_corrections`` reproduces NEMO's state, open water and
   residual ledgers; the once-after composition does not.
2. North fold every cycle (lbc cadence :435-525 with ``ldfull``): the icycle-3 top band; the F-pivot refresh
   reproduces NEMO's fold-halo moments, the periodic refresh does not.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import (
    advect_si3_prather_with_cycle_corrections,
    apply_si3_prather_source_corrections,
)
from legoesm.ice.transport import (
    advect_si3_prather_2d,
    si3_prather_open_water,
    si3_prather_pack_intensives,
    si3_prather_unpack_intensives,
)

jax.config.update("jax_enable_x64", True)

_FIX = Path(__file__).resolve().parent / "fixtures" / "orca1_t4" / "prather_multicycle_orca1"
FX = dict(np.load(str(_FIX) + ".npz"))
H = 2
TR = ("v_i", "v_s", "a_i", "oa_i", "e_s_l01", "e_s_l02", "e_s_l03", "e_i_l01", "e_i_l02", "e_i_l03",
      "szv_i_l01", "szv_i_l02", "szv_i_l03", "a_ip", "v_ip", "v_il")
KEEP = ("v_i", "v_s", "a_i", "oa_i", "e_s_l01", "e_s_l02", "e_s_l03", "e_i_l01", "e_i_l02", "e_i_l03", "szv_i_l01")
RES = ("wfx_res", "hfx_res", "sfx_res")
ARMS = ("cyc2", "cyc3")
# Pass rules (votes T1 c, T2 a): state and open water rel <= 1e-6 on the scored cells (provisional: no per-arm seed
# spread yet, and the icycle-1 20-seed spread of the advected snow fields is 4.1e-9; measured error ~6e-11, the
# once-after composition O(1)); ledgers absolute <= 1e-12 x the arm's largest |NEMO ledger| on the scored cells.
REL_TOL = 1e-6
LEDGER_ABS = 1e-12


@pytest.fixture(autouse=True)
def _fp64_policy():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(old)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise FloatingPointError("non-finite operand in oracle comparison")
    den = np.maximum(np.abs(a), np.abs(b))
    return np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)


def _expand(x):
    out = np.zeros(x.shape[:-1] + (len(TR),))
    for i, t in enumerate(KEEP):
        out[..., TR.index(t)] = x[..., i]
    return out


def _inputs(arm):
    g = lambda k: jnp.asarray(FX[f"{arm}_{k}"])  # noqa: E731
    dt, kt, fsbc, icycle, _ = FX[f"{arm}_scalars"]
    return dict(
        X=jnp.asarray(_expand(FX[f"{arm}_X"])),
        MOM=tuple(jnp.asarray(_expand(m)) for m in FX[f"{arm}_MOM"]),
        u=g("u_ice"), v=g("v_ice"), area=g("e1e2t"), e2u=g("e2u"), e1v=g("e1v"),
        wet=jnp.asarray(FX[f"{arm}_tmask1"] > 0), ato=g("ato_i"),
        dt=float(dt), kt=int(kt), fsbc=int(fsbc), icycle=int(icycle),
    )


def _per_cycle(arm, *, hard=True, **over):
    d = _inputs(arm) | over
    kw = dict(limit_sharpness_scale=math.inf, negative_area_ramp_width=0.0, open_water_floor_sharpness=math.inf)
    s = advect_si3_prather_with_cycle_corrections(
        d["X"], d["MOM"], d["ato"], d["u"], d["v"], d["area"], d["wet"], d["dt"], dx=d["e1v"], dy=d["e2u"],
        ice_step_index=d["kt"], nn_fsbc=d["fsbc"], halo_width=H, subcycles=d["icycle"], north_fold="none",
        tracer_names=TR, nlay_i=3, nlay_s=3, ponds_active=False, **(kw if hard else {}))
    return s.intensives, s.open_water, {k: getattr(s.residual_fluxes, k) for k in RES}


def _once_after(arm):
    """The composition on main before this fix: all cycles advected, then corrections and open water once."""
    d = _inputs(arm)
    c, _, _ = advect_si3_prather_2d(
        si3_prather_pack_intensives(d["X"], d["area"]), d["MOM"], d["u"], d["v"], d["area"], d["wet"], d["dt"],
        dx=d["e1v"], dy=d["e2u"], ice_step_index=d["kt"], nn_fsbc=d["fsbc"], halo_width=H,
        ice_volume_index=TR.index("v_i"), concentration_index=TR.index("a_i"), subcycles=d["icycle"],
        limit_sharpness_scale=math.inf)
    raw = si3_prather_unpack_intensives(c, d["area"], d["wet"])
    cor, res = apply_si3_prather_source_corrections(
        d["X"], raw, tracer_names=TR, nlay_i=3, nlay_s=3, cell_area_m2=1.0, halo_width=H, ponds_active=False,
        entry_intensive_contents=d["X"], contents_are_intensive=True, negative_area_ramp_width=0.0, dt_s=d["dt"],
        return_residual_fluxes=True)
    ato = si3_prather_open_water(d["ato"], d["X"][..., TR.index("a_i")], cor[..., TR.index("a_i")], d["u"], d["v"],
                                 dy_u=d["e2u"], dx_v=d["e1v"], cell_area=d["area"], dt=d["dt"] / d["icycle"],
                                 floor_sharpness=math.inf)
    return cor, ato, {k: getattr(res, k) for k in RES}


_WIN = json.loads(Path(str(_FIX) + ".json").read_text())["window"]


def _inner(a):
    w, m = int(_WIN["W"]), int(_WIN["M"])
    assert a.shape[0] == w and a.shape[1] == w
    return a[m:w - m, m:w - m]


def _misses(arm, X, ato, res):
    X, ato, res = np.asarray(X), np.asarray(ato), {k: np.asarray(v) for k, v in res.items()}
    state = _rel(_inner(X)[..., [TR.index(t) for t in KEEP]], FX[f"{arm}_nemo_X"]).max()
    ow = _rel(_inner(ato), FX[f"{arm}_nemo_ato_i"]).max()
    led = max(float(np.abs(_inner(res[k]) - FX[f"{arm}_nemo_{k}"]).max() / np.abs(FX[f"{arm}_nemo_{k}"]).max())
              for k in RES)
    return float(state), float(ow), led


def test_fixture_provenance_and_cycle_counts():
    meta = json.loads(Path(str(_FIX) + ".json").read_text())
    assert meta["oracle"].startswith("NEMO 5.0.1")
    assert [int(FX[f"{a}_scalars"][3]) for a in ARMS] == [2, 3]
    assert int(FX["fold_scalars"][3]) == 3
    for arm in ARMS:   # every ledger binds on the scored cells (the fix is exercised, not a null case)
        assert all(np.any(FX[f"{arm}_nemo_{k}"] != 0.0) for k in RES), arm


@pytest.mark.parametrize("arm", ARMS)
def test_corrections_inside_every_cycle_match_nemo(arm):
    X, ato, res = _per_cycle(arm)
    state, ow, led = _misses(arm, X, ato, res)
    assert state <= REL_TOL and ow <= REL_TOL and led <= LEDGER_ABS, (state, ow, led)


@pytest.mark.parametrize("arm", ARMS)
def test_corrections_once_after_all_cycles_miss_nemo(arm):
    state, ow, led = _misses(arm, *_once_after(arm))
    assert max(state, ow, led) > 1e-3, (state, ow, led)


def test_icycle1_per_cycle_equals_once_after():
    # One cycle: the per-cycle composition IS the previous composition (bitwise on the window).
    d = _inputs("cyc2")
    over = dict(icycle=1)
    X1, ato1, res1 = _per_cycle("cyc2", **over)
    c, _, _ = advect_si3_prather_2d(
        si3_prather_pack_intensives(d["X"], d["area"]), d["MOM"], d["u"], d["v"], d["area"], d["wet"], d["dt"],
        dx=d["e1v"], dy=d["e2u"], ice_step_index=d["kt"], nn_fsbc=d["fsbc"], halo_width=H,
        ice_volume_index=TR.index("v_i"), concentration_index=TR.index("a_i"), subcycles=1,
        limit_sharpness_scale=math.inf)
    raw = si3_prather_unpack_intensives(c, d["area"], d["wet"])
    cor, res = apply_si3_prather_source_corrections(
        d["X"], raw, tracer_names=TR, nlay_i=3, nlay_s=3, cell_area_m2=1.0, halo_width=H, ponds_active=False,
        entry_intensive_contents=d["X"], contents_are_intensive=True, negative_area_ramp_width=0.0, dt_s=d["dt"],
        return_residual_fluxes=True)
    ato = si3_prather_open_water(d["ato"], d["X"][..., TR.index("a_i")], cor[..., TR.index("a_i")], d["u"], d["v"],
                                 dy_u=d["e2u"], dx_v=d["e1v"], cell_area=d["area"], dt=d["dt"],
                                 floor_sharpness=math.inf)
    assert np.array_equal(np.asarray(_inner(X1)), np.asarray(_inner(cor)))
    assert np.array_equal(np.asarray(_inner(ato1)), np.asarray(_inner(ato)))
    assert all(np.array_equal(np.asarray(_inner(res1[k])), np.asarray(_inner(getattr(res, k)))) for k in RES)


def _fold_moments(north_fold):
    g = lambda k: jnp.asarray(FX["fold_" + k])  # noqa: E731
    dt, kt, fsbc, icycle, _ = FX["fold_scalars"]
    area = g("e1e2t")
    wet = jnp.asarray(FX["fold_tmask1"] > 0)
    _, m, _ = advect_si3_prather_2d(
        si3_prather_pack_intensives(g("X"), area), tuple(jnp.asarray(q) for q in FX["fold_MOM"]), g("u_ice"),
        g("v_ice"), area, wet, float(dt), dx=g("e1v"), dy=g("e2u"), ice_step_index=int(kt), nn_fsbc=int(fsbc),
        halo_width=H, ice_volume_index=0, concentration_index=1, subcycles=int(icycle), north_fold=north_fold,
        limit_sharpness_scale=math.inf)
    return np.stack([np.asarray(q) for q in m], 0)


def _fold_scores(mom):
    s_rows = int(FX["fold_scalars"][4])
    want = FX["fold_nemo_MOM"]
    halo = _rel(mom[:, H:-H, -H:], want[:, H:-H, -H:]).max()
    rows = _rel(mom[:, H:-H, -H - s_rows:-H], want[:, H:-H, -H - s_rows:-H]).max()
    return float(halo), float(rows)


def test_fold_refresh_every_cycle_matches_nemo_icycle3():
    halo, rows = _fold_scores(_fold_moments("orca_f_pivot"))
    assert rows <= REL_TOL and halo <= REL_TOL, (halo, rows)


def test_periodic_refresh_misses_nemo_at_the_fold_icycle3():
    halo, rows = _fold_scores(_fold_moments("none"))
    assert halo > 1e-3, (halo, rows)


def test_gradient_through_cycled_step_is_finite_and_matches_fd():
    # Smooth production defaults (no hard limits); d(total snow volume)/d(u scale) vs a central difference.
    d = _inputs("cyc2")

    def f(scale):
        X, ato, res = _per_cycle("cyc2", hard=False, u=d["u"] * scale)
        return jnp.sum(_inner(X)[..., TR.index("v_s")]) + jnp.sum(_inner(ato)) + jnp.sum(_inner(res["wfx_res"]))

    g = float(jax.grad(lambda s: f(s))(1.0))
    eps = 1e-6
    fd = (float(f(1.0 + eps)) - float(f(1.0 - eps))) / (2 * eps)
    assert np.isfinite(g) and abs(fd) > 0.0 and abs(g - fd) <= 1e-4 * abs(fd), (g, fd)
