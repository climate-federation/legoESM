"""SI3 Prather open water + residual ocean ledgers vs NEMO 5.0.1 ORCA1 (oracle-audit tier 4, U10, defect D2).

Fixture: three 16x16 windows of the recorded ORCA1 ice step (``fixtures/si3_prather_open_water_residuals_orca1.json``
holds provenance); inner 6x6 cells are scored (bitwise equal to the full-domain port at extraction).  Tolerance (vote
V4 b): per cell ``rel <= max(2 x NEMO 20-seed 1-ulp spread, 1e-13)``.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import apply_si3_prather_source_corrections
from legoesm.ice.transport import (
    SI3_PRA_OPEN_WATER_FLOOR_SHARPNESS,
    advect_si3_prather_2d,
    si3_prather_open_water,
    si3_prather_pack_intensives,
    si3_prather_unpack_intensives,
)

from legoesm import constants

FIX = np.load(Path(__file__).parent / "fixtures" / "si3_prather_open_water_residuals_orca1.npz")
TR = tuple(str(t) for t in FIX["tracers"])
DT, KT, NN_FSBC, ICYCLE = (float(x) for x in FIX["scalars"])
H, M, W = 2, 5, 16
INNER = (slice(M, W - M), slice(M, W - M))
WINDOWS = (0, 1, 2)
NEW = ("ato_i", "wfx_res", "hfx_res", "sfx_res")
SNOW = [TR.index(f"e_s_l{k:02d}") for k in (1, 2, 3)]
ICE_E = [TR.index(f"e_i_l{k:02d}") for k in (1, 2, 3)]
SALT = [TR.index(f"szv_i_l{k:02d}") for k in (1, 2, 3)]


@pytest.fixture(autouse=True)
def _fp64_policy():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(old)


def _window(w):
    return {k: FIX[f"w{w}_{k}"] for k in ("X", "MOM", "u_ice", "v_ice", "e1e2t", "e2u", "e1v", "tmask1", "ato_i")}


def _port(f, *, sharpness=math.inf):
    x = jnp.asarray(f["X"])
    area = jnp.asarray(f["e1e2t"])
    wet = jnp.asarray(f["tmask1"] > 0)
    u, v = jnp.asarray(f["u_ice"]), jnp.asarray(f["v_ice"])
    contents, _, n = advect_si3_prather_2d(
        si3_prather_pack_intensives(x, area), tuple(jnp.asarray(m) for m in f["MOM"]), u, v, area, wet, DT,
        dx=jnp.asarray(f["e1v"]), dy=jnp.asarray(f["e2u"]), ice_step_index=int(KT), nn_fsbc=int(NN_FSBC),
        halo_width=H, ice_volume_index=TR.index("v_i"), concentration_index=TR.index("a_i"), subcycles=int(ICYCLE))
    raw = si3_prather_unpack_intensives(contents, area, wet)
    cor, res = apply_si3_prather_source_corrections(
        x, raw, tracer_names=TR, nlay_i=3, nlay_s=3, cell_area_m2=1.0, halo_width=H,
        entry_intensive_contents=x, contents_are_intensive=True, dt_s=DT, return_residual_fluxes=True)
    ato = si3_prather_open_water(
        jnp.asarray(f["ato_i"]), x[..., TR.index("a_i")], cor[..., TR.index("a_i")], u, v,
        dy_u=jnp.asarray(f["e2u"]), dx_v=jnp.asarray(f["e1v"]), cell_area=area, dt=DT / ICYCLE,
        floor_sharpness=sharpness)
    out = {"ato_i": ato, "wfx_res": res.wfx_res, "hfx_res": res.hfx_res, "sfx_res": res.sfx_res}
    return {k: np.asarray(a) for k, a in out.items()}, np.asarray(raw), np.asarray(cor)


# Saturated cells (NEMO's own branch flips under 1 ulp: 20-seed spread >= 0.5) carry only roundoff-scale values
# here; there the comparison is absolute, 1e-12 x the field's largest |delta| on the ORCA1 step (vote V8 a).
SATURATED = 0.5
ABS_SCALE = {"ato_i": 0.19, "wfx_res": 6.538e-5, "hfx_res": 45.73, "sfx_res": 8.432e-3}


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise FloatingPointError("non-finite operand in oracle comparison")
    den = np.maximum(np.abs(a), np.abs(b))
    return np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)


@pytest.mark.parametrize("w", WINDOWS)
def test_open_water_and_residuals_match_nemo(w):
    out, _, _ = _port(_window(w))
    for k in NEW:
        nemo = FIX[f"w{w}_nemo_{k}_out"]
        spread = FIX[f"w{w}_spread_{k}"]
        sat = spread >= SATURATED
        tol = np.maximum(2.0 * spread, 1e-13)
        rel = _rel(out[k][INNER], nemo)
        assert np.all(rel[~sat] <= tol[~sat]), (k, float(rel[~sat].max()), np.argwhere((rel > tol) & ~sat)[:4])
        err = np.abs(out[k][INNER] - nemo)
        assert np.all(err[sat] <= 1e-12 * ABS_SCALE[k]), (k, float(err[sat].max()) if sat.any() else 0.0)


def test_nonfinite_comparison_is_rejected():
    with pytest.raises(FloatingPointError):
        _rel(np.array([np.nan]), np.array([0.0]))


def test_fixture_exercises_every_new_branch():
    """Non-vacuity of the fixture: NEMO's ledgers bind and open water clamps inside the scored cells."""
    binding = {k: sum(int((FIX[f"w{w}_nemo_{k}_out"] != 0).sum()) for w in WINDOWS) for k in NEW[1:]}
    assert min(binding.values()) >= 5, binding
    clamped = sum(int(((FIX[f"w{w}_nemo_ato_i_out"] == 0) & (FIX[f"w{w}_ato_i"][INNER] > 0)).sum()) for w in WINDOWS)
    assert clamped >= 5, clamped
    moved = sum(int((FIX[f"w{w}_nemo_ato_i_out"] != FIX[f"w{w}_ato_i"][INNER]).sum()) for w in WINDOWS)
    assert moved >= 60, moved


def test_planted_violation_is_caught():
    """Dropping the transport divergence from the open-water update must fail the oracle tolerance."""
    f = _window(0)
    out, _, cor = _port(f)
    noflux = np.maximum(0.0, f["ato_i"] - (cor[..., TR.index("a_i")] - f["X"][..., TR.index("a_i")]))
    nemo = FIX["w0_nemo_ato_i_out"]
    tol = np.maximum(2.0 * FIX["w0_spread_ato_i"], 1e-13)
    assert np.any(_rel(noflux[INNER], nemo) > tol)
    assert np.all(_rel(out["ato_i"][INNER], nemo) <= tol)


@pytest.mark.parametrize("w", WINDOWS)
def test_ledgers_close_the_corrections_per_cell(w):
    """Every unit removed by Hbig/Hsnow/zapneg is in the ledger: (raw - corrected) content == residual * dt."""
    out, raw, cor = _port(_window(w))
    d = raw - cor
    mass = (constants.rho_ice * d[..., TR.index("v_i")] + constants.rho_snow * d[..., TR.index("v_s")]
            + constants.rho_water * (d[..., TR.index("v_ip")] + d[..., TR.index("v_il")]))
    salt = constants.rho_ice * d[..., SALT].sum(-1)
    heat = -(d[..., SNOW].sum(-1) + d[..., ICE_E].sum(-1))
    inner = (slice(H, -H), slice(H, -H))
    for name, removed, ledger, scale in (("water", mass, out["wfx_res"], constants.rho_ice),
                                         ("salt", salt, out["sfx_res"], constants.rho_ice * 10.0),
                                         ("heat", heat, out["hfx_res"], 3.0e8)):
        err = np.abs(removed[inner] - ledger[inner] * DT)
        assert np.all(err <= 1e-13 * scale + 1e-12 * np.abs(removed[inner])), (name, float(err.max()))
    assert np.abs(mass[inner]).max() > 0 and np.abs(heat[inner]).max() > 0


def _periodic(f):
    """Doubly periodic copy of a window (interior wrapped into the halo) so the kernel conserves globally."""
    g = {}
    for k, a in f.items():
        if k == "MOM":
            g[k] = np.stack([np.pad(m[H:-H, H:-H], ((H, H), (H, H), (0, 0)), mode="wrap") for m in a])
        else:
            pad = ((H, H), (H, H)) + ((0, 0),) * (a.ndim - 2)
            g[k] = np.pad(a[H:-H, H:-H], pad, mode="wrap")
    g["tmask1"] = np.ones_like(g["tmask1"])  # all wet: the window's land cells would act as open sinks
    return g


@pytest.mark.parametrize("w", WINDOWS)
def test_global_budget_closes_with_residual_fluxes(w):
    """Doubly periodic window: in - out - dstorage - residual*dt*area = 0 for ice+snow mass, salt and heat."""
    f = _periodic(_window(w))
    out, _, cor = _port(f)
    area = f["e1e2t"][H:-H, H:-H]
    x0, x1 = f["X"][H:-H, H:-H], cor[H:-H, H:-H]

    def tot(c, idx, fac=1.0):
        return float(np.sum(area * fac * c[..., idx].sum(-1) if isinstance(idx, list) else area * fac * c[..., idx]))

    def mass(c):
        return (tot(c, TR.index("v_i"), constants.rho_ice) + tot(c, TR.index("v_s"), constants.rho_snow)
                + tot(c, [TR.index("v_ip"), TR.index("v_il")], constants.rho_water))

    mass0, mass1 = mass(x0), mass(x1)
    salt0, salt1 = tot(x0, SALT, constants.rho_ice), tot(x1, SALT, constants.rho_ice)
    heat0, heat1 = tot(x0, SNOW) + tot(x0, ICE_E), tot(x1, SNOW) + tot(x1, ICE_E)
    flux = {k: float(np.sum(area * out[k][H:-H, H:-H])) * DT for k in NEW[1:]}
    assert abs(mass0 - mass1 - flux["wfx_res"]) <= 1e-12 * mass0
    assert abs(salt0 - salt1 - flux["sfx_res"]) <= 1e-12 * salt0
    assert abs(heat0 - heat1 + flux["hfx_res"]) <= 1e-12 * heat0
    assert flux["wfx_res"] != 0.0 or flux["hfx_res"] != 0.0


def test_smooth_default_floor_bound_and_gradient():
    """Default smooth floor: exact for x<=0 and x>=1/s, never above the hard MAX by more than 4/(27 s); C1 grad."""
    s = SI3_PRA_OPEN_WATER_FLOOR_SHARPNESS
    assert s == 1.0e7
    f = _window(2)
    hard, _, _ = _port(f)
    smooth, _, _ = _port(f, sharpness=s)
    gap = hard["ato_i"] - smooth["ato_i"]
    assert np.all(gap >= 0.0) and np.all(gap <= 4.0 / (27.0 * s) + 1e-18)
    for k in NEW[1:]:
        assert np.array_equal(hard[k], smooth[k])

    def ow(ato):
        z = jnp.zeros_like(ato)
        return si3_prather_open_water(ato, z, z, z, z, dy_u=1.0, dx_v=1.0, cell_area=jnp.ones_like(ato), dt=1.0)

    pts = jnp.array([[-1e-7], [2e-8], [5e-8], [9e-8], [1e-3]])
    check_grads(ow, (pts,), order=1, modes=("fwd", "rev"), eps=1e-10)
    g = jax.grad(lambda a: jnp.sum(ow(a)))(pts)
    assert np.all(np.isfinite(g)) and float(g[1, 0]) > 0.0 and float(g[-1, 0]) == 1.0 and float(g[0, 0]) == 0.0


def test_gradient_through_prather_and_ledgers_is_finite():
    """jax.grad of the new outputs w.r.t. entry state and ice velocity through the full Prather chain is finite,
    nonzero, and matches a central finite difference along a random direction."""
    f = _window(0)
    a0 = jnp.asarray(f["e1e2t"])
    wet = jnp.asarray(f["tmask1"] > 0)

    def loss(x, u):
        contents, _, _ = advect_si3_prather_2d(
            si3_prather_pack_intensives(x, a0), tuple(jnp.asarray(m) for m in f["MOM"]), u, jnp.asarray(f["v_ice"]),
            a0, wet, DT, dx=jnp.asarray(f["e1v"]), dy=jnp.asarray(f["e2u"]), ice_step_index=int(KT),
            nn_fsbc=int(NN_FSBC), halo_width=H, ice_volume_index=TR.index("v_i"),
            concentration_index=TR.index("a_i"), subcycles=1)
        raw = si3_prather_unpack_intensives(contents, a0, wet)
        cor, res = apply_si3_prather_source_corrections(
            x, raw, tracer_names=TR, nlay_i=3, nlay_s=3, cell_area_m2=1.0, halo_width=H,
            entry_intensive_contents=x, contents_are_intensive=True, dt_s=DT, return_residual_fluxes=True)
        ato = si3_prather_open_water(jnp.asarray(f["ato_i"]), x[..., TR.index("a_i")], cor[..., TR.index("a_i")],
                                     u, jnp.asarray(f["v_ice"]), dy_u=jnp.asarray(f["e2u"]),
                                     dx_v=jnp.asarray(f["e1v"]), cell_area=a0, dt=DT)
        return jnp.sum(ato) + jnp.sum(res.hfx_res) * 1e-3 + jnp.sum(res.wfx_res) * 1e4

    x, u = jnp.asarray(f["X"]), jnp.asarray(f["u_ice"])
    gx, gu = jax.grad(loss, argnums=(0, 1))(x, u)
    assert np.all(np.isfinite(np.asarray(gx))) and np.all(np.isfinite(np.asarray(gu)))
    assert float(jnp.abs(gu).max()) > 0.0
    du = jnp.asarray(np.random.default_rng(0).standard_normal(u.shape)) * 1e-6
    eps = 1e-3
    fd = (loss(x, u + eps * du) - loss(x, u - eps * du)) / (2 * eps)
    ad = jnp.sum(gu * du)
    assert abs(float(fd) / float(ad) - 1.0) < 1e-3, (float(fd), float(ad))


def test_default_output_unchanged_and_dt_required():
    f = _window(1)
    x = jnp.asarray(f["X"])
    kw = dict(tracer_names=TR, nlay_i=3, nlay_s=3, cell_area_m2=1.0, halo_width=H, entry_intensive_contents=x,
              contents_are_intensive=True)
    plain = apply_si3_prather_source_corrections(x, x * 1.01, **kw)
    with_res, _ = apply_si3_prather_source_corrections(x, x * 1.01, dt_s=DT, return_residual_fluxes=True, **kw)
    assert np.array_equal(np.asarray(plain), np.asarray(with_res))
    with pytest.raises(ValueError, match="dt_s"):
        apply_si3_prather_source_corrections(x, x, return_residual_fluxes=True, **kw)


def test_zapneg_volume_ledgers_synthetic():
    """ORCA1 runs ln_pnd=F (pond branches latent there): synthetic cells for each icevar.F90 zapneg pond condition.
    A: a_ip<=0 -> v_il zapped (v_ip=0 so the pond Hbig cannot bind); B: v_ip<0 -> both zapped; C: v_il<0 only.
    wfx uses rhow.  D: v_s<0 -> snow volume (rhos) and enthalpy zapped; E: v_i<0 -> a_i=0, everything zapped (rhoi).
    These volume branches bind only at roundoff scale on the ORCA1 step, so they are pinned here."""
    x = np.zeros((9, 5, len(TR)))
    base = {"v_i": 1.0, "a_i": 0.5, "v_s": 0.1, "oa_i": 1.0, "szv_i_l01": 5.0}
    for k, v in base.items():
        x[..., TR.index(k)] = v
    for i in SNOW:
        x[..., i] = 1.0e7
    for i in ICE_E:
        x[..., i] = 1.0e8
    cells = {2: (0.0, 0.0, 0.01), 3: (0.1, -0.001, 0.003), 4: (0.1, 0.01, -0.002)}
    for row, (a_ip, v_ip, v_il) in cells.items():
        x[row, 2, TR.index("a_ip")], x[row, 2, TR.index("v_ip")], x[row, 2, TR.index("v_il")] = a_ip, v_ip, v_il
    x[5, 2, TR.index("v_s")] = -1.0e-3
    x[6, 2, TR.index("v_i")] = -1.0e-3
    xj = jnp.asarray(x)
    cor, res = apply_si3_prather_source_corrections(
        xj, xj, tracer_names=TR, nlay_i=3, nlay_s=3, cell_area_m2=1.0, halo_width=H,
        entry_intensive_contents=xj, contents_are_intensive=True, dt_s=DT, return_residual_fluxes=True)
    expected = {2: 0.01, 3: (-0.001 + 0.003), 4: -0.002}
    for row, vol in expected.items():
        assert abs(float(res.wfx_res[row, 2]) - vol * constants.rho_water / DT) <= 1e-15 * constants.rho_water / DT
        assert float(res.hfx_res[row, 2]) == 0.0 and float(res.sfx_res[row, 2]) == 0.0
    rs, ri = constants.rho_snow, constants.rho_ice
    checks = {5: (-1.0e-3 * rs, -3.0e7, 0.0), 6: (-1.0e-3 * ri + 0.1 * rs, -(3.0e7 + 3.0e8), 5.0 * ri)}
    for row, (w_exp, h_exp, s_exp) in checks.items():
        got = (float(res.wfx_res[row, 2]), float(res.hfx_res[row, 2]), float(res.sfx_res[row, 2]))
        for g, e in zip(got, (w_exp / DT, h_exp / DT, s_exp / DT), strict=True):
            assert abs(g - e) <= 1e-13 * max(abs(e), 1.0), (row, got)
    c = np.asarray(cor)
    assert c[2, 2, TR.index("v_ip")] == 0.0 and c[2, 2, TR.index("v_il")] == 0.0
    assert c[3, 2, TR.index("v_ip")] == 0.0 and c[3, 2, TR.index("v_il")] == 0.0
    assert c[4, 2, TR.index("v_ip")] == 0.01 and c[4, 2, TR.index("v_il")] == 0.0
