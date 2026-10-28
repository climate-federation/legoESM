"""Independent-oracle, JIT/AD, edge, and production-wiring tests for SI3 O13."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ice.itd import si3_ice_itd_rem


HI_MAX = np.array(
    [0.0, 0.4540167424155517, 1.1293138230836322,
     2.1414589889479725, 3.670596409915544, 99.0],
    dtype=np.float64,
)


def _inputs() -> dict[str, np.ndarray]:
    """A standalone two-column state: one mixed crossing and one first-bin melt."""
    h_b = np.array(
        [[0.25, 0.78, 1.60, 2.85, 5.10],
         [0.31, 0.83, 1.72, 2.97, 5.40]], dtype=np.float64,
    )
    h = np.array(
        [[0.31, 0.71, 1.78, 2.69, 5.55],
         [0.25, 0.91, 1.61, 3.13, 5.17]], dtype=np.float64,
    )
    a_b = np.array(
        [[0.12, 0.18, 0.21, 0.17, 0.11],
         [0.10, 0.20, 0.19, 0.16, 0.12]], dtype=np.float64,
    )
    a = a_b + np.array(
        [[0.002, -0.003, 0.001, 0.002, -0.001],
         [-0.002, 0.003, -0.001, 0.001, 0.002]], dtype=np.float64,
    )
    v = h * a
    layer_i = np.arange(1.0, 4.0)[:, None]
    layer_s = np.arange(1.0, 3.0)[:, None]
    return {
        "h_i_b": h_b,
        "a_i_b": a_b,
        "h_i": h,
        "a_i": a,
        "v_i": v,
        "v_s": 0.08 * v,
        "sv_i": 4.0 * v,
        "oa_i": 2.0 * a,
        "a_ip": 0.16 * a,
        "v_ip": 0.018 * a,
        "v_il": 0.0015 * a,
        "t_su": 266.0 + np.arange(5, dtype=np.float64)[None, :] + np.zeros((2, 1)),
        "e_s": np.stack([0.07 * v[c][None, :] * layer_s for c in range(2)]),
        "e_i": np.stack([1.3 * v[c][None, :] * layer_i for c in range(2)]),
        "szv_i": np.stack([4.0 * v[c][None, :] * layer_i for c in range(2)]),
        "hi_max": HI_MAX.copy(),
        "rn_amax_2d": np.array([0.997, 0.997], dtype=np.float64),
    }


def _glinear(lo: float, hi: float, h: float, a: float) -> tuple[float, float, float, float]:
    """Independent scalar reading of Lipscomb eq. 14; no legoESM imports."""
    if not (a > 1.0e-10 and h > 1.0e-10):
        return 0.0, 0.0, 0.0, 0.0
    left, right = lo, hi
    if h < (2.0 * left + right) / 3.0:
        right = 3.0 * h - 2.0 * left
    elif h > (left + 2.0 * right) / 3.0:
        left = 3.0 * h - 2.0 * right
    inv = 1.0 / (right - left) if right > left else 0.0
    w1 = 6.0 * a * inv
    w2 = (h - left) * inv
    return w1 * (2.0 / 3.0 - w2), 2.0 * inv * w1 * (w2 - 0.5), left, right


def _oracle_column(src: dict[str, np.ndarray], col: int) -> dict[str, np.ndarray]:
    """Scalar NumPy transcription sharing neither model code nor model constants."""
    names = ("h_i", "a_i", "v_i", "v_s", "sv_i", "oa_i", "a_ip", "v_ip", "v_il", "t_su")
    q = {name: src[name][col].copy() for name in names}
    q.update({name: src[name][col].copy() for name in ("e_s", "e_i", "szv_i")})
    original = copy.deepcopy(q)
    hb = src["h_i_b"][col]
    ab = src["a_i_b"][col]
    b = src["hi_max"]
    active = float(np.sum(q["a_i"])) > 1.0e-10
    dh = np.where(q["a_i"] > 1.0e-10, q["h_i"] - hb, 0.0)
    newb = np.zeros(6, dtype=np.float64)
    for k in range(4):
        if ab[k] > 1.0e-10 and ab[k + 1] > 1.0e-10:
            slope = (dh[k + 1] - dh[k]) / (hb[k + 1] - hb[k])
            newb[k + 1] = b[k + 1] + dh[k] + slope * (b[k + 1] - hb[k])
        elif ab[k] > 1.0e-10:
            newb[k + 1] = b[k + 1] + dh[k]
        elif ab[k + 1] > 1.0e-10:
            newb[k + 1] = b[k + 1] + dh[k + 1]
        else:
            newb[k + 1] = b[k + 1]
        active &= not (q["a_i"][k] > 1.0e-10 and q["h_i"][k] > newb[k + 1] - 1.0e-10)
        active &= not (q["a_i"][k + 1] > 1.0e-10 and q["h_i"][k + 1] < newb[k + 1] + 1.0e-10)
        active &= b[k] <= newb[k + 1] <= b[k + 2]
    newb[5] = max(b[4], 3.0 * q["h_i"][4] - 2.0 * newb[4]) if q["a_i"][4] > 1.0e-10 else b[5]
    active &= b[0] + 1.0e-10 <= hb[0] <= b[1] - 1.0e-10
    if not active:
        return original

    g00, g10, left0, right0 = _glinear(b[0], b[1], hb[0], q["a_i"][0])
    d0 = q["h_i"][0] - hb[0]
    if q["a_i"][0] > 1.0e-10 and d0 < 0.0:
        d0 = min(-d0, b[1])
        eta = min(d0, right0) - left0
        if eta > 0.0:
            lost = g10 * 0.5 * eta * eta + g00 * eta
            lost = min(lost, q["a_i"][0] * (1.0 - q["h_i"][0] / hb[0]))
            q["h_i"][0] *= q["a_i"][0] / (q["a_i"][0] - lost)
            q["a_i"][0] -= lost
            q["v_i"][0] = q["a_i"][0] * q["h_i"][0]
    elif q["a_i"][0] > 1.0e-10:
        newb[0] = min(d0, b[1])

    coeff = np.array([_glinear(newb[k], newb[k + 1], q["h_i"][k], q["a_i"][k]) for k in range(5)])
    transfers: list[tuple[bool, float, float]] = []
    for k in range(4):
        up = newb[k + 1] > b[k + 1]
        donor = k if up else k + 1
        left, right = coeff[donor, 2], coeff[donor, 3]
        emin = max(b[k + 1], left) - left if up else 0.0
        emax = min(newb[k + 1], right) - left if up else min(b[k + 1], right) - left
        emax = max(emax, emin)
        x1 = emax - emin
        x2 = 0.5 * (emax**2 - emin**2)
        x3 = (emax**3 - emin**3) / 3.0
        da = coeff[donor, 1] * x2 + coeff[donor, 0] * x1
        dv = coeff[donor, 1] * x3 + coeff[donor, 0] * x2 + da * left
        transfers.append((up, da, dv))

    ats = q["a_i"] * q["t_su"]
    for k, (up, da, dv) in enumerate(transfers):
        donor, recv = (k, k + 1) if up else (k + 1, k)
        rv = dv / q["v_i"][donor] if q["v_i"][donor] >= 1.0e-10 else 0.0
        ra = da / q["a_i"][donor] if q["a_i"][donor] >= 1.0e-10 else 0.0
        q["a_i"][donor] -= da; q["a_i"][recv] += da
        q["v_i"][donor] -= dv; q["v_i"][recv] += dv
        for name, ratio in (("v_s", rv), ("oa_i", ra), ("a_ip", ra), ("v_ip", rv), ("v_il", rv)):
            moved = q[name][donor] * ratio
            q[name][donor] -= moved; q[name][recv] += moved
        moved = ats[donor] * ra; ats[donor] -= moved; ats[recv] += moved
        for name in ("e_s", "e_i", "szv_i"):
            moved = q[name][:, donor] * rv
            q[name][:, donor] -= moved; q[name][:, recv] += moved

    for name in ("a_i", "v_i", "v_s", "oa_i", "a_ip", "v_ip", "e_s", "e_i", "szv_i"):
        q[name] = np.where(q[name] < 0.0, 0.0, q[name])
    q["v_il"] = np.where((q["v_il"] < 0.0) & (q["v_il"] > -1.0e-10), 0.0, q["v_il"])
    total = float(np.sum(q["a_i"]))
    if total > src["rn_amax_2d"][col]:
        q["a_i"] *= src["rn_amax_2d"][col] / total
    live = q["a_i"] >= 1.0e-20
    q["h_i"] = np.where(live, q["v_i"] / np.where(live, q["a_i"], 1.0), 0.0)
    q["t_su"] = np.where(live, ats / np.where(live, q["a_i"], 1.0), 273.15)
    return q


def _call_kwargs(values: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(value, jnp.float64) for key, value in values.items()}


def _model(values: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return si3_ice_itd_rem(
        **values, nn_icesal=4, ln_pnd=True, ln_pnd_lids=True,
        epsi10=1.0e-10, epsi20=1.0e-20, rt0=273.15,
    )


def test_o13_independent_scalar_oracle_true_eager_jit_and_conservation() -> None:
    src = _inputs()
    oracle = [_oracle_column(src, col) for col in range(2)]
    args = _call_kwargs(src)
    with jax.disable_jit():
        eager = _model(args)
    compiled = jax.jit(_model)(args)
    for name in ("h_i", "a_i", "v_i", "v_s", "sv_i", "oa_i", "a_ip", "v_ip", "v_il", "t_su", "e_s", "e_i", "szv_i"):
        np.testing.assert_array_equal(np.asarray(eager[name]), np.asarray(compiled[name]))
        want = np.stack([oracle[col][name] for col in range(2)])
        np.testing.assert_allclose(np.asarray(compiled[name]), want, rtol=2.0e-15, atol=2.0e-15)

    for name in ("v_i", "v_s", "oa_i", "a_ip", "v_ip", "v_il", "e_s", "e_i", "szv_i"):
        before = np.sum(src[name], axis=-1)
        after = np.sum(np.asarray(compiled[name]), axis=-1)
        np.testing.assert_allclose(after, before, rtol=3.0e-15, atol=3.0e-15)


def test_o13_jit_arguments_gradient_central_difference_and_zero_edges() -> None:
    src = _inputs()
    args = _call_kwargs(src)

    def objective(h):
        out = _model({**args, "h_i": h})
        weights = jnp.arange(1.0, 6.0, dtype=jnp.float64)
        return jnp.sum(out["a_i"] * weights) + 0.01 * jnp.sum(out["e_i"])

    h = args["h_i"]
    grad = jax.jit(jax.grad(objective))(h)
    assert bool(jnp.all(jnp.isfinite(grad)))
    direction = jnp.asarray(
        [[0.7, -0.2, 0.1, 0.3, -0.4], [-0.1, 0.5, -0.3, 0.2, 0.4]],
        dtype=jnp.float64,
    )
    eps = 2.0e-6
    central = (objective(h + eps * direction) - objective(h - eps * direction)) / (2.0 * eps)
    tangent = jnp.sum(grad * direction)
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(central), rtol=2.0e-5, atol=2.0e-7)

    edge = _inputs()
    for key in ("h_i_b", "a_i_b", "h_i", "a_i", "v_i", "v_s", "sv_i", "oa_i", "a_ip", "v_ip", "v_il", "e_s", "e_i", "szv_i"):
        edge[key][0] = 0.0
    edge["a_i_b"][1, 2] = 0.0
    edge["a_i"][1, 2] = 0.0
    edge["v_i"][1, 2] = 0.0
    edge_args = _call_kwargs(edge)
    edge_out = jax.jit(_model)(edge_args)
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in edge_out.values())
    edge_grad = jax.grad(lambda x: jnp.sum(_model({**edge_args, "h_i": x})["v_i"]))(edge_args["h_i"])
    assert bool(jnp.all(jnp.isfinite(edge_grad)))


def test_o13_production_sea_ice_call_site_fires(monkeypatch) -> None:
    """M4 plant: removing the production call makes the sentinel assertion fail."""
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.ice import sea_ice as sea_ice_mod
    from legoesm.ice.config import MeltPondConfig, SeaIceConfig, SI3PondConfig, SnowConfig
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig
    from legoesm.ice.state import distribute_dynamic_state_to_categories, init_dynamic_ice_state

    shape = (2, 2)
    grid = create_latlon_grid(n_lat=shape[0], n_lon=shape[1])
    z = jnp.zeros(shape, jnp.float64)
    base = init_dynamic_ice_state(shape, S_ice_init=2.5, T_ice_init=271.35, pond_scheme="si3_lev")
    base = base._replace(
        h_ice=base.h_ice.replace(data=jnp.full(shape, 1.5, jnp.float64)),
        concentration=base.concentration.replace(data=jnp.full(shape, 0.78, jnp.float64)),
        pond_area=base.pond_area.replace(data=jnp.full(shape, 0.18, jnp.float64)),
        pond_depth=base.pond_depth.replace(data=jnp.full(shape, 0.12, jnp.float64)),
        h_il=base.h_il.replace(data=jnp.full(shape, 0.01, jnp.float64)),
    )
    state = distribute_dynamic_state_to_categories(base, 5)
    forcing = AtmToSurface(
        sw_down=z, lw_down=z, precip_total=z, precip_snow=z,
        T_lowest=jnp.full(shape, 271.35), q_lowest=z, u_lowest=z, v_lowest=z,
        p_lowest=jnp.full(shape, 100000.0), p_surface=jnp.full(shape, 101325.0),
        rho_lowest=jnp.full(shape, 1.25), cos_zenith=z, co2_ppmv=jnp.full(shape, 400.0),
        has_radiation=z, has_precipitation=z, z_lowest=jnp.full(shape, 10.0),
    )
    pond_nml = SI3PondConfig(0.1, 0.15, 0.80, 0.1, True)
    si3_cfg = SI3ORCA2StateConfig.for_orca2_sas_ice(nn_hls=0)
    cfg = SeaIceConfig(
        n_categories=5,
        itd_remap="lipscomb2001",
        snow=SnowConfig(enabled=True, flooding=False),
        ponds=MeltPondConfig(enabled=True, scheme="si3_lev", namthd_pnd=pond_nml),
        si3_orca2=si3_cfg,
        ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
        emissivity_ice=0.0, Cd_ice=0.0, Ch_ice=0.0, Ce_ice=0.0,
        ocean_heat_transfer_coeff=0.0,
    )
    real = sea_ice_mod.si3_ice_itd_rem
    calls: list[str] = []
    monkeypatch.setattr(
        sea_ice_mod, "si3_ice_itd_rem",
        lambda **kw: (calls.append("ice_itd_rem"), real(**kw))[1],
    )
    sea_ice_mod.step_sea_ice(
        state, forcing, jnp.full(shape, 271.35), z, z, cfg, 1.0, 3600.0,
        grid=grid, rn_amax_2d=jnp.full(shape, 0.997),
    )
    assert calls == ["ice_itd_rem"]

    class PlantFired(RuntimeError):
        pass

    with monkeypatch.context() as mp:
        mp.setattr(sea_ice_mod, "si3_ice_itd_rem", lambda **kw: (_ for _ in ()).throw(PlantFired()))
        with pytest.raises(PlantFired):
            sea_ice_mod.step_sea_ice(
                state, forcing, jnp.full(shape, 271.35), z, z, cfg, 1.0, 3600.0,
                grid=grid, rn_amax_2d=jnp.full(shape, 0.997),
            )
