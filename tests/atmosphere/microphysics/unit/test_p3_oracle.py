"""P3 (one 2-moment ice category) vs the RUNNING WRF v4.8.0 P3 oracle.

Fixture ``tests/atmosphere/microphysics/oracle_fixtures/p3.npz`` was generated
by RUNNING WRF v4.8.0 ``phys/module_mp_p3.F`` (P3 v4.5.2; tag v4.8.0, SHA
06d4240ae989cc3e50af412bb472df3d9048783c) with ``p3_lookupTable_1.dat-v5.4_2momI``,
compiled with gfortran in fp64, through ``scripts/validate/oracle/p3/build.sh`` +
``make_fixture.py``, in the mp_physics = 50 configuration (one ice category,
specified droplet number nccnst = 200 /cm^3, KK2000 warm rain, Cooper
nucleation).  The oracle is fed legoESM saturation specific humidity and the
legoESM constants (the tier-1 oracle-fixture glue, recorded in ``meta``).

Tolerance (user decision 2026-10-19): max relative error <= 1e-10 in fp64 with
every existing sharpness field at its hard-switch limit.  Relative error is
``|a-b| / max(|a|, |b|)`` (0 where both are exactly 0); no floor.  The oracle's
specified droplet number is passed as ``Nc_0`` (equal to the P3Config
default, WRF ``nccnst`` = 2e8, since the user decision of 2026-10-21).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics import p3 as P3  # noqa: E402
from legoesm.atmosphere.physics.microphysics.config import P3Config  # noqa: E402
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/p3.npz"
GENERATOR = ROOT / "scripts/validate/oracle/p3/make_fixture.py"
TOL = 1.0e-10
WRF_NCCNST = 200.0e6     # WRF P3 p3_init nccnst [1/m^3]
# Every existing sharpness field -> its hard-switch limit (the oracle has hard
# switches; same limits as the tier-1 audit's HARD_LIMIT).
HARD_LIMIT = dict(ice_sigmoid_sharpness=1.0e6, cooper_supi_sharpness=1.0e9,
                  melt_sharpness=1.0e6)
HARD = P3Config(Nc_0=WRF_NCCNST)._replace(**HARD_LIMIT)
DEFAULT_SMOOTHING = P3Config(Nc_0=WRF_NCCNST)
STATE_MAP = dict(T="T", q_v="qv", q_c="qc", q_r="qr", q_i="qi", q_rim="qirim",
                 B_rim="birim", N_i="ni")
IN_KEYS = ("p", "rho", "dz", "T", "q_v", "q_c", "N_c", "q_r", "N_r", "q_i",
           "q_rim", "N_i", "B_rim")


def _fx():
    return dict(np.load(FIXTURE))


def _meta(fx):
    return json.loads(str(fx["meta"]))


def _cases():
    m = _meta(_fx())
    return [(c, dt) for c in m["cases"]
            for dt in (m["nuc_dts"] if c == "nuc" else m["dts"])]


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    d = np.maximum(np.abs(a), np.abs(b))
    return np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)


def _inputs(fx, case, dtype=jnp.float64):
    g = {k: jnp.asarray(fx[f"in_{case}_{k}"], dtype) for k in IN_KEYS}
    hyd = HydrometeorState(q_c=g["q_c"], q_r=g["q_r"], q_i=g["q_i"], q_s=g["q_rim"],
                           q_g=g["B_rim"], N_c=g["N_c"], N_r=g["N_r"], N_i=g["N_i"])
    return g, hyd


def _errors(fx, case, dt, out):
    s, prt_liq, prt_sol = out
    o = f"out_{case}_dt{dt:g}_"
    rho = fx[f"in_{case}_rho"]
    errs = {k: _rel(s[k], fx[o + v]) for k, v in STATE_MAP.items()}
    errs["N_c"] = _rel(np.asarray(s["N_c"]) / rho, fx[o + "nc"])
    errs["N_r"] = _rel(np.asarray(s["N_r"]) / rho, fx[o + "nr"])
    errs["prt_liq"] = _rel(prt_liq, fx[o + "prt_liq"])
    errs["prt_sol"] = _rel(prt_sol, fx[o + "prt_sol"])
    return {k: float(v.max()) for k, v in errs.items()}


def _step(fx, case, dt, cfg=HARD, dtype=jnp.float64):
    g, hyd = _inputs(fx, case, dtype)
    return P3.p3_step(g["T"], g["q_v"], hyd, g["p"], g["rho"], g["dz"], float(dt), cfg)


def test_fixture_provenance_and_fetched_table():
    fx = _fx()
    m = _meta(fx)
    assert m["tag"] == "v4.8.0"
    assert m["sha"] == "06d4240ae989cc3e50af412bb472df3d9048783c"
    assert m["module_mp_p3_sha256"] == (
        "716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6")
    assert m["config"]["nccnst_per_m3"] == WRF_NCCNST
    sha = hashlib.sha256(P3.p3_lookup_table_path().read_bytes()).hexdigest()
    assert sha == m["lookup_table_1_sha256"] == P3.P3_LOOKUP_TABLE_SHA256 == (
        "be1ab6fb03481e376e47c6c79d808af5d8ab069f2b242931e9c54801bad4ae84")
    # the oracle's saturation stub reproduces legoesm.thermo
    assert max(m["qsat_stub_vs_thermo_max_rel"].values()) <= TOL


def test_generator_rebuilds_the_analytic_inputs():
    """The committed generator rebuilds the sounding-free inputs (arctic_mixed,
    edge, coverage, the nucleation points) bit-for-bit, and its output-field
    list is the fixture's."""
    spec = importlib.util.spec_from_file_location("p3_make_fixture", GENERATOR)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    fx = _fx()
    assert list(gen.OUT_FIELDS) == _meta(fx)["out_fields"]
    for f in (gen.arctic_mixed, gen.edge, gen.coverage):
        c = f(None)
        for k in gen.IN_FIELDS:
            np.testing.assert_array_equal(np.asarray(c[k], float), fx[f"in_{c['name']}_{k}"][0])
    nuc = gen.nucleation_points(None)
    for k in gen.IN_FIELDS:
        np.testing.assert_array_equal(np.asarray(nuc[k], float), fx[f"in_nuc_{k}"][:, 0])


@pytest.mark.parametrize("case,dt", _cases())
def test_whole_step_matches_wrf_p3(case, dt):
    """Every post-step state field and both surface precipitation rates."""
    fx = _fx()
    worst = _errors(fx, case, dt, _step(fx, case, dt))
    print(case, dt, "max rel err", worst)
    assert max(worst.values()) <= TOL, worst


def test_fixture_exercises_the_processes():
    """Every process P3 runs for mp_physics = 50 is active somewhere."""
    fx = _fx()
    active = ("qccon", "qcevp", "qrcon", "qrevp", "qcaut", "qcacc", "nrslf",
              "qccol", "qrcol", "nrcol", "qidep", "qisub", "qinuc", "ninuc", "nislf",
              "qimlt", "qcheti", "qrheti", "qcshd", "qwgrth", "qcnuc", "qchomi",
              "qrhomi", "nrevp", "nrshdr")
    for d in active:
        n = sum(int(np.count_nonzero(fx[f"out_{c}_dt{dt:g}_{d}"])) for c, dt in _cases())
        assert n > 0, d


def test_sedimentation_needs_many_substeps_and_overflow_errors(monkeypatch):
    """dt=300 drives the oracle's adaptive sedimentation loop past 8 sub-steps
    (rain on sc_dycoms); with the fixed maximum lowered to 8 the step errors
    (the Morrison strict path), with the shipped maximum it matches the oracle."""
    fx = _fx()
    monkeypatch.setattr(P3, "_SED_SUBSTEPS_MAX", 8)
    with pytest.raises(Exception, match="sub-steps"):
        jax.block_until_ready(_step(fx, "sc_dycoms", 300.0))
    monkeypatch.undo()
    worst = _errors(fx, "sc_dycoms", 300.0, _step(fx, "sc_dycoms", 300.0))
    assert max(worst.values()) <= TOL, worst


def test_sedimentation_overflow_poisons_outputs_under_jit(monkeypatch):
    """Under jit the overflow is still fatal or, if the error is disabled by
    the environment, every output is NaN (never a silent clamp)."""
    fx = _fx()
    monkeypatch.setattr(P3, "_SED_SUBSTEPS_MAX", 8)
    g, hyd = _inputs(fx, "sc_dycoms")
    f = jax.jit(lambda T: P3.p3_step(T, g["q_v"], hyd, g["p"], g["rho"], g["dz"], 300.0, HARD))
    try:
        out = jax.block_until_ready(f(g["T"]))
    except Exception as e:  # noqa: BLE001 - equinox raises a runtime error type
        assert "sub-steps" in str(e)
        return
    assert all(bool(jnp.all(jnp.isnan(x))) for x in jax.tree_util.tree_leaves(out))


def _p3_air_mass(g):
    """P3's own air density p/(R_d T) times dz (the mass its fluxes use)."""
    return np.asarray(g["p"] / (C.R_d * g["T"]) * g["dz"])


def test_column_water_budget_closes_like_the_oracle():
    """Column vapour+condensate change + surface precipitation = 0 to
    round-off for both the port and the oracle (P3's own air mass)."""
    fx = _fx()
    for case, dt in _cases():
        g, _ = _inputs(fx, case)
        s, pl, ps = _step(fx, case, dt)
        m = _p3_air_mass(g)
        q0 = np.asarray(g["q_v"] + g["q_c"] + g["q_r"] + g["q_i"])
        q1 = np.asarray(s["q_v"] + s["q_c"] + s["q_r"] + s["q_i"])
        res = (m * (q1 - q0)).sum(-1) + np.asarray(pl + ps) * C.rho_water * dt
        o = f"out_{case}_dt{dt:g}_"
        q1o = fx[o + "qv"] + fx[o + "qc"] + fx[o + "qr"] + fx[o + "qi"]
        res_o = (m * (q1o - q0)).sum(-1) + (fx[o + "prt_liq"] + fx[o + "prt_sol"]) \
            * C.rho_water * dt
        rel = np.abs(res) / (m * q0).sum(-1)
        rel_o = np.abs(res_o) / (m * q0).sum(-1)
        print(case, dt, "water residual / column water: port", float(rel.max()),
              "oracle", float(rel_o.max()))
        assert np.all(rel <= TOL), (case, dt, rel)


def test_column_energy_budget_closes_like_the_oracle():
    """h = c_pd T + L_v q_v - L_f q_i summed over P3's air mass changes only by
    the frozen precipitation leaving the column (liquid carries no h), i.e.
    every latent-heating term has the sign of its phase change; the port's
    residual equals the oracle's (both printed)."""
    fx = _fx()
    for case, dt in _cases():
        g, _ = _inputs(fx, case)
        s, pl, ps = _step(fx, case, dt)
        m = _p3_air_mass(g)

        def h(T, qv, qi):
            return C.c_pd * T + C.L_v * qv - C.L_f * qi

        h0 = h(np.asarray(g["T"]), np.asarray(g["q_v"]), np.asarray(g["q_i"]))
        h1 = h(np.asarray(s["T"]), np.asarray(s["q_v"]), np.asarray(s["q_i"]))
        # frozen precipitation leaving the column takes its -L_f q_i with it
        res = (m * (h1 - h0)).sum(-1) - C.L_f * np.asarray(ps) * C.rho_water * dt
        o = f"out_{case}_dt{dt:g}_"
        h1o = h(fx[o + "T"], fx[o + "qv"], fx[o + "qi"])
        res_o = (m * (h1o - h0)).sum(-1) - C.L_f * fx[o + "prt_sol"] * C.rho_water * dt
        scale = (m * C.c_pd * np.asarray(g["T"])).sum(-1)
        print(case, dt, "enthalpy residual / column c_pd T: port",
              float(np.max(np.abs(res) / scale)), "oracle", float(np.max(np.abs(res_o) / scale)))
        np.testing.assert_allclose(res, res_o, rtol=0, atol=TOL * float(scale.max()))


def test_default_smoothing_gap_is_reported():
    """At the shipped (default) smoothing the port is NOT the oracle: print
    the gap (PR body) and check it stays finite."""
    fx = _fx()
    gaps = {}
    for case, dt in _cases():
        e = _errors(fx, case, dt, _step(fx, case, dt, DEFAULT_SMOOTHING))
        print(case, dt, "default-smoothing max rel err", e)
        for k, v in e.items():
            gaps[k] = max(gaps.get(k, 0.0), v)
    print("default-smoothing worst per field", gaps)


@pytest.mark.parametrize("case,dt", _cases())
def test_jit_matches_oracle(case, dt):
    fx = _fx()
    g, hyd = _inputs(fx, case)
    f = jax.jit(lambda T, qv: P3.p3_step(T, qv, hyd, g["p"], g["rho"], g["dz"],
                                         float(dt), HARD))
    worst = _errors(fx, case, dt, f(g["T"], g["q_v"]))
    print(case, dt, "jit max rel err", worst)
    assert max(worst.values()) <= TOL, worst


@pytest.mark.parametrize("case", _meta(_fx())["cases"])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("cfg", [HARD, DEFAULT_SMOOTHING], ids=["hard", "default"])
def test_gradients_finite_on_fixture_columns(case, dtype, cfg):
    fx = _fx()
    g, hyd = _inputs(fx, case, dtype)

    def loss(T, qv, qc, qr, qi):
        h = hyd._replace(q_c=qc, q_r=qr, q_i=qi)
        s, pl, ps = P3.p3_step(T, qv, h, g["p"], g["rho"], g["dz"], 300.0, cfg)
        return (jnp.sum(s["T"]) + 1e3 * jnp.sum(s["q_r"] + s["q_i"] + s["q_v"])
                + 1e3 * jnp.sum(pl + ps))

    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(
        g["T"], g["q_v"], g["q_c"], g["q_r"], g["q_i"])
    jax.clear_caches()
    for gi in grads:
        assert bool(jnp.all(jnp.isfinite(gi))), case
    assert float(jnp.max(jnp.abs(grads[0]))) > 0.0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gradients_finite_dry_column_and_extreme_supersaturation(dtype):
    """Inactive rain breakup on a dry column, nucleation at supi ~ 7
    (Cooper exponential above the cap), and the temperature where the
    inactive wet-growth denominator L_f + c_pw (T - T0) is zero;
    d/d(T, q_v, rho) finite."""
    from legoesm.thermo import saturation_specific_humidity_ice
    q8 = 8.0 * float(saturation_specific_humidity_ice(jnp.asarray(230.0),
                                                      jnp.asarray(6.0e4)))
    t_wet0 = C.T_freeze - C.L_f / C.c_pw
    for T, qv, p, rho in ((290.0, 5.0e-3, 9.0e4, 1.0), (230.0, q8, 6.0e4, 0.9),
                          (t_wet0, 1.0e-7, 1.0e5, 1.0)):
        f = lambda x: jnp.full((1, 1), x, dtype)  # noqa: E731
        z = f(0.0)
        hyd = HydrometeorState(q_c=z, q_r=z, q_i=z, q_s=z, q_g=z, N_c=z, N_r=z, N_i=z)

        def loss(T, qv, rho):
            s, _, _ = P3.p3_step(T, qv, hyd, f(p), rho, f(200.0), 20.0, HARD)
            return jnp.sum(s["T"]) + 1e3 * jnp.sum(s["q_v"]) + 1e-6 * jnp.sum(s["N_i"])

        # eager as well as jit: XLA simplifies an unselected 0/0 away under
        # jit, so only the eager gradient shows it.
        grad = jax.grad(loss, argnums=(0, 1, 2))
        for fn in (grad, jax.jit(grad)):
            g = fn(f(T), f(qv), f(rho))
            assert all(bool(jnp.all(jnp.isfinite(x))) for x in g), (T, g)


def test_div_custom_jvp_check_grads():
    from jax.test_util import check_grads
    x = jnp.asarray([1.5, -2.0, 3.0e-3])
    y = jnp.asarray([0.7, 4.0, -2.5])
    check_grads(P3._div, (x, y), order=2, modes=("fwd", "rev"))


def test_lookup_table_reader_rejects_other_versions(tmp_path):
    bad = tmp_path / "table.dat"
    bad.write_text("LOOKUP_TABLE_1-version: 5.4_3momI\n1 2 3\n")
    with pytest.raises(ValueError, match="version"):
        P3.read_p3_lookup_table_1(bad)
    short = tmp_path / "short.dat"
    short.write_text("LOOKUP_TABLE_1-version: 5.4_2momI\n1 2 3\n")
    with pytest.raises(ValueError, match="values"):
        P3.read_p3_lookup_table_1(short)


def test_uncached_table_offline_raises(tmp_path, monkeypatch):
    import urllib.error
    import urllib.request

    def offline(*a, **k):
        raise urllib.error.URLError("offline")
    monkeypatch.setenv("LEGOESM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(RuntimeError, match="not cached"):
        P3.p3_lookup_table_path()
    assert not any((tmp_path / "p3").iterdir())


def test_uncached_table_is_downloaded_and_cached(tmp_path, monkeypatch):
    import io
    import urllib.request

    good = b"VERSION 4.1\nsynthetic\n"
    monkeypatch.setattr(P3, "P3_LOOKUP_TABLE_SHA256", hashlib.sha256(good).hexdigest())
    calls = []

    def serve(url, **k):
        calls.append(url)
        return io.BytesIO(good)
    monkeypatch.setenv("LEGOESM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(urllib.request, "urlopen", serve)
    path = P3.p3_lookup_table_path()
    assert path == tmp_path / "p3" / P3.P3_LOOKUP_TABLE_NAME
    assert path.read_bytes() == good
    assert calls == [P3.P3_LOOKUP_TABLE_URL]
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]
    P3.p3_lookup_table_path()
    assert len(calls) == 1


def test_failed_download_uses_a_peer_installed_table(tmp_path, monkeypatch):
    import urllib.error
    import urllib.request

    good = b"VERSION 4.1\nsynthetic\n"
    monkeypatch.setattr(P3, "P3_LOOKUP_TABLE_SHA256", hashlib.sha256(good).hexdigest())
    target = tmp_path / "p3" / P3.P3_LOOKUP_TABLE_NAME

    def peer_wins_then_fail(*a, **k):
        target.write_bytes(good)
        raise urllib.error.URLError("rate limited")
    monkeypatch.setenv("LEGOESM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(urllib.request, "urlopen", peer_wins_then_fail)
    assert P3.p3_lookup_table_path() == target
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name]


def test_wrong_table_bytes_raise(tmp_path, monkeypatch):
    import io
    import urllib.request

    monkeypatch.setenv("LEGOESM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: io.BytesIO(b"VERSION 4.1\n"))
    with pytest.raises(RuntimeError, match="sha256"):
        P3.p3_lookup_table_path()
    assert not any((tmp_path / "p3").iterdir())
    cached = tmp_path / "p3" / P3.P3_LOOKUP_TABLE_NAME
    cached.write_text("tampered")
    with pytest.raises(RuntimeError, match="wrong sha256"):
        P3.p3_lookup_table_path()


def test_default_droplet_number_is_wrf_nccnst():
    """User decision 2026-10-21: the P3 default specified droplet number is
    WRF's nccnst (module_mp_p3.F:196)."""
    assert P3Config().Nc_0 == WRF_NCCNST


def test_fp32_cold_cloud_immersion_freezing_is_finite():
    """Tier-2 decision C8: liquid cloud at 190 K in fp32.  WRF's literal
    product cdist1*exp(aimm*(T0-T))*gamma(7+mu_c) overflows fp32 before the
    small dumh**2 factor; the log-space form keeps every output finite."""
    f = lambda x: jnp.full((1, 1), x, jnp.float32)  # noqa: E731
    z = f(0.0)
    hyd = HydrometeorState(q_c=f(2.0e-4), q_r=z, q_i=z, q_s=z, q_g=z, N_c=z, N_r=z, N_i=z)
    s, pl, ps = P3.p3_step(f(190.0), f(1.0e-6), hyd, f(2.0e4), f(0.37), f(500.0), 20.0, HARD)
    for k, v in s.items():
        assert bool(jnp.all(jnp.isfinite(v))), k
    assert bool(jnp.all(jnp.isfinite(pl))) and bool(jnp.all(jnp.isfinite(ps)))

