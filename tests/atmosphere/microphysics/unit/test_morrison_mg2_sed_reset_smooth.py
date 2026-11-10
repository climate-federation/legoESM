"""RULE 5 tests of the smoothed CAM6 MG2 droplet size reset inside the
sedimentation fall-speed coefficients (size_dist_param_liq,
micro_mg2_0.F90:2056-2120; in-cloud mean diameter held in [2, 50] um,
ncic <= qcic/1e-20), prognostic droplet number only: the specified-number
path (production) does not sediment droplets (#2216).

``MorrisonConfig.mg2_drop_sed_reset_log_width`` is the LogSumExp width (ln
space) of both limits, default 0.01 = the early reset's and the readjust's;
0 = MG2's hard clip (in the oracle tests' HARD_LIMIT).  Gaps: per fixture
column, the default's departure from that hard limit (every other switch at
its hard limit) in the whole-call droplet-number tendency (GAP) and in the
cloud-water sedimentation rate (GAP_QC), each relative to the column's
largest MG2 value of that quantity.  Same format as
test_morrison_mg2_drop_early_reset_smooth.py.  x64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402

from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import HARD_LIMIT, TOL, _rel  # noqa: E402
from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402
from tests.atmosphere.microphysics.unit import test_morrison_prognostic_nc_act as A  # noqa: E402
from tests.atmosphere.microphysics.unit import test_morrison_prognostic_nc_sed as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FXD = ROOT / "tests/atmosphere/microphysics/oracle_fixtures"
F = "mg2_drop_sed_reset_log_width"
WIDTH = 0.01
NAMES = ("mg2_prognostic_nc", "mg2_prognostic_nc_act", "mg2_prognostic_nc_sed",
         "mg2_prognostic_nc_melt", "mg2_prognostic_nc_icelim", "mg2_prognostic_nc_frz")
# GAP / GAP_QC: the sedimentation reset alone (every other switch at its hard
# limit), default vs hard.  Measured 2026-11-01; re-measure on any change of
# the reset or its width.
GAP = {
    "mg2_prognostic_nc": [5.616450710117173e-25, 2.640732016583119e-16, 3.263529440534234e-26, 4.9569208878966524e-24, 4.315215138918655e-12, 0.0],
    "mg2_prognostic_nc_act": [2.1651557344813035e-21, 9.242562058040917e-16, 7.0334686218410215e-28, 2.679405421408941e-25, 3.5064449895960335e-12, 0.0, 5.682508756471225e-28, 9.925452620000899e-23, 1.6588609454345603e-05],
    "mg2_prognostic_nc_sed": [2.1651557344813035e-21, 9.242562058040917e-16, 7.0334686218410215e-28, 2.679405421408941e-25, 3.5064449895960335e-12, 0.0, 5.682508756471225e-28, 9.925452620000899e-23, 1.6588609454345603e-05, 1.0014780483961052e-12, 0.0],
    "mg2_prognostic_nc_melt": [3.7177679963698624e-17, 3.0151092319686887e-15],
    "mg2_prognostic_nc_icelim": [9.509209159689351e-06, 3.76147016812981e-18, 1.860198571436116e-15, 5.189302913363615e-17, 4.242580835098282e-07, 5.446408028259016e-11, 8.571776780641698e-14, 2.869971904058757e-16, 1.3528856801884707e-16, 9.287549935948446e-09, 0.0, 0.0],
    "mg2_prognostic_nc_frz": [5.2709676608444104e-08, 1.178486663893354e-13, 1.556545287578363e-07, 1.5070100756849394e-16, 2.268197680825202e-15, 6.749835698644313e-16],
}
GAP_QC = {
    "mg2_prognostic_nc": [1.4131148420069516e-14, 1.1036720785513546e-15, 3.895926151980864e-15, 9.231426749457323e-15, 5.0244252648810175e-08, 0.0],
    "mg2_prognostic_nc_act": [3.419879847052067e-15, 1.2876174249765804e-15, 3.895926151980864e-15, 8.120071047420017e-16, 1.1370391416658245e-07, 0.0, 1.4111496276522527e-15, 3.505274371775327e-15, 1.4776574430447518e-05],
    "mg2_prognostic_nc_sed": [3.419879847052067e-15, 1.2876174249765804e-15, 3.895926151980864e-15, 8.120071047420017e-16, 1.1370391416658245e-07, 0.0, 1.4111496276522527e-15, 3.505274371775327e-15, 1.4776574430447518e-05, 1.5979859739100987e-14, 5.576243940600986e-15],
    "mg2_prognostic_nc_melt": [1.9203470094533595e-15, 4.293747002166408e-15],
    "mg2_prognostic_nc_icelim": [0.00029530776804399545, 3.436314896997659e-16, 4.9394148569023795e-14, 5.367558995838334e-15, 6.7777066624864436e-06, 1.3333007966571592e-09, 5.962701125922143e-14, 4.112191181818858e-15, 4.957728391345975e-15, 1.835742037127483e-07, 0.0, 0.0],
    "mg2_prognostic_nc_frz": [1.681564656447774e-08, 2.0659594268860234e-14, 1.7112186562395168e-07, 2.6671953334848736e-15, 3.3776911691972895e-15, 2.484608012765941e-15],
}


def _load(name):
    fx = dict(np.load(FXD / f"{name}.npz"))
    fx.setdefault("col_npccn", np.zeros_like(fx["col_T"]))
    return fx


def _hyd(fx):
    return {k: jnp.asarray(fx[f"col_{v}"]) for k, v in
            (("q_i", "q_i"), ("N_i", "ni_kg"), ("q_s", "q_s"), ("N_s", "ns_kg")) if f"col_{v}" in fx}


def _run(fx, width, **kw):
    cfg = _cfg(do_snow_to_graupel=False)._replace(**{F: width})
    return A._run(fx, cfg=cfg, hyd_kw=_hyd(fx), **kw)


def _sed_inputs(fx, monkeypatch, width=0.0):
    """(args, kwargs) of the real Morrison sedimentation call, and its output."""
    from legoesm.atmosphere.physics.microphysics import morrison as M
    seen = []
    real = M.mg2_cloud_sedimentation

    def spy(*a, **k):
        seen.append((a, k))
        return real(*a, **k)

    monkeypatch.setattr(M, "mg2_cloud_sedimentation", spy)
    out = _run(fx, width)
    monkeypatch.setattr(M, "mg2_cloud_sedimentation", real)
    assert len(seen) == 1
    return out, seen[0]


def _bounds_bind(args):
    """Falling cells where MG2's reset moves the number, split by which bound:
    the hard port's mean diameter (pgam+1)/lambda sits on 2 um / 50 um."""
    q_c, N_c_kg, lcldm, _, rho = args[:5]
    has_c = q_c / lcldm >= W.MG2_QSMALL
    q = jnp.where(has_c, q_c / lcldm, 0.0)
    n = jnp.where(has_c, jnp.maximum(N_c_kg / lcldm, 0.0), 0.0) * rho
    active, pgam, lam, nc_kg = W.mg2_droplet_psd(q, n, rho)
    d_mean = np.asarray((pgam + 1.0) / lam)
    moved = np.asarray(active) & ~np.isclose(np.asarray(nc_kg * rho), np.asarray(n),
                                              rtol=1e-12, atol=0.0)
    return (moved & np.isclose(d_mean, 2.0e-6, rtol=1e-12, atol=0.0),
            moved & np.isclose(d_mean, 50.0e-6, rtol=1e-12, atol=0.0))


def _gaps(fx):
    hard, soft = _run(fx, 0.0), _run(fx, WIDTH)
    out = {}
    for key, get, ora in (
            ("nc", lambda o: o.dN_c_dt, fx["ora_nctend"] * fx["col_rho"]),
            ("qc", lambda o: o.qc_budget["sedimentation"], fx["ora_qcsedten"])):
        d = np.abs(np.asarray(get(soft)) - np.asarray(get(hard)))
        out[key] = d.max(axis=1) / np.maximum(np.abs(ora).max(axis=1), 1e-300)
    return out


def test_defaults_and_hard_limit():
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    m = MorrisonConfig()
    assert getattr(m, F) == WIDTH == m.mg2_drop_early_reset_log_width
    assert ExperimentConfig._field_defaults[f"morrison_{F}"] == WIDTH
    assert HARD_LIMIT[F] == 0.0


def test_hard_limit_matches_mg2_where_the_bounds_bind(monkeypatch):
    """Whole-call droplet tendency and cloud-water sedimentation vs the RUNNING
    MG2 at the hard limit, on the sedimentation fixture, whose fall-speed reset
    has falling cells each bound moves; the default width does not match (the
    comparison can fail)."""
    fx = _load("mg2_prognostic_nc_sed")
    out, (args, _) = _sed_inputs(fx, monkeypatch)
    lo, hi = _bounds_bind(args)
    print("sed reset bound-binding cells: 2 um", int(lo.sum()), "50 um", int(hi.sum()))
    assert lo.sum() >= 1 and hi.sum() >= 1
    rho = fx["col_rho"]

    def errs(o):
        return {"nctend": _rel(np.asarray(o.dN_c_dt) / rho, fx["ora_nctend"]),
                "qcsedten": S._rel_flux(o.qc_budget["sedimentation"], fx["ora_qcsedten"],
                                        S._gross(fx, o))}

    hard = {k: float(v.max()) for k, v in errs(out).items()}
    assert max(hard.values()) <= TOL, hard
    soft = {k: float(v.max()) for k, v in errs(_run(fx, WIDTH)).items()}
    print("default vs MG2", soft)
    assert max(soft.values()) > 1e3 * TOL


@pytest.mark.parametrize("name", NAMES)
def test_default_gap_per_column_is_the_recorded_one(name):
    g = _gaps(_load(name))
    print(name, "sed-reset gap per column nc", [float(x) for x in g["nc"]],
          "qc", [float(x) for x in g["qc"]])
    for key, rec in (("nc", GAP[name]), ("qc", GAP_QC[name])):
        np.testing.assert_allclose(g[key], rec, rtol=1e-6, atol=1e-12)
        # a gap below the 1e-12 floor must not vanish (or appear) unnoticed
        assert np.array_equal(g[key] > 0.0, np.asarray(rec) > 0.0)


def test_default_width_reaches_the_sedimentation_call(monkeypatch):
    _, (_, k) = _sed_inputs(_load("mg2_prognostic_nc_sed"), monkeypatch, width=WIDTH)
    assert k["number_cap_log_width"] == WIDTH and k["slope_log_width"] == WIDTH


def _near_2um_bound(u):
    """Fall-coefficient inputs whose mean diameter sits ``3*u*WIDTH`` in ln N
    from the 2 um bound (u > 0: too many droplets, the side MG2 clips)."""
    q = 1.0e-6 * jnp.ones_like(u)
    one = jnp.ones_like(u)
    n_b = W.mg2_size_bounded_nc(q, 1.0e12 * one, one)
    return q, n_b * jnp.exp(3.0 * WIDTH * u), one


def test_gradients_through_the_fall_speed_reset():
    u = jnp.asarray([-2.0, -0.5, 0.5, 2.0])
    w = dict(number_cap_log_width=WIDTH, slope_log_width=WIDTH)
    T = 280.0 * jnp.ones_like(u)

    def f(x, **k):
        q, n, one = _near_2um_bound(x)
        fc, fnc = W.mg2_droplet_fall_coefficients(q[None], n[None], one[None], T[None],
                                                  one[None], presence_top=W.MG2_QSMALL, **k)
        return jnp.stack([jnp.log(fc[0]), jnp.log(fnc[0])])

    check_grads(lambda x: f(x, **w), (u,), order=2, modes=["rev"], eps=1e-3, atol=1e-6,
                rtol=1e-3)
    g = np.asarray(jax.grad(lambda x: jnp.sum(f(x, **w)[0]))(u))
    hard = np.asarray(jax.grad(lambda x: jnp.sum(f(x)[0]))(u))
    assert np.all(np.isfinite(g)) and np.all(g != 0.0)
    # MG2's clip pins the slope (no derivative of the fall speed in the number)
    # on its clipped side; the rounded one does not
    assert np.all(np.abs(hard[u > 0]) < 1e-2 * np.abs(g[u > 0]))


def test_whole_call_gradient_finite_at_the_default(monkeypatch):
    fx = _load("mg2_prognostic_nc_sed")
    _, (args, _) = _sed_inputs(fx, monkeypatch)
    lo, hi = _bounds_bind(args)
    binds = jnp.asarray(lo | hi)
    rho = jnp.asarray(fx["col_rho"])

    def f(nc):
        out = _run(fx, WIDTH, N_c=nc)
        return jnp.sum(jnp.where(binds, out.qc_budget["sedimentation"], 0.0))

    g = np.asarray(jax.grad(f)(jnp.asarray(fx["col_nc_kg"]) * rho))
    assert np.isfinite(g).all() and np.abs(g).max() > 0.0


def test_flat_field_validation_cli_and_decks():
    from scripts.run.run_amip import _postprocess_args, build_arg_parser, build_config_from_args

    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    from legoesm.driver.run_config_yaml import load_yaml_config
    f = f"morrison_{F}"
    kw = dict(microphysics="morrison", morrison_prognostic_droplet_number=True,
              cam6_aerosol_dir="/x")
    ExperimentConfig(**kw, **{f: 0.0}).validate_strict()
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match=f"{f} must be a finite float in"):
            ExperimentConfig(**kw, **{f: bad}).validate_strict()
    # read only on the prognostic path: the specified-number path never sediments droplets
    for k in (dict(microphysics="morrison"), dict(microphysics="kessler")):
        with pytest.raises(ValueError, match=f"{f}=0.0 is read only .* silently inert"):
            ExperimentConfig(**k, **{f: 0.0}).validate_strict()
    base = MorrisonConfig()
    assert thread_morrison_scalars(ExperimentConfig(**{f: 0.0}), "morrison", base) == \
        base._replace(**{F: 0.0})
    assert thread_morrison_scalars(ExperimentConfig(), "morrison", base) is base
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--convection", "tiedtke", "--microphysics", "morrison",
        "--morrison-mg2-drop-sed-reset-log-width", "0.0"]), parser))
    assert getattr(cfg, f) == 0.0
    amip = ROOT / "config/amip"
    arm = load_yaml_config(amip / "amip_production_prognostic_nc.yaml", build_arg_parser())
    assert arm[f] == WIDTH and arm["morrison_prognostic_droplet_number"] is True
    prod = load_yaml_config(amip / "amip_production.yaml", build_arg_parser())
    assert prod[f] == WIDTH       # named explicitly (RULE 3), inert there
    assert prod["morrison_prognostic_droplet_number"] is False


def test_specified_number_path_does_not_sediment_droplets(monkeypatch):
    """Production's path (the deck's resolved MorrisonConfig on the MG2 warm-rain
    oracle columns): the fall-speed reset is never called, so its width
    cannot change production."""
    from legoesm.atmosphere.physics.microphysics import morrison as M
    from tests.atmosphere.microphysics.unit import test_mg2_warm_rain_oracle as R

    def boom(*a, **k):
        raise AssertionError("droplet sedimentation called on the specified-number path")

    monkeypatch.setattr(M, "mg2_cloud_sedimentation", boom)
    prod = R._production_morrison()
    assert getattr(prod, F) == WIDTH and not prod.prognostic_droplet_number
    out = R._run_columns(R._fx(), prod)
    assert "sedimentation" not in out.qc_budget
