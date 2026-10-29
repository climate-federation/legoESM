"""Oracle match: OpenIFS cumulus convection CUMASTRN (``_ifs_cumastrn.cumastrn``).

Fixture ``oracle_fixtures/bechtold_ifs.npz``: the tier-2 audit's 24 columns
(16 L32, 5 L137, 3 dried-environment stress columns) and the outputs of
RUNNING OpenIFS 8f6f722f73 arpifs/phys_ec/cumastrn.F90 and its callees
(gfortran 8.5, fp64, every process of the shipped scheme on), in two arms:
``L`` legoESM constants + legoesm.thermo Tetens liquid curve, ``N`` the IFS
native constants (scripts/validate/oracle/bechtold_ifs/{build.sh,
make_fixture.py}).

Error metric (as tests/atmosphere/test_tiedtke_wrf_oracle.py): per field,
max|ours - oracle| / max|oracle| over all columns and levels.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.convection._ifs_cumastrn import cumastrn
from legoesm.atmosphere.physics.convection._ifs_cuthermo import (
    ifs_thermo, lego_thermo, satur_qsat,
)

_FIX = Path(__file__).parent / "oracle_fixtures" / "bechtold_ifs.npz"
_TOL = 1.0e-10  # user-approved oracle-match tolerance (max relative error, fp64)
_GROUPS = ("L32", "L137", "STRESS")
_REAL = {  # ours (field, index) -> oracle tag
    ("ptent", None): "PTENT", ("ptenq", None): "PTENQ", ("ptenu", None): "PTENU",
    ("ptenv", None): "PTENV", ("ptu", None): "PTU", ("pqu", None): "PQU",
    ("plu", None): "PLU", ("plude", None): "PLUDE", ("pludeli", 0): "PLUDELI1",
    ("pludeli", 1): "PLUDELI2", ("pludeli", 2): "PLUDELI3", ("pludeli", 3): "PLUDELI4",
    ("psnde", 0): "PSNDE1", ("psnde", 1): "PSNDE2", ("penth", None): "PENTH",
    ("pmflxr", None): "PMFLXR", ("pmflxs", None): "PMFLXS", ("prain", None): "PRAIN",
    ("plrain", None): "PLRAIN", ("prsud", 0): "PRSUD1", ("prsud", 1): "PRSUD2",
    ("pmfu", None): "PMFU", ("pmfd", None): "PMFD", ("pmfude_rate", None): "PMFUDE_RATE",
    ("pmfdde_rate", None): "PMFDDE_RATE", ("pcape", None): "PCAPE",
    ("pwmean", None): "PWMEAN", ("pvdiscu", None): "PVDISCU", ("pdiss", None): "PDISS",
    ("pqsen", None): "PQSEN",
}
_CONST = ("RG", "RD", "RV", "RCPD", "RLVTT", "RTT", "RA", "RLMLT", "R2ES", "R3LES",
          "R4LES", "R3IES", "R4IES")


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(_FIX))


def _thermo(fix, arm):
    c = dict(zip([str(n) for n in fix["const_names"]], fix[f"{arm}_const"]))
    th = ifs_thermo(**{k: float(c[k]) for k in _CONST})
    assert th.RPI == c["RPI"]
    if arm == "L":
        assert th == lego_thermo()  # the oracle ran exactly legoESM's constants
    return th


def _inputs(fix, g, dtype=jnp.float64):
    a = lambda k: jnp.asarray(fix[f"{g}_{k}"], dtype)
    zero = jnp.zeros_like(a("T"))
    return dict(pgaw=a("pgaw"), T=a("T"), q=a("q"), u=a("u"), v=a("v"), plitot=a("lit"),
                omega=a("omega"), ahfs=a("ahfs"), qhfl=a("qhfl"), p_full=a("p_full"),
                p_half=a("p_half"), geo_full=a("geo_full"), geo_half=a("geo_half"),
                land=a("land"), ptent=a("dT_other"), ptenq=a("dq_other"), ptenu=zero,
                ptenv=zero, ptenta=a("dT_adv"), ptenqa=a("dq_adv"))


def _run(fix, arm, g, dtype=jnp.float64, fn=cumastrn):
    th = _thermo(fix, arm)
    x = _inputs(fix, g, dtype)
    # PQSEN exactly as the driver forms it (SATUR, KFLAG = 0)
    qsat = satur_qsat(th, x["T"], x["p_full"])
    return fn(th, float(fix["dt"]), qsat=qsat, **x)


def _err(v, ref):
    v = np.asarray(v, np.float64)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(ref))):
        return np.inf
    return float(np.max(np.abs(v - ref)) / max(np.max(np.abs(ref)), 1e-300))


def _field(out, k, i):
    v = np.asarray(getattr(out, k))
    return v if i is None else v[:, i]


@pytest.mark.parametrize("arm", ["L", "N"])
@pytest.mark.parametrize("g", _GROUPS)
def test_cumastrn_matches_openifs(fix, arm, g):
    out = _run(fix, arm, g)
    o = lambda t: fix[f"{arm}_{g}_{t}"]
    ldcum = o("LDCUM") > 0
    np.testing.assert_array_equal(np.asarray(out.ldcum), ldcum)
    np.testing.assert_array_equal(np.asarray(out.ktype), o("KTYPE"))
    np.testing.assert_array_equal(np.asarray(out.kcbot)[ldcum], o("KCBOT")[ldcum])
    np.testing.assert_array_equal(np.asarray(out.kctop)[ldcum], o("KCTOP")[ldcum])
    np.testing.assert_array_equal(np.asarray(out.ldsc), o("LDSC") > 0)
    np.testing.assert_array_equal(np.asarray(out.kbotsc), o("KBOTSC"))
    errs = {t: _err(_field(out, k, i), o(t)) for (k, i), t in _REAL.items()}
    print(arm, g, "max rel err", {k: f"{v:.2e}" for k, v in errs.items()})
    bad = {k: v for k, v in errs.items() if not v <= _TOL}
    assert not bad, f"{arm}/{g}: fields above {_TOL}: {bad}"


def test_fixture_exercises_every_branch(fix):
    """Deep, shallow and mid-level clouds, a downdraught, snow, melting-era
    precipitation and sub-cloud evaporation occur in the oracle data.
    Coverage limits: omega = 0 in every column (the resolved-ascent closure
    gate is not exercised), land in {0, 1} only."""
    for arm in ("L", "N"):
        kt = np.concatenate([fix[f"{arm}_{g}_KTYPE"][fix[f"{arm}_{g}_LDCUM"] > 0]
                             for g in _GROUPS])
        assert {1, 2, 3} <= set(kt.tolist()), arm
        assert min(np.min(fix[f"{arm}_{g}_PMFD"]) for g in _GROUPS) < 0.0
        assert max(np.max(fix[f"{arm}_{g}_PMFLXS"]) for g in _GROUPS) > 0.0
        assert max(np.max(fix[f"{arm}_{g}_PSNDE2"]) for g in _GROUPS) > 0.0


def test_jit_matches_eager(fix):
    eager = _run(fix, "L", "L32")
    jitted = _run(fix, "L", "L32", fn=jax.jit(cumastrn, static_argnums=(0, 1)))
    for (k, i), _t in _REAL.items():  # same metric and tolerance as the oracle match
        assert _err(_field(jitted, k, i), _field(eager, k, i)) <= _TOL, k


@pytest.mark.parametrize("ptop0", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_gradient_finite_on_fixture_columns(fix, dtype, ptop0):
    """jax.grad of column heating, drying, momentum tendency and surface
    precipitation w.r.t. T, q, u, pressure, geopotential and the surface
    heat fluxes is finite on every L32 fixture column, also with the top
    interface pressure set to 0 (as in IFS L137)."""
    th = _thermo(fix, "L")
    x = _inputs(fix, "L32", dtype)
    if ptop0:
        x["p_half"] = x["p_half"].at[:, 0].set(0.0)
    dt = float(fix["dt"])
    keys = ("T", "q", "u", "p_full", "p_half", "geo_full", "geo_half", "ahfs", "qhfl")

    def loss(*vals):
        y = {**x, **dict(zip(keys, vals))}
        o = cumastrn(th, dt, qsat=satur_qsat(th, y["T"], y["p_full"]), **y)
        return (jnp.sum(o.ptent) + 1.0e3 * jnp.sum(o.ptenq) + jnp.sum(o.ptenu)
                + jnp.sum(o.pmflxr[:, -1] + o.pmflxs[:, -1]))

    grads = jax.grad(loss, argnums=tuple(range(len(keys))))(*(x[k] for k in keys))
    for gr in grads:
        assert np.all(np.isfinite(np.asarray(gr)))
    assert np.any(np.asarray(grads[0]) != 0.0)


@pytest.mark.parametrize("arm", ["L", "N"])
def test_column_water_budget(fix, arm):
    """Column water closes in flux form: the vapour tendency the scheme adds,
    plus the detrained condensate and detrained precipitation it hands on,
    plus the surface precipitation, sum to zero (fp64)."""
    res, dry = [], []
    for g in _GROUPS:
        out = _run(fix, arm, g)
        dp = np.diff(fix[f"{g}_p_half"], axis=1)
        dq = np.asarray(out.ptenq) - fix[f"{g}_dq_other"]
        col = np.sum(dq * dp, axis=1) / _thermo(fix, arm).RG
        res.append(col + np.sum(np.asarray(out.plude) + np.asarray(out.psnde).sum(axis=1), axis=1)
                   + np.asarray(out.pmflxr)[:, -1] + np.asarray(out.pmflxs)[:, -1])
        dry.append(col)
    # the oracle-match metric: max |residual| / max |column drying| over all columns
    rel = np.max(np.abs(np.concatenate(res))) / np.max(np.abs(np.concatenate(dry)))
    print(arm, "water residual / max column drying", rel)
    assert rel <= _TOL


_DX = 1.0e5  # m, the grid length make_fixture.py encoded in PGAW


def test_wrapper_matches_openifs(fix):
    """The BechtoldConfig.use_ifs_ascent path, seam included (pipeline
    surface fluxes negated into the IFS sign, PGAW from dx_m, hydrostatic
    geopotential, SATUR saturation, PLITOT = 0), against the oracle run on
    the same inputs with PLITOT = 0 (arm L0)."""
    from legoesm.atmosphere.physics.convection._ifs_faithful import ifs_faithful_convection
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig
    from legoesm.atmosphere.physics.convection.output import ConvectionOutput

    for g in _GROUPS:
        a = lambda k: jnp.asarray(fix[f"{g}_{k}"])
        out, mu = ifs_faithful_convection(
            a("T"), a("q"), a("p_full"), a("p_half"), a("u"), a("v"), float(fix["dt"]),
            BechtoldConfig(use_ifs_ascent=True, dx_m=_DX), a("shf_up"), a("lhf_up"),
            a("land"), a("dT_adv"), a("dq_adv"), a("dT_other"), a("dq_other"), a("omega"),
            ConvectionOutput)
        o = lambda t: fix[f"L0_{g}_{t}"]
        g_dp = lego_thermo().RG / np.diff(fix[f"{g}_p_half"], axis=1)
        errs = {
            "dT_dt": _err(out.dT_dt, o("PTENT") - fix[f"{g}_dT_other"]),
            "dq_v_dt": _err(out.dq_v_dt, o("PTENQ") - fix[f"{g}_dq_other"]),
            "du_dt": _err(out.du_dt_conv, o("PTENU")),
            "dv_dt": _err(out.dv_dt_conv, o("PTENV")),
            "dq_c_dt": _err(out.dq_c_conv_dt, g_dp * o("PLUDE")),
            "cape": _err(out.cape, o("PCAPE")),
            "M_u": _err(mu, o("PMFU")),
        }
        np.testing.assert_array_equal(np.asarray(out.convective_mask), o("LDCUM") > 0)
        print("wrapper", g, {k: f"{v:.2e}" for k, v in errs.items()})
        bad = {k: v for k, v in errs.items() if not v <= _TOL}
        assert not bad, f"wrapper/{g}: fields above {_TOL}: {bad}"


def test_wrapper_rejects_fields_it_does_not_read():
    from legoesm.atmosphere.physics.convection._ifs_faithful import _READ_FIELDS, _check_config
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    base = BechtoldConfig(use_ifs_ascent=True, dx_m=_DX)
    _check_config(base)
    f = next(k for k, v in base._asdict().items()
             if isinstance(v, float) and k not in _READ_FIELDS)
    with pytest.raises(ValueError, match=f):
        _check_config(base._replace(**{f: getattr(base, f) + 1.0}))
    with pytest.raises(ValueError, match="dx_m"):
        _check_config(base._replace(dx_m=0.0))


def test_wrapper_tunable_defaults_are_sucumf():
    """At BechtoldConfig defaults the six threaded constants equal sucumf's
    exactly, so the default path runs the oracle-matched arithmetic."""
    from legoesm.atmosphere.physics.convection._ifs_cumastrn import IfsTunables
    from legoesm.atmosphere.physics.convection._ifs_faithful import _tunables
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    assert _tunables(BechtoldConfig(use_ifs_ascent=True, dx_m=_DX)) == IfsTunables()


def _wrapper_out(fix, cfg, g="L32"):
    from legoesm.atmosphere.physics.convection._ifs_faithful import ifs_faithful_convection
    from legoesm.atmosphere.physics.convection.output import ConvectionOutput

    a = lambda k: jnp.asarray(fix[f"{g}_{k}"])
    return ifs_faithful_convection(
        a("T"), a("q"), a("p_full"), a("p_half"), a("u"), a("v"), float(fix["dt"]), cfg,
        a("shf_up"), a("lhf_up"), a("land"), a("dT_adv"), a("dq_adv"), a("dT_other"),
        a("dq_other"), a("omega"), ConvectionOutput)


def _wrapper_loss(out, mu):
    return (jnp.sum(out.dT_dt) + 1.0e3 * jnp.sum(out.dq_v_dt)
            + 1.0e3 * jnp.sum(out.dq_r_conv_dt) + jnp.sum(mu))


@pytest.mark.parametrize("field", ["epsilon_deep", "delta_deep", "epsilon_shallow",
                                   "rprcon", "dnoprc", "cape_qadv_weight"])
def test_wrapper_tunable_reaches_cumastrn(fix, field):
    """Each threaded BechtoldConfig field carries a finite, nonzero gradient
    and a non-default value changes the convective tendencies."""
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    base = BechtoldConfig(use_ifs_ascent=True, dx_m=_DX)
    v0 = getattr(base, field)
    grad = jax.grad(lambda v: _wrapper_loss(*_wrapper_out(fix, base._replace(**{field: v}))))(v0)
    assert np.isfinite(float(grad)) and float(grad) != 0.0, (field, float(grad))
    ref, _ = _wrapper_out(fix, base)
    alt, _ = _wrapper_out(fix, base._replace(**{field: 1.5 * v0 if field != "cape_qadv_weight" else 0.5}))
    assert np.max(np.abs(np.asarray(alt.dT_dt) - np.asarray(ref.dT_dt))) > 0.0, field


# --- #2209 (main): reverse-mode gradients through the clamped square roots,
# carried onto this port.  Sounding and kwargs as main's test_ifs_faithful.py.

def _deep_column(nlev=30, ps=101300.0):
    from legoesm import constants
    from legoesm.atmosphere.physics.thermodynamics import compute_moist_adiabat
    from legoesm.thermo import saturation_specific_humidity

    sig_h = np.linspace(0.0, 1.0, nlev + 1)
    p_half = (sig_h * ps)[None, :].astype(np.float32).copy()
    p_half[0, 0] = max(p_half[0, 0], 1000.0)
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    pf = p_full[0].astype(np.float64)
    T0, q0 = 300.0, 0.017
    rcpl = constants.R_d / constants.c_pd
    Tad = np.asarray(compute_moist_adiabat(
        jnp.array([T0]), jnp.array(pf)[None, :], jnp.array([q0])))[0]
    T = np.where(pf > 95000, T0 * (pf / ps) ** rcpl, Tad - 1.0)
    T = np.where(pf < 15000, np.maximum(T, 200.0), T)
    qs = np.asarray(saturation_specific_humidity(jnp.array(T), jnp.array(pf)))
    q = np.minimum(np.where(pf > 95000, q0, 0.8 * qs), q0)
    return (jnp.asarray(T[None, :], jnp.float32), jnp.asarray(q[None, :], jnp.float32),
            jnp.asarray(p_full), jnp.asarray(p_half))


def _kwargs(ncol, nlev):
    z = jnp.zeros((ncol, nlev))
    return dict(u=z, v=z, conv_prog_profile=z, conv_stoch_state=jnp.zeros(ncol),
                prng_key=None, dt=112.5, land_frac=jnp.zeros(ncol),
                shf_w_m2=jnp.full((ncol,), -20.0),
                lhf_w_m2=jnp.full((ncol,), -100.0),
                dT_dt_dyn=z, dq_dt_dyn=z)


@pytest.fixture
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _param_grad_and_fd(nlev, field):
    """d(sum dT_dt)/d(field) on the deep sounding: reverse mode and a central
    difference with relative step 1e-4."""
    from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    T, q, p_full, p_half = [a.astype(jnp.float64) for a in _deep_column(nlev=nlev)]
    kw = _kwargs(*T.shape)
    base = BechtoldConfig(use_ifs_ascent=True, dx_m=_DX)
    v = float(getattr(base, field))

    def f(x):
        out, _, _ = bechtold_convection(T, q, p_full, p_half,
                                        config=base._replace(**{field: x}), **kw)
        return jnp.sum(out.dT_dt)

    g = float(jax.grad(f)(jnp.asarray(v)))
    h = 1e-4 * v
    fd = (float(f(jnp.asarray(v + h))) - float(f(jnp.asarray(v - h)))) / (2 * h)
    return g, fd


@pytest.mark.parametrize("field", ["epsilon_deep", "rprcon"])
def test_reverse_grad_matches_fd_on_convecting_column(_x64, field):
    g, fd = _param_grad_and_fd(90, field)
    assert fd != 0.0
    assert g == pytest.approx(fd, rel=1e-3)


def _stable_column(nlev=30, ps=101300.0):
    """Dry, statically stable sounding (potential temperature rising 40 K
    across the column, 0.1 g/kg): nothing convects."""
    from legoesm import constants

    p_half = np.linspace(0.0, ps, nlev + 1)[None, :]
    p_half[0, 0] = 1000.0
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    theta = 300.0 + 40.0 * (1.0 - p_full / ps)
    T = theta * (p_full / ps) ** (constants.R_d / constants.c_pd)
    q = np.full_like(T, 1.0e-4)
    return tuple(jnp.asarray(a, jnp.float64) for a in (T, q, p_full, p_half))


@pytest.mark.parametrize("field", ["epsilon_deep", "rprcon"])
def test_reverse_grad_finite_zero_where_nothing_convects(_x64, field):
    """#2209's clamped-zero case: on a column where nothing convects the
    gradient is the finite zero the finite difference measures."""
    from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    T, q, p_full, p_half = _stable_column()
    kw = _kwargs(*T.shape)
    base = BechtoldConfig(use_ifs_ascent=True, dx_m=_DX)
    out, _, _ = bechtold_convection(T, q, p_full, p_half, config=base, **kw)
    assert not bool(np.any(np.asarray(out.convective_mask)))
    v = float(getattr(base, field))

    def f(x):
        o, _, _ = bechtold_convection(T, q, p_full, p_half,
                                      config=base._replace(**{field: x}), **kw)
        return jnp.sum(o.dT_dt)

    g = float(jax.grad(f)(jnp.asarray(v)))
    h = 1e-4 * v
    fd = (float(f(jnp.asarray(v + h))) - float(f(jnp.asarray(v - h)))) / (2 * h)
    assert np.isfinite(g)
    assert g == fd == 0.0


@pytest.mark.parametrize("g", _GROUPS)
def test_wrapper_hands_the_host_water_that_closes(fix, g):
    """What the host receives -- vapour tendency, cloud-water source and rain
    source -- sums to zero over the column.  Same metric as the oracle
    budget."""
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    out, _ = _wrapper_out(fix, BechtoldConfig(use_ifs_ascent=True, dx_m=_DX), g)
    dp = np.diff(fix[f"{g}_p_half"], axis=1)
    rg = _thermo(fix, "L").RG
    col = lambda a: np.sum(np.asarray(a) * dp, axis=1) / rg
    dry = col(out.dq_v_dt)
    res = (dry + col(out.dq_c_conv_dt) + col(out.dq_r_conv_dt)
           + col(out.dq_r_detr_dt) + col(out.dq_s_detr_dt))
    if np.max(np.abs(dry)) == 0.0:
        assert np.max(np.abs(res)) == 0.0
        return
    assert np.max(np.abs(res)) / np.max(np.abs(dry)) <= _TOL


@pytest.mark.parametrize("g", _GROUPS)
def test_wrapper_surface_rain_is_cumastrn_surface_flux(fix, g, monkeypatch):
    """S1'' (user 2026-10-28): the host's surface rain (the column sum of the
    rain source) is CUMASTRN's surface rain + snow flux, and the detrained
    rain and snow (PSNDE) go to the host's species instead."""
    from legoesm.atmosphere.physics.convection import _ifs_faithful as F
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    cap, orig = {}, F.cumastrn
    def spy(*a, **k):
        cap["o"] = orig(*a, **k)
        return cap["o"]
    monkeypatch.setattr(F, "cumastrn", spy)
    out, _ = _wrapper_out(fix, BechtoldConfig(use_ifs_ascent=True, dx_m=_DX), g)
    o = cap["o"]
    dp = np.diff(fix[f"{g}_p_half"], axis=1)
    col = lambda a: np.sum(np.asarray(a) * dp, axis=1) / _thermo(fix, "L").RG
    sfc = np.asarray(o.pmflxr)[:, -1] + np.asarray(o.pmflxs)[:, -1]
    psnde = np.asarray(o.psnde)
    pairs = ((col(out.dq_r_conv_dt), sfc),
             (col(out.dq_r_detr_dt), psnde[:, 0].sum(-1)),
             (col(out.dq_s_detr_dt), psnde[:, 1].sum(-1)))
    for got, want in pairs:
        scale = np.max(np.abs(want))
        if scale == 0.0:
            assert np.max(np.abs(got)) == 0.0
        else:
            assert np.max(np.abs(got - want)) / scale <= _TOL


def test_wrapper_publishes_the_surface_snow_flux(fix):
    """F29: the frozen surface precipitation is CUMASTRN's surface snow flux,
    within the column's surface precipitation."""
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    out, _ = _wrapper_out(fix, BechtoldConfig(use_ifs_ascent=True, dx_m=_DX))
    snow = np.asarray(out.precip_frozen_sfc)
    dp = np.diff(fix["L32_p_half"], axis=1)
    total = np.sum(np.asarray(out.dq_r_conv_dt) * dp, axis=1) / _thermo(fix, "L").RG
    assert snow.shape == total.shape and np.all(snow >= 0.0)
    assert np.any(snow > 0.0)
    assert np.all(snow <= total)


def test_clamped_square_roots_pass_nan_through(fix, monkeypatch):
    """#2209 (user-approved): the cloud-base velocity, organised-detrainment
    and Bergeron square roots let a NaN through instead of masking it to 0.
    Every one of the four call sites must reach the shared helper."""
    import inspect

    from legoesm.atmosphere.physics.convection import _ifs_cuascent as A
    from legoesm.core.profile_metrics import safe_sqrt_nan

    sites = set()

    def spy(x):
        sites.add(inspect.stack()[1].lineno)
        return safe_sqrt_nan(x)

    monkeypatch.setattr(A, "safe_sqrt_nan", spy)
    with jax.disable_jit():
        _run(fix, "L", _GROUPS[0])
    assert len(sites) == 4, sorted(sites)
