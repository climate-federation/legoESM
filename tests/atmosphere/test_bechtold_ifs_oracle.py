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
from legoesm.atmosphere.physics._ifs_thermo import (
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
    the same inputs with PLITOT = 0 (arm L0).  Runs the source no-supply
    shallow closure ("first_guess") with the exact switches
    (ifs_hard_switches=True, the RULE 5 hard limit, #2281); the default "zero"
    ramp (main #2253) and the default smooth switches are recorded departures
    whose per-column gaps are test_nosupply_default_gap_per_case and
    test_smooth_switches_default_gap_per_case."""
    from legoesm.atmosphere.physics.convection._ifs_faithful import ifs_faithful_convection
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig
    from legoesm.atmosphere.physics.convection.output import ConvectionOutput

    for g in _GROUPS:
        a = lambda k: jnp.asarray(fix[f"{g}_{k}"])
        out, mu = ifs_faithful_convection(
            a("T"), a("q"), a("p_full"), a("p_half"), a("u"), a("v"), float(fix["dt"]),
            BechtoldConfig(use_ifs_ascent=True, dx_m=_DX, ifs_shallow_nosupply="first_guess",
                           ifs_hard_switches=True),
            a("shf_up"), a("lhf_up"),
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

    tun = _tunables(BechtoldConfig(use_ifs_ascent=True, dx_m=_DX))
    # main #2253 (user 2026-10-28): the run's default no-supply shallow mode
    # is "zero", a recorded departure; the default smooth switches (RULE 5,
    # #2281) another; everything else is sucumf's.
    assert tun == IfsTunables()._replace(shallow_nosupply="zero", hard_switches=False)


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
# carried onto this port.  Sounding and kwargs as the removed reduced chain's
# tests/unit/test_ifs_faithful.py.

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


# --- main #2253 (user 2026-10-28): no-supply shallow closure, RULE 5 -------
# "first_guess" is the source and is the hard limit the oracle tests above
# run; the default "zero" closes on smooth_max(ZDHPBL, 0) of width 1 W/m2.

def _wrapper_cols(fix, g, **cfg_kw):
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    out, mu = _wrapper_out(fix, BechtoldConfig(use_ifs_ascent=True, dx_m=_DX, **cfg_kw), g)
    return np.asarray(out.dT_dt), np.asarray(out.dq_v_dt), np.asarray(mu)


def _per_column_gap(a, ref):
    """Per column: max|a - ref| / max|ref| (0 where the column's ref is 0)."""
    num = np.max(np.abs(a - ref), axis=1)
    den = np.max(np.abs(ref), axis=1)
    return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), num)


@pytest.mark.parametrize("g", _GROUPS)
def test_nosupply_first_guess_is_the_oracle(fix, g):
    dT, dq, mu = _wrapper_cols(fix, g, ifs_shallow_nosupply="first_guess",
                               ifs_hard_switches=True)
    o = lambda t: fix[f"L0_{g}_{t}"]
    assert _err(dT, o("PTENT") - fix[f"{g}_dT_other"]) <= _TOL
    assert _err(dq, o("PTENQ") - fix[f"{g}_dq_other"]) <= _TOL
    assert _err(mu, o("PMFU")) <= _TOL


# Per-column gap of the default ("zero", 1 W/m2) against the oracle, measured
# 2026-10-29 (x64, CPU): (dT_dt, dq_v_dt, M_u) per fixture column.
_NOSUPPLY_GAP = {
    "L32": (
        [1.155408049270849e-13, 6.382851606121968e-13, 9.329321598065945e-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.527033819234069e-16, 4.2997772808050026e-14, 0.0, 0.0, 4.061234644816772e-12, 8.597091978769083e-13, 1.5572484289018971e-12],
        [1.2208611804052711e-14, 8.208652670311285e-14, 3.7711641379674507e-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.768519006471387e-15, 0.0, 0.0, 7.768887789727816e-13, 5.056799592526267e-14, 1.101205278761102e-13],
        [1.4981262394187682e-14, 9.01139097311247e-14, 3.817870639265198e-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0292097072507253e-15, 6.920393684326509e-16, 0.0, 0.0, 7.545950343192183e-13, 4.6335526034809616e-14, 3.560926557233197e-14],
    ),
    "L137": (
        [0.005643546933347423, 0.0, 0.0, 0.0, 0.0],
        [0.005643546936583219, 0.0, 0.0, 0.0, 0.0],
        [0.006329754723205587, 0.0, 0.0, 0.0, 0.0],
    ),
    "STRESS": (
        [1.722849855972394e-13, 4.5420032351625236e-14, 1.775404919440501e-12],
        [3.9659361647976153e-14, 7.704682611898458e-15, 4.996926756862235e-13],
        [1.1361043895744794e-13, 6.838908281453185e-16, 5.255835484158918e-13],
    ),
}


@pytest.mark.parametrize("g", _GROUPS)
def test_nosupply_default_gap_per_case(fix, g):
    dT, dq, mu = _wrapper_cols(fix, g, ifs_hard_switches=True)
    o = lambda t: fix[f"L0_{g}_{t}"]
    got = (_per_column_gap(dT, o("PTENT") - fix[f"{g}_dT_other"]),
           _per_column_gap(dq, o("PTENQ") - fix[f"{g}_dq_other"]),
           _per_column_gap(mu, o("PMFU")))
    for a, want in zip(got, _NOSUPPLY_GAP[g]):
        np.testing.assert_allclose(a, np.asarray(want), rtol=1e-6, atol=1e-12)


def test_nosupply_guards_raise(fix):
    from legoesm.atmosphere.physics.convection._ifs_cumastrn import IfsTunables

    for bad, match in ((dict(shallow_nosupply="typo"), "unknown shallow-closure"),
                       (dict(shallow_supply_width_w_m2=-1.0), "shallow_supply_width_w_m2")):
        with pytest.raises(ValueError, match=match):
            _run(fix, "L", "L32", fn=lambda *a, **k: cumastrn(*a, tun=IfsTunables(**bad), **k))


def test_wrapper_rejects_nonpositive_epsilon_deep():
    from legoesm.atmosphere.physics.convection._ifs_faithful import _check_config
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    for v in (0.0, -1.0e-4):
        with pytest.raises(ValueError, match="epsilon_deep > 0"):
            _check_config(BechtoldConfig(use_ifs_ascent=True, dx_m=_DX, epsilon_deep=v))


def test_bechtold_hands_the_seam_positive_up_fluxes(_x64, monkeypatch):
    """#2252 on this port: bechtold_convection forwards the model's
    positive-upward fluxes unchanged; the seam negates them once into the IFS
    PAHFS/PQHFL sign (test_wrapper_matches_openifs).  A second negation at the
    bechtold call would flip the trigger's surface parcel."""
    from legoesm.atmosphere.physics.convection import _ifs_faithful as F
    from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    cap, orig = {}, F.ifs_faithful_convection
    def spy(*a, **k):
        cap.update(k)
        return orig(*a, **k)
    monkeypatch.setattr(F, "ifs_faithful_convection", spy)
    T, q, p_full, p_half = [a.astype(jnp.float64) for a in _deep_column()]
    kw = _kwargs(*T.shape)
    kw.update(shf_w_m2=jnp.full((1,), 20.0), lhf_w_m2=jnp.full((1,), 100.0))
    bechtold_convection(T, q, p_full, p_half,
                        config=BechtoldConfig(use_ifs_ascent=True, dx_m=_DX), **kw)
    assert float(cap["shf_w_m2"][0]) == 20.0 and float(cap["lhf_w_m2"][0]) == 100.0


def test_nosupply_gradient_through_the_supply_ramp(fix):
    """RULE 5 gradient test at the default: L137 column 0 is the fixture
    column whose shallow supply sits on the smooth ramp (its default gap above
    is the only nonzero one).  check_grads (order 2, reverse) of the column's
    updraught mass flux with respect to a scale on the pre-convection
    tendencies that set ZDHPBL; the derivative differs from the hard max's."""
    from legoesm.atmosphere.physics.convection._ifs_faithful import ifs_faithful_convection
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig
    from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    from jax.test_util import check_grads

    a = lambda k: jnp.asarray(fix[f"L137_{k}"])[:1]  # noqa: E731

    def f(scale, width):
        cfg = BechtoldConfig(use_ifs_ascent=True, dx_m=_DX, ifs_hard_switches=True,
                             ifs_shallow_supply_width_w_m2=width)
        _, mu = ifs_faithful_convection(
            a("T"), a("q"), a("p_full"), a("p_half"), a("u"), a("v"), float(fix["dt"]),
            cfg, a("shf_up"), a("lhf_up"), a("land"), a("dT_adv"), a("dq_adv"),
            scale * a("dT_other"), scale * a("dq_other"), a("omega"), ConvectionOutput)
        return jnp.sum(mu)

    width = BechtoldConfig().ifs_shallow_supply_width_w_m2
    one = jnp.asarray(1.0)
    check_grads(lambda s: f(s, width), (one,), order=2, modes=("rev",), eps=1e-6)
    g = float(jax.grad(lambda s: f(s, width))(one))
    g_hard = float(jax.grad(lambda s: f(s, 0.0))(one))
    assert np.isfinite(g) and g != 0.0 and g != g_hard


# --- RULE 5, faithful chain's elementwise switches (#2281) -----------------
# Per-column gap of the default smooth switches (ifs_hard_switches=False)
# against the oracle (arm L0), the no-supply closure held at the source
# ("first_guess") so only the switches differ: (dT_dt, dq_v_dt, M_u, surface
# precipitation) per fixture column, measured (x64, CPU) by switch_gaps().
_SWITCH_GAP = {
    "L32": (
        [1.155408049270849e-13, 6.381140202323673e-13, 0.0011130719528878582, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.527033819234069e-16, 4.198668525995863e-05, 0.0, 0.07037075107906984, 4.0608539898861155e-12, 1.7179434496712484e-12, 1.5572484289018971e-12],
        [1.7172552867238978e-14, 7.531650388223756e-14, 0.0010985278742337056, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.689802829971999e-05, 0.0, 0.013070933253375527, 7.768887789727816e-13, 5.056799592526267e-14, 1.101205278761102e-13],
        [1.4365594076618328e-14, 7.842070846841177e-14, 0.0011884180672240485, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0292097072507253e-15, 5.127344626210799e-05, 0.0, 0.0, 7.555190282387929e-13, 4.719359133175054e-14, 3.560926557233197e-14],
        [3.975688633527109e-15, 8.820962404225116e-14, 0.0010549443125625845, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.4518722729335297e-14, 5.1273446260288524e-05, 0.0, 0.0, 7.36695393170115e-13, 3.9271499380901464e-14, 0.0],
    ),
    "L137": (
        [1.661612930475599e-11, 0.0, 0.0, 0.0, 0.0],
        [1.2046143070100942e-13, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ),
    "STRESS": (
        [1.722849855972394e-13, 0.012694584019058777, 1.775547491707708e-12],
        [4.407539335181932e-14, 0.014005735274384364, 4.996926756862235e-13],
        [1.2783764286336544e-13, 0.015307609436948059, 5.235744369922836e-13],
        [1.0640664001275355e-15, 0.0153076094369521, 4.0989188218272814e-13],
    ),
}


def _switch_cols(fix, g, hard):
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    out, mu = _wrapper_out(fix, BechtoldConfig(use_ifs_ascent=True, dx_m=_DX,
                                               ifs_shallow_nosupply="first_guess",
                                               ifs_hard_switches=hard), g)
    return (np.asarray(out.dT_dt), np.asarray(out.dq_v_dt), np.asarray(mu),
            np.asarray(out.dq_r_conv_dt), out)


def switch_gaps(fix, g):
    """(dT_dt, dq_v_dt, M_u, surface precip) per-column gaps, default vs oracle."""
    dT, dq, mu, _, out = _switch_cols(fix, g, False)
    o = lambda t: fix[f"L0_{g}_{t}"]  # noqa: E731
    sfc = np.asarray(o("PMFLXR"))[:, -1] + np.asarray(o("PMFLXS"))[:, -1]
    dp = np.diff(fix[f"{g}_p_half"], axis=1)
    got_sfc = np.sum(np.asarray(out.dq_r_conv_dt) * dp, axis=1) / lego_thermo().RG
    return (_per_column_gap(dT, o("PTENT") - fix[f"{g}_dT_other"]),
            _per_column_gap(dq, o("PTENQ") - fix[f"{g}_dq_other"]),
            _per_column_gap(mu, o("PMFU")),
            _per_column_gap(got_sfc[:, None], sfc[:, None]))


@pytest.mark.parametrize("g", _GROUPS)
def test_smooth_switches_default_gap_per_case(fix, g):
    for a, want in zip(switch_gaps(fix, g), _SWITCH_GAP[g]):
        np.testing.assert_allclose(a, np.asarray(want), rtol=1e-6, atol=1e-12)


def test_smooth_switches_are_the_default():
    from legoesm.atmosphere.physics.convection._ifs_cumastrn import IfsTunables
    from legoesm.atmosphere.physics.convection._ifs_faithful import _tunables
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    assert BechtoldConfig().ifs_hard_switches is False
    assert _tunables(BechtoldConfig(use_ifs_ascent=True, dx_m=_DX)).hard_switches is False
    assert IfsTunables().hard_switches is True   # the source, for direct oracle calls
    with pytest.raises(ValueError, match="hard_switches"):
        cumastrn(None, 1.0, *([None] * 21), tun=IfsTunables(hard_switches=1))


def _probe(name):
    import importlib.util

    path = (Path(__file__).resolve().parents[2] / "scripts" / "validate" / "oracle"
            / "bechtold_ifs" / f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Per switch SITE (stripped source line, IfsSmooth method):
# (group, column, mode, direction, value, fd step, partners), found by
# scripts/validate/oracle/bechtold_ifs/switch_sites.py.  mode "input": the
# input perturbation ``direction`` at ``value`` puts the site inside its
# window and the test variable is that perturbation.  mode "inject" (sites no
# input perturbation reached: floors and caps real columns rarely hit; vote
# 2026-11-14): the column (optionally perturbed by ``direction``/``value``)
# with only this site's switched operand moved to mid-window at s = 1 (offset
# under stop_gradient; s spans the window in units of 1) and each ``partners``
# site held mid-window, so the switch reaches the output.
_SITE_POINTS = {
    ('* (1.0 - sm.lt(kget(pvervel, njkt5), ZOMEGA_RES, _W_OMEGA)))', 'lt'):
        ('L32', 1, 'inject', 'q', 1.0586956521739133, 0.0001, ()),
    ('llo1 = sm.lt(paph[K + 1] - kget(paph, ikd), ZPBL_DP, _W_PRES)', 'lt'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('m = sm.min(zm, r, _W_REL * r)', 'min'):
        ('L32', 12, 'input', 'omega', -11.82065217391305, 1.1745622232621571e-05, ()),
    ('sm.max(zcape - zcapdcycl, RMINCAPE * zcape, _W_CAPE))', 'max'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('weak = (sm.gt(m2, ZSHAL_MF * zmfmax / RMFCFL, _W_MF)', 'gt'):
        ('L137', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zcape_c = sm.min(zcape_c, ZCAPE_MAX, _W_CAPE)', 'min'):
        ('L32', 9, 'inject', 'q', 1.075, 0.0001, ()),
    ('zcape_c = sm.sel(br, sm.max(zcape2 - zcapdcycl + zdqcv, RMINCAPE * zcape, _W_CAPE),', 'max'):
        ('L32', 1, 'inject', None, 0.0, 0.0001, ()),
    ('zmfmx = sm.min((kget(paph, jk) - kget(paph, jk - 1)) * zcons2, RMFLIA, _W_MF)', 'min'):
        ('STRESS', 0, 'inject', 'q', 1.225, 0.0001, ()),
    ('zmfub1 = sm.max((zcape_c * zmfub) / (zheat_c * zxtau_c), ZMFUB1_MIN, _W_MF)   # ZFACCA = 1', 'max'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('ztaupbl = sm.min(kget(pgeoh, ikb) - pgeoh[K + 1], ZDCYC_DZ, _W_GEO) / (rg * zduten)', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zxtau_c = sm.max(sm.min(zxtau, ZXTAU_MAX, _W_TIME), ZXTAU_MIN, _W_TIME)', 'max'):
        ('L32', 2, 'input', 'sflux', 3.5652173913043477, 1.2682412999266035e-05, ()),
    ('* sm.ge(rtt, kget(pten, jk), _W_T)', 'ge'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('* sm.lt(kget(zqenh, jkk) / kget(pqsen, jkk), ZRH_ADV, _W_RH))', 'lt'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('* sm.min(kget(pqsen, jk) / kget(pqsen, s["kcbot"]), 1.0, _W_ONE) ** 3)', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zqex = sm.max(-ZWS_FAC * pqhfl_sfc / (zrho_s * zws_ok), ZQEXC0, _W_VAP)', 'max'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zqex = sm.min(zqex, ZQEX_MAX, _W_VAP)', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zqexc = jnp.where(km1, sm.min(sm.max(st["zqex"], zqexc, _W_VAP), ZQEX_MAX, _W_VAP),', 'max'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zqexc = jnp.where(km1, sm.min(sm.max(st["zqex"], zqexc, _W_VAP), ZQEX_MAX, _W_VAP),', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('ztex = sm.max(-ZWS_FAC * pahfs_sfc / (zrho_s * zws_ok * rcpd), ZTEXC0, _W_TEX)', 'max'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('ztex = sm.min(ztex, ZTEX_MAX, _W_TEX)', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('ztexc = jnp.where(km1, sm.min(sm.max(st["ztex"], ztexc, _W_TEX), ZTEX_MAX, _W_TEX),', 'max'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('ztexc = jnp.where(km1, sm.min(sm.max(st["ztex"], ztexc, _W_TEX), ZTEX_MAX, _W_TEX),', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zwu = sm.min(jnp.sqrt(2.0 * sm.max(kget(s["pkineu"], jk + 1), ZKE_PRC_MIN, _W_KE)),', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zxe = sm.pos(zchange - zxs, _W_MF)', 'pos'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zzzmb = sm.min(sm.max(-kget(pvervel, kk) * zrg, ZMID_MF_MIN, _W_MF), RMFLIA, _W_MF)', 'max'):
        ('L32', 12, 'inject', None, 0.0, 0.0001, ()),
    ('zzzmb = sm.min(sm.max(-kget(pvervel, kk) * zrg, ZMID_MF_MIN, _W_MF), RMFLIA, _W_MF)', 'min'):
        ('L32', 12, 'inject', None, 0.0, 0.0001, ()),
    ('de_n = sm.max(mfu1 * (1.0 - zmfun), zdmfde, _W_MF)', 'max'):
        ('L32', 1, 'inject', None, 0.0, 0.0001, ()),
    ('ent = sm.sel(sm.on(zdmfen, 0.0, _W_MF), zdmfen, zdmfde)', 'on'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('mix_d = ZMIX_DEEP * entrorg * zdz * sm.min(', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('mix_s = sm.min(0.5 * zdz * zeps, 1.0, _W_ONE)', 'min'):
        ('L32', 15, 'inject', None, 0.0, 0.0001, ()),
    ('neg = sm.lt(zbuoc, 0.0, _W_BUOC)', 'lt'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('pqenh = jnp.where(adj, sm.pos(sm.min(qup, qsup, _W_VAP) + (pqsenh - qsup), _W_VAP), pqenh)', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('ptenh = (sm.max(rcpd * pten + pgeo, rcpd * up + gup, _W_SE) - pgeoh[:K + 1]) * zorcpd', 'max'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('reset, safe_sqrt_nan(sm.pos(kget(st["zwu2h"], st["icbot"]), _W_KE)),', 'pos'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('rh = sm.min(kget(pqen, jk) / kget(pqsen, jk), 1.0, _W_ONE)', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('rhm1 = sm.min(kget(pqen, jk - 1) / kget(pqsen, jk - 1), 1.0, _W_ONE)', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zadv = sm.min(sm.pos(kget(ptenqa, jkk) * ZQADV_TAU, _W_VAP), ZQADV_MAX, _W_VAP)', 'min'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('zadv = sm.min(sm.pos(kget(ptenqa, jkk) * ZQADV_TAU, _W_VAP), ZQADV_MAX, _W_VAP)', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('zdken = sm.min((1.0 + Z_CWDRAG) * ent / mfden, 1.0, _W_ONE)', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zoe = sm.min(zoe, ZOENTR_MAX, _W_OENTR) * mfu_ic', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
}


# Sites no full-scheme column reaches (no fixture column, input
# perturbation or single-site injection gives a resolvable derivative through
# them): a direct gradient test of the smoothed expression inside its window
# at the default width (coordinator ruling 2026-11-15), with the reason.
# Values: (bound b, or a representative value where the bound is traced, and
# the width, as expressions in the site module's namespace plus RG = g and
# RCPD = c_pd; reason; optional module: "cuascent" or "thermo", else
# _ifs_cumastrn).
_SITE_SYNTHETIC = {
    ('br = 1.0 - ((1.0 - sm.le(zsatfr, ZSATFR_MAX, _W_SATFR))', 'le'):
        ('ZSATFR_MAX', '_W_SATFR', 'full-scheme point found, but check_grads through the whole scheme does not agree with autodiff to 1e-3 relative there'),
    ('m2 = sm.min(m2, zmfmax, _W_MF)', 'min'):
        ('0.3', '_W_MF', 'full-scheme point found, but check_grads through the whole scheme does not agree with autodiff to 1e-3 relative there'),
    ('zeps = (sm.pos(-pmfd_b, w_mfub)', 'pos'):
        ('0.0', '_W_REL * 1.0e-6', 'full-scheme point found, but check_grads through the whole scheme does not agree with autodiff to 1e-3 relative there'),
    ('m3 = sm.max(ZMID_MF * zkhfl / kget(pgeoh, ikb), zmfub, w_mfub) * (1.0 + zeps)', 'max'):
        ('1.0e-6', '_W_REL * 1.0e-6', 'full-scheme point found, but check_grads through the whole scheme does not agree with autodiff to 1e-3 relative there'),
    ('* sm.lt(zdh, ZSHAL_DH * rg * rcpd, rg * _W_DH))', 'lt'):
        ('ZSHAL_DH * RG * RCPD', 'RG * _W_DH', "shallow 'weak' branch: needs a shallow flux above 0.3 x the CFL limit with a cloud-base energy excess below half its scale at once; no fixture column has both"),
    ('m3 = sm.sel(sm.lt(kget(pgeoh, ikb), ZMID_GEO, _W_GEO),', 'lt'):
        ('ZMID_GEO', '_W_GEO', "mid-level cap choice by cloud-base geopotential: the fixture's one mid-level cloud has its base far above 1500 m2/s2 and its flux far below either cap"),
    ('sm.min(m3, zmfmax / (ZMID_CFL * RMFCFL), _W_MF),', 'min'):
        ('0.1', '_W_MF', "mid-level flux cap (low base): binds only for a mid-level cloud near the CFL limit; the fixture's one mid-level flux is ~1e-6"),
    ('sm.min(m3, zmfmax, _W_MF))', 'min'):
        ('0.3', '_W_MF', "mid-level flux cap: binds only for a mid-level cloud near the CFL limit; the fixture's one mid-level flux is ~1e-6"),
    ('zmfub1 = jnp.where(deep, sm.min(zmfub1, zmfmax, _W_MF), zmfub)', 'min'):
        ('0.3', '_W_MF', 'deep closure CFL cap: where it binds the final limiter re-caps the flux level by level, masking its derivative'),
    ('zheat_c = sm.max(zheat, ZHEAT_MIN, _W_HEAT)', 'max'):
        ('ZHEAT_MIN', '_W_HEAT', 'ZHEAT_MIN floor of the deep-closure heating: convecting columns heat orders of magnitude above it'),
    ('zmfub_sh = jnp.where(sh_ok, sm.min(sdiv(zdhpbl, zdh, sh_ok), zmfmax, _W_MF),', 'min'):
        ("0.3", "_W_MF", "first-guess shallow flux: the final closure rescales the profile "
                       "by ZMFUB1/ZMFUB, so the outputs are invariant to it to first order"),
    ('zheat = jnp.sum(jnp.where(llo3, sm.pos(', 'pos'):
        ("0.0", "_W_HEAT", "per-level heating summand: its window (4.7e-10) sits ~5 orders "
                         "below the ZHEAT_MIN floor of the summed heating"),
    ('zdh = rg * sm.max(zdh, ZDH_DQ * zdqmin, _W_DH)', 'max'):
        ("0.1", "_W_DH", "first-guess energy floor: feeds only the first-guess shallow "
                       "flux, which cancels in the closure rescale"),
    ('/ ((ZTAU_W0 + sm.min(pwmean, ZTAU_WMAX, _W_VEL)) * rg) * ztaures * RTAUA)', 'min'):
        ('ZTAU_WMAX', '_W_VEL',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)'),
    ('zcapdcycl = sm.max(zcapdcycl, -2.0 * zcape2, _W_CAPE)', 'max'):
        ('-2.0 * _S_CAPE', '_W_CAPE',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist) (bound traced: representative value)'),
    ('zdh = rg * sm.max(zdh, ZDH_MIN * rcpd, _W_DH)', 'max'):
        ('ZDH_MIN * RCPD', '_W_DH',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)'),
    ('zxtau_c = sm.max(sm.min(zxtau, ZXTAU_MAX, _W_TIME), ZXTAU_MIN, _W_TIME)', 'min'):
        ('ZXTAU_MAX', '_W_TIME',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)'),
    ('zwu = sm.min(jnp.sqrt(2.0 * sm.max(kget(s["pkineu"], jk + 1), ZKE_PRC_MIN, _W_KE)),', 'max'):
        ('ZKE_PRC_MIN', '_W_KE',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('zwu = sm.min(jnp.sqrt(2.0 * sm.max(kget(s["pkineu"], jk), ZKE_RAIN_MIN, _W_KE)),', 'max'):
        ('ZKE_RAIN_MIN', '_W_KE',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('zwu = sm.min(jnp.sqrt(2.0 * sm.max(kget(s["pkineu"], jk), ZKE_RAIN_MIN, _W_KE)),', 'min'):
        ('ZWU_MAX', '_W_VEL',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('lrain = kget(s["plrain"], jk + 1) * sm.pos(mfu1 - zdmfde, _W_MF) * zfac', 'pos'):
        ('0.0', '_W_MF',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('newmfub = sm.min(s["pmfub"], zmfmax, _W_MF)', 'min'):
        ('0.3', '_W_MF',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist) (bound traced: representative value)', 'cuascent'),
    ('org = sm.gt(zbuo, ZBUO_ORG, _W_BUO)', 'gt'):
        ('ZBUO_ORG', '_W_BUO',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('pqenh = jnp.where(adj, sm.pos(sm.min(qup, qsup, _W_VAP) + (pqsenh - qsup), _W_VAP), pqenh)', 'pos'):
        ('0.0', '_W_VAP',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('pw = sm.max(st["pwmean"] / jnp.maximum(ZDPMEAN_MIN, st["zdpmean"]), ZWMEAN_MIN, _W_VEL)', 'max'):
        ('ZWMEAN_MIN', '_W_VEL',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('return jnp.maximum(x, 0.0) if sm is None else sm.pos(x, _W_CONDCU)', 'pos'):
        ('0.0', '_W_CONDCU',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'thermo'),
    ('return zcond1 * sm.on(zcond, 0.0, _W_CONDCU)', 'on'):
        ('0.0', '_W_CONDCU',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'thermo'),
    ('wpr = sm.on(luk, zdnoprc, _W_COND)', 'on'):
        ('ZDNOPRC', '_W_COND',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('wre = sm.lt(zbuo, ZBUO_REINT, _W_BUO)', 'lt'):
        ('ZBUO_REINT', '_W_BUO',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('wrf = sm.on(lr, 0.0, _W_COND)', 'on'):
        ('0.0', '_W_COND',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('zchange = sm.pos(zmftest - zmfmax, _W_MF)', 'pos'):
        ('0.0', '_W_MF',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('zdmfde_f = sm.min(zdmfde, ZDETR_MAX * mfu1, _W_MF)', 'min'):
        ('ZDETR_MAX * 0.03', '_W_MF',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist) (bound traced: representative value)', 'cuascent'),
    ('zdq = sm.pos(zqold - qu, _W_COND)', 'pos'):
        ('0.0', '_W_COND',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('zdt = sm.min(sm.pos(th.RTBERCU - tuk, _W_T), th.RTBERCU - th.RTICECU, _W_T)', 'min'):
        ('th.RTBERCU - th.RTICECU', '_W_T',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('zdt = sm.min(sm.pos(th.RTBERCU - tuk, _W_T), th.RTBERCU - th.RTICECU, _W_T)', 'pos'):
        ('0.0', '_W_T',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('zkedke = sm.pos(sm.min(zkedke, 1.0, _W_ONE), _W_ONE)', 'min'):
        ('1.0', '_W_ONE',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('zkedke = sm.pos(sm.min(zkedke, 1.0, _W_ONE), _W_ONE)', 'pos'):
        ('0.0', '_W_ONE',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative, or the result differs between runs under xdist)', 'cuascent'),
    ('* sm.pos(c1, _W_CONDCU), sm)', 'pos'):
        ('0.0', '_W_CONDCU',
         'no fixture column: injecting only this site mid-window changes no scheme output when it alone is made exact (all 13 convecting columns tried)', 'thermo'),
    ('+ sm.pos(zbuof * zdz, _W_CAPEL), on)', 'pos'):
        ('0.0', '_W_CAPEL',
         'no fixture column: injecting only this site mid-window changes no scheme output when it alone is made exact (all 13 convecting columns tried)', 'cuascent'),
    ('1.0 - kt3 * sm.gt(kget(pten, s["kcbot"]), rtt - ZLSCV_DT, _W_T))', 'gt'):
        ('th.RTT - ZLSCV_DT', '_W_T',
         'no fixture column: injecting only this site mid-window changes no scheme output when it alone is made exact (all 13 convecting columns tried)', 'cuascent'),
    ('ke = sm.max((kget(s["pkineu"], jk + 1) * (1.0 - zdken) + zdkbuo) / (1.0 + zdken),', 'max'):
        ('ZKE_MIN', '_W_KE',
         'no fixture column: injecting only this site mid-window changes no scheme output when it alone is made exact (all 13 convecting columns tried)', 'cuascent'),
    ('lambda c1, c: c1 - (1.0 - sm.on(-c, 0.0, _W_CONDCU))', 'on'):
        ('0.0', '_W_CONDCU',
         'no fixture column: injecting only this site mid-window changes no scheme output when it alone is made exact (all 13 convecting columns tried)', 'thermo'),
    ('mix_d = sm.min(mix_d, 1.0, _W_ONE)', 'min'):
        ('1.0', '_W_ONE',
         'no fixture column: injecting only this site mid-window changes no scheme output when it alone is made exact (all 13 convecting columns tried)', 'cuascent'),
    ('return jnp.minimum(QSAT_CAP, x) if sm is None else sm.min(x, QSAT_CAP, _W_EPCU)', 'min'):
        ('QSAT_CAP', '_W_EPCU',
         'no fixture column: injecting only this site mid-window changes no scheme output when it alone is made exact (all 13 convecting columns tried)', 'thermo'),
    ('th, psp, t, q, lambda c: -sm.pos(-c, _W_CONDCU),', 'pos'):
        ('0.0', '_W_CONDCU',
         'no fixture column: injecting only this site mid-window changes no scheme output when it alone is made exact (all 13 convecting columns tried)', 'thermo'),
    ('wub = safe_sqrt_nan(sm.pos(kget(st["zwu2h"], jkb), _W_KE))', 'pos'):
        ('0.0', '_W_KE',
         'no fixture column: injecting only this site mid-window changes no scheme output when it alone is made exact (all 13 convecting columns tried)', 'cuascent'),
    ('zoentr_b = sm.min(zoentr_b, ZOENTR_MAX, _W_OENTR) * mfu1', 'min'):
        ('ZOENTR_MAX', '_W_OENTR',
         'no fixture column: injecting only this site mid-window changes no scheme output when it alone is made exact (all 13 convecting columns tried)', 'cuascent'),
    ('zxs = sm.pos(mfu1 - zmfmax, _W_MF)', 'pos'):
        ('0.0', '_W_MF',
         'no fixture column: injecting only this site mid-window changes no scheme output when it alone is made exact (all 13 convecting columns tried)', 'cuascent'),
    ('tc = th.RTWAT - sm.pos(th.RTWAT - t, _W_TCU)    # <= RTWAT: fraction <= 1', 'pos'):
        ('0.0', '_W_TCU',
         'successor (review round 1: caps bounded from below, fraction <= 1) of a site whose full-scheme injection point failed the per-site test (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative)', 'thermo'),
    ('zleen = sm.sel(sm.on(lit, RLMIN, _W_COND), lit * zdmfen, 0.0)', 'on'):
        ('RLMIN', '_W_COND',
         'successor (review round 1: caps bounded from below, fraction <= 1) of a site whose full-scheme injection point failed the per-site test (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative)', 'cuascent'),
    ('zlnew = Z_CLDMAX - sm.pos(Z_CLDMAX - zlnew, _W_COND)', 'pos'):
        ('0.0', '_W_COND',
         'successor (review round 1: caps bounded from below, fraction <= 1) of a site whose full-scheme injection point failed the per-site test (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative)', 'cuascent'),
    ('zlnew = luk - sm.pos(luk - zlnew, _W_COND)', 'pos'):
        ('0.0', '_W_COND',
         'successor (review round 1: caps bounded from below, fraction <= 1) of a site whose full-scheme injection point failed the per-site test (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative)', 'cuascent'),
    ('zlnew = sm.pos(zlnew, _W_COND)', 'pos'):
        ('0.0', '_W_COND',
         'successor (review round 1: caps bounded from below, fraction <= 1) of a site whose full-scheme injection point failed the per-site test (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative)', 'cuascent'),
    ('zrnew = lr - sm.pos(lr - zrnew, _W_COND)   # <= lr', 'pos'):
        ('0.0', '_W_COND',
         'successor (review round 1: caps bounded from below, fraction <= 1) of a site whose full-scheme injection point failed the per-site test (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative)', 'cuascent'),
    ('zrnew = sm.pos(zrnew, _W_COND)', 'pos'):
        ('0.0', '_W_COND',
         'successor (review round 1: caps bounded from below, fraction <= 1) of a site whose full-scheme injection point failed the per-site test (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative)', 'cuascent'),
}


@pytest.mark.parametrize("site", sorted(_SITE_SYNTHETIC), ids=lambda t: f"{t[1]}:{t[0][:60]}")
def test_switch_site_synthetic_gradient(site):
    """RULE 5 gradient test of an unreached site's smoothed expression, alone,
    at mid-window and the default width: check_grads passes and the
    derivative differs from the exact switch's."""
    from jax.test_util import check_grads
    from legoesm.atmosphere.physics._ifs_smooth import IfsSmooth
    from legoesm.atmosphere.physics import _ifs_thermo
    from legoesm.atmosphere.physics.convection import _ifs_cuascent
    from legoesm.atmosphere.physics.convection import _ifs_cumastrn as M

    sp = _probe("switch_sites")
    method = site[1]
    bexpr, wexpr, reason, *mod = _SITE_SYNTHETIC[site]
    assert reason
    mod = {"cuascent": _ifs_cuascent, "thermo": _ifs_thermo}[mod[0]] if mod else M
    th = lego_thermo()
    ns = {**vars(mod), "RG": th.RG, "RCPD": th.RCPD, "th": th}
    b, w = float(eval(bexpr, ns)), float(eval(wexpr, ns))

    def f(a, hard=False):
        sm = IfsSmooth(hard)
        args = (a, w) if method in ("pos", "neg1") else (a, jnp.asarray(b), w)
        return getattr(sm, method)(*args)

    a0 = jnp.asarray(float(sp._mid(method, b, w)))
    gs = float(jax.grad(f)(a0))
    gh = float(jax.grad(lambda a: f(a, True))(a0))
    check_grads(f, (a0,), order=1, modes=("rev",), eps=1e-3 * w, atol=1e-3 * abs(gs), rtol=1e-3)
    assert np.isfinite(gs) and abs(gs - gh) > 1e-3 * max(abs(gs), abs(gh))


_SITE_PENDING: set = set()   # none left


def test_every_switch_site_has_a_gradient_point():
    """A new or edited smooth switch in CUMASTRN must get its own point."""
    sets = (set(_SITE_POINTS), _SITE_PENDING, set(_SITE_SYNTHETIC))
    assert _probe("switch_sites").sites() == set.union(*sets)
    assert sum(map(len, sets)) == len(set.union(*sets))


@pytest.fixture(autouse=True)
def _fresh_jit():
    """Every test starts and ends with empty JAX caches: traces are cached by
    function identity, blind to the IfsSmooth methods and widths the probes
    patch, so a trace made under instrumentation (or before it) would be
    reused across tests; and XLA's CPU JIT runs out of code memory otherwise."""
    jax.clear_caches()
    yield
    jax.clear_caches()


@pytest.mark.parametrize("site", sorted(_SITE_POINTS), ids=lambda t: f"{t[1]}:{t[0][:60]}")
def test_switch_site_gradient_at_default(fix, site):
    """RULE 5 gradient test, per switch site, at the default (smooth): at the
    site's point it is inside its window, check_grads (order 1, reverse)
    passes on the scheme's heating + drying + mass flux with respect to the
    test variable, and the derivative changes when only this site is exact."""
    from jax.test_util import check_grads

    sp = _probe("switch_sites")
    g, c, mode, direction, value, h, partners = _SITE_POINTS[site]
    if mode == "input":
        s0 = value
        x_of = lambda s: sp.point_value(fix, g, c, direction, s)  # noqa: E731
        inj_of = lambda s: {}  # noqa: E731
    else:
        s0 = 1.0
        base = sp.column(fix, g, c) if direction is None else sp.point_value(
            fix, g, c, direction, value)
        x_of = lambda s: base  # noqa: E731
        inj_of = lambda s: {**{p: jnp.asarray(1.0) for p in partners}, site: s}  # noqa: E731

    def run(s, hard=()):
        with sp.instrument(hard_sites=hard, inject=inj_of(s)):
            return sp.output(fix, x_of(s))

    x0 = jnp.asarray(s0)
    pos = {}
    with sp.instrument(lambda k, d: pos.setdefault(k, []).append(d.ravel()), inject=inj_of(x0)):
        sp.output(fix, x_of(x0))
        jax.effects_barrier()
    assert np.any(sp.in_window(site, np.concatenate(pos[site])))
    gs = float(jax.grad(run)(x0))
    gh = float(jax.grad(lambda s: run(s, hard=[site]))(x0))
    # tolerances relative to this site's derivative (the default atol 1e-5
    # would pass a derivative of 1e-6 that is wrong by 90 %)
    check_grads(run, (x0,), order=1, modes=("rev",), eps=h, atol=1e-3 * abs(gs), rtol=1e-3)
    # the derivative changes by more than round-off: 1e-3 relative, and above a
    # floor 1e-12 (the loss is O(100); a site that only reaches the output at
    # round-off level gets a synthetic test, see _SITE_SYNTHETIC)
    big = max(abs(gs), abs(gh))
    assert np.isfinite(gs) and big > 1e-12 and abs(gs - gh) > 1e-3 * big


def test_ascent_smooth_cloud_to_rain_conserves_water():
    """At the smooth default the rain gains exactly what the cloud loses, and
    0 <= new cloud <= cloud, inside and around both cap windows."""
    from legoesm.atmosphere.physics._ifs_smooth import IfsSmooth
    from legoesm.atmosphere.physics.convection import _ifs_cuascent as A

    sm, w = IfsSmooth(False), A._W_COND
    for luk in (5.0e-4, A.Z_CLDMAX, A.Z_CLDMAX + 0.5 * w, 3.0 * A.Z_CLDMAX):
        z0 = jnp.linspace(-3.0 * w, 3.0 * w, 61)
        for zl in (luk + z0, A.Z_CLDMAX + z0, z0):
            for wpr in (0.0, 0.3, 1.0):
                zlnew, zprecip = A.cloud_to_rain(sm, jnp.asarray(luk), zl, wpr)
                cloud = wpr * zlnew + (1.0 - wpr) * luk
                np.testing.assert_allclose(cloud + zprecip, luk, rtol=1e-14, atol=0.0)
                assert np.all(np.asarray(zlnew) >= 0.0) and np.all(np.asarray(zlnew) <= luk)


def test_smooth_liquid_fraction_never_exceeds_one():
    """FOEALFCU's smooth upper clip sits on the cold side of RTWAT: the
    fraction stays <= 1 and is exact outside the window."""
    from legoesm.atmosphere.physics import _ifs_thermo as T
    from legoesm.atmosphere.physics._ifs_smooth import IfsSmooth

    th = lego_thermo()
    t = th.RTWAT + jnp.linspace(-3.0, 3.0, 601) * T._W_TCU
    a = np.asarray(T.foealfcu(th, t, IfsSmooth(False)))
    exact = np.asarray(T.foealfcu(th, t))
    assert np.all(a <= 1.0)
    out = np.asarray(t < th.RTWAT - T._W_TCU) | np.asarray(t >= th.RTWAT)
    np.testing.assert_allclose(a[out], exact[out], rtol=1e-12)
    assert np.any(a[~out] != exact[~out])


def test_ascent_mass_flux_width_is_the_closure_one():
    """The ascent reuses the closure's mass-flux scale (vote 2026-11-14)."""
    from legoesm.atmosphere.physics.convection import _ifs_cuascent, _ifs_cumastrn

    assert _ifs_cuascent._S_MF == _ifs_cumastrn._S_MF


def test_switch_scales_probe_covers_every_class(fix):
    """scripts/validate/oracle/bechtold_ifs/switch_scales.py (the probe the
    _S_* scales came from) samples every width class on the fixture and
    restores the widths and IfsSmooth it patched."""
    import importlib.util

    from legoesm.atmosphere.physics import _ifs_smooth
    from legoesm.atmosphere.physics.convection import _ifs_cumastrn as M

    path = (Path(__file__).resolve().parents[2] / "scripts" / "validate" / "oracle"
            / "bechtold_ifs" / "switch_scales.py")
    spec = importlib.util.spec_from_file_location("switch_scales", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    before = (M._W_MF, _ifs_smooth.IfsSmooth.max)
    scales = probe.p2(probe.measure(fix, ("L32",)))
    assert (M._W_MF, _ifs_smooth.IfsSmooth.max) == before
    assert set(scales) == set(probe._classes().values())
    assert all(np.isfinite(s) and s > 0.0 and n > 0 for s, n in scales.values())
