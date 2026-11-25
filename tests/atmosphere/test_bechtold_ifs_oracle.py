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
    # #2281) another; the corrected cloud-top humidity fixer (#2594, owner
    # 2026-11-24) a third; everything else is sucumf's.
    assert tun == IfsTunables()._replace(shallow_nosupply="zero", hard_switches=False,
                                         openifs_cloud_top_fixer=False)


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
    # drying floor 1e-9 kg m-2 s-1 (vote 2026-11-18): a barely convecting
    # group's fp64 round-off (5e-20) is not a relative error of its drying
    assert np.max(np.abs(res)) / max(np.max(np.abs(dry)), _DRY_FLOOR) <= _TOL


_DRY_FLOOR = 1.0e-9   # kg m-2 s-1


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


def test_wrapper_publishes_the_surface_snow_flux(fix, monkeypatch):
    """F29: the frozen surface precipitation is CUMASTRN's surface snow flux,
    within its surface rain + snow flux (exact; the column sum of the rain
    source equals that flux to round-off, checked by
    test_wrapper_surface_rain_is_cumastrn_surface_flux)."""
    from legoesm.atmosphere.physics.convection import _ifs_faithful as F
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    cap, orig = {}, F.cumastrn
    def spy(*a, **k):
        cap["o"] = orig(*a, **k)
        return cap["o"]
    monkeypatch.setattr(F, "cumastrn", spy)
    out, _ = _wrapper_out(fix, BechtoldConfig(use_ifs_ascent=True, dx_m=_DX))
    snow = np.asarray(out.precip_frozen_sfc)
    rain = np.asarray(cap["o"].pmflxr)[:, -1]
    np.testing.assert_array_equal(snow, np.asarray(cap["o"].pmflxs)[:, -1])
    # snow <= rain + snow  <=>  rain >= 0
    assert np.all(snow >= 0.0) and np.all(rain >= 0.0) and np.any(snow > 0.0)


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
# Re-measured for #2281 part 3 (downdraught, CUFLXN and post-fixer switches
# smoothed): L32 column 12 heating gap 0.070 -> 0.97, drying 0.013 -> 0.18 -- its whole
# precipitation flux (1.3e-9 kg m-2 s-1) lies inside the flux window
# (0.01 x 2.27e-7), so the smooth positive part suppresses it; KNOWN
# DIFFERENCE, kept (vote 2 Q6, 2026-11-19).  Other changes <= 6.6e-6.
_SWITCH_GAP = {
    "L32": (
        [1.155408049270849e-13, 3.1927949260996204e-13, 0.0011130719528878582, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.4927806518186066e-06, 4.19866852170346e-05, 0.0, 0.9734300882432095, 4.049624669431736e-12, 1.7179434496712484e-12, 3.1143115208075958e-12],
        [1.7172552867238978e-14, 1.163127531864268e-13, 0.0010985278742337056, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.911942492452407e-06, 4.689802829971999e-05, 0.0, 0.18080538279776642, 4.1134943022347197e-13, 5.056799592526267e-14, 1.101205278761102e-13],
        [1.4365594076618328e-14, 1.340269644731608e-13, 0.0011884180672240485, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.470103297401741e-15, 5.127344626193498e-05, 0.0, 0.0, 4.239592100981241e-13, 4.719359133175054e-14, 6.594308439320736e-14],
        [3.975688633527109e-15, 1.3605024188564655e-13, 0.0010549443125622503, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.621867336070376e-06, 5.127344625960713e-05, 0.0, 0.0, 4.2082974011963847e-13, 3.8886484681088704e-14, 0.0],
    ),
    "L137": (
        [1.661612930475599e-11, 0.0, 0.0, 0.0, 0.0],
        [1.2046143070100942e-13, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ),
    "STRESS": (
        [1.722849855972394e-13, 0.012694584019058777, 8.875598874530435e-13],
        [4.41178551951255e-14, 0.014005735274384364, 1.990163333564079e-13],
        [1.2783764286336544e-13, 0.015307609436948059, 2.0573300977747443e-13],
        [9.120569143950305e-16, 0.015307609436952462, 1.5953942788457935e-13],
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
# with only this site's switched operand replaced by mid-window + (s - 1) w
# (value and derivative a function of s only; vote 3 2026-11-19), s = 1, and
# each ``partners``
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
    ('return jnp.minimum(1.0, x) if sm.hard else 1.0 - sm.pos(1.0 - x, _W_ONE)', 'pos'):
        ('L32', 15, 'inject', None, 0.0, 0.0001, ()),
    ('pqenh = jnp.where(adj, sm.pos(sm.min(qup, qsup, _W_VAP) + (pqsenh - qsup), _W_VAP), pqenh)', 'pos'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('pw = sm.max(st["pwmean"] / jnp.maximum(ZDPMEAN_MIN, st["zdpmean"]), ZWMEAN_MIN, _W_VEL)', 'max'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zcapdcycl = sm.max(zcapdcycl, -2.0 * zcape2, _W_CAPE)', 'max'):
        ('L32', 1, 'inject', None, 0.0, 0.0001, ()),
    ('zdh = rg * sm.max(zdh, ZDH_MIN * rcpd, _W_DH)', 'max'):
        ('L32', 15, 'inject', None, 0.0, 0.0001, ()),
    ('zdt = sm.min(sm.pos(th.RTBERCU - tuk, _W_T), th.RTBERCU - th.RTICECU, _W_T)', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zkedke = sm.pos(sm.min(zkedke, 1.0, _W_ONE), _W_ONE)', 'min'):
        ('L32', 1, 'inject', None, 0.0, 0.0001, ()),
    ('zkedke = sm.pos(sm.min(zkedke, 1.0, _W_ONE), _W_ONE)', 'pos'):
        ('L32', 1, 'inject', None, 0.0, 0.0001, ()),
    ('zwu = sm.min(jnp.sqrt(2.0 * sm.max(kget(s["pkineu"], jk), ZKE_RAIN_MIN, _W_KE)),', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zxtau_c = sm.max(sm.min(zxtau, ZXTAU_MAX, _W_TIME), ZXTAU_MIN, _W_TIME)', 'min'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    # part 3 (CUDDRAFN, CUFLXN, post-fixers): injection for every site (vote
    # 2026-11-19, 3 of 3)
    ('bp = sm.pos(b, w)', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('return bp - sm.pos(b - a, w)', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('ddq = sm.lt(kget(zmfdq, idtop), ZDDTOP_Q * kget(zmfdq, ikd), _W_DDQF)', 'lt'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('en = -sm.pos(-en, _W_DDEN)     # bounded: en <= 0 (entrainment only)', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.01, ()),
    ('en = sm.max(en, ZDD_MIX_MIN * mfd1, _W_DDEN)', 'max'):
        ('L32', 10, 'inject', None, 0.0, 0.01, ()),
    ('frz = sm.le(tk, rtt, _W_TFRZ)', 'le'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('melt = sm.on(sk, 0.0, _W_PFLX)', 'on'):
        ('STRESS', 0, 'inject', None, 0.0, 0.0001, ()),
    ('pmfdde_rate + sm.pos(-zer, _W_RATE), pmfdde_rate)', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.01, ()),
    ('pr = sm.pos(r1, _W_PFLX)', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('ps = sm.pos(s1, _W_PFLX)', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('return r2, s2, t_new - (r1 + s1), dpmel * sm.on(r1 + s1, 0.0, _W_PFLX, strict=False)', 'on'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('sn1 = jnp.where(dsn, sm.pos(sn1, _W_PFLX), 0.0)', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('sn2 = jnp.where(dsn, sm.pos(sn2, _W_PFLX), 0.0)', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('t_new = sm.pos(r1 + s1, _W_PFLX)', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('zbuoyz = -sm.pos(-zbuoyz, _W_DDBUO)   # bounded: <= 0', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('zdrfl = -sm.pos(zrfl - zrfln, _W_PFLX)    # bounded: <= 0 (evaporation only)', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('zdrfl1 = rcpecons * sm.pos(qsk - qk, _W_QDEF) * zrcucov * spow(arg, ZEVAP_EXP) * dp', 'pos'):
        ('STRESS', 0, 'inject', None, 0.0, 0.01, ()),
    ('zneed = sm.pos(-(pqen + zmfa) / zdzt, _W_LUDE)', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('zrfln = sm.pos(zrnew, _W_PFLX)', 'pos'):
        ('L32', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zrmin = zrfl - zrcucov * sm.pos(zrhebc * qsk - qk, _W_QDEF) * zcons2 * dp', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('zrnew = sm.max(zrnew, zrmin, _W_PFLX)', 'max'):
        ('STRESS', 0, 'inject', None, 0.0, 0.0001, ()),
    ('zsnmlt = zfac * sm.pos(zten - rtt, _W_TFRZ)', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.0001, ()),
    ('ztwetb = tk - sm.pos(qsk - qk, _W_QDEF) * (ZTW1 + ZTW2 * (pk - ZTW3) - ZTW4 * (tk - ZTW5))', 'pos'):
        ('L32', 10, 'inject', None, 0.0, 0.01, ()),
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
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture)'),
    ('zwu = sm.min(jnp.sqrt(2.0 * sm.max(kget(s["pkineu"], jk + 1), ZKE_PRC_MIN, _W_KE)),', 'max'):
        ('ZKE_PRC_MIN', '_W_KE',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture)', 'cuascent'),
    ('zwu = sm.min(jnp.sqrt(2.0 * sm.max(kget(s["pkineu"], jk), ZKE_RAIN_MIN, _W_KE)),', 'max'):
        ('ZKE_RAIN_MIN', '_W_KE',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture)', 'cuascent'),
    ('lrain = kget(s["plrain"], jk + 1) * sm.pos(mfu1 - zdmfde, _W_MF) * zfac', 'pos'):
        ('0.0', '_W_MF',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture)', 'cuascent'),
    ('newmfub = sm.min(s["pmfub"], zmfmax, _W_MF)', 'min'):
        ('0.3', '_W_MF',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture) (bound traced: representative value)', 'cuascent'),
    ('org = sm.gt(zbuo, ZBUO_ORG, _W_BUO)', 'gt'):
        ('ZBUO_ORG', '_W_BUO',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture)', 'cuascent'),
    ('return jnp.maximum(x, 0.0) if sm is None else sm.pos(x, _W_CONDCU)', 'pos'):
        ('0.0', '_W_CONDCU',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture)', 'thermo'),
    ('return zcond1 * sm.on(zcond, 0.0, _W_CONDCU)', 'on'):
        ('0.0', '_W_CONDCU',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture)', 'thermo'),
    ('wpr = sm.on(luk, zdnoprc, _W_COND)', 'on'):
        ('ZDNOPRC', '_W_COND',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture)', 'cuascent'),
    ('wre = sm.lt(zbuo, ZBUO_REINT, _W_BUO)', 'lt'):
        ('ZBUO_REINT', '_W_BUO',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture)', 'cuascent'),
    ('wrf = sm.on(lr, 0.0, _W_COND)', 'on'):
        ('0.0', '_W_COND',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture)', 'cuascent'),
    ('zchange = sm.pos(zmftest - zmfmax, _W_MF)', 'pos'):
        ('0.0', '_W_MF',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture)', 'cuascent'),
    ('zdmfde_f = sm.min(zdmfde, ZDETR_MAX * mfu1, _W_MF)', 'min'):
        ('ZDETR_MAX * 0.03', '_W_MF',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture) (bound traced: representative value)', 'cuascent'),
    ('zdq = sm.pos(zqold - qu, _W_COND)', 'pos'):
        ('0.0', '_W_COND',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture)', 'cuascent'),
    ('zdt = sm.min(sm.pos(th.RTBERCU - tuk, _W_T), th.RTBERCU - th.RTICECU, _W_T)', 'pos'):
        ('0.0', '_W_T',
         'full-scheme injection point found, but the per-site test does not hold there (check_grads through the whole scheme disagrees with autodiff to 1e-3 relative; re-checked 2026-11-18 with the cache-clearing fixture)', 'cuascent'),
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
    ('melt = melt * sm.on(zten, rtt, _W_TFRZ)', 'on'):
        ('th.RTT', '_W_TFRZ',
         'no fixture column: injecting only this site mid-window changes no scheme output when it alone is made exact (all 13 convecting columns tried)'),
    ('sq = (idbas >= K - 1) * sm.gt(-kget(pmfd, idbas), 0.5 * kget(mu, idbas), _W_SQ)', 'gt'):
        ('_S_SQ', '_W_SQ',
         'no fixture column: injecting only this site mid-window changes no scheme output when it alone is made exact (all 13 convecting columns tried) (bound traced: representative value)'),
    ('RCUCOV * (1.0 + (sm.max(zrhm, ZRHM_MIN, _W_RHM) - ZRHM_MIN) / ZRHM_SCALE),', 'max'):
        ('ZRHM_MIN', '_W_RHM',
         'full-scheme injection point found, but it reaches the scheme output only at round-off level (|d loss/ds| <= 4e-12, finite difference 0; vote 3 2026-11-19)'),
    ('pmfude_rate = jnp.where(cud, pmfude_rate + sm.pos(-zer, _W_RATE), pmfude_rate)', 'pos'):
        ('0.0', '_W_RATE',
         'full-scheme injection point found, but its derivative (|d loss/ds| ~ 1e-6..1e-8 on a loss ~ 200) is resolved by finite differences only to 1.1-1.4e-3 at h = 1e-2 (round-off, error ~ 1/h), above the 1e-3 tolerance (vote 3 2026-11-19)'),
    ('zalfaw = sm.sel(sm.lt(kget(pten, jk), rtt, _W_TFRZ), 0.0, foealfcu(th, kget(pten, jk), sm))', 'lt'):
        ('th.RTT', '_W_TFRZ',
         'full-scheme injection point found, but it reaches the scheme output only at round-off level (|d loss/ds| <= 4e-12, finite difference 0; vote 3 2026-11-19)'),
    ('en = sm.max(en, -ZDD_DET_MAX * kget(pmfu, jk) - (mfd1 - zdmfde), _W_DDEN)', 'max'):
        ('0.0', '_W_DDEN',
         'full-scheme injection point found, but its derivative (|d loss/ds| ~ 2e-6..2e-7 on a loss ~ 200) is resolved by finite differences only to >= 9e-4..1.5e-3 at h = 1e-2..3e-2 (round-off, error ~ 1/h), not robustly inside the 1e-3 tolerance (vote 3 2026-11-19) (bound traced: representative value)'),
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
                zlnew, zprecip, _ = A.cloud_to_rain(
                    sm, True, jnp.asarray(luk), 0.0, jnp.asarray(luk), zl, wpr, A.ZDNOPRC)
                cloud = wpr * zlnew + (1.0 - wpr) * luk
                np.testing.assert_allclose(cloud + zprecip, luk, rtol=1e-14, atol=0.0)
                assert np.all(np.asarray(zlnew) >= 0.0) and np.all(np.asarray(zlnew) <= luk)


def test_exact_cloud_to_rain_makes_no_rain_below_the_onset():
    """Exact form: no precipitation where the cloud water is at or below the
    onset ZDNOPRC (OpenIFS sets ZPRECIP = 0 there), so existing rain is not
    fed from a level that does not precipitate."""
    from legoesm.atmosphere.physics._ifs_smooth import IfsSmooth
    from legoesm.atmosphere.physics.convection import _ifs_cuascent as A

    luk = jnp.asarray([0.5, 1.0, 2.0]) * A.ZDNOPRC
    zlnew, zprecip, pr = A.cloud_to_rain(IfsSmooth(True), True, luk, 0.2 * luk,
                                         0.8 * luk, 0.3 * luk, 1.0, A.ZDNOPRC)
    np.testing.assert_array_equal(np.asarray(pr), [False, False, True])
    np.testing.assert_array_equal(np.asarray(zprecip)[:2], 0.0)
    assert float(zprecip[2]) > 0.0


def test_smooth_mixing_fraction_keeps_the_parcel_blend_convex():
    """The test parcel's entrainment mixing fraction stays <= 1 inside the
    smooth window, so vapour and static energy of the blended parcel stay
    between the parcel's and the environment's (no negative vapour)."""
    from legoesm.atmosphere.physics._ifs_smooth import IfsSmooth
    from legoesm.atmosphere.physics.convection import _ifs_cuascent as A

    x = 1.0 + jnp.linspace(-3.0, 3.0, 601) * A._W_ONE
    m = A.mixing_fraction(IfsSmooth(False), x)
    assert np.all(np.asarray(m) <= 1.0) and np.all(np.asarray(m) >= 0.0)
    q_parcel, q_env = 1.0e-2, 1.0e-6
    q = q_parcel * (1.0 - m) + q_env * m
    assert np.all(np.asarray(q) >= q_env)
    np.testing.assert_array_equal(np.asarray(A.mixing_fraction(IfsSmooth(True), x)),
                                  np.minimum(1.0, np.asarray(x)))


@pytest.mark.parametrize("site", [
    ("zlnew = luk - sm.pos(luk - zlnew, _W_COND)", "pos"),
    ("zlnew = Z_CLDMAX - sm.pos(Z_CLDMAX - zlnew, _W_COND)", "pos"),
    ("zrnew = lr - sm.pos(lr - zrnew, _W_COND)   # <= lr", "pos"),
    ("return jnp.minimum(1.0, x) if sm.hard else 1.0 - sm.pos(1.0 - x, _W_ONE)", "pos"),
    # part 3: melt cap, detrained-precipitation caps, flux positivity fixer,
    # evaporation bound, downdraught entrainment bound, cloud-top fixer
    ("bp = sm.pos(b, w)", "pos"),   # _cap_min: melt, detrained precip, cloud-top fixer
    ("return bp - sm.pos(b - a, w)", "pos"),
    ("t_new = sm.pos(r1 + s1, _W_PFLX)", "pos"),
    ("pr = sm.pos(r1, _W_PFLX)", "pos"),
    ("zrfln = sm.pos(zrnew, _W_PFLX)", "pos"),
    ("zdrfl = -sm.pos(zrfl - zrfln, _W_PFLX)    # bounded: <= 0 (evaporation only)", "pos"),
    ("en = -sm.pos(-en, _W_DDEN)     # bounded: en <= 0 (entrainment only)", "pos"),
])
def test_host_water_closes_with_a_bounded_cap_active(fix, site):
    """RULE 0 with the smooth caps inside their windows: every level's capped
    operand is moved to mid-window (L32, smooth default) and the water the
    host receives still sums to zero over each column (fp64, _TOL)."""
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    sp = _probe("switch_sites")
    with sp.instrument(inject={site: jnp.asarray(1.0)}):
        out, _ = _wrapper_out(fix, BechtoldConfig(use_ifs_ascent=True, dx_m=_DX), "L32")
    dp = np.diff(fix["L32_p_half"], axis=1)
    col = lambda a: np.sum(np.asarray(a) * dp, axis=1) / _thermo(fix, "L").RG  # noqa: E731
    dry = col(out.dq_v_dt)
    res = (dry + col(out.dq_c_conv_dt) + col(out.dq_r_conv_dt)
           + col(out.dq_r_detr_dt) + col(out.dq_s_detr_dt))
    assert np.max(np.abs(dry)) > 0.0
    assert np.max(np.abs(res)) / np.max(np.abs(dry)) <= _TOL


def _ref_nonneg(r1, s1, dpmel):
    """cuflxn.F90:427-438 / :489-500, the three branches: (rain, snow, amount
    added to the total, melt)."""
    if r1 + s1 < 0.0:
        return 0.0, 0.0, -(r1 + s1), 0.0
    if r1 < 0.0:
        return 0.0, s1 + r1, 0.0, dpmel
    if s1 < 0.0:
        return r1 + s1, 0.0, 0.0, dpmel
    return r1, s1, 0.0, dpmel


def test_precip_positivity_fixer_conserves_and_has_the_openifs_limit():
    """RULE 0 x RULE 5 for CUFLXN's negative-flux fixer: the smooth form's
    rain + snow is the input total plus the returned deficit (which the caller
    adds to the level's precipitation source, as OpenIFS) to round-off, both
    are >= 0, and with the exact switches it is OpenIFS's three branches."""
    from legoesm.atmosphere.physics._ifs_smooth import IfsSmooth
    from legoesm.atmosphere.physics.convection import _ifs_cumastrn as M

    w = M._W_PFLX
    pts = [(a * w, b * w) for a in (-3.0, -1.0, -0.4, 0.0, 0.3, 0.7, 2.0, 50.0)
           for b in (-3.0, -0.6, 0.0, 0.2, 0.9, 5.0)]
    for r1, s1 in pts:
        for hard in (True, False):
            r2, s2, deficit, mel = (float(x) for x in M._nonneg_precip(
                IfsSmooth(hard), jnp.asarray(r1), jnp.asarray(s1), 0.25 * w))
            # deficit >= 0 at the hard limit; the C1 positive part sits up
            # to 4w/27 below T inside its window
            assert r2 >= 0.0 and s2 >= 0.0 and deficit >= (0.0 if hard else -4.0 * w / 27.0)
            assert abs(r2 + s2 - (r1 + s1) - deficit) <= 1e-12 * w
            if hard:
                np.testing.assert_allclose((r2, s2, deficit, mel), _ref_nonneg(r1, s1, 0.25 * w),
                                           rtol=1e-12, atol=1e-12 * w)
    # outside every window the smooth form is the exact one
    np.testing.assert_allclose(
        [float(x) for x in M._nonneg_precip(IfsSmooth(False), jnp.asarray(50.0 * w),
                                            jnp.asarray(5.0 * w), 0.25 * w)],
        _ref_nonneg(50.0 * w, 5.0 * w, 0.25 * w), rtol=1e-12)


def test_precip_positivity_fixer_gradient_at_default():
    """check_grads through the fixer inside its windows; the derivative
    differs from the exact switch's."""
    from jax.test_util import check_grads
    from legoesm.atmosphere.physics._ifs_smooth import IfsSmooth
    from legoesm.atmosphere.physics.convection import _ifs_cumastrn as M

    w = M._W_PFLX

    def f(r1, s1, hard=False):   # in units of w
        r2, s2, d, mel = M._nonneg_precip(IfsSmooth(hard), r1 * w, s1 * w, 0.25 * w)
        return (r2 + 2.0 * s2 + 3.0 * d + 5.0 * mel) / w

    for r1, s1 in ((0.4, 0.2), (0.5, -0.2), (-0.1, 0.6)):
        x = (jnp.asarray(r1), jnp.asarray(s1))
        check_grads(f, x, order=1, modes=("rev",), eps=1e-5, atol=1e-6, rtol=1e-6)
        g = np.asarray(jax.grad(f, argnums=(0, 1))(*x))
        gh = np.asarray(jax.grad(lambda a, b: f(a, b, True), argnums=(0, 1))(*x))
        assert np.all(np.isfinite(g)) and np.max(np.abs(g - gh)) > 1e-3 * np.max(np.abs(gh))


def test_cap_min_is_zero_with_nothing_to_cap():
    """_cap_min (smooth bounded min used by the melt cap, the detrained-
    precipitation caps and the cloud-top fixer): exactly 0 where a = 0 or
    b <= 0, across b's window (the former form b - pos(b - a) gave up to
    4w/27 there: snow melted below freezing); 0 <= f <= max(b, 0),
    f <= a + 4w/27; the hard limit is max(min(a, b), 0)."""
    from legoesm.atmosphere.physics._ifs_smooth import IfsSmooth
    from legoesm.atmosphere.physics.convection import _ifs_cumastrn as M

    w = M._W_PFLX
    sm = IfsSmooth(False)
    a = jnp.linspace(0.0, 3.0, 61)[:, None] * w
    b = jnp.linspace(-1.0, 3.0, 81)[None, :] * w
    f = np.asarray(M._cap_min(sm, a, b, w))
    np.testing.assert_array_equal(f[0], 0.0)                 # a = 0
    np.testing.assert_array_equal(f[:, np.asarray(b[0]) <= 0.0], 0.0)
    assert np.all(f >= 0.0) and np.all(f <= np.maximum(np.asarray(b), 0.0))
    assert np.all(f <= np.asarray(a) + 4.0 * w / 27.0 * (1.0 + 1e-12))
    # hard limit: pos(b) - pos(b - a) = b - (b - a) rounds (codex round 2)
    np.testing.assert_allclose(np.asarray(M._cap_min(IfsSmooth(True), a, b, w)),
                               np.maximum(np.minimum(np.asarray(a), np.asarray(b)), 0.0),
                               rtol=1e-12, atol=1e-12 * w)


def test_cap_min_gradient_at_default():
    """check_grads through _cap_min inside both windows; the derivative
    differs from the exact min's."""
    from jax.test_util import check_grads
    from legoesm.atmosphere.physics._ifs_smooth import IfsSmooth
    from legoesm.atmosphere.physics.convection import _ifs_cumastrn as M

    w = M._W_PFLX

    def f(a, b, hard=False):   # in units of w
        return M._cap_min(IfsSmooth(hard), a * w, b * w, w) / w

    for a, b in ((2.0, 0.5), (0.4, 0.9), (1.2, 1.5)):
        x = (jnp.asarray(a), jnp.asarray(b))
        check_grads(f, x, order=1, modes=("rev",), eps=1e-5, atol=1e-6, rtol=1e-6)
        g = np.asarray(jax.grad(f, argnums=(0, 1))(*x))
        gh = np.asarray(jax.grad(lambda p, q: f(p, q, True), argnums=(0, 1))(*x))
        assert np.all(np.isfinite(g)) and np.max(np.abs(g - gh)) > 1e-3 * np.max(np.abs(gh))


def test_smooth_melt_cap_is_zero_below_freezing_full_scheme(fix, monkeypatch):
    """Full scheme (L32, smooth default) with the snow flux the melt cap sees
    moved into its window (b = w/2, injected: no fixture column puts it there
    below freezing): every melt call below freezing (zsnmlt = 0) melts
    nothing, and some do sit below freezing."""
    import linecache
    import sys
    from legoesm.atmosphere.physics.convection import _ifs_cumastrn as M
    from legoesm.atmosphere.physics.convection._ifs_cumastrn import IfsTunables

    orig, rec = M._cap_min, []

    def spy(sm, a, b, w):
        fr = sys._getframe(1)
        if linecache.getline(fr.f_code.co_filename, fr.f_lineno).strip().startswith("dpmel ="):
            b = jnp.full_like(b, 0.5 * w)
            f = orig(sm, a, b, w)
            jax.debug.callback(lambda a, f: rec.append((np.asarray(a), np.asarray(f))), a, f)
            return f
        return orig(sm, a, b, w)

    monkeypatch.setattr(M, "_cap_min", spy)
    th, x = _thermo(fix, "L"), _inputs(fix, "L32", jnp.float64)
    cumastrn(th, float(fix["dt"]), qsat=satur_qsat(th, x["T"], x["p_full"]),
             tun=IfsTunables(hard_switches=False), **x)
    jax.effects_barrier()
    a = np.concatenate([np.ravel(r[0]) for r in rec])
    f = np.concatenate([np.ravel(r[1]) for r in rec])
    assert np.sum(a == 0.0) > 0
    np.testing.assert_array_equal(f[a == 0.0], 0.0)


def test_smooth_melt_is_zero_below_freezing(fix):
    """The smooth melt cap must not melt snow at T < RTT, where OpenIFS's
    melting factor ZFAC changes sign (T < RTT - 1/ZMELT_DT): surface snow of
    the fixture's frozen-precipitation columns is the oracle's."""
    from legoesm.atmosphere.physics.convection._ifs_cumastrn import IfsTunables

    th, x = _thermo(fix, "L"), _inputs(fix, "L32", jnp.float64)
    out = cumastrn(th, float(fix["dt"]), qsat=satur_qsat(th, x["T"], x["p_full"]),
                   tun=IfsTunables(hard_switches=False), **x)
    ref = np.asarray(fix["L_L32_PMFLXS"])[:, -1]
    assert np.any(ref > 0.0)
    np.testing.assert_allclose(np.asarray(out.pmflxs)[:, -1], ref, rtol=1e-3, atol=1e-12)


def test_ascent_cap_helpers_gradient_at_default():
    """check_grads through the two cap helpers themselves (not only the
    generic smooth forms), at points inside their windows."""
    from jax.test_util import check_grads
    from legoesm.atmosphere.physics._ifs_smooth import IfsSmooth
    from legoesm.atmosphere.physics.convection import _ifs_cuascent as A

    sm, wc = IfsSmooth(False), A._W_COND
    for luk, z0 in ((1.0e-3, 1.0e-3 - 0.5 * wc), (2.0 * A.Z_CLDMAX, A.Z_CLDMAX - 0.5 * wc)):
        luk = jnp.asarray(luk)
        cloud = lambda z, luk=luk: A.cloud_to_rain(sm, True, luk, 0.0, luk, z, 0.7,  # noqa: E731
                                                   A.ZDNOPRC)[0]
        check_grads(cloud, (jnp.asarray(z0),), order=1, modes=("rev",), eps=1e-3 * wc)
        g = float(jax.grad(cloud)(jnp.asarray(z0)))
        g_hard = float(jax.grad(lambda z, luk=luk: A.cloud_to_rain(
            IfsSmooth(True), True, luk, 0.0, luk, z, 0.7, A.ZDNOPRC)[0])(jnp.asarray(z0)))
        assert g > 0.0 and abs(g - g_hard) > 1e-3 * max(abs(g), abs(g_hard))
    x0 = jnp.asarray(1.0 - 0.5 * A._W_ONE)
    check_grads(lambda x: A.mixing_fraction(sm, x), (x0,), order=1, modes=("rev",),
                eps=1e-3 * A._W_ONE)
    g = float(jax.grad(lambda x: A.mixing_fraction(sm, x))(x0))
    g_hard = float(jax.grad(lambda x: A.mixing_fraction(IfsSmooth(True), x))(x0))
    assert g > 0.0 and abs(g - g_hard) > 1e-3 * max(abs(g), abs(g_hard))


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


def _mse_residual(fix, g, dT, dq, snow_sfc, ice_det, snow_det, diss):
    """Per column [W m-2]: integral of (cp dT/dt + Lv dq/dt) dp/g minus Lf x
    (surface snow + detrained ice + detrained snow) minus the convective
    kinetic-energy dissipation heating PVDISCU (LMFUVDIS) -- frozen moist
    static energy, liquid reference (cudtdqn.F90: detrained liquid/rain at
    RLVTT, ice/snow at RLSTT; surface rain/snow at RLVTT/RLSTT, :299).
    Zero if the scheme closes it; with LMFENTHCONS off (sucumf.F90:208) the
    in-cloud heating uses the mixed-phase FOELHMCU, so mixed-phase columns
    need not close."""
    th = _thermo(fix, "L")
    dp = np.diff(fix[f"{g}_p_half"], axis=1)
    col = lambda a: np.sum(np.asarray(a) * dp, axis=1) / th.RG  # noqa: E731
    return (th.RCPD * col(np.asarray(dT) - fix[f"{g}_dT_other"])
            + th.RLVTT * col(np.asarray(dq) - fix[f"{g}_dq_other"])
            - th.RLMLT * (np.asarray(snow_sfc) + np.asarray(ice_det) + np.asarray(snow_det))
            - np.asarray(diss))


def mse_residuals(fix, g):
    """(OpenIFS arm L, exact port, smooth default) column MSE residuals and
    the largest column heating [W m-2] (the metric's scale)."""
    from legoesm.atmosphere.physics.convection._ifs_cumastrn import IfsTunables

    o = lambda t: np.asarray(fix[f"L_{g}_{t}"])  # noqa: E731
    ref = _mse_residual(fix, g, o("PTENT"), o("PTENQ"), o("PMFLXS")[:, -1],
                        o("PLUDELI2").sum(1), o("PSNDE2").sum(1), o("PVDISCU"))
    th, x = _thermo(fix, "L"), _inputs(fix, g, jnp.float64)
    res = []
    for hard in (True, False):
        out = cumastrn(th, float(fix["dt"]), qsat=satur_qsat(th, x["T"], x["p_full"]),
                       tun=IfsTunables(hard_switches=hard), **x)
        res.append(_mse_residual(fix, g, out.ptent, out.ptenq, np.asarray(out.pmflxs)[:, -1],
                                 np.asarray(out.pludeli)[:, 1].sum(1),
                                 np.asarray(out.psnde)[:, 1].sum(1), out.pvdiscu))
    dp = np.diff(fix[f"{g}_p_half"], axis=1)
    heat = th.RCPD * np.sum((o("PTENT") - fix[f"{g}_dT_other"]) * dp, axis=1) / th.RG
    return ref, res[0], res[1], float(np.max(np.abs(heat)))


# Smooth default minus exact column frozen-MSE residual [W m-2] per fixture
# column (RULE 0 x RULE 5: what the smoothing adds to the column energy
# budget), measured (x64, CPU) by mse_residuals().  KNOWN DEFECT of the
# reference (claim reviewed by codex and GLM, 2026-11-18): with LMFENTHCONS
# off (sucumf.F90:208) the column does not close -- up to -228 W m-2 (1 % of
# the column heating) on the deep fixture columns; it is the source's own
# enthalpy-correction numerator (cudtdqn.F90:290-299).  The exact port
# reproduces it (first assertion).
_MSE_SMOOTH_GAP = {
    "L32": [-5.43309397471603e-10, -9.727429972627988e-10, -0.027704053871804746, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0902657382748657e-10, -0.011967617050174795, 0.0, -1.305190177128008e-10, -3.7278269360285776e-10, -6.591278634004993e-10, 1.7271485619062517e-10],
    "L137": [1.2356137814306667e-11, 0.0, 0.0, 0.0, 0.0],
    "STRESS": [1.0560086138866609e-09, 3.4830013889212523, 3.725020292222325e-10],
}


_MSE_HEAT_FLOOR = 1.0   # W m-2 (vote 2026-11-18: L137 heats < 1e-3 W m-2)


@pytest.mark.parametrize("g", _GROUPS)
def test_column_moist_static_energy_smooth_vs_exact(fix, g):
    """The exact port leaves the reference's residual (max |difference| /
    max(the group's largest column heating, 1 W m-2) <= _TOL) and the smooth
    default adds the stored per-column amount."""
    ref, hard, smooth, heat = mse_residuals(fix, g)
    if heat == 0.0:
        np.testing.assert_array_equal(hard, ref)
    else:
        assert np.max(np.abs(hard - ref)) / max(heat, _MSE_HEAT_FLOOR) <= _TOL
    np.testing.assert_allclose(smooth - hard, np.asarray(_MSE_SMOOTH_GAP[g]),
                               rtol=1e-6, atol=1e-12)


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


# --- #2594: cloud-top negative-humidity fixer (cumastrn.F90:1159-1180).  It
# fires on no fixture column at the fixture step; STRESS column 0 at ten
# times the step makes it fire in both switch modes (measured 2026-11-25:
# the only firing case of step factors 1/3/10 x four upper-humidity
# reductions x the 24 fixture columns).
_FIRE_G, _FIRE_COL, _FIRE_DT = "STRESS", 0, 10.0
_CUM_JIT = jax.jit(cumastrn, static_argnums=(0, 1), static_argnames=("tun",))


def _fire_run(fix, hard, openifs, q_scale=1.0):
    from legoesm.atmosphere.physics.convection._ifs_cumastrn import IfsTunables

    th = _thermo(fix, "L")
    x = {k: v[_FIRE_COL:_FIRE_COL + 1] for k, v in _inputs(fix, _FIRE_G).items()}
    x["q"] = x["q"] * q_scale
    return _CUM_JIT(th, _FIRE_DT * float(fix["dt"]), qsat=satur_qsat(th, x["T"], x["p_full"]),
                    tun=IfsTunables(hard_switches=hard, openifs_cloud_top_fixer=openifs), **x)


def _cloudsc_water(fix, out):
    """Column water as CLOUDSC receives it [kg m-2 s-1]: vapour tendency +
    detrained liquid and ice (PLUDELI1/2, its species sources) + detrained
    rain/snow + surface rain/snow; and the column drying (the scale)."""
    dp = np.diff(fix[f"{_FIRE_G}_p_half"][_FIRE_COL:_FIRE_COL + 1], axis=1)
    dq = np.asarray(out.ptenq) - fix[f"{_FIRE_G}_dq_other"][_FIRE_COL:_FIRE_COL + 1]
    col = np.sum(dq * dp, axis=1) / _thermo(fix, "L").RG
    li = np.asarray(out.pludeli)[:, 0] + np.asarray(out.pludeli)[:, 1]
    res = (col + np.sum(li + np.asarray(out.psnde).sum(axis=1), axis=1)
           + np.asarray(out.pmflxr)[:, -1] + np.asarray(out.pmflxs)[:, -1])
    return float(res[0]), float(col[0])


@pytest.mark.parametrize("hard", [True, False])
def test_cloud_top_fixer_takes_the_deficit_from_the_condensate(fix, hard):
    """#2594 (owner decision 2026-11-24): the default fixer REMOVES the
    humidity deficit from the detrained condensate (OpenIFS adds it), keeps
    PLUDELI1 + PLUDELI2 = PLUDE >= 0, and leaves more vapour than OpenIFS at
    the fired levels."""
    cor, ref = _fire_run(fix, hard, False), _fire_run(fix, hard, True)
    pc, pr = np.asarray(cor.plude)[0], np.asarray(ref.plude)[0]
    fired = pc != pr
    assert np.any(fired)
    assert np.all(pc[fired] < pr[fired]) and np.all(pc >= 0.0)
    li = np.asarray(cor.pludeli)[0, 0] + np.asarray(cor.pludeli)[0, 1]
    np.testing.assert_allclose(li, pc, rtol=1e-12, atol=1e-12 * np.max(pr))
    lir = np.asarray(ref.pludeli)[0, 0] + np.asarray(ref.pludeli)[0, 1]
    assert np.max(np.abs(lir - pr)) > 1e-3 * np.max(np.abs(pr - pc))
    assert np.all(np.asarray(cor.ptenq)[0][fired] > np.asarray(ref.ptenq)[0][fired])


@pytest.mark.parametrize("hard", [True, False])
def test_cloud_top_fixer_water_closes_only_when_corrected(fix, hard):
    """RULE 0, #2594: with the fixer firing, the water CLOUDSC receives
    (vapour + PLUDELI liquid/ice + detrained and surface precipitation)
    closes over the column to the oracle tolerance (fp64) in the default;
    OpenIFS's form, which rescales PLUDELI by the updated PLUDE, does not."""
    res_c, dry_c = _cloudsc_water(fix, _fire_run(fix, hard, False))
    res_r, dry_r = _cloudsc_water(fix, _fire_run(fix, hard, True))
    print("hard", hard, "water residual default / OpenIFS [kg m-2 s-1]", res_c, res_r)
    assert abs(res_c) / abs(dry_c) <= _TOL
    assert abs(res_r) / abs(dry_r) > 1.0e3 * _TOL


@pytest.mark.parametrize("hard", [True, False])
def test_cloud_top_fixer_energy_difference_is_the_openifs_mismatch(fix, hard):
    """RULE 0, #2594: the column frozen-MSE residual of OpenIFS's fixer
    exceeds the default's by exactly Lv x its PLUDELI-PLUDE mismatch (the
    default's mismatch is 0), so the corrected fixer adds nothing to the
    column energy budget.  (This column's absolute residual is #2571's.)"""
    th = _thermo(fix, "L")
    sl = slice(_FIRE_COL, _FIRE_COL + 1)
    sub = {f"{_FIRE_G}_{k}": fix[f"{_FIRE_G}_{k}"][sl] for k in ("p_half", "dT_other", "dq_other")}
    r, mism = {}, {}
    for openifs in (False, True):
        o = _fire_run(fix, hard, openifs)
        r[openifs] = _mse_residual({**fix, **sub}, _FIRE_G, o.ptent, o.ptenq,
                                   np.asarray(o.pmflxs)[:, -1], np.asarray(o.pludeli)[:, 1].sum(1),
                                   np.asarray(o.psnde)[:, 1].sum(1), o.pvdiscu)[0]
        mism[openifs] = float(np.sum(np.asarray(o.pludeli)[0, 0] + np.asarray(o.pludeli)[0, 1]
                                     - np.asarray(o.plude)[0]))
    print("hard", hard, "MSE residual default / OpenIFS [W m-2]", r[False], r[True])
    # The fixer moves only PLUDE (vapour sink, Lv x -dPLUDE via the humidity
    # tendency) and PLUDELI (heating RLVTT dLI1 + RLSTT dLI2, minus Lf dLI2
    # in the budget; RLSTT - RLMLT = RLVTT), so the residual changes by
    # RLVTT x d(LI1 + LI2 - PLUDE): the mismatch difference of the two forms.
    assert th.RLSTT - th.RLMLT == pytest.approx(th.RLVTT, rel=1e-15)
    assert mism[True] != 0.0
    assert abs(mism[False]) <= 1e-12 * abs(mism[True])
    np.testing.assert_allclose(r[True] - r[False], th.RLVTT * (mism[True] - mism[False]),
                               rtol=1e-6)


def test_cloud_top_fixer_gradient_at_default(fix):
    """RULE 5 (d): reverse-mode derivative through the smooth corrected fixer
    on the firing column agrees with finite differences (check_grads)."""
    from jax.test_util import check_grads

    def f(s):
        o = _fire_run(fix, False, False, q_scale=s)
        return jnp.sum(o.plude) + jnp.sum(o.ptenq)

    one = jnp.asarray(1.0)
    g = float(jax.grad(f)(one))
    assert np.isfinite(g) and g != 0.0
    check_grads(f, (one,), order=1, modes=("rev",), eps=1e-6, atol=1e-3 * abs(g), rtol=1e-3)


def test_cloud_top_fixer_switch_is_validated():
    from legoesm.atmosphere.physics.convection._ifs_cumastrn import IfsTunables

    with pytest.raises(ValueError, match="openifs_cloud_top_fixer"):
        cumastrn(lego_thermo(), 1.0, *([None] * 22), tun=IfsTunables(openifs_cloud_top_fixer=1))
