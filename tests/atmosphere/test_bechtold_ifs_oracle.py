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
    from legoesm.atmosphere.physics.convection._ifs_faithful import _check_config
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig

    base = BechtoldConfig(use_ifs_ascent=True, dx_m=_DX)
    _check_config(base)
    f = next(k for k, v in base._asdict().items()
             if isinstance(v, float) and k not in ("dx_m",))
    with pytest.raises(ValueError, match=f):
        _check_config(base._replace(**{f: getattr(base, f) + 1.0}))
    with pytest.raises(ValueError, match="dx_m"):
        _check_config(base._replace(dx_m=0.0))
