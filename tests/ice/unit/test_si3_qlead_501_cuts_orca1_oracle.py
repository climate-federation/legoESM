"""ORCA1 oracle: ice_flx_other's open-water freezing cuts follow NEMO 5.0.1.

NEMO 5.0.1 (icesbc.F90:391,412,416) zeroes qlead when the ice is slow
(``zvel <= 5 mm/s``) and packed (``at_i >= rn_amax - 0.01``), or when
``at_i >= 1 - epsi10``; no thickness term.  The port transcribed NEMO 5.0.2
(margin 0.001 AND ``vt_i >= 3 m``), so on 429 ORCA1 cells it kept lead freezing
NEMO 5.0.1 suppresses (tier-4 surface finding S4).  The 5.0.2 form stays
selectable (``lead_cut_version="5.0.2"``) for the C1D_OMIP_L3 5.0.2 gate.

Fixture ``fixtures/orca1_t4/si3_qlead_501_cuts.npz`` (+ ``.json``): from the
tier-4 NEMO 5.0.1 ORCA1 ice_flx_other record (kt 17281, 32 ranks): all 429
cells the 5.0.1 cut zeroes and 5.0.2 does not (``s4``), the 18 both cut
(``lf_both``), and seeded wet ice / open-water cells.  Tolerance rel 1e-12,
atol 1e-12 * max|field| (vote V11).
"""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

FIX = pathlib.Path(__file__).parent / "fixtures" / "orca1_t4" / "si3_qlead_501_cuts.npz"
TOL = 1e-12
OUTS = ("qsb_ice_bot", "fhld", "qlead")


@pytest.fixture(autouse=True)
def _fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


@pytest.fixture(scope="module")
def fx():
    z = np.load(FIX)
    return {k: z[k] for k in z.files}


def _flx(fx, version, width=0.0, **arrays):
    from legoesm.ice.sea_ice import _nemo_si3_ice_flx_other
    i = {k.split(":", 1)[1]: jnp.asarray(v) for k, v in fx.items() if k.startswith("flx_in:")}
    i.update(arrays)
    s = {k.split(":", 1)[1]: float(v) for k, v in fx.items() if k.startswith("flx_scalar:")}
    return _nemo_si3_ice_flx_other(
        ice_fraction=i["at_i"], ice_fraction_before=i["at_i_b"], ice_volume=i["vt_i"], u_ice=i["u_ice"],
        u_ice_west=i["u_ice_w"], v_ice=i["v_ice"], v_ice_south=i["v_ice_s"], u_ocean=i["ssu_m"],
        u_ocean_west=i["ssu_m_w"], v_ocean=i["ssv_m"], v_ocean_south=i["ssv_m_s"], drag_io=s["rn_Cd_io"],
        frq=i["frq_m"], qsr_ocean=i["qsr_oce"], qns_ocean=i["qns_oce"], qemp_ocean=i["qemp_oce"],
        ocean_layer_thickness=i["e3t_m"], sst_celsius=i["sst_m"], T_bottom=i["t_bo"], dt=s["rDt_ice"],
        rho_ocean=s["rho0"], c_ocean=s["rcp"], T0=s["rt0"], ice_epsilon=1.0e-10,
        max_ice_fraction=i["rn_amax_2d"], lead_cut_version=version, lead_cut_width=width, land_mask=i["smask0"])


def test_fixture_groups(fx):
    g = fx["flx_group"]
    assert (g == "s4").sum() == 429 and (g == "lf_both").sum() == 18
    assert np.all(fx["flx_in:smask0"] == 1.0)
    assert np.all(fx["flx_nemo:qlead"][g == "s4"] == 0.0)


def test_501_cuts_match_nemo_501(fx):
    out = _flx(fx, "5.0.1")
    for k in OUTS:
        p = np.asarray(out[k], np.float64); n = fx[f"flx_nemo:{k}"]
        np.testing.assert_allclose(p, n, rtol=TOL, atol=TOL * float(np.max(np.abs(n))), err_msg=k)
        assert not np.any(np.signbit(p) & (p == 0.0) & ~np.signbit(n)), k   # no -0.0 where NEMO has +0.0


def test_502_cuts_keep_lead_freezing_on_the_s4_cells(fx):
    """Control: the 5.0.2 form misses NEMO 5.0.1 on every s4 cell and agrees elsewhere."""
    g = fx["flx_group"]
    q = np.asarray(_flx(fx, "5.0.2")["qlead"])
    assert np.all(q[g == "s4"] < 0.0)
    np.testing.assert_allclose(q[g != "s4"], fx["flx_nemo:qlead"][g != "s4"], rtol=TOL,
                               atol=TOL * float(np.max(np.abs(fx["flx_nemo:qlead"]))))


def test_unknown_version_raises(fx):
    with pytest.raises(ValueError, match="lead_cut_version"):
        _flx(fx, "5.0.3")


def test_smooth_cut_converges_and_is_differentiable(fx):
    """Width 0.01 (default) is close to the hard cut away from the thresholds and has finite gradients,
    including for ice at rest (squared-speed switch, vote V13); jax.grad vs central FD (V12)."""
    g = fx["flx_group"]
    hard = np.asarray(_flx(fx, "5.0.1")["qlead"])
    soft = np.asarray(_flx(fx, "5.0.1", width=0.01)["qlead"])
    assert np.all(np.isfinite(soft))
    ow = g == "wet_open"   # open water: at_i far below the packed floor -> cut inactive
    np.testing.assert_allclose(soft[ow], hard[ow], rtol=1e-12, atol=1e-12 * float(np.max(np.abs(hard))))
    cells = np.flatnonzero(g == "s4")[:3]
    for c in cells:
        sub = {k: (v[c] if v.ndim else v) for k, v in fx.items() if k.startswith(("flx_in:", "flx_scalar:"))}
        a0 = float(sub["flx_in:at_i"])
        f = lambda a: _flx(sub, "5.0.1", width=0.01, at_i=a)["qlead"]  # noqa: E731
        gr = float(jax.grad(f)(a0)); h = 1e-6
        fd = (float(f(a0 + h)) - float(f(a0 - h))) / (2 * h)
        assert np.isfinite(gr) and abs(gr - fd) <= 1e-4 * max(abs(fd), 1e-6 * abs(float(f(a0)))), (c, gr, fd)
    rest = {k: (v[cells[0]] if v.ndim else v) for k, v in fx.items() if k.startswith(("flx_in:", "flx_scalar:"))}
    zero = jnp.asarray(0.0)
    grads = jax.grad(lambda u: _flx(rest, "5.0.1", width=0.01, u_ice=u, u_ice_w=zero, v_ice=zero,
                                    v_ice_s=zero)["qlead"])(zero)
    assert np.isfinite(float(grads))
