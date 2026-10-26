"""User decision 2026-10-22, 2a, as amended 2026-10-24, 2b: in CAM6 mode (MG2
snow: warm_rain_incloud and a prognostic snow number) the ported CAM ice
processes run with the droplet-number switch OFF too: MG2 ice
deposition/sublimation and Bergeron, Ferrier ice->snow, Hallett-Mossop
splinters, instant ice melt and CAM homogeneous freezing.  A configured ice
choice those processes would override is a start-up error (2b supersedes 2a's
silent replacement), as with the switch on.  The droplet number is not evolved
there.  x64.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics import morrison as M  # noqa: E402
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState  # noqa: E402
from legoesm.thermo import saturation_mixing_ratio_ice  # noqa: E402

from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import _production_morrison  # noqa: E402


def _cfg(**kw):
    return _production_morrison()._replace(warm_rain_scheme="kk2000_cam6", **kw)


def _cells(T_list, rhi=1.1):
    T = jnp.asarray(T_list)[:, None]
    p = jnp.full_like(T, 50000.0)
    ph = jnp.concatenate([p - 500.0, p + 500.0], axis=1)
    rho = p / (C.R_d * T)
    dz = (ph[:, 1:] - ph[:, :-1]) / (C.g * rho)
    qv = rhi * saturation_mixing_ratio_ice(T, p)
    z = jnp.zeros_like(T)
    hyd = HydrometeorState(q_c=z + 1e-4, q_r=z, q_i=z + 5e-5, q_s=z + 1e-4, q_g=z,
                           N_c=z + 5.0e7, N_r=z, N_i=z + 1.0e5, N_s=z + 1.0e4)
    return T, qv, hyd, p, ph, rho, dz, z


def _run(cfg, T_list, rhi=1.1):
    T, qv, hyd, p, ph, rho, dz, z = _cells(T_list, rhi)
    return M.morrison_microphysics(T, qv, hyd, p, ph, rho, dz, 600.0, cfg,
                                   cloud_fraction=z + 1.0)


def _spy(mp, name):
    calls = []
    real = getattr(M, name)

    def spy(*a, **k):
        calls.append((a, k))
        return real(*a, **k)

    mp.setattr(M, name, spy)
    return calls


def test_production_resolves_non_cam_ice_choices():
    c = M.resolve_morrison_flavor(_cfg())
    assert c.warm_rain_incloud and not c.prognostic_droplet_number
    assert (c.ice_to_snow_scheme, c.ice_deposition_scheme, c.wbf_scheme) == (
        "mg_ferrier", "m2005", "emergent")


def test_cam_ice_processes_replace_the_configured_ones(monkeypatch):
    dep = _spy(monkeypatch, "mg2_ice_deposition_sublimation")
    fer = _spy(monkeypatch, "ice_autoconversion_mg2")
    spl = _spy(monkeypatch, "mg2_secondary_ice_production")
    melt = _spy(monkeypatch, "mg2_instant_ice_melt_homog_freeze")
    out = _run(_cfg(publish_qc_budget=True), [250.0, 268.0, 274.0])
    assert dep and fer and spl and melt
    qb = {k: np.asarray(v) for k, v in out.qc_budget.items()}
    assert (qb["bergeron"][:2] < 0).all()                  # MG2 cloud-ice Bergeron, not emergent 0
    assert "ice_melt" in qb and qb["ice_melt"][2, 0] > 0   # ice melts instantly above 0 C
    assert (np.asarray(out.dN_c_dt) == 0.0).all()           # droplet number not evolved


def test_no_cam_ice_without_snow_number(monkeypatch):
    """Outside CAM6 mode (no prognostic snow number) nothing changes."""
    dep = _spy(monkeypatch, "mg2_ice_deposition_sublimation")
    T, qv, hyd, p, ph, rho, dz, z = _cells([250.0])
    M.morrison_microphysics(T, qv, hyd._replace(N_s=None), p, ph, rho, dz, 600.0, _cfg(),
                            cloud_fraction=z + 1.0)
    assert not dep


def test_ice_melt_mass_closes_the_cloud_water_budget():
    out = _run(_cfg(publish_qc_budget=True), [274.0, 276.0])
    qb = {k: np.asarray(v) for k, v in out.qc_budget.items()}
    resid = np.asarray(out.dq_c_dt) - sum(qb.values())
    assert np.abs(resid).max() <= 1e-12 * np.abs(np.asarray(out.dq_c_dt)).max()


def test_gradient_through_cam_ice_without_switch():
    cfg = _cfg()

    def f(T0):
        T, qv, hyd, p, ph, rho, dz, z = _cells([250.0, 268.0])
        T = T + T0
        o = M.morrison_microphysics(T, qv, hyd, p, ph, rho, dz, 600.0, cfg,
                                    cloud_fraction=z + 1.0)
        return jnp.sum(o.dq_i_dt)

    g = float(jax.grad(f)(0.0))
    h = 1e-4
    fd = (float(f(h)) - float(f(-h))) / (2 * h)
    assert np.isfinite(g) and np.isclose(g, fd, rtol=1e-3, atol=1e-6 * abs(fd)), (g, fd)


@pytest.mark.parametrize("override", [
    # liquid_from_closure off: with it on (the deck's value) main's sam-flavour
    # satadj guard refuses first, so the CAM6 refusal below would be unreached
    dict(morrison_flavor="sam", ice_to_snow_scheme="m2005_autoconv",
         liquid_from_closure=False),
    dict(wbf_scheme="bergeron_heuristic"),
    dict(ice_deposition_scheme="heuristic"),
])
def test_non_cam_ice_choice_is_refused(override):
    """2b: no silent replacement; the run refuses to start."""
    with pytest.raises(ValueError, match="CAM6 MG2 mode"):
        _run(_cfg(**override), [250.0])


def test_sam_ice_choice_on_the_deck_config_is_refused():
    """The same sam-flavour choice on the deck's own config
    (liquid_from_closure on) is refused too, by the sam satadj guard."""
    cfg = _cfg(morrison_flavor="sam", ice_to_snow_scheme="m2005_autoconv")
    assert cfg.liquid_from_closure
    with pytest.raises(ValueError, match="satadj_liquid"):
        _run(cfg, [250.0])
