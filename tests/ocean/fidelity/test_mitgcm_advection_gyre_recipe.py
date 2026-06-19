"""Tests for the legoESM-MITgcm passive-tracer advection-in-gyre recipe.

Construction / forcing / dye-init checks always run; the oracle pattern-match
against real MITgcm ``PTRACER01`` dumps is opt-in via the run dir env var
``$LEGOESM_OCEAN_FIDELITY_MITGCM_ADVGYRE_RUN`` (e.g. ``/tmp/mitgcm_advgyre/run``),
because it needs a built MITgcm.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from legoesm.ocean.fidelity import mitgcm_advection_gyre_recipe as adv

_RUN_DIR = os.environ.get("LEGOESM_OCEAN_FIDELITY_MITGCM_ADVGYRE_RUN")


def test_wind_profile_matches_mitgcm_gendata():
    """tau_x(y) = -tauMax cos(2 pi Y), Y=(j-0.5)/(ny-1): single-gyre cos2y."""
    tx = adv.advgyre_taux_ocean_profile()
    assert tx.shape == (adv.NY,)
    # cos(2 pi Y) with Y in (0,1): -tauMax at the edges, +tauMax mid-domain.
    np.testing.assert_allclose(tx[0], -adv.TAU_MAX, atol=2e-3)
    np.testing.assert_allclose(tx[adv.NY // 2], adv.TAU_MAX, atol=3e-3)


def test_closed_box_land_mask():
    """gendata.m h(end,:)=0; h(:,end)=0 -> walls on the LAST row/col only."""
    m = np.asarray(adv.build_advgyre_land_mask())
    assert m.shape == (adv.NY, adv.NX)
    assert not m[-1, :].any() and not m[:, -1].any()        # walls
    assert m[:-1, :-1].all()                                 # rest ocean


def test_dye_blob_single_cell():
    """dye(2,30)=1 (1-based Fortran) -> 0-based (y=29, x=1), all else zero."""
    d = np.asarray(adv.build_dye_field())
    assert d.shape == (adv.NY, adv.NX, 1)
    assert d.sum() == adv.DYE_AMPLITUDE
    nz = list(zip(*np.where(d > 0)))
    assert nz == [(adv.DYE_Y, adv.DYE_X, 0)]


def test_geometry_uses_beta_plane_with_mitgcm_params():
    g = adv.build_advgyre_geometry()
    assert g.n_lat == adv.NY and g.n_lon == adv.NX
    np.testing.assert_allclose(np.asarray(g.dx_T), adv.DX_M)
    y_c0 = adv.Y_ORIGIN_M + 0.5 * adv.DY_M
    np.testing.assert_allclose(
        float(g.f_T[0, 0]), adv.F0 + adv.BETA * y_c0, rtol=1e-9
    )


def test_config_selects_passive_dst3_advection():
    """MITgcm advScheme=80 -> dst3_multidim; diffKh=0; passive (beta_S=0)."""
    c = adv.build_advgyre_config()
    assert c.tracer_advection == "dst3_multidim"
    assert c.K_h == 0.0                                      # PTRACERS_diffKh=0
    assert c.eos == "linear"
    assert c.eos_linear.alpha_T == 0.0 and c.eos_linear.beta_S == 0.0
    assert c.A_h == adv.VISC_AH


def test_recipe_carries_dye_in_salinity_slot():
    """The dye is carried as the (dynamically inert) salinity field."""
    rec = adv.build_advgyre_recipe()
    assert rec.dt_s == adv.DT_S and rec.n_steps == adv.N_STEPS
    s = np.asarray(rec.state.S.data)
    assert s.shape == (adv.NY, adv.NX, 1)
    assert s.sum() == adv.DYE_AMPLITUDE
    assert s[adv.DYE_Y, adv.DYE_X, 0] == adv.DYE_AMPLITUDE


def test_wind_forcing_shape_and_sign():
    """Wind passed as +tauMax cos(...) (negated ocean stress for the atm flip)."""
    w = adv.build_advgyre_wind()
    assert w.tau_x.shape == (adv.NY, adv.NX)
    assert np.allclose(np.asarray(w.tau_y), 0.0)
    # Recipe passes the negated ocean-side stress.
    np.testing.assert_allclose(
        np.asarray(w.tau_x)[:, 0], -adv.advgyre_taux_ocean_profile(), atol=1e-12
    )


@pytest.mark.skipif(
    _RUN_DIR is None,
    reason="set LEGOESM_OCEAN_FIDELITY_MITGCM_ADVGYRE_RUN to a built MITgcm run dir",
)
def test_dst3_multidim_matches_mitgcm_advscheme80():
    """Advect the dye 4 steps in MITgcm's prescribed equilibrated velocity with
    the canonical legoESM advection operator and match MITgcm's PTRACER dumps.

    Asserts: exact tracer-mass conservation, near-unit max ratio, and an
    advection-signal-relative L2 error well under the upwind/tvd alternatives
    (dst3_multidim is the closest match to MITgcm advScheme=80)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        min_cell_to_uface,
        min_cell_to_vface,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _compute_advection_flux_div,
    )

    run = Path(_RUN_DIR)
    it0 = 259200
    ny, nx = adv.NY, adv.NX
    geom = adv.build_advgyre_geometry()
    mask2d = np.asarray(adv.build_advgyre_land_mask())

    def rd(p):
        return np.fromfile(p, dtype=">f4").reshape(ny, nx).astype(np.float64)

    u_mit = rd(run / f"U.{it0:010d}.data")
    v_mit = rd(run / f"V.{it0:010d}.data")

    uf = np.zeros((ny, nx + 1))
    uf[:, :nx] = u_mit
    uf[:, nx] = u_mit[:, 0]
    vf = np.zeros((ny + 1, nx))
    vf[:ny, :] = v_mit
    um = np.zeros((ny, nx + 1))
    vm = np.zeros((ny + 1, nx))
    for j in range(1, nx):
        um[:, j] = mask2d[:, j - 1] * mask2d[:, j]
    for i in range(1, ny):
        vm[i, :] = mask2d[i - 1, :] * mask2d[i, :]

    h = jnp.full((ny, nx, 1), adv.H_DEPTH_M)
    hu = min_cell_to_uface(h)
    hv = min_cell_to_vface(h, geom)
    mfu = jnp.asarray(uf[:, :, None]) * hu * jnp.asarray(um[:, :, None])
    mfv = jnp.asarray(vf[:, :, None]) * hv * jnp.asarray(vm[:, :, None])
    w = jnp.zeros((ny, nx, 2))
    active = jnp.asarray(mask2d[:, :, None])

    def advect(scheme):
        dye = adv.build_dye_field()
        for _ in range(adv.N_STEPS):
            dh, vt = _compute_advection_flux_div(
                dye, scheme, mfu, mfv, w, h, hu, hv, geom, adv.DT_S,
            )
            dye = jnp.where(
                active > 0.5,
                (h * dye - adv.DT_S * (dh + vt)) / jnp.maximum(h, 1e-10),
                dye,
            )
        return np.asarray(dye)[:, :, 0]

    mit0 = rd(run / f"PTRACER01.{it0:010d}.data")
    mit4 = rd(run / f"PTRACER01.{it0 + adv.N_STEPS:010d}.data")
    wet = mask2d > 0.5
    signal = np.linalg.norm((mit4 - mit0)[wet])              # advection signal

    lego = advect("dst3_multidim")
    # Mass conservation (pure advection, closed basin) — machine tight.
    np.testing.assert_allclose(lego.sum(), mit4.sum(), atol=1e-10)
    np.testing.assert_allclose(lego.sum(), 1.0, atol=1e-10)
    # Max ratio near unity.
    np.testing.assert_allclose(lego.max(), mit4.max(), rtol=2e-3)
    # Error small relative to the actual advection signal.
    err = np.linalg.norm((lego - mit4)[wet])
    assert err / signal < 0.02, f"dst3_multidim rel-to-signal {err/signal:.4f}"
    # And strictly better than the upwind alternative (discriminating).
    err_upw = np.linalg.norm((advect("upwind") - mit4)[wet])
    assert err < err_upw
