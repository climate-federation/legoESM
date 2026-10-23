"""Oracle match: longwave surface-interface temperature and level source.

The surface interface (lowest face, the wall) must carry the surface
temperature, as the docstring of ``optics_base.reconstruct_face_values``
states.  The CAM6 host sets it to (lwup/stebol)**0.25 (rrtmg_state.F90:
tint(pverp)), which equals T_sfc for a unit-emissivity surface.  RRTMGP's level
source there is ``lev_source(:,1,:) = pfrac(:,1,:) * B(tlev(:,1))``
(mo_gas_optics_rrtmgp_kernels.F90, compute_Planck_source).

Fixture ``oracle_fixtures/rrtmgp_lw_sfc_face.npz``: inputs are the arrays the
production entry ``RRTMGP.solve_columns`` hands to the LW solver (8 RFMIP
clear-sky columns, overhead layer, halos); outputs come from RUNNING
RTE+RRTMGP v1.8 (rte-rrtmgp 41c5fcd950fed09b8afe186dede266824eca7fd3,
rrtmgp-data v1.8.2 34967d68703e682305ec4da72ee22bbe1f11b198) via
scripts/validate/oracle/rrtmgp_lw_sfc_face/{build.sh,make_fixture.py}.

Scope: the UNCOMBINED surface source pfrac(layer 1) * B(T at face 1).  The
solver reads the combined source ``lw_combine_sources``, which at the surface
face is sqrt(pfrac(halo) * pfrac(layer 1)) * B(T_sfc) instead of the oracle's
pfrac(layer 1) * B(tlev(1)); that halo-pfrac gap is a separate audit finding
and is NOT covered here.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import optics_base
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

_FIX = Path(__file__).parent / "oracle_fixtures" / "rrtmgp_lw_sfc_face.npz"
_TOL = 1e-10  # user-approved oracle-match tolerance (max relative error, fp64)


@pytest.fixture(scope="module")
def case():
    z = np.load(_FIX)
    meta = json.loads(str(z["meta"]))
    opt = RRTMGP.from_legoesm_config(RRTMGPConfig(**meta["cfg"])).optics_lib
    idx = opt.gas_optics_lw.idx_gases
    vmr = {idx[str(k)]: jnp.asarray(v) for k, v in zip(z["vmr_names"], z["vmr"])}
    return z, opt, vmr


def _surface_level_source(opt, vmr, p, T, sfcT):
    @jax.jit
    def one(ig):
        ps = opt.compute_planck_sources(p, T, ig, vmr, sfc_temperature=sfcT)
        return ps["planck_src_bottom"][:, 0, 1]  # face index 1 = surface (wall)
    return jnp.stack([one(jnp.asarray(i)) for i in range(opt.n_gpt_lw)], axis=-1)


def test_surface_face_temperature_is_t_sfc(case):
    z, _, _ = case
    t_bottom, _ = optics_base.reconstruct_face_values(
        jnp.asarray(z["T"]), f_lower_bc=jnp.asarray(z["sfcT"]))
    np.testing.assert_array_equal(np.asarray(t_bottom)[:, 0, 1], z["sfcT"][:, 0])
    np.testing.assert_array_equal(z["tlev_sfc_oracle"], z["sfcT"][:, 0])


def test_surface_level_source_matches_rrtmgp_v18(case):
    z, opt, vmr = case
    ours = np.asarray(_surface_level_source(
        opt, vmr, jnp.asarray(z["p"]), jnp.asarray(z["T"]), jnp.asarray(z["sfcT"])))
    ref = z["lev_src_sfc_oracle"]
    assert ours.shape == ref.shape and np.all(ref > 0)
    rel = np.max(np.abs(ours - ref) / np.abs(ref))
    assert rel <= _TOL, f"max rel error {rel:.3e} > {_TOL}"


def test_surface_level_source_gradient_finite(case):
    z, opt, vmr = case
    p = jnp.asarray(z["p"])

    def loss(T, sfcT):
        ps = opt.compute_planck_sources(p, T, jnp.asarray(0), vmr, sfc_temperature=sfcT)
        return jnp.sum(ps["planck_src_bottom"][:, 0, 1:-1])

    gT, gS = jax.grad(loss, argnums=(0, 1))(jnp.asarray(z["T"]), jnp.asarray(z["sfcT"]))
    assert np.all(np.isfinite(gT)) and np.all(np.isfinite(gS))
    assert np.all(np.asarray(gS) > 0)  # dB/dT > 0 at the surface face
