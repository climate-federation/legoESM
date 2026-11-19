"""CAM6 MAM4 loadaer + voltonumb vs the gfortran oracle (verbatim ndrop.F90 code).

Cases / outputs: ``scripts/validate/cam6_ndrop_loadaer_oracle_{cases,expected}.txt``
(mode constants and species densities / hygroscopicities taken from the pinned CAM6
property files).  Tolerance fp64 rtol 1e-12 / atol 1e-30 (user decision v8 20a).
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics.cam6_modal_aerosol import (  # noqa: E402
    MAM4_SPECIES, cam6_loadaer, load_cam6_mode_props, voltonumb,
)

_DIR = Path(__file__).resolve().parents[4] / "scripts" / "validate"


def _read():
    lines = (_DIR / "cam6_ndrop_loadaer_oracle_cases.txt").read_text().splitlines()
    f = lambda s: np.array([float(x) for x in s.split()])
    sig, lo, hi = f(lines[1]), f(lines[2]), f(lines[3])
    n = int(lines[4])
    cases, i = [], 5
    for _ in range(n):
        m, _ns, _ph, cs, num = lines[i].split()
        cases.append((int(m), float(cs), float(num), f(lines[i + 1]), f(lines[i + 2]), f(lines[i + 3])))
        i += 4
    exp = [f(s) for s in (_DIR / "cam6_ndrop_loadaer_oracle_expected.txt").read_text().splitlines() if s.strip()]
    return sig, lo, hi, cases, exp


def test_voltonumb_matches_oracle():
    sig, lo, hi, _, exp = _read()
    np.testing.assert_allclose(np.concatenate([voltonumb(lo, sig), voltonumb(hi, sig)]), exp[0],
                               rtol=1e-12, atol=1e-30)


@pytest.mark.parametrize("k", range(8))
def test_loadaer_matches_oracle(k):
    sig, lo, hi, cases, exp = _read()
    m, cs, num, mass, dens, hyg = cases[k]
    vlo, vhi = voltonumb(lo, sig)[m - 1], voltonumb(hi, sig)[m - 1]
    got = cam6_loadaer(jnp.asarray(mass), jnp.asarray(num), jnp.asarray(cs), dens, hyg, vlo, vhi)
    np.testing.assert_allclose(np.array([float(g) for g in got]), exp[k + 1], rtol=1e-12, atol=1e-30)


def test_loadaer_grad_finite_at_empty_mode():
    sig, lo, hi, cases, _ = _read()
    m, cs, num, mass, dens, hyg = cases[7]          # vaerosol <= 1e-30 branch
    vlo, vhi = voltonumb(lo, sig)[m - 1], voltonumb(hi, sig)[m - 1]
    g = jax.grad(lambda x: sum(cam6_loadaer(x, jnp.asarray(num), jnp.asarray(cs), dens, hyg,
                                            vlo, vhi)))(jnp.asarray(mass))
    assert np.all(np.isfinite(np.asarray(g)))


@pytest.mark.skipif(not os.environ.get("LEGOESM_CAM6_AEROSOL_DIR"),
                    reason="set LEGOESM_CAM6_AEROSOL_DIR to the pinned CAM6 property-file folder")
def test_property_files_give_the_oracle_constants():
    """The loader (sha256-checked) reproduces the constants the oracle cases were built from."""
    p = load_cam6_mode_props(os.environ["LEGOESM_CAM6_AEROSOL_DIR"])
    sig, lo, hi, cases, _ = _read()
    np.testing.assert_array_equal(p.sigmag, sig)
    np.testing.assert_array_equal(p.dgnumlo, lo)
    np.testing.assert_array_equal(p.dgnumhi, hi)
    for m, _cs, _num, _mass, dens, hyg in cases:
        np.testing.assert_array_equal(p.specdens[m - 1], dens)
        np.testing.assert_array_equal(p.spechygro[m - 1], hyg)
    assert [len(s) for s in p.specdens] == [len(s) for s in MAM4_SPECIES]
