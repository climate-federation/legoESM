"""ORCA1 oracle: the N2-integral mixed-layer depth is zero on land.

NEMO 5.0.1 ``zdf_mxl`` ends with ``hmlp = gdepw(nmln) * ssmask``
(zdfmxl.F90:104).  The port's ``_nemo_mld_from_n2_integral`` returned the
unmasked gather, i.e. ``gdepw_0(nlb10)`` (1.02 m on ORCA1) on every land
column (tier-3 audit finding P8).

Fixture ``fixtures/orca1_t3/mld_landmask.npz`` (+ ``.json`` provenance): three
6x6 interior patches of the tier-3 MLD record (kt 17281, day-30 restart,
32-rank run) -- two coastal (land + wet), one seeded random -- with NEMO's
Kbb T/S, geometry, ``rn2b``, ``hmlp`` and ``nmln``.

Sub-quantity (campaign vote C6): NEMO's ``rn2b`` is fed through
``n2_override`` and the fp64 depth ladder is used, isolating the integral and
readout from the separate N2-EOS (P6) and fp32-ladder (P7) defects.  Then
``hmlp`` must match NEMO on EVERY column, land included, at rel <= 1e-12
(measured over all 32 ranks: 1.3e-16; NEMO 20-seed spread on hmlp 6.9e-16)
and ``nmln`` exactly.
"""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

FIX = pathlib.Path(__file__).parent / "fixtures" / "orca1_t3" / "mld_landmask.npz"
TOL = 1e-12


@pytest.fixture(autouse=True)
def _fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


def _eos_must_not_run(*_a):
    raise AssertionError("the n2_integral path with n2_override must not read eos_fn")


def _mld(z, n):
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import _nemo_mld
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate, create_z_star_from_thicknesses)
    tm = z[f"p{n}_tmask"]
    e3t3 = z[f"p{n}_e3t3"]
    mbkt = z[f"p{n}_mbkt"]
    e3t_1d = jnp.asarray(z["e3t_1d"])
    gdept_1d = jnp.asarray(z["gdept_1d"])
    zs = create_z_star_from_thicknesses(
        e3t_1d, gdept_1d, nemo_e3w_source="depth_difference")
    H = np.sum(e3t3 * tm, axis=-1)
    tmpl = create_partial_cell_coordinate(zs, jnp.asarray(H))
    f64 = lambda x: jnp.asarray(np.asarray(x, np.float64))  # noqa: E731
    zc = tmpl._replace(
        z_full_ref=f64(zs.z_full_ref), z_half_ref=f64(zs.z_half_ref),
        dz_ref=e3t_1d, dz_half_ref=f64(zs.dz_half_ref), t_depth_ref=gdept_1d,
        h_partial=jnp.asarray(e3t3 * tm),
        bottom_level=jnp.asarray(np.where(mbkt > 0, mbkt - 1, -1),
                                 dtype=tmpl.bottom_level.dtype),
        is_active=jnp.asarray(tm > 0, dtype=tmpl.is_active.dtype))
    hml, m_base = _nemo_mld(
        "n2_integral", jnp.asarray(z[f"p{n}_T"]), jnp.asarray(z[f"p{n}_S"]),
        jnp.asarray(tm[..., 0]), zc, _eos_must_not_run, float(z["rho_c"]),
        g=float(z["grav"]), rho_0=float(z["rho0"]), active_3d=jnp.asarray(tm),
        jacobian=jnp.asarray(1.0 + z[f"p{n}_r3t"]),
        n2_override=jnp.asarray(z[f"p{n}_rn2b"]))
    return np.asarray(hml), np.asarray(m_base) + 2


def _rel(a, b):
    d = np.maximum(np.abs(a), np.abs(b))
    return np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_mld_hmlp_matches_nemo_including_land(n):
    z = np.load(FIX)
    hml, nmln = _mld(z, n)
    ss = z[f"p{n}_ssmask"] > 0
    worst = float(np.max(_rel(hml, z[f"p{n}_hmlp"])))
    assert worst <= TOL, f"patch {n}: hmlp rel {worst:.3e} (land {int((~ss).sum())} cols)"
    bad = int(np.sum(nmln[ss] != z[f"p{n}_nmln"][ss]))
    assert bad == 0, f"patch {n}: {bad} wet nmln != NEMO"


def test_fixture_has_land_and_wet_columns():
    z = np.load(FIX)
    for n in (0, 1):
        ss = z[f"p{n}_ssmask"] > 0
        assert ss.any() and (~ss).any()
        assert np.all(z[f"p{n}_hmlp"][~ss] == 0.0)
