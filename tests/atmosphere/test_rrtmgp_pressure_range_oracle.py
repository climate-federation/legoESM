"""Oracle match: RRTMGP gas optics for layer pressures outside the table.

Upstream RTE+RRTMGP refuses a layer pressure outside [press_ref_min,
press_ref_max] (1.005 Pa .. 1096.6 hPa) when its value checks are on; with them
off its kernel clamps the pressure index to the end interval and extrapolates
the weight linearly (mo_gas_optics_rrtmgp_kernels.F90:115-117).  legoESM
follows the checks-off rule (votes V1-V6, author/codex/GLM 3-0).
Oracle tolerance: 1e-10 in the table, 1e-9 outside it (see _TOL_OUT).

Fixture ``oracle_fixtures/rrtmgp_pressure_range.npz``: three columns at 200,
250 and 300 K (inside the temperature table, so pressure is the only
out-of-range input), layers at 0.3 and 0.8 Pa, both table end nodes, 24
log-spaced in-table layers and 1.12e5 and 1.3e5 Pa; LW optical depth, LW layer
Planck source, SW optical depth and SW single-scattering albedo from RUNNING
unpatched rte-rrtmgp v1.8 with rte_config_checks(.false.) via
scripts/validate/oracle/rrtmgp_pressure_range/{build.sh,make_fixture.py}
(provenance in the fixture's ``meta``).
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

_FIX = Path(__file__).parent / "oracle_fixtures" / "rrtmgp_pressure_range.npz"
_TOL = 1e-10      # oracle-match tolerance, max relative error, fp64 (vote V5)
# Layers outside the table: measured 2.2e-10 (LW tau, 0.3 Pa, ~6 intervals out).
# Our ln-p spacing is ln p_ref[1] - ln p_ref[0], upstream's the mean spacing
# (mo_gas_optics_rrtmgp.F90:1356); the extrapolation weight amplifies that
# difference (with upstream's form patched in: 1.8e-13).  Keeping our form keeps
# the in-table path bitwise (vote: author a, GLM a, codex c).
_TOL_OUT = 1e-9
_FLOOR = 1e-30    # relative-error denominator floor for exact zeros


def _max_rel(ours, ref, mask=None):
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    rel = np.abs(ours - ref) / np.maximum(np.abs(ref), _FLOOR)
    return float(np.max(rel if mask is None else rel[mask]))


@pytest.fixture(scope="module")
def ours():
    z = np.load(_FIX)
    opt = RRTMGP.from_legoesm_config(RRTMGPConfig(include_clouds=False)).optics_lib
    p = jnp.asarray(z["play"])[:, None, :]
    t = jnp.asarray(z["tlay"])[:, None, :]
    m = jnp.asarray(z["col_dry"])[:, None, :] * 1.0e4          # molecules/cm2 -> /m2
    gases = [str(g) for g in z["gases"]]

    def vmr(lookup):
        return {lookup.idx_gases[g]: jnp.asarray(z["vmr"][..., i])[:, None, :]
                for i, g in enumerate(gases) if g in lookup.idx_gases}

    def per_g(fn, n):
        one = jax.jit(fn)
        return np.stack([np.asarray(one(jnp.asarray(i)))[:, 0, :] for i in range(n)], -1)

    vl = vmr(opt.gas_optics_lw)
    lw = opt.with_all_gpts(True, p, t, m, vl)
    vs = vmr(opt.gas_optics_sw)
    sw = opt.with_all_gpts(False, p, t, m, vs)
    out = dict(
        tau_lw=per_g(lambda g: lw.compute_lw_optical_properties(p, t, m, g)["optical_depth"],
                     opt.n_gpt_lw),
        lay_src=per_g(lambda g: lw.compute_planck_sources(
            p, t, g, vl, sfc_temperature=t[:, :, 0], level_temperature=t[:, :, 1:])["planck_src"],
            opt.n_gpt_lw),
        tau_sw=per_g(lambda g: sw.compute_sw_optical_properties(p, t, m, g)["optical_depth"],
                     opt.n_gpt_sw),
        ssa_sw=per_g(lambda g: sw.compute_sw_optical_properties(p, t, m, g)["ssa"], opt.n_gpt_sw))
    return z, out, opt


def test_fixture_leaves_the_table_at_both_ends(ours):
    z, _, opt = ours
    p_ref = np.asarray(opt.gas_optics_lw.p_ref)
    assert z["play"].min() < p_ref.min() and z["play"].max() > p_ref.max()
    assert np.all((z["tlay"] > 160.0) & (z["tlay"] < 355.0))   # pressure is the only out-of-range input


def _inside(z, opt):
    p_ref = np.asarray(opt.gas_optics_lw.p_ref)
    return (z["play"] > p_ref.min()) & (z["play"] < p_ref.max())


@pytest.mark.parametrize("name", ["tau_lw", "lay_src", "tau_sw", "ssa_sw"])
def test_out_of_table_layers_match_upstream_checks_off(ours, name):
    z, out, opt = ours
    outside = np.broadcast_to(~_inside(z, opt)[..., None], out[name].shape)
    rel = _max_rel(out[name], z[name], outside)
    assert rel <= _TOL_OUT, f"{name}: max rel error {rel:.3e} > {_TOL_OUT}"


@pytest.mark.parametrize("name", ["tau_lw", "lay_src", "tau_sw", "ssa_sw"])
def test_in_table_layers_also_match(ours, name):
    """The in-range layers alone (the bitwise-unchanged path) meet the same gate."""
    z, out, opt = ours
    rel = _max_rel(out[name], z[name], np.broadcast_to(_inside(z, opt)[..., None], out[name].shape))
    assert rel <= _TOL, f"{name}: max rel error {rel:.3e} > {_TOL}"


def test_closed_table_range_is_the_generic_interpolant(ours):
    """Inside [p_min, p_max], end nodes and their float neighbours included, the
    pressure interpolant is the generic one bit for bit (codex review: a pressure
    one ulp above p_min carries the last floor index)."""
    from legoesm.atmosphere.physics.radiation.rrtmgp.optics import gas_optics, optics_utils
    _, _, opt = ours
    p_ref = jnp.asarray(opt.gas_optics_lw.p_ref)
    lo, hi = float(p_ref.min()), float(p_ref.max())
    rng = np.random.default_rng(0)
    p = np.concatenate([[lo, np.nextafter(lo, np.inf), hi, np.nextafter(hi, 0.0)],
                        np.exp(rng.uniform(np.log(lo), np.log(hi), 64))])
    p = jnp.asarray(p)
    for off in (None, jnp.ones(p.shape, jnp.int32)):
        new = gas_optics._pressure_interpolant(p, p_ref, off)
        ref = optics_utils.create_linear_interpolant(jnp.log(p), jnp.log(p_ref), offset=off)
        for a, b in ((new.interp_low, ref.interp_low), (new.interp_high, ref.interp_high)):
            assert np.array_equal(np.asarray(a.idx), np.asarray(b.idx))
            assert np.array_equal(np.asarray(a.weight), np.asarray(b.weight))
