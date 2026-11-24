"""Geothermal heating against COMPILED NEMO 5.0.1 ``tra_bbc`` (trabbc.F90).

Fixture ``oracle_fixtures/nemo_5_0_1_trabbc.npz``: 24 synthetic columns (land,
one-level, full-depth, partial bottom cells, z* factors r3t != 0, heat flows
0..300 mW/m2 incl. NEMO's 86.4) run through the ORCA1 build's own ``trabbc.o``
(driver and votes: ``oracle_audit/batch_frazil_edmf_kf/geo``).  NEMO's live
thickness is ``e3t_3d*(1 + r3t*tmask)`` (key_qco), passed here as ``e3t_kmm``.

KNOWN DIFFERENCE: NEMO also writes ``qgh/e3t`` into level 1 of land columns
(``mbkt = 1``, ``tmask = 0``; inert, the land point is masked); ours gives 0.
Compared on wet columns (1e-12 rel, 1e-20 abs).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics import geothermal as G

jax.config.update("jax_enable_x64", True)

FIX = np.load(Path(__file__).parent / "oracle_fixtures" / "nemo_5_0_1_trabbc.npz")
NWET = FIX["nwet"]
NL = FIX["e3t_0"].shape[1]
WET = (np.arange(NL)[None, :] < NWET[:, None]).astype(float)
E3 = FIX["e3t_0"] * (1.0 + FIX["r3t"][:, None] * WET)
COLS = NWET > 0
REF = FIX["nemo_dtdt"]
RHO0, CP = float(FIX["rho_0"]), float(FIX["c_sw"])


def _nemo_written(e3=E3, hf=FIX["heatflow_mwm2"]):
    return np.asarray(G.nemo_geothermal_bottom_heating_tendency(
        jnp.asarray(e3), jnp.asarray(WET), heatflow_mwm2=jnp.asarray(hf), rho_0=RHO0, c_sw=CP))


def _general():
    return np.asarray(G.geothermal_bottom_heating_tendency(
        jnp.asarray(E3), jnp.asarray(WET), flux_wm2=jnp.asarray(FIX["heatflow_mwm2"] * 1e-3),
        rho_0=RHO0, c_sw=CP))


def _close(got, ref):
    return bool(np.all(np.abs(got - ref) <= 1e-12 * np.abs(ref) + 1e-20))


@pytest.mark.parametrize("fn", [_nemo_written, _general])
def test_matches_compiled_nemo_on_wet_columns(fn):
    assert COLS.sum() == 23 and np.count_nonzero(REF[COLS]) == 22   # one zero-flux column
    assert _close(fn()[COLS], REF[COLS])


def test_land_column_known_difference():
    assert REF[~COLS][0, 0] > 0.0 and np.all(_nemo_written()[~COLS] == 0.0)


@pytest.mark.parametrize("mutant", ["reference_thickness", "top_cell", "mw_to_w"])
def test_fixture_rejects_plausible_mistakes(mutant):
    """Non-vacuity: three plausible wrong forms each fail the comparison."""
    if mutant == "reference_thickness":
        got = _nemo_written(e3=FIX["e3t_0"])
    elif mutant == "top_cell":
        got = np.zeros_like(REF)
        got[:, 0] = FIX["heatflow_mwm2"] * 1e-3 / (RHO0 * CP * E3[:, 0])
    else:
        got = _nemo_written(hf=FIX["heatflow_mwm2"] * 1e3)
    assert not _close(got[COLS], REF[COLS])
