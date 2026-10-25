"""Direct tests for the jpl=1 lane-3 kernel probe's adapters."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.ridging import (
    SI3JPL1RidgingConfig,
    SI3JPL1RidgingState,
    SI3RdgrftState,
    si3_ice_dyn_rdgrft,
)

PROBE = (
    Path(__file__).parents[2]
    / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o6_lane3_kernel_probe.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_si3_o6_lane3_kernel_probe_test", PROBE)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

CARD = SimpleNamespace(
    ridging_config=SI3JPL1RidgingConfig(),
    nn_icesal=4,
    dt_s=30.0,
    thermodynamics=False,
    halo_width=2,
)


def test_config_is_the_resolved_ice_rheo_deck_after_nemo_override():
    cfg = probe.rdgrft_config(CARD)
    assert (cfg.rn_porordg, cfg.rn_fsnwrdg, cfg.rn_fsnwrft, cfg.rn_fpndrdg, cfg.rn_fpndrft) == (
        0.0,
        1,
        1,
        1,
        1,
    )
    assert cfg.ln_distf_exp and not cfg.ln_distf_lin and cfg.rdt_ice == 30.0


def test_ridging_adapter_is_the_o6_kernel_at_jpl1():
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    rng = np.random.default_rng(3)
    sh = (8, 7)
    a = rng.uniform(0.3, 0.95, sh)
    st = SI3JPL1RidgingState(
        ice_area=jnp.asarray(a),
        open_water_area=jnp.asarray(1.0 - a),
        ice_volume=jnp.asarray(a * 1.5),
        snow_volume=jnp.asarray(a * 0.2),
        age_content=jnp.zeros(sh),
        pond_area=jnp.zeros(sh),
        pond_volume=jnp.zeros(sh),
        pond_lid_volume=jnp.zeros(sh),
        snow_enthalpy=jnp.asarray(rng.uniform(1e6, 1e7, sh + (5,))),
        ice_enthalpy=jnp.asarray(rng.uniform(1e7, 1e8, sh + (10,))),
        ice_salt_content=jnp.asarray(rng.uniform(0.1, 1.0, sh + (10,))),
    )
    divu, delta = jnp.asarray(rng.normal(-1e-5, 1e-5, sh)), jnp.asarray(rng.uniform(1e-5, 4e-5, sh))
    sst = jnp.full(sh, -1.8)
    got, losses = probe.o6_ridging_adapter(CARD, sst)(st, divu, delta, 30.0)
    assert losses is None
    c = lambda x: x[..., None]  # noqa: E731
    z = jnp.zeros(sh)
    direct = si3_ice_dyn_rdgrft(
        SI3RdgrftState(
            a_i=c(st.ice_area),
            v_i=c(st.ice_volume),
            v_s=c(st.snow_volume),
            oa_i=c(z),
            a_ip=c(z),
            v_ip=c(z),
            v_il=c(z),
            e_s=c(st.snow_enthalpy),
            e_i=c(st.ice_enthalpy),
            szv_i=c(st.ice_salt_content),
            at_i=st.ice_area,
            ato_i=st.open_water_area,
            wfx_dyn=z,
            sfx_dyn=z,
            hfx_dyn=z,
            wfx_snw_dyn=z,
            wfx_pnd=z,
        ),
        divu_i=divu,
        delta_i=delta,
        sst_m=sst,
        sss_m=jnp.full(sh, probe.SSS_M),
        hi_max=jnp.asarray([0.0, probe.RN_HIMAX]),
        nn_hls=2,
        config=probe.rdgrft_config(CARD),
    ).state
    np.testing.assert_array_equal(np.asarray(got.ice_area), np.asarray(direct.a_i)[..., 0])
    np.testing.assert_array_equal(np.asarray(got.ice_enthalpy), np.asarray(direct.e_i)[..., 0])
    np.testing.assert_array_equal(np.asarray(got.open_water_area), np.asarray(direct.ato_i))
    assert not np.array_equal(np.asarray(got.ice_area), a)  # ridging acted


def test_summary_counts_only_informative_rows():
    rows = [
        {"name": "entry.step9.a_i", "status": "BIT-EXACT", "bitwise_nonzero_over_n": "0 / 4"},
        {"name": "entry.step9.v_i", "status": "DEBT", "bitwise_nonzero_over_n": "2 / 4"},
        {"name": "entry.step9.oa_i", "status": "DEBT", "bitwise_nonzero_over_n": "4 / 4"},
    ]
    assert probe.summarize(rows) == {
        "informative_rows": 2,
        "bit_exact_rows": 1,
        "nonexact": {"v_i": "2 / 4"},
    }
