"""Tests for the closed cloud-water budget probe and the terms it reads."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load():
    root = Path(__file__).resolve().parents[2]
    p = root / "scripts" / "validate" / "diag_qc_budget_closed.py"
    spec = importlib.util.spec_from_file_location("_qcb_probe", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_qcb_probe"] = m
    spec.loader.exec_module(m)
    return m


def test_band_rate_is_area_and_mass_weighted_inside_the_mask():
    m = _load()
    rate = np.array([[1.0, 2.0], [4.0, 0.0]])
    dp = np.full((2, 2), 100.0)
    w = np.array([0.25, 0.75])
    mask = np.array([[True, False], [True, False]])
    # (0.25*1 + 0.75*4) * 100 / 10
    assert abs(m.band_rate(rate, dp, w, 10.0, mask) - 32.5) < 1e-12
    empty = np.zeros((2, 2), dtype=bool)
    assert m.band_rate(rate, dp, w, 10.0, empty) == 0.0


def test_parser_defaults_to_the_deficit_band():
    m = _load()
    a = m.build_arg_parser().parse_args(["--config", "c", "--restart", "r"])
    assert a.band == [500.0, 800.0]
    assert a.no_graupel is False


def _tiny_state(jnp):
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    n, k = 3, 4
    return dict(
        T=jnp.full((n, k), 265.0), q_v=jnp.full((n, k), 3.0e-3),
        hyd=HydrometeorState(
            q_c=jnp.full((n, k), 5.0e-4), q_r=jnp.full((n, k), 1.0e-4),
            q_i=jnp.full((n, k), 2.0e-5), q_s=jnp.full((n, k), 1.0e-4),
            q_g=jnp.full((n, k), 1.0e-4), N_c=jnp.full((n, k), 1.0e8),
            N_r=jnp.full((n, k), 1.0e4), N_i=jnp.full((n, k), 1.0e4)),
        p_full=jnp.broadcast_to(jnp.linspace(8.0e4, 5.0e4, k), (n, k)),
        p_half=jnp.broadcast_to(jnp.linspace(8.5e4, 4.5e4, k + 1), (n, k + 1)),
        rho=jnp.full((n, k), 0.9), dz=jnp.full((n, k), 500.0))


def test_published_budget_reconstructs_the_scheme_tendency_exactly():
    """The budget must CLOSE, or a term is missing from the published set.

    Non-vacuous: drop any single term from the sum and the residual becomes
    that term's magnitude, which is far above the tolerance asserted here.
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics)
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig

    s = _tiny_state(jnp)
    cfg = MorrisonConfig(publish_qc_budget=True)
    o = morrison_microphysics(s["T"], s["q_v"], s["hyd"], s["p_full"],
                              s["p_half"], s["rho"], s["dz"], 600.0, cfg)
    assert o.qc_budget is not None
    total = sum(o.qc_budget.values())
    scale = float(jnp.max(jnp.abs(o.dq_c_dt)))
    assert scale > 0.0, "the test state must exercise the budget"
    assert float(jnp.max(jnp.abs(total - o.dq_c_dt))) <= 1.0e-12 * scale

    # Dropping a term must break closure -- proves the assertion above bites.
    partial = total - o.qc_budget["accretion"]
    assert float(jnp.max(jnp.abs(partial - o.dq_c_dt))) > 1.0e-12 * scale


def test_budget_is_off_by_default_and_changes_no_tendency():
    """The diagnostic must not perturb the model: default off, values equal."""
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics)
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig

    s = _tiny_state(jnp)
    assert MorrisonConfig().publish_qc_budget is False
    off = morrison_microphysics(s["T"], s["q_v"], s["hyd"], s["p_full"],
                                s["p_half"], s["rho"], s["dz"], 600.0,
                                MorrisonConfig())
    on = morrison_microphysics(s["T"], s["q_v"], s["hyd"], s["p_full"],
                               s["p_half"], s["rho"], s["dz"], 600.0,
                               MorrisonConfig(publish_qc_budget=True))
    assert off.qc_budget is None
    for f in ("dq_c_dt", "dq_r_dt", "dq_i_dt", "dq_s_dt", "dq_g_dt", "dT_dt"):
        assert bool(jnp.all(getattr(off, f) == getattr(on, f))), f


def test_graupel_riming_is_a_real_cloud_water_sink_that_do_graupel_removes():
    """Pins the measured faithfulness gap: CAM6's MG2 carries NO graupel.

    MG2's eight constituents are CLDLIQ, CLDICE, NUMLIQ, NUMICE, RAINQM,
    SNOWQM, NUMRAI, NUMSNO (micro_mg_cam.F90:139) and the string "graupel"
    does not occur in micro_mg2_0.F90 at all.  Its riming of cloud water by
    snow sends the rimed mass to SNOW (:1894).  Our port routes it to graupel
    instead, which is why this term exists and why turning graupel off removes
    it entirely.
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics)
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig

    s = _tiny_state(jnp)
    on = morrison_microphysics(s["T"], s["q_v"], s["hyd"], s["p_full"],
                               s["p_half"], s["rho"], s["dz"], 600.0,
                               MorrisonConfig(publish_qc_budget=True))
    off = morrison_microphysics(
        s["T"], s["q_v"], s["hyd"], s["p_full"], s["p_half"], s["rho"],
        s["dz"], 600.0,
        MorrisonConfig(publish_qc_budget=True, do_graupel=False))
    rg_on = float(jnp.sum(on.qc_budget["riming_graupel"]))
    rg_off = float(jnp.sum(off.qc_budget["riming_graupel"]))
    assert rg_on < 0.0, "supercooled state must rime cloud water onto graupel"
    assert rg_off == 0.0, "do_graupel=False must remove the graupel path"
    # Less cloud water is destroyed without the graupel route.
    assert float(jnp.sum(off.dq_c_dt)) > float(jnp.sum(on.dq_c_dt))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_budget_publication_survives_strict_sedimentation_mode():
    """Strict mode poisons every float output; the budget dict is not one.

    Non-vacuous: before the type filter, this raised
    ``ValueError`` from ``jnp.asarray`` on a dict (codex review, 2026-09-23).
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics)
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig

    s = _tiny_state(jnp)
    cfg = MorrisonConfig(publish_qc_budget=True, sed_cfl_substeps=True,
                         sed_cfl_substeps_strict=True)
    o = morrison_microphysics(s["T"], s["q_v"], s["hyd"], s["p_full"],
                              s["p_half"], s["rho"], s["dz"], 600.0, cfg)
    assert o.qc_budget is not None
    assert set(o.qc_budget) and all(
        hasattr(v, "shape") for v in o.qc_budget.values())


def test_upstream_flag_is_off_by_default():
    m = _load()
    a = m.build_arg_parser().parse_args(["--config", "c", "--restart", "r"])
    assert a.upstream is False
