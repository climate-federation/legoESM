"""Round 39: the isoneutral fold's placement, and the acquisition's admission rule.

Every test here is written so that it FAILS when the thing it checks is
removed; the two that guard a shipped shell rule EXECUTE that rule rather than
matching its text, so a rewording cannot make them vacuous.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
GATES = REPO / "scripts/validate/ocean_fidelity/testcases"
RUN_SH = GATES / "nemo_testcase_l2_gyre_round38_trazdf_kt2/run.sh"
sys.path.insert(0, str(GATES))


# --------------------------------------------------------------------------
# The acquisition's twin rule: raw identity for ONE record, consumed-field
# identity for the rest.
# --------------------------------------------------------------------------
def _raw_guard() -> str:
    """The shipped guard, extracted from run.sh and run as-is.

    Matching run.sh's TEXT would pass on a rewording that inverted the rule.
    This lifts the block between its own two markers and executes it, so the
    test fails whenever the shipped bytes stop implementing the rule.
    """
    text = RUN_SH.read_text()
    start = text.index("# THE ONE RECORD THAT MUST BE RAW-IDENTICAL")
    end = text.index("# EVERY OTHER round-37 record")
    block = text[start:end]
    assert "exit 71" in block, "the extracted block carries no refusal"
    return block


def _run_guard(tmp_path: Path, log_lines: list[str]) -> int:
    (tmp_path / "round38_raw_twin_cmp.log").write_text(
        "\n".join(log_lines) + "\n")
    script = (
        "set -uo pipefail\n"
        f'TARGET_RUN="{tmp_path}"\n'
        "KT1_RECORD=oracle_trazdf_matrix_kt00000001.bin\n"
        + _raw_guard()
        + "\nexit 0\n"
    )
    return subprocess.run(["bash", "-c", script], capture_output=True).returncode


def test_a_halo_only_difference_in_another_record_no_longer_refuses(tmp_path):
    """The defect this round fixed: a good acquisition was refused.

    Round 38's rule refused whenever ANY round-37 record differed raw.  Eight
    of that acquisition's 52 records differ in halo or undefined-slot bytes --
    uninitialised memory, which differs between two runs of the same binary --
    so the rule refused a run whose consumed fields were all identical.
    """
    assert _run_guard(tmp_path, [
        "RAW_IDENTICAL oracle_trazdf_matrix_kt00000001.bin",
        "RAW_DIFFERS   oracle_zdf_matrix_kt00000001.bin (falls through)",
    ]) == 0


def test_a_moved_kt1_trazdf_record_still_refuses(tmp_path):
    """The rule that must SURVIVE: the arm's own record is bit-pinned."""
    assert _run_guard(tmp_path, [
        "RAW_DIFFERS   oracle_trazdf_matrix_kt00000001.bin (falls through)",
        "RAW_IDENTICAL oracle_zdf_matrix_kt00000001.bin",
    ]) == 71


def test_a_missing_kt1_line_refuses(tmp_path):
    """A log that never mentions the record is a refusal, not a pass."""
    assert _run_guard(tmp_path, ["RAW_IDENTICAL oracle_rhs_kt00000001.bin"]) == 71


def test_run_sh_delegates_the_other_records_to_the_admission_gate():
    """The relaxed records must be decided by SOMETHING, and it is named."""
    text = RUN_SH.read_text()
    block = text[text.index("# EVERY OTHER round-37 record"):]
    head = block[:block.index("test -s")]
    assert '--baseline "$SOURCE_RUN"' in head, (
        "the source-run consumed-field admission is gone; the other records "
        "would then be checked by nothing")
    assert "--plant-consumed" in head, "the source admission has no plant"
    assert head.count("exit 7") >= 2, "the admission or its plant cannot refuse"


# --------------------------------------------------------------------------
# The change itself: on the WS-RK3 lane the GM/Redi slope operand set is the
# step-entry (Kbb) state, and the fold it builds on GYRE's horizontally
# uniform initial condition is EXACTLY zero -- which is NEMO's ah_wslp2 there.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _gyre_k33_call():
    """One kt=1 GYRE step, capturing the arguments of the model's K33 call.

    This runs the production step, so it costs one JAX compilation.  It is the
    only way to make the claim non-vacuously: the operands are chosen inside a
    1500-line function and the assertion has to be about what that function
    HANDED the slope routine, not about what a reader thinks it passes.
    """
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    entry_policy = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    import jax
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as M
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from nemo_testcase_l2_gyre_phase3_gate import CASE, _surface_forcings

    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    init = card.recipe.initial_state
    seen: dict[str, list] = {}

    def sink(name):
        return lambda v: seen.setdefault(name, []).append(
            np.array(v, dtype=np.float64))

    # ALL THREE GM/Redi CALL SITES, not only K33.  An independent diff review
    # showed that wrapping K33 alone leaves the density/Jacobian and the
    # tendency free to go back to the predictor state with every test green --
    # and the model's own comment calls that disagreement a CONFIRMED P1
    # defect, because K33 would then be built from a different time level than
    # the isoneutral flux it augments.
    real = {
        "K33": M.compute_isoneutral_K33_latlon,
        "jac": M.gm_redi_density_and_jacobian,
        "tend": M.gm_redi_tracer_tendency_latlon,
    }

    def wrap(tag, fn, capture_output):
        def wrapped(T, S, eta, *a, **kw):
            for label, value in (("T", T), ("S", S), ("eta", eta)):
                jax.debug.callback(sink(f"{tag}.{label}"), value)
            out = fn(T, S, eta, *a, **kw)
            if capture_output:
                jax.debug.callback(sink(tag), out)
            return out
        return wrapped

    M.compute_isoneutral_K33_latlon = wrap("K33", real["K33"], True)
    M.gm_redi_density_and_jacobian = wrap("jac", real["jac"], False)
    M.gm_redi_tracer_tendency_latlon = wrap("tend", real["tend"], False)
    try:
        freshwater, surface = _surface_forcings(card, init, 1)
        M.LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
        ).step(init, dt=card.dt_s, freshwater=freshwater,
               surface_forcing=surface)
    finally:
        M.compute_isoneutral_K33_latlon = real["K33"]
        M.gm_redi_density_and_jacobian = real["jac"]
        M.gm_redi_tracer_tendency_latlon = real["tend"]
        set_policy(entry_policy)
    assert cfg.tracer_time_integrator == "rk3_ws"
    return seen, init


def test_the_slope_operands_are_the_step_entry_state(_gyre_k33_call):
    """NEMO's ldf_slp is called on Nbb, once, outside the stage loop.

    FAILS ON THE PARENT COMMIT: the tracers it received were the forward-Euler
    predictor, 0.337 K away from step entry on 10199 of 21120 cells, and the
    ssh was the after-ssh, 2.8e-3 m away on 600 of 704.
    """
    seen, init = _gyre_k33_call
    for tag in ("K33", "jac", "tend"):
        assert [len(seen[f"{tag}.{n}"]) for n in ("T", "S", "eta")] == [1, 1, 1], (
            f"{tag} must be called exactly once per step")
        for label, entry in (("T", init.T.data), ("S", init.S.data),
                             ("eta", init.eta.data)):
            got = seen[f"{tag}.{label}"][0]
            want = np.asarray(entry, dtype=np.float64)
            unequal = int((got.view(np.uint64) != want.view(np.uint64)).sum())
            assert unequal == 0, (
                f"{tag}.{label}: {unequal} of {want.size} cells are not the "
                f"step-entry value, max |diff| "
                f"{np.abs(got - want).max():.17g}")


def test_the_fold_is_exactly_zero_on_a_horizontally_uniform_state(
        _gyre_k33_call):
    """GYRE's analytic initial T and S are horizontally uniform on wet cells.

    So NEMO's before-state slopes are exactly zero and its ah_wslp2 is
    identically 0.0 -- measured on the round-37 record, absolute maximum 0.
    legoESM's fold must be too, at the exact bar and not inside a tolerance.

    FAILS ON THE PARENT COMMIT at 9.6620886711883531e-13 m2/s.
    """
    seen, _ = _gyre_k33_call
    k33 = seen["K33"][0]
    nonzero = int(np.count_nonzero(k33))
    assert nonzero == 0, (
        f"{nonzero} of {k33.size} faces carry a fold NEMO does not; "
        f"absolute maximum {np.abs(k33).max():.17g} m2/s")


def test_the_zero_is_not_degenerate(_gyre_k33_call):
    """A routine that returns zero for everything would pass the test above.

    Perturbing one wet cell of the captured step-entry state by 1e-6 K must
    give a fold that is not zero, so the exact zero above is a property of the
    STATE and not of the routine.
    """
    import jax.numpy as jnp
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as M
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from nemo_testcase_l2_gyre_phase3_gate import CASE

    seen, init = _gyre_k33_call
    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    # THE BUMPED CELL MUST BE WET, or the arm perturbs land and proves nothing.
    assert float(np.asarray(init.land_mask.data)[10, 16]) > 0.0
    T = jnp.asarray(seen["K33.T"][0]).at[10, 16, 3].add(1e-6)
    bumped = M.compute_isoneutral_K33_latlon(
        T, jnp.asarray(seen["K33.S"][0]), jnp.asarray(seen["K33.eta"][0]),
        jnp.asarray(init.H_bathy.data), card.recipe.grid, card.recipe.z_coord,
        cfg.gm_redi, eos=cfg.eos, eos_linear=cfg.eos_linear,
        mask=jnp.asarray(init.land_mask.data), rho_0=cfg.constants.rho_0,
        g=cfg.constants.g,
        native_slope_eta=jnp.asarray(seen["K33.eta"][0]),
        u_mask=jnp.asarray(init.u_mask.data),
        v_mask=jnp.asarray(init.v_mask.data), dt=card.dt_s,
        eos_depth=getattr(cfg, "eos_depth", "insitu"))
    assert float(np.abs(np.asarray(bumped)).max()) > 0.0, (
        "a 1e-6 K bump left the fold at exactly zero, so the zero above is "
        "the routine's and proves nothing about the state")


def _ws_guard(text: str) -> str:
    """The WS-RK3 before-state assignment, lifted from the running module."""
    head = "        _T_gm_in = T_mid if _ldf_state is None else _ldf_state[0]"
    start = text.index(head)
    end = text.index("            _eta_gm_in = state.eta.data", start)
    block = text[start:end]
    return block[block.index("if ("):]


def _guard_is_scoped(text: str) -> bool:
    guard = _ws_guard(text)
    return "_ldf_state is None" in guard and '== "rk3_ws"' in guard


def test_the_modified_leapfrog_lane_still_wins():
    """``_ldf_state`` takes precedence over the WS-RK3 statement.

    The rk3_ws branch is guarded on ``_ldf_state is None``, so the
    modified-leapfrog lane -- which supplies its own Nbb operands and is the
    lane DINO runs -- must be untouched.  An independent claim review pointed
    out that the two tests above are blind to it.

    This inspects the source of the module that RUNS, and it proves it can
    fail: the same predicate is applied to two mutated copies, one with the
    ``_ldf_state`` guard deleted and one with the lane scope deleted, and both
    must be rejected.
    """
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as M
    text = Path(M.__file__).read_text()
    assert _guard_is_scoped(text)

    def mutated(old, new):
        out = text.replace(old, new)
        assert out != text, (
            "the mutation control replaced nothing, so its assertion below "
            "would pass on any code; re-anchor it on the current source")
        return out

    assert not _guard_is_scoped(
        mutated("if (_ldf_state is None\n"
                     "                and getattr(_cfg_b, "
                     '"tracer_time_integrator", "euler")\n'
                     '                == "rk3_ws"):',
                     "if (getattr(_cfg_b, \"tracer_time_integrator\", "
                     '"euler")\n                == "rk3_ws"):')), (
        "deleting the _ldf_state guard did not make this check fail, so it "
        "would not catch that regression either")
    assert not _guard_is_scoped(mutated('== "rk3_ws"', '== "euler"')), (
        "changing the lane scope did not make this check fail")
