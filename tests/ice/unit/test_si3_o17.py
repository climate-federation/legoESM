"""Structural, numerical and code-revert-plant tests for the O17 production entry.

Every plant below reverts CODE (a monkeypatched production function or a
skipped call) and must make the same production assertions fail that the
unplanted step passes; none edits an already computed result.
"""

import hashlib
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ice import sea_ice
from legoesm.ice import si3_o17 as o17_mod
from legoesm.ice.config import SeaIceConfig, build_si3_orca2_o17_config
from legoesm.ice.si3_o17 import SI3O17Card, SI3O17LiveForcing, SI3O17StepState, si3_o17_step
from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig, si3_eos_fzp, si3_lbc_lnk_orca2

from scripts.validate.ocean_fidelity.testcases import (
    nemo_si3_o17_composition_scorer as scorer,
)
from scripts.validate.ocean_fidelity.testcases import nemo_si3_o17_record_budget as budget
from scripts.validate.ocean_fidelity.testcases import nemo_si3_o17_synthetic_record as synthetic
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_acquisition.preflight import (
    ORDER,
    preflight,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_ad_probe import (
    DEFAULT_MASTER,
    _case,
)

REPO = Path(__file__).resolve().parents[3]
# Eager binary64 digests of the 7x7 mixed-regime card (one aEVP subcycle) after
# Resume 5.  Any code revert of a composed statement changes them.
PROBE_SHA256 = "7e6ff54def6cdd1b6ba878431687a00c7627e50d37af7824ee8d32481ff6ebbe"
FULL_SHA256 = "671efb4a3427c0c84f7a23d6b9f2ce25ce58d6981cf109724d0bed00eea29102"


def _leaf_rows(tree):
    rows = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.ascontiguousarray(np.asarray(leaf))
        rows.append((jax.tree_util.keystr(path), array))
    return rows


def _digest(tree) -> str:
    digest = hashlib.sha256()
    for name, array in _leaf_rows(tree):
        digest.update(f"{name}|{array.dtype.str}|{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _bit_mismatches(actual, expected) -> list[str]:
    out = []
    for (name, a), (_, b) in zip(_leaf_rows(actual), _leaf_rows(expected), strict=True):
        if a.shape != b.shape or a.tobytes() != b.tobytes():
            out.append(name)
    return out


def _assert_bitwise_equal(actual, expected) -> None:
    mismatches = _bit_mismatches(actual, expected)
    assert not mismatches, "bit mismatches:\n" + "\n".join(mismatches)


def _assert_only_signed_zeros_differ(actual, expected) -> list[str]:
    """Every compiled/eager difference is a +0/-0 pair; return the leaves that differ."""
    differing = []
    for (name, a), (_, b) in zip(_leaf_rows(actual), _leaf_rows(expected), strict=True):
        assert a.shape == b.shape, name
        if a.tobytes() == b.tobytes():
            continue
        bad = a.view(np.uint64) != b.view(np.uint64) if a.dtype == np.float64 else a != b
        assert a.dtype == np.float64 and np.all(a[bad] == 0.0) and np.all(b[bad] == 0.0), (
            f"{name}: compiled and eager differ beyond the sign of zero")
        differing.append(name)
    return differing


def _first_line(error) -> str:
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return " | ".join(lines[:2])


def _assert_digest(tree, pinned: str, baseline, label: str) -> None:
    got = _digest(tree)
    assert got == pinned, (
        f"{label} digest {got} != pinned {pinned}; changed leaves vs unplanted: "
        f"{_bit_mismatches(tree, baseline)[:12]}"
    )


@pytest.fixture(scope="module")
def mixed():
    entry, forcing, card = _case(DEFAULT_MASTER)
    return entry, forcing, card._replace(aevp=card.aevp._replace(n_subcycles=1))


def _run(entry, forcing, card, **kwargs):
    with jax.disable_jit():
        return si3_o17_step(entry, SI3O17LiveForcing(1, *forcing[1:]), card, **kwargs)


def _probe(entry, forcing, card):
    # O6 has its own admitted eager/JIT suite; omitting it keeps this boundary small.
    return _run(entry, forcing, card, _test_return_surface_probe=True,
                _plant_drop_redistribution=True)


def _scorer_projection(result):
    step_state, trace = result
    return (step_state, trace.aevp, trace.advection_state, trace.redistribution.state,
            trace.postdyn_correction, trace.surface_stress.flx_other,
            trace.finalize.correction.state, trace.finalize.final.fluxes,
            trace.finalize.final.outputs, trace.finalize.stress)


def _full_with_pre_tail(entry, forcing, card, monkeypatch):
    """Eager full step; a spy records the ice_cor(kt,2) state before the icethd tail."""
    captured = {}
    original = o17_mod.si3_o15_step

    def spy(*args, **kwargs):
        captured["o15"] = original(*args, **kwargs)
        return captured["o15"]

    monkeypatch.setattr(o17_mod, "si3_o15_step", spy)
    result = _run(entry, forcing, card)
    monkeypatch.setattr(o17_mod, "si3_o15_step", original)
    return result, captured["o15"].state


@pytest.fixture(scope="module")
def probe_baseline(mixed):
    return _probe(*mixed)


@pytest.fixture(scope="module")
def full_baseline(mixed):
    with pytest.MonkeyPatch.context() as monkeypatch:
        return _full_with_pre_tail(*mixed, monkeypatch)


@pytest.fixture(scope="module")
def full_compiled(mixed):
    entry, forcing, card = mixed
    return jax.jit(
        lambda e, d: si3_o17_step(e, SI3O17LiveForcing(1, *d), card)
    )(entry, tuple(forcing[1:]))


def _probe_checks(entry, forcing, card, probe, baseline) -> None:
    h = card.state.nn_hls
    owned = (slice(h, -h), slice(h, -h))
    smask0 = forcing.surface.smask0
    t0 = card.constants.T0
    reference_t_bo = (  # icestp.F90:136-138 evaluated from the admitted freezing point
        (si3_eos_fzp(forcing.sss_m, kbnd=0, nn_hls=h)[owned] + t0) * smask0 + t0 * (1.0 - smask0)
    )
    np.testing.assert_array_equal(
        np.asarray(probe[8]).view(np.uint64), np.asarray(reference_t_bo).view(np.uint64),
        err_msg="t_bo icestp.F90:136-138",
    )
    _assert_digest(probe, PROBE_SHA256, baseline, "surface-boundary")


def _full_checks(entry, card, result, pre_tail, baseline) -> None:
    state, trace = result
    h = card.state.nn_hls
    owned = (slice(h, -h), slice(h, -h))
    halo = np.ones(state.ice.u_ice.shape, dtype=bool)
    halo[owned] = False
    for path, leaf in jax.tree_util.tree_flatten_with_path(result)[0]:
        if np.asarray(leaf).dtype.kind == "f":
            assert np.isfinite(np.asarray(leaf)).all(), f"non-finite {jax.tree_util.keystr(path)}"
    # icethd.F90:130-134: only owned columns are gathered; t_si is A2D(0) (ice.F90:451).
    np.testing.assert_array_equal(
        np.asarray(state.ice.t_si)[halo].view(np.uint64),
        np.asarray(entry.ice.t_si)[halo].view(np.uint64),
        err_msg="halo columns ran thermodynamics (icethd.F90:130 DO_2D(0,0,0,0))",
    )
    inactive = np.asarray(trace.postdyn_correction.at_i[owned]) <= 1.0e-10
    assert np.count_nonzero(inactive)
    np.testing.assert_array_equal(
        np.asarray(state.ice.t_si[owned])[inactive].view(np.uint64),
        np.full((np.count_nonzero(inactive), 5), card.constants.T0).view(np.uint64),
        err_msg="t_si gathered scatter (icestp.F90:488, icethd.F90:518)",
    )
    corrected = trace.finalize.correction.state  # what `CALL ice_thd` returns
    aged = nemo_source_round(
        pre_tail.oa_i[owned] + nemo_source_round(pre_tail.a_i[owned] * card.state.rDt_ice)
    )
    np.testing.assert_array_equal(
        np.asarray(corrected.oa_i[owned]).view(np.uint64), np.asarray(aged).view(np.uint64),
        err_msg="icethd.F90:185 ageing after ice_cor(kt,2)",
    )
    for name in ("a_i", "v_i", "oa_i", "t_su", "e_i"):
        field = getattr(corrected, name)
        linked = si3_lbc_lnk_orca2(field, "T", 1.0, nn_hls=h, ldfull=card.fold.nfd_full_when_absent)
        np.testing.assert_array_equal(
            np.asarray(linked).view(np.uint64), np.asarray(field).view(np.uint64),
            err_msg=f"{name} halo not refreshed (icethd.F90:188-190)",
        )
    for name, nature in (("u_ice", "U"), ("v_ice", "V")):
        field = getattr(corrected, name)
        linked = si3_lbc_lnk_orca2(field, nature, -1.0, nn_hls=h, ldfull=card.fold.nfd_full_when_absent)
        np.testing.assert_array_equal(
            np.asarray(linked).view(np.uint64), np.asarray(field).view(np.uint64),
            err_msg=f"{name} halo not refreshed (icethd.F90:201)",
        )
    np.testing.assert_array_equal(  # iceupdate.F90:189-194 carry into the next step
        np.asarray(state.ice.snwice_mass).view(np.uint64),
        np.asarray(trace.finalize.final.outputs.snwice_mass).view(np.uint64),
        err_msg="snow-mass carry",
    )
    _assert_digest(_scorer_projection(result), FULL_SHA256, baseline, "full-step")


def test_o17_typed_pytree_boundary_and_explicit_card() -> None:
    assert SI3O17LiveForcing._fields[:4] == ("kt", "sss_m", "sst_m", "rn_amax_2d")
    assert not (
        {"redistribution_state", "correction_state", "thermo", "fluxes"}
        & set(SI3O17LiveForcing._fields)
    )
    assert SI3O17StepState._fields == ("ice", "aevp")
    assert SI3O17Card._fields[-1] == "core2_precipitation_closure"
    assert callable(si3_o17_step)


def test_o17_card_refuses_core2_closure_before_any_inference() -> None:
    card = SI3O17Card(*([None] * (len(SI3O17Card._fields) - 1)), True)
    with pytest.raises(ValueError, match="core2_precipitation_closure=False"):
        build_si3_orca2_o17_config(card)


def test_o17_call_order_preflight_and_swap_drop_plants() -> None:
    report = preflight(REPO)
    assert report["call_order"] == list(ORDER)
    assert report["steps"] == 100
    assert any("o4_orca2_prather_gate" in name for name in report["imported_component_tools"])
    for plant in ("swap-rheology-advection", "drop-redistribution"):
        with pytest.raises(SystemExit, match="STATUS PLANT-FIRED"):
            preflight(REPO, plant)


def test_o17_public_production_dispatch_calls_composition(monkeypatch) -> None:
    state_cfg = SI3ORCA2StateConfig.for_orca2_sas_ice(ln_icedyn=True, ln_icethd=True)
    card = object()
    config = SeaIceConfig(
        dynamics="none",
        transport="none",
        thermo_scheme="si3_orca2_o2",
        ice_constants=state_cfg.ice_constants,
        si3_orca2=state_cfg,
        core2_precipitation_closure=False,
        si3_o17=card,
    )
    entry = SI3O17StepState(None, None)
    forcing = SI3O17LiveForcing(1, *([None] * (len(SI3O17LiveForcing._fields) - 1)))
    expected = (entry, object())
    calls = []

    def planted(step_entry, live_forcing, actual_card):
        calls.append((step_entry, live_forcing, actual_card))
        return expected

    monkeypatch.setattr(sea_ice, "si3_o17_step", planted)
    actual = sea_ice.step_sea_ice(entry, forcing, None, None, None, config, 0.0, state_cfg.rDt_ice)
    assert actual == expected
    assert calls == [(entry, forcing, card)]


def test_o17_record_reader_is_self_describing_and_rejects_truncation(tmp_path) -> None:
    assert scorer.VALID_LABELS == ("INDEPENDENT", "GIVEN-NEMO-ENTRY")
    record = tmp_path / "o17_kind22_step_entry_step0001_rank0000.bin"
    field = b"v_i".ljust(32, b" ")
    end = b"__END__".ljust(32, b" ")
    payload = (
        b"O17S0001" + struct.pack("=i8i", 1, 22, 1, 0, 6, 5, 5, 10, 5)
        + field + struct.pack("=i4i", 2, 2, 3, 0, 0)
        + np.arange(6, dtype=np.float64).reshape(2, 3, order="F").tobytes(order="F")
        + end + struct.pack("=i4i", -1, 0, 0, 0, 0)
    )
    record.write_bytes(payload)
    parsed = scorer.read_record(record)
    assert parsed["meta"]["kind"] == 22
    np.testing.assert_array_equal(parsed["fields"]["v_i"], np.arange(6).reshape(2, 3, order="F"))
    record.write_bytes(payload[:143])  # one byte short of v_i's six float64 values
    with pytest.raises(ValueError, match="truncated"):
        scorer.read_record(record)


def test_o17_record_layout_is_source_derived_at_the_real_orca2_extent() -> None:
    layout = budget.record_layout()
    shapes = {kind: {name: shape for name, shape, _ in fields} for kind, fields in layout.items()}
    assert shapes[22]["v_i"] == (184, 152, 5)  # ice.F90:349 (jpi,jpj,jpl); Ni0glo=180+2*2
    assert shapes[22]["t_si"] == (180, 148, 5)  # ice.F90:451 A2D(0)
    assert shapes[22]["snwice_fmass"] == (184, 152)  # sbc_ice.F90:125 (jpi,jpj)
    assert shapes[22]["wfx_dyn"] == (180, 148)  # ice.F90:400-410 A2D(0)
    assert shapes[33]["emp"] == (184, 152) and shapes[33]["qsr"] == (180, 148)  # sbc_oce.F90:190,202
    assert shapes[32]["wndi"] == (180, 148)  # sbcblk.F90 blk_ice_1 DIMENSION(A2D(0))
    assert [shapes[34][n] for n in ("tmod_io", "utau_oce", "vtau_oce")] == [
        (184, 152), (180, 148), (180, 148)]  # iceupdate.F90:55
    report = budget.size_report()
    assert report["record_bytes_by_kind"]["34"] == 44 + 3 * 52 + 8 * (184 * 152 + 2 * 180 * 148) + 52
    assert report["o17_record_bytes_exact_max"] == 100 * sum(report["record_bytes_by_kind"].values())
    # One budget: the wrapper and master read the calculator's --shell line; the gate imports it.
    for path in ("nemo_si3_o17_acquisition/run.sh", "nemo_si3_o8_o16_master_acquisition/run.sh"):
        text = (REPO / "scripts/validate/ocean_fidelity/testcases" / path).read_text()
        assert "nemo_si3_o17_record_budget.py\" --shell" in text
    gate = (REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py").read_text()
    assert "o17_size_report(manifest[\"estimated_master_bytes\"])" in gate


def test_o17_acquisition_has_no_hard_coded_pass_and_scores_real_records() -> None:
    master = (REPO / "scripts/validate/ocean_fidelity/testcases/"
              "nemo_si3_o8_o16_master_acquisition/run.sh").read_text()
    o17 = (REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_acquisition/run.sh").read_text()
    assert '"status": "PASS"' not in master  # no literal PASS report anywhere
    assert "--self-test" not in o17
    assert "nemo_si3_o17_synthetic_record.py" in o17
    assert o17.count('"$master" --run "$evidence"') == 1
    assert 'touch "$EVIDENCE/RUN_COMPLETE"' in master


def test_o17_kt_enters_only_through_prather_sweep_parity() -> None:
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(o17_mod.si3_o17_step))
    uses = [node for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "kt"]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "id", "") == "si3_dyn_orca2"]
    passed = [kw.value for call in calls for kw in call.keywords if kw.arg == "kt"]
    assert len(uses) == len(passed) == 2 and all(u in passed for u in uses)
    assert [scorer.kt_program(kt, 1) for kt in (1, 2, 3, 4, 99, 100)] == [1, 2, 1, 2, 1, 2]


def test_o17_surface_boundary_eager_jit_and_code_revert_plants(mixed, probe_baseline, monkeypatch) -> None:
    entry, forcing, card = mixed
    _probe_checks(entry, forcing, card, probe_baseline, probe_baseline)
    compiled = jax.jit(
        lambda e, d: si3_o17_step(e, SI3O17LiveForcing(1, *d), card,
                                  _test_return_surface_probe=True, _plant_drop_redistribution=True)
    )(entry, tuple(forcing[1:]))
    # XLA turns x*convert(mask) into a select: +0 where eager/NEMO keep -0.
    assert "[14].stress2_t" in _assert_only_signed_zeros_differ(compiled, probe_baseline)
    h = card.state.nn_hls
    owned = (slice(h, -h), slice(h, -h))

    def eos_reverted(sss, **_):  # drop eos_fzp: t_bo = rt0 everywhere
        return jnp.zeros_like(sss)

    original_aevp = o17_mod.si3_orca2_aevp_solver

    def u_ice_b_reverted(*args, **kwargs):  # store_fields velocities not passed
        kwargs.update(u_ice_b=jnp.zeros_like(kwargs["u_ice_b"]), v_ice_b=jnp.zeros_like(kwargs["v_ice_b"]))
        return original_aevp(*args, **kwargs)

    original_b2, original_other = o17_mod.si3_blk_ice_2, o17_mod.si3_ice_flx_other
    pre_dynamics_a_i = entry.ice.a_i[owned]

    def a_i_b_reverted(original):  # second store_fields (icestp.F90:184) skipped
        def call(*args, **kwargs):
            kwargs["at_i_b"] = jnp.sum(pre_dynamics_a_i, axis=-1)
            if "a_i_b" in kwargs:
                kwargs["a_i_b"] = pre_dynamics_a_i
            return original(*args, **kwargs)
        return call

    plants = {
        "t_bo": {"si3_eos_fzp": eos_reverted},
        "u_ice_b": {"si3_orca2_aevp_solver": u_ice_b_reverted},
        "a_i_b": {"si3_blk_ice_2": a_i_b_reverted(original_b2),
                  "si3_ice_flx_other": a_i_b_reverted(original_other)},
    }
    for label, patches in plants.items():
        with monkeypatch.context() as patched:
            for name, replacement in patches.items():
                patched.setattr(o17_mod, name, replacement)
            planted = _probe(entry, forcing, card)
        with pytest.raises(AssertionError) as fired:
            _probe_checks(entry, forcing, card, planted, probe_baseline)
        print(f"PLANT {label}: {_first_line(fired.value)}")


def test_o17_full_step_production_invariants(mixed, full_baseline) -> None:
    entry, forcing, card = mixed
    result, pre_tail = full_baseline
    _full_checks(entry, card, result, pre_tail, _scorer_projection(result))
    with pytest.raises(ValueError, match="ln_ice_embd=F"):
        si3_o17_step(entry, SI3O17LiveForcing(1, *forcing[1:]), card._replace(ln_ice_embd=True))


def test_o17_full_step_compiled_differs_from_eager_only_in_signed_zeros(full_baseline, full_compiled) -> None:
    # 7 x 7 card only: at 184 x 152 the eager/jit probe also measures value differences.
    differing = _assert_only_signed_zeros_differ(full_compiled, full_baseline[0])
    print("COMPILED/EAGER signed-zero-only leaves:", differing)
    assert "[0].aevp.stress2_t" in differing


@pytest.mark.xfail(strict=True, reason=(
    "compiled XLA rewrites x*convert(mask) into select and stores +0 where NEMO and eager "
    "store -0 (dynamics.py zmsk/ice_mask); kernel barrier is a pending user decision"))
def test_o17_full_step_compiled_equals_eager_bitwise(full_baseline, full_compiled) -> None:
    _assert_bitwise_equal(full_compiled, full_baseline[0])


@pytest.mark.parametrize("plant", ["thermodynamic_tail", "icethd_tail_lbc", "owned_gather"])
def test_o17_full_step_code_revert_plants_fail(plant, mixed, full_baseline, monkeypatch) -> None:
    entry, forcing, card = mixed
    baseline = _scorer_projection(full_baseline[0])
    if plant == "thermodynamic_tail":  # skip the icethd.F90:183-201 tail call
        monkeypatch.setattr(o17_mod, "_thermodynamic_tail", lambda state, card: state)
    elif plant == "icethd_tail_lbc":  # skip the tail's lbc_lnk (icethd.F90:188-190,201)
        monkeypatch.setattr(o17_mod, "si3_lbc_lnk_orca2", lambda field, *a, **k: field)
    else:  # gather over the full jpi x jpj grid instead of DO_2D(0,0,0,0)
        monkeypatch.setattr(o17_mod, "_owned_mask", lambda shape, h: jnp.ones(shape, dtype=bool))
    result, pre_tail = _full_with_pre_tail(entry, forcing, card, monkeypatch)
    with pytest.raises(AssertionError) as fired:
        _full_checks(entry, card, result, pre_tail, baseline)
    print(f"PLANT {plant}: {_first_line(fired.value)}")


def test_o17_scorer_on_writer_shaped_records_and_plants(mixed, tmp_path, monkeypatch) -> None:
    """Real scorer on a model-built, writer-extent record (2 steps), then plants."""
    entry, forcing, card = mixed
    runner = synthetic.memoized(scorer.make_runner(card, "eager"))
    synthetic.generate(tmp_path / "rec", 2, entry, forcing, card, runner)
    loads = []
    original_load = scorer.load_step
    monkeypatch.setattr(scorer, "load_step", lambda *a: loads.append(a[1:]) or original_load(*a))
    kwargs = dict(execution="eager", expect_steps=2, card=card, runner=runner)
    labels = list(scorer.VALID_LABELS)
    report = scorer.score(tmp_path / "rec", labels, **kwargs)
    assert report["status"] == "PASS", report["labels"]["GIVEN-NEMO-ENTRY"]["steps"]
    assert loads == [(1, 0), (2, 0)]  # streamed: one step's records at a time
    records = scorer.record_paths(tmp_path / "rec/record_100step/o17_master_records")
    t_si = scorer.read_record(records[(1, 0)][22])["fields"]["t_si"]
    assert t_si.shape == (3, 3, 5)  # A2D(0) record extent, padded by the scorer

    # Padding cannot reach an owned output: a planted halo t_si gives the same owned record.
    step1 = scorer.load_step(records[(1, 0)], 1, 0)
    given = scorer.entry_state(step1, card)
    halo = np.ones(given.ice.t_si.shape[:2], bool)
    halo[2:-2, 2:-2] = False
    padded = given._replace(ice=given.ice._replace(
        t_si=jnp.where(jnp.asarray(halo)[..., None], 123.0, given.ice.t_si)))
    out, trace = scorer.make_runner(card, "eager")(padded, scorer.live_forcing(step1, card))
    assert scorer.score_step(out, trace, step1, card, "GIVEN-NEMO-ENTRY")["pass"]

    def plant_record(kind, name, mutate):
        path = records[(2, 0)][kind]
        rec = scorer.read_record(path)
        mutate(rec["fields"][name])
        synthetic.write_record(path, kind, 2, 0, (7, 7, 5, 10, 5), list(rec["fields"].items()))

    import shutil
    pristine = tmp_path / "pristine"
    shutil.copytree(tmp_path / "rec", pristine)

    def restore():
        shutil.rmtree(tmp_path / "rec")
        shutil.copytree(pristine, tmp_path / "rec")

    def ulp(value):
        value.flat[0] = np.nextafter(value.flat[0], np.inf)

    def signed_zero(value):
        index = tuple(np.argwhere((value == 0.0) & ~np.signbit(value))[0])
        value[index] = -0.0

    def nan(value):
        value.flat[0] = np.nan

    for kind, name, mutate, label in ((28, "t_si", ulp, "GIVEN-NEMO-ENTRY"),
                                      (28, "wfx_bog", signed_zero, "GIVEN-NEMO-ENTRY"),
                                      (28, "qns", nan, "INDEPENDENT")):
        plant_record(kind, name, mutate)
        planted = scorer.score(tmp_path / "rec", labels, **kwargs)
        assert planted["status"] == "FAIL"
        assert planted["labels"][label]["status"] == "FAIL"
        assert f"step_exit.{name}" in planted["labels"][label]["steps"]["2:0"]["failed_fields"]
        restore()

    # Item 1: no padding of A2D(0) entries / no owned alignment -> cannot pass.
    with monkeypatch.context() as patched:
        patched.setattr(scorer, "_as_full", lambda value, full, h: jnp.asarray(value))
        with pytest.raises((TypeError, ValueError)):
            scorer.score(tmp_path / "rec", labels, **kwargs)
    original_compare = scorer.compare
    with monkeypatch.context() as patched:
        patched.setattr(scorer, "compare", lambda a, e, full, h: original_compare(a, e, (0, 0), h))
        assert scorer.score(tmp_path / "rec", labels, **kwargs)["status"] == "FAIL"
    # Item 2: kind 27 is the state `ice_thd` returns; a wrong boundary fails.
    original_fields = scorer.scored_fields

    def wrong_boundary(out, trace):
        return [(k, n, getattr(trace.advection_state, n) if k == 27 else v)
                for k, n, v in original_fields(out, trace)]

    with monkeypatch.context() as patched:
        patched.setattr(scorer, "scored_fields", wrong_boundary)
        assert scorer.score(tmp_path / "rec", labels, **kwargs)["status"] == "FAIL"
