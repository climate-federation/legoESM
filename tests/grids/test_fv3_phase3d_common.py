"""Gates for the shared 3-D phase helpers (``fv3_phase3d_common``).

Every function in that module is a GATE, so every test here has the same
shape: one call that must PASS and one that must RAISE.  A gate tested
only on the passing side is a gate that cannot fail, which is the defect
class this campaign has hit most often.

Scope note.  ``CSW_OUT_LIKE``'s agreement with the NumPy spec's private
``_out_like`` is ALREADY asserted by
``test_fv3_cgrid_phase_3d.py::test_csw_out_like_matches_the_spec``, and
that assertion reads ``fv3_cgrid_phase_3d.CSW_OUT_LIKE``, which is now
this module's object re-exported.  It is not duplicated here; what IS
asserted here is the re-export identity, which is the thing the move
could have broken.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core import fv3_phase3d_common as common
from legoesm.core.fv3_native_state_3d import field_shape

jax.config.update("jax_enable_x64", True)

N, NG, KM = 12, 3, 4


class _Ctx:
    """The two attributes ``validate_stacked`` reads off a real ctx."""

    n, ng = N, NG


def _stacked(name, *, n=N, ng=NG, km=KM):
    return jnp.zeros((6,) + field_shape(name, n, ng, km), dtype=jnp.float64)


# ---------------------------------------------------------------- dtype

def test_require_uniform_accepts_f64_and_skips_none():
    common.require_f64_jax("t", {"a": jnp.zeros(3, dtype=jnp.float64),
                                 "b": None})


def test_require_uniform_accepts_uniform_f32():
    # The gate is now a UNIFORMITY gate (coarse fp32/fp64 policy,
    # 2026-08-28): a uniformly-f32 phase is ACCEPTED (was rejected under
    # the old strict-f64 gate).
    common.require_uniform_float_jax(
        "t", {"a": jnp.zeros(3, dtype=jnp.float32),
              "b": jnp.zeros(2, dtype=jnp.float32)})


def test_require_uniform_rejects_mixed_dtypes():
    # The real hazard the gate now catches: an f64 operand (a leaked
    # metric/workspace) mixed into an f32 phase -> silent promotion.
    with pytest.raises(TypeError, match="MIXED float dtypes"):
        common.require_uniform_float_jax(
            "t", {"a": jnp.zeros(3, dtype=jnp.float32),
                  "b": jnp.zeros(3, dtype=jnp.float64)})


def test_require_uniform_rejects_non_float():
    with pytest.raises(TypeError, match="float32 or float64"):
        common.require_uniform_float_jax(
            "t", {"a": jnp.zeros(3, dtype=jnp.int32)})


def test_require_uniform_skips_only_WEAK_0d_scalars():
    # A WEAK-typed 0-dim scalar (a python-float timestep dt / damping
    # coeff kgb) is weak-promoting and NOT part of the field uniformity
    # invariant: it must NOT trip the gate among f32 fields.
    common.require_uniform_float_jax(
        "t", {"delp": jnp.zeros(3, dtype=jnp.float32),
              "dt": jnp.asarray(120.0),          # weak f64 (python float)
              "kgb": jnp.asarray(0.15)})         # weak f64
    # ...but a STRONG-f64 0-dim (an f64 constant that "went strong") among
    # f32 fields STILL trips it -- the blind spot the wholesale 0-dim skip
    # left, now closed (codex+GLM+Claude, increment 2).
    with pytest.raises(TypeError, match="MIXED float dtypes"):
        common.require_uniform_float_jax(
            "t", {"delp": jnp.zeros(3, dtype=jnp.float32),
                  "coeff": jnp.asarray(0.15, dtype=jnp.float64)})  # STRONG
    # ...and a 1-D f64 field among f32 fields STILL trips it.
    with pytest.raises(TypeError, match="MIXED float dtypes"):
        common.require_uniform_float_jax(
            "t", {"delp": jnp.zeros(3, dtype=jnp.float32),
                  "metric": jnp.zeros(3, dtype=jnp.float64)})


def test_require_f64_alias_points_at_uniform_gate():
    # Back-compat: the historical name is an alias, so the ~40 call sites
    # and the private copies keep working.
    assert common.require_f64_jax is common.require_uniform_float_jax


# ----------------------------------------------------------------- bool

def test_require_bool_accepts_a_bool():
    common.require_bool("t", "hydrostatic", True)


@pytest.mark.parametrize("bad", [1, 0, "yes", None, np.bool_(True)])
def test_require_bool_rejects_truthy_non_bools(bad):
    # np.bool_ is included deliberately: it is truthy and prints as True,
    # so a caller passing one would otherwise select a branch silently.
    with pytest.raises(TypeError, match="must be a bool"):
        common.require_bool("t", "hydrostatic", bad)


# ------------------------------------------------------------------- km

def test_require_km_returns_a_python_int():
    got = common.require_km("t", np.int64(5))
    assert got == 5 and type(got) is int


@pytest.mark.parametrize("bad", [4.0, "4", True, None])
def test_require_km_rejects_non_ints(bad):
    with pytest.raises(TypeError, match="must be a Python int"):
        common.require_km("t", bad)


def test_require_km_rejects_zero():
    with pytest.raises(ValueError, match="must be >= 1"):
        common.require_km("t", 0)


def test_require_km_rejects_a_tracer():
    # km is a loop trip count AND an array extent; a traced km would fail
    # much deeper, inside a shape computation.
    with pytest.raises(TypeError, match="must be a Python int"):
        jax.jit(lambda k: common.require_km("t", k))(3)


# ----------------------------------------------------------------- nord

def test_require_nord_accepts_an_integral_float():
    assert common.require_nord("t", "nord", 2.0) == 2


def test_require_nord_names_the_offending_parameter():
    # The name argument is the reason this function was promoted with a
    # superset signature: a module with two damping orders must be able
    # to say WHICH one is wrong.
    with pytest.raises(ValueError, match="nord_w must be an integral"):
        common.require_nord("t", "nord_w", 2.7)


def test_require_nord_rejects_negative():
    with pytest.raises(ValueError, match="must be >= 0"):
        common.require_nord("t", "nord", -1)


def test_require_nord_rejects_bool():
    with pytest.raises(ValueError, match="must be an integral"):
        common.require_nord("t", "nord", True)


# ------------------------------------------------------- validate_stacked

def test_validate_stacked_accepts_declared_shapes():
    cont = {nm: _stacked(nm) for nm in ("delp", "u", "v")}
    common.validate_stacked("t", cont, _Ctx(), KM, ("delp", "u", "v"),
                            what="states")


def test_pkc_maps_to_the_interface_shape_the_c_stage_allocates():
    """`pkc` is the C-stage FULL interface pressure, (m_a, m_a, km+1) at
    fv3_native_cgrid_phase_3d.py:253 -- the shape field_shape declares
    for pk/gz.  The NH D-grid tail validates csw_press through this
    table, and the lookup raised "unknown field 'pkc'" without it."""
    assert common.CSW_OUT_LIKE["pkc"] == "pk"
    cont = {"pkc": _stacked("pk")}
    common.validate_stacked("t", cont, _Ctx(), KM, ("pkc",),
                            what="csw_press")
    bad = {"pkc": _stacked("delp")}          # (m_a, m_a, km), not km+1
    with pytest.raises(ValueError, match=r"csw_press\['pkc'\]"):
        common.validate_stacked("t", bad, _Ctx(), KM, ("pkc",),
                                what="csw_press")


def test_validate_stacked_maps_csw_names_through_the_table():
    # `delpc` shares `delp`'s shape and `divg_d` shares `divgd`'s; the
    # mapping is what makes the C-grid outputs checkable at all.
    cont = {"delpc": _stacked("delp"), "divg_d": _stacked("divgd")}
    common.validate_stacked("t", cont, _Ctx(), KM, ("delpc", "divg_d"),
                            what="csw_outs")


def test_validate_stacked_rejects_a_stagger_slip():
    # (m_a, m_b) vs (m_b, m_a) BROADCASTS later instead of raising --
    # the whole reason this gate exists.
    cont = {"u": _stacked("v")}
    with pytest.raises(ValueError, match=r"states\['u'\] has shape"):
        common.validate_stacked("t", cont, _Ctx(), KM, ("u",), what="states")


def test_validate_stacked_rejects_the_numpy_lane_container():
    with pytest.raises(TypeError, match="face-stacked dict"):
        common.validate_stacked("t", [{}] * 6, _Ctx(), KM, ("delp",),
                                what="states")


def test_validate_stacked_names_the_missing_key():
    with pytest.raises(KeyError, match="missing"):
        common.validate_stacked("t", {"delp": _stacked("delp")}, _Ctx(),
                                KM, ("delp", "pt"), what="states")


# ---------------------------------------------------------- stack_levels

def test_stack_levels_puts_km_at_axis_2():
    per_level = [jnp.zeros((5, 7), dtype=jnp.float64) for _ in range(KM)]
    out = common.stack_levels("t", "q", per_level)
    assert out.shape == (5, 7, KM)


def test_stack_levels_puts_km_at_axis_2_for_a_slotted_slab():
    # The reason the axis is 2 and not -1: an allflux slab (i, j, slot)
    # must become (i, j, km, slot), the oracle's allflux_x(i,j,k,iq).
    per_level = [jnp.zeros((5, 7, 5), dtype=jnp.float64) for _ in range(KM)]
    out = common.stack_levels("t", "allflux_x", per_level)
    assert out.shape == (5, 7, KM, 5)


def test_stack_levels_level0_reference_mode_catches_a_late_level():
    per_level = [jnp.zeros((5, 7), dtype=jnp.float64) for _ in range(KM)]
    per_level[2] = jnp.zeros((5, 8), dtype=jnp.float64)
    with pytest.raises(ValueError, match=r"level 2 output 'q'.*level 0 has"):
        common.stack_levels("t", "q", per_level)


def test_stack_levels_want2d_mode_catches_a_wrong_level_zero():
    # The strict mode's whole advantage: level 0 is checked too, so a
    # uniformly wrong stack is caught instead of being taken as the
    # reference.
    per_level = [jnp.zeros((5, 8), dtype=jnp.float64) for _ in range(KM)]
    with pytest.raises(ValueError,
                       match=r"level 0 output 'q'.*container expects"):
        common.stack_levels("t", "q", per_level, (5, 7))


def test_stack_levels_want2d_mode_accepts_the_declared_shape():
    per_level = [jnp.zeros((5, 7), dtype=jnp.float64) for _ in range(KM)]
    assert common.stack_levels("t", "q", per_level, (5, 7)).shape == (5, 7, KM)


def test_stack_levels_rejects_an_empty_level_list():
    with pytest.raises(ValueError, match="empty per-level list"):
        common.stack_levels("t", "q", [])


def test_stack_levels_is_a_pure_index_copy_under_jit():
    # jnp.stack has no `x*y + z` for XLA to contract into an FMA, so the
    # jit and eager lowerings must agree BITWISE.  This is the premise
    # the phase modules' assembly-step gates rest on.
    per_level = [jnp.asarray(np.random.default_rng(k).normal(size=(5, 7)))
                 for k in range(KM)]
    eager = common.stack_levels("t", "q", per_level)
    jitted = jax.jit(lambda p: common.stack_levels("t", "q", list(p))
                     )(tuple(per_level))
    assert np.array_equal(np.asarray(eager), np.asarray(jitted))


# ----------------------------------------------------------- stack_faces

def test_stack_faces_stacks_the_face_axis_first():
    per_face = [{"u": jnp.zeros((5, 7, KM), dtype=jnp.float64),
                 "v": jnp.zeros((7, 5, KM), dtype=jnp.float64)}
                for _ in range(6)]
    out = common.stack_faces("t", per_face)
    assert out["u"].shape == (6, 5, 7, KM)
    assert out["v"].shape == (6, 7, 5, KM)


def test_stack_faces_rejects_a_per_face_key_split():
    per_face = [{"u": jnp.zeros((2, 2), dtype=jnp.float64)} for _ in range(6)]
    per_face[3] = {"u": jnp.zeros((2, 2), dtype=jnp.float64),
                   "w": jnp.zeros((2, 2), dtype=jnp.float64)}
    with pytest.raises(KeyError, match="face 4 produced keys"):
        common.stack_faces("t", per_face)


def test_stack_faces_rejects_a_wrong_face_count():
    per_face = [{"u": jnp.zeros((2, 2), dtype=jnp.float64)} for _ in range(5)]
    with pytest.raises(ValueError, match="expected 6 per-face dicts"):
        common.stack_faces("t", per_face)


# ------------------------------------------------------ the promotion itself

def test_the_phase_modules_use_this_module_and_keep_no_private_copy():
    """The point of the promotion: one definition, no sixth copy.

    Both existing 3-D phase modules must resolve the helpers to THIS
    module's objects, and must no longer carry a private copy under the
    old underscore name -- otherwise a future edit fixes one copy and
    leaves the others, which is exactly the state this replaced.
    """
    from legoesm.core import fv3_cgrid_phase_3d as cg
    from legoesm.core import fv3_dsw_phase_3d as ds

    shared = ("require_f64_jax", "require_bool", "require_km",
              "require_nord", "validate_stacked", "stack_levels",
              "stack_faces")
    for mod in (cg, ds):
        for nm in shared:
            assert getattr(mod, nm) is getattr(common, nm), (mod.__name__, nm)
            assert not hasattr(mod, "_" + nm), (
                f"{mod.__name__} still carries a private {nm} copy")


def test_csw_out_like_is_the_same_object_the_cgrid_module_exports():
    from legoesm.core import fv3_cgrid_phase_3d as cg

    assert cg.CSW_OUT_LIKE is common.CSW_OUT_LIKE
    assert "CSW_OUT_LIKE" in cg.__all__


# ------------------------------------------- build_batched_gs (C2a view)
#
# Gates for the face-batched context view the vmap arms consume.  Same
# shape as everything above: one PASS and one RAISE per gate, and the
# raising side is what makes the common-mode assert non-vacuous -- a
# silent broadcast of face 1's statics would be plausible wrong physics,
# not an error message.

from legoesm.core.fv3_duo_sw_core import GridFlags  # noqa: E402


class _BatchCtx:
    """The three attributes ``build_batched_gs`` touches on a real ctx
    (``gs6``, ``flags6``, and a settable cache attribute)."""

    def __init__(self, gs6, flags6):
        self.gs6 = tuple(gs6)
        self.flags6 = tuple(flags6)


def _fake_gs6():
    """Six tiny per-face metric dicts, DISTINCT per face so a face
    mix-up in the stack cannot hide behind symmetry."""
    return [
        {"dx": jnp.asarray(np.arange(12.0).reshape(3, 4) + 100.0 * t),
         "dy": jnp.asarray(np.arange(20.0).reshape(4, 5) + 100.0 * t)}
        for t in range(6)
    ]


def _fake_flags6(**overrides):
    base = dict(bounded_domain=True, grid_type=0)
    base.update(overrides)
    return [GridFlags(da_min=0.1 + 0.01 * t, da_min_c=0.2 + 0.01 * t,
                      **base) for t in range(6)]


def test_batched_gs_roundtrip_is_exact():
    """Unstacking the batched view equals the per-face dicts EXACTLY --
    ``jnp.stack`` is a pure index copy, so bitwise, not allclose."""
    gs6 = _fake_gs6()
    view = common.build_batched_gs(_BatchCtx(gs6, _fake_flags6()))
    assert set(view["gs"]) == {"dx", "dy"}
    assert view["unstacked_keys"] == ()
    for key in ("dx", "dy"):
        assert view["gs"][key].shape == (6,) + gs6[0][key].shape
        for t in range(6):
            assert np.array_equal(np.asarray(view["gs"][key][t]),
                                  np.asarray(gs6[t][key])), (key, t)


def test_batched_gs_da_min_arrays_are_per_face():
    """The two per-face flag fields batch as (6,) f64 -- and DIFFERING
    per-face values must NOT trip the common-mode gate (the control that
    keeps that gate from over-firing)."""
    flags6 = _fake_flags6()
    view = common.build_batched_gs(_BatchCtx(_fake_gs6(), flags6))
    assert view["da_min6"].shape == (6,)
    assert view["da_min_c6"].shape == (6,)
    assert view["da_min6"].dtype == jnp.float64
    for t in range(6):
        assert float(view["da_min6"][t]) == flags6[t].da_min
        assert float(view["da_min_c6"][t]) == flags6[t].da_min_c
    # the shared statics carry every OTHER GridFlags field, once
    assert set(view["flags"]) == (set(GridFlags._fields)
                                  - set(common.PER_FACE_FLAG_FIELDS))
    assert view["flags"]["bounded_domain"] is True


def test_batched_gs_refuses_a_per_face_key_split():
    gs6 = _fake_gs6()
    del gs6[3]["dy"]
    with pytest.raises(KeyError, match=r"gs6\[3\].*dy"):
        common.build_batched_gs(_BatchCtx(gs6, _fake_flags6()))


def test_batched_gs_common_mode_assert_fires_and_names_the_field():
    """Non-vacuity of the GLM guard: ONE differing static flag on ONE
    face must raise, naming the field -- not broadcast face 1's value."""
    flags6 = _fake_flags6()
    flags6[4] = flags6[4]._replace(grid_type=4)
    with pytest.raises(ValueError, match="grid_type"):
        common.build_batched_gs(_BatchCtx(_fake_gs6(), flags6))


def test_batched_gs_skips_a_shape_mismatched_key_and_records_it():
    """A key whose shape differs across faces cannot share a batch axis;
    it stays loop-path-only and is NAMED in ``unstacked_keys`` (a
    vmapped kernel needing it then fails loudly with a KeyError, never
    with broadcast face-1 values)."""
    gs6 = _fake_gs6()
    gs6[2]["dx"] = jnp.zeros((5, 4), dtype=jnp.float64)
    view = common.build_batched_gs(_BatchCtx(gs6, _fake_flags6()))
    assert view["unstacked_keys"] == ("dx",)
    assert "dx" not in view["gs"]
    assert "dy" in view["gs"]


def test_batched_gs_is_cached_by_identity_of_gs6_and_flags6():
    """Built once per context; a clone that swaps in a FRESH flags6
    tuple must get a FRESH view (the stale-cache hazard the identity
    check exists for)."""
    ctx = _BatchCtx(_fake_gs6(), _fake_flags6())
    v1 = common.build_batched_gs(ctx)
    assert common.build_batched_gs(ctx) is v1
    ctx.flags6 = tuple(_fake_flags6())   # equal values, new tuple
    v3 = common.build_batched_gs(ctx)
    assert v3 is not v1
