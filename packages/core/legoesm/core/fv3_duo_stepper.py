"""FV3 duo-grid km=1 acoustic stepper -- JAX lane (the composition).

Functional, jit-compilable, differentiable twin of
``legoesm.core.fv3_native_duo_stepper``'s km=1 shallow-water assembly.
This module introduces NO numerics of its own: every arithmetic
operation happens inside a kernel that is already ported and gated
elsewhere.  What it contributes is the ORDER -- which kernel runs when,
which exchange fires where, and which halo is deliberately NOT
refreshed.

Authority chain (``docs/atmosphere/fv3_duo_jax_lane_strategy.md`` §2):
the NumPy lane ``fv3_native_duo_stepper`` is the SPECIFICATION (hop B);
the pinned Fortran
``/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean``
(``model/dyn_core.F90``, md5 ``e5a5fab9…``, 3128 lines) is quoted only
where a line number is needed to say WHY a step exists.  Every citation
below was re-derived with ``awk``/``sed`` against that tree (R7), not
copied from a comment.

The pieces this module composes -- none of them re-implemented here:

* ``legoesm.core.fv3_duo_sw_core`` -- ``c_sw``, ``d_sw1_duo`` …
  ``d_sw6_duo``;
* ``legoesm.grids.fv3_duo_halos`` -- ``ext_scalar_sixface``,
  ``ext_vector_dgrid_sixface``, ``ext_vector_cgrid_sixface`` and BOTH
  duo barriers (``average_allflux_shared_edges`` = barrier 1,
  ``dyn_core.F90:872``; ``average_shared_edge_bgrid`` = barrier 2,
  ``:984``);
* ``legoesm.core.fv3_pgrad`` -- ``geopk``, ``p_grad_c``,
  ``one_grad_p``.

The step, in the oracle's own order
-----------------------------------
``full_acoustic_step_sixface`` is four things, and the fourth is the one
that is easy to get wrong while looking right:

1. :func:`acoustic_step_sixface` -- entry exchanges, then
   ``c_sw`` (``:489``) -> ``geopk``/``p_grad_c`` (``:533``/``:629``) ->
   the post-``p_grad_c`` duo exchanges (``:652`` divgd, ``:655``
   uc/vc) -> ``d_sw1`` -> BARRIER 1 -> ``d_sw2`` -> ``d_sw3`` ->
   BARRIER 2 -> the inline KE assembly -> ``d_sw4`` -> ``d_sw5`` ->
   ``d_sw6``;
2. the post-step ``delp``/``pt`` halo refresh --
   ``ext_scalar(delp)``/``ext_scalar(pt)`` at ``dyn_core.F90:1336-1337``;
3. per face ``geopk`` on the D-grid call (``:1401``), the external-mode
   filter ``divg2`` (``:1310-1330``) and ``one_grad_p`` (``:1531``);
4. **NO post-step vector refresh.**  ``dyn_core.F90:1332-1338``
   refreshes ONLY ``delp`` and ``pt``; the D-wind ``ext_vector`` runs at
   the NEXT step's entry (``:468-472``).  The returned state's D-wind
   halos are therefore STALE-BY-ONE, and a twin that "helpfully"
   refreshes them here is a DIVERGENCE that will look more correct than
   the faithful version.  ``test_fv3_duo_stepper.py``'s
   ``test_returned_d_wind_halo_is_stale_by_one`` pins it, together with
   the non-vacuity check that a refresh WOULD have changed those cells.

The entry A-scalar exchange is gated on the first substep of an outer
step -- ``dyn_core.F90:432-439``, ``if ( it==1 )`` at ``:432`` -- while
``ext_vector`` at ``:471``, the post-``p_grad_c`` pair and the tail
refresh fire on EVERY substep.  :func:`advance_duo_outer_step` is where
that cadence lives (``entry_ascalar=(it == 0)``).

State layout (the binding contract, ``fv3_duo_halos`` docstring §1-§6)
---------------------------------------------------------------------
The public state is a dict of FACE-STACKED arrays, never a list of six
per-face dicts::

    {"delp": (6, ma, ma), "pt": (6, ma, ma),
     "u":    (6, ma, mb), "v":  (6, mb, ma)}            ma = n+2*ng
                                                        mb = n+2*ng+1

with an optional ``"w"`` ``(6, ma, ma)`` on input.  The step RETURNS
exactly the four keys above, mirroring the NumPy twin, which also drops
``w`` and every other carried key (``fv3_native_duo_stepper.py:1025``).
:func:`states_to_jax` / :func:`states_to_numpy` are the ONLY sanctioned
boundary adapters to the NumPy lane's list-of-dicts container.

Lists appear inside a function body only, as the six per-face operands
the SW-core kernels take BY CONSTRUCTION (they are single-face
routines).  Nothing crosses a function boundary as a list.

Deviations from the NumPy spec, each deliberate and each testable
-----------------------------------------------------------------
D1. **Ext-bundle lane only.**  The NumPy twin also carries an "interim"
    index-copy exchange path, which its own ``exchange_post_pgrad_sixface``
    documents as the non-faithful measurement opt-in and which REFUSES to
    run at ``nord > 0`` without an explicit ``ext_exclude``.  This lane
    implements the faithful path and raises on the rest, rather than
    silently selecting different halo physics (dispatch hardening).

D2. **No environment-driven numerics.**  ``LEGOESM_DUO_ENTRY_ASCALAR``,
    ``LEGOESM_DUO_PG_BVERTEX``, ``LEGOESM_DUO_AVG_B_ENDPOINTS`` and
    ``LEGOESM_DUO_CORNER_MODE`` change what the NumPy lane computes.
    A production lane whose numerics depend on the environment is a
    silent-divergence trap, so this module RAISES when one of them is
    set (naming the explicit argument that replaces it) instead of
    quietly disagreeing with a NumPy run made under the same variable.
    Same doctrine, and the same shape, as
    ``build_jax_duo_halo_tables``'s ``LEGOESM_DUO_CORNER_MODE`` guard.

D3. **``dt`` stays DYNAMIC.**  ``fv3_pgrad``'s jit factories pin
    ``dt2``/``dt`` STATIC, which would retrace the whole step on every
    new time step.  The stepper therefore calls the raw ``p_grad_c`` /
    ``one_grad_p`` kernels, whose bodies use the time step only
    arithmetically (``fv3_pgrad.py:1122`` ``dt2 * rdxc[...] / (...)``,
    ``:1308`` ``dt / (wk[...] + wk[...])``) with no Python branch on it.
    Static-vs-dynamic is a jit POLICY, not numerics; the values are
    identical.  ``d_ext`` by contrast MUST stay static -- ``one_grad_p``
    branches on it in Python (``fv3_pgrad.py:1268`` ``if d_ext > 0.0``).

D4. **The per-face loop is a Python loop, not ``vmap``.**  R1a permits
    vectorising only after a dependence argument, and the faces ARE
    independent -- but so is the NumPy lane's ``for t in range(1, 7)``,
    and the literal translation of a loop over six independent single-
    face kernel calls is six calls.  ``vmap`` would additionally require
    every per-face gridstruct stacked; it is an optimisation, not a
    translation, and is deferred.

D5. **``duo_pad_scalars`` is not ported.**  It is the legacy
    ``use_k2e_scalars`` measurement path, which the NumPy lane's own
    ``_check_exchange_flags`` makes mutually exclusive with the ext
    bundle and which its comment records as MEASURED WORSE.  A context
    carrying that flag raises here.

D6. **``ctx["step_dump"]`` is not supported.**  It is a Python callback
    at dyn_core stage boundaries; under ``jit`` it would see tracers.
    A context carrying one raises rather than silently dropping the
    instrument.

OPEN DEFECT (job 9404093, C12, dt=450 s, d_ext=0)
--------------------------------------------------
The JITTED step and the EAGER step disagree on ``u`` by
**7.510956730778894e-07 relative** after one step; ``delp``, ``pt`` and
``v`` agree to better than 1e-12.  The EAGER lane matches the NumPy
authority to 1e-11 on all four fields, so this is a property of the
compiled lowering and NOT a port defect -- but it is nine orders above
the f64 ULP, so the usual "XLA contracts mul+add into an FMA"
explanation does not cover it and the residual is UNEXPLAINED, not
agreement.  Every multi-step growth number in
``tests/grids/test_fv3_duo_stepper.py`` is downstream of it.  The three
``test_jit_gap_localiser_*`` gates there bisect it: stage chain vs
D-grid tail, then ``geopk_sw_1lev_d`` vs ``one_grad_p_1lev``.

Differentiability
-----------------
No ``donate_argnums`` anywhere (it conflicts with reverse-mode AD,
which is the point of this lane).  No ``jnp.where`` is added here -- the
stepper contains no data-dependent branch of its own; every switch it
inherits lives inside a gated kernel.  Grid metrics, halo tables and
``hs`` are STATIC (carried by :class:`DuoStepperContext`, hashed by
identity), so differentiating a step means differentiating with respect
to the state fields only.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.fv3_duo_sw_core import (
    GridFlags,
    c_sw,
    d_sw1_duo,
    d_sw2_duo,
    d_sw3_duo,
    d_sw4_duo,
    d_sw5_duo,
    d_sw6_duo,
)
from legoesm.core.fv3_pgrad import geopk, one_grad_p, p_grad_c
from legoesm.core.fv3_phase3d_common import require_uniform_float_jax
from legoesm.grids.fv3_duo_halos import (
    average_allflux_shared_edges,
    average_shared_edge_bgrid,
    build_jax_duo_halo_tables,
    ext_scalar_sixface,
    ext_scalar_sixface_allk,
    ext_vector_cgrid_sixface,
    ext_vector_cgrid_sixface_allk,
    ext_vector_dgrid_sixface,
    stack6,
)

__all__ = [
    "SWConfig",
    "SW_CFG_DEFAULT",
    "SW_CFG_CASE8",
    "DuoStepperContext",
    "build_jax_duo_stepper_context",
    "states_to_jax",
    "states_to_numpy",
    "geopk_sw_1lev",
    "geopk_sw_1lev_d",
    "p_grad_c_1lev",
    "one_grad_p_1lev",
    "exchange_post_pgrad_sixface",
    "exchange_post_pgrad_sixface_allk",
    "csw_step_sixface",
    "dsw12_step_sixface",
    "acoustic_step_sixface",
    "full_acoustic_step_sixface",
    "advance_duo_outer_step",
    "run_duo_sw",
    "make_geopk_sw_1lev_d_jit",
    "make_one_grad_p_1lev_jit",
    "make_csw_step_sixface_jit",
    "make_dsw12_step_sixface_jit",
    "make_acoustic_step_sixface_jit",
    "make_full_acoustic_step_sixface_jit",
]

# The stepper's own state keys, in the order the NumPy twin returns
# them (fv3_native_duo_stepper.py:1025).
_STATE_KEYS = ("delp", "pt", "u", "v")

# dyn_core.F90:855 `do iq=1,4+nq` -- barrier 1's slot count.  The km=1
# SW lane carries nq = 1: `fv3_duo_sw_core.d_sw1_duo` allocates its
# allflux stacks with `nq = 1` (fv3_duo_sw_core.py:2012) and the NumPy
# stepper calls `average_allflux_shared_edges(afx6, afy6, 1, n, ng)`
# (fv3_native_duo_stepper.py:684).  Not a free parameter: the two must
# agree or the barrier refuses the slot axis.
_STEPPER_NQ = 1


# ---------------------------------------------------------------------
# D2 -- the environment guards
# ---------------------------------------------------------------------

def _check_diagnostic_env(skip_b_endpoints: bool) -> None:
    """Refuse the NumPy lane's env-gated diagnostic modes.

    Each variable below changes what the NumPy stepper COMPUTES.  This
    lane takes the corresponding choice as an explicit argument, so a
    JAX run made with one of them exported would silently disagree with
    the NumPy run it is being compared against -- the exact failure the
    halo module's ``LEGOESM_DUO_CORNER_MODE`` guard exists to prevent
    (that one is enforced inside ``build_jax_duo_halo_tables`` and is
    not re-checked here).
    """
    if os.environ.get("LEGOESM_DUO_ENTRY_ASCALAR", "") == "off":
        raise ValueError(
            "LEGOESM_DUO_ENTRY_ASCALAR=off is set: the NumPy stepper "
            "then drops the ENTRY ext_scalar(delp/pt) while keeping the "
            "tail refresh.  This lane takes that choice as the explicit "
            "`entry_ascalar=False` argument and does not read the "
            "environment; unset the variable and pass the argument, or "
            "the two lanes silently disagree.")
    if os.environ.get("LEGOESM_DUO_PG_BVERTEX", "") == "mean2":
        raise ValueError(
            "LEGOESM_DUO_PG_BVERTEX=mean2 is set: the NumPy "
            "one_grad_p_1lev then runs the non-faithful bvertex_mean2 "
            "B-vertex screen (fv3_native_duo_stepper.py:946).  The JAX "
            "fv3_pgrad.one_grad_p does NOT implement it (its module "
            "docstring, deviation 3), so this lane cannot reproduce "
            "that run.")
    if (os.environ.get("LEGOESM_DUO_AVG_B_ENDPOINTS", "") == "local"
            and not skip_b_endpoints):
        raise ValueError(
            "LEGOESM_DUO_AVG_B_ENDPOINTS=local is set but the halo "
            "tables were built with skip_b_endpoints=False: barrier 2 "
            "would blend the B-edge endpoints in this lane and not in "
            "the NumPy one.  Pass "
            "build_jax_duo_stepper_context(..., skip_b_endpoints=True) "
            "to reproduce that diagnostic run.")


# ---------------------------------------------------------------------
# stage configuration
# ---------------------------------------------------------------------

class SWConfig(NamedTuple):
    """The stage knobs, hashable BY VALUE so they can be jit-static.

    Every field is read by a Python ``if`` or a Python power somewhere
    in the ``d_sw`` chain (``damp_v``/``damp_t`` feed ``damp_c > 1e-4``
    inside ``fv_tp_2d``; ``d4_bg``/``nord`` feed
    ``(da_min_c*d4_bg)**(nord+1)``; the ``hord``s select formulas), which
    is exactly why the NumPy lane's plain ``dict`` cannot cross the jit
    boundary unchanged.

    The DEFAULTS are the NumPy lane's ``_SW_CFG_DEFAULT``, restated
    rather than imported because a cross-module PRIVATE import is banned
    by ``tests/test_no_private_cross_imports.py``;
    ``test_sw_config_defaults_match_the_numpy_lane`` is what makes the
    restatement safe (it goes red if either side moves).
    """

    hord_tr: int = 8
    hord_vt: int = 6
    hord_tm: int = 6
    hord_dp: int = 6
    hord_mt: int = 6
    nord_v: int = 1
    damp_v: float = 0.2
    dddmp: float = 0.2
    d2_bg: float = 0.0
    d4_bg: float = 0.12
    nord: int = 1

    @classmethod
    def from_mapping(cls, cfg) -> SWConfig:
        """``SWConfig`` from the NumPy lane's dict, defaults for the rest.

        Mirrors ``cfg = dict(_SW_CFG_DEFAULT); cfg.update(sw_cfg or {})``
        (fv3_native_duo_stepper.py:657).  An unknown key RAISES rather
        than being dropped: the NumPy lane would carry it into ``cfg``
        where a typo'd knob is simply never read, and a knob that is
        silently ignored is a different run wearing the same name.
        """
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        unknown = sorted(set(cfg) - set(cls._fields))
        if unknown:
            raise ValueError(
                f"SWConfig.from_mapping: unknown knobs {unknown}; the "
                f"stage configuration is exactly {list(cls._fields)}")
        return cls()._replace(**dict(cfg))


SW_CFG_DEFAULT = SWConfig()

SW_CFG_CASE8 = SWConfig(
    # Zenodo C48.sw.case8 fms.out damping block + fv_core_nml, mirrored
    # from the NumPy lane's SW_CFG_CASE8 (a PUBLIC symbol there, so this
    # one is cross-checked against it by test, not restated blind):
    # del-6 (nord=2) bg 0.12, vorticity damping OFF, dddmp 0, hords 8.
    hord_tr=8, hord_vt=8, hord_tm=8, hord_dp=8, hord_mt=8,
    nord_v=2, damp_v=0.0, dddmp=0.0, d2_bg=0.0, d4_bg=0.12, nord=2,
)


def _resolve_cfg(sw_cfg) -> SWConfig:
    if sw_cfg is None:
        return SW_CFG_DEFAULT
    if isinstance(sw_cfg, SWConfig):
        return sw_cfg
    raise TypeError(
        "sw_cfg must be an SWConfig (hashable, so it can be a jit-static "
        "argument) or None; a plain dict cannot cross the jit boundary. "
        "Convert with SWConfig.from_mapping(<dict>).")


# ---------------------------------------------------------------------
# the static context
# ---------------------------------------------------------------------

class DuoStepperContext:
    """Static (never traced) half of the NumPy lane's ``ctx``.

    Identity-hashable ON PURPOSE, exactly like
    :class:`legoesm.grids.fv3_duo_halos.DuoHaloTables`: it is passed as
    a **static** jit argument so the halo index tables, the six
    gridstructs' metric arrays and the surface geopotential become
    jaxpr constants.  Consequence to respect: two structurally identical
    bundles compile twice, and NOTHING in here is differentiable -- grid
    metrics are constants in this lane, not parameters.  Build one per
    resolution and reuse it.
    """

    __slots__ = ("n", "ng", "npx", "m_a", "bd", "tab", "gs6", "flags6",
                 "hs6", "duogrid", "_batched_gs")

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):                              # pragma: no cover
        return (f"DuoStepperContext(n={self.n}, ng={self.ng}, "
                f"npx={self.npx}, duogrid={self.duogrid})")


def build_jax_duo_stepper_context(ctx: dict, *,
                                  skip_b_endpoints: bool = False,
                                  spmd_mesh=None,
                                  dtype=None,
                                  nq: int = _STEPPER_NQ,
                                  ) -> DuoStepperContext:
    """Convert the NumPy ``build_six_face_duo_context`` dict ONCE.

    Setup stays NumPy (strategy §2 / the halo module's doctrine): the
    gridstruct builders, the ext context and the k2e tables all run in
    the NumPy lane and their output is frozen here into static jax
    arrays and index tables.  Call this ONCE, outside any time loop.

    Refuses, loudly, every context this lane does not implement:

    * ``use_ext_bundle=False`` (D1) -- there is no ``ectx`` to build
      halo tables from, and the interim index-copy exchanges are the
      NumPy lane's own documented non-faithful path;
    * a step-time ``ext_exclude`` family (``ascalar``/``dvec``/
      ``divgd``/``cvec``) -- each substitutes a different exchange
      inside the step.  The build-time families (``metrics``, ``f0``)
      are consumed by ``build_six_face_duo_context`` before this point
      and are accepted;
    * ``use_k2e_scalars`` (D5) and ``step_dump`` (D6).

    A context built with ``use_ext_metrics=True`` carries extended-
    lattice gridstruct metrics that its own ``ectx`` does not describe;
    ``build_jax_duo_halo_tables``'s gridstruct cross-check refuses it,
    and that measurement lane is not supported here.
    """
    _check_diagnostic_env(skip_b_endpoints)

    if ctx.get("use_k2e_scalars"):
        raise ValueError(
            "use_k2e_scalars is set: that is the legacy k2e A-scalar pad "
            "(fv3_native_duo_stepper.duo_pad_scalars), which the NumPy "
            "lane records as MEASURED WORSE in isolation and makes "
            "mutually exclusive with the ext bundle.  It is not ported "
            "to the JAX lane.")
    if ctx.get("step_dump") is not None:
        raise ValueError(
            "ctx['step_dump'] is a Python callback at the dyn_core stage "
            "boundaries; under jit it would be handed tracers.  The JAX "
            "lane does not implement it -- run the NumPy stepper for a "
            "stage dump, or read the stage dicts this module returns.")
    if not ctx.get("use_ext_bundle"):
        raise ValueError(
            "the JAX duo stepper implements the EXT-BUNDLE lane only "
            "(dyn_core.F90:437-438/471/652/655/1336-1337).  Build the "
            "context with build_six_face_duo_context(..., "
            "use_ext_bundle=True).  The NumPy lane's interim index-copy "
            "exchanges are its own documented non-faithful measurement "
            "path (exchange_post_pgrad_sixface FAILS CLOSED on them at "
            "nord > 0), so they are not mirrored here.")
    step_time = {"ascalar", "dvec", "divgd", "cvec"}
    excluded = step_time.intersection(ctx.get("ext_exclude", ()))
    if excluded:
        raise ValueError(
            f"ext_exclude={sorted(excluded)} selects the interim "
            f"index-copy exchange inside the step; the JAX lane "
            f"implements the faithful ext_scalar/ext_vector path only "
            f"(D1).  The build-time families ('metrics', 'f0') are "
            f"consumed by build_six_face_duo_context and are accepted.")

    n = int(ctx["n"])
    ng = int(ctx["ng"])
    gs6 = ctx["gs6"]
    if len(gs6) != 6:
        raise ValueError(
            f"ctx['gs6'] holds {len(gs6)} faces, not 6 -- a short list "
            f"would silently step a subset of the cube")

    # nq: how many passenger tracers the barrier stacks carry.  Default
    # _STEPPER_NQ = 1, the frozen deck.  The tables are built PER nq (the
    # allflux slot axis is 4+nq and tab.allflux_slots indexes it), so a
    # different nq needs its own context -- it cannot be changed after.
    if int(nq) < 1:
        raise ValueError(f"nq={nq}: the duo lane carries at least one "
                         f"passenger tracer")
    tab = build_jax_duo_halo_tables(ctx["ectx"], gs6, nq=int(nq),
                                    skip_b_endpoints=skip_b_endpoints,
                                    dtype=dtype)
    if spmd_mesh is not None:
        # ENGINEERING knob (M3): route every step-side halo exchange
        # through the SPMD arm built on this mesh.  The mesh SHAPE picks
        # the arm -- ('face',) = the O(halo) whole-face shard_map ring;
        # ('face','tile_i','tile_j') = the tiled port's (6,kt,kt) tile
        # arm (duo_tiled_port_scope.md M3; kt=1 is the G0 bridge).  Any
        # other axis layout is refused by the respective builder.
        # Selects no scientific configuration; the SPMD exchanges are
        # bitwise-equal to the certified path (test_fv3_duo_spmd /
        # test_fv3_duo_tiled).  Default None keeps the certified
        # single-device trace byte-identical.
        if tuple(spmd_mesh.axis_names) == ("face",):
            from legoesm.grids.fv3_duo_spmd import build_ring_comm
            tab.ring_comm = build_ring_comm(tab, spmd_mesh)
        else:
            from legoesm.grids.fv3_duo_spmd import build_tile_comm
            tab.tile_comm = build_tile_comm(tab, spmd_mesh)

    out = DuoStepperContext()
    out.n, out.ng = n, ng
    out.npx = n + 1
    out.m_a = n + 2 * ng
    out.bd = ctx["bd"]
    out.tab = tab
    # ARRAYS-ONLY gridstruct per face (the traced half of the NumPy
    # lane's mixed dict); the scalars/bools go to GridFlags, which is
    # hashable by value.  Both halves are constants in this lane.
    # dtype (2026-08-28, fp32/mixed increment 2): the gridstruct METRICS
    # (areas, cos/sin, edge lengths, rd*c, ...) are fp64 host constants.
    # Under the coarse fp32 policy they enter kernels alongside f32 state
    # and would silently promote it back to f64 (and trip the uniformity
    # gates), so cast every INEXACT metric array to the run's storage
    # dtype. Default None keeps f64 -> byte-identical certified path.
    # (Integer index metrics are left untouched.)
    def _metric(v):
        a = jnp.asarray(v)
        if dtype is not None and jnp.issubdtype(a.dtype, jnp.inexact):
            return a.astype(dtype)
        return a
    out.gs6 = tuple({k: _metric(v) for k, v in gs.items()
                     if isinstance(v, np.ndarray)} for gs in gs6)
    out.flags6 = tuple(GridFlags.from_gs(gs) for gs in gs6)
    # duo requires bounded metrics: fv_arrays.F90:1512
    # `bounded_domain = regional .or. nested .or. duogrid` -- the
    # implication runs duogrid => bounded, and c_sw enforces it.  The
    # stage assembler checks it here too because it is publicly callable
    # with fabricated stage outputs, i.e. without passing c_sw's guard.
    for t, fl in enumerate(out.flags6):
        if not fl.bounded_domain:
            raise ValueError(
                f"gs6[{t}] has bounded_domain=False: the duo kernels "
                f"require it on every face (fv_arrays.F90:1512); build "
                f"the context with oracle_conventions=True")
    hs6 = ctx.get("hs6")
    m_a = out.m_a
    _hs_dtype = dtype if dtype is not None else jnp.float64
    if hs6 is None:
        # the NumPy twin's `np.zeros_like(delp)` default, materialised
        # once instead of per call (same values, same shape)
        out.hs6 = jnp.zeros((6, m_a, m_a), dtype=_hs_dtype)
    else:
        # Topography follows the run's storage dtype (increment 2). For
        # fp64 (dtype=None) it is uncast, byte-identical. For fp32 it is
        # cast so the gz integral and the NH carry it seeds are uniform;
        # the per-op MIXED step (a later increment) keeps the gz/energy
        # region fp64 and would pass an f64 hs6 view instead.
        out.hs6 = stack6([np.asarray(h) for h in hs6]).astype(_hs_dtype)
    require_uniform_float_jax("build_jax_duo_stepper_context", {"hs6": out.hs6})
    out.duogrid = True
    # Lazily filled by fv3_phase3d_common.build_batched_gs (the
    # face-batched vmap arm's stacked view of gs6/flags6); None = not
    # built.  Initialised here so slot-copying clones (the tests') never
    # hit an unset-slot AttributeError.
    out._batched_gs = None
    return out


# ---------------------------------------------------------------------
# lane-boundary state adapters
# ---------------------------------------------------------------------

def states_to_jax(states6, *, keep_w: bool = True) -> dict:
    """NumPy lane list-of-six-dicts -> the face-stacked JAX container.

    Only the prognostics the step reads are carried: ``delp``/``pt``/
    ``u``/``v`` and, when present and ``keep_w``, ``w``.  Every other
    key of the NumPy IC dicts (the analytic ``ua``/``va``/… ) is dropped
    -- the NumPy step drops them too, at its first return.

    Does NOT cast: an f32 initial condition RAISES here rather than
    being silently promoted, because a promoted f32 IC is a run that
    lost bits before the first step and no downstream check can see it.
    The same gate is what fires when ``jax_enable_x64`` is off, since
    ``jnp.asarray`` of an f64 numpy array then returns f32.
    """
    if len(states6) != 6:
        raise ValueError(
            f"states_to_jax expects six faces, got {len(states6)}")
    out = {k: stack6([np.asarray(s[k]) for s in states6])
           for k in _STATE_KEYS}
    n_w = sum("w" in s for s in states6)
    if n_w not in (0, 6):
        # dropping w on the strength of one missing face would run five
        # faces on their real w and one on zeros, silently
        raise ValueError(
            f"states_to_jax: 'w' is present on {n_w} of 6 faces; it must "
            f"be on all of them or none")
    if keep_w and n_w == 6:
        out["w"] = stack6([np.asarray(s["w"]) for s in states6])
    require_uniform_float_jax("states_to_jax", out)
    return out


def states_to_numpy(states: dict) -> list:
    """The face-stacked JAX container -> the NumPy lane's list of dicts.

    Concrete arrays only (it calls ``np.asarray``): this is a lane
    boundary, not something to call from inside a traced region.
    """
    arrs = {k: np.asarray(v) for k, v in states.items()}
    return [{k: arrs[k][t] for k in arrs} for t in range(6)]


# ---------------------------------------------------------------------
# km=1 SW pressure helpers -- adapters over fv3_pgrad, no new numerics
# ---------------------------------------------------------------------

def _geopk_sw_adapter(delp2d, hs, bd, pt, *, cg: bool):
    """Shared body of the two km=1 SW ``geopk`` adapters.

    Mirrors ``fv3_native_duo_stepper._geopk_sw_adapter`` argument for
    argument: it folds the ``-DSW_DYNAMICS`` convention (``akap = 1``,
    ``ptop = 0``, ``pt`` defaulting to 1, no ``cp_air``, no
    ``peln``/``pkz``) and the 2-D<->3-D staging onto the ONE km-general
    kernel in :mod:`legoesm.core.fv3_pgrad`.  It carries no numerics of
    its own.

    ``unwritten_fill=0.0`` (not the oracle's 1e30 sentinel) is the NumPy
    twin's documented choice, kept so the two lanes' halo slots are
    comparable; ``cp_air=1.0`` is inert because the SW branch drops the
    factor entirely.  ``bounded_domain=False`` is likewise the NumPy
    twin's literal, on the bounded lane too -- the duo range widening
    comes from ``duogrid=True``.
    """
    require_uniform_float_jax("geopk_sw_1lev", {"delp": delp2d, "hs": hs})
    delp3 = jnp.asarray(delp2d)[:, :, None]
    pt3 = (jnp.ones_like(delp3) if pt is None
           else jnp.asarray(pt)[:, :, None])
    out = geopk(delp3, pt3, hs, bd, km=1, ptop=0.0, akap=1.0,
                cp_air=1.0, cg=cg, duogrid=True, computehalo=False,
                npx=bd.ie + 1, npy=bd.je + 1, a2b_ord=4,
                bounded_domain=False, sw_dynamics=True,
                unwritten_fill=0.0)
    return out["pk"], out["gz"]


def geopk_sw_1lev(delpc, hs, bd, pt=None):
    """``dyn_core.F90`` geopk (2660-2790), SW_DYNAMICS, km=1, CG=.true.

    The C-grid call site is ``dyn_core.F90:533``.  Returns
    ``(pkc, gz)``, both ``(m_a, m_a, 2)``.
    """
    return _geopk_sw_adapter(delpc, hs, bd, pt, cg=True)


def geopk_sw_1lev_d(delp, hs, bd, pt=None):
    """geopk at the D-grid call site (``dyn_core.F90:1401``), CG=.false.

    Same km=1 SW formulas as the C version; the ranges widen to
    ``is-2..ie+2`` and ``delp``'s halos must be freshly exchanged (the
    oracle runs ``ext_scalar(delp/pt)`` at ``:1336-1337`` immediately
    before this call, which is exactly what
    :func:`full_acoustic_step_sixface` does).

    OPEN ITEM inherited from the NumPy twin, NOT introduced here:
    ``dyn_core.F90:1402`` passes ``computehalo=.true.`` at this site,
    which on the duo lane would extend the write box to the full data
    domain.  The km=1 NumPy path has always used the un-extended
    ``is-2..ie+2`` box; mirroring it keeps hop B a pure translation, and
    re-opening it would be a hop-A change (out of scope, strategy §10).
    """
    return _geopk_sw_adapter(delp, hs, bd, pt, cg=False)


def p_grad_c_1lev(dt2, delpc, pkc, gz, uc, vc, gs: dict, bd):
    """``dyn_core.F90`` p_grad_c (2073-2132), km=1, hydrostatic.

    Call site ``dyn_core.F90:629``.  FUNCTIONAL (R4): the NumPy adapter
    mutates ``uc``/``vc`` through ``[:, :, None]`` views; this one
    RETURNS ``(uc, vc)``.  ``delpc`` is unread on the hydrostatic branch
    and is kept for interface fidelity.

    ``dt2`` may be a TRACER here (D3): the kernel uses it only as a
    multiplier.
    """
    uc3, vc3 = p_grad_c(dt2, jnp.asarray(delpc)[:, :, None], pkc, gz,
                        jnp.asarray(uc)[:, :, None],
                        jnp.asarray(vc)[:, :, None], gs, bd, npz=1,
                        hydrostatic=True)
    return uc3[:, :, 0], vc3[:, :, 0]


def one_grad_p_1lev(u, v, pkc, gz, divg2, gs: dict, bd, npx: int,
                    npy: int, *, dt, d_ext: float = 0.02):
    """``dyn_core.F90`` one_grad_p (2347-2480), km=1 hydrostatic SW.

    Call site ``dyn_core.F90:1531``.  FUNCTIONAL (R4): returns
    ``(u, v)``.

    ALIASING: the NumPy adapter has to COPY ``pkc``/``gz`` on entry,
    because the shared kernel is faithful to dyn_core and mutates them
    in place through ``a2b_ord4(replace=.true.)``; the JAX kernel
    returns new arrays instead, so the caller's ``pkc``/``gz`` are
    untouched by construction and the copy has nothing to protect.  The
    B-node ``pk``/``gz`` the kernel returns are DISCARDED here, exactly
    as both NumPy call sites discard them.

    ``dt`` may be a TRACER (D3); ``d_ext`` may NOT -- ``one_grad_p``
    branches on it in Python (``fv3_pgrad.py:1268``).
    """
    u_out, v_out, _pk, _gz = one_grad_p(
        jnp.asarray(u)[:, :, None], jnp.asarray(v)[:, :, None], pkc, gz,
        divg2, None, gs, bd, npx=npx, npy=npy, npz=1, dt=dt, ptop=0.0,
        akap=1.0, hydrostatic=True, a2b_ord=4, d_ext=d_ext, ng=bd.ng,
        duogrid=True)
    return u_out[:, :, 0], v_out[:, :, 0]


# ---------------------------------------------------------------------
# the post-p_grad_c duo exchanges
# ---------------------------------------------------------------------

def exchange_post_pgrad_sixface(ctx: DuoStepperContext, divgd6, uc6, vc6,
                                *, nord: int):
    """``dyn_core.F90:652`` ``ext_scalar(divgd, …, 1,1)`` (gated on
    ``flagstruct%nord > 0``) and ``:655`` ``ext_vector(uc, vc, …,
    1,0,0,1)`` (UNGATED on the duo lane).

    THE GATE IS ``nord``, the DIVERGENCE-damping order ``d_sw5`` runs,
    not ``nord_v``, the vorticity-damping order: upstream seeds
    ``nord_k = flagstruct%nord`` (``:749``) and derives the per-level
    ``nord_v(k) = min(2, flagstruct%nord)`` only later (``:757``), and
    the exchange is gated on the GLOBAL flag.  The shipped decks set
    both to the same value, so gating on the wrong one is invisible
    there -- which is why it is stated.

    Returns ``(divgd6, uc6, vc6)`` face-stacked.  Every face-stacked
    operand is ``(6, …)``.
    """
    nord = _validated_nord(nord)
    tab = ctx.tab
    if nord > 0:
        divgd6 = ext_scalar_sixface(divgd6, tab, "B")
    uc6, vc6 = ext_vector_cgrid_sixface(uc6, vc6, tab)
    return divgd6, uc6, vc6


def _validated_nord(nord) -> int:
    """A damping ORDER is integral by construction; the deck dicts mix
    ints and floats, so ``int()`` would round 2.7 to 2 without a word.
    Reject instead (mirrors the NumPy guard)."""
    if nord != int(nord):
        raise ValueError(
            f"nord must be an integral damping order, got {nord!r}")
    return int(nord)


def exchange_post_pgrad_sixface_allk(ctx: DuoStepperContext, divgd6k,
                                     uc6k, vc6k, *, nord: int):
    """k-batched :func:`exchange_post_pgrad_sixface` (v2a): the same
    two exchanges -- ``ext_scalar(divgd, …, 1,1)`` gated on ``nord >
    0``, ``ext_vector(uc, vc, …, 1,0,0,1)`` ungated -- over
    ``(6, i, j, K)`` stacks with the level axis TRAILING, one batched
    exchange call per operand instead of one per level.  The gate
    semantics live here exactly as in the per-level helper; see its
    docstring for the nord-not-nord_v argument.
    """
    nord = _validated_nord(nord)
    tab = ctx.tab
    if nord > 0:
        divgd6k = ext_scalar_sixface_allk(divgd6k, tab, "B")
    uc6k, vc6k = ext_vector_cgrid_sixface_allk(uc6k, vc6k, tab)
    return divgd6k, uc6k, vc6k


# ---------------------------------------------------------------------
# stage assemblers
# ---------------------------------------------------------------------

def _stack_faces(per_face: list) -> dict:
    """Six per-face output dicts -> one dict of ``(6, …)`` stacks.

    All six dicts carry the same keys and, within one key, the same
    per-face shape (contract §2: the FACE axis is stackable, the stagger
    axis is not -- which is why this stacks per key and never across
    keys).

    A key that is ``None`` on ALL six faces is carried through as
    ``None``: the hydrostatic arms of ``d_sw2_duo`` and ``d_sw5_duo``
    return ``w = None`` (an ABSENCE -- the branch never ran), and
    materialising that as zeros would be a plausible-looking value where
    the twin has nothing, which is precisely the sentinel failure this
    repo has been bitten by.  A key that is ``None`` on SOME faces is a
    per-face lane split and raises.
    """
    keys = tuple(per_face[0])
    for t, d in enumerate(per_face[1:], start=1):
        if tuple(d) != keys:
            raise ValueError(
                f"face {t} returned keys {tuple(d)}, face 0 returned "
                f"{keys} -- the stage outputs must be uniform to stack")
    out = {}
    for k in keys:
        vals = [d[k] for d in per_face]
        n_none = sum(v is None for v in vals)
        if n_none == 6:
            out[k] = None
        elif n_none:
            raise ValueError(
                f"{k}: {n_none} of 6 faces returned None -- the faces "
                f"took different branches, which cannot be stacked and "
                f"is not something this lane should ever do")
        else:
            out[k] = jnp.stack(vals)
    return out


def _state_w(states: dict):
    """``st.get("w", st["pt"] * 0.0)`` -- the NumPy twin's own default.

    ``pt * 0.0`` rather than ``zeros_like``: the twin writes exactly
    that, and it is NOT the same expression -- it propagates a NaN
    wherever ``pt`` carries one, which is the lane's tripwire and must
    not be quietly repaired here.
    """
    if "w" in states:
        w = states["w"]
        require_uniform_float_jax("_state_w", {"w": w})
        return w
    return states["pt"] * 0.0


def csw_step_sixface(ctx: DuoStepperContext, states: dict, dt2,
                     *, duogrid: bool = True) -> dict:
    """The certified duo ``c_sw`` on every face (``dyn_core.F90:489``).

    dyn_core's ``divgd`` + ``uc``/``vc`` exchanges happen POST-
    ``p_grad_c`` (``:652``/``:655``), so they are NOT applied here --
    :func:`dsw12_step_sixface` runs them at their own point in the
    cadence.  (The NumPy twin carries an ``exchange=True`` opt-in for
    standalone halo tests, which substitutes the interim index-copy
    helpers at a point where upstream has NO duo exchange at all; it is
    not mirrored -- D1.)

    Returns the stacked ``c_sw`` outputs: ``delpc``, ``ptc``, ``wc``,
    ``uc``, ``vc``, ``ua``, ``va``, ``ut``, ``vt``, ``divg_d``.
    """
    require_uniform_float_jax("csw_step_sixface",
                     {k: states[k] for k in _STATE_KEYS})
    npx = ctx.npx
    w6 = _state_w(states)
    # R1a: the six faces are INDEPENDENT -- c_sw reads only its own
    # face's fields and gridstruct and writes only its own outputs, so
    # no iteration reads a location another writes.  Kept as a literal
    # loop rather than vmapped (D4).
    outs = [c_sw(states["delp"][t], states["pt"][t], w6[t],
                 states["u"][t], states["v"][t], ctx.gs6[t], ctx.bd,
                 npx, npx, dt2, duogrid=duogrid,
                 bounded_domain=ctx.flags6[t].bounded_domain)
            for t in range(6)]
    return _stack_faces(outs)


def dsw12_step_sixface(ctx: DuoStepperContext, states: dict,
                       csw_outs: dict, dt, sw_cfg=None) -> dict:
    """geopk(SW, km=1) + ``p_grad_c`` per face (``:533``/``:629``), the
    post-``p_grad_c`` duo exchanges (``:652``/``:655``), ``d_sw1`` per
    face, BARRIER 1 (``:872``, ``mpp_get_boundary`` with
    ``gridtype=CGRID_NE`` at ``:874``, then the 0.5*(mine + neighbour)
    blend), then ``d_sw2`` per face on the AVERAGED flux slots.

    Workspace choice, carried verbatim from the NumPy twin: ``d_sw1``'s
    ``ut``/``vt`` workspaces enter as ZEROS (``workspace_sentinel=0.0``)
    -- the DEFINED analog of upstream's uninitialised stack, which the
    always-fire panel-edge blocks read.

    Returns the stacked ``d_sw1`` outputs with the AVERAGED
    ``allflux_x``/``allflux_y``, the ``d_sw2``-updated ``delp``/``pt``,
    and the exchanged ``uc``/``vc``/``divg_d``.
    """
    cfg = _resolve_cfg(sw_cfg)
    bd, npx, n, m_a = ctx.bd, ctx.npx, ctx.n, ctx.m_a

    uc6 = csw_outs["uc"]
    vc6 = csw_outs["vc"]
    # --- geopk + p_grad_c per face (R1a: face-independent) -----------
    pg = []
    for t in range(6):
        pkc, gz = geopk_sw_1lev(csw_outs["delpc"][t], ctx.hs6[t], bd,
                                pt=csw_outs["ptc"][t])
        pg.append(p_grad_c_1lev(0.5 * dt, csw_outs["delpc"][t], pkc, gz,
                                uc6[t], vc6[t], ctx.gs6[t], bd))
    uc6 = jnp.stack([p[0] for p in pg])
    vc6 = jnp.stack([p[1] for p in pg])

    divgd6, uc6, vc6 = exchange_post_pgrad_sixface(
        ctx, csw_outs["divg_d"], uc6, vc6, nord=int(cfg.nord))

    # --- d_sw1 per face ---------------------------------------------
    zx = jnp.zeros((npx, n), dtype=jnp.float64)
    zy = jnp.zeros((n, npx), dtype=jnp.float64)
    zcx = jnp.zeros((npx, m_a), dtype=jnp.float64)
    zcy = jnp.zeros((m_a, npx), dtype=jnp.float64)
    w6 = _state_w(states)
    s1 = [d_sw1_duo(states["delp"][t], states["pt"][t], w6[t],
                    uc6[t], vc6[t], zx, zy, zcx, zcy,
                    ctx.gs6[t], ctx.flags6[t], bd, npx, npx, dt=dt,
                    hord_tr=cfg.hord_tr, hord_vt=cfg.hord_vt,
                    hord_tm=cfg.hord_tm, hord_dp=cfg.hord_dp,
                    nord_v=cfg.nord_v, nord_t=0,
                    damp_v=cfg.damp_v, damp_t=0.0,
                    nq=int(ctx.tab.nq),
                    workspace_sentinel=0.0)
          for t in range(6)]
    s1s = _stack_faces(s1)

    # --- BARRIER 1 (dyn_core.F90:872) --------------------------------
    afx6, afy6 = average_allflux_shared_edges(s1s["allflux_x"],
                                              s1s["allflux_y"], ctx.tab)

    # --- d_sw2 per face, on the AVERAGED slots -----------------------
    s2 = [d_sw2_duo(s1s["delp"][t], s1s["pt"][t], afx6[t], afy6[t],
                    ctx.gs6[t], ctx.flags6[t], bd)
          for t in range(6)]
    s2s = _stack_faces(s2)
    return {**s1s, "allflux_x": afx6, "allflux_y": afy6,
            "delp": s2s["delp"], "pt": s2s["pt"],
            "uc": uc6, "vc": vc6, "divg_d": divgd6}


def acoustic_step_sixface(ctx: DuoStepperContext, states: dict, dt,
                          sw_cfg=None, entry_ascalar: bool = True) -> dict:
    """ONE full duo acoustic STAGE CHAIN on all six faces.

    Entry exchanges (``dyn_core.F90:432-439`` A-scalars, gated on
    ``it == 1``; ``:468-472`` the D-vector, EVERY substep) -> ``c_sw``
    -> geopk/PG-C -> the post-PG exchanges -> ``d_sw1`` -> BARRIER 1 ->
    ``d_sw2`` -> ``d_sw3`` -> BARRIER 2 (``:984``, ``gridtype=BGRID_NE``
    at ``:986``) on ``(ubb, vbbtemp)`` -> the inline KE assembly ->
    ``d_sw4`` -> ``d_sw5`` -> ``d_sw6``.

    ``entry_ascalar`` is the ``it == 1`` gate and is STATIC: it selects
    whether two exchanges run at all, and getting it wrong changes every
    step after the first.  See :func:`advance_duo_outer_step`.

    Returns the stacked per-face new state ``delp``/``pt``/``u``/``v``
    plus the diagnostics the D-grid tail consumes (``ke``, ``wk``,
    ``divg_d`` and, load-bearing for the external-mode filter,
    ``delpc`` -- ``d_sw5``'s SAVED divergence).
    """
    require_uniform_float_jax("acoustic_step_sixface",
                     {k: states[k] for k in _STATE_KEYS})
    cfg = _resolve_cfg(sw_cfg)
    bd, npx, tab = ctx.bd, ctx.npx, ctx.tab
    m_a, ng = ctx.m_a, ctx.ng

    # --- entry exchanges (dyn_core.F90:437-438, :471) ----------------
    delp6, pt6 = states["delp"], states["pt"]
    if entry_ascalar:
        delp6 = ext_scalar_sixface(delp6, tab, "A")
        pt6 = ext_scalar_sixface(pt6, tab, "A")
    u6, v6 = ext_vector_dgrid_sixface(states["u"], states["v"], tab)
    st = {**states, "delp": delp6, "pt": pt6, "u": u6, "v": v6}

    # --- c_sw, then the whole C-grid + d_sw1/d_sw2 half --------------
    csw = csw_step_sixface(ctx, st, 0.5 * dt)
    s12 = dsw12_step_sixface(ctx, st, csw, dt, sw_cfg=cfg)

    # --- d_sw3 per face (R1a: face-independent) ----------------------
    s3 = _stack_faces([
        d_sw3_duo(st["u"][t], st["v"][t], s12["uc"][t], s12["vc"][t],
                  ctx.gs6[t], ctx.flags6[t], bd, npx, npx, dt=dt,
                  hord_mt=cfg.hord_mt)
        for t in range(6)])

    # --- BARRIER 2 (dyn_core.F90:984) --------------------------------
    # the oracle loads tempfx1 <- ubb and tempfy1 <- vbbtemp (:971-981),
    # so ubb is the x-like member and vbbtemp the y-like one
    ubb6, vbbtemp6 = average_shared_edge_bgrid(s3["ubb"], s3["vbbtemp"],
                                               tab)

    ring = slice(ng, ng + npx)
    outs = []
    for t in range(6):
        # dyn_core's inline KE assembly between d_sw3 and d_sw4, on the
        # AVERAGED B arrays.  The 0.0 background is the NumPy twin's
        # (`np.full(..., 0.0)`), not the 1e30 workspace sentinel.
        kee = s3["ubbtemp"][t] * vbbtemp6[t]
        ke = jnp.zeros((m_a + 1, m_a + 1), dtype=jnp.float64)
        ke = ke.at[ring, ring].set(0.5 * (kee + ubb6[t] * s3["vbb"][t]))
        s4 = d_sw4_duo(st["u"][t], st["v"][t], s12["ut"][t],
                       s12["vt"][t], ke, ctx.flags6[t], bd, npx, npx,
                       dt=dt)
        s5 = d_sw5_duo(s12["delp"][t], st["u"][t], st["v"][t],
                       s12["uc"][t], s12["vc"][t], csw["ua"][t],
                       csw["va"][t], s12["divg_d"][t],
                       s12["crx_adv"][t], s12["cry_adv"][t],
                       s12["xfx_adv"][t], s12["yfx_adv"][t],
                       s12["ra_x"][t], s12["ra_y"][t], s4["ke"],
                       ctx.gs6[t], ctx.flags6[t], bd, npx, npx, dt=dt,
                       hord_vt=cfg.hord_vt, nord=cfg.nord,
                       dddmp=cfg.dddmp, d2_bg=cfg.d2_bg,
                       d4_bg=cfg.d4_bg, d_con=0.0)
        s6 = d_sw6_duo(st["u"][t], st["v"][t], s5["ut"], s5["vt"],
                       s5["ke"], s5["wk"], s5["vortfluxx"],
                       s5["vortfluxy"], ctx.gs6[t], ctx.flags6[t], bd,
                       npx, npx, nord_v=cfg.nord_v, damp_v=cfg.damp_v,
                       d_con=0.0)
        outs.append({"delp": s12["delp"][t], "pt": s12["pt"][t],
                     "u": s6["u"], "v": s6["v"],
                     "ke": s5["ke"], "wk": s5["wk"],
                     "divg_d": s5["divg_d"], "delpc": s5["delpc"]})
    return _stack_faces(outs)


def full_acoustic_step_sixface(ctx: DuoStepperContext, states: dict, dt,
                               d_ext: float = 0.02, sw_cfg=None,
                               entry_ascalar: bool = True) -> dict:
    """The complete acoustic step INCLUDING the D-grid tail.

    ``acoustic_step_sixface`` -> the ``delp``/``pt`` halo refresh
    (``dyn_core.F90:1336-1337``) -> per face the D geopk (``:1401``),
    the external-mode filter and ``one_grad_p`` (``:1531``) back to
    covariant winds.

    **The returned D-wind halos are STALE-BY-ONE, deliberately.**
    ``dyn_core.F90:1332-1338`` refreshes ONLY ``delp`` and ``pt``; the
    D-vector ``ext_vector`` runs at the NEXT step's entry (``:468-472``,
    the call at ``:471``).  A twin that refreshes the winds here is a
    DIVERGENCE, not a fix, and it will look more correct than the
    faithful version -- so it is pinned by a test, with a non-vacuity
    assertion that the refresh would have changed those cells.

    ``d_ext`` defaults to 0.02 to keep the signature identical to the
    NumPy twin's (a different default would make the two backends
    silently disagree, which is a worse failure than an inconvenient
    default).  It is NOT the oracle value: every duo deck resolves
    ``D_EXT = 0.0``, so oracle comparisons must pass ``d_ext=0.0`` --
    a nonzero value has already produced one retracted claim in this
    campaign.  ``d_ext`` is jit-STATIC (``one_grad_p`` branches on it in
    Python).
    """
    require_uniform_float_jax("full_acoustic_step_sixface",
                     {k: states[k] for k in _STATE_KEYS})
    bd, npx, ng, tab = ctx.bd, ctx.npx, ctx.ng, ctx.tab

    stage = acoustic_step_sixface(ctx, states, dt, sw_cfg=sw_cfg,
                                  entry_ascalar=entry_ascalar)

    # --- post-step delp/pt refresh (dyn_core.F90:1336-1337) ----------
    delp6 = ext_scalar_sixface(stage["delp"], tab, "A")
    pt6 = ext_scalar_sixface(stage["pt"], tab, "A")

    sl = slice(ng, ng + npx)
    u6, v6 = stage["u"], stage["v"]
    outs_u, outs_v = [], []
    for t in range(6):
        pkc, gz = geopk_sw_1lev_d(delp6[t], ctx.hs6[t], bd, pt=pt6[t])
        # divg2 = d_ext * da_min_c * the SAVED divergence, at km=1.
        # dyn_core.F90:1310-1330 forms
        #   wk = sum_k ptc, divg2 = d2_divg * sum_k(ptc*vt) / wk
        # with d2_divg = flagstruct%d_ext * gridstruct%da_min_c (:1311);
        # at npz=1 the mass weight ptc cancels exactly, leaving
        # d2_divg*vt(:,:,1).  That cancellation is the NumPy lane's
        # (hop A) km=1 form, mirrored here verbatim, not re-derived.
        divg2 = (d_ext * float(ctx.flags6[t].da_min_c)
                 * stage["delpc"][t][sl, sl])
        uu, vv = one_grad_p_1lev(u6[t], v6[t], pkc, gz, divg2,
                                 ctx.gs6[t], bd, npx, npx, dt=dt,
                                 d_ext=d_ext)
        outs_u.append(uu)
        outs_v.append(vv)
    # NO post-step vector refresh -- see the docstring.  Consumers
    # needing fresh D halos re-enter the step (which exchanges at its
    # own entry) or call ext_vector_dgrid_sixface themselves.
    return {"delp": delp6, "pt": pt6,
            "u": jnp.stack(outs_u), "v": jnp.stack(outs_v)}


def advance_duo_outer_step(ctx: DuoStepperContext, states: dict,
                           dt_atmos, n_split: int, d_ext: float = 0.02,
                           sw_cfg=None, *, step_fn=None) -> dict:
    """One ``dt_atmos`` block = ``n_split`` acoustic steps.

    ``dt = dt_atmos / n_split`` with the UPSTREAM exchange schedule: the
    entry A-scalar ``ext_scalar(delp, pt)`` fires only on the FIRST
    inner step (``dyn_core.F90:432`` gates it on ``it == 1``), while the
    D-vector entry (``:471``), the post-PG pair (``:652``/``:655``) and
    the tail refresh (``:1336-1337``) fire on every inner step.

    ``step_fn`` (keyword-only, defaults to
    :func:`full_acoustic_step_sixface`) is the COMPILATION seam.  Jitting
    THIS routine unrolls ``n_split`` whole steps into one program, which
    at production ``n_split = 7`` is a seven-times-larger graph to
    compile for no gain; passing
    ``step_fn=make_full_acoustic_step_sixface_jit()`` instead compiles
    ONE step (twice -- ``entry_ascalar`` is static) and reuses it.  It
    exists so callers get that without re-implementing the ``it == 1``
    cadence, which must live in exactly one place.  The signature stays
    positionally identical to the NumPy twin's.

    ``n_split`` is a Python trip count and therefore jit-STATIC when this
    routine itself is jitted; a different ``n_split`` is a different
    unrolled program.
    """
    if n_split < 1:
        raise ValueError(f"n_split must be >= 1, got {n_split}")
    step = full_acoustic_step_sixface if step_fn is None else step_fn
    dt = dt_atmos / n_split
    for it in range(n_split):
        states = step(ctx, states, dt, d_ext=d_ext, sw_cfg=sw_cfg,
                      entry_ascalar=(it == 0))
    return states


def run_duo_sw(ctx: DuoStepperContext, states: dict, dt, nsteps: int,
               d_ext: float = 0.02, sw_cfg=None, *, step_fn=None) -> dict:
    """``nsteps`` back-to-back full acoustic steps.

    Mirrors the NumPy twin, INCLUDING its flat cadence: every step here
    runs the entry A-scalar exchange, where
    :func:`advance_duo_outer_step` runs it once per ``dt_atmos`` block.
    The two are different exchange schedules and are not
    interchangeable.

    A Python loop, not ``lax.scan`` over time: the caller normally wants
    the intermediate states (the runners sample daily), and the first
    step's input pytree may carry a ``w`` leaf that the returned state
    does not, which a scan carry cannot express.  Pass
    ``step_fn=make_full_acoustic_step_sixface_jit()`` for the compiled
    path -- see :func:`advance_duo_outer_step` for why the seam is on
    the STEP and not on the loop.
    """
    if nsteps < 0:
        raise ValueError(f"nsteps must be >= 0, got {nsteps}")
    step = full_acoustic_step_sixface if step_fn is None else step_fn
    for _ in range(nsteps):
        states = step(ctx, states, dt, d_ext=d_ext, sw_cfg=sw_cfg)
    return states


# ---------------------------------------------------------------------
# jit policies -- the ONE place each entry point's staticness is decided
# ---------------------------------------------------------------------
# Common to all of them: `ctx` is STATIC (identity-hashable; it carries
# the halo index tables, the six gridstructs and `hs`, all constants in
# this lane), `sw_cfg`/`entry_ascalar`/`d_ext` are STATIC (each is read
# by a Python branch or a Python power), and `dt` is DYNAMIC so a new
# time step does not recompile.  No `donate_argnums` anywhere: buffer
# donation conflicts with reverse-mode AD, which is the point of the
# lane.

def make_geopk_sw_1lev_d_jit(fn=geopk_sw_1lev_d):
    """Static: ``bd`` only.  ``delp``/``hs``/``pt`` dynamic."""
    return jax.jit(fn, static_argnums=(2,))


def make_one_grad_p_1lev_jit(fn=one_grad_p_1lev):
    """Static: ``bd``/``npx``/``npy`` + ``d_ext`` (``one_grad_p``
    branches on it in Python).  ``dt`` stays DYNAMIC, per D3."""
    return jax.jit(fn, static_argnums=(6, 7, 8),
                   static_argnames=("d_ext",))


def make_csw_step_sixface_jit(fn=csw_step_sixface):
    """Static: ctx + duogrid.  ``states``/``dt2`` dynamic."""
    return jax.jit(fn, static_argnums=(0,), static_argnames=("duogrid",))


def make_dsw12_step_sixface_jit(fn=dsw12_step_sixface):
    """Static: ctx + sw_cfg.  ``states``/``csw_outs``/``dt`` dynamic."""
    return jax.jit(fn, static_argnums=(0, 4),
                   static_argnames=("sw_cfg",))


def make_acoustic_step_sixface_jit(fn=acoustic_step_sixface):
    """Static: ctx + sw_cfg + entry_ascalar.  ``states``/``dt`` dynamic."""
    return jax.jit(fn, static_argnums=(0, 3, 4),
                   static_argnames=("sw_cfg", "entry_ascalar"))


def make_full_acoustic_step_sixface_jit(fn=full_acoustic_step_sixface):
    """Static: ctx + d_ext + sw_cfg + entry_ascalar.

    ``d_ext`` is listed in BOTH ``static_argnums`` and
    ``static_argnames`` so it is static whether the caller passes it
    positionally or by keyword (jax resolves the two independently when
    both are given -- ``jax/_src/api_util.py:420-423``).
    """
    return jax.jit(fn, static_argnums=(0, 3, 4, 5),
                   static_argnames=("d_ext", "sw_cfg", "entry_ascalar"))


