"""C-grid half-step of the duo acoustic cadence, km-general -- JAX lane.

Functional, jit-compilable, differentiable twin of
``legoesm.core.fv3_native_cgrid_phase_3d``.  Like its NumPy
specification, this module introduces **no numerics of its own**: every
arithmetic operation happens inside a kernel that is already ported and
gated elsewhere (``fv3_duo_sw_core.c_sw``, ``fv3_pgrad.geopk`` /
``p_grad_c``, ``fv3_nh_core.update_dz_c`` / ``riem_solver_c``).  What it
contributes is the CADENCE -- which kernel runs at which level, on which
face, and in which order.

Authority chain (``docs/atmosphere/fv3_duo_jax_lane_strategy.md`` §2):
``fv3_native_cgrid_phase_3d`` is the SPECIFICATION (hop B).  The pinned
Fortran
``/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean``
is quoted only where a line number says WHY a step exists.  **Every
citation in this file was re-derived with ``sed``/``grep`` against that
tree** (R7) rather than copied from a comment, including the ones the
NumPy spec already carries -- one of them turned out to be wrong (see
"Where the spec's prose is not verifiable" below).

ORACLE CADENCE, verified line by line
-------------------------------------
::

    dyn_core.F90:339   do it=1,n_split
                :488     do k=1,npz
                :489       call c_sw(...)                 <- per LEVEL
                :533     call geopk(..., .true., ...)     <- CG=.true.
                :584     call update_dz_c(...)            } NH only,
                :590     call Riem_Solver_C(...)          }  :583-594
                :629     call p_grad_c(dt2, npz, delpc, pkc, gz, uc, vc,
                                       bd, rdxc, rdyc, hydrostatic)

WHY ``c_sw`` IS LOOPED OVER LEVELS AND THE PRESSURE CHAIN IS NOT
---------------------------------------------------------------
``c_sw``'s dummies are 2-D -- ``dimension(bd%isd:bd%ied, bd%jsd:bd%jed)``
and friends at ``sw_core.F90:84-87`` -- so the oracle wraps it in
``do k=1,npz`` (``:488``).  The faithful lift is to call the certified
2-D kernel once per level, NOT to write new stage math.

``geopk`` and ``p_grad_c`` are km-general and take the whole column, so
they are called ONCE per face with ``km``.  ``geopk``'s running
top-down ``p1d`` accumulator is a recurrence whose SUM ORDER is part of
the bit-exact contract; a per-level loop here would break it.
``p_grad_c`` has no inter-k coupling at all (``dyn_core.F90:2100`` opens
``do k=1,npz`` and the only vertical reads are the bracketing interfaces
``k`` and ``k+1``).

CONVENTIONS THIS MODULE SETS FOR THE 3-D JAX LAYER
--------------------------------------------------
This is the first Phase-2 (3-D) module; everything above the kernels is
otherwise unported, so the choices below are the ones later 3-D modules
copy.  Each is a decision, not an accident:

C1. **State is ONE dict of FACE-STACKED arrays**, never a list of six
    per-face dicts::

        {"delp": (6, m_a, m_a, km), "pt": (6, m_a, m_a, km),
         "w":    (6, m_a, m_a, km),
         "u":    (6, m_a, m_b, km), "v": (6, m_b, m_a, km)}

    with ``m_a = n + 2*ng``, ``m_b = m_a + 1``, face ``t`` = Fortran tile
    ``t+1``, and Fortran index ``1-ng`` at numpy index 0 on both
    horizontal axes (``fv3_duo_halos``'s binding contract).  Per-field
    shapes come from ``fv3_native_state_3d.field_shape`` -- IMPORTED, so
    the two lanes cannot drift (C7).  This mirrors the km=1
    ``fv3_duo_stepper`` contract and satisfies strategy §7a, which
    requires the canonical PyTree to be stated AND pinned by a test.
    Lists appear inside a function body only, as the per-face operands
    the single-face kernels take by construction; nothing crosses a
    public boundary as a list.  :func:`state_3d_to_jax` /
    :func:`state_3d_to_numpy` are the ONLY sanctioned adapters.

C2. **The face loop and the LEVEL loop are Python loops, not ``vmap``.**
    R1a permits vectorising only after a dependence argument, and both
    loops ARE independent (the arguments are written at each site).  But
    so is the NumPy lane's ``for t`` / ``for k``, and the literal
    translation of a loop over independent single-face, single-level
    kernel calls is that many calls -- the same reasoning as the km=1
    stepper's deviation D4.  ``vmap`` over levels is an OPTIMISATION,
    and it is not free of risk: batching adds an axis that XLA may fuse
    and contract differently, so it needs its own equivalence gate
    before it can be called a translation.  Deferred, deliberately.
    Cost of the choice, stated: at ``km = 4`` this traces 24 ``c_sw``
    bodies, so compile time is minutes rather than seconds.

C2a. **The FACE loop has an OPT-IN vmapped arm** (face-batching ladder
    steps 1-2), ``batched=True`` on each phase function, built on
    :func:`legoesm.core.fv3_phase3d_common.build_batched_gs`'s stacked
    view of the six gridstructs.  Motivation is measured, not
    aesthetic: under SPMD the per-face traced reads ``x[t]`` are the
    2-GPU wall (a masked select + all-reduce per read, and every device
    computes all six faces); ``vmap`` over a leading ``(6,)`` axis lets
    GSPMD partition the face axis with zero communication.  The DEFAULT
    is ``batched=False`` -- the certified loop path, byte for byte --
    and the equivalence gate C2 asked for is
    ``test_*_batched_matches_loop`` (rtol 1e-13 / atol 1e-12; the
    few-ulp slack is XLA reassociating across the added batch axis).
    The LEVEL loop stays a Python loop on both arms.

C3. **Static/dynamic split.**  ``ctx``, ``km`` and every deck constant
    or Python-branch selector are STATIC; the field data and ``dt2`` are
    DYNAMIC.  ``dt2`` stays dynamic (the stepper's deviation D3
    generalised) so a new time step does not recompile the phase.  That
    is a jit POLICY, not numerics, and it was verified rather than
    assumed -- every callee uses the time step arithmetically only:
    ``fv3_nh_core.py:843`` ``rdt = 1.0 / dt`` is update_dz_c's ONLY use;
    ``fv3_nh_core.py:203-204`` ``t1g = gama*2.0*dt*dt`` / ``rdt = 1.0/dt``
    are sim1_solver's, with ``:258``/``:265``/``:279`` multipliers;
    ``fv3_pgrad.py`` p_grad_c multiplies by ``dt2``.  No Python ``if``
    on the time step anywhere in that chain.  (``fv3_nh_core``'s and
    ``fv3_pgrad``'s OWN jit factories pin ``dt`` static; that is their
    policy for a standalone kernel call, and it is not a constraint on
    the body.)

C4. **In-place mutation in the spec becomes a RETURNED key** (R4).  The
    NumPy lane mutates ``uc``/``vc`` inside ``csw_outs`` (p_grad_c's
    oracle contract, ``dyn_core.F90:2073-2132``) and mutates ``gz``/
    ``ws3`` in the NH stage.  Here every one of those comes back in the
    returned dict and the CALLER threads it forward.  A key that the
    spec mutates and this lane does not return would be a silent
    divergence, so the rule is mechanical: if the spec writes it, it is
    in the returned dict.

C5. **A data-dependent Python guard cannot exist under ``jit``**, so it
    becomes a STATIC ``check_*`` keyword that RAISES on a tracer rather
    than silently skipping.  See :func:`cgrid_pressure_phase_3d`'s
    ``check_delpc``.  Silently skipping is the failure this repo has
    been bitten by most often; an explicit opt-out is greppable at every
    call site.

C6. **Stage observation is by RETURN, not by callback.**  The spec takes
    a ``stage_hook(name, payload)`` for the oracle stage boundaries S04
    (post C-grid geopk) and S05 (post ``p_grad_c``).  Under ``jit`` that
    hook would be handed tracers -- exactly what
    ``build_jax_duo_stepper_context`` refuses for ``ctx['step_dump']``.
    It is NOT ported.  Nothing is lost: S04 is the returned ``pk``/``gz``
    and S05 is the returned ``uc``/``vc``, byte for byte.

C7. **Name and shape tables are IMPORTED from the spec, never retyped.**
    ``CSW_OUT_2D``, ``STATE_FIELDS``, ``field_shape`` and
    ``require_no_remap_needed`` come from the NumPy modules.  A retyped
    copy is a place for the two lanes to drift silently;
    ``CSW_OUT_LIKE`` is the one mapping that had to be re-stated (the
    spec's ``_out_like`` is private, and a private cross-module import
    is banned by a CI ratchet), and a drift test pins it against the
    spec's copy.

Non-hydrostatic
---------------
:func:`cgrid_pressure_phase_3d` REFUSES ``hydrostatic=False``, mirroring
the spec (``fv3_native_cgrid_phase_3d.py:141-145``).  The NH C-grid
pressure chain is a different chain, not a flag on this one, and it is
:func:`cgrid_nh_pressure_phase_3d`.

Where the spec's prose is not verifiable
----------------------------------------
The spec gives its NH refusal the reason "fv3_native_pgrad refuses the
NH p_grad_c branch".  Read at the point of use, it does not: NumPy
``p_grad_c`` implements the non-hydrostatic arm in full
(``fv3_native_pgrad.py``, ``if hydrostatic: ... else: wk = delpc[...]``),
and so does the JAX twin.  The refusal is kept -- it is right -- but for
the reason that IS checkable: ``geopk`` produces ``pk = p**akap``, while
NH ``p_grad_c`` consumes the FULL interface pressure ``pef = pe2 + pem``
that ``Riem_Solver_C`` produces (``nh_utils.F90:404``).  Feeding one
where the other is expected is a wrong quantity, not a missing branch.

Differentiability
-----------------
No ``donate_argnums`` (it conflicts with reverse-mode AD, which is the
point of this lane).  **No ``jnp.where`` is added by this module** --
R1b's dead-arm hazard needs a data-dependent select, and this module
contains none: its only branches are Python ``if``s on static
configuration, and its only array operations are ``jnp.stack`` /
``jnp.asarray`` index copies.  Every switch it inherits lives inside a
kernel gated in its own test module.  Grid metrics, halo tables and
``hs`` are STATIC (carried by ``DuoStepperContext``, hashed by
identity), so differentiating a phase means differentiating with respect
to the state fields only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from legoesm.core.fv3_duo_sw_core import c_sw
from legoesm.core.fv3_native_cgrid_phase_3d import CSW_OUT_2D
from legoesm.core.fv3_native_state_3d import (
    STATE_FIELDS,
    field_shape,
    require_no_remap_needed,
)
from legoesm.core.fv3_nh_core import riem_solver_c, update_dz_c
from legoesm.core.fv3_pgrad import geopk, p_grad_c
from legoesm.core.fv3_phase3d_common import (
    batch_size,
    CSW_OUT_LIKE,
    build_batched_gs,
    require_bool,
    require_f64_jax,
    require_km,
    require_nord,
    stack_faces,
    stack_levels,
    validate_stacked,
)
from legoesm.grids.fv3_duo_halos import stack6, unstack6

__all__ = [
    "CSW_OUT_2D",
    "CSW_OUT_LIKE",
    "state_3d_to_jax",
    "state_3d_to_numpy",
    "csw_phase_3d",
    "cgrid_pressure_phase_3d",
    "cgrid_nh_pressure_phase_3d",
    "make_csw_phase_3d_jit",
    "make_cgrid_pressure_phase_3d_jit",
    "make_cgrid_nh_pressure_phase_3d_jit",
]

# `CSW_OUT_LIKE` -- which field's shape each c_sw output shares -- now
# lives in `fv3_phase3d_common` (imported above) because
# `validate_stacked` needs it and every 3-D phase module imports that
# one; defining it here and importing it there would be a cycle.  It
# stays in this module's `__all__` so existing importers do not move,
# and `test_csw_out_like_matches_the_spec` still pins it against the
# NumPy spec's private `_out_like`, so the two cannot drift.

# What the hydrostatic and NH pressure stages must be handed.  Stated as
# data so the entry gate names the missing key instead of raising a bare
# KeyError three frames deep inside a kernel.
_PRESSURE_IN_HYDRO = ("delpc", "ptc", "uc", "vc")
_PRESSURE_IN_NH = ("delpc", "ptc", "uc", "vc", "ut", "vt", "wc")


# ---------------------------------------------------------------------
# entry gates
# ---------------------------------------------------------------------

def _require_a2b_ord(fname: str, a2b_ord) -> int:
    """Dispatch hardening: ``geopk`` selects a different B-grid
    interpolation per value, and anything outside {2, 4} would fall
    through to whichever arm the kernel's own guard happens to take."""
    if a2b_ord not in (2, 4):
        raise ValueError(
            f"{fname}: unknown a2b_ord={a2b_ord!r} (expected 2 or 4)")
    return int(a2b_ord)


def _refuse_nh_pressure(fname: str, hydrostatic: bool) -> None:
    """Mirror of ``fv3_native_cgrid_phase_3d.py:141-145``.

    A SEPARATE FUNCTION on purpose, not an inline ``raise``: a test can
    monkeypatch it to a no-op and show that the call then RUNS -- i.e.
    that the refusal is load-bearing and not decoration on an
    unreachable path.  An inline raise cannot be shown non-vacuous.

    Reason restated so it is checkable: ``geopk`` yields ``pk =
    p**akap``, while the NH ``p_grad_c`` consumes the FULL interface
    pressure ``pef = pe2 + pem`` that ``Riem_Solver_C`` produces
    (``nh_utils.F90:404``).  See the module docstring for why the spec's
    own stated reason is not verifiable in the current NumPy code.
    """
    if hydrostatic:
        return
    raise NotImplementedError(
        f"{fname}: only the hydrostatic branch is implemented here "
        f"(mirroring the NumPy specification, "
        f"fv3_native_cgrid_phase_3d.py:141-145); refusing to silently "
        f"run hydrostatic formulas for NH. geopk produces pk = p**akap, "
        f"but the non-hydrostatic p_grad_c consumes the FULL interface "
        f"pressure pef = pe2 + pem from Riem_Solver_C "
        f"(nh_utils.F90:404) -- a different quantity, not a flag. Use "
        f"cgrid_nh_pressure_phase_3d.")


def _require_grid_type_zero(fname: str, ctx) -> None:
    """The spec calls ``c_sw`` WITHOUT ``grid_type``, i.e. at 0.

    Faithful (both lanes take the kernel default), and silent if the
    gridstruct disagrees -- so it is made loud here instead.  The duo
    context sets ``gs["grid_type"] = 0``
    (``fv3_native_duo_stepper.py:225``), so this cannot fire on any
    supported configuration; it fires only when someone hands the phase
    a doubly-periodic or cartesian gridstruct whose ``grid_type`` the
    spec would then ignore.
    """
    bad = [t for t in range(batch_size(ctx)) if ctx.flags6[t].grid_type != 0]
    if bad:
        raise ValueError(
            f"{fname}: faces {bad} carry grid_type "
            f"{[ctx.flags6[t].grid_type for t in bad]}, but the C-grid "
            f"phase calls c_sw at the kernel default grid_type=0 (the "
            f"NumPy spec does not pass the argument at all, "
            f"fv3_native_cgrid_phase_3d.py:87-90). Running anyway would "
            f"silently use cubed-sphere formulas on a non-cubed-sphere "
            f"gridstruct.")


def _concrete_or_raise(fname: str, knob: str, arr, why: str):
    """Return ``arr`` as NumPy, or refuse if it is a jit tracer.

    C5.  A data-dependent Python check cannot run under ``jit``; the
    only two honest options are "raise" and "skip", and skipping
    silently is how a guard becomes decoration.  The caller opts out on
    purpose by passing ``{knob}=False``.
    """
    if isinstance(arr, Tracer):
        raise ValueError(
            f"{fname}: {knob}=True is a DATA-DEPENDENT check and this "
            f"call is being traced (jit/jvp/vjp/vmap), where the value "
            f"is not available. {why} Run the check once eagerly, then "
            f"pass {knob}=False for the traced lane -- it is refused "
            f"here rather than silently skipped, so every opt-out is "
            f"visible at its call site.")
    return np.asarray(arr)


# ---------------------------------------------------------------------
# lane-boundary state adapters (C1)
# ---------------------------------------------------------------------

def state_3d_to_jax(state6) -> dict:
    """NumPy lane's list of six per-face dicts -> the face-stacked dict.

    Carries EVERY key the per-face dicts hold (unlike the km=1
    stepper's adapter, which drops all but four): the 3-D C-grid phase
    reads ``w`` unconditionally, and the NH lane also carries ``delz``,
    so a key-dropping adapter here would silently truncate the state.

    Does NOT cast.  An f32 field RAISES rather than being promoted,
    because a promoted f32 initial condition is a run that lost bits
    before the first step and no downstream check can recover them.  The
    same gate fires when ``jax_enable_x64`` is off, since
    ``jnp.asarray`` then truncates at array creation.
    """
    if not isinstance(state6, (list, tuple)) or len(state6) != 6:
        raise ValueError(
            f"state_3d_to_jax: expected 6 per-face dicts, got "
            f"{type(state6).__name__} of length "
            f"{len(state6) if hasattr(state6, '__len__') else '?'}")
    keys = tuple(state6[0])
    for t, face in enumerate(state6):
        extra = set(face) ^ set(keys)
        if extra:
            raise KeyError(
                f"state_3d_to_jax: face {t + 1} has keys {sorted(face)}, "
                f"face 1 has {sorted(keys)}; a per-face key split would "
                f"stack a different field into the same slot")
    out = {k: stack6([face[k] for face in state6]) for k in keys}
    require_f64_jax("state_3d_to_jax", out)
    return out


def state_3d_to_numpy(states: dict) -> list:
    """Face-stacked dict -> the NumPy lane's list of six per-face dicts.

    The exact inverse of :func:`state_3d_to_jax` (pinned by a round-trip
    test), so a JAX result can be fed to a NumPy-lane consumer without
    anyone re-deriving the unstacking at a call site.

    Does NOT cast either: it never silently widens on the way out, so a
    precision-losing run cannot be made to look f64 to a downstream
    comparison. (The dtype gates now enforce UNIFORMITY, so an f32 leaf
    MIXED among f64 raises here; a genuine fp64-vs-fp32 run intent is
    caught at the model boundary, not by widening.)
    """
    if not isinstance(states, dict):
        raise TypeError(
            f"state_3d_to_numpy: expected the face-stacked dict, got "
            f"{type(states).__name__}")
    require_f64_jax("state_3d_to_numpy", states)
    per_key = {k: unstack6(v) for k, v in states.items()}
    return [{k: np.asarray(per_key[k][t]) for k in states}
            for t in range(6)]


# ---------------------------------------------------------------------
# stage 1 -- c_sw over levels and faces (dyn_core.F90:488-489)
# ---------------------------------------------------------------------

def csw_phase_3d(ctx, states: dict, dt2, km, *, nord: int = 2,
                 duogrid: bool = True, hydrostatic: bool = True,
                 remap_follows: bool = False,
                 batched: bool = False) -> dict:
    """Per-level ``c_sw`` on all six faces; returns 3-D C-grid outputs.

    ``ctx`` is the JAX lane's ONE static context,
    :class:`legoesm.core.fv3_duo_stepper.DuoStepperContext`, built once
    by ``build_jax_duo_stepper_context``.  A second context type for one
    unit would be a parallel system; this one already carries ``n``,
    ``ng``, ``bd``, the six gridstructs and their static flags, and the
    halo tables the NEXT unit needs at ``dyn_core.F90:652``/``:655``.

    ``dt2`` is the half step (``dyn_core.F90:489`` passes ``dt2``) and is
    DYNAMIC (C3).  ``nord`` comes from the deck: the shipped duo decks
    run ``nord = 2``, which is why the ``divgd`` exchange downstream is
    required at all (``dyn_core.F90:652`` gates it on
    ``flagstruct%nord > 0``).

    ``hydrostatic=False`` selects c_sw's NH arm: ``w`` rides the same
    upwind fluxes as ``delp``/``pt`` and the advected cell-centre ``wc``
    is collected -- it is the ``omga`` argument ``Riem_Solver_C``
    consumes (``dyn_core.F90:591``, intent(in) there, so the C stage
    never writes it back).  ``w`` is REQUIRED in ``states`` on both
    arms, because the spec's ``level_slice`` reads ``lev["w"]``
    unconditionally.

    Returns the face-stacked ``c_sw`` outputs: ``delpc``, ``ptc``,
    ``uc``, ``vc``, ``ua``, ``va``, ``ut``, ``vt``, ``divg_d``, plus
    ``wc`` on the NH arm.  ``delp``/``pt``/``w`` corner ghosts that
    ``c_sw`` fills internally are NOT returned -- ``c_sw`` does not
    export them, in either lane.

    ``batched`` (STATIC, default False) selects the vmap-over-faces arm
    (C2a): same kernel, same level loop, the face loop replaced by
    ``jax.vmap`` over :func:`build_batched_gs`'s stacked view.  False is
    the certified loop path, untouched.
    """
    km = require_km("csw_phase_3d", km)
    require_bool("csw_phase_3d", "remap_follows", remap_follows)
    require_no_remap_needed(km, remap_follows=remap_follows)
    for nm, vv in (("duogrid", duogrid), ("hydrostatic", hydrostatic),
                   ("batched", batched)):
        require_bool("csw_phase_3d", nm, vv)
    nord = require_nord("csw_phase_3d", "nord", nord)
    _require_grid_type_zero("csw_phase_3d", ctx)

    validate_stacked("csw_phase_3d", states, ctx, km, STATE_FIELDS,
                      what="states")
    require_f64_jax("csw_phase_3d",
                     {k: states[k] for k in STATE_FIELDS}
                     | {"dt2": jnp.asarray(dt2)})

    n, ng, npx = ctx.n, ctx.ng, ctx.npx
    names = CSW_OUT_2D + (() if hydrostatic else ("wc",))

    if batched:
        return _csw_phase_3d_batched(ctx, states, dt2, km, names=names,
                                     nord=nord, duogrid=duogrid,
                                     hydrostatic=hydrostatic)

    per_face = []
    # R1a, face axis: `c_sw` reads only face t's own fields and
    # gridstruct and writes only face t's outputs -- no iteration reads
    # a location another writes, and there is no shared buffer (this
    # lane is functional, so the NumPy lane's `fort` 1-based views onto
    # one buffer do not exist here at all).  Kept as a literal loop
    # rather than vmapped: convention C2.
    for t in range(batch_size(ctx)):
        per_level = {name: [] for name in names}
        # R1a, level axis: `c_sw`'s dummies are 2-D
        # (sw_core.F90:84-87), so a level CANNOT read another level --
        # the oracle's own `do k=1,npz` (dyn_core.F90:488) is what makes
        # the per-level call the faithful lift.  Outputs land in
        # disjoint `[:, :, k]` slots.  Independent, therefore
        # vectorisable, therefore NOT vectorised here (C2): the premise
        # is recorded so a later `vmap` has something to stand on.
        for k in range(km):
            got = c_sw(states["delp"][t][:, :, k],
                       states["pt"][t][:, :, k],
                       states["w"][t][:, :, k],
                       states["u"][t][:, :, k],
                       states["v"][t][:, :, k],
                       ctx.gs6[t], ctx.bd, npx, npx, dt2,
                       nord=nord, hydrostatic=hydrostatic,
                       duogrid=duogrid,
                       bounded_domain=ctx.flags6[t].bounded_domain)
            for name in names:
                if name not in got:
                    raise KeyError(
                        f"c_sw returned no {name!r}; keys are "
                        f"{sorted(got)}. The 3-D assembler must not "
                        f"silently drop a stage output.")
                per_level[name].append(jnp.asarray(got[name]))
        acc = {}
        for name in names:
            want2d = field_shape(CSW_OUT_LIKE[name], n, ng, km)[:2]
            acc[name] = stack_levels(
                f"csw_phase_3d[face {t + 1}]", name, per_level[name],
                want2d)
        per_face.append(acc)
    return stack_faces("csw_phase_3d", per_face)


def _csw_phase_3d_batched(ctx, states, dt2, km, *, names, nord,
                          duogrid, hydrostatic) -> dict:
    """The vmap-over-faces arm of :func:`csw_phase_3d` (C2a).

    Entry gates already ran in the caller.  The FACE loop becomes one
    ``jax.vmap`` per level over the batched gridstruct view; the LEVEL
    loop stays Python, exactly as on the loop path (the oracle's own
    ``do k=1,npz``, dyn_core.F90:488).  ``bounded_domain`` is passed as
    ONE static value: it is uniform across faces by
    ``build_batched_gs``'s common-mode gate, and ``c_sw`` branches on it
    in Python, so it could not be batched anyway.

    in_axes: the five state planes and the gridstruct dict are 0 (per
    face); ``bd``/``npx``/``dt2``/``nord``/``hydrostatic``/``duogrid``/
    ``bounded_domain`` are closed over (face-invariant -- ``dt2`` is
    traced but shared, the rest are static Python values).
    """
    bview = build_batched_gs(ctx)
    bounded = bview["flags"]["bounded_domain"]
    n, ng, npx, bd = ctx.n, ctx.ng, ctx.npx, ctx.bd

    def one_face(delp2, pt2, w2, u2, v2, gs_t):
        return c_sw(delp2, pt2, w2, u2, v2, gs_t, bd, npx, npx, dt2,
                    nord=nord, hydrostatic=hydrostatic, duogrid=duogrid,
                    bounded_domain=bounded)

    vf = jax.vmap(one_face, in_axes=(0, 0, 0, 0, 0, 0))
    per_level = {name: [] for name in names}
    for k in range(km):
        got = vf(states["delp"][:, :, :, k], states["pt"][:, :, :, k],
                 states["w"][:, :, :, k], states["u"][:, :, :, k],
                 states["v"][:, :, :, k], bview["gs"])
        for name in names:
            if name not in got:
                raise KeyError(
                    f"c_sw returned no {name!r}; keys are "
                    f"{sorted(got)}. The 3-D assembler must not "
                    f"silently drop a stage output.")
            per_level[name].append(got[name])
    out = {}
    for name in names:
        want = (batch_size(ctx),) + field_shape(CSW_OUT_LIKE[name], n, ng,
                                               km)[:2]
        for k, arr in enumerate(per_level[name]):
            if tuple(arr.shape) != want:
                raise ValueError(
                    f"csw_phase_3d[batched]: level {k} output {name!r} "
                    f"has shape {arr.shape}, the face-batched container "
                    f"expects {want}. A stagger or window mismatch here "
                    f"would broadcast, not raise.")
        # Level axis at position 3 == the loop path's per-face axis 2
        # with the face axis prepended, so both arms return the SAME
        # (6, i, j, km) layout.
        out[name] = jnp.stack(per_level[name], axis=3)
    return out


# ---------------------------------------------------------------------
# stage 2a -- hydrostatic pressure (dyn_core.F90:533 then :629)
# ---------------------------------------------------------------------

def cgrid_pressure_phase_3d(ctx, csw_outs: dict, km, *, dt2, ptop: float,
                            akap: float, cp_air: float, a2b_ord: int = 4,
                            hydrostatic: bool = True,
                            remap_follows: bool = False,
                            check_delpc: bool = True,
                            batched: bool = False) -> dict:
    """C-grid ``geopk`` then ``p_grad_c``, once per face over the column.

    ``dyn_core.F90:533`` calls geopk with ``CG = .true.`` (the literal
    ``.true.`` is on that line, verified); ``:629`` then calls
    ``p_grad_c``.  In the oracle ``p_grad_c`` MUTATES ``uc``/``vc`` IN
    PLACE, so the arrays handed in are the ones the D-grid phase reads;
    here they come back in the returned dict and the caller threads them
    (R4 / convention C4).

    Returns ``{"pk", "gz", "pe", "peln", "pkz", "uc", "vc"}``,
    face-stacked.  ``pk``/``gz`` ARE the oracle's stage boundary S04 and
    ``uc``/``vc`` ARE S05, which is why no ``stage_hook`` is ported
    (C6): a Python callback under ``jit`` receives tracers.

    ``sw_dynamics=False`` and real ``ptop``/``akap``/``cp_air`` -- the
    km=1 SW adapters in ``fv3_duo_stepper`` pass
    ``sw_dynamics=True, ptop=0, akap=1, cp_air=1``, which is the
    shallow-water degenerate case and NOT what a 3-D hydrostatic column
    wants.

    ``check_delpc`` (C5): ``geopk`` integrates ``p1d = ptop +
    cumsum(delpc)`` and takes ``log(p1d)``.  A NON-POSITIVE ``delpc`` is
    perfectly FINITE, so it sails through every ``isfinite`` check and
    only turns into NaN one stage later inside the log -- which is
    exactly how a sub-step blow-up got mis-attributed twice in the NumPy
    lane.  The check therefore fails at the source, naming face and
    level.  It is data-dependent, so under ``jit``/``jvp``/``vjp`` it
    RAISES instead of silently skipping; pass ``check_delpc=False`` to
    opt out on purpose.
    """
    km = require_km("cgrid_pressure_phase_3d", km)
    for nm, vv in (("hydrostatic", hydrostatic),
                   ("remap_follows", remap_follows),
                   ("check_delpc", check_delpc),
                   ("batched", batched)):
        require_bool("cgrid_pressure_phase_3d", nm, vv)
    require_no_remap_needed(km, remap_follows=remap_follows)
    _refuse_nh_pressure("cgrid_pressure_phase_3d", hydrostatic)
    a2b_ord = _require_a2b_ord("cgrid_pressure_phase_3d", a2b_ord)

    validate_stacked("cgrid_pressure_phase_3d", csw_outs, ctx, km,
                      _PRESSURE_IN_HYDRO, what="csw_outs")
    require_f64_jax("cgrid_pressure_phase_3d",
                     {k: csw_outs[k] for k in _PRESSURE_IN_HYDRO}
                     | {"hs6": ctx.hs6, "dt2": jnp.asarray(dt2)})

    bd = ctx.bd
    if check_delpc:
        _check_delpc_positive("cgrid_pressure_phase_3d",
                              csw_outs["delpc"], bd)

    if batched:
        return _cgrid_pressure_phase_3d_batched(
            ctx, csw_outs, km, dt2=dt2, ptop=ptop, akap=akap,
            cp_air=cp_air, a2b_ord=a2b_ord)

    per_face = []
    # R1a, face axis: geopk and p_grad_c read one face's delpc/ptc/hs
    # and write one face's outputs; the C-grid phase has NO cross-face
    # exchange (those are dyn_core.F90:652/:655, the next unit).
    for t in range(batch_size(ctx)):
        got = geopk(csw_outs["delpc"][t], csw_outs["ptc"][t], ctx.hs6[t],
                    bd, km=km, ptop=ptop, akap=akap, cp_air=cp_air,
                    cg=True, duogrid=True, computehalo=False,
                    npx=bd.ie + 1, npy=bd.je + 1, a2b_ord=a2b_ord,
                    bounded_domain=False, sw_dynamics=False)
        uc_t, vc_t = p_grad_c(dt2, csw_outs["delpc"][t], got["pk"],
                              got["gz"], csw_outs["uc"][t],
                              csw_outs["vc"][t], ctx.gs6[t], bd,
                              npz=km, hydrostatic=True)
        per_face.append({**got, "uc": uc_t, "vc": vc_t})
    return stack_faces("cgrid_pressure_phase_3d", per_face)


def _cgrid_pressure_phase_3d_batched(ctx, csw_outs, km, *, dt2, ptop,
                                     akap, cp_air, a2b_ord) -> dict:
    """The vmap-over-faces arm of :func:`cgrid_pressure_phase_3d` (C2a).

    Entry gates (including ``check_delpc``) already ran in the caller.
    One ``jax.vmap`` replaces the face loop; the per-face body is
    IDENTICAL to the loop path's -- ``geopk`` then ``p_grad_c``, same
    keyword values.

    in_axes: ``delpc``/``ptc``/``hs``/``uc``/``vc`` and the gridstruct
    dict are 0 (per face).  Closed over (face-invariant): ``bd`` and
    every ``geopk`` keyword (static Python values -- ``geopk`` takes no
    gridstruct at all), plus ``dt2`` (traced but shared) and
    ``npz=km``/``hydrostatic`` for ``p_grad_c``.
    """
    bview = build_batched_gs(ctx)
    bd = ctx.bd

    def one_face(delpc_t, ptc_t, hs_t, uc_t, vc_t, gs_t):
        got = geopk(delpc_t, ptc_t, hs_t, bd, km=km, ptop=ptop,
                    akap=akap, cp_air=cp_air, cg=True, duogrid=True,
                    computehalo=False, npx=bd.ie + 1, npy=bd.je + 1,
                    a2b_ord=a2b_ord, bounded_domain=False,
                    sw_dynamics=False)
        uc_o, vc_o = p_grad_c(dt2, delpc_t, got["pk"], got["gz"],
                              uc_t, vc_t, gs_t, bd, npz=km,
                              hydrostatic=True)
        return {**got, "uc": uc_o, "vc": vc_o}

    return jax.vmap(one_face, in_axes=(0, 0, 0, 0, 0, 0))(
        csw_outs["delpc"], csw_outs["ptc"], ctx.hs6,
        csw_outs["uc"], csw_outs["vc"], bview["gs"])


def _check_delpc_positive(fname: str, delpc6, bd) -> None:
    """The spec's ``delpc > 0`` gate, over the COMPUTE window, per face.

    Window transcribed from ``fv3_native_cgrid_phase_3d.py:164-165``
    (``is_-isd .. ie-isd``, ``js-jsd .. je-jsd``), not re-derived.
    """
    arr = _concrete_or_raise(
        fname, "check_delpc", delpc6,
        "geopk would take log(ptop + cumsum(delpc)), so a non-positive "
        "layer mass is FINITE here and only becomes NaN one stage later.")
    i0, i1 = bd.is_ - bd.isd, bd.ie - bd.isd + 1
    j0, j1 = bd.js - bd.jsd, bd.je - bd.jsd + 1
    for t in range(arr.shape[0]):
        win = arr[t][i0:i1, j0:j1, :]
        if not np.all(np.isfinite(win)) or win.min() <= 0.0:
            k = int(np.argmin(win.min(axis=(0, 1))))
            raise ValueError(
                f"{fname}: face {t + 1}: delpc entering geopk has min "
                f"{win.min():.6g} at level {k} (must be > 0). geopk will "
                f"take log(ptop + cumsum(delpc)); a non-positive layer "
                f"mass is finite and would surface as NaN one stage "
                f"later.")


# ---------------------------------------------------------------------
# stage 2b -- NH pressure (dyn_core.F90:584, :590, then :629)
# ---------------------------------------------------------------------

def cgrid_nh_pressure_phase_3d(ctx, csw_outs: dict, gz6, ws3_6, km, *,
                               dt2, ptop: float, akap: float,
                               cp_air: float, p_fac: float, a_imp: float,
                               dp0, zs6, remap_follows: bool = False,
                               batched: bool = False) -> dict:
    """NH C-grid pressure: ``update_dz_c`` -> ``Riem_Solver_C`` -> NH
    ``p_grad_c``, per face (``dyn_core.F90:584``, ``:590``, then
    ``:629``).

    Replaces the hydrostatic ``geopk`` chain of
    :func:`cgrid_pressure_phase_3d`.  The CALLER owns the ``gz``/``zh``
    cadence around this: the first substep seeds ``gz[.., km] = zs``,
    rebuilds ``gz`` from ``delz``, duo-exchanges ``gz`` and copies it
    into ``zh``; later substeps restore ``gz = zh``.  ``gz6`` arrives
    here in HEIGHT form and leaves as GEOPOTENTIAL (``Riem_Solver_C``
    rebuilds it from ``hs = phis``), which is exactly what the NH
    ``p_grad_c`` consumes.

    Corner flags are hardcoded FALSE, mirroring the spec: ``duogrid``
    forces ``bounded_domain = .true.`` (``fv_arrays.F90:1512``) and
    ``fv_grid_utils.F90:219-229`` then leaves all four flags false --
    the ``if`` at ``:224`` is ``grid_type < 3 .and. .not.
    bounded_domain``, so ``update_dz_c``'s ``fill_4corners`` calls stay
    live but inert, faithfully.  (On an ``oracle_conventions=True``
    context the gridstruct's own flags are already False, so the
    hardcode overrides nothing there.)

    ``ws3_6`` are per-face PADDED 2-D arrays ``update_dz_c`` fills and
    ``Riem_Solver_C`` reads (``ws3`` in dyn_core; intent(in) there).
    ``wc`` (the c_sw advected ``w``, dyn_core's ``omga`` at ``:591``) is
    intent(in) to the Riemann solve and never written back.

    R4 / C4 -- everything the spec mutates in place comes back:
    ``{"pkc", "gz", "ws3", "uc", "vc"}``, face-stacked.  ``pkc`` is the
    FULL C-stage interface pressure ``pe2 + pem``; ``gz`` is the
    REBUILT geopotential; ``delpc``/``ptc`` are unchanged and live on in
    ``csw_outs``.
    """
    km = require_km("cgrid_nh_pressure_phase_3d", km)
    require_bool("cgrid_nh_pressure_phase_3d", "remap_follows",
                  remap_follows)
    require_bool("cgrid_nh_pressure_phase_3d", "batched", batched)
    require_no_remap_needed(km, remap_follows=remap_follows)

    validate_stacked("cgrid_nh_pressure_phase_3d", csw_outs, ctx, km,
                      _PRESSURE_IN_NH, what="csw_outs")

    n, ng = ctx.n, ctx.ng
    m_a = n + 2 * ng
    gz6 = jnp.asarray(gz6)
    ws3_6 = jnp.asarray(ws3_6)
    zs6 = jnp.asarray(zs6)
    dp0 = jnp.asarray(dp0)
    nb = batch_size(ctx)
    want_gz = (nb,) + field_shape("gz", n, ng, km)
    if gz6.shape != want_gz:
        raise ValueError(
            f"cgrid_nh_pressure_phase_3d: gz6 has shape {gz6.shape}, "
            f"expected {want_gz} (km+1 INTERFACES, not km levels)")
    for nm, arr in (("ws3_6", ws3_6), ("zs6", zs6)):
        if arr.shape != (nb, m_a, m_a):
            raise ValueError(
                f"cgrid_nh_pressure_phase_3d: {nm} has shape "
                f"{arr.shape}, expected {(nb, m_a, m_a)} (the PADDED 2-D "
                f"plane update_dz_c indexes from is-ng)")
    if dp0.shape != (km,):
        raise ValueError(
            f"cgrid_nh_pressure_phase_3d: dp0 has shape {dp0.shape}, "
            f"expected {(km,)} (one reference thickness per layer)")
    require_f64_jax("cgrid_nh_pressure_phase_3d",
                     {k: csw_outs[k] for k in _PRESSURE_IN_NH}
                     | {"gz6": gz6, "ws3_6": ws3_6, "zs6": zs6,
                        "dp0": dp0, "hs6": ctx.hs6,
                        "dt2": jnp.asarray(dt2)})

    bd = ctx.bd
    # The JAX NH kernels take a STATIC (is_, ie, js, je, ng) int tuple
    # instead of the NumPy lane's `bd` object -- hashable BY VALUE, so an
    # equal-but-fresh bounds never costs a retrace.
    bounds = (bd.is_, bd.ie, bd.js, bd.je, bd.ng)
    npx = ctx.npx
    pkc_shape = field_shape("pk", n, ng, km)

    if batched:
        return _cgrid_nh_pressure_phase_3d_batched(
            ctx, csw_outs, gz6, ws3_6, km=km, dt2=dt2, ptop=ptop,
            akap=akap, cp_air=cp_air, p_fac=p_fac, a_imp=a_imp,
            dp0=dp0, zs6=zs6, bounds=bounds, npx=npx,
            pkc_shape=pkc_shape)

    per_face = []
    # R1a, face axis: each face's update_dz_c / Riem_Solver_C /
    # p_grad_c read and write that face's arrays only.  Functional
    # throughout, so unlike the NumPy lane there is no shared gz6/ws3_6
    # buffer for one face's write to reach another's read.
    for t in range(nb):
        gz_t, ws_t = update_dz_c(
            bounds, km, dt2, dp0, zs6[t], ctx.gs6[t]["area"],
            csw_outs["ut"][t], csw_outs["vt"][t], gz6[t], ws3_6[t],
            npx, npx,
            sw_corner=False, se_corner=False,
            ne_corner=False, nw_corner=False,
            grid_type=ctx.flags6[t].grid_type)
        # `pkc` is intent(out)-shaped scratch in the oracle; allocate it
        # here rather than asking the caller for a buffer (R4).
        # dtype follows storage (fp32/fp64), from csw_outs["delpc"]
        pkc0 = jnp.zeros(pkc_shape, dtype=csw_outs["delpc"][t].dtype)
        gz_t, pkc_t = riem_solver_c(
            1, dt2, bounds, km, akap, cp_air, ptop, ctx.hs6[t],
            csw_outs["wc"][t], csw_outs["ptc"][t], csw_outs["delpc"][t],
            gz_t, pkc0, ws_t, p_fac, a_imp)
        uc_t, vc_t = p_grad_c(dt2, csw_outs["delpc"][t], pkc_t, gz_t,
                              csw_outs["uc"][t], csw_outs["vc"][t],
                              ctx.gs6[t], bd, npz=km, hydrostatic=False)
        per_face.append({"pkc": pkc_t, "gz": gz_t, "ws3": ws_t,
                         "uc": uc_t, "vc": vc_t})
    return stack_faces("cgrid_nh_pressure_phase_3d", per_face)


def _cgrid_nh_pressure_phase_3d_batched(ctx, csw_outs, gz6, ws3_6, *,
                                        km, dt2, ptop, akap, cp_air,
                                        p_fac, a_imp, dp0, zs6, bounds,
                                        npx, pkc_shape) -> dict:
    """The vmap-over-faces arm of :func:`cgrid_nh_pressure_phase_3d`.

    Entry gates already ran in the caller.  One ``jax.vmap`` replaces
    the face loop; the per-face body is IDENTICAL to the loop path's --
    ``update_dz_c`` -> ``riem_solver_c`` -> NH ``p_grad_c``.

    ``grid_type`` is passed as ONE static value: the loop path reads
    ``ctx.flags6[t].grid_type`` per face, and ``build_batched_gs``'s
    common-mode gate guarantees the six are equal (it RAISES otherwise),
    so the shared value is the same value -- and ``update_dz_c``
    branches on it in Python, so it could not be batched anyway.  The
    corner flags are the loop path's hardcoded ``False`` (see the
    caller's docstring).

    in_axes: ``zs``/``ut``/``vt``/``gz``/``ws3``/``hs``/``wc``/``ptc``/
    ``delpc``/``uc``/``vc`` and the gridstruct dict (``area`` for
    ``update_dz_c``, the metric arrays for ``p_grad_c``) are 0 (per
    face).  Closed over (face-invariant): ``bounds``/``km``/``npx``/
    ``akap``/``cp_air``/``ptop``/``p_fac``/``a_imp``/``bd`` (static
    Python values -- ``a_imp`` selects a Python branch inside
    ``riem_solver_c``), plus ``dt2`` and ``dp0`` (traced but shared: one
    half step and ONE reference-thickness column for all six faces,
    exactly as on the loop path).
    """
    bview = build_batched_gs(ctx)
    grid_type = bview["flags"]["grid_type"]
    bd = ctx.bd

    def one_face(zs_t, ut_t, vt_t, gz_t, ws_t, hs_t, wc_t, ptc_t,
                 delpc_t, uc_t, vc_t, gs_t):
        gz1, ws1 = update_dz_c(
            bounds, km, dt2, dp0, zs_t, gs_t["area"], ut_t, vt_t,
            gz_t, ws_t, npx, npx,
            sw_corner=False, se_corner=False,
            ne_corner=False, nw_corner=False,
            grid_type=grid_type)
        # dtype follows storage (fp32/fp64), from delpc_t
        pkc0 = jnp.zeros(pkc_shape, dtype=delpc_t.dtype)
        gz2, pkc_t = riem_solver_c(
            1, dt2, bounds, km, akap, cp_air, ptop, hs_t, wc_t,
            ptc_t, delpc_t, gz1, pkc0, ws1, p_fac, a_imp)
        uc_o, vc_o = p_grad_c(dt2, delpc_t, pkc_t, gz2, uc_t, vc_t,
                              gs_t, bd, npz=km, hydrostatic=False)
        return {"pkc": pkc_t, "gz": gz2, "ws3": ws1,
                "uc": uc_o, "vc": vc_o}

    return jax.vmap(one_face, in_axes=(0,) * 12)(
        zs6, csw_outs["ut"], csw_outs["vt"], gz6, ws3_6, ctx.hs6,
        csw_outs["wc"], csw_outs["ptc"], csw_outs["delpc"],
        csw_outs["uc"], csw_outs["vc"], bview["gs"])


# ---------------------------------------------------------------------
# jit factories -- ONE policy per public routine, declared here so a
# call site can never invent a different static split.
#
# `ctx` is STATIC (it is hashed by identity and carries the six
# gridstructs, the static flags and `hs`, all constants in this lane);
# `km` and every deck constant / Python-branch selector are STATIC;
# `dt2` and the field data are DYNAMIC (C3).  Each `km`-style argument
# appears in BOTH `static_argnums` and `static_argnames` so it is static
# whether the caller passes it positionally or by keyword -- jax
# resolves the two independently when both are given.  No
# `donate_argnums` anywhere: buffer donation conflicts with reverse-mode
# AD, which is the point of the lane.
# ---------------------------------------------------------------------

def make_csw_phase_3d_jit(fn=csw_phase_3d):
    """Static: ``ctx``, ``km``, ``nord``, ``duogrid``, ``hydrostatic``,
    ``remap_follows``.  ``states`` and ``dt2`` dynamic."""
    return jax.jit(fn, static_argnums=(0, 3),
                   static_argnames=("km", "nord", "duogrid",
                                    "hydrostatic", "remap_follows",
                                    "batched"))


def make_cgrid_pressure_phase_3d_jit(fn=cgrid_pressure_phase_3d):
    """Static: ``ctx``, ``km`` and every keyword except ``dt2``.

    ``check_delpc`` is static because it is a Python branch; note that
    the DEFAULT (``True``) makes a traced call RAISE by design (C5) --
    pass ``check_delpc=False`` for the compiled lane, having run the
    check once eagerly.
    """
    return jax.jit(fn, static_argnums=(0, 2),
                   static_argnames=("km", "ptop", "akap", "cp_air",
                                    "a2b_ord", "hydrostatic",
                                    "remap_follows", "check_delpc",
                                    "batched"))


def make_cgrid_nh_pressure_phase_3d_jit(fn=cgrid_nh_pressure_phase_3d):
    """Static: ``ctx``, ``km``, ``ptop``, ``akap``, ``cp_air``,
    ``p_fac``, ``a_imp``, ``remap_follows``.

    ``a_imp`` MUST be static -- ``riem_solver_c`` branches on it in
    Python (``if not (a_imp > 0.999): raise``).  ``dp0``/``zs6``/``gz6``/
    ``ws3_6``/``csw_outs``/``dt2`` are dynamic.
    """
    return jax.jit(fn, static_argnums=(0, 4),
                   static_argnames=("km", "ptop", "akap", "cp_air",
                                    "p_fac", "a_imp", "remap_follows",
                                    "batched"))
